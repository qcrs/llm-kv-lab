# P1-M1-T2-IMPL-01A-R1 实施记录

## 1. 为什么出现 R1

原 01A 方案分别调用 `append_block_ids(..., overwrite=True)` 写 retained row，再调用 `overwrite=False` 写同一步新块。固定源码中的 `StagedWriteTensor` descriptor 容量按 `num_rows` 配置；同一 request 在一次 publication window 产生两个 descriptor，`max_num_reqs=1` 时触发 broadcast capacity error。因此原方案保留为历史 DESIGN_CONFLICT evidence，本 R1 只修复组合方式。

## 2. 原方案与新方案

原方案：`retained overwrite + normal append + single apply`，实际是 two descriptors。新方案在 Worker CPU 侧把 `retained_block_ids` 与同一步 `same_step_new_block_ids` 拼成 `final_physical_row`，只调用一次 `append_block_ids(..., overwrite=True)`，再复用既有 `apply_staged_writes()`。

## 3. Task Identity 与范围

- Slice：`P1-M1-T2-IMPL-01A-R1`
- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/physical-kv-reclaim-v026`
- HEAD：`568afb3a13806beb53bb2e6bd518269357b237c0`
- 范围：single KV cache group、whole-block reclaim、Worker execution view；不做 01B、T3、transport、ownership、free/reuse。

## 4. Source Facts

| 标签 | 文件/函数 | 源码行为 | 本 slice 意义 |
|---|---|---|---|
| SOURCE_FACT | `vllm/v1/worker/gpu/block_table.py::BlockTables.append_block_ids` | `overwrite=True` 从 column 0 stage row，并立即更新 CPU `num_blocks`；`overwrite=False` 从当前 frontier 开始。 | final row 必须一次 overwrite，避免第二 descriptor。 |
| SOURCE_FACT | `vllm/v1/worker/gpu/buffer_utils.py::StagedWriteTensor` | staged descriptor 数量按 row capacity 管理，`stage_write` 与 `apply_write` 分离。 | 解释历史 two-descriptor conflict。 |
| SOURCE_FACT | `vllm/v1/worker/gpu/states.py::RequestState` | `effective_kv_len` 是独立 GPU state；`num_computed_tokens_np` 提供 logical CPU mirror。 | reclaim 只 shrink physical state，logical 不回退。 |
| SOURCE_FACT | `model_runner.py::prepare_attn` | slot mapping 使用 cache-position channel。 | R1 不改变 S2/S3 data path。 |

## 5. Design Choices

- `DESIGN_CHOICE`：CPU materialized retained IDs，复用 existing overwrite staged-write。
- `DESIGN_CHOICE`：不修改 `block_table.py` 或 `buffer_utils.py`，不扩大 descriptor capacity，不做 second publication。
- `DESIGN_CHOICE`：reclaim event 允许一次 `effective_kv_len.gpu[req_index].item()` 做 old-E validation；不新增 CPU physical mirror。
- `DESIGN_CHOICE`：same-step new blocks 是 pre-step allocation delta，不能并入 `new_effective_kv_len`。
- `DESIGN_CHOICE`：不 free candidate blocks，不修改 Scheduler canonical ownership。

## 6. 逐文件修改说明

### `vllm/v1/worker/gpu/model_runner.py`

修改前 `_commit_reclaim_transition()` 会为 retained row 与正常 append 分别准备写入，无法满足 descriptor capacity。修改后该 helper 接收 `same_step_new_block_ids`，在全部 validation 通过后构造 `final_physical_row`，仅执行一次 `append_block_ids(req_index, (final_physical_row,), overwrite=True)`；只有 physical shrink 时 stage `effective_kv_len`。逻辑状态 `num_computed_tokens_np` 只读。新增了 request/group/count/duplicate/length/modulo/no-op 校验。替换：two-stage mutation → one final-row mutation。删除：无。

### `tests/v1/worker/test_gpu_reclaim_commit.py`

新增真实 CUDA `RequestState`、`BlockTables` oracle，覆盖 canonical row、same-step B6、descriptor count、no-op、invalid transition、normal growth 与 boundary slot mapping。修改：历史 conflict 场景改为 single-write regression。替换：two descriptor expectation → one descriptor expectation。删除：无。

## 7. Helper 说明

`GPUModelRunner._commit_reclaim_transition(request_id, retained_block_ids, new_effective_kv_len, expected_old_num_blocks, same_step_new_block_ids)` 的输入是已 materialize 的 block IDs；输出没有返回值，mutation 以 staged state 形式发布。preconditions 是 request 存在、single group、旧 block 数匹配、retained 非空且唯一、new E 合法且 whole-block 对齐、logical/physical 对齐。校验全部先于 mutation。无 shrink 且无 new blocks 时 strict no-op；无 shrink 但有 new blocks 时仍生成一次 final-row replacement。该 helper 不负责 producer policy、canonical subset 验证、transport、free 或 ACK。

## 8. 状态转换与 Same-Step Append

Before：`L=94, E=94, blocks=[10,11,12,13,14,15]`。Reclaim retained `[10,11,14,15]`、`new_E=62`；同一步 new `[16]`。最终一次 overwrite 后为 `[10,11,14,15,16]`，`num_blocks=5`，`L=94`、`E=62`。KV payload 没有 copy。若先把 active frontier 缩到 4 再以 `overwrite=False` 追加，会产生第二 descriptor；因此 R1 在 CPU 侧预先合并，避免 B6 stranded at old column 6。

## 9. Validation Contract

Worker 可验证 request existence、single group、旧 block count、retained 非空/唯一/数量、new E 正值且不增长、whole-block shrink、block alignment、`new_E <= logical` 及 logical/physical modulo。失败时 staged row、`num_blocks.np`、`effective_kv_len`、logical state 均不变。Worker 不能验证 retained IDs 是否为 old canonical row 的有序子集，也不能验证 partial-tail identity；这些留给未来 CPU control-plane producer，不增加 GPU readback 或 CPU row mirror。

## 10. Tests

| 测试 | 输入/期望 | 结果 |
|---|---|---|
| `test_canonical_reclaim_transition` | `[10..15]` → `[10,11,14,15]`, E 94→62, L 94 | 历史运行 PASS |
| `test_single_write_final_row_avoids_descriptor_conflict` | retained + `[16]` → `[10,11,14,15,16]`, descriptor=1 | 历史运行 PASS |
| `test_noop_identity_does_not_stage_mutation` | keep-all, E 不变 | PASS |
| `test_no_reclaim_with_new_delta_composes_one_final_row` | old row + `[16]`, descriptor=1 | PASS |
| invalid count/duplicate/new-E/logical alignment tests | fail-before-mutation | PASS |
| `test_normal_growth_after_reclaim_keeps_two_progress_states_independent` | 94/62 → 95/63 | PASS |
| `test_boundary_crossing_maps_position_64_to_new_block` | positions 62,63,64 → slots 254,255,256 | PASS |

历史完整运行：`25 passed, 15 warnings`。本次收尾复跑因 NVIDIA driver unavailable 为 `25 skipped`，见 raw log；未覆盖结果不推翻此前通过的 CUDA oracle。

## 11. Descriptor Evidence

旧机制：同 request 两次 `append_block_ids` → 2 descriptors → `max_num_reqs=1` capacity conflict。新机制：`retained + same-step new → final row → one overwrite`，测试明确断言 staged descriptor count 为 1，且 `apply_staged_writes()` 成功。

## 12. Evidence

- `/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a-r1/00-source-identity.log`
- `/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a-r1/01-source-review.log`
- `/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a-r1/02-targeted-tests.log`
- `/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a-r1/03-regression-tests.log`
- `/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a-r1/04-pycompile-diff-check.log`

## 13. Git Diff Summary

累计 worktree diff 为 7 个 production files、174 additions、4 deletions，包含 S1/S2/S3/Review Fix/R1，不能全部归因 R1。R1 semantic delta 仅限 `model_runner.py` helper 的 final-row composition 与 `test_gpu_reclaim_commit.py` 的对应 oracle；本 slice 没有 production 删除。最终 `git diff --check` 通过。

## 14. 未做事项与 Remaining Risks

未做 Scheduler transport、`SchedulerOutput`、reclaim producer、ownership shrink、`BlockPool.free`、reuse、ACK、allocator effective occupancy、retention policy、GPU compaction、DCP/PCP/cascade/spec/CUDA Graph、01B/T3。当前只证明 Worker staged final-row primitive；真实 execute-model transport/order 仍是 `P1-M1-T2-IMPL-01B` 的范围。

## 15. Next Allowed Action

Review P1-M1-T2-IMPL-01A-R1 implementation, tests, evidence, and diff tracker.
