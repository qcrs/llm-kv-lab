# P1-M1-T2-IMPL-01A 实施记录

## 1. Task Identity

| 字段 | 值 |
|---|---|
| Task ID | `P1-M1-T2-IMPL-01A` |
| 名称 | Worker Reclaim Commit Primitive |
| 日期 | 2026-08-28（Asia/Shanghai） |
| Worktree | `/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim` |
| Branch | `p1/physical-kv-reclaim-v026` |
| Pinned HEAD | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| 起始状态 | 保留 S1/S2/S3 + Web Review Fix 未提交修改 |
| 执行结论 | `DESIGN_CONFLICT`；未完成 Slice acceptance |

本 Slice 只尝试 Worker physical-view staged transition，不包含 transport、ownership、free、reuse、ACK、allocator 或 retention policy。

## 2. 本 Slice 要解决的问题

S1 建立 persistent `effective_kv_len`，S2 让 KV write 使用 `cache_positions`，S3 让 normal FA2 read extent 使用 guarded `effective_kv_seq_lens`。T2-01A 原计划增加一个 Worker primitive，把 control plane 已准备好的 retained physical block IDs 提交为新的 execution view：

```text
[B0 B1 B2 B3 B4 B5], effective=94
                 |
                 v
[B0 B1 B4 B5], effective=62
```

这一步只改变 Worker metadata indirection 和 physical progress，不复制 KV payload，也不改变 logical `num_computed_tokens=94`。Scheduler canonical ownership 仍保留旧 blocks，B2/B3 只是 future candidate，不在本 Slice free。

## 3. Approved Design

批准方案选择 CPU-materialized retained IDs，并复用 MRV2 `append_block_ids(..., overwrite=True)`，因为 `max_model_len=2048`、`block_size=16` 时整行约 128 个 `int32` IDs，没必要引入 GPU keep-index compaction kernel、临时 buffer 或新 ABI。

批准的 same-step composition 还要求：

```text
stage reclaim overwrite
-> num_blocks CPU frontier 变为 4
-> stage normal append B6 with overwrite=False
-> single apply_staged_writes()
-> [B0 B1 B4 B5 B6]
```

本轮 source/runtime test 证明：frontier 更新成立，但底层 staged-write capacity 无法一般性承载同 row 的两条 staged writes。因此批准组合不能按原假设直接完成。

## 4. Source Facts

### SOURCE_FACT 1

- 文件：`vllm/v1/worker/gpu/block_table.py`
- 类/函数：`BlockTables.append_block_ids()`，L107-L120
- 源码行为：`overwrite=True` 使用 `start=0`；`overwrite=False` 从 `num_blocks.np` 读取 start；每次调用通过 `stage_write()` 增加一条 staged write，并立即更新 `num_blocks.np`。
- 本 Slice 意义：active CPU frontier 能先缩短再让 normal append 选择 column 4。

### SOURCE_FACT 2

- 文件：`vllm/v1/worker/gpu/buffer_utils.py`
- 类/函数：`StagedWriteTensor.__init__()` / `stage_write()` / `apply_write()`，L130-L201
- 源码行为：staged write descriptors 会 append 到 Python lists；descriptor UVA buffers `write_indices`、`write_starts`、`write_cu_lens` 的 capacity 固定为 `num_rows`。`apply_write()` 把全部 staged descriptors copy 到这些 buffers 后才 launch kernel。
- 本 Slice 意义：设计隐含每 publication window 每 row 至多一条 staged write；同一个 request 的 reclaim overwrite + normal append 是两条 descriptor。

### OBSERVED 1

- 测试：`test_reclaim_then_normal_append_uses_new_frontier`
- 对真实 `BlockTables(max_num_reqs=1)` 连续 stage overwrite 和 append 后执行 `apply_staged_writes()`。
- 结果：`_staged_write_indices=[0,0]` 无法 copy 到 capacity 1 的 buffer，抛出 `ValueError: could not broadcast input array from shape (2,) into shape (1,)`。
- 结论：批准的 single-publication same-step composition 不能由现有 primitive 一般性保证。

### SOURCE_FACT 3

- 文件：`vllm/v1/worker/gpu/states.py`
- 类/函数：`RequestState.effective_kv_len` / `apply_staged_writes()`
- 源码行为：S1 physical state 是独立 GPU `StagedWriteTensor`，支持 `stage_write_elem()` 后通过 `apply_write()` 发布。
- 本 Slice 意义：absolute physical shrink 本身可以 staged；冲突位于 block-table 同 row 双写组合。

## 5. 修改文件总表

| 文件 | 修改类型 | 函数/类 | 修改目的 |
|---|---|---|---|
| `vllm/v1/worker/gpu/model_runner.py` | 新增（未通过 acceptance） | `GPUModelRunner._commit_reclaim_transition()` | 校验并 stage single-group Worker physical transition |
| `tests/v1/worker/test_gpu_reclaim_commit.py` | 新增（失败 evidence 保留） | 01A targeted tests | 覆盖 canonical transition、composition、no-op、invalid atomicity、normal growth |
| `vllm/v1/worker/gpu/block_table.py` | 未修改但依赖 | `append_block_ids()` / `apply_staged_writes()` | existing row overwrite/frontier/publication primitive |
| `vllm/v1/worker/gpu/buffer_utils.py` | 未修改但依赖 | `StagedWriteTensor` | existing staged descriptor storage/publication |

替换：无。删除：无。

## 6. 逐文件详细修改说明

### `vllm/v1/worker/gpu/model_runner.py`

修改前没有 Worker reclaim commit helper。尝试新增 `_commit_reclaim_transition()`，输入 request ID、materialized retained block IDs、absolute new effective length 和 expected old block count。

新增 validation：request existence、single group、old count、non-empty、retained count、duplicate IDs、positive/new-not-growing effective length、whole-block count/shrink consistency、block alignment、logical upper bound 和 logical/physical alignment。validation 全部发生在 staged mutation 前；identity transition 直接返回。

validation 后尝试：

```text
append_block_ids(... overwrite=True)
effective_kv_len.stage_write_elem(...)
```

替换：无。删除：无。风险：helper 尚未获得 same-step composition acceptance，当前不能视为可交付实现。

### `tests/v1/worker/test_gpu_reclaim_commit.py`

新增 181 行 focused CUDA tests，使用真实 `RequestState` 和 `BlockTables`。canonical transition、no-op、多项 invalid-before-mutation、existing S1 normal growth 已得到通过观察；same-step composition 触发 source-level capacity conflict。另有两个 test 因 validation error message 顺序与预期 regex 不同而失败；核心 conflict 触发后没有继续修改它们。

替换：无。删除：无。

## 7. 新增 Helper 说明

`GPUModelRunner._commit_reclaim_transition()`：

- 输入：`request_id`、`retained_block_ids`、`new_effective_kv_len`、`expected_old_num_blocks`；
- 输出：无；成功时只 stage mutation；
- preconditions：active request、single group、whole-block-consistent transition；
- mutation：stage block row overwrite 与 absolute physical length；
- no-op：block count 和 effective length 均不变时直接 return；
- 不负责：producer decision、canonical ordered-subset validation、partial-tail block identity、transport、apply ordering、ownership/free/reuse/ACK。

该 helper 当前属于冲突现场的未接受 patch，不应在 Web review 前接入 runtime。

## 8. 状态转换 Before / After

单独 canonical transition 已观察通过：

```text
Before:
logical=94
physical=94
blocks=[B0 B1 B2 B3 B4 B5]

Transition:
retained=[B0 B1 B4 B5]
new_E=62

After publication:
logical=94
physical=62
blocks=[B0 B1 B4 B5]
```

KV payload 没有 copy。但 Slice 的必要 acceptance 还包含 same-step append，因此单独 transition PASS 不能让 01A PASS。

## 9. Same-Step Append Composition

目标结果是 `[B0 B1 B4 B5 B6]`，避免 B6 使用旧 frontier column 6。CPU `num_blocks.np` 的确在 reclaim stage 后变为 4，因此 append 的 start 选择正确。

实际阻塞发生在 publication：overwrite 和 append 为同一 row 产生两条 staged descriptors，而 `StagedWriteTensor` descriptor buffers 只按 `num_rows` 分配。`max_num_reqs=1` 时两条 descriptor 必然 overflow；生产 batch 即使 `max_num_reqs>1`，全 batch 额外双写也可能超过总 capacity，而且现有 contract 没有证明同-row descriptors 的 publication ordering。

可能选项必须重新 review：

1. 在 `BlockTables` 层合并同 row overwrite+append 为一条 materialized staged write；
2. 扩展 `StagedWriteTensor` descriptor capacity 并证明同-row ordering；
3. 在 normal append 前先 publication reclaim，再进行第二 publication，并审计 stream/order；
4. 让 01B producer 预合成 retained+new IDs，但这改变 01A/01B contract。

这些都超出当前 allowed files 或 approved ordering。

## 10. Validation Contract

Worker helper 可以验证 request existence、single group、count、duplicate、positive/shrink、whole-block alignment 和 logical不回退。Worker 当前不能验证 retained IDs 是否为 old canonical row 的有序子集，也不能验证 active partial-tail block identity 被保留；这些需要 future CPU control-plane producer 基于 canonical old row 验证。

本轮没有增加 GPU→CPU block-table readback、CPU mirror 或 GPU gather kernel。读取 `effective_kv_len.gpu.item()` 发生同步，但这是 isolated internal primitive 的当前尝试；其 runtime placement/performance 仍需 review。

## 11. Tests

执行结果：`20 passed, 3 failed, 15 warnings in 17.48s`。

- canonical reclaim transition：PASS；
- same-step append composition：FAIL，发现 staged descriptor capacity conflict；
- no-op identity：PASS；
- old count mismatch / duplicate / inconsistent length 等 invalid-before-mutation：多数 PASS；
- 两个 validation-message tests：FAIL，实际更早被 whole-block consistency 拒绝；
- S1 effective-state regression suite：PASS；
- `py_compile`：PASS；
- `git diff --check`：PASS。

因为 same-step composition 是硬 acceptance，Slice 结论为 `DESIGN_CONFLICT`，不是 partial PASS。

## 12. Evidence

```text
/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a/00-source-identity.log
/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a/01-focused-source-audit.log
/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a/02-targeted-tests.log
/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a/03-pycompile.log
/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a/04-git-diff.log
/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a/05-final-status.log
```

## 13. Git Diff Summary

相对 pinned HEAD 的 accumulated tracked production diff 当前为：

```text
7 files changed, 161 insertions(+), 4 deletions(-)
```

其中 01A 在 `model_runner.py` 增加约 66 行 helper；新增 untracked test 文件 181 行。其余 tracked diff 属于既有 S1/S2/S3 + review-fix。没有修改或删除 `block_table.py`、`buffer_utils.py`。

## 14. 未做事项

没有实现 transport、`SchedulerOutput` field、ownership shrink、`BlockPool.free`、reuse、ACK、allocator change、retention policy、GPU compaction、01B 或 T3。没有修改 third_party，也没有 commit。

## 15. Remaining Risks / Next Slice

当前不是直接进入 01B，而是先由 Web review 本 DESIGN_CONFLICT，选择 same-step composition 的安全表达方式。只有重新批准 01A contract/allowed files 并完成 01A 后，未来 `P1-M1-T2-IMPL-01B` 才处理 prepared transition transport 与 `execute_model()` integration ordering。

```text
Next Allowed Action:
Review P1-M1-T2-IMPL-01A evidence and implementation record.
```
