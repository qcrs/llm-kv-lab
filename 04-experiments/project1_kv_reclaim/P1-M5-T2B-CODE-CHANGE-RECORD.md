# P1-M5-T2B 代码变更记录

## Slice Summary

- 目的：在冻结的 M4/M5-T1/T2A contract 上实现真实的 post-forward V2 payload compaction、Worker BlockTable staged next-state、absolute `effective_kv_len` override 和 `CompactionResultData` publication。
- Slice：`P1-M5-T2B`
- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/v2-token-compaction-v026`
- Source HEAD before：`1f04cdca301fa72d20046a791672cf265bf9df09`
- Source HEAD after：`1f04cdca301fa72d20046a791672cf265bf9df09`（未创建 commit）
- 本轮没有修改 M4 primitive、Scheduler canonical ownership 或 BlockPool。
- Runtime mutation introduced：YES，仅限批准的 T2B Worker payload compaction、BlockTable staged overwrite 和 post-update absolute override。
- KV payload mutation introduced：YES，仅在所有 T2A deterministic validation 成功后调用 `compact_paged_kv_triton_2d()`。
- Ownership mutation introduced：NO；没有 Scheduler reconcile、free、reuse 或 deferred-free。
- Files Added：NONE（本记录和 raw evidence 位于 worktree 外层 workspace evidence 目录）。
- Files Modified：`vllm/v1/worker/gpu/model_runner.py`、`vllm/v1/worker/gpu/input_batch.py`、`tests/v1/worker/test_gpu_compaction_preparation.py`、`tests/v1/worker/test_gpu_effective_kv_state.py`。
- Files Deleted：NONE。

## 1. 设计边界与实现结论

- `SOURCE_FACT`：M4 production primitive 为 `compact_paged_kv_triton_2d()`，输入为 `[B,H,N,C]`，`shape[2]` 是 physical `block_size`，返回 `(new_effective_kv_len, new_num_blocks)`。
- `APPROVED_DESIGN`：执行顺序为 `model.forward()` → T2A source fence → 所有 layer M4 compaction → BlockTable staged overwrite → absolute `effective_kv_len` override/result publication → `sample_tokens()` → `post_update()`。
- `IMPLEMENTATION_DECISION`：使用 `_PreparedCompaction` 作为 execute-model-local descriptor；使用 batch-indexed `effective_kv_len_override_valid/value`，由现有 `idx_mapping` 在 `post_update()` 中写回 persistent `req_state_idx`。
- `IMPLEMENTATION_DECISION`：保留 `BlockTables.append_block_ids(..., overwrite=True)` 作为 staged overwrite path，不新增 immediate GPU metadata API。
- `IMPLEMENTATION_DECISION`：普通 `post_update()` 路径创建 zero override tensors，避免 Triton kernel 接收 `None` pointer；这不改变普通路径语义。

## 2. 逐文件变更审计

### CHG-01

File：`vllm/v1/worker/gpu/model_runner.py`

Change type：`MODIFIED`

Final symbols / lines：

- `_PreparedCompaction`：L129-L139
- `GPUModelRunner._prepare_v2_compactions`：L1812-L1945
- `GPUModelRunner._execute_v2_compactions`：L1948-L2000
- `GPUModelRunner.execute_model` T2B seam：L2232-L2257
- `GPUModelRunner.sample_tokens` result/override handoff：L2274-L2279、L2328-L2336、L2361-L2369
- `ExecuteModelState`：L2517-L2527

Previous responsibility：已有 M5-T1/T2A plan extraction 和 source preparation；没有 post-forward destructive execution、Worker next-state stage、override 或 result publication。

New responsibility：在 model forward 完成后消费已验证 descriptor，跨所有 `self.kv_caches` 调用 M4 primitive，成功后 stage retained BlockTable prefix，创建 batch-indexed absolute override，并把 completed `CompactionResultData` 送入 `ExecuteModelState`/`ModelRunnerOutput`。

Exact change：

- `ADD` `CompactionResultData` import 和 `_execute_v2_compactions()`。
- `EXTEND` `_prepare_v2_compactions()`，加入 MVP exact page-count guard：`source_num_blocks == cdiv(source_effective_kv_len, block_size)`。
- `REPLACE/EXTEND` `execute_model()` post-forward seam，使 T2B 在 `ExecuteModelState` publication 前执行。
- `EXTEND` `sample_tokens()`、`postprocess_sampled()` 和 `ExecuteModelState` 以携带 override/result。

Why required：避免用 capacity-only 检查把非 exact active source page row 交给 M4；确保 payload compact 完成后才发布 Worker physical next-state 和 result。

Requirement mapping：T2A exact page-count guard；T2B post-forward M4 execution；all-layer success-before-stage；retained prefix；absolute E override；result transport。

Classification：`SOURCE_FACT_DRIVEN` + `DESIGN_DECISION`

Runtime mutation introduced：`YES`，仅在 `_execute_v2_compactions()` 内，并且发生在全部 T2A validation 之后。

Compatibility impact：M4/T1/V1 contract 未改变；PP、free/reuse、Scheduler reconcile 不在 MVP scope。

Tests covering it：`tests/v1/worker/test_gpu_compaction_preparation.py` 的 T2A/T2B focused tests；`tests/v1/worker/test_gpu_kv_compaction.py`；`tests/v1/core/test_reclaim_transport.py`。

### CHG-02

File：`vllm/v1/worker/gpu/input_batch.py`

Change type：`MODIFIED`

Final symbols / lines：

- `_post_update_kernel`：L495-L564
- `post_update`：L566-L626

Previous responsibility：按 `computed_delta` 增量更新 `num_computed_tokens` 和 `effective_kv_len`。

New responsibility：在逻辑 token counter 独立推进的同时，若当前 batch row 有 V2 absolute override，则以 `override_value` 直接提交 persistent `effective_kv_len`。

Exact change：

- `ADD` 两个 kernel pointer 参数和两个可选 Python tensor 参数。
- `EXTEND` kernel：先处理 `computed_delta`，再执行 `override_valid` 分支；override 不嵌套在 delta 分支中。
- `ADD` 普通路径 zero validity/value tensor materialization，避免 Triton `NoneType` pointer 编译失败。

Why required：V2 的 `K` 已是当前 forward 后 compaction 的最终 physical length，必须得到 `E_next = K`，不能变成 `K + delta`；即使 `computed_delta == 0` 也必须提交 override。

Requirement mapping：T2B absolute E override；normal/V1 increment compatibility；`req_state_idx != batch_idx` mapping preservation。

Classification：`DESIGN_DECISION`

Runtime mutation introduced：`YES`，是批准的 persistent Worker effective-state commit；不是 execute-model 内直接写 E。

Compatibility impact：无 override 时普通/V1 行为保持原有增量语义；现有 V1 reclaim/effective-state regression 通过。

Tests covering it：`tests/v1/worker/test_gpu_effective_kv_state.py` 的 baseline、V1 divergence、V2 `computed_delta == 0` 和 `computed_delta != 0` tests。

### CHG-03

File：`tests/v1/worker/test_gpu_compaction_preparation.py`

Change type：`MODIFIED`

Final symbols / lines：

- `test_prepare_v2_compaction_rejects_insufficient_source_capacity`：L155-L159
- `test_execute_v2_compaction_compacts_all_layers_and_stages_next_state`：L182-L224
- `test_execute_v2_compaction_does_not_publish_success_after_layer_failure`：L226-L258

Previous responsibility：覆盖 T2A source fence，但没有 exact page-count assertion 或 T2B execution-level focused coverage。

New responsibility：覆盖 exact page-count failure、两层 fake M4 execution、retained prefix/staged overwrite、batch-indexed override/result，以及 layer failure 的 no-stage/no-success-result semantics。

Exact change：

- `REPLACE` capacity test 的 expected error 为 exact page count。
- `ADD` fake M4 multi-layer success test。
- `ADD` second-layer injected failure test。

Why required：验证 T2B control/data seam，而不是重新测试 M4 kernel correctness；显式保留“无 rollback claim”语义。

Requirement mapping：T2A exact page count；multi-layer payload compaction；retained BlockTable prefix；failure injection；no successful result publication。

Classification：`TEST_ONLY`

Runtime mutation introduced：`NO`（测试 fake mutation 仅在 isolated fixture 内）。

Compatibility impact：仅增加 focused coverage。

Tests covering it：本文件自身；执行结果见 raw log。

### CHG-04

File：`tests/v1/worker/test_gpu_effective_kv_state.py`

Change type：`MODIFIED`

Final symbols / lines：

- `test_v2_override_is_absolute_when_computed_delta_is_zero`：L89-L113
- `test_v2_override_does_not_add_computed_delta`：L115-L139

Previous responsibility：覆盖普通 effective KV state advancement 和 V1 divergence。

New responsibility：验证 V2 absolute override 在 `computed_delta == 0` 与 `computed_delta == 1` 时都得到 `E = K`，同时逻辑 `num_computed_tokens` 按 delta 独立推进。

Exact change：`ADD` 两个 CUDA focused tests。

Why required：防止 frozen V2 semantics 回退为 `K + delta` 或把 absolute override 错误嵌套到 delta 分支。

Requirement mapping：T2B absolute E override、logical/physical independence。

Classification：`TEST_ONLY`

Runtime mutation introduced：`NO`（测试只操作 isolated `RequestState` fixture）。

Compatibility impact：无生产兼容性变化。

Tests covering it：本文件；执行结果见 raw log。

## 3. 运行时 mutation audit

- KV payload：`YES`，仅通过唯一 production primitive `compact_paged_kv_triton_2d()`；不重新实现 gather/writeback。
- Worker `effective_kv_len`：`execute_model()` 内不直接写 persistent E；由 `post_update()` 使用 absolute override 提交。
- Worker BlockTable：payload 所有 layer 调用成功后通过 `append_block_ids(..., overwrite=True)` staged retained prefix。
- Scheduler canonical state：`NO`。
- BlockPool free/reuse/deferred free：`NO`。
- Rollback：`NO`。若首次 M4 write 后后续 layer kernel 失败，抛错且不发布成功 result；已发生的 payload mutation 不宣称可回滚。
- M4 invocation：`YES`，T2A validation 后、`ExecuteModelState` publication 前。

## 4. 测试与静态检查

- T2A/T2B focused + M5-T1 transport：`25 passed, 15 warnings`。
- M4 regression：`52 passed, 0 skipped, 15 warnings`。
- V1 reclaim regression：`14 passed, 15 warnings`。
- Effective KV state regression：`15 passed, 15 warnings`。
- `ruff check`（4 个 changed Python files）：`All checks passed!`。
- `py_compile`（4 个 changed Python files）：exit code `0`。
- `git diff --check`：exit code `0`。

## 5. 证据与限制

完整命令、时间、Git 状态、source anchors、测试 stdout 和 diff 摘要见：

`04-experiments/project1_kv_reclaim/raw/P1-M5-T2B-IMPLEMENTATION.log`

本 Slice 不包括：Scheduler ownership reconcile、safe free/reuse、deferred free、rollback、retention policy、PP/多 KV group/其他非 MVP 配置和性能测量。

Next Allowed Action：`WEB_REVIEW_CURRENT_SLICE`。

Proposed Next Action：由 Web review 决定是否进入 M5-T3；本 agent 不自动开始下一 Slice。
