# P1-M5-T2B-HARDEN-01 代码变更记录

## 1. Slice 概要

- 目的：硬化 T2B 的多 request transaction ordering，并清理已经失效的中间 carrier。
- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/v2-token-compaction-v026`
- Start HEAD：`1f04cdca301fa72d20046a791672cf265bf9df09`
- End HEAD：`1f04cdca301fa72d20046a791672cf265bf9df09`（未创建 commit）
- 生产 KV primitive 未修改；Scheduler ownership/free 未修改。
- 本轮没有进入 M5-T3。

## 2. Hardened 事项

- `DONE`：所有 request × 所有 layer 的 payload compaction 完成后，才进入 BlockTable/override/result metadata phase。
- `DONE`：删除 `ExecuteModelState.prepared_compactions` dead carrier；`_PreparedCompaction` 只在 `execute_model()` 内完成生命周期。
- `DONE`：删除 exact page-count guard 后的 redundant capacity-only guard。
- `DONE`：`step_seq` 从 `CompactionPlanData` 传播到 `_PreparedCompaction` 和 `CompactionResultData`。
- `NOT_DONE`：可选 `batch_idx` 命名清理；本轮未做无必要的机械重命名。

## 3. 逐文件审计

### CHG-01

File：`vllm/v1/worker/gpu/model_runner.py`

Change type：`MODIFIED`

Final symbols / line anchors：

- `_PreparedCompaction`：L129-L140
- `_prepare_v2_compactions`：L1812-L1978
- `_execute_v2_compactions`：L1981-L2042
- `execute_model` T2B seam：L2265-L2289
- `ExecuteModelState`：L2550-L2563

Previous responsibility：T2A prepared descriptor 会被建立；T2B 按 request 完成 payload、metadata 和 result；`ExecuteModelState` 还保存已消费的 prepared descriptor。

New responsibility：先执行所有 request × 所有 KV layer 的 M4 payload phase；全部成功后再执行 metadata/result phase。`step_seq` 被保留到 result；prepared descriptor 不跨 `execute_model()`/`sample_tokens()`。

Exact change：

- `ADD` `_PreparedCompaction.step_seq`。
- `DELETE` exact-page guard 后的 `source_num_blocks * block_size < source_effective_kv_len` capacity-only guard。
- `REPLACE` `_execute_v2_compactions()` 为显式两阶段：phase 1 收集 validated tensors，phase 2 staged overwrite、override 和 result 构造。
- `EXTEND` result 构造以 `prepared.step_seq` 填充 `CompactionResultData.step_seq`。
- `DELETE` `ExecuteModelState.prepared_compactions` 字段及其构造参数。

Why required：避免 request A metadata 已 staged 后 request B payload failure；避免无效 carrier 延长 descriptor 生命周期；保证 P1 MVP exact page-count invariant 与 frozen M4 input contract 一致。

Requirement mapping：HARDEN-01A/B/C/D。

Classification：`SOURCE_FACT_DRIVEN` + `DESIGN_DECISION`

Runtime mutation introduced：`YES`，只保留批准的 T2B payload compaction、staged BlockTable overwrite 和 post-update override；没有新增 ownership mutation。

Compatibility impact：M4 primitive、V1 reclaim contract、Scheduler canonical state 均保持不变；现有 MVP 之外配置仍不支持。

Tests covering it：`tests/v1/worker/test_gpu_compaction_preparation.py`。

### CHG-02

File：`tests/v1/worker/test_gpu_compaction_preparation.py`

Change type：`MODIFIED`

Final symbols / line anchors：

- `_plan`：L40-L64
- `test_execute_v2_compaction_compacts_all_layers_and_stages_next_state`：L182-L229
- `test_execute_v2_compaction_does_not_publish_success_after_layer_failure`：L232-L264
- `test_execute_v2_compaction_payload_failure_does_not_stage_prior_request`：L267-L315

Previous responsibility：覆盖 T2A fence 和单 request T2B fake execution。

New responsibility：额外验证 `step_seq` propagation，以及多 request payload failure 时 prior request 不进入 metadata phase、`num_blocks` 不变化；不对 payload rollback 作断言。

Exact change：

- `EXTEND` `_plan` 支持 `step_seq`。
- `ADD` success test 的 `step_seq == 17` 断言。
- `ADD` multi-request failure/no-metadata-commit regression。

Why required：直接覆盖 HARDEN-01A/D 的 failure boundary 和 result identity。

Requirement mapping：multi-request ordering、no success result after payload failure、optional step sequence。

Classification：`TEST_ONLY`

Runtime mutation introduced：`NO`（fake primitive 只修改 isolated test fixture）。

Compatibility impact：增加 focused regression，不改变生产行为以外的测试契约。

Tests covering it：本文件。

### CHG-03

File：`04-experiments/project1_kv_reclaim/P1-M5-T2B-HARDEN-01-CODE-CHANGE-RECORD.md`

Change type：`ADDED`

Final symbol：本文档。

Previous responsibility：无。

New responsibility：记录本 Slice 每个 production/test 修改、设计映射、line anchors、限制和验证。

Classification：`EVIDENCE_ONLY`

Runtime mutation introduced：`NO`

### CHG-04

File：`04-experiments/project1_kv_reclaim/raw/P1-M5-T2B-HARDEN-01.log`

Change type：`ADDED`

Final symbol：raw evidence log。

Previous responsibility：无。

New responsibility：保存起始/结束身份、source anchors、命令和原始测试结果。

Classification：`EVIDENCE_ONLY`

Runtime mutation introduced：`NO`

## 4. Transaction / lifetime 结论

```text
Plan
  → _prepare_v2_compactions()
  → _PreparedCompaction
  → _execute_v2_compactions()
  → prepared descriptor lifetime ends inside execute_model()

Execute
  → effective_kv_len_override_valid/value
  → ExecuteModelState
  → post_update()

Execute
  → CompactionResultData
  → ExecuteModelState
  → ModelRunnerOutput
```

- 后续 payload runtime failure 可能导致 earlier layer/request payload 已变化；MVP 不实现 rollback。
- 所有 payload 操作成功前不会调用 `append_block_ids(..., overwrite=True)`。
- payload failure 后不会返回成功 result。
- Scheduler canonical state 不变；没有 free/reuse/deferred free。

## 5. 验证结果

- T2A/T2B focused + M5-T1 transport：`26 passed, 15 warnings`。
- M4 focused regression：`52 passed, 0 skipped, 15 warnings`。
- V1/effective-state 组合：当前环境 NVIDIA driver 不可用，`29 skipped`；该结果不作为 CUDA correctness 通过证据。
- `ruff check`：`All checks passed!`。
- `py_compile`：exit code `0`。
- `git diff --check`：exit code `0`。

## 6. 已知限制

- no rollback MVP；
- no Scheduler reconcile yet；
- no safe free/reuse/deferred free；
- PP>1 not supported by current P1 MVP；
- exact-page invariant 仅是 P1 current MVP invariant，不是 universal vLLM invariant；
- no performance claim。

Next Allowed Action：`WEB_REVIEW_CURRENT_SLICE`。
