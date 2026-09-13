# P1-M5-T3 实现记录：Scheduler Canonical Reconciliation

## Slice Summary

- 目的：将 Worker 已完成的 V2 compaction receipt 折回 Scheduler canonical ownership 与 physical `effective_kv_len`。
- Source HEAD before：`1f04cdca301fa72d20046a791672cf265bf9df09`
- Source HEAD after：同一 HEAD（未创建 commit）。
- 修改范围：Scheduler reconciliation、KV manager/coordinator dense-prefix detach helper、focused unit test。
- Runtime mutation：YES，仅 Scheduler canonical `req_to_blocks`、`num_cached_block`、request `effective_kv_len`；无 Worker/KV payload mutation。
- Ownership mutation：YES，removed suffix 转入 Scheduler `deferred_frees`；未立即调用 `BlockPool.free_blocks()`。

## Change Ledger

### CHG-01

- File：`vllm/v1/core/single_type_kv_cache_manager.py`
- Change type：ADD
- Symbol：`SingleTypeKVCacheManager.reconcile_compacted_blocks`
- Final lines：见 `P1-M5-T3-IMPLEMENT-01.log` 的 `rg`/`nl` evidence。
- Previous responsibility：仅有 V1 `reconcile_reclaimed_blocks`，会立即释放 removed blocks。
- New responsibility：V2 按 canonical row prefix 截断 ownership，返回 removed `KVCacheBlock`，不释放。
- Exact operation：增加 request/source/new block count validation；提交 `req_to_blocks[:new_num_blocks]`；限制 `num_cached_block`；返回 suffix。
- Why required：V2 result 不携带 Worker freed IDs，Scheduler 必须从 canonical row 推导 suffix。
- Requirement mapping：T3 canonical prefix reconcile、removed suffix release handoff、no immediate free。
- Classification：SOURCE_FACT_DRIVEN + DESIGN_DECISION
- Runtime mutation introduced：YES，Scheduler manager ownership metadata；生命周期为 `update_from_output` commit phase。
- Compatibility impact：V1 helper unchanged；prefix/spec/PP 等非 MVP 未扩展。
- Tests covering it：`tests/v1/core/test_compaction_reconciliation.py`。

### CHG-02

- File：`vllm/v1/core/kv_cache_coordinator.py`
- Change type：ADD
- Symbol：`KVCacheCoordinator.reconcile_compacted_blocks`
- Final lines：见 raw log。
- Previous responsibility：无 V2 dense-prefix forwarding API。
- New responsibility：single KV group guard 后转发到 `SingleTypeKVCacheManager`。
- Exact operation：新增 coordinator forwarding method。
- Why required：保持 `KVCacheManager` 对 manager 内部 ownership 的封装。
- Requirement mapping：T3 V2-specific ownership helper。
- Classification：SOURCE_FACT_DRIVEN
- Runtime mutation introduced：NO（仅转发；下层执行 metadata mutation）。
- Compatibility impact：MVP single KV group；multi-group 明确拒绝。
- Tests covering it：manager focused tests；Scheduler tests。

### CHG-03

- File：`vllm/v1/core/kv_cache_manager.py`
- Change type：ADD
- Symbol：`KVCacheManager.reconcile_compacted_blocks`
- Final lines：见 raw log。
- Previous responsibility：无 V2 dense-prefix API。
- New responsibility：向 coordinator 暴露 V2 canonical detach 接口。
- Exact operation：新增 thin forwarding method。
- Why required：Scheduler 不直接访问 coordinator/manager ownership internals。
- Requirement mapping：T3 ownership authority boundary。
- Classification：SOURCE_FACT_DRIVEN
- Runtime mutation introduced：NO（下层执行）。
- Compatibility impact：不改变现有 V1 `reconcile_reclaimed_blocks`。
- Tests covering it：Scheduler focused tests。

### CHG-04

- File：`vllm/v1/core/sched/scheduler.py`
- Change type：EXTEND / REPLACE
- Symbols：`_PreparedCompactionReconciliation`、`_prepare_compaction_reconciliations`、`_commit_compaction_reconciliations`、`update_from_output`、`schedule`。
- Final lines：见 raw log。
- Previous responsibility：只做 result duplicate-ID 检查；V2 result 未进行 canonical reconcile；V1/V2 同步分支可叠加。
- New responsibility：Phase 0/1 全量 admission+validation，Phase 2 全量 canonical commit；V2 E absolute `K`；V1/V2 same-step XOR fence；same-step preemption 清理 local plan 已保留。
- Exact operation：新增 source-E/page-count/row-count/new-shape/step_seq validation；所有 active result 准备完毕后统一调用 V2 helper；removed suffix 进入 `deferred_frees`；V2 request 跳过 V1/additive E branch；schedule materialization 处拒绝 V1+V2 coexistence。
- Why required：避免 half-commit，保持 logical `num_computed_tokens` 不变，并防止 `K+q`。
- Requirement mapping：T3 result admission、all-validation-before-commit、absolute E、V1/V2 XOR、deferred release。
- Classification：SOURCE_FACT_DRIVEN + DESIGN_DECISION
- Runtime mutation introduced：YES，Scheduler canonical ownership/E/deferred queue；无 Worker payload mutation。
- Compatibility impact：normal additive path与V1 reclaim path保持；MVP single KV group / no prefix/spec/PP>1。
- Tests covering it：现有 T1/T2/V1 tests + new helper tests；完整结果受当前无 NVIDIA driver 环境限制。

## Test / Evidence Summary

- `py_compile`：通过。
- `git diff --check`：通过。
- focused helper pytest：`4 passed, 9 warnings`。
- Scheduler lifecycle focused pytest：`8 failed, 3 passed`；当前环境无法从 NVIDIA driver/NVML 推断 `current_platform.device_type`，失败为环境 blocker，不是断言失败。
- `ruff`：仓库既有 UP038/E501 问题；本轮新增代码无独立 lint error。

## Scope / Limitations

- 未修改 Worker、M4 primitive、Scheduler free/reuse policy。
- 无 cross-component rollback；Worker payload 可能在后续 failure 前已部分改变。
- Scheduler reconcile 仅处理当前 P1 MVP single KV group。
- prefix caching、spec decode、PP>1、multi-group 未支持。
- 未声称性能收益。
