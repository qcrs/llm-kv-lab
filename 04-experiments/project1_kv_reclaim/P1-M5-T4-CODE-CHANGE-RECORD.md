# P1-M5-T4 — Unified Physical Block Release / Reuse Closure

## 1. Source identity

- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/v2-token-compaction-v026`
- Start HEAD：`1f04cdca301fa72d20046a791672cf265bf9df09`
- 本轮未创建 commit；T2/T3 及既有 Worker dirty changes 保留。
- 约束配置：MRV2、同步调度、PP=1、single KV group、KV connector/prefix/spec/CUDA Graph 关闭。

## 2. Problem statement

T3 已完成 V1/V2 canonical ownership detach，但 V2 在同步 MVP 中把 removed blocks 放入 `(sched_step_seq + 1, removed)`；由于 `defer_block_free=False` 时该序列不是 generic engine-step counter，V2 removed suffix 不能可靠进入 allocator free queue。T4 统一 result-time partial-transition release：manager/coordinator 只 detach，Scheduler 在 `update_from_output()` 完成 seam 统一调用 `BlockPool.free_blocks()`。

## 3. Upstream source facts

- upstream `update_from_output()` 位于 GPU write completion seam。
- upstream deferred-free protocol 仅在 `defer_block_free` 为真时推进 `processed_step_seq` 并 drain。
- `_free_request_blocks()`、`_free_cow_retained_blocks()` 仍保留 upstream deferred-free 生命周期。

## 4. V1 before / after

Before：`SingleTypeKVCacheManager.reconcile_reclaimed_blocks()` 更新 canonical row 后直接 `block_pool.free_blocks(reversed(removed_blocks))`，并返回 block IDs。

After：保留 validation、`req_to_blocks` replacement、`num_cached_block` clamp 和 ownership-before-release 顺序；改为返回 `list[KVCacheBlock]`。`Scheduler.update_from_output()` 提交 `effective_kv_len` 后调用 `_release_reconciled_blocks()`。

## 5. V2 before / after

Before：`reconcile_compacted_blocks()` detach suffix 后由 Scheduler append `(sched_step_seq + 1, removed)`。

After：V2 result-time commit 直接调用同一 `_release_reconciled_blocks()`；`deferred_frees` 不再接收 T4 partial-transition entry。

## 6. Why `sched_step_seq + 1` was invalid

在 `defer_block_free=False` 的同步 MVP 中，`sched_step_seq`/`processed_step_seq` 属于 conditional overlap/connector protocol，并不持续代表普通 Scheduler step。使用 `+1` 会留下无法满足的 fence，导致 V2 allocator capacity 不恢复。

## 7. Exact files / functions

- `vllm/v1/core/single_type_kv_cache_manager.py`：`reconcile_reclaimed_blocks()` 返回 detached objects；`reconcile_compacted_blocks()` 保持 detach-only。
- `vllm/v1/core/kv_cache_coordinator.py`：V1/V2 reconcile 返回 `list[KVCacheBlock]`。
- `vllm/v1/core/kv_cache_manager.py`：V1/V2 reconcile 返回 `list[KVCacheBlock]`。
- `vllm/v1/core/sched/scheduler.py`：`_release_reconciled_blocks()`；V1/V2 result-time wiring；deferred drain guard restoration。
- `tests/v1/core/test_compaction_reconciliation.py`：detach-only 与 Scheduler immediate release helper coverage。

## 8. Release authority after T4

`SingleTypeKVCacheManager` 决定哪些 blocks 不再 canonical-owned；`Scheduler` 决定 result-time release timing；`BlockPool` 执行实际 free/reuse。Removed block 在进入 free queue 前已从 canonical row 中移除。

## 9. Acceptance tests

- `test_reconcile_compacted_blocks_detaches_prefix_without_freeing`
- `test_reconcile_compacted_blocks_rejects_invalid_shape`
- `test_active_plan_without_result_is_rejected_before_mutation`
- `test_result_without_plan_is_rejected`
- `test_scheduler_releases_detached_blocks_without_deferred_fence`
- Existing `tests/v1/core/test_reclaim_ownership.py` remains the real Scheduler/BlockPool V1 release/reuse oracle but is `ENV_BLOCKED` here because NVML/device inference is unavailable.

## 10. Test output

- `tests/v1/core/test_compaction_reconciliation.py`: `7 passed`.
- `tests/v1/core/test_reclaim_ownership.py`: `4 failed` during fixture setup with `RuntimeError: Failed to infer device type`; no assertion failure observed (`ENV_BLOCKED`).
- `py_compile`: pass.
- `git diff --check`: pass after whitespace cleanup.
- `ruff`: existing repository lint noise in unrelated `UP038` sites and pre-existing scheduler docstring E501; no T4-specific lint error identified.

## 11. Remaining unsupported modes

async scheduling、PP>1、KV connector、prefix caching、spec decode、multi-group、offload、CUDA Graph 等未验证，不宣称支持。未运行 real-engine reuse smoke；本轮无性能 claim。

## 12. DESIGN_CONFLICT

NONE。当前同步 MVP 的 `update_from_output()` 是有效 completion seam；无需修改 Worker、T2/T3 contract 或 deferred-free protocol。

## 14. Change ledger

### CHG-01

- File：`vllm/v1/core/single_type_kv_cache_manager.py`
- Change type：`REPLACE`
- Symbol：`reconcile_reclaimed_blocks()`
- Old responsibility：更新 canonical row 后在 manager 内调用 `BlockPool.free_blocks()`，并返回 block IDs。
- Exact operation：删除 manager 内 free；返回 `removed_blocks: list[KVCacheBlock]`。
- New responsibility：只完成 validation、canonical ownership detach 和 `num_cached_block` clamp；把 release authority 交给 Scheduler。
- Why：统一 V1/V2 partial-transition ownership/release authority，满足 INV-T4-1/2/3。
- Classification：`SOURCE_FACT_DRIVEN`
- Runtime mutation：是；result-time 更新 `req_to_blocks`，owner 为 `SingleTypeKVCacheManager`。
- Compatibility：normal/V1/V2 保持 transition contract；prefix/spec/PP/async 未扩展。
- Tests：`test_reconcile_reclaimed_blocks_detaches_without_manager_free`。

### CHG-02

- File：`vllm/v1/core/kv_cache_coordinator.py`、`vllm/v1/core/kv_cache_manager.py`
- Change type：`EXTEND`
- Symbol：`reconcile_reclaimed_blocks()`、`reconcile_compacted_blocks()`
- Old responsibility：V1 返回 `list[int]`；V2 无统一 typed detached-object release channel。
- Exact operation：两层 API 改为返回 `list[KVCacheBlock]`，仅 topology validation/forwarding，不 free。
- New responsibility：向 Scheduler 传递真实 detached ownership objects。
- Why：为统一 release helper 提供对象级输入。
- Classification：`SOURCE_FACT_DRIVEN`
- Runtime mutation：否（仅返回通道与类型）。
- Compatibility：single-group MVP；未改变 Worker/T2/T3 contract。

### CHG-03

- File：`vllm/v1/core/sched/scheduler.py`
- Change type：`ADD`
- Symbol：`_release_reconciled_blocks()`
- Old responsibility：不存在 V1/V2 统一 partial-transition release helper。
- Exact operation：新增 helper，对非空 `removed` 调用 `block_pool.free_blocks(reversed(removed))`。
- New responsibility：Scheduler 在 result-time completion seam 决定并执行 detached block release。
- Why：闭合 detach→allocator free→reuse 链路，且不修改 `_free_request_blocks()` / `_free_cow_retained_blocks()`。
- Classification：`DESIGN_DECISION`
- Runtime mutation：是；BlockPool free queue 由 Scheduler 更新。
- Compatibility：仅同步 MVP partial transition；upstream deferred-free protocol unchanged。

### CHG-04

- File：`vllm/v1/core/sched/scheduler.py`
- Change type：`EXTEND`
- Symbol：`update_from_output()` V1/V2 result-time branches
- Old responsibility：V1 manager 内 free；V2 append `(sched_step_seq + 1, removed)`。
- Exact operation：V1 reconcile 返回值后调用 helper；V2 canonical commit 后调用同 helper，删除 P1-specific deferred fence。
- New responsibility：V1/V2 均为 ownership commit → E commit → Scheduler release。
- Why：修复 V2 synchronous MVP allocator reclaim gap；保持 validation-before-commit。
- Classification：`SOURCE_FACT_DRIVEN`
- Runtime mutation：是；`req_to_blocks`、`effective_kv_len`、free queue 在 `update_from_output()` result-time 变更。
- Compatibility：normal path unchanged；V1/V2 partial transition；prefix/spec/PP/async unsupported。

### CHG-05

- File：`vllm/v1/core/sched/scheduler.py`
- Change type：`REPLACE`
- Symbol：deferred-free drain guard in `update_from_output()`
- Old responsibility：无论 `defer_block_free` 是否启用都调用 drain，并在 false 时不推进 processed seq。
- Exact operation：恢复 upstream 分支：仅当 `defer_block_free and total_num_scheduled_tokens > 0` 时推进 `processed_step_seq` 并 drain。
- New responsibility：保持 conditional overlap/connector protocol，不将其当 generic step counter。
- Why：满足 INV-T4-7，避免 P1 partial release 污染 upstream lifecycle。
- Classification：`SOURCE_FACT_DRIVEN`
- Runtime mutation：是；`processed_step_seq` / deferred queue 仅 conditional protocol owner 为 Scheduler。
- Compatibility：normal/V1/V2 partial release unchanged；async/connector 仍不宣称支持。

### CHG-06

- File：`tests/v1/core/test_compaction_reconciliation.py`
- Change type：`TEST_ONLY`
- Symbol：detach/release focused tests
- Old responsibility：仅覆盖 V2 detach、shape validation 和 T3 admission。
- Exact operation：增加 manager-level V1 detach-only test与 Scheduler helper reverse-order/no-deferred-fence test。
- New responsibility：覆盖 manager 不 immediate-free 及 Scheduler release channel。
- Classification：`TEST_ONLY`
- Runtime mutation：否。
- Tests：7→8 tests in this file.

## 15. Required audit answers

- V1 before/after：manager 从 detach+free 改为 detach-only；Scheduler result-time free。
- V2 before/after：`(sched_step_seq + 1, removed)` 删除；Scheduler result-time immediate release（同步 MVP）。
- `sched_step_seq + 1` 不能作为 V2 fence：`defer_block_free=False` 时序列不推进，不是 generic engine-step counter。
- Release authority：manager 决定 WHICH，Scheduler 决定 WHEN，BlockPool 执行 free/reuse。
- Deferred-free protocol：upstream `_free_request_blocks()`、`_free_cow_retained_blocks()`、`_drain_deferred_frees()` 及 conditional guard 保持。
- DESIGN_CONFLICT：`NONE`。
