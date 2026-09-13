# P1-M5-T3-HARDEN-01 Scheduler Reconciliation Acceptance Hardening

## Slice Summary

- 基线 HEAD：`1f04cdca301fa72d20046a791672cf265bf9df09`（未创建 commit）。
- H1：same-step preemption rollback 现在清理 local `compaction_plans`。
- H2：active current-step Plan 必须有对应 Result；Result→Plan admission 保持。
- H3：新增无 GPU platform 初始化依赖的 Scheduler-admission focused tests。
- 未修改 Worker、M4 primitive 或 deferred-free `sched_step_seq + 1` fence。

### CHG-01

File: `vllm/v1/core/sched/scheduler.py`
Change type: `EXTEND` / `REPLACE`
Symbol: `Scheduler.schedule` rollback；`_prepare_compaction_reconciliations`
Old line range: pre-hardening anchors见 raw log。
Final line range: rollback L617-L621；admission L1724-L1730。

Previous code: rollback 清理 scheduled token、new blocks、V1 transition 等，但遗漏 current-step `compaction_plans`；admission 只验证 Result→Plan。
Exact operation: 新增 `compaction_plans.pop(preempted_req_id, None)`；建立 `plan_ids`/`result_ids`，对仍 active 且缺 Result 的 Plan 抛出 `ValueError`。
New responsibility: 防止 stale V2 plan 到达 Worker；防止 active Plan 在无 receipt 时进入 normal additive E path。
Why: acceptance findings F1/F3。
Requirement mapping: H1/H2。
Classification: `SOURCE_FACT_DRIVEN` + `DESIGN_DECISION`。
Runtime mutation introduced: YES，仅 current-step map cleanup；H2 validation 在任何 canonical commit 前执行。
Compatibility: normal/V1 unchanged；V2 admission strengthened；prefix/spec/PP/async unchanged and out of MVP scope。
Tests: `test_active_plan_without_result_is_rejected_before_mutation`；existing transport tests。

### CHG-02

File: `tests/v1/core/test_compaction_reconciliation.py`
Change type: `TEST_ONLY` / `EXTEND`
Symbols: `_scheduler_for_admission`、`test_active_plan_without_result_is_rejected_before_mutation`、`test_result_without_plan_is_rejected`。
Final line range: lines 1-75。
Previous responsibility: 仅有 manager dense-prefix helper tests。
Exact operation: 增加轻量 scheduler admission fixture 与 H2 missing-plan/missing-result tests，同时保留 helper tests。
New responsibility: 验证 Result↔Plan admission 和 active Plan completeness 在 mutation 前失败。
Classification: `TEST_ONLY`。
Runtime mutation introduced: NO。
Compatibility: 仅测试，不改变生产状态。

## Verification

- focused hardening tests：`6 passed, 25 warnings`。
- py_compile：通过。
- `git diff --check`：通过。
- Scheduler lifecycle tests仍受 NVIDIA/NVML device detection 环境 blocker 影响，未伪造 PASS。
- ruff：新测试 import/line issues 已修正；仓库既有 `scheduler.py` E501 与其他旧 UP038 findings 仍存在。

## Limitations

no cross-component rollback；single KV group MVP；prefix caching/spec decode/PP>1 unsupported；no generalized V1+V2 composition；deferred-free `+1` unchanged；safe-free/reuse remains M5-T4；no performance claim。
