# P1-M5-T1 代码变更记录

## Slice Summary

- Purpose：在真实 vLLM MRV2 源码中建立 V2 `CompactionPlanData` / `CompactionResultData` 契约和双向 transport seam；本轮不执行 compaction。
- Source HEAD before：`9ea5d804ca6cc3a7a7212d83d626acb1ddd023ea`
- Source HEAD after：同一 HEAD（本轮未创建 commit）。
- Branch：`p1/v2-token-compaction-v026`
- Files Added：`04-experiments/project1_kv_reclaim/records/P1-M5-T1-SOURCE-MAP.md`、本记录文件。
- Files Modified：4 个 production Python 文件、1 个 test 文件。
- Files Deleted：NONE
- Production LOC added/deleted：`109/2`（`git diff --numstat`；其中 2 行为 import replacement）。
- Test LOC added/deleted：`91/2`。
- Architecture Deviations：NONE。
- Runtime Mutation Introduced? **NO**
- KV Payload Mutation Introduced? **NO**
- Ownership Mutation Introduced? **NO**

## 1. Source Fact 与实现选择

- SOURCE_FACT：`SchedulerOutput` 定义于 `vllm/v1/core/sched/output.py:208`，由 `Scheduler.schedule()` 在 `scheduler.py:1188` 构造。
- SOURCE_FACT：MRV2 `GPUModelRunner.execute_model()` 在 `model_runner.py:2004` 调用 model forward，随后在 `:2022` 发布 `ExecuteModelState`。
- SOURCE_FACT：`ModelRunnerOutput` 是 `vllm/v1/outputs.py:234` 的序列化 Worker→Scheduler carrier；EngineCore 在 `core.py:601` 和 `:703` 调用 `Scheduler.update_from_output()`。
- SOURCE_FACT：`ReclaimTransitionData` 位于 `output.py:114`，通过 V1 `CachedRequestData.reclaim_transitions` 传输。
- APPROVED_DESIGN：V2 plan 是 Scheduler/control plane → Worker 的 post-forward plan；V2 result 是 Worker → Scheduler 的独立 post-forward result；V1 contract 保持 V1-only。
- IMPLEMENTATION_DECISION：plan 使用 `dict[str, CompactionPlanData]`，因为现有 SchedulerOutput 以 request-indexed dict 组织 request payload；result 使用 `list[CompactionResultData]`，因为现有 ModelRunnerOutput 是 per-step serialized result carrier，且不将 allocator-authoritative freed IDs 放入结果。
- IMPLEMENTATION_DECISION：复用现有 runtime 的 step ordering 作为默认关联保证，同时保留可选 `step_seq` 字段以便显式 producer 在需要时携带 fence；没有新增 CUDA event、线程或 lifetime manager。

## 2. Change Entries

### CHG-01

File：`vllm/v1/core/sched/output.py`

Symbol：`CompactionPlanData`、`CompactionResultData`

Change Type：ADD

Pre-change Lines：L112-L118 之后无 V2 contract。

Post-change Lines：`CompactionPlanData` L120-L135；`CompactionResultData` L137-L151。

Before：只有 V1 `ReclaimTransitionData`，表达 pre-forward whole-block command。

After：新增两个 `@dataclass(frozen=True)`。Plan 字段为 `request_id`、`keep_member_indices`、`expected_source_effective_kv_len`、`expected_source_num_blocks`、可选 `step_seq`；Result 字段为 `request_id`、两个 expected source 字段、`new_effective_kv_len`、`new_num_blocks`、可选 `step_seq`。

Exact Change：ADD 两个独立数据类型及 docstring；未修改或扩展 `ReclaimTransitionData`。

Reason：建立冻结架构要求的 V2 双向语义边界。

Design Mapping：APPROVED_DESIGN 的“separate V2 plan/result contract”；Plan source extent 明确是 POST-FORWARD extent（例如 32 + 16 = 48），不是 forward-entry E；Result 不携带 freed IDs authority。

Behavioral Effect：对象可被 Python dataclass 序列化/传输；不触发任何 KV 或 ownership 操作。

Non-Effect / Out of Scope：不做 source-E runtime validation、不改 BlockTable、不改 E、不 free、不调用 M4 primitive。

Tests：`test_v2_compaction_contract_fields_and_source_semantics`。

Raw Evidence：T1 pytest `7 passed`；`py_compile` rc=0。

### CHG-02

File：`vllm/v1/core/sched/output.py`

Symbol：`SchedulerOutput.compaction_plans` / `SchedulerOutput.make_empty`

Change Type：EXTEND

Pre-change Lines：`SchedulerOutput` L208-L291，末尾字段为 `num_spec_tokens_to_schedule` L277；`make_empty()` 无 V2 字段。

Post-change Lines：字段 L313-L315；`make_empty()` L318-L330（default factory 自动提供空 dict）。

Before：SchedulerOutput 不能携带 V2 plan。

After：增加 `compaction_plans: dict[str, CompactionPlanData] = field(default_factory=dict)`。

Exact Change：ADD optional request-indexed carrier；未重写已有字段或 constructor 参数。

Reason：让零或多个计划沿现有 SchedulerOutput transport 传播，并保持 normal path identity。

Design Mapping：APPROVED_DESIGN 的 Scheduler → SchedulerOutput → Worker plan direction。

Behavioral Effect：所有既有构造调用继续使用空 dict default；显式 plan 可按 request ID 访问。

Non-Effect / Out of Scope：不 materialize CUDA tensor；不执行或验证 physical source state。

Tests：`test_v2_plan_scheduler_output_to_worker_preserves_request_mapping`、既有 V1 transport tests。

Raw Evidence：T1 pytest `7 passed`。

### CHG-03

File：`vllm/v1/core/sched/scheduler.py`

Symbol：`Scheduler._prepared_compaction_plans`、`Scheduler.schedule()`、`Scheduler._set_prepared_compaction_plan()`

Change Type：ADD / EXTEND

Pre-change Lines：`_prepared_reclaim_plans` L201；`schedule()` request locals L459-L464、scheduled request commit L642-L655、`SchedulerOutput(...)` L1188-L1203；无 V2 prepared ingress。

Post-change Lines：state L201-L202；locals L459-L462；one-shot materialization L646-L652；output argument L1203；helper L1450-L1465。

Before：只有 V1 prepared reclaim plan ingress/materialization。

After：增加 `_prepared_compaction_plans`；显式 producer 可调用 `_set_prepared_compaction_plan()`；只有 request 实际进入当前 scheduled set 时才 pop 并放入 `SchedulerOutput.compaction_plans`；preempted request 清理对应 pending plan。

Exact Change：ADD V2 map/helper；EXTEND schedule locals、scheduled request branch 和 output constructor；V1 `_prepared_reclaim_plans` 语句未删除或改义。

Reason：提供 policy-neutral、schedule-time binding seam，而不是在 Scheduler 内置 retention policy。

Design Mapping：APPROVED_DESIGN 的 external explicit plan producer、Scheduler-side binding、post-forward source semantics；step fence 只作为可选数据字段，不引入独立 lifetime protocol。

Behavioral Effect：显式 plan 可从 Scheduler 进入本轮 output；未调度 request 的 plan 不会误传；normal output 仍为空 mapping。

Non-Effect / Out of Scope：不检查 actual E，不改 request E、`req_to_blocks`、BlockPool、deferred free 或 V1 transition。

Tests：`test_v2_prepared_plan_is_materialized_only_for_scheduled_request`；既有 `test_reclaim_transport.py` V1 tests。

Raw Evidence：T1 transport/V1 suite `7 passed`。

### CHG-04

File：`vllm/v1/outputs.py`

Symbol：`ModelRunnerOutput.compaction_results`

Change Type：EXTEND

Pre-change Lines：`ModelRunnerOutput` L234-L281，无 V2 result field。

Post-change Lines：`compaction_results` L283-L286。

Before：ModelRunnerOutput 只携带 sampled/logprob/connector 等既有结果。

After：增加 `list[CompactionResultData]` default-empty carrier；注释明确 freed IDs 不是 authority。

Exact Change：ADD optional per-step result list；保留 `with_kv_conn_output_only()` 及其他字段。

Reason：复用现有 serialized Worker→Scheduler result boundary，不改 EngineCore execution path。

Design Mapping：APPROVED_DESIGN 的独立 Worker→Scheduler `CompactionResultData` 与 canonical Scheduler ownership。

Behavioral Effect：dummy/真实 ModelRunnerOutput 都可携带 V2 result；默认路径无结果。

Non-Effect / Out of Scope：本 Slice 不由 Worker 自动生成 result，不做 reconcile/free。

Tests：`test_v2_result_model_runner_output_to_scheduler_carrier`。

Raw Evidence：T1 pytest `7 passed`。

### CHG-05

File：`vllm/v1/worker/gpu/model_runner.py`

Symbol：`GPUModelRunner.extract_compaction_plans`、`GPUModelRunner.execute_model`、`ExecuteModelState`

Change Type：ADD / EXTEND

Pre-change Lines：`execute_model()` L1773-L1800；`ExecuteModelState` L2287-L2294 无 V2 plan。

Post-change Lines：helper L1776-L1789；read seam L1791-L1800；state publication field L2022-L2030；`ExecuteModelState.compaction_plans` L2294。

Before：execute_model 只消费既有 SchedulerOutput，并在 forward 后发布不含 plan 的 ExecuteModelState。

After：新增只读 `extract_compaction_plans()`，校验 dict、value type 和 key/request identity；execute_model 入口读取并把同一 mapping 保存到 ExecuteModelState。

Exact Change：ADD structural extraction helper；EXTEND execute_model 和 NamedTuple；没有插入 compaction call。

Reason：让 Worker 能看到并保存 plan，给后续 M5-T2 post-forward seam 留出明确 publication boundary。

Design Mapping：APPROVED_DESIGN 的 GPUModelRunner post-forward consumer；当前只 transport/read/store，ExecuteModelState 仍在 forward 完成后发布。

Behavioral Effect：错误 carrier shape fail-closed；合法 plan 不丢失、不重排。

Non-Effect / Out of Scope：没有 `compact_paged_kv_triton_2d()`、Triton invocation、KV payload/row/E mutation 或 result generation。

Tests：`test_v2_plan_scheduler_output_to_worker_preserves_request_mapping`。

Raw Evidence：T1 pytest `7 passed`；M4 suite `52 passed`。

### CHG-06

File：`vllm/v1/core/sched/scheduler.py`

Symbol：`Scheduler._validate_compaction_results` / `Scheduler.update_from_output`

Change Type：ADD / EXTEND

Pre-change Lines：`update_from_output()` L1702-L1720（修改前 L1661-L1679）。

Post-change Lines：helper L1691-L1700；调用 L1702-L1707。

Before：Scheduler 未读取 ModelRunnerOutput 的 V2 result carrier。

After：在既有 result consumer 入口只检查 result request ID 不重复，然后继续既有逻辑。

Exact Change：ADD structural no-op validator；EXTEND update entry；未加入 canonical reconcile。

Reason：建立 Worker result → Scheduler receiver seam，同时保持本 Slice 的 transport-only 边界。

Design Mapping：APPROVED_DESIGN 的 Worker post-forward result publication；canonical row/E/free 延后到 M5-T3。

Behavioral Effect：重复 result request ID fail-closed；合法 result 可穿过既有 carrier，不改变 scheduler state。

Non-Effect / Out of Scope：不验证 actual source E，不修改 `request.effective_kv_len`、`req_to_blocks`、BlockPool 或 deferred frees。

Tests：`test_v2_result_model_runner_output_to_scheduler_carrier`；V1/M4 regressions。

Raw Evidence：V1 Worker regression `14 passed`；M4 `52 passed`。

### CHG-07

File：`tests/v1/core/test_reclaim_transport.py`

Symbol：V2 transport tests

Change Type：ADD / REPLACE

Pre-change Lines：已有 4 个 V1 transport tests，文件 L1-L124。

Post-change Lines：新增 tests L18-L98；原 V1 tests 从 L101 起，imports L4-L13。

Before：仅覆盖 V1 `ReclaimTransitionData` transport。

After：新增 contract field/source semantics、SchedulerOutput→Worker mapping、multiple requests (A/C, B absent)、result carrier→Scheduler validation、Scheduler prepared plan materialization 5 个测试。

Exact Change：ADD 5 tests；REPLACE import block ordering；既有 V1 assertions 保留。

Reason：覆盖 T1 A–F transport correctness 和 V1 isolation。

Design Mapping：仅验证 data shape/request identity，不模拟 M5-T2 payload correctness。

Behavioral Effect：测试失败可直接定位 field loss、request cross-wire、duplicate result 或 V1 regression。

Non-Effect / Out of Scope：不执行真实 KV compaction；GPU unavailable 时既有 fixture tests 仍反映环境限制。

Tests：本文件 `7 passed`。

Raw Evidence：`CUDA_VISIBLE_DEVICES=1 ... pytest -q tests/v1/core/test_reclaim_transport.py`。

## 3. 验证记录

- Source identity：`git branch --show-current`=`p1/v2-token-compaction-v026`；HEAD=`9ea5d804ca6cc3a7a7212d83d626acb1ddd023ea`；开始时工作树 clean。
- Import identity（`PYTHONPATH=$PWD`）：`vllm.__file__` 指向 implementation worktree 的 `vllm/__init__.py`；`output.py` 与 `model_runner.py` 同一 worktree。
- `CUDA_VISIBLE_DEVICES=1 ... -m pytest -q tests/v1/core/test_reclaim_transport.py`：`7 passed, 15 warnings`。
- `CUDA_VISIBLE_DEVICES=1 ... -m pytest -q tests/v1/worker/test_gpu_reclaim_commit.py`：`14 passed, 15 warnings`。
- `CUDA_VISIBLE_DEVICES=1 ... -m pytest -q tests/v1/worker/test_gpu_kv_compaction.py`：`52 passed, 15 warnings`，0 skip。
- `/home/qcrs/learning/llm-kv-lab/.venv/bin/python -m py_compile <5 changed Python files>`：PASS，rc=0。
- `/home/qcrs/learning/llm-kv-lab/.venv/bin/python -m ruff check <5 changed Python files>`：无法执行，环境报 `No module named ruff`。
- 备用 `/opt/miniconda/bin/ruff check <5 changed Python files>`：仅报告三个 pre-existing 超长行（`output.py:178`、`output.py:194`、`scheduler.py:1443`）；本 Slice 新增行无 ruff error。
- `git diff --check`：PASS，rc=0。

## 4. Runtime Mutation Audit

- KV payload mutation：NO
- Worker BlockTable/row mutation：NO
- Worker `effective_kv_len` mutation：NO
- Scheduler request/canonical ownership mutation：NO
- BlockPool free / deferred-free / safe-free change：NO
- Triton/M4 compaction invocation：NO

## 5. Exact Change Manifest

MODIFIED:

- `worktrees/p1-vllm-reclaim/vllm/v1/core/sched/output.py` — production；ADD V2 dataclasses，EXTEND SchedulerOutput。
- `worktrees/p1-vllm-reclaim/vllm/v1/core/sched/scheduler.py` — production；ADD prepared plan/result validation，EXTEND schedule/update seams。
- `worktrees/p1-vllm-reclaim/vllm/v1/outputs.py` — production；EXTEND ModelRunnerOutput result carrier。
- `worktrees/p1-vllm-reclaim/vllm/v1/worker/gpu/model_runner.py` — production；ADD plan extraction，EXTEND ExecuteModelState。
- `worktrees/p1-vllm-reclaim/tests/v1/core/test_reclaim_transport.py` — test；ADD V2 contract/transport tests，REPLACE import ordering。

ADDED:

- `04-experiments/project1_kv_reclaim/records/P1-M5-T1-SOURCE-MAP.md` — docs/evidence；真实 pre-change source map。
- `04-experiments/project1_kv_reclaim/records/P1-M5-T1-CODE-CHANGE-RECORD.md` — docs/evidence；本记录。

DELETED:

- NONE

## 6. Closure Boundary

本记录证明 T1 的 plan/result contract、双向 transport、request identity isolation、V1 isolation 和静态/回归证据；不证明 M5-T2 的 post-forward physical transition、source fence runtime validation、BlockTable/E commit、Scheduler canonical reconcile 或 safe-free。下一动作固定为 `WEB_REVIEW_CURRENT_SLICE`；停止在 M5-T1，不进入 M5-T2。
