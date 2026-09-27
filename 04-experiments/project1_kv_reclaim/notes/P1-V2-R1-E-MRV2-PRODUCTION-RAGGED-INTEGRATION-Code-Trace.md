# P1-V2-R1-E MRV2 Production Ragged Identity Reconciliation Code Trace

## 1. Slice 状态

- 当前状态：`PASS_PENDING_WEB_REVIEW`；不创建 commit，不进入下一 Slice。
- Worktree：`worktrees/p1-vllm-reclaim`
- Branch：`p1/v2-token-compaction-v026`
- Start HEAD：`7ed376375479c3de2fc392cb8356d35aa695a7dc`
- 本轮重点：Scheduler preempt → resume → MRV2 Full Snapshot → 新 persistent `req_idx` materialization，以及真实 A100 MRV2 Ragged 主链。

## 2. 修改追踪

| File / Symbol | Change Type | Before | After | Why / Invariant | Tests / Evidence | Status |
|---|---|---|---|---|---|---|
| `vllm/v1/worker/gpu/model_runner.py` / `_apply_ragged_kv_updates` | MODIFY | Ragged transport 不在 MRV2 lifecycle | 按当前 `req_id_to_index` materialize Snapshot/Delta；缺 transport、slot、identity 或 scheduled state 时 fail closed | Scheduler 是 ownership authority；Snapshot 不依赖旧 row | `test_gpu_model_runner_v2_ragged.py`；A100 logs | PASS |
| `vllm/v1/worker/gpu/model_runner.py` / `_remove_request` | MODIFY | 删除请求不清理 Ragged mirror | 先取旧 persistent `req_idx`，再清理 Worker mirror row | preempt/finish 后旧 row 不得污染 slot reuse | resume lifecycle test | PASS |
| `vllm/v1/worker/gpu/model_runner.py` / `_prepare_ragged_step` | MODIFY | 无 MRV2 Ragged step producer | 使用 `InputBatch.idx_mapping_np` gather persistent rows，构造当前 step views | 禁止 `range(num_reqs)`；StepViews 仅为 per-step derived state | non-contiguous slot test | PASS |
| `vllm/v1/worker/gpu/model_runner.py` / `execute_model`, `sample_tokens` | MODIFY | 无 MRV2 Ragged success-tail | 保存 forward-entry source descriptor，并在 postprocess 成功后 commit Worker E | Worker commit 使用 req_idx/version/source E/target E，不重新从 SchedulerOutput 推导 | focused tests | PASS |
| `vllm/v1/worker/gpu/warmup.py` / `warmup_kernels` | MODIFY | generic warmup 生成无 Ragged transport 的 synthetic `SchedulerOutput`，启动时触发 fail closed | Ragged active 时跳过 generic V2 warmup，不创建 fake Ragged allocator/mirror state；真实请求负责执行 Ragged path | profile/startup 不伪造 physical ownership | `04-mrv2-startup-port18082.log`, `...port18083.log` | PASS |
| `vllm/v1/worker/gpu_worker.py` / runner selection | REVERT | `page_group_size != None` 强制 MRV1 | Ragged 与 Dense 均按 `use_v2_model_runner` 进入 MRV2 | 禁止 Production Ragged fallback 到 MRV1 | A100 startup log | PASS |
| `vllm/config/vllm.py` / Ragged validation | ADD | Ragged + MRV1 可静默运行 | Ragged active 且未启用 MRV2 时抛出明确错误 | fail closed，不隐藏架构错误 | config test | PASS |
| `vllm/v1/worker/gpu_model_runner.py` | REVERT | E 阶段 MRV1 Ragged lifecycle、reshape、ForwardContext wiring | 删除 E 专属 MRV1 production path，保留原有 MRV1 非本 Slice 内容 | MRV2 是唯一 Production Ragged runner | diff / focused tests | PASS |
| `vllm/v1/worker/gpu/ragged_kv_state.py` / `reindex` | DELETE | MRV1 condense/reindex API | 删除无调用者的 `reindex()` | MRV2 mirror key 是 persistent `req_idx`，normal lifecycle 不 reindex | state tests | PASS |
| `tests/v1/worker/test_gpu_model_runner_v2_ragged.py` | ADD | 无 MRV2 Ragged runner tests | Snapshot、Delta、gather、commit、fail-closed、preempt/resume/slot-change 测试 | 覆盖 frozen lifecycle contract | `03-focused-tests-resume.txt` | PASS |
| `tests/v1/core/test_scheduler.py` / priority helper | MODIFY | helper 只能构造 Dense spec | 增加可选 `kv_cache_spec`，供 Ragged preemption test 复用 | 不新增第二套 Scheduler 测试架构 | resume test | PASS |

## 3. 四条运行链

### Chain A — Scheduler

`Scheduler.schedule()` → `KVCacheManager.allocate_slots()` → `RaggedAttentionManager.plan_capacity()` → `apply_capacity_plan()` → `RaggedKVUpdateData`。

在 `scheduler.py:1145-1196`，MRV2 将 `scheduled_resumed_reqs` 合并到 `scheduled_new_reqs`；new/resumed request 导出 Full Snapshot，running request 才导出 Allocation Delta。Scheduler canonical `effective_lens/page_rows/state_version` 未由 Worker 改写。

### Chain B — MRV2 Worker

`SchedulerOutput` → `RequestState.add_requests()` 获得当前 `req_idx` → `_apply_ragged_kv_updates()` 通过 `req_id_to_index` materialize → `RaggedWorkerPhysicalState` → `InputBatch.idx_mapping_np` → `gather()` → `RaggedClusterStepView` → `build_ragged_step_views()`。

Preempt 时旧 row 清理；resume 时 request 重新 add，Snapshot 完整替换新 row。测试显式使用 A 的旧 slot `7`、resume 后新 slot `2`，并验证旧 row 全零。

### Chain C — Execution

MRV2 `execute_model()` → `set_forward_context(ragged_step_views, ragged_layer_indices)` → `Attention.forward()` → `unified_kv_cache_update()` → `ragged_kv_cache_update()` → `unified_attention_with_output()` → `ragged_attention_forward()` → FlashAttention。

A100 instrumented evidence 观察到 `Ragged KV write entered` 与 `Ragged attention forward entered`；instrumentation 已在证据采集后删除。

### Chain D — Commit

forward-entry source descriptor（`req_idx/state_version/source E/target E`）→ model forward → sampling/postprocess success → Worker E commit → `ModelRunnerOutput` → `Scheduler.update_from_output()` → Scheduler canonical E commit。

当前 Worker success commit 已使用 forward-entry capture；没有在 sample/postprocess 阶段重新从 `SchedulerOutput` 推导 source transaction。

## 4. Resume 生命周期证据

新增测试真实驱动 priority Scheduler：A 首次分配，B 到达并触发 A preempt，B 完成后 A resume。观察到：

- preempt output：`preempted_req_ids == {"A"}`，A 不在当前 scheduled token map；Worker 旧 slot 清理。
- resume output：A 出现在 `scheduled_new_reqs`，`scheduled_cached_reqs` 为空。
- `ragged_kv_updates.snapshots == {"A"}`，`allocations == {}`。
- resume Snapshot 的 `state_version/page_rows` 是 Scheduler 新分配的 canonical 状态，page IDs 与旧 Snapshot 不同。
- 将当前 `req_id_to_index["A"]` 改为新 slot `2` 后，Snapshot 的 `rows/counts/effective_lens/state_version` 全量落在 slot `2`；旧 slot `7` 保持清空。

这证明 Snapshot materialization 跟随当前 persistent `req_idx`，不依赖 resume 前的历史 row。

## 5. 测试分类

- `PASS`：MRV2 Ragged worker/state/layout/manager/planner/attention focused tests，51 passed；新增 resume lifecycle 通过。
- `TEST_STALENESS`：旧 Dense scheduler/worker fixtures 默认引用本地不存在的 `facebook/opt-125m`，且当前环境网络代理缺少 `socksio`；失败发生在 `ModelConfig` 构造，未进入本次 Ragged 代码。
- `CUDA_REQUIRED`：此前隐藏 CUDA 的旧 tests 需要 UVA/CUDA 初始化；不能作为 CPU regression 结论。
- `REGRESSION`：当前未观察到与本 Slice 代码相关的 focused CPU regression。
- `A100 startup 初次失败`：generic V2 warmup 缺少 Ragged transport，属于 MRV2/Ragged lifecycle integration bug；通过跳过 Ragged generic warmup 修复后启动成功。

## 6. Evidence Index

- CPU：`04-experiments/project1_kv_reclaim/raw/p1-v2-r1-e-mrv2-production-ragged-integration/03-focused-tests-resume.txt`
- Resume：`03-resume-focused.txt`（首个测试通过；同命令中的旧默认 fixture 失败按 `TEST_STALENESS` 分类）
- Startup：`04-mrv2-startup-port18082.log`、`04-mrv2-startup-port18083.log`
- Single prefill/decode：`05-single-request.json`、`05-single-request-instrumented.json`
- Boundary：`06-boundary.json`
- Ragged parity：`07-ragged-parity.json`
- Dense parity：`07-dense-parity.json`
- Mixed：`09-mixed-A.json`、`09-mixed-B.json`
- Concurrent：`10-two-requests-A.json`、`10-two-requests-B.json`
- Finish/reuse：`11-finish-reuse-startup.log`、`11-finish-A.json`、`11-finish-reuse-B.json`

## 7. 最终证据闭环（2026-09-27）

### 7.1 Exact token parity

使用本地 `LLM.generate()` 直接读取 `RequestOutput.outputs[0].token_ids`，同一 Qwen3-0.6B、prompt、seed、`temperature=0`、12 output tokens：

```text
Dense MRV2 : [12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279]
Ragged MRV2: [12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279]
exact_equal=true
```

原始证据：`raw/p1-v2-r1-e-final-evidence-closure/05-final-a100-token-parity-smoke.log`。

### 7.2 True same-step mixed

一次真实 Scheduler step 同时包含：A decode `num_scheduled_tokens=1`、source `E=(8,...,8)`；B new prefill `num_scheduled_tokens=13`、source `E=(0,...,0)`。MRV2 记录 `idx_mapping_np=[1,0]`、`query_start_loc=[0,1,14]`，post-write member seq lens 为 A=`9`、B=`13`；同一 Ragged forward 记录 `requests=2`，并进入真实 KV write 与 attention forward。

临时 instrumentation 已删除。原始证据：`raw/p1-v2-r1-e-final-evidence-closure/03-mixed-step-trace.log`。

### 7.3 Scheduler physical page free/reuse

新增 focused test 观察 canonical `RaggedAttentionManager` 与共享 `BlockPool`：initial free=`11`；A ownership=`(1,2,3,4)`、free=`7`；A finish 后 canonical A state 消失、free=`11`；B ownership=`(4,3)`、free=`9`。本次自然观察到 released ID overlap `[3,4]`，但测试不依赖固定 page ID。

原始证据：`raw/p1-v2-r1-e-final-evidence-closure/04-focused-tests-final.log`。

### 7.4 Warmup scope note

`Ragged generic V2 warmup skip` 已验证于当前 `enforce-eager` / CUDA-Graph-disabled E scope；未来 CUDA Graph/compile warmup requires separate design.

### 7.5 当前 Gate 判断

- Architecture Gate：`PASS`
- CPU Gate：`PASS`
- Resume Gate：`PASS`
- A100 Startup Gate：`PASS`
- A100 Ragged Execution Gate：`PASS`
- Exact Token Parity Gate：`PASS`（generated token IDs exact equality）
- True Mixed-Step Gate：`PASS`（同一 Scheduler step 的 decode + new prefill + Ragged write/read）
- Physical Page Reuse Gate：`PASS`（canonical free capacity 回升，后续 B allocation；额外观察到自然 ID overlap）
- Dense/Ragged Parity Gate：`PASS`
- Mixed/Concurrency Gate：`PASS`
- Reuse Gate：`PASS`
- Overall：`PASS_PENDING_WEB_REVIEW`

## 8. Next Allowed Action

`WEB_REVIEW_CURRENT_SLICE`

Proposed Next Action：由 Web review 决定是否接受当前证据并关闭 Slice；不进入 non-uniform compaction。
