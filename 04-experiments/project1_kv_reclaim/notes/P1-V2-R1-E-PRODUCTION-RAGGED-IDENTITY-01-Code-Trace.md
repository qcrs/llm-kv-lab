# P1-V2-R1-E-PRODUCTION-RAGGED-IDENTITY-01 代码变更记录

## Slice 范围

本轮把已经通过 D synthetic CUDA execution 的 Ragged 组件接入 vLLM production identity 初始化链。只覆盖 activation、spec、manager registry、coordinator 单 `BlockPool` 和真实 Engine startup；Scheduler/Worker step transport、ForwardContext metadata、unified write/read dispatch 尚未完成。

## 变更记录

### 1. `vllm/engine/arg_utils.py`

- **Symbol**：`EngineArgs.page_group_size`、`EngineArgs.add_cli_args()`、`EngineArgs.create_engine_config()`
- **Change Type**：`MODIFY`
- **Before**：`CacheConfig.page_group_size` 没有显式 EngineArgs/CLI 入口。
- **After**：新增 `--page-group-size`，并把值传入 `CacheConfig`。
- **Why**：E 的唯一 public activation predicate 必须能从真实 Engine 配置显式到达 Core。
- **Contract**：`page_group_size is not None` 才激活 Ragged；`None` 保留 Dense。
- **Test**：真实 Qwen startup 日志观察到 `page_group_size: 2`。
- **Status**：通过 startup config resolution。

### 2. `vllm/config/vllm.py`

- **Symbol**：`VllmConfig.validate_ragged_core()`
- **Change Type**：`ADD`
- **Before**：没有 engine-wide Ragged unsupported-mode gate。
- **After**：集中拒绝 TP/PP/DCP/PCP、prefix caching、spec decode、async scheduling、KV offload、CUDA Graph、MLA、sliding-window 和 quantized KV。
- **Why**：避免 Ragged 在部分组件已激活、部分组件仍走 Dense 的半启用状态。
- **Contract**：E 支持矩阵；默认 `page_group_size=None` 不改变行为。
- **Test**：真实启动第一次使用默认 async 时被明确拒绝；显式 `--no-async-scheduling` 后继续完成初始化。
- **Status**：通过 fail-fast 和 positive startup。

### 3. `vllm/model_executor/layers/attention/attention.py`

- **Symbol**：`Attention.get_kv_cache_spec()`
- **Change Type**：`MODIFY`
- **Before**：full decoder attention 始终返回 `FullAttentionSpec`。
- **After**：激活 Ragged 时返回 `RaggedAttentionSpec`，并在 layer boundary 检查 full attention、`head_size_v == head_size` 和 `page_group_size` contract。
- **Why**：让 planner 从真实 Attention layer identity 生成 Ragged group，而不是 synthetic-only spec。
- **Contract**：D geometry；FA2、BF16/FP16、full causal decoder scope。
- **Test**：真实 Qwen3-0.6B 使用 FlashAttention 2 完成 KV cache initialization。
- **Status**：startup 通过；runtime request 尚未通过。

### 4. `vllm/v1/core/single_type_kv_cache_manager.py`

- **Symbol**：`register_all_kvcache_specs()`
- **Change Type**：`ADD`
- **Before**：`RaggedAttentionSpec` 没有 manager registry entry。
- **After**：注册 `RaggedAttentionSpec -> RaggedAttentionManager`。
- **Why**：复用现有 coordinator/registry，不创建第二套 allocator。
- **Contract**：一个 Ragged manager 复用 coordinator-owned `BlockPool`。
- **Test**：CPU coordinator smoke 构造出 `RaggedAttentionManager`。
- **Status**：通过。

### 5. `vllm/v1/core/kv_cache_coordinator.py`

- **Symbol**：`KVCacheCoordinator.__init__()` manager construction
- **Change Type**：`MODIFY`
- **Before**：所有 manager 都只按 spec 直接构造，Ragged placement 没有注入。
- **After**：对 `RaggedAttentionSpec` 按 group `layer_names` 构造 identity `MemberPlacementMap`，并仍使用同一个 coordinator `BlockPool`。
- **Why**：完成 production identity 的静态 placement 注入，不改变 ownership authority。
- **Contract**：Scheduler/Coordinator 仍是 canonical physical owner；placement 只表达 static topology。
- **Test**：CPU smoke 观察 `RaggedAttentionManager`、`num_gpu_blocks=32`、`num_clusters=16`。
- **Status**：通过。

### 6. `vllm/v1/core/ragged_kv_cache_manager.py`

- **Symbol**：`RaggedAttentionManager.__init__()`
- **Change Type**：`MODIFY`
- **Before**：docstring 仍声明未注册 production，且不能接收 coordinator 通用构造参数。
- **After**：更新 production identity 语义，接受并明确忽略通用 sizing/zeroing 参数，拒绝 DCP/PCP 非 1。
- **Why**：保持 coordinator 通用构造接口，同时不让 Ragged manager 获得额外 ownership authority。
- **Contract**：当前 E scope 为 TP/PP/DCP/PCP=1。
- **Test**：CPU coordinator smoke。
- **Status**：通过。

## 当前边界

尚未修改 Scheduler、`SchedulerOutput.ragged_kv_updates` 生产填充、`GPUModelRunner` mirror/StepViews、ForwardContext layer metadata、unified write/read dispatch 或 successful-tail frontier commit。因此当前结果不是 E PASS。

## 本轮继续执行的 production 接线

### [E-07] Scheduler vector allocation and Ragged transport

File:
`vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/sched/scheduler.py`

Symbol:
`KVCacheManager.allocate_slots()`、`Scheduler.schedule()`、`SchedulerOutput.ragged_kv_updates`

Change Type:
`MODIFY`

Before:
生产 scheduler 仍调用 Dense `KVCacheCoordinator` allocation；`RaggedAttentionManager` 的 vector API 未被真实 request path 使用。

After:
Ragged activation 下，`allocate_slots()` 以 `source effective_lens[C] + num_new_tokens` 构造 `RaggedCapacityPlan`，调用 `plan_capacity()` / `apply_capacity_plan()`，Dense `KVCacheBlocks` transport 保持为空，physical rows 通过 `RaggedKVUpdateData` 传输。新/resumed request 使用 snapshot，running request 使用 allocation delta。

Why:
避免把 cluster-major physical rows 塞进 Dense `KVCacheGroup` outer axis，并保持 Scheduler 为唯一 physical ownership authority。

Dense Correspondence:
`KVCacheManager.allocate_slots()` 的原 Dense coordinator branch 未改变；`page_group_size=None` 时 `ragged_manager is None`。

Ragged Meaning:
完成 plan → reserve → transport 的 production control-plane 接线；reserve 不推进 `effective_lens`。

Invariant:
`effective_lens` 与 capacity 分离；同一 request 同一步不会同时发送 snapshot 和 delta。

Dependency:
A/B12/C/D frozen physical state、vector state、Snapshot/Delta contract。

Tests:
`py_compile`、`git diff --check`；focused pytest 被环境缺少 `tblib` 阻断。

Evidence:
`raw/p1-v2-r1-e-production-ragged-identity-01/02-static-validation.txt`

Status:
`NOT YET PROVEN`（真实 request 未运行）。

### [E-08] Worker mirror lifecycle and step staging

File:
`vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu/ragged_kv_state.py`

Symbol:
`GPUModelRunner._update_states()`、`GPUModelRunner.initialize_kv_cache()`、`RaggedWorkerPhysicalState.reindex()`、`RaggedStepViews` staging。

Change Type:
`ADD` / `MODIFY`

Before:
Worker 没有 production Ragged mirror materialization；scheduler output 中的 snapshot/delta 没有消费者。

After:
初始化阶段创建一次 `RaggedWorkerPhysicalState` 和 placement tensors；`_update_states()` 在 batch condense 后应用 snapshot/delta；每步从 mirror gather cluster rows/effective lengths 并构造 `RaggedStepViews`。

Why:
Worker 只作为可丢弃 mirror，step metadata 只作为 derived execution view。

Dense Correspondence:
Dense `InputBatch.block_table` 和 Dense slot mapping 保持原路径；Ragged 分支只在 `RaggedAttentionSpec` 激活时存在。

Ragged Meaning:
将 Scheduler canonical rows 映射到本轮 active request 顺序，保持 write 使用 source E、read 使用 post-write seq lens。

Invariant:
不在 Worker 分配或释放 page；不在每 step 重建 `MemberPlacementMap` tensors。

Dependency:
B12 ownership、C placement/geometry、D `build_ragged_step_views()`。

Tests:
`py_compile`；真实 worker 初始化和 request 未运行。

Evidence:
`raw/p1-v2-r1-e-production-ragged-identity-01/01-source-audit.txt`

Status:
`NOT YET PROVEN`。

### [E-09] ForwardContext and unified Ragged dispatch

File:
`vllm/forward_context.py`, `vllm/model_executor/layers/attention/attention.py`, `vllm/v1/worker/gpu_model_runner.py`

Symbol:
`ForwardContext.ragged_step_views`、`unified_kv_cache_update()`、`unified_attention_with_output()`。

Change Type:
`ADD` / `MODIFY`

Before:
unified seams 只调用 Dense `impl.do_kv_cache_update()` / `impl.forward()`。

After:
Ragged context 存在时，unified seams 直接调用已有 `ragged_kv_cache_update()` / `ragged_attention_forward()`；Dense branch 不变。Ragged worker 跳过 Dense block-table metadata builder。

Why:
复用 D 已验证的 `reshape_and_cache_flash()` 和 `flash_attn_varlen_func()`，不新增 custom op，不重写 `FlashAttentionImpl`。

Dense Correspondence:
无 Ragged views 时保留原 unified implementation。

Ragged Meaning:
确保一轮只发生一次正确 Ragged write/read，禁止 silently fallback 到 Dense slot mapping。

Invariant:
physical cache 继续使用 zero-copy virtual view；layer-local member slice 由 placement-derived metadata 给出。

Dependency:
D FA2 execution contract、physical/virtual geometry。

Tests:
`py_compile`、`git diff --check`；真实 CUDA execution 未完成。

Evidence:
`raw/p1-v2-r1-e-production-ragged-identity-01/02-static-validation.txt`

Status:
`NOT YET PROVEN`。

### [E-10] Successful frontier commit

File:
`vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/core/sched/scheduler.py`

Symbol:
`GPUModelRunner.sample_tokens()`、`Scheduler.update_from_output()`。

Change Type:
`MODIFY`

Before:
Ragged worker/scheduler 没有真实 production success commit。

After:
forward 成功返回后，worker 在 sampling 前提交 `effective_lens`；scheduler 只在收到 `ModelRunnerOutput` 后提交 canonical `effective_lens`。target 为 source vector 加本步 scheduled token 数。

Why:
明确 reserve-before-success invariant，避免 plan/reserve 阶段提前推进 E。

Dense Correspondence:
Dense request 不进入 Ragged commit branch。

Ragged Meaning:
建立 execute success → Worker commit / Scheduler commit 的顺序。

Invariant:
失败或未收到 output 时不存在这两个 commit 调用；Worker 不成为 allocator authority。

Dependency:
B12 state_version、D source-E/post-write-E contract。

Tests:
仅完成静态检查；E7 focused test 尚未在当前缺失依赖环境中执行。

Evidence:
`raw/p1-v2-r1-e-production-ragged-identity-01/02-static-validation.txt`

Status:
`NOT YET PROVEN`。

## ADD / MODIFY / REPLACE / DELETE 总结

ADD:
- `ForwardContext.ragged_step_views` / `ragged_layer_indices`。
- `KVCacheManager` Ragged manager access、pending delta transport、snapshot/export/commit helpers。
- `RaggedWorkerPhysicalState.reindex()`。
- Worker Ragged mirror、step view staging 和 success commit 分支。

MODIFY:
- Scheduler vector allocation、snapshot/delta population 和 successful canonical commit。
- Unified attention seams 的 Ragged typed dispatch。
- Ragged manager/registry/coordinator 的 E0 activation（此前已有 local worktree 修改）。

REPLACE:
- Ragged 激活时 Dense allocation/metadata dispatch 被专用 Ragged branch 替代；Dense 默认路径未替换。

### [E-11] 真实 A100 生命周期修复与 parity 负证据

File:
`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/sched/scheduler.py`、`vllm/v1/worker/gpu_model_runner.py`

Symbol:
`KVCacheManager.allocate_slots()`、`Scheduler.update_from_output()`、Ragged step scalar derivation

Change Type:
`MODIFY`

Before:
Ragged allocation 把所有非 `None` 的 `new_computed_blocks` 都当作 cached KV，真实首次 request 传入空容器时被错误拒绝；Scheduler 在 request 已 abort/finish 并释放 canonical state 后仍尝试 Ragged frontier commit；step producer 通过 GPU `.max().item()` 取得 `max_kv_len`。

After:
仅当 `new_computed_blocks` 实际包含 block 时拒绝 cached KV；对正常输出路径已判定 `request is None` 或 `request.is_finished()` 的 request 跳过 Ragged commit；`max_kv_len` 从 CPU Worker mirror 计算，避免每步 GPU→CPU scalar sync。

Why:
这是 source-native vLLM 首次请求和 aborted in-flight request 的实际接口语义，必须适配；否则真实 A100 request 会在 Scheduler allocation 或 commit 阶段崩溃。

Dense Correspondence:
Dense `KVCacheBlocks` allocation 和 normal output skip 逻辑保持不变；`page_group_size=None` 不进入这些 Ragged 分支。

Ragged Meaning:
修复了 plan/reserve/transport 的空 computed-container 兼容性，以及 finish/free 与 success output 竞态；不改变 physical geometry 或 ownership authority。

Invariant:
active request 缺少 Ragged canonical state 仍 fail-closed；只有 forward success 后才推进 active request 的 effective frontier；Worker 仍不是 allocator authority。

Dependency:
B12 Scheduler authority、D source-E/post-write-E、E Snapshot/Delta lifecycle。

Tests:
`tests/v1/core/test_ragged_kv_cache_manager.py`、`tests/v1/worker/test_ragged_kv_state.py`、`tests/v1/attention/test_ragged_execution.py`：`35 passed, 7 skipped`；真实 A100 单请求与双并发请求返回 HTTP 200；Dense/Ragged greedy parity 未通过。

Evidence:
`raw/p1-v2-r1-e-production-ragged-identity-01/04-a100-request.log`、`05-a100-concurrency.log`、`06-dense-ragged-parity.log`。

Status:
`IN_PROGRESS`

## 当前结论

- 真实 A100 production initialization：`PASS`。
- 真实 Ragged single request：`PASS`。
- 真实 Ragged multi-step decode smoke：request 返回 `200`，但输出语义尚未独立证明。
- 真实 Ragged two-request concurrency：`PASS`（修复 commit-after-free 后无 EngineCore crash）。
- Dense/Ragged identity parity：`FAIL / NOT CLOSED`；同一 greedy prompt 输出不同，说明当前 Ragged attention read/write 仍存在 correctness 问题。
- Slice verdict：`IN_PROGRESS`，不得报告 `PASS / CLOSED`。

DELETE:
`NONE`。

## 当前真实调用链（source-level）

Chain A — Scheduler allocation:
`Scheduler.schedule()` → `KVCacheManager.allocate_slots()` → `RaggedAttentionManager.plan_capacity()` → `apply_capacity_plan()` → `RaggedKVUpdateData` → `SchedulerOutput`。

Chain B — Worker materialization:
`GPUModelRunner._update_states()` → `apply_snapshot()` / `apply_allocation_delta()` → `RaggedWorkerPhysicalState.gather()` → `build_ragged_step_views()`。

Chain C — Attention:
`GPUModelRunner.execute_model()` → `set_forward_context()` → `Attention.forward()` → `unified_kv_cache_update()` → `ragged_kv_cache_update()`；随后 `unified_attention_with_output()` → `ragged_attention_forward()` → `flash_attn_varlen_func()`。

Chain D — Commit:
`GPUModelRunner.sample_tokens()` after successful forward → Worker `commit_effective_lens()`；`Scheduler.update_from_output()` → Scheduler `commit_ragged_effective_lens()`。

## 当前验证结论

- `python -m py_compile`：通过。
- `git diff --check`：通过。
- `ruff --select I,F,E9`：通过。
- focused pytest：`35 passed, 7 skipped`。
- 固定 `.venvs/vllm-v026-torch211-cu129-py310/bin/python` 环境可用；未下载或安装依赖。
- 真实 A100 request-level engine：通过 V1 Ragged runner 完成初始化、prefill、decode、commit 和 teardown。
- Dense/Ragged identity parity：相同 Qwen3-0.6B、greedy、BF16、prompt、`max_tokens=2` 输出 token IDs 均为 `[3555, 525]`，通过。

## 新增 E 修复记录

### [E-12] Ragged production runner 与 physical cache reshape

File:
`vllm/v1/worker/gpu_worker.py`、`vllm/v1/worker/gpu_model_runner.py`

Symbol:
`GPUWorker.use_v2_model_runner`、`GPUModelRunner._reshape_kv_cache_tensors()`

Change Type:
`MODIFY`

Before:
Ragged 配置仍选择 V2 runner；真实 request 未进入 E 已实现的 V1 SchedulerOutput/mirror/StepViews 路径。切换到 V1 后，cache reshape 仍按 Dense `[P,Hkv,B,2D]` 解释 Ragged backing。

After:
`page_group_size` 非空时 fail-closed 选择 V1 runner；Ragged cache backing 按 `[P,Hp,B,2D]` 建立 zero-copy view，Dense 继续使用 V2 runner 和原有 reshape。

Why:
真实 A100 日志首次显示 `Using V2 Model Runner`，且 V1 路径在 cache 初始化处暴露 shape mismatch；这是 request-level parity 失败的实际入口遗漏。

Invariant:
physical geometry、virtual block equation、Scheduler authority 和 zero-copy physical→virtual view 不变。

Tests:
真实 A100 Qwen3-0.6B request；focused Ragged tests；Dense baseline parity。

Evidence:
`raw/p1-v2-r1-e-production-ragged-identity-01/07-a100-v1-ragged-parity.log`

Status:
`PASS`

### [E-13] InputBatch compaction 尾部空槽兼容

File:
`vllm/v1/worker/gpu_model_runner.py`

Symbol:
`GPUModelRunner._update_states()`

Change Type:
`MODIFY`

Before:
Ragged mirror reindex 列表直接遍历 `InputBatch.req_ids`，尾部合法 `None` 空槽导致真实 teardown/decode step `KeyError`。

After:
只对非空 request row 生成 source indices，保持 `InputBatch.condense()` 的现有语义。

Why:
V1 production request 首次完整运行暴露该普通 batch compaction 边界。

Status:
`PASS`

## Overengineering audit（当前）

- 第二套 placement authority：未新增；Worker 使用一次性 placement tensors。
- 第二套 request ID allocator：未新增。
- Worker allocator authority：未新增；Worker 只 apply/remove mirror。
- 提前推进 E：source-level 与真实单请求运行均保持 reserve 后、forward success 才 commit。
- 每 step 重建 placement tensor：未新增；placement tensors 在 KV cache initialization 创建。
- cluster axis 塞入 KVCacheGroup axis：未新增；snapshot/delta 使用独立 Ragged transport。
- identity scalar shortcut：未新增；仍使用 `effective_lens[C]` / `page_counts[C]`。
- KV cache duplicate copy：未新增；Ragged helper 使用 virtual zero-copy view。
- 未使用 abstraction：新增字段/分支均有 worker/attention consumer，并由真实 A100 request 触发。
- Dense hot path 污染：仅增加 `ragged_manager is not None` 分支；Dense branch 保持原逻辑。

## 当前 Verdict

`P1-V2-R1-E-PRODUCTION-RAGGED-IDENTITY-01 = PASS_PENDING_WEB_REVIEW`

原因：E production request path 已在真实 A100 上完成 V1 Ragged runner 初始化、Ragged write/read、multi-step decode、成功提交和 teardown；Dense/Ragged identity parity 已通过。仍等待 Web review 后才能关闭 Slice。

Next Allowed Action:
`WEB_REVIEW_CURRENT_SLICE`
