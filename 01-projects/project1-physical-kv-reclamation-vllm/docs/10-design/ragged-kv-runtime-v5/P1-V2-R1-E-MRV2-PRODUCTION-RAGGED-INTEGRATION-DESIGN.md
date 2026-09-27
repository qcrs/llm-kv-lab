# P1-V2-R1-E — MRV2 Production Ragged Identity 接入设计

> 目标：在 **vLLM v0.26.0 MRV2**（`vllm/v1/worker/gpu/model_runner.py`）中完成 Production Ragged Identity request lifecycle，修正当前 E Slice 将 Ragged 强制切换到 MRV1 的架构偏移。
>
> 本文是实现规格，不进入 non-uniform compaction，不重新设计 A/B/C/D 已冻结 contract。

---

## 0. 审计基线与结论

### 0.1 本次审计源码

- `qcrs/vllm`
  - branch: `p1/v2-token-compaction-v026`
  - audited HEAD: `7ed376375479c3de2fc392cb8356d35aa695a7dc`
  - commit: `feat(ragged-kv): wire production identity runtime path`
  - parent: `e78de910720a95dfdb34d1ea3d73bb902fc0bca7`
- `qcrs/llm-kv-lab`
  - `main` HEAD: `275aa614f62a70dd4c3139aae4964da9fbe3c816`
  - E Code Trace / PROJECT_STATE 已审计
- `aiha-lab/tangram`
  - 用于 Ragged execution/state 结构对照，不作为本项目 ownership authority 设计来源。

### 0.2 最终架构判定

P1 的 production target 必须固定为：

```text
Dense  ─┐
        ├──> MRV2: vllm/v1/worker/gpu/model_runner.py
Ragged ─┘
```

禁止：

```text
Dense  -> MRV2
Ragged -> MRV1 / vllm/v1/worker/gpu_model_runner.py
```

`page_group_size` 是 KV runtime/layout feature，不得改变 ModelRunner generation。

### 0.3 当前 E 的真实状态

当前 commit `7ed3763` 已经完成了大量正确的 E control-plane / dispatch 接线：

```text
CLI / Config
  -> RaggedAttentionSpec
  -> RaggedAttentionManager
  -> Scheduler vector allocation
  -> SchedulerOutput.ragged_kv_updates
  -> ForwardContext Ragged fields
  -> unified Ragged KV write/read seam
```

但 Worker lifecycle 实装在 **MRV1**，并在 `gpu_worker.py` 中通过：

```python
self.use_v2_model_runner = (
    vllm_config.use_v2_model_runner
    and vllm_config.cache_config.page_group_size is None
)
```

强制 Ragged 走 MRV1。

因此当前 E 不是设计整体错误，而是：

> **Scheduler / transport / Attention seam 基本可保留；Worker runtime integration 接错了 runner generation。**

这次修复应是 **MRV2 reconciliation**，不是重写 Ragged。

---

# 1. Frozen Architecture：本轮禁止修改的核心 contract

## 1.1 Authority 边界

```text
Scheduler
  = canonical physical ownership authority

Worker
  = discardable physical-state mirror

RaggedStepViews
  = current-step derived execution metadata

Attention / FA2
  = execution consumer
```

禁止：

- Worker 自己分配/free physical page；
- Worker 报 allocator-authoritative freed IDs；
- Attention backend 持有 request lifetime ownership；
- 用 Dense `KVCacheGroup` outer axis 伪装 Ragged cluster axis。

---

## 1.2 Scheduler canonical state

每个 Ragged request：

```text
RaggedRequestPhysicalState
  state_version
  effective_lens[C]
  page_rows[C][variable depth]
```

其中：

```text
page_counts[C] = len(page_rows[c])
```

`C` 是 physical clusters，不是 KVCacheGroup 数量。

---

## 1.3 Transaction ordering

必须继续保持：

```text
PLAN
  -> RESERVE
  -> TRANSPORT
  -> EXECUTE
  -> SUCCESS
  -> COMMIT
```

即：

```text
allocated capacity != valid effective frontier
```

示例：

```text
source E = [32,32]
reserve 后 page_counts 可增长
但 forward success 前 E 仍必须是 [32,32]
forward + output 成功后才 -> [33,33]
```

---

## 1.4 Physical / Virtual geometry

继续冻结：

```text
physical cache: [P, Hp, B, 2D]
virtual cache:  [P*Hp, 1, B, 2D]

virtual_block = physical_page * Hp + column
virtual_slot  = virtual_block * B + offset
```

physical -> virtual 必须 zero-copy。

---

## 1.5 Vector state 不得退化

Identity 阶段虽然通常：

```text
E[0] == E[1] == ... == E[C-1]
```

production contract 仍必须是：

```text
effective_lens[C]
page_counts[C]
page_rows[C]
state_version
```

不得为了 Identity 再退回 scalar `effective_kv_len` 作为 Ragged authority。

---

# 2. 当前 commit 7ed3763：哪些保留，哪些必须撤

## 2.1 KEEP — 保留现有 E 实现

以下设计与 MRV2 target 一致，原则上不要重写：

### Config / activation

- `vllm/engine/arg_utils.py`
  - `--page-group-size`
- `vllm/config/vllm.py`
  - `validate_ragged_core()`

### Spec / Core / Scheduler

- `Attention.get_kv_cache_spec() -> RaggedAttentionSpec`
- `RaggedAttentionSpec -> RaggedAttentionManager` registry
- `KVCacheCoordinator` identity `MemberPlacementMap`
- `KVCacheManager.allocate_slots()` Ragged vector branch
- `RaggedPageAllocationDeltaData`
- `RaggedRequestStateSnapshotData`
- `RaggedKVUpdateData`
- `Scheduler.schedule()` snapshot/delta population
- `Scheduler.update_from_output()` success-only canonical E commit

### Execution foundation

- `gpu/attn_utils.py::_reshape_kv_cache()` 中 C 阶段已经实现的 Ragged physical view
- `ragged_layout.py`
- `ragged_forward.py`

### Shared Attention seam

- `ForwardContext.ragged_step_views`
- `ForwardContext.ragged_layer_indices`
- `unified_kv_cache_update()` Ragged dispatch
- `unified_attention_with_output()` Ragged dispatch

这些是 runner-neutral 或 MRV2-compatible 的能力。

---

## 2.2 REVERT — 当前 MRV1 shortcut 必须撤销

### A. `vllm/v1/worker/gpu_worker.py`

必须删除 Ragged 强制选择 MRV1 的逻辑。

恢复：

```python
self.use_v2_model_runner = vllm_config.use_v2_model_runner
```

Ragged 不应参与 runner generation selection。

此外建议在 central Ragged validation 中明确：

```text
Ragged production requires MRV2
```

如果用户显式设置 MRV1，则 fail closed，而不是偷偷支持。

---

### B. `vllm/v1/worker/gpu_model_runner.py`（MRV1）

E commit 中加入的以下 Ragged production 代码不应继续作为正式实现：

- `ragged_worker_state`
- `ragged_member_to_cluster`
- `ragged_member_to_column`
- `ragged_layer_indices`
- `ragged_step_views`
- `_update_states()` 中 snapshot/delta/reindex
- `_prepare_inputs()` 中 RaggedStepViews staging
- `execute_model()` Ragged Dense-metadata bypass
- `sample_tokens()` Ragged Worker E commit
- `_reshape_kv_cache_tensors()` Ragged reshape
- MRV1 initialize Ragged runtime state

这些逻辑中的**语义**可作为 MRV2 port 参考，但不能保留成第二套 production runtime。

---

### C. `RaggedWorkerPhysicalState.reindex()`

该 API 是为 MRV1 persistent `InputBatch.condense()` 服务的。

MRV2 不应该使用它。

MRV2 的 request lifetime identity 已经由：

```text
RequestState.req_id_to_index
RequestState.index_to_req_id
free_indices
```

提供稳定 slot。

建议：

- 如果代码搜索确认 `reindex()` 只被 MRV1 E 使用：删除；
- 如果为了保留历史测试暂不删除：明确标记 MRV2 禁用，不得从新实现调用。

优先选择删除，避免留下错误 lifecycle abstraction。

---

# 3. 为什么 MRV2 与 MRV1 的接法不能机械迁移

这是本次 reconciliation 最重要的地方。

## 3.1 MRV1

MRV1 的 InputBatch 是 persistent batch，存在：

```text
remove request
  -> holes / None
  -> InputBatch.condense()
  -> row reorder
```

所以之前真实 A100 暴露了：

```text
trailing None slot
  -> Ragged mirror reindex KeyError
```

MRV1 修 `condense + reindex` 是合理的 MRV1 bug fix。

---

## 3.2 MRV2

MRV2 不使用这个模型。

MRV2 有 persistent：

```text
RequestState
  req_id_to_index
  index_to_req_id
  free_indices
```

每一步临时构造：

```text
InputBatch
  req_ids
  idx_mapping_np
```

其中：

```text
batch_idx -> req_state_idx
```

所以 MRV2 的正确结构应是：

```text
Persistent Ragged Worker mirror
  row index = RequestState req_idx

              ↓ current step

InputBatch.idx_mapping_np

              ↓ gather

RaggedClusterStepView

              ↓ derive

RaggedStepViews
```

**不需要 mirror condense，不需要 mirror reindex。**

---

# 4. MRV2 的真实 production lifecycle

当前 MRV2 `GPUModelRunner` 的核心链：

```text
execute_model(scheduler_output)
  |
  |-- finish_requests()
  |-- add_requests()
  |-- update_requests()
  |
  |-- prepare_inputs()
  |-- prepare_attn()
  |-- model_state.preprocess_state()
  |-- model_state.prepare_attn()
  |
  |-- set_forward_context()
  |-- model(...)
  |
  |-- build ExecuteModelState
  `-- return

sample_tokens()
  |
  |-- sample()
  |-- build ModelRunnerOutput / AsyncOutput
  |-- postprocess_sampled()
  `-- return output
```

Ragged 应嵌入这个 lifecycle，而不是新增第二套 runner。

---

# 5. Target MRV2 Ragged lifecycle

最终应形成：

```text
Scheduler canonical Ragged state
        |
        | plan / reserve
        v
SchedulerOutput.ragged_kv_updates
        |
        v
MRV2 execute_model()
        |
        | finish_requests()
        | add_requests()
        | update_requests()
        |
        | apply Ragged snapshot/delta
        v
RaggedWorkerPhysicalState
        |
        | InputBatch.idx_mapping_np
        v
RaggedClusterStepView (source state)
        |
        v
build_ragged_step_views()
        |
        v
ForwardContext
        |
        +--> ragged_kv_cache_update()
        |
        `--> ragged_attention_forward()
                 |
                 v
            model forward success
                 |
                 v
MRV2 sample_tokens()/postprocess success
                 |
                 v
Worker commit_effective_lens()
                 |
                 v
ModelRunnerOutput reaches Scheduler
                 |
                 v
Scheduler commit_ragged_effective_lens()
```

---

# 6. MRV2 具体实现设计

## 6.1 `gpu_worker.py` — 只恢复正常 runner selection

### MODIFY / REVERT

文件：

```text
vllm/v1/worker/gpu_worker.py
```

目标：

```python
self.use_v2_model_runner = vllm_config.use_v2_model_runner
```

禁止 `page_group_size` 改 runner。

### Config hardening

建议 `validate_ragged_core()` 增加：

```text
if Ragged active and MRV2 disabled:
    reject
```

这样：

```text
Ragged + MRV2 = supported E path
Ragged + MRV1 = explicit unsupported
```

---

# 7. MRV2 initialization

文件：

```text
vllm/v1/worker/gpu/model_runner.py
```

## 7.1 `__init__`

只声明 persistent runtime fields：

```python
self.ragged_worker_state = None
self.ragged_member_to_cluster = None
self.ragged_member_to_column = None
self.ragged_layer_indices = None
```

不在这里假定最终 KV group layout；真实 geometry 要等 `initialize_kv_cache()`。

---

## 7.2 `initialize_kv_cache()`

在 `kv_cache_config` 确定后识别 Ragged group。

E scope：

```text
exactly one KV cache group
exactly one RaggedAttentionSpec group
no mixed Dense/Ragged
```

建立：

```python
placement = MemberPlacementMap.identity(
    num_layers=len(group.layer_names),
    num_kv_heads=spec.num_kv_heads,
    page_group_size=spec.page_group_size,
)
```

Worker placement 必须和 Scheduler Coordinator 使用相同来源：

```text
kv_cache_group.layer_names order
```

不得通过：

- `static_forward_context` dict 的偶然遍历顺序；
- layer name string 解析；
- model module traversal 顺序；

重新定义 layer identity。

建立：

```python
self.ragged_worker_state = RaggedWorkerPhysicalState(
    max_num_reqs=self.max_num_reqs,
    placement=placement,
    max_pages_per_cluster=cdiv(self.max_model_len, spec.block_size),
    block_size=spec.block_size,
)
```

placement tensors 只 materialize 一次：

```python
self.ragged_member_to_cluster,
self.ragged_member_to_column = placement_to_tensors(
    placement,
    device=self.device,
)
```

layer mapping：

```python
self.ragged_layer_indices = {
    layer_name: layer_idx
    for layer_idx, layer_name in enumerate(group.layer_names)
}
```

并检查：

```text
all layer_names exist in static_forward_context
per-layer num_kv_heads == Ragged spec expectation
```

---

## 7.3 KV tensor allocation / reshape

**不要在 MRV2 model_runner 新写一套 reshape。**

当前 C 阶段已经在：

```text
vllm/v1/worker/gpu/attn_utils.py::_reshape_kv_cache()
```

完成：

```text
RaggedAttentionSpec
  -> [P,Hp,B,2D]
```

MRV2 `initialize_kv_cache()` 原本就调用：

```text
init_kv_cache()
  -> _allocate_kv_cache()
  -> _reshape_kv_cache()
```

因此继续复用即可。

这是撤销 MRV1 E-12 duplicate reshape 的主要原因。

---

# 8. SchedulerOutput -> MRV2 Worker mirror

建议在 MRV2 runner 内新增一个很窄的 helper：

```python
_apply_ragged_kv_updates(scheduler_output)
```

不要新增第二套 runtime manager。

## 8.1 调用顺序

`execute_model()`：

```text
finish_requests()
add_requests()
update_requests()
_apply_ragged_kv_updates()
```

为什么必须在 add/update 之后？

因为 snapshot/delta 是 `req_id` keyed，但 Worker mirror row 绑定的是 MRV2 persistent `req_idx`；必须先保证当前 request 已经拥有合法 RequestState slot。

---

## 8.2 Snapshot

```python
req_idx = self.req_states.req_id_to_index.get(req_id)
if req_idx is None:
    fail closed
self.ragged_worker_state.apply_snapshot(req_idx, snapshot)
```

snapshot 是 authoritative replacement。

---

## 8.3 Delta

```python
req_idx = self.req_states.req_id_to_index.get(req_id)
if req_idx is None:
    fail closed
self.ragged_worker_state.apply_allocation_delta(req_idx, delta)
```

Delta 自己已经验证：

```text
expected state_version
expected source E
expected source counts
```

不要绕过这些 fence。

---

## 8.4 Dense behavior

Dense path：

```text
ragged_worker_state is None
scheduler_output.ragged_kv_updates is None
```

行为保持原样。

Ragged path如果 transport 缺失：

```text
fail closed
```

禁止 fallback 到 Dense block table。

---

# 9. Request removal / slot reuse

MRV2 `_remove_request(req_id)` 是 mirror lifecycle 的正确清理点。

当前顺序：

```text
model_state.remove_request
req_states.remove_request -> req_idx
other states remove
```

Ragged 应使用 `req_states.remove_request()` 返回的旧 `req_idx`：

```python
req_idx = self.req_states.remove_request(req_id)
if req_idx is None:
    return False
if self.ragged_worker_state is not None:
    self.ragged_worker_state.remove_request(req_idx)
```

Worker remove 只清 mirror：

```text
rows -> 0
counts -> 0
E -> 0
state_version -> -1
```

**绝不调用 BlockPool.free。**

Scheduler 才负责 canonical page free。

这是防止 request slot reuse 时发生 stale/ABA 污染的关键。

---

# 10. Current-step Ragged metadata staging

建议新增窄 helper：

```python
_prepare_ragged_step(input_batch)
```

返回：

```text
RaggedStepViews
+ source RaggedClusterStepView
```

后者用于 success commit 精确绑定 forward-entry source state。

---

## 10.1 active request indexing

使用：

```python
req_indices = input_batch.idx_mapping_np[:input_batch.num_reqs]
cluster_view = self.ragged_worker_state.gather(req_indices)
```

禁止：

```text
gather(range(num_reqs))
```

因为 MRV2 persistent request slot 不保证当前 batch requests 位于 `[0,num_reqs)`。

这是 MRV1 -> MRV2 port 中最容易出现的严重 bug。

---

## 10.2 GPU derived tensors

从 mirror：

```text
cluster_rows     [R,C,MaxPages]
source E         [R,C]
state_versions   [R]
```

只为当前 step materialize：

```python
cluster_rows_gpu
source_e_gpu
```

placement tensors必须复用初始化缓存。

---

## 10.3 query information

直接复用 MRV2 `InputBatch`：

```text
input_batch.query_start_loc
input_batch.num_scheduled_tokens
input_batch.num_tokens
```

关键值：

```python
num_actual_tokens = input_batch.num_tokens
max_query_len = int(input_batch.num_scheduled_tokens.max())
```

`max_kv_len` 从 CPU mirror 计算：

```python
max_kv_len = int(
    (cluster_view.effective_lens
     + input_batch.num_scheduled_tokens[:, None]).max()
)
```

避免前一轮 E 曾暴露的：

```text
GPU .max().item()
-> per-step GPU->CPU synchronization
```

---

## 10.4 构建 RaggedStepViews

调用已经通过 D real-CUDA 验证的：

```python
build_ragged_step_views(
    cluster_rows_gpu,
    source_e_gpu,
    input_batch.query_start_loc[:input_batch.num_reqs + 1],
    cached_member_to_cluster,
    cached_member_to_column,
    Hp,
    B,
    num_actual_tokens=input_batch.num_tokens,
    max_query_len=max_query_len,
    max_kv_len=max_kv_len,
)
```

保持：

```text
write addresses use source E
attention read seq lens use source E + query_len
```

---

# 11. MRV2 attention preparation：Ragged 必须绕开 Dense BlockTables

当前 Dense MRV2：

```text
prepare_attn()
  -> gather Dense BlockTables
  -> compute Dense slot mappings

model_state.prepare_attn()
  -> Dense backend metadata
```

Ragged 不应调用这条路径。

原因：Scheduler 在 Ragged mode 故意把 Dense `KVCacheBlocks` transport 置空，真实 physical rows 在 `ragged_kv_updates` 中。

如果继续 Dense prepare_attn：

```text
empty / stale Dense block table
  -> bogus slot_mapping / metadata
  -> accidental Dense fallback
```

---

## 11.1 Target branch

在 `execute_model()` real batch path：

```text
input_batch = prepare_inputs(...)

if Ragged:
    ragged_views, ragged_source = _prepare_ragged_step(input_batch)
    attn_metadata = {}
    slot_mappings_by_layer = {}
    skip Dense prepare_attn()
    skip Dense model_state.prepare_attn()
else:
    existing Dense path unchanged
```

E gate 已经拒绝：

- MLA
- sliding window
- hybrid/state-space model path
- TP/PP/DCP/PCP > 1
- spec decode

所以当前 Qwen full-attention E scope 下，不需要为 Ragged 创建 fake Dense metadata。

---

# 12. ForwardContext

真实 model forward：

```python
with set_forward_context(
    attn_metadata,
    ...,
    slot_mapping=slot_mappings_by_layer,
    ragged_step_views=ragged_views,
    ragged_layer_indices=self.ragged_layer_indices,
):
    model(...)
```

Dense：

```text
ragged_step_views = None
ragged_layer_indices = None
```

Ragged：

```text
attn_metadata = {}
slot_mapping = {}
ragged_step_views != None
```

Attention seam 已经存在：

```text
unified_kv_cache_update
  -> ragged_kv_cache_update

unified_attention_with_output
  -> ragged_attention_forward
```

这部分保留即可。

---

# 13. Layer identity 必须单一来源

`ragged_forward.py` 使用：

```python
member_start = layer_idx * num_kv_heads
```

所以 `layer_idx` 必须与 Scheduler `MemberPlacementMap` 的 layer order exact 对齐。

唯一允许来源：

```text
kv_cache_group.layer_names
```

即：

```python
ragged_layer_indices = {
    name: i
    for i, name in enumerate(group.layer_names)
}
```

禁止通过 layer name 数字解析或 module traversal 重建。

---

# 14. Dummy / profile path

这一部分容易误判，但源码已经确认。

MRV2：

```text
profile_run()
  -> _dummy_run(skip_attn=True, is_profile=True)
```

该路径：

```text
attn_metadata = None
```

FA2：

```python
if attn_metadata is None:
    # Profiling run.
    return output.fill_(0)
```

所以 profiling 不会访问或 reshape Ragged KV cache。

结论：

- profile dummy 不需要 Ragged Scheduler ownership；
- 不要为了 profile 创建 fake snapshot/page rows；
- `ragged_step_views=None` 即可；
- current E 已经禁止 CUDA Graph，所以暂时不需要支持真实-attention dummy capture。

如果 Ragged 配置进入一个 `dummy_run` 且 `skip_attn=False` 的未审计路径：

```text
fail closed / NOT SUPPORTED
```

不要静默制造 fake Ragged state。

---

# 15. Worker success-only frontier commit

这是 E correctness closure 的核心。

## 15.1 不在 execute 前 commit

以下阶段均不得推进 Worker E：

```text
Scheduler reserve
Worker snapshot/delta apply
_prepare_ragged_step
before model()
```

---

## 15.2 保存 exact forward-entry source

`_prepare_ragged_step()` 返回的 `RaggedClusterStepView` 应保存进 `ExecuteModelState`：

```text
source effective_lens
source state_versions
```

这是实际生成 write slots 的 source state。

不要在 success commit 时重新猜 source。

建议 `ExecuteModelState` 增加：

```python
ragged_source_state: RaggedClusterStepView | None
```

这是必要的 execution transaction evidence，不是第二套 authority。

---

## 15.3 commit seam

MRV2 `sample_tokens()` 中：

```text
sample success
  -> postprocess_sampled success
  -> Worker Ragged commit
  -> output return
```

建议 commit 在 `postprocess_sampled()` 成功后、最终 return 前完成。

原因：

- forward 成功但 sampling/postprocess 抛异常时，不应让 Worker E 单边推进；
- Scheduler 只会在拿到成功 ModelRunnerOutput 后推进 canonical E；
- Worker/Scheduler commit condition 应尽可能对齐。

每个 request：

```python
req_idx = int(input_batch.idx_mapping_np[batch_idx])
source_e = tuple(ragged_source_state.effective_lens[batch_idx])
state_version = int(ragged_source_state.state_versions[batch_idx])
q_len = int(input_batch.num_scheduled_tokens[batch_idx])
target_e = tuple(v + q_len for v in source_e)

ragged_worker_state.commit_effective_lens(
    req_idx,
    state_version,
    source_e,
    target_e,
)
```

不修改 state_version。

---

# 16. Scheduler canonical commit

当前 `Scheduler.update_from_output()` 已有逻辑原则上保留。

其语义是：

```text
ModelRunnerOutput arrived successfully
  -> locate scheduler snapshot/delta source
  -> commit canonical E = source E + num_scheduled_tokens
```

并已修复真实 E 中发现的 aborted/finished race：

```python
if request is None or request.is_finished():
    continue
```

该修复应保留。

当前 E 禁止 speculative decoding，所以 scheduled token 数与本轮真实 KV write 数一致。

---

# 17. Scalar `effective_kv_len` 在 MRV2 Ragged 中的定位

当前 MRV2 已有历史 P1/V1 scalar：

```text
RequestState.effective_kv_len
InputBatch.cache_positions
InputBatch.effective_kv_seq_lens
```

它不能成为 R1 Ragged authority。

MRV2 Ragged execution必须只用：

```text
RaggedWorkerPhysicalState.effective_lens[C]
```

当前 identity 阶段 scalar 和 vector 数值通常相等，可以让 scalar state继续随现有 generic postprocess 更新以减少改动。

但：

```text
Ragged slots / seq lens / member visibility
```

禁止读取 scalar `cache_positions/effective_kv_seq_lens`。

下一阶段进入 non-uniform E 后，scalar 只能是 legacy compatibility state，不能表达真实 Ragged physical frontier。

---

# 18. Tangram：借鉴点与明确不采用点

Tangram 提供了重要事实：

```text
Ragged paging -> explicit group axis
per-group effective lengths
per-group depth may become non-uniform
execution needs grouped address translation
```

其 `RaggedBlockTable` 使用：

```text
[request, head_group, block_depth]
```

这支持我们的基本判断。

但本项目不复制以下部分。

## 18.1 不复制 uniform append reconstruction

Tangram `_append_row_grouped()` 通过 flat IDs + existing counts 反推 uniform target depth。

本项目已经有：

```text
appended_page_counts[C]
```

显式表达每个 cluster 的 append 数量，更适合真正 non-uniform state。

---

## 18.2 不复制 Worker snapshot/restore authority

Tangram persistent InputBatch 需要：

```text
snapshot_row()
restore_row()
```

因为 batch row 会被 compact/drop/re-add。

MRV2 已有 stable RequestState slot；本项目还有 Scheduler Full Snapshot 作为 authoritative recovery。

因此：

```text
Scheduler Snapshot
-> Worker mirror replacement
```

而不是 Worker 自己维护长期恢复 authority。

---

## 18.3 不复制 Worker freed IDs

Tangram 会返回：

```text
compression_freed_block_ids
```

本项目冻结：

```text
Worker reports shape transition
Scheduler derives detached pages from canonical KVCacheBlock rows
Scheduler/BlockPool performs free
```

保持 allocator authority 单一。

---

# 19. 当前发现的 bug / 风险清单

## P0 — 必须修

### P0-1 Runner generation drift

当前 E commit：Ragged 强制 MRV1。

修复：删除 fallback，Ragged target=MRV2。

### P0-2 MRV2 没有消费 Ragged transport

Scheduler 已生产 `ragged_kv_updates`，但 MRV2 `gpu/model_runner.py` 当前没有任何 Ragged consumer。

不修会导致：

```text
Scheduler reserve pages
Worker no mirror
Worker no StepViews
Attention cannot run real Ragged request
```

### P0-3 MRV1 duplicate reshape

C 已经在 MRV2 `gpu/attn_utils.py` 实现正确 Ragged reshape。

MRV1 再写一套只会形成双实现。

---

## P1 — correctness / lifecycle

### P1-1 不能 port `InputBatch.condense()` 逻辑

MRV1-specific。

MRV2 应使用：

```text
req_state_idx + idx_mapping_np
```

### P1-2 active batch 不能 `gather(range(num_reqs))`

MRV2 persistent slots可能为：

```text
req-A -> slot 7
req-B -> slot 2
```

当前 batch：

```text
idx_mapping_np = [7,2]
```

必须按 `[7,2]` gather。

### P1-3 remove 后必须清 mirror slot

否则 slot 被新 request 复用时可能继承旧 rows/version/E。

### P1-4 layer order 必须与 placement authority一致

使用 `kv_cache_group.layer_names` exact order。

### P1-5 禁止 GPU scalar sync 计算 max_kv_len

使用 CPU mirror + q lens。

### P1-6 Ragged actual path 必须绕开 Dense prepare_attn

禁止 empty Dense table 产生 bogus metadata。

### P1-7 dummy/profile 不得创建假 ownership

profiling依赖 `attn_metadata=None -> output.fill_(0)`；保持即可。

### P1-8 Worker commit 必须绑定 forward-entry source

建议保存 `RaggedClusterStepView` 到 `ExecuteModelState`，而不是 sample 时从 Scheduler transport 重新猜。

---

## P2 — 工程/后续性能，不阻塞 E correctness

- 每 step CPU NumPy mirror -> CUDA tensor materialization 有 H2D 成本；
- `[R,C,MaxPages]` mirror 是预分配大矩阵，长 context 可能占用较多 host memory；
- member block table 每 step派生会有 metadata construction cost；
- placement/step metadata未来可 staging/pinned/fused。

这些都不要在 E reconciliation 中优化。

---

# 20. 文件级修改矩阵

| File | Action | 设计 |
|---|---|---|
| `vllm/config/vllm.py` | MODIFY SMALL | 保留 gate；新增 Ragged requires MRV2 fail-fast |
| `vllm/v1/worker/gpu_worker.py` | REVERT | 删除 page_group_size -> MRV1 fallback |
| `vllm/v1/worker/gpu_model_runner.py` | REVERT E RAGGED PARTS | MRV1 不作为正式 Ragged runtime |
| `vllm/v1/worker/gpu/ragged_kv_state.py` | MODIFY SMALL | 删除/停用 `reindex()`；保留 snapshot/delta/commit/gather |
| `vllm/v1/worker/gpu/model_runner.py` | MAIN MODIFY | MRV2 mirror lifecycle、step staging、ForwardContext、success commit |
| `vllm/v1/worker/gpu/attn_utils.py` | KEEP | 继续作为唯一 MRV2 Ragged physical reshape owner |
| `vllm/forward_context.py` | KEEP / HARDEN OPTIONAL | 已有 Ragged fields |
| `vllm/model_executor/layers/attention/attention.py` | KEEP / HARDEN OPTIONAL | 已有 Ragged unified dispatch |
| `vllm/v1/core/kv_cache_manager.py` | KEEP | Scheduler vector reserve/transport producer |
| `vllm/v1/core/sched/scheduler.py` | KEEP | snapshot/delta + canonical success commit |
| `vllm/v1/core/ragged_kv_cache_manager.py` | KEEP | canonical Scheduler state |
| `vllm/v1/attention/backends/ragged_layout.py` | KEEP | D frozen execution metadata |
| `vllm/v1/attention/backends/ragged_forward.py` | KEEP | D frozen write/read |

---

# 21. 推荐 MRV2 内部 helper 边界

为了不把 `gpu/model_runner.py` 写成大量 Ragged-specific inline logic，建议只增加以下窄方法：

```text
_init_ragged_runtime(kv_cache_config)
_apply_ragged_kv_updates(scheduler_output)
_prepare_ragged_step(input_batch)
_commit_ragged_step(input_batch, source_state)
```

不要创建：

- `RaggedRuntimeManager` singleton；
- 第二套 request allocator；
- 第二套 placement registry；
- 第二套 KV cache allocator。

四个 helper 已足够。

如果想进一步减少 `model_runner.py` 行数，可以把纯 state transform helper 放进现有 `gpu/ragged_kv_state.py`，但 request slot lookup / lifecycle hook 仍应由 MRV2 runner 控制。

---

# 22. MRV2 acceptance matrix

E 只有完成以下测试才允许 `PASS / CLOSED`。

## E-MRV2-01 — Runner identity

真实日志必须出现：

```text
Using V2 Model Runner
```

并且 Ragged request 仍能执行。

这是本轮第一 Gate。

---

## E-MRV2-02 — Single prefill + decode

A100：

```text
/data/models/Qwen3-0.6B
BF16
page_group_size=2
max_model_len=128
max_num_seqs=1
gpu_memory_utilization~0.2
async=False
prefix=False
enforce_eager=True
```

证明：

```text
Scheduler snapshot
-> MRV2 mirror
-> StepViews
-> Ragged write/read
-> Worker commit
-> Scheduler commit
```

---

## E-MRV2-03 — Dense/Ragged identity parity

同模型/同 prompt/同 sampling：

```text
Dense MRV2
vs
Ragged MRV2 page_group_size=2
```

使用 greedy：

```text
temperature=0
```

至少 token IDs exact equal。

此前 MRV1 `[3555,525] == [3555,525]` 只能作为 golden oracle；最终必须获得 MRV2 parity。

---

## E-MRV2-04 — Block boundary

覆盖：

```text
15 -> 16 -> 17
```

或等价 B=16 boundary。

验证：

```text
source E
page allocation
Worker E commit
Scheduler E commit
```

一致。

---

## E-MRV2-05 — Mixed step

必须明确制造：

```text
Request A = decode
Request B = newly-arrived prefill
```

在同一 scheduler step。

不是普通“两请求并发”替代。

---

## E-MRV2-06 — Request slot non-contiguous mapping

Focused test人为构造：

```text
req-A -> persistent slot 7
req-B -> persistent slot 2
InputBatch.idx_mapping_np = [7,2]
```

证明 gather 和 commit 都操作 7/2，而不是 0/1。

这是 MRV2 port 的关键 oracle。

---

## E-MRV2-07 — Finish / slot reuse

```text
A req_idx = X
A mirror contains pages/E/version
A finish
-> mirror[X] cleared
B reuses req_idx = X
-> B must start from clean mirror
-> snapshot establishes new state
```

防止 ABA/stale mirror。

---

## E-MRV2-08 — Scheduler page free / reuse

记录 A physical page IDs：

```text
A finish
-> Scheduler canonical free
-> BlockPool free capacity increase
-> B normal allocation
-> naturally reuses >=1 released ID
```

Worker 不参与 free。

---

## E-MRV2-09 — Reserve-before-success

Focused unit：

```text
source E=[32,32]
reserve grows capacity
assert Worker/Scheduler E unchanged
commit success
assert E -> target
```

---

## E-MRV2-10 — Profile startup

真实 Engine startup必须通过 MRV2 profile run。

确认：

```text
dummy profile
attn_metadata=None
no Ragged ownership fabricated
no shape crash
```

---

# 23. Regression

至少：

```text
Ragged manager tests
Ragged worker-state tests
Ragged layout/execution tests
MRV2 RequestState/InputBatch focused tests
MRV2 model runner focused tests
Dense attn_utils regression
Dense Qwen MRV2 smoke
```

静态：

```text
python -m py_compile changed files
git diff --check
ruff --select I,F,E9 changed files
```

---

# 24. Fail-closed conditions

Ragged active 时以下情况必须失败：

```text
MRV1 selected
missing Ragged transport
snapshot request has no RequestState slot
delta request has no RequestState slot
stale version/E/counts
missing layer index
mixed Dense/Ragged group
unsupported parallelism
unsupported prefix/spec/async/CUDA Graph/MLA/sliding/quantized KV
```

禁止 silent Dense fallback。

---

# 25. 本轮明确不做

```text
non-uniform compaction policy
KV importance / eviction scoring
Ragged compaction execution
physical relocation
trailing-page reclaim policy
prefix caching
async scheduler
spec decode
TP/PP/DCP/PCP > 1
CUDA Graph
quantized KV
MLA
sliding window
Triton/custom new kernel
performance tuning
```

E reconciliation 只解决：

> **Production Ragged Identity 在正确的 MRV2 runtime 中闭环。**

---

# 26. Codex 实现顺序

建议一次完整推进，但按以下顺序调试：

```text
Step 1
REVERT MRV1 fallback
+ central MRV2 gate

Step 2
MRV2 initialize Ragged mirror/placement/layer mapping

Step 3
MRV2 apply Scheduler snapshot/delta
+ remove/slot reuse cleanup

Step 4
MRV2 gather via InputBatch.idx_mapping_np
+ build RaggedStepViews

Step 5
MRV2 bypass Dense attn metadata
+ pass Ragged ForwardContext

Step 6
MRV2 success-only Worker commit

Step 7
focused tests

Step 8
A100 MRV2 startup / request / parity / mixed / reuse

Step 9
document + evidence audit
```

不要在 Step 4 失败时重新切回 MRV1。

---

# 27. Code Trace 要求

继续更新：

```text
P1-V2-R1-E-PRODUCTION-RAGGED-IDENTITY-01-Code-Trace.md
```

增加一个明确章节：

```text
E-MRV2-RECONCILIATION
```

记录：

```text
REVERT
- gpu_worker MRV1 fallback
- MRV1 Ragged production wiring
- duplicate MRV1 reshape
- MRV1 condense/reindex dependency

ADD/MODIFY
- MRV2 initialize runtime
- MRV2 transport apply
- MRV2 step staging
- MRV2 ForwardContext producer
- MRV2 success commit

KEEP
- Scheduler canonical control plane
- ForwardContext schema
- Attention unified dispatch
- C/D geometry/execution helpers
```

MRV1 A100 parity evidence保留为：

```text
HISTORICAL / GOLDEN FEASIBILITY ORACLE
```

不得删除，但不得作为 MRV2 E closure evidence。

---

# 28. 最终 closure 条件

只有看到完整链：

```text
MRV2 selected
  -> Ragged physical cache initialized
  -> Scheduler vector reserve
  -> SchedulerOutput snapshot/delta
  -> MRV2 persistent mirror
  -> idx_mapping gather
  -> RaggedStepViews
  -> unified Ragged KV write/read
  -> sampling/postprocess success
  -> Worker vector E commit
  -> Scheduler vector E commit
  -> finish/free/reuse
```

并且：

```text
Dense MRV2 token IDs == Ragged MRV2 token IDs
```

才允许：

```text
P1-V2-R1-E-PRODUCTION-RAGGED-IDENTITY-01
= PASS / CLOSED
```

否则继续：

```text
PASS_PENDING_WEB_REVIEW
```

或：

```text
IN_PROGRESS / BLOCKED_BY_<FACT>
```

---

# 29. 最终设计一句话

本次修改不是“把 MRV1 代码复制进 MRV2”，而是：

> **保留已经成立的 Scheduler-owned Ragged control plane 和 D 阶段 execution substrate，使用 MRV2 自己的 persistent RequestState slot + ephemeral InputBatch.idx_mapping 重新实现 Worker mirror lifecycle；Ragged 只改变 KV physical state/address/execution metadata，不改变 ModelRunner generation。**

这应作为后续 Codex CLI 实现的冻结架构。
