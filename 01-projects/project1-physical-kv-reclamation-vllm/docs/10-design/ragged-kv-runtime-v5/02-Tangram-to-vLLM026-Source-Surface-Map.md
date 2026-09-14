# Tangram → qcrs/vLLM 0.26 Source Surface Map — v4

本文件不是“文件名对照表”，而是后续实现的 **semantic port map**。每个 surface 都回答四件事：Tangram 做什么、qcrs 当前事实是什么、我们怎么做、何时允许修改。

## 1. Porting Strategy Legend

```text
PORT       = 机制与实现结构都可大量复用
ADAPT      = 保留语义，按 MRv2 API 改写
REDESIGN   = Tangram语义有用，但当前 v0.26 必须重建
REUSE-P1   = 直接继承当前 P1 已验证 contract/oracle
DELAY      = Core之后再做
SKIP       = 当前项目不做
FREEZE-OUT = startup明确拒绝
```

## 2. Complete Surface Matrix

| # | Tangram Surface | Tangram Semantic | qcrs/v0.26 Target | v4 Action | Round |
|---:|---|---|---|---|---|
| 1 | `CacheConfig.page_group_size` | Ragged enable/geometry | `vllm/config/cache.py` | ADAPT | R1-A1 |
| 2 | `RaggedAttentionSpec` | page bytes / group geometry | `v1/kv_cache_interface.py` | ADAPT | R1-A1 |
| 3 | compression config validation | reject incompatible paths | config + attention spec | ADAPT | R1-A1 |
| 4 | ragged `get_num_blocks` | global page count | `core/kv_cache_utils.py` | ADAPT | R1-A2 |
| 5 | shared `KVCacheTensor` | one global physical backing | `get_kv_cache_config_from_groups` | PORT/ADAPT | R1-A2 |
| 6 | ragged cache shape | column-major page | `_reshape_kv_cache` / backend storage | REDESIGN | R1-A3 |
| 7 | `as_virtual_block_view` | `(page,col)→virtual block` | new `ragged_layout.py` | PORT/ADAPT | R1-C |
| 8 | old `RaggedBlockTable` | per-(req,group) rows | new MRv2 `RaggedBlockTables` | REDESIGN | R1-B |
| 9 | ragged slot mapping | group/member physical write slots | MRv2 block-table kernels | REDESIGN | R1-C/D1 |
| 10 | `ragged_forward` decode | member-major FA | existing attention seam | ADAPT | R1-D2 |
| 11 | `ragged_forward` prefill | token→member materialization | existing attention seam | ADAPT | R1-D3 |
| 12 | Tangram integrated KV write | member KV scatter | `unified_kv_cache_update` | REDESIGN | R1-D1 |
| 13 | metadata overlays | per-layer/member seq view | metadata builder / forward context | ADAPT | R1-D2/D3 |
| 14 | per-group effective len | physical state | new `RaggedKVState` | REDESIGN | R2 |
| 15 | compression executor | plan/writeback | new minimal control plane | SELECTIVE | R4 |
| 16 | Torch writeback | correctness backend | P1 reference + Tangram reference | REUSE-P1 | R3 |
| 17 | Triton in-place writeback | optimized paged→paged | compression module | DELAY | R8 |
| 18 | BudgetScope uniform | fixed per-group count | R4 control plane | PORT | R4/R6 |
| 19 | layer/global scope | non-uniform threshold | optional TP1 | DELAY | R4-B |
| 20 | KeyDiff or simple scorer | positions to retain | one scorer only | SELECTIVE | R4 |
| 21 | FastKVZip | checkpoint/gated scorer | none | SKIP | — |
| 22 | AOT clustering | reduce max-pool waste | tooling + member map | ADAPT | R7 |
| 23 | TP kept-length MAX | physical boundary sync | `torch.distributed` | PORT/ADAPT | R6 |
| 24 | custom ragged op | piecewise split seam | existing v0.26 ops | SKIP FIRST | R9 optional |
| 25 | piecewise CG | dynamic metadata isolation | native compilation config | REUSE | R1-F/R9 |
| 26 | old row snapshot/restore | old persistent-batch lifecycle | MRv2 persistent row | MOSTLY SKIP | R5 |
| 27 | preemption reset | recompute lifecycle | MRv2 preempt/remove | ADAPT | R5 |
| 28 | ring allocator | high page-count CPU optimization | `block_pool.py` | DELAY | post-R7 |
| 29 | prefix-cache disable | mutable compressed layout | startup gate | PORT | R1-A1 |
| 30 | DCP/PCP | context-parallel geometry | unsupported | FREEZE-OUT | — |
| 31 | MLA/SWA/hybrid | different KV semantics | unsupported | FREEZE-OUT | — |
| 32 | quantized KV | packed/scaled physical layout | unsupported | FREEZE-OUT | — |

## 3. qcrs/v0.26 End-to-End Mapping

### 3.1 Startup Geometry

```text
CacheConfig
→ Attention.get_kv_cache_spec
→ RaggedAttentionSpec
→ get_kv_cache_spec
→ KVCacheGroupSpec
→ get_kv_cache_config_from_groups
→ KVCacheConfig
→ init_attn_backend
→ initialize_kv_cache
→ _allocate_kv_cache
→ _reshape_kv_cache
→ bind_kv_cache
```

Primary files：

```text
vllm/config/cache.py
vllm/model_executor/layers/attention/attention.py
vllm/v1/kv_cache_interface.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/worker/gpu/attn_utils.py
vllm/v1/worker/gpu/model_runner.py
```

### 3.2 Scheduler / Ownership

```text
Scheduler.schedule
→ KVCacheManager.allocate_slots
→ KVCacheCoordinator
→ SingleTypeKVCacheManager
→ BlockPool
→ KVCacheBlocks / RaggedBlockDelta
→ SchedulerOutput
```

Ragged 必须显式拥有：

```text
request
→ group_flat
→ ordered physical page row
```

不要让“一个 flat id list”成为没有 counts/geometry 的隐式协议。

### 3.3 Worker Staging

```text
GPUModelRunner.execute_model
→ finish/free/add/update request state
→ apply RequestState staged writes
→ apply RaggedBlockTables staged writes
→ gather active rows
→ compute member/group slot mappings
→ prepare_attn
```

### 3.4 KV Write

```text
QKV projection / K,V generated
→ unified_kv_cache_update
→ dense/ragged dispatch
→ member-major K,V
→ virtual slots
→ reshape_and_cache_flash
```

### 3.5 Attention Read

```text
unified_attention_with_output
→ dense/ragged dispatch
→ member-major Q
→ virtual block table
→ member seq lens
→ flash_attn_varlen_func
→ inverse output layout
```

### 3.6 Reclamation

```text
CompactionPlan
→ Worker compaction
→ RaggedCompactionResult
→ Scheduler validates source fence/counts
→ canonical rows trim
→ BlockPool free
→ future allocate_slots reuses page
```

## 4. Files That Should Stay Small

### `gpu/model_runner.py`

只保留 orchestration hook：

```text
maybe_create_ragged_runtime
ragged_runtime.prepare_step
ragged_runtime.post_forward_compaction
```

复杂逻辑放新模块，避免把 central runner 变成 Tangram mixin 的复制品。

### `attention.py`

只在现有：

```text
unified_kv_cache_update
unified_attention_with_output
```

附近做最小 dispatch。

### `scheduler.py`

R1 不加 compression/free；R3 才引入 per-group compaction receipt/reconciliation。

## 5. New Modules Recommended

```text
vllm/v1/worker/gpu/ragged_block_table.py
vllm/v1/worker/gpu/ragged_kv_state.py
vllm/v1/worker/gpu/ragged_runtime.py
vllm/v1/attention/backends/ragged_layout.py
vllm/v1/attention/backends/ragged_forward.py
vllm/v1/core/ragged_kv_cache_manager.py   # if separate manager is cleaner
vllm/v1/attention/compression/...          # from R4 only
```

## 6. Surfaces That Must Not Be Mixed

```text
Dense/P1 RequestState.effective_kv_len
!=
RaggedKVState.effective_kv_lens
```

```text
Dense BlockTables
!=
RaggedBlockTables
```

```text
semantic Hkv
!=
storage Hp
```

```text
logical model position
!=
physical cache position per group
```

这四个 separation 是避免后续 `if ragged` 污染整个 vLLM data plane 的关键。
