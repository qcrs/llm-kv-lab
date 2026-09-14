# vLLM 0.26 End-to-End Source Chain & Modification Map

本文件是后续源码学习和 debug 的主入口：**任何 bug 先定位属于哪一条链，再看局部代码。**

---

# 1. Chain A — Startup / KV Geometry

```text
VllmConfig
  ├─ ModelConfig
  ├─ CacheConfig
  └─ ParallelConfig
        ↓
model construction
        ↓
Attention.__init__
        ↓
Attention.get_kv_cache_spec(vllm_config)
        ↓
worker attn_utils.get_kv_cache_spec()
        ↓
KVCacheSpec registry / grouping
        ↓
kv_cache_utils.get_kv_cache_groups()
        ↓
get_kv_cache_config_from_groups()
        ↓
KVCacheConfig
        ↓
GPUModelRunner.initialize_kv_cache()
        ↓
init_attn_backend()
        ↓
_allocate_kv_cache()
        ↓
_reshape_kv_cache()
        ↓
bind_kv_cache()
```

## Dense invariant

```text
spec.num_kv_heads = physical storage heads = semantic KV heads
```

## Ragged change

```text
semantic Hkv
!=
storage Hp
```

修改点：

```text
R1-A1 spec
R1-A2 global planner/backing
R1-A3 storage layout
```

最容易错：

```text
page bytes double-count layers
shared backing != shared page namespace
logical shape correct but physical stride wrong
```

---

# 2. Chain B — Scheduler / Allocation / Ownership

```text
EngineCore.step
        ↓
Scheduler.schedule
        ↓
select running / waiting requests
        ↓
compute num_new_tokens
        ↓
KVCacheManager.allocate_slots
        ↓
KVCacheCoordinator.get_num_blocks_to_allocate
        ↓
SingleTypeKVCacheManager.allocate_new_blocks
        ↓
BlockPool
        ↓
KVCacheBlocks / new block IDs
        ↓
SchedulerOutput
```

## Dense assumption to break

```text
one request row depth
```

## Ragged target

```text
request
→ group_flat
→ ordered physical page row
```

需要回答：

```text
谁拥有 canonical row？ Scheduler
谁能 free？ Scheduler side allocator
Worker 能做什么？ payload + worker physical view transition
```

---

# 3. Chain C — SchedulerOutput → Worker Persistent State

```text
GPUModelRunner.execute_model(scheduler_output)
        ↓
finish_requests / preempt remove
        ↓
free_states
        ↓
add_requests
        ↓
update_requests
        ↓
RequestState staged writes
        ↓
BlockTables / RaggedBlockTables staged writes
        ↓
apply_staged_writes
```

Ragged 新增：

```text
RaggedBlockTables
RaggedKVState
```

重要区分：

```text
req_idx = worker persistent slot
request_id = scheduler identity
physical page IDs = allocator ownership
```

---

# 4. Chain D — Input Logical Position / Physical Cache Position

Dense/P1 当前已经建立：

```text
positions        = model/RoPE logical coordinate
cache_positions  = physical append coordinate
```

Ragged 后：

```text
positions[token]
cache_positions[group, token]
```

不要让 logical position 跟随 compaction renumber。

---

# 5. Chain E — Block Table Gather / Slot Mapping

Dense：

```text
persistent BlockTables
        ↓
gather_block_tables(idx_mapping)
        ↓
input_block_tables
        ↓
compute_slot_mappings(cache_positions)
```

Ragged：

```text
persistent rows [R*GT,M]
        ↓
gather active (request,group) rows
        ↓
member/group block views
        ↓
virtual slot mapping
```

slot 公式：

```text
p = E[group] + query_offset
page = row[p // B]
col = member column
vbid = page*Hp + col
slot = vbid*B + p%B
```

---

# 6. Chain F — Attention Metadata

```text
DefaultModelState.prepare_attn
        ↓
build_attn_metadata
        ↓
AttentionGroup metadata builder
        ↓
CommonAttentionMetadata
        ↓
backend-specific metadata
        ↓
set_forward_context
```

Ragged 需要额外 overlay：

```text
member query_start_loc
member seq lens
member virtual block tables
per-layer/member mapping
```

不要在每层 forward 临时 Python 创建大量 GPU tensor；R9 要把这些变 persistent buffers。

---

# 7. Chain G — KV Write

当前 v0.26：

```text
Attention.forward
        ↓
unified_kv_cache_update
        ↓
backend reshape_and_cache_flash / update kernel
```

Ragged adapter：

```text
K/V [T,Hkv,D]
        ↓
member-major [T*Hkv,1,D]
        ↓
virtual slots
        ↓
standard reshape_and_cache_flash
```

这条链先于 attention read closure。

---

# 8. Chain H — Attention Read

```text
Attention.forward
        ↓
unified_attention_with_output
        ↓
FlashAttentionImpl.forward
        ↓
flash_attn_varlen_func
```

Ragged decode：

```text
Q [T,Hq,D]
→ [T*Hkv,Hq/Hkv,D]
```

Ragged prefill：

```text
token-major
→ member-major materialization
→ FA
→ inverse output copy
```

---

# 9. Chain I — Post-Forward Physical Compaction

R3/R4：

```text
keep plan
        ↓
reference/Triton gather
        ↓
payload writeback to group-row prefix
        ↓
new E[group]
        ↓
new group page count
        ↓
worker row tail detach
        ↓
RaggedCompactionResult
```

注意：此时 Worker 还不能把 canonical page 返回 allocator。

---

# 10. Chain J — Scheduler Reconciliation / Free / Reuse

```text
ModelRunnerOutput
        ↓
Scheduler.update_from_output
        ↓
validate result carrier
        ↓
validate plan/source step fence
        ↓
validate expected source group counts
        ↓
prepare reconciliation
        ↓
mutate canonical req→group rows
        ↓
BlockPool.free(...)
        ↓
future KVCacheManager.allocate_slots
        ↓
released block ID reused
```

这条链是项目最重要的 systems evidence。

---

# 11. Chain K — True Preemption

```text
Scheduler pressure
→ preempt request
→ free canonical KV
→ preempted_req_ids
→ worker remove/reset
→ waiting queue
→ later recompute
```

Ragged reset surface：block rows / E / scores / pending plans。

---

# 12. Chain L — TP2

```text
ModelConfig.get_num_kv_heads(parallel_config)
        ↓
rank-local Hkv
        ↓
rank-local groups/member map
        ↓
local scorer/positions
        ↓
collective physical retained-length agreement
        ↓
rank-compatible page depth/free boundary
```

---

# 13. Chain M — Piecewise CUDA Graph

```text
compiled/captured compute
→ split op: unified_kv_cache_update
→ dynamic Ragged write eager
→ split op: unified_attention_with_output
→ dynamic Ragged attention eager
→ compiled/captured compute
```

R9 优化的是：

```text
metadata allocation
pointer stability
CPU staging
launch gaps
```

不是重新实现 torch.compile。

---

# 14. Debug Routing Table

| Symptom | First Chain | First Things to Check |
|---|---|---|
| OOM/startup block count wrong | A | page bytes, layer divisor, shared backing |
| block ID duplication | B/C | scheduler group counts, placement order |
| KV write wrong cell | D/E/G | cache position, slot mapping, stride |
| decode output wrong | H | member Q layout, seq lens, block table |
| prefill only wrong | H | member-major permutation/query_start_loc |
| after compaction next token wrong | I/H | E commit timing, row trim, append position |
| free pages not increase | J | scheduler reconciliation not worker row |
| double free/reuse crash | J/K | source fence, ownership authority |
| only TP2 wrong | L | local Hkv, group matching, MAX sync |
| only CG wrong | M | fresh tensors/pointers, shape class, CPU sync |

这张表后续可以直接作为 debug checklist。


# v5 Critical Chain — Post-Compression Allocation

R3 之后必须额外追踪：

```text
Worker compaction result new_E[GT]/new_counts[GT]
        ↓
Scheduler reconciliation
        ↓
Ragged scheduler-side physical state commit
        ↓
next Scheduler.allocate_slots
        ↓
per-group required_pages[g] = ceil((E[g]+q)/B)
        ↓
allocate only groups crossing page boundary
        ↓
RaggedBlockDelta
        ↓
Worker append exact group rows
```

如果这条链缺失，系统可能在一次 compaction 后看似 free 成功，但下一次 decode 又按 logical/max depth 把短 group 扩回来。
