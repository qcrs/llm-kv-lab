> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Proposed Data Structures, APIs & Ownership — v5 Canonical Addendum

这不是最终代码签名，而是实施前应冻结的接口边界。目的：避免所有 ragged 逻辑散到 `GPUModelRunner` / generic manager。

## 1. `RaggedAttentionSpec`

职责：只描述 static geometry。

```python
RaggedAttentionSpec(
    block_size,
    num_kv_heads,      # semantic local Hkv
    head_size,
    dtype,
    page_group_size,   # storage Hp
    ... v0.26 required AttentionSpec fields
)
```

派生：

```text
num_head_groups_per_layer = Hkv/Hp
page_size_bytes = 2*B*Hp*D*dtype_size
```

不保存 runtime rows / lengths。

## 2. `RaggedGeometry`

建议新增不可变 helper：

```text
num_layers_local
num_kv_heads_local
page_group_size
num_groups_per_layer
num_groups_total
block_size
head_size
```

提供：

```text
flat_group(layer,g)
layer_group(flat_group)
member(layer,kv_head)
identity_cluster/member column
```

所有模块共用，避免重复算 index。

## 3. `RaggedBlockTables`

职责：worker execution row，不拥有 allocator free authority。

建议接口：

```text
add_request(req_idx, group_block_ids)
append_pages(req_idx, per_group_new_ids)
remove_request(req_idx)
move_request(src_idx,dst_idx)

get_group_row(req_idx,flat_group)
get_counts(req_idx)

build_group_slot_mapping(...)
get_active_block_table_views(...)

trim_tails(req_idx,new_counts) -> candidate_detached_ids
```

注意：`trim_tails` 返回的是 worker看到的 detached IDs，不等于 allocator已经 free。

## 4. `RaggedKVState`

职责：physical sequence length execution state。

```text
effective_kv_lens_cpu [R,GT] int32
persistent GPU/UVA staging as needed
```

接口：

```text
initialize_request(req,E0)
prepare_step(req_indices,q_lens)
commit_compaction(req_idx,new_E)
reset(req_idx)
```

逻辑 progress继续由原 `RequestState.num_computed_tokens` 管理。


## 4A. `RaggedSchedulerPhysicalState` — v5 REQUIRED

这是 v5 新增的关键 canonical state。Ragged non-uniform compaction 后，下一步 allocator 不能继续只看 logical `request.num_computed_tokens`，否则不同 group 会被重新按统一 logical depth 扩容。

建议不要把 vector state 塞进 generic `Request`，而是由 scheduler-side Ragged KV manager 持有：

```text
req_to_effective_lens[req_id] -> int32 [GT]
req_to_group_counts[req_id]   -> int32 [GT]
req_to_group_block_ids        -> canonical ordered rows
```

它是 physical allocation 的 authority；Worker `RaggedKVState` 只是 execution mirror。

每一步追加 `q` 个 token 时：

```text
required_E[g]     = E[g] + q
required_pages[g] = ceil(required_E[g] / B)
need[g]           = max(required_pages[g] - count[g], 0)
```

allocator 只申请 `sum(need[g])` 个新 page，再 deterministic scatter 到对应 group row。

这使 non-uniform depth 在后续 decode 中保持 non-uniform，而不是被下一次 allocation 重新拉齐。

Tangram 当前使用 `compress_max_eff_seq_len` 作为 post-compression allocation 的 scalar upper bound；本项目 v5 选择更严格的 per-group scheduler state，因为目标是证明长期 non-uniform physical ownership，而不仅是一次压缩后的保守容量。

## 5. `RaggedLayout`

纯函数，不有状态：

```text
as_virtual_cache_view(kv_cache, Hp)
member_virtual_block_table(...)
member_virtual_slots(...)
member_seq_lens(...)
identity_member_maps(...)
```

这里必须尽量接近 Tangram，方便直接 differential testing。

## 6. `RaggedAttentionRuntime`

这是 ModelRunner-facing façade，避免主 runner膨胀。

职责：

```text
init geometry/state/tables
consume scheduler block updates
prepare per-step ragged metadata
provide per-layer virtual write slots
apply post-forward compaction result
handle remove/preempt lifecycle
```

`GPUModelRunner` 理想只出现极少 hook。

## 7. Attention adapter

分 write/read：

```text
ragged_kv_cache_update(...)
ragged_attention_forward(...)
```

不要让 `ragged_attention_forward` 同时拥有 scheduler state mutation。

## 8. Scheduler/Manager ownership

### Scheduler authority

拥有：

```text
physical page ownership
canonical request→group rows
free/reuse
preemption
```

### Worker authority

拥有：

```text
payload bytes
execution block-table view
per-group E
member metadata
```

### Policy

只产生 decision，不拥有 pages。

## 9. Compaction plan/result

建议从 P1 scalar扩为 vector：

```text
RaggedCompactionPlan
- request_id
- step_seq/source_version
- expected_source_E[GT]
- expected_source_counts[GT]
- keep representation

RaggedCompactionResult
- request_id
- source version
- new_E[GT]
- new_counts[GT]
```

尽量不要在 transport里发送完整 freed IDs；Scheduler自己从 canonical row + new_counts 推导，这与 P1 authority保持一致。

## 10. Allocation API shape

Tangram 的 manager一次按 `num_required_blocks*num_groups` 申请，首版 qcrs 可以借这个思想，但长期更清晰的是：

```text
required_count[g] = ceil(target_physical_len[g]/B)
need[g] = max(required_count[g] - current_count[g], 0)

pool.allocate(sum need)
→ deterministic scatter into group rows
```

R1 identity时所有 group target相同；R2以后自然支持 non-uniform append。

## 11. Null page

R1/R2 FullAttention-only 不需要 sliding-window front null semantics。保留 BlockPool null page约定，但不要提前把 Tangram sliding-window nulling引入 Core。

## 12. Prefix cache

Ragged Core禁用 APC，因为：

```text
hash chain assumes token/block structure
ragged groups can have different retained members/depths
in-place payload compaction mutates page content identity
```

以后若研究 reuse，需要独立设计“compressed-state identity”，不是这次项目目标。
