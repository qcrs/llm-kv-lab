# Ragged Step Metadata — Shape, Lifetime & Attention Contract

**Status:** CANONICAL for R1-C/D1/D2/D3/R9

---

# 1. Goal

避免 Ragged 实现把 metadata 临时散落在：

```text
GPUModelRunner
FA MetadataBuilder
Attention.forward
compression executor
```

统一定义一个 step 内每份 tensor 的：

```text
semantic shape
owner
producer
consumer
lifetime
CG stability requirement
```

---

# 2. Static Geometry

```text
R       = active requests
L       = local attention layers
Hkv     = local KV heads
Hq      = local query heads
Hp      = page_group_size
G       = Hkv/Hp
GT      = L*G
Q       = total scheduled query tokens
M       = max pages per group row
```

Member：

```text
member = (request, layer, kv_head)
```

Attention sequence member 在一个 layer 内通常是：

```text
(request, kv_head)
```

因此每 layer FA sees：

```text
R * Hkv member sequences
```

不是 `R*GT` attention sequences；GT 是 allocator/block-row axis，Hkv 是 member/attention axis。这个 distinction 必须写死。

---

# 3. Persistent State

## Worker ownership table

```text
ragged_block_rows:
[Rmax*GT, M] int32
```

logical：

```text
[Rmax, GT, M]
```

## Worker physical frontier

```text
effective_lens:
[Rmax, GT] int32
```

## Scheduler canonical mirror

对应：

```text
req→E[GT]
req→counts[GT]
req→rows[GT]
```

---

# 4. Step Request Mapping

MRv2 existing：

```text
idx_mapping[batch_req] -> persistent req_idx
query_start_loc[R+1]
positions[Q]
```

Ragged 不复制 request identity；复用以上 mapping。

---

# 5. Group-Major Block Table View

从 persistent rows gather：

```text
step_group_block_table:
[R, GT, M]
```

不建议 Attention 每 layer重新从 persistent table gather。

可以一次 gather，再切：

```text
layer_group_table[layer]
→ [R, G, M]
```

---

# 6. Member Virtual Block Table

一个 group page 包含 `Hp` columns。

对于 layer 内 KV head `h`：

```text
g = h // Hp
col = h % Hp
vbid = physical_page_id * Hp + col
```

构建：

```text
member_block_table[layer]:
[R*Hkv, M]
```

row order freeze：

```text
member_row = request_batch_idx * Hkv + kv_head
```

这样 decode/prefill 共用同一 block-table semantics。

---

# 7. Member Effective Length

allocator frontier：

```text
E_group[R,G]
```

同 group 内 Hp members 共享 page depth，但 scorer 可以在 group 内选择不同 retained token positions；Core physical effective length 首先定义为 group retained width：

```text
member_seq_len(request,h)
= E_group[request, group(h)]
```

如果未来允许同 group 内不同 retained count，则 physical page depth仍由 max member length决定，但 FA `seqused_k` 必须 per-member；Core 首版建议 keep count 在 page group 内对齐，只允许 keep positions 不同。这与 Tangram group-page physical contract更容易一致，也降低 R2/R3复杂度。

---

# 8. KV Write Slot Mapping

Dense：

```text
slot_mapping[Q]
```

Ragged：

```text
member_slot_mapping[Q * Hkv]
```

row/token order freeze：

```text
for token in token-major Q:
  for kv_head in 0..Hkv-1:
      emit member slot
```

对应 K/V reshape：

```text
K,V [Q,Hkv,D]
→ [Q*Hkv,1,D]
```

需要 synthetic oracle 验证每个 element 写到：

```text
physical_page
column
block_offset
```

正确位置。

---

# 9. Decode Q Layout

```text
Qdense [R,Hq,D]
q_per_kv = Hq/Hkv
```

reshape：

```text
[R,Hkv,q_per_kv,D]
→ [R*Hkv,q_per_kv,D]
```

member query start：

```text
[0,1,2,...,R*Hkv]
```

Decode 是最先实现的 read path。

---

# 10. Prefill / Mixed Q Layout

不能简单 reshape。

输入 token-major：

```text
[Q,Hq,D]
```

需要根据 request token span：

```text
query_start_loc
```

构造 member-major：

```text
req0/head0 tokens...
req0/head1 tokens...
...
req1/head0 tokens...
```

输出再 inverse scatter 回 token-major。

Core 接受 materialization；禁止为了 zero-copy 过早引入复杂 fused transpose。

---

# 11. Metadata Ownership

推荐新增：

```text
RaggedStepViews
```

只描述当前 step derived metadata：

```text
step_group_block_tables
member_block_tables_by_layer
member_slot_mapping
member_seq_lens_by_layer
member_query_start_loc
member_max_seq_len
```

它不拥有 canonical state，不修改 allocator。

---

# 12. CUDA Graph Lifetime

R1 eager 可以临时 tensor。

R9 将以下转为 persistent buffers：

```text
member_block_table buffers
member_slot_mapping buffer
member_seq_lens buffer
member_query_start_loc buffer
```

每步只 update contents。

必须记录：

```text
data_ptr before replay
data_ptr after replay
```

不能只验证 output。

---

# 13. Critical Shape Distinction

项目中必须区分三个 axis：

```text
GT = layer × page-group
  allocator / ownership axis

Hkv
  attention member axis per layer

Q×Hkv
  KV write member-token axis
```

混淆这三个 axis 是最容易出现 silent correctness bug 的地方。

---

# 14. Minimal Acceptance

```text
A1 group flatten/unflatten oracle
A2 member row ordering oracle
A3 virtual block table oracle
A4 virtual slot oracle
A5 decode member-major Q oracle
A6 prefill pack/inverse-pack equality
A7 persistent buffer address stability at R9
```
