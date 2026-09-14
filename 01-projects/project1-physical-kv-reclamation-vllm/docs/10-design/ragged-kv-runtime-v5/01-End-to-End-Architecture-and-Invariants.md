> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# End-to-End Architecture and Invariants

## 1. 先看两套系统的基本差异

### vLLM 0.26 / P1 当前抽象

当前 MRv2 的关键数据面是：

```text
Scheduler
  ↓ block IDs
RequestState / InputBatch
  ↓
BlockTables
  persistent: [kv_group][request, blocks]
  ↓ gather
input_block_tables
  ↓
slot_mapping[kv_group, token]
  ↓
CommonAttentionMetadata
  ↓
FlashAttention
```

P1 在这条链路上额外引入：

```text
num_computed_tokens = logical progress

effective_kv_len = physical retained length

positions       = logical coordinate
cache_positions = physical write coordinate
```

这套拆分必须保留，因为 non-uniform runtime 本质上仍然需要“模型位置”和“物理 KV 位置”解耦。

### Tangram 抽象

Tangram 将一个普通 KV page：

```text
[block_size × all KV heads]
```

拆成：

```text
[block_size × page_group_size heads]
```

并把 request block row 从：

```text
[request, block]
```

提升为：

```text
[request, layer × head_group, block]
```

因此不同 `(layer, group)` 可以拥有不同 page depth。

核心变化不是 scorer，而是 representation。

---

## 2. Identity Ragged Paging 的 memory invariant

设：

- `B` = block_size
- `Hkv` = KV heads per rank
- `Hp` = page_group_size
- `D` = head_size
- `S` = dtype bytes
- `G = Hkv / Hp`

Dense page：

```text
P_dense = 2 × B × Hkv × D × S
```

Ragged page：

```text
P_ragged = 2 × B × Hp × D × S
```

每个 token-range 需要 `G` 个 ragged pages：

```text
G × P_ragged = P_dense
```

所以 Identity 模式下理论总 KV bytes 不应因为 Ragged 本身下降：

```text
smaller page × more pages = same bytes
```

这条 invariant 是 R1 的根。任何“开启 ragged 立刻省内存”的结果都应该先被视为 accounting bug，而不是性能收益。

---

## 3. Group flatten 约定

推荐 v0.26 统一使用：

```text
group_flat = local_layer_id * groups_per_layer + group_in_layer
```

于是：

```text
G_total = num_local_layers × groups_per_layer
```

worker physical state：

```text
block_table[request, group_flat, block_slot]
effective_len[request, group_flat]
slot_mapping[group_flat, scheduled_token]
```

不要让不同模块发明自己的 flatten 顺序；`RaggedLayoutSpec` 应作为共享 contract。

---

## 4. 为什么建议 MRv2-native flattened row

Tangram 的旧 `RaggedBlockTable` 本身是 3D：

```text
[max_requests, G_total, max_blocks]
```

v0.26 `BlockTables` 已经有成熟的：

- `StagedWriteTensor`
- `FusedStagedWriter`
- persistent GPU pointer table
- CUDA-Graph-safe persistent `input_block_tables`
- Triton gather
- Triton slot mapping

因此更好的迁移方式是把 ragged request/group 看成“虚拟 row”：

```text
flat_row = request_idx * G_total + group_flat
```

physical storage：

```text
[max_requests * G_total, max_blocks_per_group]
```

逻辑 view：

```text
[max_requests, G_total, max_blocks_per_group]
```

这样可以最大化复用 MRv2 row-oriented staging / gather primitive，而不是把旧 Worker V1 的 BlockTable class 整体移植回来。

### 该设计的关键验证点

- request swap/move 是否需要同时移动连续 `G_total` rows；
- `num_blocks` 必须变为每 `(request, group)` 一份；
- gather kernel 的 batch axis 需要映射为 `(batch_request, group)`；
- slot mapping kernel 需要接受 group-specific physical position；
- padded/capture rows 必须保证地址稳定。

---

## 5. Physical position 与 logical position

Ragged compression 后：

```text
logical position: 0 ... original sequence position
physical member:  0 ... retained_length(group)-1
```

它们不能再用同一个 scalar coordinate 表示。

必须坚持：

```text
positions
= model / RoPE / causality 的逻辑位置

cache positions / slot mapping
= physical compacted KV 的位置
```

P1 已经建立这一拆分，所以这里是高价值 reuse，而不是推翻。

---

## 6. Ragged Attention 的真正 sequence abstraction

标准 attention：

```text
one request = one varlen sequence
```

Tangram-style ragged attention：

```text
one (request, KV head) = one varlen sequence
```

每个 KV head 对应它的 GQA query-head bundle。

例如本 rank 有 4 KV heads，2 requests：

```text
R0/H0
R0/H1
R0/H2
R0/H3
R1/H0
R1/H1
R1/H2
R1/H3
```

FlashAttention 最终看到 8 个 varlen sequences。

### 为什么不用改 FA kernel

物理 page 中一个 `physical_page` 有 `Hp` 个 column。对 member/head 可以构造：

```text
virtual_block_id = physical_block_id * Hp + column
```

从 FA 的视角，每个 virtual block 仍是一个标准 single-KV-head paged block。

这就是 Head Group Page 的关键价值：

> 改 physical placement 和 metadata，不必发明新的 attention math kernel。

---

## 7. Decode 与 Prefill 必须分开理解

### Decode

`q_len=1` 时：

```text
[token, kv_head, ...]
```

可以直接 reshape 为 member-major；这是 ragged attention 最容易先跑通的路径。

### Prefill / Mixed

不同 request query length 不同，token-major Q/K/V 无法通过一个 view 变成所有 member contiguous sequence。

需要：

```text
token-major Q/K/V
 → reshape
 → permute
 → contiguous/member-major
 → FA2
 → inverse copy to token-major output
```

因此只跑通 decode 不能宣布 R1 完成。

---

## 8. Compression transaction

真正系统闭环必须严格分层：

```text
Scorer
  ↓ scores
BudgetScope
  ↓ per-(layer,group) count
Keep Decision
  ↓ retained positions
CompressionExecutor
  ↓ payload compaction
RaggedBlockTable trim
  ↓ freed physical IDs
ModelRunnerOutput
  ↓
Scheduler reconciliation
  ↓ canonical ownership update
BlockPool free
  ↓
new request reuse
```

### ownership invariant

Worker 可以发现哪些 page 已经 physically detached，但 Scheduler 必须是 request→block ownership 的 canonical source of truth。

这正是 P1 最值得复用的部分。

---

## 9. TP invariant

TP 下每个 rank 保存自己的 KV-head shard。

允许：

```text
rank0 local heads keep positions A/B/C
rank1 local heads keep positions D/E/F
```

但如果 Scheduler 给各 rank 一个共享的 logical block-allocation decision，则 physical group depth/free boundary 需要一致。

最低风险方案：

```text
local keep decision
 → local kept length
 → TP all_reduce(MAX) kept length
 → rank-local writeback to common physical depth
 → every rank frees same logical page slots
```

R6 只实现 `uniform` budget scope，避免跨 rank score threshold / cluster-member aggregation。

---

## 10. Preemption invariant

必须区分：

### Persistent-batch row removal

请求只是本 step 没在 active persistent batch，但 KV ownership 仍在。

MRv2 persistent row 在 ordinary unscheduled step 中继续存在，因此 Core 不需要 snapshot/restore；只有未来 worker 主动丢弃但 scheduler 仍持有 KV 的特殊 lifecycle 才需要 exact snapshot。

### Scheduler preemption

请求 KV 被真正 free，之后通过 recomputation 恢复。

此时不能 restore 旧 physical row；必须 reset：

- effective group lengths
- compressor state
- retained-score state
- cached keep lengths
- transient block-table snapshot

这是 R5 true-preemption 的核心 correctness distinction。

---

## 11. CUDA Graph invariant

v0.26 MRv2 已经以 `unified_attention_with_output` 为 attention eager-break / piecewise compilation seam。

因此目标不是 Full CUDA Graph，而是：

```text
Captured graph piece
  ↓
Eager Ragged Attention island
  ↓
Captured graph piece
```

Ragged runtime 动态 metadata 不进入 full graph；稳定的 GEMM/FFN/RMSNorm/Projection 继续 capture。

需要优化的是 eager island 的尺寸、metadata allocation、H2D sync，而不是强行消灭 graph break。

---

# v2 Addendum — 新增四个一级不变量

1. **Global Page Namespace Invariant**：Ragged block ID表示一个真实 `(any local layer, any head-group)` page，整个 worker只有一个 page-ID namespace。
2. **Write/Read Ordering Invariant**：current-step K/V必须经 `unified_kv_cache_update` 完成后，attention read才能消费；Ragged不得破坏 MRv2 dummy-dependency顺序。
3. **Zero-Copy Virtualization Invariant**：`virtual_block=p*Hp+c` 必须是 storage view，不是 per-step KV copy；首版采用 ragged-specific column-major packed-content layout。
4. **Authority Invariant**：worker payload/state transformation成功不等于 allocator free；只有 Scheduler canonical reconcile 后 physical page才算 released。

详见 `13-...KV-Write-Seam.md` 与 `14-...Transaction-Protocol.md`。

