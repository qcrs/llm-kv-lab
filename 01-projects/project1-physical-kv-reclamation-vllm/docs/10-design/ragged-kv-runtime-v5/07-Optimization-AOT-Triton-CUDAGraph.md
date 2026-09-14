# Optimization Program — AOT Clustering / Triton Writeback / Piecewise CUDA Graph — v4

优化顺序不是“实现难度”，而是 **简历价值 × 主线相关性 × 可测性**：

```text
AOT per-layer clustering
> Piecewise CUDA Graph hardening
> Triton in-place if profiler says material
> Bulk allocator if profiler says material
```

---

# 1. AOT Head Clustering — Capacity Optimization

## 1.1 Problem

一个 Head-Group page 的 physical depth 由组内 retained length 最大值决定：

```math
pages(group) = ceil(max_i K_i / block_size)
```

因此相邻 grouping 如果混合：

```text
head A retained 90%
head B retained 30%
```

B 仍然随 A 占用接近 90% page depth。

这不是 scorer 质量问题，是 placement / fragmentation 问题。

## 1.2 First Scope

只做 per-layer clustering：

```text
within each layer:
Hkv heads → groups of Hp similar-retention heads
```

不先做 cross-layer cluster，因为那会重新打开 layer ownership / TP shard mapping / physical placement。

## 1.3 Pipeline

```text
page_group_size=1
→ pilot workload
→ retention profile [L,Hkv]
→ aggregate / stability check
→ cluster similar heads
→ cluster_of / column_of
→ runtime member map
→ benchmark physical pages
```

## 1.4 Map Invariants

```text
bijection: every (layer,head) maps to one (cluster,column)
each cluster exactly Hp members
column range [0,Hp)
all runtime local heads covered
```

## 1.5 Metrics

必须比较：

```text
Per-head Ideal
Adjacent Grouping
AOT Grouping
```

报告：

```text
allocated pages
ideal pages
residual waste
KV bytes
max concurrency
request/token throughput
```

不要直接引用 Tangram 的具体百分比作为自己的结果。

---

# 2. Piecewise CUDA Graph — Runtime/CPU Optimization

## 2.1 Existing v0.26 Asset

当前 v0.26 已经有：

```text
CUDAGraphMode.PIECEWISE
unified_attention_with_output splitting
unified_kv_cache_update splitting/exclusion
persistent dummy block-table/slot buffers
```

因此首版 Ragged 不需要新增 custom op 只为了 CG。

## 2.2 First Architecture

```text
Captured compute piece
→ eager Ragged KV update
→ eager Ragged attention metadata/read
→ captured compute piece
```

目标是 dynamic Ragged state 不破坏其余大块模型 compute 的 capture。

## 2.3 R1-F Smoke

只验证：

```text
Ragged Identity + Piecewise CG correctness
```

失败不阻塞 Core；记录原因。

## 2.4 R9 Hardening

需要：

```text
RaggedStepBuffers
persistent_member_block_tables
persistent_virtual_slots
persistent_member_seq_lens
persistent_member_query_start_loc
persistent_group_effective_lens
```

CPU metadata 优先：

```text
pinned memory / UVA
staged copy
no per-layer Python tensor allocations
```

## 2.5 Dangerous Patterns

```text
torch.empty every layer/step
Python list → GPU tensor every step
fresh block-table tensors
fresh seq-len tensors
GPU→CPU .item in hot path
shape/data-dependent Python branch inside capture region
```

## 2.6 Compare

```text
Dense + Piecewise CG
Ragged eager
Ragged + Piecewise CG
```

Metrics：

```text
TPOT
decode token/s
CPU step time
metadata build time
CUDA launch gaps
attention eager-island time
```

只有实测改善才能写“降低 launch/CPU overhead”。

---

# 3. Triton In-place Writeback — Data Movement Optimization

## 3.1 Reference First

R3/R4 继续使用：

```text
PyTorch/reference gather
or P1 Triton gather
→ independent scratch
→ writeback
```

这是 correctness oracle。

## 3.2 In-place Safety

对 sorted keep positions：

```math
s_j >= j
```

destination 不会位于 source 之后。

分类：

```text
source >= kept
→ safe parallel move

source < kept
→ endangered prefix
→ sequential read-before-write tiles
```

最后再 zero-pad trailing slots。

## 3.3 Only Implement If

profile shows：

```text
compaction boundary latency material
or scratch allocation/memory traffic material
```

## 3.4 Metrics

```text
scratch bytes
bytes moved estimate
kernel latency
GB/s
compression-boundary latency
TPOT/end-to-end impact
```

若 e2e 影响 <1%，把结果定位成 kernel engineering，不夸大 serving speedup。

---

# 4. Bulk int32 Ring Allocator — Optional CPU Optimization

Ragged page 变小以后 page count 大增，可能造成：

```text
KVCacheBlock object churn
Python list operations
free-queue overhead
GC pressure
```

Tangram 用 int32 bulk allocator/ring 规避这些成本。

本项目必须先用当前 allocator 做 Core correctness；只有 profile 显示 CPU allocator pressure 才 port。

Metrics：

```text
allocate/free CPU time
Python object count
step CPU time
throughput under high concurrency
```

---

# 5. Optimization Stop Rule

任何优化都必须有：

```text
hypothesis
baseline profile
implementation
micro evidence
end-to-end evidence
trade-off / negative result
```

没有 profiler hypothesis 的优化不进入主线。
