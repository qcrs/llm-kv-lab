# Breadth Engineering — TP=2 + True Preemption — v4

本文件只讨论两个最有简历价值的 breadth：**Tensor Parallel correctness** 与 **serving lifecycle correctness**。它们必须在 Core R3/R4 已经稳定后进入。

---

# 1. TP=2：不是“多卡跑起来”

项目要证明的是：

```text
rank-local KV geometry
+ local member/head-group addressing
+ local retention
+ rank-consistent physical shape transition
+ single scheduler ownership contract
```

## 1.1 Authoritative Local Geometry

使用：

```python
model_config.get_num_kv_heads(parallel_config)
```

不要自己假设 `global_Hkv / TP`。

定义：

```text
Hkv_local = vLLM authoritative local KV heads
Hp = page_group_size
Glocal = Hkv_local / Hp
```

startup：

```text
Hkv_local % Hp == 0
```

## 1.2 Rank-local vs Rank-consistent

### Rank-local

允许每个 rank 不同：

```text
KV head shard
score
keep positions
KV payload
member map within local shard
```

### Rank-consistent

必须 scheduler-visible compatible：

```text
retained group depth
page detach boundary
free/reuse timing
request lifecycle
```

原因：Scheduler 只有一个 request 语义，不能 rank0 已 free、rank1 仍读。

## 1.3 First Supported TP Contract

只做：

```text
TP=2
uniform budget scope
identity or local per-layer grouping
```

不做：

```text
layer/global budget under TP
cross-rank AOT cluster
DCP/PCP
PP
```

## 1.4 Kept-Length Synchronization

如果 local scorer 导致：

```text
rank0 kept=95
rank1 kept=111
```

则 physical depth 取 conservative common boundary：

```text
all_reduce(MAX) → 111
```

然后：

```text
pages = ceil(111/block_size)
```

注意：这不要求保留 positions 一致，只要求 physical capacity boundary compatible。

## 1.5 Debug-only Rank Contract

建议开发阶段 free 前做一次小规模：

```text
all_gather(new_group_counts)
all_gather(detached_page_count)
all_gather(row_geometry_checksum)
```

只作为 debug/assert，不进入性能 hot path。

## 1.6 TP Acceptance

```text
TP2 dense vs ragged identity parity
TP2 uniform compression correctness
rank-local member addressing correct
rank-consistent group depth
scheduler free count exact
request B reuses released pages
request A continues decode
```

TP2 只有全部满足才能写简历。

---

# 2. True Preemption：MRv2-specific Lifecycle

## 2.1 三个不同事件不能混

```text
A. request 本 step 未被调度
B. request 从 worker persistent batch slot移除，但 scheduler仍拥有KV
C. scheduler true preemption：KV free，之后recompute
```

当前项目真正需要标准化的是 C。

## 2.2 Ordinary Unscheduled Step

MRv2 persistent state：

```text
RequestState[req_idx]
RaggedBlockTables[req_idx]
RaggedKVState[req_idx]
```

只要 scheduler 没有 preempt/free，它们都不应该 reset，也不需要 snapshot/restore。

## 2.3 True Preemption

预期链：

```text
Scheduler encounters KV pressure
→ chooses request to preempt
→ canonical KV ownership freed
→ preempted_req_ids emitted
→ worker finish/remove path
→ Ragged state reset
→ request returns waiting
→ later scheduled as recompute
→ state rebuilt from canonical scheduler allocation
```

## 2.4 Worker Reset Surface

必须清：

```text
RaggedBlockTables row/counts
RaggedKVState E[r,g]
compression req state
slot score state
pending plan/result state
per-request scratch handles if any
```

不能清：

```text
logical request token history retained by scheduler/request object
```

## 2.5 No Stale Restore

preemption 之后：

```text
old physical page IDs
old compressed group depths
old slot scores
```

都不再合法。

Resume 必须走 recompute/new allocation，而不是 restore stale row。

## 2.6 Pressure Test

设计一个可重复 workload：

```text
small num_gpu_blocks_override
Req A long prefill/decode
Req B admitted
force pressure
A or B preempted
freed pages observed
preempted request later recomputes
output compared with no-pressure reference
```

## 2.7 Acceptance

```text
preemption event observed
BlockPool ownership released
worker ragged state reset
recomputed row has no stale page IDs
request output correct after resume
no page leak / double free
```

---

# 3. Why These Two Breadth Items Matter

TP=2 证明：

```text
distributed inference
rank-local geometry
collective synchronization
```

Preemption 证明：

```text
serving lifecycle
logical state vs physical state separation
allocator ownership correctness
```

这两个比 PP、更多 scorer、KV connector 更适合作为当前简历项目的 breadth。
