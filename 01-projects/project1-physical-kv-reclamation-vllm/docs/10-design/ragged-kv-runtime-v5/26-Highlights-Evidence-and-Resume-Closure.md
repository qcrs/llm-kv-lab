# Project Highlights, Evidence & Resume Closure — v4

本项目的简历价值来自 **系统闭环**，不是“实现了多少 Tangram feature”。所有 bullet 必须绑定自己的 evidence。

---

# 1. S-tier Highlights — 必须完成

## Highlight S1 — MRv2-native Ragged KV Representation

从：

```text
request → one block row → all KV heads
```

扩展为：

```text
request
→ layer
→ head-group
→ independent physical page row
```

涉及：

```text
KV spec
capacity planner
global physical pool
MRv2 staged BlockTables
slot mapping
attention metadata
```

Evidence：R1-E real-engine identity。

## Highlight S2 — Virtual Single-Head FlashAttention Adaptation

```text
virtual_block = physical_page*Hp + column
```

在 dedicated stride contract 下 zero-copy 映射为标准 single-head paged cache，从而复用 FA2 kernel，不改 attention math。

Evidence：R1-A3 stride oracle + R1-D2/D3 equality。

## Highlight S3 — Scheduler-Authoritative Physical Reclamation

```text
plan
→ worker payload transform
→ result receipt
→ scheduler source-fence validation
→ canonical ownership trim
→ pool free
```

Evidence：R3 transaction log。

## Highlight S4 — Non-uniform Decision → Real Capacity

必须证明：

```text
K[group A] != K[group B]
→ group page counts differ
→ BlockPool free ↑
→ another request reuses exact page IDs
→ first request continues decode
```

这是项目最强证据。

---

# 2. A-tier Breadth

## TP=2

体现：

```text
Tensor Parallel head sharding
rank-local KV geometry
collective synchronization
rank-consistent physical free boundary
```

Evidence：R6 TP2 identity/compression/reuse closure。

## Preemption

体现：

```text
logical request lifecycle
physical ownership lifecycle
compression state lifecycle
```

Evidence：KV-pressure preempt→recompute→continue。

## Piecewise CUDA Graph

体现：

```text
torch.compile split boundary
dynamic metadata eager island
persistent buffer address
CPU launch optimization
```

Evidence：Dense piecewise vs Ragged eager vs Ragged piecewise profile。

---

# 3. B-tier Optimization

## AOT Per-Layer Clustering

推荐作为默认 optimization，因为直接解决 Ragged physical fragmentation。

Evidence：

```text
per-head ideal
adjacent group
AOT group
physical pages / residual waste / concurrency
```

## Triton In-place

只有 profiler 支持才做。

Evidence：scratch bytes / DRAM / compaction latency。

## Bulk Allocator

只有 CPU allocator pressure 明显才做。

Evidence：alloc/free CPU time / object churn / step CPU time。

---

# 4. Resume Claim → Required Evidence Matrix

| Resume Claim | Earliest Round | Required Evidence | Forbidden Before |
|---|---|---|---|
| Implemented Ragged KV Runtime | R1-E | real model dense/ragged identity | synthetic only |
| Physical KV reclamation | R3 | BlockPool free + exact reuse | row shrink only |
| Non-uniform KV compression runtime | R4-B | automatic unequal group depths + R3 closure | manual plan only |
| TP=2 support | R6 | two-rank correctness + synchronized physical depth | torchrun startup |
| Preemption-safe | R5 | real pressure preempt/recompute | unit reset only |
| CUDA Graph optimized | R9 | TPOT/CPU/launch evidence | no-crash smoke |
| Reduced fragmentation via AOT | R7 | own page-waste ablation | Tangram paper number |
| Triton in-place acceleration | R8 | own microbench/profile | kernel implemented only |

---

# 5. Resume Wording — by Completion Stage

## After R3

> 基于 vLLM 0.26 Model Runner V2 设计实现 Head-Group Ragged KV Cache Runtime，将 KV page 从全 KV heads 的统一 block 细化为 per-head-group physical allocation，并通过 virtual block addressing 复用 FlashAttention paged kernel。

> 建立 per-group physical KV state 与 scheduler-authoritative reclamation transaction，实现 token-level compaction 后真实 GPU page 回收和跨请求 block reuse，并验证原请求持续 decode 正确性。

## After R4-B

> 将 retention policy 与物理 KV runtime 解耦，以单一 deterministic scorer 驱动 per-layer/per-head-group non-uniform retention，闭环 decision → compaction → allocator capacity。

## After TP2

> 扩展 Ragged KV Runtime 至 TP=2，基于 rank-local KV head geometry 构建 head-group paging，并通过 TP collective 对齐压缩后的 scheduler-visible physical free boundary，保证跨 rank page ownership 一致。

## After Piecewise CG

> 将动态 Ragged KV 路径接入 vLLM piecewise torch.compile/CUDA Graph，复用 KV-update / attention splitting seam，通过 persistent metadata buffers 与 eager-island 隔离降低 decode runtime/launch overhead。

## After AOT

> 基于 offline per-head retention profile 实现 AOT head clustering，降低 head-group max-pool fragmentation，并量化 physical page、并发容量与 serving throughput 变化。

---

# 6. 90-second Interview Story

1. vLLM PagedAttention 解决 sequence-level fragmentation，但一个 physical page 仍绑定一个 layer 内全部 KV heads。
2. non-uniform retention 若不同 heads 保留长度不同，Dense page 仍无法独立释放。
3. 我把 page 改成 head-group page，并给 `(request,layer,group)` 独立 block row。
4. 用 `(physical_page,column)` → virtual single-head block 复用 FA2 paged kernel。
5. v0.26 MRv2 已将 KV write/read 分开，因此分别重建 Ragged KV update 与 member-major attention read。
6. Worker 只做 payload transform，Scheduler 仍是 allocator authority；通过 source fence 后才真正 free。
7. 最后用另一请求复用 released page，并验证原请求继续 decode。
8. Breadth 再扩展 TP2 / Preemption / Piecewise CG。

---

# 7. Final Evidence Tree

```text
EVIDENCE/
  R0-baseline/
  R1-identity/
    geometry.txt
    stride-oracle.txt
    synthetic-write.txt
    decode-equality.txt
    prefill-equality.txt
    engine-parity.jsonl
  R3-reclaim-reuse/
    ownership-before.txt
    compaction-result.txt
    ownership-after.txt
    free-pages.txt
    reuse-proof.txt
    continuation.txt
  R4-policy/
  R5-preemption/
  R6-tp2/
  R7-aot/
  R8-triton/
  R9-cudagraph/
  R10-benchmark/
```

面试时任何 claim 都能落到具体证据，而不是口头描述。
