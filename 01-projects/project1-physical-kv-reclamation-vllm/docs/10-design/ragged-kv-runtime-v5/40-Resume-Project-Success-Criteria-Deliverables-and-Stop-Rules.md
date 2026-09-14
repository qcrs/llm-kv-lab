# Resume Project Success Criteria, Deliverables & Stop Rules

**Status:** CANONICAL project-management contract

---

# 1. What Counts as a Successful Project

项目成功不是“把 Tangram feature 全搬过来”。

最低强闭环：

```text
Ragged Identity representation works
+ unequal per-group physical frontier works
+ manual compaction physically frees pages
+ another request reuses them
+ original request continues decode
```

这已经足以成为真实 AI Infra 简历项目。

旗舰闭环：

```text
Core
+ one real policy
+ preemption
+ TP=2
+ piecewise CG compatibility/hardening
+ one measured optimization
```

---

# 2. Deliverables by Tier

## Tier 1 — Systems Core

必须产出：

```text
architecture.md
source-chain.md
R1 identity evidence
R2 unequal-depth evidence
R3 free/reuse evidence
correctness tests
```

## Tier 2 — Policy Closure

```text
one scorer integration
quality sanity result
policy/runtime separation doc
```

## Tier 3 — Breadth

```text
preemption pressure trace
TP2 trace
piecewise CG trace
```

## Tier 4 — Optimization

优先 AOT：

```text
profile.npz
cluster map
fragmentation report
before/after serving benchmark
```

---

# 3. Resume Claim Gate

| Claim | 必须拥有的 evidence |
|---|---|
| implemented Ragged KV runtime | real-engine identity + source diff |
| physical reclamation | free-count + exact page IDs + reuse |
| non-uniform allocation | unequal group counts survive future decode |
| TP=2 | 2-rank run + physical-boundary consistency |
| CUDA Graph optimization | eager vs piecewise performance data |
| AOT reduces fragmentation | own measured pages/waste, not Tangram number |
| Triton improves compaction | microbenchmark + profiler |

没有对应 evidence 就降低措辞。

---

# 4. Recommended Resume Story

标题：

```text
Non-Uniform Ragged KV Cache Runtime for vLLM 0.26
```

一句话：

> 基于 vLLM Model Runner V2 重构 per-layer/per-head-group KV physical runtime，通过 Head-Group Paging 和 virtual block addressing 支持非均匀 KV retention 的真实 GPU page reclamation，并扩展 scheduler ownership、FlashAttention、TP 与 piecewise CUDA Graph。

---

# 5. Stop Rules

为了避免项目无限扩展，满足下列条件就停止某方向：

### Scorer

```text
1 个真实 scorer闭环后停止
```

### TP

```text
TP2 uniform闭环后停止
```

不做 layer/global distributed policy。

### CUDA Graph

如果 piecewise compatibility PASS 但性能提升不显著：

```text
保留 compatibility claim
停止继续压 Full CG
```

### Triton

如果 compaction <1% e2e：

```text
可以保留 micro kernel demo
不再花大量时间 e2e 优化
```

### Bulk Allocator

如果 CPU profiler 中 allocator/object overhead 不显著：

```text
不 port ring buffer
```

---

# 6. When to Abandon a Slice

任何 Slice 出现：

```text
requires opening >2 frozen unsupported subsystems
or
cannot produce independent correctness oracle
or
requires changing Dense behavior globally
```

应标记：

```text
DESIGN_CONFLICT
```

回到前一 Gate，不硬顶。

---

# 7. Final Benchmark Set

控制规模，不做论文复现：

```text
B0 Dense eager/piecewise
B1 Ragged identity eager
B2 Ragged identity piecewise
B3 manual nonuniform
B4 one policy
B5 preemption pressure
B6 TP2 uniform
B7 AOT clustered
```

模型优先：

```text
small Qwen for correctness/debug
one larger model for capacity/performance if resources allow
```

---

# 8. Final Interview Assets

最终至少准备：

```text
1 张 architecture diagram
1 张 allocation/reclamation lifecycle diagram
1 张 TP state diagram
1 张 CUDA Graph eager-island diagram
1 张 benchmark table
1 个 exact free/reuse log
1 个 profiler screenshot/trace summary
```

这些比堆更多 feature 更有价值。
