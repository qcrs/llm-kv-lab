> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Test, Evidence, Benchmark and Resume Closure

原则：测试少而关键。每个 test 必须证明一个系统 invariant，而不是追求数量。

---

## 1. Test Pyramid

### L0 Pure semantics

CPU/PyTorch：

- page-size math
- group flatten/unflatten
- member map bijection
- virtual block/slot mapping
- keep plan

### L1 GPU primitive

- ragged slot mapping
- decode reshape
- prefill member-major transform
- reference compaction
- Triton writeback

### L2 Runtime component

- RaggedBlockTables staged write/gather
- RaggedKVState updates
- scheduler reconciliation
- true-preemption state reset / recompute invalidation

### L3 Real engine

- prefill/decode
- continuous batch
- physical free/reuse
- TP2
- preemption
- CUDA Graph

不要在每轮都跑完整 test suite。

---

## 2. 必须保留的 Evidence Ledger

每个 milestone 建议保存：

```text
commit SHA
command
model
GPU / CUDA / torch
vLLM config
expected invariant
observed evidence
pass/fail
known limitations
```

真实性能实验再加：

```text
warmup policy
repeat count
mean/p50/p95
GPU memory before/after
profiler trace path
```

---

## 3. R1 Evidence

### 必须证明

- Dense / Ragged Identity memory accounting 等价。
- Dense / Ragged outputs 等价。
- slot mapping/virtual block mapping 正确。
- multi-request prefill/decode 正确。

### 不应宣称

- memory saving
- throughput gain

R1 的目的只是 representation correctness。

---

## 4. R3 Evidence — 项目最重要截图/日志

建议记录一个非常直观的 block lifecycle：

```text
A before:
G0 [11,12,13,14]
G1 [21,22,23,24]

A after compact:
G0 [11,12]
G1 [21,22,23]

freed = [13,14,24]
BlockPool free: 100 → 103

B allocate:
... includes one or more of [13,14,24]

A next decode: PASS
```

这比单个 `torch.cuda.memory_allocated()` 更能证明真实 physical reclamation。

---

## 5. Performance Metrics

### Capacity

```text
physical pages owned
free blocks
KV bytes
head-equivalent retained bytes
max concurrency
```

### Serving

```text
request throughput
output token throughput
TTFT
TPOT
Goodput under chosen SLO（可选）
```

### Compression overhead

```text
score time
keep-plan time
writeback time
scheduler reconciliation time
```

### Runtime overhead

```text
CPU step time
CUDA launch gaps
piecewise graph replay coverage
```

---

## 6. AOT Benchmark

固定：

- same model
- same scorer
- same budget
- same page_group_size

只变：

```text
identity/adjacent map
vs
AOT cluster map
```

报告：

```text
ideal pages
actual pages
waste fraction
max concurrency
throughput
```

---

## 7. Triton Benchmark

两个 level：

### Micro

x-axis：

```text
source length
keep ratio
page_group_size
```

比较：

```text
Torch/reference
P1 scratch Triton
in-place Triton
```

指标：

- us / boundary
- GB/s effective moved bytes
- scratch bytes

### End-to-end

在同 workload：

```text
compression off
reference writeback
in-place writeback
```

只在真实 end-to-end 有变化时写入 resume 性能数字。

---

## 8. CUDA Graph Benchmark

必须拆：

```text
Dense+CG
Ragged eager
Ragged+CG
```

否则无法区分 Ragged 本身与 CUDA Graph 的影响。

重点看 decode TPOT；prefill 往往不是 CUDA Graph 最敏感区域。

---

## 9. Quality Evaluation

项目不是 compression algorithm paper，不需要大规模复刻所有质量表。

推荐：

- RULER 2–3 个 representative tasks
- SCBench 少量 long-context cases
- 若使用 KeyDiff，可选一组可重复的 benchmark

目标：确认系统 port 没引入额外质量损失，并展示 retention/quality trade-off。

---

## 10. 最终 Resume Bullet 模板

完成 Core + TP/Preemption：

```text
基于 vLLM 0.26 Model Runner V2 重构 Non-Uniform Ragged KV Cache Runtime，
将 request-level Paged KV 扩展为 per-layer/per-head-group physical pages，
打通 Ragged FlashAttention、KV compaction、Scheduler/BlockPool ownership
reconciliation 与 physical page reuse，并支持 TP=2 uniform compression 及
preemption/recompute 生命周期。
```

完成优化后再加：

```text
基于 retention profile 实现 AOT head clustering 降低 group-level KV
fragmentation；使用 Triton in-place paged writeback 减少 KV compaction
scratch/数据搬运；基于 vLLM piecewise CUDA Graph 隔离动态 Ragged Attention
为 eager island，优化 decode runtime overhead。
```

性能数字必须用你自己的实验结果替换，不能引用 Tangram README 数字。

---

## 11. 面试时必须能回答

1. PagedAttention 为什么不等于 physical reclamation？
2. 为什么 Ragged page 本身不省总 KV bytes？
3. 为什么一个 `(request, KV head)` 可以变成 FA varlen sequence？
4. virtual block ID 怎么构造？
5. 为什么 prefill 需要 copy 而 decode 多数只需 reshape？
6. Scheduler 和 Worker 谁拥有 block ownership？
7. 为什么 TP uniform 比 layer/global 容易？
8. 为什么 preemption 不能 restore stale ragged row？
9. AOT clustering 优化的到底是什么浪费？
10. in-place compaction 为什么不会覆盖未读 source？
11. 为什么选择 piecewise CUDA Graph 而不是 full graph？
12. 如果 profiler 显示 compaction 只占 0.5%，你还会不会继续优化 Triton？为什么？

如果这些能清楚回答，这个项目就不是“照着 Tangram 改代码”。

---

# v2 Addendum — Mandatory performance accounting

最终所有 performance图表必须包含 `Dense → Ragged Identity → Ragged+Compression → +Optimization`。Ragged Identity 是 representation tax baseline，不能省略。Tangram reference speedup如果使用 uncompressed-ragged baseline，只能作为参考方法，不能与自己的 vanilla vLLM baseline混写。详见 `17-Performance-Accounting-and-Benchmark-Matrix.md`。

