# Resume Technical Stack & Evidence Matrix

本文件回答：**这个项目到底体现哪些 AI Infra 技术栈，以及每个技术点怎样在源码/实验中被证明。**

---

# 1. Technical Stack Matrix

| Stack | Concrete Project Surface | Source/Implementation | Evidence |
|---|---|---|---|
| vLLM V1/MRv2 | scheduler→worker execution | Scheduler / GPUModelRunner | real engine traces |
| Continuous Batching | mixed active requests | Scheduler + InputBatch | A decode + B prefill |
| PagedAttention | page table / physical slots | BlockPool / RaggedBlockTables | slot oracle |
| KV Cache Management | ownership / free / reuse | manager / scheduler | exact released ID reuse |
| FlashAttention 2 | varlen / GQA / paged KV | flash_attn backend | dense/ragged equality |
| GPU Data Layout | shape/stride/alias | ragged_layout | zero-copy oracle |
| Triton | gather/writeback/slot mapping | P1/R8 kernels | reference equality + microbench |
| CUDA Memory | physical page backing | KVCacheTensor/global pool | byte/page accounting |
| PyTorch Custom Ops | attention/update seam | unified_* ops | dispatch integration |
| torch.compile | piecewise partition | compilation config | compile trace |
| CUDA Graph | replay-safe buffer | R9 buffers | TPOT/launch profile |
| UVA / Pinned Memory | metadata staging | MRv2 staged buffers | pointer/CPU timing |
| Tensor Parallel | local head shard | ParallelConfig/ModelConfig | TP2 tests |
| NCCL Collectives | retained boundary sync | all_reduce(MAX) | rank consensus |
| Serving Preemption | free/recompute | scheduler lifecycle | pressure test |
| State Transaction | source fence / commit | P1→Ragged reconcile | stale receipt test |
| AOT Optimization | head placement | clustering tools | fragmentation ablation |
| Performance Engineering | TTFT/TPOT/throughput | benchmark ledger | B0–B8 |

---

# 2. Source Chain → Interview Knowledge

## Scheduler / Continuous Batching

你应该能解释：

```text
为什么 Scheduler 先 allocate_slots
为什么 BlockPool 是 canonical physical owner
为什么 Worker 不能直接 free
为什么 preemption 与 ordinary unscheduled 不同
```

## BlockTable / PagedAttention

你应该能手算：

```text
position → depth/offset → physical page → slot
```

Ragged 后再加：

```text
head → group/column → virtual block
```

## FlashAttention / GQA

你应该能解释：

```text
为什么 one KV head member 对应 q_per_kv query heads
为什么 decode可reshape
为什么 prefill需要materialize
```

## torch.compile / CUDA Graph

你应该能解释：

```text
为什么 dynamic metadata不能直接放full graph
splitting op是什么
piecewise graph保留了什么收益
fixed address为什么重要
```

## TP

你应该能解释：

```text
local KV heads如何取得
为什么 keep positions不必一致
为什么 free boundary必须一致
为什么用MAX不是SUM
```

---

# 3. Evidence Levels

### L0 — Source Understanding

只有源码分析，不能写“实现”。

### L1 — Unit / Synthetic

可以写“实现组件”，不能写“runtime supported”。

### L2 — Real Engine Correctness

可以写“implemented runtime”。

### L3 — Serving Closure

包含 continuous batch / reclaim / reuse / preemption，可以写“serving system”。

### L4 — Distributed / Compiler Breadth

TP2 / CG 等，可以写 distributed/runtime breadth。

### L5 — Performance Claim

必须 own benchmark。

---

# 4. Project Claim Ladder

```text
R1-E
→ Ragged KV Runtime

R3
→ Physical KV Reclamation Runtime

R4-B
→ Non-Uniform KV Compression Runtime

R5 + R6
→ Serving-Lifecycle + Distributed Breadth

R7/R9
→ Capacity / Runtime Optimization
```

不要提前跳级。

---

# 5. Suggested Final Resume Project Header

**Non-Uniform Ragged KV Cache Runtime for vLLM 0.26**  
`vLLM / PagedAttention / FlashAttention-2 / Triton / CUDA Graph / Tensor Parallel / GPU Memory Management`

如果最终 TP2 完成，可以改为：

**Distributed Non-Uniform Ragged KV Cache Runtime for vLLM 0.26**

---

# 6. Metrics to Put on Resume

优先：

```text
physical pages reduced X%
max concurrency improved X%
TPOT overhead/reduction X%
compaction latency X us/ms
fragmentation reduced X%
```

次级：

```text
unit test count
LOC
```

不要把测试数量当主要项目成果。

所有 `%` 必须来自自己的 benchmark，不引用 Tangram 数字。
