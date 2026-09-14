> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Performance Accounting & Benchmark Matrix

最终结果必须能回答：**收益来自 compression 释放容量，还是 Ragged representation 本身？优化恢复了多少 Ragged tax？**

## 1. Mandatory ladder

所有核心 performance结果至少有四列：

```text
A Dense v0.26
B Ragged Identity (compression off)
C Ragged + Compression
D Ragged + Compression + Optimization(s)
```

不要只报 `C vs B`，也不要只报 `C vs A`。

### Identity tax

```text
Tax_metric = B / A - 1
```

重点测：TTFT、TPOT、prefill throughput、decode throughput、CPU step time。

### Compression capacity gain

重点看物理量，不只看 latency：

```text
owned physical pages
free pages
peak KV bytes
max admitted/running requests
preemptions
goodput under SLO
```

## 2. R7 AOT matrix

固定 model/scorer/budget/Hp：

```text
Per-head ideal (offline lower bound)
Adjacent grouping
Per-layer AOT grouping
Cross-layer AOT (optional)
```

报告：

```text
allocated head-slots / pages
residual over-allocation
rank stability
real BlockPool page count after compression
```

Tangram README 中的 42.7%→2.9%只能作为 reference motivation，不能写成自己的结果。

## 3. R8 Triton matrix

Micro：

```text
Torch/scratch reference latency
Triton in-place latency
bytes moved estimate
scratch peak bytes
K (kept length), Hp, D, num clusters sweep
```

End-to-end：只看 compression boundary占 overall request time多少；若很小，明确写“micro speedup未转化为 serving speedup”。

## 4. R9 CUDA Graph matrix

```text
Dense + Piecewise CG
Ragged + enforce_eager
Ragged + Piecewise CG basic
Ragged + Piecewise CG + persistent metadata
```

测：

- TPOT / decode tok/s；
- CPU model-runner step time；
- Nsight/Torch profiler GPU idle gaps；
- per-step metadata allocation count；
- eager island duration。

## 5. TP / Preemption matrix

TP2：同一 prompt/config，TP1/TP2 correctness、free counts、rank checksum、communication overhead。  
Preemption：故意小 KV pool产生压力，报告 preemption count、recompute tokens、throughput；再对比 compression是否降低 pressure。

## 6. Quality budget

只做能回答“compression有没有把模型明显弄坏”的小规模 quality evidence：RULER/SCBench 子集或项目统一 long-context cases。不要把整个论文 benchmark复刻变成项目主任务。

## 7. Final report table template

| Config | KV pages peak | Free after compaction | Max concurrency | TTFT | TPOT | Throughput | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | | | | | | | |
| Ragged Identity | | | | | | | |
| + Compression | | | | | | | |
| + AOT | | | | | | | |
| + Triton | | | | | | | |
| + Piecewise CG | | | | | | | |

只有自己的实验数据填入；reference paper数字单独标注。
