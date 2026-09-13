# P1 Capacity / Serving Benchmark Plan

正式 benchmark 只在 G3 correctness 前置通过后运行。

## Modes

- Native baseline / feature off。
- V1 whole-block reclaim。
- V1-R copy fallback（若批准）。
- V2 token compaction（G5）。

## Independent Variables

prompt length、decode length、retention ratio/target、block size（若 baseline允许）、batch/concurrency、fixed KV pool budget。一次实验只改变一个主变量。

## Metrics

Allocator deterministic：

- prefill peak owned blocks before reclaim；
- post-reclaim effective physical KV tokens；
- post-reclaim blocks/request；
- time-to-capacity-recovery (final-prefill commit → blocks published free)；
- BlockPool free blocks；
- released-ID reuse；
- pool utilization/conservation；
- allocation rejects。

Serving conditional：

- admitted/max concurrent requests；
- preemption count（仅 compatibility Gate 后）；
- TTFT、TPOT、E2E；
- request/s、token/s、goodput optional；
- reclaim/compaction latency；
- quality。

CUDA/HBM：

记录 fixed KV tensor/pool allocation，但不把 internal free pages 写成 HBM returned。可报告 peak process memory 作为 observed secondary metric，必须解释 allocator model。

## Workloads

1. Deterministic two-request reuse。
2. **Rolling/staggered arrival**：A prefill→reclaim→B arrival→A decode+B prefill，证明 recovered pages 对后续 arrival 有用。
3. Fixed-pool capacity ramp（以 post-reclaim steady/rolling state 为主）。
4. Simultaneous long-prefill pressure：作为 limitation/negative regime，明确 Core 在 reclaim point 前不能救首次 prefill peak。
5. Context × retention sweep。
6. Quality subset：Needle/RULER/LongBench 的小而清晰集合（需冻结版本）。

## Run Protocol

每个 run：Git SHA、env、model/tokenizer、backend、flags、pool budget、shape/batch、warmup、repeats、seed、command、raw/processed。GPU contamination/thermal/JIT 异常标 invalid，不删。

## Result Interpretation

- pages↓是确定 system capability；
- concurrency/preemption 受 workload/scheduler影响；
- TPOT可能因 shorter attention改善，也可能被 bookkeeping抵消；
- V2 compaction有 crossover；
- negative regime 必须画出。

## Claim Gate

任何数字沿：report figure → processed row → aggregation version → raw → manifest/config → Git/env/model。缺一环不进简历。

## v1.2 Claim Boundary

Core 不允许从 capacity benchmark 推导：

```text
single-request max prefill context ↑
first-prefill peak KV ↓
```

除非未来另有 prefill-time reclaim implementation + evidence。P1 当前最强的容量证据应是：**reclaim 后 pages 被真实发布并被后续/并发请求复用**。
