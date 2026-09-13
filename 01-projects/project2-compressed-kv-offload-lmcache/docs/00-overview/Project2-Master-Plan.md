# Project 2 Master Plan

## Runtime Entry

只有 P1-G3 V1_RUNTIME_CORE_DONE accepted 后，P2-M0 才可 review。解锁不等于执行。

## Milestones

| M | Purpose | Gate |
|---|---|---|
| M0 | Environment / ABI / Connector Freeze | P2-G0 |
| M1 | Raw filesystem-L2 baseline | P2-G1 |
| M2 | V1 Block-INT8 Serde | P2-G2 |
| M3 | vLLM E2E Reuse + Correctness | P2-G3 Core Done |
| M4 | V2 Triton Codec | P2-G4 |
| M5 | Crossover / NSYS / NCU | P2-G5 Strong Done |
| M6 | Optional V3 | explicit GO |
| M7 | Delivery | P2-G7 |

## Gate Boundaries

G0：pins/ABI/native imports/backend/connector smoke。
G1：raw FS-L2 store/load/file/bytes/reuse。
G2：format/oracle/estimator/async wrapper；fixed-size + FSL2 per-object/per-store exact byte proof。
G3：compressed E2E load/resume/quality；scoped FSL2 size invariant保持成立。
G4：Triton encode/decode vs PyTorch，source/destination device known。
G5：kernel micro/NCU与actual path/NSYS分离；crossover/negative regimes。
G6：item-specific source/lifetime/profiler。
G7：reproduction/claim audit。

## Deterministic Claims

serialized L2 bytes↓、file size↓、L1→L2 serialized bytes↓；在**外部人为固定 byte budget/quota**下可推导可容纳 object 数↑，但这不是 FSL2 backend 自动 admission capacity。

## Conditional/Forbidden

store/load/TTFT/throughput conditional。L1 raw capacity、first D2H、HBM、speedup、beat TurboQuant not Core。

## Current

Blocked；Parent P2-M0-T1；Slice NONE；no next runtime action。

## v1.2 Codec / Filesystem Scope

- `Block-INT8` 的 Block 是 LMCache object-domain quantization block：`K/V × layer × KV head × fixed token group × head_dim`；不要求与 vLLM BlockPool page 对齐。
- FSL2 latency/crossover 必须绑定 buffered-warm、buffered large-working-set/new-key、或 verified O_DIRECT regime。
- Upstream LMCache real FS Serde E2E / TurboQuant tests 优先作为 test-pattern reference。
