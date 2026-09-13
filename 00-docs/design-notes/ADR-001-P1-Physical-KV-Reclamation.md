# ADR-001 — P1 Physical KV Reclamation

Status：ACCEPTED_HIGH_LEVEL
Implementation approval：NONE

## Decision

P1 固定为 vLLM v0.26.0 上的 Physical KV Cache Reclamation。V1 使用 whole-block zero-copy row reconstruction，必要时回退 Tangram-style copy compaction；V2 使用 PyTorch oracle + staged Triton paged gather/scatter；V3 仅做 profiler/reference-backed hardening。

## Why

问题是 logical retention 后 physical ownership 仍占 page。项目成功由 real BlockPool reuse、logical/physical decoupling 与 safe-free correctness 定义，不由必然 TPOT speedup 定义。

## Invariants

- logical position 不重编号；
- physical cache position 从 effective occupancy append；
- worker 停止引用后 scheduler 才 free；
- canonical ownership 由 KV manager 持有；
- free page 不等于 HBM 返回系统。

## Not Approved Here

具体字段/API、patch files、attention metadata adapter、trigger、output channel 与任意 Slice。

