# P1 M4–M7 Parent Task Cards

所有 Parent Task 均遵循：没有 one-shot Approved Slice Prompt 时 REVIEW_ONLY；候选 S1/S2 只是 PLANNED_DECOMPOSITION。Allowed Files 需在 M0 映射为实际路径；超出即停止。

## P1-M4-T1 — PyTorch Token Compaction Oracle

- Goal：paged gather/scatter slow oracle with exact keep semantics。
- Learning：token index→source block/offset→destination prefix。
- Read First：design 06；vLLM paged layout。
- Anchors：KV cache shape/strides/block table。
- Approved Design Needed：keep contract、scratch/destination layout、tolerance。
- Allowed：project oracle/tests only。
- Read-only：runtime/kernel。
- Out：Triton/performance。
- Patch：reference functions + invalid atomic tests。
- Tests：random/tiny/multiple heads/tails/invalid。
- Evidence：tensor snapshots/error。
- Gate：deterministic oracle。
- Failure：layout unknown/alias mutation。
- Fallback：contiguous extracted tensor oracle。
- Next：M4-T2。
- Candidate：layout audit; gather oracle; scatter oracle。

## P1-M4-T2 — Triton Paged Gather

- Goal：source paged K/V → contiguous scratch。
- Learning：Triton pointer arithmetic and masked vector load。
- Read First：vLLM triton_reshape_and_cache_flash; oracle。
- Anchors：exact KV strides/dtype/head dims。
- Approved Design Needed：program granularity/tile supported shapes。
- Allowed：new project kernel/launcher/tests; no runtime integration。
- Read-only：upstream attention runtime。
- Out：scatter/fusion/E2E claims。
- Patch：one kernel + launcher + tests。
- Tests：vs PyTorch across shapes, OOB guard, unsupported fail-fast。
- Evidence：correctness raw and compiler config。
- Gate：correctness/tolerance all supported shapes。
- Failure：incorrect stride, nonfinite, excessive compile variants。
- Rollback：PyTorch oracle。
- Next：M4-T3。
- Candidate：launcher contract; kernel correctness; edge matrix。

## P1-M4-T3 — Scatter / Atomicity / Primitive Gate

- Goal：scratch→compacted paged slots，完整 primitive。
- Learning：destination mapping与source/destination lifetime。
- Read First：design 06、upstream store primitive。
- Anchors：reshape/cache launcher and slot mapping。
- Approved Design Needed：reuse vs adapted scatter、scratch owner。
- Allowed：kernel/launcher/tests。
- Out：general in-place fusion。
- Tests：gather+scatter vs oracle、invalid no mutation、overlap prohibited。
- Evidence：paged snapshots/test matrix。
- Gate：G4 correctness。
- Failure：destination overwrites unread source、trailing garbage visible。
- Fallback：upstream-style scatter/PyTorch。
- Next：M5-T1。
- Candidate：scatter; composed primitive; Gate close。

## P1-M5-T1 — V2 Runtime Transaction Integration

- Goal：invoke staged compaction and reuse V1 ACK/free transaction。
- Learning：GPU completion/lifetime before ownership publication。
- Read First：V1 transaction records、V2 primitive。
- Anchors：worker post-forward hook、stream/event、output channel。
- Approved Design Needed：trigger/scratch owner/completion fence。
- Allowed：approved runtime files。
- Out：fused kernel/async।
- Patch：feature-gated V2 path only。
- Tests：single/multi request、V1/V2 off/on、cleanup。
- Evidence：state/stream/IDs。
- Gate：runtime correctness and no premature free。
- Failure：race/alias/leak。
- Rollback：V1 default / PyTorch reference。
- Next：M5-T2。
- Candidate：hook; completion; E2E。

## P1-M5-T2 — Kernel Microbenchmark and NCU

- Goal：measure gather/scatter separately and explain bottleneck。
- Learning：DRAM throughput/occupancy/registers vs E2E。
- Read First：profiling manual。
- Anchors：kernel names/NVTX。
- Approved Design Needed：shape matrix/timing protocol。
- Allowed：bench/profile/evidence only。
- Out：serving claim/tuning unrelated kernels。
- Tests：correctness before timing；warmup/repeats。
- Evidence：raw timings/NCU reports/commands。
- Gate：reproducible measurement and negative regimes。
- Failure：JIT included/uncontrolled clocks/aggregation missing。
- Rollback：mark invalid run。
- Next：M5-T3。
- Candidate：bench; NCU; tuning decision。

## P1-M5-T3 — V2 E2E / NSYS / Cost Model

- Goal：saved physical bytes vs compaction latency、attention/serving effect。
- Learning：micro vs critical path。
- Read First：capacity and Nsys manuals。
- Anchors：NVTX reclaim/gather/scatter/attention。
- Approved Design Needed：workload matrix。
- Allowed：benchmark/profile/docs。
- Out：guaranteed speedup claim。
- Tests：quality/correctness gate then performance。
- Evidence：Nsys/raw metrics/crossover/limits。
- Gate：G5 full explanation。
- Failure：trace lacks source identity or bytes。
- Rollback：V2 remains non-default correct primitive。
- Next：M6 explicit GO or M7。
- Candidate：data map; E2E sweep; conclusion。

## P1-M6-T1 — Optional Hardening Selection

- Goal：select exactly one V3 item from async/periodic/preemption/graph/fused/policy。
- Learning：profiler/reference-driven scope。
- Read First：design 07 and current evidence。
- Anchors：item-specific。
- Approved Design Needed：mandatory explicit GO。
- Allowed/Tests/Evidence：new Task Card created only after GO。
- Out：multiple V3 parallel changes。
- Gate：compatibility/correctness first。
- Failure：no profiler/reference signal。
- Fallback：skip with reason。
- Next：M7。

## P1-M7-T1 — Reproduction and Technical Report

- Goal：clean-shell reproduce, architecture/report, artifact chain。
- Learning：evidence-backed systems story。
- Allowed：delivery/docs/scripts packaging; no new feature。
- Tests：one-command smoke + required suites。
- Evidence：release manifest。
- Gate：reproducible and limitations explicit。
- Next：M7-T2。

## P1-M7-T2 — Resume / Interview Claim Audit

- Goal：map each claim to source/diff/test/raw/profiler。
- Allowed：50-delivery/interview。
- Tests：claim audit checklist。
- Gate：no unmeasured number/upstream ownership inflation。
- Next：project close decision。

