# P2 M3–M5 Parent Task Cards

当前 P2 runtime 全部 BLOCKED_BY_PROJECT1_CORE。以下是未来 Parent Task contract；候选 Slice 仅 PLANNED_DECOMPOSITION，不是授权。

## P2-M3-T1 — Compressed L2 Load / Deserialize Correctness

- Goal：stored compressed object loads into correct raw MemoryObj。
- Learning：load reverse lifecycle/bitmap/completion。
- Read First：M2 records、design 04。
- Anchors：FSL2 load→wrapper→deserializer。
- Approved Design Needed：tolerance/error policy。
- Allowed：tests/evidence/minimal approved fixes。
- Out：vLLM generation/performance。
- Tests：round-trip files、missing/truncated/corrupt/unsupported。
- Evidence：four bytes equality、tensor error、cleanup。
- Gate：no partial publication，completion exact。
- Failure：wrong layout/dtype、temp leak。
- Rollback：raw path。
- Next：M3-T2。
- Candidate：load success；failure atomicity。

## P2-M3-T2 — vLLM Cache-Hit Resume with Block-INT8

- Goal：real connector reuse via compressed FS-L2，request resumes。
- Learning：lossy tensor correctness vs token exactness。
- Read First：M0 connector/M1 raw/M3 load。
- Anchors：connector load/wait/layer restore。
- Approved Design Needed：workload/model/quality metrics。
- Allowed：harness/evidence/minimal glue。
- Out：performance/Triton。
- Tests：cold/raw/compressed、repeated hits、multiple requests、feature off。
- Evidence：hit provenance、generation、quality/error、file/report bytes。
- Gate：correct completion/no corruption and invariant。
- Failure：L1 masks L2、slot mapping mismatch、generation severe corruption。
- Fallback：adjust scale dtype/retention not architecture；raw baseline。
- Next：M3-T3。
- Candidate：single hit；multi hit；quality.

## P2-M3-T3 — V1 Core Release Review

- Goal：assemble ABI/raw/format/byte/E2E evidence。
- Learning：哪些收益deterministic，哪些conditional。
- Allowed：docs/state/delivery only。
- Out：auto Triton。
- Tests：clean-shell minimal reproduce。
- Gate：P2-G3 Core Done。
- Failure：missing raw chain、size invariant not exact/unexplained、PCIe claim。
- Rollback：BLOCKED with exact missing Task。
- Next：M4-T1 or delivery，user choice。
- Candidate：artifact audit；reproduction；release。

## P2-M4-T1 — Triton Encode Oracle

- Goal：CUDA BF16→scales+INT8 payload。
- Learning：reduction/tiling/layout offsets。
- Read First：design 05、TurboQuant store、PyTorch oracle。
- Approved Design Needed：supported shapes/grid/tile。
- Allowed：new kernel/launcher/tests; no E2E。
- Out：decode/placement/PCIe claim。
- Tests：vs PyTorch、zeros/extremes/strides/unsupported fail-fast。
- Evidence：correctness/compiler config。
- Gate：supported matrix correct。
- Failure：offset/precision/nonfinite。
- Fallback：PyTorch V1。
- Next：M4-T2。
- Candidate：launcher; kernel; edge tests。

## P2-M4-T2 — Triton Decode Oracle

- Goal：serialized CUDA→BF16 destination。
- Learning：scale/payload coalescing and cast。
- Read First：TurboQuant decode、M4-T1 format。
- Approved Design Needed：destination layout/device。
- Allowed：decode kernel/tests。
- Out：E2E。
- Tests：vs PyTorch quantized result/tolerance、round-trip。
- Evidence：error matrix。
- Gate：encode+decode correct。
- Failure：shape/stride/dtype mismatch。
- Fallback：PyTorch deserialize。
- Next：M4-T3。
- Candidate：decode; composed round-trip。

## P2-M4-T3 — Actual Serde Placement Integration

- Goal：connect Triton under same Serde contract and record src/dst device/staging。
- Learning：kernel fast不等于E2E/PCIe快。
- Read First：TurboQuant serializer staging、design 05。
- Anchors：MemoryObj device checks/temp allocations/copies。
- Approved Design Needed：device fallback/default policy。
- Allowed：minimal codec integration/instrumentation/tests。
- Out：pre-D2H hook。
- Tests：CPU source、CUDA source（supported）、fallback、same file bytes。
- Evidence：copy/device trace、four-byte invariant。
- Gate：G4 actual placement known。
- Failure：raw H2D+D2H added/unbounded temp。
- Rollback：PyTorch V1 default；Triton benchmark-only。
- Next：M5-T1。
- Candidate：device observer；CUDA path；CPU fallback/Gate。

## P2-M5-T1 — Codec Microbenchmark / NCU

- Goal：encode/decode GB/s and kernel bottleneck。
- Learning：kernel line separated from actual system。
- Read First：profiling manual。
- Approved Design Needed：shape/timing matrix。
- Allowed：bench/profile/evidence。
- Out：L2/serving claim。
- Tests：correctness before timing；warmup/repeats。
- Evidence：raw/NCU/commands/negative regimes。
- Gate：valid reproducible micro results。
- Failure：JIT/transfer included incorrectly。
- Next：M5-T2。
- Candidate：bench; NCU; tuning decision。

## P2-M5-T2 — NSYS Data-Movement Map

- Goal：map GPU paged→L1→Serde→temp→FS and reverse。
- Learning：L2 bytes vs PCIe bytes。
- Read First：profiling manual/design 05。
- Anchors：NVTX connector/extract/serialize/memcpy/L2/resume。
- Approved Design Needed：capture workload/process。
- Allowed：instrument/profile/evidence。
- Out：pre-D2H implementation。
- Tests：raw/PyTorch/Triton comparable runs。
- Evidence：memcpy sizes/device/timeline/CPU thread/event。
- Gate：actual placement documented，不推断。
- Failure：capture misses process/copy attribution。
- Fallback：software counters + targeted recapture。
- Next：M5-T3。
- Candidate：instrument; raw map; codec map.

## P2-M5-T3 — Codec / I/O / Serving Crossover

- Goal：context/object size sweep, theoretical vs measured crossover。
- Learning：when compression pays and why。
- Read First：crossover benchmark。
- Approved Design Needed：matrix/statistics。
- Allowed：benchmark/evidence/docs。
- Out：guaranteed speedup/PCIe claim without proof。
- Tests：correctness/quality gate per run。
- Evidence：raw/processed/curves/negative regimes。
- Gate：G5 strong done。
- Failure：L1 hit masks FS、uncontrolled cache、metric conflation。
- Rollback：invalid run/limited claim。
- Next：M6 explicit GO or M7。
- Candidate：FS synthetic; E2E sweep; conclusion。

