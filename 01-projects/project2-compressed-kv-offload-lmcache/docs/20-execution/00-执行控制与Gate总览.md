# P2 Execution Control and Gates

当前 P2 runtime 全部 BLOCKED_BY_PROJECT1_CORE。以下是未来 Parent Task contract；候选 Slice 仅 PLANNED_DECOMPOSITION，不是授权。

| M | Theme | Parent Tasks | Gate |
|---|---|---|---|
| M0 | Environment / ABI / Connector Freeze | unlock+identity, source map, connector reuse | G0 |
| M1 | Raw FS-L2 Baseline | raw store/load, byte/reuse baseline | G1 |
| M2 | V1 Block-INT8 Serde | format/oracle, wrapper integration | G2 |
| M3 | vLLM E2E Reuse / Correctness | compressed load/resume/quality/bytes | G3 Core Done |
| M4 | V2 Triton Codec | encode/decode, serde placement | G4 |
| M5 | Crossover / NSYS / NCU | micro/actual path/E2E | G5 |
| M6 | Optional V3 | explicit GO | G6 |
| M7 | Delivery | report/reproduction/claims | G7 |

Gate无partial pass。G0前不写codec，G1前不启Serde，G2前不跑compressed E2E，G3后才做performance，M6需GO。Slice PASS不关闭Parent。

G0必须证明torch/native ABI与backend没有silent fallback。G2/G3必须满足fixed-size + FSL2 scoped byte invariant（per-object + per-store；control pre-existing keys）。G5必须分开kernel、L2 I/O、PCIe与serving。

