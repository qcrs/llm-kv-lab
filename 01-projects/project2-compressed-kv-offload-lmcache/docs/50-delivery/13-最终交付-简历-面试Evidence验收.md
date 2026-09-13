# P2 Final Delivery / Resume / Interview

## Deliver

pins/ABI/source map；raw connector/FS baseline；format spec；unit/async/E2E；four-byte invariant；Triton micro/NCU；actual movement NSYS；crossover；reproduction；negative/limits；artifact index。

## Resume

V1 evidence后：implemented custom fixed-size Block-INT8 KV Serde for LMCache real filesystem-L2 path，verified FSL2-scoped exact serialized bytes and cache-hit load/resume。

V2 evidence后：added Triton encode/decode and separated kernel, L2 I/O and actual data movement to identify crossover。

只有V3 NSYS证据后：moved codec ahead of raw D2H and verified reduced transfer bytes。

## Interview

解释Connector/RequestTracker/slot、MemoryObj/Serde/wrapper lifecycle、size invariant、code-doc drift、CPU/CUDA staging、L2 vs L1 vs PCIe、crossover、why INT8 simple、why not beat TurboQuant、V3 five gates。

数字只从artifact chain。

## v1.2 Claim Boundary

- `Block-INT8` 的 block 是 codec quant group，不等于 vLLM allocator page；
- FSL2 可以确定证明 `.data` / serialized bytes下降；
- “固定预算下可存更多 object”必须写成 external byte-budget 下的 derived capacity，不写成 FSL2 自动 admission；
- filesystem crossover 必须带 `buffered-warm / buffered-large-working-set / O_DIRECT` 等 regime 标签；
- first-D2H / PCIe reduction仍只允许来自 NSYS evidence。
