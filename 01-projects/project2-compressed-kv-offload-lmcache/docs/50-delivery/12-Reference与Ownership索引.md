# P2 Reference and Ownership

vLLM provides KV Connector/layer hooks/paged runtime。

LMCache provides connector implementation、RequestTracker/slot mapping、MemoryObj、Serde/AsyncSerdeProcessor、wrapper、L1/L2 lifecycle、filesystem、FP8/TurboQuant。

CacheGen provides motivation/history。newer vLLM provides V3 page/pinned/pool/fence reference。

We implement fixed Block-INT8 spec/serde/estimator/factory、PyTorch/Triton codec、E2E/byte/crossover/profiling。

禁止写“给LMCache首次增加compression”“实现LMCache async/filesystem”“发明quant algorithm”“减少2x PCIe”无NSYS证据。

## v1.2 Provenance Ledger

实现期间持续维护 `docs/90-reference/REFERENCE-PORTING-LEDGER.md`：记录 reference repo/commit/symbol、使用类型（API / semantic / test / kernel pattern / adapted code）和 project-owned target。最终面试/README 的 ownership 声明必须能回溯到该 ledger。
