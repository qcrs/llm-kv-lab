# P2 Environment Snapshot

All runtime values TO_FREEZE except target/pins：

- A100 80GB target；
- vLLM 568afb3...；
- LMCache a7afade...；
- torch 2.11 ABI required；
- single machine/GPU；
- filesystem L2。

Python/exact torch/CUDA/driver/Triton/native modules/model/config/FS root TO_FREEZE after unlock。No old environment inheritance。

