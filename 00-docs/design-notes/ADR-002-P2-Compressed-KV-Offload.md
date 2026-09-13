# ADR-002 — P2 Compressed KV Offload

Status：ACCEPTED_HIGH_LEVEL
Implementation approval：NONE

## Decision

P2 固定为 vLLM + LMCache 真实 reuse / filesystem-L2 上的 custom fixed-size Block-INT8 Serde。V2 增加 Triton codec 与 placement/crossover 分析；pre-D2H compression 属于 optional V3。

## Why

Core 可确定证明 serialized L2 bytes 和 fixed-budget L2 footprint，且能复用 Connector、MemoryObj、Serde、AsyncSerdeProcessor、SerdeL2AdapterWrapper 与 filesystem backend。generic Serde 的 first D2H 语义不确定，不能作为 Core claim。

## Invariants

- vLLM pin 568afb3...；
- LMCache baseline a7afade...；
- torch/native extension ABI gate；
- estimated size = temp object = file bytes = reported bytes（fixed V1 尽量 exact）；
- L2 bytes 与 PCIe bytes 分开；
- reference ownership 不冒充自研。

## Not Approved Here

format offsets、scale layout、registration name、kernel shape、placement hook 与任何 Slice。

