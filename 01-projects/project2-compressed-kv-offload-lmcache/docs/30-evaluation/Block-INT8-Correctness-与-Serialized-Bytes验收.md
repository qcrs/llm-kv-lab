# Block-INT8 Correctness / Serialized Bytes Acceptance

## 1. Tensor / Format Oracle

PyTorch reference quant/dequant；shape/dtype；payload/scales/metadata offsets；zero/near-zero/extreme/nonfinite policy；multiple quant groups/heads/K/V；unsupported fail-fast。

Metrics：MAE、RMSE、max error、cosine；quantized payload/scale 与 reference 对齐。Lossy round-trip 不要求 BF16 bitwise identity。

## 2. Quant Group Must Be Frozen First

测试命名和 oracle 必须使用 codec-domain `quant_group`；不允许把它称为 vLLM “physical KV block”。M1 需要冻结 LMCache `MemoryLayoutDesc` axis 与 fixed token-group G。Gate 记录：

```text
MemoryObj shape/layout/device
chunk/object token span
K/V/head mapping
approved quant_group definition
```

## 3. Scoped Byte Hard Invariant — FSL2 Only

仅适用于 **fixed-size Block-INT8 + FSL2Adapter**。

### Per object

```text
estimated serialized size
== allocated temp serialized MemoryObj bytes
== corresponding filesystem .data file bytes
```

### Per store operation

```text
sum(newly persisted object bytes)
== L2StoreResult.bytes_transferred
```

测试必须使用 isolated namespace/clean files，或显式扣除 already-existing keys，因为 FSL2Adapter 对已存在 key 会 skip store。

同时记录 raw BF16 object bytes与compression ratio。理论 bit width不能替代 physical bytes。

其它 backend（Redis/Mooncake/NIXL等）不继承这个 equality，除非重新审计 framing/alignment/accounting。

## 4. Async / Failure

submit→pending→event→query exactly once；temp release；serialize失败不inner-store；partial/corrupt file不publish raw destination；missing file clear error；no leak/deadlock。

## 5. E2E

A store compressed FS object；B real L2 load/deserialize/restore；cache-hit provenance；request generation completes；quality/error recorded；raw/FP8作为 baseline。

## 6. Failure Triage

```text
unit fail
→ format/math/group

unit pass / file mismatch
→ estimator/temp/backend/pre-existing-key

file/load correct / tensor wrong
→ deserialize layout

tensor correct / generation wrong
→ slot mapping/connector or lossy quality
```

## Logical File Length vs Disk Allocation

Exact byte invariant 使用：

```text
os.stat(path).st_size
```

表示 `.data` logical file length。不要用 `du` / allocated filesystem blocks 与 serialized payload 做 exact equality；后者受 filesystem block allocation、sparse/extents、metadata影响，可作为 storage-system secondary metric 另报。
