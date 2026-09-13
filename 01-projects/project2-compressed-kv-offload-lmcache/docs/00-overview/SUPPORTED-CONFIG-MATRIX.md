# P2 Supported Configuration Matrix

P2 当前 runtime 仍 `BLOCKED_BY_PROJECT1_CORE`。本表只定义未来 V0/V1/V2 的 support boundary，不授权任何 probe/patch。

| Dimension | V0 Baseline | V1 Block-INT8 Serde | V2 Triton Codec | V3 |
|---|---|---|---|---|
| vLLM | 0.26.0 / `568afb3...` | same | same | newer offload only as reference |
| LMCache | `a7afade...` | same | same | same baseline, scoped port only |
| Torch ABI | 2.11.x family, exact TO_FREEZE | same | same | same unless separately rebuilt/proven |
| GPU | A100 80GB | A100 80GB | A100 80GB | same first |
| L2 backend | local filesystem | local filesystem | local filesystem for E2E | other backends separate audit |
| Serde | disabled | custom fixed-size INT8 | same serialized contract | adaptive format optional |
| Quantization granularity | N/A | codec-domain `K/V × layer × KV head × fixed token group × head_dim`; exact G TO_FREEZE | same frozen group | lower-bit later |
| Relation to vLLM BlockPool page | N/A | **no required alignment** | no required alignment | only if future connector-side codec needs it |
| Scale | N/A | FP16 candidate, TO_APPROVE | same | FP32 fallback if numeric evidence requires |
| Serialized length | raw | fixed deterministic | fixed deterministic | variable format not assumed |
| MemoryObj source device | TO_OBSERVE | TO_OBSERVE | explicitly profiled | GPU-side pre-D2H only after gate |
| L1 raw retention | upstream/default behavior | no automatic reduction claim | same | policy change separate |
| First D2H | raw/unknown placement | no reduction claim | no reduction claim without NSYS | V3-D target |
| Connector | LMCacheConnectorV1 | same | same | layerwise hooks / offload reference |
| Multi-node/RDMA/Mooncake/NIXL | OUT | OUT | OUT | optional separate project scope only |

## Byte Proof Scope

V1/V2 fixed-size + filesystem L2 才使用 exact object/file invariant。其它 backend 不继承该 invariant，除非重新 source audit。

## Filesystem Capacity / Benchmark Scope

`FSL2Adapter` 是 deterministic byte oracle，但当前 baseline 本身没有 fixed max-capacity admission。V1 可测：

```text
serialized bytes ↓
.data file size ↓
```

以及在“人为固定 byte budget / quota model”下推导可容纳 object 数；不得把后者描述成 FSL2 runtime 自动 admission improvement。

I/O latency benchmark 还必须标注 buffered/page-cache/O_DIRECT regime，避免把 Linux page cache 命中当成磁盘性能。
