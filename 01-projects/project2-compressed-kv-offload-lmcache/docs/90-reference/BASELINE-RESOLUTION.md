# P2 Baseline Resolution / Canonical Errata

本文件指定 P2 v3 baseline 的 canonical interpretation，防止版本、backend 和 byte-accounting 语义在后续实现中被过度泛化。

## R1 — Implementation baseline 固定在 LMCache `a7afade...`

Canonical implementation baseline：

```text
vLLM 0.26.0 / 568afb3a13806beb53bb2e6bd518269357b237c0
LMCache / a7afadebb9248b62b5c533ce2c12297e9d94fc4a
Torch ABI family / 2.11.x
```

LMCache 该 commit 的 `pyproject.toml` build requirement 为 `torch==2.11.0`。后续 `f9addd...` snapshot 只作为 reference comparison，不得未经 rebuild/ABI proof 直接替换 implementation baseline。

## R2 — Block-INT8 的 `Block` 是 codec quantization block

Compatibility baseline 的 TurboQuant tests 使用 LMCache `MemoryLayoutDesc` KV object（示例 `[2, num_layers, num_tokens, hidden_dim]`，`hidden_dim = num_heads × head_dim`）定义 codec layout。因此 canonical P2 V1 不再以“尽量对齐 vLLM physical block”为目标。

Canonical quant group：

```text
K/V × layer × KV head × fixed token group G × head_dim
```

exact axis/layout/G 仍需 M1 从 same-commit source/runtime freeze；若实际 object layout不同，只重新映射 LMCache object-domain axis，不回到 vLLM BlockPool page coupling。

## R3 — Exact byte equality 只限定 V1 fixed-size + filesystem L2

对于 V1 的固定尺寸 format 与 `FSL2Adapter`，目标 invariant 是：

### Per-object

```text
estimate_serialized_size(layout)
== allocated serialized MemoryObj byte length
== corresponding filesystem `.data` logical file size (`stat.st_size`)
```

### Per-store operation

```text
sum(newly persisted object byte lengths)
== L2StoreResult.bytes_transferred
```

注意：如果 key 已经存在，FS adapter 会 skip store，因此 operation-level `bytes_transferred` 只统计本次实际新写入 bytes。测试必须清理 namespace 或记录 pre-existing key 状态。

这不是 Redis/Mooncake/NIXL/variable-format 的永久 universal invariant。其它 backend 的 framing/alignment/accounting 必须单独审计。

## R4 — Generic Serde 不等于 first D2H compression

V1/V2 Core 只允许确定 claim：

- serialized L2 bytes decrease；
- filesystem object size decrease；
- under an externally imposed fixed byte budget，serialized object size下降可推导 stored-object count increase；
- inner L2 persisted payload bytes decrease。

注意：FSL2 baseline 自身没有 fixed max-capacity admission，因此不得把 derived byte-budget capacity写成 backend 自动 admission。

默认不允许 claim：

- L1 raw CPU footprint decrease；
- first GPU→CPU D2H bytes decrease；
- cache-hit TTFT/throughput improvement。

只有 NSYS 明确证明 codec 位于 raw D2H 之前，才允许声称 PCIe byte reduction。

## R5 — Pre-D2H 是 reference-backed high-cost V3，不是 Core

newer vLLM offload subsystem提供 canonical GPU page view、pinned host memory、stream/event pools、transfer fences 等机制 reference，因此 pre-D2H 不再是“完全未知”。但 pinned v0.26 + LMCache integration 仍需 substantial porting，保持 V3 optional。

## R6 — Source doc 与 same-commit code 冲突时以 code + E2E byte proof 为准

LMCache FP8 Serde 实际 implementation 的 estimator 是 exact 1 byte/element，并明确说明过度估算会导致 wrapper 分配更大的 serialized MemoryObj，从而侵蚀 L2 saving。后续任何 Serde contract 必须同时核对：

```text
design doc
+ same-commit code
+ object/file/bytes evidence
```

## R7 — Filesystem crossover 必须绑定 I/O regime

FSL2 normal path受 Linux page cache影响；O_DIRECT又有 alignment/fallback条件。因此所有 FS latency/crossover figure 必须标记 buffered-warm、buffered-large-working-set/new-key 或 verified-O_DIRECT。Storage footprint可以独立作为 deterministic evidence，但 filesystem latency不能脱离 cache/backend state做 universal claim。

