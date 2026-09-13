# 04 — V1 Block-INT8 Serde Format / Object Lifecycle

## 1. Goal

V1 不是“发明 KV quantization”。它实现一个透明、固定尺寸、可 oracle 的 INT8 Serde，让真实 LMCache filesystem L2持久化更少 bytes，同时保持 load/deserialize/resume path正确。

## 2. Quantization Group — 固定在 LMCache Object Domain

P2 不需要把 codec group 与 vLLM allocator physical page 对齐。LMCache compatibility baseline 的 TurboQuant tests 明确以 `MemoryLayoutDesc` KV object 语义工作，示例 layout 为：

```text
[2, num_layers, num_tokens, hidden_dim]
hidden_dim = num_kv_heads × head_dim
```

因此 V1 canonical candidate 是：

```text
quant_group
=
K/V
× one layer
× one KV head
× fixed token group G
× head_dim
```

`Block-INT8` 里的 **Block = codec quantization block**，不是 vLLM `BlockPool` block。

M0/M1 需要从 same-commit `MemoryLayoutDesc` / runtime object确认实际 shape、K/V axis、layer/token/head layout；若真实 layout 与上述 reference 不同则重新映射 object-domain axis，但**不把 codec重新耦合回 vLLM physical page**。

### Why this is preferred

- Serde 的 contract 本来在 LMCache object layer；
- Connector/RequestTracker 已负责 vLLM physical slots ↔ LMCache object bridge；
- codec size/offset/kernel tiling可直接由 `MemoryLayoutDesc + head_dim + G` 推导；
- P2 与 P1 的 allocator/page ownership保持解耦；
- 未来 chunk size 改变也不要求 allocator page size跟着 codec 变化。

V1 只需 freeze 一个固定 `G`（例如候选 16，但正式值由 M1 approval），不做 data-dependent group。

## 3. Numeric Format Draft

每个 quant_group：

```text
amax  = max(abs(x))
scale = max(amax / 127, eps)
q     = clamp(round(x / scale), -127, 127)
xhat  = q * scale
```

- symmetric INT8；
- scale candidate = FP16；
- K/V优先 separate scale；
- nonfinite input policy必须显式冻结；
- FP32 scale只是 numeric fallback，不是 architecture change。

## 4. Serialized Layout

V1 只用 fixed deterministic layout：

```text
fixed metadata (only if needed)
scale region
INT8 payload region
```

所有 offsets由：

```text
MemoryLayoutDesc + frozen codec config
```

推导，不依赖 tensor values。

禁止 V1：pickle、entropy coding、variable-length payload、data-dependent header、mixed bit、outlier side channel。

## 5. Scoped Exact-Size Contract

### Only for V1/V2 fixed format + FSL2Adapter

Per object：

```text
estimate_serialized_size(layout)
== allocated serialized temp MemoryObj byte length
== corresponding `.data` logical file length (`stat().st_size`)
```

Per store operation：

```text
sum(bytes of newly persisted objects)
== L2StoreResult.bytes_transferred
```

FS adapter会 skip already-existing key，因此测试必须控制 namespace/pre-existing state。

这不是其它 backend 的 universal invariant。

## 6. Store Lifecycle — Upstream Owns Async Plumbing

```text
raw L1 read lock
→ SerdeL2AdapterWrapper reserves serialized temp
→ AsyncSerdeProcessor invokes BlockInt8Serializer
→ serialize completion
→ temp becomes source for inner FSL2Adapter
→ FS writes full temp byte_array
→ L2 completion
→ temp release
```

我们的 Serializer 不管理：thread、eventfd、L1 lock、FS coroutine、temp cleanup。

## 7. Load Lifecycle

```text
caller raw destination
+ wrapper serialized temp
→ FSL2Adapter loads `.data` into temp
→ BlockInt8Deserializer
→ raw destination MemoryObj
→ completion/temp release
→ connector resume
```

Load必须在 raw destination publish/consume前完成；partial/corrupt file不能静默发布“成功” KV。

## 8. Implementation Ownership

We implement：

```text
Block-INT8 format spec
Serializer / Deserializer
exact estimator
factory registration
PyTorch oracle
format/Serde tests
E2E byte/quality evidence
```

Upstream owns：

```text
MemoryObj
AsyncSerdeProcessor
SerdeL2AdapterWrapper
L1 locking/lifetime
FSL2Adapter
vLLM connector/slot mapping
```

## 9. Failure Tree

- layout unknown → stop at M1, no format freeze；
- quant unit fail → math/group/offset；
- unit correct but file size mismatch → estimator/temp allocation/backend；
- file/load bytes correct but tensor wrong → deserialize/layout；
- tensor correct but generation wrong → connector/slot mapping/quality；
- quality degraded only → lossy trade-off，不叫 lifecycle corruption。

## 10. Upstream Test Pattern to Reuse

Compatibility baseline 已有两类非常直接的 test reference：

1. `tests/v1/distributed/serde/test_turboquant.py`：验证 `MemoryLayoutDesc`、head_dim、serialized-size、invalid layout 和 GPU codec configuration；
2. `tests/v1/distributed/serde/test_serde_fs_e2e.py`：真实跑 L1 write → serialize → filesystem store → clear L1 → prefetch/load → deserialize，并检查文件存在、round-trip quality 与 temp/L1 memory cleanup。

P2 V1 应先复用这些 test **structure**，替换为 Block-INT8，再增加 exact bytes assertions；之后才进入 vLLM connector E2E。不要从零另造 async/FS test harness。
