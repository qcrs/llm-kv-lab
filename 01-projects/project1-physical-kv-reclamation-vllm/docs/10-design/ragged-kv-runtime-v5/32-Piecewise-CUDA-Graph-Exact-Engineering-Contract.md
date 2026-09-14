# Piecewise CUDA Graph for Ragged KV — Exact Engineering Contract

**Goal:** 不追求 Full CUDA Graph。复用 vLLM 0.26 native piecewise compilation，把动态 Ragged KV 操作留在 eager island，同时保持其余模型 compute 的 graph/compile 收益。

---

# 1. Existing v0.26 Architecture to Reuse

关键 splitting seam：

```text
vllm::unified_kv_cache_update
vllm::unified_attention_with_output
```

当前 compilation config 已把 attention custom op 作为 piecewise boundary，并对 KV cache update 有 graph exclusion logic。

因此首版 **不新增**：

```text
vllm::unified_attention_ragged
```

除非 R9 profiler/compile trace证明已有 seam无法满足。

---

# 2. Target Execution Shape

```text
Captured/Compiled Piece A
  embedding / qkv / rope / other compute
        ↓
Eager Island 1
  Ragged KV update
        ↓
Eager Island 2
  Ragged metadata-dependent FA2
        ↓
Captured/Compiled Piece B
  output projection / FFN / later compute
```

---

# 3. R1-F Compatibility Smoke

R1-F 只回答：

```text
Can identity Ragged run under native PIECEWISE mode correctly?
```

不做：

```text
persistent metadata optimization
custom op redesign
full graph
performance claim
```

允许状态：

```text
PASS
DEFERRED(reason)
```

---

# 4. R9 Persistent Buffer Contract

建议新对象：

```text
RaggedStepBuffers
```

至少预分配：

```text
member_block_tables
member_virtual_slots
member_seq_lens
member_query_start_loc
group_effective_lens
member→group/column maps
prefill permutation indices if bounded
```

原则：

```text
contents may change
addresses must remain stable for replay-facing tensors
```

---

# 5. CPU→GPU Metadata

优先：

```text
UVA-backed metadata
pinned CPU buffers
async/staged H2D
```

避免：

```text
list → torch.tensor(cuda) every step
new torch.empty per layer
CPU sync from GPU shape values
```

---

# 6. Decode Fast Path

decode q_len=1 是 CG 最值得优化的 path。

目标：

```text
precomputed member ordering
persistent decode query_start_loc template
persistent member table output buffers
only overwrite active prefix
```

尽量让 Python 工作与 `num_layers × Hkv` 解耦。

---

# 7. Prefill Policy

Prefill/mixed metadata更动态。

首版允许：

```text
prefill falls back to eager metadata/materialization
```

不要为了 capture prefill 拉高复杂度。

R9 的重点是 decode TPOT。

---

# 8. Tests

## C1 Address Stability

连续 N steps：

```text
data_ptr(member_block_tables) stable
data_ptr(member_slots) stable
...
```

## C2 Replay Correctness

same batch shape，多次不同 requests/content，输出与 eager reference一致。

## C3 Shape Bucket

不同 decode batch sizes命中预期 capture bucket，不读取 stale padded slots。

## C4 Mixed Batch Fallback

mixed/prefill path如果不 capture，必须正确 fallback，而不是错误 replay。

---

# 9. Performance Experiment

三组：

```text
A Dense + Piecewise CG
B Ragged eager
C Ragged + Piecewise CG
```

固定：model / request length / batch / GPU / cache budget。

记录：

```text
TPOT
tokens/s
CPU step wall time
metadata construction time
attention eager-island GPU time
CUDA launch count/gaps
```

---

# 10. Success Definition

CG 成功不是：

```text
no crash
```

而是：

```text
correctness
+ replay-safe pointers
+ measurable reduction in CPU/launch overhead
+ explainable remaining eager island
```

如果 C 与 B 没有改善，保留 compatibility claim，不写 optimization claim。

---

# 11. Full CUDA Graph Decision

当前项目默认：

```text
NO-GO
```

只有以下同时满足才重新评估：

```text
Piecewise已经稳定
profile证明eager island仍是主要瓶颈
metadata shapes可以静态化
实现成本不影响R10 closure
```
