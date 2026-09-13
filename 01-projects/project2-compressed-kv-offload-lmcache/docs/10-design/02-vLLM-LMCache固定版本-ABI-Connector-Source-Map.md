# 02 — vLLM / LMCache Fixed Baseline, ABI and Connector Source Map

## 1. Implementation Baselines

```text
vLLM 0.26.0
commit 568afb3a13806beb53bb2e6bd518269357b237c0

LMCache
commit a7afadebb9248b62b5c533ce2c12297e9d94fc4a

Torch ABI family
2.11.x
```

LMCache compatibility commit 的 `pyproject.toml` build requirement 明确是 `torch==2.11.0`。later snapshot `f9addd...` 是 source comparison，不自动升级为 implementation baseline。

## 2. M0 ABI Gate

不能只测 `import lmcache`。必须记录：

- exact Python/torch/CUDA/Triton；
- vLLM import path/version/commit；
- LMCache import path/commit；
- native extension真正 import/load；
- CUDA/Triton backend是否真正启用；
- 是否发生 silent fallback；
- worktree dirty status。

任意 native symbol mismatch / fallback ambiguity → stop，不能进入 codec。

## 3. vLLM Connector Surface

Pinned vLLM `LMCacheConnectorV1` 已提供 worker-side：

```text
register_kv_caches
start_load_kv
wait_for_layer_load
save_kv_layer
wait_for_save
```

scheduler-side connector lifecycle由 upstream处理。P2 不新造 vLLM↔external cache bridge。

## 4. LMCache Request / Block / Slot Bridge

`vllm_v1_adapter.py` 的 RequestTracker/ReqMeta维护 request token/block state并从 allocated block IDs + block_size生成 slot mapping。P2 custom codec不重写 slot mapping framework。

M0/M1 仍需冻结 actual：

```text
MemoryObj layout
MemoryObj device
chunk/object boundaries
K/V/head dimension interpretation
```

这些决定 `quant_group`，不能从“vLLM physical block”字面猜。

## 5. Serde / L2 Source Map

| Layer | Fixed source role | P2 ownership |
|---|---|---|
| Serde ABC | Serializer/Deserializer/size contract | implement Block-INT8 variant |
| AsyncSerdeProcessor | task/thread/event lifecycle | reuse |
| SerdeL2AdapterWrapper | temp object + serialize→L2 / L2→deserialize | reuse |
| FSL2Adapter | real `.data` persistence + byte accounting | use as deterministic backend |
| FP8 serde | minimal fixed-size template | compare/reference |
| TurboQuant | Triton codec + CPU/CUDA staging reference | compare/reference |

## 6. Code > Doc Rule

如果 design doc 与 same-commit implementation有差异：

```text
same-commit code
+ E2E object/file/byte evidence
```

优先于旧设计文字。FP8 estimator 的 exact 1 byte/element 就是已知例子。

## 7. v1.2 Object-Domain Codec/Test Anchors

Pinned LMCache compatibility baseline 的 TurboQuant tests将 KV `MemoryLayoutDesc` 示例写成：

```text
[2, num_layers, num_tokens, hidden_dim]
hidden_dim = num_kv_heads × head_dim
```

因此 P2 M1/M2 的 source audit 应冻结的是 **LMCache object axes / head_dim / token group**，而不是寻找 vLLM physical page boundary。Connector/RequestTracker 已经承担 physical-slot→object bridge。

同时 pinned `tests/v1/distributed/serde/test_serde_fs_e2e.py` 是真实 filesystem Serde lifecycle reference，应作为 Block-INT8 V1 test harness 的首要结构参考。
