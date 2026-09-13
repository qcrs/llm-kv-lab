# GitHub Targeted Revalidation — v1.1 Hardening

Date: 2026-08-24 Asia/Taipei

Purpose: 只核实本次文档修订依赖的关键 public-source facts。不是 local worktree/runtime evidence；M0 仍需在实际 implementation worktree 复核。

## P1 — pinned vLLM `568afb3...`

### `BlockTable.add_row()`

File: `vllm/v1/worker/block_table.py`

Observed source semantic:

```text
add_row(block_ids, row_idx)
→ num_blocks_per_row[row_idx] = 0
→ append_row(block_ids, row_idx)
```

Conclusion: arbitrary supplied physical block-id list can rebuild active row. This primitive is A/exact reference; reclaim-specific transition remains B/adaptation.

GitHub:
https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/block_table.py

### Normal runtime uses `add_row()`

File: `vllm/v1/worker/gpu_input_batch.py`

`InputBatch.add_request()` sets `num_computed_tokens_cpu` and calls:

```text
self.block_table.add_row(request.block_ids, req_index)
```

Conclusion: full-row reconstruction is not an unused helper.

GitHub:
https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/gpu_input_batch.py

### Qwen3 positional transform order

File: `vllm/model_executor/models/qwen3.py`

`Qwen3Attention.forward()` applies:

```text
q, k = self.rotary_emb(positions, q, k)
```

before:

```text
attn_output = self.attn(q, k, v)
```

Conclusion: standard Qwen3 RoPE path is a suitable first validation target for preserving logical model positions while compacting physical KV storage. This does **not** prove the same for ALiBi/relative bias/mRoPE/dual-chunk paths.

GitHub:
https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/model_executor/models/qwen3.py

## P1 — Tangram `8e1cbfa...`

File: `vllm/v1/core/kv_cache_manager.py`

Observed:

```text
if effective_num_cached_tokens is not None:
    num_tokens_need_slot =
        effective_num_cached_tokens
        + num_new_tokens
        + num_lookahead_tokens
```

Conclusion: allocation after compression can size from effective physical occupancy rather than logical computed progress; this is direct reference for P1 allocation accounting.

GitHub:
https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/core/kv_cache_manager.py

## P2 — LMCache `a7afade...`

### Torch ABI baseline

File: `pyproject.toml`

Build requirement includes:

```text
torch==2.11.0
```

Conclusion: this compatibility commit is appropriate as source baseline for vLLM 0.26 / torch 2.11 family; local native-extension import still requires M0 proof.

GitHub:
https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/pyproject.toml

### FP8 exact estimator

File: `lmcache/v1/distributed/serde/fp8.py`

Observed implementation returns exactly one serialized byte per tensor element and comments that inflated estimates inflate persisted L2 bytes because the wrapped adapter stores the whole MemoryObj.

Conclusion: fixed-size custom Serde should use exact estimator where possible.

GitHub:
https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/serde/fp8.py

### Filesystem L2 byte accounting

File: `lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py`

Observed store path:

```text
buf = obj.byte_array
size = len(buf)
write buf
bytes_written += size
L2StoreResult(success, bytes_written)
```

It also skips a key if its target file already exists.

Conclusion: for isolated/clean V1 fixed-size FSL2 tests, per-object temp byte length can be compared with `.data` file size, while per-store `bytes_transferred` must account only newly written objects. The equality is backend-scoped, not universal.

GitHub:
https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py

## Authority Note

These are `SOURCE_FACT` from public pinned/reference repositories for documentation hardening. They do not replace:

```text
actual local worktree identity
actual import path
runtime shape/device observation
unit/E2E evidence
profiler evidence
```

Those remain M0+ tasks.
