# GitHub Targeted Revalidation — v1.2 Final Contract

Date: 2026-08-24 Asia/Taipei

Purpose: support the final documentation-contract hardening. These are public pinned/reference-source facts, **not** substitutes for P1/P2 local M0 source/runtime evidence.

## P1 — pinned vLLM `568afb3...`

### 1. `BlockTable.add_row()` is a row-rebuild primitive, but does not clear old tail

`BlockTable.add_row()` resets `num_blocks_per_row[row]` to zero and appends the supplied IDs. `append_row()` writes only the new active slice. Therefore arbitrary full-row reconstruction is a real upstream primitive, but a shorter live replacement can leave stale values beyond the new active count.

**Contract consequence:** P1 M1/M2 must prove stale tail is unreachable by all consumers or use a thin clear-old-range + rebuild semantic. “Primitive exists” is not the same as “live reclaim composite transition is already proven.”

Reference: `vllm/v1/worker/block_table.py`.

### 2. Hybrid block mapping exists

Pinned `BlockTable` supports `kernel_block_size != kv_manager block_size` and computes `blocks_per_kv_block`. This would complicate the one logical/allocator block ↔ one block-table-column assumption.

**Contract consequence:** Core freezes `kernel_block_size == kv_manager_block_size`, `blocks_per_kv_block == 1`, DCP/PCP=1 and CP interleave=1. Other layouts are compatibility work, not Core.

Reference: `vllm/v1/worker/block_table.py`.

### 3. Runner already has append vs full-replacement state transitions

Pinned `GPUModelRunner._update_states()` appends new IDs for normal running requests and replaces the full block-id list when resuming from preemption.

**Conclusion:** reclaim-specific replacement is an adaptation of existing runtime state patterns, not a new BlockTable architecture.

Reference: `vllm/v1/worker/gpu_model_runner.py`.

### 4. Tangram still validates logical-position / physical-slot split

Tangram keeps model positions derived from logical computed progress while compression-aware slot positions derive from effective sequence length, and returns compression/effective-length/free-ID results back through worker→scheduler control flow.

**Conclusion:** P1 logical/physical split, effective allocation and worker-ACK lifecycle remain reference-backed.

References: Tangram audited commit `8e1cbfa...`, `gpu_model_runner.py`, `kv_cache_manager.py`, scheduler/outputs paths.

## P2 — LMCache `a7afade...`

### 5. TurboQuant tests define codec around LMCache object layout

Compatibility-baseline tests describe KV `MemoryLayoutDesc` as `[2, num_layers, num_tokens, hidden_dim]` and derive `hidden_dim = num_heads × head_dim` for codec size/config tests.

**Contract consequence:** P2 `Block-INT8` should define a codec-domain block (`K/V × layer × KV head × fixed token group × head_dim`) rather than trying to align with vLLM `BlockPool` pages.

Reference: `tests/v1/distributed/serde/test_turboquant.py`.

### 6. Real filesystem Serde E2E test already exists

Pinned LMCache `test_serde_fs_e2e.py` exercises L1 write → serialize → real filesystem store → clear L1 → prefetch/load → deserialize → verify, including file existence, round-trip quality and memory cleanup.

**Contract consequence:** P2 V1 should adapt this test pattern before inventing a new async/FS harness.

Reference: `tests/v1/distributed/serde/test_serde_fs_e2e.py`.

### 7. FSL2 byte accounting is deterministic, but FSL2 has no fixed capacity admission

FSL2 stores `obj.byte_array`, counts actual newly written bytes, and skips already-existing keys. The backend reports no max capacity by default and does not provide a global-eviction capacity signal.

**Contract consequence:** `.data` / serialized byte reduction is a deterministic runtime fact; “more objects under fixed N-byte budget” is a derived/external-quota result, not automatic FSL2 admission capacity.

Reference: `lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py`.

### 8. Filesystem I/O regime matters

FSL2 normal I/O is buffered and therefore affected by Linux page cache. `use_odirect=True` only represents O_DIRECT when alignment requirements are met; otherwise the implementation can fall back to buffered I/O.

**Contract consequence:** crossover figures must label buffered-warm, large-working-set/new-key, or verified-O_DIRECT regimes. Storage footprint is independent of this latency regime.

Reference: `fs_l2_adapter.py`.

## Governance / License Note

Audited vLLM, Tangram, Sparse-vLLM and LMCache reference repositories are Apache-2.0. Semantic/reference reuse is distinct from direct code adaptation. Any `ADAPTED_CODE` or `VERBATIM_CODE` must be recorded in each project `REFERENCE-PORTING-LEDGER.md` and preserve applicable copyright/header/NOTICE requirements.

## Authority Note

These observations are documentation-hardening SOURCE_FACTs from public repos. M0 still must freeze actual local worktree/import/environment/signatures and may supersede this package if fixed local source differs.
