# R1-A1 Source Trace

## `vllm/config/cache.py`

ORIGINAL ROLE:
  KV cache configuration schema and structural configuration hash.
ADDED:
  `CacheConfig.page_group_size: int | None = None` with positive-value validation.
CHANGED:
  None; the field remains included in `compute_hash()` factors because it is not
  listed among ignored runtime/derived fields.
REMOVED:
  None.
WHY:
  Define future Ragged physical page width `Hp` while preserving `None` Dense
  behavior and avoiding public runtime activation in this slice.
UPSTREAM:
  Programmatic engine/cache configuration.
DOWNSTREAM NOW:
  Focused config/spec tests.
DOWNSTREAM FUTURE:
  Attention spec dispatch and Ragged activation gate.

## `vllm/v1/kv_cache_interface.py`

ORIGINAL ROLE:
  KV storage contracts, page-byte accounting, memory accounting, and spec merge
  behavior.
ADDED:
  `RaggedAttentionSpec(AttentionSpec)` with `Hkv`, `Hp`, per-layer group count,
  Hp-sized page bytes, identity max-memory accounting, and Ragged-preserving
  merge.
CHANGED:
  None in existing Dense specs; the pre-existing local docstring modification is
  retained.
REMOVED:
  None.
WHY:
  Keep semantic model geometry (`Hkv`) separate from physical page geometry
  (`Hp`) without binding ownership or placement to Dense managers.
DOES NOT OWN:
  `MemberPlacementMap`, page IDs, allocator, retained positions, scheduler
  authority, attention dispatch, or runtime manager state.

## R1-A1 Boundary

NOW:
  `CacheConfig.page_group_size` -> `RaggedAttentionSpec` definition/tests -> STOP

FUTURE ACTIVATION:
  `Attention.get_kv_cache_spec` -> explicit registry registration -> real Ragged
  manager/planner -> global physical page pool and cluster rows.
