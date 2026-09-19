# R1-A2 Exact Implementation Contract — Physical Namespace / Planner / Global Backing

**Parent:** P1-V2-DESIGN-REVIEW-01  
**Slice:** R1-A2  
**Activation:** FORBIDDEN — this slice must not enable production Ragged startup  
**Dependency:** R1-A1 definition exists  
**Parallelism:** may proceed independently of B1

---

# 0. Objective

Implement and prove:

```text
RaggedAttentionSpec
+
available worker KV memory
→
Nphys physical page slots
+
one global raw-backing descriptor
```

No Scheduler ownership logic is introduced here.

---

# 1. Frozen Semantics

Given:

```text
P = RaggedAttentionSpec.page_size_bytes
M = available_memory
```

compute:

```text
Nphys = floor(M / P)
```

Ragged meaning:

```text
KVCacheConfig.num_blocks = Nphys
```

Generate:

```text
KVCacheTensor(
    size = Nphys * P,
    shared_by = all supported local Ragged layer names,
)
```

When upstream BlockPool is later used:

```text
page 0 = NULL
allocatable pages = Nphys - 1
```

---

# 2. Primary Files

Expected primary modifications:

```text
vllm/v1/core/kv_cache_utils.py
tests/v1/core/test_ragged_kv_cache_planner.py
```

Potential narrow supporting modification only if necessary:

```text
vllm/v1/kv_cache_interface.py
```

If isolated raw-allocation validation is required:

```text
vllm/v1/worker/gpu/attn_utils.py
```

may be exercised or minimally adjusted, but no runtime activation is allowed.

---

# 3. Planner Dispatch

Branch on resolved Spec:

```text
RaggedAttentionSpec
```

not raw:

```text
CacheConfig.page_group_size
```

Config is intent; Spec is resolved storage contract.

---

# 4. Supported A2 Planner Shape

R1 A2 may require exactly one synthetic Ragged KVCacheGroupSpec.

That group may contain multiple local supported Attention layers.

Physical clusters remain internal to future Ragged manager/worker state.

Do not create one KVCacheGroup per physical cluster.

---

# 5. Capacity / Concurrency Audit

Audit at least:

```text
get_num_blocks
get_kv_cache_config_from_groups
get_max_concurrency_for_kv_cache_config
_pool_bytes_per_block or equivalent
startup capacity reporting/logging
```

Any formula consuming `KVCacheConfig.num_blocks` must be checked for Dense logical-depth assumptions.

For R1 identity:

```text
G = Hkv/Hp
C = L*G
D = ceil(max_model_len/B)
```

one full-length request needs approximately:

```text
D * C
```

physical pages, subject to NULL/reservation details.

A concurrency helper must not treat `Nphys` as Dense logical block count.

---

# 6. num_gpu_blocks_override

R1 behavior:

```text
Ragged Spec + num_gpu_blocks_override
→ explicit unsupported error
```

Do not silently reinterpret a Dense logical-block override as physical-page count.

No new public CLI is required.

---

# 7. Raw Backing Descriptor

For supported local layer names:

```text
layer_names = [...]
```

return exactly one descriptor:

```text
KVCacheTensor(
    size = P * Nphys,
    shared_by = layer_names,
)
```

No per-layer division.

No independent per-layer Ragged backing.

---

# 8. Identity Memory Oracle

For identity placement:

```text
G = Hkv/Hp
P_ragged = P_dense/G
C = L*G
```

Then:

```text
D * C * P_ragged
=
D * L * P_dense
```

This equality is mandatory test evidence.

R1 changes allocation granularity, not identity total KV bytes.

---

# 9. NULL Page Accounting

Backing contains all `Nphys` slots including page 0.

Later BlockPool reserves page 0.

Do not subtract NULL from raw backing size.

---

# 10. Dense Regression

When no Ragged Spec is supplied, existing Dense planner behavior remains unchanged.

Do not change:

```text
Dense get_num_blocks semantics
FullAttention grouping
Dense per-layer tensor structure
```

---

# 11. Forbidden Changes

R1-A2 must not modify/activate:

```text
Attention.get_kv_cache_spec production dispatch
production Ragged Registry mapping
FullAttentionManager
RaggedAttentionManager production hookup
BlockPool implementation
Scheduler allocation
SchedulerOutput Ragged carriers
Worker RaggedBlockTables
FA2 read/write
Ragged physical reshape
compression/policy
CUDA Graph
Tangram ring allocator
```

---

# 12. Required Tests

## T1 — Exact physical page count

Synthetic `available_memory` and known `P_ragged` produce exact `Nphys`.

## T2 — One global descriptor

Assert:

```text
len(kv_cache_tensors) == 1
size == Nphys * P
shared_by == expected local Ragged layers
```

## T3 — Identity memory invariant

Dense identity bytes equal Ragged identity bytes.

## T4 — Concurrency semantic guard

Construct a case where using `Nphys` as Dense logical depth overestimates concurrency by roughly `L*G`; assert corrected Ragged accounting.

## T5 — Dense regression

Existing FullAttention planner output remains unchanged.

## T6 — Override rejection

Ragged + `num_gpu_blocks_override` fails explicitly.

---

# 13. Acceptance Evidence

Required:

```text
targeted pytest
py_compile
git diff --check
focused diff audit
Dense planner regression
no-activation audit
```

No engine smoke is required because production activation is intentionally absent.

---

# 14. Handoff Condition

A2 is complete only when:

```text
Given a RaggedAttentionSpec,
planner output defines a worker-local physical page namespace
and one shared raw backing,
without changing Scheduler ownership or activating Ragged startup.
```
