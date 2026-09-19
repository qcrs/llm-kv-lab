# R1-A1 Minimal Implementation Contract v2 — Macro-Audited

**Project:** Physical KV Cache Reclamation for vLLM 0.26  
**Round:** R1 — Identity Ragged Paging  
**Slice:** R1-A1 — Spec / Config Contract  
**Status:** READY FOR IMPLEMENTATION  
**Supersedes:** `R1-A1-Minimal-Implementation-Contract.md`  
**Primary goal:** freeze the semantic/storage contract without prematurely activating an incomplete Ragged runtime.

---

# 0. Macro Decision

After re-auditing:

- `aiha-lab/tangram` at reference commit `6fa551fc8f6edcc118a2a39b3554ee1520e91edd`;
- `qcrs/llm-kv-lab` Project-1 canonical v5 documents;
- target `qcrs/vllm` v0.26-derived MRv2 source;

R1-A1 is narrowed to a **definition slice**, not an engine-activation slice.

The previous draft allowed:

```text
CacheConfig.page_group_size
→ Attention.get_kv_cache_spec()
→ RaggedAttentionSpec
→ stop before Registry
```

That is useful for source learning, but as a production implementation boundary it creates a half-enabled feature:

```text
configuration accepts Ragged
→ Attention emits RaggedAttentionSpec
→ Registry has no real Ragged manager
→ engine necessarily fails later
```

The v2 contract therefore freezes:

```text
R1-A1 = Config + Spec + geometry/accounting contract

R1-A2 entry / activation gate =
Attention dispatch
+ explicit Registry registration
+ real Ragged manager/planner consumption
```

This avoids temporary binding to `FullAttentionManager` and avoids exposing a user-facing feature whose runtime ownership semantics do not yet exist.

---

# 1. Long-Term Architecture This Slice Must Preserve

The future system is not merely:

```text
layer × adjacent head-group
```

The long-term abstraction is:

```text
semantic member
    = (local_layer, local_kv_head)

MemberPlacementMap
    member -> (physical_cluster_id, column)

physical cluster/page row
    request × cluster × depth -> page_id
```

For R1 identity placement only:

```text
Hkv = semantic local KV heads
Hp  = heads stored per physical page
G   = Hkv / Hp
L   = local supported attention layers

C = L * G

cluster_id = layer_idx * G + head // Hp
column     = head % Hp
```

But **`cluster_id = layer*G+group` is not a permanent architectural invariant**.

It is the R1 identity placement function.

Future AOT placement may change only `MemberPlacementMap` while preserving:

```text
physical page pool
physical cluster row abstraction
virtual-block addressing
scheduler physical ownership protocol
```

This is intentionally aligned with Tangram's explicit `(layer, head) -> (cluster, column)` map, while keeping the first implementation per-layer/identity only.

---

# 2. Frozen Symbols and Semantics

```text
Hkv = semantic KV head count on current rank
Hp  = physical page width in KV-head columns
G   = Hkv / Hp
B   = block_size in tokens
Dk  = K head dimension
Dv  = V head dimension
S   = dtype bytes
L   = local supported attention layers
C   = physical cluster count
```

R1 identity:

```text
C = L * G
```

Future:

```text
C is owned by placement geometry;
do not require every consumer to recompute C as L*G.
```

`RaggedAttentionSpec.num_kv_heads` MUST remain `Hkv`.

Do not encode `Hp` into `num_kv_heads`.

---

# 3. Ragged Page Geometry

Generic page payload bytes:

```math
P_{ragged}
= B \times Hp \times (D_k + D_v) \times S
```

Current R1 gate:

```text
Dk == Dv == D
```

therefore:

```math
P_{ragged}
= 2 \times B \times Hp \times D \times S
```

Dense per-layer page:

```math
P_{dense}
= B \times Hkv \times (D_k + D_v) \times S
```

Identity invariant:

```math
G \times P_{ragged} = P_{dense}
```

R1 itself saves no KV bytes.

---

# 4. Class Hierarchy Decision

Frozen:

```text
KVCacheSpec
  ↑
AttentionSpec
  ├── FullAttentionSpec
  └── RaggedAttentionSpec
```

Use:

```python
class RaggedAttentionSpec(AttentionSpec):
    ...
```

Do not use:

```python
class RaggedAttentionSpec(FullAttentionSpec):
    ...
```

## Reason

In target vLLM, Spec inheritance participates in runtime classification through Registry/MRO and uniform-type grouping.

Ragged mathematical attention is still Full Attention, but Ragged **storage and ownership** are not Dense FullAttention storage semantics.

Future registration must be explicit:

```text
RaggedAttentionSpec
→ RaggedAttentionManager
→ uniform_type_base_spec = RaggedAttentionSpec
```

No MRO fallback to `FullAttentionManager` is allowed as the canonical design.

---

# 5. R1-A1 Production Files in Scope

## 5.1 `vllm/config/cache.py`

Add internal opt-in storage geometry:

```python
page_group_size: int | None = None
```

Preferred validation if compatible with the current config model:

```python
page_group_size: int | None = Field(default=None, gt=0)
```

Semantics:

```text
None
→ existing Dense behavior

positive Hp
→ Ragged geometry requested by a later activation slice
```

### Important requirements

1. Default MUST remain `None`.
2. Do not copy Tangram's product default `4`; this project must preserve upstream Dense behavior unless explicitly enabled.
3. Do not expose a public CLI switch in R1-A1 if that would imply the engine feature is usable already.
4. `page_group_size` affects storage/runtime structure and MUST NOT be accidentally excluded from relevant configuration hashing.
5. Do not mutate prefix caching/backend/parallel settings in this slice.

---

## 5.2 `vllm/v1/kv_cache_interface.py`

Add:

```python
@dataclass(frozen=True, kw_only=True)
class RaggedAttentionSpec(AttentionSpec):
    page_group_size: int
    head_size_v: int | None = None
```

### `__post_init__`

Permanent geometry invariants only:

```text
Hp > 0
Hkv % Hp == 0
G >= 1
```

Normalize:

```text
head_size_v is None
→ head_size_v = head_size
```

Do not encode R1 temporary gates such as TP1, FA2-only, unquantized-only into the Spec's permanent geometry contract.

### Derived group count

```python
@property
def num_head_groups_per_layer(self) -> int:
    return self.num_kv_heads // self.page_group_size
```

This property is a **per-layer geometry helper**, not the future global physical cluster namespace.

### `real_page_size_bytes`

Prefer extending the target v0.26 `AttentionSpec` page-size layering rather than overriding higher-level padded/auxiliary semantics unnecessarily.

```python
@property
def real_page_size_bytes(self) -> int:
    return (
        self.block_size
        * self.page_group_size
        * (self.head_size + self.head_size_v)
        * get_dtype_size(self.dtype)
    )
```

### `max_memory_usage_bytes`

For one layer at identity retention:

```text
num_depths
× groups_per_layer
× ragged_page_bytes
```

Conceptually:

```python
num_depths = cdiv(max_model_len, self.block_size)
return (
    num_depths
    * self.num_head_groups_per_layer
    * self.page_size_bytes
)
```

The result must equal Dense per-layer max memory for the same model/config under R1 constraints.

### `merge`

Ragged merge must keep Ragged identity and reject incompatible page geometry.

At minimum verify compatible:

```text
block_size
num_kv_heads
head_size
head_size_v
dtype
kv_quant_mode
page_size_padded / stride-relevant inherited fields
page_group_size
```

Return `RaggedAttentionSpec`, not `FullAttentionSpec`.

Do not silently merge different `Hp` values.

---

# 6. Files Explicitly NOT Modified in R1-A1 v2

Do not modify production activation paths yet:

```text
vllm/model_executor/layers/attention/attention.py
    Attention.get_kv_cache_spec()

vllm/v1/kv_cache_spec_registry.py

vllm/v1/core/single_type_kv_cache_manager.py

vllm/v1/core/kv_cache_manager.py

vllm/v1/core/kv_cache_utils.py

BlockPool
Scheduler
BlockTables / MultiGroupBlockTable
ModelRunner KV allocation/reshape
FlashAttention backend
slot mapping
KV write/read kernels
```

Reason:

Ragged activation is not valid until there is a real consumption path for:

```text
Ragged Spec
→ global physical page planner
→ Ragged manager-owned canonical state
→ worker physical rows
```

Do not create a fake transition by registering Ragged to `FullAttentionManager`.

---

# 7. A1 Validation Ownership

R1-A1 implements only geometry/config validation.

## Permanent Spec-local validation

```text
Hp > 0
Hkv % Hp == 0
```

## Deferred to activation / A2 entry

The following require runtime or layer/global facts and should not be hardwired into A1 Spec:

```text
FlashAttention backend
HND / Ragged-specific physical layout support
FullAttention-only
KVQuantMode.NONE
head_size_v == head_size
TP1 / PP1 / DCP1 / PCP1
prefix cache off
spec decode off
connector/offload off
MLA/SWA/hybrid off
```

These become explicit activation gates when `Attention.get_kv_cache_spec()` is wired to emit Ragged in production.

---

# 8. High-Value Tests Only

Do not build a branch-coverage test matrix.

Keep approximately three focused tests.

## T1 — Canonical geometry + identity accounting

Example:

```text
Hkv=8
Hp=2
B=16
Dk=Dv=128
BF16
```

Verify:

```text
spec.num_kv_heads == 8
spec.page_group_size == 2
G == 4
G * P_ragged == P_dense
max_memory_identity == dense max memory
```

This is the core semantic test.

## T2 — `dataclasses.replace` preservation

The target worker path later performs:

```python
replace(spec, indexes_kv_by_block_stride=...)
```

Verify on a Ragged Spec that:

```text
type remains RaggedAttentionSpec
page_group_size preserved
head_size_v preserved
only requested inherited field changes
```

This is more valuable than enumerating many trivial invalid values.

## T3 — One representative invalid geometry

Example:

```text
Hkv=8
Hp=3
```

Must reject.

Do not separately test `Hp=-1`, `Hp=0`, `Hp=5`, etc. unless required by existing project conventions.

Optional tiny config assertion may be folded into T1:

```text
CacheConfig.page_group_size default is None
```

---

# 9. Source-Chain Record — Required

Maintain a compact file/slice record:

```text
R1-A1-SOURCE-TRACE.md
```

or an equivalent section in the canonical slice record.

For every changed file record:

```text
PATH
ORIGINAL ROLE
ADDED
CHANGED
REMOVED
WHY
UPSTREAM INPUT
DOWNSTREAM CONSUMER
DEFERRED ACTIVATION
```

Example:

```text
vllm/config/cache.py
- Added: page_group_size=None
- Changed: config schema/hash implications only
- Removed: none
- Role: describe requested Ragged physical page width
- Downstream now: RaggedAttentionSpec unit construction
- Production activation: deferred to A2 entry
```

```text
vllm/v1/kv_cache_interface.py
- Added: RaggedAttentionSpec
- Role: storage geometry/accounting contract
- Does NOT own: placement map, page IDs, allocator, scheduler ownership, retained positions
```

---

# 10. STOP Boundary

R1-A1 v2 ends at:

```text
CacheConfig geometry
+
RaggedAttentionSpec geometry/accounting
+
unit evidence
```

It stops **before**:

```text
Attention production dispatch
Registry registration
Manager instantiation
planner page-count semantics
physical backing
cluster rows
member map runtime consumption
```

---

# 11. Next Activation Gate

The next activation step must be designed as one coherent seam, not separate hacks:

```text
Attention.get_kv_cache_spec()
        ↓
RaggedAttentionSpec
        ↓
Registry explicit Ragged type
        ↓
RaggedAttentionManager
        ↓
Global Page Pool
        ↓
Physical cluster ownership rows
```

Recommended long-term registration:

```text
manager_class = RaggedAttentionManager
uniform_type_base_spec = RaggedAttentionSpec
```

Reuse lower-level BlockPool/free-queue primitives where correct, but do not reuse Dense request→one-row ownership semantics as the canonical Ragged model.

---

# 12. Forward-Compatibility Requirements for A2+

These are not implemented in A1, but A1 must not make them harder.

## 12.1 Physical cluster namespace

Canonical future state should use:

```text
C = number of physical clusters
```

not permanently hardcode every API to:

```text
GT = L * G
```

R1 identity has `C = L*G`; future placement maps may differ.

## 12.2 Member placement is separate from Spec

`RaggedAttentionSpec` owns page geometry.

A separate `MemberPlacementMap` owns:

```text
(layer, kv_head)
→ (cluster_id, column)
```

## 12.3 Ragged layout must have one source of truth

A2/A3 should introduce a dedicated layout helper/contract owning:

```text
physical shape
stride contract
virtual block view
(page,column) <-> virtual_block
address oracle
```

Do not duplicate virtual-block arithmetic across planner, ModelRunner, KV write, FA metadata, and compressor.

## 12.4 HND/NHD

Target FA backend can expose multiple physical stride orders.

R1's zero-copy virtual block requires column-major/HND-like physical storage:

```text
[Bphys, Hp, N, 2D]
```

Therefore activation must either:

```text
A. hard-gate to the audited HND layout in R1
```

or preferably later:

```text
B. give Ragged its own explicit physical layout hook independent of Dense's global HND/NHD choice
```

Do not silently run Ragged over an NHD backing where `(page,column)` cannot be flattened zero-copy.

## 12.5 Scheduler canonical physical state

Do not copy Tangram's scalar-max frontier as the final model.

Future manager state should preserve per-cluster:

```text
E[request, cluster]
counts[request, cluster]
rows[request, cluster]
```

This is necessary to keep non-uniform groups non-uniform after later decode allocation.

## 12.6 Performance path remains replaceable

Member-major FA2 is the correctness adapter, not permanent canonical storage state.

Keep:

```text
cluster-major canonical ownership
→ derived member-major attention views
```

so a later profiler-driven fused/head-group-aware attention path can replace the derived adapter without redesigning allocator state.

---

# 13. Acceptance

R1-A1 v2 is PASS only if:

```text
[ ] CacheConfig adds opt-in Hp with Dense default unchanged
[ ] RaggedAttentionSpec directly extends AttentionSpec
[ ] Hkv and Hp remain semantically separate
[ ] canonical page bytes are correct
[ ] identity max-memory invariant matches Dense
[ ] merge preserves Ragged type and Hp
[ ] dataclasses.replace preserves Ragged fields/type
[ ] one representative invalid geometry rejects
[ ] no production Attention dispatch activated
[ ] no Registry/Manager/allocator modifications
[ ] no fake FullAttentionManager reuse
[ ] source-chain record lists exact file-level changes
```

---

# 14. Sources Re-Audited

Reference repository:

```text
aiha-lab/tangram
commit: 6fa551fc8f6edcc118a2a39b3554ee1520e91edd
```

Key reference files:

```text
vllm/v1/kv_cache_interface.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/worker/ragged_block_table.py
vllm/v1/attention/backends/ragged_layout.py
vllm/config/compression.py
benchmarks/tangram/speedup/run_speedup.sh
```

Target project:

```text
qcrs/llm-kv-lab
01-projects/project1-physical-kv-reclamation-vllm/
```

Target runtime source:

```text
qcrs/vllm
branch: p1/v2-token-compaction-v026
```

