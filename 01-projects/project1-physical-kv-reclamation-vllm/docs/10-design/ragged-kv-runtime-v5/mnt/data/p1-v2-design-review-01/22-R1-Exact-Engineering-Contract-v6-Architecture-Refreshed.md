# R1 Exact Engineering Contract — General Ragged Runtime / Identity Behavior

**Project:** Physical KV Cache Reclamation for vLLM  
**Target baseline:** vLLM v0.26.0  
**Status:** PROPOSED CANONICAL — architecture refreshed after P1-V2-DESIGN-REVIEW-01  
**Non-goal:** This document does not authorize source modification by itself.

---

# 0. R1 Goal

R1 SHALL establish the **general Ragged physical runtime architecture** while executing only the identity/uniform behavior.

```text
General Ragged Runtime
    + Identity MemberPlacementMap
    + Uniform physical frontier values
    + No compaction/reclamation
```

R1 is not allowed to build a uniform-only API that must be redesigned in R2.

Canonical variables:

```text
Hkv = semantic KV heads on current rank
Hp  = semantic KV members per physical page
B   = cache block size
C   = physical cluster count
M   = maximum page depth per cluster row
```

R1 identity only:

```text
G = Hkv / Hp
C = L_local * G

cluster = layer * G + head // Hp
column  = head % Hp
```

The identity arithmetic is an implementation of `MemberPlacementMap`, not a permanent runtime contract.

---

# 1. Global Architectural Invariants

## 1.1 Semantic head identity is preserved

```text
RaggedAttentionSpec.num_kv_heads = Hkv
```

Never rewrite semantic `num_kv_heads` to `Hp` to trick the backend.

## 1.2 Scheduler canonical ownership

```text
request
→ physical cluster
→ physical depth
→ physical page ID
```

Canonical per-request state:

```text
RaggedRequestPhysicalState
├─ effective_lens[C]
├─ page_counts[C]
└─ ordered_page_rows[C]
```

## 1.3 Physical page identity

Within one worker-local pool:

```text
page_id = one true fixed-size physical page slot
```

## 1.4 Worker is not allocator authority

```text
Scheduler RaggedAttentionManager = canonical ownership authority
Worker Ragged state              = execution mirror
BlockPool                        = allocator/free/refcount machinery
```

## 1.5 Backend-neutral address

Canonical address:

```text
(page_id, column, block_offset)
```

FA2 virtual block ID:

```text
vbid = page_id * Hp + column
```

is only an adapter encoding.

---

# 2. R1 Runtime Activation Rule

`CacheConfig.page_group_size != None` means only:

```text
Ragged requested
```

Production Ragged activation is forbidden until R1-E.

Before R1-E, all slices are validated by direct construction/injection of Ragged objects.

Do not make normal startup return `RaggedAttentionSpec` early.

---

# 3. R1-A1 — Spec / Config Definition

Define:

```text
CacheConfig.page_group_size
RaggedAttentionSpec(AttentionSpec)
```

Permanent Spec invariants:

```text
Hp > 0
Hkv % Hp == 0
num_kv_heads == Hkv
```

Ordinary non-quantized page payload:

```text
P_ragged
=
B * Hp * (Dk + Dv) * dtype_size
```

R1 identity:

```text
G * P_ragged == P_dense
```

R1 runtime limitations such as FA2-only, KVQuantMode.NONE, Dk==Dv, TP1, etc. are capability gates, not permanent Spec invariants.

A1 forbidden:

```text
production Attention dispatch
Registry activation
Manager activation
Planner activation
Worker runtime changes
```

---

# 4. R1-A2 — Physical Namespace / Planner / Global Backing

Given:

```text
P = RaggedAttentionSpec.page_size_bytes
M = available worker KV memory
```

compute:

```text
Nphys = floor(M / P)
```

Ragged meaning:

```text
KVCacheConfig.num_blocks = Nphys physical page slots
```

Generate one raw-allocation descriptor:

```text
KVCacheTensor(
    size = Nphys * P,
    shared_by = all supported local Ragged Attention layers,
)
```

With upstream BlockPool later:

```text
page 0 = NULL
allocatable IDs = [1, Nphys)
```

A2 must audit every planning/reporting consumer that assumes Dense `num_blocks` semantics, including concurrency/capacity calculation and override semantics.

R1 rejects ambiguous `num_gpu_blocks_override` use under Ragged mode.

A2 does **not** activate scheduler ownership.

---

# 5. R1-A3 — Ragged Physical Layout

Initial physical logical view:

```text
[Bphys, Hp, B, Dk + Dv]
```

For R1 `Dk == Dv`:

```text
[Bphys, Hp, B, 2D]
```

Required FA2 compatibility view:

```text
[Bphys * Hp, 1, B, 2D]
```

This must be zero-copy.

If current backend/global stride does not permit zero-copy `(page,column)` flattening:

```text
STOP
```

and introduce a dedicated Ragged layout/reshape seam.

Central Ragged layout helper owns:

```text
physical shape
stride contract
virtual view
(page,column)↔vbid
address oracle
identity placement helper
```

No duplicate virtual-block arithmetic across unrelated modules.

---

# 6. R1-B1 — Scheduler Ragged Ownership Manager

Use a dedicated:

```text
RaggedAttentionManager
```

Canonical state:

```text
req_id
→ E[C]
→ page_counts[C]
→ page_rows[C]
```

R1 values are uniform but APIs are vector-based.

Reuse upstream BlockPool.

Ragged BlockPool ID semantics:

```text
block_id == physical page_id
```

Canonical planning:

```text
target_page_counts[c]
=
ceil(target_E[c] / B)

need[c]
=
max(target_page_counts[c] - current_counts[c], 0)
```

Allocation protocol:

```text
plan
→ reserve sum(need) pages
→ scatter by explicit need[C]
→ candidate state
→ atomic commit
```

R1 may have an identity convenience adapter:

```text
target_E[c] = E[c] + q
```

but scalar logical tokens are not the permanent physical API.

---

# 7. R1-B2 — Scheduler↔Worker Transport / Persistent Mirror

Worker persistent state:

```text
page_rows    [Rmax,C,M] int32
page_counts  [Rmax,C]   int32
effective_E  [Rmax,C]   int32
```

Use existing request persistent indexing and `idx_mapping`.

Transport distinguishes:

```text
Full Snapshot
Incremental Allocation Delta
```

Full Snapshot concept:

```text
E[C]
counts[C]
flat_page_ids[]
```

Allocation Delta concept:

```text
expected_source_counts[C]
appended_counts[C]
flat_new_page_ids[]
```

All snapshot/delta apply operations are:

```text
validate-all
→ build candidate
→ commit-all
```

Worker does not free allocator pages.

---

# 8. R1-C — Placement / Address Oracle

Given semantic member:

```text
(layer, head)
```

obtain:

```text
(cluster, column)
=
MemberPlacementMap[layer, head]
```

For physical position `p`:

```text
depth        = p // B
block_offset = p % B
page_id      = row[req, cluster, depth]
```

Canonical address:

```text
(page_id, column, block_offset)
```

C MUST remain backend-neutral.

---

# 9. R1-D1 — FA2-Compatible KV Write Adapter

Model K/V:

```text
[Q,Hkv,D]
```

R1 may materialize:

```text
[Q*Hkv,1,D]
```

Current adapter derives:

```text
vbid = page_id*Hp + column
slot = vbid*B + block_offset
```

and reuses existing cache-write primitive where valid.

The `Q*Hkv` slot mapping is not a permanent Ragged ABI.

---

# 10. R1-D2 — Decode Read Adapter

R1 may transform:

```text
Q [R,Hq,D]
→ [R,Hkv,q_per_kv,D]
→ [R*Hkv,q_per_kv,D]
```

with derived member block tables and member-visible lengths.

Canonical ownership remains cluster-major.

---

# 11. R1-D3 — Prefill / Mixed Read Adapter

R1 accepts correctness-first materialization:

```text
token-major
→ member-major
→ FA2
→ inverse scatter
```

Do not over-optimize before correctness/parity.

---

# 12. R1-E — Runtime Activation + Real-Engine Identity Gate

R1-E is the first production activation point.

Only after A2/A3/B1/B2/C/D are ready may:

```text
CacheConfig.page_group_size
```

cause:

```text
Attention.get_kv_cache_spec()
→ RaggedAttentionSpec
```

Activation chain:

```text
requested config
↓
capability validation
↓
RaggedAttentionSpec
↓
Registry
↓
RaggedAttentionManager
↓
Ragged planner
↓
BlockPool physical namespace
↓
global backing
↓
Worker cluster mirror
↓
Ragged layout
↓
execution adapter
```

Registry contract:

```text
RaggedAttentionSpec
→ manager_class = RaggedAttentionManager
→ uniform_type_base_spec = RaggedAttentionSpec
```

R1 activation gates:

```text
decoder FullAttention only
all supported decoder layers Ragged
KVQuantMode.NONE
Dk == Dv
Hkv % Hp == 0
FA2 / validated Ragged layout
TP1 / PP1 / DCP1 / PCP1
prefix cache OFF
spec decode OFF
async scheduling OFF
connector/offload OFF
MLA/SWA/hybrid OFF
num_gpu_blocks_override unsupported
```

Unsupported combinations fail fast.

No silent Dense fallback.

Identity real-engine acceptance:

```text
prefill parity
decode parity
multiple requests
cross-block growth
Hp variants
request finish/free
no hidden Dense fallback
```

---

# 13. R1-F — Piecewise CUDA Graph Smoke

Compatibility smoke only.

No latency claim without profiler/benchmark evidence.

---

# 14. Updated R1 DAG

```text
                          R1-A1
                  Spec / Config Definition
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
          R1-A2                      R1-B1
 Physical Namespace /            Scheduler Ownership /
 Planner / Global Backing        Ragged Manager
             │                         │
             ▼                         ▼
          R1-A3                      R1-B2
 Physical Layout /               Transport / Worker
 Zero-Copy View                  Persistent Mirror
             │                         │
             └────────────┬────────────┘
                          ▼
                        R1-C
              Placement / Address Oracle
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
           R1-D1        R1-D2        R1-D3
             └────────────┼────────────┘
                          ▼
                        R1-E
                          │
                          ▼
                        R1-F
```

A2 and B1 can proceed independently.

---

# 15. R2 Reframing

R2 is:

```text
Non-Uniform Frontier Enablement
```

The vector architecture already exists in R1.

R2 may change values:

```text
R1: E=[32,32,32]
R2: E=[32,57,21]
```

but may not require ownership API redesign.

---

# 16. R3 Reclamation Contract

R3 uses the same canonical request state.

Worker reports completed physical shape.

Scheduler:

```text
validate source vectors
→ derive detached page IDs from canonical rows
→ commit new E/counts/rows
→ free detached pages
→ later reuse
```

Never:

```text
Worker free first
→ Scheduler learns later
```

---

# 17. Core Success Criterion

R1 architecture is accepted only if R2/R3 can be implemented without changing:

```text
physical page identity
canonical request ownership shape
snapshot/delta transport model
cluster-major worker state
MemberPlacementMap contract
backend-neutral physical address contract
```

Later rounds may replace:

```text
allocator implementation
storage container
transport encoding
FA2 adapter
metadata materialization
```

without replacing the architecture.
