# P1-V2-DESIGN-REVIEW-01 — Ragged Runtime Architecture Freeze

**Project:** Physical KV Cache Reclamation for vLLM  
**Target:** vLLM v0.26.0 / MRV2  
**Status:** ARCHITECTURE FREEZE CANDIDATE  
**Purpose:** Freeze the interfaces required before R1-A2 / R1-B1 / R1-B2 implementation.

---

# 0. Executive Verdict

The project SHALL NOT build a uniform-only R1 runtime and redesign it later.

The frozen direction is:

```text
R1 = general Ragged physical runtime architecture
     + identity MemberPlacementMap
     + uniform frontier values

R2 = enable non-uniform frontier values on the same interfaces

R3 = prove compaction → detach → free → reuse → continue
```

The architecture is frozen around four boundaries:

```text
D1 Physical Page Namespace
D2 Scheduler Canonical Ownership
D3 Scheduler↔Worker State / Execution Boundary
D4 Runtime Activation Boundary
```

---

# 1. Source-Derived Facts

The following facts are grounded in the reviewed vLLM/Tangram sources.

1. Dense vLLM planner/block management treats one scheduler block ID as a logical depth index reused across grouped layer-local physical KV tensors.
2. `KVCacheConfig.num_blocks` is consumed by both memory-planning/reporting code and Scheduler/BlockPool construction; changing its meaning is not a worker-only change.
3. Upstream `BlockPool` already provides fixed-size ID allocation/free/refcount machinery suitable for a correctness-first physical-page allocator.
4. `KVCacheSpecRegistry` maps Spec type to Manager type and grouping base, and walks MRO.
5. Dense Worker `BlockTables` uses the vLLM KV-cache-group axis; that axis is not equivalent to our physical-cluster axis.
6. Tangram demonstrates both a global Ragged backing and Ragged-aware allocation proportional to head-group count; this confirms planner-only changes are insufficient.
7. Tangram's `snapshot_row()/restore_row()` demonstrates that non-uniform rows cannot be reconstructed from a flat Dense-style block list without row-boundary metadata.

Our design adopts these facts but does not copy Tangram's ring allocator or Worker-authoritative snapshot mechanism.

---

# 2. Global Design Principle

Canonical state describes **facts**, not current implementation containers.

Freeze:

```text
physical page identity
cluster-major ownership
frontier/capacity separation
Scheduler authority
snapshot/delta transport semantics
MemberPlacementMap
backend-neutral physical address
production activation boundary
```

Do not freeze the architecture to:

```text
np.ndarray
list[list[KVCacheBlock]]
Tangram ring allocator
FA2 member-major metadata
identity head grouping
```

---

# 3. Canonical Terminology

```text
Hkv = semantic KV heads on the current rank
Hp  = semantic KV members stored per physical page
B   = cache block size
C   = physical cluster count
M   = maximum page depth per physical-cluster row
```

R1 identity only:

```text
G = Hkv / Hp
C = L_local * G

IdentityMemberPlacementMap(layer, head):
    cluster = layer * G + head // Hp
    column  = head % Hp
```

`C = L*G` is not a permanent ownership invariant.

---

# 4. D1 — Physical Namespace Freeze

## D1-01

Ragged `page_id` is unique within one worker-local physical pool.

## D1-02

Under Ragged:

```text
KVCacheConfig.num_blocks = Nphys
```

where `Nphys` is total physical page slots, not Dense logical token-block depth count.

## D1-03

When reusing upstream BlockPool:

```text
page 0 = reserved NULL page
allocatable page IDs = [1, Nphys)
```

## D1-04

R1 reuses upstream BlockPool allocation/free/refcount machinery.

No Tangram ring allocator is required for Core R1.

## D1-05

One Ragged `KVCacheTensor` backs all supported local Ragged Attention layers.

## D1-06

Canonical ownership semantics are:

```text
request → physical cluster → physical depth → physical page ID
```

## D1-07

Ragged uses a dedicated `RaggedAttentionManager`.

Do not permanently inject Ragged state into `FullAttentionManager`.

## D1-08

Planner dispatches on resolved `RaggedAttentionSpec`, not the raw config knob.

---

# 5. D2 — Scheduler Ownership Freeze

## D2-01

One request owns one coherent state object:

```text
RaggedRequestPhysicalState
├─ effective_lens[C]
├─ page_counts[C]
└─ ordered_page_rows[C]
```

## D2-02

`C` is the permanent physical ownership axis.

## D2-03

`MemberPlacementMap` is separate from Scheduler ownership.

Scheduler does not need to know which semantic `(layer,head)` belongs to a cluster.

## D2-04

Effective frontier and owned capacity are separate state:

```text
E[c] <= page_counts[c] * B
```

Do not permanently require equality with `ceil(E/B)` because future reservation/speculation may over-own capacity temporarily.

## D2-05

Canonical allocation targets a frontier vector:

```text
target_E[C]
```

not a scalar logical token count.

## D2-06

Allocation is:

```text
plan
→ reserve physical pages
→ deterministic per-cluster scatter
→ commit
```

## D2-07

Per-cluster appended counts are explicit.

Never infer equal split from `len(new_ids) / C`.

## D2-08

Compaction/reclamation uses the same canonical request state.

No second reclaim ownership subsystem.

## D2-09

Reconciliation is atomic across all clusters.

## D2-10

Page IDs are pool-local identities; distributed-global uniqueness is not assumed.

## D2-11

R1 prefix caching is disabled, but the ownership model does not prohibit future refcounted sharing.

---

# 6. D3 — Scheduler↔Worker Freeze

## D3-01

Scheduler is canonical ownership authority.

Worker is an execution mirror.

## D3-02

Transport distinguishes:

```text
Full Snapshot
Incremental Allocation Delta
```

## D3-03

New/resumed/re-added requests are reconstructable from Scheduler-owned full snapshots.

Do not rely on Worker-owned row snapshots as canonical recovery state.

## D3-04

Worker persistent physical state is cluster-major:

```text
page_rows    [Rmax,C,M]
page_counts  [Rmax,C]
effective_E  [Rmax,C]
```

## D3-05

Ragged transport gets an explicit cluster axis.

Do not overload Dense `block_ids` tuple semantics or create fake KVCacheGroups.

## D3-06

Backend-neutral step state remains cluster-major.

## D3-07

`MemberPlacementMap` is the only semantic-member → `(cluster,column)` authority.

## D3-08

Canonical physical address is:

```text
(page_id, column, block_offset)
```

FA2 `vbid = page_id*Hp + column` is adapter-only.

## D3-09

Snapshot/delta application is validate-all → candidate-state → atomic-commit.

## D3-10

Worker never frees allocator-owned pages directly.

## D3-11

Fresh-page zeroing may continue to transport flat physical page IDs, but the worker implementation must interpret them against Ragged global-backing semantics.

## D3-12

Persistent cluster rows are gathered once per step; layer/member execution views are derived afterward.

---

# 7. D4 — Runtime Activation Freeze

## D4-01

`CacheConfig.page_group_size` is requested intent, not runtime mode.

## D4-02

`RaggedAttentionSpec` is the resolved storage-type marker.

## D4-03

`Attention.get_kv_cache_spec()` is the primary production activation seam.

## D4-04

Permanent Spec validation and temporary R1 runtime gates are separate.

Permanent:

```text
Hp > 0
Hkv % Hp == 0
```

R1 capability gates include:

```text
Full decoder Attention only
KVQuantMode.NONE
Dk == Dv
FA2 / validated Ragged layout
TP1 / PP1 / DCP1 / PCP1
prefix cache OFF
spec decode OFF
async scheduling OFF
connector/offload OFF
MLA/SWA/hybrid OFF
```

## D4-05

Registry mapping is:

```text
RaggedAttentionSpec
→ RaggedAttentionManager
→ uniform_type_base_spec = RaggedAttentionSpec
```

Never fall back to `FullAttentionManager`.

## D4-06

Planner consumes resolved Spec type.

## D4-07

Every consumer of `KVCacheConfig.num_blocks` must be audited for Dense semantic assumptions.

## D4-08

R1 rejects ambiguous `num_gpu_blocks_override` under Ragged mode.

## D4-09

Ragged remains one vLLM KVCacheGroup containing internal `C` physical clusters.

## D4-10

Generic Coordinator/KVCacheManager should not own physical-cluster topology.

## D4-11

Activation is fail-fast.

No silent Dense fallback or incompatible config mutation.

## D4-12

Backend/layout capability is validated before production emits a Ragged Spec.

## D4-13

Once a Ragged Spec exists, downstream storage path stays Ragged.

## D4-14

R1 rejects Dense/Ragged mixed decoder layers and hybrid KV configurations.

## D4-15

Production Ragged activation occurs only at R1-E after A2/A3/B1/B2/C/D are complete.

---

# 8. Frozen Runtime Architecture

```text
Retention Policy / Algorithm
        │
        ▼
RetentionPlan
        │
        ▼
MemberPlacementMap
semantic member → (cluster,column)
        │
        ▼
RaggedAttentionManager
req → E / counts / rows
        │
        ▼
BlockPool
physical page IDs
        │
        ▼
Global Ragged Backing
        │
        ▼
Worker Persistent Cluster State
        │
        ▼
RaggedClusterStepView
        │
        ▼
Execution Adapter
FA2 today / replaceable later
```

---

# 9. New R1 Dependency DAG

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
           Write        Decode       Prefill/Mixed
             └────────────┼────────────┘
                          ▼
                        R1-E
            Runtime Activation + Identity Parity
                          │
                          ▼
                        R1-F
                 Piecewise CG Smoke
```

A2 and B1 may proceed independently.

B2 follows B1 semantics but does not require A3.

C is the first convergence point.

---

# 10. R2 / R3 Reframing

## R2

R2 is now:

```text
Non-Uniform Frontier Enablement
```

The vector architecture already exists in R1.

R1:

```text
E=[x,x,x]
counts=[n,n,n]
```

R2:

```text
E=[x,y,z]
counts=[a,b,c]
```

No ownership API redesign is allowed at R2.

## R3

R3 proves:

```text
Worker compaction
→ result
→ Scheduler atomic reconciliation
→ ownership detach
→ BlockPool free
→ another request reuses page
→ original request continues decode
```

No second state subsystem.

---

# 11. Stop Conditions

Stop implementation and return to design review if any slice requires:

```text
physical clusters represented as KVCacheGroups
num_kv_heads rewritten to Hp
Worker allocator free
uniform-only ownership API
equal-split reconstruction
Ragged branches scattered across generic runtime files
silent Dense fallback
hidden physical-layout copy
```

---

# 12. Freeze Verdict

`P1-V2-DESIGN-REVIEW-01` core architecture is sufficiently defined to cut exact implementation slices for:

```text
R1-A2
R1-B1
R1-B2
```

This document authorizes implementation-contract generation only. Production Ragged activation remains forbidden until R1-E.
