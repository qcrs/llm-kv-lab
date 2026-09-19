# v6 Implementation Slices — File Ownership & Dependency DAG

**Status:** PROPOSED CANONICAL EXECUTION MAP  
**Design basis:** D1 Physical Namespace, D2 Scheduler Ownership, D3 Scheduler↔Worker Contract, D4 Runtime Activation

---

# 1. Dependency DAG

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
                          │
                          ▼
                         R2
             Non-Uniform Frontier Enablement
                          │
                          ▼
                         R3
             Reclaim / Free / Reuse / Continue
                          │
                          ▼
                         R4
                     One Policy
```

A2 and B1 may proceed independently.

B2 depends on B1 state/transport semantics but does not require A3.

C is the first convergence point.

---

# 2. Global Execution Rule

Before R1-E:

```text
NO production Ragged activation
```

Normal startup must not return `RaggedAttentionSpec` from `Attention.get_kv_cache_spec()`.

Earlier slices are tested via direct construction/injection.

---

# 3. R1-A1 — Spec / Config Definition

Primary:

```text
vllm/config/cache.py
vllm/v1/kv_cache_interface.py
focused Ragged Spec tests
```

Scope:

```text
CacheConfig.page_group_size
RaggedAttentionSpec
geometry/page bytes
merge
permanent Spec validation
```

Forbidden:

```text
Registry activation
Attention production dispatch
planner activation
manager
BlockPool
Worker
FA
```

---

# 4. R1-A2 — Physical Namespace / Planner / Global Backing

Primary design surface:

```text
vllm/v1/core/kv_cache_utils.py
vllm/v1/kv_cache_interface.py
vllm/v1/worker/gpu/attn_utils.py   # only if isolated raw-allocation validation requires it
```

Audit-only semantic consumers:

```text
kv_cache_coordinator.py
block_pool.py
capacity/concurrency helpers
startup capacity reporting
```

Scope:

```text
Ragged planner branch on RaggedAttentionSpec
Nphys = available/page_bytes
KVCacheConfig.num_blocks = Nphys physical slots
one KVCacheTensor shared by supported local Ragged layers
NULL page accounting
Ragged capacity/concurrency math
reject ambiguous num_gpu_blocks_override
```

Forbidden:

```text
Attention activation
Ragged manager production hookup
Worker Ragged tables
FA layout
ring allocator
BlockPool redesign
```

---

# 5. R1-A3 — Physical Layout / Zero-Copy View

Suggested new module:

```text
vllm/v1/attention/backends/ragged_layout.py
```

Minimal worker integration:

```text
vllm/v1/worker/gpu/attn_utils.py
```

Scope:

```text
[Bphys,Hp,B,2D]
zero-copy [Bphys*Hp,1,B,2D]
stride/address oracle
(page,column)↔vbid
IdentityMemberPlacementMap helper
```

Forbidden:

```text
Scheduler ownership
policy
compaction
production activation
hidden contiguous copy
```

---

# 6. R1-B1 — Scheduler Ragged Ownership Manager

Preferred new module:

```text
vllm/v1/core/ragged_kv_cache_manager.py
```

Tests:

```text
tests/v1/core/test_ragged_kv_cache_manager.py
```

Scope:

```text
RaggedRequestPhysicalState
RaggedCapacityPlan
RaggedPageAllocationDelta
C-vector state
BlockPool physical-page allocation/free
plan→reserve→commit
snapshot export
request free
synthetic compaction reconciliation
synthetic non-uniform tests
```

Forbidden:

```text
production Registry activation
Attention dispatch
Worker tables
FA2
policy
ring allocator
prefix cache
```

---

# 7. R1-B2 — Scheduler↔Worker Transport / Persistent Mirror

Primary:

```text
vllm/v1/core/sched/output.py
vllm/v1/worker/gpu/ragged_block_table.py    # suggested
vllm/v1/worker/gpu/model_runner.py          # minimal apply hook
```

Optional helper:

```text
vllm/v1/worker/gpu/ragged_kv_state.py
```

Scope:

```text
Full Snapshot carrier
Allocation Delta carrier
[Rmax,C,M] page rows
[Rmax,C] counts
[Rmax,C] E
atomic apply
request reset/re-add
idx_mapping gather
backend-neutral RaggedClusterStepView
```

Forbidden:

```text
FA2 member-major tables as canonical storage
Worker allocator free
C fake KVCacheGroups
production activation
compression policy
```

---

# 8. R1-C — Placement / Address Oracle

Primary:

```text
ragged_layout.py
MemberPlacementMap
RaggedClusterStepView helpers
```

Defines:

```text
(layer,head) → (cluster,column)
(cluster position) → (page_id,column,offset)
```

No Scheduler allocation logic.

No FA kernel modification.

---

# 9. R1-D1 — KV Write Adapter

Scope:

```text
derive member write targets
optional Q*Hkv member slot mapping
reuse existing reshape/cache primitive where valid
synthetic exact-address oracle
```

FA2 metadata is adapter-only.

---

# 10. R1-D2 — Decode Read Adapter

Scope:

```text
member block-table derivation
member effective lengths
decode Q adapter
Dense-vs-Ragged identity parity
```

---

# 11. R1-D3 — Prefill / Mixed Adapter

Scope:

```text
token-major → member-major pack
FA2
inverse scatter
variable request query lengths
```

Correctness first; materialization allowed.

---

# 12. R1-E — Production Runtime Activation

Primary activation surface:

```text
vllm/model_executor/layers/attention/attention.py
vllm/v1/kv_cache_spec_registry.py
vllm/v1/core/single_type_kv_cache_manager.py
Ragged manager module
planner
worker init/runtime dispatch
```

First slice allowed to enable:

```text
CacheConfig.page_group_size
→ RaggedAttentionSpec
```

Requires:

```text
capability validation
Registry mapping
Ragged planner
Ragged manager
physical BlockPool namespace
global backing
Worker persistent mirror
layout
FA2 adapter
```

No silent Dense fallback.

---

# 13. R1-F — Piecewise CUDA Graph Smoke

Compatibility only.

No performance claim without evidence.

---

# 14. R2 — Non-Uniform Frontier Enablement

R2 does not add vector architecture.

It enables divergence of existing vectors:

```text
R1: E=[x,x,x], counts=[n,n,n]
R2: E=[x,y,z], counts=[a,b,c]
```

Required proof:

```text
allocation after non-uniform state
must not re-expand shorter clusters to max depth
```

---

# 15. R3 — Reclamation Closure

Required:

```text
worker compaction
→ scheduler result
→ atomic reconciliation
→ ownership detach
→ BlockPool free
→ another request reuses physical page
→ original request continues decode
```

No second ownership subsystem.

---

# 16. File Hygiene Rule

If implementation scatters:

```python
if page_group_size is not None:
```

or generic:

```python
if ragged:
```

across many scheduler/model-runner/attention files, STOP and review.

Expected branch concentration:

```text
Spec selection
Planner
Manager type
Worker Ragged state construction
Ragged execution adapter
```

---

# 17. Immediate Exact Slices

After this docs set is committed:

```text
R1-A2
R1-B1
R1-B2
```

A2 and B1 can be implemented independently.

B2 follows B1 semantics.

None may activate production Ragged mode.
