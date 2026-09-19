# R1-A1 Reasoning / Source Trace / Architecture Analysis v2 — Macro-Audited

**Project:** Physical KV Cache Reclamation for vLLM 0.26  
**Purpose:** explain not only what R1-A1 should implement, but why the architecture is chosen from the perspective of R2–R9, Tangram's real implementation, and the target vLLM 0.26 MRv2 runtime.  
**Supersedes:** `R1-A1-Reasoning-Source-Trace-and-Architecture-Analysis.md`

---

# 0. Executive Judgment

The project's main direction is sound and should **not** be replaced by a direct Tangram port.

The strongest existing decisions are:

```text
1. RaggedAttentionSpec is a distinct storage type.
2. One global physical page pool is shared across local attention layers.
3. Scheduler/KV manager remains canonical physical ownership authority.
4. Worker owns execution view + payload transformation, not free authority.
5. Scheduler future state is per-group/per-cluster rather than a single scalar frontier.
6. MRv2-native persistent/staged row infrastructure is reused instead of porting old BlockTable wholesale.
7. Identity Ragged is proven before compression/policy.
```

However, macro re-audit found four architectural adjustments worth making now:

```text
A. Replace permanent `GT = L*G` semantics with a generic physical-cluster axis C.
B. Make Ragged physical layout/addressing a single-source abstraction and explicitly gate HND/NHD.
C. Split Spec definition from production runtime activation; avoid half-enabled A1.
D. Treat member-major FA2 as a correctness adapter, not the final performance architecture.
```

These changes do not invalidate the project's direction; they reduce future redesign risk.

---

# 1. What Tangram Actually Implements

Reference pin:

```text
aiha-lab/tangram
6fa551fc8f6edcc118a2a39b3554ee1520e91edd
```

The useful lesson is not “copy Tangram files.”

It is to identify which abstractions survived an end-to-end implementation.

---

## 1.1 Ragged is a new KV storage type

Tangram defines:

```text
RaggedAttentionSpec(AttentionSpec)
```

not:

```text
RaggedAttentionSpec(FullAttentionSpec)
```

Its page bytes use `page_group_size`, while `num_kv_heads` remains semantic Hkv.

This directly supports our decision:

```text
Attention math = Full Attention
KV storage type = Ragged
```

The storage type should not inherit Dense ownership semantics merely because the attention equation is unchanged.

---

## 1.2 Tangram uses one global raw backing

Tangram's planner detects Ragged and allocates one contiguous KV tensor shared by all participating attention layers.

The physical page ID therefore identifies one page in a global worker-local namespace rather than “block depth interpreted separately inside every layer tensor.”

This is critical for reclamation:

```text
free page from L7/G2
→ page returns to one global pool
→ another layer/group/request can reuse it
```

A per-layer pool would preserve avoidable fragmentation and make cross-layer capacity borrowing impossible.

Therefore our A2 global page namespace is a strong long-term choice.

---

## 1.3 Tangram adds a group axis to the block table

Its `RaggedBlockTable` stores roughly:

```text
[request, total_head_groups, block_slot]
```

and keeps per-group counts.

The reason is fundamental:

```text
compressed groups can have different depths
```

A Dense row:

```text
request -> [block0, block1, ...]
```

cannot represent that state.

Our v0.26 adaptation:

```text
physical [R*C, M]
logical  [R,C,M]
```

is therefore conceptually correct.

The difference is implementation style:

```text
Tangram: native 3D RaggedBlockTable
our target MRv2: flatten request×cluster into row-oriented persistent infrastructure
```

This is a good adaptation rather than a deviation.

---

## 1.4 Tangram's allocator is Ragged-aware, but its frontier is still conservative

Tangram modifies the generic single-type manager with a Ragged branch:

```text
req_to_block_ids: request -> flat int32 physical page IDs
```

Initial allocation uses:

```text
required_token_blocks * total_head_groups
```

After compression, the scheduler passes a scalar effective cached length (`compress_max_eff_seq_len`) so allocation uses post-compression occupancy rather than logical `num_computed_tokens`.

This solves an important bug:

```text
logical sequence length remains large
but resident physical KV was compressed
```

However a scalar max frontier cannot fully preserve permanently non-uniform group growth.

If:

```text
E = [33, 57, 21]
```

one scalar frontier loses which group actually crosses a page boundary next.

Therefore our v5 design:

```text
E[request, cluster]
counts[request, cluster]
rows[request, cluster]
```

owned by a Ragged-specific manager is a stronger architecture for the project's stated systems claim.

This is one place where **we should deliberately not copy Tangram literally**.

---

# 2. Dense vLLM Ownership and Why Ragged Breaks It

Dense FullAttention benefits from a synchronization invariant:

```text
all participating layers
all KV heads
share the same logical token-block depth for one request
```

So scheduler ownership can collapse dimensions into:

```text
request -> block depth row
```

A block ID is interpreted within each layer-local KV tensor.

Ragged future ownership breaks this:

```text
request
  -> physical cluster
      -> independent page depth
```

The important change is not the scorer and not compaction itself.

It is the **allocator unit and ownership namespace**.

Dense allocator unit:

```text
cross-layer logical token-block depth
```

Ragged allocator unit:

```text
one actual physical page
```

This is why the global page pool and per-cluster rows must be architectural, not temporary implementation details.

---

# 3. Head Group vs Physical Cluster

The previous documents frequently used:

```text
GT = L * G
flat_group = layer * G + group
```

This is correct for R1 identity placement but too strong as a permanent abstraction.

Tangram's placement layer provides the important clue: it can explicitly map every semantic member:

```text
member = (layer, kv_head)
```

to:

```text
(cluster_id, column)
```

and its cluster map may span layers.

Therefore the canonical architecture should distinguish:

```text
G
= per-layer geometric number of Hp-sized groups

C
= total physical clusters/page rows in the placement
```

R1 identity:

```text
C = L * G
cluster_id = layer*G + head//Hp
column = head%Hp
```

Future placement:

```text
MemberPlacementMap decides cluster_id/column
```

This keeps AOT grouping, arbitrary permutation, and possible cross-layer clustering from forcing a BlockTable redesign.

---

# 4. Why Cross-Layer Clustering Should Be Architecturally Possible but Not Implemented Early

Cross-layer clustering can improve page utilization if members with similar retained lengths are paired even when they live in different layers.

But it complicates:

```text
layer-local execution
TP shard mapping
compression scoring lifecycle
physical ownership debugging
metadata slicing
```

So the implementation roadmap should remain:

```text
R1 identity placement
→ R4 automatic per-layer non-uniform state
→ R7 AOT per-layer clustering
→ cross-layer only if evidence justifies it
```

The key macro change is not to implement cross-layer now.

It is to avoid data structures that make it impossible later.

Hence:

```text
canonical rows use cluster_id
placement owns layer/head -> cluster
```

rather than every subsystem permanently recomputing `layer*G+group`.

---

# 5. Spec vs Placement vs Retention

Three concepts must remain separate.

## 5.1 `RaggedAttentionSpec` — geometry

Owns:

```text
Hkv
Hp
block_size
Dk/Dv
dtype
page bytes
max identity memory
```

Does not own:

```text
which heads share a page
which tokens are retained
which page IDs are allocated
```

## 5.2 `MemberPlacementMap` — spatial placement

Owns:

```text
(layer, kv_head)
→ (cluster_id, column)
```

Identity is one map.

AOT clustering is another map.

## 5.3 Retention / Compaction state — token dimension

Owns:

```text
which historical KV entries survive
physical retained width E
keep positions/indices
```

This separation is important because optimizing head placement and choosing token importance are different research/system problems.

---

# 6. Physical Layout: Keep the Current Direction, but Add a Stronger Boundary

The target qcrs/vLLM FlashAttention backend defines logical Dense KV shape:

```text
(B, H, N, 2D)
```

and may choose different physical stride orders for HND and NHD.

Our Ragged virtual-block trick requires:

```text
physical page p + column c
→ contiguous single-head block
```

so that:

```text
vbid = p * Hp + c
```

is a zero-copy address transform.

For the target packed K/V format, the R1 physical shape:

```text
[Bphys, Hp, N, 2D]
```

is appropriate **only if its actual strides are column-major/HND-compatible**.

If actual physical layout is NHD-like:

```text
[Bphys, N, Hp, 2D]
```

`(page,column)` cannot be flattened into one contiguous virtual-block axis without a copy.

Therefore the architecture must not merely say “FA2 only.”

It must say one of:

```text
R1: require audited HND-compatible Ragged physical layout
```

and later preferably:

```text
Ragged owns an explicit layout hook independent of Dense global HND/NHD setting
```

---

# 7. Introduce One Ragged Layout Source of Truth

Tangram has a dedicated `ragged_layout.py` that centralizes:

```text
physical shape
virtual view
identity member map
cluster-map member map
cluster page gather
```

This is a good architectural pattern to borrow.

For our target v0.26, A2/A3 should introduce an equivalent narrow abstraction, e.g.:

```text
RaggedLayoutSpec / ragged_layout.py
```

owning:

```text
physical cache shape
stride oracle
as_virtual_block_view
virtual_block_id(page,column)
member_to_cluster/member_to_col identity map
address decode oracle
```

Do not duplicate:

```text
page*Hp+column
```

in scheduler, worker table code, FA builder, write path, and compressor independently.

This reduces silent layout divergence later.

---

# 8. Registry and Manager: Target v0.26 Needs a Cleaner Design Than Tangram

The target qcrs/vLLM includes a KVCacheSpec Registry that maps:

```text
Spec type
→ Manager type
→ uniform-type base
```

Therefore inheritance has runtime consequences.

Long-term registration should be explicit:

```text
RaggedAttentionSpec
→ RaggedAttentionManager
→ uniform_type_base_spec = RaggedAttentionSpec
```

Why not `FullAttentionManager`?

Because future canonical state differs:

```text
Dense:
request -> one depth row

Ragged:
request -> cluster -> independent depth/pages
```

Why not copy Tangram's `self.ragged` branches throughout `SingleTypeKVCacheManager`?

Because this project wants a stronger per-cluster physical-state contract, and the target Registry already gives us a clean dispatch boundary.

Reuse should happen below the semantic boundary:

```text
BlockPool / free queue / common primitives
```

not by pretending Dense and Ragged ownership are the same manager state.

---

# 9. A1 Activation Boundary — Revised Decision

The previous A1 draft proposed wiring:

```text
Attention.get_kv_cache_spec()
→ RaggedAttentionSpec
```

while intentionally stopping before Registry/Manager.

This is acceptable as a temporary source-learning experiment, but it is not the cleanest implementation slice because the feature becomes constructible but not consumable.

Revised slicing:

## A1 — Definition

```text
CacheConfig.page_group_size
RaggedAttentionSpec
geometry/accounting/merge
replace() preservation
```

No production activation yet.

## A2 entry — Activation + physical substrate

Wire coherently:

```text
Attention.get_kv_cache_spec()
→ RaggedAttentionSpec
→ explicit Registry Ragged type
→ RaggedAttentionManager
→ global physical page planner/backing
```

This is a larger seam, but every activated object now has a real consumer.

The implementation can still be split internally, but the branch should not pretend the user-facing feature is runnable before this seam closes.

---

# 10. Scheduler Physical State — Keep the v5 Design

The project's `38-Scheduler-Side-Ragged-Physical-State-and-Allocation-Contract.md` is one of the strongest parts of the current architecture.

Keep:

```text
RaggedSchedulerPhysicalState
owned by Ragged-specific manager
```

with canonical vectors per request:

```text
E[C]
counts[C]
rows[C]
```

R1 identity:

```text
all E equal
```

R2+:

```text
E can diverge
```

Allocation:

```text
target_count[c] = ceil((E[c] + q) / B)
need[c] = target_count[c] - counts[c]
```

This is better aligned with true long-term physical reclamation than a scalar max effective length.

---

# 11. Worker Metadata — Keep Canonical State Cluster-Major

The current v5 distinction is good:

```text
C / physical cluster axis
= allocator ownership

Hkv / member axis per layer
= attention semantics

Q*Hkv
= KV write member-token axis
```

Do not store member-expanded attention tables as the canonical allocator state.

Instead:

```text
cluster-major canonical rows
→ step-derived member block tables
→ FA adapter
```

This becomes crucial for future performance work.

---

# 12. Performance Lesson from Tangram

Tangram's own speedup benchmark script explicitly warns that its compression baseline uses Ragged paging and is not the same as vanilla vLLM; it records a case where vanilla `page_group_size=None` is substantially faster than uncompressed `page_group_size=4` at very long context.

The exact number is Tangram's own measurement, not ours, so it must not be reused as our result.

But the architectural lesson is important:

> finer physical ownership can introduce substantial runtime overhead even when it saves no memory.

Likely overhead sources include:

```text
more physical page IDs
larger block/metadata surfaces
member-major expansion
extra packing/scattering
per-step metadata construction
more CPU allocator/bookkeeping work
attention adapter overhead
```

Therefore our optimization roadmap must not assume:

```text
compaction kernel is the only important optimization target
```

---

# 13. Correctness Path vs Performance Path

Freeze two layers of implementation intent.

## Correctness path

R1/R2/R3:

```text
cluster-major physical rows
→ derive member-major block table / seq lens
→ existing FA2 primitive
```

This is narrow, auditable, and reference-backed.

## Performance path

Only after profiling:

```text
metadata fusion
persistent derived buffers
head-group-aware builder
custom/fused attention adapter
reduced member expansion
bulk allocator
Triton writeback
```

The project should allow the dominant profile result to choose the optimization.

Therefore revise the old priority assumption:

```text
AOT clustering
> Piecewise CG
> Triton writeback
```

into a profile-driven performance gate where **attention-path overhead is also a first-class candidate**.

Do not prematurely commit to a custom attention kernel, but do not architect it out either.

---

# 14. Global Page Pool and Utilization

Keep the global pool.

For Ragged:

```text
num_physical_pages
= floor(available_memory / P_ragged)
```

No extra `/ num_layers` because the page is now the true allocator unit and pages are shared across layer/cluster ownership.

This gives the future allocator maximum freedom:

```text
freed L7/Cx page
→ reused by any compatible cluster/request
```

This is exactly the kind of choice that improves future utilization rather than only making R1 convenient.

However, downstream accounting must stop assuming:

```text
num_blocks * block_size
```

is automatically “logical token capacity.”

In Ragged mode `num_blocks` means physical page count.

A2 should audit metrics/admission/accounting APIs that still interpret it as Dense token-block capacity.

---

# 15. Prefix Cache / Spec Decode / Connector Scope

The current R1 gates remain appropriate.

Disable or reject:

```text
prefix caching
spec decode
KV connector/offload
hybrid/SWA/MLA
DCP/PCP
TP>1 initially
```

These are not claims that Ragged fundamentally cannot support them.

They isolate the first proof.

In particular, prefix caching is difficult because hash/block-prefix reuse assumes a uniform request-level block-depth abstraction that Ragged intentionally breaks.

Do not mix that problem into R1.

---

# 16. Revised Round-Level Mental Model

```text
R1-A1
Geometry contract only

R1-A2/A3
Activate Ragged type
Global page pool
Dedicated layout contract
Zero-copy virtual page view

R1-B
Cluster-major ownership table on MRv2 persistent rows

R1-C/D
Derived member views
KV write/read correctness

R1-E
Engine identity parity

R2
Per-cluster physical frontier and non-uniform allocation

R3
Manual compact -> reconcile -> free -> reuse -> continue

R4
Real retention policy

R5/R6
Preemption / TP

R7
AOT placement optimization

R8/R9
Profile-driven runtime optimization
including attention-path overhead, metadata, allocator, writeback, CG
```

---

# 17. Decisions Kept / Changed

## Keep

```text
RaggedAttentionSpec(AttentionSpec)
global page namespace
single shared raw backing
scheduler canonical ownership
worker execution mirror
MRv2 flattened physical rows
virtual-block FA reuse
identity-before-compression
per-cluster scheduler E/counts
AOT per-layer first
profile-gated optimizations
```

## Change / clarify

```text
OLD:
GT=L*G is the permanent group axis

NEW:
C is the physical cluster axis;
R1 identity happens to have C=L*G
```

```text
OLD:
A1 production dispatch can stop at Registry boundary

NEW:
A1 is definition-only;
production activation is wired with a real Ragged consumer at A2 entry
```

```text
OLD:
FA2-only gate is enough for layout

NEW:
R1 must also guarantee HND/column-major-compatible physical stride,
or use a dedicated Ragged layout hook
```

```text
OLD:
late optimization emphasis mainly AOT/CG/Triton writeback

NEW:
profile must also evaluate member-major attention/metadata overhead
```

---

# 18. Source Trace Used for This Re-Audit

## Tangram

```text
aiha-lab/tangram
commit 6fa551fc8f6edcc118a2a39b3554ee1520e91edd

vllm/v1/kv_cache_interface.py
  RaggedAttentionSpec(AttentionSpec)

vllm/v1/core/kv_cache_utils.py
  single global Ragged backing / page pool semantics

vllm/v1/core/single_type_kv_cache_manager.py
  ragged allocator branch / req_to_block_ids

vllm/v1/core/kv_cache_manager.py
  effective_num_cached_tokens allocation path

vllm/v1/worker/ragged_block_table.py
  request × head-group × depth state
  non-uniform counts / compaction

vllm/v1/attention/backends/ragged_layout.py
  column-major physical layout
  virtual block view
  arbitrary Member -> cluster/column mapping

vllm/config/compression.py
  Ragged validation / prefix-cache incompatibility

benchmarks/tangram/speedup/run_speedup.sh
  explicit warning about uncompressed Ragged overhead vs vanilla baseline
```

## qcrs Project-1

```text
qcrs/llm-kv-lab

22-R1-Exact-Engineering-Contract.md
38-Scheduler-Side-Ragged-Physical-State-and-Allocation-Contract.md
39-Ragged-Step-Metadata-Shape-Lifetime-and-Attention-Contract.md
01-End-to-End-Architecture-and-Invariants.md
03-Round-by-Round-Implementation-Plan.md
07-Optimization-AOT-Triton-CUDAGraph.md
13-Ragged-Physical-Layout-and-MRv2-KV-Write-Seam.md
14-Runtime-State-Machine-Ownership-and-Transaction-Protocol.md
```

## Target runtime

```text
qcrs/vllm
branch p1/v2-token-compaction-v026

vllm/v1/worker/gpu/attn_utils.py
  KV raw allocation / reshape / replace(spec)

vllm/v1/attention/backends/flash_attn.py
  logical shape (B,H,N,2D)
  HND/NHD stride order

vllm/v1/kv_cache_spec_registry.py
  Spec -> Manager / uniform type dispatch
```

---

# 19. Final Architecture Principle

The project should optimize for **future allocator freedom** and **replaceable execution adapters**.

Canonical state should represent what must remain true across future implementations:

```text
semantic members
placement map
physical cluster ownership
physical frontier
page IDs
transaction authority
```

Derived execution views may change:

```text
member-major FA2 today
fused/head-group-aware attention tomorrow
```

That boundary is the most important macro design choice from this re-audit.

