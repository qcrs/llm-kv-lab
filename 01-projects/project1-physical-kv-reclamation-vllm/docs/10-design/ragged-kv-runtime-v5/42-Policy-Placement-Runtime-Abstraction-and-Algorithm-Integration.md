# Policy–Placement–Runtime Architecture for Non-Uniform Ragged KV

**Project:** `qcrs/llm-kv-lab`  
**Program:** vLLM 0.26 Non-Uniform Ragged KV Runtime  
**Document role:** Canonical architecture addendum / replacement guidance  
**Status:** PROPOSED CANONICAL DESIGN UPDATE  
**Scope:** R1–R7 architecture, policy integration, member placement, algorithm roadmap  
**Non-goal:** This document does not expand R1 implementation scope or authorize source modification by itself.

---

# 0. Executive Decision

The Ragged KV project is not defined as “an implementation of one KV eviction algorithm”.

The canonical system objective is:

> **Build a non-uniform physical KV runtime that decouples semantic KV retention decisions from physical KV placement and reclamation, so multiple head-/layer-aware KV eviction or compression algorithms can reuse the same allocator, compaction, FA2 read/write, ownership reconciliation, and page-reuse path.**

The architecture is frozen into three primary layers:

```text
Retention Policy / Algorithm
        │
        │ semantic decision:
        │ importance / budget / keep positions
        ▼
Unified RetentionPlan
        │
        ▼
Placement / Clustering
        │
        │ semantic (layer, head)
        │        ↓
        │ physical (cluster, column)
        ▼
MemberPlacementMap
        │
        ▼
Ragged Runtime
        │
        ├─ physical page allocation
        ├─ member-aware KV write
        ├─ member-major FA2 read
        ├─ compaction
        ├─ per-group effective length
        ├─ scheduler ownership reconciliation
        └─ BlockPool free / reuse
```

The runtime must not know *why* a KV member was retained.

The policy must not know *how* a physical page is allocated or freed.

The placement layer is the explicit bridge between them.

This separation becomes a first-class architecture invariant.

---

# 1. Why This Update Is Necessary

The existing v5 design already contains the required pieces:

- R1 uses identity head grouping;
- R2 creates per-group physical state;
- R3 proves physical reclamation;
- R4 connects one real compression policy;
- R7 introduces AOT per-layer clustering.

However, the previous document structure can be read as:

```text
R1 = contiguous grouping runtime
...
R7 = optional grouping optimization
```

That framing is too weak.

The stronger and more accurate interpretation is:

```text
General Member Placement Architecture
        │
        ├─ R1: Identity placement implementation
        │
        └─ R7: AOT clustered placement implementation
```

Therefore:

> **Identity grouping is not the architecture. Identity grouping is the first implementation of the architecture.**

The runtime must not make the long-term assumption that:

```text
physical group == contiguous semantic head IDs
```

R1 may use that mapping, but the mapping must be isolated behind one explicit contract.

---

# 2. Canonical Terminology

## 2.1 Semantic member

A semantic KV member is:

```text
member = (local_layer_id, semantic_kv_head_id)
```

It belongs to the mathematical model.

For a layer with:

```text
Hkv = 8
```

the semantic members remain:

```text
h0 h1 h2 h3 h4 h5 h6 h7
```

No physical layout may change their mathematical identity.

---

## 2.2 Physical page width

```text
Hp = page_group_size
```

`Hp` is the number of semantic KV members stored in one physical page.

It is a **physical resource-management granularity**, not a model-semantic parameter.

For:

```text
Hkv = 8
Hp  = 2
```

there are:

```text
G = Hkv / Hp = 4
```

physical groups per layer in the R1 identity case.

---

## 2.3 Physical cluster

A physical cluster is a set of exactly `Hp` semantic members sharing one physical page frontier.

For a per-layer placement:

```text
cluster c
  ├─ column 0 → semantic member A
  └─ column 1 → semantic member B
```

The cluster is the allocation/reclamation unit along the token-depth axis.

Members in the same cluster do **not** need to be mathematically equally important.

They only share the same physical page-depth envelope.

---

## 2.4 Physical frontier

For one cluster:

```text
E[request, cluster]
```

is the effective compacted KV length that determines page depth:

```text
required_pages(cluster)
=
ceil(E[cluster] / block_size)
```

If members inside a cluster have different retained lengths, the physical envelope is generally:

```text
E_cluster = max(E_member_i)
```

unless the policy/runtime contract explicitly normalizes them to one shared length.

This is the origin of **max-pool over-allocation**.

---

# 3. New Architecture Invariant: Semantic–Physical Placement Separation

The runtime SHALL represent the mapping:

```text
semantic member
(layer, head)

→

physical placement
(cluster, column)
```

through an explicit placement abstraction.

Canonical conceptual interface:

```text
MemberPlacementMap

inputs:
    local_layer_id
    semantic_kv_head_id

outputs:
    cluster_id
    column_in_cluster
```

Canonical tensor form:

```text
member_to_cluster: [num_local_layers * Hkv]
member_to_col:     [num_local_layers * Hkv]
```

or equivalent structured representation.

Required invariants:

```text
1. every semantic member maps to exactly one physical (cluster, column)

2. every physical (cluster, column) is occupied by exactly one semantic member

3. every cluster contains exactly Hp members

4. column is in [0, Hp)

5. the map is immutable during one serving lifetime unless an explicit
   migration protocol is introduced in a future program

6. Attention semantic head identity is preserved regardless of physical placement
```

---

# 4. R1 Is the Identity Special Case

R1 remains intentionally narrow.

R1 SHALL NOT:

- profile head importance;
- cluster heads by score;
- load arbitrary cluster maps;
- perform online regrouping;
- implement cross-layer placement;
- delete KV;
- change retention budgets.

R1 implements only:

```text
IdentityMemberPlacementMap
```

For local layer `L`, semantic head `h`:

```text
groups_per_layer = Hkv / Hp

cluster =
    L * groups_per_layer
    + h // Hp

column =
    h % Hp
```

Example:

```text
Hkv = 8
Hp  = 2

h0 → g0 / col0
h1 → g0 / col1

h2 → g1 / col0
h3 → g1 / col1

h4 → g2 / col0
h5 → g2 / col1

h6 → g3 / col0
h7 → g3 / col1
```

This mapping is only one valid `MemberPlacementMap`.

### R1 architecture rule

All R1 address generation should consume the placement contract through one centralized helper or precomputed map.

Do not scatter long-term assumptions such as:

```python
group = head // Hp
column = head % Hp
```

across:

- KV write;
- FA2 read;
- block-table builders;
- compaction;
- slot mapping;
- scheduler reconciliation.

The arithmetic may be used internally by the Identity map implementation, but must not become the runtime's permanent semantic contract.

---

# 5. Placement Must Be Transparent to Attention Semantics

Suppose AOT placement later produces:

```text
g0 = {h0, h1}
g1 = {h3, h7}
g2 = {h2, h5}
g3 = {h4, h6}
```

Then:

```text
semantic order:
h0 h1 h2 h3 h4 h5 h6 h7
```

has not changed.

Only physical storage placement changes.

Example map:

```text
h0 → g0 / col0
h1 → g0 / col1
h2 → g2 / col0
h3 → g1 / col0
h4 → g3 / col0
h5 → g2 / col1
h6 → g3 / col1
h7 → g1 / col1
```

FA2/member-major execution must still construct the correct semantic query-to-KV relationship.

Therefore:

> **Physical clustering is a storage and allocation transformation, not an Attention-head permutation in model semantics.**

---

# 6. Unified RetentionPlan Contract

To support multiple eviction algorithms without rewriting the physical runtime, policy output must be normalized into one runtime-facing representation.

The canonical conceptual object is:

```text
RetentionPlan
```

Recommended semantics:

```text
RetentionPlan
├─ member_keep_indices
│    semantic retained source positions per (layer, head)
│
├─ member_kept_lengths
│    number of retained KV entries per semantic member
│
├─ protected_prefix
│    optional sink / prompt-protection region
│
├─ protected_tail
│    optional recent-window protection
│
├─ budget_metadata
│    optional algorithm/debug information
│
└─ policy_generation
     source-fence / lifecycle identifier if needed
```

The exact Python schema must be frozen only when R4 implementation starts.

The architecture contract is more important than field names.

---

# 7. Retention Content and Physical Frontier Are Different

This distinction is mandatory.

Two heads may retain the same number of KV entries while retaining different positions:

```text
h0 keep = {0, 2, 8, 15, ...}
h1 keep = {1, 4, 9, 13, ...}

len(h0) = len(h1) = 32
```

They can still share:

```text
physical depth = 32
```

after member-wise compaction.

Therefore the system must conceptually separate:

```text
Retention Content
= which semantic KV entries survive

Physical Frontier
= how many physical page-depth slots the cluster needs
```

For a cluster `c`:

```text
frontier[c]
=
max(member_kept_length[m] for m in members(c))
```

Then:

```text
required_pages[c]
=
ceil(frontier[c] / block_size)
```

This enables per-head keep sets while retaining group-granular physical allocation.

---

# 8. Why Grouping Exists

`Hp` controls a systems trade-off.

## Hp = Hkv

```text
all heads share one physical frontier
```

Advantages:

- minimum metadata;
- minimum allocator objects;
- dense-like execution.

Disadvantage:

- almost no head-wise physical reclamation freedom.

## Hp = 1

```text
one head = one physical cluster
```

Advantages:

- per-head ideal physical frontier;
- no grouping fragmentation.

Disadvantages:

- maximum page count;
- maximum block-table rows;
- maximum allocator/bookkeeping pressure;
- larger member-major metadata;
- potentially higher runtime overhead.

## 1 < Hp < Hkv

This is the intended compromise:

```text
reclamation granularity
vs
runtime/control-plane overhead
```

The optimal `Hp` is an empirical systems parameter and must not be claimed without benchmark evidence.

---

# 9. Why AOT Clustering Matters

Adjacent semantic heads are not guaranteed to have similar retention behavior.

Example:

```text
contiguous group:
{h2, h3}

h2 ideal retained length = 1024
h3 ideal retained length = 256
```

With shared physical pages:

```text
cluster frontier ≈ 1024
```

The smaller member is physically over-provisioned.

This is not a scorer error.

It is a **placement fragmentation** problem.

AOT clustering changes:

```text
adjacent grouping
```

into:

```text
similar-retention grouping
```

while keeping fixed physical page width `Hp`.

---

# 10. R7 Is Upgraded from “Optimization” to “Placement System”

The old interpretation:

```text
R7 = AOT per-layer clustering optimization
```

is replaced with:

> **R7 = AOT Placement Compiler + Arbitrary Per-Layer Member Placement**

R7 has two responsibilities.

## R7-A — Retention Profiling

Run a pilot configuration with maximal observation granularity:

```text
page_group_size = 1
```

Collect:

```text
retention_profile[layer, head]
```

The profile may contain:

- mean retained ratio;
- median retained ratio;
- failure-conditioned retention;
- rank stability;
- optional workload-class metadata.

The runtime profile source must match the policy being clustered.

A map built for one scorer is not automatically assumed optimal for another scorer.

---

## R7-B — Placement Compiler

Input:

```text
retention_profile [L, Hkv]
Hp
```

Output:

```text
cluster_of [L, Hkv]
column_of  [L, Hkv]
```

First supported scope:

```text
per-layer only
```

Within each layer:

```text
sort heads by retention behavior
→ partition into groups of Hp
→ assign column
```

Required invariants:

```text
bijection
every cluster exactly Hp members
every local semantic head covered
map immutable during serving
TP-local ownership respected
```

---

# 11. R7-X Cross-Layer Placement Is a Separate Stretch Program

Cross-layer clustering is useful, but it is not required to support most head-/layer-aware eviction algorithms.

Example:

```text
cluster c =
(L2, h1)
(L7, h5)
(L19, h3)
(L25, h0)
```

Potential benefit:

```text
global retention-similarity
→ lower max-pool fragmentation
```

But it reopens major systems assumptions:

- layer ownership;
- physical row meaning;
- execution order;
- per-layer compression boundary;
- scheduler state representation;
- TP shard placement;
- map/profile comparability;
- score normalization across layers.

It also introduces a scientific requirement:

```text
retention score in Layer A
must be meaningfully comparable
with retention score in Layer B
```

That is not automatically true for per-layer-normalized algorithms.

Therefore:

```text
Arbitrary per-layer placement
= canonical architecture capability

Cross-layer placement
= R7-X stretch / research extension
```

Cross-layer support must not block Core or Flagship closure.

---

# 12. No Online Dynamic Regrouping in the Core Program

The project SHALL NOT initially support:

```text
step t:
g0 = {h0, h1}

step t+N:
g0 = {h0, h5}
```

Online regrouping requires:

- live KV migration;
- page ownership transfer;
- block-table rewrite;
- member-map versioning;
- synchronization with in-flight Attention;
- scheduler/worker transactional agreement;
- CUDA Graph address/lifetime handling.

This is a separate research problem.

The supported model is:

```text
offline / startup:
build placement map

online serving:
placement map frozen
retention state changes
```

---

# 13. Revised Round Structure

The existing R0–R10 plan remains, with the following semantic update.

## R1 — General Placement Contract + Identity Implementation

R1 remains identity-only.

New acceptance invariant:

```text
all member-aware runtime consumers
depend on one placement contract

identity arithmetic is centralized
and not duplicated as a permanent assumption
```

R1 still proves:

```text
no compression
same semantic Attention
different physical representation
```

---

## R2 — Per-Group Physical State

No algorithm yet.

Establish:

```text
E[request, cluster]
counts[request, cluster]
canonical rows
```

Different clusters can have unequal frontier.

---

## R3 — Manual Reclamation Transaction

Manual:

```text
member keep sets / target lengths
```

drive:

```text
compaction
→ physical frontier
→ trim
→ scheduler reconcile
→ free
→ cross-request reuse
```

R3 proves physical capability independent of scorer quality.

---

## R4 — Policy Adapter Program

R4 should no longer be framed as “hard-wire one scorer into the runtime”.

It should introduce:

```text
Policy Adapter
        ↓
RetentionPlan
        ↓
existing R3 executor
```

### R4-A

One deterministic policy, simple boundary.

Goal:

```text
prove policy → RetentionPlan → physical runtime contract
```

### R4-B

Enable real non-uniform layer/head/member output.

Goal:

```text
automatic policy
→ unequal member/group retention
→ unequal physical page depth
→ real reclaim/reuse
```

### R4-C optional

Second policy using the same adapter interface.

Goal:

```text
prove algorithm replaceability
without allocator/write/read redesign
```

This is strong evidence that the architecture is real rather than a one-policy code path.

---

## R7 — AOT Placement Compiler

R7 now owns:

```text
retention profiling
+ per-layer clustering
+ arbitrary placement map loading
+ identity vs AOT comparison
```

R7 does NOT own:

```text
new Attention math
new allocator
new compaction transaction
```

Those must reuse R1–R4 infrastructure.

---

# 14. Algorithm Integration Taxonomy

Not every KV method fits this runtime equally well.

The natural target class is:

> **Any method whose decision can be normalized into per-semantic-member retained positions and/or retained lengths, after which compaction can convert the decision into a smaller dense physical frontier.**

Algorithms are divided into four integration classes.

---

## Class A — Native Fit

Characteristics:

```text
outputs keep positions and/or per-head budget
training-free or light-weight
can compact survivors into prefix
does not require sparse-hole Attention kernel
```

These are the best project targets.

Examples:

- SnapKV
- Ada-KV used with a base selector
- KeyDiff-style deterministic scorer
- Random Attention
- HeadKV-style head-level budget allocation

---

## Class B — Strong Fit but Adds Layer/Head Regimes

Characteristics:

```text
different layers or head classes use different retention rules
```

Examples:

- PyramidKV
- RazorAttention
- DuoAttention

These strongly validate the semantic-to-physical separation, but require more control-plane structure than Class A.

---

## Class C — Fit After Repeated-Eviction State Is Mature

Characteristics:

```text
eviction decisions evolve during decode
score state must be accumulated
boundary repeats many times
```

Examples:

- H2O
- TOVA

These should not be first R4 policies because they mix policy correctness with lifecycle and repeated-reclaim complexity.

---

## Class D — Not a Direct Fit

Characteristics:

```text
requires sparse-hole addressing
or changes Attention computation itself
or introduces precision/offload state beyond retention
```

Examples:

- arbitrary sparse-attention masks without compaction;
- mixed-precision KV quantization;
- CPU/NVMe KV offload;
- online dynamic head regrouping.

These need additional runtime dimensions and should not be advertised as zero-change policy adapters.

---

# 15. Recommended Algorithm Roadmap

## Priority S — SnapKV

Why it fits:

```text
per-head important KV positions
observation-window-derived selector
prompt/prefill compression boundary
no training required
open-source reference
```

Most importantly, SnapKV naturally exercises:

```text
member-specific keep indices
```

which is more valuable than a policy that only changes one scalar length.

Recommended role:

```text
R4-B first serious per-head keep-plan policy
```

Do not port the whole Hugging Face implementation.

Extract only:

```text
score / vote semantics
→ RetentionPlan
```

and reuse the Ragged executor.

---

## Priority S — Ada-KV + SnapKV

Ada-KV is particularly aligned with this project because its key idea is:

```text
adaptive budget allocation across attention heads
```

rather than equal head budgets.

That maps almost exactly to the runtime capability being built:

```text
semantic head-wise budget
        ↓
member retained length
        ↓
cluster physical frontier
        ↓
AOT placement / max-pool fragmentation
```

This makes Ada-KV a very strong Flagship integration.

Recommended role:

```text
R4-C or post-R4 flagship policy
```

Preferred experiment:

```text
SnapKV uniform-head budget
vs
SnapKV + Ada-KV head-wise budget

under:
1. Hp=1 per-head ideal
2. Hp>1 adjacent grouping
3. Hp>1 AOT grouping
```

This directly demonstrates why R7 exists.

---

## Priority A — PyramidKV

PyramidKV dynamically allocates different cache sizes across layers.

It is a strong match for:

```text
layer-aware non-uniform physical depth
```

Recommended role:

```text
validate LayerScope / layer-wise allocation
```

It is less direct than Ada-KV for testing head grouping, but excellent for proving:

```text
one scalar request length is insufficient
```

and that:

```text
E[layer, group]
```

is a real systems requirement.

---

## Priority A — Random Attention

Random Attention is valuable for a different reason.

It is signal-free:

```text
no attention score
no value statistic
no learned selector
```

and can generate per-KV-head random keep sets with recent protection.

Therefore it is an unusually clean test of:

```text
policy independence
```

If Random Attention and SnapKV both reuse exactly the same:

```text
RetentionPlan
→ compaction
→ reclamation
```

path, the architecture claim becomes much stronger.

Recommended role:

```text
algorithm-replaceability / reasoning workload extension
```

It is not the first R4 policy because the reference method focuses on generated-token eviction and repeated decode boundaries, which should come after basic prefill-boundary compression is stable.

---

## Priority A- / B+ — DuoAttention

DuoAttention divides heads into:

```text
Retrieval Heads:
full KV history

Streaming Heads:
sink + recent window
```

This is conceptually an excellent match for Ragged physical depth:

```text
retrieval head cluster → long frontier
streaming head cluster → bounded frontier
```

It strongly validates head-type-dependent physical allocation.

However, it is closer to an attention-regime specialization than a generic top-k eviction selector.

Recommended role:

```text
advanced static head-regime integration
```

not first R4.

---

## Priority B+ — RazorAttention

RazorAttention also separates retrieval heads from local heads and preserves long history selectively.

Why useful:

```text
tests extreme head-wise asymmetry
```

Why later:

```text
compensation-token semantics
and method-specific logic
add policy complexity
```

Use it only after the generic head-regime path is established.

---

## Priority B+ — HeadKV

HeadKV is highly aligned with:

```text
head-level importance
global KV budget allocation
```

It is attractive after Ada-KV because it provides another head-aware budget policy with a different importance signal.

Recommended role:

```text
optional second head-budget algorithm
```

not required for Core.

---

## Priority B — KeyDiff

KeyDiff remains a practical early deterministic scorer.

Advantages:

- Tangram already has a serving-oriented implementation;
- deterministic behavior is good for debugging;
- no extra learned checkpoint is required;
- useful for validating score → keep-plan plumbing.

Recommended role:

```text
R4-A control-plane bring-up
```

If implementation time is limited:

```text
KeyDiff first
→ SnapKV / Ada-KV for flagship
```

---

## Priority C — H2O

H2O dynamically preserves:

```text
heavy hitters + recent tokens
```

It is valuable for testing repeated online eviction.

But this requires:

- accumulated attention-score state;
- repeated compression boundaries;
- long-lived policy state;
- repeated free/regrow behavior;
- stronger lifecycle validation.

Recommended only after the core repeated-boundary runtime is stable.

---

## Priority C — TOVA

TOVA-style token omission based on current attention is also useful for online eviction testing.

It should be treated similarly to H2O:

```text
late lifecycle stress test
```

rather than an early algorithm.

---

## Defer — FastKVZip

Tangram supports FastKVZip, but the current project should not use it as the first integration because gate/checkpoint or hidden-state capture paths can introduce unrelated runtime/compile side effects.

It may be useful later for Tangram comparison, but it should not define the architecture.

---

# 16. Recommended Minimal Algorithm Set

Do not turn the project into a scorer zoo.

The strongest minimal portfolio is:

```text
Policy 0:
Manual / deterministic keep plan
→ R3 correctness oracle

Policy 1:
KeyDiff or simple deterministic scorer
→ R4-A control-plane bring-up

Policy 2:
SnapKV
→ per-head keep-position integration

Policy 3:
SnapKV + Ada-KV
→ head-wise adaptive budget
→ flagship non-uniform physical reclamation

Optional Policy 4:
Random Attention
→ prove policy independence
→ reasoning workload / repeated decode extension
```

If schedule is tight, stop after:

```text
SnapKV + Ada-KV
```

once physical reclaim/reuse and AOT grouping evidence are complete.

---

# 17. The Most Important Experiment for R7

The R7 benchmark should not only compare throughput.

It should directly isolate placement quality.

For one fixed policy, preferably Ada-KV-enhanced SnapKV:

```text
A. Per-head ideal
   Hp = 1

B. Adjacent grouping
   Hp > 1
   identity map

C. AOT per-layer grouping
   Hp > 1
   clustered map
```

Measure:

```text
ideal_member_slots
allocated_member_slots
residual_waste
physical_pages
KV_bytes
max_concurrency
compression-boundary latency
decode TPOT
request throughput
quality
```

Define:

```text
placement_waste
=
allocated capacity - per-head ideal capacity
```

This allows a clean systems statement:

> AOT placement recovers part of the fragmentation introduced by head-group paging without returning to the metadata/allocator cost of `Hp=1`.

Do not claim improvement until measured.

---

# 18. Policy Compatibility Contract

A policy is considered directly compatible if it can provide:

```text
semantic member identity
+
retained positions or retained count
+
explicit protected regions if required
+
stable lifecycle semantics
```

The policy must NOT:

- directly free BlockPool pages;
- directly mutate Scheduler canonical ownership;
- directly assume physical cluster IDs;
- encode `head // Hp` as semantic truth;
- bypass the source-fenced compaction transaction.

The runtime must NOT:

- inspect scorer-specific statistics;
- know SnapKV/Ada-KV/H2O-specific formulas;
- choose model-quality budgets;
- reorder semantic heads implicitly.

---

# 19. Revised Project Claim

Avoid:

> Implemented a head-group KV eviction algorithm in vLLM.

Prefer:

> **Built a non-uniform Ragged KV runtime for vLLM 0.26 that decouples semantic KV retention policy from physical member placement and page ownership. The system maps per-layer/per-head retention plans onto head-group physical pages, performs member-aware compaction, scheduler-validated reclamation and cross-request page reuse, and supports pluggable retention policies plus AOT head clustering.**

If only identity placement is completed:

> Implemented the identity specialization of a generalized semantic-member-to-physical-cluster placement contract.

If R7 is completed:

> Added AOT per-layer placement compilation that groups heads with similar retention behavior to reduce head-group max-pool fragmentation.

---

# 20. Updated Closure Levels

## Core Runtime

```text
R1 + R2 + R3
```

Proves:

```text
general physical substrate
manual non-uniform state
real free/reuse
continuation correctness
```

No algorithm-quality claim required.

---

## Policy-Integrated Core

```text
R4-A + R4-B
```

Proves:

```text
real policy
→ unified RetentionPlan
→ automatic non-uniform physical reclamation
```

---

## Flagship System

Recommended:

```text
R1–R4
+
SnapKV / Ada-KV integration
+
R7 AOT per-layer placement
```

Proves:

```text
policy
placement
runtime
```

as independent layers.

---

## Stretch

```text
Random Attention repeated decode
DuoAttention-style head regimes
TP2
Cross-layer placement
Piecewise CUDA Graph hardening
Triton compaction
```

None of these should block the flagship closure.

---

# 21. Required Documentation Changes

The following existing documents should be updated semantically.

## `01-End-to-End-Architecture-and-Invariants.md`

Add first-class invariant:

```text
Policy–Placement–Runtime Separation Invariant
```

State that semantic `(layer, head)` identity is independent from physical `(cluster, column)` placement.

---

## `03-Round-by-Round-Implementation-Plan.md`

Change R1 wording from only:

```text
Identity Ragged Paging
```

to:

```text
General Member Placement Contract
+ Identity Ragged Paging Implementation
```

Change R7 from:

```text
AOT Per-Layer Clustering
```

to:

```text
AOT Placement Compiler
+ Arbitrary Per-Layer Member Placement
```

Add:

```text
R7-X Cross-Layer Placement = stretch
```

---

## `05-Core-R2-R4-NonUniform-Reclamation-and-Compression.md`

Replace scorer-centric R4 contract with:

```text
Policy Adapter
→ RetentionPlan
→ Keep/Compaction Executor
```

Explicitly separate:

```text
member keep indices
member kept lengths
cluster physical frontier
```

---

## `07-Optimization-AOT-Triton-CUDAGraph.md`

Promote AOT clustering from a local optimization into:

```text
placement-layer optimization
```

Retain its performance role:

```text
reduce max-pool fragmentation
```

but connect it to the general `MemberPlacementMap`.

---

## `22-R1-Exact-Engineering-Contract.md`

Keep R1 scope unchanged.

Replace the implicit permanent assumption:

```text
head -> h//Hp, h%Hp
```

with:

```text
R1 implementation:
IdentityMemberPlacementMap

general runtime contract:
semantic member -> cluster,column
```

R1 acceptance should include:

```text
identity map generation exact
all R1 member-aware consumers use centralized placement helper/map
no arbitrary cluster-map loading yet
```

---

# 22. Non-Goals

This architecture update does NOT authorize:

```text
R1 arbitrary clustering
R1 policy implementation
R1 cross-layer ownership
online dynamic regrouping
new sparse Attention kernel
quantized ragged pages
offload
mixed representation pages
```

Those remain separate slices/programs.

---

# 23. Decision Summary

Frozen decisions:

```text
1. R1 identity grouping is a special case, not the final placement model.

2. semantic KV member identity and physical cluster placement are separate.

3. MemberPlacementMap becomes a first-class architecture contract.

4. R1 only implements IdentityMemberPlacementMap.

5. R7 implements arbitrary per-layer AOT placement.

6. Cross-layer placement remains R7-X stretch.

7. policy output is normalized through RetentionPlan.

8. Retention content and physical frontier are distinct concepts.

9. allocator / FA2 / compaction / ownership logic must not depend on a
   particular eviction algorithm.

10. the flagship policy path should prioritize:
    SnapKV → Ada-KV-enhanced head-wise budgeting → AOT placement comparison.

11. Random Attention is a valuable later proof of policy independence and
    reasoning-workload compatibility.

12. H2O/TOVA are repeated-eviction lifecycle extensions, not first policies.
```

---

# 24. Recommended Next Engineering Action

Do not start R7 now.

The immediate R1 action is only:

```text
audit all planned R1-C / D1 / D2 / D3 member-addressing surfaces

→ define one IdentityMemberPlacementMap helper/representation

→ ensure all member-aware addressing consumes it

→ keep cluster-map loading / AOT clustering disabled
```

Then continue the frozen R1 sequence.

This prevents a future R7 rewrite without expanding current scope.

---

# References / Source Basis

Project source basis:

- `qcrs/llm-kv-lab`
  - `01-End-to-End-Architecture-and-Invariants.md`
  - `03-Round-by-Round-Implementation-Plan.md`
  - `05-Core-R2-R4-NonUniform-Reclamation-and-Compression.md`
  - `07-Optimization-AOT-Triton-CUDAGraph.md`
  - `22-R1-Exact-Engineering-Contract.md`

Tangram reference:

- `aiha-lab/tangram`
  - `vllm/v1/attention/backends/ragged_layout.py`
  - `tools/head_group_clustering/`
  - `vllm/v1/attention/compression/`
  - Tangram scorer/benchmark paths

Algorithm references:

- SnapKV: https://arxiv.org/abs/2404.14469
- Ada-KV: https://arxiv.org/abs/2407.11550
- Ada-KV implementation: https://github.com/FFY0/AdaKV
- PyramidKV: https://arxiv.org/abs/2406.02069
- H2O: https://github.com/FMInference/H2O
- HeadKV: https://github.com/FYYFU/HeadKV
- DuoAttention: https://github.com/mit-han-lab/duo-attention
- RazorAttention: https://arxiv.org/abs/2407.15891
- Random Attention: https://github.com/SalesforceAIResearch/Random-Attention

