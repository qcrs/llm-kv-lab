# P1-V2 R1-A3 — Placement & Physical Address Architecture Freeze

**Project:** Physical KV Cache Reclamation for vLLM  
**Phase:** P1-V2 / R1-A3  
**Document type:** Architecture Freeze / Source Review / C1 Handoff  
**Date:** 2026-09-20  
**Status:** **APPROVED ARCHITECTURE WITH REQUIRED PRE-C1 CONTRACT HARDENING**

---

## 0. Source Baseline and Review Scope

### 0.1 P1 implementation baseline

```text
Repository:
qcrs/vllm

Branch:
p1/v2-token-compaction-v026

Reviewed HEAD:
47e2d319b2c37934d21613c06856ffcb4e28a15b

vLLM baseline:
568afb3a13806beb53bb2e6bd518269357b237c0

Branch relation:
7 commits ahead of baseline
0 commits behind baseline
```

Key P1 commits in the reviewed line include:

```text
cd444b72...  feat(p1): freeze V1 whole-block KV reclamation core
9ea5d804...  p1-v2: close M4 token-level KV compaction data plane
1f04cdca...  p1: prepare post-forward V2 compaction sources
bd5a0e95...  feat(p1): close V2 KV compaction runtime reconciliation
4b8e7177...  feat(p1-v2): add ragged KV control-plane state and transport
47e2d319...  feat(p1-v2): add ragged KV spec and planning support
```

Primary reviewed P1 sources:

- [`vllm/config/cache.py`](https://github.com/qcrs/vllm/blob/47e2d319b2c37934d21613c06856ffcb4e28a15b/vllm/config/cache.py)
- [`vllm/v1/kv_cache_interface.py`](https://github.com/qcrs/vllm/blob/47e2d319b2c37934d21613c06856ffcb4e28a15b/vllm/v1/kv_cache_interface.py)
- [`vllm/v1/core/kv_cache_utils.py`](https://github.com/qcrs/vllm/blob/47e2d319b2c37934d21613c06856ffcb4e28a15b/vllm/v1/core/kv_cache_utils.py)
- [`vllm/v1/core/ragged_kv_cache_manager.py`](https://github.com/qcrs/vllm/blob/47e2d319b2c37934d21613c06856ffcb4e28a15b/vllm/v1/core/ragged_kv_cache_manager.py)
- [`vllm/v1/core/sched/output.py`](https://github.com/qcrs/vllm/blob/47e2d319b2c37934d21613c06856ffcb4e28a15b/vllm/v1/core/sched/output.py)
- [`vllm/v1/worker/gpu/ragged_kv_state.py`](https://github.com/qcrs/vllm/blob/47e2d319b2c37934d21613c06856ffcb4e28a15b/vllm/v1/worker/gpu/ragged_kv_state.py)
- [`vllm/v1/worker/gpu/attn_utils.py`](https://github.com/qcrs/vllm/blob/47e2d319b2c37934d21613c06856ffcb4e28a15b/vllm/v1/worker/gpu/attn_utils.py)
- [`tests/v1/core/test_ragged_kv_cache_manager.py`](https://github.com/qcrs/vllm/blob/47e2d319b2c37934d21613c06856ffcb4e28a15b/tests/v1/core/test_ragged_kv_cache_manager.py)
- [`tests/v1/worker/test_ragged_kv_state.py`](https://github.com/qcrs/vllm/blob/47e2d319b2c37934d21613c06856ffcb4e28a15b/tests/v1/worker/test_ragged_kv_state.py)

### 0.2 Tangram reference baseline

```text
Repository:
aiha-lab/tangram

Pinned reference:
6fa551fc8f6edcc118a2a39b3554ee1520e91edd
```

Primary reviewed Tangram sources:

- [`vllm/v1/attention/backends/ragged_layout.py`](https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/attention/backends/ragged_layout.py)
- [`vllm/v1/worker/ragged_block_table.py`](https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/worker/ragged_block_table.py)
- [`vllm/v1/attention/backends/ragged_forward.py`](https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/attention/backends/ragged_forward.py)
- [`vllm/v1/attention/backends/flash_attn.py`](https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/attention/backends/flash_attn.py)
- [`vllm/v1/worker/compression_model_runner_mixin.py`](https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/worker/compression_model_runner_mixin.py)
- [`vllm/v1/core/single_type_kv_cache_manager.py`](https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/core/single_type_kv_cache_manager.py)
- [`vllm/v1/core/kv_cache_utils.py`](https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/core/kv_cache_utils.py)
- [`vllm/v1/kv_cache_interface.py`](https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/kv_cache_interface.py)
- [`vllm/v1/worker/gpu_model_runner.py`](https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/worker/gpu_model_runner.py)

### 0.3 Review rule

This document deliberately distinguishes:

```text
SOURCE FACT
= what the reviewed source actually does

CURRENT PROJECT DESIGN
= the architecture already established by P1 A1/A2/B12

PROPOSED FREEZE
= the contract approved by this A3 review
```

Tangram is used only as **execution/data-path feasibility evidence**. Its control-plane design is not adopted wholesale.

---

# 1. Executive Decision

## 1.1 Overall verdict

P1-V2 does **not** require an architectural restart.

The current direction is structurally sound:

```text
semantic KV members
        ↓
fixed placement
        ↓
cluster-owned physical page rows
        ↓
Scheduler canonical ownership
        ↓
Worker mirror
        ↓
execution-time address materialization
        ↓
Ragged physical backing
        ↓
FlashAttention-compatible virtual block view
```

The current code is, however, still split at the exact point A3 is intended to freeze:

```text
A1/A2/B12
= geometry + planning + control-plane state

missing
= placement + physical address algebra + production backing materialization
```

That missing layer is narrow enough to fix without redesigning A1/A2/B12.

## 1.2 Status matrix

| Area | Status | Decision |
|---|---|---|
| `RaggedAttentionSpec` geometry | **PASS_WITH_CHANGE** | Geometry is correct. Clarify physical-layout semantics and R1 supported execution subset; do not split the class now. |
| Ragged memory planning / global backing | **PASS** | `page_size_bytes`, total-memory accounting, and single global backing direction are coherent. |
| Dense-vs-Ragged identity memory invariant | **PASS** | Current formula preserves total uncompressed KV bytes. |
| B12 Scheduler ownership authority | **PASS** | Scheduler remains canonical allocator/ownership authority. |
| B12 Worker discardable mirror | **PASS** | Keep worker state non-authoritative. |
| Reserve → Write → Commit frontier ordering | **PASS** | Correct and should remain frozen. |
| `num_clusters` contract | **BLOCK until A3 hardening** | It is currently an independent constructor argument. It must be derived from placement geometry. |
| Member placement abstraction | **BLOCK until A3 hardening** | No canonical `MemberPlacementMap` exists yet. |
| Snapshot/Delta stale fencing | **PASS_WITH_CHANGE** | Shape fences are insufficient against ownership ABA; add one physical-state generation. |
| Snapshot XOR Delta | **PASS_WITH_CHANGE** | Freeze XOR and enforce it in the transport type. |
| Prefix-retained compaction pages | **PASS / FREEZE** | This is already the scheduler reconciliation semantics and should become an explicit V2 invariant. |
| Physical GPU backing | **BLOCK for GPU execution** | Planner allocates `Hp`-wide pages, but current worker reshape is still backend/Dense `Hkv` geometry. |
| Virtual block addressing | **PASS / ADOPT** | Adopt Tangram's algebra as a P1 execution address contract, not as allocator identity. |
| FlashAttention kernel rewrite | **DEFER / NOT REQUIRED** | The frozen layout is explicitly chosen so FA can consume virtual single-head blocks without a kernel rewrite. |
| Prefix caching for compressed Ragged state | **DEFER** | Current manager correctly fails closed. |
| Async free / CUDA stream fencing | **DEFER to hardening** | Address contract is compatible, but logical generation does not replace CUDA completion fencing. |

## 1.3 The one concrete execution mismatch

### SOURCE FACT

P1 A1/A2 currently plans one Ragged physical page as:

```text
page bytes
=
B × Hp × (Dk + Dv) × sizeof(dtype)
```

and allocates one global raw backing tensor shared by all Ragged layers.

However, current `qcrs/vllm` worker code in `vllm/v1/worker/gpu/attn_utils.py::_reshape_kv_cache` still sends:

```python
kv_cache_spec.num_kv_heads   # Hkv
```

to the normal backend `get_kv_cache_shape(...)` path.

There is no current `RaggedAttentionSpec` special case there.

### Consequence

If Ragged runtime activation were enabled now, Planner and Worker would disagree about what one `block_id/page_id` indexes:

```text
Planner:
page = Hp KV heads

Generic execution reshape:
block = Hkv KV heads
```

This is a real GPU execution blocker.

### Decision

This is **not** an A1/A2 architecture conflict because the current config explicitly describes Ragged runtime activation as deferred and the Ragged manager is not production-registered.

The correction is:

> Keep the `Hp` planner geometry. Later make the Worker materialize the corresponding Ragged backing shape.

Do **not** “fix” this by reverting the planner to Dense `Hkv` pages.

---

# 2. Current Architecture Reconstruction

## 2.1 Spec layer

### SOURCE FACT

Current `RaggedAttentionSpec` states:

```text
num_kv_heads
= semantic local KV-head count Hkv

page_group_size
= physical page width Hp
```

and exposes:

```text
num_head_groups_per_layer
= Hkv / Hp
```

Its current page byte formula is effectively:

\[
\text{page\_bytes}
=
B \cdot H_p \cdot (D_k + D_v) \cdot s
\]

where `s` is dtype bytes.

For the usual `Dk = Dv = D` case:

\[
\text{page\_bytes}
=
2BH_pDs
\]

The code also requires:

```text
Hkv % Hp == 0
```

### CURRENT PROJECT DESIGN

The spec deliberately does not own:

```text
placement
page IDs
allocation
runtime dispatch
```

This separation is correct.

### PROPOSED FREEZE

Keep this separation.

`RaggedAttentionSpec` defines **Ragged physical page geometry**, while `MemberPlacementMap` defines **which semantic members occupy those page columns**.

---

## 2.2 Planner layer

### SOURCE FACT

When Ragged planning is active, `get_kv_cache_config_from_groups(...)` currently:

1. requires exactly one Ragged KV cache group;
2. rejects mixed Ragged/non-Ragged groups;
3. obtains `page_size` from `RaggedAttentionSpec.page_size_bytes`;
4. computes:

```text
num_blocks = available_memory // page_size
```

5. emits one:

```text
KVCacheTensor(
    size=num_blocks * page_size,
    shared_by=all_ragged_layer_names,
)
```

### Interpretation

This means the intended allocator namespace is:

```text
one worker/rank-global physical page pool
```

not:

```text
one page-ID namespace per layer
```

A `physical_page_id` must therefore identify one physical page in a **single global backing**.

That is exactly the right direction for cross-layer/member placement.

---

## 2.3 Scheduler ownership layer

### SOURCE FACT

`RaggedAttentionManager` currently owns:

```text
request_id
→ RaggedRequestPhysicalState
    ├── effective_lens[C]
    └── page_rows[C][variable depth]
```

It performs:

```text
plan_capacity()
→ apply_capacity_plan()
→ commit_effective_lens()
```

The reserve step grows `page_rows` without prematurely advancing `effective_lens`.

The manager also explicitly rejects prefix caching for this path.

### CURRENT PROJECT DESIGN

```text
Scheduler
= canonical physical ownership authority
```

This is correct and should be preserved.

### Current weakness

`RaggedAttentionManager.__init__(..., num_clusters)` currently accepts `num_clusters` as an independent positive integer.

That is too weak once placement becomes a formal architecture object.

`C` must not be independently authored by Scheduler code.

---

## 2.4 Transport layer

### SOURCE FACT

Current DTOs include:

```text
RaggedRequestStateSnapshotData
RaggedPageAllocationDeltaData
RaggedCompactionResultData
RaggedKVUpdateData
```

Current Snapshot contains:

```text
request_id
effective_lens
page_counts
flat_page_ids
```

Current Delta contains:

```text
request_id
expected_source_effective_lens
expected_source_page_counts
appended_page_counts
flat_new_page_ids
```

Current compaction result already contains an optional field named `state_version`, but Snapshot and Delta do not, and the field is not yet a coherent end-to-end generation contract.

### SOURCE FACT: production wiring status

The current control-plane test manually attaches:

```python
output.ragged_kv_updates = RaggedKVUpdateData(...)
```

The reviewed production `scheduler.py` does not yet produce this transport path.

So B12 is presently a **tested control-plane substrate**, not a fully wired production path.

That is consistent with the current “production execution not ready” status.

---

## 2.5 Worker mirror

### SOURCE FACT

`RaggedWorkerPhysicalState` stores:

```text
rows           [Rmax, C, M]
counts         [Rmax, C]
effective_lens [Rmax, C]
```

in NumPy arrays.

It supports:

```text
apply_snapshot
apply_allocation_delta
commit_effective_lens
remove_request
gather
```

and rejects malformed or shape-stale mutations atomically.

### CURRENT PROJECT DESIGN

```text
Worker
= discardable physical-state mirror
```

This is correct.

The NumPy representation is also useful as a CPU oracle even after a production GPU-facing representation is added.

---

## 2.6 Execution gap

The missing chain is currently:

```text
(layer, kv_head)
        ↓
canonical member identity
        ↓
placement
        ↓
cluster / column
        ↓
request page row
        ↓
physical page id
        ↓
virtual block / slot
        ↓
Ragged GPU backing
        ↓
FA write/read metadata
```

A3 freezes this chain.

---

# 3. Tangram Reference Analysis

## 3.1 What Tangram proves

Tangram provides strong source-backed evidence that the following execution model is feasible:

```text
semantic member
= (layer, KV head)

member
→ (cluster, column)

physical backing
= [2, P, Hp, B, D]

virtual_block_id
= physical_page_id * Hp + column
```

and that this can be presented to FlashAttention as a standard single-KV-head paged layout.

That is the most important reference result for P1 A3.

---

## 3.2 Member identity and placement

Tangram explicitly defines a flat member row as:

```text
m = layer * num_kv_heads + head
```

and uses:

```text
member_to_cluster[m]
member_to_col[m]
```

Its identity mapping is:

```text
cluster = layer * (Hkv / Hp) + floor(head / Hp)
column  = head % Hp
```

Tangram also validates that the mapping is a bijection over full `(cluster, column)` slots.

The source explains why:

```text
duplicate slot
→ two heads write the same physical column
→ corruption

empty column
→ page capacity exists that no member owns
→ geometry/placement contract has diverged
```

### P1 decision

**ADOPT the abstraction and the R1 bijection invariant.**

Do not adopt Tangram's offline clustering algorithm or arbitrary graph machinery in this slice.

---

## 3.3 Column-major physical pages

Tangram's physical shape is:

```text
[2, num_physical_blocks, page_group_size, block_size, head_size]
```

The critical dimension order is:

```text
P → Hp → B
```

not:

```text
P → B → Hp
```

because `(physical_page, column)` must itself be one contiguous virtual single-head block.

Tangram then reshapes:

```text
[2, P, Hp, B, D]
→
[2, P * Hp, B, 1, D]
```

without moving payload.

### P1 decision

**ADOPT the same physical ordering for the R1 execution contract.**

This choice is not cosmetic; it is what makes the virtual block arithmetic correspond to a contiguous physical region.

---

## 3.4 Virtual block addressing

Tangram's helpers implement:

```text
virtual_block_id
=
physical_page_id * Hp + column
```

and:

```text
virtual_slot
=
virtual_block_id * B + block_offset
```

The virtual IDs are then supplied through normal FlashAttention block-table and slot-mapping interfaces.

### P1 decision

**ADOPT the arithmetic exactly.**

But freeze one semantic distinction:

```text
physical_page_id
= allocator / ownership identity

virtual_block_id
= derived execution address
```

The Scheduler must never allocate, free, or reconcile `virtual_block_id`.

---

## 3.5 RaggedBlockTable

Tangram's `RaggedBlockTable` adds a head-group/cluster axis and explicitly contains two different physical-row operations:

```text
compression:
keep front, free tail

sliding-window:
keep tail positions, null front
```

This is useful feasibility evidence that non-uniform row depth can be staged efficiently.

### P1 decision

Use it as a **production staging/data-structure reference** only.

P1 should not copy Tangram's allocator ownership model.

---

## 3.6 FlashAttention reuse

Tangram's `flash_attn.py` converts a Ragged cache into a virtual single-head view and then calls the existing cache write / varlen attention path.

The key idea is:

```text
physical layout adaptation
+
metadata address adaptation

NOT

FlashAttention algorithm rewrite
```

### P1 decision

This is the preferred execution direction.

A3 therefore explicitly does **not** introduce a new FA kernel contract.

---

## 3.7 Ragged prefill/decode execution

Tangram's `ragged_forward.py` treats each:

```text
(request, KV-head member)
```

as one varlen sequence carrying the member's GQA query-head group.

It separates:

```text
uniform decode
→ reshape-friendly

prefill / mixed
→ token-major to member-major materialization/copy
```

### P1 decision

This is execution feasibility evidence, not an A3 requirement.

C1 only freezes addresses. Prefill/decode tensor reshaping belongs to later execution work.

---

## 3.8 What P1 should explicitly not copy from Tangram

Do not copy the following as architecture requirements:

1. Tangram's full compression policy/scorer stack.
2. Its offline/AOT cluster-map production pipeline.
3. Its Worker-side allocation/free bookkeeping as P1's authority model.
4. Its hybrid sliding-window/full-attention complexity.
5. Its exact `RaggedBlockTable` lifecycle.
6. Its custom-op / piecewise-CUDA-Graph implementation now.
7. Its compression executor as a prerequisite for address translation.

The reusable invariant is much smaller:

```text
member placement
+
column-major pages
+
virtual addressing
+
standard FA-compatible view
```

---

# 4. Frozen Member / Placement Contract

## 4.1 MemberIdentity

For R1, a semantic KV member is frozen as:

```text
MemberIdentity
=
(local_layer_idx, local_kv_head_idx)
```

where:

```text
0 <= local_layer_idx < L
0 <= local_kv_head_idx < Hkv
```

Definitions:

- `L` = number of Ragged KV layers local to this Worker/PP stage and represented by this placement map.
- `Hkv` = semantic KV head count local to this TP rank.
- `Hp` = `page_group_size`.

The canonical flat member index is:

\[
m = layer \cdot H_{kv} + head
\]

### Important scope rule

`layer_idx` is the stable **local Ragged KV layer order**, not an uncontrolled model-module integer.

If a future PP mapping needs a global model layer ID, that is separate metadata and does not change the flat member algebra.

---

## 4.2 MemberPlacementMap

Freeze a model/rank-level immutable descriptor conceptually equivalent to:

```python
@dataclass(frozen=True)
class MemberPlacementMap:
    num_layers: int
    num_kv_heads: int
    page_group_size: int

    member_to_cluster: tuple[int, ...]
    member_to_column: tuple[int, ...]

    @property
    def num_members(self) -> int: ...

    @property
    def num_clusters(self) -> int: ...
```

This object is:

```text
model/rank physical-layout metadata
```

not:

```text
per-request state
```

Every request under the same Ragged execution configuration uses the same placement map.

---

## 4.3 Derivation of C

Freeze:

\[
M = L \cdot H_{kv}
\]

and:

\[
C = M / H_p
\]

Therefore:

```text
num_clusters
MUST be derived from MemberPlacementMap
```

and must no longer be an independent user/caller-selected runtime parameter.

### R1 validation

```text
M % Hp == 0
```

must hold.

---

## 4.4 Bijection invariant

For R1:

```text
members
→ (cluster, column)
```

must be a bijection onto:

```text
{0 ... C-1} × {0 ... Hp-1}
```

Therefore:

1. `len(member_to_cluster) == L * Hkv`;
2. `len(member_to_column) == L * Hkv`;
3. every cluster ID is in `[0, C)`;
4. every column ID is in `[0, Hp)`;
5. no two members map to the same `(cluster, column)`;
6. every `(cluster, column)` slot has exactly one member;
7. every cluster contains exactly `Hp` members;
8. no sparse/empty page column is allowed in R1.

This is deliberately stricter than a future arbitrary placement graph.

---

## 4.5 Cluster semantics

A cluster is the physical allocation/retention unit for the members placed into its `Hp` columns.

Therefore all members in one cluster share:

```text
page_rows[c]
effective_lens[c]
```

This is an important semantic consequence.

R1 does **not** support two members inside the same cluster having different retained sequence lengths.

Future per-head independence is still possible without changing this control-plane abstraction:

```text
option A:
Hp = 1

option B:
place members requiring independent retention into different clusters
```

So this restriction does not force a later control-plane rewrite.

---

## 4.6 Default identity placement

For R1, the default placement should be deterministic and policy-free:

```text
cluster = layer * (Hkv // Hp) + (head // Hp)
column  = head % Hp
```

No AOT clustering algorithm is required.

A future policy may supply another valid bijection.

---

## 4.7 Concrete example

Given:

```text
L   = 2
Hkv = 4
Hp  = 2
```

then:

```text
num_members  = 2 * 4 = 8
num_clusters = 8 / 2 = 4
```

Default map:

| m | member | cluster | column |
|---:|---|---:|---:|
| 0 | `(layer0, head0)` | 0 | 0 |
| 1 | `(layer0, head1)` | 0 | 1 |
| 2 | `(layer0, head2)` | 1 | 0 |
| 3 | `(layer0, head3)` | 1 | 1 |
| 4 | `(layer1, head0)` | 2 | 0 |
| 5 | `(layer1, head1)` | 2 | 1 |
| 6 | `(layer1, head2)` | 3 | 0 |
| 7 | `(layer1, head3)` | 3 | 1 |

The critical point is:

```text
C = 4
```

is not separately configured.

---

# 5. Frozen Physical Page Layout

## 5.1 Physical page identity

Freeze:

> One `physical_page_id` is one `BlockPool`-managed Ragged physical page in the worker-global Ragged backing.

For active ownership:

```text
physical_page_id > 0
```

because vLLM block ID `0` remains reserved/null semantics.

The page ID namespace is global across:

```text
requests
clusters
layers
heads
```

for that backing/BlockPool.

There is no separate `(layer, page_id)` allocator identity.

---

## 5.2 Logical K/V page geometry

The most general geometry already implied by the current P1 spec is:

```text
K page:
[Hp, B, Dk]

V page:
[Hp, B, Dv]
```

with one paired physical page ID owning both K and V planes.

Page bytes:

\[
B \cdot H_p \cdot (D_k + D_v) \cdot s
\]

### R1 execution-supported representation

For the current FlashAttention-oriented MVP, freeze the execution subset:

```text
Dk == Dv == D
```

and represent the contiguous physical backing as:

```text
[2, P, Hp, B, D]
```

where:

```text
axis 0 = K/V
axis 1 = physical page id
axis 2 = page column
axis 3 = token offset in page
axis 4 = head dimension
```

If a future model requires `Dk != Dv`, retain the same physical-page identity and addressing semantics but materialize separate K/V planes; do not distort the R1 `[2,...]` tensor shape.

---

## 5.3 Why the column dimension must precede block_size

Freeze physical order:

```text
[P, Hp, B, D]
```

not:

```text
[P, B, Hp, D]
```

because the execution contract requires:

```text
(physical_page_id, column)
```

to identify one contiguous `B × D` single-head block.

Then:

```text
[2, P, Hp, B, D]
```

can be viewed as:

```text
[2, P * Hp, B, 1, D]
```

without moving KV payload.

---

## 5.4 Alignment across P1 layers

The following terms are now frozen to refer to the **same physical object**:

```text
RaggedAttentionSpec page
KVCacheTensor planned page unit
BlockPool KVCacheBlock / block_id
Scheduler page_rows element
Worker mirror rows element
GPU backing P-axis element
```

This equality is a core invariant.

There must never be a state where:

```text
Planner page
= Hp-head object
```

while:

```text
Worker block
= Hkv-head object
```

---

## 5.5 Dense identity/no-compression memory proof

Let:

```text
L  = local Ragged layers
N  = number of token blocks per request
   = ceil(max_effective_len / B)
Hkv = semantic KV heads per layer
Hp  = heads per physical page
s   = dtype bytes
```

Dense per-layer memory:

\[
N \cdot B \cdot H_{kv} \cdot (D_k + D_v) \cdot s
\]

Ragged pages per layer/depth:

\[
H_{kv}/H_p
\]

Ragged memory:

\[
N \cdot \frac{H_{kv}}{H_p}
\cdot B \cdot H_p \cdot (D_k + D_v) \cdot s
\]

Therefore:

\[
\text{Ragged bytes}
=
\text{Dense bytes}
\]

before compression/retention changes the number of owned pages.

Across `L` layers both sides are simply multiplied by `L`.

This is the correct identity baseline:

> Ragged paging changes partitioning and ownership granularity, not baseline KV information volume.

---

# 6. Frozen Address Translation Contract

## 6.1 Inputs

Given:

```text
request_id
layer_idx
local_kv_head_idx
effective_position
placement
request physical state
```

where:

```text
request physical state
=
page_rows[C]
effective_lens[C]
```

address resolution follows one canonical chain.

---

## 6.2 Canonical formula

### Step 1 — semantic member

\[
m = layer \cdot H_{kv} + head
\]

### Step 2 — placement

\[
c = member\_to\_cluster[m]
\]

\[
col = member\_to\_column[m]
\]

### Step 3 — effective page coordinate

\[
page\_index = \lfloor e / B \rfloor
\]

\[
block\_offset = e \bmod B
\]

where `e = effective_position`.

### Step 4 — allocator-owned physical page

\[
p = page\_rows[c][page\_index]
\]

### Step 5 — execution virtual block

\[
virtual\_block\_id = p \cdot H_p + col
\]

### Step 6 — execution virtual slot

\[
virtual\_slot
=
virtual\_block\_id \cdot B + block\_offset
\]

Equivalently:

\[
virtual\_slot
=
(pH_p + col)B + block\_offset
\]

---

## 6.3 Physical tensor coordinate

For the R1 `[2, P, Hp, B, D]` backing:

```text
K location
= backing[0, p, col, block_offset, :]

V location
= backing[1, p, col, block_offset, :]
```

The K/V plane does not alter placement or virtual block identity.

---

## 6.4 Read frontier vs reserved write capacity

Do not conflate:

```text
owned capacity
```

with:

```text
committed effective frontier
```

For a historical/read address:

```text
effective_position < effective_lens[c]
```

must hold.

For a planned write:

```text
page_index < len(page_rows[c])
```

must hold after reserve, even though:

```text
effective_position >= current effective_lens[c]
```

may be valid until write completion and frontier commit.

Therefore the core address resolver should resolve **owned physical capacity**; read/write visibility is a separate validation rule.

This preserves the already-correct:

```text
Reserve
→ Write
→ Commit effective frontier
```

transaction.

---

## 6.5 Where virtual addresses are generated

Freeze responsibility as:

```text
Scheduler
    owns physical page IDs and page rows
    does NOT materialize virtual block/slot addresses

Worker / execution metadata layer
    combines:
        placement
        worker page-row mirror
        per-step effective positions
    and materializes virtual block tables / virtual slots
```

This matches the architecture boundary:

```text
control plane
= ownership

data plane
= address materialization
```

and avoids shipping execution-specific expanded tables through Scheduler transport.

---

## 6.6 Concrete numerical example

Use the earlier placement:

```text
L   = 2
Hkv = 4
Hp  = 2
B   = 16
```

Suppose request physical rows are:

```text
cluster 0: [10, 11]
cluster 1: [20, 21, 22]
cluster 2: [30, 31]
cluster 3: [40, 41]
```

Resolve:

```text
layer = 1
head  = 2
effective_position = 21
```

Member:

```text
m = 1 * 4 + 2 = 6
```

From placement:

```text
member 6
→ cluster 3
→ column 0
```

Effective page coordinate:

```text
page_index  = 21 // 16 = 1
block_offset = 21 % 16 = 5
```

Physical page:

```text
page_rows[3][1] = 41
```

Virtual block:

```text
virtual_block_id
= 41 * 2 + 0
= 82
```

Virtual slot:

```text
virtual_slot
= 82 * 16 + 5
= 1317
```

Physical payload coordinate:

```text
K = backing[0, 41, 0, 5, :]
V = backing[1, 41, 0, 5, :]
```

No Scheduler-side virtual ID is required.

---

# 7. Versioning / Transaction Contract

## 7.1 Why current shape fencing is insufficient

Current stale detection uses combinations of:

```text
expected_effective_lens
expected_page_counts
```

Those values identify **shape**, not ownership identity.

An ABA is possible in principle:

```text
State A
E/counts = X
pages = [10, 11, ...]

→ compaction/free/reuse/rebuild

State B
...

→ later

State C
E/counts = X
pages = [50, 51, ...]
```

An old Delta or result referring to A can satisfy C's shape check even though it refers to the wrong physical pages.

This becomes especially unsafe once:

```text
async GPU work
preemption/resume
request re-add
page reuse
```

are allowed.

Therefore a generation fence is required now, before execution contracts are built on top of B12.

---

## 7.2 Minimal correct design

Freeze exactly **one** additional generation:

```text
state_version
```

but define its meaning narrowly:

> `state_version` is the generation of the request's physical layout / payload interpretation, not a second counter for every frontier update.

Do **not** add a separate `frontier_version`.

The effective frontier remains fenced by the actual value:

```text
expected_effective_lens
```

This is the minimal design that closes physical ABA without introducing two coupled clocks.

---

## 7.3 What increments state_version

`state_version++` whenever a mutation can make the same `(effective_lens, page_counts)` refer to a different physical state or payload interpretation.

### Increment

```text
reserve that appends physical pages
    page_rows changes

compaction / repacking
    surviving logical members may move to new offsets
    and/or page_rows shrinks

canonical ownership replacement / rebuild
    page identity changes

request physical state destroyed and later recreated
    new incarnation must not reuse the old generation
```

### Do not increment

```text
normal commit_effective_lens after successful append write
```

Reason:

- owned pages do not change;
- historical payload mapping does not change;
- the frontier is already value-fenced by `expected_effective_lens`;
- the reserve generation is still the correct physical layout generation.

### Snapshot / mirror operations

```text
export snapshot
apply snapshot on Worker
Worker remove from persistent batch
Worker re-add/resync from same Scheduler canonical state
```

must **not** advance the Scheduler generation.

They materialize an existing canonical generation.

---

## 7.4 Request re-use / incarnation rule

A `request_id` must never restart from a previously used generation while the Scheduler process is alive.

Implementation can keep one lightweight last-generation ledger per request ID.

Conceptually:

```text
old request physical state ended at v=12

same request_id later obtains a new canonical physical incarnation
→ version >= 13
```

No separate incarnation counter is needed.

`state_version` is sufficient.

---

## 7.5 Required transport semantics

### Snapshot

Add:

```text
state_version
```

A Snapshot means:

> Replace the Worker mirror with this exact canonical state generation.

Worker does not require source-generation equality for a full snapshot; the snapshot is the resynchronization authority.

### Allocation Delta

Add:

```text
expected_source_state_version
new_state_version
```

Worker accepts only if:

```text
worker.state_version == expected_source_state_version
```

plus the existing expected E/count checks.

After atomically appending pages:

```text
worker.state_version = new_state_version
```

Scheduler is the producer of `new_state_version` because Scheduler owns the page allocation mutation.

### Frontier commit

`commit_effective_lens(...)` should additionally require:

```text
expected_state_version
```

but normal frontier commit does not increment it.

Therefore a write result cannot commit against a physically replaced row even if its old `effective_lens` happens to match.

### Compaction result

The current optional ambiguous field:

```text
state_version
```

should be replaced/clarified as:

```text
expected_source_state_version
```

Compaction reconciliation validates:

```text
source version
source E
source page counts
```

before changing canonical rows.

Successful compaction advances the canonical physical generation exactly once.

The Worker must not invent freed page IDs; it still reports final shape only.

---

## 7.6 Normal write transaction ordering

Freeze:

```text
S0:
Scheduler canonical state
(page_rows, E, version=v)

1. Plan
   pure calculation against v / E / counts

2. Reserve
   Scheduler allocates new physical pages if required
   page_rows changes
   state_version advances when ownership changes
   Delta carries source and new version

3. Worker apply Delta
   validates source version + E + counts
   installs page IDs

4. Address materialization
   Worker derives virtual blocks / slots

5. Execute KV write
   data lands in reserved capacity

6. Completion
   synchronous MVP: forward completion
   future async: CUDA event / stream fence

7. Commit effective frontier
   validates expected state_version + expected E
   E advances
   state_version remains the current physical generation
```

The key invariant remains:

> Capacity may be owned before data is committed visible.

---

## 7.7 Compaction transaction ordering

Freeze:

```text
1. Start from exact source:
   version=v
   E=source_E
   counts=source_counts

2. Worker executes compaction payload movement
   survivors are packed into retained prefix pages

3. Completion fence
   payload movement must be complete before detached pages are reusable

4. Worker reports:
   expected source version
   expected source E/counts
   new E/counts

5. Scheduler validates exact source

6. Scheduler derives:
   retained rows = old_row[:new_count]
   detached rows = old_row[new_count:]

7. Scheduler commits new canonical physical state
   state_version advances once

8. Scheduler returns detached physical pages to BlockPool
```

Future asynchronous execution adds a CUDA completion fence to step 3; it does not change the logical state contract.

---

# 8. Compaction Physical Invariants

## 8.1 Prefix-retained page invariant

Freeze this as a formal P1 V2 invariant:

For every cluster `c`:

```text
old_row[c]
=
[p0, p1, ..., pn]
```

and a compaction result with `new_count[c] = k` implies:

```text
new_row[c]
=
old_row[c][:k]
```

Detached pages are exactly:

```text
old_row[c][k:]
```

Example:

```text
[B10, B11, B12, B13]
→
[B10, B11]
```

Allowed payload movement:

```text
surviving KV
→ packed into B10/B11
```

Not allowed in P1 V2:

```text
[B10, B13]
```

or:

```text
allocate B50/B51 and relocate survivors there
```

---

## 8.2 Meaning for the compaction kernel

The future payload compaction implementation must treat the destination as:

```text
the existing physical prefix pages
```

It may rewrite data inside them.

It must not:

```text
allocate pages
choose destination page IDs
free pages
change Scheduler ownership directly
```

This keeps allocator authority out of GPU execution code.

---

## 8.3 Meaning for BlockPool free

BlockPool release remains Scheduler-side.

The Worker reports:

```text
source shape
final shape
```

not:

```text
freed page IDs
```

Scheduler derives detached IDs from its canonical row.

This property should remain frozen.

---

## 8.4 Meaning for reconciliation

Because retention is prefix-based in physical-page space, Scheduler reconciliation is deterministic:

```text
candidate_rows[c]
= old_rows[c][:new_count[c]]
```

No arbitrary page-set comparison is required.

This is one of the main reasons not to support arbitrary relocation in V2.

---

## 8.5 Meaning for address stability

Be precise about what is stable.

### Stable

```text
physical page IDs of the retained page prefix
```

### Not necessarily stable

```text
a surviving original token/member's old offset
```

Compaction intentionally repacks survivors, so a survivor may move from one effective position/slot to another **inside the retained page prefix**.

After compaction, addressing is always recomputed from:

```text
new effective position
→ retained page row
→ virtual address
```

This is correct.

---

# 9. Worker Execution-State Direction

## 9.1 Keep NumPy mirror as a reference oracle

The current `RaggedWorkerPhysicalState` should not be thrown away when GPU execution work starts.

It has high value as:

```text
CPU reference state
atomicity oracle
transport validation oracle
resync test target
C1 address-resolution test substrate
```

### Decision

Keep it.

Do not attempt to make NumPy itself the production attention metadata path.

---

## 9.2 Production direction

Later Worker execution should converge toward a staged representation conceptually like:

```text
host canonical/mirror metadata
        ↓
pinned/UVA staged tensors
        ↓
Ragged block-table views
        ↓
GPU metadata builder
        ↓
member virtual block tables / slots
```

Likely production tensors include the equivalent of:

```text
cluster_rows   [Rmax, C, M]
page_counts    [Rmax, C]
effective_lens [Rmax, C]
member_to_cluster [L*Hkv]
member_to_column  [L*Hkv]
```

The important architecture rule is not the exact buffer class.

It is:

> GPU-facing staging is a materialization of the Worker mirror; it is not a second ownership authority.

---

## 9.3 Why this direction extends cleanly

### Token-level compaction

Non-uniform cluster lengths are already represented.

### Per-head/per-layer retention

Placement controls grouping. `Hp=1` provides fully independent members if needed.

### Preemption / resume

Snapshot + state generation rebuilds the disposable Worker state.

### Async GPU execution

Generation checks solve logical stale state; CUDA events solve completion ordering.

They are separate concerns.

### CUDA Graph

The address algebra is static. Only step tables/lengths vary. A future staged table or piecewise graph path does not change the ownership contract.

### Prefix reuse

The current Ragged manager correctly disables prefix caching.

A compacted active-request state may no longer correspond to the ordinary contiguous token-prefix semantics required by standard prefix cache hashing. Prefix reuse must be designed explicitly later, not inherited accidentally.

### LMCache / offload

A physical page now has an explicit geometry and identity, which is a usable future serialization/transfer unit.

No current offload integration is needed.

### KV quantization / heterogeneous representation

A future page descriptor may add:

```text
representation / pool / codec metadata
```

without changing:

```text
member → cluster/column
request → page_rows
```

The address-to-storage final step can later become representation-aware.

---

# 10. RaggedAttentionSpec vs Attention Semantics vs Compression Policy

Three independent axes must remain conceptually separate:

```text
Attention semantics:
Full / Sliding / MLA / ...

Physical KV layout:
Dense / Ragged

Compression policy:
random / SnapKV-like / importance / ...
```

## 10.1 Current naming imperfection

`RaggedAttentionSpec` currently packages enough attention-spec fields plus Ragged physical geometry that the type name can look like a new attention semantic.

It is not.

In P1 R1:

```text
semantic attention
= Full attention baseline

physical KV layout
= Ragged
```

---

## 10.2 Should it be split now?

Possible theoretically cleaner design:

```text
FullAttentionSpec
+
RaggedPhysicalLayoutSpec
```

### Decision: **DO NOT REFACTOR NOW**

Reasons:

1. vLLM's KV cache planning already uses `KVCacheSpec` as the integration point for memory geometry.
2. Splitting it now would propagate through spec grouping, planner dispatch, manager selection, worker initialization, and backend selection before any execution benefit is obtained.
3. P1 is a bounded resume project, not a framework-wide type-system redesign.
4. The current class can express the required R1 geometry correctly.
5. A future need to compose Ragged layout with multiple semantic attention types would provide a concrete reason to extract a separate physical-layout object.

### Required documentation clarification

Freeze the meaning as:

> `RaggedAttentionSpec` is the P1 MVP vLLM integration wrapper for Ragged KV physical geometry. It does not define a compression policy and should not be interpreted as a new attention algorithm.

---

# 11. Explicit Non-Goals

R1-A3 does not implement or require:

```text
compression scorer
AOT clustering algorithm
learned placement
arbitrary sparse cluster graph
production prefix caching
LMCache
offload
remote KV
KV quantization
heterogeneous storage pools
FlashAttention kernel rewrite
new attention algorithm
full async runtime
CUDA event plumbing
piecewise CUDA Graph implementation
production Ragged prefill/decode reshaping
GPU compaction kernel changes
formal benchmark
```

Also not required now:

```text
FullAttentionSpec + RaggedPhysicalLayoutSpec refactor
```

---

# 12. Required Code Adjustments Before C1

This section distinguishes:

```text
A. contract hardening needed before C1 is accepted
B. GPU execution work that must NOT be pulled into C1
```

## 12.1 MUST — canonical placement object

### Suggested location

```text
new neutral module:
vllm/v1/ragged_kv_layout.py
```

The exact filename may vary, but it should not live inside FlashAttention implementation code because both Scheduler-side validation and Worker execution need the same contract.

### Add

```text
MemberPlacementMap
identity/default placement constructor
bijection validation
flat member index helper
num_clusters derivation
```

### Reason

Today the system has `C`-shaped state without a canonical object proving what each cluster/column means.

### Expected change

All code that needs `C` obtains:

```text
placement.num_clusters
```

rather than independently authoring `num_clusters`.

---

## 12.2 MUST — bind RaggedAttentionManager to placement-derived C

### File

```text
vllm/v1/core/ragged_kv_cache_manager.py
```

### Current issue

```python
RaggedAttentionManager(..., num_clusters: int)
```

accepts any positive integer.

### Required direction

Prefer conceptually:

```python
RaggedAttentionManager(..., placement: MemberPlacementMap)
```

then:

```text
self.num_clusters = placement.num_clusters
```

At minimum, if API shape is temporarily kept, constructor validation must prove supplied C equals placement-derived C. Long term, there should be only one source.

---

## 12.3 MUST — add physical generation fencing

### Files

```text
vllm/v1/core/ragged_kv_cache_manager.py
vllm/v1/core/sched/output.py
vllm/v1/worker/gpu/ragged_kv_state.py
```

### Required model changes

Add canonical physical-state generation to Scheduler state and Worker mirror.

Transport semantics:

```text
Snapshot:
state_version

Allocation Delta:
expected_source_state_version
new_state_version

Compaction Result:
expected_source_state_version
```

Normal frontier commits validate `expected_state_version` but do not need another frontier counter.

### Reason

Close ownership/payload ABA without introducing multiple generation clocks.

---

## 12.4 MUST — enforce Snapshot XOR Delta

### File

```text
vllm/v1/core/sched/output.py
```

### Required invariant

For one `RaggedKVUpdateData`:

```text
set(snapshots) ∩ set(allocations) == ∅
```

### Recommended enforcement

Add a transport-level validation such as `__post_init__`.

### Same-step materialization rule

If a request requires full materialization and also receives new reserved pages in the same Scheduler step:

```text
perform Scheduler-side reserve first
→ emit one final Snapshot of the post-reserve canonical state
```

Do not emit:

```text
Snapshot(req) + Delta(req)
```

in the same output.

There is no compelling current use case that requires both.

---

## 12.5 MUST — update CPU tests

### Files

```text
tests/v1/core/test_ragged_kv_cache_manager.py
tests/v1/worker/test_ragged_kv_state.py
```

Add tests for:

```text
placement-derived C
invalid duplicate (cluster,column)
empty column / non-bijection
version-stale Delta despite equal E/counts
version-stale compaction despite equal E/counts
request remove/re-add generation non-reuse
Snapshot XOR Delta rejection
```

Most importantly add an explicit ABA test:

```text
A: same E/counts, pages A, version v
C: same E/counts, pages C, version v+N
old message from A
→ reject
```

---

## 12.6 SHOULD — clarify R1 execution-supported geometry

### File

```text
vllm/v1/kv_cache_interface.py
```

Document:

```text
semantic geometry supports Dk/Dv byte accounting
R1 FlashAttention execution path initially requires Dk == Dv
```

No large refactor is required.

---

## 12.7 DO NOT MODIFY FOR C1 — GPU backing reshape

### File

```text
vllm/v1/worker/gpu/attn_utils.py
```

### Known blocker

Current code routes `RaggedAttentionSpec` through normal backend shape creation using `Hkv`.

### Future required execution change

Before GPU execution, add a Ragged-specific shape path equivalent in semantics to:

```text
[2, P, Hp, B, D]
```

for the supported R1 shape.

Tangram's `column_major_cache_shape(...)` is the direct feasibility reference.

### Why not now

C1 is CPU reference addressing. Pulling GPU allocation/reshape into C1 would collapse the intended architecture-first sequencing.

---

## 12.8 DO NOT MODIFY FOR C1 — block-table/attention path

Defer these to later execution slices:

```text
vllm/v1/worker/gpu/block_table.py
vllm/v1/worker/gpu/input_batch.py
attention metadata builder
FlashAttention integration path
```

Future work will materialize:

```text
cluster page rows
→ member virtual block tables
→ member virtual slots
```

Do not rewrite FlashAttention kernels.

---

## 12.9 DO NOT MODIFY NOW — unrelated scope

Do not touch:

```text
compression scorer
KV compaction kernel
prefix cache
LMCache
KV Connector
offload
quantization
CUDA Graph policy
async free
```

A3 does not require them.

---

# 13. Snapshot / Delta Exclusivity Freeze

## Decision

Freeze:

```text
For any request in one SchedulerOutput:

Snapshot XOR Delta
```

## Why

A Snapshot means:

```text
ignore prior Worker mirror
materialize exact canonical state
```

A Delta means:

```text
Worker must already hold the exact source generation
append only this capacity
```

Those are mutually exclusive synchronization modes.

Allowing both creates avoidable ambiguity:

```text
Does Delta source refer to pre-Snapshot state?
post-Snapshot state?
Is application order semantically significant?
```

None of that is needed.

## Same-step case

A legitimate same-step workflow can always be expressed as:

```text
Scheduler canonical mutations
→ final Snapshot
```

for a materializing request.

Therefore XOR is both simpler and sufficient.

---

# 14. Long-Term Architecture Audit

## 14.1 Token-level compaction

**Compatible.**

Compaction changes effective positions and row depths but continues to resolve addresses through the same placement/page-row algebra.

---

## 14.2 Non-uniform per-head/per-layer retention

**Compatible with one constraint.**

Non-uniformity is cluster-granular.

Members that must diverge cannot share a cluster.

`Hp=1` is the fully independent limit.

No Scheduler control-plane redesign is required.

---

## 14.3 Preemption / resume

**Compatible.**

Scheduler retains canonical state; Worker can discard and reconstruct from a versioned Snapshot.

---

## 14.4 Async GPU execution

**Compatible but not solved by state_version alone.**

Need two independent safeguards:

```text
logical stale-state fence
= state_version + expected E/counts

physical completion fence
= CUDA event / stream completion
```

Never free detached physical pages merely because the logical result has arrived if GPU writes are still in flight.

---

## 14.5 Piecewise CUDA Graph

**Compatible.**

The placement and virtual-address formulas are static algebra. Future captured/eager boundaries only change where tables are staged/materialized.

---

## 14.6 Future prefix reuse

**Deferred by design.**

Standard prefix-cache semantics cannot be assumed for a lossy/compacted Ragged active-request state.

Keep fail-closed until semantic compatibility is explicitly designed.

---

## 14.7 LMCache / offload

**Compatible.**

The architecture exposes a clean future transfer object:

```text
physical page ID
+
physical page geometry
+
representation metadata later
```

No ownership redesign is implied.

---

## 14.8 Quantization / heterogeneous representation

**Compatible.**

Future address translation may become:

```text
physical page id
→ representation descriptor
→ pool + offset
```

while the current:

```text
member → cluster → page row
```

control-plane remains valid.

---

# 15. Next Slice Definition — R1-C1 CPU Reference Addressing

## 15.1 Slice name

```text
P1-V2 R1-C1 — CPU Reference Physical Address Resolution
```

## 15.2 Goal

Implement a pure CPU/reference oracle for the frozen A3 algebra.

The core API should be conceptually equivalent to:

```python
resolve_kv_address(
    layer_idx,
    local_kv_head_idx,
    effective_position,
    placement,
    physical_state,
)
```

and return:

```text
member
cluster
column
page_index
physical_page_id
virtual_block_id
block_offset
virtual_slot
```

Recommended result type:

```python
@dataclass(frozen=True)
class ResolvedKVAddress:
    member: int
    cluster: int
    column: int
    page_index: int
    physical_page_id: int
    virtual_block_id: int
    block_offset: int
    virtual_slot: int
```

No GPU pointer is returned in C1.

---

## 15.3 Scope

C1 may implement:

```text
MemberPlacementMap
identity placement
placement validation
pure address formula
capacity bounds checks
CPU tests
```

If the A3 generation/XOR hardening is not landed separately first, treat it as a small `C1-PRE` contract patch before the resolver is accepted.

---

## 15.4 C1 non-goals

C1 must not implement:

```text
GPU KV allocation
Ragged cache tensor reshape
GPU slot-mapping kernels
FlashAttention changes
RaggedBlockTable production path
compaction kernel
KV write kernel
prefill/decode member-major conversion
CUDA Graph
benchmark
prefix cache
LMCache
```

---

## 15.5 C1 validation rules

The resolver must reject:

```text
layer outside [0, L)
head outside [0, Hkv)
negative effective_position
invalid/non-bijective placement
page_index beyond owned row depth
null physical page id 0 as an active address
```

Do not require:

```text
effective_position < effective_lens[c]
```

inside the lowest-level capacity resolver, because a reserved write address may intentionally lie beyond the current committed frontier.

Instead test/read helpers should explicitly distinguish:

```text
committed read visibility
vs
reserved write capacity
```

---

## 15.6 Required C1 tests

### Test 1 — canonical identity example

```text
L=2, Hkv=4, Hp=2, B=16
```

Verify all eight member mappings and `C=4`.

### Test 2 — numerical virtual address

Use:

```text
cluster3 row = [40,41]
member=(layer1,head2)
effective_position=21
```

Expect:

```text
m=6
c=3
col=0
page_index=1
p=41
virtual_block=82
offset=5
virtual_slot=1317
```

### Test 3 — column distinction

Same physical page and effective position, two members in different columns must differ by exactly one virtual block:

```text
vb(col+1) = vb(col) + 1
```

### Test 4 — page-depth bounds

Addressing beyond a cluster's owned row rejects even if another cluster has a deeper row.

### Test 5 — non-uniform cluster depths

Verify independent rows such as:

```text
counts = [2,4,2,3]
```

resolve correctly.

### Test 6 — custom valid bijection

Use a non-identity valid placement and prove the same formula follows the supplied map rather than hardcoded layer-major grouping.

### Test 7 — malformed maps

Reject:

```text
duplicate (cluster,column)
missing slot
out-of-range cluster
out-of-range column
wrong member count
```

### Test 8 — dense identity memory algebra

Keep/extend the current A1 test proving:

```text
num_head_groups_per_layer * ragged_page_bytes
== dense_page_bytes
```

### Test 9 — version ABA regression

Even though not part of the numerical resolver itself, C1 gate should not pass until an equal-shape/different-generation stale message is rejected.

### Test 10 — Snapshot XOR Delta

Transport rejects the same request appearing in both maps.

---

## 15.7 C1 acceptance criteria

C1 is **PASS** only if:

1. `MemberPlacementMap` is the single placement source of truth.
2. `C` is derived, not freely authored.
3. The canonical address formula is implemented once in a pure CPU helper.
4. All formulas are tested against explicit numerical examples.
5. Non-uniform cluster row depths work.
6. No GPU/FA implementation is introduced.
7. State generation ABA protection is present.
8. Snapshot/Delta XOR is enforced.
9. Existing B12 state/transport tests remain green after contract changes.
10. A short evidence record captures files changed and exact test commands/results.

---

## 15.8 C1 evidence expected from Codex

```text
SOURCE / DIFF
- exact files changed
- new placement/address types
- no GPU execution files changed unless explicitly authorized

TESTS
- targeted placement/address tests
- B12 manager tests
- B12 worker-state tests

EVIDENCE
- one printed L=2/Hkv=4/Hp=2 resolution example
- one custom-placement example
- one ABA rejection example
- one Snapshot XOR Delta rejection example
```

C1 proves:

```text
address semantics are correct
```

It does **not** claim:

```text
production GPU execution is correct
```

---

# 16. Post-C1 Execution Roadmap Boundary

After C1 passes, the next execution work should proceed in layers rather than jumping directly into FA changes.

Recommended order:

```text
C1
CPU address oracle

→ C2
Ragged GPU backing materialization
[2,P,Hp,B,D]

→ C3
Worker staged cluster block tables
+ virtual block/slot materialization

→ C4
KV write/read integration using existing FA interfaces

→ C5
prefill/decode correctness

→ C6
compaction payload execution integration

→ later
async fencing / CUDA Graph / performance hardening
```

The exact numbering can be adjusted by the project plan, but the dependency direction should remain.

---

# 17. Final Verdict

## A1/A2 是否需要重构？

**No.**

Verdict:

```text
A1 Ragged Spec
= PASS_WITH_CHANGE

A2 Planner
= PASS
```

Required changes are contract hardening and execution materialization, not a planner redesign.

The current page-byte formula and global backing direction are correct.

---

## B12 是否可以冻结？

**Yes, the B12 architecture can be frozen.**

Freeze:

```text
Scheduler = canonical ownership authority
Worker = discardable mirror
Reserve → Write → Commit
Worker reports compaction shape, not allocator-authoritative freed IDs
Scheduler derives detached pages and frees them
```

But before production execution, B12 transport must receive the A3 amendments:

```text
placement-derived C
state_version physical-generation fence
Snapshot XOR Delta validation
```

These are narrow contract completions, not a B12 redesign.

---

## RaggedAttentionSpec 是否保留？

**Yes. Keep it for the MVP.**

Do not refactor now into:

```text
FullAttentionSpec + RaggedPhysicalLayoutSpec
```

Document that Ragged is a physical KV layout specialization, not a compression policy or a new attention algorithm.

---

## MemberPlacementMap 应该怎么定义？

Freeze:

```text
member = (local_layer_idx, local_kv_head_idx)

m = layer * Hkv + head

member_to_cluster[m]
member_to_column[m]

C = L * Hkv / Hp
```

R1 requires a complete bijection onto all `C × Hp` slots.

`num_clusters` is derived and must not remain an arbitrary caller knob.

---

## physical page layout 应该是什么？

R1 supported physical backing:

```text
[2, P, Hp, B, D]
```

with column-major page layout:

```text
P → Hp → B → D
```

Semantic generalization remains:

```text
K: [P,Hp,B,Dk]
V: [P,Hp,B,Dv]
```

Page ID is the same BlockPool/Scheduler/Worker/GPU-backing identity.

---

## 是否采用 virtual_block_id？

**Yes.**

Freeze:

```text
virtual_block_id
= physical_page_id * Hp + column

virtual_slot
= virtual_block_id * B + offset
```

But virtual IDs are execution-derived addresses only.

They are not allocator identities and never become Scheduler ownership objects.

---

## 是否必须现在加入 state_version？

**Yes.**

Shape-only fencing has an ownership ABA hole.

Use one generation:

```text
state_version
```

for physical-layout/payload generations.

Do not add a second frontier generation.

Continue using `expected_effective_lens` to fence the normal committed frontier.

---

## 当前代码具体需要修改哪些地方？

### Before C1 acceptance

```text
1. new neutral Ragged layout module
   - MemberPlacementMap
   - validation
   - derived C

2. ragged_kv_cache_manager.py
   - remove/free-proof arbitrary C
   - bind state to placement
   - physical state generation

3. sched/output.py
   - version fields
   - Snapshot XOR Delta validation
   - clarify compaction version semantics

4. worker/gpu/ragged_kv_state.py
   - mirror state_version
   - generation validation
   - placement-derived C

5. targeted CPU tests
   - placement
   - ABA
   - XOR
```

### Before GPU execution, but not C1

```text
6. worker/gpu/attn_utils.py
   - Ragged-specific [2,P,Hp,B,D] backing reshape

7. Worker block-table / metadata staging
   - cluster rows
   - member virtual blocks
   - virtual slots

8. attention integration
   - virtual view
   - reuse existing FA cache/write/read APIs
```

### Do not modify now

```text
FA kernel
compression scorer
LMCache
offload
prefix caching
full async runtime
CUDA Graph policy
```

---

## 修改后是否可以正式进入 R1-C1？

**Yes.**

Final gate:

```text
A3 ARCHITECTURE
= FREEZE

A3 CONTRACT HARDENING
= REQUIRED

R1-C1 CPU REFERENCE ADDRESSING
= READY AFTER / WITH THE REQUIRED A3 CONTRACT PATCH

GPU PRODUCTION EXECUTION
= STILL BLOCKED
```

---

# 18. Final Architecture in One Diagram

```text
                   ┌──────────────────────────────┐
                   │ RaggedAttentionSpec          │
                   │ B / Hkv / Hp / D / dtype    │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │ MemberPlacementMap           │
                   │ m=(layer,head)               │
                   │ → (cluster,column)           │
                   │ C derived, full bijection    │
                   └──────────────┬───────────────┘
                                  │
           model/rank static      │
──────────────────────────────────┼──────────────────────────────────
           request dynamic        │
                                  ▼
                   ┌──────────────────────────────┐
                   │ Scheduler canonical state    │
                   │ page_rows[C]                 │
                   │ effective_lens[C]            │
                   │ state_version                │
                   └──────────────┬───────────────┘
                                  │ Snapshot XOR Delta
                                  ▼
                   ┌──────────────────────────────┐
                   │ Worker discardable mirror    │
                   │ rows / counts / E / version │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │ CPU/GPU address materializer │
                   │ p = row[c][e//B]             │
                   │ vb = p*Hp + col              │
                   │ vs = vb*B + e%B              │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │ Ragged GPU backing           │
                   │ [2, P, Hp, B, D]             │
                   └──────────────┬───────────────┘
                                  │ view
                                  ▼
                   ┌──────────────────────────────┐
                   │ virtual single-head blocks   │
                   │ [2, P*Hp, B, 1, D]          │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │ existing FA interfaces       │
                   │ no kernel rewrite required   │
                   └──────────────────────────────┘
```

The architectural boundary to preserve is:

> **Scheduler owns physical pages. Worker derives physical addresses. FlashAttention consumes virtualized addresses.**

That is the contract that should now be frozen and used as the entry point for R1-C1.
