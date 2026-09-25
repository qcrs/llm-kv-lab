# R1-E Production Ragged Identity Architecture and Integration Plan

**Project:** Physical KV Cache Reclamation / Ragged KV Runtime for vLLM v0.26.0  
**Phase:** R1-E — Production Ragged Identity  
**Status:** CANONICAL DESIGN / DESIGN ONLY / NOT IMPLEMENTED  
**Implementation repository:** `qcrs/vllm`  
**Implementation branch:** `p1/v2-token-compaction-v026`  
**Reviewed D final HEAD:** `1ce56192c730ebb1859270d5b2c46253c2cf65fc`  
**D main implementation:** `219b4e449aed50d2df4532dc5d32ba56d07c7f32`  
**Pinned upstream baseline:** `568afb3a13806beb53bb2e6bd518269357b237c0`  
**Project repository:** `qcrs/llm-kv-lab@9d7bb91d97e5c63245c99893da001f57ae6cf985`  
**Tangram reference:** `aiha-lab/tangram@6fa551fc8f6edcc118a2a39b3554ee1520e91edd`

> This document is the architecture contract for a future
> `P1-V2-R1-E-PRODUCTION-RAGGED-IDENTITY-01` implementation slice.
> It does not authorize implementation, does not activate Ragged production
> execution, and does not change `PROJECT_STATE.md`.

---

# 0. Repository Audit and Input Authority

## 0.1 qcrs/vllm

**SOURCE FACT**

```text
branch:
p1/v2-token-compaction-v026

remote branch HEAD:
1ce56192c730ebb1859270d5b2c46253c2cf65fc

parent:
219b4e449aed50d2df4532dc5d32ba56d07c7f32
```

The branch therefore matches the requested D final closure exactly:

```text
219b4e449aed50d2df4532dc5d32ba56d07c7f32
= D main implementation

1ce56192c730ebb1859270d5b2c46253c2cf65fc
= D multi-layer closure test
```

No E implementation commit exists after this D closure on the reviewed branch.

GitHub exposes committed branch state, not the local worktree's uncommitted
`git status`. Therefore:

```text
remote committed status:
D-final branch identity verified

local worktree status:
NOT OBSERVABLE THROUGH GITHUB CONNECTOR
```

This is not a design blocker because this round does not modify the vLLM
worktree.

## 0.2 qcrs/llm-kv-lab

**SOURCE FACT**

```text
branch:
main

HEAD:
9d7bb91d97e5c63245c99893da001f57ae6cf985

remote committed status:
latest project-doc commit audited

local worktree status:
NOT OBSERVABLE THROUGH GITHUB CONNECTOR
```

The latest commit updates D/project documentation, but several runtime control
documents still describe D as pending Web review and retain older reviewed
HEADs.

## 0.3 Documentation state

**DOC_STATE_STALE**

The following source-of-truth mismatch exists:

- the actual implementation branch is at D final
  `1ce56192c730ebb1859270d5b2c46253c2cf65fc`;
- `44-R1-D-Ragged-GPU-Execution-Architecture-Freeze.md` still records an
  older reviewed implementation HEAD;
- `PROJECT_STATE.md`, `CURRENT-CONTEXT.md`, and `CHATGPT-HANDOFF.md`
  still describe D as `PASS_PENDING_WEB_REVIEW`.

Per the E design prompt, stale documentation does **not** override:

```text
qcrs/vllm actual branch
+
D evidence
+
this E design contract
```

No control-plane state is changed in this design round.

---

# 1. Executive Decision

R1-E is a production-wiring task, not another Ragged-math task.

The frozen A/B/C/D foundation is sufficient. The minimum viable production
architecture is:

```text
CacheConfig.page_group_size
        ↓
central Ragged Core activation validation
        ↓
Attention.get_kv_cache_spec()
FullAttentionSpec → RaggedAttentionSpec
        ↓
existing one-group / one-global-backing planner
        ↓
existing KVCacheCoordinator
        ↓
registered RaggedAttentionManager
        ↓
same coordinator-owned BlockPool
        ↓
Scheduler vector capacity path
        ↓
SchedulerOutput.ragged_kv_updates
        ↓
GPUModelRunner-owned Worker mirror
        ↓
reusable GPU staging + one global RaggedStepViews
        ↓
layer-local derived Ragged metadata
        ↓
existing ForwardContext dictionaries
        ↓
existing unified_kv_cache_update
        ↓
D ragged_kv_cache_update
        ↓
existing unified_attention_with_output
        ↓
D ragged_attention_forward
        ↓
successful execute_model tail: Worker frontier commit
        ↓
successful ModelRunnerOutput: Scheduler frontier commit
```

There is no second allocator, no Worker-side ownership authority, no second
persistent GPU ownership database, and no new custom attention op.

## Direct answers Q1–Q18

| Question | Decision |
|---|---|
| Q1. Minimum production activation boundary? | `page_group_size is not None`, guarded by one engine-wide Core validator plus one layer/backend validation boundary. Activation is atomic across Spec → Planner → Scheduler → Worker → metadata → write/read. |
| Q2. How does `RaggedAttentionManager` enter the allocator? | Register it as the manager for `RaggedAttentionSpec` inside the existing coordinator. The coordinator continues to create exactly one `BlockPool` and injects that pool into the Ragged manager. |
| Q3. How does normal append allocate? | Read canonical per-cluster `source_E`; for request query length `q`, compute `target_E[c]=source_E[c]+q`; run `plan_capacity(target_E) → apply_capacity_plan()`. Never derive Ragged capacity from `num_computed_tokens`, `max(E)`, or scalar logical length. |
| Q4. When does Scheduler commit E? | In `Scheduler.update_from_output`, only after a successful `ModelRunnerOutput` returns. `schedule()` reserves capacity only. |
| Q5. When does Worker commit E? | At successful `GPUModelRunner.execute_model` tail, after forward/sampling has completed and immediately before returning the successful result. A forward exception must not advance the mirror. |
| Q6. How do Snapshot/Delta enter ModelRunner lifecycle? | Reuse `SchedulerOutput.ragged_kv_updates`. Remove finished/preempted mirrors first, add/update normal request slots, then apply new-request Snapshots and running-request Allocation Deltas against the assigned `req_idx`. |
| Q7. Who owns `RaggedWorkerPhysicalState`? | `GPUModelRunner`. It owns request-index lifecycle and SchedulerOutput ingestion; `DefaultModelState` should not own transport/physical state. |
| Q8. Who owns placement GPU tensors? | `GPUModelRunner` as model-lifetime static topology. Construct/upload once and reuse. |
| Q9. Where is `RaggedStepViews` built? | Once per real step in `GPUModelRunner`, after active request order/query lengths are known and after Worker mirror updates are applied. |
| Q10. How does Ragged enter ForwardContext? | Reuse existing `attn_metadata[layer_name]` and `slot_mapping[layer_name]`; publish layer-local derived Ragged metadata. Do not add a new top-level ForwardContext field. |
| Q11. How do write/read reuse unified seams? | Existing `unified_kv_cache_update` and `unified_attention_with_output` dispatch on an explicit Ragged metadata type. Dense branches remain the existing calls. |
| Q12. Modify `FlashAttentionImpl`? | No. D already proves direct reuse of `reshape_and_cache_flash` and FA2 varlen entry points. |
| Q13. Modify `Attention.forward()`? | Its call/order structure should remain unchanged. The same file changes `get_kv_cache_spec` and the two existing unified-op bodies only. |
| Q14. Add a custom op? | No. Existing unified ops already provide the required graph/order seam. |
| Q15. Avoid parsing numeric layer names? | Use stable `kv_cache_group.layer_names` initialization order to create `layer_name → member slice`. Ragged correctness never calls `extract_layer_index(prefix)`. |
| Q16. Dummy/profile? | Profile-before-KV-init skips Ragged attention; any post-init dummy that executes attention uses synthetic dummy step metadata and never consumes/persists Worker request state. |
| Q17. How is real Engine identity proven? | E0–E7: startup, single prefill parity, B=16 crossing, continuous batching, mixed prefill/decode, chunked prefill if stable, finish/reuse, and free-page leak check against Dense. |
| Q18. What survives future compaction unchanged? | Activation, planner/global backing, vector normal allocation, Scheduler→Worker transport shape, Worker mirror ownership, GPU staging, StepViews production path, ForwardContext seam, write/read dispatch, and post-forward commit boundaries. Compaction only changes canonical Ragged state/reconciliation. |

---

# 2. Frozen Foundation — Do Not Reopen in E

## A — Representation

**FROZEN CONTRACT**

```text
Hkv = semantic KV heads
Hp  = physical page KV-head width
M   = L * Hkv
C   = M / Hp
```

`MemberPlacementMap` is the only semantic-to-physical topology authority:

```text
(layer, semantic KV head)
→
(cluster, column)
```

## B — Ownership

**FROZEN CONTRACT**

```text
Scheduler = canonical physical ownership authority
Worker    = discardable mirror
```

Canonical Scheduler state:

```text
state_version
effective_lens[C]
page_rows[C][variable depth]
```

Lifecycle:

```text
plan_capacity
→ apply_capacity_plan
→ GPU write/read
→ commit_effective_lens
```

Existing reconciliation/free/snapshot/allocation-delta contracts remain valid.

## C — Address / Storage

**FROZEN CONTRACT**

```text
physical: [P, Hp, B, 2D]
virtual:  [P*Hp, 1, B, 2D]

virtual_block = physical_page * Hp + column
```

The virtual view aliases physical storage; no copy is allowed.

## D — GPU Execution

**FROZEN CONTRACT**

Real CUDA has already demonstrated:

```text
RaggedStepViews
group physical slots
member virtual slots
reshape_and_cache_flash
FA2 decode
FA2 prefill
FA2 mixed
non-uniform E
custom placement
multi-layer layer_idx > 0
```

Execution layout remains:

```text
write:
token-major / head-minor

decode read:
request-major / head-minor

prefill/mixed:
head-major / request-minor

cache:
zero-copy virtual view
```

E only connects these proven components to real request lifecycle.

---

# 3. Production Chain Audit

| Chain stage | File / symbol | Current behavior | Required E behavior |
|---|---|---|---|
| Config | `vllm/config/cache.py::CacheConfig.page_group_size` | Field exists; doc says runtime activation deferred. | Remains the single activation knob. |
| CLI | `vllm/engine/arg_utils.py::EngineArgs`, cache construction | `page_group_size` is not an EngineArgs field and is not passed into `CacheConfig(...)`. | Add explicit experimental `--page-group-size` and pass it through. |
| Spec | `attention.py::Attention.get_kv_cache_spec` | Full decoder attention returns `FullAttentionSpec`. | Under validated activation, return `RaggedAttentionSpec`. |
| Planner | `kv_cache_utils.py::get_kv_cache_config_from_groups` | Ragged path already requires exactly one all-Ragged group and creates one `KVCacheTensor(shared_by=all layers)`. | Reuse unchanged. |
| Backing allocation | `attn_utils.py::_allocate_kv_cache` | One raw tensor object is installed for every `shared_by` layer. | Reuse; assert alias invariant in tests, not per step. |
| Backing reshape | `attn_utils.py::_reshape_kv_cache` | Ragged aliases each layer to `[P,Hp,B,2D]` on the same raw storage. | Reuse unchanged. |
| Binding | `worker/utils.py::bind_kv_cache` | Each layer alias is bound to `Attention.kv_cache`; `runner_kv_caches` contains multiple aliases. | Treat runner list as compatibility aliases, never as ownership multiplicity. |
| Manager selection | `single_type_kv_cache_manager.py::register_all_kvcache_specs` | Ragged manager is not registered. | Register `RaggedAttentionSpec → RaggedAttentionManager`. |
| Coordinator | `kv_cache_coordinator.py::KVCacheCoordinator` | Creates one BlockPool and injects it into all managers; generic APIs are scalar. | Preserve one pool; pass centralized placement to Ragged manager and expose a narrow Ragged manager/vector seam. |
| Ragged manager | `ragged_kv_cache_manager.py` | Vector state/plan/apply/commit/free/snapshot are implemented; scalar allocate methods intentionally throw. | Reuse vector contract; do not emulate scalar Dense allocation. |
| KV facade | `kv_cache_manager.py::allocate_slots` | Dense scalar token/block contract. | Add narrow Ragged vector allocation facade; Dense `allocate_slots` stays unchanged. |
| Scheduler | `scheduler.py::schedule` | Calls scalar `allocate_slots`; does not fill `ragged_kv_updates`. | When Ragged active, use vector capacity path and emit Snapshot/Delta. |
| Transport | `sched/output.py` | Snapshot/Delta/RaggedKVUpdate DTOs already exist. | Reuse exactly; no second lifecycle protocol. |
| Worker mirror | `worker/gpu/ragged_kv_state.py` | Complete discardable mirror implementation exists but has no production owner. | `GPUModelRunner` owns and wires it. |
| Step staging | `model_runner.py` | Dense BlockTables/slot mappings only. | Gather active Ragged mirror state into reusable GPU staging and build one global StepViews. |
| Metadata | `attn_utils.py`, `DefaultModelState.prepare_attn` | Dense metadata/slot mappings keyed by layer. | Produce layer-local Ragged metadata keyed by the same layer names. |
| ForwardContext | `forward_context.py` | Holds `attn_metadata` and `slot_mapping` dictionaries. | Reuse unchanged. |
| Write | `attention.py::unified_kv_cache_update` | Calls backend `do_kv_cache_update`. | Explicit Ragged metadata → D Ragged write; else byte/semantic-identical Dense call. |
| Read | `attention.py::unified_attention_with_output` | Calls `self.impl.forward`. | Explicit Ragged metadata → D Ragged read; else existing backend call. |
| Worker frontier | `GPUModelRunner.execute_model` | No Ragged mirror commit. | Commit at successful execute tail only. |
| Scheduler frontier | `Scheduler.update_from_output` | No Ragged canonical commit. | Commit canonical E only after successful worker output. |

---

# 4. Phase A — Production Activation

## 4.1 Activation condition

**E DESIGN DECISION**

The only public activation predicate is:

```python
cache_config.page_group_size is not None
```

No other hidden environment variable or shape sniffing is allowed.

Production activation remains OFF until every E wiring component is present.

## 4.2 Centralized validation boundaries

### Boundary 1 — engine-wide Ragged Core configuration

**E DESIGN DECISION**

Validate once after final vLLM config resolution:

```text
page_group_size > 0
TP = 1
PP = 1
DCP = 1
PCP = 1
prefix caching OFF
spec decode OFF
async scheduling OFF
KV connector/offload OFF
CUDA Graph OFF / eager
no MLA
no sliding-window / hybrid cache mode
KV quant mode = NONE
```

This is the only engine-wide unsupported-mode gate.

### Boundary 2 — attention-layer/backend compatibility

**E DESIGN DECISION**

At `Attention.get_kv_cache_spec()`, while the layer/backend facts are
available, validate only layer-local facts:

```text
decoder full causal attention
FlashAttention / FA2
head_size_v == head_size
Hkv % Hp == 0
no sliding_window
not MLA
KVQuantMode.NONE
```

Do not repeat these checks in Scheduler, Worker, metadata helpers, or D
execution helpers.

For the E acceptance run, require:

```text
Hp < Hkv
```

so “identity” proves real Ragged grouping rather than Dense-equivalent
`Hp == Hkv`.

## 4.3 CLI activation

### Option A — programmatic/internal only

Pros:

- smallest public API surface;
- zero CLI stability commitment.

Cons:

- real Engine reproduction requires custom Python configuration plumbing;
- less transparent acceptance;
- weakens the project demonstration because the production feature cannot be
  invoked through normal vLLM arguments.

### Option B — explicit experimental `--page-group-size`

Pros:

- reproducible real-Engine invocation;
- matches the already-existing CacheConfig concept;
- makes Dense/Ragged A/B tests straightforward;
- misuse is contained by centralized fail-fast validation.

Risks:

- exposes an experimental API before future prefix/TP/graph support;
- requires maintaining one EngineArgs pass-through.

### Decision

**E DESIGN DECISION — Option B**

Add an explicitly documented experimental `--page-group-size`. The source
audit shows CacheConfig fields are not automatically all surfaced: EngineArgs
manually declares cache arguments and manually constructs `CacheConfig`.
Therefore the existing field is not currently sufficient for CLI activation.

The flag is not a promise of broad compatibility; unsupported modes fail at
the central Core gate.

---

# 5. Phase B — Planner and Global Backing

## 5.1 Existing planner is already the correct E architecture

**SOURCE FACT**

For an all-Ragged configuration, `get_kv_cache_config_from_groups()` already:

1. requires exactly one Ragged KV cache group;
2. computes page count using Ragged page bytes;
3. creates one global `KVCacheTensor`;
4. sets `shared_by` to all Ragged layer names.

This is the desired production layout and should not be redesigned.

## 5.2 Raw backing ownership

**SOURCE FACT**

`_allocate_kv_cache()` allocates one raw tensor for the shared KVCacheTensor,
then inserts the same tensor object under every layer name.

`_reshape_kv_cache()` creates per-layer Ragged tensor views with the frozen
physical geometry while preserving shared storage.

`bind_kv_cache()` binds those aliases to each `Attention.kv_cache`.

Therefore:

```text
raw backing owner:
GPUModelRunner KV cache allocation lifetime

layer Attention.kv_cache:
alias only

runner self.kv_caches entries:
compatibility aliases only
```

## 5.3 Global backing alias invariant

**FROZEN CONTRACT FOR E**

```text
G1. Exactly one physical allocation backs the Ragged group.

G2. Every Ragged Attention.kv_cache aliases that same untyped storage.

G3. A physical page ID denotes one page in the global backing, not one page
    per layer alias.

G4. Allocation/free is performed once through BlockPool.

G5. Any bulk memory operation over layer aliases must either operate on the
    global backing once or deduplicate by storage identity.

G6. self.kv_caches length is not an ownership count.
```

Current `copy_kv_cache_blocks_inplace()` already deduplicates by storage
pointer, which is compatible with this invariant.

---

# 6. Phase C — Scheduler / Allocator Integration

This is the highest-risk E seam.

## 6.1 Why simply registering the Ragged manager is incorrect

**SOURCE FACT**

The existing coordinator's generic path calls:

```text
manager.get_num_blocks_to_allocate(...)
manager.allocate_new_blocks(...)
```

The Ragged manager intentionally raises:

```text
RuntimeError("Ragged manager requires vector capacity API")
```

because its state is:

```text
effective_lens[C]
page_rows[C][variable depth]
```

rather than one scalar Dense row.

## 6.2 Option A — integrate into existing coordinator

```text
KVCacheCoordinator
  └── one BlockPool
      └── RaggedAttentionManager

KVCacheManager
  └── narrow vector capacity facade

Scheduler
  └── Ragged branch calls vector facade
```

Advantages:

- preserves one BlockPool;
- preserves one manager hierarchy and existing free/preemption lifecycle;
- Ragged state stays next to the object that owns physical blocks;
- future compaction reconciliation already lands in the same manager state.

Risk:

- coordinator construction must provide `MemberPlacementMap`;
- generic scalar APIs must never be called when Ragged is active.

## 6.3 Option B — separate Scheduler-side Ragged manager

```text
Scheduler
  ├── Dense KVCacheManager / BlockPool
  └── separate Ragged manager / ownership
```

This creates or strongly encourages:

- duplicate BlockPool state;
- duplicate request ownership;
- split free/preemption paths;
- two sources of physical truth;
- future compaction reconciliation ambiguity.

### Decision

**E DESIGN DECISION — Option A**

Ragged becomes a first-class manager under the existing coordinator, sharing
the coordinator-created BlockPool.

No second Scheduler-side allocator is allowed.

## 6.4 Manager registration

**E DESIGN DECISION**

Register:

```text
RaggedAttentionSpec
→
RaggedAttentionManager
```

as its own KV-cache-spec family.

Because the manager constructor also needs placement, coordinator construction
must detect the Ragged group, obtain the centralized identity placement, and
pass `placement=` only to that manager.

Do not weaken `RaggedAttentionManager` by making placement optional.

## 6.5 Narrow KVCacheManager vector facade

**E DESIGN DECISION**

Do not overload Dense `allocate_slots()` with vector semantics.

Expose a narrow Ragged path conceptually equivalent to:

```text
get_ragged_state(request_id)
plan_ragged_capacity(request_id, target_E)
apply_ragged_capacity_plan(plan)
snapshot_ragged_state(request_id)
commit_ragged_effective_lens(...)
```

The facade owns admission checks against the same BlockPool/free-block count so
Scheduler does not reach around KVCacheManager.

The exact Python symbol names may be adjusted during implementation review;
the semantic boundary is frozen.

## 6.6 Normal append is vector-first forever

**FROZEN CONTRACT**

For request `r` and this-step query length `q_r`:

```text
source_E_r[c]
=
Scheduler Ragged canonical state effective_lens[c]

target_E_r[c]
=
source_E_r[c] + q_r
```

Then:

```text
plan_capacity(request_id, target_E_r)
→ admission check using plan.total_new_pages
→ apply_capacity_plan(plan)
```

Examples:

```text
today, identity:
source E = [32,32,32,32]
q = 1
target E = [33,33,33,33]

future, post-compaction:
source E = [17,32,9,48]
q = 1
target E = [18,33,10,49]
```

The same code path handles both.

The following are forbidden as Ragged allocation authority:

```text
request.num_computed_tokens
max(E)
logical sequence length
scalar effective_kv_len
```

Logical progress may continue to serve ordinary Scheduler token semantics; it
is not physical Ragged capacity authority.

---

# 7. Request Lifecycle and Transport

## 7.1 New request

```text
no canonical Ragged state
→ empty E = [0...0]
→ target E = source E + q
→ plan/apply capacity
→ canonical rows now exist, E remains source
→ emit full Snapshot
→ Worker installs snapshot into assigned req_idx
```

**E DESIGN DECISION**

New requests send **Snapshot**, not Delta.

Reason: Delta assumes an already-established source version, E vector, row
counts, and page ordering at Worker. A new request has no such Worker baseline.
Snapshot establishes the state atomically and prevents accidental req_idx stale
state from becoming the base.

## 7.2 Running request

```text
existing canonical source E
→ target E = source E + q
→ plan/apply
```

If pages are appended:

```text
emit RaggedPageAllocationDeltaData
```

If no pages are appended:

```text
ragged transport = None for this request
```

No transport is required merely to advance E. The Worker already owns the
source mirror, gets `q` from the normal SchedulerOutput/input-batch lifecycle,
and commits its mirror only after successful execution.

## 7.3 Preempted / resumed request

**SOURCE FACT**

GPUModelRunner already treats `preempted_req_ids` as request removals along
with finished requests.

**E DESIGN DECISION**

Ragged follows the same lifecycle:

```text
preemption:
Scheduler KV free
→ RaggedAttentionManager.free_request()
→ physical pages return to the one BlockPool
→ SchedulerOutput.preempted_req_ids
→ Worker remove_request(req_idx)

resume:
new canonical Ragged state
→ fresh capacity allocation
→ Snapshot
```

A resumed request must never rely on the old Worker mirror.

## 7.4 Finished request

```text
Scheduler:
existing request-finish/free path
→ Ragged manager free_request()
→ BlockPool free

Worker:
finished_req_ids
→ RaggedWorkerPhysicalState.remove_request(req_idx)
→ ordinary req_idx release
```

The mirror must be cleared before the req_idx becomes available for a new
request.

## 7.5 SchedulerOutput ordering

**E DESIGN DECISION**

Do not create a parallel request protocol.

Use the existing SchedulerOutput lifecycle:

```text
1. determine finished/preempted IDs
2. schedule requests and reserve Ragged capacity
3. build scheduled_new_reqs / scheduled_cached_reqs
4. attach ragged_kv_updates:
     new/resumed → Snapshot
     running + new pages → Delta
     running + no pages → none
5. Worker removes finished/preempted state
6. Worker creates/updates ordinary request state
7. Worker applies Ragged Snapshot/Delta to assigned req_idx
8. build execution metadata
```

---

# 8. Worker Mirror Ownership and Lifetimes

## 8.1 Worker owner alternatives

### Option A — GPUModelRunner

Pros:

- already owns SchedulerOutput ingestion;
- owns req_id↔req_idx lifecycle;
- owns add/update/finish/preemption sequence;
- owns KV cache and per-step GPU input preparation;
- can clear mirror before req_idx reuse.

### Option B — DefaultModelState

Cons:

- model-specific state object is downstream of request transport;
- does not own the physical allocation protocol;
- would couple generic request ownership to model-specific attention prep.

### Decision

**E DESIGN DECISION — GPUModelRunner**

`RaggedWorkerPhysicalState` is constructed and owned by GPUModelRunner when
Ragged Core is active.

It remains discardable: no Worker operation can allocate/free physical pages.

## 8.2 Ownership / Lifetime Matrix

| Object | Authority | Owner | Device | Lifetime | Mutable | Source of truth |
|---|---|---|---|---|---|---|
| `MemberPlacementMap` | static topology contract | Ragged runtime initialization | CPU | model | no | yes for semantic→physical topology |
| `layer_name → member slice` | derived from placement + group order | GPUModelRunner | CPU | model | no | derived |
| `member_to_cluster` | placement | GPUModelRunner | CUDA | model | no | derived/cached |
| `member_to_column` | placement | GPUModelRunner | CUDA | model | no | derived/cached |
| Scheduler `RaggedRequestPhysicalState` | Scheduler | RaggedAttentionManager | CPU | request | yes | **yes** for physical ownership/frontier |
| `RaggedWorkerPhysicalState` | Scheduler mirror | GPUModelRunner | CPU/NumPy | request slot | yes | no |
| reusable cluster-row staging | Worker mirror | GPUModelRunner | CUDA | model/runner | yes active slice | no |
| reusable effective-E staging | Worker mirror | GPUModelRunner | CUDA | model/runner | yes active slice | no |
| global `RaggedStepViews` | derived | GPUModelRunner | CUDA | step | no after build | no |
| layer-local Ragged metadata | derived | GPUModelRunner / attn metadata builder helper | CUDA/Python refs | step | no | no |
| physical KV backing | BlockPool IDs + Worker storage | GPUModelRunner allocation | CUDA | engine | data mutable | storage only; ownership is Scheduler |
| `Attention.kv_cache` | alias | Attention layer | CUDA | model | data mutable | no |

---

# 9. Static Placement Initialization

## 9.1 Do not parse layer names

**E DESIGN DECISION**

Ragged correctness must not depend on a layer name looking like:

```text
model.layers.17....
```

Use:

```text
kv_cache_group.layer_names
+
stable initialization order
```

to establish:

```text
layer_name
→ semantic layer ordinal
→ member slice [ordinal*Hkv : (ordinal+1)*Hkv]
```

The formula is centralized in the placement initializer only.

Existing generic `bind_kv_cache()` may still use `extract_layer_index` for
its compatibility `runner_kv_caches` ordering; E Ragged correctness must not
consume that ordering as semantic identity.

## 9.2 Identity today, custom placement tomorrow

**E DESIGN DECISION**

E Core uses only identity placement, constructed by one shared initializer.

Do not scatter:

```python
cluster = layer * ...
```

through Scheduler/Worker/Attention.

Future custom placement changes only the initialization source that produces
`MemberPlacementMap`; Scheduler vector ownership, Worker mirror, staging,
StepViews, and Attention dispatch remain the same.

**DEFERRED**

If custom placement becomes externally configurable, its serialized topology
must be made an engine-initialization input so Scheduler and Worker consume the
same topology. E identity does not need that transport yet.

---

# 10. GPU Staging Strategy

## Option A — allocate tensors every step

```text
NumPy gather
→ torch.tensor(...)
→ fresh CUDA allocations
```

Pros: minimal code.  
Cons: allocator overhead, pointer churn, poor future CUDA Graph seam, needless
Python/CUDA object creation.

## Option B — reusable ModelRunner staging

```text
Worker mirror
→ reusable host/NumPy active view
→ copy active slice into preallocated CUDA buffers
→ derive StepViews
```

Pros:

- no persistent second ownership database;
- stable buffer addresses;
- lower per-step allocation pressure;
- compatible with eventual graph metadata bufferization;
- clean separation of request state and step state.

Cost: modest fixed staging memory and more initialization code.

## Option C — force Ragged state into Dense BlockTables/InputBatch storage

Rejected because Dense row geometry is not the canonical Ragged state and would
reintroduce scalar assumptions.

### Decision

**E DESIGN DECISION — Option B**

GPUModelRunner owns reusable staging buffers for at least:

```text
active cluster rows
active source effective_lens
```

Only the active request slice is copied each step.

D-derived member tables/slots/lens may remain step-lifetime derived tensors in
R1-E eager mode.

**DEFERRED**

Full pointer-stable member metadata for CUDA Graph is a later optimization; E
must not pre-build a graph transaction system.

---

# 11. Global StepViews vs Layer-local Execution Metadata

## Option 1 — global StepViews + numeric layer index in Attention

Pros:

- matches D helper signature directly;
- one Python object.

Cons:

- Attention must know a global numeric layer ordinal;
- encourages name parsing;
- semantic slicing logic leaks into execution layer;
- less consistent with existing `dict[layer_name → metadata]`.

## Option 2 — build global once, publish layer-local derived metadata

```text
one global RaggedStepViews
        ↓
for layer_name in kv_cache_group.layer_names:
    layer member slots
    layer member block table
    layer member seq lens
        ↓
attn_metadata[layer_name]
slot_mapping[layer_name]
```

Tensor slicing can be views; it does not require copying global page state.

### Decision

**E DESIGN DECISION — Option 2**

Build the expensive/global transformation once, then expose a small immutable
layer-local Ragged metadata object.

Conceptually:

```text
RaggedLayerStepMetadata
  member_block_table   [R,Hkv,N]
  member_seq_lens      [R,Hkv]
  member_slot_mapping  [Q,Hkv]
  query_start_loc
  num_actual_tokens
  max_query_len
  max_kv_len
  page_group_size
  block_size
```

This is a projection of global StepViews, not canonical state.

D math is not redesigned; E adds a thin layer-local adapter over the already
proven transformations.

---

# 12. Existing ForwardContext Seam

## Option A — reuse current dictionaries

```text
attn_metadata[layer_name]
slot_mapping[layer_name]
```

## Option B — add `ragged_step_views` / ownership data to ForwardContext

This would make ForwardContext a second large runtime-state container and force
Attention to recover layer slicing itself.

### Decision

**E DESIGN DECISION — Option A**

No new top-level ForwardContext field.

Dense:

```text
attn_metadata[layer_name] = existing backend metadata
slot_mapping[layer_name]  = existing Dense slot mapping
```

Ragged:

```text
attn_metadata[layer_name] = explicit RaggedLayerStepMetadata
slot_mapping[layer_name]  = layer member virtual slots
```

The explicit metadata class is the runtime discriminator.

No shape sniffing is allowed.

---

# 13. Write / Read Dispatch

## 13.1 Write

Current:

```text
unified_kv_cache_update
→ attn_layer.impl.do_kv_cache_update
```

E:

```text
if isinstance(attn_metadata, RaggedLayerStepMetadata):
    ragged_kv_cache_update(...)
else:
    existing Dense do_kv_cache_update(...)
```

The exact helper can be a thin layer-local wrapper over the D implementation.

## 13.2 Read

Current:

```text
unified_attention_with_output
→ self.impl.forward(...)
```

E:

```text
if isinstance(attn_metadata, RaggedLayerStepMetadata):
    ragged_attention_forward(...)
else:
    self.impl.forward(...)
```

## 13.3 Ordering

**FROZEN CONTRACT**

Preserve:

```text
unified_kv_cache_update
→ kv_cache_dummy_dep
→ unified_attention_with_output
```

This existing dependency is sufficient.

## 13.4 What is explicitly not changed

```text
FlashAttentionImpl              NO CHANGE
Attention.forward call order    NO CHANGE
new Ragged custom op            NOT ADDED
Dense backend dispatch          UNCHANGED
```

Tangram's `unified_attention_ragged` custom op is therefore not ported.

---

# 14. Post-forward Effective Frontier Commit

Capacity and visibility are separate transactions.

## 14.1 Scheduler schedule phase

```text
source E
→ reserve pages for target E
→ ownership may grow
→ E MUST remain source E
```

The Scheduler may continue its ordinary logical
`request.num_computed_tokens` accounting; that scalar is not Ragged physical
visibility.

## 14.2 Worker commit

### Alternatives

A. Commit immediately after KV write.  
B. Commit after attention forward.  
C. Commit at successful `execute_model` tail.

### Decision

**E DESIGN DECISION — C**

Commit the Worker mirror only after the entire Worker step has completed
successfully, immediately before returning its successful output.

Reasons:

- if forward raises, E stays unchanged;
- if later Worker-side sampling/postprocessing raises, mirror also stays
  unchanged;
- the mirror is not needed for the current attention because StepViews already
  carries post-write `source_E + q` visibility;
- the next scheduling step cannot begin in synchronous Core before the current
  output returns.

## 14.3 Scheduler commit

### Option A — infer success from normal ModelRunnerOutput

```text
schedule reserves
→ worker completes
→ update_from_output receives normal output
→ commit source_E + q
```

### Option B — Worker sends `RaggedStepCommitData`

This duplicates information already available in the synchronous step and
creates a protocol that is only necessary for future multiple-in-flight
execution.

### Decision

**E DESIGN DECISION — Option A**

No new ACK/commit DTO in R1-E.

In `Scheduler.update_from_output`, for every successfully scheduled Ragged
request:

```text
source_E = current Ragged canonical effective_lens
q        = scheduler_output.num_scheduled_tokens[request_id]
target_E = source_E + q
commit_effective_lens(
    expected_state_version=current ownership version,
    expected_source_effective_lens=source_E,
    new_effective_lens=target_E,
)
```

Because R1 Core disables async scheduling and allows one in-flight batch,
`SchedulerOutput + ModelRunnerOutput` is an adequate completion boundary.

## 14.4 Future async seam

**DEFERRED**

Multiple in-flight batches may require:

```text
step_seq
generation/version fence
per-step pending frontier
ordered commit/rollback
```

The current DTOs already reserve optional `step_seq` fields in related
contracts, but E must not implement an async transaction manager now.

---

# 15. Capacity vs Visibility — Canonical State Transition Table

Assume request source `E=S`, this-step query length `q`, target `T=S+q`.

| Moment | Scheduler E | Scheduler pages | Worker E | Worker rows | GPU cache |
|---|---|---|---|---|---|
| Before scheduling | S | source capacity | S | source rows | history through S valid |
| After capacity reserve | **S** | capacity ≥ T | S | source rows | new pages may contain stale bytes but are invisible |
| After Worker applies Snapshot/Delta | S | capacity ≥ T | **S** | mirrors reserved rows | unchanged visibility |
| Forward entry / StepViews built | S | capacity ≥ T | S | reserved rows | derived write slots target positions; read lens will be T after write |
| After KV write | S | capacity ≥ T | S | reserved rows | new K/V physically written; canonical visibility still S |
| After attention | S | capacity ≥ T | S | reserved rows | attention has consumed visibility T for this step |
| After Worker commit | S | capacity ≥ T | **T** | reserved rows | Worker ready for next step |
| After Scheduler update_from_output | **T** | capacity ≥ T | T | reserved rows | canonical parity restored |

**FROZEN CONTRACT**

Owned capacity may lead effective visibility. Effective visibility may never
lead owned capacity.

---

# 16. New-page Zeroing

## 16.1 Source audit

**SOURCE FACT**

`KVBlockZeroer` currently builds zeroing metadata only for
`FullAttentionSpec`; Ragged is skipped.

The Ragged backing is shared across all layer aliases.

## 16.2 Correctness analysis

Under R1-E Core:

```text
prefix cache OFF
KVQuantMode.NONE
no hybrid attention
no connector
FA2 seqused_k/member seq lens bounds visibility
every newly visible current-step slot is written before read
```

Therefore stale bytes beyond the effective frontier are not observable by the
attention kernel.

### Decision

**E DESIGN DECISION**

Ragged new physical pages do **not** require zeroing for R1-E correctness.

Do not route global Ragged page IDs through Dense `KVBlockZeroer` merely for
symmetry.

This also avoids the risk of zeroing the same shared page once per layer alias.

**DEFERRED**

A future debug-determinism or kernel-hardening mode may add Ragged-aware
zeroing. If added, it must operate once per global backing/page ID and
deduplicate aliases by storage identity.

---

# 17. Dummy / Profile / Warmup Semantics

A real Engine must start before it can serve a request.

## 17.1 Required distinction

### Profile before KV backing is available

Do not require Scheduler-owned Ragged request state.

Use the existing ability to skip attention for the memory-profile path when
the backing is not initialized.

### Post-init dummy attention

If a warmup/dummy path executes attention, construct a synthetic
dummy-step metadata object from dummy input sizes and valid in-range physical
pages.

Properties:

```text
not inserted into Scheduler Ragged state
not inserted into RaggedWorkerPhysicalState
not committed
not transported
lifetime = dummy call only
```

### CUDA graph

R1-E activation requires eager / CUDA Graph OFF, so graph capture is not an E
acceptance requirement.

**DESIGN RISK**

A startup path that silently forces ordinary Dense dummy block tables into a
Ragged attention call is invalid and must be caught by E0 startup tests.

---

# 18. Prefix Cache / Connector / CoW

**E DESIGN DECISION**

Central fail-fast for R1-E:

```text
prefix cache OFF
KV connector/offload OFF
CoW/prefix partial-hit path unsupported
```

Do not add compatibility code in E.

Future seams:

- prefix reuse must teach prefix ownership/hash semantics about clustered pages;
- connectors must serialize/transfer clustered physical page state;
- CoW must copy one global physical page exactly once.

These are adapters to the ownership model, not reasons to duplicate ownership
now.

---

# 19. Continuous Batching and Chunked Prefill

**FROZEN E REQUIREMENT**

The wiring must support variable per-request `q`.

For active request `r`:

```text
target_E_r = source_E_r + q_r
```

No E code may assume:

```text
q == 1
all q are equal
only one request
decode-only
```

D already proves the mixed mathematical layout; E must preserve active-request
order from InputBatch through Ragged gather and layer-local metadata.

---

# 20. Four Critical Sequence Diagrams

## 20.1 New request prefill

```text
Scheduler
  | q = prompt chunk
  | source_E = 0
  v
KVCacheManager / RaggedAttentionManager
  | plan_capacity(target_E=q)
  v
BlockPool
  | allocate physical pages once
  v
Scheduler Ragged state
  | rows installed; E still 0
  | snapshot()
  v
SchedulerOutput.ragged_kv_updates
  | Snapshot
  v
GPUModelRunner
  | assign req_idx
  | Worker.apply_snapshot()
  | gather active request
  | stage rows/E to CUDA
  v
RaggedStepViews
  | derive write slots and post-write member lens
  v
ForwardContext[layer_name]
  | layer-local Ragged metadata
  v
Attention
  | unified_kv_cache_update
  |   → D Ragged write
  | unified_attention_with_output
  |   → D FA2 Ragged read
  v
GPUModelRunner execute tail
  | Worker E: 0 → q
  v
ModelRunnerOutput
  v
Scheduler.update_from_output
  | canonical E: 0 → q
```

## 20.2 Running decode

```text
Scheduler canonical E = [e0,e1,...]
  | q=1
  | target_E = E+1
  v
plan/apply capacity
  | if boundary crossed → allocation delta
  | else → no Ragged transport
  v
Worker existing mirror
  | optional apply delta
  | gather source E
  v
StepViews
  | exact per-cluster physical slots
  v
write → read
  v
Worker tail commit E+1
  v
Scheduler output commit E+1
```

## 20.3 Mixed batch

```text
Request A: decode qA=1
source_E_A = [17,32,9,48]

Request B: prefill qB=7
source_E_B = [0,0,0,0]

Scheduler:
  target_A = [18,33,10,49]
  target_B = [7,7,7,7]
  reserve independently from each vector

Worker active order:
  [A, B]

query_start_loc:
  [0, 1, 8]

global StepViews:
  preserves each request's own source E and q
  → member slots
  → member seq lens

per-layer metadata:
  A decode + B prefill coexist in one FA2 mixed path

successful tail:
  Worker commits A target and B target

Scheduler update:
  canonical A/B targets committed
```

## 20.4 Finish / free / reuse

```text
Request A finishes
  ↓
Scheduler normal finish path
  ↓
RaggedAttentionManager.free_request(A)
  ↓
same BlockPool.free_blocks(A pages)
  ↓
SchedulerOutput.finished_req_ids={A}
  ↓
GPUModelRunner
  ↓
RaggedWorkerPhysicalState.remove_request(A req_idx)
  ↓
ordinary req_idx becomes reusable

Request B later arrives
  ↓
plan_capacity(B)
  ↓
same BlockPool may return physical page IDs formerly owned by A
  ↓
Snapshot establishes B's new ownership
  ↓
new writes occur before B exposes those positions
```

---

# 21. Architecture Alternative Decisions

| Concern | Option A | Option B | Decision | Rejected reason |
|---|---|---|---|---|
| Ragged allocator | Existing coordinator + same BlockPool + vector facade | Separate Scheduler Ragged manager | **A** | B creates split-brain ownership/lifecycle. |
| Worker mirror owner | GPUModelRunner | DefaultModelState | **GPUModelRunner** | ModelState does not own SchedulerOutput/req_idx physical lifecycle. |
| Placement GPU owner | GPUModelRunner model-lifetime tensors | FA metadata builder owns global placement | **GPUModelRunner** | Placement is global cross-layer topology, not FA-builder-local policy. |
| Step staging | fresh CUDA tensors each step | reusable runner buffers | **reusable buffers** | fresh allocations create pointer churn and future graph dead-end. |
| Execution metadata | global views + numeric layer idx in Attention | global build + layer-local projections | **layer-local projections** | avoids name parsing and matches source-native layer dictionaries. |
| ForwardContext | reuse current metadata/slot dictionaries | add global Ragged field | **reuse current** | avoids a second runtime-state container. |
| Scheduler commit | normal ModelRunnerOutput completion | explicit Ragged ACK DTO | **normal completion** | synchronous E has one in-flight batch; ACK is redundant. |
| Activation | internal programmatic only | explicit experimental CLI | **explicit CLI** | internal-only weakens reproducibility; central gates control misuse. |

---

# 22. Tangram Comparison Matrix

| Concern | Tangram approach | qcrs current foundation | Reuse / Adapt / Reject | Reason |
|---|---|---|---|---|
| `page_group_size` activation | Attention-layer Ragged branch | CacheConfig field exists but production activation deferred | **Adapt** | qcrs needs atomic Core gates and normal EngineArgs pass-through. |
| RaggedBlockTable | dedicated Worker-side Ragged table abstraction | Scheduler canonical Ragged page rows + discardable Worker mirror already frozen | **Reject ownership model** | qcrs must not create Worker allocation authority. |
| group slot mapping | group-aware physical→member slots | D `group_physical_slots` + member virtual slots proven | **Reuse concept; keep qcrs D** | D is already source-backed and CUDA-validated. |
| member map lifetime | cached static member maps in metadata builder | D helper supports cached placement tensors | **Reuse principle** | place global tensors on GPUModelRunner instead of copying exact owner. |
| metadata builder | Ragged data embedded in FA metadata, plus per-layer overlays | qcrs has existing layer-keyed ForwardContext seam | **Adapt** | publish explicit qcrs layer-local Ragged metadata. |
| ForwardContext | Tangram-specific integration | qcrs already has attn_metadata/slot_mapping dicts | **Reuse qcrs seam** | no new top-level state needed. |
| attention custom op | `unified_attention_ragged` | qcrs already has unified write/read ops | **Reject** | extra op duplicates an existing ordering/dispatch seam. |
| write/read coupling | Ragged eager op handles specialized execution | qcrs D has separate proven write/read helpers plus dummy dep | **Reuse qcrs** | preserves upstream-like write-before-read structure. |
| layer index | Tangram stores/derives numeric layer index | qcrs can use `kv_cache_group.layer_names` order | **Reject parsing** | model-name syntax must not be a correctness assumption. |
| physical layout | Tangram column-major rank/layout differs | qcrs frozen `[P,Hp,B,2D]` + virtual alias | **Reject port** | C/D storage contract is already closed. |
| physical ownership | Tangram architecture is optimized around its compression runtime | qcrs Scheduler authority + manager state frozen | **Reject port** | ownership is a project invariant. |
| compaction lifecycle | Tangram integrates its own compressor/scorer lifecycle | qcrs has `reconcile_compaction` and source-E vector contract | **Adapt ideas only** | E identity must remain compaction-agnostic. |

---

# 23. File / Symbol Impact Map

| File | Symbol | Current role | E change | Why | Risk | Future dependency |
|---|---|---|---|---|---|---|
| `vllm/config/cache.py` | `CacheConfig.page_group_size` | existing deferred knob | **NO CHANGE** | field already expresses geometry | low | future support matrix |
| `vllm/engine/arg_utils.py` | `EngineArgs`, `CacheConfig(...)` construction | public CLI/config bridge | MODIFY | expose experimental flag and pass-through | low | API stability |
| `vllm/config/vllm.py` | config validation | engine-wide resolved config | MODIFY | central Core fail-fast | medium | TP/async/prefix future relaxations |
| `vllm/model_executor/layers/attention/attention.py` | `get_kv_cache_spec` | selects Dense spec | MODIFY | switch to Ragged under activation | high | future backend support |
| same | `unified_kv_cache_update` | unified write seam | MODIFY narrow branch | invoke D Ragged write | high | quantized Ragged later |
| same | `unified_attention_with_output` | unified read seam | MODIFY narrow branch | invoke D Ragged read | high | FA3/other backends later |
| same | `Attention.forward` | write/read ordering | **NO SEMANTIC CHANGE** | existing seam is sufficient | critical regression if touched | compile path |
| `vllm/v1/kv_cache_interface.py` | `RaggedAttentionSpec` | frozen geometry | **NO CHANGE expected** | already sufficient | low | custom topology remains external |
| `vllm/v1/core/kv_cache_utils.py` | planner/config creation | one Ragged global backing | **NO CHANGE expected** | desired architecture exists | medium regression test | future mixed groups |
| `vllm/v1/core/single_type_kv_cache_manager.py` | registry | built-in manager registration | MODIFY | register Ragged manager | medium/circular import | future generic spec registry |
| `vllm/v1/core/kv_cache_coordinator.py` | manager construction / one BlockPool | coordinator authority | MODIFY narrow Ragged construction/accessor | pass placement while preserving one pool | high | custom placement |
| `vllm/v1/core/kv_cache_manager.py` | allocation facade | Dense scalar API | MODIFY | add narrow vector Ragged capacity API | **critical** | future compaction normal append |
| `vllm/v1/core/ragged_kv_cache_manager.py` | plan/apply/commit/snapshot/free | canonical Ragged state | MINIMAL / ideally NO CORE LOGIC CHANGE | existing contract already correct | medium | non-uniform compaction |
| `vllm/v1/core/sched/output.py` | Ragged Snapshot/Delta/Update | transport DTOs | **NO CHANGE** | sufficient protocol exists | low | async may use step_seq later |
| `vllm/v1/core/sched/scheduler.py` | `schedule` | admission/allocation | MODIFY | vector allocation + transport | **critical** | async/prefix |
| same | `update_from_output` | successful completion boundary | MODIFY | canonical E commit | **critical** | future async fence |
| `vllm/v1/worker/gpu/ragged_kv_state.py` | Worker mirror | complete mirror primitive | **NO CHANGE expected** | methods already sufficient | medium tests | optional staging helpers |
| `vllm/v1/worker/gpu/model_runner.py` | init/request lifecycle/execute | Worker runtime owner | MODIFY | mirror, placement tensors, staging, updates, StepViews, Worker commit, dummy | **critical** | graph/TP |
| `vllm/v1/worker/gpu/model_states/default.py` | `prepare_attn` | Dense model-specific metadata | **NO CHANGE preferred** | keep Ragged production glue in runner/attn_utils | low | alternate models |
| `vllm/v1/worker/gpu/attn_utils.py` | metadata helpers / KV init | cache materialization + metadata | MODIFY narrow helpers | layer-local Ragged metadata/slot map; retain existing backing path | high | other Ragged backends |
| `vllm/forward_context.py` | `ForwardContext` | per-forward metadata container | **NO CHANGE** | reuse existing dictionaries | low | none |
| `vllm/v1/attention/backends/ragged_layout.py` | StepViews | D derived metadata | MODIFY additive | add layer-local projection type/helper | medium | custom placement |
| `vllm/v1/attention/backends/ragged_forward.py` | D write/read helpers | proven CUDA execution | MODIFY thin adapter only | consume layer-local projection without parsing layer idx | high | non-uniform stays same |
| `vllm/v1/attention/backends/flash_attn.py` | Dense FA implementation | standard backend | **NO CHANGE** | E dispatch happens before backend Dense call | critical to preserve | future native integration |
| `vllm/v1/worker/utils.py` | `bind_kv_cache`, copy/zero utils | alias binding/utilities | **NO CHANGE expected** | E must not depend on alias list ordering | medium | future generic cleanup |
| `vllm/v1/outputs.py` | `ModelRunnerOutput` | worker→scheduler success boundary | **NO CHANGE** | no ACK DTO needed | low | async may extend protocol |

---

# 24. Recommended Internal Implementation Order

This is one future implementation slice, not nine separately authorized slices.

| Step | Work | Main files | Dependency | Local acceptance | Rollback boundary |
|---|---|---|---|---|---|
| 1 | Add central activation validation + CLI plumbing, but keep actual production spec switch disabled until final integration | arg_utils.py, vllm.py | none | unsupported matrix fails once, Dense untouched | remove plumbing |
| 2 | Register Ragged manager and integrate placement-aware coordinator construction/vector facade | single_type manager, coordinator, kv_cache_manager | A/B foundation | one BlockPool identity; vector allocation unit tests | registry/facade revert |
| 3 | Scheduler vector allocation and Snapshot/Delta production | scheduler.py | step 2 | new/running/no-growth/preempt tests | branch remains activation-off |
| 4 | GPUModelRunner mirror lifecycle | model_runner.py | step 3 DTOs | new/delta/remove/req_idx reuse tests | no attention dispatch yet |
| 5 | static placement tensors + reusable staging + global StepViews | model_runner.py, ragged_layout.py | step 4 | mixed q/non-uniform E metadata tests | no production read/write yet |
| 6 | layer-local metadata + existing ForwardContext dictionaries | attn_utils.py, ragged_layout.py | step 5 | layer-name/order tests; no numeric parsing | Dense builder unchanged |
| 7 | existing unified write/read dispatch | attention.py, ragged_forward.py | step 6 | D focused CUDA reused through production seam | Dense branches exact |
| 8 | Worker/Scheduler post-success commits + dummy/profile semantics | model_runner.py, scheduler.py | step 7 | failure does not commit; startup dummy tests | activation still switchable |
| 9 | atomically enable `get_kv_cache_spec` switch and run real Engine identity acceptance | attention.py + tests/scripts | all above | E0–E7 | `page_group_size=None` remains Dense escape hatch |

**E DESIGN DECISION**

The actual production switch should land only after the internal pieces are
wired, so intermediate commits cannot leave:

```text
Ragged spec + Dense scheduler
```

or:

```text
Ragged scheduler + Dense attention
```

as a usable configuration.

---

# 25. E Identity Acceptance Matrix

Target environment:

```text
model: /data/models/Qwen3-0.6B
GPU: A100 80GB
backend: FA2
dtype: BF16
block_size: 16
eager: true
TP=1 PP=1 DCP=1 PCP=1
prefix cache: OFF
spec decode: OFF
async scheduling: OFF
KV connector/offload: OFF
CUDA Graph: OFF
```

Choose a model-valid `Hp` satisfying:

```text
0 < Hp < Hkv
Hkv % Hp == 0
```

so the run genuinely exercises Ragged physical grouping.

| Gate | Scenario | Oracle | Expected | Evidence |
|---|---|---|---|---|
| C0 | `page_group_size=None` | baseline Dense tests | unchanged behavior | Dense focused regression |
| C1 | unsupported Ragged config | central validator | fail-fast before runtime allocation | validation tests |
| E0 | Engine startup | no exception + inspected KV config | one Ragged group, one backing, all aliases same storage | startup log + source assertions |
| E1 | single prefill | Dense vs Ragged same prompt, deterministic decode | logits close within BF16 tolerance; output tokens identical | captured logits/tokens |
| E2 | block crossing | prompt/decode crosses token 16 boundary | correct page growth/slots and Dense parity | manager + Engine trace |
| E3 | continuous batching | two requests with different arrival/progress | both advance correct per-request E; Dense parity | scheduler/worker trace |
| E4 | mixed prefill + decode | one decode request + one prefill request same step | variable q path correct; no shared-q assumption | step metadata + outputs |
| E5 | chunked prefill | force stable chunking config | repeated chunks advance vector E correctly | trace; may be marked blocked only if baseline cannot deterministically trigger |
| E6 | finish/reuse | finish A, then admit B | A mirror removed; Scheduler pages freed; B can reuse physical IDs safely | page-ID/refcount evidence |
| E7 | leak check | all requests complete | free-page count returns to expected baseline | BlockPool counters |
| W1 | failure before successful return | injected unit-level forward exception | neither Worker nor Scheduler frontier committed | targeted failure test |
| M1 | shared backing alias | all Ragged layer kv_cache views | same storage pointer, no per-layer ownership duplication | initialization test |
| M2 | placement lifetime | multiple steps | same cached placement GPU tensor pointers | instrumentation/unit test |
| M3 | non-uniform synthetic source | worker/manager focused test | normal append uses source vector without scalar collapse | targeted unit test |

“Engine can generate text” alone is not sufficient evidence.

---

# 26. Future-readiness Audit

| Future capability | E interface status | Future work |
|---|---|---|
| real non-uniform compaction | **directly compatible** | update canonical Ragged state/reconciliation only; next normal append already consumes vector E |
| physical page free/reuse | **directly compatible** | existing reconcile/free + same BlockPool |
| automatic scoring | **compatible via adapter** | scorer changes compaction policy, not normal execution wiring |
| custom placement | **compatible if initializer seam preserved** | replace identity placement source; serialize topology if externally configured |
| TP2 | **needs adapter/refactor** | shard semantic members/physical pages and define cross-rank placement; Core currently fail-fast |
| CUDA Graph | **needs metadata-buffer work** | reusable staging helps, but step-derived member tensors need stable bufferization/capture contract |
| async scheduling | **needs protocol adapter** | step_seq/generation fence and pending frontiers |
| prefix reuse | **substantial adapter** | clustered hash/reuse/CoW ownership semantics |
| LMCache/offload | **substantial adapter** | connector must understand global clustered pages and topology |
| FA3/other backend | **backend adapter** | D currently proves FA2 only |

Future-ready here means correct ownership/interfaces, not premature feature
implementation.

---

# 27. Core Invariants

```text
I1. Scheduler is the sole allocation/free authority.

I2. Worker Ragged physical state is a discardable mirror.

I3. Physical ownership capacity and effective visibility are separate.

I4. Normal append always starts from per-cluster canonical source E.

I5. Spec, planner, allocator, and runtime use the same Hp geometry.

I6. One global physical page namespace is preserved.

I7. MemberPlacementMap is the only semantic→physical topology authority.

I8. Placement GPU tensors are model-lifetime cached and never rebuilt per step.

I9. RaggedStepViews and layer-local metadata are derived step state, never
    canonical ownership.

I10. page_group_size=None preserves the existing Dense path.

I11. Existing write-before-read dependency is preserved.

I12. Future non-uniform E uses the same normal append/execution path.

I13. A new/resumed Worker request is established by Snapshot, never by a Delta
     against unknown state.

I14. Worker and Scheduler E advance only after successful execution boundaries.

I15. Shared layer aliases never multiply physical ownership/free/zero/copy work.
```

---

# 28. Top Risks

| Risk | Severity | Probability | Detection | Mitigation |
|---|---|---:|---|---|
| Dense scalar allocator accidentally remains active | Critical | High without explicit branch | Ragged manager RuntimeError / allocation unit tests | explicit Scheduler vector path; never call Dense allocate_slots |
| Scheduler E committed in `schedule()` | Critical | Medium | injected forward failure shows frontier advanced | reserve capacity only; commit in update_from_output |
| Worker/Scheduler frontier divergence | Critical | Medium | per-step version/E assertions | Worker tail commit + Scheduler successful-output commit |
| req_idx reused with stale mirror | Critical | Medium | finish/reuse test | clear Ragged mirror before req_idx release; new request Snapshot |
| layer/member ordering mismatch | Critical | Medium | multi-layer/custom-order test | group.layer_names-based centralized map; no name parsing |
| shared backing treated as per-layer caches | Critical | Medium | storage pointer and free/copy count tests | global alias invariant; dedup bulk ops |
| per-step placement tensor reconstruction | Medium | Medium | pointer identity/instrumentation | GPUModelRunner model-lifetime cached tensors |
| dummy/profile startup crash | High | Medium | E0 startup | explicit dummy/profile semantics; no request mirror dependency |
| Dense path regression | Critical | Low/Medium | `page_group_size=None` regression suite | narrow typed branch; no Dense helper rewrite |
| mixed prefill/decode q mismatch | Critical | Medium | E4 + D mixed tests | query_start_loc-driven per-request q, never q=1 assumption |
| new-page stale bytes become visible | High | Low under Core | NaN/stale sentinel test | visibility bounded by post-write member lens; write before read |
| manager registration/placement constructor cycle | High | Medium | startup/unit import test | lazy/local built-in registration and coordinator-only placement injection |

---

# 29. Why E Should Remain One Implementation Slice

**E DESIGN DECISION**

Keep:

```text
P1-V2-R1-E-PRODUCTION-RAGGED-IDENTITY-01
```

as one implementation slice.

The source audit does **not** show a reason to split:

1. coordinator integration can be unit-tested while production activation
   remains off;
2. Worker mirror lifecycle can be wired/tested before attention dispatch;
3. no independent architecture migration must be released as a half-active
   runtime.

The work spans many files, but file count is not a valid slice boundary.

The atomic final activation requirement is easier to enforce as one slice.

---

# 30. Recommended E Architecture

```text
                        ┌────────────────────────────┐
                        │ CacheConfig.page_group_size│
                        └──────────────┬─────────────┘
                                       │
                         central Core activation gates
                                       │
                                       v
                     ┌────────────────────────────────┐
                     │ Attention.get_kv_cache_spec()   │
                     │ → RaggedAttentionSpec           │
                     └───────────────┬────────────────┘
                                     │
                                     v
                     ┌────────────────────────────────┐
                     │ KV cache planner                │
                     │ one Ragged group                │
                     │ one global KVCacheTensor        │
                     └───────────────┬────────────────┘
                                     │
                                     v
                ┌──────────────────────────────────────────┐
                │ KVCacheCoordinator                       │
                │ ONE BlockPool                            │
                │ └─ RaggedAttentionManager                │
                │    canonical E[C] + page_rows[C]         │
                └───────────────────┬──────────────────────┘
                                    │ vector capacity
                                    v
                     ┌────────────────────────────────┐
                     │ Scheduler.schedule              │
                     │ source_E + q → target_E         │
                     │ reserve capacity, E unchanged   │
                     └───────────────┬────────────────┘
                                     │ Snapshot / Delta
                                     v
                     ┌────────────────────────────────┐
                     │ SchedulerOutput                 │
                     │ ragged_kv_updates               │
                     └───────────────┬────────────────┘
                                     │
                                     v
          ┌────────────────────────────────────────────────────┐
          │ GPUModelRunner                                     │
          │ RaggedWorkerPhysicalState (discardable CPU mirror) │
          │ static placement GPU tensors                       │
          │ reusable step staging                              │
          └───────────────────┬────────────────────────────────┘
                              │ active gather
                              v
                     ┌────────────────────────────────┐
                     │ global RaggedStepViews          │
                     └───────────────┬────────────────┘
                                     │ layer-name keyed projection
                                     v
                     ┌────────────────────────────────┐
                     │ ForwardContext                  │
                     │ attn_metadata[layer_name]       │
                     │ slot_mapping[layer_name]        │
                     └───────────────┬────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    v                                 v
       unified_kv_cache_update          unified_attention_with_output
                    │                                 │
                    v                                 v
          D Ragged KV write                 D Ragged FA2 read
                    └────────────────┬────────────────┘
                                     │
                                     v
                          successful execute tail
                                     │
                            Worker E commit
                                     │
                                     v
                            ModelRunnerOutput
                                     │
                                     v
                         Scheduler.update_from_output
                                     │
                          Scheduler canonical E commit
```

---

# 31. Rejected Alternatives

1. **Separate Scheduler-side Ragged allocator**  
   Rejected because it creates duplicate ownership/BlockPool/lifecycle.

2. **Replicate scalar sequence length into every cluster as permanent logic**  
   Rejected because the first real compaction immediately breaks it.

3. **Store canonical page ownership on GPU**  
   Rejected because Scheduler is the authority; GPU state is only step-derived.

4. **Use layer-name regex/index parsing for correctness**  
   Rejected because semantic topology must not depend on model naming syntax.

5. **Put global StepViews into a new ForwardContext field**  
   Rejected because existing per-layer metadata dictionaries are already the
   natural execution seam.

6. **Add `unified_attention_ragged` custom op like Tangram**  
   Rejected because qcrs already has two unified ops with the required
   write-before-read dependency.

7. **Modify FlashAttentionImpl directly**  
   Rejected because D already proves the exact FA2 primitives required.

8. **Explicit Worker ACK DTO for every step**  
   Rejected in synchronous Core because normal ModelRunnerOutput already proves
   successful completion.

9. **Zero every newly allocated Ragged page in E**  
   Rejected as a correctness requirement because invisible stale bytes are
   never read under the Core contract.

10. **Fresh `torch.tensor(MemberPlacementMap)` every step**  
    Rejected because placement is static model topology.

---

# 32. Implementation Handoff — Draft

> DRAFT ONLY. Not executable until ChatGPT Web review approves this document.

## Slice

```text
P1-V2-R1-E-PRODUCTION-RAGGED-IDENTITY-01
```

## Scope

Wire the already-proven Ragged A/B/C/D components into the real vLLM v0.26
Engine request lifecycle under a strict single-GPU/eager/FA2 identity
configuration.

## Ordered work

```text
1. add Core activation validation and explicit page-group-size EngineArgs plumbing
2. register RaggedAttentionManager under existing coordinator
3. add KVCacheManager narrow vector-capacity facade
4. Scheduler source_E+q allocation and Snapshot/Delta production
5. GPUModelRunner RaggedWorkerPhysicalState lifecycle
6. static placement GPU tensors and reusable staging
7. global StepViews + layer-local metadata publication
8. existing unified write/read typed dispatch
9. Worker tail commit + Scheduler update_from_output commit
10. dummy/profile startup compatibility
11. atomically enable RaggedAttentionSpec production switch
12. run E0–E7 + Dense regression
```

## Expected implementation files

Primary expected modifications:

```text
vllm/engine/arg_utils.py
vllm/config/vllm.py
vllm/model_executor/layers/attention/attention.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/sched/scheduler.py
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/attn_utils.py
vllm/v1/attention/backends/ragged_layout.py
vllm/v1/attention/backends/ragged_forward.py
tests/... focused E tests
```

Expected no-change / regression-only surfaces:

```text
vllm/config/cache.py
vllm/v1/kv_cache_interface.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/core/sched/output.py
vllm/v1/worker/gpu/ragged_kv_state.py
vllm/v1/worker/gpu/model_states/default.py
vllm/forward_context.py
vllm/v1/attention/backends/flash_attn.py
vllm/v1/outputs.py
```

Implementation may change an expected no-change file only if a source-backed
need is discovered; that requires recording the reason, not silently widening
scope.

## Acceptance

Must close:

```text
Core config/spec gate
one-global-backing invariant
one-BlockPool allocator invariant
vector normal append
Snapshot/Delta lifecycle
req_idx remove/reuse
static placement lifetime
mixed/chunked metadata
write/read production dispatch
Worker post-success commit
Scheduler post-success commit
E0–E7 real Qwen identity
Dense page_group_size=None regression
```

## Out of scope

```text
real compaction policy
scoring
TP2
PP>1
DCP/PCP>1
async scheduling
CUDA Graph
prefix cache / APC
CoW
KV Connector / LMCache
quantized Ragged KV
FA3
performance optimization
```

## Stop conditions

Stop and report `DESIGN_CONFLICT` if implementation discovers any of:

```text
1. existing coordinator cannot preserve one BlockPool while exposing Ragged
   vector allocation without changing ownership semantics;

2. real ForwardContext/unified-op path cannot invoke D helpers without a new
   custom op or broad Attention backend rewrite;

3. Scheduler has more than one in-flight physical frontier under the claimed
   synchronous Core configuration;

4. shared backing is not actually one storage allocation under the activated
   production planner;

5. request finish/preemption lifecycle can reuse req_idx before the Worker
   mirror can be reliably removed;

6. real Qwen FA2 layer order cannot be mapped from kv_cache_group.layer_names
   without model-name parsing.
```

Otherwise stop after evidence collection and return to Web review. Do not
auto-enter compaction or R2/R3.

---

# 33. Design Verdict

```text
E_DESIGN_READY_FOR_WEB_REVIEW
```

R1-E remains:

```text
DESIGN ONLY
NOT IMPLEMENTED
PRODUCTION ACTIVATION OFF
```
