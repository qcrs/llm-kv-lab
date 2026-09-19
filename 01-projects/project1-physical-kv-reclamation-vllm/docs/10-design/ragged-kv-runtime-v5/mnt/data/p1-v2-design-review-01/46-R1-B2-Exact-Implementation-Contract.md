# R1-B2 Exact Implementation Contract — Scheduler↔Worker Transport / Persistent Cluster Mirror

**Parent:** P1-V2-DESIGN-REVIEW-01  
**Slice:** R1-B2  
**Activation:** FORBIDDEN  
**Dependency:** B1 state/snapshot/delta semantics frozen  
**Does not require:** A3 physical layout or FA2 execution adapter

---

# 0. Objective

Implement transport and Worker persistent physical metadata so any Scheduler-owned Ragged request state, including synthetic non-uniform state, can be reconstructed and incrementally updated.

No Attention read/write logic is part of B2.

---

# 1. Transport Principle

Do not overload Dense:

```text
NewRequestData.block_ids
CachedRequestData.new_block_ids
```

with physical-cluster semantics.

Those fields encode the vLLM KVCacheGroup axis.

Ragged physical clusters are a separate axis and require explicit transport.

---

# 2. Proposed Transport Objects

Exact field names may vary slightly; semantics are frozen.

## Full Snapshot

```text
RaggedRequestStateSnapshotData
├─ effective_lens: list[int]        # C
├─ page_counts: list[int]           # C
└─ flat_page_ids: list[int]         # sum(counts)
```

## Allocation Delta

```text
RaggedPageAllocationDeltaData
├─ expected_source_page_counts: list[int]   # C
├─ appended_page_counts: list[int]          # C
└─ flat_new_page_ids: list[int]
```

## SchedulerOutput aggregate

Preferred conceptual shape:

```text
ragged_kv_updates
├─ snapshots: dict[req_id, snapshot]
└─ allocations: dict[req_id, delta]
```

Dense transport remains unchanged.

---

# 3. Full Snapshot Usage

Use snapshot when Worker cannot safely reconstruct exact cluster rows from an existing local row:

```text
new request
resume
re-add after persistent-row removal
explicit resync/rebuild
```

Flattening is deterministic cluster-major order.

No Worker-private authoritative snapshot.

---

# 4. Worker Persistent State

Preferred dedicated abstraction:

```text
RaggedWorkerPhysicalState
```

or equivalent narrow combination with `RaggedBlockTables`.

Persistent shapes:

```text
page_rows:
[Rmax,C,M] int32

page_counts:
[Rmax,C] int32

effective_lens:
[Rmax,C] int32
```

Reuse existing request-state row indexing and `idx_mapping`.

Do not introduce another request-ID→slot allocator.

---

# 5. Suggested Files

Primary:

```text
vllm/v1/core/sched/output.py
vllm/v1/worker/gpu/ragged_block_table.py
vllm/v1/worker/gpu/model_runner.py
```

Optional narrow helper:

```text
vllm/v1/worker/gpu/ragged_kv_state.py
```

Tests:

```text
tests/v1/core/test_ragged_transport.py
tests/v1/worker/test_ragged_block_table.py
```

Keep generic runner modifications minimal and type-directed.

---

# 6. Snapshot Apply

Validate before mutation:

```text
len(E) == C
len(counts) == C
sum(counts) == len(flat_page_ids)
E[c] >= 0
counts[c] >= 0
E[c] <= counts[c] * B
page IDs valid when namespace bounds are available
non-null page IDs unique inside request state
```

Construct candidate rows first.

Then atomically publish:

```text
rows
counts
E
```

No partial write.

---

# 7. Allocation Delta Apply

Validate:

```text
current_counts == expected_source_page_counts
len(appended_counts) == C
sum(appended_counts) == len(flat_new_page_ids)
new IDs valid
new IDs unique
new IDs not already active in request rows
```

Build candidate rows/counts and commit together.

---

# 8. Request Lifecycle

Required Worker lifecycle:

```text
add from full snapshot
increment via allocation delta
remove/reset request row
persistent request-index reuse
re-add from Scheduler snapshot
```

On slot reuse:

```text
old rows/counts/E must be cleared before new snapshot commit
```

No stale tail may remain observable.

---

# 9. RaggedBlockTables API

Preferred semantic API:

```text
apply_snapshot(req_idx, snapshot)
apply_allocation_delta(req_idx, delta)
reset_request(req_idx)
gather_step_rows(idx_mapping, ...)
get_page_counts(...)
```

Internal implementation may use `StagedWriteTensor` / UVA helpers when compatible.

Do not expose an identity-only API as the sole permanent path.

An identity convenience wrapper is allowed only if it compiles down to the general snapshot path.

---

# 10. Backend-Neutral Step Gather

B2 must provide/prove:

```text
persistent [Rmax,C,M]
+ idx_mapping
→ step [R,C,M]
```

and equivalent gather for:

```text
E
counts
```

Do not yet derive:

```text
FA2 vbid
member block tables
member slot mappings
member-major Q
```

Those belong to C/D.

---

# 11. Production Wiring Boundary

B2 may define data structures and Worker apply hooks, but normal Dense scheduler/worker behavior remains unchanged because production Ragged activation is still OFF.

Tests may directly construct Ragged update objects and invoke isolated Worker helpers.

No real Ragged Engine startup is required.

---

# 12. Zeroing Interface

B2 may preserve a flat new-physical-page-ID zeroing transport where existing API semantics already match.

Do not implement Ragged global-backing zeroing semantics in B2 if that belongs to A2/A3/E integration.

The transport should remain compatible with later page-size-aware Ragged zeroing.

---

# 13. Worker Free Is Forbidden

No B2 method may free BlockPool pages based on Worker metadata.

Worker removal/reset clears the mirror only.

Physical page release is Scheduler/B1 authority.

---

# 14. Forbidden Changes

Do not:

```text
activate Attention Ragged Spec dispatch
activate production Registry path
modify BlockPool
free pages from Worker
encode C as num_kv_cache_groups
scatter identity formula layer*G+head//Hp throughout Worker
add FA2 member-major execution metadata
implement compression policy
move/compact KV payload
```

---

# 15. Required Tests

## T1 — Uniform full snapshot

Exact reconstruction.

## T2 — Non-uniform full snapshot

Different counts per cluster reconstruct exact rows.

## T3 — Invalid flat length

Atomic failure.

## T4 — Invalid E/capacity relation

Atomic failure.

## T5 — Uniform incremental delta

All clusters append correctly.

## T6 — Non-uniform incremental delta

Only selected clusters append.

## T7 — Stale source-count delta

Atomic failure with unchanged state.

## T8 — Duplicate new page ID

Rejected.

## T9 — Remove/reset + slot reuse

No stale rows/counts/E.

## T10 — Re-add from Scheduler snapshot

Exact state restored without Worker-private snapshot.

## T11 — Active-step gather

`idx_mapping` returns exact `[R,C,M]` rows/counts/E.

## T12 — Dense BlockTables regression

Normal Dense path unchanged.

---

# 16. Acceptance Evidence

Required:

```text
core transport tests
worker Ragged metadata tests
existing Dense BlockTables regression
py_compile
git diff --check
focused no-FA/no-activation audit
```

GPU only if the selected staged-write implementation requires it; CPU/reference oracles remain mandatory.

---

# 17. Handoff Condition

B2 is accepted when:

```text
Any Scheduler-owned Ragged request state,
including synthetic non-uniform state,
can be serialized as snapshot/delta,
applied atomically to a Worker cluster-major mirror,
removed,
and reconstructed later without Worker-owned authoritative snapshot state.
```

Then R1-C can derive backend-neutral physical addresses from the mirror.
