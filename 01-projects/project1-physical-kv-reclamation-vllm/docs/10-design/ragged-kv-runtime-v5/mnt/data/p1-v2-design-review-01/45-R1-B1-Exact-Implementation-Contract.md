# R1-B1 Exact Implementation Contract — Scheduler Ragged Ownership Manager

**Parent:** P1-V2-DESIGN-REVIEW-01  
**Slice:** R1-B1  
**Activation:** FORBIDDEN — normal Engine startup must not route to this manager yet  
**Dependency:** R1-A1 RaggedAttentionSpec definition  
**Parallelism:** may proceed independently of A2/A3

---

# 0. Objective

Implement the Scheduler-side **general Ragged physical ownership model** while running only R1 identity behavior.

R1 must already prove synthetic non-uniform states through the same API so R2 does not require redesign.

---

# 1. Canonical State

Introduce one coherent request-state abstraction:

```text
RaggedRequestPhysicalState
├─ effective_lens[C]
├─ page_counts[C]
└─ ordered_page_rows[C]
```

Required invariants:

```text
len(E) == C
len(counts) == C
len(rows) == C
counts[c] == len(rows[c])
E[c] <= counts[c] * B
```

R1 identity initializes uniform vectors.

The concrete Python container is not the architectural contract.

---

# 2. Preferred Module Boundary

Suggested new module:

```text
vllm/v1/core/ragged_kv_cache_manager.py
```

Potential definitions:

```text
RaggedRequestPhysicalState
RaggedCapacityPlan
RaggedPageAllocationDelta
RaggedAttentionManager
```

Avoid large permanent `if ragged` branches in `single_type_kv_cache_manager.py`.

---

# 3. Manager Type

The intended final manager is:

```text
RaggedAttentionManager
```

compatible with Registry/Coordinator architecture at R1-E.

B1 does not activate production registration/dispatch.

---

# 4. Static Inputs

Manager receives authoritative:

```text
C
B
BlockPool
RaggedAttentionSpec / required storage metadata
```

Manager must not permanently derive cluster semantics from:

```text
layer_idx
head_idx
layer*G+group
```

`MemberPlacementMap` belongs outside Scheduler ownership.

---

# 5. BlockPool Use

Reuse upstream `BlockPool`.

Ragged interpretation:

```text
allocated block ID == physical page ID
```

No ring allocator or second free queue.

---

# 6. Pure Capacity Planning

Implement a no-mutation planning primitive equivalent to:

```text
plan_capacity(
    request_id,
    target_effective_lens[C],
)
→ RaggedCapacityPlan
```

Derived:

```text
target_counts[c] = ceil(target_E[c]/B)
need[c] = max(target_counts[c] - current_counts[c], 0)
```

Planning records enough source state to validate application.

No BlockPool mutation during pure planning.

---

# 7. R1 Identity Adapter

R1 convenience may accept:

```text
append_q
```

and derive:

```text
target_E[c] = current_E[c] + append_q
```

for all clusters.

This must call the vector planning path, not a separate scalar implementation.

---

# 8. Reserve / Scatter / Commit

Given `need[C]`:

```text
reserve sum(need) pages
```

Scatter by explicit counts.

Example:

```text
need=[1,0,2,0]
ids=[100,101,102]
```

becomes:

```text
c0 += [100]
c1 += []
c2 += [101,102]
c3 += []
```

Build candidate state before commit.

No equal-split inference.

---

# 9. Snapshot Export

Expose exact Scheduler-owned snapshot semantics:

```text
E[C]
page_counts[C]
cluster-major flat page IDs
```

This is sufficient for Worker reconstruction in B2.

---

# 10. Request Release

On request release:

```text
collect active non-null pages
remove canonical request state
free pages through BlockPool
```

Tests must verify page count restoration and later reuse by another request.

---

# 11. Compaction/Reconciliation Primitive

B1 should define/prove the Scheduler-side reconciliation primitive even before real Worker compaction is integrated.

Input concept:

```text
request_id
expected_source_E[C]
expected_source_counts[C]
new_E[C]
new_counts[C]
```

Rules:

```text
validate all source state first
new_counts[c] <= old_counts[c] for shrink path
new_E[c] <= new_counts[c] * B
```

Derive detached pages from canonical rows.

Then:

```text
commit all new rows/E/counts
→ free detached pages
```

No partial mutation.

This synthetic primitive is mandatory because it proves R3 does not require a new ownership architecture.

---

# 12. Prefix / Cache-Hit Boundary

R1 prefix caching remains disabled.

Do not implement:

```text
hash lookup
common-prefix sharing
cache-hit logic
```

B1 may preserve future refcount compatibility by avoiding a global-exclusive-page assumption.

---

# 13. Forbidden Changes

Do not:

```text
activate Registry mapping in normal startup
modify Attention.get_kv_cache_spec
change Dense FullAttentionManager semantics
modify Worker block tables
add SchedulerOutput Ragged carriers
touch FA2
add policy/scoring
add Tangram ring allocator
represent physical clusters as KVCacheGroups
```

---

# 14. Required Tests

## T1 — Identity initialization

Uniform E/counts/rows valid.

## T2 — Uniform append crossing boundary

All clusters need one page.

## T3 — Uniform append inside existing capacity

No allocation.

## T4 — Synthetic non-uniform capacity plan

Example:

```text
E=[32,56,20,48]
counts=[2,4,2,3]
target=[33,57,21,49]
need=[1,0,0,1]
```

## T5 — Explicit scatter

Non-uniform `need` maps flat IDs to exact rows.

## T6 — Source mismatch atomic failure

No state mutation, no free.

## T7 — Synthetic compaction reconciliation

Different clusters shrink by different amounts.

## T8 — Request free/reuse

Another request receives a released physical page ID.

## T9 — Two requests interleaved

No cross-request corruption.

## T10 — Dense manager regression

Existing manager behavior unchanged.

---

# 15. Acceptance Evidence

Required:

```text
targeted core tests
relevant BlockPool/single-type-manager regression
py_compile
git diff --check
focused no-activation audit
```

No GPU or engine smoke required.

---

# 16. Handoff Condition

B1 is accepted only if:

```text
R1 identity uses the same vector ownership APIs
that synthetic non-uniform states already exercise successfully.
```

If R2 would need to redesign B1 to introduce `C`-vector semantics, B1 fails acceptance.
