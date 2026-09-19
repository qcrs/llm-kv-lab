# Scheduler-Side Ragged Physical State & Allocation Contract v2

**Status:** PROPOSED CANONICAL — P1-V2-DESIGN-REVIEW-01  
**Role:** Scheduler ownership / capacity / reconciliation contract  
**Key rule:** R1 uses general vector state with identity/uniform values; R2 only enables divergence.

---

# 0. Authority Model

```text
Scheduler RaggedAttentionManager
= canonical physical ownership + canonical physical frontier

BlockPool
= physical page allocator/free/refcount machinery

Worker
= execution mirror

Retention/compression policy
= decision producer only
```

Exactly one subsystem owns physical page assignment: Scheduler-side Ragged manager.

---

# 1. Static Topology

Scheduler allocation needs:

```text
C = physical cluster count
B = block size
```

It does not need to know semantic `(layer,head)` placement.

R1 identity topology builder may produce:

```text
G = Hkv/Hp
C = L_local * G
```

but Manager APIs consume `C`, not the flatten formula.

---

# 2. Canonical Per-Request State

Conceptual object:

```text
RaggedRequestPhysicalState
```

Fields:

```text
effective_lens[C]
page_counts[C]
ordered_page_rows[C]
```

For cluster `c`:

```text
E[c]
= committed physical KV frontier

page_counts[c]
= owned active page capacity count

page_rows[c][d]
= physical page ID at depth d
```

Required invariants:

```text
len(E) == C
len(counts) == C
len(rows) == C
counts[c] == len(rows[c])
E[c] >= 0
counts[c] >= 0
E[c] <= counts[c] * B
```

Do not permanently require:

```text
counts[c] == ceil(E[c]/B)
```

because future reservation/speculation may temporarily over-own capacity.

---

# 3. Container Is Not the Contract

Allowed implementation representations include:

```text
nested rows
flat IDs + counts
CSR-style row_ptr + IDs
```

The semantic API must always expose ordered cluster rows and counts.

Never depend on allocation order to rediscover cluster boundaries.

---

# 4. Page Identity

Under Ragged:

```text
BlockPool block_id == physical page_id
```

IDs are worker-pool-local.

R1:

```text
page 0 = NULL
allocatable = [1,Nphys)
```

---

# 5. Frontier and Capacity Are Separate

Example:

```text
B = 16
E[c] = 32
counts[c] = 3
```

means:

```text
valid physical KV = 32 tokens
owned capacity     = 48 token slots
```

This is legal architecture even if settled R1 usually owns exactly two pages.

---

# 6. Canonical Capacity Planning

Permanent planning input:

```text
target_effective_lens[C]
```

Derived:

```text
target_counts[c]
=
ceil(target_E[c] / B)

need[c]
=
max(target_counts[c] - current_counts[c], 0)

total_need
=
sum(need)
```

R1 identity convenience:

```text
target_E[c] = current_E[c] + q
```

for all `c`.

That convenience must reuse the vector path.

---

# 7. Plan → Reserve → Commit

Planning does not mutate state.

Conceptual plan:

```text
RaggedCapacityPlan
├─ source_E[C]
├─ source_counts[C]
├─ target_E[C]
├─ target_counts[C]
├─ appended_counts[C]
└─ total_new_pages
```

Execution:

```text
plan
→ BlockPool reserve total_new_pages
→ deterministic scatter using appended_counts[C]
→ candidate state
→ atomic commit
```

---

# 8. Deterministic Scatter

Example:

```text
appended_counts = [1,0,2,0]
flat_new_ids    = [100,101,102]
```

Result:

```text
cluster0 += [100]
cluster1 += []
cluster2 += [101,102]
cluster3 += []
```

Forbidden canonical assumption:

```text
len(new_ids) % C == 0
```

---

# 9. Allocation Delta

Conceptual result/transport:

```text
RaggedPageAllocationDelta
├─ expected_source_page_counts[C]
├─ appended_page_counts[C]
└─ flat_new_page_ids[]
```

Validate:

```text
current_counts == expected_source_counts
sum(appended_counts) == len(flat_new_ids)
new IDs valid
new IDs unique in delta
new IDs do not duplicate active IDs of same request
```

Then commit candidate state atomically.

---

# 10. R1 Identity Example

```text
C=4
B=16
E=[32,32,32,32]
counts=[2,2,2,2]
```

One new token:

```text
target_E=[33,33,33,33]
target_counts=[3,3,3,3]
need=[1,1,1,1]
```

Pool returns:

```text
[20,77,31,90]
```

Each cluster appends one physical page.

---

# 11. R2 Example Without API Change

Current:

```text
E=[32,56,20,48]
counts=[2,4,2,3]
```

Next token:

```text
target_E=[33,57,21,49]
target_counts=[3,4,2,4]
need=[1,0,0,1]
```

Pool returns:

```text
[110,111]
```

Only cluster0 and cluster3 grow.

No R2 API redesign.

---

# 12. Step Frontier Vocabulary

A step may contain:

```text
E_start
  ↓ normal writes
E_produced
  ↓ optional compaction
E_final
```

R1 identity:

```text
E_final == E_produced
```

R3 compaction:

```text
E_final <= E_produced
```

This vocabulary is mandatory for same-step append + compaction reasoning.

---

# 13. Reconciliation / Compaction

Worker result conceptually reports:

```text
expected_source_E[C]
expected_source_counts[C]
new_E[C]
new_counts[C]
```

Scheduler validates all clusters before mutation.

Detached pages:

```text
detached[c]
=
old_row[c][new_counts[c] : old_counts[c]]
```

Candidate row:

```text
new_row[c]
=
old_row[c][:new_counts[c]]
```

Commit order:

```text
validate all
→ build all candidate state
→ commit rows/counts/E
→ free detached pages
```

No partial cluster commit.

---

# 14. Failure Atomicity

Any mismatch or invalid state implies:

```text
zero canonical mutation
zero BlockPool free
```

Examples:

```text
source E mismatch
source counts mismatch
invalid page ID
duplicate page ID
invalid growth/shrink
row/count inconsistency
```

---

# 15. Request Release

On finish/preemptive final release:

```text
collect all active non-null physical pages
remove request canonical state
free pages through BlockPool
```

The logic must not assume uniform row depth.

---

# 16. Prefix Sharing Boundary

R1 prefix caching is OFF.

Do not freeze the architecture to globally-exclusive pages forever because BlockPool refcounting may support future shared-prefix semantics.

R1 does require no accidental duplicate non-null page references inside one request's active physical positions.

---

# 17. RaggedAttentionManager Boundary

Manager owns:

```text
C
request states
capacity planning
reservation/scatter
snapshot export
request free
compaction reconciliation
```

Manager does not own:

```text
MemberPlacementMap
FA2 virtual block arithmetic
worker tensor layout
retention scoring/policy
```

---

# 18. R1 Acceptance

Required tests:

```text
identity initialization
uniform cross-block append
no-growth append
synthetic non-uniform capacity planning
explicit non-uniform scatter
source mismatch atomic failure
duplicate page rejection
request release
released page reuse by another request
two-request interleave
NULL page exclusion
dense manager regression
```

Synthetic non-uniform tests are mandatory in R1 even while production values remain uniform.
