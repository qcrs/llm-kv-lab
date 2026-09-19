# Ragged Scheduler↔Worker Transport, Step Metadata & Attention Contract v2

**Status:** PROPOSED CANONICAL — P1-V2-DESIGN-REVIEW-01  
**Role:** Scheduler transport / Worker mirror / backend-neutral step view / FA2 adapter contract

---

# 0. Fundamental Separation

The runtime is split into five layers:

```text
1. Scheduler canonical ownership
2. Scheduler → Worker state transport
3. Worker persistent physical mirror
4. Backend-neutral cluster-major step view
5. Execution adapter (FA2 member-major in R1)
```

Only layer 5 is FA2-specific.

---

# 1. Static Geometry

```text
R   = active requests in current step
C   = physical cluster count
Hp  = page width in semantic members
B   = cache block size
M   = maximum page depth of one cluster row
Q   = total scheduled query tokens
Hkv = semantic KV heads per local layer
Hq  = semantic query heads per local layer
```

R1 identity only:

```text
G = Hkv/Hp
C = L_local * G
```

---

# 2. MemberPlacementMap

Canonical semantic mapping:

```text
(local_layer, semantic_kv_head)
→ (cluster, column)
```

R1 identity:

```text
cluster = layer * G + head // Hp
column  = head % Hp
```

Consumers must not scatter this arithmetic throughout ModelRunner/FA/compaction code.

---

# 3. Scheduler Canonical State

Authority:

```text
req
→ E[C]
→ page_counts[C]
→ page_rows[C]
```

Worker is never authoritative merely because it owns the GPU-side mirror.

---

# 4. Transport Modes

Dense fields such as:

```text
NewRequestData.block_ids
CachedRequestData.new_block_ids
```

encode one list per vLLM KVCacheGroup.

Do not reinterpret them as one list per physical cluster.

Ragged transport explicitly carries a cluster axis and distinguishes:

```text
Full Snapshot
Incremental Allocation Delta
```

---

# 5. Full Snapshot

Conceptual object:

```text
RaggedRequestStateSnapshotData
├─ effective_lens[C]
├─ page_counts[C]
└─ flat_page_ids[]
```

Used for:

```text
new request
resume
re-add after worker persistent-row removal
worker state rebuild/resync
future migration/recovery
```

Flattening order is cluster-major:

```text
cluster0 row
cluster1 row
...
cluster C-1 row
```

`page_counts` reconstructs row boundaries.

No Worker-private authoritative snapshot is required.

---

# 6. Incremental Allocation Delta

Conceptual object:

```text
RaggedPageAllocationDeltaData
├─ expected_source_page_counts[C]
├─ appended_page_counts[C]
└─ flat_new_page_ids[]
```

Used for continuously-running requests that only gain physical capacity.

Worker validates source counts before append.

No equal-split assumption.

---

# 7. Effective Frontier Transport

Normal append does not need to retransmit a full `E[C]` vector solely because a page boundary was crossed.

However:

```text
Full Snapshot
```

must include current `E[C]`.

And compaction/reclaim must explicitly validate/report source and final frontier vectors.

---

# 8. Worker Persistent Mirror

Conceptual state:

```text
RaggedWorkerPhysicalState
```

Persistent shapes:

```text
page_rows    [Rmax,C,M] int32
page_counts  [Rmax,C]   int32
effective_E  [Rmax,C]   int32
```

Reuse existing request-lifetime indexing:

```text
req_id → persistent req_idx
```

Current step reuses existing:

```text
idx_mapping[batch_req] → persistent req_idx
```

No second request-slot allocator.

---

# 9. Snapshot Apply Atomicity

Before mutation validate:

```text
len(E) == C
len(counts) == C
sum(counts) == len(flat_page_ids)
E[c] >= 0
counts[c] >= 0
E[c] <= counts[c] * B
page IDs valid when pool bounds are available
non-null IDs unique inside request state
```

Build candidate rows first.

Then atomically publish:

```text
rows
counts
E
```

---

# 10. Delta Apply Atomicity

Validate:

```text
worker counts == expected_source_page_counts
len(appended_counts) == C
sum(appended_counts) == len(flat_new_page_ids)
new IDs valid
new IDs unique
new IDs not already active in the request
```

Build candidate state, then commit all.

---

# 11. Backend-Neutral Step Gather

Gather persistent cluster state once:

```text
persistent rows [Rmax,C,M]
+ idx_mapping
→ step_cluster_rows [R,C,M]
```

Likewise:

```text
step_effective_E [R,C]
step_page_counts [R,C]
```

Do not gather from persistent rows independently for every Attention layer.

---

# 12. RaggedClusterStepView

Conceptual backend-neutral step object:

```text
RaggedClusterStepView
├─ cluster_rows
├─ effective_lens
├─ page_counts
└─ request/step mapping metadata as needed
```

It does not contain FA2 virtual block IDs.

It does not own scheduler state.

---

# 13. Canonical Physical Address

For member `(layer,head)`:

```text
(cluster,column)
=
MemberPlacementMap[layer,head]
```

For physical position `p`:

```text
depth        = p // B
block_offset = p % B
page_id      = cluster_rows[req,cluster,depth]
```

Canonical target:

```text
(page_id,column,block_offset)
```

---

# 14. FA2 Virtual Encoding

R1 FA2 adapter may encode:

```text
vbid = page_id * Hp + column
slot = vbid * B + block_offset
```

This is not allocator identity or permanent ABI.

---

# 15. FA2 Member Block Tables

For a semantic member mapped to `(c,col)` and physical row:

```text
[p0,p1,p2,...]
```

adapter derives:

```text
[p0*Hp+col,
 p1*Hp+col,
 p2*Hp+col,
 ...]
```

Temporary logical shape:

```text
[R,Hkv,M]
```

FA2 adapter may flatten:

```text
[R*Hkv,M]
```

R1 member row ordering:

```text
request_batch_idx * Hkv + semantic_kv_head
```

This is execution ordering, not physical ownership ordering.

---

# 16. Member Effective Length

R1 page-group visibility:

```text
member_effective_len(request, member)
=
cluster_effective_len[
    request,
    placement(member).cluster
]
```

Future per-member visible lengths may be added to the execution adapter without changing scheduler page ownership.

---

# 17. KV Write Addressing

Canonical write target:

```text
physical_pos = E_start[cluster] + query_offset
```

then derive:

```text
depth
block_offset
page_id
column
```

R1 FA2 adapter may materialize:

```text
member_slot_mapping [Q*Hkv]
```

Future fused writeback may consume `(page_id,column,block_offset)` directly.

---

# 18. Decode Adapter

R1 may reshape:

```text
Qdense [R,Hq,D]
→ [R,Hkv,q_per_kv,D]
→ [R*Hkv,q_per_kv,D]
```

with member block tables and member-visible lengths.

Do not define Ragged semantics as “R*Hkv model sequences.”

---

# 19. Prefill / Mixed Adapter

Correctness-first R1 path:

```text
token-major
→ materialized member-major
→ FA2
→ inverse scatter
```

This is replaceable later.

---

# 20. Compaction Direction

Scheduler → Worker:

```text
snapshot / allocation delta
```

Worker → Scheduler:

```text
physical compaction result
```

Worker reports shape/state transition, not allocator-authoritative page IDs to free.

Scheduler derives detached pages from canonical rows.

---

# 21. Worker Free Is Forbidden

Required order:

```text
Worker compacts
→ reports final shape
→ Scheduler validates/reconciles
→ Scheduler commits ownership
→ Scheduler frees detached pages
```

---

# 22. Fresh-Page Zeroing

A flat physical-page-ID list remains a good generic transport for fresh-page zeroing.

Cluster metadata is unnecessary.

But worker zeroing must interpret IDs against the Ragged global backing and Ragged page size, not Dense layer-local tensor semantics.

---

# 23. CUDA Graph Boundary

R1 eager may use temporary execution-adapter metadata.

Persistent address-stable buffers are a later CG-hardening concern.

Do not make FA2 metadata part of Scheduler↔Worker canonical protocol.

---

# 24. Acceptance

Required tests:

```text
full snapshot reconstruction
non-uniform snapshot reconstruction
uniform allocation delta
non-uniform allocation delta
stale source-count rejection
duplicate ID rejection
remove/reset + request-slot reuse
re-add from Scheduler snapshot
step gather exactness
MemberPlacementMap oracle
physical address oracle
FA2 vbid oracle
member block-table oracle
member slot-mapping oracle
decode adapter oracle
prefill pack/inverse-pack equality
```
