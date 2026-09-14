# Preemption / Recompute Lifecycle — Exact Engineering Contract

**Goal:** 证明 Ragged physical state 在真实 serving pressure 下不会因为 request lifecycle 变化而出现 stale page、double free 或错误 restore。

---

# 1. State Layers

必须显式区分：

```text
Logical Request State
  token history
  request status
  num_computed_tokens

Scheduler Physical Ownership
  req → group rows
  BlockPool ref/free state

Worker Physical View
  RaggedBlockTables
  RaggedKVState E[r,g]

Compression State
  score buffers
  boundary state
  pending plan/result
```

这些层的生命周期不同。

---

# 2. Event Taxonomy

## Event A — Not Scheduled This Step

```text
request仍running/owned
worker persistent state保留
```

动作：无。

## Event B — Worker Slot Rearrangement

如果未来出现纯 worker-side row move，但 scheduler仍拥有KV，则搬 persistent state即可；不 free。

## Event C — True Scheduler Preemption

```text
scheduler frees owned KV
request status→preempted/waiting
worker receives preempted_req_id
worker runtime state removed
later recompute
```

这是 R5 必须完成的事件。

---

# 3. Reset Contract on True Preemption

必须清除：

```text
RaggedBlockTables[req_idx,*]
RaggedBlockTables counts
RaggedKVState E[req_idx,*]
compression score rows
compression boundary metadata
pending compaction plan/result
per-request temporary references
```

必须确保：

```text
freed page IDs 不再出现在 worker active rows
```

---

# 4. Re-admission / Recompute

preempted request 后续被重新调度时：

```text
scheduler allocates/reuses new canonical blocks
→ worker add/update request
→ overwrite row from scheduler delta
→ E initialized according to recompute progress
→ model recomputes KV
```

不能：

```text
restore old compressed block IDs
restore old E
restore old score slots
```

---

# 5. Why Tangram Snapshot/Restore Is Not the Default Here

旧 Tangram 某些 persistent-batch removal场景需要 snapshot ragged row，因为 row topology non-uniform。

MRv2 当前 persistent request-indexed data plane下，普通“本 step未调度”不删除 row。

因此 snapshot/restore 不应成为主线设计；只有未来明确出现“scheduler仍拥有KV但worker主动drop persistent row”的场景才增加。

---

# 6. Pressure Test Design

通过小 pool 强制：

```text
num_gpu_blocks_override = small value
```

Workload：

```text
A long prefill/decode
B arrives
pool pressure
scheduler preempts one request
other request makes progress
preempted request later recomputes
```

同时运行 no-pressure reference。

---

# 7. Evidence to Capture

```text
preemption event/request id
ownership before preemption
BlockPool free count before/after
worker row before/reset
E before/reset
new allocation on resume
output tokens vs reference
final free count
```

---

# 8. Failure Injections

## P1 stale row not reset

预期 test 能检测：resume reads freed/reused block。

## P2 E not reset

预期 physical write跳到错误 offset。

## P3 score state not reset

新 sequence 继承旧 token importance。

## P4 pending compaction result arrives after preemption

Scheduler source fence 应把 stale receipt 标记 stale/ignore，而不是应用到新 incarnation。

这条尤其重要。

---

# 9. Acceptance

```text
true preemption reproduced
canonical pages freed
worker row/count/E reset
stale receipt cannot mutate new state
resume recomputes
output correct
no double free
no page leak
```

---

# 10. Resume Claim Gate

只有真实 pressure test PASS 才写：

> 完善 Ragged KV request lifecycle，区分 transient unscheduled 与 scheduler preemption/recompute，避免 stale per-group physical state，并在 KV-pressure workload 下验证释放、重算与继续解码。
