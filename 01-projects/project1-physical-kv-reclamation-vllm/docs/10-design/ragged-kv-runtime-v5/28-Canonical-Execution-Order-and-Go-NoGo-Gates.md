> **v5 canonical addendum:** R2+ allocation/reclamation 必须遵循 `38-Scheduler-Side-Ragged-Physical-State-and-Allocation-Contract.md`；metadata shape/lifetime 遵循 `39-Ragged-Step-Metadata-Shape-Lifetime-and-Attention-Contract.md`。

# Canonical Execution Order & Go/No-Go Gates — v5

这是后续执行的唯一推荐顺序。每个 Gate 的意义是：**该层不再存在未解释的 correctness hole**。

---

## Gate 0 — Baseline Freeze

```text
R0
```

GO：

```text
clean dense smoke reproducible
P1 focused tests reproducible
source/env pins saved
```

---

## Gate 1 — Spec Contract

```text
R1-A1
```

GO：

```text
page bytes exact
Hkv/Hp geometry exact
unsupported combinations reject
dense path unchanged
```

NO-GO：如果 Ragged semantic Hkv 与 storage Hp 仍混在一个含义里。

---

## Gate 2 — Global Pool Contract

```text
R1-A2
```

GO：

```text
one global page namespace
capacity math exact
shared backing exact
```

NO-GO：若仍是每 layer 独立 page-ID namespace，却假装成 global pool。

---

## Gate 3 — Layout / Stride Contract

```text
R1-A3
```

GO：

```text
virtual view zero-copy
physical/virtual address oracle exact
```

NO-GO：

```text
reshape silently copies
(page,column) not contiguous
virtual-block formula depends on undocumented stride
```

这是 v4 引入、v5 继续冻结的硬 Gate。

---

## Gate 4 — RaggedBlockTables

```text
R1-B
```

GO：

```text
per-group rows independent
counts independent
staged update/gather correct
persistent pointers stable
```

---

## Gate 5 — Virtual Slot / KV Write

```text
R1-C
R1-D1
```

GO：

```text
member slot oracle exact
known K/V lands in exact physical cell
```

---

## Gate 6 — Attention Read

```text
R1-D2
R1-D3
```

GO：

```text
decode synthetic equality
prefill synthetic equality
mixed uneven-q equality
```

---

## Gate 7 — Real Identity

```text
R1-E
```

GO：

```text
dense/ragged real model parity
continuous batching
chunked prefill
no page leak
```

R1-E 未过：禁止进入 R2/R3/scorer。

---

## Gate 7.5 — Piecewise Smoke

```text
R1-F
```

可 PASS / DEFERRED。

这里只验证 compatibility；不是 Core blocker。

---

## Gate 8 — Per-Group Physical State

```text
R2
```

GO：

```text
same logical token
→ different group physical write positions
→ correct group reads

scheduler-side canonical E/counts exists
→ next allocation is computed per group
→ only boundary-crossing groups receive new pages
→ shorter groups do not regrow to max/logical depth
```

---

## Gate 9 — Physical Reclamation

```text
R3
```

必须原子满足：

```text
nonuniform target lengths
payload correct
worker rows shrink
scheduler canonical ownership matches
BlockPool free increases
freed page exact reuse
original request continues decode
```

只有这一 Gate 通过，项目才可以正式称：

> Physical KV Reclamation Runtime

---

## Gate 10 — Automatic Policy

```text
R4
```

GO：

```text
one scorer/control plane
policy generates keep plan
same R3 physical closure
```

自动 non-uniform 完成后才可以称：

> Non-Uniform KV Compression Runtime

---

## Gate 11 — Lifecycle

```text
R5
```

GO：

```text
true preemption observed
state reset exact
recompute correct
no stale page ownership
```

---

## Gate 12 — TP2

```text
R6
```

GO：

```text
local head geometry correct
rank-local attention correct
physical depths compatible
free/reuse correct
```

---

## Gate 13 — AOT

```text
R7
```

GO：own page-waste ablation shows meaningful reduction or a well-explained negative result。

---

## Gate 14 — Triton Optimization

```text
R8
```

只在 profile gate 通过后进入。

GO：reference equality + microbenchmark + e2e accounting。

---

## Gate 15 — Piecewise CG Hardening

```text
R9
```

GO：

```text
replay-safe persistent metadata
correctness
CPU/launch/TPOT evidence
```

---

## Gate 16 — Resume Closure

```text
R10
```

每条简历 bullet 必须能指向一份 evidence。

---

# Hard Stop Rules

```text
R1-E before scorer
R3 before AOT
reference before in-place Triton
Ragged eager before CUDA Graph optimization
TP1 Core before TP2
profiler before allocator/kernel optimization
```
