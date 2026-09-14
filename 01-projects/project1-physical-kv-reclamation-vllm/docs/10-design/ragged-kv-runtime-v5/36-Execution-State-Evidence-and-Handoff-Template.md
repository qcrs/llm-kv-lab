# Execution State / Evidence / Web↔Codex Handoff Template — v4

这个模板用于每个 Slice，避免项目再次出现“代码改了很多，但不知道是否真正 closed”。

---

# 1. Slice State Header

```text
Project: Ragged KV Runtime for vLLM 0.26
Parent Round: R?-?
Slice ID: R?-?-IMPL-??
Status: OPEN / IN_PROGRESS / BLOCKED / PASS / DESIGN_CONFLICT
Branch:
HEAD before:
HEAD after:
Reference pins:
```

---

# 2. Frozen Goal

一句话：

```text
This slice proves ____________.
```

禁止写一堆 feature。

---

# 3. Source Facts

```text
qcrs fact 1:
file/function:
what it currently does:

Tangram fact 1:
file/function:
reference semantic:
```

必须把：

```text
SOURCE FACT
vs
OUR DESIGN CHOICE
```

分开。

---

# 4. Invariants

```text
I1:
I2:
I3:
```

每个 invariant 最好能被 test/assert/evidence检查。

---

# 5. Allowed Files

```text
MODIFY:
- ...

NEW:
- ...

DO NOT TOUCH:
- ...
```

---

# 6. Implementation Plan

按最小 dependency 顺序：

```text
Step 1
Step 2
Step 3
```

不要同一 Slice 同时修改多个独立高风险 subsystem。

---

# 7. Minimal Tests

只写高价值 tests：

```text
T1 happy-path semantic
T2 boundary/shape
T3 dense regression
T4 one failure/unsupported gate
```

除非 bug 需要，不做过度 permutation testing。

---

# 8. Evidence Required

```text
command:
result:
artifact/log:
key assertion:
```

对于 runtime Slice，要至少保存：

```text
config
source SHA
command
stdout summary
important state dump
```

---

# 9. Acceptance

```text
[ ] A1
[ ] A2
[ ] A3
```

全部满足才 PASS。

---

# 10. Failure / Fallback

```text
Observed failure:
Which invariant failed:
Source/API mismatch or design conflict:
Fallback:
Whether next Slice is blocked:
```

如果出现 `DESIGN_CONFLICT`，停止扩散 patch，回 Web 做 architecture review。

---

# 11. Web → Codex Prompt Skeleton

```text
你现在执行 Ragged KV Runtime 的 <SLICE>。

Pinned source:
- qcrs/vllm ...
- Tangram ...

Goal:
<one sentence>

Source facts:
<facts>

Frozen design:
<design>

Allowed files:
<files>

Do not touch:
<files/features>

Acceptance:
<exact list>

Tests:
<small list>

Evidence required:
<commands/state dumps>

If implementation reveals a contradiction with frozen design, stop with DESIGN_CONFLICT; do not broaden scope.
```

---

# 12. Codex → Web Handoff Skeleton

```text
STATUS:
PASS / BLOCKED / DESIGN_CONFLICT

CHANGED FILES:
...

IMPLEMENTED:
...

TESTS:
...

EVIDENCE:
...

INVARIANTS VERIFIED:
...

KNOWN LIMITATIONS:
...

DESIGN QUESTIONS:
...

HEAD:
...
```

---

# 13. Round Closure Ledger

| Round | Status | Core Evidence | Resume Claim Enabled? |
|---|---|---|---|
| R1 | OPEN | identity engine | no |
| R2 | OPEN | unequal E[r,g] | no |
| R3 | OPEN | free/reuse/continue | physical reclamation |
| R4 | OPEN | automatic policy | non-uniform runtime |
| R5 | OPEN | preemption pressure | lifecycle-safe |
| R6 | OPEN | TP2 closure | distributed |
| R7 | OPEN | fragmentation ablation | AOT optimization |
| R8 | OPEN | kernel microbench | Triton optimization |
| R9 | OPEN | TPOT/launch profile | CG optimization |
| R10 | OPEN | final benchmark | final resume |

每完成一个 Slice，只更新对应 row，不提前改“最终状态”。
