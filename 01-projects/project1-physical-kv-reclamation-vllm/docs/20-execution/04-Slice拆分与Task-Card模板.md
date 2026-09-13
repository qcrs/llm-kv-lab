# Slice Decomposition and Task Card Template

当前没有 Approved Slice。本文件只说明如何在 Web review 后从 Parent Task 创建一个 Slice。

## Decomposition Rule

一个 Slice 应在一次 review cycle 内完成，通常只包含以下一种：

- read-only source audit；
- one pure oracle/test scaffold；
- one state transition；
- one output/manager seam；
- one kernel + launcher；
- one approved experiment；
- one evidence/release close。

不要把 source audit、architecture decision、runtime integration、benchmark 放进同一 Slice。

## Candidate ID

例如 P1-M0-T1 可以候选拆为 repo identity、environment/import identity、Gate close，但只有 Web + 用户选定一个并输出 one-shot prompt 后，ID 才从 PLANNED_DECOMPOSITION 变 APPROVED。

## Parent Task Card Template

~~~text
Parent Task ID / Status
Goal
Learning Goal
Prerequisites
Known SOURCE_FACT / OBSERVED
HYPOTHESIS / TO_VERIFY
Read First
Source Anchors
Approved Design Needed
Allowed Files
Read-only Files
Out Of Scope
Expected Patch Shape
Tests
Evidence
Pass/Fail Gate
Failure Signals
Rollback/Fallback
Handoff
Next Task
Candidate Slice Boundaries (not approved)
~~~

## One-Shot Slice Prompt Template

~~~text
Execution Mode: SLICE_EXECUTE
Approved Slice ID:
Parent Task:
User-approved Goal:
Approved Designs:
Source Identity:
Read First:
Allowed Files:
Read-only Files:
Expected Patch:
Exact Tests:
Evidence Path:
Stop Conditions:
What Is Not Authorized:
Parent Task remains NOT CLOSED.
Do not create or execute the next Slice.
~~~

若 Approved Designs 为空但 Slice 需要架构选择，Codex 必须 REVIEW_ONLY。

