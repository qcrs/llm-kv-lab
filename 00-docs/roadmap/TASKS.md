# Cross-Project Task Index

本文件只控制跨项目顺序；详细状态以项目 PROJECT_STATE 为准。

| Project | Parent Task | Status | Runtime Authority | Next Condition |
|---|---|---|---|---|
| P1 | P1-M0-T1 Source/Environment/Worktree Gate | REVIEW_REQUIRED | REVIEW_ONLY | Web + user review then optional Slice Prompt |
| P1 | P1-M0-T2 Source Map Freeze | BLOCKED | NONE | M0-T1 PASS |
| P1 | P1-M1..M7 | BLOCKED | NONE | previous Gates |
| P2 | P2-M0-T1 ABI/Connector/Worktree Gate | BLOCKED_BY_PROJECT1_CORE | NONE | P1-G3 V1_RUNTIME_CORE_DONE accepted |
| P2 | P2-M1..M7 | BLOCKED_BY_PROJECT1_CORE | NONE | P1 unlock + previous Gates |

Current active Parent Task：P1-M0-T1。

Current approved Slice：NONE。

Only Next Allowed Action：ChatGPT Web + 用户 review P1 M0 gate。

