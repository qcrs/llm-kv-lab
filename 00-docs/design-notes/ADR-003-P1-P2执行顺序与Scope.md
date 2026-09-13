# ADR-003 — P1 / P2 Execution Order and Scope

Status：ACCEPTED

## Decision

P1 为 primary active project。P2 文档现在建立，但 runtime 保持 BLOCKED_BY_PROJECT1_CORE。P1-G3 V1_RUNTIME_CORE_DONE 的 accepted Gate 是 P2 runtime 的唯一解锁条件。

同一时间只允许一个 runtime Parent Task IN_PROGRESS。解锁 P2 不等于自动启动 P2；仍需 P2 M0 ABI/source/worktree review。

## Rationale

两个项目都修改/依赖 vLLM KV runtime。并行推进会造成 source identity、环境、evidence 和用户学习负担混乱。先完成 P1 Core 能复用 owner/lifetime/slot mapping 的理解，但不能复用 P1 runtime result 冒充 P2 evidence。

## Revisit

只有用户 + ChatGPT Web 在 P1-G3 V1_RUNTIME_CORE_DONE Gate review 时，基于时间预算和交付目标修改顺序。

