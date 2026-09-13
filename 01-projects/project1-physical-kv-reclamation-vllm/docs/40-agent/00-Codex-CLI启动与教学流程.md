# P1 Codex CLI Startup and Teaching Flow

## Startup

从项目 root 启动 review；实现会话只能从 PROJECT_STATE 冻结的 implementation worktree 启动。禁止从 study/reference tree 启动。

每次输出：

~~~text
Project / Parent Task / Slice / Mode
Learning Goal
Source Identity
Approved Designs
Known Facts / TO_VERIFY
Read First / exact anchors
Allowed / Read-only / Out of Scope
Expected Evidence / Gate
Next Allowed Action
~~~

当前：P1-M0-T1、REVIEW_ONLY、Slice NONE。

## Teaching Flow

~~~text
concept / why
→ fixed source caller/state/owner
→ user prediction
→ tiny hand-worked case
→ oracle/test
→ approved minimal patch
→ evidence
→ Web diff review and learning note
~~~

Codex 的解释服务于执行事实；详细系统教学、架构批准和结果解释由 Web + 用户完成。

## Stop Before Patch

没有 source identity、Approved Slice、Approved Design（若需要）、allowed files、test/evidence path 任一项，保持 REVIEW_ONLY。

