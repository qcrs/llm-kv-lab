# Current Workspace State

## Current Phase

Documentation / Source Freeze

Canonical package phase: `NEW_PROJECT_DOCUMENTATION_FINAL_CONTRACT_V1_2`

## Primary Active Project

Project 1 — Physical KV Cache Reclamation for vLLM

## Current Position

~~~yaml
current_parent_task: P1-M0-T1
execution_mode: REVIEW_ONLY
current_slice: NONE
parent_task_status: REVIEW_REQUIRED
slice_status: NOT_STARTED
approved_implementation_slice: NONE
~~~

## Approved High-Level Decisions

- P1 project identity and v3 baseline scope.
- P2 project identity and v3 baseline scope.
- P1 V1 / V2 / V3 boundary.
- P2 V0 / V1 / V2 / V3 boundary.
- P1-G3 V1_RUNTIME_CORE_DONE is the runtime review unlock gate for P2; it is not P1 full-project completion.
- ChatGPT Web / Work owns architecture review; Codex is Repository Execution Agent.
- PROJECT_STATE is authoritative detailed project state.

这些是高层项目/流程决策，不等于任何 implementation API、owner、field name 或 patch 已批准。

## Project 1

状态：DOCUMENTATION_READY / REVIEW_ONLY。

当前 Parent Task：P1-M0-T1 — Source / Environment / Worktree Freeze Gate。

当前 Slice：NONE。

Approved Design：仅 ADR-001 的高层范围；实现级设计 NONE。

## Project 2

状态：BLOCKED_BY_PROJECT1_CORE。

P2-M0-T1 及全部 runtime/environment/ABI 操作均 blocked。允许阅读与文档 review，不允许创建 runtime fork、安装环境或修改 vLLM/LMCache。

## Next Allowed Action

ChatGPT Web + 用户 review P1 M0 source/environment/worktree gate，核对 implementation worktree、branch、environment、source identity、允许的只读 probe 与 evidence schema；review 后再决定是否生成第一个明确 Slice Prompt。

## Explicitly Forbidden Now

- 创建或执行 Implementation Slice；
- 修改 vLLM / LMCache source；
- 正式 benchmark / profiler capture；
- 把候选 Task 或 Slice 标 PASS；
- 解锁 P2 runtime；
- 用旧 QCache / KV Retention 设计补全新项目。



## v1.2 Final-Contract Note

完整 P1/P2 v3 baseline 已嵌入各项目 `docs/90-reference/baselines/`；canonical corrections 见 `BASELINE-RESOLUTION.md`。P1 增加 Qwen3/RoPE positional gate、partial-tail invariant，并把 explicit reclaim_generation 降为 conditional draft。P2 将 quant granularity改为待 layout 冻结的 quant_group，并将 exact byte equality限定为 fixed-size FSL2 scope。

Final-contract additions：P1 增加 post-prefill peak/non-claim、hybrid-block/DCP/PCP/cascade gate、live-row stale-tail seam、V2 current-member index domain；P2 将 codec block冻结为 LMCache object-domain quant group，并增加 FS page-cache/O_DIRECT benchmark regimes与 FSL2 capacity claim scope。Codex state write权限收紧为 factual-only，Core M0–M5 默认禁止 AUTO_TASK。
