# LLM KV Lab — P1 Reclaim / P2 Offload 文档工作区 v1.2 Final Contract

这是两个 LLM Inference / AI Infra 简历项目的研发控制面，不含功能实现代码。

- Project 1：Physical KV Cache Reclamation for vLLM
- Project 2：Compressed KV Offload for vLLM + LMCache

## Start Here

1. `AGENTS.md`：workspace 协作与权限契约
2. `CURRENT.md`：当前阶段、Parent Task、Slice 和阻塞
3. `00-docs/INDEX.md`：路线、学习与 evidence 导航
4. `01-projects/project1-physical-kv-reclamation-vllm/PROJECT_STATE.md`
5. P1 Master Plan 与当前 M0 Task Card

旧 QCache / KV Retention 项目仅保存在 `archive/legacy-projects/`，状态为
`ARCHIVED`，不是当前技术设计来源。

本包已经建立项目定义、Source Map、设计基线、Milestone / Parent Task / Slice / Gate、Agent 执行契约、实验纪律、runtime context sync、handoff pack 和交付审计。

## 当前状态

~~~text
Current Phase: NEW_PROJECT_DOCUMENTATION_FINAL_CONTRACT_V1_2
Primary Active Project: Project 1
Current Parent Task: P1-M0-T1
Execution Mode: REVIEW_ONLY
Current Slice: NONE
Approved Implementation Slice: NONE
Project 2: BLOCKED_BY_PROJECT1_CORE
~~~

本状态不授权 Codex 修改 vLLM 或 LMCache。下一步只能由 ChatGPT Web + 用户审查 P1 M0 的 source/environment/worktree gate，之后再决定是否创建第一个明确 Slice Prompt。

## 阅读顺序

1. CURRENT.md
2. AGENTS.md
3. REPO_PINS.md
4. 00-docs/roadmap/00-跨项目总控执行手册.md
5. 00-docs/roadmap/02-ChatGPT-Web-Work-Codex协作与文档角色矩阵.md
6. 当前项目 PROJECT_STATE.md
7. 当前项目 `docs/90-reference/BASELINE-RESOLUTION.md`
8. 当前项目 `SUPPORTED-CONFIG-MATRIX.md`、Master Plan、SOURCE-MANIFEST 和 M0 Task Card
9. 必要 design / source anchors；需要完整历史 reasoning 时再读 `docs/90-reference/baselines/`

## 两条权威链

技术事实与状态：

~~~text
fixed source / raw evidence
→ accepted ADR / APPROVED_DESIGN
→ PROJECT_STATE.md
→ Master Plan
→ Task Card
→ derived context / prompt
~~~

Agent 行为：

~~~text
root AGENTS.md
→ workspace collaboration matrix
→ project AGENTS.md
→ Task Card
→ one-shot Slice Prompt
~~~

Prompt 永远不能覆盖 fixed source。CURRENT-CONTEXT、CHATGPT-HANDOFF 与 LATEST handoff pack 都是派生入口，不是新的事实源。

## 目录职责

- 00-docs：跨项目路线、ADR、source reading、profiling、weekly、interview 和 context pack。
- 01-projects：两个项目的设计、执行、评测、Agent、交付、process、learning、decisions。
- 04-experiments：raw/processed evidence 与 handoff 入口；不放设计结论。
- 05-models：模型身份与许可索引；不打包模型本体。

详细文件清单见 PACKAGE-MANIFEST.md，迁移边界见 MIGRATION-NOTES.md。



## v1.1 Repair Scope (historical hardening retained)

- preserved full P1/P2 v3 technical baselines in-package；
- added canonical baseline resolutions；
- expanded Core design docs toward implementation-spec depth；
- corrected P1 zero-copy reference level and generation over-design；
- added Qwen3/RoPE and partial-tail hard gates；
- scoped P2 quant_group and filesystem byte invariants；
- normalized Unicode filenames；
- kept `REVIEW_ONLY`, no Approved Slice, no runtime source modification。

## v1.2 Final Contract

本包在 v1.1 hardened 基础上完成开工前最后一次 contract hardening。项目方向与 V1/V2/V3 不变；本次只收紧：P1 prefill-peak/block-granularity/live-row seam，P2 object-domain codec/FS benchmark claim，以及 Web↔Codex state authority/provenance。当前仍为 `REVIEW_ONLY`，Approved Implementation Slice = NONE。
