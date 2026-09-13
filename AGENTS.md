# Workspace Agent Contract

## 0. Documentation and Evidence Writing Policy

除临时对话中的思考过程外，所有写入仓库的新建或修改的工程文档、实验报告、分析报告、
源码阅读笔记、Task/Slice 执行记录、code change record、handoff 和状态说明必须使用中文。
源码 symbol、class/function/variable、文件路径、
命令、raw stdout/stderr、Git 输出、错误信息、Profiler 原始输出和配置字段名保持
英文原样，例如：`num_computed_tokens`、`BlockTables.compute_slot_mappings()`、
`ModelRunnerOutput`。

`raw/*.log`、raw command output 和 terminal stdout/stderr 是不可翻译的原始证据；
中文解释只能写入 report、analysis note 或 handoff；不得因为模板或上游习惯把正式记录写成英文。
详细不等于把完整 raw log
复制进报告：raw log 保存完整事实，报告引用日志并提炼关键观察、分析和限制。

正式实验、验证报告或 Gate evidence 至少回答：

1. 为什么做：问题、不确定性、对应 Gate/invariant；
2. 验证什么：明确 H1/H2 或 invariant；
3. 环境与版本：workspace、worktree、branch、HEAD、Python、Torch、CUDA、GPU、
   model、dtype、关键配置和 env；
4. 怎么做：步骤、命令、脚本、输入和场景；
5. 观察到什么：标记 `OBSERVED` 并引用 raw evidence；
6. 结果说明什么：从 source/runtime 观察到项目结论的推理；
7. 结果不能说明什么：明确未覆盖的 correctness、ownership、性能或支持范围；
8. 下一步为什么是那个动作：绑定 remaining `TO_VERIFY`、Gate 和权限边界。

正式报告推荐使用以下结构：

```text
# 实验 / 验证名称
## 1. 背景与验证动机
## 2. 本轮问题
## 3. 假设 / Invariant
## 4. 环境与版本
## 5. 测试设计
## 6. 执行步骤与命令
## 7. 原始观察
## 8. 结果分析
## 9. 该结果证明了什么
## 10. 该结果没有证明什么
## 11. 与项目设计 / Gate 的关系
## 12. Remaining TO_VERIFY
## 13. Evidence Index
## 14. 下一步
```

源码阅读笔记不得只是函数目录；应围绕问题、ownership、persistent/per-step state、
data/control flow、source anchor、设计原因和 P1 relevance 组织。每个关键 symbol
尽量说明创建者、修改者、生命周期、消费者、shape/device、publication point 和
与 P1 的关系。

除非用户明确要求英文，Codex final handoff 默认使用中文；symbol、command 和 raw
evidence 路径保持英文。

## 0.1 AGENTS Scope Clarification

根 AGENTS 与 P1 AGENTS 约束本 workspace/P1 control plane。`worktrees/p1-vllm-reclaim/AGENTS.md`
及其下级 AGENTS 是上游 vLLM Repo B 的贡献规范，本轮不修改它们，避免污染 pinned
implementation worktree；上游规则与本 workspace 的 REVIEW_ONLY、Slice boundary 和
source safety 规则同时适用。

本文件是工作区级最高 Agent 行为契约。它约束 ChatGPT Web / Work、Codex CLI 与用户如何协作，但不创造技术事实。

## 1. 项目顺序与范围

~~~text
P1 documentation ready
→ P1 M0 source/environment/worktree freeze
→ P1 V1_RUNTIME_CORE_DONE
→ P2 runtime unlock
→ P2 V0/V1/V2
~~~

P1 是唯一 active runtime project。P2 文档完整存在，但在 P1 V1_RUNTIME_CORE_DONE 的 accepted Gate 前保持 BLOCKED_BY_PROJECT1_CORE。不得并行扩展两个 runtime scope。

本工作区不继承旧 QCache 或旧 KV Retention 技术架构。迁移边界见 MIGRATION-NOTES.md。

## 2. 角色

ChatGPT Web / Work + 用户负责 Mentor / Architect / Reviewer：

- 源码教学、系统认知、Source Fact 与 Design Choice 分离；
- 架构取舍、风险分析、Task / Slice 拆分；
- 明确 Approved Design 与 Approved Slice；
- 实验设计、diff review、profiling 解释、学习笔记；
- 项目总结、简历与面试 evidence 审计。

Codex CLI 是 Repository Execution Agent：

- fixed source audit；
- 执行已批准的最小 Slice；
- minimal patch、unit/integration test、已批准 experiment；
- raw evidence、Git diff、source anchors、execution record；
- handoff pack 与事实性 PROJECT_STATE 更新。

Codex 不得静默决定 architecture policy，也不得把一个 Parent Task 当作黑箱自动做完。

## 3. 执行模式

- REVIEW_ONLY：默认于没有明确 Approved Slice 时。允许 read、rg、source audit、git diff/test inspection；禁止改 source、批准设计或进入实现。
- SLICE_EXECUTE：只执行用户明确批准的一个 Slice ID。不得自动完成 Parent Task、进入下一 Slice 或扩大实验矩阵。
- AUTO_TASK：只有用户明确说“自动完成这个 Task”或等价授权时启用；仍受 Gate、stop condition 与记录要求约束。**P1/P2 Core 的 M0–M5 默认禁止 AUTO_TASK**：source freeze、state contract、runtime integration、correctness 与 profiler-critical task 必须保持 Slice 级 review。AUTO_TASK 只推荐用于已经批准的机械性 benchmark sweep、重复 correctness matrix、profiling 采集或 packaging/docs refresh。

用户只说“进入 M1-T2”“继续做 M1-T2”“看看这一部分”，不构成 AUTO_TASK，也不构成具体 Slice 批准。

## 4. Parent Task 与 Slice

Parent Task 使用 P1-Mx-Ty / P2-Mx-Ty。Slice 使用 P1-Mx-Ty-Sz / P2-Mx-Ty-Sz。

- Parent Task 描述完整目标与 Gate。
- Slice 是一次会话可验证的最小仓库变更或只读审计。
- Slice PASS 不等于 Parent Task PASS。
- Parent Task 只有 Gate evidence 完整且用户要求收尾/关闭后才能 PASS。
- 当前没有任何 Approved Implementation Slice。

20-execution 中的候选 Slice 边界是 PLANNED_DECOMPOSITION，不是已创建或获批准的 Slice。

## 5. 事实与设计状态

统一标签：

- SOURCE_FACT：fixed source 直接确认。
- OBSERVED：具体命令、测试、trace 或 profiler 原始产物观察。
- DESIGN_DRAFT：候选设计，尚未批准。
- APPROVED_DESIGN：用户 + ChatGPT Web 明确确认。
- HYPOTHESIS：可证伪解释或性能预测。
- TO_VERIFY：尚未确认，必须绑定最小 source audit / test / experiment。

SOURCE_FACT 不等于 DESIGN_DRAFT；DESIGN_DRAFT 不等于 APPROVED_DESIGN。Codex 可以发现事实、提出 draft，但不能自行升级设计。

## 6. DESIGN_CONFLICT

Approved Design 与 fixed source 冲突时立即停止，并落盘：

~~~text
# DESIGN_CONFLICT
Approved assumption:
Observed source fact:
Exact source anchors:
Why they conflict:
Possible options:
Files modified so far:
Safe rollback needed:
Tests already run:
No further patch was made.
~~~

不得自行选择一个新架构继续 patch。

## 7. Source of Truth

技术事实 / 项目状态：

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
→ 00-docs/roadmap/02 collaboration matrix
→ project AGENTS.md
→ Task Card
→ one-shot Slice Prompt
~~~

PROJECT_STATE.md 是项目唯一 authoritative detailed state。CURRENT.md 是跨项目 current control；CURRENT-CONTEXT、CHATGPT-HANDOFF、LATEST pack 是派生入口。

## 8. Session 启停规则

开始前必须：

1. 读取 CURRENT.md、REPO_PINS.md、根与项目 AGENTS。
2. 读取项目 PROJECT_STATE、Master Plan、当前 Parent Task。
3. 核对 repo/worktree/branch/HEAD/status、环境和 import source identity。
4. 明确 mode、Approved Design、Approved Slice、allowed/read-only files、Gate 和不做项。
5. 上一 Gate smoke 失败则不开始新 Slice。

结束时必须：

1. 保存 source/test/experiment record；
2. 更新 raw evidence index 与 handoff；
3. 仅按 State Write Whitelist 事实性更新 PROJECT_STATE；
4. Slice 执行结束时 `Next Allowed Action` 必须固定为 `WEB_REVIEW_CURRENT_SLICE`（或 blocked 时等价 Web review 状态）；Codex 只能另写 `Proposed Next Action`，不能批准下一步；
5. 不自动创建下一 Slice、PASS Parent Gate、解锁 P2 或修改 Approved Design。

## 9. Teaching-First / Source-First / Evidence-Driven

顺序固定：

~~~text
Source
→ Reference
→ Correctness
→ Runtime
→ Benchmark
→ Profiler
→ Conclusion
~~~

修改前先解释 caller/callee、owner、state、shape/layout、CPU/GPU boundary、publication point、failure 与 fallback。先预测再 patch；预测不符时回到 source/state model，不扩大代码。

## 10. 实验与性能纪律

每次实验固定记录 Git SHA、environment、model、workload、shape、batch、config、warmup、repeats、command、raw result、processed result、conclusion。

- microbenchmark、E2E benchmark、profiler evidence 分开；
- 一次只改变一个核心变量；
- negative result 和 meaningful failure 必须保留；
- 未测数字不得进入 README/简历；
- P1 free BlockPool page 不等于 CUDA HBM 返回系统；
- P2 generic Serde 不等于 first D2H bytes 下降。

每项优化记录 Hypothesis、Baseline、Change、Correctness、Metric、Profiler、Decision、Negative regime、Next step。

## 11. 文档归属

- learning：概念卡、源码 walkthrough、用户复述、问题库、反思；不是状态。
- decisions：项目内部稳定取舍。
- 00-docs/design-notes：跨项目 accepted ADR。
- 04-experiments：raw/processed evidence。
- profiling-notes / evaluation：profiling 方法与结论。
- 50-delivery：交付、简历、面试审计。
- PROJECT_STATE：只保存当前事实与状态，不写教程。

## 12. Context Compression Rule

压缩只能保留：

- Source Facts；
- Approved Designs；
- Design Drafts；
- Evidence；
- Current State；
- Open Questions；
- Constraints。

禁止：

- inference → fact；
- draft → approved；
- 删除 negative result / meaningful failure；
- TO_VERIFY → conclusion；
- 自动解决 design question；
- 因省略限制而改变设计结论。

## 13. 安全边界

不得覆盖用户 dirty changes、修改 study tree、把 reference repo 当 implementation source、使用破坏性 Git 命令、未经批准安装/重建依赖、运行正式 benchmark 或扩大支持矩阵。发现身份或权限不清时停止。



## 14. Technical Baseline Preservation (introduced in v1.1; retained in v1.2)

每个项目 `docs/90-reference/baselines/` 保存完整 v3 technical reasoning snapshot。它是 READ_ONLY_REFERENCE_SNAPSHOT，不是 state/approval。若摘要文档与 snapshot 冲突，先读 `BASELINE-RESOLUTION.md`；若与 fixed source冲突，以 fixed source为准。Agent不得因为 split design doc 较短而重新发明已由 baseline+resolution确定的 reference-backed contract。

## 15. PROJECT_STATE State Write Whitelist (v1.2)

Codex 对 `PROJECT_STATE.md` 的写权限是 **factual recorder**，不是 workflow authority。

### Codex MAY update

- current Slice factual status (`IN_PROGRESS/PASS/FAIL/BLOCKED`)；
- modified files / commands / tests / raw evidence paths；
- observed source/environment facts 与 exact anchors；
- last-good commit、rollback need、failure reason；
- `Proposed Next Action`（只是一条建议）。

### Codex MUST NOT approve/change

- `APPROVED_DESIGN`；
- Parent Task / Gate PASS；
- next Parent Task / cross-project unlock；
- project scope / support matrix；
- `Next Allowed Action` 为某个新的执行 Slice；
- P1→P2 runtime unlock。

Slice 执行结束时 Codex 应写：

```text
Next Allowed Action: WEB_REVIEW_CURRENT_SLICE
Proposed Next Action: <optional proposal>
```

只有用户 + ChatGPT Web/Work review 后，才能把 `Next Allowed Action` 改成新的 Approved Slice / Gate action。

## 16. Reference Interpretation Precedence (v1.2)

技术事实仍以 `fixed source / raw evidence` 最高。对于 package 内的 **reference/design interpretation**，读取顺序固定：

```text
fixed local source / raw evidence
→ accepted ADR / APPROVED_DESIGN / PROJECT_STATE
→ docs/90-reference/BASELINE-RESOLUTION.md
→ current docs/10-design + current Master Plan / Task Card
→ SOURCE-MANIFEST / REFERENCE-PORTING-LEDGER
→ docs/90-reference/baselines/* historical snapshot
```

历史 baseline 用于保留 reasoning，不直接提供当前 implementation contract。若 `BASELINE-RESOLUTION` 已声明某段被 superseded，Codex 不得从 historical snapshot 恢复旧结论。

## 17. Governance Files Are Read-Only During Normal Implementation

普通 source-audit / implementation / test / benchmark Slice 中，Codex 不得修改以下 governance/authority 文件：

```text
root AGENTS.md
CURRENT.md
REPO_PINS.md
00-docs/design-notes/ADR-*
project AGENTS.md
project docs/00-overview/SUPPORTED-CONFIG-MATRIX.md
project docs/00-overview/Project*-Master-Plan.md
project docs/90-reference/BASELINE-RESOLUTION.md
```

这些文件只有在用户 + ChatGPT Web/Work 明确创建 **documentation-governance Slice** 时才允许修改。Codex 永远不得通过修改治理文件扩大自己的权限、改变 Gate、解锁 P2 或把 draft 升为 approved。

`CURRENT.md` 是跨项目控制面，默认只由用户 + ChatGPT Web/Work 更新；Codex 的执行结果通过 project `PROJECT_STATE` / handoff 上报。

## 18. Archive Rule

`archive/` is **HISTORICAL ONLY**, **NOT CURRENT SOURCE OF TRUTH**, and
**DO NOT USE FOR CURRENT P1/P2 IMPLEMENTATION**. Codex may inspect it to preserve
history or recover provenance, but archived QCache/KV Retention technical decisions
must not be used as the current implementation contract.
