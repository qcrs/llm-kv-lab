# ChatGPT Web / Work 与 Codex 协作及文档角色矩阵

## 协作循环

~~~text
ChatGPT Web + 用户
  source teaching / architecture review
  → Approved Design + Approved Slice Prompt
Codex
  minimal repository execution
  → diff / tests / raw evidence / handoff
ChatGPT Web + 用户
  diff review / interpretation / learning note
  → approve, revise, fallback or stop
~~~

Codex 不是 milestone black box。没有 Approved Slice 时保持 REVIEW_ONLY。

## 职责矩阵

| 工作 | 用户 + ChatGPT Web/Work | Codex CLI |
|---|---|---|
| 源码教学/整体认知 | Owner | 提供 fixed anchors |
| Source Fact | Review/解释 | 调查与记录 |
| Architecture policy | 明确批准 | 只能 draft/停止等待 |
| Task/Slice 拆分 | Owner | 可提出候选 |
| Minimal patch | Review | 执行批准 Slice |
| Test/experiment | 设计与解释 | 执行批准矩阵 |
| Raw evidence/diff | Review | Owner |
| PROJECT_STATE | 决策 review / Next Allowed Action owner | 仅事实字段；执行后 Next Allowed Action 固定回 Web review |
| 学习笔记 | Owner | 提供材料 |
| 简历 claim | evidence 审计 | 不得自动发布 |

## 文档角色

| 文档 | 角色 | Authority |
|---|---|---|
| root AGENTS | 工作区 Agent 行为 | Agent chain #1 |
| CURRENT | 跨项目唯一当前阶段 | Workspace control |
| project AGENTS | 项目边界/stop rule | Agent chain |
| PROJECT_STATE | 项目详细状态 | Authoritative project state |
| accepted ADR | 稳定高层/架构取舍 | Below raw source/evidence |
| Master Plan | scope、milestone、Gate | Below PROJECT_STATE |
| Task Card | Parent Task contract | 本任务范围 |
| one-shot Slice Prompt | 单次授权 | 不能覆盖上层 |
| CURRENT-CONTEXT | Web 快速接入 | Derived |
| CHATGPT-HANDOFF | 最近执行入口 | Derived |
| Artifact Index | evidence path index | Index only |
| learning | 理解过程 | Non-state |
| delivery | 可验证表达 | Non-state |
| 04-experiments | raw/processed evidence | Evidence truth |

## Approved Design 最小格式

- Decision ID / status / date；
- problem and source facts；
- chosen option / rejected options；
- owner、state、invariant、publication/commit point；
- failure behavior、fallback、scope；
- impacted Parent Task；
- explicit implementation fields still TO_VERIFY。

只有用户 + ChatGPT Web 明确说 approved 才能进入 PROJECT_STATE 的 Approved Designs。

## Review Checklist

ChatGPT Web 每次收到 handoff 检查：

1. source identity 与 Slice scope；
2. diff 是否仅允许文件；
3. 新事实是否有 anchor/evidence；
4. draft 是否被错误升级；
5. test 是否覆盖 failure/feature-off；
6. negative result 是否保留；
7. Parent Task 是否被过早 PASS；
8. Next Allowed Action 是否唯一。

## PROJECT_STATE 写权限补充

Codex 是 recorder，不是 state-machine approver。它可以记录 Slice 的事实性结果并提出 `Proposed Next Action`，但不得通过更新 `PROJECT_STATE` 间接批准下一 Slice、Parent Gate 或 P2 unlock。任何执行 Slice 结束后，唯一允许的执行态下一步是：

```text
Next Allowed Action = WEB_REVIEW_CURRENT_SLICE
```

Web + 用户 review 后才生成新的 one-shot Slice Prompt。
