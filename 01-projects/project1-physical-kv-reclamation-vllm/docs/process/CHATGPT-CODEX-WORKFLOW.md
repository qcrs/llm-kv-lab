# P1 ChatGPT Web / Work ↔ Codex Workflow

## Loop

1. Web + 用户读取 PROJECT_STATE 与 raw evidence，讲解 source/state/owner。
2. 形成 Design Draft；明确 source facts 与 open questions。
3. 用户 + Web 明确 APPROVED_DESIGN。
4. Web 生成一个 one-shot Approved Slice Prompt。
5. Codex SLICE_EXECUTE minimal patch/test/evidence。
6. Codex 更新 Slice record、PROJECT_STATE 事实字段、runtime handoff、LATEST pack。
7. Web review diff/结果/失败，形成 learning note。
8. 用户确认后才决定下一 Slice或 Parent Task Gate。

没有第 3/4 步时 REVIEW_ONLY。“进入 Parent Task”不授权自动做完。

## Runtime Files

- PROJECT_STATE：authoritative detailed state。
- runtime/CURRENT-CONTEXT：derived Web compression。
- runtime/CHATGPT-HANDOFF：latest execution entry。
- ENVIRONMENT-SNAPSHOT / SOURCE-IDENTITY：identity snapshots。
- ARTIFACT-INDEX：path index only。
- LATEST pack：transport entry only。

冲突：PROJECT_STATE + raw evidence优先。

## Handoff Pack

04-experiments/project1_kv_reclaim/handoffs/LATEST 固定五文件。大型 raw artifact 不复制；FILES 引用。历史 truth 是 Slice record + original evidence，不是 LATEST。

## Compression

只保留 Facts/Approved/Drafts/Evidence/State/Open Questions/Constraints；不得升级、推断、删失败或改 TO_VERIFY。

## v1.2 Review Boundary

Codex 完成 Slice 后不决定下一 Slice。它只提交 handoff/evidence，并把 `Next Allowed Action` 归位到 `WEB_REVIEW_CURRENT_SLICE`；Web + 用户 review 后才批准 next action。Core M0–M5 保持 Slice-level collaboration，不把 architecture-critical Parent Task 交给 AUTO_TASK 黑箱执行。

