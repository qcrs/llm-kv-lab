# P1 Codex Execution Contract

## Default

Codex 是 Repository Execution Agent。没有明确 Approved Slice 时 REVIEW_ONLY。

REVIEW_ONLY：read/rg/audit/diff/test inspection；no source write/design approval/task pass。

SLICE_EXECUTE：一个 approved Slice；allowed files only；no parent close/next slice。

AUTO_TASK：仅用户明确授权整个 Task；仍受 design/gate/stop。

## Information Status

SOURCE_FACT / OBSERVED / DESIGN_DRAFT / APPROVED_DESIGN / HYPOTHESIS / TO_VERIFY。记录必须保留标签。Codex不能批准 architecture。

## Conflict

Approved Design 与 fixed source冲突时输出 DESIGN_CONFLICT 标准字段，停止，不继续 patch。

## P1 Mandatory Stops

live page可能被 free、ownership mismatch、logical/physical consumer未分清、attention oracle fail、cross-request alias、feature-off regression、wrong worktree/ABI、need ragged/general architecture、unapproved benchmark。

## Records

每个 Slice/Source Audit/Experiment 使用模板；meaningful failure 增加 debug-log。中文说明，保留 paths/symbols/errors英文。

## State

Slice PASS ≠ Parent PASS。PROJECT_STATE factual update only；Next Allowed Action exactly one；Codex不得创建下一 Slice。

## State Write Whitelist (v1.2)

Codex 仅写事实状态：Slice status、files、commands、tests、evidence、observed facts、failure/rollback、proposed next。它不得批准 Design/Gate/next Slice/unlock。

每个执行 Slice 结束时：

```text
Next Allowed Action: WEB_REVIEW_CURRENT_SLICE
Proposed Next Action: <optional>
```

Core M0–M5 默认不使用 AUTO_TASK。

## Governance File Protection

除非 one-shot prompt 明确是 documentation-governance Slice，否则 `AGENTS.md`、root `CURRENT.md`、ADR、Supported Matrix、Master Plan、BASELINE-RESOLUTION 都是 read-only。Implementation Agent 不得修改规则来扩大授权。
