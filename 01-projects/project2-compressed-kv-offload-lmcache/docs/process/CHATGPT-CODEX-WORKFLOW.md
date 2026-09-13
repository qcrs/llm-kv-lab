# P2 Web ↔ Codex Workflow

P1 unlock → Web review P2 M0 → source facts/drafts → explicit Approved Design → one Approved Slice → Codex minimal execution → tests/evidence/state/handoff → Web review → next decision。

当前在第一箭头前，P2 runtime blocked。PROJECT_STATE authoritative；runtime context/handoff/LATEST derived。Context compression and no-auto-next rules inherit root。

## v1.2 Review Boundary

Codex 完成 Slice 后不决定下一 Slice。它只提交 handoff/evidence，并把 `Next Allowed Action` 归位到 `WEB_REVIEW_CURRENT_SLICE`；Web + 用户 review 后才批准 next action。Core M0–M5 保持 Slice-level collaboration，不把 architecture-critical Parent Task 交给 AUTO_TASK 黑箱执行。

