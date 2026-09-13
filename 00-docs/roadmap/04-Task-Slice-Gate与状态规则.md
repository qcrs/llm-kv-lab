# Task / Slice / Gate 与状态规则

## 1. ID

- Project：P1 / P2。
- Milestone：P1-M0。
- Parent Task：P1-M0-T1。
- Slice：P1-M0-T1-S1。
- Gate：P1-G0。

20-execution 中列出的 S1/S2 是候选拆分语义，状态为 PLANNED_DECOMPOSITION；只有 one-shot prompt 明确 ID、Approved Design、allowed files、tests 和 Gate 后，才成为 Approved Slice。

## 2. 状态

Parent Task：

- BLOCKED
- REVIEW_REQUIRED
- READY
- IN_PROGRESS
- VERIFY
- PASS
- FAIL
- DEFERRED
- SKIPPED_WITH_REASON

Slice：

- NONE
- PROPOSED
- APPROVED
- IN_PROGRESS
- PASS
- FAIL
- BLOCKED

同时只能有一个 runtime Slice IN_PROGRESS。当前所有 Slice 均 NONE/未创建。

## 3. Gate

Gate 不允许 PARTIAL PASS。所需 evidence 缺失即 BLOCKED。

每个 Gate report 包含：

- decision；
- active pins/environment；
- required evidence；
- reproduction；
- failures；
- scope confirmation；
- Last Good Commit；
- Next Allowed Action。

## 4. Parent Task Card 必填字段

- Goal / Learning Goal；
- Read First / Source Anchors；
- Approved Design Needed；
- Allowed Files / Read-only Files；
- Out Of Scope；
- Expected Patch Shape；
- Tests；
- Evidence；
- Pass/Fail Gate；
- Failure Signals；
- Rollback / Fallback；
- Handoff；
- Next Task。

## 5. Slice Prompt 必填字段

- exact Slice ID 与 mode；
- parent task；
- user-approved goal；
- Approved Designs 列表；
- exact source identity；
- read first / allowed / read-only；
- expected patch shape；
- exact tests/evidence；
- stop conditions；
- what is not authorized；
- no auto-next statement。

缺任何 architecture-critical 字段，Codex 保持 REVIEW_ONLY。

## 6. 状态更新

Slice 完成只更新 Slice Status 和事实性字段。Parent Task Gate 需单独 review。Codex 完成 Slice 后必须把 `Next Allowed Action` 设为 `WEB_REVIEW_CURRENT_SLICE`；它可以另写 `Proposed Next Action`，但 Proposed 不得冒充 Allowed。新的执行动作只能由 Web + 用户 review 后批准。

## 7. DESIGN_CONFLICT 与失败

冲突或 meaningful failure：

- 保存 raw error / debug-log；
- 停止 patch；
- 记录 modified files 和 rollback need；
- 不删 negative result；
- 不自行换设计；
- handoff 给 ChatGPT Web + 用户。

## 8. AUTO_TASK 限制

P1/P2 Core 的 M0–M5 默认不允许 AUTO_TASK。即使用户希望加快推进，也应先把 architecture-critical Parent Task 拆成明确 Slice，并逐 Slice 经过 Web review。AUTO_TASK 优先用于：已批准的 benchmark sweep、重复测试矩阵、profiler capture、机械 packaging / docs refresh。
