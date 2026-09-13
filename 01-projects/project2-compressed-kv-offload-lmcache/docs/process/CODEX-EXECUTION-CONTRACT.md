# P2 Codex Execution Contract

Default REVIEW_ONLY。P1-G3未accepted时禁止P2 runtime action。SLICE_EXECUTE只执行一个approved Slice；AUTO_TASK需明确授权。

Mandatory stops：pin/ABI/import mismatch、silent fallback、source/doc conflict、size invariant、temp lifecycle、slot/generation、unproven PCIe assumption、V3 five gates、dirty conflict。

Slice PASS≠Parent PASS；Codex不批准format/placement，不创建next Slice。使用templates并保留negative/failure。

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
