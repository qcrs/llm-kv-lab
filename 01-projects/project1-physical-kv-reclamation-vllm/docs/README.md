# P1 Documentation Map

- 00-overview：定义、Master Plan、SOURCE-MANIFEST。
- 10-design：P1 v3 技术 baseline 的分层详细设计。
- 20-execution：Milestone、Parent Task、候选 Slice 分解与 Gate。
- 30-evaluation：correctness/capacity/profiling。
- 40-agent：Codex 启动、Task 手册、fallback。
- 50-delivery：reference ownership、交付/claim 审计。
- 90-reference：新 baseline traceability；不放旧 QCache active design。
- process：Web↔Codex、runtime sync、record templates。

阅读 current task 前先读项目 PROJECT_STATE。

## Current Implementation Records

- `04-experiments/project1_kv_reclaim/records/m1/P1-M1-T1-IMPLEMENTATION-INDEX.zh-CN.md`：`P1-M1-T1` 的 S1/S2/S3 统一入口。
- `04-experiments/project1_kv_reclaim/records/m1/P1-M1-T1-S1-IMPLEMENTATION-RECORD.zh-CN.md`：MRV2 persistent `effective_kv_len` state plumbing。
- `04-experiments/project1_kv_reclaim/records/m1/P1-M1-T1-S2-S3-IMPLEMENTATION-RECORD.zh-CN.md`：logical/physical write/read split 与 Web review-fix 最终状态。



## v1.2 Final-Contract Baseline

完整 v3 reasoning snapshot：`docs/90-reference/baselines/`（从项目根路径理解；若当前文件已在 docs 内则相对路径按目录导航）。Canonical corrections：`docs/90-reference/BASELINE-RESOLUTION.md`。Snapshot 是 read-only reference，不自动批准 implementation API。

## v1.2 Reference Entry

Reference provenance / porting boundary：`90-reference/REFERENCE-PORTING-LEDGER.md`。P1 Core benchmark 特别注意 final-prefill 后 reclaim 的 capacity scope。
