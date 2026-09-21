# P1 Artifact Index

只登记入口与用途，不复制 raw 内容。`PROJECT_STATE.md` 是当前状态权威文件。

Canonical V1 code checkpoint：`cd444b72ceecb2496bf015baf8c18bf883151fa1`；tag：`p1-v1-core-accepted`。

| Artifact | 类型 | 状态 | 用途 |
|---|---|---|---|
| `docs/process/runtime/P1-V1-CORE-END-TO-END-ARCHITECTURE.md` | canonical | FROZEN | S1→T3 runtime architecture entry |
| `docs/process/runtime/P1-V1-CORE-ACCEPTED-SOURCE-MANIFEST.md` | canonical | FROZEN | accepted production/test/harness files and symbols |
| `docs/process/runtime/P1-V1-CORE-SOURCE-IDENTITY.md` | canonical | FROZEN | upstream + dirty diff recovery identity |
| `docs/process/runtime/CURRENT-CONTEXT.md` | derived entry | CURRENT | 5–10 分钟状态恢复 |
| `docs/process/runtime/CHATGPT-HANDOFF.md` | handoff | CURRENT | V2 design-review handoff |
| `04-experiments/project1_kv_reclaim/notes/P1-V1-CORE-FREEZE-01.md` | canonical report | ACCEPTED | freeze decision and rules |
| `04-experiments/project1_kv_reclaim/notes/P1-V1-CORE-GATE-REVIEW-01.md` | gate report | ACCEPTED | I1–I18 Gate basis |
| `04-experiments/project1_kv_reclaim/notes/P1-M1-T3-IMPL-01A-Physical-Ownership-Reconciliation-实施记录.md` | supporting report | PASS | T3 implementation oracles |
| `04-experiments/project1_kv_reclaim/notes/P1-M1-T3-INTEGRATION-CLOSURE-真实Engine生命周期验证报告.md` | supporting report | PASS | real Engine lifecycle closure |
| `04-experiments/project1_kv_reclaim/raw/p1-v1-core-gate-review-01/` | raw evidence | PRESERVED | Gate source/test audit logs |
| `04-experiments/project1_kv_reclaim/raw/m1-t3-impl-01a/` | raw evidence | PRESERVED | T3 targeted/regression logs |
| `04-experiments/project1_kv_reclaim/raw/m1-t3-integration-closure/` | raw evidence | PRESERVED | GPU2 real Engine smoke |
| `04-experiments/project1_kv_reclaim/raw/p1-v1-core-freeze-01/` | raw evidence | CURRENT | repository identity and freeze checks |
| `04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01b-web-review-fix/` | historical/supporting | PRESERVED | T2 transport closure |
| `04-experiments/project1_kv_reclaim/raw/p1-v1-core-checkpoint-doc-sync-01/` | raw evidence | CURRENT | checkpoint identity and document sync |
| `04-experiments/project1_kv_reclaim/notes/P1-V2-R1-C-EXECUTION-FOUNDATION-01-Code-Trace.md` | supporting report | PASS_PENDING_WEB_REVIEW | C0–C4 execution-addressing change trace |
| `04-experiments/project1_kv_reclaim/raw/p1-v2-r1-c-execution-foundation-01-final/` | raw evidence | CURRENT | C0–C4 focused tests, source identity, layout/address evidence |
| `04-experiments/project1_kv_reclaim/handoffs/LATEST/` | handoff | CURRENT | current Slice state, diff, files and test summary |

## 分类规则

`canonical` 是后续 V1/V2 review 的首要入口；`supporting` 提供 slice 细节；`raw evidence`
保留不可翻译的原始命令输出；`historical` 不得覆盖当前 accepted 结论。
