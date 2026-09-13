# Package Manifest — v1.2 Final Contract

## Identity

- Package: `llm-kv-lab-p1-reclaim-p2-offload-docs-v1.2-final-contract`
- Generated / final-contract hardened: `2026-08-24 Asia/Taipei`
- Purpose: two-project documentation / collaboration / state / evidence / Web↔Codex handoff workspace
- Runtime source code changes: **NONE**
- Formal experiments / profiler captures: **NONE**
- Approved implementation slices: **NONE**
- Current mode: **REVIEW_ONLY**
- P2 runtime: **BLOCKED_BY_PROJECT1_CORE**

## Technical Baseline Preservation

- P1/P2 full v3 audited reasoning snapshots are preserved under each project `docs/90-reference/baselines/`.
- `BASELINE-RESOLUTION.md` defines canonical superseding interpretations without rewriting historical snapshots.
- Historical snapshots are `READ_ONLY_REFERENCE_SNAPSHOT`; fixed local source/raw evidence remains the highest technical authority.

### Snapshot checksums

| Snapshot | Bytes | SHA256 |
|---|---:|---|
| P1 v3 | 87143 | `a798f4cb42b990b917ca390ed6998e883607bb06cd44d906a8e281d75c07fc6b` |
| P2 v3 | 74599 | `2d12e222974bf644b7cef790e1122630fd4f9d97ef70627e5c02bdad7626bbdf` |

## v1.2 Final-Contract Hardening

1. P1 reclaim scope explicitly starts after final-prefill commit; no first-prefill peak-KV or single-request max-context claim.
2. P1 Core freezes DCP/PCP=1, manager/kernel block size 1:1, `blocks_per_kv_block=1`, CP interleave=1, and cascade attention off until separately audited.
3. `BlockTable.add_row()` remains an exact primitive, while live shorter-row stale-tail observability is a mandatory M1/M2 integration seam.
4. P1 feature-off identity and feature-on/retention=1 no-op identity are separate regression gates.
5. P1 V2 uses `keep_member_indices` in the current retained/member-sequence index domain.
6. P1 capacity evaluation includes rolling/staggered arrivals, prefill peak, post-reclaim ownership, recovery latency and simultaneous-prefill negative regime.
7. P2 `Block-INT8` Block is a LMCache object-domain codec quantization block: K/V × layer × KV head × fixed token group × head_dim; no required vLLM BlockPool-page alignment.
8. P2 FSL2 capacity language is scoped: serialized/file bytes are runtime facts; object count under fixed bytes is an external-quota/derived result.
9. P2 FS crossover requires buffered-warm, large-working-set/new-key, or verified-O_DIRECT regime labels.
10. P2 exact `.data` equality uses logical `stat.st_size`, not `du`/physical allocated blocks.
11. Codex PROJECT_STATE writes are factual-only; after an execution Slice the allowed next step returns to `WEB_REVIEW_CURRENT_SLICE`.
12. Core M0–M5 defaults to Slice-level collaboration; AUTO_TASK is reserved for approved mechanical work.
13. Governance/authority files are read-only during normal implementation so an execution Agent cannot self-modify policy.
14. P1/P2 include `REFERENCE-PORTING-LEDGER.md` to track API/semantic/test/kernel/code provenance and attribution.
15. P2 source/test references include pinned LMCache real filesystem Serde E2E and TurboQuant object-layout/size tests.

## Public-Source Revalidation

- Current targeted public-source revalidation: `00-docs/source-reading/GITHUB-REVALIDATION-v1.2.md`.
- Historical `GITHUB-REVALIDATION-v1.1.md` is retained for audit history.
- Public-source revalidation does not replace local M0 source/environment/import/runtime freeze.

## Authority / Collaboration

```text
fixed local source / raw evidence
→ accepted ADR / APPROVED_DESIGN / PROJECT_STATE
→ BASELINE-RESOLUTION
→ current design / Master Plan / Task Card
→ SOURCE-MANIFEST / REFERENCE-PORTING-LEDGER
→ historical v3 baseline snapshot
```

- ChatGPT Web/Work + user own architecture, Gate decisions, next-Slice approval and cross-project unlock.
- Codex is repository execution/evidence agent; it cannot approve designs, Gates, next Slices or P2 unlock.
- Root `CURRENT.md` is cross-project control and is not modified by Codex during normal implementation.

## Runtime Sequence

- P1 is the only active runtime project.
- Current Parent Task: `P1-M0-T1`.
- Current Slice: `NONE`.
- `P1-G3 V1_RUNTIME_CORE_DONE` is the P2 runtime review unlock gate, not full P1 completion.
- `P1-G5 P1_MAIN_PROJECT_STRONG_DONE` is the strong P1 completion gate including V2.

## File Inventory

- `00-docs/INDEX.md`
- `00-docs/context-packs/PACK-MANIFEST.md`
- `00-docs/context-packs/工作区目录结构与网页版认知说明.md`
- `00-docs/design-notes/ADR-001-P1-Physical-KV-Reclamation.md`
- `00-docs/design-notes/ADR-002-P2-Compressed-KV-Offload.md`
- `00-docs/design-notes/ADR-003-P1-P2执行顺序与Scope.md`
- `00-docs/design-notes/README.md`
- `00-docs/interview/README.md`
- `00-docs/profiling-notes/INDEX.md`
- `00-docs/profiling-notes/优化记录模板.md`
- `00-docs/roadmap/00-跨项目总控执行手册.md`
- `00-docs/roadmap/01-共同工程规范与环境门.md`
- `00-docs/roadmap/02-ChatGPT-Web-Work-Codex协作与文档角色矩阵.md`
- `00-docs/roadmap/03-P1-P2文档地图与学习路径.md`
- `00-docs/roadmap/04-Task-Slice-Gate与状态规则.md`
- `00-docs/roadmap/README.md`
- `00-docs/roadmap/TASKS.md`
- `00-docs/source-reading/GITHUB-REVALIDATION-v1.1.md`
- `00-docs/source-reading/GITHUB-REVALIDATION-v1.2.md`
- `00-docs/source-reading/README.md`
- `00-docs/weekly/README.md`
- `01-projects/README.md`
- `01-projects/project1-physical-kv-reclamation-vllm/AGENTS.md`
- `01-projects/project1-physical-kv-reclamation-vllm/PROJECT_STATE.md`
- `01-projects/project1-physical-kv-reclamation-vllm/README.md`
- `01-projects/project1-physical-kv-reclamation-vllm/decisions/README.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/00-overview/00-项目总览与导航.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/00-overview/Project1-Master-Plan.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/00-overview/SOURCE-MANIFEST.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/00-overview/SUPPORTED-CONFIG-MATRIX.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/10-design/01-问题定义-目标-贡献边界.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/10-design/02-vLLM-v0.26固定源码与Paged-KV-Reclaim-Source-Map.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/10-design/03-V1-Whole-Block-Zero-Copy-Reclamation详细设计.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/10-design/04-Logical-Physical-KV-State与Attention-Slot-Semantics.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/10-design/05-KV-Ownership-Allocation-Worker-Scheduler-ACK-Safe-Free.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/10-design/06-V2-Token-Level-Triton-Paged-KV-Compaction详细设计.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/10-design/07-V3-Async-Periodic-Preemption-CUDAGraph-Hardening.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/20-execution/00-执行控制与Gate总览.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/20-execution/01-M0-M1-Parent-Task-Cards.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/20-execution/02-M2-M3-Parent-Task-Cards.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/20-execution/03-M4-M7-Parent-Task-Cards.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/20-execution/04-Slice拆分与Task-Card模板.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/30-evaluation/Capacity-Serving-Benchmark完整评测方案.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/30-evaluation/Correctness-Oracle-与-Multi-Request-Reuse验收.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/30-evaluation/Nsight-Systems-NCU-Profiling手册.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/40-agent/00-Codex-CLI启动与教学流程.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/40-agent/10-Agent-Task执行手册.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/40-agent/11-风险-Fallback-Decision-Tree.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/50-delivery/12-Reference与Ownership索引.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/50-delivery/13-最终交付-简历-面试Evidence验收.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/90-reference/BASELINE-RESOLUTION.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/90-reference/README.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/90-reference/REFERENCE-PORTING-LEDGER.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/90-reference/baselines/P1-v3-Implementation-Audited-BASELINE.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/90-reference/baselines/README.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/README.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/CHATGPT-CODEX-WORKFLOW.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/CODEX-EXECUTION-CONTRACT.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/history/README.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/runtime/ARTIFACT-INDEX.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/runtime/CHATGPT-HANDOFF.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/runtime/CURRENT-CONTEXT.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/runtime/ENVIRONMENT-ACTIVATION.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/runtime/ENVIRONMENT-SNAPSHOT.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/runtime/SOURCE-IDENTITY.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/templates/CODEX-EXPERIMENT-RECORD.template.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/templates/CODEX-SLICE-RECORD.template.md`
- `01-projects/project1-physical-kv-reclamation-vllm/docs/process/templates/CODEX-SOURCE-AUDIT.template.md`
- `01-projects/project1-physical-kv-reclamation-vllm/learning/README.md`
- `01-projects/project2-compressed-kv-offload-lmcache/AGENTS.md`
- `01-projects/project2-compressed-kv-offload-lmcache/PROJECT_STATE.md`
- `01-projects/project2-compressed-kv-offload-lmcache/README.md`
- `01-projects/project2-compressed-kv-offload-lmcache/decisions/README.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/00-overview/00-项目总览与导航.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/00-overview/Project2-Master-Plan.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/00-overview/SOURCE-MANIFEST.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/00-overview/SUPPORTED-CONFIG-MATRIX.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/10-design/01-问题定义-目标-贡献边界.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/10-design/02-vLLM-LMCache固定版本-ABI-Connector-Source-Map.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/10-design/03-V0-Connector-Reuse与Filesystem-L2-Baseline.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/10-design/04-V1-Block-INT8-Serde-Format与Object-Lifecycle.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/10-design/05-V2-Triton-Codec-Placement与Crossover详细设计.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/10-design/06-V3-Layerwise-Pipeline-Resource-Pool-Transfer-Scheduling.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/10-design/07-V3-Pre-D2H-Compressed-Transfer-Optional-Design.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/20-execution/00-执行控制与Gate总览.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/20-execution/01-M0-M2-Parent-Task-Cards.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/20-execution/02-M3-M5-Parent-Task-Cards.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/20-execution/03-M6-M7-Parent-Task-Cards.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/20-execution/04-Slice拆分与Task-Card模板.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/30-evaluation/Block-INT8-Correctness-与-Serialized-Bytes验收.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/30-evaluation/Codec-IO-Crossover-Benchmark完整方案.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/30-evaluation/FILESYSTEM-BENCHMARK-REGIMES.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/30-evaluation/Nsight-Data-Movement与NCU-Profiling手册.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/40-agent/00-Codex-CLI启动与教学流程.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/40-agent/10-Agent-Task执行手册.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/40-agent/11-风险-Fallback-Decision-Tree.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/50-delivery/12-Reference与Ownership索引.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/50-delivery/13-最终交付-简历-面试Evidence验收.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/90-reference/BASELINE-RESOLUTION.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/90-reference/README.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/90-reference/REFERENCE-PORTING-LEDGER.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/90-reference/baselines/P2-v3-Implementation-Audited-BASELINE.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/90-reference/baselines/README.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/README.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/CHATGPT-CODEX-WORKFLOW.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/CODEX-EXECUTION-CONTRACT.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/history/README.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/runtime/ARTIFACT-INDEX.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/runtime/CHATGPT-HANDOFF.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/runtime/CURRENT-CONTEXT.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/runtime/ENVIRONMENT-ACTIVATION.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/runtime/ENVIRONMENT-SNAPSHOT.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/runtime/SOURCE-IDENTITY.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/templates/CODEX-EXPERIMENT-RECORD.template.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/templates/CODEX-SLICE-RECORD.template.md`
- `01-projects/project2-compressed-kv-offload-lmcache/docs/process/templates/CODEX-SOURCE-AUDIT.template.md`
- `01-projects/project2-compressed-kv-offload-lmcache/learning/README.md`
- `04-experiments/README.md`
- `04-experiments/project1_kv_reclaim/README.md`
- `04-experiments/project1_kv_reclaim/handoffs/LATEST/CHATGPT-HANDOFF.md`
- `04-experiments/project1_kv_reclaim/handoffs/LATEST/DIFF-SUMMARY.md`
- `04-experiments/project1_kv_reclaim/handoffs/LATEST/FILES.txt`
- `04-experiments/project1_kv_reclaim/handoffs/LATEST/PROJECT_STATE-SNAPSHOT.md`
- `04-experiments/project1_kv_reclaim/handoffs/LATEST/TEST-SUMMARY.md`
- `04-experiments/project1_kv_reclaim/results/README.md`
- `04-experiments/project2_kv_offload/README.md`
- `04-experiments/project2_kv_offload/handoffs/LATEST/CHATGPT-HANDOFF.md`
- `04-experiments/project2_kv_offload/handoffs/LATEST/DIFF-SUMMARY.md`
- `04-experiments/project2_kv_offload/handoffs/LATEST/FILES.txt`
- `04-experiments/project2_kv_offload/handoffs/LATEST/PROJECT_STATE-SNAPSHOT.md`
- `04-experiments/project2_kv_offload/handoffs/LATEST/TEST-SUMMARY.md`
- `04-experiments/project2_kv_offload/results/README.md`
- `05-models/README.md`
- `AGENTS.md`
- `CHANGELOG-v1.1.md`
- `CHANGELOG-v1.2.md`
- `CURRENT.md`
- `DOCUMENTATION-AUDIT.md`
- `MIGRATION-NOTES.md`
- `PACKAGE-MANIFEST.md`
- `README.md`
- `REPO_PINS.md`
- `TREE.txt`
