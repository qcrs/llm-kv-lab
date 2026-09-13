# Migration Notes

## Purpose

本包继承旧 llm-kv-lab 的协作“操作系统”，但重置两个项目的技术身份。旧包是 process/template reference，不是新技术 baseline。

## Preserved Process Primitives

- root AGENTS + project AGENTS layering；
- CURRENT + authoritative PROJECT_STATE；
- ChatGPT Web / Work ↔ Codex CLI 分工；
- Parent Task / Slice / Gate；
- REVIEW_ONLY / SLICE_EXECUTE / AUTO_TASK；
- Source Fact / Design Draft / Approved Design；
- DESIGN_CONFLICT stop；
- source-first、predict-before-patch、Teaching-First；
- one minimal Task / Slice at a time；
- runtime context sync；
- CURRENT-CONTEXT、CHATGPT-HANDOFF、SOURCE-IDENTITY、ENVIRONMENT-SNAPSHOT；
- Artifact Index 与 LATEST handoff pack；
- context compression rule；
- learning / decisions / evidence / delivery 分离；
- experiment manifest、negative result、fallback、Gate evidence；
- Task Card、source audit、slice record、experiment record。

## Explicitly Not Inherited

- old QCache technical architecture；
- native INT4 canonical + BF16 recent shadow；
- old mixed attention / LSE merge / RHT milestones；
- old physical demotion roadmap；
- old Trace-Driven KV Retention / T-LRU / simulator architecture；
- old project task IDs、states、results and performance claims；
- old QOffload or nano-vLLM implementation assumptions；
- any old Approved Design that conflicts with P1/P2 v3。

## New Technical Baselines

P1：Physical KV Cache Reclamation for vLLM，以 logical/physical decoupling、whole-block zero-copy reclaim、worker ACK 后 safe-free、real BlockPool reuse、V2 staged Triton paged-KV compaction 为主线。

P2：Compressed KV Offload for vLLM + LMCache，以真实 Connector/reuse、filesystem L2、custom fixed-size Block-INT8 Serde、exact serialized bytes、V2 Triton codec 与 data-movement crossover 为主线。

## Conflict Rule

- 技术冲突：P1/P2 v3 > 旧项目文档。
- process 冲突：orientation active root/project AGENTS 与 process > 旧 package template。
- source 冲突：fixed source / raw evidence > 所有设计和 prompt。

旧输入不会复制进 90-reference 作为 active 文档，避免后续 Codex 被旧项目名称和 claim 污染。新 SOURCE-MANIFEST 只登记其“结构参考”角色。

## Workspace Migration Execution (2026-08-24)

本次 v1.2 contract 已安装到 workspace 根层。实际归档映射如下：

| 原 active 路径 | 归档路径 | 处理 |
|---|---|---|
| `01-projects/project1-qcache-vllm/` | `archive/legacy-projects/project1-qcache-vllm/` | 整体移动，含旧 docs、learning、worktree 与项目实验 |
| `01-projects/project2-kv-retention/` | `archive/legacy-projects/project2-kv-retention/` | 整体移动，含旧 docs、learning 与项目实验 |
| `04-experiments/project1_pkv/` | `archive/legacy-projects/project1_pkv/` | raw/processed 历史 evidence 保留 |
| `04-experiments/project2_kv_retention/` | `archive/legacy-projects/project2_kv_retention/` | raw/processed 历史 evidence 保留 |
| `00-docs/roadmap/` | `archive/legacy-projects/legacy-00-docs-roadmap/` | 旧路线整体归档 |
| `00-docs/design-notes/` | `archive/legacy-projects/legacy-00-docs-design-notes/` | 旧 ADR 整体归档 |

ZIP 中的 `00-docs/`、`01-projects/`、`04-experiments/`、`05-models/` 作为 active
control plane 安装；现有 source-reading、profiling、weekly、interview 和通用
学习资料保留并与新入口合并。没有修改 `third_party/`，没有运行 benchmark、NSYS、NCU
或任何实现 Slice。

文档不建立 SHA256、immutable manifest、content-hash gate 或 doc-lock；Git history
和本迁移记录提供版本追踪。



## v1.1 Hardening Migration

Unlike v1, v1.1 embeds the full P1/P2 v3 technical baselines to prevent reasoning loss. `BASELINE-RESOLUTION.md` records superseding interpretations without mutating historical snapshots. Escaped Unicode filenames were normalized. No old QCache/KV Retention technical architecture was reintroduced.
