# Workspace Documentation Index

## Start Here

Read the root `AGENTS.md` and `CURRENT.md` first. The active projects are P1
Physical KV Cache Reclamation and P2 Compressed KV Offload; archived project
designs live under `../archive/` and are historical only.

- roadmap：跨项目顺序、工程 Gate、协作矩阵、Task/Slice/状态规则。
- design-notes：三个 accepted 高层 ADR；不含 implementation approval。
- source-reading：源码 walkthrough 的写作与证据规则。
- profiling-notes：优化记录索引与模板。
- weekly：周计划/复盘，不改变 PROJECT_STATE。
- interview：只登记已验证 delivery evidence。
- context-packs：网页版快速接入说明；均为 derived context。

项目文档入口：

- ../01-projects/project1-physical-kv-reclamation-vllm/
- ../01-projects/project2-compressed-kv-offload-lmcache/

状态入口：

- ../CURRENT.md
- Project 1 PROJECT_STATE.md
- Project 2 PROJECT_STATE.md



## v1.2 Final-Contract Baseline Entries

- P1: `../01-projects/project1-physical-kv-reclamation-vllm/docs/90-reference/BASELINE-RESOLUTION.md`
- P1 full snapshot: `../01-projects/project1-physical-kv-reclamation-vllm/docs/90-reference/baselines/`
- P1 support matrix: `../01-projects/project1-physical-kv-reclamation-vllm/docs/00-overview/SUPPORTED-CONFIG-MATRIX.md`
- P2: `../01-projects/project2-compressed-kv-offload-lmcache/docs/90-reference/BASELINE-RESOLUTION.md`
- P2 full snapshot: `../01-projects/project2-compressed-kv-offload-lmcache/docs/90-reference/baselines/`
- P2 support matrix: `../01-projects/project2-compressed-kv-offload-lmcache/docs/00-overview/SUPPORTED-CONFIG-MATRIX.md`

- Final public-source revalidation: `source-reading/GITHUB-REVALIDATION-v1.2.md`
- P1 porting ledger: `../01-projects/project1-physical-kv-reclamation-vllm/docs/90-reference/REFERENCE-PORTING-LEDGER.md`
- P2 porting ledger: `../01-projects/project2-compressed-kv-offload-lmcache/docs/90-reference/REFERENCE-PORTING-LEDGER.md`
- P2 FS benchmark regimes: `../01-projects/project2-compressed-kv-offload-lmcache/docs/30-evaluation/FILESYSTEM-BENCHMARK-REGIMES.md`
