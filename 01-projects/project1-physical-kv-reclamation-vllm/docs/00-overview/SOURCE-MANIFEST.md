# P1 Source Manifest

## Technical Design Baseline

Primary input：P1-Physical-KV-Cache-Reclamation-for-vLLM-Reference-Backed-Design-v3-Implementation-Audited.md。

其技术结论优先于旧 QCache 文档。原输入未复制为 active Task；本目录把它拆成 overview/design/execution/evaluation。

## Implementation Source

- vLLM v0.26.0。
- commit 568afb3a13806beb53bb2e6bd518269357b237c0。
- local worktree/import identity：`worktrees/p1-vllm-reclaim/vllm`；当前 runner 为 MRV2 (`VLLM_USE_V2_MODEL_RUNNER=1`)。
- current MRV2 source map：`vllm/v1/worker/gpu/model_runner.py`、`gpu/states.py`、`gpu/block_table.py`；完整 F1-F7 结论见 `../90-reference/MRV2-PORT-RESOLUTION.md`。

## Reference Hierarchy

| Role | Repository / Commit | Provides | Does Not Provide |
|---|---|---|---|
| UPSTREAM | vLLM / 568afb3... | allocator、BlockPool、BlockTable、runner、slot、attention、Triton paged address | P1 reclaim transaction |
| REFERENCE | Tangram / 8e1cbfa... | effective occupancy、logical/slot split、worker ACK、copy compaction | drop-in v0.26 patch |
| REFERENCE | Sparse-vLLM / c146433... | ownership/test methodology | P1 runtime implementation |
| REFERENCE | newer vLLM / 0ecc284... | deferred free/lifetime hardening | P1 baseline dependency |
| REFERENCE | NVIDIA KVPress | periodic trigger/policy | allocator/ownership |

## Ownership

UPSTREAM PROVIDES 不写成 WE IMPLEMENT。P1 自己实现 scoped v0.26 reclaim subsystem、uniform state split、safe reuse、V2 compaction、tests/benchmark/profiling。

## Source Truth Rule

same-commit implementation > design doc/comment > external reference > hypothesis。current-main/newer snapshot 只能 REFERENCE。local fixed source 与 v3 audit 不同则 DESIGN_CONFLICT / source re-audit。

## Reference Gaps

明确保留：

- arbitrary keep-index Triton paged gather 没有 exact drop-in；
- current v0.26 attention effective-length adapter 是 API adaptation；
- preferred zero-copy integration 仍需 E2E oracle；
- performance benefit 需测。

每项都有 oracle/fallback，不把 gap 写成已解决。



## v1.2 Final-Contract Baseline

完整 v3 reasoning snapshot：`docs/90-reference/baselines/`（从项目根路径理解；若当前文件已在 docs 内则相对路径按目录导航）。Canonical corrections：`docs/90-reference/BASELINE-RESOLUTION.md`。Snapshot 是 read-only reference，不自动批准 implementation API。

## v1.2 Composite-Seam Audit Targets

“primitive 已存在”不等于“组合 transition 已证明”。M0–M2 必须显式关闭：

- MRV2 `BlockTables.append_block_ids(..., overwrite=True)` live reclaim 后 stale tail 是否可观察；
- manager/kernel block size 是否一对一；
- DCP/PCP/cascade 是否完全关闭；
- final-prefill reclaim 的 peak-capacity边界；
- V2 `keep_member_indices` 的 current-member index domain。

具体 reference 使用方式与 ownership 见 `../90-reference/REFERENCE-PORTING-LEDGER.md`。
