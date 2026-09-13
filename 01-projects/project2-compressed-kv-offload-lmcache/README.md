# Project 2 — Compressed KV Offload for vLLM + LMCache

副标题：Custom Block-INT8 L2 Serde and Data-Movement Crossover Analysis

第二项目，定位于 KV Reuse、External KV Cache、LMCache、Memory Hierarchy、GPU↔CPU Data Movement、Serde、Triton、PCIe / Storage Profiling。

当前状态 BLOCKED_BY_PROJECT1_CORE。文档存在不等于 runtime 已启动；不得冻结环境、创建 worktree、修改 vLLM/LMCache 或运行 baseline。

## One Sentence

在真实 vLLM KV Connector + LMCache reuse + filesystem L2 路径中，加入 fixed-size Block-INT8 Serde，确定性验证 serialized L2 bytes，并用 Triton/NSYS/NCU 分离 codec、I/O 与 data movement 的 crossover。

## Boundary

- V0：ABI / Connector reuse / raw filesystem L2。
- V1：custom Block-INT8 format + Serde + exact size + E2E load/resume。
- V2：Triton encode/decode + placement/crossover。
- V3：layerwise pipeline、resource pool、transfer scheduling；pre-D2H optional。

generic Serde 不自动减少 first D2H；later snapshot 不自动替换 implementation baseline。

入口：PROJECT_STATE → Master Plan → SOURCE-MANIFEST → design → execution。



## v1.2 Final-Contract Baseline

完整 v3 reasoning snapshot：`docs/90-reference/baselines/`（从项目根路径理解；若当前文件已在 docs 内则相对路径按目录导航）。Canonical corrections：`docs/90-reference/BASELINE-RESOLUTION.md`。Snapshot 是 read-only reference，不自动批准 implementation API。
