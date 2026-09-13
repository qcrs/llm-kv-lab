# Project 2 Agent Contract

继承 root AGENTS 与 workspace collaboration matrix。

## Blocked State

P2 runtime = BLOCKED_BY_PROJECT1_CORE。P1 G3 V1_RUNTIME_CORE_DONE accepted 前，仅允许阅读和文档 review。禁止 P2 worktree/environment/ABI probe、connector run、source patch、experiment。解锁后仍从 P2-M0 review 开始，不自动执行。

## Scope After Unlock

- vLLM 568afb3...；
- LMCache a7afade... compatibility baseline；
- torch 2.11 native ABI Gate；
- LMCacheConnectorV1 / RequestTracker / slot mapping；
- local filesystem L2 raw baseline；
- fixed-size Block-INT8 Serde；
- fixed-size + FSL2 scoped serialized byte invariant；
- V2 PyTorch/Triton codec and actual placement；
- crossover/NSYS/NCU；
- optional V3 reference-backed pipeline/resource/transfer。

不做 RDMA/Mooncake/NIXL/multi-node/P-D、重新设计 connector/async framework、复杂 quant research、默认 L1 policy、未证据 first-D2H claim。

## Hard Invariants

1. vLLM/LMCache/torch/native extension identity一致，无 silent fallback。
2. V0 connector、raw FS-L2 与 V1 codec 分层 debug。
3. MemoryObj layout从 fixed source/runtime freeze，不猜。
4. fixed V1 format layout/size deterministic。
5. V1 fixed-size + FSL2：per-object `estimate == temp bytes == .data bytes`；per-store `sum(new writes) == bytes_transferred`，并控制 already-existing keys。
6. L2 bytes、L1 CPU footprint、GPU↔CPU PCIe bytes 分开。
7. generic Serde只承诺 L1→L2 serialized representation。
8. Serializer/Deserializer遵守 AsyncSerdeProcessor/wrapper lifecycle，不重写异步系统。
9. CPU/CUDA staging实际 placement由 profiler决定。
10. pre-D2H 只有 V3 gate 后可做。

## Authority

Codex可发现 source/runtime facts和提出 draft，不能批准 format/placement/pipeline。没有 Approved Slice时 REVIEW_ONLY。P2 blocked时即使用户说“进入 P2-M0”，仍需先确认 P1 unlock。

## DESIGN_CONFLICT

ABI/pin不兼容、code与design doc漂移、MemoryObj device/layout与设计冲突、estimator/file size contract不成立、generic Serde source已经/尚未 CPU staged 与批准假设不同，均停止并按根模板交接。

## Records

使用本项目 process templates与 04-experiments/project2_kv_offload。Parent Task/Slice/next规则同根契约。



## Baseline Reading Rule

技术设计 review / source audit 前，按顺序读取：

1. `docs/90-reference/BASELINE-RESOLUTION.md`；
2. 当前相关 `docs/10-design/*`；
3. 需要完整 reasoning 时读取 `docs/90-reference/baselines/*-BASELINE.md`；
4. 最终以 fixed local source / raw evidence 为准。

不得因为 split design doc 没有重复 baseline 的全部 reasoning，就把已经 reference-backed 的 contract重新当成空白设计。

## v1.2 Codec-Domain Rule

P2 的 `Block-INT8` 中 “Block” 默认指 **codec quantization block**，不是 vLLM `BlockPool` physical page。Core format优先在 LMCache `MemoryLayoutDesc` object domain 定义：

```text
K/V × layer × KV head × fixed token group × head_dim
```

除非 fixed source 明确要求，否则不得为了“对齐 vLLM page”把 Serde重新耦合回 allocator layout。

Filesystem L2 的 deterministic claim 是 serialized/file bytes 下降；`FSL2Adapter` 本身没有固定 capacity admission，因此“固定预算下能装更多 object”只能写成外部 byte-budget 的 derived result，不得写成 backend 自动 admission capacity。
