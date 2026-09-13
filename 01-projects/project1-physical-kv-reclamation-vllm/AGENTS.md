# Project 1 Agent Contract

## Documentation / Evidence Requirements

除临时对话中的思考过程外，P1 所有新建或修改的工程文档、source-reading、实验/验证报告、
Task/Slice record、code change record、handoff 和状态说明必须使用中文；源码 symbol、
路径、命令、raw stdout/stderr、Git/profiler 原始输出保持英文。`raw/*.log` 不翻译、
不改写，报告只引用关键观察和日志路径。英文模板不得直接作为正式记录交付。

正式 P1 validation 或 Gate report 必须说明：验证动机、问题、假设/invariant、完整
环境与版本、测试设计、执行命令、`OBSERVED` 原始结果、结果推理、未覆盖范围、
remaining `TO_VERIFY` 以及下一步的权限理由。推荐章节为：背景与动机、问题、假设、
环境、测试设计、执行、观察、分析、证明内容、未证明内容、Gate 关系、TO_VERIFY、
Evidence Index、下一步。

P1 source-reading 必须围绕 ownership、persistent/per-step state、data/control flow
和 source anchors 组织；对关键 symbol 记录创建者、修改者、生命周期、消费者、
shape/device、publication point 与 P1 relevance，而不是只列函数名称。

继承工作区根 AGENTS.md。技术事实冲突遵循 fixed source/raw evidence → accepted ADR/APPROVED_DESIGN → PROJECT_STATE → Master Plan → Task Card；行为遵循 root AGENTS → workspace matrix → 本文件 → Task Card → Slice Prompt。

## Scope

### Current runner identity

P1 当前 runtime target 是 MRV2：`VLLM_USE_V2_MODEL_RUNNER=1`，source 为
`vllm/v1/worker/gpu/model_runner.py`。`vllm/v1/worker/gpu_model_runner.py`、
`CachedRequestState`、旧 `GPUInputBatch` seam 只可作为 historical reference；
当前映射与 F1-F7 结论见 `docs/90-reference/MRV2-PORT-RESOLUTION.md`。

P1 只处理：

- vLLM v0.26.0 pinned source；
- logical/physical KV state split；
- whole-block reclaim、BlockTable row reconstruction；
- worker ACK → scheduler ownership commit → BlockPool reuse；
- effective allocation / slot mapping / attention KV length；
- V2 PyTorch/Triton compaction；
- correctness/capacity/profiling/delivery；
- optional V3 hardening only after Core/V2 Gate。

不处理 LMCache、P2 runtime、RDMA、multi-node、TP/PP/DP>1、prefix caching/spec/async/CUDA Graph（MVP）、ragged per-head/layer paging或复杂 policy research。

## Project-Specific Hard Invariants

1. logical_num_computed_tokens 不因 reclaim 回退。
2. logical model/RoPE position 不重编号。
3. physical_cache_position 从 effective/physical_kv_len 追加。
4. attention 只看到 retained physical KV length，但非 attention consumer 不被全局 physical seq_len 污染。
5. worker commit 是第一 commit point；scheduler canonical ownership commit 是第二 commit point。
6. BlockPool free 只能发生在 worker 不再引用 candidate pages 后。
7. freed IDs 必须验证属于该 request，禁止 duplicate/null/foreign free。
8. allocation 按 physical/effective occupancy 继续增长。
9. feature-off identity 与 feature-on/retention=1 no-op identity 是两类独立 regression gate。
10. free BlockPool page 不代表 CUDA allocation 返回系统。

## Modes

当前 REVIEW_ONLY，Current Slice NONE。候选 task/slice decomposition 不是授权。任何 source 修改必须有 Approved Slice ID、Approved Design、allowed files、tests、evidence 与 stop conditions。

## P1 DESIGN_CONFLICT Triggers

- v0.26 source 没有所宣称的 BlockTable/runner transition；
- attention backend 无法以最小方式消费 effective KV length；
- scheduler/worker commit ordering 与 source lifecycle 冲突；
- candidate page 仍可能被 in-flight step 访问；
- prefix/shared ownership 未关闭或无法证明；
- patch 需要把 P1 扩展为 Tangram ragged architecture。

触发后按根契约停止，不自行降级；fallback 也需要 Web + 用户确认。

## Session Read First

CURRENT → REPO_PINS → 本项目 PROJECT_STATE → Master Plan → current Parent Task → SOURCE-MANIFEST → 1–3 exact source anchors → git/env/source identity。

## Records

- read-only audit：CODEX-SOURCE-AUDIT；
- approved source change：CODEX-SLICE-RECORD；
- experiment：CODEX-EXPERIMENT-RECORD；
- handoff：runtime/CHATGPT-HANDOFF + 04-experiments/project1_kv_reclaim/handoffs/LATEST。

结束不得自动进入下一 Slice或标 Parent Task PASS。



## Baseline Reading Rule

技术设计 review / source audit 前，按顺序读取：

1. `docs/90-reference/BASELINE-RESOLUTION.md`；
2. 当前相关 `docs/10-design/*`；
3. 需要完整 reasoning 时读取 `docs/90-reference/baselines/*-BASELINE.md`；
4. 最终以 fixed local source / raw evidence 为准。

不得因为 split design doc 没有重复 baseline 的全部 reasoning，就把已经 reference-backed 的 contract重新当成空白设计。

## v1.2 Additional Core Gates

V1/V2 Core 还要求：

```text
DCP = 1
PCP = 1
kernel_block_size == kv_manager_block_size
blocks_per_kv_block == 1
cp_kv_cache_interleave_size == 1
cascade attention = OFF / explicitly unsupported until separately audited
```

并明确：Core reclaim 发生在 final-prefill commit 之后，因此 **不降低该 request 首次 full-prefill 的 peak KV requirement，也不自动提高 single-request maximum prefill context**。项目确定收益发生在 reclaim point 之后的 allocator capacity/reuse。

MRV2 `BlockTables.append_block_ids(..., overwrite=True)` 只证明 row replacement primitive 存在；live reclaim 还必须在 M1/M2 证明旧 active-row tail 不可被后续 consumer 观察，或采用 clear-old-range + rebuild 的薄 replacement semantic。不得把 primitive 存在等价成组合 transition 已完全 proven。
