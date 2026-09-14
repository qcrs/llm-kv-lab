> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Architecture Decision Records (ADR)

状态：`PROPOSED/FROZEN` 表示进入对应 Slice 前不得无理由修改；若源码实验推翻，需要新增 superseding ADR，而不是静默改口径。

## ADR-001 — Base branch

**FROZEN:** 新 Ragged branch 从 P1/V2 pinned branch派生，但 P1 scalar path保持冻结 reference；Ragged不继续污染 scalar abstraction。

## ADR-002 — Global page namespace

**FROZEN:** Ragged physical page ID 在所有 local `(layer,head-group)` 间全局唯一。

## ADR-003 — One shared raw backing

**PROPOSED:** FullAttention Ragged Core使用一个 raw KV backing，被 local attention layers共享；page ID决定真正物理位置。

## ADR-004 — MRv2 virtual rows

**FROZEN for first implementation:** persistent block table内部以 `[R*G_total,Bmax]` row-oriented storage实现，对外暴露 group-aware API。

## ADR-005 — Separate RaggedKVState

**FROZEN:** `E[r,g]` 独立于 P1 `effective_kv_len[r]`；禁止同一个字段根据 mode改变 rank/shape。

## ADR-006 — Preserve split KV write/read seam

**FROZEN:** Ragged KV update继续由 `unified_kv_cache_update` 顺序化，attention read继续走 `unified_attention_with_output` eager split。

## ADR-007 — Ragged-specific column-major page layout

**PROPOSED:** BF16/FP16首版使用 `[B,Hp,N,2D]` contiguous，使 `(page,column)` zero-copy flatten；不改变 dense全局 cache layout。

## ADR-008 — No new FlashAttention mathematical kernel

**FROZEN:** 通过 virtual block/member sequences复用 FA varlen/paged primitive；只写 layout/adapter，不重写 attention math kernel。

## ADR-009 — Scheduler is ownership authority

**FROZEN:** Worker报告 completed physical shape；Scheduler从 canonical rows导出 freed IDs并执行 pool free。

## ADR-010 — Hard startup gates

**FROZEN:** 未审计 feature直接报错，不 silent fallback；Core only FullAttention/FA2/BF16-or-FP16/TP1/PP1/no-prefix/no-spec/no-DCP-PCP。

## ADR-011 — Piecewise CUDA Graph only

**FROZEN:** Full CG不是目标；Ragged attention可 eager，稳定模型片段 capture/replay。

## ADR-012 — Standard distributed breadth = TP2 uniform

**FROZEN:** Layer/global budget under TP2是 stretch；标准 breadth只保证 uniform count + cross-rank physical-depth reconciliation。

## ADR-013 — AOT staging

**FROZEN:** R7-A per-layer clustering是标准优化；cross-layer/global map只在 arbitrary cluster ownership已稳定后进入 R7-B。

## ADR-014 — Optimizations are evidence-driven

**FROZEN:** AOT关注 capacity；Triton关注 compaction movement；CG关注 CPU/launch。任何 optimization不以正 speedup作为项目完成依赖。
