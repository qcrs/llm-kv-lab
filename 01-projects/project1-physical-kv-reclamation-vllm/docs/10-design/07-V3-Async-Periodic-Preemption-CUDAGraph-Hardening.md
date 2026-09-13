# 07 — P1 V3 Runtime Hardening / Optimization

V3 不定义项目成败。只有 V1 V1_RUNTIME_CORE_DONE + V2 P1_MAIN_PROJECT_STRONG_DONE 后，且 user + Web 明确 GO，才进入。

## Priority A — Async-safe Deferred Free

newer vLLM 已提供 processed-safe boundary / deferred-free analogous reference。目标不是速度，而是：

```text
async scheduling on
+ request-specific in-flight fence
+ no stale page alias
+ candidate page eventually reusable
```

禁止用 global device synchronize伪装 lifetime correctness。

## Priority B — Periodic Decode Reclaim

先做 whole-block periodic path：

```text
compression_interval
target_physical_tokens
min_reclaimable_blocks / hysteresis
```

触发语义可借 KVPress `DecodingPress`；physical ownership/lifetime仍由 P1 runtime contract负责。稳定后才让 token-level V2 compaction挂 periodic trigger。

## Priority C — Preemption / Resume

Preemption 丢失 old physical cache 时，reclaim physical mapping应 reset；resume 使用现有 full block-list replacement，并重新建立 effective state。只有通过该 compatibility test 后，benchmark 才允许声称 preemption reduction。

## Priority D — CUDA Graph

优先固定 buffer address/shape，以 in-place metadata content update 兼容 graph。成功标准是 correctness + recapture behavior可解释，不是必然 speedup。

## Priority E — Fused Paged→Paged Compaction

只在 NCU/NSYS证明 scratch write/read 是 dominant overhead 时进入。最大风险是 source/destination overlap race。优先只支持可证明 non-overlap case；若需要复杂 general dependency scheduler，保留 staged V2。

## Priority F — One Better Policy

最多增加一个 non-trivial policy（如 K-norm/attention-score）证明 runtime policy-agnostic。不要把项目转成 eviction algorithm benchmark。

## Priority G — Ragged per-layer/per-head

Tangram 有完整 reference，但这是第二套更大 architecture：effective length从 scalar/request变成 per-layer/head-group，连带改变 allocator、BlockTable、metadata、worker result。只有时间明显富余才做。
