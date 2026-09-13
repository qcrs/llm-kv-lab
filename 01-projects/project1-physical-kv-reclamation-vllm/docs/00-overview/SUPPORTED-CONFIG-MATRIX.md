# P1 Supported Configuration Matrix

此表是 implementation gate，不是“未来可能支持”的愿望清单。M0 必须把 TO_FREEZE 项冻结为具体 evidence。

| Dimension | V1 Runtime Core | V2 Triton Core | V3 / Later |
|---|---|---|---|
| vLLM | v0.26.0 / `568afb3...` | same | newer source only as reference |
| GPU | A100 80GB | A100 80GB | same first |
| Topology | single GPU, TP=PP=DP=DCP=PCP=1 | same | distributed later only if explicitly approved |
| KV manager ↔ kernel block | `kernel_block_size == kv_manager_block_size`, `blocks_per_kv_block == 1` | same | hybrid block splitting requires separate audit |
| CP interleave | `cp_kv_cache_interleave_size == 1` | same | later only |
| Cascade attention | OFF / exact pinned flag TO_FREEZE | OFF | separate compatibility audit |
| Model | **Qwen3-family first target, exact model TO_FREEZE** | same | other model families require positional-semantic audit |
| Positional semantics | standard RoPE decoder path | same | ALiBi/relative bias/mRoPE/dual-chunk TO_VERIFY |
| Attention | causal dense full attention | same | other attention types OUT |
| Attention backend | exact pinned FA backend TO_FREEZE | same | backend portability later |
| KV groups | exactly 1 | exactly 1 | ragged/multi-group V3-G |
| KV dtype | BF16 | BF16 source/dest | quantized KV-cache modes out |
| Prefix caching | OFF | OFF | separate future compatibility |
| Spec decode | OFF | OFF | future only |
| Async scheduling | OFF | OFF | V3-A deferred-free |
| CUDA Graph | eager / graph OFF | eager first | V3-D |
| Reclaim trigger | after final prefill forward commit | same | periodic decode V3-B |
| First-prefill peak KV | **not reduced by Core** | same | prefill-time reclaim would be separate scope |
| Single-request max prefill context | **no improvement claim** | same | separate prefill-time design required |
| First post-reclaim q_len | 1 | 1 | multi-token query requires separate causal/position audit |
| Retention policy | whole-block sink + recent baseline | arbitrary sorted token keep indices for compaction | one extra policy at most |
| Prefix/shared page ownership | none | none | requires refcount/shared ownership design |
| Preemption claim | no claim until compatibility gate | same | V3-C |

## Hard Fail-Fast Rule

任何 runtime 配置不满足 V1/V2 当前 support cell 时，必须 fail fast 或保持 feature off；不得“看起来能跑”就把 unsupported path 当成支持。

## Positional-Semantic Gate

首个模型必须确认：

1. Q/K 的 positional transform 在 KV 写入前已经按 logical `positions` 应用；
2. attention backend 不依赖 compact physical index 重新生成 key logical position bias；
3. q_len=1 decode 下 retained keys 全部是历史 key，compact order 保持 original retained order；
4. dual-chunk、mRoPE 或其它特殊 position path 未启用。

Pinned Qwen3 source 是当前 reference target；M0 仍需在实际 local worktree 核对 exact config/model/backend。

## Post-Prefill Capacity Scope

Core V1/V2 先完成原始 prompt prefill，随后才 reclaim。因此：

```text
prefill peak KV requirement = upstream/full prompt requirement
post-reclaim owned KV pages < pre-reclaim pages   # if retention < 1
```

这能为后续 arrival / decode workload 回收 allocator capacity，但不能把一个本来放不下的首次长 prompt 变成可执行。容量 benchmark 必须至少包含 staggered/rolling arrival，而不能只靠同时启动多个 long-prefill request。
