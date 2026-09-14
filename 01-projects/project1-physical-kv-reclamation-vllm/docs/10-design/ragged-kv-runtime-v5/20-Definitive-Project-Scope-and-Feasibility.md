> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Definitive Project Scope & Feasibility — v3

这份文档是 v3 的**最终 scope contract**。如果其他文档和这里冲突，以这里为准。

## 1. 我们到底在做什么

项目不是“把 Tangram patch 搬到 vLLM 0.26”，也不是“重新发明一种 KV scorer”。准确目标是：

> 在 `qcrs/vllm@p1/v2-token-compaction-v026` 的 vLLM 0.26 Model Runner V2 上，实现一个 **Ragged / Non-Uniform KV Cache Runtime**：把 request-scalar 的 physical KV state 扩展成 per-(layer, head-group) state，使不同 head-group 能拥有不同 retained token 数和不同 physical page depth，并将这种 non-uniform retention 闭环到真实 GPU page reclamation / reuse。

系统主链：

```text
Dense vLLM 0.26
request → one logical block row → all local KV heads share depth

            ↓ representation redesign

Ragged runtime
request
  └─ layer
      └─ head-group
          ├─ independent page row
          ├─ independent physical length E[r,l,g]
          └─ shared global physical-page namespace

            ↓ policy / manual decision

keep positions / target depth
  → payload compaction
  → trim group tail pages
  → worker result
  → scheduler canonical reconcile
  → BlockPool free
  → another request reuses the pages
```

Tangram 是**reference implementation / test oracle**；P1 是**ownership transaction 和 compaction 基础**；v0.26 MRv2 是**最终运行时约束**。

## 2. 五个必须重建的 Tangram 机制

项目真正依赖 Tangram 的只有五个系统机制。它们都能在 Tangram 当前 main 找到明确源码出处。

### M1 — Head-Group Page

一个 physical page 不再装全部 `Hkv`，而只装 `Hp=page_group_size` 个 KV heads。

```text
Dense page bytes  = 2 * B * Hkv * D * dtype_size
Ragged page bytes = 2 * B * Hp  * D * dtype_size
G = Hkv / Hp
G * RaggedPageBytes = DensePageBytes
```

Tangram reference：
- `vllm/v1/kv_cache_interface.py::RaggedAttentionSpec`
- `RaggedAttentionSpec.page_size_bytes`
- `RaggedAttentionSpec.num_head_groups_per_layer`

这是 R1-A1 的直接 reference。

### M2 — Global Physical Page Pool

Ragged page ID 必须在所有 local layers / head-groups 之间共享一个 physical namespace，否则某 group 释放的 page 不能被另一 group/layer 立即复用。

Tangram reference：
- `vllm/v1/core/kv_cache_utils.py::get_num_blocks`
- `get_kv_cache_config_from_groups` 的 ragged branch
- `vllm/v1/core/block_pool.py` ragged ring path
- `vllm/v1/core/single_type_kv_cache_manager.py` ragged allocation path

v0.26 MRv2 不能照搬，必须 redesign planner/backing/manager integration。

### M3 — Virtual Single-Head Paged View

Tangram 的关键技巧不是改 FlashAttention kernel，而是把 `(physical_page, column)` 映射成一个 virtual block：

```text
virtual_block_id = physical_page_id * Hp + column
```

于是一个 head-column 可以看成标准的 single-KV-head paged cache。

Tangram reference：
- `vllm/v1/attention/backends/ragged_layout.py::as_virtual_block_view`
- `member_virtual_block_table`
- `member_virtual_slots`
- `tests/v1/attention/ragged_reference.py`
- `tests/v1/attention/test_ragged_layout.py`

这是 R1-C / R1-D 的算法 reference。

### M4 — One `(request, KV head)` = one FA varlen sequence

每个 KV head member 携带其 GQA query-head group：

```text
member sequence = (request, kv_head)
num_q_per_kv = Hq / Hkv
```

Decode q_len=1 时 reshape 很便宜；prefill / mixed batch 因 request 长度不同，需要 token-major → member-major materialization。

Tangram reference：
- `vllm/v1/attention/backends/ragged_forward.py::_ragged_decode_forward`
- `_ragged_member_major_forward`
- `vllm/v1/attention/backends/flash_attn.py` ragged metadata path

这是 R1-D2 / D3 的直接 reference。

### M5 — Per-group compaction → physical page free

不同 `(layer,group)` 可以压到不同长度：

```text
E[r,l,g] → ceil(E/B) pages
```

payload 先 compact 到 retained prefix，再 trim trailing page IDs。

Tangram reference：
- `vllm/v1/worker/ragged_block_table.py::compact_after_compress_all_layers`
- `vllm/v1/attention/compression/executor.py`
- `vllm/v1/core/single_type_kv_cache_manager.py::free_blocks_by_ids`

P1 reference：
- existing `CompactionPlanData / CompactionResultData`
- scheduler-owned physical reconciliation
- V2 gather/writeback oracle

这是 R2/R3/R4 的核心。

## 3. 哪些东西我们明确不搬

为了保证能完成，首版禁止把 Tangram 的全部功能一起搬入。

```text
NOT CORE:
- scorer zoo
- FastKVzip gate checkpoint
- multi-turn serving
- sliding-window/hybrid support
- MLA
- prefix caching
- KV Connector / LMCache
- DCP / PCP
- PP
- speculative decode
- full CUDA Graph
- cross-layer AOT clustering
```

这些都不是证明“Ragged physical runtime”成立的必要条件。

## 4. Core completion 的最低充分条件

### Functional Core

必须完成：

```text
R1 Identity Ragged Runtime
R2 Per-group physical state E[r,l,g]
R3 Manual non-uniform reclaim + real free/reuse
R4-A one real scorer + one safe budget path
```

必须有四组证据：

1. **Identity correctness**：compression off 时 dense vs ragged real-engine output 等价。
2. **Non-uniform correctness**：同一 request 至少两个 groups 拥有不同 `E` 和 block count，request 能继续 decode。
3. **Allocator closure**：trimmed physical pages 回到 pool，第二 request 实际拿到其中至少一个 ID。
4. **Policy closure**：一个真实 scorer 产生 keep decision，通过同一 data plane 完成 compaction/free。

只要这四条闭环，这个项目已经成立。

### Flagship Complete

增加：

```text
R4-B automatic non-uniform budget assignment
```

即 policy 本身而不是人工 target 产生不同 group depths。

## 5. 最稳妥的实现模型

### Semantic model

保持模型语义：

```text
Hkv = local semantic KV heads
Hp  = physical page group width
G   = Hkv / Hp
```

`RaggedAttentionSpec.num_kv_heads` 仍然是 `Hkv`，不要改成 `Hp`。

### Storage model

v0.26 当前 FA cache shape 和 Tangram 基线不同，因此首版建议做 **ragged-specific physical shape**，而不是硬套 normal backend shape：

```text
physical ragged backing view:
[Bphys, Hp, block_size, 2*D]

virtual single-head view:
[Bphys*Hp, 1, block_size, 2*D]

FA internal transpose/split 后：
K/V [Bphys*Hp, block_size, 1, D]
```

必须满足 zero-copy alias：

```text
virtual[p*Hp+c, 0, t, :]
    aliases
physical[p, c, t, :]
```

这是 v0.26-native adaptation，不是 Tangram 原样 shape。

## 6. 可实现性判断

| 子系统 | 科学不确定性 | 工程风险 | Tangram reference | 完成建议 |
|---|---:|---:|---|---|
| Spec/page math | 低 | 低 | 强 | 必做 |
| Global pool/backing | 低 | 中 | 强 | 必做 |
| Ragged block table | 低 | 高 | 强 | 必做，MRv2 redesign |
| virtual addressing | 低 | 中 | 很强+tests | 必做 |
| KV write | 低 | 高 | 间接 | 必做，v0.26-specific |
| decode FA | 低 | 中 | 很强 | 必做 |
| prefill/mixed FA | 低 | 高 | 很强 | 必做 |
| per-group E | 低 | 中 | 强 | 必做 |
| manual reclaim | 低 | 中 | 强 + P1 | 必做 |
| one scorer | 低 | 中 | 强 | 必做 |
| TP2 uniform | 低 | 中高 | 强 | breadth |
| preemption | 低 | 中 | 强 | breadth |
| AOT clustering | 低 | 中 | 强/tools | optimization |
| in-place Triton | 低 | 中高 | 强 | optimization |
| Piecewise CG | 低 | 中 | 强 + v0.26 seam | optimization |

结论：这个项目的主要风险是 **data-layout / state-consistency engineering**，不是算法是否存在。因此用严格 Slice / oracle / fallback 可以显著提高完成概率。

## 7. Stop-loss 原则

如果某个 optimization 卡住，不得阻塞 Core：

```text
AOT clustering fails → identity/per-layer adjacent map still valid
Triton in-place fails → keep P1/Torch scratch compaction backend
Piecewise CG performance negative → report eager-island tax, keep eager correctness
TP2 fails → core TP1 still成立
Preemption breadth fails → standard finish/abort lifecycle仍可交付
```

唯一不可降级的部分是 R1 identity + R2/R3 physical closure。
