# 04 — Logical / Physical KV State 与 Attention / Slot Semantics

## 1. Four Semantics Must Be Split

| Semantic | Meaning | Reclaim changes it? | Primary consumers |
|---|---|---:|---|
| `logical_num_computed_tokens` | request/model committed progress | No | scheduler, sampling, phase, model length |
| `effective_kv_len` | retained valid physical KV token count | Yes | allocation sizing, physical frontier |
| `logical_position` | original model position | No | RoPE/model positional transform |
| `physical_cache_position` | compact storage index | Yes | slot mapping / append location |

Example：

```text
before: logical=128, effective=128
after reclaim: logical=128, effective=64
next model position = 128
next cache position = 64
after one committed decode: logical=129, effective=65
```

P1 的核心不是“改一个 seq_len”，而是拆开 upstream 原本复用同一 counter 的不同 semantic consumers。

## 2. Pinned Upstream Coupling (MRV2)

Pinned MRV2 `GPUModelRunner` / `BlockTables` 路径中，logical position currently drives：

```text
positions
→ block index = position // block_size
→ BlockTable physical ID
→ slot
```

而 `num_computed_tokens` 同时还参与 model position、seq length、request progression。若 reclaim 后直接把它改成 physical length，会破坏 model semantics。

MRV2 当前 seam 的 observed pattern 是：

```text
positions_np     = logical progress + query offset
slot_positions   = effective physical length + query offset
attention length = separate effective/member KV length
global seq state = remains logical where required
```

P1 uniform MVP 可在此处设计一个 narrow semantic adapter；这是 candidate seam，尚未批准实现。

## 3. Positional-Semantic Gate — 不能泛化到所有模型

“retained K 可以换 physical slot 而不改 logical position”对首个支持模型必须有 source proof。

Pinned Qwen3 `Qwen3Attention.forward()` 的关键顺序：

```text
q, k = self.rotary_emb(positions, q, k)
attn_output = self.attn(q, k, v)
```

因此 standard Qwen3 RoPE decoder path 中，写入 KV 的 K 已经按 original logical `positions` 变换。物理重排/compact 不应该把 token 7000 重新解释成 position 2500。

### V1 first-support gate

必须冻结：

- Qwen3-family decoder-only exact model；
- standard RoPE；
- causal attention；
- no dual-chunk attention；
- no encoder/bidirectional embedding mode；
- attention backend 不从 compact physical index 重新构造 key relative bias。

下面全部不属于 V1/V2 默认支持：

```text
ALiBi
relative-position bias
mRoPE / multimodal special position path
dual-chunk attention
backend-specific logical-key-position reconstruction
```

若未来支持，必须重新设计 metadata/oracle；不能只删掉 fail-fast。

## 4. `prepare_inputs()` Target Contract (MRV2)

对 request i：

```text
logical_positions_i = logical_num_computed_i + query_offset
cache_positions_i   = effective_kv_len_i + query_offset
```

用途：

- `logical_positions` → model forward / RoPE；
- `cache_positions` → `BlockTable.compute_slot_mapping()`；
- `attention_effective_kv_len` → attention metadata；
- existing/global `seq_lens` → 保持 logical，除非 exact consumer audit 证明可局部替换。

禁止：

```text
self.seq_lens = effective_kv_len   # global overwrite
num_computed_tokens = effective_kv_len
```

## 5. Attention Contract under q_len=1

V1/V2 Core 把 post-reclaim query 限制为 `q_len=1`，这是重要 correctness simplification：

- retained keys 全部来自 query 之前的历史 token；
- compact physical order只需要保持 retained original relative order；
- causal mask 不需要为同一 query batch 中多个新 token 恢复原 logical key/query boundary。

如果未来支持 `q_len>1`，必须重新审计 causal metadata/position boundaries，不能自动沿用 V1 contract。

## 6. Partial Tail and Capacity

必须区分：

```text
valid retained KV tokens = effective_kv_len
allocated pages          = ceil(effective_kv_len / block_size)
page capacity            = allocated pages * block_size
```

最后一页未满时：

```text
effective_kv_len < page capacity
```

slot mapping 使用 `effective_kv_len` 作为 append frontier；attention effective length也只包含 valid token，不包含 unused tail slots。

## 7. Persistent Batch / Row Identity

State 绑定 `req_id` 和 current request state，不绑定可复用 `req_index` 本身。InputBatch condense/move/swap 后要保持：

```text
req_id → effective_kv_len
req_id → current BlockTable row
req_id → logical_num_computed
```

一致。

显式 `reclaim_generation` 不是当前 invariant；只有 source audit 证明现有 request/state identity 不足时再批准。

## 8. Required Invariants

```text
0 <= effective_kv_len <= logical_num_computed
ceil(effective_kv_len / block_size) <= active block count/capacity contract
append cache position never points to freed page
retained token order stable
logical position never renumbered because of compaction
feature-off path identical to upstream
```

对于 V2 token-level keep，`keep_member_indices` 只索引 **current retained/member sequence**，并保持当前 member order；physical compact index不是新的 logical position。若未来 periodic reclaim policy 需要 original logical position，应另维护 member→logical metadata，而不是把 member index当 model position。

## 9. Failure-Oriented Tests

必须主动构造：

- logical counter误用为 physical frontier；
- physical counter误用为 RoPE position；
- global seq_len 被物理长度覆盖；
- partial tail off-by-one；
- row condense/swap 后 effective state 跟错 req；
- retention=1 与 upstream 不一致；
- q_len>1 意外进入；
- unsupported positional mode 意外进入。

任何一个都应在进入 capacity/performance benchmark 前解决。

## v1.2 Scope Boundary — 首次 Prefill Peak 与 Block Granularity

`effective_kv_len` 只在 reclaim commit 后变短。final-prefill 之前，logical progress 与 physical occupancy 仍按 upstream full-attention path增长。因此 P1 Core 不声称降低首次 full-prefill peak，也不声称提升 single-request maximum prompt length。

同时 Core 只支持一对一 block granularity：

```text
KV manager block == attention kernel block
blocks_per_kv_block == 1
DCP/PCP == 1
```

这样 `effective_kv_len // block_size` 与 BlockTable column 的语义唯一。hybrid block splitting / context-parallel interleave 需要单独重新证明 slot mapping 和 ownership accounting。
