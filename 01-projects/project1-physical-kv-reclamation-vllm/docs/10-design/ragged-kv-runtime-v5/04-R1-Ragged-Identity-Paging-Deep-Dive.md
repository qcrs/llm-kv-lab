> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# R1 — Ragged Identity Paging Deep Dive v2

R1 的定义仍然只有一句：**不丢任何 KV，只换 physical representation，并保持 dense 语义。** 但 v2 把原来隐藏在一刀里的 global pool 与 KV write path 单独拆出。

## 1. R1 首版环境合同

```text
GPU              A100 / sm80
Attention        FlashAttention 2
TP / PP          1 / 1
DCP / PCP        1 / 1
Model            Qwen3 full-attention model
KV dtype         BF16 / FP16
KVQuantMode      NONE
head_size_v      == head_size
page_group_size  2 or 4
cluster map      identity
compression      OFF
prefix cache     OFF
spec decode      OFF
KV connector     OFF
offload          OFF
CUDA graph       eager first
```

如果任一 feature 未经审计，不允许 silent fallback，启动时报错。

## 2. R1-A1 — Spec / Registry / Startup Gates

### 目标

建立 semantic spec，而不碰 runtime allocation：

```text
Hkv = local KV heads
Hp  = page_group_size
G   = Hkv / Hp
```

Ragged page bytes：

```text
P_r = block_size × Hp × 2 × head_size × dtype_size
```

v0.26 packed K/V layout把 K/V 放在最后的 `2D` content 中，等价 byte formula 不变。

### v0.26-native 注意点

`AttentionSpec` 比 Tangram fork 的旧 spec 多：`kv_quant_mode`, `page_size_padded`, `indexes_kv_by_block_stride`, `head_size_v` 等，因此不要复制旧类定义。新 spec 应通过 `KVCacheSpecRegistry` 注册到专用 `RaggedFullAttentionManager`（或等价 manager），避免修改通用 manager 的每个分支。

### Acceptance

- `Hp > 0`；`Hkv % Hp == 0`。
- quantized KV、MLA、SWA/hybrid、DCP/PCP 等首版明确拒绝。
- `P_r = P_dense / G`。
- Dense path 在配置关闭时无行为变化。

## 3. R1-A2 — Global Page-ID Namespace / Shared Raw Backing

这是上一版低估的部分。

普通 v0.26 的 block sizing 本质上按“一个 logical block ID 对多个 layer 的 footprint”预算；Ragged 则把 physical page 本身变为 `(layer, head-group)` 独立单位，因此 block ID 必须是全局物理 page ID。

### Capacity equation

设 KV 可用显存 `M`：

```text
Dense:
  one allocator block footprint ≈ L × P_dense
  N_dense ≈ M / (L × P_dense)

Ragged:
  one allocator page footprint = P_r
  N_ragged_global ≈ M / P_r
```

一个未压缩 request 的 page consumption：

```text
L × G × ceil(seq_len / block_size)
```

因此 identity 状态总 bytes 与 dense 等价，而 allocator 的 ID 数量会增大约 `L×G` 级别。

### Backing tensor

首版建议：

```text
one raw byte storage
  shared_by = all local full-attention layers
```

每个 layer 的 `kv_cache` view 都指向同一 backing；真正的 layer/group 区分来自 RaggedBlockTables 分配的 disjoint page IDs，而不是 per-layer allocation。

### 不能直接复用 packed-layout 的原因

v0.26 `_get_kv_cache_config_packed` 的 `block_stride/offset` 允许多个 group/slot 在一个 backing 中打包，但这些 group 可以拥有**独立 block-table namespace**。Ragged 需要的是**一个全局 physical-page namespace**。两者可共享 allocator plumbing 思路，但语义不能混同。

### Acceptance

- fixed `kv_cache_memory_bytes` 下总 allocated bytes 可精确计算。
- two layers 的 raw tensor `data_ptr`/storage alias 符合设计。
- page IDs 在所有 `(layer,group)` 间不重复。
- concurrency reporting 不多除/少除一个 `L`。

## 4. R1-B — MRv2-native RaggedBlockTables

Tangram 的 semantic shape 是：

```text
[R, G_total, Bmax]
slot_mapping = [G_total, T]
```

MRv2 推荐 physical storage：

```text
block_rows.gpu: [R * G_total, Bmax]
num_blocks:     [R, G_total]
```

逻辑 helper 再 expose 3D view。

为什么：`StagedWriteTensor` 的 kernel 本来就是 first-dimension row indexed。把 `(request,group)` 变成 virtual row，能继续复用 staged write、persistent address、UVA write descriptors 和 Triton row gather。

### 核心 API

```text
set_request_uniform(req, pages)
append_uniform(req, newly_allocated_pages)
append_per_group(req, pages_by_group)     # R2启用
get_group_row(req, g)
gather_active_batch(idx_mapping)
compute_group_slots(...)
trim_tail_per_group(...)                  # R3启用
```

不要让调用方自己计算 flat row。

## 5. R1-C — v0.26 Ragged Physical Layout

Tangram 的不变量必须保留：

```text
virtual_block = physical_page * Hp + column
```

但 v0.26 cache 是 packed K/V content，而不是 Tangram 老 `[2,...,D]` shape。

### Proposed storage

首版 ragged page 使用：

```text
logical/physical contiguous:
[B, Hp, N, 2D]
```

也就是 page 内 column 在 token 维之前（HND-like / column-major）。

零拷贝 virtual view：

```text
[B * Hp, 1, N, 2D]
```

随后当前 FA code 的：

```text
.transpose(1,2).split(D,-1)
```

得到：

```text
K,V: [B*Hp, N, 1, D]
```

正好是 standard single-KV-head paged cache。

### 为什么普通 NHD 不行

NHD physical order 相当于 `[B,N,Hp,2D]`；`N` 插在 `B` 与 `Hp` 之间，无法通过简单 reshape 把 `(B,Hp)` 变成连续 virtual block axis。若强行 reshape 可能 copy 或地址语义错误。

因此首版不应依赖全局 `VLLM_KV_CACHE_LAYOUT` 切换所有 dense path，而应为 Ragged 创建专属 reshape/view helper。

### 必须先做的测试

对每个 `(p,col,offset,content)`：

```text
physical[p,col,offset]
== virtual[p*Hp+col,0,offset]
```

并验证二者 storage alias，而不仅是数值相等。

## 6. R1-D1 — Ragged KV Write Seam

MRv2 当前明确把 KV write 与 attention read 分开：

```text
Attention.forward
  ├─ unified_kv_cache_update(...)
  │    └─ FlashAttentionImpl.do_kv_cache_update
  └─ unified_attention_with_output(...)
       └─ FlashAttentionImpl.forward
```

这个 ordering 通过 dummy dependency 保证 torch.compile 不重排。

### 冻结方案

保留这条 contract。Ragged 只改变 update body：

```text
K/V token-major
  ↓ layer-local member reshape
K/V member-major
  ↓ group physical slots → member virtual slots
reshape_and_cache_flash
  ↓
virtual cache view [B*Hp,1,N,2D]
```

不要把隐藏的 KV mutation 偷塞进 read op；那会失去当前 MRv2 已经设计好的 side-effect/data-dependency seam。

### Write-path differential oracle

构造 distinct K/V values，使每个 token/head 都可识别；执行 update 后逐 `(token,head)` 用 independent address equation 读回。此测试必须在任何 FA read 测试前 PASS。

## 7. R1-D2 — Decode Read

Decode `q_len=1`：

```text
[num_reqs,Hq,D]
→ [num_reqs*Hkv,q_per_kv,D]
```

`block_table`、`seq_lens`、`slot_mapping` 都是 member sequences。

风险：v0.26 `FlashAttentionImpl.forward` 内一些逻辑根据原始 `self.num_heads/self.num_kv_heads` 建 descale/sink/ALiBi tensors。首版全部 hard-gate 掉；必要时加 `RaggedFlashAttentionAdapter`，只复用 `flash_attn_varlen_func`，不要硬骗 normal impl 的 static head attributes。

## 8. R1-D3 — Prefill / Mixed

变长请求没有 zero-copy member-major view：

```text
Q token-major
  → view [T,Hkv,q_per_kv,D]
  → permute [Hkv,T,q_per_kv,D]
  → contiguous
```

K/V 同理，输出 inverse copy。这个 copy 是 identity-ragged 的结构性税，必须在 R10 单独计账。

## 9. R1-E / R1-F Exit

R1-E 至少覆盖：fresh prefill、decode continuation、2–3 requests continuous batch、chunked prefill。  
R1-F 只验证 piecewise CUDA Graph：ragged attention eager，其他 graph pieces 可 capture/replay；不在这里优化 metadata allocation。

R1 全部关闭前，禁止开始 compression。

---

# v4 Mandatory Addendum — R1-A3 Layout Gate

在本文件原有 R1-A2 与 R1-B 之间，必须插入：

```text
R1-A3 Physical Layout / Stride Contract
```

Canonical details见：

```text
22-R1-Exact-Engineering-Contract.md
34-Physical-Layout-Stride-and-Virtual-Block-Exact-Contract.md
```

未证明 zero-copy virtual block 之前，不进入 KV write/read。

