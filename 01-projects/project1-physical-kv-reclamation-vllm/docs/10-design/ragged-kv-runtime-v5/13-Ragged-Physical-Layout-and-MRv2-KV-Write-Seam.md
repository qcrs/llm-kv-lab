> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# v0.26 Ragged Physical Layout + MRv2 KV Write Seam

这是 v2 新增的最重要设计文档。它解决上一版尚未冻结的两个问题：**Ragged bytes 到底怎么落在 v0.26 FA cache 中，以及新 token 到底在哪里写入。**

## 1. Source contracts

### v0.26 MRv2

```text
Attention.forward
  ├─ unified_kv_cache_update(key,value,layer)
  │    └─ impl.do_kv_cache_update(...slot_mapping)
  │         └─ reshape_and_cache_flash
  │
  └─ unified_attention_with_output(... dummy_dep ...)
       └─ impl.forward(...)
            └─ flash_attn_varlen_func
```

`unified_attention_with_output` 本身是 piecewise graph eager break。这个 seam 值得保留。

### Tangram invariant

```text
physical page p + column c
→ virtual block v = p*Hp + c
```

FA 仍看到普通 single-KV-head paged blocks，数学 kernel 不变。

## 2. Why `[B,N,Hp,2D]` is insufficient

若物理连续布局为 NHD：

```text
[B, N, Hp, 2D]
```

内存线性次序在同一 page 中先走 token `N` 再走 head column。要把 `(B,Hp)` 合成一个 virtual-block axis 时，中间隔着 `N`，不能通过一个无拷贝 reshape 得到：

```text
[B*Hp, N, 1, 2D]
```

因此 virtual-block trick要求 page 内 column 是较外层维度。

## 3. Proposed v0.26-native physical shape

首版 BF16/FP16：

```text
Ragged KV logical/physical contiguous
[B, Hp, N, 2D]
```

其中：

```text
B  = number of global physical pages
Hp = heads per physical page
N  = block_size
2D = packed K and V content
```

virtual view：

```text
[B*Hp, 1, N, 2D]
```

现有 FA code继续：

```text
virtual.transpose(1,2)
→ [B*Hp,N,1,2D]
split(D,-1)
→ K/V [B*Hp,N,1,D]
```

### Required invariant

```text
virtual[p*Hp+c, 0, t, :]
ALIASES
physical[p, c, t, :]
```

必须检查 storage pointer/stride，而不是仅 `torch.equal`。

## 4. Spec semantics vs storage semantics

不要把 `RaggedAttentionSpec.num_kv_heads` 改成 `Hp`。它应该继续描述 model semantic local Hkv，供 validation/member geometry 使用；真正 reshape backing 时，ragged-special branch 使用 `page_group_size` 作为 **storage head width**。

因此：

```text
semantic Hkv  → groups_per_layer = Hkv/Hp
storage page  → Hp columns
```

这两个概念必须在 API 中命名区分。

## 5. KV write path

本 step 对 request r 有 `q_len` 个新 token。对于 layer l，每个 KV head h 对应 member：

```text
member = l*Hkv + h
cluster = member_to_cluster[member]
column  = member_to_col[member]
```

R1 identity map：

```text
cluster = l*G + floor(h/Hp)
column = h % Hp
```

group physical write coordinate：

```text
physical_pos[g, token] = E[r,g] + local_query_offset
physical_page = block_table[r,g, physical_pos//N]
offset = physical_pos % N
```

member virtual slot：

```text
virtual_slot = (physical_page*Hp + column)*N + offset
```

### D1 data transform

Input K/V：

```text
[T, Hkv, D]
```

`reshape_and_cache_flash` 在 virtual view中每个 sequence只有一个 KV head，所以要形成 member rows：

```text
K_member [Hkv*T, 1, D]
V_member [Hkv*T, 1, D]
slot_member [Hkv*T]
```

Decode 可低成本 reshape；prefill/mixed若 token packing跨多个 req，需要按 `(req,head)` sequence order准确构造，而不是简单全 batch `permute(H,T)` 后假设 request边界消失。

## 6. Ordering contract

强制保留：

```text
KV write op
   ↓ dummy dependency
attention read op
```

理由：

- torch.compile 需要看见 side-effect ordering；
- CUDA Graph/piecewise splitting已有成熟入口；
- compaction 是 post-forward mutation，不应与 current-step write 合并；
- 后续 P1 stale fence逻辑依赖“post-forward source state”语义。

## 7. Read path adapter

推荐结构：

```text
unified_attention_with_output
  → if ragged:
       ragged_forward_adapter(layer, q,k,v,out,metadata,kv_cache)
    else:
       existing impl.forward
```

`ragged_forward_adapter` 负责：

1. slice actual tokens；
2. token-major→member-major；
3. layer-specific virtual block table / seq lens；
4. virtual cache view；
5. 调用最窄的 FA varlen primitive；
6. output member-major→token-major。

首版不建议直接调用 full normal `impl.forward` 并伪装 head counts，因为其 descale/sink/cascade branches有静态 model-head假设。

## 8. D1/D2/D3 Tests

### D1 write-only

每 token/head 写 unique encoded value：

```text
value = f(req,token,head,plane,dim)
```

写后 independent address decoder逐 cell读取。

### D2 decode

- 1 req / 2 req；
- heterogeneous history lengths；
- compare dense torch/FA output；
- no compression。

### D3 prefill

- request lengths `[5,13,2]` 等非对齐组合；
- block-boundary crossing；
- chunked prefill；
- validate member `cu_seqlens_q`；
- padded dummy rows不能产生 valid slot。

## 9. Future layout extensions

Quantized KV、`head_size_v != head_size`、sink/ALiBI、FA3/4、HND/NHD selectable layout 都在 Core 后再开。首版 physical layout contract必须保持极窄，否则一个 shape bug会同时污染 attention、compaction与allocator证据。
