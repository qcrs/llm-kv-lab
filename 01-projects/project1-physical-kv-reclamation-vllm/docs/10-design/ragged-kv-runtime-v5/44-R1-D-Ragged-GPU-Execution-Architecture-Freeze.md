# R1-D Ragged GPU Execution — vLLM 链路深剖、Tangram 对照与架构冻结

**Project:** Physical KV Cache Reclamation / Ragged KV Runtime for vLLM v0.26.0  
**Implementation repository:** `qcrs/vllm`  
**Implementation branch:** `p1/v2-token-compaction-v026`  
**Reviewed implementation HEAD:** `db3cc1179f53e2dc8df2096ed17b5ec86ef033d3`  
**Pinned upstream baseline:** `568afb3a13806beb53bb2e6bd518269357b237c0`  
**Project repository:** `qcrs/llm-kv-lab`  
**Tangram reference:** `aiha-lab/tangram@6fa551fc8f6edcc118a2a39b3554ee1520e91edd`

**Recommended repository location**

```text
01-projects/project1-physical-kv-reclamation-vllm/
└── docs/
    └── 10-design/
        └── ragged-kv-runtime-v5/
            └── 44-R1-D-Ragged-GPU-Execution-Architecture-Freeze.md
```

---

# 1. Executive Decision

进入 D 前，当前项目已经把三个基础层分别建立起来：

```text
A
physical page / global pool / member placement

B
Scheduler canonical ownership
Worker discardable physical-state mirror
reserve / commit / reconciliation

C
physical address oracle
[P,Hp,B,2D] physical backing
[P*Hp,1,B,2D] zero-copy virtual view
cluster → member block/slot/seq-len transforms
```

当前真正缺失的是：

> **把已经确定的 Ragged physical state 真正送进 GPU KV write 和 FlashAttention read。**

旧 canonical 文档把这一层拆成：

```text
R1-D1 KV Write
R1-D2 Decode Read
R1-D3 Prefill / Mixed Read
```

重新审计当前 `qcrs/vllm` v0.26 execution seam、现有 C 产物和 Tangram pinned implementation 后，本次冻结：

```text
R1-D1 + R1-D2 + R1-D3
          ↓
一个完整实现 Slice：

P1-V2-R1-D-RAGGED-GPU-EXECUTION-01
```

内部仍保留三个 **acceptance area**：

```text
D-WRITE
D-DECODE
D-PREFILL-MIXED
```

但它们不再是三个独立 Slice。

D 的边界同时明确：

```text
D
= synthetic / focused GPU execution correctness closure

E
= production vLLM Engine activation + lifecycle wiring + real-model identity
```

因此 D 不开启 production `page_group_size` Ragged runtime，不进入真实 Engine。

---

# 2. 为什么现在适合一次完成整个 D

把 write / decode / prefill-mixed 合成一个 Slice 是否合理，取决于错误边界是否已经足够清楚。

C 已经关闭以下高风险基础问题：

```text
Ragged page byte accounting       CLOSED
physical cache geometry           CLOSED
zero-copy virtual-block view      CLOSED
member placement                  CLOSED
physical/virtual address equation CLOSED
virtual slot equation             CLOSED
real CUDA storage alias           CLOSED
```

因此 D 失败时，问题已经能够被定位为：

```text
1. group/member slot producer 错
2. K/V flatten ordering 错
3. member sequence metadata 错
4. prefill/mixed packing 错
5. FA2 primitive integration 错
```

不再需要同时怀疑 allocator、stride、page bytes 或 cache hidden copy。

这正是“一次完整做 D”成立的前提。

人为继续拆成 D1/D2/D3 反而容易出现：

```text
D1 发明一套 metadata
D2 再改一次 shape / ordering
D3 再重写一次 prefill ordering
```

最终造成不必要返工。

---

# 3. 先理解当前 vLLM v0.26 Dense execution chain

D 不是重新发明一个 attention runtime。

正确思路是：

> **先理解 vLLM Dense 主链，再只替换 Ragged 必须替换的几何。**

当前 qcrs/v0.26 关键执行链：

```text
Scheduler
   ↓
GPUModelRunner.prepare_inputs
   ↓
prepare_pos_seq_lens
   ↓
BlockTables.compute_slot_mappings
   ↓
GPUModelRunner.prepare_attn
   ↓
DefaultModelState.prepare_attn
   ↓
build_attn_metadata
   ↓
build_slot_mappings_by_layer
   ↓
set_forward_context
   ↓
Attention.forward
   ├─ unified_kv_cache_update
   │      ↓
   │  FlashAttentionImpl.do_kv_cache_update
   │      ↓
   │  reshape_and_cache_flash
   │
   └─ unified_attention_with_output
          ↓
      FlashAttentionImpl.forward
          ↓
      flash_attn_varlen_func
```

这条链是 D 阶段最值得真正掌握的 vLLM execution path。

---

# 4. `prepare_pos_seq_lens`：logical position 与 physical cache position

当前 Project 1 V1 已经在：

```text
vllm/v1/worker/gpu/input_batch.py
```

建立：

```text
positions
cache_positions
```

两个坐标。

当前 kernel 的核心关系：

```text
positions
= num_computed_tokens + local_query_offset

cache_positions
= effective_kv_len + local_query_offset
```

于是：

```text
positions
负责模型逻辑位置 / RoPE

cache_positions
负责 KV physical append coordinate
```

没有 reclaim 时：

```text
positions == cache_positions
```

reclaim 后：

```text
positions != cache_positions
```

这是 KV reclamation runtime 很核心的思想：

> **模型语义位置不必等于 KV storage 位置。**

但是 V1 的：

```text
effective_kv_len
```

还是 request-level scalar。

Ragged V2 进一步把它升级为：

```text
E[r, cluster]
```

因此同一个 token q 对不同 cluster：

```text
physical_position[q,c]
=
E[r,c] + local_query_offset
```

所以 Dense 的：

```text
cache_positions[Q]
```

不再足够。

Ragged execution 需要：

```text
physical_positions[Q,C]
```

---

# 5. Dense `BlockTables.compute_slot_mappings` 本质是什么

Dense PagedAttention 的地址翻译可以概括成：

```text
token physical position
        ↓
block_depth = position // B
offset      = position % B
        ↓
request block table
        ↓
physical block id
        ↓
slot = block_id * B + offset
```

因此它完成：

```text
(request, position)
→ physical KV slot
```

Dense 输出：

```text
slot_mapping[Q]
```

Ragged 并没有推翻这个思想。

Ragged 只是扩展为：

```text
(request, cluster, physical_position)
→ group physical slot
```

也就是：

```text
group_physical_slots[Q,C]
```

再利用 C 已经冻结的：

```text
(cluster,column)
→ member
→ virtual block
```

得到：

```text
member_virtual_slots[Q,M]
```

所以 Ragged addressing 可以理解为：

> **Dense slot mapping 的 group-aware 扩展 + virtual block re-encoding。**

---

# 6. Dense KV Write seam

当前 qcrs v0.26 已经明确把：

```text
KV cache mutation
```

和：

```text
attention read
```

拆成两次调用。

`Attention.forward()` 中：

```text
unified_kv_cache_update
        ↓
kv_cache_dummy_dep
        ↓
unified_attention_with_output
```

write 链：

```text
Attention.forward
↓
unified_kv_cache_update
↓
get_attention_context
↓
layer_slot_mapping
↓
FlashAttentionImpl.do_kv_cache_update
↓
reshape_and_cache_flash
```

Dense 输入语义：

```text
K/V:
[Q,Hkv,D]

slot_mapping:
[Q]

cache:
[P,Hkv,B,2D]
```

一个 slot 对应一个 token position，而这个 token 的所有 Hkv heads 一起写入 Dense page 的 head axis。

---

# 7. Ragged write 为什么必须变成 member write

Ragged physical cache：

```text
[P,Hp,B,2D]
```

C 已经证明它可以 zero-copy 看成：

```text
[P*Hp,1,B,2D]
```

其中：

```text
virtual_block
=
physical_page * Hp + column
```

从 FlashAttention / cache kernel 视角：

```text
一个 virtual block
只有一个 KV head
```

因此 Ragged write 不能继续使用：

```text
一个 token slot → Hkv heads
```

而必须把：

```text
(token, semantic KV head)
```

变成独立 write item。

所以：

```text
K/V [Q,Hkv,D]
        ↓
reshape
        ↓
[Q*Hkv,1,D]
```

对应：

```text
member virtual slots
[Q*Hkv]
```

然后继续使用现有：

```text
reshape_and_cache_flash
```

不需要重新写 KV scatter kernel。

---

# 8. Write 不需要 prefill member-major packing

这是一个非常容易和 read 混淆的地方。

KV write 是：

```text
scatter
```

它不需要 varlen sequence semantics。

只要：

```text
K/V row i
对应 slot i
```

即可。

因此 write 统一固定：

```text
token-major / head-minor
```

顺序：

```text
token0/head0
token0/head1
...
token0/head(Hkv-1)

token1/head0
...
```

所以：

```text
K/V [Q,Hkv,D]
→ reshape [Q*Hkv,1,D]

member_slots [Q,Hkv]
→ flatten [Q*Hkv]
```

无论：

```text
decode
prefill
mixed
```

都可以使用同一 write path。

因此：

> **prefill/mixed 的复杂 packing只属于 Attention read，不属于 KV write。**

---

# 9. Dense Attention Read seam

write 完成后：

```text
unified_attention_with_output
```

从 ForwardContext 取得：

```text
attn_metadata
kv_cache
```

调用：

```text
FlashAttentionImpl.forward
```

最后进入：

```text
flash_attn_varlen_func
```

其核心 metadata 包括：

```text
Q
paged K/V
query_start_loc
block_table
seqused_k / seq_lens
max_query_len
max_seq_len
```

Dense 语义：

```text
一个 request
=
一个 FA varlen sequence
```

于是：

```text
block_table [R,MaxPages]
seq_lens    [R]
query_start [R+1]
```

---

# 10. Ragged read：`(request, KV-head)` 才是一个 FA sequence

对于某一 layer，未来不同 KV heads 所属 cluster 可以具有不同：

```text
page row
effective length E
```

因此：

```text
request
```

不能再作为唯一 sequence。

Ragged 中：

```text
(request, KV head)
```

才是一条 paged sequence。

每 layer：

```text
number of FA sequences
=
R * Hkv
```

而 GQA：

```text
q_per_kv
=
Hq / Hkv
```

所以每个 member sequence：

```text
Q:
[q_len, q_per_kv, D]

K/V:
single-head paged cache
```

这与 virtual cache：

```text
[P*Hp,1,B,2D]
```

正好一致。

---

# 11. Decode read 为什么简单

uniform decode：

```text
每 request query_len = 1
```

输入：

```text
Q [R,Hq,D]
```

可以直接：

```text
[R,Hkv,q_per_kv,D]
→
[R*Hkv,q_per_kv,D]
```

本项目冻结 member sequence order：

```text
request-major
head-minor
```

即：

```text
req0/head0
req0/head1
...
req0/head(Hkv-1)
req1/head0
...
```

对应：

```text
member_block_table
[R,Hkv,MaxPages]
→ [R*Hkv,MaxPages]

member_seq_lens
[R,Hkv]
→ [R*Hkv]

member_query_start_loc
=
[0,1,2,...,R*Hkv]
```

因此 decode 可以是一个低成本 reshape fast path。

---

# 12. Prefill / mixed 为什么必须 materialize

假设：

```text
req0 q_len = 3
req1 q_len = 1
```

token-major tensor 顺序：

```text
req0 token0
req0 token1
req0 token2
req1 token0
```

但 member FA sequence 需要：

```text
req0/head0:
 token0 token1 token2

req0/head1:
 token0 token1 token2

...

req1/head0:
 token0

req1/head1:
 token0
```

因此 prefill/mixed 无法通过单一 `.view()` 得到 member sequence-major layout。

R1 冻结：

```text
允许显式 gather / copy
```

原则：

```text
先正确
后 profile
```

不要提前写：

```text
fused Triton transpose
复杂 zero-copy trick
```

那些应该等 profile 后再决定。

---

# 13. Canonical metadata order 与 FA execution order 要分开

进一步对照 Tangram 的 prefill/mixed 实现后，这里不强行要求：

```text
decode
prefill
mixed
```

都使用同一个 FA sequence linearization。

真正应该冻结的是：

## Canonical derived metadata

C/D 内部继续保持：

```text
member_block_table:
[R,M,MaxPages]

member_seq_lens:
[R,M]

member_slot_mapping:
[Q,M]
```

某 layer semantic slice：

```text
[R,Hkv,...]
```

也就是 canonical tensor axis始终：

```text
request-major
head-minor
```

这保证：

```text
MemberPlacementMap semantic member index
m = layer * Hkv + head
```

不会因为执行模式而改变。

但是：

> **FA kernel的 sequence linear order允许 decode 和 prefill/mixed选择各自最简单的执行 layout。**

这是 execution ordering，不是 ownership/placement semantics。

---

# 14. Decode 与 Prefill/Mixed 的执行 ordering

## 14.1 Uniform decode

decode：

```text
Q = R
query_len = 1
```

最便宜的是直接：

```text
Q [R,Hkv,q_per_kv,D]
→ reshape
[R*Hkv,q_per_kv,D]
```

对应 sequence order：

```text
req0/head0
req0/head1
...
req1/head0
...
```

即：

```text
request-major / head-minor
```

metadata同样直接：

```text
[R,Hkv,N] → [R*Hkv,N]
[R,Hkv]   → [R*Hkv]
```

没有额外 copy。

---

## 14.2 Prefill / mixed

prefill/mixed 的 token-major tensor：

```text
[Q,Hkv,q_per_kv,D]
```

如果强行生成：

```text
req0/head0 tokens...
req0/head1 tokens...
...
```

需要复杂 gather index。

Tangram pinned implementation展示了一个更简单的合法 execution order：

```text
head0 / req0 tokens...
head0 / req1 tokens...
...
head1 / req0 tokens...
head1 / req1 tokens...
...
```

即：

```text
head-major / request-minor
```

因为原始 Q token-major 已经按照 request span排列，所以只需要：

```text
[Q,Hkv,q_per_kv,D]
→ permute(1,0,2,3)
→ contiguous
→ [Hkv*Q,q_per_kv,D]
```

而不需要维护一套复杂 token gather map。

block table：

```text
canonical [R,Hkv,N]
→ permute(1,0,2)
→ [Hkv,R,N]
→ [Hkv*R,N]
```

seq lens：

```text
canonical [R,Hkv]
→ transpose
→ [Hkv,R]
→ [Hkv*R]
```

query start：

对每个 head h，将原始 request query starts平移：

```text
query_start_loc_h
=
query_start_loc[:R] + h * Q
```

按：

```text
head-major / request-minor
```

flatten，并追加：

```text
Hkv * Q
```

作为最后 tail。

因此：

```text
canonical metadata order
仍然统一为 [R,Hkv,...]

FA execution order
decode      = request-major/head-minor
prefill/mix = head-major/request-minor
```

这个模式比强制单一 linear order更简单，也更接近 Tangram 已验证实现。

---

# 14.3 Output inverse

prefill/mixed FA output：

```text
[Hkv*Q,q_per_kv,D]
```

先：

```text
view(Hkv,Q,q_per_kv,D)
```

再：

```text
permute(1,0,2,3)
```

恢复：

```text
[Q,Hkv,q_per_kv,D]
→ [Q,Hq,D]
```

因此 forward 和 inverse 使用同一显式 transpose关系，不需要第二套独立 gather/scatter公式。

这也是采用 mode-specific FA execution ordering 的主要工程收益。

---

# 15. D 真正新增的 physical-slot producer

C 已经实现：

```text
group physical slot
→ member virtual slot
```

D 需要补齐前半段：

```text
Worker E + cluster page rows
→ group physical slot
```

D 输入：

```text
cluster_rows
[R,C,MaxPages]

source_effective_lens
[R,C]

query_start_loc
[R+1]
```

对于 token q：

```text
r = request_of_token(q)

local_offset
=
q - query_start_loc[r]
```

然后：

```text
physical_position[q,c]
=
source_E[r,c] + local_offset
```

再：

```text
page_depth
=
physical_position // B

offset
=
physical_position % B

page
=
cluster_rows[r,c,page_depth]

group_physical_slot[q,c]
=
page * B + offset
```

输出：

```text
[Q,C]
```

这就是 Dense：

```text
BlockTables.compute_slot_mappings
```

在 Ragged geometry 下的对应物。

---

# 16. 最重要的 frontier distinction：source E vs post-write E

D 最容易发生 silent correctness bug 的地方之一是把：

```text
写地址 frontier
```

和：

```text
attention read visibility
```

混成同一个 E。

写当前 query：

```text
source_E = 32
q_len = 3
```

应写：

```text
position 32
position 33
position 34
```

所以 write address使用：

```text
source_E
```

但是 attention read发生在 write 之后。

此时 cache 中有效长度：

```text
post_write_E
=
source_E + q_len
=
35
```

因此 FA：

```text
seqused_k / seq_lens
```

必须使用：

```text
post_write_group_seq_lens[r,c]
=
source_E[r,c] + query_len[r]
```

然后再：

```text
member_seq_lens
=
gather(post_write_group_seq_lens)
```

冻结为：

```text
WRITE:
source E

READ:
source E + current query_len
```

---

# 17. 这为什么与 causal prefill 兼容

一个 member：

```text
history length = E
current query length = q
```

KV 总有效长度：

```text
E + q
```

FA causal varlen 会把 query 对齐到 KV sequence尾部。

所以 query token j 自动只能看到：

```text
历史 E
+
当前 chunk [0..j]
```

这正是 autoregressive chunked prefill 需要的语义。

因此 D 不需要手写二维 causal mask。

只要正确构造：

```text
cu_seqlens_q
seqused_k
block_table
causal=True
```

即可。

---

# 18. Tangram：应该借什么，不应该借什么

## 18.1 借 `RaggedBlockTable` 的 slot mapping思想

Tangram 明确把：

```text
block_table:
[request, group, page-depth]

slot_mapping:
[group, token]
```

作为一等结构。

其 grouped slot mapping本质就是：

```text
group position
→ block table
→ physical slot
```

这验证了我们的：

```text
[Q,C] group physical slot
```

抽象是合理的。

但不要 port 整个 `RaggedBlockTable`。

因为 qcrs 已经建立：

```text
Scheduler canonical ownership
+
RaggedWorkerPhysicalState
```

再 port Tangram ownership table 会制造第二套 state authority。

结论：

```text
借 slot math
不借 ownership architecture
```

---

## 18.2 借 Tangram 的 cached member maps

Tangram 的 `FlashAttentionMetadataBuilder` 缓存：

```text
_member_to_cluster
_member_to_col
```

一次构建，多 step复用。

这与当前已经冻结的：

```text
MemberPlacementMap
→ initialize once
→ cached GPU tensors
→ every-step reuse
```

一致。

因此 D 必须把 C 的 low-level execution helper收口成显式 tensor输入。

---

# 19. C helper API 在 D 应做一次必要收口

当前 C helper：

```text
member_virtual_block_table(..., placement)
member_virtual_slots(..., placement)
member_seq_lens(..., placement)
```

内部会调用：

```text
placement_to_tensors()
```

作为 C reference/convenience API没有问题。

但 production-oriented D 不应继续如此。

D 建议改成：

```text
member_virtual_block_table(
    physical_table,
    member_to_cluster,
    member_to_column,
    page_group_size,
)

member_virtual_slots(
    physical_slots,
    member_to_cluster,
    member_to_column,
    page_group_size,
    block_size,
)

member_seq_lens(
    physical_seq_lens,
    member_to_cluster,
)
```

保留：

```text
placement_to_tensors(MemberPlacementMap)
```

只用于：

```text
initialization
construction
tests
```

这样 API本身就不会允许 hot path无意间每 step创建 tensor。

---

# 20. cached placement tensors 的 production owner

Tangram 把 cached maps放在：

```text
FlashAttentionMetadataBuilder
```

这个 owner 很合理，因为 builder：

```text
跨 step 存活
知道 device
知道 AttentionSpec
知道 layer set
本身负责生成 attention metadata
```

因此 qcrs 后续 E 推荐：

```text
FlashAttentionMetadataBuilder
owns:
member_to_cluster
member_to_column
```

D 本身不需要完成真实 builder integration。

D 只要求：

```text
execution helper显式接 cached tensors
测试模拟 initialize-once / reuse
```

不要为了 D 新建：

```text
RaggedGeometry
PlacementCacheManager
global singleton
```

---

# 21. Tangram 的 dedicated custom op 不直接照搬

Tangram pinned implementation使用：

```text
unified_attention_ragged
```

把 member-major transform 和 attention execution封装在一个 dedicated custom op 中。

但 qcrs 当前 v0.26 已经有：

```text
unified_kv_cache_update
        ↓ dummy dependency
unified_attention_with_output
```

并且 write/read已经分离。

因此 qcrs 冻结：

```text
NO new unified_attention_ragged custom op
```

未来 E：

```text
unified_kv_cache_update
├ Dense → existing do_kv_cache_update
└ Ragged → ragged_kv_cache_update

dummy dependency

unified_attention_with_output
├ Dense → existing impl.forward
└ Ragged → ragged_attention_forward
```

这样能保留：

```text
v0.26 source-native architecture
torch.compile side-effect ordering
future post-forward compaction semantics
```

---

# 22. D 的统一 `RaggedStepViews`

D 建议在：

```text
vllm/v1/attention/backends/ragged_layout.py
```

增加一个小型 immutable derived carrier：

```text
RaggedStepViews
```

它不是新的 ownership state。

推荐字段：

```text
cluster_block_table
[R,C,MaxPages]

member_block_table
[R,M,MaxPages]

member_slot_mapping
[Q,M]

member_seq_lens
[R,M]

query_start_loc
[R+1]

num_actual_tokens
Q
```

必要时携带极少数静态 scalar：

```text
page_group_size
block_size
num_layers
num_kv_heads
```

但避免复制已经在 Spec/placement存在的信息。

禁止把这些塞进 step view：

```text
state_version
page_counts authority
Scheduler allocator object
BlockPool
request lifecycle owner
```

它只是：

```text
B state
→ derived execution tensors
```

---

# 23. D-WRITE：冻结设计

建议新增：

```text
vllm/v1/attention/backends/ragged_forward.py
```

提供：

```text
ragged_kv_cache_update(...)
```

输入语义：

```text
key   [Q,Hkv,D]
value [Q,Hkv,D]

physical_cache
[P,Hp,B,2D]

layer_member_slots
[Q,Hkv]
```

处理：

```text
physical cache
→ as_virtual_block_view
→ [P*Hp,1,B,2D]

→ transpose/split
→ K/V cache [P*Hp,B,1,D]

key/value
→ reshape
→ [Q*Hkv,1,D]

slot
→ flatten
→ [Q*Hkv]

→ existing reshape_and_cache_flash
```

重要：

```text
cache virtual view
必须 zero-copy
```

但是：

```text
current-step key/value reshape
允许 materialize
```

因为 fused QKV split产生的 K/V 不一定 contiguous。

不要为了强制 current-step K/V zero-copy增加复杂实现。

---

# 24. D-DECODE：冻结设计

提供统一 read helper：

```text
ragged_attention_forward(...)
```

uniform decode走 fast path：

```text
Q [R,Hq,D]
→ [R,Hkv,q_per_kv,D]
→ [R*Hkv,q_per_kv,D]
```

metadata：

```text
block_table:
[R,Hkv,N]
→ [R*Hkv,N]

seq_lens:
[R,Hkv]
→ [R*Hkv]

query_start_loc:
arange(R*Hkv+1)
```

cache：

```text
physical
→ virtual
→ K/V [P*Hp,B,1,D]
```

然后调用最窄：

```text
flash_attn_varlen_func
```

不要把 Ragged 强塞入整个 normal：

```text
FlashAttentionImpl.forward
```

并伪装：

```text
num_kv_heads=1
```

因为 normal impl还携带：

```text
model semantic Hkv
descale shape
DCP/cascade
其他 backend branches
```

D Core没有必要承担这些复杂度。

---

# 25. D-PREFILL/MIXED：冻结设计

当：

```text
max_query_len > 1
```

走 member-major materialization。

但不创建复杂 gather-index系统。

输入：

```text
Q token-major:
[Q,Hq,D]
```

先 view：

```text
[Q,Hkv,q_per_kv,D]
```

然后：

```text
permute(1,0,2,3)
→ contiguous
→ [Hkv*Q,q_per_kv,D]
```

对应 FA sequence order：

```text
head-major / request-minor
```

metadata：

```text
member_block_table layer slice:
[R,Hkv,N]
→ permute(1,0,2)
→ [Hkv*R,N]

member_seq_lens:
[R,Hkv]
→ transpose
→ [Hkv*R]
```

member query starts：

```text
for h in 0..Hkv-1:
    query_start_loc[:R] + h*Q

flatten
append Hkv*Q
```

FA output：

```text
[Hkv*Q,q_per_kv,D]
→ view(Hkv,Q,q_per_kv,D)
→ permute(1,0,2,3)
→ [Q,Hkv,q_per_kv,D]
→ [Q,Hq,D]
```

首版允许这里的：

```text
permute + contiguous
```

因为 materialization 是 current-step Q/output，而不是整个 KV cache。

不要优化成 Triton fused pack，除非后续 profile证明值得。

---

# 26. D 完整 execution architecture

```text
RaggedWorkerPhysicalState
        │
        │ gather active requests
        ▼
cluster_rows [R,C,N]
source_E     [R,C]
query_start_loc [R+1]
        │
        ▼
build_ragged_step_views
        │
        ├─ physical positions [Q,C]
        ├─ group slots        [Q,C]
        ├─ member slots       [Q,M]
        ├─ member block table [R,M,N]
        └─ post-write lens    [R,M]
        │
        ├─────────────────────────────────────┐
        │                                     │
        ▼                                     ▼
D-WRITE                                  D-READ
        │                                     │
layer slots [Q,Hkv]                   layer block table
        │                             layer seq lens
flatten [Q*Hkv]                       member q starts
        │                                     │
K/V reshape                                  ┌┴─────────────┐
[Q*Hkv,1,D]                          decode fast      prefill/mixed
        │                            reshape           pack/copy
virtual cache                              │                 │
[P*Hp,1,B,2D]                             └──────┬──────────┘
        │                                       ▼
reshape_and_cache_flash           flash_attn_varlen_func
        │                                       │
        └──────────── shared physical KV ───────┘
```

---

# 27. D 支持范围

D Core冻结：

```text
FA2
A100-class CUDA
BF16 / FP16
KVQuantMode.NONE
head_size_v == head_size
full causal decoder attention
TP=1
PP=1
DCP=1
PCP=1
eager correctness
no prefix cache
no speculative decode
```

D 不声称支持：

```text
quantized KV
MLA
sliding window
attention sinks
ALiBi
soft-cap variants
cross attention
TP2
CUDA Graph
```

不要为了“未来可能需要”提前写这些分支。

---

# 28. D 明确不做的事情

D 不修改 production lifecycle：

```text
Attention.get_kv_cache_spec production activation
Scheduler Ragged dispatch
SchedulerOutput production wiring
GPUModelRunner production Ragged initialization
DefaultModelState production gather
ForwardContext production Ragged dispatch
real Engine model run
compression/scorer
physical reclamation
free/reuse
CUDA Graph
benchmark
```

如果 D implementation发现必须大范围修改这些文件才能完成 synthetic execution closure：

```text
DESIGN_CONFLICT
```

说明 D/E boundary破坏，不要硬顶。

---

# 29. File ownership freeze

D 推荐主要修改：

```text
vllm/v1/attention/backends/ragged_layout.py
```

职责：

```text
cached-tensor-based member transforms
group physical position/slot construction
RaggedStepViews
prefill/mixed pack mapping
```

新增：

```text
vllm/v1/attention/backends/ragged_forward.py
```

职责：

```text
ragged_kv_cache_update
ragged decode read
ragged prefill/mixed read
```

测试：

```text
tests/v1/attention/test_ragged_execution.py
tests/v1/attention/ragged_reference.py
```

必要时调整：

```text
tests/v1/test_ragged_kv_layout.py
```

来覆盖 C helper signature change。

D 不应触碰：

```text
scheduler.py
ragged_kv_cache_manager.py
sched/output.py
GPUModelRunner production flow
Attention.get_kv_cache_spec
```

---

# 30. 独立 Write oracle

构造：

```text
known cluster rows
known source E
known placement
known K/V payload
```

payload编码：

```text
f(token, head, plane, dim)
```

调用：

```text
ragged_kv_cache_update
```

测试不能用 production virtual helper反推 expected address。

必须调用 C1：

```text
resolve_kv_address
```

独立计算：

```text
page
column
offset
```

然后直接读取：

```text
physical_cache[page,column,offset,:D]
physical_cache[page,column,offset,D:]
```

验证 K/V exact。

这一个 test同时证明：

```text
group slots
member slots
K/V flatten order
virtual cache
reshape_and_cache_flash
physical storage mapping
```

---

# 31. 独立 Attention reference

read 测试不要使用另一份 production helper作为 expected。

tests-only：

```text
tests/v1/attention/ragged_reference.py
```

构造 logical K/V history。

对每：

```text
request
KV head
GQA query-head group
```

直接 PyTorch：

```text
scores = Q @ K.T * scale
apply causal mask
prob = softmax(scores)
out = prob @ V
```

再组合回 token-major输出。

这样：

```text
Ragged FA2
```

与：

```text
independent mathematical reference
```

比较。

---

# 32. 必要 D tests

测试不需要很多，但必须覆盖真正不同的机制。

## T1 Group physical slot

验证：

```text
source E
+ local offset
→ page / offset
→ physical slot
```

包含一个 block-boundary case。

## T2 Real CUDA write

真实：

```text
reshape_and_cache_flash
```

执行。

逐 cell验证 exact physical location。

## T3 Decode FA2

至少：

```text
1 request
2 requests
GQA ratio > 1
block boundary
```

与 independent PyTorch reference close。

## T4 Prefill

例如：

```text
q_lens = [3]
```

验证 pack / FA / inverse scatter。

## T5 Uneven multi-request prefill

例如：

```text
q_lens = [3,2]
```

## T6 Mixed

例如：

```text
q_lens = [1,3]
```

## T7 Non-uniform E

至少一个：

```text
req0 E = [15,31,8,20]
req1 E = [17,5,32,9]
```

证明同一 token不同 cluster写入不同 physical depth，read也使用正确 per-member visibility。

## T8 Custom placement

一个合法 non-identity `MemberPlacementMap`。

至少证明 write exact cells；read可加一个小 case。

## T9 Regression

C focused tests继续 PASS。

Dense source path没有被修改。

---

# 33. 不要过度测试 / 过度 validation

继续遵守当前项目原则：

```text
test contract
not every if
```

不需要：

```text
几十种 Hp
几十个边界
非法 dtype 大全
人为 malformed tensor fuzz
```

validation 也不要重复：

```text
placement bijection
quant mode
head_size_v
page padding
state_version
ownership capacity
```

这些已经由 A/B/C boundary保证。

D 只检查直接可能产生 silent execution corruption、且上游没有更合适 owner的条件。

能由 Torch自然给出明确 shape error的，不要额外包三层 check。

---

# 34. Hidden copy contract

必须区分：

## Cache

```text
[P,Hp,B,2D]
→ [P*Hp,1,B,2D]
```

绝对 zero-copy。

## Current-step Q/K/V

可以：

```text
reshape
gather
temporary member-major buffer
```

尤其 fused QKV split可能产生 non-contiguous K/V。

这类 current-step小 tensor copy是允许的。

不要将：

```text
cache hidden copy prohibition
```

错误扩展成：

```text
所有 QKV 都必须 zero-copy
```

---

# 35. D acceptance gate

D 只有以下全部成立才 CLOSED：

```text
A. group physical position exact

B. group physical slot exact

C. production-oriented member helper显式接 cached placement tensors
   不在每 step执行 torch.tensor(MemberPlacementMap)

D. known K/V
   → real reshape_and_cache_flash
   → exact physical(page,column,offset)

E. decode:
   Ragged FA2 ≈ independent PyTorch reference

F. prefill:
   Ragged FA2 ≈ independent PyTorch reference

G. mixed uneven-q:
   Ragged FA2 ≈ independent PyTorch reference

H. at least one non-uniform E case

I. at least one custom placement case

J. C regressions remain PASS

K. Dense source path untouched

L. no production activation
```

通过后：

```text
R1-D = PASS / CLOSED
```

下一阶段：

```text
R1-E
Production Activation + Real Engine Identity
```

---

# 36. E 留下来的事情

D 完成以后，E 才允许把这些 pieces接进真实 runtime：

```text
cache_config.page_group_size
→ RaggedAttentionSpec

Scheduler canonical Ragged manager
→ SchedulerOutput

Worker Ragged mirror
→ active step gather

FlashAttentionMetadataBuilder
→ cached member_to_cluster/member_to_column
→ RaggedStepViews

ForwardContext
→ Ragged member slots/metadata

unified_kv_cache_update
→ Ragged write dispatch

unified_attention_with_output
→ Ragged read dispatch
```

然后真实：

```text
Qwen3-0.6B
A100
FA2
BF16
eager
```

比较：

```text
Dense vs Ragged identity
token IDs / logits
```

并验证：

```text
single request
block crossing
continuous batching
mixed prefill+decode
chunked prefill
no page leak
```

只有 E 通过以后，才可以声称：

```text
Ragged KV runtime 已经真正进入 vLLM Engine
```

---

# 37. D 阶段最值得学习的 vLLM 知识

## 37.1 Logical vs physical position

```text
RoPE / model semantics
≠
KV storage coordinate
```

这是 compression/reclamation runtime 的核心。

## 37.2 PagedAttention addressing

理解：

```text
position
→ block table
→ block ID / offset
→ slot mapping
```

能把 vLLM KV memory management从“函数调用”提升到“地址模型”。

## 37.3 ForwardContext

模型 Attention module不直接拥有当前 batch dynamic metadata。

真正链路：

```text
ModelRunner build metadata
→ set_forward_context
→ Attention custom op读取
```

这是 runtime state进入 model graph的重要 seam。

## 37.4 KV write / read separation

当前 qcrs v0.26：

```text
unified_kv_cache_update
→ dependency
→ unified_attention_with_output
```

对：

```text
torch.compile
mutation ordering
post-forward compaction
```

都很重要。

## 37.5 FlashAttention varlen

FA 真正关心：

```text
有多少 sequences
每个 sequence query多长
KV有效长度多少
对应哪些 paged blocks
```

它并不要求一个 sequence必须等于“一个用户 request”。

因此我们才能把：

```text
(request, KV-head)
```

重新解释成 FA sequence。

## 37.6 GQA + virtual single-head cache

模型：

```text
Hq > Hkv
```

virtual cache每 member只有：

```text
1 KV head
```

因此每 member sequence携带：

```text
q_per_kv = Hq/Hkv
```

个 query heads。

这正是 Ragged read adapter成立的关键。

---

# 38. 最终架构认知

当前项目不是：

```text
把 Tangram code复制到 vLLM
```

而是：

```text
qcrs Scheduler-owned physical lifecycle
+
Tangram virtual-block storage insight
+
qcrs v0.26 native split write/read seam
```

最终：

```text
Scheduler canonical ownership
        ↓
Worker Ragged physical mirror
        ↓
RaggedStepViews
        ↓
┌──────────────────────┬──────────────────────┐
│                      │                      │
KV Write               Attention Read
│                      │
member virtual slots   member sequences
│                      │
reshape_and_cache      flash_attn_varlen
│                      │
└────────── shared virtual cache ──────────────┘
        ↓
physical [P,Hp,B,2D]
```

D 要证明：

> **这套 execution geometry 在真实 CUDA 和 FA2 primitive 上正确。**

E 才证明：

> **它已经真正进入 vLLM Engine lifecycle。**

---

# 39. Frozen Decision Summary

```text
R1-D
= ONE SLICE

Slice:
P1-V2-R1-D-RAGGED-GPU-EXECUTION-01
```

内部 acceptance：

```text
D-WRITE
D-DECODE
D-PREFILL-MIXED
```

但不拆成独立任务。

冻结：

```text
canonical metadata:
[R,Hkv,...] request-major/head-minor

write:
token-major/head-minor flatten
prefill不需要额外 write packing

decode FA execution:
request-major/head-minor reshape fast path

prefill/mixed FA execution:
head-major/request-minor
permute + contiguous + inverse permute

placement tensors:
initialize once
cached
helper显式接 tensor

cache virtual view:
zero-copy

current-step Q/K/V:
必要时允许 copy

kernel reuse:
reshape_and_cache_flash
flash_attn_varlen_func

new custom op:
NO

production Engine activation:
NO in D
```

只有出现以下真正 architecture conflict 才允许重新拆 D：

```text
1. FA2 paged read无法消费 frozen virtual-cache geometry；

2. qcrs separated write/read seam无法承载 member-major read，
   且必须同时打开多个 frozen unsupported subsystem；

3. prefill/mixed需要改变 physical address contract，
   而不仅仅是数据 packing。
```

仅仅：

```text
代码量多
测试量多
```

不是拆分理由。

---

# 40. Go / No-Go

基于当前源码与 evidence：

```text
C real CUDA zero-copy alias    PASS
qcrs v0.26 write seam          AUDITED
qcrs v0.26 read seam           AUDITED
Tangram grouped slots          AUDITED
Tangram member-sequence path   AUDITED
Worker per-cluster state       AVAILABLE
```

结论：

```text
R1-D DESIGN: GO
Execution: ONE COMPLETE SLICE
```

D 完成后必须停在：

```text
PASS_PENDING_WEB_REVIEW
```

不得自动进入 `R1-E`。
