# P1-V2-R1-D：Ragged KV + FlashAttention2 详细理解笔记

> 主题：Ragged KV Cache 如何复用现有 FlashAttention2（FA2）  
> 重点：理解 FA2 需要什么、Ragged 当前有什么、为什么要做 shape/sequence 转换、Decode 与 Prefill/Mixed 为什么分开处理  
> 说明：本文重点是“理解链路”，不是只记代码语法。

---

# 1. 本轮最核心的结论

整个 `ragged_attention_forward()` 的本质不是重新设计 Attention，而是做一层 **Ragged → FA2 的接口适配**。

```text
Ragged 自己的 KV 组织方式
        ↓
重新组织 Query + block_table + seq_lens + query_start_loc
        ↓
把 (request, real_kv_head)
包装成 FA2 能理解的 pseudo-sequence
        ↓
调用标准 FlashAttention2
        ↓
把输出重新恢复成模型标准 [Q, Hq, D]
```

最关键的一句话：

```text
Dense：
一个 request 就可以直接作为一条 FA2 sequence。

Ragged：
由于 virtual KV cache 的显式 KV-head 维退化成 1，
必须把一个 request 按 real_kv_head 拆成多条 pseudo-sequence：

pseudo-sequence = (request, real_kv_head)
```

---

# 2. 先明确符号

后续统一使用：

```text
R     = num_requests
Q     = num_actual_tokens，本 step 的总 Query token 数
Hq    = Query head 数
Hkv   = KV head 数
G     = queries_per_kv_head = Hq / Hkv
D     = head_size
B     = block_size
Pmax  = 每条 KV history 最多拥有多少个 blocks
V     = virtual block 总数
```

例如：

```text
R   = 2

Request0:
q_len = 3

Request1:
q_len = 2

Q   = 5
Hq  = 8
Hkv = 2
G   = 4
D   = 128
```

---

# 3. 第一层基础：FA2 到底需要什么

## 3.1 FA2 处理的不是“一个大矩阵”

在 varlen Attention 中，FA2 接收的是若干条相互独立的 Attention sequence。

可以把一条 FA2 sequence 理解成：

```text
一段连续的 Query
+
与它对应的一条完整 K/V history
```

对第 i 条 sequence，逻辑上计算：

```text
Attention(Q_i, K_i, V_i)
```

不同 sequence 之间互不做 Attention。

---

## 3.2 FA2 的 Q

FA2 的 Query 一般被 packed 到一起：

```text
q.shape = [total_query_tokens, Hq_FA, D]
```

例如：

```text
sequence0:
3 个 Query token

sequence1:
2 个 Query token
```

则：

```text
q.shape = [5, Hq_FA, D]
```

q 内部可能是：

```text
seq0_q0
seq0_q1
seq0_q2
seq1_q0
seq1_q1
```

---

## 3.3 `cu_seqlens_q`

因为 q 被 packed 在一起，FA2 需要知道不同 sequence 的边界。

例如：

```text
cu_seqlens_q = [0, 3, 5]
```

表示：

```text
sequence0 = q[0:3]
sequence1 = q[3:5]
```

因此：

```text
cu_seqlens_q
=
Query sequence 边界表
```

---

## 3.4 paged KV cache

Paged KV 模式下，K/V 通常类似：

```text
K cache:
[num_blocks, B, Hkv_FA, D]

V cache:
[num_blocks, B, Hkv_FA, D]
```

这是一个全局 KV block pool。

它本身不知道：

```text
哪个 block 属于哪一条 sequence
```

因此需要：

```text
block_table
```

---

## 3.5 `block_table`

```text
block_table.shape
=
[num_sequences, Pmax]
```

例如：

```text
sequence0 -> [20,24,28]
sequence1 -> [31,40,45]
```

含义：

```text
block_table[i]
=
第 i 条 sequence 的 KV block 路径
```

---

## 3.6 `seqused_k / seq_lens`

`block_table` 只能告诉 FA2：

```text
这条 sequence 使用了哪些 blocks
```

不能告诉它最后一个 block 实际用了多少 token。

例如：

```text
B = 16

block_table[0] = [20,24,28]
```

最多能容纳：

```text
3 × 16 = 48
```

但实际可能只有：

```text
34 个有效 KV token
```

所以需要：

```text
seq_lens[0] = 34
```

因此：

```text
block_table = 在哪里

seq_lens = 实际有效长度
```

---

## 3.7 一条 FA2 sequence 完整需要的信息

对于 sequence i：

```text
Q:
q[cu_seqlens_q[i] : cu_seqlens_q[i+1]]

KV 地址:
block_table[i]

KV 有效长度:
seq_lens[i]
```

然后 FA2 执行：

```text
Attention(Q_i, K_i, V_i)
```

---

# 4. Dense 下为什么简单

## 4.1 Dense KV cache

Dense 当前 layer 的 KV cache 可以理解为：

```text
[num_blocks, B, Hkv, D]
```

关键是：

```text
Hkv 仍然是一个显式 Tensor 维度
```

例如：

```text
block20:

token0:
    KV head0
    KV head1

token1:
    KV head0
    KV head1
...
```

所以一个 request 的：

```text
block_table = [20,24,28]
```

这一行就可以同时服务所有 KV heads。

---

## 4.2 Dense 下一个 request 就是一条 FA2 sequence

假设：

```text
Hq = 4
Hkv = 2
```

GQA：

```text
Q head0、Q head1 -> KV head0
Q head2、Q head3 -> KV head1
```

Dense 可以一次给 FA2：

```text
Q = [Lq, 4, D]
K = [Lk, 2, D]
V = [Lk, 2, D]
```

FA2 内部按 head 独立计算：

```text
O_h0 = Attention(Q_h0, K_h0, V_h0)
O_h1 = Attention(Q_h1, K_h0, V_h0)

O_h2 = Attention(Q_h2, K_h1, V_h1)
O_h3 = Attention(Q_h3, K_h1, V_h1)
```

注意：

```text
sequence ≠ head
```

Dense 中：

```text
sequence = request

sequence 内部还包含多个 heads
```

---

# 5. 为什么 Ragged 不一样

## 5.1 Ragged physical cache

当前 Ragged backing storage：

```text
physical_cache
=
[P, Hp, B, 2D]
```

其中不再显式存在：

```text
layer
kv_head
```

而是由 placement metadata 表达，例如：

```text
member_to_cluster
member_to_column
```

一个 member 对应：

```text
(layer, kv_head)
```

---

## 5.2 Virtual KV cache

通过 `_virtual_kv_cache()`，Ragged backing storage 被 zero-copy 地解释为：

```text
key_cache
=
[V, B, 1, D]

value_cache
=
[V, B, 1, D]
```

最关键的是：

```text
Hkv_FA = 1
```

真实 KV head 身份已经不再由：

```text
cache[..., head, :]
```

表达。

而是进入：

```text
virtual block id
```

中。

---

# 6. 为什么 virtual cache 只有一个 head 仍然可以算

假设：

```text
Hq  = 4
Hkv = 2
G   = 2
```

真实模型：

```text
Qh0,Qh1 -> KVh0
Qh2,Qh3 -> KVh1
```

Dense 一次计算：

```text
Group0:
Qh0,Qh1
×
KVh0

Group1:
Qh2,Qh3
×
KVh1
```

这两个 KV-head group 数学上本来就是彼此独立的。

因此可以拆成两个 Attention 问题：

```text
Attention(
    Q_group0,
    K_h0,
    V_h0
)

Attention(
    Q_group1,
    K_h1,
    V_h1
)
```

Ragged 正是利用了这一点。

它把 Dense 原本在 FA2 内部隐式处理的 KV-head group，显式展开成多个 pseudo-sequence。

---

# 7. Ragged 的核心转换：pseudo-sequence

假设：

```text
Request0:

real KV head0
-> virtual blocks [20,24,28]

real KV head1
-> virtual blocks [31,35,39]
```

virtual cache：

```text
[V,B,1,D]
```

如果仍然：

```text
sequence0 = Request0
```

那么：

```text
block_table[0]
```

到底应该填：

```text
[20,24,28]
```

还是：

```text
[31,35,39]
```

一条 FA2 sequence 只有一行 block_table，而 virtual cache 又只有 1 个显式 KV head，因此无法同时表达两个真实 KV heads。

所以 Ragged 改成：

```text
pseudo-sequence0 = Request0 / real KV head0
pseudo-sequence1 = Request0 / real KV head1
```

因此：

```text
num_FA2_sequences
=
R * Hkv
```

---

# 8. 每条 pseudo-sequence 的 Query 是什么

假设：

```text
Hq  = 8
Hkv = 2
G   = 4
```

那么：

```text
Q head0~3 -> real KV head0

Q head4~7 -> real KV head1
```

所以：

```text
pseudo-sequence0 = R0/H0
```

其中：

```text
Q:
R0 中属于 KV0 的 G=4 个 Query heads

K/V:
R0/H0 的 virtual KV history
```

pseudo-sequence1：

```text
Q:
R0 中属于 KV1 的 G=4 个 Query heads

K/V:
R0/H1 的 virtual KV history
```

因此每条 pseudo-sequence 对 FA2 来说仍然是合法 GQA：

```text
Hq_FA = G
Hkv_FA = 1
```

也就是：

```text
G 个 Query heads
共享 1 个 KV head
```

---

# 9. 当前 layer 进入 Attention 前已有的数据

当前 layer 切片后：

```text
query
=
[Q, Hq, D]
```

---

```text
layer_block_table
=
[R, Hkv, Pmax]
```

含义：

```text
layer_block_table[r,h,:]
=
request r
当前 layer 的 real KV head h
对应的 virtual block 路径
```

---

```text
layer_seq_lens
=
[R, Hkv]
```

含义：

```text
layer_seq_lens[r,h]
=
request r
real KV head h
当前有效 KV 长度
```

---

```text
views.query_start_loc
=
[R+1]
```

描述：

```text
原始 packed query 中
每个 request 的 Query token 边界
```

---

# 10. 为什么 Decode 和 Prefill/Mixed 要分开

## 10.1 Decode

纯 Decode：

```text
每个 request 当前只有 1 个 Query token
```

所以：

```text
每个 (request, kv_head)
也只有 1 个 Query token
```

例如：

```text
R0/H0 -> 1 token
R0/H1 -> 1 token
R1/H0 -> 1 token
R1/H1 -> 1 token
```

每条 pseudo-sequence 只有一个元素，因此不需要重新聚集多个 Query token。

可以直接 reshape。

---

## 10.2 Prefill / Mixed

例如：

```text
R0 q_len = 3
R1 q_len = 2
```

对于：

```text
R0/H0
```

这条 pseudo-sequence 有：

```text
R0t0/H0
R0t1/H0
R0t2/H0
```

3 个 Query token。

FA2 要求：

```text
一条 sequence 的 Query
是 packed_query 中一段连续区域
```

因此：

```text
R0t0/H0
R0t1/H0
R0t2/H0
```

必须连续。

原始 Query 却是 token-major：

```text
R0t0/H0
R0t0/H1

R0t1/H0
R0t1/H1

R0t2/H0
R0t2/H1
```

同一个 H0 的 Query 被 H1 插开。

所以必须重排。

---

# 11. 为什么 Mixed/Prefill 要把 Hkv 放到前面

假设只有一个 request，3 个 token，2 个 KV heads。

定义：

```text
A = t0/H0
B = t0/H1

C = t1/H0
D = t1/H1

E = t2/H0
F = t2/H1
```

原始 `[token, head]`：

```text
        H0    H1

t0      A     B
t1      C     D
t2      E     F
```

原始逻辑顺序：

```text
A B C D E F
```

但 pseudo-sequence 需要：

```text
R0/H0:
A C E

R0/H1:
B D F
```

所以希望：

```text
A C E | B D F
```

原因不是 FA2 规定 “H 必须放前面”。

真正原因是：

```text
A、C、E 都必须使用同一行 R0/H0 的 block_table，
所以它们必须属于同一条 FA2 pseudo-sequence。

FA2 又用 q[start:end] 表示一条 sequence，
因此 A、C、E 必须在 packed q 中连续。
```

而最简单的实现方法就是：

```text
[token, head]
↓
[head, token]
```

因此：

```text
[Q, Hkv, ...]
↓
[Hkv, Q, ...]
```

---

# 12. Prefill/Mixed：构造 packed_query

原始：

```text
query.shape
=
[Q, Hq, D]
```

例如：

```text
[5,8,128]
```

---

## 12.1 第一次 reshape

```python
query.reshape(
    Q,
    Hkv,
    G,
    D,
)
```

得到：

```text
[Q,Hkv,G,D]
```

例如：

```text
[5,2,4,128]
```

含义：

```text
[token,
 real_kv_head_group,
 query_head_inside_group,
 dim]
```

这里只是把：

```text
Hq
```

拆成：

```text
Hkv × G
```

没有改变 token/head 的排列关系。

---

## 12.2 `permute(1,0,2,3)`

```text
[Q,Hkv,G,D]
↓
[Hkv,Q,G,D]
```

这是整个 Mixed/Prefill packing 中真正体现算法意图的核心。

它把：

```text
token-major
```

变成：

```text
head-major
```

例如：

```text
H0:
R0t0
R0t1
R0t2
R1t0
R1t1

H1:
R0t0
R0t1
R0t2
R1t0
R1t1
```

因为 Q 轴内部原本已经是：

```text
R0所有token
R1所有token
...
```

所以每个 head 区域内部自动形成：

```text
H0/R0
H0/R1
H1/R0
H1/R1
```

这些 pseudo-sequence。

---

## 12.3 `.contiguous()`

`permute()` 一般只改变：

```text
shape + stride
```

而不立即搬动底层 storage。

所以：

```text
.contiguous()
```

会按当前 head-major 逻辑顺序重新整理出连续内存。

可以理解为：

```text
逻辑上：
H0 所有 token
H1 所有 token

↓

物理内存也真正按这个顺序排好
```

---

## 12.4 最后 reshape

```python
.reshape(
    Hkv * Q,
    G,
    D,
)
```

得到：

```text
packed_query.shape
=
[Hkv*Q, G, D]
```

例如：

```text
[10,4,D]
```

注意：

```text
Hkv*Q
=
所有 pseudo-sequence 的 Query token 总数
```

并不是 sequence 数。

真正 sequence 数：

```text
Hkv*R
```

---

# 13. packed_query 的最终顺序

例子：

```text
R0 q_len = 3
R1 q_len = 2
Hkv = 2
```

最终：

```text
index0 = H0/R0t0
index1 = H0/R0t1
index2 = H0/R0t2

index3 = H0/R1t0
index4 = H0/R1t1

index5 = H1/R0t0
index6 = H1/R0t1
index7 = H1/R0t2

index8 = H1/R1t0
index9 = H1/R1t1
```

因此：

```text
sequence0 = H0/R0 = packed_query[0:3]
sequence1 = H0/R1 = packed_query[3:5]
sequence2 = H1/R0 = packed_query[5:8]
sequence3 = H1/R1 = packed_query[8:10]
```

---

# 14. `packed_output`

Mixed/Prefill：

```python
packed_output = torch.empty_like(packed_query)
```

因为 Query 已经真正被重排成 head-major。

FA2 输出也会按 head-major 顺序产生。

而模型最后需要：

```text
[Q,Hq,D]
```

所以必须先用一个临时 packed_output，之后再恢复。

Decode 不需要这一步，因为 Decode 可以让：

```text
packed_output
```

直接作为：

```text
output
```

的 view。

---

# 15. block_table 的转换

原来：

```text
layer_block_table
=
[R,Hkv,Pmax]
```

例如：

```text
R0:
H0 -> [20,24,28]
H1 -> [31,35,39]

R1:
H0 -> [50,54,58]
H1 -> [61,65,69]
```

原始顺序：

```text
R0/H0
R0/H1
R1/H0
R1/H1
```

但 packed_query 已经采用：

```text
H0/R0
H0/R1
H1/R0
H1/R1
```

所以 block_table 也必须改成完全一致的顺序。

---

## 15.1 permute

```python
layer_block_table.permute(1,0,2)
```

得到：

```text
[Hkv,R,Pmax]
```

---

## 15.2 contiguous + reshape

```python
.contiguous()
.reshape(Hkv * R, -1)
```

最终：

```text
block_table
=
[Hkv*R,Pmax]
```

例如：

```text
row0 = H0/R0
row1 = H0/R1
row2 = H1/R0
row3 = H1/R1
```

---

# 16. seq_lens 的转换

原始：

```text
layer_seq_lens
=
[R,Hkv]
```

例如：

```text
[
    [34,18],
    [51,27],
]
```

含义：

```text
R0/H0 = 34
R0/H1 = 18

R1/H0 = 51
R1/H1 = 27
```

执行：

```python
layer_seq_lens.transpose(0,1)
```

变：

```text
[Hkv,R]
```

内容：

```text
[
    [34,51],
    [18,27],
]
```

然后：

```text
reshape(-1)
```

得到：

```text
[34,51,18,27]
```

对应：

```text
H0/R0
H0/R1
H1/R0
H1/R1
```

再次与 packed_query / block_table 完全一致。

---

# 17. 为什么要构造新的 query_start_loc

原始：

```text
views.query_start_loc
=
[R+1]
```

例如：

```text
[0,3,5]
```

表示：

```text
R0 -> query[0:3]
R1 -> query[3:5]
```

但是现在 FA2 看到的是：

```text
H0/R0
H0/R1
H1/R0
H1/R1
```

所以需要新的：

```text
cu_seqlens_q
```

来描述这 4 条 pseudo-sequence。

---

# 18. `head_offsets`

```python
head_offsets =
torch.arange(Hkv) * Q
```

如果：

```text
Hkv = 2
Q = 5
```

得到：

```text
[0,5]
```

含义：

```text
H0 区域起点 = 0

H1 区域起点 = 5
```

---

# 19. 复制 request 起点到每个 head 区域

原始：

```text
views.query_start_loc = [0,3,5]
```

去掉最后终点：

```text
[0,3]
```

这是每个 request 的起点。

广播：

```text
[[0,3]]
+
[
 [0],
 [5]
]
```

得到：

```text
[
 [0,3],
 [5,8]
]
```

表示：

```text
H0/R0 start = 0
H0/R1 start = 3

H1/R0 start = 5
H1/R1 start = 8
```

flatten：

```text
[0,3,5,8]
```

最后补总结束位置：

```text
Hkv * Q = 10
```

得到：

```text
query_start_loc
=
[0,3,5,8,10]
```

---

# 20. FA2 如何解释新的 query_start_loc

```text
sequence0:
q[0:3]
=
H0/R0

sequence1:
q[3:5]
=
H0/R1

sequence2:
q[5:8]
=
H1/R0

sequence3:
q[8:10]
=
H1/R1
```

所以：

```text
num_sequences
=
Hkv * R
```

而：

```text
total_query_tokens
=
Hkv * Q
```

这两个一定不要混。

---

# 21. 调用 `flash_attn_varlen_func`

最终：

```python
flash_attn_varlen_func(
    q=packed_query,
    k=key_cache,
    v=value_cache,
    out=packed_output,
    cu_seqlens_q=query_start_loc,
    max_seqlen_q=max_query_len,
    seqused_k=seq_lens,
    max_seqlen_k=views.max_kv_len,
    softmax_scale=softmax_scale,
    causal=True,
    block_table=block_table,
    fa_version=fa_version,
)
```

可以翻译成：

```text
FA2：

我给你若干条 pseudo-sequence。

每一条：
- Query 区间已经由 cu_seqlens_q 指定
- KV block 地址由 block_table[i] 指定
- KV 有效长度由 seq_lens[i] 指定
- Query heads 数量为 G
- KV heads 数量为 1
```

---

# 22. 一条 pseudo-sequence 完整例子

比如：

```text
sequence0 = H0/R0
```

FA2 收到：

```text
Q:
packed_query[0:3]

shape:
[3,G,D]
```

K/V：

```text
virtual cache:
[V,B,1,D]
```

地址：

```text
block_table[0]
```

长度：

```text
seq_lens[0]
```

所以逻辑上计算：

```text
R0 当前所有 Query token 中
属于 real KV head0 的 G 个 Query heads

×

R0 / KV head0 的完整 KV history
```

这和 Dense 内部对应 KV-head group 的计算数学上相同。

---

# 23. `max_seqlen_q`

```text
max_seqlen_q
=
所有 pseudo-sequence 中最大 Query 长度
```

如果原 request：

```text
3,2
```

拆成：

```text
3,2,3,2
```

最大值仍为：

```text
3
```

所以直接沿用：

```text
views.max_query_len
```

---

# 24. `max_seqlen_k`

类似：

```text
seq_lens =
[34,51,18,27]
```

则：

```text
max_seqlen_k
=
最大 KV 长度
```

例如：

```text
51
```

---

# 25. causal=True

拆成 pseudo-sequence 并不会破坏 causal 关系。

例如：

```text
t0
t1
t2
```

拆成：

```text
H0:
t0
t1
t2

H1:
t0
t1
t2
```

token 顺序仍然保持一致。

因此每个 pseudo-sequence 内仍可以正确执行 causal Attention。

---

# 26. FA2 输出是什么顺序

FA2 把结果写入：

```text
packed_output
=
[Hkv*Q,G,D]
```

其顺序与 packed_query 一样：

```text
H0/R0
H0/R1
H1/R0
H1/R1
```

所以模型最终不能直接拿来作为：

```text
[Q,Hq,D]
```

必须逆转换。

---

# 27. 为什么只有 Mixed/Prefill 要 unpack

Decode 前面：

```python
packed_output = output.reshape_as(packed_query)
```

所以：

```text
packed_output
```

本身就是最终 output 的 view。

FA2 写 packed_output，就是直接写最终 output。

---

Mixed/Prefill 前面：

```python
packed_output = torch.empty_like(packed_query)
```

这是独立 buffer。

所以必须恢复。

---

# 28. output 恢复过程

代码：

```python
output.copy_(
    packed_output.view(
        Hkv,
        Q,
        G,
        D,
    )
    .permute(1,0,2,3)
    .reshape_as(output)
)
```

---

## 28.1 view

原来：

```text
[Hkv*Q,G,D]
```

view 成：

```text
[Hkv,Q,G,D]
```

仍然是：

```text
head-major
```

---

## 28.2 permute

```text
[Hkv,Q,G,D]
↓
[Q,Hkv,G,D]
```

恢复：

```text
token-major
```

---

## 28.3 reshape

因为：

```text
Hq = Hkv * G
```

所以：

```text
[Q,Hkv,G,D]
↓
[Q,Hq,D]
```

恢复模型标准输出。

---

# 29. PACK / UNPACK 完整对称关系

进入 FA2：

```text
[Q,Hq,D]

↓ reshape

[Q,Hkv,G,D]

↓ permute

[Hkv,Q,G,D]

↓ contiguous + reshape

[Hkv*Q,G,D]
```

---

FA2 输出：

```text
[Hkv*Q,G,D]
```

恢复：

```text
↓ view

[Hkv,Q,G,D]

↓ permute

[Q,Hkv,G,D]

↓ reshape

[Q,Hq,D]
```

因此：

```text
PACK:
token-major -> head-major

UNPACK:
head-major -> token-major
```

---

# 30. WRITE 路径与 READ 路径的统一理解

Ragged 的 WRITE 和 READ 其实非常对称。

## WRITE

Dense：

```text
一个 token
→ 一个 slot

所有 heads
通过显式 head 维写入
```

Ragged：

```text
一个 (token, kv_head)
→ 一个 virtual slot
```

所以：

```text
[T,Hkv,D]
→
[T*Hkv,1,D]
```

---

## READ

Dense：

```text
一个 request
→ 一行 block_table

所有 heads
通过显式 head 维读取
```

Ragged：

```text
一个 (request, kv_head)
→ 一行 virtual block_table
```

所以：

```text
[R,Hkv,Pmax]
→
[R*Hkv,Pmax]
```

因此可以记：

```text
WRITE：

Dense：
token

Ragged：
(token, head)


READ：

Dense：
request

Ragged：
(request, head)
```

---

# 31. RaggedStepViews 在整条链路里的作用

主要结构：

```text
cluster_block_table
[R,C,Pmax]

member_block_table
[R,M,Pmax]

member_slot_mapping
[Q,M]

member_seq_lens
[R,M]

query_start_loc
[R+1]
```

其中：

```text
M = L * Hkv
```

member：

```text
member
=
(layer, kv_head)
```

当前 layer 使用：

```text
member_start = layer_idx * Hkv
member_end   = member_start + Hkv
```

得到：

```text
layer_block_table
=
[R,Hkv,Pmax]

layer_seq_lens
=
[R,Hkv]

layer_slots
=
[Q,Hkv]
```

所以：

```text
M
=
全局 member/address lookup 空间

Hkv
=
当前 layer 的真实执行空间
```

---

# 32. C / M / Hkv 三个维度不要混

```text
C
=
physical grouping / cluster space
```

```text
M
=
global semantic member space
=
L * Hkv
```

```text
Hkv
=
当前 layer 的 KV-head execution space
```

链路：

```text
cluster metadata
[R,C,...]

↓ placement mapping

member metadata
[R,M,...]

↓ 当前 layer slice

[R,Hkv,...]

↓ FA2 pseudo-sequence packing

[R*Hkv,...]
```

---

# 33. 物理地址、virtual 地址、sequence id 不要混

这是一个非常重要的区分。

## physical page

```text
physical_page_id
```

是真实 Ragged backing storage 的 page。

---

## virtual block

```text
virtual_block
=
physical_page * Hp + column
```

它解决：

```text
某个 member/head 的数据
在 virtual cache 哪里
```

即：

```text
where
```

---

## FA2 sequence id

例如：

```text
sequence_id
=
某个 (request, kv_head)
```

它解决：

```text
这次 Attention 问题属于谁
```

即：

```text
who
```

所以：

```text
virtual address
≠
sequence id
```

---

# 34. 为什么已经有 virtual block 还需要 R*Hkv sequence

因为 virtual block 只能回答：

```text
数据在哪里
```

不能回答：

```text
哪些 Query 应该和这一组 KV history 做 Attention
```

FA2 仍然需要：

```text
sequence i
```

来把：

```text
一段连续 Q
+
一行 block_table
+
一个 seq_len
```

绑定在一起。

所以：

```text
R*Hkv
```

不是为了“消掉 R”。

而是把：

```text
(request, kv_head)
```

二维身份重新编号成：

```text
FA2 sequence_id
```

---

# 35. `reshape / view / permute / transpose / contiguous`

## reshape

```text
主要改变 shape。
尽量共享 storage，必要时可能 copy。
```

例如：

```text
[Q,Hq,D]
→
[Q,Hkv,G,D]
```

这里主要是把：

```text
Hq
```

拆成：

```text
Hkv × G
```

---

## view

```text
严格要求当前 storage/stride
允许直接共享底层内存。
```

不会主动帮你重排数据。

---

## permute

```text
重新排列维度顺序。
通常只修改 shape/stride，
不立即复制 storage。
```

例如：

```text
[Q,Hkv,G,D]
→
[Hkv,Q,G,D]
```

---

## transpose(a,b)

```text
只交换两个指定维度。
```

二维时：

```text
transpose(0,1)
≈
permute(1,0)
```

---

## contiguous

```text
按照当前逻辑顺序
真正重新整理连续物理内存。
```

常见组合：

```text
permute
→
contiguous
→
reshape
```

---

# 36. 为什么不能只用 view

因为：

```text
view
```

只能改变 shape，不会主动把：

```text
A B C D E F
```

重新排列成：

```text
A C E B D F
```

Mixed/Prefill 真正需要的是：

```text
token-major
→
head-major
```

这是顺序变化，不只是 shape 变化。

所以必须有：

```text
permute
```

来改变逻辑维度顺序。

---

# 37. 为什么显式 `.contiguous().reshape()`

理论上某些情况下：

```text
permute().reshape()
```

可能让 PyTorch 自动 copy。

但这里显式：

```text
permute
.contiguous()
.reshape
```

语义更清楚：

```text
1. 我要 head-major
2. 我明确允许这里 materialize
3. 然后再 flatten
```

同时符合当前 D slice 的工程边界：

```text
current-step Q/K/V 可以 materialize

cache 本身必须 zero-copy
```

---

# 38. 常见困惑 1：FA2 sequence 是不是一个 head

不是。

```text
sequence
=
一组互不串扰的 Attention token 区间
```

Dense：

```text
sequence = request
```

一条 sequence 内仍有多个 heads。

Ragged：

```text
pseudo-sequence = (request, real_kv_head)
```

这是为了适配 virtual cache `Hkv_FA=1` 的工程包装。

---

# 39. 常见困惑 2：为什么一个 pseudo-sequence 只有 1 个 KV head 也能算

因为：

```text
G 个 Query heads
共享 1 个 KV head
```

本身就是合法 GQA。

例如：

```text
Q shape:
[Lq,G,D]

K/V shape:
[Lk,1,D]
```

可以计算：

```text
G 个 Query heads
分别和同一个 KV head 做 Attention
```

---

# 40. 常见困惑 3：为什么 Dense 不拆成 R/H0、R/H1

因为 Dense 的：

```text
K/V tensor
```

保留了显式：

```text
Hkv
```

维度。

FA2 本身就可以在一条 request sequence 内按 head 独立计算。

Ragged virtual cache：

```text
[...,1,D]
```

已经看不到真实 Hkv，所以只能通过：

```text
pseudo-sequence
+
不同 block_table row
```

恢复真实 head 身份。

---

# 41. 常见困惑 4：为什么 Mixed 要 H 放前面

不是 FA2 要求：

```text
H 必须第一维
```

而是：

```text
同一 (request, kv_head)
的多个 Query token
必须在 packed_query 中连续。
```

当前实现最简单的方法：

```text
[Q,H]
→
[H,Q]
```

这样同一个 head 的所有 token 聚在一起，而 Q 内部原本又已经按 request 连续，因此自然得到：

```text
H0/R0
H0/R1
H1/R0
H1/R1
```

---

# 42. 常见困惑 5：block_table 是不是“历史”

更准确地说：

```text
block_table
=
当前完整 KV sequence 的 block 地址映射
```

既可以包含之前已经缓存的 KV，也可以包含当前 step 刚写进去的新 KV。

所以：

```text
Query
=
当前 step 要计算输出的 token

KV
=
到当前时刻完整可见的上下文
```

---

# 43. 常见困惑 6：Q_len 和 KV_len 为什么不同

一个 request 可以有：

```text
q_len = 当前 step 处理多少个 Query token
```

同时：

```text
kv_len = 到当前时刻完整 KV history 有多长
```

例如：

```text
过去已有 100 个 KV token
当前 Decode 新增 1 个 token

q_len  = 1
kv_len = 101
```

或者：

```text
过去已有 100
当前 chunked prefill 新增 32

q_len  = 32
kv_len = 132
```

这两者不能混。

---

# 44. 常见困惑 7：为什么 Decode 可以直接 reshape

因为 Decode：

```text
每个 (request, kv_head)
只有 1 个 Query token
```

不存在：

```text
同一 pseudo-sequence 的多个 token
被其他 head 插开
```

的问题。

所以不需要真正重排 token。

---

# 45. 常见困惑 8：为什么 Prefill/Mixed 要 materialize

因为 Mixed/Prefill：

```text
同一 pseudo-sequence
可能有多个 Query token
```

原始 query 是 token-major。

要变成 head-major，就需要：

```text
permute
+
contiguous
```

真正重排。

---

# 46. 当前 D Slice 的理解边界

当前 D 的重点是：

```text
focused / synthetic real-CUDA
WRITE + READ closure
```

并且：

```text
production Ragged Engine
仍然没有正式打开
```

因此当前可以确认的是：

```text
Ragged metadata
→ virtual cache
→ WRITE
→ READ / FA2
→ output restore
```

这一条 focused execution chain 已经打通。

但不要把当前代码过度解释成：

```text
生产 Engine 已经完成所有 scheduler / model runner wiring
```

当前没有做这一层生产生命周期闭环。

---

# 47. P1-V2-R1-D 最终理解总结

这一阶段真正完成的是：

```text
Ragged physical KV layout
        ↓
zero-copy virtual KV view
        ↓
现有 cache kernel WRITE
        ↓
RaggedStepViews 生成 READ metadata
        ↓
按当前 layer 提取 Hkv members
        ↓
把 (request, real_kv_head)
包装成 FA2 pseudo-sequence
        ↓
构造：
packed_query
block_table
seq_lens
cu_seqlens_q
        ↓
调用标准 FA2
        ↓
把 packed_output
恢复成 [Q,Hq,D]
```

WRITE 侧的核心：

```text
(token, kv_head)
→
virtual token / virtual slot
```

READ 侧的核心：

```text
(request, kv_head)
→
FA2 pseudo-sequence
```

所以整个 Ragged 方案最统一的理解是：

```text
真实 head 身份
不再依赖 cache 的显式 head 维，

而是被编码进：

WRITE：
virtual slot identity

READ：
virtual block_table / pseudo-sequence identity
```

---

# 48. 最终一句话

```text
P1-V2-R1-D 的关键不是实现新的 Attention Kernel，

而是通过地址重映射和 sequence 重包装，

让一种原本不是标准 Dense KV layout 的 Ragged KV Cache，
仍然能够复用现有的 cache write kernel 和 FlashAttention2。

WRITE 时：
把 head 编进 slot。

READ 时：
把 head 编进 pseudo-sequence 和 block_table。

最终 Attention 数学保持不变。
```
