# P1-V2 Ragged KV Physical / Virtual Layout 与 Addressing 深度笔记

> **Project:** Physical KV Cache Reclamation for vLLM — V2 Token-Level Compaction  
> **Branch:** `qcrs/vllm:p1/v2-token-compaction-v026`  
> **Scope:** C1–C4 execution-addressing foundation  
> **核心源码：**
>
> - `vllm/v1/ragged_kv_layout.py`
> - `vllm/v1/attention/backends/ragged_layout.py`
> - `vllm/v1/core/ragged_kv_cache_manager.py`
> - `vllm/v1/worker/gpu/attn_utils.py`
>
> 本文只固定当前 C 阶段已经建立的 **Ragged physical layout、ownership、physical address、virtual address 和 metadata transform**。  
> 当前 C Slice **尚未完成 production Ragged KV write / FlashAttention read wiring**；这些属于后续 D1 / D2。

---

# 1. 先给出最终 mental model

整个 C1–C4 可以压缩成一句话：

> **Ragged KV 的真实数据存在 `[P, Hp, B, 2D]` 中；Scheduler 以 cluster 为单位管理 physical page chain；一个 semantic member 通过 placement 找到 `(cluster,column)`；token position 通过 page depth 找到 physical page；最终真实地址是 `(page,column,offset)`。为了复用 vLLM 原有 block/slot 风格接口，再把 `(page,column)` flatten 为 virtual block，把 `(page,column,offset)` flatten 为 virtual slot。**

最核心关系：

```text
semantic identity
(layer, kv_head)
        │
        ▼
      member
        │ placement
        ▼
(cluster, column)

physical_position
        │ divmod(B)
        ▼
(page_depth, offset)

cluster + page_depth
        │ page_rows / active_row
        ▼
physical_page_id

最终真实地址：
(page, column, offset)
```

然后：

```text
virtual_block
= page * Hp + column

virtual_slot
= virtual_block * B + offset

因此：

virtual_slot
= (page * Hp + column) * B + offset
```

---

# 2. 关键符号

| 符号 | 含义 |
|---|---|
| `L` | 当前 Ragged layout 覆盖的 layer 数 |
| `Hkv` | 每层 local KV head 数 |
| `M` | semantic member 数，`M = L * Hkv` |
| `Hp` | `page_group_size`，一个 physical page 横向容纳的 member 数 |
| `C` | cluster 数，`C = M / Hp` |
| `P` | global physical page pool 中的 page 数 |
| `B` | `block_size`，每个 member 在一个 page 内最多保存的 token 数 |
| `D` | head size |
| `R` | request 数 |
| `Q` | 某个 metadata helper 中待处理的 query/write positions 数；不是 page 容量 |
| `MaxPages` | batched page row padding 后的最大 page depth |

---

# 3. Dense 和 Ragged 的根本差别

## 3.1 Dense mental model

FlashAttention Dense KV 的逻辑 shape 通常可理解为：

```text
[P, Hkv, B, 2D]
```

一个 Dense block 可以理解成：

```text
一个 block
=
B 个 token
×
这一层全部 Hkv heads 的 KV
```

同一个 KV cache group 中，不同 layer 可以共享同一套 block-table 语义：

```text
block_table = [7, 11, 20]

layer0_cache[7]
layer1_cache[7]
layer2_cache[7]
```

这里 block id 相同，但不同 layer 通常通过不同 layer cache view/storage 体现 layer identity。

---

## 3.2 Ragged physical layout

当前 C 实现冻结：

```text
physical_cache.shape
=
[P, Hp, B, 2D]
```

含义：

```text
P   = physical page
Hp  = 一个 page 中的 member columns
B   = 每个 member 在这个 page 中的 token offsets
2D  = K + V
```

例如：

```text
Hp = 2
B  = 16
```

一个 physical page：

```text
physical page 7

column0:
    token offset 0
    token offset 1
    ...
    token offset 15

column1:
    token offset 0
    token offset 1
    ...
    token offset 15
```

真实 Tensor 地址：

```text
physical_cache[7, 0, 5, :]
physical_cache[7, 1, 5, :]
```

分别表示 page7 中两个不同 member 的 token offset5。

---

# 4. member、cluster、physical page 到底是什么关系

这是整个设计最重要的概念之一。

## 4.1 member

一个 member 是一个 semantic KV sequence：

```text
member
≈
(layer, kv_head)
```

例如：

```text
L = 1
Hkv = 4
Hp = 2
```

则：

```text
member0 = layer0/head0
member1 = layer0/head1
member2 = layer0/head2
member3 = layer0/head3
```

---

## 4.2 cluster

cluster 不是 physical Tensor 的一维。

它是：

> **Hp 个 members 的 allocation / ownership 分组。**

例如 identity placement：

```text
cluster0
├── member0 → column0
└── member1 → column1

cluster1
├── member2 → column0
└── member3 → column1
```

因此：

```text
cluster
= 一组共享同一条 physical page chain 的 members
```

---

## 4.3 physical page

一个 physical page 是：

> **某个 cluster 在某一个 page depth 上实际占用的一块 GPU KV storage。**

假设：

```text
B = 16
cluster0 effective_len = 27
```

cluster0 需要两页：

```text
cluster0
depth0 → page10
depth1 → page12
```

那么：

```text
page10:
    column0 → member0 token 0~15
    column1 → member1 token 0~15

page12:
    column0 → member0 token 16~26
    column1 → member1 token 16~26
```

所以：

> **一个 page 不是整个 cluster；它是 cluster 在一个 B-token depth 上的一段物理存储。**

---

# 5. 为什么说 cluster 是 ownership / metadata 概念

真实 Tensor：

```text
physical_cache[P, Hp, B, 2D]
```

Tensor 本身只有：

```text
page
column
token offset
KV vector
```

它没有：

```text
request
cluster
layer
head
```

这些语义由外部 metadata 决定。

例如 Scheduler authoritative state：

```text
request A

cluster0 → [page10, page12]
cluster1 → [page20, page25]
```

这里 `page_rows` / `active_row` 才告诉我们：

```text
page10 / page12 当前属于 request A 的 cluster0
page20 / page25 当前属于 request A 的 cluster1
```

因此：

```text
cluster
不是一块 Tensor storage

cluster
    ↓ owns / references
一条 physical page chain
```

也就是：

```text
request
  │
  ├─ cluster0
  │    ├─ member0 col0
  │    ├─ member1 col1
  │    └─ pages [10,12,...]
  │
  └─ cluster1
       ├─ member2 col0
       ├─ member3 col1
       └─ pages [20,25,...]
```

---

# 6. `MemberPlacementMap`：semantic member 到 physical slot 的静态映射

当前 placement authority：

```python
MemberPlacementMap
```

核心：

```text
member_to_cluster
member_to_column
```

例如：

```text
member_to_cluster = [0,0,1,1]
member_to_column  = [0,1,0,1]
```

意味着：

```text
member0 → cluster0,col0
member1 → cluster0,col1
member2 → cluster1,col0
member3 → cluster1,col1
```

因此 layer/head identity 不再依赖：

```text
哪个 layer Tensor
+
Hkv dimension
```

而是显式变成：

```text
(layer, head)
    ↓
member
    ↓
(cluster,column)
```

---

# 7. C1：从 semantic token position 得到真实 physical address

`resolve_kv_address()` 的逻辑：

```text
(layer,head)
    ↓ placement
member
    ↓
(cluster,column)

physical_position
    ↓ divmod(B)
(page_depth,offset)

active_row[cluster][page_depth]
    ↓
physical_page_id
```

最终得到：

```text
(page,column,offset)
```

例如：

```text
B = 16
physical_position = 21
```

则：

```text
page_depth = 21 // 16 = 1
offset     = 21 % 16  = 5
```

若：

```text
cluster3 page chain = [40,41]
```

则：

```text
page_depth1 → page41
```

若 member 在：

```text
cluster3,col0
```

最终真实地址：

```text
physical_cache[41,0,5,:]
```

---

# 8. C2：为什么需要 virtual view

当前真实 Ragged storage：

```text
[P, Hp, B, 2D]
```

但 vLLM 原有 Dense-style kernel / metadata 接口更自然地理解：

```text
[block, head, token, ...]
```

所以 C2 采取：

```text
physical:
[P, Hp, B, 2D]

        ↓ zero-copy .view()

virtual:
[P*Hp, 1, B, 2D]
```

核心思想：

> **把每一个 `(physical page, column)` 伪装成一个 single-member virtual block。**

---

# 9. physical page + column 如何变成 virtual block

公式：

```text
virtual_block
=
physical_page * Hp + column
```

例如：

```text
Hp = 2
```

映射：

```text
(page0,col0) → block0
(page0,col1) → block1

(page1,col0) → block2
(page1,col1) → block3

...

(page7,col0) → block14
(page7,col1) → block15
```

所以：

```text
physical_cache[7,0,:,:]
```

和：

```text
virtual_cache[14,0,:,:]
```

指向相同 bytes。

同理：

```text
physical_cache[7,1,:,:]
=
virtual_cache[15,0,:,:]
```

---

# 10. 为什么这是 zero-copy

C2 代码：

```python
physical_cache.view(
    P * Hp,
    1,
    B,
    2D,
)
```

没有：

```text
clone
copy
contiguous fallback
```

前提是 physical cache contiguous。

对于 contiguous：

```text
[P, Hp, B, 2D]
```

前两维可以直接 flatten：

```text
(P,Hp)
→
P*Hp
```

因此 virtual view 只是新的 shape / stride interpretation。

数据没有搬。

---

# 11. 最核心：virtual slot 到底是什么

physical world 中，一个完整 KV cell 的坐标是：

```text
(page, column, offset)
```

但 Dense-style write 接口习惯使用：

```text
slot = block * B + offset
```

所以先：

```text
virtual_block
= page * Hp + column
```

再：

```text
virtual_slot
= virtual_block * B + offset
```

得到：

```text
virtual_slot
=
(page * Hp + column) * B + offset
```

展开：

```text
=
page * Hp * B
+ column * B
+ offset
```

这实际上就是 contiguous `[P,Hp,B]` 前三维的 linear cell index。

因此：

> **virtual slot 不是凭空制造的新地址；它就是 `(page,column,offset)` 在 virtual view 下的线性编码。**

---

# 12. physical slot 和 virtual slot 必须严格区分

## physical slot

定义：

```text
physical_slot
=
page * B + offset
```

它只编码：

```text
(page,offset)
```

没有 column。

因此它是：

> **cluster-level 的不完整地址。**

例如：

```text
B = 16
page7
offset5
```

得到：

```text
physical_slot = 7*16+5 = 117
```

但 `117` 并不能告诉你：

```text
page7,col0
还是
page7,col1
```

---

## virtual slot

placement 给 member 补上 column：

```text
member0 → col0
member1 → col1
```

于是：

```text
physical slot117
        │
        ▼
page7,offset5
        │
        ├── member0 col0
        │       ↓
        │    (7,0,5)
        │       ↓
        │    virtual block14
        │       ↓
        │    virtual slot229
        │
        └── member1 col1
                ↓
             (7,1,5)
                ↓
             virtual block15
                ↓
             virtual slot245
```

计算：

```text
member0:
virtual_block = 7*2+0 = 14
virtual_slot  = 14*16+5 = 229

member1:
virtual_block = 7*2+1 = 15
virtual_slot  = 15*16+5 = 245
```

对应真实地址：

```text
physical_cache[7,0,5,:]
↔
virtual_cache[14,0,5,:]
↔
virtual slot229

physical_cache[7,1,5,:]
↔
virtual_cache[15,0,5,:]
↔
virtual slot245
```

---

# 13. 为什么一个 physical slot 可以展开成多个 virtual slots

因为一个 cluster 内：

```text
cluster0
├─ member0 col0
└─ member1 col1
```

同一个 token 对这些 members：

```text
page 相同
offset 相同
column 不同
```

所以 cluster-level physical slot：

```text
page7 offset5
→ physical slot117
```

可以同时对应：

```text
member0 → (7,0,5)
member1 → (7,1,5)
```

因此一个 cluster physical slot 最终展开成 `Hp` 个 member virtual slots。

---

# 14. C4：为什么不能只 virtualize KV Tensor

C2 已经把 KV cache：

```text
[P,Hp,B,2D]
```

变成：

```text
[P*Hp,1,B,2D]
```

但如果 metadata 仍然使用 physical cluster semantics：

```text
cluster page table
cluster slots
cluster seq lens
```

kernel 仍然不知道如何正确索引 virtual cache。

因此 metadata 也必须同步切换到：

```text
member virtual block table
member virtual slots
member seq lens
```

这就是 C4 的目的。

---

# 15. `placement_to_tensors()`

原始 placement 是 Python tuples：

```text
member_to_cluster
member_to_column
```

C4 需要执行 Tensor operation：

```text
index_select
torch.where
torch.div
torch.remainder
```

所以：

```python
placement_to_tensors(...)
```

把 tuple materialize 成 Torch tensor。

注意：

```text
.to(...)
```

只改变 dtype / device；

```text
.view(...)
```

才改变 shape。

而 production 设计要求后续 D1/D2 应：

```text
初始化一次
→ cache placement tensors
→ hot path reuse
```

不应每 step 反复构造。

---

# 16. `member_virtual_block_table()`

输入：

```text
physical_cluster_table
shape = [R,C,MaxPages]
```

例如：

```text
request0:

cluster0 → [10,12,0]
cluster1 → [20,25,0]
```

其中 `0` 是 NULL/padding page sentinel。

placement：

```text
member0 → cluster0,col0
member1 → cluster0,col1
member2 → cluster1,col0
member3 → cluster1,col1
```

---

## Step 1：cluster → member expansion

```python
cluster_table =
    physical_cluster_table.index_select(
        1,
        member_to_cluster
    )
```

若：

```text
member_to_cluster = [0,0,1,1]
```

原：

```text
cluster0 → [10,12,0]
cluster1 → [20,25,0]
```

展开：

```text
member0 → [10,12,0]
member1 → [10,12,0]
member2 → [20,25,0]
member3 → [20,25,0]
```

shape：

```text
[R,C,MaxPages]
↓
[R,M,MaxPages]
```

---

## Step 2：补 column

```python
column =
    member_to_column
    .to(dtype=...)
    .view(1,-1,1)
```

若：

```text
member_to_column = [0,1,0,1]
```

变成：

```text
[1,M,1]
```

这样可以和：

```text
[R,M,MaxPages]
```

broadcast。

逻辑上相当于：

```text
member0 → [0,0,0]
member1 → [1,1,1]
member2 → [0,0,0]
member3 → [1,1,1]
```

---

## Step 3：physical page → virtual block

```python
virtual_table
=
cluster_table * Hp + column
```

例如：

```text
member0:
[10,12] col0
→ [20,24]

member1:
[10,12] col1
→ [21,25]

member2:
[20,25] col0
→ [40,50]

member3:
[20,25] col1
→ [41,51]
```

---

## Step 4：保护 NULL page

若 physical page padding 是：

```text
0
```

不能直接算：

```text
0 * Hp + col1 = 1
```

否则 NULL page 会误变成有效 virtual block。

所以：

```python
torch.where(
    cluster_table == 0,
    0,
    virtual_table,
)
```

保证：

```text
physical NULL 0
→
virtual NULL 0
```

---

# 17. `member_virtual_slots()`

输入：

```text
physical_slots
shape = [Q,C]
```

这里：

> `Q` 是该 helper 中待处理的 query/write positions 数，不是 page 中 token 数。

例如测试：

```text
Q = 2
C = 2
```

可能：

```text
q0:
cluster0 → page7 offset5
cluster1 → page11 offset15

q1:
cluster0 → PAD
cluster1 → page19 offset0
```

physical slot 编码：

```text
slot = page * B + offset
```

因此：

```text
q0:
cluster0 → 7*16+5
cluster1 → 11*16+15
```

---

## Step 1：cluster slot → member slot

```python
member_slots =
    physical_slots.index_select(
        1,
        member_to_cluster
    )
```

例如：

```text
cluster0 slot117
cluster1 slot325
```

展开：

```text
member0 → 117
member1 → 117
member2 → 325
member3 → 325
```

---

## Step 2：反解 physical slot

```python
page   = floor(member_slots / B)
offset = member_slots % B
```

例如：

```text
117
→ page7 offset5
```

---

## Step 3：placement 补 column

```text
member0 → col0
member1 → col1
```

得到完整 physical coordinates：

```text
member0 → (7,0,5)
member1 → (7,1,5)
```

---

## Step 4：变成 virtual slot

```python
virtual_slots
=
(page * Hp + column) * B + offset
```

即：

```text
(page,column,offset)
↓
virtual block + offset
↓
virtual slot
```

---

## Step 5：保护 PAD slot

write-side sentinel：

```text
-1 = PAD_SLOT_ID
```

因此：

```python
torch.where(
    member_slots == -1,
    -1,
    virtual_slots
)
```

保证：

```text
physical PAD -1
→
virtual PAD -1
```

---

# 18. `member_seq_lens()`

输入：

```text
physical_seq_lens
shape = [R,C]
```

例如：

```text
request0:

cluster0 effective_len = 32
cluster1 effective_len = 20
```

由于同一 cluster 内的 members 当前共享同一 physical lifecycle/frontier：

```text
cluster0
├─ member0 len32
└─ member1 len32

cluster1
├─ member2 len20
└─ member3 len20
```

所以只需：

```python
physical_seq_lens.index_select(
    1,
    member_to_cluster
)
```

得到：

```text
[R,M]
```

例如：

```text
[32,32,20,20]
```

这里不需要 column，因为 sequence length 与 column 无关。

---

# 19. C1–C4 的完整转换图

```text
                         SEMANTIC WORLD

                      (layer, kv_head)
                              │
                              ▼
                           member
                              │
                       MemberPlacementMap
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                 cluster              column
                    │                   │
                    │                   │
physical_position   │                   │
       │             │                   │
       ▼             │                   │
(depth, offset)      │                   │
       │             │                   │
       └──────► page_rows[cluster][depth]
                              │
                              ▼
                         physical page
                              │
                              ▼

                    COMPLETE PHYSICAL ADDRESS

                     (page,column,offset)
                              │
                              │
                  virtual_block = page*Hp+column
                              │
                              ▼
                         virtual block
                              │
                  virtual_slot = block*B+offset
                              │
                              ▼
                         virtual slot
```

---

# 20. Data 和 Metadata 必须一起转换

```text
DATA

physical cache
[P,Hp,B,2D]
      │
      ▼ zero-copy
virtual cache
[P*Hp,1,B,2D]
```

同时：

```text
METADATA

physical block table
[R,C,MaxPages]
      │
      ▼
member virtual block table
[R,M,MaxPages]


physical slots
[Q,C]
      │
      ▼
member virtual slots
[Q,M]


physical seq lens
[R,C]
      │
      ▼
member seq lens
[R,M]
```

核心原则：

> **不能只改 cache view 而不改 metadata；Data 和 Metadata 必须处于同一个 address space。**

---

# 21. 为什么 virtual address 不是“假的”

非常重要。

physical contiguous layout：

```text
[P,Hp,B,2D]
```

忽略最后 `2D`，一个 KV cell：

```text
(page,column,offset)
```

对应的 linear cell index：

```text
page * Hp * B
+ column * B
+ offset
```

整理：

```text
(page * Hp + column) * B + offset
```

这正好就是：

```text
virtual_slot
```

所以：

> **virtual address 不是另外维护一份 KV，也不是重新分配地址；它是 physical contiguous layout 的另一种坐标编码。**

---

# 22. 一个完整 concrete example

参数：

```text
Hp = 2
B  = 16

cluster0:
    member0 → col0
    member1 → col1

cluster0 pages:
    depth0 → page7
    depth1 → page9
```

处理 member1 的：

```text
physical_position = 5
```

则：

```text
depth  = 5 // 16 = 0
offset = 5 % 16  = 5

page = page_rows[cluster0][0]
     = 7

column = 1
```

真实 physical address：

```text
physical_cache[7,1,5,:]
```

virtual：

```text
virtual_block
= 7*2+1
= 15

virtual_slot
= 15*16+5
= 245
```

对应：

```text
physical_cache[7,1,5,:]
=
virtual_cache[15,0,5,:]
=
virtual slot245
```

若只看 cluster-level physical slot：

```text
physical_slot
= 7*16+5
= 117
```

则：

```text
physical slot117
        │
        ▼
page7,offset5
        │
        ├── member0 col0
        │      ↓
        │   virtual slot229
        │
        └── member1 col1
               ↓
            virtual slot245
```

这说明：

```text
physical_slot
= cluster-level incomplete address

virtual_slot
= member-level complete flattened address
```

---

# 23. NULL / PAD sentinel 必须区分

当前两个不同 sentinel：

## block table / page row

```text
0
```

表示 NULL / padding page。

所以：

```text
physical page0 padding
不能做
0*Hp+column
```

必须保持：

```text
0 → 0
```

---

## slot mapping

```text
-1
```

表示 PAD_SLOT_ID / no-write。

所以：

```text
-1
```

不能最终变成某个有效 virtual slot。

必须：

```text
-1 → -1
```

---

# 24. 当前 C 阶段已经证明什么

当前源码和 C Slice evidence 已经建立：

1. `MemberPlacementMap` 是唯一 static placement authority。
2. scalar `resolve_kv_address()` 能从：
   ```text
   (layer,head,physical_position)
   ```
   得到：
   ```text
   physical page / column / offset
   virtual block / virtual slot
   ```
3. Ragged physical cache shape 是：
   ```text
   [P,Hp,B,2D]
   ```
4. physical cache 可以 zero-copy view 为：
   ```text
   [P*Hp,1,B,2D]
   ```
5. `member_virtual_block_table()` 可以把：
   ```text
   [R,C,MaxPages]
   →
   [R,M,MaxPages]
   ```
6. `member_virtual_slots()` 可以把：
   ```text
   [Q,C]
   →
   [Q,M]
   ```
7. `member_seq_lens()` 可以把：
   ```text
   [R,C]
   →
   [R,M]
   ```
8. vector transform 已通过 scalar oracle 对照测试。
9. CUDA alias smoke 已证明 physical / virtual view 指向同一 storage。

---

# 25. 当前还没有证明什么

必须避免把 C foundation 当成 production integration。

当前尚未完成：

```text
actual Ragged KV write
→ D1

actual FlashAttention read
→ D2 / D3

Scheduler / ModelRunner production wiring

real Engine Ragged dispatch

prefill / mixed workload

prefix cache

quantized Ragged

unequal K/V head size

TP2

CUDA Graph hot-path integration
```

所以当前最准确的状态是：

> **地址体系、layout、zero-copy view 和 metadata transform 已经建立；真正 kernel integration 尚未接通。**

---

# 26. 后续审 D1 / D2 时应该检查什么

以后看实现，不需要重新从头理解，只需要检查这些 invariant。

## D1 — KV Write

必须确认：

```text
cluster physical slot
    ↓
member_virtual_slots()
    ↓
member virtual slot
    ↓
write kernel
    ↓
virtual cache [P*Hp,1,B,2D]
    ↓ alias
physical cache [P,Hp,B,2D]
```

重点检查：

- slot 的 `[Q,C] → [Q,M]` shape 是否与真正 write input 对齐；
- `(layer,head)` 到 member 的 ordering 是否一致；
- `-1` sentinel 是否 preserved；
- placement tensor 是否 hot-path reuse；
- 是否发生 hidden copy；
- write kernel 是否真的把 virtual block 当 single-head block 使用。

---

## D2 — Attention Read

必须确认：

```text
physical cluster page table
    ↓
member_virtual_block_table()
    ↓
[R,M,MaxPages]

physical seq lens
    ↓
member_seq_lens()
    ↓
[R,M]

virtual cache
[P*Hp,1,B,2D]
```

然后 read-side adapter 必须保证：

```text
member-major query
block table
seq lens
virtual cache
```

四者 ordering 完全一致。

重点检查：

- FlashAttention 实际需要的 batch/query organization 如何从 request/member 展开；
- `M` 是否作为 execution sequence 维处理；
- block table shape 是否与 FA varlen API 一致；
- causal / query_start_loc / seq_lens 是否保持语义；
- member flatten 后的 output 如何还原到原 `(layer,head)` organization。

---

# 27. 最后记住这 6 句话

1. **member 是 `(layer,kv_head)` 的 semantic KV sequence。**
2. **cluster 是 Hp 个 members 的 allocation / ownership group，不是 Tensor dimension。**
3. **一个 physical page 是 cluster 在一个 page depth 上的 B-token storage slice。**
4. **真实 Ragged 地址是 `(page,column,offset)`。**
5. **virtual block = `(page,column)` flatten；virtual slot = `(page,column,offset)` flatten。**
6. **virtual 不是复制数据，而是让旧 Dense-style block/slot 接口理解同一块 Ragged physical storage。**

---

# 28. 一张最终总图

```text
                    semantic member
                   (layer, kv_head)
                           │
                           ▼
                    MemberPlacementMap
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
               cluster            column
                  │                 │
                  │                 │
        physical_position           │
                  │                 │
                  ▼                 │
          depth = pos // B          │
          offset = pos % B          │
                  │                 │
                  ▼                 │
        page_rows[cluster][depth]    │
                  │                 │
                  ▼                 │
                page                │
                  └────────┬────────┘
                           ▼
                 (page,column,offset)
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
     REAL PHYSICAL                 VIRTUAL ENCODING

physical_cache                    virtual_block
[P,Hp,B,2D]                       = page*Hp+column
                                     │
physical[page,                    virtual_slot
         column,                   = block*B+offset
         offset,:]                    │
            │                         ▼
            └──────── same bytes ─ virtual_cache
                                  [P*Hp,1,B,2D]
```

---

## 当前阶段的判断基线

以后如果某段新代码声称支持 Ragged KV execution，可以直接问它四个问题：

```text
1. 它的 semantic member 最终映射到哪个 cluster / column？
2. cluster 的 token position 最终通过哪条 page row 找到 physical page？
3. kernel 看到的是 physical address 还是 virtual address？
4. cache view、block table、slot mapping、seq lens 是否使用同一套 address semantics？
```

只要其中任何一个答案模糊，Ragged execution 就可能存在 semantic mismatch。

