# R1 A/B Implemented Foundation and C Entry

**Project:** Physical KV Cache Reclamation / Ragged KV Runtime for vLLM v0.26.0  
**Repository:** `qcrs/vllm`  
**Branch:** `p1/v2-token-compaction-v026`  
**Pinned upstream baseline:** `568afb3a13806beb53bb2e6bd518269357b237c0`  
**Current reviewed branch head:** `018e68f47f3bcdfb0b935f5ffe0e579b033c6268`  
**Reference:** `aiha-lab/tangram@6fa551fc8f6edcc118a2a39b3554ee1520e91edd`

**Recommended repository location**

```text
01-projects/project1-physical-kv-reclamation-vllm/
└── docs/
    └── 10-design/
        └── ragged-kv-runtime-v5/
            └── 43-R1-A-B-Implemented-Foundation-and-C-Entry.md
```

---

# 1. 这份文档解决什么问题

当前 Project 1 已经完成了相当多 Ragged KV 的基础工作，但“设计阶段名称”和“实际代码推进顺序”已经发生了一定偏移。

旧 canonical 路线大体写作：

```text
A1 Spec
→ A2 Global Page Pool
→ A3 Physical Layout
→ B Ragged Block Table
→ C Addressing
→ D1 KV Write
→ D2/D3 Attention Read
```

实际推进过程中，为了先解决 allocator authority、non-uniform physical state、Scheduler/Worker 同步等问题，B/control-plane 和后续 A3 contract alignment 被提前实现。

因此目前不能简单理解成：

```text
A 完成
B 完成
所以 C 之后只剩 GPU kernel
```

更准确的当前状态是：

```text
A:
static geometry / page accounting / placement contract
基本成立

B:
Scheduler canonical ownership
+ Worker discardable mirror
+ transport/reconciliation contract
已经建立

但：

真实 GPU backing layout
virtual block materialization
KV write
FA read
production wiring

仍未闭环
```

本文件的目的就是在进入 C 前，把已经完成的 A/B 基础重新梳理成一条清楚的系统链路。

---

# 2. 总体问题：为什么 Dense vLLM 的 KV 结构不能直接支持我们的目标

Dense vLLM 的基本语义可以简化为：

```text
一个 layer
    ↓
一张 KV cache

一个 block/page
    ↓
同时包含该 layer 的全部 local KV heads
```

若：

```text
Hkv = 8
B   = 16
D   = 128
```

Dense 一个 page 大致承载：

```text
8 heads × 16 tokens × K/V payload
```

所以 allocator 看到的 block 粒度，本质仍然绑定于“整个 layer 的 token block”。

这意味着：

```text
某几个 KV heads 的历史信息已经不重要
```

并不能直接转化成：

```text
只释放这些 heads 对应的一部分显存
```

因为它们和其他 heads 仍在同一 Dense physical page 中。

Project 1 的根本改变是：

```text
Dense page
[layer, all Hkv heads, B tokens]

↓

Ragged physical page
[Hp heads, B tokens]
```

其中：

```text
Hp < Hkv
```

可以成立。

于是一个 layer 被拆成：

```text
G = Hkv / Hp
```

个独立 physical head groups。

再进一步：

```text
不同 group
可以拥有不同的 effective length
可以拥有不同数量的 physical pages
```

这才让 token/head-group 级 physical reclamation 有可能真正释放 BlockPool page。

因此 A/B 两阶段不是“加几个数据结构”，而是在回答两个根本问题：

```text
A:
一个 Ragged physical page 到底是什么？

B:
这些 page 归谁、如何分配、如何在 Scheduler 和 Worker 间保持一致？
```

---

# 3. A 阶段：定义“一个 physical page 到底是什么”

A 阶段主要解决 static geometry、memory accounting 和 topology。

它不负责真正执行 attention。

可以理解成：

```text
A = Representation Contract
```

也就是先把未来 runtime 使用的物理单位定义准确。

---

# 4. A1 — RaggedAttentionSpec：区分 semantic Hkv 和 storage Hp

A1 的核心不是增加一个新的 Spec 名字，而是第一次明确区分：

```text
semantic geometry
vs
storage geometry
```

定义：

```text
Hkv = 当前 rank 上模型语义上的 KV head 数量
Hp  = 一个 physical Ragged page 实际容纳的 KV-head columns
```

例如：

```text
Hkv = 8
Hp  = 2
```

则：

```text
groups_per_layer = Hkv / Hp = 4
```

重要的是：

```text
RaggedAttentionSpec.num_kv_heads
仍然是 Hkv
```

不能为了 storage 简单把它改成 Hp。

原因是 Hkv 仍然描述：

```text
模型 Q/K/V geometry
GQA geometry
semantic head identity
layer/head mapping
```

而 Hp 只是：

```text
physical storage width
```

所以：

```text
semantic Hkv
≠
physical page width Hp
```

这是后面所有设计成立的基础。

## 4.1 page size

Dense page 的真实 memory size 大致是：

```text
2 × B × Hkv × D × dtype_size
```

Ragged page：

```text
2 × B × Hp × D × dtype_size
```

所以：

```text
ragged_page_bytes
=
dense_page_bytes × Hp / Hkv
```

例如：

```text
Hkv=8
Hp=2
```

一个 Ragged page 只有 Dense page 的：

```text
1 / 4
```

但一个 Dense layer-token-block 等价于：

```text
4 Ragged pages
```

所以在 identity/no-compression 条件下：

```text
4 × Ragged page bytes
=
1 × Dense page bytes
```

R1 identity 阶段并不是靠“少存 KV”节约内存，而是先改变 physical allocation granularity。没有 compression 时，总 KV bytes 理论上仍与 Dense 等价；真正收益要等不同 group 深度开始分叉之后出现。

---

# 5. A2 — Global Physical Page Pool

如果只是把每个 layer 自己的 block 切成小块，但 allocator 仍然使用 per-layer block-ID namespace，后面的 Scheduler ownership、Worker block table、compaction、free/reuse 和 virtual addressing 都会变复杂。

因此 A2 冻结：

```text
physical page ID
必须是 global physical namespace
```

例如：

```text
page 41
```

表示整个 Ragged backing 中唯一的 physical page 41，而不是“某 layer 的第 41 个 block”。

Scheduler 因此可以统一表达：

```text
request A
cluster 3
owns pages [40,41,42]
```

而无需额外携带 layer-pool/head-group-pool namespace。

## 5.1 Planner 的变化

Ragged 已经把 `(layer, head-group)` 编入 cluster/member topology，因此 planner 不再建立 per-layer independent physical pool，而是：

```text
one global Ragged page pool
```

page count：

```text
P = available_memory // ragged_page_size
```

raw backing：

```text
KVCacheTensor
size = P × ragged_page_size
shared_by = all supported Ragged layers
```

未来多个 layer 的 kv-cache view 应 alias 同一 global backing，而不是复制多个 physical pool。

---

# 6. A3 实际推进：MemberPlacementMap

设计文档早期把 A3 更多描述为 physical layout / stride，但实际实现中在进入 GPU layout 前先暴露了一个更基础的问题：

```text
semantic member
到底映射到哪个 physical cluster / column？
```

因此当前 branch 已经先引入：

```python
MemberPlacementMap
```

它回答：

```text
(layer_idx, kv_head_idx)
        ↓
member
        ↓
(cluster, column)
```

定义：

```text
M = L * Hkv
C = M / Hp
```

并要求：

```text
M 个 semantic members
↔
C × Hp 个 physical slots
```

形成完整 bijection。

## 6.1 identity placement

R1 默认：

```text
cluster = layer * (Hkv/Hp) + head // Hp
column  = head % Hp
```

例如：

```text
L=2
Hkv=4
Hp=2
```

得到：

```text
layer0/head0 → cluster0,col0
layer0/head1 → cluster0,col1
layer0/head2 → cluster1,col0
layer0/head3 → cluster1,col1
layer1/head0 → cluster2,col0
layer1/head1 → cluster2,col1
layer1/head2 → cluster3,col0
layer1/head3 → cluster3,col1
```

## 6.2 为什么不能让各模块自己算 identity 公式

如果 Scheduler、Worker、attention helper 都自己重算：

```python
cluster = layer * groups_per_layer + head // Hp
column = head % Hp
```

未来研究 AOT clustering、importance-aware grouping 或其他合法 placement 时，每个模块都必须跟着改。

因此冻结：

```text
MemberPlacementMap
= 唯一 placement authority
```

其他模块只消费：

```text
member_to_cluster
member_to_column
```

而不重新推 placement policy。

---

# 7. A 阶段当前还没有真正完成什么

虽然 static geometry、page accounting、placement 已经建立，但真实 GPU physical layout 还没有闭环。

当前 FA2 native Dense cache contract 是：

```text
[P,Hkv,B,2D]
```

而 Ragged physical page 未来需要：

```text
[P,Hp,B,2D]
```

并 zero-copy 解释成：

```text
[P*Hp,1,B,2D]
```

用于：

```text
virtual_block = page * Hp + column
```

这一部分设计已经冻结，但实际 GPU materialization/alias oracle 要由接下来的 C execution foundation 关闭。

因此当前更准确的状态是：

```text
A static representation contract
= 基本完成

A GPU realization
= C 阶段关闭
```

这是当前实际推进顺序相对旧 DAG 的一个重要调整。

---

# 8. B 阶段：定义“这些 physical pages 到底归谁”

A 回答 page 是什么、member 在哪里；B 回答：

```text
page 谁拥有
request 当前拥有多少
什么时候扩容
什么时候有效
什么时候可以 free
Scheduler 和 Worker 如何同步
```

因此：

```text
B = Physical Ownership / State Control Plane
```

---

# 9. 为什么 Worker-only block table 不够

如果只在 Worker 上维护 cluster rows 和 effective lengths，GPU 执行知道怎么读写，但 Scheduler/allocator 不知道：

```text
某个 cluster 现在还有多少 physical capacity
下一 token 是否需要新 page
哪些 page 还能继续用
compaction 后哪些 page 可以 free
```

例如：

```text
B=16

cluster0:
E=31
pages=2

cluster1:
E=55
pages=4

cluster2:
E=19
pages=2
```

下一 token：

```text
target E = [32,56,20]
```

都不用申请 page。

再下一 token：

```text
target E = [33,57,21]
```

只有 cluster0 跨过 32→33，需要新增一个 page。

如果 Scheduler 只知道 `request.num_computed_tokens` 或 `max(E)`，就无法长期保持 non-uniform depth。

因此 B 阶段建立 Scheduler-side canonical physical state。

---

# 10. Scheduler canonical state

当前核心结构：

```python
RaggedRequestPhysicalState
```

语义：

```text
request
├── state_version
├── effective_lens[C]
└── page_rows[C][variable depth]
```

例如：

```text
request A

cluster0:
E=32
pages=[10,11]

cluster1:
E=48
pages=[20,21,22]
```

即：

```text
effective_lens = [32,48]

page_rows =
[
  [10,11],
  [20,21,22],
]
```

page counts 从：

```text
len(page_rows[c])
```

派生，不再维护另一份独立 authority。

---

# 11. reserve 和 commit 被明确拆开

B 阶段一个重要设计是：

```text
拥有 capacity
≠
里面的 KV 已经有效
```

因此生命周期拆成：

```text
plan_capacity()
↓
apply_capacity_plan()
↓
GPU write
↓
commit_effective_lens()
```

## 11.1 plan_capacity

例如：

```text
B=16

current:
E=[32,48]
counts=[2,3]

target:
[48,64]
```

则：

```text
required_counts=[3,4]
append=[1,1]
```

这里只计算，不改变 ownership。

## 11.2 apply_capacity_plan

真正向 BlockPool 申请 page。

例如：

```text
before:
version=4
E=32
pages=[10,11]
```

reserve 一个 page：

```text
after:
version=5
E=32
pages=[10,11,12]
```

E 仍然是 32，因为 page 12 只是 owned capacity，还没有保证写入有效 KV。

## 11.3 commit_effective_lens

GPU write 成功后：

```text
version=5
E=48
pages=[10,11,12]
```

ownership 没变，因此 state_version 不变。

可以概括成：

```text
apply_capacity_plan
= 我拥有这些 pages 了

commit_effective_lens
= 这些 pages 里现在有多少 KV 真正有效
```

---

# 12. B 阶段为 non-uniform continuation 做准备

如果 compaction 只是压完一次少几个 block，但下一 decode allocator 又根据 logical sequence length 把所有 group 补回同样深度，reclamation 很快会被吃掉。

当前 Scheduler state 让后续 allocation 可以直接逐 cluster 使用：

```text
target_E[g] = current_E[g] + q
```

再计算：

```text
required_pages[g]
```

因此未来可以长期保持：

```text
cluster0 depth=4
cluster1 depth=7
cluster2 depth=3
```

并继续 decode。

这也是 B 阶段相对“一次性压缩 demo”最重要的系统价值之一。

---

# 13. Scheduler authority 与 Worker mirror

为了避免出现两个 allocator authority，当前冻结：

```text
Scheduler
= canonical physical ownership authority

Worker
= discardable execution mirror
```

链路：

```text
Scheduler canonical state
        ↓
Snapshot / Delta
        ↓
Worker mirror
        ↓
GPU execution
```

Worker 不自行决定 free 哪个 BlockPool page。

---

# 14. Snapshot 与 Allocation Delta

B 阶段定义两种 Scheduler → Worker state transfer。

## Snapshot

完整 authoritative replacement：

```text
state_version
effective_lens
page_counts
flat_page_ids
```

适用于首次 materialize、resync、resume/re-add、未来 migration。

语义：

```text
不要基于旧状态推
直接变成 Scheduler 给出的完整状态
```

## Allocation Delta

用于已有 Worker mirror 上的 incremental append：

```text
expected_source_state_version
new_state_version
expected_source_E
expected_source_counts
appended_counts
new page IDs
```

语义：

```text
你必须先是 source state
才能应用 delta
```

所以：

```text
Snapshot = replace
Delta    = mutate
```

同一 request 在同一次 update 中不能同时存在 Snapshot + Delta，避免 application ordering ambiguity。

---

# 15. Worker physical mirror

当前：

```python
RaggedWorkerPhysicalState
```

维护：

```text
rows           [R,C,MaxPages]
counts         [R,C]
effective_lens [R,C]
state_versions [R]
```

它不是 allocator，也不是第二份 canonical ownership，只是 Scheduler physical state 在 Worker 上的可丢弃镜像。

未来 C/D 阶段构造 Torch/GPU execution metadata 时，也不应该再建立一套新的 ownership authority，而应该：

```text
Worker mirror
    ↓ derive
step execution metadata
```

---

# 16. Compaction reconciliation

Worker 做完 compaction 后不能直接 free page IDs，只报告：

```text
source state
new E
new page counts
```

Scheduler 根据 canonical row 推导：

```text
retained = old_row[:new_count]
detached = old_row[new_count:]
```

例如：

```text
old=[10,11,12,13]
new_count=2
```

则固定：

```text
retain=[10,11]
detach=[12,13]
```

这就是 prefix-retention invariant。

其本质是：

```text
allocator authority 不能转移给 Worker
```

---

# 17. state_version 的角色

`state_version` 不是 Ragged paging 数学上必需的概念，而是当前架构：

```text
Scheduler canonical state
→ transport
→ Worker mirror
→ result/reconcile
```

产生的 generation fence。

例如：

```text
旧状态:
version=3
E=[32]
count=[2]
pages=[10,11]
```

经过 free/reallocate 后可能再次出现：

```text
新状态:
version=9
E=[32]
count=[2]
pages=[50,51]
```

只看 E/count shape 一样，但 ownership 已经不同。

因此：

```text
version 3 != version 9
```

可以拒绝 stale transition。

当前应保留这一核心 generation fence，但不要继续扩展成大量独立 epoch/version。

---

# 18. 为什么 B 看起来比 Tangram 更复杂

Tangram 的 Ragged runtime 更接近：

```text
Worker / ModelRunner local state
→ RaggedBlockTable
→ CompressionExecutor
→ local compact
```

很多操作位于同一 execution domain 内同步完成，因此不需要完全相同的 Scheduler canonical ownership + Worker discardable mirror + generation transport contract。

Project 1 刻意保留：

```text
allocator authority 在 Scheduler
```

因为项目目标不仅是 KV payload 被压缩，而是要证明：

```text
BlockPool 中真实 physical pages
被正确 reclaim / free / reuse
```

所以：

```text
Tangram
更偏 local execution architecture

P1
更强调 scheduler-owned physical lifecycle
```

---

# 19. A 和 B 合在一起后，系统现在已经知道什么

Static side 已经知道：

```text
Hkv
Hp
B
D
member → cluster,column
```

Scheduler side 已经知道：

```text
request
→ cluster
→ ordered physical page row

request
→ per-cluster E
```

Worker side 可以得到：

```text
active request rows
counts
E
state generation
```

但是还不能完整回答：

```text
layer1/head2 的 physical_position=21
到底落在 GPU backing 的哪个 cell？
```

这正是 C 要关闭的问题。

---

# 20. 为什么下一阶段必须进入 C，而不是继续扩 B

当前再继续添加更多 Scheduler transport、更多 version、更多 validation 或更多 lifecycle bookkeeping，收益已经很低。

真正 blocker 已经转移到 execution plane：

```text
physical page ID
        ↓
actual tensor layout
        ↓
virtual block
        ↓
virtual slot
        ↓
KV write
        ↓
FA read
```

因此现在应停止继续扩大 B control-plane，进入：

```text
R1-C-EXECUTION-FOUNDATION
```

---

# 21. C 要承接 A/B 的什么

C 不重新定义 page ownership、placement policy、state lifecycle。

它只负责把 A/B 已经确定的状态翻译成 execution address：

```text
A:
MemberPlacementMap
(layer,head)
→ cluster,column

B:
request + cluster
→ page row

C:
physical_position
→ page_depth,offset
→ physical_page
→ virtual_block
→ virtual_slot
```

最终得到：

```text
(layer,head,physical_position)
↓
exact GPU KV address
```

---

# 22. C 之后 D1 才真正写 KV

未来 D1：

```text
K/V [Q,Hkv,D]
```

转换为：

```text
[Q*Hkv,1,D]
```

C 提供对应：

```text
member virtual slots [Q*Hkv]
```

cache：

```text
physical [P,Hp,B,2D]
```

zero-copy：

```text
virtual [P*Hp,1,B,2D]
```

再尽量复用已有：

```text
reshape_and_cache_flash
```

所以完整职责可以压缩为：

```text
A
定义 physical unit / topology

B
定义 ownership / lifecycle

C
定义 addressing / storage view

D
真正执行 KV write/read
```

---

# 23. 当前已经证明什么

目前 A/B focused tests 主要证明：

```text
Ragged page byte accounting
global page planning
placement bijection
Scheduler capacity planning
page ownership append
effective frontier commit
Worker snapshot/delta reconstruction
stale transition rejection
compaction prefix retention
free/reuse control-plane semantics
```

这些主要是：

```text
CPU / control-plane contracts
```

---

# 24. 当前还没有证明什么

尚未证明：

```text
actual Ragged GPU physical layout
zero-copy virtual view
actual Ragged KV write
actual FlashAttention Ragged read
prefill/mixed member-major transformation
real engine Dense/Ragged output parity
real physical reclaim end-to-end
performance improvement
```

所以当前不能描述为：

```text
Ragged KV runtime 已经完成
```

更准确的是：

```text
Ragged physical-state / ownership foundation
已经建立

execution path
正在进入 C/D
```

---

# 25. 文档与实现记录建议放在哪里

## 25.1 本文：A/B 基线分析

建议保存：

```text
llm-kv-lab/
01-projects/
project1-physical-kv-reclamation-vllm/
docs/
10-design/
ragged-kv-runtime-v5/
43-R1-A-B-Implemented-Foundation-and-C-Entry.md
```

职责：

```text
解释 A/B 已经完成什么
解释为什么这些基础存在
冻结进入 C 前的系统认知
```

## 25.2 后续 C implementation trace

不要继续把大量 implementation log 写回本文。

建议单独保存：

```text
docs/20-execution/
P1-V2-R1-C-EXECUTION-FOUNDATION-01-Code-Trace.md
```

如果未来建立稳定的 records/evidence 子目录，也可以迁移到那里。

其职责是：

```text
具体改了哪些文件
ADD / MODIFY / REPLACE / DELETE
测试结果
源码证据
remaining blockers
```

这样可以保持：

```text
10-design
= architecture / canonical reasoning

20-execution
= slice implementation / evidence trace
```

不要把两类内容重新混在一起。

---

# 26. 最终认知

进入 C 前，可以用四句话概括当前项目：

```text
A1/A2:
定义了更细粒度的 Ragged physical page，
并建立 global page namespace。

Placement alignment:
定义 semantic (layer,head)
如何对应 physical (cluster,column)。

B:
Scheduler 成为 page ownership authority，
Worker 持有 discardable execution mirror，
并建立 reserve / commit / reconcile 生命周期。

C:
下一步不再研究 ownership，
而是把已经确定的 physical state
真正映射成 v0.26 FA2 可消费的 GPU address。
```

系统图：

```text
Model semantic KV
(layer,head)
        │
        ▼
MemberPlacementMap
        │
        │ A
        ▼
(cluster,column)
        │
        ├──── Scheduler canonical state ────┐
        │                                  │
        │              B                   ▼
        │                         ordered physical pages
        │                                  │
        └──────────────────────── physical_position
                                           │
                                           │ C
                                           ▼
                              page / column / offset
                                           │
                                           ▼
                                virtual block / slot
                                           │
                                           │ D1/D2
                                           ▼
                                  KV Write / FA Attention
```

从现在开始，Project 1 的主要风险已经从：

```text
physical ownership 语义是否成立
```

转向：

```text
这些 ownership 是否能够
无 hidden copy、无地址错误地
落到真实 GPU execution path
```

这就是下一阶段 C/D 应集中解决的问题。
