# P1-M1-T2：从 vLLM 原生 KV 链路到 Whole-Block Reclaim  
## ——源码分析、问题定位、关键 Contract、DESIGN_CONFLICT 与 R1 实现详解

> 项目：**Physical KV Cache Reclamation for vLLM**  
> 副标题：**Decoupled Logical/Physical KV Positions with Real Block Reuse**  
> 固定源码：vLLM v0.26.0  
> Pinned commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> 当前工作分支：`p1/physical-kv-reclaim-v026`
>
> 本文不是简单记录“改了哪些代码”，而是从一个问题出发：

> **v2 补充说明**
>
> 本版重点重写“代码修改与实验”部分，明确区分：
>
> - 为什么原生 `update_requests()` 缺少 reclaim commit 能力；
> - `_commit_reclaim_transition()` 为什么新增、为什么位于 `GPUModelRunner`；
> - 它目前**尚未接入 `execute_model()` runtime**，R1 只验证 Worker primitive；
> - 01A helper 与 upstream normal append 如何组合后触发 two-descriptor conflict；
> - R1 如何重构已有 helper，而不是再新增第二套函数；
> - focused CUDA oracle 如何构造真实 `RequestState + BlockTables`、如何复现 conflict、如何验证 descriptor=1、slot mapping 与 fail-before-mutation。
>
>
> **如果一个 RUNNING request 的中间 KV blocks 被真实回收，vLLM 原来的执行链到底在哪里失效？我们应该在哪一层拆 logical / physical 语义，又应该在哪一层提交新的 physical block view？**

---

# 0. 这份笔记解决什么问题

整个 P1 到目前为止，最容易被误解成：

```text
“从 block table 中删掉 B2、B3，
然后把这两个 block free 掉。”
```

实际上，这只是结果的一小部分。

真正困难的是：

```text
Request 的 logical token progress 不能回退；
但它在 GPU 中实际保留的 KV token 数减少了；
未来 token 的 RoPE position 仍然必须使用 logical position；
未来 KV 写入位置却必须从压缩后的 physical frontier 继续；
Attention 读取长度也必须改成 physical valid length；
Scheduler 的 block ownership / allocator 又暂时还是另一套 truth。
```

所以本项目真正面对的不是单纯的“删除 blocks”，而是一个 **execution semantics + metadata ownership + allocator ownership** 的解耦问题。

本文重点回答：

1. 为什么原生 vLLM 的 `num_computed_tokens` 无法直接支持 reclaim；
2. 为什么必须引入 `effective_kv_len`；
3. 为什么 `positions` 不能直接改成 physical；
4. 为什么 KV WRITE 和 KV READ 必须分别修；
5. 为什么 T2 可以只改 Worker physical view，而暂时不 free；
6. 为什么 whole-block reclaim 是 V1 的必要限制；
7. 为什么第一次 01A 实现虽然逻辑正确，却在 `StagedWriteTensor` publication 层失败；
8. R1 为什么改成 `retained + same-step new → final row → ONE overwrite`；
9. 目前 production code 到底新增、修改、替换、删除了什么；
10. 通过这个问题，我们实际看清了 vLLM 哪些核心运行链路。

---

# 1. 先定义真正的问题：reclaim 不是“少几个 block”这么简单

考虑一个 request：

```text
logical tokens computed = 94
block_size = 16

Worker block row:
[B0 B1 B2 B3 B4 B5]
```

因为：

```text
ceil(94 / 16) = 6
```

最后一个 block `B5` 只使用了 14 个 token slot。

现在我们希望 reclaim 两个完整中间 blocks：

```text
remove:
[B2 B3]

retain:
[B0 B1 B4 B5]
```

此时不能说 request 只计算了 62 个 logical tokens。

模型语义仍然是：

```text
logical_num_computed = 94
```

但是 Worker 真正保留的 physical KV token 数变成：

```text
effective_kv_len
= 94 - 2 * 16
= 62
```

所以 reclaim 后的正确状态是：

```text
Logical/model truth:
L = 94

Physical KV truth:
E = 62

Worker execution block view:
[B0 B1 B4 B5]
```

这里已经出现了第一个关键结论：

> **logical progress 和 physical KV occupancy 不再是同一个值。**

这也是 P1 的根问题。

---

# 2. 从这个问题看，vLLM 至少存在三种不同的 truth

经过 S1/T2 的源码梳理，可以把整个系统分成三层：

```text
┌────────────────────────────────────────────┐
│ 1. Logical / model truth                   │
│    num_computed_tokens                     │
│                                            │
│    回答：模型已经走到哪个 token position？ │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ 2. Worker physical execution truth         │
│    effective_kv_len                        │
│    + Worker active BlockTable view         │
│                                            │
│    回答：GPU 当前真正保留/访问哪些 KV？    │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ 3. Scheduler allocator / ownership truth   │
│    req_to_blocks / BlockPool ownership     │
│                                            │
│    回答：哪些 physical blocks 仍归 request │
│          所有，哪些可以真正 free/reuse？   │
└────────────────────────────────────────────┘
```

P1 的复杂性就在于：

```text
这三个 truth 最开始完全重合，
reclaim 以后必须允许它们阶段性分离。
```

T2 当前只做到：

```text
Worker physical execution truth
发生变化
```

而没有进入：

```text
Scheduler canonical ownership shrink
BlockPool.free
reuse
```

后者是 T3。

---

# 3. 从 Scheduler 开始看：`num_computed_tokens` 到底是什么

## 3.1 Scheduler 先构造当前 step 的 start-state

在 `Scheduler.schedule()` 中，当前 request 的：

```text
request.num_computed_tokens
```

用于决定本轮从哪里继续计算、需要多少 slot、要调度多少 token。

SchedulerOutput 构造时，对于 continuing cached request，会通过：

```text
CachedRequestData.num_computed_tokens
```

把当前 step 的 logical start-state 送给 Worker。

核心位置：

```text
vllm/v1/core/sched/scheduler.py
    _make_cached_request_data()
```

概念上：

```text
Scheduler CPU:
L = 94

        ↓ SchedulerOutput

Worker:
current step logical base = 94
```

## 3.2 为什么 Scheduler 自己随后会先变成 `L + q`

Scheduler 构造完当前 step 的 output 后，会执行：

```text
_update_after_schedule()
```

做 optimistic advancement：

```text
request.num_computed_tokens += num_scheduled_tokens
```

这是 Scheduler 为“下一次 schedule”准备的状态。

所以同一时间可能存在：

```text
当前 SchedulerOutput:
L = 94

Scheduler 内部下一轮 optimistic state:
L = 95
```

这不是 bug。

它说明：

> Scheduler 的 CPU state 是调度控制状态；Worker GPU state 是执行进度状态。二者不是简单 lock-step scalar。

这也是为什么后面不能随意把 Scheduler 的 absolute logical value拿来覆盖 Worker physical state。

---

# 4. Worker 的 persistent state：为什么 S1 必须加 `effective_kv_len`

核心文件：

```text
vllm/v1/worker/gpu/states.py
```

原始 Worker 已经有：

```text
num_computed_tokens
```

它用于 GPU 侧 execution progress。

S1 新增：

```text
RequestState.effective_kv_len
```

类型仍是 GPU persistent staged state：

```text
StagedWriteTensor[int32]
```

它的语义是：

```text
当前已经真正存在的有效 physical KV token 数
```

新 request 初始化：

```text
effective_kv_len = initial num_computed_tokens
```

正常情况下二者相等：

```text
L = E
```

future reclaim 之后才允许：

```text
L != E
```

---

# 5. 为什么 S1 只增加 state，不立刻修改 Attention

这是一个非常重要的工程拆分。

S1 只解决：

```text
physical progress 能否独立存在并正确推进？
```

不解决：

```text
physical progress 是否已经被 slot mapping / Attention 使用？
```

所以 S1 后：

```text
RequestState.effective_kv_len
    ├─ add_request 初始化
    ├─ normal post_update + actual delta
    └─ future reclaim 可独立 shrink
```

但是暂时没有 execution consumer。

这是一种很好的最小 Slice 方法：

```text
先建立 state lifecycle
再接 producer/consumer
```

否则如果同时改：

```text
state + position + slot mapping + attention
```

一旦结果错误，很难知道是哪一层出了问题。

---

# 6. post-update：logical 和 physical 为什么正常时一起前进

核心位置：

```text
vllm/v1/worker/gpu/input_batch.py
    post_update()
    _post_update_kernel()
```

原始逻辑只更新：

```text
num_computed_tokens += computed_delta
```

S1 后：

```text
computed_delta
    ├─ num_computed_tokens += delta
    └─ effective_kv_len   += delta
```

这里必须使用：

```text
computed_delta
```

而不能简单使用：

```text
query_len
```

因为真正完成的 token 数可能和计划调度数不同，例如 future spec decode rejection 等机制。

在当前 V1 scope 中 spec decode 关闭，但这里仍沿用原生正确的 commit semantics：

> **只有 forward / sampling 后真正被接受的 token，才进入 persistent execution progress。**

因此：

```text
reclaim 后：
L=94, E=62

normal step actual delta=1

post_update:
L=95
E=63
```

这就是 S1 最核心的 independent-divergence oracle。

---

# 7. S1 暴露出的下一个问题：原生 `positions` 同时承担了两个语义

S1 建立：

```text
L = 128
E = 64
```

以后，继续看：

```text
prepare_inputs()
    ↓
prepare_pos_seq_lens()
```

会发现原生 vLLM 通过 `num_computed_tokens` 产生：

```text
positions
seq_lens
```

而 `positions` 后面有两个完全不同的 consumer：

```text
positions
   ├─ model / RoPE
   └─ BlockTables.compute_slot_mappings()
```

这在原生：

```text
logical == physical
```

时没有问题。

但 reclaim 后：

```text
logical=128
physical=64
```

就冲突了。

---

# 8. 为什么不能把 `positions` 直接改成 physical

假设：

```text
logical base = 128
physical base = 64
query_len = 4
```

模型真正需要的 RoPE positions 是：

```text
[128,129,130,131]
```

如果为了 physical KV 写入方便，把 positions 改成：

```text
[64,65,66,67]
```

那么模型语义就被破坏。

因为 reclaim 只是改变：

```text
KV 被放在哪里
```

而不是改变：

```text
这些 token 在序列中的 logical position
```

所以必须拆：

```text
logical positions
vs
physical cache positions
```

这就是 S2。

---

# 9. S2：一个 kernel 同时生成 logical / physical 两套位置

核心 producer：

```text
vllm/v1/worker/gpu/input_batch.py
    _prepare_pos_seq_lens_kernel()
    prepare_pos_seq_lens()
```

S2 后对同一个 request：

```text
logical base = num_computed_tokens
physical base = effective_kv_len
```

生成：

```text
positions
= logical + local_offset

cache_positions
= physical + local_offset
```

同时：

```text
seq_lens
= logical + query_len

effective_kv_seq_lens
= physical + query_len
```

例如：

```text
L = 128
E = 64
q = 4
```

得到：

```text
positions:
[128,129,130,131]

cache_positions:
[64,65,66,67]

seq_lens:
132

effective_kv_seq_lens:
68
```

这四个值分别属于两个 domain。

---

# 10. 四个坐标系：理解 P1 必须彻底分开

后续几乎所有 bug，都来自把下面四个概念混在一起。

## 10.1 Logical token position

```text
128
129
130
...
```

属于模型序列语义。

用于：

```text
RoPE
model position
logical lifecycle
```

## 10.2 Physical cache sequence position

例如 reclaim 后：

```text
64
65
66
...
```

表示：

```text
这是当前压缩后的 KV 序列中的第几个 token。
```

用于找到：

```text
physical block-table column
+
offset inside block
```

## 10.3 Block-table column

```text
physical_pos // block_size
```

例如：

```text
64 // 16 = 4
```

表示 Worker compacted row 的 column 4。

## 10.4 Raw block / slot

Block table 再把 column 翻译成 physical block ID。

例如：

```text
row:
[B0 B1 B4 B5 B6]

column 4
→ B6
```

最终 raw slot：

```text
block_id * block_size + offset
```

所以：

```text
logical position
≠ cache position
≠ block-table column
≠ raw slot
```

这四层必须分开。

---

# 11. S2 修的是 KV WRITE，不是 KV READ

`cache_positions` 最终进入：

```text
GPUModelRunner.prepare_attn()
    ↓
BlockTables.compute_slot_mappings()
```

然后得到：

```text
slot_mapping
```

再用于 KV cache write：

```text
reshape_and_cache_flash
```

所以 S2 本质是：

```text
KV WRITE addressing correction
```

链路：

```text
effective_kv_len
      ↓
cache_positions
      ↓
compute_slot_mappings
      ↓
slot_mapping
      ↓
KV write
```

但是 Attention read 还没有修。

---

# 12. 为什么 S2 做完仍然不够：Attention 可能继续读 logical 长度

即使新 token 已经写到了正确的 compact physical slot：

```text
cache_position=64
→ compact row column 4
```

如果 Attention backend 仍然认为：

```text
KV sequence length = logical seq_len
```

例如：

```text
132
```

而实际 physical KV 只有：

```text
68
```

那么 READ extent 仍然错误。

因此 S3 的问题是：

> **paged KV write address 已经 physical 化，但 Attention 的 valid KV read length 也必须 physical 化。**

---

# 13. S3：为什么不能把 generic `seq_lens` 整体改成 physical

源码 audit 发现：

```text
seq_lens
```

不是只有 FA consumer。

它还属于 generic runtime metadata，并被其他逻辑使用。

因此不能简单：

```text
seq_lens = effective physical length
```

否则会污染非 Attention logical semantics。

所以 S3 的设计是：

```text
generic logical seq_lens
继续保留

新增：
effective_kv_seq_lens
作为 optional physical channel
```

然后在最靠近 approved backend 的地方切换。

---

# 14. S3 的完整 Attention transport chain

physical read-length channel：

```text
RequestState.effective_kv_len
        ↓
_prepare_pos_seq_lens_kernel
        ↓
InputBuffers.effective_kv_seq_lens
        ↓
InputBatch.effective_kv_seq_lens
        ↓
DefaultModelState.prepare_attn()
        ↓
build_attn_metadata()
        ↓
CommonAttentionMetadata
        ↓
FlashAttentionMetadataBuilder
        ↓
FlashAttentionMetadata.seq_lens
        ↓
seqused_k
        ↓
FA2
```

最终：

```text
S2 = WRITE address
S3 = READ extent
```

这是理解 paged KV correctness 最重要的一组拆分。

---

# 15. 为什么 physical channel 要尽量晚地接管

P1 V1 最终没有把整个：

```text
CommonAttentionMetadata.seq_lens
```

变成 physical。

而是在：

```text
normal FA2
DCP=1
non-cascade
```

这一条 approved path 中，由 FlashAttention builder 使用：

```text
effective_kv_seq_lens
```

原因是：

```text
generic runtime metadata
属于跨 backend contract；

physical KV read length
属于当前 backend specialization。
```

越早把 generic field physical 化，影响面越大。

这也是 S3 review-fix 最核心的 architecture discipline。

---

# 16. 到这里 execution semantics 才真正支持 logical / physical divergence

S1/S2/S3 合起来，建立了下面的 contract：

| 语义 | persistent | per-forward | consumer |
|---|---|---|---|
| logical | `num_computed_tokens` | `positions`, `seq_lens` | model / RoPE / logical runtime |
| physical | `effective_kv_len` | `cache_positions`, `effective_kv_seq_lens` | slot mapping / FA KV read |

例如：

```text
L=94
E=62
q=3
```

当前 forward：

```text
logical positions:
94,95,96

physical cache positions:
62,63,64

logical seq_len:
97

physical effective KV seq_len:
65
```

模型看到的 token 仍然是：

```text
94,95,96
```

KV write/read 则使用：

```text
62,63,64 / 65
```

这才让 physical compaction 在语义上成为可能。

---

# 17. 进入 T2：现在终于可以真正改变 Worker block view

S1/S2/S3 解决的是：

```text
如果 logical / physical 不一样，
execution 怎么跑？
```

T2 开始解决：

```text
Worker block table 到底怎么从 old layout
变成 retained layout？
```

canonical example：

```text
old:
[B0 B1 B2 B3 B4 B5]

keep indices:
[0,1,4,5]

retained:
[B0 B1 B4 B5]

candidate freed:
[B2 B3]
```

注意：

```text
candidate freed
```

在 T2 只是：

```text
“未来可以 free 的候选”
```

而不是：

```text
“已经可以被 BlockPool 复用”
```

真正 free/reuse 属于 T3。

---

# 18. 为什么 whole-block reclaim 是 V1 必须的限制

这个限制不是为了简单而随便定的，而是来自 allocator cadence。

假设：

```text
S = block_size
L = logical progress
E = physical progress

whole-block reclaim:
L - E = R * S
```

所以：

```text
L mod S = E mod S
```

这意味着 logical 和 physical 虽然绝对长度不同，但它们跨 block boundary 的节奏保持一致。

canonical：

```text
L=94
E=62
S=16

94 mod 16 = 14
62 mod 16 = 14
```

下一步：

```text
L=95
E=63
```

再下一步：

```text
L=96
E=64
```

logical 和 physical 同时跨 boundary。

这正是 A3 allocator delta audit 证明的关键。

---

# 19. 为什么 `new_effective_kv_len` 是 62，不是 64

原 request：

```text
L=94
```

最后 block 只有：

```text
14 valid tokens
```

reclaim 两个完整 blocks：

```text
94 - 2*16 = 62
```

所以：

```text
new_E = 62
```

不能写：

```text
retained block count * 16
= 4*16
= 64
```

因为最后一个 retained block 不是 full。

这也说明：

> **block capacity 与 valid token count 是两个概念。**

---

# 20. BlockTables：T2 最关键的源码位置

核心文件：

```text
vllm/v1/worker/gpu/block_table.py
```

核心对象：

```text
BlockTables
```

其中：

```text
block_tables[i]
```

是 persistent GPU block-table row。

而：

```text
num_blocks.np
```

是 CPU side active-prefix frontier。

关键函数：

```python
append_block_ids(req_index, new_block_ids, overwrite)
```

语义：

```text
overwrite=True
→ start=0
→ 从 row 头重建 active prefix

overwrite=False
→ start=current num_blocks.np
→ 从 active frontier 后 append
```

并且函数调用时就会更新：

```text
num_blocks.np
```

---

# 21. `StagedWriteTensor`：为什么“stage”不是立即改 GPU

核心文件：

```text
vllm/v1/worker/gpu/buffer_utils.py
```

`StagedWriteTensor` 不存在一个完整的：

```text
CPU mirror of the GPU row
```

它有的是：

```text
persistent .gpu tensor

+
CPU/Python staged descriptor lists:
_staged_write_indices
_staged_write_starts
_staged_write_contents
_staged_write_cu_lens
```

所以：

```text
stage_write()
```

不是立即修改 GPU row。

它只是记录：

```text
“稍后要对 row X，从 column Y，写这些内容。”
```

真正 publication 在：

```text
apply_write()
```

---

# 22. `UvaBackedTensor` 与 `StagedWriteTensor` 为什么不能混淆

`num_blocks` 使用的是：

```text
UvaBackedTensor
```

它确实有：

```text
CPU source-of-truth .np
```

所以：

```text
num_blocks.np
```

可以作为 Worker CPU 当前 active frontier。

但是 block-table row 本身：

```text
StagedWriteTensor
```

没有完整 CPU row mirror。

这是后面为什么“只有 keep indices 时，如果强行走现有 CPU staged-write API，可能需要 GPU→CPU 读 actual IDs”的根源。

最终 V1 选择：

```text
control-plane 直接 materialize retained block IDs
```

因此无需 D2H block-table readback。

---

# 23. 为什么 shorter overwrite 后 stale tail 没关系

例如原 GPU backing row：

```text
[B0 B1 B2 B3 B4 B5]
```

overwrite retained：

```text
[B0 B1 B4 B5]
```

可能实际 backing memory 变成：

```text
[B0 B1 B4 B5 B4 B5]
```

因为 overwrite 只重写前四列，并不主动把旧 tail 清零。

但是：

```text
num_blocks = 4
```

所以 active prefix 是：

```text
[B0 B1 B4 B5]
```

`gather_block_tables()` 只按照 active count gather。

因此：

```text
tail 中旧的整数 ID
只是 stale metadata
不是 active ownership duplication
```

A4 oracle 专门覆盖了这个 stale-tail boundary。

---

# 24. 第一次 Worker reclaim 设计为什么选择 materialized retained IDs

有两种主要方向。

## 24.1 Strategy A：CPU/control-plane materialized retained IDs

例如：

```text
old:
[B0 B1 B2 B3 B4 B5]

keep:
[0,1,4,5]

producer 直接得到：
[B0 B1 B4 B5]
```

Worker 只需要：

```text
overwrite retained row
```

优点：

```text
不需要 GPU gather kernel
不需要 temp buffer
不需要 GPU→CPU block-table readback
不需要处理 in-place compaction hazard
复用现有 BlockTables primitive
```

当前 V1：

```text
max_model_len=2048
block_size=16
```

最多约：

```text
128 block IDs
```

`int32` 整行约：

```text
512 bytes
```

因此 full metadata row materialization 在 V1 是合理 correctness-first 方案。

这不是在声称它性能最好。

它只是：

```text
实现面最小
正确性最容易证明
```

---

# 25. 为什么没有选择 GPU keep-index compaction

另一种方案是：

```text
Worker 收到 keep_indices=[0,1,4,5]

GPU:
old row
+
keep_indices
→ gather retained row
```

这在架构上完全可行。

而且 keep indices 本身可以直接 CPU→GPU 传输，并不天然要求 D2H。

但是需要新增：

```text
GPU-side compact primitive
temp/two-phase gather
```

因为 naive：

```text
row[j] = row[keep[j]]
```

可能产生 source/destination overlap hazard。

V1 没有必要为了 512B 级 metadata row 引入这一层复杂度，因此 defer 到 V2。

---

# 26. 真正棘手的问题：同一步 Scheduler 可能已经分配了 B6

这是 T2 实现中最关键的 integration 问题。

假设 reclaim 前：

```text
Worker:
[B0 B1 B2 B3 B4 B5]

L=94
E=94
```

reclaim 后：

```text
retained:
[B0 B1 B4 B5]

new_E=62
```

这一轮 scheduled query 如果跨 physical boundary：

```text
cache positions:
62,63,64
```

那么 position 64 需要一个新的 physical block。

Scheduler 在 Worker 收到当前 step 之前，已经通过：

```text
allocate_slots()
```

分配：

```text
new_block_ids=[B6]
```

所以最终 Worker 在 forward 前必须看到：

```text
[B0 B1 B4 B5 B6]
```

---

# 27. `new_block_ids` 的真实语义：它是 continuing request 的 current-step delta

Scheduler：

```text
req_to_new_blocks[req]
```

只记录当前 step 新分配的 blocks。

`_make_cached_request_data()` 把它变成：

```text
CachedRequestData.new_block_ids
```

对于 continuing request，它不是完整 row：

```text
[B0 B1 B2 B3 B4 B5 B6]
```

而只是：

```text
[B6]
```

Worker 原本已经有 persistent old row，因此 upstream 正常路径只需要 delta append。

---

# 28. 原生 Worker continuing path

核心位置：

```text
vllm/v1/worker/gpu/model_runner.py
    update_requests()
```

概念：

```python
for req in scheduled_cached_reqs:
    update CPU logical mirror

    if req_new_block_ids is not None:
        block_tables.append_block_ids(
            req_index,
            req_new_block_ids,
            overwrite=False,
        )
```

所以普通 request：

```text
old:
[B0 B1 B2 B3 B4 B5]

new=[B6]

→ append from current frontier 6

final:
[B0 B1 B2 B3 B4 B5 B6]
```

这就是原生 delta update。

---

# 29. 为什么 reclaim request 不能直接跳过整个 `update_requests()`

`update_requests()` 不只 append blocks。

它还负责：

```text
1. 更新 num_computed_tokens_np
2. 更新 num_computed_prefill_tokens
3. zero freshly allocated cache blocks
4. apply copy-on-write block copies
```

因此未来 01B 的正确 integration 不是：

```text
if reclaim:
    skip update_requests()
```

而是：

> **保留整个 update_requests 生命周期，只替换该 request 的 block-table mutation 分支。**

这点对后续代码位置判断非常重要。

---

# 30. 第一次 01A 的想法：先 reclaim，再 normal append

最初 approved route：

```text
Step 1:
overwrite retained row

[B0 B1 B2 B3 B4 B5]
→
[B0 B1 B4 B5]

num_blocks:
6 → 4

Step 2:
normal append B6

start = current num_blocks = 4

→
[B0 B1 B4 B5 B6]

num_blocks:
4 → 5

Step 3:
single apply_staged_writes()
```

从 **frontier semantics** 看，这完全合理。

甚至解决了另一个错误顺序。

---

# 31. 为什么“先 append B6，再 reclaim”不行

如果 upstream append 先执行：

```text
old num_blocks=6

append B6 at column 6
num_blocks=7
```

然后 reclaim overwrite：

```text
[B0 B1 B4 B5]
num_blocks=4
```

GPU backing 可能变：

```text
col0 col1 col2 col3 col4 col5 col6
 B0   B1   B4   B5   B4   B5   B6
```

此时：

```text
active prefix count = 4
```

B6 虽然没被覆盖，但被 stranded 在 active prefix 之外。

physical position 64：

```text
column = 64 // 16 = 4
```

它需要：

```text
active col4
```

而 B6 却在旧的：

```text
col6
```

所以错误不在“B6 被覆盖”，而在：

> **B6 被写到了旧 frontier 后面，最终不属于新的 active prefix。**

---

# 32. 关键发现：01A 的逻辑没错，publication primitive 却不支持

真实 runtime oracle 发现：

```text
append_block_ids(retained, overwrite=True)
```

会产生一条 staged descriptor：

```text
descriptor #1
```

然后：

```text
append_block_ids([B6], overwrite=False)
```

又产生：

```text
descriptor #2
```

这两个 descriptor 都属于同一个 publication window。

问题在：

```text
StagedWriteTensor
```

的 descriptor UVA buffer capacity 是按：

```text
num_rows
```

分配的。

真实测试：

```text
BlockTables(max_num_reqs=1)
```

capacity：

```text
1
```

但同一个 request stage 两次：

```text
_staged_write_indices=[0,0]
```

数量：

```text
2
```

于是 `apply_staged_writes()` 时出现：

```text
ValueError:
could not broadcast input array from shape (2,)
into shape (1,)
```

这就是：

```text
P1-M1-T2-IMPL-01A
= DESIGN_CONFLICT
```

---

# 33. 这个 DESIGN_CONFLICT 到底说明了什么

它没有推翻：

```text
logical/physical decoupling
```

没有推翻：

```text
whole-block reclaim
```

没有推翻：

```text
retained row [B0 B1 B4 B5]
```

也没有推翻：

```text
B6 应该落在 col4
```

它只说明：

> **我们用“两条 staged mutation”表达一个 request 的最终 block-row state，与现有 StagedWriteTensor 的 publication contract 不匹配。**

所以问题属于：

```text
mutation-expression / publication mechanism
```

而不是：

```text
KV reclaim semantics
```

这是非常重要的定位。

---

# 34. 为什么不修改 `StagedWriteTensor`

发现 conflict 后有几个候选方案：

```text
A. 合并成一条 final-row write
B. 做两次 publication
C. 扩大 StagedWriteTensor descriptor capacity
D. 新增 GPU-side compaction
```

最终选择 A。

不选 C 的原因：

```text
会修改 vLLM generic infrastructure；
需要重新审计全部 caller；
multi-group / fused writer / ordering 都会被影响；
只是为了 P1 一个 reclaim integration，不值得。
```

不选 B 的原因：

```text
增加第二个 publication boundary；
引入额外 kernel / stream / ordering seam；
破坏原来 execute_model 中
“metadata mutation → single publication → prepare inputs”
的结构。
```

所以最小正确方案是：

```text
把同一个 request 最终应该长什么样
在 CPU 先算好，
然后一次 publication。
```

---

# 35. R1：final-row one-write composition

新设计：

```text
retained:
[B0 B1 B4 B5]

same-step new:
[B6]

        ↓ CPU compose

final_physical_row:
[B0 B1 B4 B5 B6]

        ↓

ONE
append_block_ids(
    overwrite=True
)

        ↓

ONE staged descriptor
```

从：

```text
two ordered mutations
```

变成：

```text
one composed mutation
```

这是 R1 的核心。

---

# 36. `retained_block_ids` 与 `same_step_new_block_ids` 的 contract

这两个输入不能语义重叠。

`retained_block_ids` 必须表示：

```text
old Worker physical row
经过 reclaim
但尚未合入 current-step allocation delta
```

即：

```text
post-reclaim
/
pre-current-step-new-delta
```

canonical：

```text
retained:
[B0 B1 B4 B5]
```

`same_step_new_block_ids` 是：

```text
当前 step Scheduler allocate_slots
相对于 pre-step Worker row 新产生的 tail delta
```

canonical：

```text
[B6]
```

最终：

```text
retained + new
→ final row
```

不能让 retained 已经包含 B6，否则会：

```text
[B0 B1 B4 B5 B6]
+
[B6]

→ duplicate B6
```

这个 producer/transport contract 真正接入 Scheduler 属于 01B。

---

# 37. R1 中 `effective_kv_len=62` 为什么仍然正确

R1 最终 block row：

```text
[B0 B1 B4 B5 B6]
```

共有 5 个 block。

但：

```text
effective_kv_len != 5 * 16
```

因为 B6 是：

```text
current step 提前准备的 physical capacity
```

不是：

```text
16 个已经 computed 的 KV token
```

reclaim base：

```text
E=62
```

当前 query：

```text
cache positions:
62,63,64
```

forward 后如果实际完成 3 token：

```text
post_update:
E = 62 + 3 = 65
```

所以：

```text
block count = allocated capacity
effective_kv_len = valid completed KV length
```

二者不能混淆。

---

# 38. boundary slot-mapping oracle 为什么重要

R1 test 验证：

```text
cache positions:
[62,63,64]

slots:
[254,255,256]
```

假设：

```text
B5 id=15
B6 id=16
S=16
```

则：

```text
pos62:
block B5
offset14
slot=15*16+14=254

pos63:
block B5
offset15
slot=255

pos64:
block B6
offset0
slot=16*16=256
```

这个 test 不只是检查 row 内容。

它真正证明：

> **S2 产生的 physical cache position 穿过 compacted block frontier 后，确实会命中 same-step 新 block B6。**

这把：

```text
R1 block-table mutation
```

与：

```text
S2 slot mapping
```

连接起来了。

---

# 39. 为什么突然新增 `_commit_reclaim_transition()`：先从“原生代码缺了什么能力”看

上一版如果只写：

```text
model_runner.py
新增 _commit_reclaim_transition()
```

确实会像“突然多了一个很长的函数”。

正确理解方式不是先看这个函数，而是先看：

> **原生 `GPUModelRunner` 已经有什么 request-update 能力；P1 reclaim 又缺哪一种能力。**

---

## 39.1 原生 `update_requests()` 已经能做什么

源码位置：

```text
vllm/v1/worker/gpu/model_runner.py
GPUModelRunner.update_requests()
```

source audit 中的核心逻辑是：

```python
def update_requests(self, scheduler_output):
    reqs = scheduler_output.scheduled_cached_reqs
    num_computed_tokens_np = self.req_states.num_computed_tokens_np

    for req_id, num_computed_tokens, req_new_block_ids in zip(
        reqs.req_ids,
        reqs.num_computed_tokens,
        reqs.new_block_ids,
    ):
        req_index = self.req_states.req_id_to_index[req_id]

        # 更新 logical CPU mirror
        num_computed_tokens_np[req_index] = num_computed_tokens

        # continuing request 的正常 block 增长
        if req_new_block_ids is not None:
            self.block_tables.append_block_ids(
                req_index,
                req_new_block_ids,
                overwrite=False,
            )

    # 后面还有：
    # - num_computed_prefill_tokens bookkeeping
    # - new block zeroing
    # - CoW block copy
```

它非常适合原生 vLLM 的世界：

```text
旧 Worker block row 不需要缩短
只可能在 tail 继续增长
```

例如：

```text
old:
[B0 B1 B2 B3 B4 B5]

current-step new:
[B6]

update_requests():
append B6

result:
[B0 B1 B2 B3 B4 B5 B6]
```

因此 upstream 只需要：

```text
append-only delta update
```

---

## 39.2 P1 reclaim 需要的却不是 append-only update

P1 的 Worker physical view 需要发生：

```text
[B0 B1 B2 B3 B4 B5]
        ↓
[B0 B1 B4 B5]
```

这里不是：

```text
append tail
```

而是：

```text
replace current active row
```

同时还要：

```text
effective_kv_len:
94 → 62
```

logical：

```text
num_computed_tokens:
94 → 94
```

所以我们需要一个原生代码里不存在的 operation：

> **“对一个 active continuing request，校验并 stage 一次 physical-view replacement，同时绝对更新 physical progress，但不动 logical progress。”**

这个 operation 就是：

```text
_commit_reclaim_transition()
```

它不是 retention policy。

它不决定：

```text
删哪些 blocks
```

它也不是 allocator。

它不负责：

```text
free / reuse
```

它更像一个：

```text
Worker-side metadata transaction primitive
```

即：

```text
上游已经决定好了“新的 physical view 应该是什么”
        ↓
Worker 负责安全提交这个 view
```

---

# 40. 为什么这个函数加在 `GPUModelRunner`，而不是 `BlockTables`

这个位置是经过源码职责判断之后选的。

## 40.1 它同时需要操作两个不同对象

reclaim commit 不只是改：

```text
BlockTables
```

还必须改：

```text
RequestState.effective_kv_len
```

也就是：

```text
physical page mapping
+
physical valid length
```

需要一起变化。

而 `BlockTables` 本身只应该知道：

```text
row 怎么 stage / publish
num_blocks frontier 怎么维护
slot mapping 怎么查
```

它不应该知道：

```text
logical token progress
effective_kv_len
whole-block reclaim invariant
request lifecycle
```

如果把 reclaim semantics 塞进：

```text
block_table.py
```

会让一个 generic page-table primitive 开始理解 P1 的 request-level policy。

这不是合理职责边界。

---

## 40.2 `GPUModelRunner` 正好同时持有这些 state

`GPUModelRunner` 已经拥有：

```text
self.req_states
self.block_tables
```

它同时知道：

```text
request_id → req_index

logical RequestState

physical effective_kv_len

Worker BlockTables
```

所以它是最自然的：

```text
request-level physical transition coordinator
```

---

## 40.3 为什么它紧挨着 `update_requests()`

source audit 中：

```text
update_requests()
结束于约 line 873

_commit_reclaim_transition()
紧接着从约 line 874 开始
```

这不是说：

```text
update_requests()
当前已经调用了它
```

而是因为它和 `update_requests()` 属于同一个职责区域：

```text
“existing request 的 Worker state update”
```

把它放在这里方便后续 01B 做：

```text
update_requests()
    ├─ normal request
    │    → 原 append path
    │
    └─ reclaim request
         → _commit_reclaim_transition(...)
```

所以：

> **函数的位置是在为 future integration seam 做准备，但 R1 当前还没有真正 wiring 进去。**

这一点非常重要。

---

# 41. 当前 runtime 主链到底有没有走这个函数？

## 41.1 答案：R1 当前没有

当前真正 production runtime 主链仍然是：

```text
GPUModelRunner.execute_model()
        │
        ├─ update_pp_decode_requests()
        ├─ finish_requests()
        ├─ free_states()
        ├─ add_requests()
        ├─ update_requests()
        ├─ block_tables.apply_staged_writes()
        │
        ├─ prepare_inputs()
        ├─ prepare_attn()
        └─ forward
```

R1 没有新增：

```text
SchedulerOutput.reclaim_transition
```

也没有在：

```text
update_requests()
```

里增加：

```text
if reclaim:
    _commit_reclaim_transition(...)
```

因此当前 production execution 中：

```text
execute_model()
```

并不会凭空触发 reclaim。

---

## 41.2 那这个函数现在存在的意义是什么

它目前是一个：

```text
isolated Worker primitive
```

用 targeted tests 证明：

```text
如果 future 01B 把一个合法 reclaim decision
送到 Worker，

Worker 是否已经有一个正确、
可 publication、
不会产生 two-descriptor conflict 的 commit primitive？
```

这就是 01A / 01A-R1 的 Slice 边界。

换句话说：

```text
01A/R1:
“枪本身能不能正确开火？”

01B:
“真正 runtime 在哪里、什么时候、拿什么参数调用它？”
```

不能把这两刀混起来。

---

# 42. 这个函数为什么看起来很长：它本质上是一个 transaction boundary

R1 的 helper 输入概念上是：

```text
request_id

retained_block_ids

new_effective_kv_len

expected_old_num_blocks

same_step_new_block_ids
```

它可以拆成四个阶段理解：

```text
Phase A：找到当前 Worker state

Phase B：验证 transition 合法性

Phase C：构造 final physical row

Phase D：只 stage mutation，不负责 publication
```

长的主要原因其实是：

```text
validation 多
```

不是 mutation 本身复杂。

真正 mutation 只有很小一段。

---

# 43. `_commit_reclaim_transition()` 逐段拆开：到底每一部分在干嘛

下面不是要求记代码，而是理解每组代码为什么存在。

---

## 43.1 Phase A：找到 request 当前 Worker row

先验证：

```text
request_id 是否仍在 req_id_to_index
```

因为 future transport 到 Worker 时，request 可能理论上已经：

```text
finished / preempted / removed
```

V1 当前 scope 会限制 continuing RUNNING，但 commit primitive 仍然必须防御 stale transition。

然后：

```text
req_index = req_id_to_index[request_id]
```

从 request ID 找到：

```text
RequestState row
BlockTable row
```

例如：

```text
request-a
→ row 0
```

---

## 43.2 Phase B1：读取 commit 前的 old state

helper 需要知道：

```text
old_num_blocks
old_effective_kv_len
logical_num_computed_tokens
block_size
```

各自作用：

### `old_num_blocks`

来自：

```text
block_tables.num_blocks.np[0, req_index]
```

它表示：

```text
当前 Worker active block-table prefix 有多少列
```

canonical：

```text
6
```

### `old_effective_kv_len`

当前 R1 correctness-first 版本从：

```text
effective_kv_len.gpu[req_index].item()
```

读取。

canonical：

```text
94
```

### `logical_num_computed_tokens`

用于确保：

```text
physical reclaim 不能把 logical progress 搞坏
```

canonical：

```text
94
```

### `block_size`

canonical：

```text
16
```

这些值组合起来，就是 commit 前的 validation snapshot。

---

## 43.3 Phase B2：为什么有 `expected_old_num_blocks`

输入：

```text
expected_old_num_blocks
```

不是为了计算 retained row。

它是一个：

```text
validation fence
```

例如 producer 认为：

```text
我是在 old row 有 6 blocks 时做出的 reclaim decision
```

于是 transport：

```text
expected_old_num_blocks = 6
```

Worker 真正收到时如果发现：

```text
num_blocks.np = 7
```

说明：

```text
producer decision
和
Worker current state
已经不在同一个版本
```

这时不能“尽量执行”。

必须：

```text
fail-before-mutation
```

所以它本质是：

```text
optimistic concurrency / stale-decision fence
```

---

## 43.4 Phase B3：retained list validation 为什么这么多

要检查：

```text
retained 非空

retained_count <= old_count

retained IDs 无 duplicate
```

这些都属于：

```text
Worker 能在不读取完整 old GPU row 的条件下做的本地 sanity checks
```

但是 Worker 当前**不能**证明：

```text
retained IDs 真的是 old canonical row 的有序子集
```

例如理论上：

```text
old:
[B0 B1 B2 B3 B4 B5]

传来：
[B0 B7 B4 B5]
```

仅看：

```text
count
duplicate
```

无法知道 B7 是不是原 row 成员。

为什么不直接从 GPU 把 old row 读回来验证？

因为那会引入：

```text
GPU→CPU block-table readback
+
同步
```

而未来 producer 本身就在 Scheduler/control-plane CPU 上拥有 canonical row。

所以这个 validation 应该在 producer 做。

因此 R1 helper 的原则是：

```text
能本地便宜验证的 → Worker 验证
需要 canonical old row 的 → future producer 验证
```

---

## 43.5 Phase B4：whole-block invariant validation

核心公式：

```text
physical_shrink
=
old_E - new_E
```

canonical：

```text
94 - 62 = 32
```

预期 shrink：

```text
expected_shrink
=
(old_num_blocks - retained_num_blocks) * block_size
```

canonical：

```text
(6 - 4) * 16
= 32
```

两者必须相等：

```text
old_E - new_E
==
removed_blocks * block_size
```

这就是在代码层钉死：

```text
V1 only supports whole-block reclaim
```

如果传：

```text
new_E = 63
```

则：

```text
94 - 63 = 31
```

和：

```text
2 * 16 = 32
```

不一致。

必须拒绝。

---

## 43.6 为什么还要检查 logical / physical modulo

whole-block reclaim 必须保持：

```text
L % S == E % S
```

canonical：

```text
94 % 16 = 14
62 % 16 = 14
```

这是后续：

```text
Scheduler allocation cadence
和
Worker physical boundary cadence
```

仍然一致的基础。

所以 helper 的 validation 不是“为了代码稳一点随便加”。

它是在 production boundary 上 enforce 我们此前 A3/A4 已经证明的 design invariant。

---

# 44. 真正 mutation 其实只有两件事

所有 validation 都通过后，R1 才开始 mutation。

---

## 44.1 第一件事：CPU compose final physical row

输入：

```text
retained:
[B0 B1 B4 B5]

same-step new:
[B6]
```

构造：

```text
final_physical_row:
[B0 B1 B4 B5 B6]
```

如果：

```text
same_step_new_block_ids is None
```

则：

```text
final = retained
```

这里完全不 copy KV payload。

只是在 CPU 上拼一个：

```text
block-ID metadata list
```

---

## 44.2 第二件事：ONE block-table overwrite

调用：

```python
self.block_tables.append_block_ids(
    req_index,
    (final_physical_row,),
    overwrite=True,
)
```

这个调用会：

```text
stage row write from column 0
+
更新 CPU num_blocks.np 到 final row length
```

canonical：

```text
num_blocks:
6 → 5
```

重点：

```text
只有一次 append_block_ids 调用
```

所以：

```text
BlockTables staged descriptor count = 1
```

这正是 R1 修复 DESIGN_CONFLICT 的核心。

---

## 44.3 第三件事：stage physical length shrink

如果是真的 reclaim：

```text
old_E != new_E
```

再：

```text
effective_kv_len.stage_write_elem(
    req_index,
    new_E,
)
```

canonical：

```text
94 → 62
```

注意这和 BlockTables 是：

```text
两个不同 StagedWriteTensor 对象
```

原 DESIGN_CONFLICT 不是“整个 Worker 只能有一条 staged write”。

问题是：

```text
同一个 BlockTables row
在同一 publication window
被表达成两条 block-table descriptors
```

R1 之后 block table 只有一条 descriptor。

`effective_kv_len` 自己有独立 publication state，不冲突。

---

# 45. 一个特别重要的点：helper 只负责 stage，不负责 apply

`_commit_reclaim_transition()` 名字容易让人误解成：

```text
调用它以后 GPU 状态已经立即全部生效
```

实际上不是。

它做的是：

```text
stage transition
```

真正 publication 仍复用已有机制：

```text
block_tables.apply_staged_writes()
req_states.apply_staged_writes()
```

为什么这么设计？

因为 vLLM 原来的 mutation model 本来就是：

```text
CPU/control code
先描述一批 metadata changes

        ↓

统一 publication

        ↓

prepare_inputs / prepare_attn / forward
```

我们不应该为了 reclaim 打破这套模型。

---

# 45.1 所以 future 01B 的完整 Worker 链应该长这样

当前还没实现，但根据已经审计的 integration seam，目标是：

```text
execute_model()
│
├─ finish_requests()
├─ free_states()
├─ add_requests()
│
├─ update_requests()
│    │
│    ├─ request A: no reclaim
│    │      ↓
│    │   原 upstream append(new_block_ids)
│    │
│    └─ request B: has reclaim transition
│           │
│           ├─ retained_block_ids
│           ├─ same-step new_block_ids
│           │
│           ▼
│       _commit_reclaim_transition()
│           │
│           ├─ validate
│           ├─ compose final row
│           ├─ stage ONE block-table overwrite
│           └─ stage effective_kv_len shrink
│
├─ block_tables.apply_staged_writes()
├─ req-state publication（按现有 state lifecycle）
│
├─ prepare_inputs()
│    ├─ logical positions
│    └─ physical cache_positions
│
├─ prepare_attn()
│    └─ slot mapping uses new final block row
│
└─ forward
```

再次强调：

```text
这张图是 01B 的目标 wiring；
R1 当前只实现中间那个 primitive。
```

---

# 45.2 为什么不在 01A 就直接改 `update_requests()`

因为如果直接改主链，一旦失败，你无法快速判断：

```text
是 reclaim transition 本身错？

还是 SchedulerOutput transport 错？

还是 request routing 错？

还是 same-step new_block_ids 合并错？

还是 staged publication 错？
```

01A 的做法是先把问题缩成：

```text
给我一个已经准备好的 transition，
Worker 能不能正确 commit？
```

这叫：

```text
primitive-first validation
```

R1 再进一步只解决：

```text
same-step final-row publication
```

等这个 PASS，01B 才去接 transport/routing。

这也是为什么现在会看到一个：

```text
“还没被主链调用的 helper”
```

它不是死代码设计。

它是 slice-driven development 的中间状态。

---

# 45.3 01A 的实验到底怎么做：不是跑完整模型，而是造一个最小真实 Worker state

测试文件：

```text
tests/v1/worker/test_gpu_reclaim_commit.py
```

它不是 E2E：

```text
Scheduler
→ EngineCore
→ Transformer
→ Attention
→ Sampler
```

而是 focused CUDA oracle。

测试直接构造：

```text
真实 RequestState
+
真实 BlockTables
+
production GPUModelRunner._commit_reclaim_transition
```

目的：

> **精准验证 Worker commit primitive，不让完整模型的其他变量干扰。**

---

# 45.4 Test Setup：怎么造 canonical old state

测试 helper `make_runner()` 概念上做：

```text
RequestState(
    max_num_reqs=1,
    max_model_len=256,
    max_num_batched_tokens=8,
    num_speculative_steps=0,
    ...
)
```

然后：

```text
add request-a
logical=94
physical=94
```

再构造真实：

```text
BlockTables(
    block_size=16,
    max_num_reqs=1,
    ...
)
```

把：

```text
[10,11,12,13,14,15]
```

写入 row 0。

然后：

```text
apply_staged_writes()
synchronize()
```

得到稳定初始状态：

```text
request-a → req_index 0

L=94
E=94

num_blocks=6

GPU block row:
[10,11,12,13,14,15]
```

为什么 `max_num_reqs=1`？

不仅为了测试简单。

更重要的是：

```text
StagedWriteTensor descriptor capacity = num_rows
```

设成 1 可以最确定地暴露：

```text
同一 request 两条 block-table descriptors
```

这个 conflict。

---

# 45.5 01A 是怎么发现 DESIGN_CONFLICT 的

原 approved composition 想做：

```text
1. helper:
   overwrite retained row

2. upstream-style normal append:
   append B6

3. apply once
```

测试真实执行：

```text
BlockTables(max_num_reqs=1)

stage overwrite
→ descriptor #1

stage append
→ descriptor #2
```

此时 Python staged metadata：

```text
_staged_write_indices = [0, 0]
```

但 publication buffer capacity：

```text
1
```

`apply_staged_writes()` 时真实报：

```text
ValueError:
could not broadcast input array from shape (2,)
into shape (1,)
```

所以这个 experiment 不是：

```text
“我想象两条 descriptor 可能有问题”
```

而是：

```text
真实 BlockTables
+
真实 StagedWriteTensor
+
真实 apply path
```

直接把原 design 打爆。

这就是 01A 被判：

```text
DESIGN_CONFLICT
```

的依据。

---

# 45.6 R1 实验怎么证明问题真的被修了

R1 不再：

```text
stage retained
stage B6
```

而是先：

```text
final =
[10,11,14,15]
+
[16]

=
[10,11,14,15,16]
```

然后调用一次 helper。

最关键测试：

```text
test_single_write_final_row_avoids_descriptor_conflict
```

验证至少三层东西。

### 第一层：apply 前 descriptor 数量

直接检查：

```text
staged block-table descriptor count == 1
```

这证明：

```text
问题机制本身已经消失
```

而不只是“这次刚好没报错”。

### 第二层：真实 publication

调用：

```text
block_tables.apply_staged_writes()
req_states.apply_staged_writes()
synchronize()
```

要求：

```text
不再出现 capacity error
```

### 第三层：publication 后最终状态

验证：

```text
active row:
[10,11,14,15,16]

num_blocks=5

effective=62

logical=94
```

这证明：

```text
one descriptor
```

不是以牺牲最终语义为代价换来的。

---

# 45.7 boundary slot-mapping 实验怎么做、为什么它比“row 对了”更强

测试：

```text
test_boundary_crossing_maps_position_64_to_new_block
```

不是只 assert：

```text
row[4] == 16
```

而是进一步走真实：

```text
BlockTables.compute_slot_mappings()
```

给 physical cache positions：

```text
[62,63,64]
```

结果：

```text
[254,255,256]
```

展开：

```text
62:
B5(id=15), offset14
→ 15*16+14
→ 254

63:
B5(id=15), offset15
→ 255

64:
B6(id=16), offset0
→ 256
```

这说明：

```text
R1 final row
+
S2 physical cache_positions
+
原生 compute_slot_mappings
```

三层真正接上了。

所以这个 test 已经跨过：

```text
“metadata list 看起来正确”
```

进入：

```text
“真正 KV write address 会落到正确新 block”
```

这一层。

---

# 45.8 invalid-before-mutation 实验是在验证什么

例如输入：

```text
old:
L=94
E=94
6 blocks

retained:
4 blocks

却传：
new_E=63
```

whole-block shrink 应该是：

```text
32
```

实际：

```text
31
```

helper 应该拒绝。

但是只“抛异常”还不够。

测试还要确认：

```text
num_blocks.np 没变

block-table staged descriptor 没增加

effective_kv_len 没变

logical 没变
```

这证明：

```text
validation
真的发生在 mutation 之前
```

而不是：

```text
先改一半
再发现参数不合法
```

这就是 transaction primitive 最重要的 atomicity contract。

---

# 45.9 normal-growth regression 为什么也在这个测试里

reclaim 后：

```text
L=94
E=62
```

再调用原 production：

```text
post_update()
```

模拟一个正常 token 成功完成。

预期：

```text
L=95
E=63
```

这个测试的意义是：

> R1 不能只让“reclaim 那一瞬间”状态正确，还必须保证它留下的 divergence state 能继续被 S1 normal execution 正确推进。

所以它是在把：

```text
T2 primitive
```

重新接回：

```text
S1 lifecycle
```

---

# 45.10 当前实验没有证明什么

R1 tests 很强，但它仍然不是 E2E。

它没有证明：

```text
Scheduler 会正确产生 retained_block_ids

SchedulerOutput 会正确 transport

update_requests 会正确识别 reclaim request

normal request 不会误走 full-row path

reclaim request 不会再执行 normal append

candidate freed blocks 会安全 free/reuse

T3 allocator 会正确工作
```

这些明确属于：

```text
01B
+
T3
```

所以 R1 PASS 的准确含义是：

> **Worker commit primitive PASS，不是完整 reclaim runtime PASS。**

---

# 45.11 代码修改到底“新增 / 修改 / 替换 / 删除”了什么

这一点要分 01A 和 R1 看。

## 01A：第一次新增 primitive

### `model_runner.py`

原来：

```text
没有任何 request-level reclaim commit primitive
```

01A 新增：

```text
_commit_reclaim_transition()
```

约 66 行。

它的目标是：

```text
验证 prepared reclaim transition
+
stage retained row
+
stage new effective length
```

注意：

> **01A helper 自己并没有把 upstream normal append 也写在函数内部。**

two-descriptor conflict 来自我们原计划的 overall composition：

```text
helper stage retained row
+
之后仍让 upstream normal path stage B6
```

两条 block-table mutation 进入同一 publication window。

这是上一版笔记需要纠正的地方。

---

## R1：没有再新增第二个 reclaim 函数

R1 做的是：

```text
重构已有 _commit_reclaim_transition()
```

新增概念输入：

```text
same_step_new_block_ids
```

把原来的：

```text
“只知道 retained row”
```

升级为：

```text
“知道 retained row
 + 当前 step new delta
 → 自己构造最终 row”
```

从而把 overall mutation：

```text
helper overwrite
+
upstream append
```

替换成：

```text
helper ONE final-row overwrite
```

所以代码层更准确的描述是：

```text
新增函数：
发生在 01A

R1：
修改/重构该函数
不是又新增一套函数
```

---

## `test_gpu_reclaim_commit.py`

01A 新增 focused test 文件。

R1 在这个文件上：

```text
保留 canonical / invalid / lifecycle tests

把原 conflict case
改造成 final-row single-write regression

新增 descriptor=1 assertion

新增 same-step B6 final-row test

新增 boundary slot-mapping test
```

---

## production 删除

```text
无
```

但是：

```text
01A helper 的旧 mutation semantics
被 R1 逻辑替换
```

所以：

```text
“没有删除文件/函数”
```

不等于：

```text
“旧逻辑还在并行存在”
```

历史保存的是：

```text
实施记录
raw evidence
DESIGN_CONFLICT 结论
```

production helper 已经被 R1 重构。

---

# 45.12 最后把“代码为什么这样改”压缩成一张图

```text
原生 vLLM
────────────────────────────────────
continuing request
update_requests()
    ↓
append-only delta

能力：
[B0..B5] + [B6]
→ [B0..B6]


P1 新需求
────────────────────────────────────
需要：
[B0..B5]
→ [B0 B1 B4 B5]

同时：
E 94→62
L 保持94

原生没有这个 request-level operation


所以 01A 新增
────────────────────────────────────
GPUModelRunner
._commit_reclaim_transition()

职责：
“prepared physical transition
的 Worker commit primitive”


原 01A composition
────────────────────────────────────
helper:
retained overwrite
        ↓ descriptor #1

upstream:
append B6
        ↓ descriptor #2

single publication
        ↓
capacity conflict


R1 重构
────────────────────────────────────
helper 收到：
retained
+
same-step new

        ↓ CPU compose

final row

        ↓

ONE overwrite
ONE block-table descriptor

        ↓

existing publication

        ↓

S2 cache_positions
compute_slot_mappings
正确命中 B6
```

这才是这段代码修改的完整因果链。

---


# 46. 为什么 `effective_kv_len.gpu.item()` 曾经被单独拿出来审计

当前 R1 helper 为了 validation，需要知道：

```text
old_E
```

当前 persistent physical truth 在：

```text
effective_kv_len.gpu
```

所以实现中可以：

```python
effective_kv_len.gpu[req_index].item()
```

CPU 读取 GPU scalar。

这在 correctness 上没问题。

问题只是：

```text
可能形成 CPU/GPU synchronization point
```

因此曾额外验证：

```text
能不能从 CPU 已有状态
L + pre-step active N + S
恢复 old E
```

---

# 47. CPU inference audit 的结论是什么

whole-block V1 下：

```text
L mod S = E mod S
```

如果：

```text
N = ceil(E/S)
```

那么：

```python
r = L % S

if r == 0:
    E = N * S
else:
    E = (N - 1) * S + r
```

例如：

```text
L=94
N=4
S=16

E=(4-1)*16+14
=62
```

exhaustive oracle 覆盖整个：

```text
max_model_len=2048
```

空间并通过。

但是这只是：

```text
Validated Optimization Option
```

不是 R1 blocker。

R1 correctness-first 暂时允许 reclaim event 读取一次 `.gpu.item()`，没有新增 `effective_kv_len_np` CPU mirror。

---

# 48. 为什么必须使用 pre-append `N`

这个 audit 还发现一个非常重要的 ordering 事实。

reclaim 后：

```text
L=94
E=62
N=4
```

可以推出：

```text
E=62
```

如果当前 step 已经把 B6 append 进 Worker active row：

```text
N=5
```

却仍使用旧 logical base：

```text
L=94
```

公式会推出：

```text
E=78
```

错误。

所以 old-E inference 若未来落地，必须发生：

```text
读取 old num_blocks.np
        ↓
validation
        ↓
compose final row
        ↓
overwrite 更新 num_blocks
```

而不能在 append 后再推。

这个 ordering 与 R1/01B 的 integration seam 恰好一致。

---

# 49. 为什么目前不新增 `effective_kv_len_np`

表面上可以给 physical state 再加一个 CPU mirror：

```text
effective_kv_len_np
```

但这样马上会引入：

```text
GPU post_update 后 CPU 怎么同步？
spec reject 怎么同步？
reclaim 怎么同步？
哪个才是 authoritative truth？
```

S1 当时故意没有加入它。

所以当前更合理的选择是：

```text
GPU persistent physical truth
继续单一维护
```

而不是为了一个 rare validation path 增加双 truth。

---

# 50. 从这个问题重新看 vLLM 的完整运行链

下面这张图，是本文最值得记住的 vLLM 视角。

```text
┌──────────────────────────────────────────────────────────┐
│ Scheduler CPU                                            │
│                                                          │
│ Request.num_computed_tokens                              │
│      │                                                   │
│      ├─ decide num_new_tokens                            │
│      ├─ allocate_slots()                                 │
│      │      ↓                                            │
│      │   current-step new_block_ids                      │
│      │                                                   │
│      └─ _make_cached_request_data()                      │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
              SchedulerOutput / CachedRequestData
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Worker CPU                                               │
│ GPUModelRunner                                           │
│                                                          │
│ add_requests() / update_requests()                       │
│      │                                                   │
│      ├─ update logical CPU mirror                        │
│      ├─ normal request: delta block append               │
│      └─ future reclaim request: final-row overwrite      │
│                                                          │
│ BlockTables.num_blocks.np = active physical frontier     │
│                                                          │
│ block_tables.apply_staged_writes()                       │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Worker GPU persistent state                              │
│                                                          │
│ num_computed_tokens.gpu      logical base                │
│ effective_kv_len.gpu         physical base               │
│ BlockTables.gpu              physical page translation   │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
                 prepare_inputs()
                         │
           ┌─────────────┴──────────────┐
           ▼                            ▼
 logical producer                  physical producer
 positions                        cache_positions
 seq_lens                         effective_kv_seq_lens
           │                            │
           ▼                            ▼
 model / RoPE                compute_slot_mappings
                                        │
                                        ▼
                                    KV WRITE

 effective_kv_seq_lens
           │
           ▼
 CommonAttentionMetadata
           │
           ▼
 FA2 builder
           │
           ▼
 seqused_k
           │
           ▼
 KV READ
                         │
                         ▼
                      forward
                         │
                         ▼
                    post_update
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
logical += actual delta          physical += actual delta
```

---

# 51. reclaim request 在 future 01B 中应该怎么进入这条链

未来 01B 理想 integration：

```text
CachedRequestData
    │
    ├─ req_id
    ├─ logical base
    ├─ normal same-step new_block_ids
    └─ reclaim transition metadata
              │
              ▼
update_requests()
              │
              ├─ normal request
              │      ↓
              │   upstream append
              │
              └─ reclaim request
                     │
                     ├─ retained row
                     ├─ same-step new delta
                     │
                     ▼
                  final row
                     │
                     ▼
                 ONE overwrite
```

关键：

```text
不能跳过 update_requests 整体
```

只替换该 request 的：

```text
block-table mutation
```

其余 upstream bookkeeping 保留。

---

# 52. 01B 仍然必须回答的问题

R1 只证明 Worker primitive。

还没解决：

## 52.1 谁生成 reclaim decision

需要 control-plane producer 基于 canonical old row 得到：

```text
retained_block_ids
candidate_freed_block_ids
new_effective_kv_len
expected_old_num_blocks
```

## 52.2 producer 的时间点

必须确保：

```text
retained
```

不包含 current-step：

```text
new_block_ids
```

否则会 duplicate。

## 52.3 transport ABI

要决定如何通过：

```text
SchedulerOutput
```

把 prepared transition 送到 Worker。

## 52.4 update_requests routing

必须做到：

```text
reclaim request:
final-row overwrite
skip its original normal block append

non-reclaim request:
original append unchanged
```

## 52.5 T3 还没有开始

candidate freed blocks 此时仍不能 reuse。

---

# 53. 为什么 T3 不能简单“Scheduler 也把 row 从 6 缩成 4”

这是后续非常关键的 allocator 问题。

当前 upstream allocator 大体基于：

```text
required blocks
=
ceil((logical + scheduled)/S)
```

减：

```text
Scheduler 当前拥有的 block 数
```

如果 T2 后 Scheduler 立即：

```text
ownership count:
6 → 4
```

但 logical：

```text
L=94
```

那么 allocator 会看到：

```text
ceil(94/16) - 4
= 6 - 4
= 2
```

错误地认为还缺两个 blocks。

所以 T3 不只是：

```text
free B2/B3
```

它还必须引入 Scheduler-side：

```text
effective physical occupancy
```

使 allocator 用 physical progress 计算需求。

这也是为什么 T2/T3 必须拆开。

---

# 54. A3 为什么允许 T2 暂时 Scheduler/Worker 不一致

T2 期间：

```text
Scheduler canonical ownership:
[B0 B1 B2 B3 B4 B5]

Worker execution view:
[B0 B1 B4 B5]
```

看起来两边 block count 不一样。

但因为 whole-block reclaim：

```text
L mod S = E mod S
```

所以未来 block-growth delta cadence 仍然一致。

即：

```text
Scheduler 什么时候认为需要一个 new block
```

与：

```text
Worker physical frontier 什么时候跨 block boundary
```

仍然同步。

A3 对：

```text
Q=1..64
```

做过 oracle，全部通过。

这就是 T2 temporary divergence 能成立的数学依据。

---

# 55. 当前测试 evidence 到底证明了什么

## S1/S2/S3

最终 targeted suite：

```text
13 passed
```

证明：

```text
persistent logical / physical state
position split
slot mapping split
FA2 read-length split
scope guard / fallback
```

## T2 A4

pure contract oracle：

```text
canonical transition PASS
modulo alignment PASS
block-growth equality PASS
stale-tail boundary PASS
identity PASS
duplicate/order/partial-tail invalid PASS
```

## 01A

真实结果：

```text
20 passed
3 failed
```

核心失败：

```text
same-step two-descriptor publication conflict
```

因此：

```text
DESIGN_CONFLICT
```

## 01A-R1

历史完整 CUDA run：

```text
25 passed
15 warnings
```

最后收尾复跑时 NVIDIA driver unavailable：

```text
25 skipped
```

因此 evidence 应准确描述为：

```text
R1 曾在 CUDA 环境完成 25 PASS；
最终 close-out rerun 因 driver unavailable 被 skip；
py_compile PASS；
git diff --check PASS。
```

不能把：

```text
25 skipped
```

写成新的 CUDA PASS。

---

# 56. R1 当前 Web Review 结论

技术上，R1 可以接受。

它解决了原 blocker：

```text
2 descriptors
→
1 descriptor
```

并且：

```text
没有改 generic StagedWriteTensor
没有新增 second publication
没有 GPU compaction
没有改 allocator/ownership
```

建议文档中保持三条精确措辞：

1. `same_step_new_block_ids` 是：
   **current-step Scheduler allocation delta relative to the pre-step Worker row**；
2. normal non-reclaim request：
   **仍然走 upstream delta append，不应该因为 helper 能做 full-row replacement 就全部重写整行**；
3. 历史 01A：
   **保留的是 DESIGN_CONFLICT evidence / 记录；production helper 已由 R1 重构。**

---

# 57. 通过这个问题，我们真正学到的 vLLM 架构

## 57.1 Scheduler state 和 Worker execution state 本来就不是同一层

Scheduler 负责：

```text
计划
预算
allocation
request lifecycle
```

Worker 负责：

```text
真实 model execution
GPU resident state
physical addressing
```

所以 logical / physical 解耦并不是“违反 vLLM 架构”。

相反，是顺着已有层次给每一层正确的 truth。

---

## 57.2 persistent state 与 per-forward metadata 必须分开

`effective_kv_len`：

```text
request lifetime state
```

`cache_positions`：

```text
current forward per-token derived metadata
```

`effective_kv_seq_lens`：

```text
current forward per-request derived metadata
```

不能为了“字段少”把它们合并。

---

## 57.3 paged KV 本质是两层信息

Attention 的 paged KV read 需要：

```text
page translation
+
valid read extent
```

即：

```text
block table
+
effective sequence length
```

只改其中一个都不完整。

---

## 57.4 KV WRITE 与 KV READ 是两个独立 correctness surface

WRITE：

```text
cache_positions
→ slot_mapping
```

READ：

```text
block_table + effective length
→ FA
```

这也是为什么 S2/S3 必须拆。

---

## 57.5 Metadata mutation 也有 runtime contract

`BlockTables.append_block_ids()` 看起来只是 Python helper。

但底下其实是：

```text
stage descriptors
→ UVA publication buffers
→ GPU mutation kernel
```

所以不能只看：

```text
最终 row 数学上是不是对
```

还必须看：

```text
现有 mutation primitive 能不能合法表达这个 transition
```

01A 就是一个典型例子：

```text
state transition 正确
publication expression 错误
```

---

## 57.6 “最小 patch”不等于“少看源码”

真正最小的正确 patch，往往需要先读更多源码。

这次为了只改：

```text
model_runner.py 一个 Worker helper
```

实际上必须审计：

```text
Scheduler producer
CachedRequestData
RequestState
BlockTables
StagedWriteTensor
execute_model ordering
slot mapping
FA metadata
post_update
allocator cadence
```

否则很容易在局部看起来合理，但真实 runtime contract 不成立。

---

# 58. 当前项目状态压缩

```text
S1
Persistent logical / physical progress
PASS

        ↓

S2
Logical model positions
vs
Physical KV write positions
PASS

        ↓

S3
Logical runtime seq_lens
vs
Physical FA2 read extent
PASS

        ↓

T2 A1-A4
Whole-block transition contract
PASS

        ↓

01A
retained overwrite
+
normal append
=
two descriptors
DESIGN_CONFLICT

        ↓

01A-R1
retained
+
same-step new
=
final physical row
=
ONE overwrite
PASS

        ↓

NEXT
01B
Scheduler reclaim decision transport
+
update_requests integration

        ↓

T3
Scheduler effective occupancy
+
canonical ownership reconciliation
+
BlockPool free/reuse
```

---

# 59. 最终应该记住的 canonical example

```text
Before
────────────────────────────────────

Logical:
L = 94

Physical:
E = 94

Worker row:
[B0 B1 B2 B3 B4 B5]


Reclaim Decision
────────────────────────────────────

remove:
[B2 B3]

retained:
[B0 B1 B4 B5]

new_E:
94 - 2*16 = 62


Current-Step Scheduler Allocation
────────────────────────────────────

same-step new:
[B6]


Worker R1 Composition
────────────────────────────────────

retained + new

[B0 B1 B4 B5]
+
[B6]

→

[B0 B1 B4 B5 B6]


Before Forward
────────────────────────────────────

logical base:
94

physical base:
62

block-table active count:
5

注意：
5 blocks 是 capacity，
不是 80 valid KV tokens。


Current Query
────────────────────────────────────

q=3

logical positions:
94,95,96

physical cache positions:
62,63,64


Slot Mapping
────────────────────────────────────

62 → B5 offset14
63 → B5 offset15
64 → B6 offset0


Attention Read
────────────────────────────────────

effective KV seq len:
62 + 3 = 65


Post Update
────────────────────────────────────

如果实际完成 3 tokens：

L:
94 → 97

E:
62 → 65
```

这就是当前 P1 execution-side whole-block reclaim contract 的完整闭环。

---

# 60. 当前最准确的一句话总结

P1 到 01A-R1 为止，并不是在“实现一个删除 KV block 的函数”。

真正完成的是：

> **把 vLLM 原本隐式绑定在 `num_computed_tokens / positions / seq_lens / block table` 上的 logical 与 physical KV 语义拆开，然后在不复制 KV payload 的前提下，为 continuing RUNNING request 建立一个可安全 publication 的 Worker physical-row transition primitive。**

原 01A 进一步证明：

> **正确的状态转换不等于正确的 runtime mutation expression。**

R1 最终通过：

```text
retained + same-step allocation delta
→ final physical row
→ ONE overwrite descriptor
```

解决了 publication 层的真实冲突。

现在 execution-side primitive 已基本清楚，下一步真正进入的是：

```text
Scheduler decision producer
→ transport ABI
→ update_requests routing
```

而不是继续改 BlockTables 或 StagedWriteTensor。

---

# 附录 A：关键源码地图

| 问题 | 文件 | 核心对象/函数 |
|---|---|---|
| Scheduler logical state | `vllm/v1/core/sched/scheduler.py` | `schedule`, `_make_cached_request_data`, `_update_after_schedule` |
| Scheduler allocator | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` |
| per-type allocation | `vllm/v1/core/single_type_kv_cache_manager.py` | `get_num_blocks_to_allocate`, `allocate_new_blocks` |
| Worker persistent state | `vllm/v1/worker/gpu/states.py` | `RequestState` |
| logical/physical producer | `vllm/v1/worker/gpu/input_batch.py` | `_prepare_pos_seq_lens_kernel`, `prepare_pos_seq_lens` |
| post-forward commit | `vllm/v1/worker/gpu/input_batch.py` | `post_update`, `_post_update_kernel` |
| Worker orchestration | `vllm/v1/worker/gpu/model_runner.py` | `add_requests`, `update_requests`, `prepare_inputs`, `prepare_attn`, `execute_model` |
| block-table state | `vllm/v1/worker/gpu/block_table.py` | `BlockTables`, `append_block_ids`, `apply_staged_writes`, `compute_slot_mappings` |
| staged publication | `vllm/v1/worker/gpu/buffer_utils.py` | `StagedWriteTensor`, `UvaBackedTensor` |
| generic attention metadata | `vllm/v1/attention/backend.py` | `CommonAttentionMetadata` |
| metadata builder | `vllm/v1/worker/gpu/attn_utils.py` | `build_attn_metadata` |
| model-state transport | `vllm/v1/worker/gpu/model_states/default.py` | `prepare_attn` |
| FA2 specialization | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionMetadataBuilder`, FA forward metadata |

---

# 附录 B：本笔记依据的工程记录 / evidence

主要依据：

```text
P1-M1-T1-S1-MRV2-num-computed-tokens与Effective-KV-State生命周期详解-v2.md

P1-M1-T1-S2-S3-vLLM-Logical-Physical-KV-解耦源码与实现详解-v3.md

P1-M1-T1-S1-IMPLEMENTATION-RECORD.zh-CN.md

P1-M1-T1-S2-S3-IMPLEMENTATION-RECORD.zh-CN.md

m1-t2-s1-blocktable-audit.log

m1-t2-a2-commit-frontier-audit.log

m1-t2-a3-allocator-delta-audit.log

m1-t2-a4-transition-oracle.log

m1-t2-01a-redesign-source-audit.log

m1-t2-effective-inference-audit.log

P1-M1-T2-IMPL-01A-Worker-Reclaim-Commit-Primitive-实施记录.md

P1-M1-T2-IMPL-01A-R1-Final-Row-Reclaim-Composition-实施记录.md
```

当前 R1 raw evidence root：

```text
/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a-r1/
```

其中包括：

```text
00-source-identity.log
01-source-review.log
02-targeted-tests.log
03-regression-tests.log
04-pycompile-diff-check.log
05-git-diff.log
06-final-status.log
```

---

# 附录 C：当前明确不应混入 01A-R1 的事项

```text
Scheduler reclaim producer
SchedulerOutput reclaim ABI
ownership shrink
BlockPool.free
candidate reuse
Worker ACK
allocator effective occupancy
retention policy
GPU keep-index compaction
Triton KV payload compaction
DCP/PCP/cascade/spec decode/CUDA Graph generalization
```

这些不是“忘了做”，而是被刻意留到后续 Slice，以保持每一层 contract 可独立验证。
