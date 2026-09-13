# 07-vLLM ModelRunner 执行数据面与 Paged KV 读写主链

> 固定版本：vLLM v0.26.0  
> commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> 当前配置：A100 80GB / Qwen3-0.6B / TP=1 / PP=1 / DP=1 / Offline LLM / vLLM V1 Engine / V2 Model Runner（MRV2）  
> 主要源码路径：`vllm/v1/worker/gpu/model_runner.py`  
> Attention Backend：当前 CUDA/A100 普通 decoder attention 路径下，`FLASH_ATTN` 是默认最高优先级候选；最终是否实际选中仍取决于 `validate_configuration()` 与运行环境依赖是否通过。  
> 本文定位：**承接上一篇 Scheduler / KV 控制面笔记，从 `SchedulerOutput` 开始进入 MRV2 ModelRunner，打通 Scheduler decision → GPU 输入组织 → block table / slot mapping → KV write → KV read → FlashAttention 的完整数据面链路。**

---

# 1. 这一阶段到底要解决什么问题

上一篇已经把 Scheduler / KV 控制面梳理到一个比较完整的程度：

```text
EngineCore.step()
↓
Scheduler.schedule()
↓
决定：
- 哪些 request 本轮执行
- 每个 request 执行多少 token
- 哪些 request 获得新的 KV blocks
- 哪些 request 被 preempt / finish
↓
SchedulerOutput
```

但是到了这里仍然存在一个巨大的断层：

```text
SchedulerOutput
↓
???
↓
GPU kernel
```

Scheduler 认识的是：

```text
request
num_scheduled_tokens
physical block ids
request lifecycle
```

GPU kernel 真正需要的却是：

```text
input_ids
positions
query_start_loc
seq_lens
block_table
slot_mapping
KV cache tensor
Attention metadata
```

因此这一阶段真正要回答的是：

> **Scheduler 做出的逻辑调度 / KV 资源决策，究竟如何被翻译成 GPU 可以直接执行的数据结构与物理地址？**

这就是 ModelRunner 的核心价值。

本文最终要打通下面这条链：

```text
SchedulerOutput
↓
MRV2 GPUModelRunner
↓
Persistent Request State
↓
Per-step InputBatch
↓
block_table / slot_mapping
↓
Attention Metadata
↓
QKV projection
↓
KV write
↓
Paged KV Cache
↓
KV read
↓
FlashAttention
```

---

# 2. 先建立 ModelRunner 的宏观定位

整个 vLLM V1 Engine 可以先压缩成：

```text
Frontend
↓
EngineCore
↓
Scheduler
↓
Executor
↓
Worker
↓
ModelRunner
↓
Model / Attention Backend / GPU Kernel
```

各层职责不是简单的“逐层调用”，而是解决不同变化维度：

```text
EngineCore
= orchestration / 主循环编排

Scheduler
= scheduling policy + request state + KV resource decision

Executor
= deployment / execution topology

Worker
= device / distributed / model runtime environment

ModelRunner
= execution-side state + GPU input preparation + model execution
```

对 ModelRunner 最准确的理解是：

> **ModelRunner 是 Scheduler 控制面与 GPU 数据面之间的 execution bridge。**

它不是简单地做：

```text
model(input)
```

而是先完成一次巨大的“表示转换”：

```text
Scheduler 世界
request A 本轮算 1 token
request B 本轮算 128 token
A 的逻辑 block 1 → physical block 18
B 新增 physical block 31,32

                    ↓ ModelRunner

GPU 世界
flattened input tokens
positions
query_start_loc
seq_lens
block_table
slot_mapping
backend-specific metadata
```

因此，ModelRunner 本质上包含两类工作：

```text
Control → Data translation
+
Actual model execution
```

前者是我们这次最重要的学习对象。

---

# 3. 当前一定要先区分：vLLM V1 Engine ≠ Model Runner V1

这个是本阶段非常重要的版本校正。

当前环境是：

```text
vLLM V1 Engine
+
V2 Model Runner（MRV2）
```

这里两个 V1/V2 不是同一个维度。

```text
vLLM V1
= Engine architecture generation
= EngineCore / Scheduler / KVCacheManager / Worker 等整体架构

Model Runner V1 / V2
= vLLM V1 Engine 内部 execution-side 的两套实现
```

当前主路径应该读：

```text
vllm/v1/worker/gpu/model_runner.py
```

而不是：

```text
vllm/v1/worker/gpu_model_runner.py
```

后者是 MRV1。

---

# 4. 为什么 MRV2 要重新设计 ModelRunner 状态

MRV1 的核心问题并不是“有 persistent state”这件事。

Serving runtime 必然需要保存跨 step 状态。

真正的问题是：

> **persistent request state 与 dynamic batch row 绑定得太紧。**

Continuous batching 的 batch 每轮都可能发生：

```text
Step t:
[A, B, C, D]

Step t+1:
[A, D, E]

Step t+2:
[D, E, F, G]
```

如果 request state 直接绑定 batch row：

```text
row 0 = A
row 1 = B
row 2 = C
row 3 = D
```

当 B/C finish，E 加入时，就会出现大量：

```text
row move
batch compaction
block table move
sampler state move
metadata update
cached request state synchronization
```

MRV2 的核心思想是把：

```text
Persistent Request State
```

和：

```text
Current-step Batch Layout
```

彻底分开。

因此 MRV2 的基本模型是：

```text
Persistent Request State
= stable request-indexed storage

Per-step InputBatch
= current scheduler decision 对 persistent state 的一个 gather view
```

可以类比：

```text
req_states
= database table

InputBatch
= SELECT / gather result of this step
```

这就是 MRV2 最核心的 architecture idea 之一。

---

# 5. MRV2 `execute_model()` 的宏观阶段

从源码来看，`GPUModelRunner.execute_model()` 不应该被理解成一个“大 forward”。

它可以拆成五个阶段：

```text
SchedulerOutput
↓

① persistent state synchronization
   finish_requests()
   free_states()
   add_requests()
   update_requests()
   block_tables.apply_staged_writes()

↓

② current-step input preparation
   prepare_inputs()
   → InputBatch

↓

③ KV / Attention addressing preparation
   prepare_attn()
   → block_tables
   → slot_mappings

↓

④ backend metadata preparation
   model_state.prepare_attn()
   → backend-specific attn_metadata

↓

⑤ model forward / attention / sampling
```

这个阶段划分比记住每个函数的具体代码更重要。

---

## 5.1 把真实 `execute_model()` 骨架放回这五个阶段

当前 MRV2 主路径可以用下面这段骨架理解：

```text
if not dummy_run:
    update_pp_decode_requests()

    finish_requests(scheduler_output)
    free_states(scheduler_output)
    add_requests(scheduler_output)
    update_requests(scheduler_output)

    block_tables.apply_staged_writes()

    if total_num_scheduled_tokens == 0:
        return ...

input_batch = prepare_inputs(scheduler_output, batch_desc)

block_tables, slot_mappings = prepare_attn(input_batch)

model_state.preprocess_state(...)

attn_metadata = model_state.prepare_attn(
    input_batch,
    ...,
    block_tables,
    slot_mappings,
    ...
)

model forward
```

这段代码最值得注意的不是调用顺序本身，而是它揭示了一个非常清晰的 phase boundary：

```text
SchedulerOutput
↓
先同步 persistent state
↓
再构造 per-step view
↓
再构造 addressing
↓
再转 backend metadata
↓
最后才真正 model forward
```

也就是说：

> `execute_model()` 前半段很大一部分工作并不是 GPU 数学计算，而是在构建“这一轮 GPU 计算所依赖的执行世界”。

---

## 5.2 为什么一定是“先 persistent state，后 InputBatch”

假设 Scheduler 本轮让一个 PREEMPTED request A resume。

SchedulerOutput 里已经带了：

```text
A 的 token state
A 当前 block ids
A 本轮 num_scheduled_tokens
```

如果先 `prepare_inputs()`，再 `add_requests()`：

```text
prepare_inputs
↓
查 req_id_to_index[A]
```

此时 A 根本还不存在于 ModelRunner persistent state。

所以正确顺序必须是：

```text
Scheduler lifecycle transition
↓
同步 execution-side state
↓
当前 step 才能从 persistent storage gather
```

同理，finished/preempted request 必须先 remove，否则 current batch / buffer state 可能还残留旧 execution state。

---

## 5.3 为什么 `block_tables.apply_staged_writes()` 在 `prepare_inputs()` 前后位置很敏感

这一层反映的是一个通用 runtime 原则：

> **在某份 state 即将成为后续 GPU input 的 source-of-data 之前，必须确保当前 step 累积的 pending writes 已经变成 consumer 可见状态。**

因此 staged write 不是独立的小技巧，而是 phase ordering 的一部分：

```text
produce state deltas
↓
commit / apply
↓
consumer reads
```

如果顺序反过来：

```text
consumer reads
↓
apply writes
```

当前 step 就可能使用上一轮旧数据。

所以看到：

```text
apply_staged_writes()
```

时应该把它理解成：

```text
state synchronization barrier / commit point
```

而不只是“批量 copy API”。

---

# 6. 第一阶段：Scheduler state 如何同步到 ModelRunner

Scheduler 才是 request scheduling state 的权威持有者。

ModelRunner 持有的是：

> **为了执行而维护的 execution-side state。**

因此每个 step 开头必须根据 `SchedulerOutput` 更新本地状态。

核心分三类。

## 6.1 Finished / Preempted request

源码：

```text
finish_requests()
```

当前 MRV2 中，preempted request 在 execution side 也按照“本次 execution state 生命周期结束”处理。

概念上：

```text
Scheduler Request
仍然存在
↓
status = PREEMPTED
↓
Scheduler 后续可能重新调度
```

但 ModelRunner：

```text
当前 active execution state
↓
_remove_request()
↓
释放 req_idx / execution-side state
```

因此：

```text
Scheduler Request 生命周期
≠
ModelRunner active execution-state 生命周期
```

这是理解 preemption 的关键。

---

## 6.2 New / Resumed request

Scheduler producer 端真实逻辑是：

```text
WAITING request   → scheduled_new_reqs
PREEMPTED request → scheduled_resumed_reqs
```

但 MRV2 路径会把：

```text
new + resumed
```

统一打包进：

```text
NewRequestData
```

所以 ModelRunner 消费侧看起来是：

```text
NewRequestData
↓
add_requests()
↓
重新建立 execution-side state
```

这解释了为什么：

```text
PREEMPTED request
在 Scheduler 不是“新 request”

但在 MRV2 execution side
可以按照 fresh execution state 重新加入
```

这不是语义冲突，而是两个生命周期层级不同。

---

## 6.3 Continuing RUNNING request

这些 request 已经存在于 ModelRunner persistent state。

Scheduler 只需要传递当前变化：

```text
num_computed_tokens
new block ids
其他 step delta
```

ModelRunner：

```text
update_requests()
```

只 patch 变化部分。

因此 MRV2 同时具备：

```text
Persistent state
+
Incremental updates
```

避免每一轮重新构建所有 request state。

---

# 7. `RequestState`：MRV2 persistent request state 的核心

`RequestState` 本质上是一组按 `req_idx` 索引的持久数组 / Tensor。

重要结构：

```python
self.req_id_to_index: dict[str, int]
self.index_to_req_id: dict[int, str]
self.free_indices = list(range(max_num_reqs))
```

当 request 加入：

```text
req_id
↓
free_indices.pop()
↓
req_idx
```

得到：

```text
req_id ↔ req_idx
```

例如：

```text
req_idx 255 → A
req_idx 254 → B
req_idx 253 → C
```

注意：

> `req_idx` 不是 scheduler request id，也不是 batch row。

它只是：

> **ModelRunner persistent state storage 的 slot index。**

request active execution state 存活期间通常稳定；finish / preempt 后归还 `free_indices`，未来可以被其他 request 重用。

---

# 8. 为什么 `req_idx` 可以稳定，而当前 batch 顺序不断变化

假设 persistent state：

```text
req_idx 255 = A
req_idx 254 = B
req_idx 253 = C
req_idx 252 = D
```

这一轮 Scheduler 只调：

```text
[A, D, C]
```

ModelRunner 不移动 A/D/C 在 persistent storage 中的位置。

而是构造：

```text
idx_mapping = [255, 252, 253]
```

表示：

```text
current batch row 0 → persistent row 255 → A
current batch row 1 → persistent row 252 → D
current batch row 2 → persistent row 253 → C
```

这就是 MRV2 最重要的数据组织方式：

```text
persistent layout 不动
current execution view 每轮重建
```

好处：

```text
1. 减少 batch compaction / row movement
2. request lifecycle 与 batch layout 解耦
3. 更适合 continuous batching
4. 更适合 async / pipeline execution
5. 降低跨 step state synchronization 复杂度
```

---

# 9. `RequestState` 中到底保存哪些东西

可以分成三组。

## 9.1 Token sequence state

例如：

```text
all_token_ids
prompt_len
prefill_len
total_len
```

它们描述：

```text
这个 request 到目前为止已经知道哪些 token
sequence 逻辑长度是多少
```

---

## 9.2 Computation progress state

例如：

```text
num_computed_prefill_tokens
num_computed_tokens
num_computed_tokens_np
```

这些字段把 Scheduler 的“计算 frontier”同步到 execution side。

核心语义：

```text
known token frontier
vs
computed token frontier
```

ModelRunner 需要知道：

```text
这一轮应该从哪个 sequence position 开始计算
```

---

## 9.3 Runtime state

例如：

```text
last_sampled_tokens
max_seq_len
draft_tokens
next_prefill_tokens
```

用于：

```text
模型输入构造
采样
prefill/decode continuation
speculative-related state
```

当前主线只需理解这些属于 execution runtime state。

---

# 9.4 为什么已经“统一 Prefill / Decode”了，`RequestState` 里仍然有 `prefill_len`、`num_computed_prefill_tokens`、`prepare_prefill_inputs()`

这一点非常容易产生误解。

前面说：

```text
vLLM V1 不再把 Prefill / Decode
当成 Scheduler 顶层的两个互斥执行模式
```

并不意味着：

```text
Prompt token 与 Generated token
在 execution side 已经完全没有语义差异
```

真正统一的是：

> **Scheduler 如何描述“这一轮要做多少计算”。**

没有被抹掉的是：

> **这些待计算 token 到底来自哪里。**

---

## 9.4.1 用一个 request 看清楚

假设 prompt：

```text
[P0 P1 P2 P3 P4 P5]
```

当前：

```text
all_token_ids
= [P0 P1 P2 P3 P4 P5]

prefill_len
= 6

num_computed_tokens
= 3
```

说明：

```text
P0 P1 P2
已经 forward

P3 P4 P5
token ID 已经知道
但还没有 forward
```

Scheduler 这一轮只需要说：

```text
A 本轮 schedule 2 tokens
```

它不必声明：

```text
A 当前 mode = PREFILL
```

因为：

```text
known frontier = 6
computed frontier = 3
```

自然就知道还有 token 要计算。

但是 ModelRunner 真正构造 `input_ids` 时必须回答：

```text
这两个 token 的 token_id 从哪里拿？
```

答案是：

```text
all_token_ids[3:5]
→ P3 P4
```

这就是 `prepare_prefill_inputs()` 的工作。

所以它不是：

> “重新把整个 batch 判定成 Prefill mode。”

而是：

> **在统一 batch 中，把那些“token ID 已经存在于 prompt/history、但尚未计算”的 token 填入当前 input buffer。**

---

## 9.4.2 Decode 时数据来源不同

假设 prompt 已经全部计算完：

```text
prefill_len = 6
num_computed_tokens = 6
```

上一轮模型采样得到：

```text
O0
```

于是：

```text
all_token_ids
= [P0 P1 P2 P3 P4 P5 O0]
```

Scheduler 仍然只看到：

```text
known = 7
computed = 6
↓
schedule 1 token
```

Scheduler 逻辑与 prefill 没有本质不同。

但 ModelRunner input preparation 知道：

```text
position 6
已经越过 prefill_len
```

这个 token 是上一轮 runtime/sample 产生的 generated token。

因此“统一”的位置是：

```text
Scheduler:
都统一成 schedule N tokens
```

而“仍然存在差异”的位置是：

```text
ModelRunner:
这些 N 个 token 的具体来源和准备方法不同
```

---

## 9.4.3 一个 batch 可以同时有 Prefill 与 Decode token

假设当前 step：

```text
A:
prompt 已结束
本轮计算上一轮 sampled token O20
→ 1 token

B:
prompt 还有 128 token 没计算
→ 128 tokens

C:
prompt 已结束
本轮计算 sampled token O5
→ 1 token
```

Scheduler 统一输出：

```text
num_scheduled_tokens
= [1, 128, 1]
```

ModelRunner 最终 flatten：

```text
input_ids =
[
 A_O20,
 B_Pxxx ... B_Pxxx,
 C_O5
]
```

然后：

```text
query_start_loc
= [0, 1, 129, 130]
```

所以一个 `InputBatch` 内完全可能同时存在：

```text
decode-like query
+
chunked-prefill query
```

这正是为什么 V1 不适合在 batch 顶层做：

```python
if batch.is_prefill:
    ...
else:
    ...
```

---

## 9.4.4 `num_computed_prefill_tokens` 为什么仍然有意义

`num_computed_tokens` 是：

```text
整个 sequence 的 execution frontier
```

而 `num_computed_prefill_tokens` 关注的是：

```text
原始 prompt / prefill 区间本身的进度
```

例如：

```text
prompt_len = 1000
```

在 chunked prefill 中：

```text
Step 1:
num_computed_tokens = 256
num_computed_prefill_tokens = 256

Step 2:
num_computed_tokens = 512
num_computed_prefill_tokens = 512

...

Prompt 全完成后:
num_computed_tokens = 1000
num_computed_prefill_tokens = 1000
```

之后进入 generation：

```text
算到 O0:
num_computed_tokens = 1001

但：
num_computed_prefill_tokens
仍然停在 prompt boundary 附近
```

因此：

```text
num_computed_tokens
= whole sequence frontier

num_computed_prefill_tokens
= prompt-specific frontier
```

两个字段描述不同语义，不冲突。

---

## 9.4.5 `prepare_prefill_inputs()` 这个名字应该怎么理解

不要理解成：

```text
“当前 GPUModelRunner 正在进入 Prefill 模式”
```

应该理解成：

```text
“当前 step 的 flattened input 中，
哪些 token 需要从尚未消费完的 prefill token source 填入？”
```

也就是说它是：

```text
workload-specific helper
```

而不是：

```text
top-level execution mode switch
```

最终准确表述是：

> **vLLM V1 消除的是 Prefill/Decode 的“顶层调度模式二分”，没有消除 prompt/generated token 的数据语义差异。`prepare_prefill_inputs()` 处理的是后者。**

---

# 10. `RequestState` 的数据放置：UVA、StagedWrite、CPU mirror 到底在解决什么

这一节不能只记“有些东西放 UVA，有些东西放 GPU”。

真正的问题是：

> **MRV2 的 persistent state 既要让 CPU 很方便地维护，又要让 GPU 在执行前能够低成本地读取；而 serving 每一个 step 都会产生大量非常小的状态更新。**

如果所有状态都采用最朴素的方式：

```text
CPU 修改一个字段
↓
立刻 H2D copy
↓
再修改另一个字段
↓
再 H2D copy
↓
...
```

那么 decode 场景会产生大量细碎的 CPU→GPU 操作。

GPU kernel 可能只执行几十微秒，但 CPU metadata path 却被：

```text
Python dispatch
CUDA enqueue
small memcpy
synchronization
```

拖慢。

因此 MRV2 对 persistent state 做了更加细致的数据放置和更新策略。

---

## 10.1 先区分三种东西：真正 GPU Tensor、UVA-backed、StagedWriteTensor

可以先建立下面的抽象：

```text
A. GPU resident tensor
   数据主要驻留 HBM
   ↓
   GPU 高频访问

B. UVA-backed tensor
   数据主要驻留 Host memory
   ↓
   GPU 可以通过统一虚拟地址空间访问

C. StagedWriteTensor
   重点不是“放在哪里”
   而是“更新什么时候真正提交”
```

所以：

```text
UVA
= placement / visibility 问题

StagedWrite
= update batching / synchronization 问题
```

这两个概念不要混在一起。

---

## 10.2 UVA 是什么

UVA = Unified Virtual Addressing。

这里不要把它和 CUDA Unified Memory 混淆。

在当前这套 runtime 心智模型里，可以把 UVA 理解为：

```text
Host memory
     │
     │ page-locked / GPU-visible address
     ▼
GPU 可以通过统一地址空间直接寻址
```

也就是说，有些 persistent state 不需要复制一份完整副本长期驻留 HBM。

例如：

```python
self.all_token_ids = StagedWriteTensor(
    ...,
    uva_instead_of_gpu=True,
)
```

`all_token_ids` 的逻辑 shape 类似：

```text
[max_num_reqs, max_model_len]
```

如果：

```text
max_num_reqs = 256
max_model_len = 128K
```

那么：

```text
256 × 131072
≈ 33.5M entries
```

即便 token id 是 int32，也已经约 128 MiB。

更重要的是：

> GPU 每一轮并不需要把 256 个 request 的 128K token history 全量高速扫描。

它通常只需要当前 scheduled request 的某些 token range。

因此这种状态非常符合：

```text
长期存在
+
CPU 经常更新 / bookkeeping
+
GPU 只局部读取
```

的特征。

于是把它放在 Host/UVA-backed storage，比无条件长期占用 HBM 更合理。

---

## 10.3 UVA 不是“更快的 GPU memory”

这一点必须明确。

GPU 访问：

```text
HBM
```

通常远快于访问 Host memory。

所以不能得到：

```text
UVA 很方便
⇒ 所有状态都应该放 UVA
```

真正的 data placement 原则是：

```text
谁访问？
访问频率多高？
访问量多大？
CPU 是否频繁修改？
GPU 是否要求高带宽？
是否值得长期占 HBM？
```

例如：

```text
KV Cache
GPU 每层大量读取
→ 必须是 GPU/HBM 主数据

Q/K/V activation
GPU 高频计算
→ GPU/HBM

all_token_ids
长期存在、CPU 维护、GPU 局部读取
→ UVA 很合适

num_blocks
CPU bookkeeping 高频，数据量非常小
→ UVA 也很合适
```

因此 MRV2 的思想不是：

```text
大 tensor → UVA
小 tensor → GPU
```

而是：

> **根据访问模式进行 placement。**

---

## 10.4 为什么 `num_blocks` 这么小还要 `UvaBackedTensor`

源码中：

```python
self.num_blocks = UvaBackedTensor(
    (num_kv_cache_groups, max_num_reqs)
)
```

假设：

```text
1 KV cache group
256 requests
```

也就几百个整数。

显然这里不是为了省几 KB HBM。

真正原因是 CPU 经常要做：

```python
start = self.num_blocks.np[i, req_index]
```

例如：

```text
A 当前已有 3 个 blocks
本轮 Scheduler 又给了 [25, 31]

start = 3
↓
从 logical block column 3 开始 append
```

如果 authoritative `num_blocks` 只放 GPU，那么 CPU 每次读取：

```text
这个 request 当前有几个 block？
```

都可能导致：

```text
GPU → CPU synchronization
```

这是 serving runtime 非常忌讳的。

所以：

```text
num_blocks
= CPU frequently-managed
  + GPU-visible metadata
```

用 UVA-backed storage 很合理。

---

## 10.5 `StagedWriteTensor` 到底是什么

`stage_write()` 可以先简单理解成：

> **先记录“我要修改什么”，但不要每发生一个小修改就立刻触发一次独立 GPU-visible update。**

例如新 request A 加入，可能连续做：

```python
self.total_len.stage_write_elem(...)
self.all_token_ids.stage_write(...)
self.num_computed_tokens.stage_write_elem(...)
```

语义可以想成：

```text
Pending updates:

A.total_len = 1024
A.all_token_ids[0:1024] = [...]
A.num_computed_tokens = 0
```

然后又来 request B：

```text
B.total_len = 1536
B.all_token_ids[0:1536] = [...]
B.num_computed_tokens = 0
```

这些更新先进入 staged/pending 状态。

重点是：

```text
stage_write
≠ request 完成后才写

stage_write
= 当前 execution step 内先累积小更新
```

---

## 10.6 `apply_staged_writes()` 到底什么时候发生

这是最容易理解错的地方。

不是：

```text
Request start
↓
一直 stage 几百个 decode step
↓
Request finish
↓
才 apply
```

如果这样做，GPU 当然永远看不到最新状态。

正确的生命周期更接近：

```text
一个 GPUModelRunner.execute_model() step
─────────────────────────────────────

SchedulerOutput 到达

↓
finish / add / update requests

↓
产生很多 staged writes

A token state 修改
A computed 修改
B token state 修改
B length 修改
...

↓
apply_staged_writes()

↓
本 step 后续 prepare_inputs / forward
能够读取更新后的 state

↓
GPU execution
```

所以 staged write 的 batching boundary 主要是：

> **一次 execution step 中，把多个细碎的小更新集中提交。**

而不是 request lifetime。

---

## 10.7 为什么这样比“每改一次立即 copy”更好

serving 的一个关键特征是：

```text
request 数量很多
+
每个 request 每 step 只改一点点
```

比如 decode：

```text
A append 1 token
B append 1 token
C append 1 token
...
```

单个数据变化非常小。

但小 copy 的固定成本并不会变成 0：

```text
dispatch
enqueue
memory operation setup
synchronization bookkeeping
```

所以可能出现：

```text
1000 次 × 4 byte copy
```

比：

```text
一次批量提交
```

昂贵很多。

`StagedWriteTensor` 就是在解决：

> **metadata update 粒度太碎的问题。**

---

## 10.8 `num_computed_tokens_np` 又为什么存在

这类 CPU mirror 的目的也类似。

CPU 在 input preparation / bookkeeping 时经常需要知道：

```text
request 已经算到哪个 token position
```

如果每次都从 GPU tensor 读：

```text
CPU wants value
↓
GPU sync
↓
read scalar
```

那么代价可能远大于读这个 int 本身。

于是保留：

```text
GPU-visible state
+
CPU-side numpy mirror
```

让 CPU logic 可以直接读取本地状态。

因此：

```text
UVA
StagedWrite
CPU mirror
```

虽然实现形式不同，但它们都服务于同一个 production runtime 目标：

> **减少 CPU↔GPU metadata path 的同步、复制和 dispatch overhead。**

---

## 10.9 这一节最终应该形成的认知

不要记成：

```text
MRV2 为了省显存用了 UVA
```

这太窄。

更准确是：

```text
MRV2 persistent state
不是简单“全放 GPU”

而是根据访问模式拆分：

高带宽 GPU hot state
→ HBM

CPU 高频维护、GPU少量访问
→ UVA-backed

大量细碎更新
→ staged writes

CPU bookkeeping 常用值
→ CPU mirror
```

这说明 MRV2 的设计已经不只是：

```text
model execution
```

而是在认真优化：

```text
CPU metadata path
+
memory placement
+
GPU execution path
```

这也是 production inference runtime 与 toy runtime 的重要差异。

---

# 11. `BlockTables`：KV allocation decision 在 ModelRunner 的持久表示

Scheduler / KVCacheManager 决定：

```text
request logical block
→ physical KV block id
```

例如：

```text
Request A
logical block 0 → physical block 7
logical block 1 → physical block 18
logical block 2 → physical block 25
```

ModelRunner 需要把这份 mapping 保存下来。

这就是 persistent `BlockTables`。

可以理解成二维表：

```text
persistent row = req_idx
column = logical block index
value = physical block id
```

例如：

```text
req_idx 255(A): [7, 18, 25, ...]
req_idx 254(B): [4, 9, ...]
req_idx 253(C): [31, 32, ...]
```

这一步非常重要：

> Scheduler 管理的是 block ownership；ModelRunner 保存的是 GPU execution 所需的 block mapping representation。

---

# 12. New / resumed 与 continuing request 为什么分别 `overwrite` / `append`

`BlockTables.append_block_ids()` 里有两种典型模式。

## New / resumed

```text
overwrite=True
```

意味着：

```text
从 logical block 0 开始
重新建立当前 request 的完整 mapping
```

例如 resumed request：

```text
[7,18,25]
```

直接重新写入 ModelRunner persistent block table。

为什么？

因为 preemption 后 execution-side state 已经被删除，resume 是 fresh execution state。

---

## Continuing request

```text
overwrite=False
```

例如原来：

```text
[7,18]
```

Scheduler 本轮新分配：

```text
[25]
```

ModelRunner：

```text
append
↓
[7,18,25]
```

因此：

```text
SchedulerOutput 中的 new block ids
= current-step delta

ModelRunner BlockTables
= persistent full mapping
```

这也是“persistent state + delta update”的典型体现。

---

# 13. `prepare_inputs()`：从 Request 世界转向 Token 世界

前面还都在处理 request state。

真正准备 GPU 输入时，ModelRunner 必须把 request batch 变成 flattened token batch。

假设当前 SchedulerOutput：

```text
A → 本轮算 1 token
B → 本轮算 32 token
C → 本轮算 128 token
```

那么：

```text
num_scheduled_tokens = [1,32,128]
```

ModelRunner 最终需要构造类似：

```text
[A0,
 B0 ... B31,
 C0 ... C127]
```

这就是 flattened ragged token buffer。

---

## 13.1 `prepare_inputs()` 的真实核心不是“创建一个 batch”，而是建立两张坐标系

源码里最关键的几步是：

```python
num_tokens_per_req = scheduler_output.num_scheduled_tokens
req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)

idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
idx_mapping_np = np.fromiter(
    idx_mapping_iter,
    dtype=np.int32,
    count=num_reqs,
)

idx_mapping = async_copy_to_gpu(
    idx_mapping_np,
    device=self.device,
)

np.cumsum(
    num_scheduled_tokens,
    out=query_start_loc_np[1:num_reqs + 1],
)
```

这段其实同时建立两张不同的坐标系：

```text
Request coordinate system
-------------------------
current batch row
↓
idx_mapping
↓
persistent req_idx

Token coordinate system
-----------------------
current batch row
↓
query_start_loc
↓
flattened token range
```

所以 `prepare_inputs()` 的意义不是简单：

```text
list[request]
→ tensor
```

而是：

> **把 request-level scheduler decision 映射成 GPU 可以消费的 flattened token coordinate system。**

---

## 13.2 为什么需要两张 mapping，而不是一张

假设：

```text
Persistent:
req_idx 255 = A
req_idx 254 = B
req_idx 253 = C
req_idx 252 = D
```

当前 Scheduler：

```text
A → 1 token
D → 32 tokens
C → 128 tokens
```

那么第一张 mapping：

```text
idx_mapping
= [255,252,253]
```

回答：

```text
当前 batch row 对应 persistent state 的哪一行？
```

第二张 mapping：

```text
query_start_loc
= [0,1,33,161]
```

回答：

```text
当前 batch row 对应 flattened token buffer 的哪一段？
```

因此：

```text
batch row 1
```

同时可以解释成：

```text
persistent state:
idx_mapping[1] = 252 → D

flattened tokens:
query_start_loc[1:3] = [1,33]
→ query[1:33] 属于 D
```

这两个 mapping 联合起来，ModelRunner 才能把：

```text
stable request state
```

和：

```text
temporary flattened GPU tokens
```

接在一起。

---

## 13.3 `prepare_prefill_inputs()` 在这条链中的真实位置

源码会把类似这些 persistent state 传入：

```text
next_prefill_tokens
all_token_ids
prefill_len
num_computed_tokens
idx_mapping
query_start_loc
```

所以它面对的不是：

```text
“整个 batch 是不是 Prefill？”
```

而是：

```text
“对于当前 flattened batch 中，
哪些 token range 需要从 persistent prompt/token history 中填出来？”
```

可以把它想成一个 scatter/gather input materialization helper：

```text
Persistent Request State
all_token_ids
prefill boundary
computed frontier
        │
        │ idx_mapping
        │ query_start_loc
        ▼
Current input_ids buffer
```

它只负责把“应该进入本轮模型 forward 的 token id”真正填入连续 GPU input buffer。

---

## 13.4 为什么不是直接把 `all_token_ids` 整行送给模型

因为 `all_token_ids` 保存的是：

```text
整个 request 已知 token history
```

而本轮 forward 只执行：

```text
num_scheduled_tokens
```

那一小段。

例如：

```text
A:
all_token_ids 长度 = 8192
本轮 schedule = 1

B:
all_token_ids 长度 = 4096
本轮 schedule = 128
```

真正 GPU forward 需要的是：

```text
A 当前 1 token
+
B 当前 128 tokens
```

而不是：

```text
8192 + 4096
```

全部重新输入。

所以：

```text
persistent token history
≠
current model input
```

`prepare_inputs()` 就是在做这个切片与 materialization。

---

## 13.5 `positions` 又是怎么来的

input token id 只告诉模型：

```text
“这个 token 是谁”
```

但 Attention / RoPE 等还需要：

```text
“这个 token 在 sequence 的什么位置”
```

所以对于 A：

```text
num_computed_tokens = 100
schedule = 1
```

当前 position：

```text
100
```

对于 B：

```text
num_computed_tokens = 256
schedule = 4
```

当前 positions：

```text
256,257,258,259
```

flatten 后：

```text
positions =
[100, 256,257,258,259]
```

这张 token-level position table 后面会继续参与：

```text
slot_mapping
RoPE / positional computation
Attention metadata
```

所以 `positions` 是连接：

```text
execution frontier
```

和：

```text
token physical KV addressing
```

的重要桥梁。

---

## 13.6 `seq_lens` 为什么和 `positions` 看起来相似却完全不同

例如 B：

```text
本轮计算 positions
= [256,257,258,259]
```

如果这 4 个 token 写入后都应该被当前 causal attention 看见，那么：

```text
seq_len
≈ 260
```

所以：

```text
positions
= token-level coordinates

seq_lens
= request-level visible KV frontier
```

后面：

```text
slot_mapping
```

使用 position 定位“每个 token 写哪”。

而：

```text
FlashAttention
```

使用 seq_len 判断“整个 request 读多少历史 KV”。

这再次体现：

```text
token-level write semantics
vs
request-level read semantics
```

# 14. `idx_mapping`：连接 current batch 与 persistent state

`prepare_inputs()` 先根据当前 request order：

```text
req_ids = [...]
```

查：

```text
req_states.req_id_to_index
```

得到：

```text
idx_mapping
```

例如：

```text
Persistent:
255=A
254=B
253=C
252=D

Current step:
[A,D,C]

idx_mapping = [255,252,253]
```

这张 mapping 是 MRV2 整个 gather model 的关键。

后续很多数据都可以根据：

```text
current batch row
↓
idx_mapping
↓
persistent req_idx
```

拿到。

---

# 15. `query_start_loc`：ragged query batch 的边界表

假设：

```text
A 本轮 1 token
D 本轮 32 token
C 本轮 128 token
```

那么：

```text
num_scheduled_tokens = [1,32,128]
```

前缀和：

```text
query_start_loc = [0,1,33,161]
```

语义：

```text
query[0:1]    → A
query[1:33]   → D
query[33:161] → C
```

这解决了一个问题：

> GPU 不需要为不同长度 request 保留一个规则二维矩阵，而可以把 token flatten 后，用 prefix offsets 描述边界。

这就是 varlen / ragged batch 的核心 representation。

---

# 16. `positions` 与 `seq_lens`

两个字段容易混。

## `positions`

粒度：

```text
每一个当前 scheduled token
```

表示该 token 在自己 sequence 中的逻辑 position。

例如：

```text
A 当前计算 position 20
B 当前计算 position 100~131
```

那么 positions 是 flattened token 级数组。

---

## `seq_lens`

粒度：

```text
每一个 request
```

表示当前 request Attention 能看到的 sequence / KV 长度。

例如：

```text
A seq_len = 21
B seq_len = 132
```

后面 Attention kernel 通过它知道：

```text
每个 request 应该读取多少有效 KV token
```

---

# 17. `InputBatch` 的真正含义

因此 MRV2 的 `InputBatch` 不应该被理解成一个长期对象。

它是：

> **当前 step execution view。**

核心内容包括：

```text
idx_mapping
num_scheduled_tokens
query_start_loc
positions
seq_lens
is_prefilling
其他当前 step metadata
```

所以：

```text
Persistent Request State
        +
SchedulerOutput
        ↓
prepare_inputs()
        ↓
Per-step InputBatch
```

这就是 MRV2 的第二个关键转换。

---

# 18. `prepare_attn()`：从 token layout 转向 KV addressing

有了当前 step token layout 后，下一步是解决：

> 当前这些 token 到底对应 KV Cache 的哪些物理位置？

`GPUModelRunner.prepare_attn()` 做两件核心工作：

```text
1. gather current block tables
2. compute slot mappings
```

因此输出：

```text
block_tables
slot_mappings
```

这两个东西一定要严格区分。

---

# 19. `BlockTables` 的三层表示：persistent layout、current batch layout、token-level slot

这一部分是整个 MRV2 最容易“看代码都认识，但脑子里还是混”的地方。

原因是源码里同时出现：

```text
self.block_tables
input_block_tables
slot_mappings
```

它们都和 KV 地址有关，但粒度完全不同。

先给最终结论：

```text
Persistent block table
= req_idx × logical_block → physical_block

Current input block table
= batch_idx × logical_block → physical_block

Slot mapping
= current flattened token → physical KV slot
```

这三层不能混。

---

## 19.1 Persistent `self.block_tables`：按稳定 req_idx 存

假设 MRV2 persistent request slots：

```text
req_idx 255 → A
req_idx 254 → B
req_idx 253 → C
req_idx 252 → D
```

对应的 block ownership：

```text
A: [7,18,25]
B: [4,9]
C: [31,32]
D: [20,26]
```

那么 persistent table 可以想成：

```text
row 255: [7,18,25,...]
row 254: [4,9,...]
row 253: [31,32,...]
row 252: [20,26,...]
```

这里 row 的语义是：

```text
persistent req_idx
```

而不是：

```text
current batch row
```

所以它可以跨 step 保持稳定。

---

## 19.2 为什么不能让当前 Attention 直接使用 persistent table

假设这一轮 Scheduler 只选择：

```text
[A, D, C]
```

Current batch：

```text
batch_idx 0 → A
batch_idx 1 → D
batch_idx 2 → C
```

但是 persistent rows 是：

```text
A → 255
D → 252
C → 253
```

如果 kernel 直接使用 persistent table，它每一次读取都需要：

```text
batch_idx
↓
idx_mapping
↓
req_idx
↓
persistent block table row
```

这会把 ModelRunner 的 persistent-state layout 泄漏到 Attention backend。

而 backend 更希望看到：

```text
block_table[0]
就是 current request 0 的 block table

block_table[1]
就是 current request 1 的 block table
```

所以 ModelRunner 在 forward 前做一次：

```text
persistent layout
↓ gather
current batch layout
```

---

## 19.3 `gather_block_tables()` 到底做什么

它不 gather KV data。

这一点一定要明确：

```text
gather_block_tables()
≠ 把 physical KV blocks 拼成连续 KV tensor
```

它只是 gather **地址表的 rows**。

例如：

```text
Persistent:

255 A → [7,18,25]
254 B → [4,9]
253 C → [31,32]
252 D → [20,26]
```

当前：

```text
req_ids = [A,D,C]
idx_mapping = [255,252,253]
```

于是：

```text
gather_block_tables(idx_mapping)
```

生成：

```text
input_block_tables

batch row 0(A): [7,18,25]
batch row 1(D): [20,26]
batch row 2(C): [31,32]
```

可以把它理解为 SQL：

```text
persistent block table
= database table indexed by req_idx

gather_block_tables
= SELECT rows 255,252,253
  AND reorder as [A,D,C]
```

最终得到 backend 更容易消费的 dense batch representation。

所以直觉里的：

```text
“拼成一个 batch、拼成好读取的模式”
```

基本对。

但应该精确成：

> **拼的是 block-address metadata，不是 K/V 数据本身。**

---

## 19.4 为什么这个 gather 很符合 MRV2 的设计

MRV2 最核心的思想就是：

```text
persistent storage layout
不要随着 dynamic batching 搬来搬去
```

因此：

```text
A/B/C/D 在 req_states 中不搬

每一轮只建立：
idx_mapping
+
gathered execution view
```

如果不这样，而是每轮把 persistent table 本身 compact 成 batch layout：

```text
finish B
↓
搬 D
搬 C
搬各种 metadata
```

就重新回到了 MRV1 更容易出现的 row-management 问题。

所以：

```text
gather_block_tables
```

实际上是 MRV2：

```text
stable persistent state
+
ephemeral execution view
```

这一架构思想在 KV addressing 上的具体体现。

---

## 19.5 `input_block_tables` 为什么是 per-step state

它只对当前 step 的 batch order 有意义。

例如：

```text
Step t:
[A,D,C]

input block table:
row0=A
row1=D
row2=C
```

下一轮：

```text
Step t+1:
[D,E,A]
```

那么应该重新得到：

```text
row0=D
row1=E
row2=A
```

所以它不能被当成 request 的长期 ownership truth。

长期 truth 仍然是：

```text
persistent BlockTables
```

---

## 19.6 `slot_mapping` 为什么又不直接使用 gathered table

这一点很有意思。

KV read 路径：

```text
persistent BlockTables
↓ gather_block_tables()
↓
current input block table
↓
FlashAttention
```

但 KV write 的 `compute_slot_mappings()` 可以直接：

```text
idx_mapping
+
positions
+
persistent block table
↓
slot_mapping
```

原因是两个 consumer 的粒度不同。

Attention read 想要：

```text
“当前 batch request 0 的整个 logical sequence block mapping”
```

所以适合 batch-row table。

KV write 想要：

```text
“当前 flattened token i 最终写哪个 slot”
```

一旦 slot 算出来，后面根本不需要 request row 结构了。

所以：

```text
                            Persistent BlockTables
                              /               \
                             /                 \
                            ▼                   ▼
             gather_block_tables()      compute_slot_mappings()
                    │                          │
                    ▼                          ▼
        current batch block table          slot_mapping
                    │                          │
                    ▼                          ▼
                KV READ                    KV WRITE
```

这说明：

> 同一份 persistent block ownership 会被 ModelRunner 转换成不同 consumer 最合适的 execution representation。

---

## 19.7 `num_blocks` 为什么必须同时存在

Persistent block table 通常预分配最大宽度：

```text
[max_num_reqs, max_num_blocks]
```

但每个 request 当前只有前 N 个 column 有效。

例如：

```text
row A:
[7,18,25, ?, ?, ?, ...]
```

必须知道：

```text
有效长度 = 3
```

所以 `num_blocks[req_idx]` 用来追踪：

```text
这个 request 当前 block table 有多少有效项
```

当 continuing request 新增：

```text
[31,32]
```

就知道从：

```text
column 3
```

开始 append：

```text
[7,18,25,31,32]
```

这也是为什么 `num_blocks` 是 persistent bookkeeping state，而 `input_block_tables` 是 per-step execution view。

---

## 19.8 New / resumed 和 continuing 为什么一个 overwrite、一个 append

这也可以从 persistent lifecycle 理解。

### New / resumed

ModelRunner 侧 execution state 是 fresh 的：

```text
req_idx 新分配 / 重建
```

因此 Scheduler 给过来的 block IDs 应该从 logical block 0 重建：

```text
overwrite=True
```

例如：

```text
[7,18,25]
```

直接成为：

```text
logical 0→7
logical 1→18
logical 2→25
```

### Continuing

已有：

```text
[7,18]
```

本轮只新增：

```text
[25]
```

所以：

```text
overwrite=False
```

变成：

```text
[7,18,25]
```

这再次说明：

```text
SchedulerOutput
偏 current-step delta

ModelRunner persistent BlockTables
保存 full execution mapping
```

---

## 19.9 这一节最终应该形成的图

```text
Scheduler / KVCacheManager
physical block ownership
        │
        ▼
SchedulerOutput block ids
        │
        ▼
Persistent BlockTables
(req_idx × logical block)
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
idx_mapping                       positions
        │                              │
        ▼                              │
gather_block_tables()                  │
        │                              │
        ▼                              │
current batch block table              │
(batch_idx × logical block)            │
        │                              ▼
        │                      compute_slot_mappings()
        │                              │
        ▼                              ▼
    KV READ                    current token → slot
                                       │
                                       ▼
                                   KV WRITE
```

这里最重要的认知不是记函数名，而是：

> **同一份 persistent KV ownership，根据 consumer 不同，被转换成 sequence-level read representation 和 token-level write representation。**

---

# 20. `slot_mapping` 到底怎么计算

当前主路径 CP_SIZE=1 时，核心公式非常直接。

对于当前 token：

```text
position
↓
logical_block = position // block_size
block_offset  = position % block_size
↓
physical_block = block_table[req_idx][logical_block]
↓
physical_slot = physical_block * block_size + block_offset
```

因此地址链是：

```text
current scheduled token
↓
batch_idx
↓ idx_mapping
persistent req_idx
↓
sequence position
↓
logical block
↓ persistent block table
physical block
↓
physical slot
```

---

# 21. 一个完整的 slot_mapping 例子

假设：

```text
block_size = 16
Request A
position = 20
```

那么：

```text
logical_block = 20 // 16 = 1
block_offset  = 20 % 16 = 4
```

Persistent block table：

```text
A logical block 0 → physical block 7
A logical block 1 → physical block 18
```

所以：

```text
physical_block = 18
```

最后：

```text
slot = 18 * 16 + 4
     = 292
```

得到：

```text
slot_mapping[token_i] = 292
```

注意：

```text
292
```

不是 CUDA pointer。

它是：

> **Paged KV Cache flattened token-slot index。**

---

# 22. 为什么 block ID 不是 GPU 地址

Scheduler 分配的：

```text
KVCacheBlock(block_id)
```

属于 control-plane physical block identifier。

例如：

```text
block_id = 18
```

它的语义是：

```text
KV Cache block pool 中第 18 个 physical block
```

而不是：

```text
0x7f.... GPU virtual address
```

真正访问 GPU tensor 时，需要结合：

```text
kv_cache base tensor
block id
block offset
head index
head dimension
stride/layout
```

才能形成底层 memory address。

因此应保持三层概念：

```text
physical block id
↓
physical slot id
↓
actual tensor memory address
```

---

# 23. 为什么还要 `kernel_block_size`

Control-plane 的 KV block abstraction 与 Attention kernel 能接受的 block layout 不一定完全一致。

因此初始化 Attention backend 时会调用：

```text
prepare_kernel_block_sizes()
```

它的目标是：

> 在一个 KV cache group 中找到所有 backend 都支持的 kernel block size。

FlashAttention 当前明确要求：

```text
kernel block size = MultipleOf(16)
```

所以存在：

```text
KV control-plane block abstraction
        ↓
backend/kernel block compatibility
        ↓
kernel_block_size
```

这是典型的 control plane abstraction 与 kernel layout requirement 之间的适配层。

---

# 24. `DefaultModelState.prepare_attn()`：为什么 ModelRunner 后面还有 ModelState

`GPUModelRunner.prepare_attn()` 做的是通用 KV addressing：

```text
block_tables
slot_mappings
```

但不同模型 / Attention backend 还需要不同 metadata。

因此下一层：

```text
DefaultModelState.prepare_attn()
```

负责组织：

```text
query_start_loc
seq_lens
max_query_len
max_seq_len
block_tables
slot_mappings
positions
is_prefilling
...
```

然后交给：

```text
build_attn_metadata()
```

所以职责边界：

```text
GPUModelRunner.prepare_attn()
= generic KV addressing

ModelState.prepare_attn()
= model/backend metadata orchestration
```

---

# 25. `CommonAttentionMetadata`：ModelRunner 的通用 Attention 语言

`build_attn_metadata()` 首先构造：

```text
CommonAttentionMetadata
```

其字段可以分成四组。

## Request / token boundary

```text
num_reqs
num_actual_tokens
query_start_loc
max_query_len
```

## Sequence state

```text
seq_lens
max_seq_len
positions
is_prefilling
```

## KV addressing

```text
block_table_tensor
slot_mapping
kv_cache_config
```

## Attention semantics

```text
causal
sliding-window related state
其他 model-specific state
```

所以它的角色是：

> **Backend-independent Attention execution description。**

---

# 26. 为什么不能直接把 CommonAttentionMetadata 给所有 backend

因为不同 backend 需要不同数据布局 / planner / workspace / metadata object。

概念上：

```text
FlashAttention
需要自己的 metadata

FlashInfer
需要自己的 metadata

Triton Attention
也可能需要另一套 metadata
```

因此 vLLM 设计：

```text
CommonAttentionMetadata
↓
Backend-specific MetadataBuilder
↓
Backend-specific AttentionMetadata
```

这相当于：

```text
ModelRunner 通用语言
↓
translator
↓
Backend 方言
```

---

# 27. `AttentionGroup`：Layer / KV Spec / Backend 的绑定对象

`AttentionGroup` 核心字段：

```text
backend
layer_names
kv_cache_spec
kv_cache_group_id
metadata_builders
```

它本质上是在表达：

> **哪些 layer 可以共享同一种 Attention Backend / KV Cache 语义 / metadata 构造规则。**

分组 key：

```text
backend.full_cls_name()
+
layer_kv_cache_spec
+
num_heads_q
```

所以只有：

```text
Backend 相同
KV Cache Spec 相同
Q-head 数相同
```

的 layer 才放进同一个 AttentionGroup。

---

# 28. 为什么要分 AttentionGroup

如果所有层都强行共享一个 metadata builder，会出现问题：

```text
不同 attention type
不同 KV layout
不同 sliding-window/full-attn
不同 backend
不同 Q-head configuration
```

都可能需要不同 metadata。

因此 `AttentionGroup` 把变化维度隔离：

```text
Model layers
↓ group by compatible execution semantics
AttentionGroup
↓
Backend-specific metadata builder
```

这样模型代码无需知道：

```text
FlashAttention metadata 长什么样
FlashInfer metadata 长什么样
```

这是 backend abstraction 的关键。

---

# 29. MetadataBuilder 为什么是持久对象，而不是每 step new 一个

`AttentionGroup.create_metadata_builders()` 在初始化阶段创建 builder。

运行时：

```text
get_metadata_builder(0)
```

直接复用。

原因是 builder 可能自己持有：

```text
persistent GPU buffers
CUDA Graph-related buffers
workspace
planner state
```

如果每 step 都：

```text
Builder(...)
```

会带来：

```text
CPU allocation overhead
GPU buffer allocation
CUDA Graph state conflict
```

因此这里又体现 production runtime 的原则：

> 尽可能把稳定对象初始化一次，把 per-step path 压缩成更新和复用。

---

# 30. Attention Backend 是怎么被选出来的

这一段我们实际追踪了完整选择链。

起点是普通 Attention layer：

```text
Attention.__init__()
```

如果外部没有显式指定：

```text
attn_backend = None
```

则：

```text
get_attn_backend(...)
```

参数包括：

```text
head_size
dtype
kv_cache_dtype
use_mla
has_sink
attn_type
sliding_window
...
```

然后构造：

```text
AttentionSelectorConfig
```

再进入：

```text
current_platform.get_attn_backend_cls()
```

当前平台是：

```text
CUDAPlatform
```

---

# 31. CUDA backend selection 的两层逻辑

CUDA 平台的选择由两件事情组成：

## 31.1 Priority Policy

```text
_get_backend_priorities()
```

回答：

> 在当前硬件类别 / Attention 类型下，我们希望优先使用谁？

当前普通 non-MLA、A100 SM80 路径优先级：

```text
1. FLASH_ATTN
2. FLASHINFER
3. TRITON_ATTN
4. FLEX_ATTENTION
5. TURBOQUANT
```

---

## 31.2 Capability Validation

每个候选 backend：

```text
validate_configuration(...)
```

判断：

```text
GPU capability 是否支持
head size 是否支持
dtype 是否支持
KV cache dtype 是否支持
block size 是否支持
sliding window 等 feature 是否支持
依赖是否可 import
```

然后从合法候选中选 priority 最小的。

因此准确模型是：

```text
Priority Policy
+
Capability Filtering
↓
Final Backend
```

不是：

```text
A100 → 写死 FlashAttention
```

---

# 32. 为什么这个 backend selection 设计更合理

Production runtime 的硬件 / 模型组合太复杂：

```text
A100 / H100 / Blackwell
BF16 / FP16 / FP8 KV
MLA / MHA / GQA
full attention / sliding window
不同 block size
不同 dependency availability
```

如果写成：

```python
if A100:
    use FlashAttention
```

很快就不可维护。

而现在：

```text
Candidate priority
↓
Backend 自己声明 capability
↓
统一 validate
↓
选择合法的最高优先级实现
```

扩展性明显更好。

---

# 33. `FlashAttentionBackend` 自己到底是什么

`FlashAttentionBackend` 不是 kernel。

它更像：

> **Capability description + Factory / registry entry。**

它声明：

```text
supported_dtypes
supported_kv_cache_dtypes
supported kernel block sizes
sliding window support
non-causal support
attention type support
```

并暴露：

```text
get_builder_cls()
→ FlashAttentionMetadataBuilder

get_impl_cls()
→ FlashAttentionImpl
```

因此结构：

```text
FlashAttentionBackend
├─ capability
├─ MetadataBuilder factory
└─ Impl factory
```

---

# 34. FlashAttention KV Cache 的逻辑 shape

`FlashAttentionBackend.get_kv_cache_shape()`：

```text
(num_blocks,
 num_kv_heads,
 block_size,
 2 * head_size)
```

代码注释：

```text
logical (B, H, N, 2*D)
```

其中这里的 `B`：

> 是 blocks，不是 request batch。

以 Qwen3 为例：

```text
num_kv_heads = 8
head_size = 128
block_size = 16
```

则：

```text
[num_blocks, 8, 16, 256]
```

最后：

```text
256 = K 128 + V 128
```

即 K/V 在 content dimension 中打包。

---

# 35. Logical shape ≠ Actual memory layout

FlashAttentionBackend 还提供：

```text
get_kv_cache_stride_order()
```

说明：

```text
logical shape
(B,H,N,2D)
```

和：

```text
actual memory stride/layout
```

不是同一个概念。

这是 GPU programming 中必须建立的认知：

```text
Tensor shape
≠
physical memory order
```

Attention kernel 可能为了 memory coalescing / TMA / vectorization 对 stride 有额外要求。

当前阶段只需要知道这一层存在，不需要继续深入 layout kernel。

---

# 36. 一个关键发现：FlashAttention forward 不负责 KV write

`FlashAttentionBackend`：

```text
forward_includes_kv_cache_update = False
```

这个 flag 非常重要。

它意味着：

```text
FlashAttentionImpl.forward()
```

不会自己负责：

```text
把当前 K/V 写入 Paged KV Cache
```

因此 vLLM 把执行拆成：

```text
QKV projection
│
├─ K/V
│   ↓
│ KV Cache Update
│   ↓
│ Paged KV Cache
│
└─ Q
    ↓
Attention Compute
    ↑
    KV Cache
```

所以：

```text
KV write
≠
Attention compute
```

这是后续做 KV 量化 / compression / offload 非常重要的边界。

---

# 37. `forward_includes_kv_cache_update` 为什么设计成 backend capability

基类默认允许 backend 自己完成 KV update。

不同 backend 可以选择：

```text
模式 A：fused
backend.forward()
= KV update + Attention

模式 B：decoupled
framework KV update
↓
backend.forward() 只做 Attention
```

FlashAttention 当前走模式 B。

这样的好处是：

```text
不同 backend 可以按照自身 kernel 能力选择融合程度
```

而不是由上层强制所有 backend 采用同一种 execution pattern。

---

# 38. KV Write 的 Python 调用点：`unified_kv_cache_update()`

Attention layer forward 中，K/V reshape 后：

```text
key   → [num_tokens, num_kv_heads, head_size]
value → [num_tokens, num_kv_heads, head_size_v]
```

如果：

```text
backend.forward_includes_kv_cache_update == False
```

则调用：

```text
unified_kv_cache_update(key, value, layer_name)
```

这里最开始令人困惑的是：

```text
参数里为什么没有：
kv_cache
slot_mapping
block_table
```

答案在下一层。

---

# 39. `get_attention_context(layer_name)`：运行时上下文汇合点

`unified_kv_cache_update()` 内部：

```text
get_attention_context(layer_name)
```

从 `ForwardContext` 取出：

```text
attn_metadata
attn_layer
kv_cache
layer_slot_mapping
```

这说明：

```text
layer_name
```

在 runtime 中不仅是日志名字，而是一个 key。

它把当前 Attention layer 对应的：

```text
metadata
layer object
KV Cache tensor
slot mapping
```

关联起来。

因此：

```text
ModelRunner prepare metadata
↓
ForwardContext
↓
Attention layer 运行时按 layer_name 取回
```

形成一条隐式 context data path。

---

# 40. 为什么 `attn_metadata[layer_name]` 是 dict

前面 `build_attn_metadata()` 最终：

```text
attn_metadata[layer_name] = metadata
```

现在 `get_attention_context(layer_name)`：

```text
attn_metadata_raw[layer_name]
```

两头正好对上。

因此设计不是偶然：

```text
prepare 阶段
layer_name → metadata

forward 阶段
layer_name → metadata
```

这样多 Attention layer / 多 backend group 也可以共存。

---

# 41. `kv_cache_dummy_dep` 到底是什么

`unified_kv_cache_update()` 最后返回一个空 Tensor：

```text
torch.empty(0, ...)
```

并传给：

```text
unified_attention_with_output(...,
    kv_cache_dummy_dep=...)
```

它没有数学意义。

源码已经明确说明它用于：

```text
signal a side effect
+
建立 KV update → Attention 的 data dependency
+
保证 torch.compile 保留顺序
```

为什么必须这样？

因为真实依赖是：

```text
KV update
修改 shared KV cache memory
↓
Attention
读取 shared KV cache memory
```

这是一种 side-effect dependency。

编译器如果只看显式 Tensor dataflow，可能无法安全判断两者顺序。

所以 dummy dependency 相当于显式告诉 compiler：

```text
write KV first
↓
then Attention
```

这是非常典型的 compiler/runtime engineering 设计。

---

# 42. 真正 KV write 落到 `FlashAttentionImpl.do_kv_cache_update()`

`unified_kv_cache_update()`：

```text
attn_layer.impl.do_kv_cache_update(
    layer,
    key,
    value,
    kv_cache,
    layer_slot_mapping,
)
```

对于当前 backend：

```text
attn_layer.impl
= FlashAttentionImpl
```

因此：

```text
FlashAttentionImpl.do_kv_cache_update()
```

就是 backend runtime 层真正处理 KV write 的函数。

---

# 43. `do_kv_cache_update()` 如何把 packed KV Cache 拆成 K/V view

原始 logical cache：

```text
[num_blocks, num_kv_heads, block_size, 2D]
```

执行：

```text
transpose(1,2)
```

得到：

```text
[num_blocks, block_size, num_kv_heads, 2D]
```

再：

```text
split(head_size, dim=-1)
```

得到：

```text
key_cache
[num_blocks, block_size, num_kv_heads, D]

value_cache
[num_blocks, block_size, num_kv_heads, D]
```

例如 Qwen3：

```text
K Cache:
[num_blocks, 16, 8, 128]

V Cache:
[num_blocks, 16, 8, 128]
```

这里第一次把：

```text
control-plane physical block
```

与：

```text
GPU KV Cache tensor view
```

真正连接起来。

---

# 44. 真正 scatter write：`reshape_and_cache_flash()`

`do_kv_cache_update()` 最终调用：

```text
reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    ...
)
```

其参数可以按语义分：

```text
What to write:
key / value

Where to write:
key_cache / value_cache

Address:
slot_mapping

Storage format:
kv_cache_dtype / k_scale / v_scale
```

这就是 KV write path 的最核心接口。

---

# 45. `reshape_and_cache_flash()` 本身仍然只是 Python wrapper

它内部只是：

```text
torch.ops._C_cache_ops.reshape_and_cache_flash(...)
```

所以最终链：

```text
FlashAttentionImpl.do_kv_cache_update()
↓
reshape_and_cache_flash()
↓
torch.ops._C_cache_ops.reshape_and_cache_flash
↓
C++ / CUDA custom op
↓
GPU kernel
```

到这里如果当前目标是理解 vLLM architecture，就不必继续钻 CUDA kernel。

因为上层语义已经完全确定。

---

# 46. KV write kernel 的核心语义

虽然真实 CUDA kernel 会并行实现，但可以用下面伪代码理解：

```text
for token i:
    slot = slot_mapping[i]

    block  = slot // block_size
    offset = slot % block_size

    key_cache[block, offset, :, :]   = key[i]
    value_cache[block, offset, :, :] = value[i]
```

例如：

```text
slot_mapping[i] = 292
block_size = 16
```

则：

```text
block  = 292 // 16 = 18
offset = 292 % 16 = 4
```

最终：

```text
K_cache[18, 4, :, :] ← current K
V_cache[18, 4, :, :] ← current V
```

这就是 Scheduler 的 physical block decision 最终如何变成 HBM 中实际写入位置。

---

# 47. 为什么 KV write 不需要 block_table

这是一个非常值得留下来的设计结论。

`slot_mapping` 已经是：

```text
position + block_table
↓
预先算好的 token-level physical address index
```

所以真正 write 时不需要重新：

```text
position
→ logical block
→ block table lookup
→ physical block
```

而直接：

```text
slot_mapping
→ write destination
```

因此：

```text
Write path
= token → physical slot
```

而不是 sequence-level lookup。

---

# 48. KV Read Path：`FlashAttentionImpl.forward()`

写入结束以后，普通 decoder Attention 开始读取 KV Cache。

函数参数虽然包含：

```text
query
key
value
kv_cache
attn_metadata
```

但普通 decoder path 真正传给 FlashAttention 的是：

```text
q = query
k = key_cache
v = value_cache
```

而不是：

```text
k = current key
v = current value
```

这说明：

> 当前 token 的 K/V 已经先写入 KV Cache，Attention 随后直接从完整 KV Cache 中读取。

---

# 49. 为什么当前 token 自己也能从 KV Cache 中读到

因为顺序是：

```text
QKV projection
↓
new K/V
↓
KV cache update
↓
current K/V 已经进入 KV Cache
↓
Attention read
```

例如 decode token 20：

```text
写之前 KV Cache:
K0 ... K19

写 K20/V20 后:
K0 ... K20

然后：
Q20 × K0...K20
```

因此无需额外拼接：

```text
historical KV + current KV
```

因为完整 cache 已经包含 current KV。

`kv_cache_dummy_dep` 则保证 compiler 不能把 read 放到 write 前面。

---

# 50. FlashAttention read path 的关键参数

普通 `not use_cascade` 路径：

```text
cu_seqlens_q = attn_metadata.query_start_loc
seqused_k    = attn_metadata.seq_lens
max_seqlen_q = attn_metadata.max_query_len
max_seqlen_k = attn_metadata.max_seq_len
block_table  = attn_metadata.block_table
```

最终：

```text
flash_attn_varlen_func(
    q=query,
    k=key_cache,
    v=value_cache,
    cu_seqlens_q=...,
    seqused_k=...,
    block_table=...,
    ...
)
```

所以 read path 依赖：

```text
Q
+
KV Cache
+
block_table
+
seq_lens
+
ragged-query metadata
```

---

# 51. `block_table` 在 read path 中到底做什么

假设 Request A：

```text
logical block 0 → physical 7
logical block 1 → physical 18
logical block 2 → physical 25
```

于是：

```text
block_table[A] = [7,18,25]
```

Attention 要读：

```text
sequence position 0~15
→ logical block 0
→ physical block 7

position 16~31
→ logical block 1
→ physical block 18

position 32~...
→ logical block 2
→ physical block 25
```

所以一个 request 的 KV 完全不需要在 HBM 中连续。

这是 Paged KV 的本质：

```text
logical sequence contiguous
↓
physical KV storage non-contiguous
↓
block_table 负责翻译
```

---

# 52. `seq_lens` 为什么同样必不可少

只有 block table 还不够。

例如：

```text
block_size = 16
A 有两个 block
```

并不意味着 A 一定有：

```text
32 valid KV tokens
```

可能只有：

```text
21 tokens
```

所以：

```text
block_table
告诉 kernel：去哪些 physical blocks

seq_lens
告诉 kernel：其中有多少 token 真正有效
```

二者共同描述：

```text
logical sequence → physical KV data
```

---

# 53. `query_start_loc` 在 Attention read path 的意义

当前 batch flatten：

```text
A: 1 token
B: 32 token
C: 4 token
```

则：

```text
query_start_loc = [0,1,33,37]
```

FlashAttention 由此知道：

```text
query[0:1]   → A
query[1:33]  → B
query[33:37] → C
```

因此：

```text
query_start_loc
= flattened Q 中每个 request 的 boundary
```

这是 varlen attention 能同时处理不同 query length 的基础。

---

# 54. `max_query_len` / `max_seq_len` 的角色

这两个不是主要寻址结构。

```text
max_query_len
= 当前 batch 中最大的 per-request query length

max_seq_len
= 当前 batch 中最大的 sequence / KV length
```

它们更多用于：

```text
kernel planning
launch configuration
workspace sizing
backend specialization
```

真正每个 request 的边界仍由：

```text
query_start_loc
seq_lens
```

描述。

---

# 55. 最重要的区别：Write 用 slot_mapping，Read 用 block_table

这是本阶段必须牢牢记住的一句话。

## Write 的问题

```text
“当前这个 token 的 K/V 应该写到哪里？”
```

答案：

```text
slot_mapping[token]
```

因此：

```text
WRITE
current token
↓
physical slot
```

---

## Read 的问题

```text
“这个 request 的整个历史 sequence 在哪些 physical blocks？”
```

答案：

```text
block_table[request]
+
seq_lens[request]
```

因此：

```text
READ
logical sequence
↓
physical blocks
```

最终压缩：

```text
WRITE:
token → slot_mapping → KV Cache

READ:
request sequence → block_table + seq_lens → KV Cache
```

---

# 56. Prefill / Decode 为什么在这里看起来“不分了”：统一的是 execution representation，不是 workload 本身

这一点必须建立非常精确的认知。

nano-vLLM 中常见的是：

```python
if context.is_prefill:
    flash_attn_varlen_func(...)
else:
    flash_attn_with_kvcache(...)
```

所以很容易形成：

```text
Prefill
= 一套执行模式

Decode
= 另一套执行模式
```

这是合理的教学抽象，因为两种 workload 的确不同。

但是当前 vLLM V1 + MRV2 + FlashAttention 路径里，普通 decoder 最终统一进入：

```text
flash_attn_varlen_func(...)
```

这里发生的不是：

```text
Prefill / Decode 计算性质消失
```

而是：

> **上层不再要求先把整个 batch 分类成“Prefill batch”或“Decode batch”，而是把每个 request 本轮需要执行的 token 数编码进统一 ragged representation。**

---

## 56.1 Prefill 和 Decode 的数学 workload 仍然不同

典型 Prefill：

```text
query_len = 128 / 512 / 1024 ...
KV length 也在快速增长

特点：
many queries × many keys
```

典型 Decode：

```text
query_len ≈ 1
KV length = 1K / 8K / 32K ...

特点：
very few queries × long historical KV
```

这两个 workload：

```text
算术强度
memory bandwidth 压力
parallelism
最佳 tile / split 策略
```

都可能不同。

所以绝不能记成：

```text
vLLM V1 认为 Prefill 和 Decode 一样
```

---

## 56.2 V1 Scheduler 为什么不愿意强类型区分

真实 continuous batching 可能同一 step：

```text
A → Decode            1 token
B → Chunked Prefill 128 tokens
C → Decode            1 token
D → Chunked Prefill  64 tokens
```

如果顶层强制：

```text
if Prefill batch:
...
else Decode batch:
...
```

那么这个 batch 根本无法归类。

V1 更自然地表示：

```text
num_scheduled_tokens
= [1,128,1,64]
```

然后 flatten：

```text
query_start_loc
= [0,1,129,130,194]
```

所以每个 request 自己的 query length 已经编码在：

```text
query_start_loc[i+1] - query_start_loc[i]
```

里。

---

## 56.3 Prefill / Decode 差异如何被 metadata 编码

### Decode request A

```text
q_len = 1
seq_len = 4096
```

### Chunked-prefill request B

```text
q_len = 128
seq_len = previous_len + 128
```

最终 backend 拿到：

```text
query_start_loc
seq_lens
max_query_len
max_seq_len
block_table
```

它不需要额外一个：

```text
batch_mode = PREFILL / DECODE
```

才能知道 workload 形状。

因此：

```text
Mode flag
```

被更一般的：

```text
shape + offsets + lengths
```

取代。

---

## 56.4 为什么这和 `prepare_prefill_inputs()` 不矛盾

这里必须再次区分两个层次。

### Scheduler / Attention execution representation

统一成：

```text
每 request 本轮 N tokens
```

### Input materialization

仍然必须知道：

```text
这些 token id 从哪里来
```

Prompt 尚未消费完：

```text
从 all_token_ids 的 prefill range 中拿
```

Generated continuation：

```text
从上一轮 sampled/runtime token state 中拿
```

所以：

```text
prepare_prefill_inputs()
```

仍然合理。

它解决的是：

```text
token source semantics
```

不是：

```text
top-level batch mode
```

---

## 56.5 为什么统一 interface 之后底层仍然可以 specialization

统一：

```text
flash_attn_varlen_func(...)
```

只说明：

```text
Python/backend API boundary
```

被统一。

并不能推出：

```text
所有输入最后执行完全同一个 CUDA kernel configuration
```

底层完全可以根据：

```text
max_query_len
seq_len
num_heads / num_kv_heads
paged block table
FA version
hardware
scheduler metadata
```

选择不同 execution specialization。

因此准确架构是：

```text
Scheduler
↓
unified token scheduling

ModelRunner
↓
unified ragged representation

Attention Backend API
↓
unified varlen/paged interface

Kernel / dispatch layer
↓
workload-specific specialization
```

这叫：

> **把 specialization 下沉到更适合做 specialization 的层级。**

---

## 56.6 nano-vLLM 与 vLLM V1 的设计差异

nano-vLLM 更像：

```text
                 Attention caller
                      │
                explicit mode
               /             \
              ▼               ▼
         Prefill            Decode
              │               │
         varlen API      kvcache API
```

优点：

```text
直观
容易学习
容易看出 workload 差异
```

vLLM V1 更像：

```text
Scheduler
↓
per-request token counts
↓
mixed ragged batch
↓
common attention representation
↓
backend/kernel specialization
```

优点：

```text
更适合 continuous batching
更容易混合 chunked prefill + decode
减少上层 mode branching
backend 可以集中做 dispatch / specialization
```

所以不是谁“更正确”，而是抽象目标不同。

nano 更强调教学和简洁。

vLLM 更强调：

```text
production mixed workload
+
backend extensibility
+
continuous batching
```

---

## 56.7 一句话钉死

以后看到：

```text
prepare_prefill_inputs
is_prefilling
num_computed_prefill_tokens
```

不要觉得这和“V1 统一 Prefill/Decode”冲突。

正确理解是：

> **V1 统一的是调度和执行表示，不是抹掉 Prompt/Generated token 的语义，也不是消灭 Prefill/Decode 的 workload 差异。**

---

# 57. Prefill 如何被 metadata 表达

假设：

```text
A 本轮 chunked prefill 128 token
B 本轮 chunked prefill 64 token
```

则：

```text
num_scheduled_tokens = [128,64]
query_start_loc = [0,128,192]
max_query_len = 128
seq_lens = [A_current_len, B_current_len]
```

本质是：

```text
q_len > 1
```

的 ragged workload。

---

# 58. Decode 如何被 metadata 表达

假设三个 decode request：

```text
A → 1 token
B → 1 token
C → 1 token
```

则：

```text
num_scheduled_tokens = [1,1,1]
query_start_loc = [0,1,2,3]
max_query_len = 1
seq_lens = [1025,2050,513]
```

这就是典型：

```text
Q 很短
KV 很长
```

的 decode workload。

所以 Prefill / Decode 的区别仍然存在，只是转化成：

```text
shape + metadata semantics
```

---

# 59. 为什么 production vLLM 更适合统一成 ragged workload

真实 continuous batching 中，一轮可能是：

```text
A → Decode          1 token
B → Chunked Prefill 128 token
C → Decode          1 token
D → Chunked Prefill 64 token
```

这时候整个 batch 很难简单定义成：

```text
Prefill batch
```

或：

```text
Decode batch
```

但统一表示非常自然：

```text
num_scheduled_tokens = [1,128,1,64]
query_start_loc = [0,1,129,130,194]
```

所以 vLLM V1 更倾向表达：

> **每个 request 本轮执行多少 token。**

而不是：

> **整个 batch 是 Prefill 还是 Decode。**

这与 Scheduler 的统一 token scheduling 思想完全一致。

---

# 60. API 统一不等于 kernel 完全相同

需要特别避免一个错误结论：

```text
都调用 flash_attn_varlen_func
⇒ Prefill/Decode 底层完全相同
```

这是错的。

准确理解：

```text
Python / backend interface
统一成 varlen + paged API

但底层仍可以根据：
q_len
seq_len
GQA shape
hardware
FA version
paged KV layout
scheduler metadata
```

进行 workload-specific specialization。

因此：

```text
nano-vLLM
= 上层显式 specialization

vLLM V1
= 上层统一 representation
  + 更底层 specialization
```

这是一种 abstraction boundary 的变化，而不是计算性质消失。

---

# 61. Scheduler 与 Attention 的统一思想其实是一致的

Scheduler 前面已经看到：

```text
不强制：
Prefill request / Decode request

而主要维护：
num_computed_tokens
num_tokens
num_scheduled_tokens
```

也就是：

```text
这一轮这个 request 还要算多少 token
```

进入 ModelRunner：

```text
num_scheduled_tokens
↓
query_start_loc
↓
ragged Q layout
```

进入 Attention：

```text
ragged Q
+
paged KV
↓
unified varlen attention interface
```

所以 V1 整体路线可以总结：

```text
Control plane:
Prefill / Decode
→ token scheduling problem

Data plane:
Prefill / Decode
→ query-length / sequence-length problem

Kernel plane:
→ workload-specific specialization
```

这是一个非常漂亮的统一设计。

---

# 62. 从 Scheduler 到 GPU KV Write 的完整闭环

现在可以正式画完整写链：

```text
Scheduler.schedule()
↓
决定 request 本轮 num_scheduled_tokens
↓
KVCacheManager.allocate_slots()
↓
SingleTypeKVCacheManager
↓
BlockPool
↓
physical block IDs
↓
SchedulerOutput
↓
MRV2 add/update requests
↓
persistent BlockTables
↓
prepare_inputs()
↓
positions / idx_mapping
↓
prepare_attn()
↓
position + persistent block table
↓
slot_mapping
↓
Model forward
↓
QKV projection
↓
current K/V
↓
unified_kv_cache_update()
↓
get_attention_context(layer_name)
↓
kv_cache + layer_slot_mapping
↓
FlashAttentionImpl.do_kv_cache_update()
↓
reshape_and_cache_flash()
↓
_C_cache_ops CUDA kernel
↓
GPU Paged KV Cache
```

这就是：

> **Scheduler control-plane block decision 如何真正成为 GPU HBM 中 K/V 的物理落点。**

---

# 63. 从 Scheduler 到 FlashAttention KV Read 的完整闭环

读链：

```text
Scheduler
↓
request block ownership / sequence progress
↓
SchedulerOutput
↓
ModelRunner persistent BlockTables
↓
current InputBatch
↓
gather current block_table
↓
CommonAttentionMetadata
↓
FlashAttentionMetadata
↓
FlashAttentionImpl.forward()
↓
Q
+
key_cache/value_cache
+
block_table
+
seq_lens
+
query_start_loc
↓
flash_attn_varlen_func()
↓
Paged KV read + Attention
↓
Attention Output
```

这里的核心是：

> **logical request sequence 不需要物理连续；block_table 把 logical sequence 翻译成 distributed physical KV blocks。**

---

# 64. 一个端到端具体例子

假设：

```text
block_size = 16
Request A 已知长度 = 21
当前本轮需要计算 position 20
```

Scheduler/KV control plane：

```text
A block table:
logical 0 → physical 7
logical 1 → physical 18
```

ModelRunner `prepare_attn()`：

```text
position = 20
logical_block = 1
offset = 4
physical_block = 18
slot = 18*16+4 = 292
```

KV Write：

```text
QKV projection
↓
K20 / V20
↓
slot_mapping = 292
↓
K_cache[18,4,:,:] = K20
V_cache[18,4,:,:] = V20
```

然后 Attention Read：

```text
Q20
+
block_table[A] = [7,18]
+
seq_len[A] = 21
↓
FlashAttention
```

kernel 从：

```text
physical block 7
读取 logical position 0~15

physical block 18
读取 logical position 16~20
```

最终：

```text
Q20 attends to K0...K20
```

这一个例子把：

```text
logical token position
physical block
physical slot
KV write
KV read
```

全部串起来。

---

# 65. 为什么 ModelRunner 不是“简单跑模型”

现在可以重新回答这个问题。

如果 ModelRunner 只是：

```text
model(input_ids)
```

那么 model 自己就必须知道：

```text
Scheduler request state
block allocation
continuous batching
preemption
request reorder
paged KV
backend metadata
CUDA Graph padding
```

这样模型代码会与 runtime 强耦合。

现在 vLLM 把这些职责放到 ModelRunner：

```text
Scheduler decision
↓
ModelRunner translation
↓
backend-neutral model execution representation
↓
Attention backend
```

好处：

```text
model architecture
和
serving runtime scheduling
和
attention kernel backend
```

可以相对独立演化。

---

# 66. 为什么 ModelRunner 需要维护 persistent state，而不是每 step 全从 Scheduler 重建

每轮全部重建当然逻辑简单，但成本很高：

```text
token state copy
block table rebuild
GPU metadata copy
sampler state rebuild
request mapping rebuild
```

Serving 每个 decode step 都非常短，CPU overhead 很容易吞掉 GPU kernel 优化收益。

因此 MRV2 选择：

```text
persistent execution state
+
small per-step delta
+
per-step gathered view
```

这是典型 production inference runtime 设计。

---

# 67. 为什么 Scheduler 与 ModelRunner 都要保存“看起来相似”的状态

这不是简单重复。

两者语义不同。

Scheduler：

```text
authoritative logical state
```

关心：

```text
request lifecycle
priority
num_computed_tokens
KV ownership
admission / preemption
```

ModelRunner：

```text
execution representation
```

关心：

```text
GPU-visible token state
req_idx
block table tensor
input buffer
metadata
```

所以是：

```text
Control-plane state
↓ synchronization
Execution-side state
```

不是两个地方互相不知道谁是真的。

---

# 68. 为什么 block table 要同时存在 control plane 与 execution side

Scheduler/KVCacheManager 中：

```text
req_to_blocks
```

用于：

```text
allocation
free
prefix cache ownership
preemption
resource accounting
```

ModelRunner 中：

```text
BlockTables
```

用于：

```text
GPU kernel addressing
```

两者虽然表达的 mapping 相似，但职责完全不同：

```text
Scheduler block state
= resource management truth

ModelRunner block table
= execution data structure
```

这是 control/data plane 分离的典型例子。

---

# 69. 为什么 `slot_mapping` 应该在 ModelRunner 计算，而不是 Scheduler

Scheduler 知道：

```text
request
block ownership
num scheduled tokens
```

但它不应该承担 GPU layout 细节：

```text
flattened token ordering
positions buffer
kernel block size
GPU batch padding
backend-specific input layout
```

`slot_mapping` 是：

```text
logical scheduling state
+
GPU execution layout
```

的交界产物。

所以放在 ModelRunner 更合理。

否则 Scheduler 会被迫知道过多 GPU execution detail。

---

# 70. 为什么 `AttentionGroup` 不放在 Scheduler

同理，Scheduler 不应该知道：

```text
FlashAttention
FlashInfer
Triton Attention
metadata builder
CUDA Graph workspace
```

这些都是 execution/backend concerns。

因此：

```text
Scheduler
→ 只产生通用 execution contract

ModelRunner / AttentionGroup
→ 将其绑定到具体 backend
```

这保持了 Scheduler 的 backend neutrality。

---

# 71. 为什么 KV write 与 Attention read 分离对 KV 项目很重要

当前 FlashAttention backend：

```text
KV write
= reshape_and_cache_flash

Attention read
= flash_attn_varlen_func
```

这意味着未来做 Quantized Paged KV Cache 时可以分别考虑：

```text
Write-side transformation
BF16 K/V
↓
quantize / pack
↓
low-bit KV Cache
```

和：

```text
Read-side transformation
low-bit paged KV
↓
dequant / fused attention
↓
Attention
```

这比把所有逻辑硬塞进一个 Attention kernel 更容易逐步演进。

例如项目可以分阶段：

```text
V0:
先改 KV write format

V1:
实现 gather + dequant

V2:
进一步 fused low-bit decode attention
```

因此这条源码主链与后续 KV 项目是直接相关的。

---

# 72. 当前最重要的对象所有权关系

建议用下面关系图记忆：

```text
EngineCore
├─ Scheduler
│   ├─ Request objects
│   ├─ waiting/running
│   └─ KVCacheManager
│       └─ BlockPool
│
└─ Executor
    └─ Worker
        └─ GPUModelRunner
            ├─ req_states
            ├─ BlockTables
            ├─ InputBuffers
            ├─ ModelState
            └─ Model / Attention layers
                ├─ AttentionBackend
                ├─ AttentionImpl
                └─ kv_cache tensor
```

这里必须保持：

```text
Scheduler owns scheduling truth
ModelRunner owns execution representation
Attention layer owns/accesses runtime KV tensor
Backend owns implementation policy/capability
Impl executes backend-specific runtime
```

---

# 73. 当前最重要的数据对象生命周期

## Cross-step persistent

```text
Scheduler Request
Scheduler KV ownership
ModelRunner RequestState
ModelRunner persistent BlockTables
Attention layer KV Cache Tensor
MetadataBuilder object
```

## Per-step

```text
SchedulerOutput
InputBatch
idx_mapping
query_start_loc
current input block tables
slot_mapping
CommonAttentionMetadata
FlashAttentionMetadata
Q/K/V
Attention output
```

这一区分非常重要。

很多源码看起来“重复”，实际上就是：

```text
persistent state
vs
current-step delta/view
```

不同。

---

# 74. 当前最容易混淆的几个概念

## `request_id`

Scheduler / user-level request identity。

## `req_idx`

ModelRunner persistent execution-state slot。

## `batch_idx`

当前 step InputBatch 中的 row index。

## `logical block index`

sequence 内第几个 KV block。

## `physical block id`

BlockPool 分配的 KV block identifier。

## `physical slot id`

`physical_block * block_size + offset`。

## actual GPU address

由 KV tensor base + stride/layout + indexes 最终产生，不能与 block/slot id 混淆。

---

# 75. 当前最容易混淆的三个“长度”

## `num_scheduled_tokens`

```text
当前 step 每个 request 实际计划计算多少 token
```

## `query_len`

在 Attention 中基本对应当前 step 该 request 的 query token 数。

## `seq_len`

当前 Attention 可见的完整 sequence / KV 长度。

例如 decode：

```text
query_len = 1
seq_len = 4096
```

例如 chunked prefill：

```text
query_len = 128
seq_len = previous_history + 128
```

---

# 76. 当前最重要的四个 mapping

```text
req_id → req_idx
```

连接 Scheduler identity 与 ModelRunner persistent state。

```text
batch_idx → req_idx
```

即 `idx_mapping`，连接 current step batch 与 persistent state。

```text
logical block → physical block
```

即 block table。

```text
current token → physical slot
```

即 slot mapping。

把这四张 mapping 理清，ModelRunner 大部分代码就不会再散。

---

# 77. 我们这次源码追踪实际是怎么走的

这部分记录追踪过程，而不是只记录最终答案。

最开始我们已有：

```text
EngineCore.step
→ Scheduler.schedule
→ model_executor.execute_model
→ Worker.execute_model
→ GPUModelRunner.execute_model
```

真正进入 MRV2 后，没有直接钻 model forward，而是先问：

> SchedulerOutput 到了 ModelRunner 后，persistent request state 怎么维护？

因此先追：

```text
finish_requests()
add_requests()
update_requests()
```

由此确认：

```text
new/resumed 重建 execution state
continuing request patch
preempted execution state 删除
```

---

# 78. 第二步追踪：为什么 MRV2 不需要频繁移动 batch row

然后看：

```text
RequestState
req_id_to_index
index_to_req_id
free_indices
```

再追到：

```text
prepare_inputs()
```

发现：

```text
current req_ids
↓
req_id_to_index
↓
idx_mapping
```

这一步明确了 MRV2 的核心设计：

```text
persistent req_idx layout
+
per-step gather view
```

从而解决了之前单纯看函数时“为什么又有 RequestState、又有 InputBatch”的困惑。

---

# 79. 第三步追踪：Scheduler 的 block IDs 怎么变成 GPU 可用地址

接下来追：

```text
BlockTables
```

先确认：

```text
append_block_ids(overwrite=True/False)
```

再看：

```text
gather_block_tables()
compute_slot_mappings()
```

由此建立：

```text
persistent block table
↓ gather
current input block table
```

以及：

```text
position
+
block table
↓
slot_mapping
```

这一步第一次把 Scheduler/KV 控制面与 GPU KV addressing 接起来。

---

# 80. 第四步追踪：为什么 prepare_attn 后面还有 ModelState

看到：

```text
GPUModelRunner.prepare_attn()
```

只返回：

```text
block_tables
slot_mappings
```

于是继续追：

```text
self.model_state.prepare_attn(...)
```

当前普通 decoder-only Qwen3 进入：

```text
DefaultModelState
```

继续得到：

```text
build_attn_metadata()
```

由此确认：

```text
Runner
= addressing preparation

ModelState
= model/backend metadata orchestration
```

---

# 81. 第五步追踪：`build_attn_metadata()` 到底返回什么

分析 `build_attn_metadata()` 后确认：

```text
for each KV cache group
↓
CommonAttentionMetadata
↓
for each AttentionGroup
↓
MetadataBuilder.build()
↓
metadata
↓
for layer_name in group.layer_names
attn_metadata[layer_name] = metadata
```

于是我们知道最终返回的是：

```text
layer_name → backend-specific metadata
```

而不是一个全局 metadata。

这也为后面 `get_attention_context(layer_name)` 埋下了伏笔。

---

# 82. 第六步追踪：AttentionGroup 怎么来的

搜索：

```text
AttentionGroup(
```

定位：

```text
init_attn_backend()
```

发现三阶段：

```text
Phase 1:
按 backend + KV spec + num_heads_q 分组

Phase 2:
协调 kernel block size

Phase 3:
创建 MetadataBuilder + 判断 CUDA Graph support
```

由此确定 `AttentionGroup` 不是运行时临时 group，而是初始化阶段建立的 backend binding structure。

---

# 83. 第七步追踪：Qwen3 到底用哪个 Backend

从：

```text
AttentionGroup
↓
attn_layers[layer_name].get_attn_backend()
```

继续追：

```text
Attention.get_attn_backend()
↓
return self.attn_backend
```

说明真正选择发生在 Attention 初始化。

然后追：

```text
self.attn_backend = get_attn_backend(...)
```

再进入：

```text
vllm/v1/attention/selector.py
```

最终：

```text
current_platform.get_attn_backend_cls()
```

当前就是 CUDAPlatform。

---

# 84. 第八步追踪：CUDAPlatform 怎么选择 backend

先看到：

```text
get_valid_backends()
```

再看到：

```text
_get_backend_priorities()
```

当前 A100 / non-MLA 普通路径：

```text
FLASH_ATTN
FLASHINFER
TRITON_ATTN
FLEX_ATTENTION
TURBOQUANT
```

随后逐个：

```text
validate_configuration()
```

因此没有直接武断地说：

```text
A100 必然 FlashAttention
```

而是得到更准确的结论：

```text
FLASH_ATTN 是默认最高优先级候选；
最终需通过 capability / import validation。
```

这体现了源码追踪中非常重要的方法：

> 不要从 candidate priority 直接跳到 runtime fact。

---

# 85. 第九步追踪：FlashAttentionBackend 的职责

定位：

```text
registry.py
FLASH_ATTN
→ vllm.v1.attention.backends.flash_attn.FlashAttentionBackend
```

再看 `FlashAttentionBackend`：

确认：

```text
get_builder_cls()
→ FlashAttentionMetadataBuilder

get_impl_cls()
→ FlashAttentionImpl
```

并看到：

```text
forward_includes_kv_cache_update = False
```

于是追踪方向从“直接看 FlashAttention forward”改成：

> 先找 KV write 到底在哪里。

这是一次非常重要的追踪路线调整。

---

# 86. 第十步追踪：KV write 到底在哪里

搜索：

```text
forward_includes_kv_cache_update
```

定位：

```text
vllm/model_executor/layers/attention/attention.py
```

看到：

```text
unified_kv_cache_update(key, value, layer_name)
```

于是问题变成：

```text
为什么这里没有 slot_mapping / kv_cache 参数？
```

继续追：

```text
get_attention_context(layer_name)
```

发现 runtime context 会返回：

```text
attn_layer
kv_cache
layer_slot_mapping
```

到这里 KV write 的上层依赖全部汇合。

---

# 87. 第十一步追踪：真正 GPU KV write

继续：

```text
attn_layer.impl.do_kv_cache_update()
```

当前：

```text
FlashAttentionImpl.do_kv_cache_update()
```

最后：

```text
reshape_and_cache_flash()
↓
torch.ops._C_cache_ops.reshape_and_cache_flash
```

于是完整确认：

```text
slot_mapping
最终进入 C++/CUDA custom op
决定当前 K/V 的 physical write slot
```

到此 KV write path 架构追踪完成。

---

# 88. 第十二步追踪：Attention 怎么读历史 KV

最后回到：

```text
FlashAttentionImpl.forward()
```

关键观察：

```text
flash_attn_varlen_func(
    q=query,
    k=key_cache,
    v=value_cache,
    block_table=block_table,
    seqused_k=seq_lens,
)
```

于是确认：

```text
current K/V 已经在 forward 前写入 KV cache
Attention 直接读取完整 paged KV cache
```

同时进一步确认：

```text
WRITE → slot_mapping
READ  → block_table + seq_lens
```

整个 Scheduler → KV write/read 数据面正式闭环。

---

# 89. 第十三步追踪：为什么 nano-vLLM 分 Prefill/Decode，这里不分

看到统一：

```text
flash_attn_varlen_func()
```

后，出现新的架构问题：

```text
nano-vLLM:
Prefill → varlen
Decode  → with_kvcache

为什么 vLLM V1 不显式分？
```

进一步分析得到：

```text
vLLM V1 Scheduler
已经把 Prefill/Decode 统一成：
每个 request 本轮执行多少 token
```

ModelRunner 再编码成：

```text
query_start_loc
max_query_len
seq_lens
```

所以 mixed batch 可以自然表达。

最终 Attention backend 统一消费：

```text
ragged query + paged KV
```

而真正 workload specialization 下沉到更底层 kernel / dispatch。

---

# 90. 这次源码阅读中最重要的方法论

这次如果直接从 `GPUModelRunner.execute_model()` 一路逐函数往下钻，很容易再次陷入碎片。

更有效的方法实际上是不断提出“跨层问题”。

例如：

```text
Scheduler 的 block id 怎么真正写到 GPU KV？
```

于是链自然变成：

```text
SchedulerOutput
→ BlockTables
→ slot_mapping
→ KV update
→ CUDA op
```

再例如：

```text
Attention 为什么能读非连续 KV？
```

于是链自然变成：

```text
block_table
→ FlashAttention metadata
→ flash_attn_varlen_func
```

因此后续继续读 vLLM 应尽量采用：

```text
先提出真实 runtime 问题
↓
沿 producer → representation → consumer 追
↓
确认 ownership / lifetime / semantics
↓
再决定是否值得深入 kernel
```

而不是：

```text
看到函数 A
→ 读完
→ 调函数 B
→ 又读完
→ 最后不知道为什么读
```

---

# 91. 推荐的源码追踪模板

以后遇到一个核心对象，可以固定问六个问题。

## 1. Why

为什么需要它？没有它会有什么问题？

## 2. Owner

谁持有它？生命周期在哪一层？

## 3. Producer

谁创建 / 更新它？

## 4. Semantics

它到底代表 persistent state，还是 current-step delta？

## 5. Consumer

谁真正使用它？用于 scheduling、addressing 还是 kernel execution？

## 6. Transition

它如何转成下一层 representation？

例如 `slot_mapping`：

```text
Why:
当前 token write 需要 physical destination

Owner:
ModelRunner / BlockTables current-step buffer

Producer:
compute_slot_mappings()

Semantics:
current-step token → physical slot

Consumer:
reshape_and_cache_flash

Transition:
slot id → KV tensor scatter address
```

用这种方法就不会只记住函数名。

---

# 92. 当前 ModelRunner 架构最终压缩成五层

以后复习 MRV2，只需要先恢复下面五层。

```text
                     GPUModelRunner
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
        ▼                 ▼                  ▼
① Persistent State   ② Step View       ③ KV Addressing

req_states           InputBatch         BlockTables
block_tables         idx_mapping        ↓
                     query_start_loc     block_table
                     positions           slot_mapping
                     seq_lens
        │                 │                  │
        └─────────────────┼──────────────────┘
                          ▼
                 ④ Backend Metadata

               CommonAttentionMetadata
                          ↓
                   AttentionGroup
                          ↓
                 MetadataBuilder
                          ↓
               FlashAttentionMetadata
                          │
                          ▼
                    ⑤ GPU Execution
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
        KV Write                    KV Read
            │                           │
K/V + slot_mapping        Q + block_table + seq_lens
            │                           │
            ▼                           ▼
       KV Cache  ◄──────────── FlashAttention
```

---

# 93. 再把 EngineCore 到 GPU 压缩成最终大图

```text
EngineCore.step()
│
▼
Scheduler.schedule()
│
│ 决定：
│ - request
│ - num_scheduled_tokens
│ - KV blocks
│ - lifecycle transitions
│
▼
SchedulerOutput
│
▼
Executor / Worker
│
▼
GPUModelRunner.execute_model()
│
├──────────────────────────────────────────────┐
│ 1. Persistent state synchronization          │
│                                              │
│ Scheduler request/block state                │
│      ↓                                       │
│ req_states + persistent BlockTables          │
├──────────────────────────────────────────────┤
│ 2. Current step view                         │
│                                              │
│ prepare_inputs()                             │
│      ↓                                       │
│ InputBatch                                   │
│ idx_mapping                                  │
│ query_start_loc                              │
│ positions                                    │
│ seq_lens                                     │
├──────────────────────────────────────────────┤
│ 3. KV addressing                             │
│                                              │
│ prepare_attn()                               │
│      ↓                                       │
│ current block_table                          │
│ slot_mapping                                 │
├──────────────────────────────────────────────┤
│ 4. Backend metadata                          │
│                                              │
│ CommonAttentionMetadata                      │
│      ↓                                       │
│ AttentionGroup                               │
│      ↓                                       │
│ FlashAttentionMetadataBuilder                │
│      ↓                                       │
│ FlashAttentionMetadata                       │
├──────────────────────────────────────────────┤
│ 5. Model / Attention execution               │
│                                              │
│ QKV projection                               │
│      │                                       │
│      ├──────── KV WRITE ─────────────┐        │
│      │ K/V                          │        │
│      │ + slot_mapping               │        │
│      │ ↓                            │        │
│      │ reshape_and_cache_flash      │        │
│      │ ↓                            │        │
│      │ GPU Paged KV Cache           │        │
│      │                              │        │
│      └──────── KV READ ─────────────┤        │
│                                     │        │
│ Q + block_table + seq_lens          │        │
│          +                          │        │
│       KV Cache                      │        │
│          ↓                          │        │
│ flash_attn_varlen_func() ◄──────────┘        │
│          ↓                                   │
│ Attention Output                             │
└──────────────────────────────────────────────┘
│
▼
后续 Transformer / LM Head / Sampling
│
▼
ModelRunnerOutput
│
▼
EngineCore.update_from_output()
│
▼
Scheduler state reconciliation
```

---

# 94. 当前阶段最终必须建立的十个结论

## 结论 1

```text
ModelRunner
```

不是“model.forward wrapper”，而是：

```text
Scheduler control plane
→ GPU execution data plane
```

的转换桥。

## 结论 2

MRV2 的核心设计是：

```text
persistent request state
与
per-step batch view
分离
```

## 结论 3

```text
req_idx
```

是 persistent execution-state slot，不是 batch row。

## 结论 4

```text
idx_mapping
```

把 current batch row 映射回 persistent req_idx。

## 结论 5

```text
block_table
```

描述：

```text
request logical block → physical block
```

## 结论 6

```text
slot_mapping
```

描述：

```text
current token → physical KV slot
```

## 结论 7

KV write：

```text
K/V + slot_mapping
→ reshape_and_cache_flash
→ GPU KV Cache
```

## 结论 8

KV read：

```text
Q + block_table + seq_lens + KV Cache
→ FlashAttention
```

## 结论 9

Prefill / Decode 差异没有消失，而是：

```text
explicit mode
→ shape / metadata workload
```

并允许 mixed batch。

## 结论 10

整个 vLLM V1 的一个核心趋势是：

```text
控制面统一 token scheduling
数据面统一 ragged execution representation
kernel 层再做 specialization
```

---

# 95. 当前阶段真正学到的东西

这一阶段最重要的并不是知道：

```text
FlashAttentionImpl.forward 在哪一行
```

而是建立了下面这条完整因果链：

```text
Scheduler 为什么给 request 分 block
↓
block ownership 如何进入 ModelRunner
↓
为什么 ModelRunner 要 persistent BlockTables
↓
为什么 current batch 还要 gather
↓
为什么 write 需要 slot_mapping
↓
slot_mapping 如何由 position + block table 得到
↓
为什么 KV write 与 Attention forward 分开
↓
当前 K/V 如何真正 scatter 到 GPU cache
↓
为什么 Attention read 又不用 slot_mapping
↓
block_table 如何支持 non-contiguous paged KV read
↓
为什么 Prefill/Decode 可以统一成 ragged workload
```

也就是说，我们已经从“知道 vLLM 有 Scheduler / KV Cache / ModelRunner / FlashAttention”推进到：

> **能够解释这些模块为什么存在、谁产生什么状态、状态如何跨层转换、最终如何落到真实 GPU KV Cache 读写。**

这才是当前源码桥接学习阶段真正需要建立的 runtime architecture 认知。

---

# 96. 一句话总括

```text
Scheduler 决定“算谁、算多少、给哪些 KV blocks”；

MRV2 ModelRunner 把这个控制面计划同步成 persistent execution state，
再构造当前 step 的 ragged InputBatch；

BlockTables 把 logical sequence block 映射到 physical KV block，
slot_mapping 再把当前 scheduled token 映射到具体 physical KV slot；

Attention Layer 用 slot_mapping 把新 K/V 写入 Paged KV Cache，
FlashAttention 再用 block_table + seq_lens 从 Paged KV Cache 读取完整历史；

因此 ModelRunner 是整个 vLLM V1 中连接 Scheduler 控制面、Paged KV 地址空间与 GPU Attention kernel 的核心执行桥。
```
