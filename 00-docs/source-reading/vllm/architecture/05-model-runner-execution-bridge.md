# vLLM 源码桥接笔记：ModelRunner 执行桥与后续追踪路线

> 固定版本：vLLM v0.26.0  
> commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> 当前配置：TP=1 / PP=1 / DP=1 / V2 Model Runner / Offline LLM  
> 目标方向：LLM inference / KV Cache  
> 本文用途：**直接承接原笔记第 42～44 节**，补齐 `Worker → GPUModelRunnerV2` 已经确认的内容，并重新校正之后的源码学习顺序。

---

# 45. `Worker.execute_model()`：Worker 不是模型本身，而是 execution runtime adapter

上一阶段已经确认：

```text
EngineCore
↓
Executor
↓
WorkerWrapperBase
↓
CUDA Worker
↓
GPUModelRunnerV2
```

新的真实问题是：

> `CUDA Worker.execute_model()` 收到 `SchedulerOutput` 后，到底自己做什么，哪些事情又交给 ModelRunner？

实际源码确认：

```python
def execute_model(
    self,
    scheduler_output: SchedulerOutput
) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
    ...
    output = self.model_runner.execute_model(
        scheduler_output,
        intermediate_tensors
    )
    ...
```

## 45.1 当前配置下的真实主路径

当前：

```text
TP=1
PP=1
DP=1
V2 Model Runner
普通生成模型
```

因此 Worker 中大量 PP / SP / distributed 特殊分支都不会进入。

当前主路径可以压缩成：

```text
SchedulerOutput
    ↓
CUDA Worker.execute_model()
    ↓
检查 Worker-level runtime 状态
    ↓
intermediate_tensors = None
    ↓
profile annotation
    ↓
GPUModelRunnerV2.execute_model(
    scheduler_output,
    None
)
```

所以：

> **Worker 在架构上不是纯转发，但在当前 PP=1 路径下，它非常接近一个带 distributed/runtime 管理能力的薄包装层。**

---

## 45.2 为什么还需要 Worker 这一层？

如果 Executor 直接调用：

```text
GPUModelRunnerV2
```

那么 ModelRunner 就要自己处理：

```text
Pipeline Parallel communication
rank / local_rank
distributed group
device/runtime lifecycle
profiling / sync checking
intermediate tensor send/recv
```

这些都不是“模型如何执行”本身的问题。

因此 vLLM 把职责拆成：

```text
Worker
=
当前 rank 的 execution runtime
+
distributed communication adapter
+
ModelRunner lifecycle owner

ModelRunner
=
一次 batch 如何真正变成 model execution
```

这是前面“不同变化维度正交拆分”的继续。

---

## 45.3 当前不需要深入的 Worker 分支

当前 PP=1，可以直接标记：

```text
OUT-OF-SCOPE

_pp_send_work
pipeline_parallel_size > 1
irecv_tensor_dict()
isend_tensor_dict()
SP / all_gather
IntermediateTensors 跨 PP stage
```

这些以后学习 PP 时再回来。

---

# 46. `GPUModelRunnerV2.execute_model()`：不是“跑模型”这么简单

进入 `GPUModelRunnerV2.execute_model()` 后，第一眼会看到一个非常长的函数。

如果逐行看，很容易迷失。

第一遍必须先问：

> 为什么 ModelRunner 需要这么长的 `execute_model()`？

原因是：

> **SchedulerOutput 不是可以直接送进 Qwen3.forward() 的 GPU Tensor。**

Scheduler 给出的东西是控制面执行计划，例如：

```text
哪些 request 本轮执行
每个 request 执行多少 token
哪些 request 新增 / finished / resumed
最新 block IDs
已经计算多少 token
```

模型真正需要的却是：

```text
input_ids
positions
batch layout
block_table
slot_mapping
attention metadata
CUDA execution mode
```

所以 ModelRunner 的核心职责是：

> **把 Scheduler 的“执行计划”翻译成一次真实 GPU forward 所需的数据结构和 Tensor。**

---

# 47. `GPUModelRunnerV2.execute_model()` 的 7 段主骨架

把 speculative decode、Mamba、multimodal、Connector、PP、EPLB 等当前无关分支全部砍掉后，可以压缩成：

```text
SchedulerOutput
    ↓
① _update_states()
   更新 persistent batch / request states
    ↓
② _prepare_inputs()
   准备本轮 token execution 信息
    ↓
③ _get_slot_mappings()
   计算本轮 KV 写入位置
    ↓
④ _build_attention_metadata()
   构造 Attention Backend 所需 metadata
    ↓
⑤ _preprocess()
   得到 input_ids / positions / model kwargs
    ↓
⑥ _model_forward()
   真正模型 forward
    ↓
⑦ compute_logits()
   保存 execute_model_state
    ↓
return None
    ↓
later: sample_tokens()
```

第一遍只需要记住这一层级。

不要立刻深入每个 helper。

---

# 48. 为什么 `execute_model()` 最后可能 `return None`

当前 V2 async runtime 中，`execute_model()` 与 sampling 被拆开。

大致是：

```text
execute_model()
    ↓
model forward
    ↓
logits
    ↓
保存到 self.execute_model_state
    ↓
return None

之后：

sample_tokens()
    ↓
消费 execute_model_state
    ↓
产生真正采样结果
```

源码开头甚至会检查：

```text
如果上一次 execute_model_state 还没被 sample_tokens() 消费，
就不能再次 execute_model()
```

所以必须修正一个容易形成的错误理解：

```text
错误：
execute_model()
= forward + sampling + ModelRunnerOutput 一次完成

当前 V2 async 路径更接近：
execute_model()
= submission / forward side

sample_tokens()
= result consumption / sampling side
```

这与之前理解的 async scheduling 是一致的：

```text
execution submission
≠
result consumption
```

---

# 49. Persistent Batch：为什么 ModelRunner 不每轮重建 batch

`GPUModelRunnerV2` 内部维护：

```text
self.requests
self.input_batch
```

它们不是同一个概念。

## 49.1 `self.requests`

可以先理解为：

> ModelRunner 对 request 的本地长期状态缓存。

大致：

```text
request_id
    ↓
CachedRequestState
    ├── prompt_token_ids
    ├── output_token_ids
    ├── num_computed_tokens
    ├── block_ids
    └── sampling state
```

---

## 49.2 `self.input_batch`

可以先理解为：

> 当前真正准备执行的 persistent batch 的 CPU-side representation。

它面向执行，所以包含：

```text
req_ids
token arrays
num_computed_tokens
block_table
sampling metadata
...
```

后续：

```text
_prepare_inputs()
_get_slot_mappings()
```

都会依赖它。

---

## 49.3 为什么叫 Persistent Batch

Continuous batching 下，相邻 step 往往高度重叠。

例如：

```text
Step 1:
A B C

Step 2:
A B C

Step 3:
A B D
```

如果每一步都从零做：

```text
重新创建 A/B/C
重新复制 token
重新生成全部 block table
重新生成全部 metadata
```

CPU overhead 会非常大。

Persistent Batch 选择：

```text
旧 batch
+
Scheduler 本轮增量变化
↓
remove / update / add
```

例如：

```text
Step 2 → Step 3

C remove
D add
A/B 不动
```

所以它本质是一个 production optimization：

> **利用连续 batch 高重叠特性，增量维护 execution-side batch state。**

---

# 50. `_update_states()`：真正做的是 Scheduler 状态 → ModelRunner 状态同步

`_update_states(scheduler_output)` 的 docstring 已经说明：

```text
Update cached states and persistent batch with SchedulerOutput.

之后 _prepare_inputs() 使用这些状态构造 GPU Tensor。
```

第一遍不要陷入 speculative decode 分支。

整个函数只保留 6 件事：

```text
① 删除 finished request
        ↓
② 从 persistent batch 移除本轮 unscheduled request
        ↓
③ 创建 new request 的 CachedRequestState
        ↓
④ 更新 running / resumed request
        ↓
⑤ 同步新的 block IDs / block_table
        ↓
⑥ add new/resumed request
        ↓
condense / refresh
```

---

# 51. `_update_states()` 中三个 request 状态必须区分

## 51.1 Finished

```text
self.requests
    删除

self.input_batch
    删除
```

含义：

> ModelRunner 侧生命周期结束。

注意：

这里是 ModelRunner 本地状态删除。

KVCacheManager 是否 free KV block，是控制面的另外一件事情。

---

## 51.2 当前没被 Scheduler 选中

```text
self.requests
    保留

self.input_batch
    remove
```

原因：

> request 之后还可能重新被调度。

所以：

```text
“不在当前 execution batch”
≠
“request 生命周期结束”
```

---

## 51.3 当前被调度

```text
self.requests
    保留

self.input_batch
    保留 / 更新
```

这说明：

```text
Request lifetime
和
Current batch membership
```

是两个不同维度。

---

# 52. SchedulerOutput 中的 new / cached request 到底是什么意思

ModelRunner 收到的 SchedulerOutput 会区分：

```text
scheduled_new_reqs
scheduled_cached_reqs
finished_req_ids
num_scheduled_tokens
```

这里：

```text
cached request
```

第一遍不要理解成：

```text
Prefix Cache hit request
```

这里更接近：

> **ModelRunner 已经认识、已经拥有 CachedRequestState 的 request。**

所以：

```text
new request
→ ModelRunner 第一次建立 CachedRequestState

cached/running request
→ 更新已有 CachedRequestState
```

---

# 53. 我们已经第一次看到 block ID 如何进入执行侧

新 request 建立状态时：

```text
SchedulerOutput
    ↓
new_req_data.block_ids
    ↓
CachedRequestState.block_ids
```

已有 request 获得新 block：

```text
SchedulerOutput
    ↓
new_block_ids
    ↓
append 到 CachedRequestState.block_ids
    ↓
input_batch.block_table.append_row(...)
```

因此已经得到一条非常重要的 KV 主线：

```text
KVCacheManager / BlockPool
          ↓
Scheduler 侧拿到 block IDs
          ↓
SchedulerOutput
          ↓
GPUModelRunnerV2._update_states()
          ↓
CachedRequestState.block_ids
          ↓
input_batch.block_table
```

这证明：

> **block allocation 不发生在 ModelRunner。**

ModelRunner 消费 Scheduler/KV 控制面已经决定好的 block IDs。

---

# 54. 到这里为什么应该暂停继续钻 ModelRunner

目前继续往下看：

```text
input_batch.block_table
_get_slot_mappings()
_build_attention_metadata()
Attention
```

会遇到大量我们还没有真正建立语义的字段：

```text
num_scheduled_tokens
num_computed_tokens
new_block_ids
scheduled_cached_reqs
resumed_req_ids
KV groups
block table
```

问题不在代码太难，而在于：

> **我们已经开始研究“消费者如何消费 SchedulerOutput”，但还没有正式研究“SchedulerOutput 为什么会被生产成这样”。**

也就是说，我们沿 runtime caller/callee 一路向下追得太快，跳过了概念依赖。

---

# 55. 当前真正缺失的一大块：控制面

目前已经深入理解：

```text
EngineCore
→ Executor
→ Worker
→ ModelRunner
```

但是中间真正决定 runtime 语义的部分还没有系统建立：

```text
EngineCore
    ↓
Scheduler
    ↓
KVCacheManager
    ↓
BlockPool
    ↓
SchedulerOutput
```

所以当前不是“重新从头”。

而是：

> **停止继续向 GPU 数据面下钻，补齐 Scheduler / KV 控制面，然后再回来连接 ModelRunner。**

---

# 56. 重新建立 vLLM KV 主线的大架构

整个生产推理主路径先分成三层：

```text
┌────────────────────────────────────┐
│ 1. Control Plane                   │
│                                    │
│ EngineCore                         │
│ Scheduler                          │
│ KVCacheManager                     │
│ BlockPool                          │
│                                    │
│ 回答：本轮 WHAT TO RUN？           │
└─────────────────┬──────────────────┘
                  │
           SchedulerOutput
                  │
                  ▼
┌────────────────────────────────────┐
│ 2. Execution Preparation           │
│                                    │
│ Executor                           │
│ Worker                             │
│ GPUModelRunnerV2                   │
│ Persistent Batch                   │
│                                    │
│ 回答：如何把计划变成 GPU 输入？    │
└─────────────────┬──────────────────┘
                  │
                  │
       input_ids / positions
       block_table / slot_mapping
       attention metadata
                  │
                  ▼
┌────────────────────────────────────┐
│ 3. GPU Data Plane                  │
│                                    │
│ Model                              │
│ Attention Backend                  │
│ KV Tensor                          │
│ FlashAttention / Triton / CUDA     │
│                                    │
│ 回答：GPU HOW TO EXECUTE？         │
└────────────────────────────────────┘
```

核心不是记类名，而是理解：

```text
Control Plane
    决策

Execution Preparation
    翻译

GPU Data Plane
    执行
```

---

# 57. 为什么 production vLLM 必须这样拆

假设 Scheduler 直接控制 GPU Tensor：

```text
Scheduler
↓
CUDA KV Tensor
↓
Attention
```

那么 Scheduler 就必须理解：

```text
CUDA storage
KV Tensor layout
Attention backend
block table representation
slot mapping
CUDA Graph
device synchronization
```

这样 Scheduler 与 GPU backend 会严重耦合。

反过来，如果 ModelRunner 自己决定 KV allocation：

```text
ModelRunner
↓
自己选择 block
```

那么控制面就无法统一知道：

```text
还有多少 KV capacity
哪个 request 可以 admission
哪个 request 必须 preempt
哪个 prefix 可以 reuse
```

所以生产系统把它拆开：

```text
Scheduler / KVCacheManager
    决定资源和计划

SchedulerOutput
    跨层 contract

ModelRunner
    把计划翻译成 GPU metadata

Attention
    执行真实数据访问
```

这才是这条源码链真正值得学习的架构思想。

---

# 58. Scheduler 为什么存在

以后正式进入 Scheduler 前，先固定一个最基本的系统问题：

```text
GPU 同时存在：

A：长 Prompt，Prefill 200 tokens
B：正在 Decode，下一步 1 token
C：新 Prompt，Prefill 500 tokens

但本轮 token budget = 256
```

不可能三个 request 全部无限制运行。

必须决定：

```text
谁这轮运行？
每个 request 跑多少 token？
谁继续 waiting？
谁可能被 preempt？
```

这就是：

```text
Scheduler
```

Scheduler 解决的是：

> **有限计算预算下，本轮哪些 request 计算多少 token。**

---

# 59. KVCacheManager 为什么存在

即使 Scheduler 想：

```text
A：128 tokens
B：1 token
```

也不能立即执行。

这些 token 会产生新的 KV。

必须确认：

```text
128 + 1 个新 token
需要多少 KV slots / blocks？

GPU KV capacity 够不够？
```

因此需要：

```text
KVCacheManager
```

Scheduler 和 KVCacheManager 的关系：

```text
Scheduler
    “我想算这些 token”
          ↓
KVCacheManager
    “KV 空间允许吗？”
          ↓
BlockPool
    “这些 block 可以分给你”
```

最终调度决策实际上同时受：

```text
Compute token budget
+
KV memory budget
```

约束。

这也是为什么 KV Cache 是 LLM serving 的核心资源对象。

---

# 60. BlockPool 为什么又单独存在

GPU KV Cache 被划分为固定 page/block。

概念上：

```text
KV Cache
├── block 0
├── block 1
├── block 2
├── block 3
...
```

BlockPool 负责管理这些资源记录：

```text
哪些 block free
哪些正在使用
block ID
refcount
cached/free 状态
```

所以：

```text
KVCacheManager
= KV 生命周期 / allocation policy

BlockPool
= 实际 block resource pool
```

注意：

```text
block_id
≠
CUDA pointer
```

block ID 先是控制面稳定编号。

真正怎样映射到 GPU KV Tensor，是之后 ModelRunner / Attention 数据面的事情。

---

# 61. SchedulerOutput 为什么存在

当 Scheduler 和 KVCacheManager 已经完成本轮决策后：

```text
A：本轮 64 tokens
B：本轮 1 token
A：新增 block [17,18,19,20]
B：继续使用已有 blocks
C：继续 waiting
D：finished
```

这些信息必须交给执行侧。

这就是：

```text
SchedulerOutput
```

第一遍可以直接理解成：

> **Scheduler → execution side 的本轮执行计划 / contract。**

因此它自然需要包含：

```text
哪些 request 执行
每个 request 本轮多少 token
哪些 new / cached / resumed
哪些 finished
最新 block IDs
当前 computed progress
```

等信息。

这样再回头看：

```text
GPUModelRunnerV2._update_states(SchedulerOutput)
```

它的意义就会非常自然：

> **把控制面本轮计划同步到执行面的 persistent batch。**

---

# 62. 后续学习路线正式校正

从现在开始，不继续向下钻 `_get_slot_mappings()`。

学习顺序改为：

```text
阶段 A
Scheduler 心智模型

阶段 B
Scheduler.schedule() 主路径

阶段 C
KVCacheManager / BlockPool

阶段 D
重新理解 SchedulerOutput contract

阶段 E
返回 GPUModelRunnerV2

阶段 F
block_table / slot_mapping

阶段 G
Attention Backend / KV read-write
```

这是当前最重要的路线修正。

---

# 63. 阶段 A：先建立 Scheduler 心智模型

暂时不读完整 `schedule()`。

先回答：

```text
1. Scheduler 持有哪些 request？
2. waiting / running 分别是什么意思？
3. Request 用什么字段表达“已经算了多少”？
4. 每一轮 Scheduler 为什么要决定 num_scheduled_tokens？
5. token budget 是什么？
6. Prefill / Decode 为什么可以统一成“本轮算多少 token”？
7. SchedulerOutput 为什么是最终产物？
```

第一阶段目标不是代码细节，而是建立：

```text
Request state
+
Compute budget
↓
Scheduler
↓
num_scheduled_tokens per request
```

KVCacheManager 此时先当黑盒：

```text
“检查 KV capacity / allocate blocks”
```

---

# 64. 阶段 B：再看 `Scheduler.schedule()` 主 control flow

有了心智模型后，才进入源码。

第一遍只看：

```text
running requests
waiting requests
token budget
KVCacheManager.allocate_slots()
preemption
SchedulerOutput construction
```

跳过：

```text
Connector
speculative
multimodal
DP special branch
structured output
metrics
```

目标是能回答：

```text
为什么 A 本轮得到 64 tokens？
为什么 B 本轮得到 1 token？
为什么 C 继续 waiting？
为什么 Scheduler 在某处必须调用 KVCacheManager？
```

---

# 65. 阶段 C：正式学习 KVCacheManager / BlockPool

Scheduler 主链理解后，再进入 KV。

核心问题：

```text
1. 一个 token 为什么需要新的 KV slot？
2. block_size=16 的工程含义是什么？
3. request 的 token 如何映射到 block？
4. allocate_slots() 为什么需要 num_new_tokens？
5. BlockPool 怎样拿出 free block ID？
6. partial block 如何继续追加？
7. request finish 时 block 怎么回收？
8. preemption 时 block 生命周期如何变化？
9. Prefix Cache block 和 active request block 有什么区别？
```

最后形成：

```text
tokens to compute
↓
KV demand
↓
KVCacheManager
↓
BlockPool
↓
block IDs
```

---

# 66. 阶段 D：重新看 SchedulerOutput

等 Scheduler + KV 学完，再重新看 SchedulerOutput。

此时不是背 dataclass 字段，而是问：

```text
哪些字段来自 Scheduler request decision？
哪些字段来自 KV allocation？
哪些字段用于增量同步 ModelRunner？
哪些字段只是特殊 feature？
```

目标形成：

```text
Scheduler internal state
+
KVCacheManager allocation
↓
SchedulerOutput contract
↓
Executor / Worker / ModelRunner
```

---

# 67. 阶段 E：再回来学习 GPUModelRunnerV2

到这里重新回来：

```text
GPUModelRunnerV2.execute_model()
```

此时：

```text
_update_states()
```

不再需要逐行钻。

因为已经知道：

```text
SchedulerOutput
= 本轮计划

_update_states
= 增量同步计划
```

真正值得进入的是：

```text
Persistent Batch
↓
block_table
↓
slot_mapping
↓
Attention metadata
```

---

# 68. 阶段 F：block_table 与 slot_mapping

这一步是项目未来最重要的数据路径之一。

目标要回答：

```text
Scheduler block ID
↓
如何写进 input_batch.block_table？

block_table
到底表达什么？

当前新 token
↓
如何计算 slot_mapping？

slot_mapping
为什么是 KV write address？

历史 KV read
为什么主要依赖 block_table？
```

最终建立：

```text
logical token position
↓
request block table
↓
physical KV block ID
↓
slot mapping
↓
GPU KV storage
```

---

# 69. 阶段 G：Attention Backend

最后进入：

```text
Attention Layer
Attention Backend
KV Tensor
```

只追一条当前真实 backend。

回答：

```text
1. 新 K/V 在哪里写入 KV Cache？
2. slot_mapping 如何被使用？
3. 历史 K/V 如何通过 block_table 被读取？
4. KV Tensor 真正在哪里分配？
5. KV layout / dtype 由谁决定？
6. Prefill / Decode 在 backend 层如何体现？
```

做到这里，才算真正把：

```text
Scheduler control plane
→ KV resource management
→ ModelRunner execution preparation
→ Attention KV data path
```

完整闭环。

---

# 70. 为什么这条路线直接服务未来 Quantized Paged KV Cache 项目

未来如果要实现：

```text
Recent BF16
+
History INT4
```

会同时影响三层。

## Control Plane

需要表达：

```text
logical block 当前是什么状态？
需要多少资源？
什么时候 migration？
```

可能落在：

```text
KVCacheManager
Scheduler metadata
```

---

## Execution Preparation

需要把 logical block 解释成：

```text
BF16 pool physical block
or
INT4 pool physical block
```

可能涉及：

```text
ModelRunner
block_table
additional metadata
```

---

## GPU Data Plane

Attention 必须知道：

```text
这个 block 是 BF16 还是 INT4？
packed data 在哪？
scale 在哪？
如何读取和 dequantize？
```

涉及：

```text
Attention Backend
Triton / CUDA
```

所以学 vLLM 的目标不是：

```text
会解释某个 300 行函数
```

而是：

> **知道一个 KV 系统优化应该把 control state、physical storage、runtime metadata 和 GPU kernel 分别放在哪里。**

---

# 71. 之后的固定源码学习模板

为了防止再次掉进“函数分析模式”，之后每进入一个新组件，都必须先完成下面 6 个问题：

```text
① 当前系统问题是什么？
        ↓
② 为什么需要这个组件？
        ↓
③ 它拥有哪类状态？
        ↓
④ 上游给它什么？
        ↓
⑤ 它给下游什么？
        ↓
⑥ 再进入具体源码
```

源码函数只回答：

```text
这个设计具体是怎么实现的？
```

而不是用源码本身代替架构理解。

---

# 72. 每次 rg 的新纪律

以后仍然保持：

> 一次只给一个 `rg`

但进入 `rg` 前必须先回答：

```text
我们为什么要找这个 symbol？
它处在整条路径的哪一层？
我们期待通过它验证什么？
```

例如：

```text
错误方式：

rg "def schedule"
→ 开始看 2000 行 Scheduler
```

正确方式：

```text
真实问题：
多个 request 竞争 token/KV budget 时，
Scheduler 如何决定每个 request 本轮算几个 token？

↓
因此我们才定位：
Scheduler.schedule()
```

---

# 73. 当前已经完成到哪里

截至目前，可以认为已经完成：

```text
Frontend / EngineCore process boundary
EngineCoreProc
Executor abstraction
UniProcExecutor
WorkerWrapperBase
CUDA Worker
WorkerBase
GPUModelRunnerV2 creation

Runtime:
SchedulerOutput
→ Executor
→ Worker
→ GPUModelRunnerV2.execute_model() entry

ModelRunner 第一层：
persistent batch 的存在原因
_update_states() 的总体职责
block IDs 会被同步进 execution-side block_table
execute_model / sample_tokens 在 V2 async 路径中分离
```

这些内容保留，不需要重新学。

---

# 74. 当前明确还没有真正完成的部分

不要因为已经在 ModelRunner 里看到了这些名字，就误以为已经掌握：

```text
Scheduler 正式主路径
waiting / running 状态机
num_computed_tokens 真实语义
num_scheduled_tokens 真实产生过程
token budget
chunked prefill
preemption

KVCacheManager
BlockPool
allocate_slots()
free lifecycle
prefix cache lifecycle

SchedulerOutput 完整 contract

input_batch.block_table 内部
slot_mapping
attention metadata
Attention Backend
KV write/read
```

这些才是之后的主线。

---

# 75. 下一阶段正式起点

下一阶段**不再从 ModelRunner 开始**。

第一个问题固定为：

> **假设现在同时存在 3 个请求：一个正在 Decode、一个长 Prompt 正在 Prefill、一个刚到达，为什么 vLLM 必须有 Scheduler？Scheduler 每一轮真正要决定什么？**

先通过一个具体三请求例子建立：

```text
Request
waiting / running
Prefill / Decode
num_computed_tokens
num_scheduled_tokens
token budget
KV budget
SchedulerOutput
```

的直觉。

在这个心智模型建立之前：

```text
不读 KVCacheManager
不读 BlockPool
不读 block_table
不读 slot_mapping
不继续 ModelRunner
```

---

# 76. 后续完整追踪路线

最终固定追踪顺序：

```text
① Request / Scheduler 心智模型
        ↓
② Scheduler.schedule()
        ↓
③ token budget / waiting / running
        ↓
④ KVCacheManager.allocate_slots()
        ↓
⑤ BlockPool / block lifecycle
        ↓
⑥ SchedulerOutput
        ↓
⑦ Executor / Worker
        ↓
⑧ GPUModelRunnerV2 persistent batch
        ↓
⑨ block_table
        ↓
⑩ slot_mapping
        ↓
⑪ Attention metadata
        ↓
⑫ Attention Backend
        ↓
⑬ KV write / paged KV read
        ↓
⑭ ModelRunnerOutput / sample_tokens
        ↓
⑮ Scheduler state update
```

这才是之后真正要闭环的 production vLLM KV execution path。

---

# 77. 当前阶段一句话总结

当前不是：

> “ModelRunner 看不懂，所以退回去重新学。”

而是：

> **已经提前确认了 execution-side 的形状，现在暂停向下钻，补齐 Scheduler / KV 控制面的生产语义，然后再回来连接 block_table、slot_mapping 和 Attention 数据路径。**

最终目标始终不变：

```text
真实问题
→ 架构动机
→ object ownership
→ control-plane decision
→ SchedulerOutput contract
→ execution-side translation
→ GPU KV data path
→ dynamic evidence
→ 为什么这样设计
```

这比单纯“看完源码”更重要。
