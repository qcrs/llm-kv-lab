# vLLM v0.26.0 整体架构与 KV 主链总纲

> 固定版本：vLLM v0.26.0  
> Commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> 当前实验：A100 80GB / Qwen3-0.6B / Offline `LLM` / TP=1 / PP=1 / DP=1  
> Engine：vLLM V1  
> Model Runner：V2 Model Runner（MRV2）  
> 学习目标：不是“看完 vLLM”，而是建立一套能够支撑 KV Cache 项目开发的生产级推理 Runtime 心智模型。

---

# 0. 这份文档解决什么问题

前面的 00～07 笔记分别解决了：

```text
Linux Process / Thread
↓
Frontend → EngineCore Process
↓
EngineCore / EngineCoreProc 分层
↓
Executor / Worker / ModelRunner
↓
Scheduler / KVCacheManager / BlockPool
↓
SchedulerOutput
↓
MRV2 execution state
↓
block_table / slot_mapping
↓
KV write / paged KV read / Attention Backend
```

单篇阅读时，每个模块已经比较清楚，但容易出现一个问题：

> 知道很多局部函数，却没有形成“一个请求为什么能从 token 变成 GPU HBM 中的一页 KV，再被下一轮 Attention 找回来”的整体认知。

所以本文不再按源码文件组织，而是只围绕一个问题展开：

> **一个 Request 在 vLLM 中如何被持续调度、获得 KV 空间、执行模型、写入 Paged KV、读取历史 KV、产生新 token，再进入下一轮？**

最终要形成的是下面这一条闭环：

```text
Request state
    ↓
Scheduler.schedule()
    ↓
本轮 token decision
    ↓
KVCacheManager.allocate_slots()
    ↓
BlockPool physical block IDs
    ↓
SchedulerOutput
    ↓
Executor / Worker
    ↓
MRV2 GPUModelRunner
    ↓
persistent state → per-step InputBatch
    ↓
block_table + slot_mapping
    ↓
Attention metadata
    ↓
KV WRITE + PAGED KV READ
    ↓
ModelRunnerOutput
    ↓
Scheduler.update_from_output()
    ↓
Request state 下一轮
```

这也是当前 vLLM 定向桥接学习真正需要掌握的主路径。

---

# 1. 先建立最高层心智模型：vLLM 是一个“持续运行的状态机”

不要把 vLLM 理解成：

```text
request
→ model.forward()
→ output
```

生产 Serving Runtime 更接近：

```text
State_t
  ↓
Decision_t
  ↓
Execution_t
  ↓
Result_t
  ↓
State_t+1
```

其中：

```text
State
= Request 当前拥有多少 token
+ 已经计算到哪里
+ 当前 KV block ownership
+ waiting / running / preempted 状态
+ execution-side persistent state

Decision
= 本轮执行哪些 Request
+ 每个 Request 执行多少 token
+ 是否需要新增 KV block

Execution
= 构造 GPU input
+ model forward
+ KV write/read
+ sampling

Result
= sampled token
+ completion / stop
+ execution result
```

因此 EngineCore 的主循环才会是：

```text
Scheduler current state
        ↓
Scheduler.schedule()
        ↓
SchedulerOutput
        ↓
Executor / Worker / ModelRunner
        ↓
ModelRunnerOutput
        ↓
Scheduler.update_from_output()
        ↓
Scheduler next state
```

**理解 vLLM 的关键不是“谁调用了谁”，而是“哪一层持有什么长期状态，哪一层只描述当前 step”。**

---

# 2. 整体分成三个世界：Process、Control Plane、Data Plane

## 2.1 Process / Runtime deployment world

回答：

> 代码运行在哪个 Linux Process？

当前 Offline `LLM` + V1 multiprocessing 路径：

```text
Frontend Python Process
│
│  LLM / LLMEngine / SyncMPClient
│
└──────────── ZMQ ────────────┐
                              ▼
                     EngineCore Process
```

动态实验已经看到：

```text
python smoke_06b.py
└── VLLM::EngineCore
```

因此：

```text
Process
≠ Python Object
≠ Thread
```

一个类名中出现：

```text
Proc
Worker
Executor
```

都不能据此判断它是一个 Linux Process。

真正判断 Process creation，要找到类似：

```python
multiprocessing.Process(...)
proc.start()
```

---

## 2.2 CPU Control Plane

回答：

> 现在应该让谁执行？执行多少？需要多少 KV 空间？

主要对象：

```text
EngineCore
Scheduler
Request
KVCacheManager
Coordinator
SingleTypeKVCacheManager
BlockPool
KVCacheBlock
SchedulerOutput
```

核心逻辑：

```text
Request progress
↓
compute demand
↓
KV capacity demand
↓
physical block allocation
↓
execution plan
```

---

## 2.3 GPU Execution / Data Plane

回答：

> 已经决定好的这一批工作，怎么真正变成 GPU Tensor、KV 地址和 Attention execution？

主要对象：

```text
Executor
WorkerWrapperBase
CUDA Worker
MRV2 GPUModelRunner
RequestState
InputBatch
BlockTables
Attention metadata
KV cache tensors
Attention Backend
```

核心逻辑：

```text
SchedulerOutput
↓
execution-side state
↓
GPU input representation
↓
physical KV addressing
↓
model / Attention kernel
```

---

# 3. 初始化链和 Runtime 链必须彻底分开

这是前面阅读源码时最容易混乱的地方。

## 3.1 初始化链：搭建对象图

```text
LLM(...)
↓
LLMEngine
↓
EngineCoreClient.make_client()
↓
SyncMPClient
↓
MPClient
↓
launch_core_engines()
↓
CoreEngineProcManager
↓
proc.start()
↓
EngineCore Linux Process
↓
EngineCoreProc(...)
↓
EngineCore.__init__()
│
├── Executor
│    ↓
│  WorkerWrapperBase
│    ↓
│  CUDA Worker
│    ↓
│  GPUModelRunnerV2
│
├── KV cache initialization
│
└── Scheduler
```

这时只是：

```text
对象创建
device initialization
model load
KV cache allocation/init
IPC setup
```

**还没有开始处理 Request。**

---

## 3.2 Runtime 链：真正处理 Request

```text
EngineCore step_fn
↓
Scheduler.schedule()
↓
SchedulerOutput
↓
Executor.execute_model()
↓
Worker.execute_model()
↓
GPUModelRunner.execute_model()
↓
GPU execution / sampling
↓
ModelRunnerOutput
↓
Scheduler.update_from_output()
```

所以不要因为初始化阶段先看到了 ModelRunner，就产生：

```text
EngineCore
→ ModelRunner
→ Scheduler 被绕过
```

这种错误理解。

**初始化路径描述“对象怎么建立”；Runtime 路径描述“请求怎么运行”。**

---

# 4. Frontend 为什么会出现一个独立 EngineCore Process

当前 `LLM(...)` 入口大致经过：

```text
LLM.__init__()
↓
LLMEngine.from_engine_args()
↓
EngineCoreClient.make_client()
```

当前环境中：

```text
VLLM_ENABLE_V1_MULTIPROCESSING = True
asyncio_mode = False
```

因此：

```text
multiprocess_mode=True
asyncio_mode=False
↓
SyncMPClient
```

这里必须区分两个完全不同维度：

```text
multiprocess_mode
= Frontend 与 EngineCore 是否独立 Process

asyncio_mode
= Frontend Client 是否采用 asyncio 调用模型
```

所以：

```text
asyncio_mode=False
```

并不等于：

```text
Engine 内部没有 asynchronous scheduling
```

这是两个不同概念。

---

# 5. `SyncMPClient → launch_core_engines → CoreEngineProcManager` 各自做什么

可以只记三句话。

## `SyncMPClient`

> Frontend 怎么同步调用独立 Engine。

主要屏蔽：

```text
ZMQ
request / output transport
output queue thread
```

---

## `launch_core_engines()`

> 根据配置决定应该启动什么 Engine topology。

它回答：

```text
Local Process？
Ray Actor？
DP 几个 Engine？
是否需要 Coordinator？
```

它属于 deployment policy / orchestration。

---

## `CoreEngineProcManager`

> 已经决定使用 Local Process 后，具体怎么管理 Process 生命周期。

包括：

```text
create
start
ready
monitor
shutdown
```

因此：

```text
launch_core_engines
= 选择方案

CoreEngineProcManager
= 执行这个方案
```

可以理解成：

```text
Policy vs Mechanism
```

---

# 6. `EngineCoreProc` 和 `EngineCore` 不是两个 Runtime Object

源码关系：

```python
class EngineCoreProc(EngineCore):
```

所以：

```text
EngineCoreProc IS-A EngineCore
```

不是：

```text
EngineCoreProc HAS-A EngineCore
```

Linux EngineCore Process 内部大致是：

```text
EngineCore Linux Process
│
└── engine_core = EngineCoreProc(...)
        │
        ├── EngineCore 的核心 inference runtime 能力
        │    ├── Scheduler
        │    ├── Executor
        │    ├── KV initialization
        │    └── step / runtime state
        │
        └── EngineCoreProc 增加的 Process adapter 能力
             ├── ZMQ
             ├── input/output queue
             ├── thread
             ├── handshake
             ├── signal
             └── shutdown
```

所以最准确的定义是：

```text
EngineCore
= 与具体 Process / ZMQ 相对解耦的 inference runtime core

EngineCoreProc
= 能作为独立后台 ZMQ Process 运行的 EngineCore
```

---

# 7. 为什么还需要 Executor / Worker / ModelRunner 三层

如果 EngineCore 直接调用：

```text
GPUModelRunner
```

那么 EngineCore 会同时知道：

```text
CUDA / CPU / XPU
UniProc / Multiproc / Ray
TP / PP / DP
rank / local_rank
RPC
GPU lifecycle
model lifecycle
```

生产 Runtime 会严重耦合。

所以 vLLM 把不同变化维度拆开：

```text
EngineCore
= runtime orchestration

Scheduler
= scheduling policy + request/KV control

Executor
= deployment / execution topology abstraction

WorkerWrapper
= lifecycle / proxy / lazy construction

Worker
= current execution rank + device / distributed runtime

ModelRunner
= execution-side state + GPU input preparation + model execution
```

这套拆分的核心不是“层多”，而是：

> **让不同变化维度可以独立演化。**

例如：

```text
换 Executor backend
不需要重写 Scheduler

换 GPU / CPU Worker
不需要改变 EngineCore contract

换 Model Runner V1 / V2
不需要重写整个 Runtime
```

---

# 8. 当前为什么是 `UniProcExecutor`，但仍然有两个 Linux Process

当前：

```text
TP=1
PP=1
DP=1
world_size=1
```

所以：

```text
distributed_executor_backend = "uni"
↓
Executor.get_class()
↓
UniProcExecutor
```

但：

```text
UniProcExecutor
≠ 整个 vLLM 只有一个 Linux Process
```

因为：

```text
Frontend / EngineCore Process boundary
```

和：

```text
EngineCore 内部 model execution topology
```

是两个维度。

当前正确结构：

```text
Frontend Linux Process

        │ ZMQ

EngineCore Linux Process
│
├── EngineCoreProc Object
├── Scheduler Object
├── UniProcExecutor Object
├── WorkerWrapperBase Object
├── CUDA Worker Object
└── GPUModelRunnerV2 Object
```

在当前 UniProc 路径里，Worker / ModelRunner 都是同一个 EngineCore Linux Process 内部的 Python Object。

---

# 9. EngineCore 的真正职责：驱动“决策 → 执行 → 回收”闭环

当前主路径可以压缩为：

```python
scheduler_output = self.scheduler.schedule(...)

future = self.model_executor.execute_model(
    scheduler_output,
    non_block=True,
)

model_output = future.result()

engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output,
    model_output,
)
```

因此：

```text
Scheduler
= decide / control

Executor + Worker + ModelRunner
= execute

Scheduler.update_from_output()
= result reconciliation

EngineCore
= orchestration loop
```

EngineCore 自己不负责：

```text
request priority policy
free block queue
slot_mapping calculation
Attention math
```

它负责的是：

> **让几个长期存在的 subsystem 按正确顺序持续推进。**

---

# 10. Scheduler 是一个长期存在的 stateful controller

Scheduler 持有的主状态可以分成三类。

## 10.1 Request lifecycle

```text
requests
waiting
running
finished bookkeeping
preempted bookkeeping
```

因此：

```text
waiting / running
```

描述的是 Request 的调度生命周期。

它们不是：

```text
waiting = Prefill
running = Decode
```

例如一个 Chunked Prefill Request 完全可能已经在：

```text
running
```

---

## 10.2 Scheduling constraints

例如：

```text
max_num_running_reqs
max_num_scheduled_tokens
max_model_len
token_budget
```

回答：

```text
这一轮最多可以运行多少 Request？
这一轮最多可以计算多少 token？
```

---

## 10.3 KV resource interface

Scheduler 持有：

```text
KVCacheManager
```

因此 Scheduler 决定：

```text
我要让 Request A 本轮算 N token
```

但不会直接：

```text
从 free queue pop 一个 physical block
```

真正 KV resource mechanics 交给 KVCacheManager / concrete manager / BlockPool。

---

# 11. 理解 V1 Scheduler 的核心：两个 frontier

一个 Request 最重要的不是“它是 Prefill 还是 Decode”，而是两个位置。

## 11.1 Known-token frontier

```text
num_tokens
```

表示：

> Request 当前已经拥有的真实 token sequence 长度。

普通非 speculative 路径中大致是：

```text
Prompt tokens
+
已经接受的 generated output tokens
```

例如：

```text
Prompt = 100
已经 sample 20 个 output

num_tokens = 120
```

---

## 11.2 Computed frontier

```text
num_computed_tokens
```

表示：

> 当前有多少 token position 已经被模型执行 frontier 覆盖。

例如：

```text
Prompt 总长 = 1000
只执行了前 256 token

num_tokens = 1000
num_computed_tokens = 256
```

说明：

```text
Request 已经知道完整 1000 token
但 model execution 只推进到 256
```

这两个 frontier 的差值就是 Scheduler 最基本的 work demand。

在当前普通路径下可以近似理解：

```text
pending work
≈ num_tokens_with_spec - num_computed_tokens
```

然后再被：

```text
token budget
model length
chunked prefill
KV capacity
feature-specific constraints
```

继续裁剪。

---

# 12. 为什么 V1 Scheduler 可以统一 Prefill 与 Decode

源码设计的核心是：

```text
Scheduler 不需要先问：
“这是 Prefill request 还是 Decode request？”

而是问：
“这个 Request 的 computed frontier
还落后 known frontier 多少？”
```

## Prefill

```text
num_tokens = 1000
num_computed_tokens = 400

pending = 600
```

这表现为大段 token work。

## Decode

假设：

```text
Prompt 已全部计算
刚刚 sample 出 O0

num_tokens = 101
num_computed_tokens = 100
```

此时：

```text
pending = 1
```

下一轮只需要 forward 这个刚生成的 O0，再 sample O1。

所以 Scheduler 层：

```text
Prefill / Decode
→ 都统一为“推进多少 token”
```

但这不意味着两者完全没有区别。

不同层看到的是不同 abstraction：

```text
Control Plane
Prefill / Decode
→ token scheduling problem

ModelRunner / Attention metadata
→ query length / seq length problem

Kernel Plane
→ workload-specific specialization
```

因此内部仍然出现 `prefill` 命名 helper 或不同 kernel specialization，并不和统一 Scheduler 矛盾。

---

# 13. `num_scheduled_tokens`：本 step 的 delta，不是 Request 总长度

必须把长期 state 与 step decision 分开。

```text
num_tokens
= persistent known sequence length

num_computed_tokens
= persistent/optimistic execution frontier

num_scheduled_tokens[req]
= 当前 Scheduler step 给这个 Request 的 computation delta
```

例如：

```text
A Decode:
num_scheduled_tokens[A] = 1

B Chunked Prefill:
num_scheduled_tokens[B] = 255
```

总 batch：

```text
total_num_scheduled_tokens = 256
```

这个值决定 execution-side 很多 per-step Tensor / metadata 的规模。

---

# 14. Scheduler 为什么还要询问 KVCacheManager

Scheduler 即使决定：

```text
A 本轮算 256 token
```

也不能直接认为这个决策可执行。

因为每个新计算 position 都需要合法的 KV destination。

因此存在第二个约束：

```text
Compute budget
≠
KV capacity
```

一个 Request 的执行需求需要转成：

```text
本 step 后，我的 KV coverage 至少要到哪里？
```

普通单类型主路径可以先近似理解为：

```text
target KV coverage
≈ num_computed_tokens(before step)
 + num_scheduled_tokens
```

更复杂 feature（external computed tokens、skipped tokens、lookahead 等）会调整精确语义；当前主路径只保留这个核心。

---

# 15. KV 控制面分层：为什么不是一个 BlockManager 就结束

当前关键层次：

```text
Scheduler
↓
KVCacheManager
↓
Coordinator
↓
SingleTypeKVCacheManager
↓
BlockPool
↓
KVCacheBlock
```

每一层回答不同问题。

---

## 15.1 `KVCacheManager`

> 面向 Scheduler 的 Request-level KV resource interface。

它关心：

```text
这个 Request 需要多少 KV coverage？
是否能够 allocate？
finish/preempt 时如何 free？
```

它不负责 Scheduler priority。

---

## 15.2 `Coordinator`

> 协调一个或多个 concrete KV manager。

生产框架不能默认：

```text
所有 layer
都只有一种完全相同的 KV layout
```

所以中间需要协调层。

当前普通 Qwen3 full-attention 路径可以先把它理解成：

```text
上层统一入口
↓
下层 concrete manager
```

---

## 15.3 `SingleTypeKVCacheManager`

> 某一类 KV Cache 的 token → block coverage 规则，以及 Request 的 block ownership。

关键 persistent state：

```text
req_to_blocks[request]
```

它回答：

```text
Request A 当前已经拥有多少 block？
为了 target coverage 总共需要多少 block？
这次还需要新增几个？
```

---

## 15.4 `BlockPool`

> 全局 physical KV block metadata resource pool。

负责：

```text
free block queue
get_new_blocks()
free_blocks()
refcount
cached block metadata
```

它不关心：

```text
A 是 Prefill 还是 Decode
A 的 token budget 是多少
```

---

## 15.5 `KVCacheBlock`

> 一个 physical KV block 的控制面 identity / metadata record。

它可以包含：

```text
block_id
ref_cnt
block_hash
...
```

但：

```text
KVCacheBlock object
≠ CUDA Tensor page
≠ CUDA pointer
```

这是整个 KV 主链最重要的边界之一。

---

# 16. 从 token coverage 到新增 physical block

假设：

```text
block_size = 16
```

Request 当前已经持有 2 个 block：

```text
num_req_blocks = 2
```

即最多能够覆盖：

```text
32 token slots
```

本轮后 target coverage 变成：

```text
41 tokens
```

那么总共需要：

```text
ceil(41 / 16) = 3 blocks
```

因此：

```text
num_required_blocks = 3
num_req_blocks = 2
num_new_blocks = 1
```

然后：

```text
SingleTypeKVCacheManager
↓
BlockPool.get_new_blocks(1)
↓
physical block ID，例如 18
↓
append 到 req_to_blocks[A]
```

所以必须区分：

```text
num_scheduled_tokens
= 本轮计算多少 token

num_new_blocks
= 本轮为了满足 KV coverage 需要新增多少 page
```

二者没有一一对应关系。

例如：

```text
Decode 1 token
```

如果仍处于当前 partial block 内：

```text
num_scheduled_tokens = 1
num_new_blocks = 0
```

完全正常。

---

# 17. `block_id` 为什么是控制面与数据面的桥梁，但不是 GPU 地址

Scheduler / BlockPool 得到：

```text
physical block ID = 18
```

这个 18 是一个稳定的页编号。

它不是：

```text
0x7f... CUDA pointer
```

控制面只需要表达：

```text
Request A 的 logical block 1
→ physical block 18
```

GPU 数据面则拥有真实 KV Tensor，例如概念上：

```text
K_cache[physical_block, offset, ...]
V_cache[physical_block, offset, ...]
```

因此：

```text
block ID
= stable page identity

KV Tensor
= real GPU storage
```

这就是为什么控制面可以管理资源而不直接操作 CUDA pointer。

---

# 18. SchedulerOutput：Control Plane → Execution Plane 的协议边界

这是整个架构最值得重视的对象之一。

Scheduler 内部拥有很多复杂 state，但不会把整个 Scheduler 对象交给 ModelRunner。

它构造一个：

```text
SchedulerOutput
```

只传本轮执行真正需要的 control delta。

核心语义可以分为：

```text
1. 哪些 Request 本轮执行
2. 每个 Request 本轮执行多少 token
3. new / resumed Request 的初始化数据
4. continuing Request 的增量更新
5. 新增 block IDs / KV maintenance instruction
6. finished / preempted lifecycle changes
```

因此可以理解成：

```text
Scheduler internal world
      ↓ serialize / summarize current decision
SchedulerOutput
      ↓
Execution world
```

这也是未来做 QCache 时非常可能需要扩展的 control-plane contract。

---

# 19. New / Resumed / Continuing：为什么要分增量同步

Scheduler producer 视角：

```text
WAITING 首次 admission
→ new

PREEMPTED 再次进入 execution
→ resumed

已经 RUNNING 并继续执行
→ continuing
```

进入 MRV2 后：

```text
new + resumed
→ 都需要建立 fresh execution-side state

continuing
→ 只 patch delta
```

因此：

```text
Scheduler Request 生命周期
≠
ModelRunner active execution-state 生命周期
```

尤其 preemption：

```text
Scheduler：
Request 仍然存在，只是 PREEMPTED，后面可以 resume

MRV2：
当前 active execution state 被 remove

resume 后：
重新 add execution state
```

这是生产 Runtime 中非常典型的分层生命周期。

---

# 20. Executor / Worker 这一段 Runtime bridge 到底做什么

当前路径：

```text
EngineCore
↓
UniProcExecutor.execute_model()
↓
collective_rpc("execute_model")
↓
WorkerWrapperBase
↓
CUDA Worker.execute_model()
↓
GPUModelRunnerV2.execute_model()
```

其中：

## Executor

```text
HOW / WHERE TO EXECUTE
```

当前 UniProc 很薄，但 abstraction 必须存在，因为其他 topology 可以是 MP / Ray 等。

## Worker

```text
当前 rank / device 的 execution runtime adapter
```

负责：

```text
device lifecycle
distributed context
PP communication
profiling/runtime checks
ModelRunner ownership
```

当前 PP=1 路径中很多 distributed branch 不进入，所以 Worker 看起来比较薄。

## ModelRunner

```text
把 SchedulerOutput 翻译成 GPU execution world
```

它不是简单：

```text
model(input)
```

而是：

```text
control state synchronization
+
per-step input construction
+
KV addressing
+
Attention metadata
+
model execution
```

---

# 21. 版本校正：V1 Engine ≠ Model Runner V1

当前环境一定要写清：

```text
vLLM V1 Engine
+
V2 Model Runner（MRV2）
```

两者不是一个版本维度。

当前 MRV2 主路径：

```text
vllm/v1/worker/gpu/model_runner.py
```

不要再用 MRV1：

```text
vllm/v1/worker/gpu_model_runner.py
```

中的：

```text
CachedRequestState
persistent self.input_batch
_update_states()
```

去解释当前 MRV2。

MRV2 最关键的 architecture change 是：

```text
persistent request state

与

current-step InputBatch

分离
```

---

# 22. MRV2 的核心：Persistent State 与 Per-step View 分离

Serving batch 会不断变化：

```text
Step t:
[A, B, C, D]

Step t+1:
[A, D, E]

Step t+2:
[D, E, F, G]
```

如果长期 request state 和 batch row 强绑定，就需要频繁：

```text
row move
compaction
metadata move
block table move
sampler state move
```

MRV2 选择：

```text
Persistent Request State
= stable request-indexed storage

Per-step InputBatch
= 当前 Scheduler decision 对 persistent state 的 gather view
```

可以类比：

```text
req_states
= database table

InputBatch
= SELECT 当前需要执行的 request rows
```

这是 MRV2 最重要的心智模型。

---

# 23. MRV2 `execute_model()` 五阶段

整体不要逐行记，先记五层：

```text
SchedulerOutput
↓

① Persistent state synchronization
   finish_requests()
   free_states()
   add_requests()
   update_requests()
   block_tables.apply_staged_writes()

↓

② Current-step input preparation
   prepare_inputs()
   → InputBatch

↓

③ KV addressing preparation
   prepare_attn()
   → current block_tables
   → slot_mappings

↓

④ Backend metadata preparation
   model_state.prepare_attn()
   → backend-specific Attention metadata

↓

⑤ Model / Attention execution
   QKV
   KV write
   paged KV read
   logits / sampling
```

这五段比任何单个 helper 都重要。

---

# 24. 为什么必须先同步 persistent state，再构建 InputBatch

假设 Scheduler 本轮让 PREEMPTED Request A resume。

SchedulerOutput 已经告诉 ModelRunner：

```text
A 要重新执行
A 当前 token state
A 当前 block IDs
A 本轮 num_scheduled_tokens
```

如果直接：

```text
prepare_inputs()
```

那么 execution-side persistent state 中可能还没有 A。

所以必须：

```text
Scheduler lifecycle transition
↓
add/update/remove persistent execution state
↓
commit staged writes
↓
当前 step 才 gather InputBatch
```

这里体现一个非常通用的系统原则：

> **producer 的 state delta 必须先 commit，consumer 才能构造本轮 view。**

---

# 25. `StagedWriteTensor` / `apply_staged_writes()` 怎么理解

Serving 的 metadata update 特征是：

```text
很多 Request
×
每个 step 每个 Request 只改一点点
```

比如 Decode：

```text
A append 1 token
B append 1 token
C append 1 token
...
```

如果每个 scalar change 都立即触发一次 GPU-visible update，固定开销会很高。

所以 MRV2 会：

```text
当前 execute_model step
↓
连续 stage 很多细碎修改
↓
apply_staged_writes()
↓
后续 input preparation / GPU execution 读取到一致的新状态
```

注意：

```text
stage_write
≠ 等 Request finish 才提交
```

它的 batching boundary 是：

```text
一个 execution step
```

而不是 request lifetime。

---

# 26. Per-step `InputBatch` 到底包含什么

Persistent state 解决：

```text
Request 长期状态在哪里
```

`InputBatch` 解决：

```text
这一轮到底执行哪些 Request、哪些 token
```

关键内容包括：

```text
idx_mapping
positions
query_start_loc
seq_lens
本轮 flattened token representation
```

可以把它理解成：

```text
Persistent state
     ↓ gather by current SchedulerOutput
Per-step InputBatch
```

所以在 MRV2 中：

```text
InputBatch
≠ 长期 Request owner
```

它是当前 step 的 execution view。

---

# 27. ModelRunner 中最值得掌握的四张 mapping

整个 MRV2 很多代码可以压缩成四类 mapping。

## 27.1 `req_id → req_idx`

```text
Scheduler Request identity
→ persistent execution-state row
```

解决：

```text
A 的长期 execution metadata 存在哪一行？
```

---

## 27.2 `current batch row → persistent req_idx`

即：

```text
idx_mapping
```

解决：

```text
当前 step 第 i 个 request
应该去 persistent storage gather 哪一行？
```

---

## 27.3 `logical KV block → physical block ID`

即：

```text
block_table
```

解决：

```text
Request logical history 的第 j 个 block
实际在 GPU KV pool 的哪个 physical page？
```

---

## 27.4 `current token → physical KV slot`

即：

```text
slot_mapping
```

解决：

```text
本轮新计算出来的 K/V
到底写进哪个 physical slot？
```

把这四张 mapping 理清，ModelRunner 就不再是一堆散乱 metadata。

---

# 28. `block_table`：历史 KV 的逻辑序列 → physical page mapping

假设：

```text
block_size = 16
```

Request A 的 KV：

```text
logical block 0 → physical block 7
logical block 1 → physical block 18
logical block 2 → physical block 31
```

那么：

```text
block_table[A] = [7, 18, 31]
```

这张表表达：

```text
Request logical sequence
如何映射到非连续 physical KV pages
```

所以 Paged KV 不要求：

```text
一个 Request 的历史 KV
在 HBM 物理连续
```

只要 block table 可以恢复逻辑顺序即可。

---

# 29. `slot_mapping`：当前 K/V 的写地址

对于 position：

```text
position = p
```

在当前普通 paged layout 下可理解为：

```text
logical_block = p // block_size
offset        = p % block_size
physical      = block_table[logical_block]
slot          = physical * block_size + offset
```

`slot_mapping` 就是把本轮每个 current token 映射到这样的 physical slot。

因此：

```text
block_table
= 历史序列怎么找到 physical blocks

slot_mapping
= 当前新 K/V 应该写到哪个 physical slot
```

这是二者最核心的区别。

---

# 30. 一个最重要的具体例子

假设：

```text
block_size = 16
Request A 当前执行 position = 20

block table:
logical 0 → physical 7
logical 1 → physical 18
```

那么：

```text
position 20
↓
logical_block = 20 // 16 = 1
offset        = 20 % 16  = 4
↓
physical_block = 18
↓
slot = 18 * 16 + 4 = 292
```

所以当前模型算出的：

```text
K20 / V20
```

写入：

```text
physical slot 292
```

概念上：

```text
K_cache[18, 4, ...] = K20
V_cache[18, 4, ...] = V20
```

随后 Attention 读取历史 KV 时，并不会要求：

```text
physical block 7 后面必须正好是 18
```

它只需要：

```text
block_table = [7,18]
```

就可以按照逻辑顺序访问历史 KV。

这个例子基本把 Paged KV 的核心闭环全部串起来了。

---

# 31. 为什么 `prepare_attn()` 后面还需要 `model_state.prepare_attn()`

Runner 层先完成：

```text
通用 addressing preparation
```

例如：

```text
block_tables
slot_mappings
positions
seq_lens
```

但是具体 Attention backend 还需要自己的 metadata 格式。

因此继续经过：

```text
CommonAttentionMetadata
↓
AttentionGroup
↓
Backend MetadataBuilder
↓
backend-specific AttentionMetadata
```

当前普通 decoder Qwen3 路径中，笔记追到了 FlashAttention metadata builder 方向。

因此可以理解：

```text
ModelRunner
= 产生通用 execution / addressing facts

ModelState / AttentionGroup / Builder
= 把这些事实适配成具体 backend contract
```

这又是一次“稳定语义与 backend-specific implementation 分离”。

---

# 32. Attention Backend 为什么决定 KV 数据路径的一部分

Paged KV 管理并不意味着：

```text
永远有一个固定 paged_attention_v1/v2 kernel
```

当前 v0.26 已经不应该继续寻找旧的：

```text
paged_attention_v1.cu
paged_attention_v2.cu
```

项目相关路径更接近：

```text
KVCacheManager / BlockPool
→ 继续提供分页 block 管理

ModelRunner
→ 构造 block_table / slot_mapping

active Attention Backend
→ 决定 KV layout / metadata contract / paged read path

对应 cache op
→ 完成 KV write
```

因此：

> **Paged KV 是数据组织方式，不等于某一个固定 kernel 名字。**

---

# 33. 当前 Attention Backend 的确认边界

当前 07 笔记已经确认：

```text
A100 / CUDA / 普通 decoder attention
```

下 `FLASH_ATTN` 是默认高优先级候选之一，并已经沿其实现追了 metadata / write / read 路径。

但是必须保留严格边界：

> **最终当前实验是否真实选中 FLASH_ATTN，需要启动日志 / trace / profiler 进一步动态确认。**

所以本文中的 FlashAttention 主链是：

```text
当前已追踪的 active-candidate implementation path
```

而不是未经动态证据就声称：

```text
本机每一次运行都已经证明使用 FLASH_ATTN
```

---

# 34. KV WRITE：Scheduler 的 block decision 最终怎样落到 HBM

完整写链可以压缩为：

```text
Scheduler.schedule()
↓
num_scheduled_tokens
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
slot_mapping
↓
Model forward
↓
QKV projection
↓
current K/V
↓
KV cache update path
↓
K/V + slot_mapping
↓
cache write op
↓
GPU Paged KV Cache
```

在当前 07 追踪的 FlashAttention 路径中进一步表现为：

```text
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
cache op / CUDA implementation
↓
GPU KV Tensor
```

最重要的不是记函数名，而是理解：

```text
Control Plane 分配的是 page identity
↓
ModelRunner 生成 token-level slot
↓
cache op 才真正写 GPU bytes
```

---

# 35. PAGED KV READ：为什么读取主要依赖 block_table

完整读链：

```text
Scheduler
↓
Request block ownership / sequence progress
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
backend-specific AttentionMetadata
↓
Attention forward
↓
Q
+
key_cache / value_cache
+
block_table
+
seq_lens
+
query_start_loc
↓
paged KV read
↓
Attention output
```

为什么读取不是只靠 `slot_mapping`？

因为：

```text
slot_mapping
回答当前这几个新 token 写哪里
```

而 Attention 需要读：

```text
整个逻辑历史 sequence 的 KV
```

所以需要：

```text
block_table
+
seq_lens
```

去恢复历史 logical sequence。

---

# 36. Prefill / Decode 到数据面以后到底还有什么区别

Scheduler 层统一：

```text
都是 num_scheduled_tokens
```

但 ModelRunner / Attention 仍然看到不同 execution shape。

## Chunked Prefill

可能：

```text
Request B
本轮 q_len = 255
历史 KV + current K/V
```

## Decode

通常：

```text
Request A
本轮 q_len = 1
历史 seq_len 很长
```

所以：

```text
Scheduler abstraction
= 统一 token work

Execution representation
= ragged / variable query lengths

Kernel specialization
= 仍可以针对 q_len、seq_len、prefill/decode behavior 优化
```

因此：

> **统一调度 abstraction，不等于抹掉 workload 差异。**

---

# 37. 执行完成后，状态如何回到 Scheduler

ModelRunner / sampling 得到：

```text
sampled_token_ids
```

通过：

```text
ModelRunnerOutput
```

回到：

```text
Scheduler.update_from_output()
```

Scheduler 根据：

```text
req_id_to_index
```

找到每个 Request 的 generated tokens。

重要职责分界：

```text
ModelRunner
= token producer

Scheduler / Request
= sequence state + lifecycle owner
```

Scheduler 本身不负责做模型 sampling。

---

# 38. 为什么 Prefill 通常没有 generated token

假设 Chunked Prefill：

```text
Prompt = 1000
num_computed_tokens = 256
本轮 schedule 256
```

ModelRunner：

```text
forward positions [256,512)
```

但仍然没有完成 Prompt，所以通常：

```text
generated_token_ids = []
```

此时 Request 的变化主要是：

```text
computed frontier
256 → 512
```

而不是：

```text
sequence length 增加
```

必须严格区分：

```text
Computation progress
= num_computed_tokens grows

Sequence growth
= sample output → num_tokens grows
```

---

# 39. 一个完整 Chunked Prefill step

初始：

```text
Prompt = 1000
num_tokens = 1000
num_computed_tokens = 256
```

Scheduler：

```text
pending = 744
↓
本轮 schedule 256
```

KV control plane：

```text
确保 target KV coverage 能覆盖到 512
必要时新增 physical blocks
```

Scheduler 发出：

```text
SchedulerOutput
```

ModelRunner：

```text
构造 positions [256,512)
↓
slot_mapping
↓
forward
↓
写入这些 token 的 KV
↓
generated_token_ids = []
```

结果回收后：

```text
num_tokens = 1000
num_computed_tokens = 512
```

下一轮：

```text
pending = 488
```

这就是 Chunked Prefill 的稳定推进方式。

---

# 40. 一个完整 Decode step

假设：

```text
Prompt P0...P99 已完成
已经 sample O0

_all_token_ids = [P0 ... P99, O0]
num_tokens = 101
num_computed_tokens = 100
```

注意：

```text
O0 已经“知道”
但 O0 还没有作为 input forward
```

Scheduler：

```text
pending = 1
↓
num_scheduled_tokens = 1
```

ModelRunner：

```text
forward(O0)
↓
写 O0 对应的新 KV
↓
Attention 读取历史 KV
↓
logits
↓
sample O1
```

`ModelRunnerOutput`：

```text
generated_token_ids = [O1]
```

Scheduler：

```text
append_output_token_ids(O1)
```

新状态：

```text
num_tokens = 102
num_computed_tokens = 101
```

下一轮：

```text
pending = 1
```

所以 autoregressive decode 的本质是两个 frontier 交替推进：

```text
compute O0
↓
sample O1
↓
known frontier +1
↓
下一轮 compute O1
```

---

# 41. 为什么 Scheduler 的进度更新看起来是 optimistic

当前笔记已经追到：

```text
schedule-time progress
+
result-time reconciliation
```

即 Scheduler 为了支持：

```text
async scheduling
multiple in-flight batches
pipeline overlap
```

可能在 work 发出时先推进逻辑 execution frontier，而不是每次都等 GPU 完成后才更新。

因此需要：

```text
num_in_flight_tokens
SchedulerOutput
ModelRunnerOutput
update_from_output()
```

共同维护：

```text
计划状态
vs
真实执行结果
```

可以理解：

```text
SchedulerOutput
= 我计划 / 发出了什么

ModelRunnerOutput
= 执行面实际产生了什么

update_from_output()
= reconcile 二者
```

当前学习阶段不需要深入所有 speculative correction / PP branch，但要知道这种 time semantics 存在。

---

# 42. Request finish / preemption 时，控制面和执行面分别发生什么

## Finish

Scheduler：

```text
Request lifecycle end
↓
KVCacheManager free ownership
↓
finished_req_ids
```

MRV2：

```text
finished_req_ids
↓
_remove_request(req_id)
↓
remove execution-side RequestState / model state
```

---

## Preemption

Scheduler：

```text
free current KV ownership
num_computed_tokens reset / adjust
status → PREEMPTED
重新进入 waiting
```

MRV2：

```text
preempted_req_ids
↓
把当前 active execution state remove
```

以后 resume：

```text
Scheduler Request 重新被调度
↓
NewRequestData
↓
MRV2 fresh add execution state
```

因此再强调：

```text
Scheduler Request lifetime
≠
MRV2 active execution-state lifetime
```

---

# 43. KV block free 为什么不等于 GPU bytes 立即清零

控制面 free 的主要含义是：

```text
这个 physical block identity
不再被当前 Request ownership 持有
↓
可以重新进入 free / evictable resource management
```

它不必等价于：

```text
立即 memset 整个 GPU page = 0
```

因此必须区分：

```text
logical/resource lifecycle
vs
physical byte content
```

这对未来做：

```text
KV migration
低比特 pool
cache reuse
```

非常关键。

---

# 44. 为什么 QCache 项目不能只修改一个模块

如果未来实现：

```text
BF16 recent KV
+
INT4 history KV
```

这不是简单：

```text
在 Attention 前 quantize 一下
```

因为当前 vLLM 已经明确拆成控制面与数据面。

至少涉及几个层次。

## 44.1 Logical KV state / policy

需要表达：

```text
哪个 logical block
当前是 BF16 recent
还是 INT4 history
是否迁移中
是否可释放
```

这属于控制面 / resource state 问题。

---

## 44.2 Physical storage

Worker / backend 侧真正需要：

```text
BF16 KV Pool
INT4 packed KV Pool
scale / zero-point metadata
```

这是 GPU storage 问题。

---

## 44.3 Scheduler → Worker protocol

如果 migration / dtype state 会影响 execution，SchedulerOutput 需要能够表达：

```text
本轮新增什么 block
哪些 block 要迁移
哪些 metadata execution side 需要更新
```

这是 control contract 问题。

---

## 44.4 ModelRunner addressing / metadata

现有：

```text
block table
→ physical block ID
```

未来 mixed pool 可能需要进一步表达：

```text
logical block
→ dtype / pool / physical index / scale metadata
```

这取决于最终 QCache 设计，不能在学习阶段提前拍死。

---

## 44.5 KV write

新产生的 K/V 可能先：

```text
写 BF16 recent pool
```

历史 block 满足条件后：

```text
quantize + pack + migrate
```

这里才真正进入 Triton / CUDA data path。

---

## 44.6 History Attention read

Attention 必须能够同时读取：

```text
BF16 recent KV
+
packed INT4 history KV
```

需要：

```text
unpack / dequant
或
直接 low-bit Attention
```

所以最终一定会触及 Attention Backend / kernel contract。

---

# 45. 一张最终完整大图

```text
┌────────────────────────────────────────────────────────────┐
│ Frontend Linux Process                                     │
│                                                            │
│ LLM / LLMEngine                                            │
│       │                                                    │
│       ▼                                                    │
│ SyncMPClient                                               │
└───────┼────────────────────────────────────────────────────┘
        │ ZMQ
════════╪══════════════ OS PROCESS BOUNDARY ══════════════════
        ▼
┌────────────────────────────────────────────────────────────┐
│ EngineCore Linux Process                                   │
│                                                            │
│ EngineCoreProc IS-A EngineCore                             │
│                                                            │
│               ┌──────────────────────┐                     │
│               │      EngineCore      │                     │
│               │   orchestration      │                     │
│               └──────────┬───────────┘                     │
│                          │                                 │
│                          ▼                                 │
│               ┌──────────────────────┐                     │
│               │      Scheduler       │                     │
│               │                      │                     │
│               │ Request lifecycle    │                     │
│               │ token budget         │                     │
│               │ num_scheduled_tokens │                     │
│               └──────────┬───────────┘                     │
│                          │                                 │
│                          ▼                                 │
│               ┌──────────────────────┐                     │
│               │   KVCacheManager     │                     │
│               └──────────┬───────────┘                     │
│                          ▼                                 │
│                     Coordinator                            │
│                          ▼                                 │
│              SingleTypeKVCacheManager                     │
│                          │                                 │
│              req_to_blocks / coverage                     │
│                          ▼                                 │
│                    ┌───────────┐                           │
│                    │ BlockPool │                           │
│                    └─────┬─────┘                           │
│                          │ physical block IDs              │
│                          ▼                                 │
│                  SchedulerOutput                           │
│                          │                                 │
│                          ▼                                 │
│                    UniProcExecutor                         │
│                          ▼                                 │
│                  WorkerWrapperBase                         │
│                          ▼                                 │
│                     CUDA Worker                            │
│                          ▼                                 │
│                 MRV2 GPUModelRunner                        │
│                          │                                 │
│      ┌───────────────────┼────────────────────┐            │
│      ▼                   ▼                    ▼            │
│ Persistent State     Per-step View       KV Addressing     │
│ req_states           InputBatch          BlockTables       │
│ block tables         idx_mapping         block_table       │
│                      positions           slot_mapping      │
│                      query_start_loc                       │
│                      seq_lens                              │
│      └───────────────────┼────────────────────┘            │
│                          ▼                                 │
│                Attention Metadata                          │
│                          ▼                                 │
│                    Model Forward                           │
│                          │                                 │
│             ┌────────────┴────────────┐                    │
│             ▼                         ▼                    │
│          KV WRITE                  KV READ                  │
│ K/V + slot_mapping      Q + block_table + seq_lens        │
│             │                         │                    │
│             ▼                         │                    │
│     GPU Paged KV Cache  ◄─────────────┘                    │
│             │                                              │
│             └────── Attention / logits / sampling          │
│                              │                             │
│                              ▼                             │
│                      ModelRunnerOutput                     │
│                              │                             │
│                              ▼                             │
│                   Scheduler.update_from_output()           │
│                              │                             │
│                              ▼                             │
│                         next State                         │
└────────────────────────────────────────────────────────────┘
```

---

# 46. 每一层只记一句话

如果以后忘了源码，只恢复下面这些句子。

### Frontend / `LLMEngine`

> 接收用户请求，并通过 Client 使用 EngineCore。

### `SyncMPClient`

> Frontend 如何同步访问独立 EngineCore Process。

### `EngineCoreProc`

> 让 EngineCore 具备独立 Process / ZMQ 生命周期能力。

### `EngineCore`

> 持续驱动 schedule → execute → reconcile 的 runtime orchestration。

### `Scheduler`

> 决定本 step 哪些 Request 推进多少 token，并维护 Request 生命周期。

### `KVCacheManager`

> 把 Request 的计算需求转换为 KV resource demand。

### `SingleTypeKVCacheManager`

> 维护某类 KV 的 Request block ownership 和 token→block coverage 规则。

### `BlockPool`

> 管理全局 physical KV block metadata resource。

### `SchedulerOutput`

> Control Plane 给 Execution Plane 的本 step 增量协议。

### `Executor`

> 把 execution plan 映射到具体 execution topology。

### `Worker`

> 当前 rank/device 的执行环境与 ModelRunner owner。

### `MRV2 GPUModelRunner`

> 把 SchedulerOutput 翻译成 GPU 可以执行的数据结构、地址和 metadata。

### `block_table`

> Request logical history block → physical KV block。

### `slot_mapping`

> 当前 token → physical KV write slot。

### Attention Backend

> 消费 backend metadata、KV Tensor 与 page mapping，完成 KV write/read 和 Attention。

---

# 47. 最容易产生的 12 个错误理解

## 错误 1

```text
UniProcExecutor
= 整个 vLLM 单进程
```

正确：

```text
它只描述 EngineCore 内 execution topology。
Frontend / EngineCore 仍可跨 Process。
```

---

## 错误 2

```text
Worker = Linux Process
```

当前 TP=PP=DP=1 UniProc 路径中：

```text
Worker 是 EngineCore Process 内 Python Object。
```

---

## 错误 3

```text
EngineCoreProc 里面又持有一个 EngineCore Object
```

正确：

```text
EngineCoreProc IS-A EngineCore。
```

---

## 错误 4

```text
waiting = Prefill
running = Decode
```

正确：

```text
waiting/running = scheduling lifecycle
Prefill/Decode = execution progress behavior
```

---

## 错误 5

```text
num_tokens = 已经 forward 的 token 数
```

正确：

```text
num_tokens = 当前已知真实 sequence 长度。
```

---

## 错误 6

```text
num_computed_tokens = 已生成 output token 数
```

正确：

```text
它表示 execution frontier。
```

---

## 错误 7

```text
num_scheduled_tokens = 本 Request 总共要算多少
```

正确：

```text
它只是当前 step 的 computation delta。
```

---

## 错误 8

```text
scheduled 1 token
= 一定分配一个新 KV block
```

正确：

```text
只有 target coverage 跨 block boundary 才需要新增 block。
```

---

## 错误 9

```text
KVCacheBlock = GPU page pointer
```

正确：

```text
KVCacheBlock 是 control-plane physical block metadata identity。
```

---

## 错误 10

```text
block_table = 当前 token 写地址
```

正确：

```text
block_table = logical history blocks → physical blocks
slot_mapping = current tokens → write slots
```

---

## 错误 11

```text
MRV2 InputBatch 是 persistent batch
```

正确：

```text
MRV2 persistent request state 与 per-step InputBatch 分离。
```

---

## 错误 12

```text
V1 Scheduler 统一 Prefill/Decode
= 底层没有 Prefill/Decode 差异
```

正确：

```text
Control Plane 统一 token abstraction；
Data/Kernel Plane 仍存在 workload specialization。
```

---

# 48. 对 QCache 最重要的五个架构认知

## 48.1 控制面 metadata 和 GPU bytes 必须分开设计

不能把：

```text
BF16 / INT4 residency
```

只当一个 Tensor dtype 参数。

还涉及：

```text
logical block state
physical pool residency
migration state
scale metadata
```

---

## 48.2 Scheduler 不能直接 launch Triton migration kernel

因为 Scheduler 属于：

```text
Control Plane
```

它应该表达：

```text
what transition should happen
```

真正 GPU migration 属于：

```text
Worker / ModelRunner / backend execution
```

---

## 48.3 BlockPool 的 free/ownership 不等于物理 Tensor 格式管理

未来如果有：

```text
BF16 Pool
INT4 Pool
Scale Pool
```

必须认真设计：

```text
control-plane block identity
如何映射到不同 physical storage
```

不能简单假设现有一个 `block_id` 就天然包含所有 mixed-dtype 信息。

---

## 48.4 `block_table` 与 `slot_mapping` 是未来低比特数据路径的关键边界

因为最终：

```text
history read
依赖 logical→physical mapping

new KV write
依赖 token→physical slot mapping
```

无论是 migration、双池还是低比特 Attention，都绕不开这两个地址体系。

---

## 48.5 Attention Backend 才是真正消费 KV layout 的地方之一

低比特 KV 真正能不能带来收益，最终取决于：

```text
packed storage
↓
write cost
↓
history read pattern
↓
unpack/dequant
↓
Attention kernel
```

所以 KV quantization 不能只停留在 Scheduler / BlockPool。

---

# 49. 当前还没有被动态证明的地方

静态主链已经基本闭环，但以下内容仍应该通过实验补证据。

## 49.1 实际 Attention Backend

确认：

```text
selected backend
cache dtype
KV layout / shape / stride
write symbol
read symbol
```

不能只看默认优先级。

---

## 49.2 token → block → slot 数值关系

至少抽几个 token 验证：

```text
position
logical_block
physical_block
expected_slot
actual_slot_mapping
```

确保：

```text
expected == actual
```

---

## 49.3 KV lifecycle

验证：

```text
new request
→ allocate blocks
→ partial block append
→ decode crossing block boundary
→ finish/free
→ block ID later reused
```

并同时观察：

```text
req_to_blocks
free block count
ref_cnt
```

---

# 50. 当前学习阶段的最终停止点

现在不再需要继续无边界阅读：

```text
Prefix Cache hash internals
KV Connector / LMCache
PP implementation
Speculative Decode
Multimodal
Mamba
Ray
完整 CUDA Graph internals
完整 FlashAttention kernel source
API Server
```

除非项目后续真正需要它们。

当前 vLLM 桥接阶段完成标准应该是：

```text
1. 能从 Request 解释到 Scheduler token decision
2. 能从 scheduled token 解释到 KV block demand
3. 能从 BlockPool 解释到 SchedulerOutput
4. 能从 SchedulerOutput 解释到 MRV2 persistent state
5. 能从 MRV2 解释到 InputBatch
6. 能解释 block_table / slot_mapping
7. 能解释 KV write / history read
8. 能解释 ModelRunnerOutput 如何回到 Scheduler
9. 能指出 QCache 分别影响哪些 control/data plane hooks
10. 用最小 trace 实验验证关键 mapping 与 backend
```

达到这里，就应该进入项目，而不是继续把 vLLM 当成一个需要“读完”的课程。

---

# 51. 最终复习版：30 秒恢复整条链

以后只要记住下面这段即可恢复整个架构：

```text
Frontend 把 Request 送到 EngineCore。

EngineCore 是 runtime 驱动者，
不断做 schedule → execute → reconcile。

Scheduler 持有 Request lifecycle，
通过 num_tokens 与 num_computed_tokens 判断还剩多少 work，
再按 token budget 决定本轮 num_scheduled_tokens。

计算 token 还必须有 KV 空间，
所以 Scheduler 通过 KVCacheManager → concrete manager → BlockPool
把 target token coverage 转成 physical block IDs。

Scheduler 把本轮 request/token/block/lifecycle delta
编码为 SchedulerOutput。

Executor / Worker 把计划送到 MRV2 GPUModelRunner。

MRV2 先同步 persistent RequestState，
再为当前 step gather 出 InputBatch，
构造 block_table 与 slot_mapping，
再转成 Attention Backend metadata。

QKV 计算后：
slot_mapping 决定 current K/V 写到哪个 physical KV slot；
block_table + seq_lens 决定 Attention 如何按逻辑顺序读取历史 Paged KV。

模型产生 logits / sampled token 后，
ModelRunnerOutput 回到 Scheduler.update_from_output()，
更新 Request sequence 与 lifecycle，
形成下一轮 State。
```

如果这段能够不看笔记完整解释出来，当前 vLLM 主链认知就已经建立。

---

# 52. 源文件对应关系

本总纲基于当前已经完成的学习材料重新组织，主要对应：

```text
00-runtime-debug-basics.md
→ Linux Process / PID / PPID / Thread 基础

01-process-architecture-and-control-plane.md
→ LLM → SyncMPClient → launch_core_engines → EngineCore Process

02-enginecore-runtime-layering.md
→ EngineCoreProc / EngineCore / Manager 分层

03-enginecore-to-executor-execution.md
→ EngineCore Runtime 分层补充

04-executor-worker-execution-chain.md
→ Executor / WorkerWrapper / Worker / ModelRunner

05-model-runner-execution-bridge.md
→ Runtime execution bridge 与学习路径校正

06-scheduler-kv-control-plane.md
→ Scheduler / Request / KVCacheManager / BlockPool / SchedulerOutput / result reconciliation

07-model-runner-paged-kv-data-plane.md
→ MRV2 / InputBatch / block_table / slot_mapping / Attention metadata / KV write/read

02-vLLM三周定向桥接学习手册
→ 整体 Gate、学习边界与 QCache hook 目标
```

---

# 53. 一句话结束

> **vLLM V1 的核心不是“Scheduler 调模型”，而是一个长期运行的状态闭环：Scheduler 用 token frontier 做控制决策，KVCacheManager/BlockPool 把计算需求变成 page resource，SchedulerOutput 把控制面增量传给 MRV2，ModelRunner 再把 request/block 语义翻译成 GPU 的 per-step Tensor、block_table 和 slot_mapping，Attention Backend 最终据此写入和读取 Paged KV；执行结果再回到 Scheduler，推动下一轮状态。**
