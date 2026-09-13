# 10-vLLM E6 日志驱动的完整 Runtime 调用链总桥接

> **v2 补充说明（基于 E6 阅读过程中暴露的问题）**
>
> 本版在原文基础上重点补齐五个认知断点：
>
> 1. `launch_core_engines()` 与 `CoreEngineProcManager` 为什么同时存在；
> 2. `EngineCoreProc` 到底是“进程适配器”还是“配置层”，以及 input/output queue / thread 的职责；
> 3. Scheduler 在 `BEFORE_COMMIT / AFTER_COMMIT` 前后的 feasibility、preemption、rollback 与 logical commit 边界；
> 4. 为什么 CUDA/CPU 是异步的，但 E6 Python Trace 看起来仍是顺序的；`batch_queue` 与多 step 到底在流水什么；
> 5. Request finish、async queue drain 与 EngineCore Process shutdown 是三个不同阶段。
>
> 本版继续坚持：**E6 日志直接证明的事实**、**源码可确认的调用关系**、**架构层面的解释/推断**分开描述，避免把 host-side trace 误当成 GPU hardware timeline。

> 固定版本：vLLM v0.26.0  
> Commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> 环境：A100 80GB / Qwen3-0.6B / TP=1 / PP=1 / DP=1 / Offline `LLM` / V1 Engine / V2 Model Runner / `block_size=16` / eager  
> 实验日志：`e6-default-bf16.log`  
> 本文定位：**不再把源码笔记和实验日志分开看，而是把 E6 日志当成“Runtime 执行轨迹”，逐段对应到 Linux Process、Python Object、源码函数、阶段边界和下一跳。**

---

# 0. 为什么需要这份文档

前面的 00～09 笔记已经分别学过：

```text
Linux Process / Thread
↓
Frontend → EngineCore Process
↓
EngineCoreProc / EngineCore
↓
Executor / Worker / ModelRunner
↓
Scheduler / KVCacheManager / BlockPool
↓
SchedulerOutput
↓
MRV2 InputBatch
↓
block_table / slot_mapping
↓
Attention Backend
↓
KV write / paged KV read
```

但只读这些静态文档时，很容易出现一个问题：

> 知道 `step_with_batch_queue()` 是什么，也知道 `prepare_attn()` 是什么，但看到真实日志里的 `[BRIDGE][CORE][CALL_BEGIN]`、`[BRIDGE][ADDR][MAPPING]`、`[BRIDGE][ATTN][KV_WRITE]` 时，不知道自己现在“走到源码的哪一层了”。

所以本文改变阅读方式：

```text
以前：
源码文件 → 函数 → 理解

现在：
真实日志 → 这是哪个阶段？
        → 哪个函数打印？
        → 谁调用这个函数？
        → 它下一步调用谁？
        → 这一行证明什么？
```

目标是让以后看到任何一段 E1～E8 日志时，都能快速定位：

```text
我现在在哪个 Linux Process？
我现在在哪个 Python Object？
我现在处于初始化还是 Request Runtime？
这是控制面、执行准备还是 GPU 数据面？
这行 log 对应哪个函数？
它的上游和下游是什么？
```

---

# 1. 先建立一张“日志 → 源码”的总地图

整个 E6 可以分成两条完全不同的时间线：

```text
============================================================
时间线 A：Initialization
============================================================
Frontend
↓
创建 EngineCore Process
↓
EngineCoreProc.run_engine_core()
↓
EngineCoreProc(...)
↓
EngineCore.__init__()
↓
Executor / Worker / ModelRunner
↓
Model load
↓
Dummy/Profile forward
↓
KV Cache allocation
↓
Warmup forward
↓
Scheduler / batch queue ready

============================================================
时间线 B：Serving Runtime
============================================================
Frontend add_request
↓
ZMQ
↓
EngineCoreProc input thread
↓
input_queue
↓
busy loop
↓
Scheduler.add_request
↓
step_with_batch_queue()
↓
Scheduler.schedule()
↓
KV allocation
↓
SchedulerOutput
↓
Executor
↓
Worker
↓
MRV2
↓
prepare_inputs
↓
prepare_attn
↓
block_table / slot_mapping
↓
Attention
↓
KV write / read
↓
sample
↓
batch_queue
↓
Scheduler.update_from_output()
```

**第一条必须建立的规则：Initialization log 和 Serving Runtime log 不能混着解释。**

---

# 2. 一张最重要的 Process / Object / Function 总图

当前实验不是“所有东西都在一个 Python 进程”。

```text
┌────────────────────────────────────────────────────────────┐
│ Frontend Linux Process                                     │
│ PID = 1563437（E6 CLIENT_SEND 中可见）                     │
│                                                            │
│ LLM                                                        │
│ LLMEngine                                                  │
│ SyncMPClient                                               │
│                                                            │
│ SyncMPClient.add_request()                                 │
│      │                                                     │
│      └──── ZMQ SEND ───────────────────────────────────┐    │
└────────────────────────────────────────────────────────│────┘
                                                         │
======================= OS PROCESS BOUNDARY =============│====
                                                         ▼
┌────────────────────────────────────────────────────────────┐
│ EngineCore Linux Process                                   │
│ PID = 1564298                                              │
│                                                            │
│ EngineCoreProc Object                                      │
│ IS-A EngineCore                                            │
│                                                            │
│ ├── input thread                                           │
│ │   └── process_input_sockets()                            │
│ │                                                         │
│ ├── busy loop                                              │
│ │   └── run_busy_loop()                                    │
│ │       └── self.step_fn()                                 │
│ │           └── step_with_batch_queue()                    │
│ │                                                         │
│ ├── Scheduler Object                                       │
│ │   └── KVCacheManager → Coordinator → BlockPool           │
│ │                                                         │
│ └── UniProcExecutor Object                                 │
│     └── WorkerWrapperBase                                  │
│         └── CUDA Worker                                    │
│             └── GPUModelRunner V2                          │
│                 └── Model / Attention Backend              │
│                     └── FlashAttentionImpl                 │
└────────────────────────────────────────────────────────────┘
```

所以看到：

```text
(EngineCore pid=1564298)
```

首先应该反应：

> 这行 log 已经不在 Frontend Python Process，而是在独立 EngineCore Linux Process 中。

---

# 3. Phase 0：Frontend 是怎么把 EngineCore Process 创建出来的

E6 日志本身从 EngineCore 已经运行后开始大量输出，所以“Process 是怎么来的”主要来自前面的 01/02 静态笔记和 Linux PID 实验。

完整链：

```text
用户脚本
↓
LLM(...)
↓
LLM.__init__()
↓
LLMEngine.from_engine_args()
↓
Executor.get_class(vllm_config)
↓
LLMEngine(...)
↓
EngineCoreClient.make_client()
↓
当前：multiprocess_mode=True, asyncio_mode=False
↓
SyncMPClient
↓
MPClient.__init__()
↓
launch_core_engines()
↓
CoreEngineProcManager
↓
context.Process(
    target=EngineCoreProc.run_engine_core,
    name="EngineCore",
    ...
)
↓
proc.start()
↓
Linux 创建新的 EngineCore Process
↓
EngineCoreProc.run_engine_core(...)
```

真正创建 OS Process 的边界不是：

```python
context.Process(...)
```

而是：

```python
proc.start()
```

`context.Process(...)` 只是创建 Python Process Object；`start()` 才让操作系统真正产生新的 PID，并执行：

```python
EngineCoreProc.run_engine_core(...)
```

因此运行时关系是：

```text
Frontend Process
│
│ proc.start()
▼
EngineCore Process
    │
    └── target = EngineCoreProc.run_engine_core()
```

---


## 3.1 为什么不是 `SyncMPClient → proc.start()`，中间还要有 `launch_core_engines() → CoreEngineProcManager`

这是阅读初始化链时非常容易产生的第一个疑问：

```text
SyncMPClient
↓
launch_core_engines()
↓
CoreEngineProcManager
↓
Process(...)
↓
proc.start()
```

看起来好像多了两层。

最准确的理解不是：

```text
因为 DP，所以专门多了一层
```

而是：

```text
launch_core_engines()
= Engine deployment topology 的选择 / 编排层

CoreEngineProcManager
= 已选择 Local Process 方案以后，
  具体管理这些 Process 生命周期的机制层
```

### 3.1.1 `launch_core_engines()` 回答的是“应该启动什么”

它面对的是部署维度，而不是模型计算维度。

概念上要回答：

```text
当前 Engine 用 Local multiprocessing 还是 Ray？

DP 是否 > 1？
需要几个 EngineCore？

哪些 Engine 是本地启动？
哪些可能由外部系统管理？

是否需要 DP Coordinator？

启动阶段的 handshake / address 怎么组织？
```

所以 DP 的确是它需要处理的一个原因，但不是唯一原因。

当前 E6：

```text
TP=1
PP=1
DP=1
Local multiprocessing
```

最后会收敛到很简单的：

```text
launch_core_engines()
↓
选择 Local EngineCore Process
↓
CoreEngineProcManager
↓
启动 1 个 EngineCore Process
```

如果以后是更复杂拓扑，`launch_core_engines()` 仍然是同一个入口，只是可能选择不同 manager / coordinator / engine count。

因此它是一个典型的：

```text
Policy / Orchestration
```

层。

### 3.1.2 `CoreEngineProcManager` 回答的是“Local Process 具体怎么活”

一旦已经决定：

```text
我要启动 Local Process 版 EngineCore
```

仍然有一组和 inference math 完全不同的问题：

```text
Process Object 怎么创建？
target 是谁？

什么时候 proc.start()？

PID 出现以后，Engine 是否真的 Ready？

Model / CUDA / KV 初始化失败怎么办？

Frontend 退出以后谁负责结束这些子进程？

terminate / join / cleanup 怎么统一处理？
```

这些属于：

```text
Process Lifecycle
```

所以 `CoreEngineProcManager` 的三类核心职责可以记成：

```text
creation
readiness
shutdown
```

其中：

```text
context.Process(...)
= 创建 Python Process Object

proc.start()
= 真正跨过 OS Process boundary，创建 Linux child process

handshake / startup wait
= 确认 child 不只是“活着”，而是 Runtime 已经可用
```

这里要牢记：

```text
Liveness != Readiness
```

EngineCore PID 已经存在，并不代表：

```text
CUDA 已初始化
模型已加载
KV Cache 已建立
IPC 已 Ready
```

### 3.1.3 为什么 Manager 必须在父进程一侧

时间顺序决定了它不能放在 EngineCoreProc 自己里面：

```text
Parent / Frontend Process
↓
创建 child process
↓
child process 才开始执行
↓
child 内部才可能创建 EngineCoreProc Object
```

因此：

> **EngineCoreProc 不可能在自己存在之前负责创建自己所在的 Linux Process。**

所以必须存在 parent-side lifecycle manager：

```text
CoreEngineProcManager
```

来做 create/start/readiness/shutdown。

### 3.1.4 三层不要再混

```text
SyncMPClient
= Frontend 如何和独立 Engine 通信

launch_core_engines()
= 应该部署什么 Engine topology

CoreEngineProcManager
= Local Process topology 具体怎么创建、确认 Ready、结束
```

可以压缩成：

```text
Client Adapter
↓
Deployment Orchestrator
↓
Process Lifecycle Manager
↓
OS Process Boundary
```


# 4. Phase 1：EngineCore Process 的真正入口 `run_engine_core()`

新 Linux Process 启动以后，不是直接进入 `Scheduler.schedule()`。

它先进入：

```text
EngineCoreProc.run_engine_core()
```

可以把这层理解为：

> **EngineCore Linux Process 的 Python main entry。**

核心结构是：

```text
EngineCoreProc.run_engine_core()
↓
engine_core = EngineCoreProc(...)
↓
初始化完成
↓
engine_core.run_busy_loop()
```

而：

```python
class EngineCoreProc(EngineCore)
```

所以 `EngineCoreProc(...)` 只创建一个 Object，不是：

```text
EngineCoreProc Object
↓
再创建一个 EngineCore Object
```

更准确：

```text
一个 EngineCoreProc Object
=
EngineCore 的核心 Runtime 能力
+
EngineCoreProc 的 Process / ZMQ / queue / thread 能力
```

---

# 5. Phase 2：`EngineCoreProc.__init__()` 为什么又进入 `EngineCore.__init__()`

`EngineCoreProc.__init__()` 会做 Process adapter 相关初始化，然后：

```python
super().__init__(...)
```

这里的 `super().__init__()` 不是创建另一个对象，而是在**当前同一个 EngineCoreProc Object 上初始化 EngineCore 父类那部分 state**。

最终对象类似：

```text
EngineCoreProc Object
│
├── input_queue             ← EngineCoreProc
├── output_queue            ← EngineCoreProc
├── ZMQ sockets             ← EngineCoreProc
├── input/output threads    ← EngineCoreProc
│
├── model_executor          ← EngineCore.__init__
├── scheduler               ← EngineCore.__init__
├── KV runtime config       ← EngineCore.__init__
├── batch_queue             ← EngineCore.__init__
└── step_fn                 ← EngineCore.__init__
```

`EngineCore.__init__()` 当前主骨架：

```python
self.model_executor = executor_class(vllm_config)

kv_cache_config = self._initialize_kv_caches(vllm_config)

self.scheduler = Scheduler(...)
```

所以初始化阶段首先会看到 Executor / Worker / ModelRunner / Model load / KV Cache init 的日志，然后真正 Request Runtime 才开始。

---


## 5.1 `EngineCoreProc` 到底是什么：不是“配置层”，而是 Process / IPC Runtime Adapter

原文写：

```text
一个 EngineCoreProc Object
=
EngineCore 的核心 Runtime 能力
+
EngineCoreProc 的 Process / ZMQ / queue / thread 能力
```

这个结论是对的，但还需要把“这些能力为什么存在”解释得更具体。

### 5.1.1 `EngineCore` 与 `EngineCoreProc` 分别解决什么问题

父类 `EngineCore` 关心：

```text
Scheduler
Executor
KV runtime
step / step_with_batch_queue
Request state progression
schedule → execute → reconcile
```

也就是：

> **如果已经有人把 Request 送到我面前，我怎么把推理 Runtime 跑起来。**

`EngineCoreProc` 增加的是：

```text
ZMQ socket
input/output transport
input_queue / output_queue
input/output threads
handshake
signal
shutdown
run_busy_loop()
```

它解决的是：

> **怎样把这个 EngineCore 变成一个长期运行在独立 Linux Process 中、可以和 Frontend 通过 IPC 通信的服务。**

因此最准确的定义是：

```text
EngineCore
= inference runtime core

EngineCoreProc
= process-enabled / ZMQ-enabled EngineCore
```

### 5.1.2 为什么收到 ZMQ 请求后不直接在 socket thread 里调用 Scheduler

E6 ingress trace 已经把这一层暴露出来：

```text
Frontend
[CLIENT_SEND]
↓ ZMQ
EngineCore input thread
[IPC_RECV]
↓
[IPC_ENQUEUE]
↓
input_queue
↓
EngineCore busy loop
[CORE_DEQUEUE]
↓
Scheduler.add_request()
[SCHED_ADD_DONE]
```

这里的关键不是“多一个 queue”，而是隔离两个执行上下文：

```text
I/O thread
= 负责尽快收消息

EngineCore busy loop
= 负责串行推进 Runtime / Scheduler 状态
```

如果 ZMQ receive thread 收到消息后直接修改：

```text
Scheduler.waiting
Scheduler.running
Request state
```

就会把：

```text
通信线程
```

和：

```text
核心 inference state machine
```

直接耦合起来，需要面对更多并发状态同步问题。

当前结构则更像：

```text
IPC producer
↓
thread-safe / process-local queue
↓
Runtime consumer
```

所以 `input_queue` 可以理解为：

> **IPC 世界与 EngineCore state-machine 世界之间的缓冲 / ownership transfer boundary。**

### 5.1.3 output 方向也是同样思想

概念上：

```text
Scheduler / EngineCore 产生 EngineCoreOutputs
↓
output-side queue / transport
↓
ZMQ
↓
Frontend SyncMPClient
↓
用户侧 LLM.generate()
```

也就是说 `EngineCoreProc` 不负责“生成 token 的算法”，而负责：

```text
这些 Runtime 结果怎么跨 Process boundary 送出去
```

### 5.1.4 `EngineCoreProc` 是否负责“配置”？

这里要修正一个容易造成误解的说法：

```text
EngineCoreProc != configuration manager
```

核心配置主要来自：

```text
EngineArgs
↓
VllmConfig
↓
作为 child process 启动参数传入
↓
EngineCoreProc / EngineCore
↓
Executor / Worker / Scheduler / KV runtime
```

`EngineCoreProc` 确实会持有/使用一些 process-local 信息，例如：

```text
IPC addresses
rank/process context
socket/queue wiring
signal/shutdown state
```

但这些更适合叫：

```text
process-local runtime wiring
```

而不是“统一配置系统”。

### 5.1.5 最准确的对象图

```text
EngineCore Linux Process
│
└── engine_core = EngineCoreProc(...)
    │
    ├── inherited EngineCore state
    │   ├── Scheduler
    │   ├── Executor
    │   ├── KV runtime
    │   ├── batch_queue
    │   └── step_fn
    │
    └── EngineCoreProc-specific state
        ├── ZMQ transport
        ├── input_queue
        ├── output_queue
        ├── input/output threads
        ├── run_busy_loop
        ├── handshake
        └── signal / shutdown
```

只有：

```text
一个 EngineCoreProc Object
```

不存在：

```text
EngineCoreProc Object
↓
内部再 new 一个 EngineCore Object
```

`super().__init__()` 只是初始化同一个对象的父类部分。


# 6. 用 E6 启动日志把 EngineCore 初始化链穿起来

E6：

```text
(EngineCore pid=1564298) INFO ... [core.py:116]
Initializing a V1 LLM engine (v0.26.0) with config: ...
```

这行可以定位为：

```text
Linux EngineCore Process
↓
EngineCoreProc(...)
↓
super().__init__()
↓
EngineCore.__init__()
↓
开始构建 inference runtime object graph
```

它的意义不是“开始处理第一个 Request”，而是：

> **当前 EngineCore Process 正在搭建长期存在的 Runtime。**

---

# 7. `parallel_state` 日志对应 Worker/device 初始化阶段

随后：

```text
[parallel_state.py:1615]
world_size=1 rank=0 local_rank=0 ... backend=nccl

[parallel_state.py:1946]
rank 0 ... DP rank 0, PP rank 0, PCP rank 0, TP rank 0
```

这对应初始化调用链中的：

```text
EngineCore.__init__()
↓
self.model_executor = UniProcExecutor(...)
↓
Executor.__init__()
↓
UniProcExecutor._init_executor()
↓
WorkerWrapperBase(...)
↓
init_worker()
↓
Actual CUDA Worker
↓
Worker.init_device()
↓
distributed/device initialization
```

当前：

```text
TP=1
PP=1
DP=1
```

仍然会初始化 distributed/rank abstraction，因为 Worker 层的接口设计并不是“只有多卡才有 rank”。

---

# 8. 为什么是 `UniProcExecutor`，但日志仍来自独立 EngineCore Process

当前：

```text
world_size=1
↓
distributed_executor_backend="uni"
↓
Executor.get_class()
↓
UniProcExecutor
```

但是：

```text
UniProcExecutor
≠ 整个 vLLM 只有一个 Linux Process
```

两个维度：

```text
Frontend / EngineCore Process boundary
→ V1 multiprocessing

EngineCore 内部 Worker execution topology
→ uni / mp / ray / external
```

当前结构：

```text
Frontend Linux Process
        │ ZMQ
        ▼
EngineCore Linux Process
        │
        └── UniProcExecutor Python Object
```

---

# 9. `Using V2 Model Runner` 对应 Worker 创建 ModelRunner

E6：

```text
[gpu_worker.py:378] Using V2 Model Runner
```

它对应：

```text
UniProcExecutor._init_executor()
↓
WorkerWrapperBase.init_worker()
↓
创建 CUDA Worker Object
↓
Worker.init_device()
↓
根据当前配置选择 V2 Model Runner
↓
self.model_runner = GPUModelRunnerV2(...)
```

这条日志的重要性是：

> 后面所有 `[BRIDGE][MRV2]`、`prepare_attn()`、`vllm/v1/worker/gpu/model_runner.py` 的分析，确实属于当前 active runtime，不是读错了 MRV1 文件。

---

# 10. `Loading model from scratch...` 对应 Worker/ModelRunner load_model 链

E6：

```text
[model_runner.py:285] Loading model from scratch...
```

当前初始化链可以继续写成：

```text
UniProcExecutor._init_executor()
│
├── driver_worker.init_worker()
├── driver_worker.init_device()
│      └── CUDA Worker
│          └── GPUModelRunnerV2(...)
│
└── driver_worker.load_model()
       ↓
Worker.load_model()
       ↓
self.model_runner.load_model()
       ↓
Qwen3ForCausalLM 权重真正加载
```

所以：

```text
Loading model from scratch
```

不是 Runtime 每 step 都发生，而是 Engine 初始化一次性的 model lifecycle。

---

# 11. Attention Backend selection 发生在 model/runtime 初始化阶段

E6：

```text
[cuda.py:482]
Using FLASH_ATTN attention backend out of potential backends:
['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']

[flash_attn.py:776]
Using FlashAttention version 2
```

可以挂到：

```text
ModelRunner / model layer initialization
↓
Attention layer 创建 / backend selector
↓
CUDA platform candidate backend selection
↓
validate configuration
↓
FLASH_ATTN selected
↓
FlashAttentionImpl
↓
FA2
```

这发生在正式 Request 前。

所以以后看到：

```text
Using FLASH_ATTN
```

应该把它归到：

```text
Initialization / Backend Binding
```

而不是：

```text
Call0 的 Attention forward
```

---

# 12. 第一组 `_dummy_req_0` 为什么没有 `CALL_BEGIN / Scheduler` 日志

E6：

```text
[BRIDGE][MRV2][EXECUTE_BEGIN]
total_scheduled=256
per_req={'_dummy_req_0': 256}

[ATTN][KV_WRITE]
kv_shape=(0,)
slot_shape=None

[ATTN][KV_READ]
block_table_shape=None
```

这里非常关键。

如果这是真实 Request Runtime，正常应该看到：

```text
CALL_BEGIN
↓
SCHEDULE_BEGIN
↓
Scheduler.schedule
↓
EXEC_SUBMIT_BEGIN
↓
MRV2 EXECUTE_BEGIN
```

但这里直接出现：

```text
MRV2 EXECUTE_BEGIN
```

没有：

```text
CORE CALL
Scheduler
```

这说明它不是 EngineCore busy-loop 驱动的用户 Request transaction，而是 **Engine initialization/profile 路径直接驱动 Worker / ModelRunner 做 synthetic forward**。

其目的主要是 profiling activation/runtime memory，为之后确定可用于 KV Cache 的显存预算服务。

此时：

```text
kv_shape=(0,)
slot_mapping=None
block_table=None
```

正因为真正 KV Cache 尚未完成创建。

所以：

```text
_dummy_req_*
= initialization/profile synthetic request
≠ user request
```

---

# 13. Dummy/Profile 之后才出现真正的 KV Cache capacity

E6：

```text
[gpu_worker.py:560]
Available KV cache memory: 14.61 GiB

[kv_cache_utils.py:2177]
GPU KV cache size: 136,736 tokens
```

这对应 EngineCore 初始化中的：

```text
EngineCore.__init__()
↓
_initialize_kv_caches(vllm_config)
↓
Executor / Worker profile available memory
↓
get_kv_cache_configs(...)
↓
根据 available memory + KV spec
计算 num_blocks
↓
真正 allocate / initialize KV cache
```

当前：

```text
136736 tokens / block_size16
= 8546 blocks
```

后面 Attention runtime 果然看到：

```text
kv_shape=(8546,8,16,256)
```

因此初始化日志形成一个前后闭环：

```text
profile
↓
可用 KV memory = 14.61 GiB
↓
capacity = 136736 tokens
↓
num_blocks = 8546
↓
Attention KV Tensor first dim = 8546
```

---

# 14. `_warmup_0_`：KV 已创建后的模型热身路径

之后：

```text
[MRV2][EXECUTE_BEGIN]
total_scheduled=2
per_req={'_warmup_0_': 2}
```

这仍然不是用户 Request。

但与 `_dummy_req_0` 不同的是，此时已经看到：

```text
[ADDR][BLOCK_TABLE] block_table=[1]
[ADDR][MAPPING] position=0 → slot16
[ATTN][KV_WRITE] kv_shape=(8546,8,16,256)
[ATTN][KV_READ] block_table=[1,...]
```

说明：

```text
_dummy_req_
→ KV 尚未正式可用的 profile

_warmup_
→ KV cache 已创建，真实 addressing/data path 热身
```

依然没有用户 runtime 的：

```text
CLIENT_SEND
CALL_BEGIN
Scheduler.queue snapshot
```

因此不要把 warmup 中的 block1 当成用户 Request 之后“继承”的 block ownership。

---

# 15. `CORE INIT`：EngineCore Runtime 的执行模式终于固定

初始化末尾：

```text
[BRIDGE][CORE][INIT]
async_scheduling=True
use_v2_model_runner=True
pp_size=1
max_concurrent_batches=2
batch_queue_size=2
batch_queue_enabled=True
step_fn=step_with_batch_queue
```

这条 trace 对应 `EngineCore.__init__()` 里 runtime orchestration state 已经建立。

核心推导：

```text
PP=1
+
V2 Model Runner
+
async_scheduling=True
↓
max_concurrent_batches=2
↓
batch_queue_size=2
↓
batch_queue enabled
↓
self.step_fn = self.step_with_batch_queue
```

因此从这一刻以后，真正 Request 到来时：

```text
EngineCoreProc.run_busy_loop()
```

不会调用：

```text
EngineCore.step()
```

而是动态调用：

```text
self.step_fn()
→ step_with_batch_queue()
```

---

# 16. 到这里：Initialization 主链完整闭环

可以把 E6 启动日志压成：

```text
Frontend LLM(...)
↓
SyncMPClient / launch_core_engines
↓
CoreEngineProcManager
↓
proc.start()
════════════════ Process Boundary ════════════════
↓
EngineCoreProc.run_engine_core()
↓
EngineCoreProc(...)
↓
super().__init__()
↓
EngineCore.__init__()
│
├── UniProcExecutor
│    ↓
│  WorkerWrapperBase
│    ↓
│  CUDA Worker
│    ↓
│  GPUModelRunnerV2
│    ↓
│  model load
│    ↓
│  Attention backend = FLASH_ATTN / FA2
│
├── profile dummy forward
│    ↓
│  determine available memory
│
├── _initialize_kv_caches()
│    ↓
│  14.61 GiB
│    ↓
│  136736 token capacity
│    ↓
│  8546 blocks
│
├── warmup model
│
├── Scheduler(...)
│
└── step_fn = step_with_batch_queue
```

这时 Engine 才真正 ready。

---

# 17. Phase 3：真实 Request 从 Frontend 到 Scheduler ownership

现在进入 E6 的真正用户 Request。

首先脚本打印：

```text
prompt_tokens: 31
block_size: 16
prompt_blocks: 2
prompt_remainder: 15
```

随后：

```text
[BRIDGE][INGRESS][CLIENT_SEND]
pid=1563437
req=0-a2c39288
```

注意 PID：

```text
1563437
```

不是 EngineCore PID `1564298`。

所以这条 log 在：

```text
Frontend Linux Process
```

对应函数：

```text
SyncMPClient.add_request()
↓
self._send_input(ADD, request)
```

`CLIENT_SEND` 表示：

> Frontend 正式把这个 Request 交给多进程 transport。

---

# 18. `IPC_RECV / IPC_ENQUEUE`：进入 EngineCoreProc input thread

接下来：

```text
(EngineCore pid=1564298)
[BRIDGE][INGRESS][IPC_RECV]
thread=Thread-2 (process_input_sockets)
req=0-a2c39288
```

对应函数：

```text
EngineCoreProc.process_input_sockets()
```

调用环境：

```text
EngineCore Linux Process
└── input thread
    └── process_input_sockets()
```

这里接收 ZMQ multipart，decode Request。

紧接着：

```text
[IPC_ENQUEUE]
input_qsize_after=1
```

对应：

```python
self.input_queue.put_nowait((request_type, request))
```

所以：

```text
CLIENT_SEND
↓ ZMQ
IPC_RECV
↓
IPC_ENQUEUE
```

只证明：

> Request 已经到 EngineCore Process 并进入 local input_queue。

**还没有进入 Scheduler。**

---

# 19. 为什么还需要 `CORE_DEQUEUE / SCHED_ADD_DONE`

接下来：

```text
[BRIDGE][INGRESS][CORE_DEQUEUE]
req=0-a2c39288
known_before=[]
```

对应：

```text
EngineCoreProc busy-loop context
↓
从 local input_queue 取 request
↓
EngineCoreProc._handle_client_request()
```

ADD 分支：

```python
self.add_request(req, request_wave)
```

然后：

```text
[SCHED_ADD_DONE]
known_after=['0-a2c39288']
```

这表示：

```text
EngineCore.add_request(...)
↓
Scheduler.add_request(...)
↓
Request 正式进入 Scheduler.requests / waiting ownership
```

因此最准确的 Ingress 状态机是：

```text
CLIENT_SEND
= Frontend 已发送

IPC_RECV
= EngineCore input thread 已收到

IPC_ENQUEUE
= 已进入 local input_queue

CORE_DEQUEUE
= busy-loop 开始处理

SCHED_ADD_DONE
= Scheduler 正式认识这个 Request
```

---

# 20. Phase 4：`CALL_BEGIN` 到底在哪个函数、代表什么

现在来到用户最关心的：

```text
[BRIDGE][CORE][CALL_BEGIN]
call=0
path=step_with_batch_queue
queue_len=0
queue_capacity=2
scheduler_has_requests=True
```

它应该理解为：

```text
EngineCoreProc.run_busy_loop()
↓
完成本轮 input_queue drain
↓
发现 Scheduler 有 Request / batch_queue 可能有 outstanding work
↓
调用 self.step_fn()
↓
当前 step_fn = step_with_batch_queue
↓
进入 EngineCore.step_with_batch_queue()
↓
CALL_BEGIN
```

所以：

> `[CORE][CALL_BEGIN]` 不是 `run_busy_loop()` 本身的入口；它是 **busy loop 调用本轮 EngineCore runtime step 后，`step_with_batch_queue()` 这一轮 transaction 的入口标记**。

这里的 `call=0` 是 instrumentation 自己的 EngineCore step-call 序号，不是 Request ID，也不是 GPU kernel 序号。

`queue_len=0` 表示当前还没有 outstanding batch transaction。

`scheduler_has_requests=True` 表示 Scheduler 现在拥有需要推进的 Request。

---

# 21. `SCHEDULE_BEGIN`：正式进入 Scheduler control plane

随后：

```text
[CORE][SCHEDULE_BEGIN]
call=0
queue_len=0
```

它对应 `step_with_batch_queue()` 内：

```text
准备调用
self.scheduler.schedule(...)
```

所以链路已经是：

```text
EngineCoreProc.run_busy_loop()
↓
EngineCore.step_with_batch_queue()
↓
CALL_BEGIN
↓
SCHEDULE_BEGIN
↓
Scheduler.schedule()
```

从这里开始进入 **CPU control plane**。

---

# 22. `QUEUE_SNAPSHOT`：我们已经进入 `Scheduler.schedule()` 内部

日志：

```text
[SCHED][QUEUE_SNAPSHOT]
sched_call=0
known_requests=['0-a2c39288']
running=[]
waiting=['0-a2c39288']
```

Trace 位置：

```text
vllm/v1/core/sched/scheduler.py
Scheduler.schedule()
```

意义：

```text
Scheduler.requests
= [R]

waiting
= [R]

running
= []
```

这是新 Request 第一次调度，所以它还没有 admission 成为 running。

这行属于：

```text
Control Plane / Scheduler state observation
```

不是 ModelRunner batch。

---

# 23. `[PREFIX][LOOKUP]` 为什么 Prefix Cache OFF 仍有日志

E6：

```text
[PREFIX][LOOKUP]
status=WAITING
num_tokens=31
num_computed_before=0
local_hit_tokens=0
hit_block_ids=None
```

实验设置：

```text
enable_prefix_caching=False
```

因此结果自然是：

```text
local_hit_tokens=0
```

这一步在 waiting admission 逻辑附近，核心语义是：

```text
新 Request 在 schedule 时
先确定是否已有 computed prefix
↓
本实验没有
↓
有效 computed frontier 仍然是 0
```

所以接下来整个 31-token Prompt 都需要自己计算。

---

# 24. `BEFORE_COMMIT`：Scheduler + KV 控制面已经做完“本轮决策”

E6：

```text
[SCHED][BEFORE_COMMIT]
kind=NEW
num_tokens=31
num_computed=0
num_in_flight=0
pending=31
scheduled=31
new_block_ids=([1,2],)
```

这行非常重要。

它不是 Scheduler 刚开始考虑这个 Request，而是已经经历了：

```text
Scheduler.schedule()
↓
计算 candidate token work
↓
pending = 31
↓
token_budget 允许 31
↓
KVCacheManager.allocate_slots(...)
↓
Coordinator
↓
SingleTypeKVCacheManager
↓
BlockPool
↓
分配 physical block 1,2
↓
形成 num_scheduled_tokens=31
↓
BEFORE_COMMIT trace
```

因此：

```text
new_block_ids=[1,2]
```

是 **KV 控制面已经完成 physical block ownership 决策** 的证据。

但这时还没有进入 ModelRunner，也还没有计算 `slot_mapping`。

---

# 25. 为什么 31 token 需要两个 physical blocks

```text
block_size=16
prompt_tokens=31
```

所以：

```text
logical block0
positions 0...15

logical block1
positions 16...30
```

需要：

```text
ceil(31/16)=2 blocks
```

Scheduler/KV control plane 分配：

```text
logical block0 → physical block1
logical block1 → physical block2
```

此刻只是 ownership：

```text
Request R owns physical blocks [1,2]
```

还不是 GPU token-level address。

---

# 26. `AFTER_COMMIT`：为什么 GPU 还没跑，`num_computed` 已经变成31

日志：

```text
[SCHED][AFTER_COMMIT]
scheduled=31
num_tokens=31
placeholders=1
num_computed=31
num_in_flight=31
```

对应 Scheduler 的 schedule-time commit：

```text
_update_after_schedule()
```

它做的是逻辑状态提交：

```text
num_computed_tokens += scheduled
num_in_flight_tokens += scheduled
维护 output placeholder
```

所以：

```text
BEFORE:
computed=0
in_flight=0

AFTER:
computed=31
in_flight=31
```

**这不是说 GPU 已经执行完31个 token。**

更准确：

> Scheduler 已经把这31个 token 当成“已提交出去、不能在下一轮重复 schedule 的逻辑 work”。

这是 async scheduling 能继续 ahead-of-result scheduling 的基础。

---


## 26.1 调度“成功”、preemption、rollback 与 `BEFORE_COMMIT / AFTER_COMMIT` 的精确边界

阅读 E6 Call0 时很容易产生一个感觉：

```text
BEFORE_COMMIT
↓
AFTER_COMMIT
↓
SCHEDULE_END
```

是不是说明“到了这里调度成功已经是必然的”？

这个判断需要分层。

### 26.1.1 在 Scheduler 正常资源控制语义里：大体是的

`BEFORE_COMMIT` 不是刚开始尝试 schedule Request。

在到达这里之前，当前 request 的候选 work 已经经历了类似：

```text
计算 pending / candidate tokens
↓
token budget 裁剪
↓
KVCacheManager.allocate_slots(...)
↓
如果当前 KV capacity 不允许
    ↓
    Scheduler policy 处理等待 / preemption / retry
↓
形成当前可执行的 token 数与 new block ids
↓
BEFORE_COMMIT
```

所以在 E6：

```text
scheduled=31
new_block_ids=([1,2],)
```

说明控制面已经得到了一份可提交的 decision。

### 26.1.2 KV allocation “失败”发生在 Scheduler 内部，不是到 GPU 再失败

资源压力下可能出现：

```text
allocate_slots(...)
↓
None / 当前不可满足
```

这类正常 capacity failure 不应该形成一个“不可能执行”的 `SchedulerOutput` 再交给 ModelRunner。

而是 Scheduler 内部处理：

```text
选择 victim
↓
preempt
↓
释放 / 调整 KV ownership
↓
撤销 tentative scheduling bookkeeping
↓
恢复 token budget
↓
retry / 重新形成最终 schedule decision
```

因此：

```text
preemption / rollback
```

属于：

```text
Scheduler.schedule() 内部控制流
```

而：

```text
SCHEDULE_END
```

表示最终 `SchedulerOutput` 已经形成并跨到 execution side。

### 26.1.3 为什么叫 tentative decision / rollback

考虑同一 Scheduler step：

```text
token_budget=256

先暂定 A = 100 tokens
↓
再处理 B
↓
发现当前 KV capacity 不足
↓
Scheduler 决定 preempt A
```

那么 A 之前占用的 step-level bookkeeping 必须撤销，例如概念上：

```text
num_scheduled_tokens[A]
req_to_new_blocks[A]
本 step token budget
scheduled_running_reqs 中的记录
```

否则最终 `SchedulerOutput` 会带着一份已经不成立的计划。

所以 Scheduler 的形成过程更像：

```text
candidate
↓
feasibility / resource check
↓
可能 rollback / retry
↓
final decision
↓
commit
```

### 26.1.4 `AFTER_COMMIT` 仍然不等于“GPU 执行成功”

E6：

```text
AFTER_COMMIT
num_computed=31
num_in_flight=31
```

这里 commit 的是：

```text
Scheduler logical state
```

即：

> 这 31 个 token 已经提交给 execution pipeline，下一轮 Scheduler 不应该把它们重复 schedule。

它不是：

```text
CUDA kernel 已完成
KV 已物理写完
ModelRunnerOutput 已 reconcile
```

所以需要严格区分：

```text
Scheduler feasibility success
≠ execution success
≠ result reconciliation success
```

### 26.1.5 `AFTER_COMMIT` 以后还可能出现什么“失败”

至少要分成两类：

```text
A. 正常资源不足 / admission failure
   → Scheduler 内部处理
   → wait / preempt / rollback / retry

B. execution/runtime failure
   → CUDA error / Worker exception / backend crash 等
   → 已经不是普通 scheduling policy 可以消化的问题
```

E6 没有出现 B。

### 26.1.6 “预期下一轮”究竟预期了什么

Async scheduling 下出现：

```text
placeholders=1
```

不能理解成 Scheduler 在预测：

```text
下一 token 的 token_id 是什么
```

Scheduler 只是在逻辑上知道：

```text
这一批 forward / sample 会产生一个可供后续 position 消费的 output token
```

所以可以提前预留：

```text
next output position / dependency
```

更准确叫：

```text
logical-ahead scheduling based on outstanding output placeholder
```

而不是：

```text
token-value prediction
```

真实 sampled token 仍然要等后面的：

```text
ModelRunnerOutput
↓
Scheduler.update_from_output()
↓
TOKEN_APPEND
```

才正式进入 Request 的真实 token sequence。


# 27. `SCHEDULE_END`：`SchedulerOutput` 已经形成

```text
[CORE][SCHEDULE_END]
call=0
total_scheduled=31
per_req={'0-a2c39288':31}
```

它对应：

```text
Scheduler.schedule()
↓
return SchedulerOutput
↓
回到 EngineCore.step_with_batch_queue()
```

此时跨越关键 protocol boundary：

```text
Control Plane
Scheduler / KVCacheManager / BlockPool
        │
        │ SchedulerOutput
        ▼
Execution Plane
Executor / Worker / ModelRunner
```

后面的执行侧不会重新决定：

```text
“要不要给 R 两个 block？”
```

它消费的就是 Scheduler 已决定好的结果。

---

# 28. `EXEC_SUBMIT_BEGIN`：EngineCore 开始把 SchedulerOutput 交给 Executor

```text
[CORE][EXEC_SUBMIT_BEGIN]
call=0
total_scheduled=31
```

函数位置仍然是：

```text
EngineCore.step_with_batch_queue()
```

下一句主调用：

```python
self.model_executor.execute_model(
    scheduler_output,
    non_block=True,
)
```

当前 `self.model_executor` 是：

```text
UniProcExecutor Object
```

所以真正调用链：

```text
EngineCore.step_with_batch_queue()
↓
UniProcExecutor.execute_model(scheduler_output)
↓
collective_rpc("execute_model")
↓
run_method(driver_worker,...)
↓
WorkerWrapperBase.execute_model()
↓
Actual CUDA Worker.execute_model()
↓
GPUModelRunnerV2.execute_model()
```

---

# 29. `MRV2 EXECUTE_BEGIN`：说明 SchedulerOutput 已经真正抵达 ModelRunner

日志：

```text
[BRIDGE][MRV2][EXECUTE_BEGIN]
total_scheduled=31
per_req={'0-a2c39288':31}
```

位置：

```text
vllm/v1/worker/gpu/model_runner.py
GPUModelRunner.execute_model()
```

所以日志已经动态把这条静态链证明了：

```text
SchedulerOutput(total=31)
↓
EngineCore EXEC_SUBMIT_BEGIN(total=31)
↓
Executor / Worker bridge
↓
MRV2 EXECUTE_BEGIN(total=31)
```

`31` 在三层保持一致，说明 execution contract 没有被重新解释成别的 work 数量。

---

# 30. MRV2 进入后先做什么：不是立刻 Qwen forward

`GPUModelRunnerV2.execute_model()` 可以先拆成：

```text
SchedulerOutput
↓
① synchronize persistent request state
   finish / add / update
↓
② apply staged block-table writes
↓
③ prepare_inputs()
   current-step InputBatch
↓
④ prepare_attn()
   block_tables + slot_mappings
↓
⑤ model_state.prepare_attn()
   backend-specific metadata
↓
⑥ model forward
↓
⑦ sampling state
```

E6 新增的 `[ADDR]` trace 正好位于第④阶段。

---

# 31. `[ADDR][BATCH]` 对应 `GPUModelRunner.prepare_attn()`

日志：

```text
[ADDR][BATCH]
num_reqs=1
num_tokens=31
group=0
block_size=16
```

它的直接函数上下文：

```python
def prepare_attn(self, input_batch):
    block_tables = self.block_tables.gather_block_tables(...)

    slot_mappings = self.block_tables.compute_slot_mappings(...)

    # TRACE HERE

    return block_tables, slot_mappings
```

所以：

```text
[ADDR][...]
```

不在 Scheduler，也不在 Attention kernel。

它属于：

> **ModelRunner execution-preparation / KV addressing plane。**

这层的任务是把 Scheduler 给出的 physical block ownership 翻译成 GPU 可消费 metadata。

---

# 32. `[ADDR][BLOCK_TABLE]`：Scheduler 的 block ownership 已进入数据面

日志：

```text
[ADDR][BLOCK_TABLE]
req=0-a2c39288
block_table=[1,2]
```

这和 Scheduler：

```text
new_block_ids=([1,2],)
```

对上。

但语义发生了转化：

```text
Scheduler / KV control plane：
R 获得 physical blocks [1,2]

ModelRunner：
logical block0 → physical1
logical block1 → physical2

表示为：
block_table=[1,2]
```

所以 `block_table` 是：

```text
logical sequence block index
→ physical KV block id
```

的 per-step execution view。

---

# 33. `gather_block_tables()` 在这里做什么

MRV2 的 persistent block state 按稳定 `req_idx` 保存。

当前 step 的 batch row 不一定等于 persistent row。

因此：

```text
persistent BlockTables
        +
idx_mapping
        ↓
gather_block_tables()
        ↓
current batch block table
```

单 Request E6 看起来只是：

```text
[1,2]
```

但 production continuous batching 中，这个 gather 是：

> 从稳定 request-indexed storage 中抽出本轮真正参与执行的 Request rows。

---

# 34. `[ADDR][MAPPING]` 对应 `compute_slot_mappings()`

这就是用户最关心的第二个例子。

日志：

```text
[ADDR][MAPPING]
position=16
logical_block=1
physical_block=2
offset=0
expected_slot=32
actual_slot=32
match=True
```

函数调用链：

```text
GPUModelRunner.prepare_attn()
↓
self.block_tables.compute_slot_mappings(
    input_batch.idx_mapping,
    input_batch.query_start_loc,
    input_batch.positions,
    ...
)
↓
_compute_slot_mappings_kernel Triton kernel
```

当前 `CP_SIZE=1` 时 kernel 真实公式：

```text
block_indices = positions // block_size
block_offsets = positions % block_size

block_numbers = block_table[req][block_indices]

slot_ids = block_numbers * block_size + block_offsets
```

也就是：

```text
logical_block = position // 16
physical_block = block_table[logical_block]
offset = position % 16
slot = physical_block*16 + offset
```

---

# 35. E6 Call0 的 31 条 MAPPING 实际在证明什么

可以不用逐条死记，只分两段：

## physical block1

```text
position 0...15
logical block 0
physical block 1
slot 16...31
```

例如：

```text
position0
→ logical0
→ physical1
→ offset0
→ slot16

position15
→ logical0
→ physical1
→ offset15
→ slot31
```

## physical block2

```text
position 16...30
logical block 1
physical block 2
slot 32...46
```

边界：

```text
position15 → block1 slot31
position16 → block2 slot32
```

全部：

```text
expected_slot == actual_slot
```

所以 Phase E 真正动态证明：

> Scheduler 的 block ownership 不是停留在“block ID metadata”，ModelRunner 已经把它精确翻译成每个 scheduled token 的 physical KV slot。

---

# 36. `prepare_attn()` 后为什么还不直接调用 FlashAttention

`prepare_attn()` 返回：

```text
block_tables
slot_mappings
```

随后 MRV2：

```text
build_slot_mappings_by_layer(slot_mappings,...)
↓
model_state.prepare_attn(
    input_batch,
    block_tables,
    slot_mappings,
    ...
)
↓
backend-specific AttentionMetadata
```

分工：

```text
GPUModelRunner.prepare_attn()
= 通用 Paged KV addressing

ModelState.prepare_attn()
= 把通用 metadata 转成 backend 所需 metadata
```

最终在 model forward 时，通过 forward context 让每个 Attention layer 能取回：

```text
当前 layer 的 slot_mapping
当前 layer 的 attn_metadata
当前 layer 的 kv_cache Tensor
```

---

# 37. 进入 Qwen model forward 后：为什么同时有 WRITE 和 READ

每一层 Attention 对当前 token 都需要完成两件事：

```text
1. 当前 token 新生成的 K/V 要写入 KV Cache
2. 当前 query 要读取整个历史 KV 做 Attention
```

所以数据面天然分成：

```text
WRITE PATH
current K/V → exact physical slot

READ PATH
current Q → historical logical sequence → physical KV blocks
```

E6 正好同时打了两条 trace。

---

# 38. `[ATTN][KV_WRITE]` 对应哪个函数

日志：

```text
[ATTN][KV_WRITE]
layer=model.layers.0.self_attn.attn
impl=FlashAttentionImpl
key_shape=(31,8,128)
value_shape=(31,8,128)
slots=[16...46]
```

Trace 放在 generic Attention KV update dispatch 附近：

```text
vllm/model_executor/layers/attention/attention.py
unified_kv_cache_update()
```

上游模型调用：

```text
Qwen Attention layer forward
↓
Attention.forward(...)
↓
unified_kv_cache_update(key,value,layer_name)
```

函数内部：

```text
get_attention_context(layer_name)
↓
取到：
- attn_layer
- kv_cache
- layer_slot_mapping
↓
attn_layer.impl.do_kv_cache_update(...)
```

当前 `impl`：

```text
FlashAttentionImpl
```

所以真正下一跳：

```text
FlashAttentionImpl.do_kv_cache_update()
```

---

# 39. 为什么 `[KV_WRITE] slots` 和 `[ADDR][MAPPING]` 完全一致非常重要

ADDR：

```text
ModelRunner 计算：
slots=[16,17,...,46]
```

Attention：

```text
KV_WRITE 实际收到：
slots=[16,17,...,46]
```

这把两篇静态笔记之间原本的“断层”补上了：

```text
Scheduler
new_block_ids=[1,2]
↓
ModelRunner
block_table=[1,2]
↓
compute_slot_mappings
slots=[16...46]
↓
forward context
↓
unified_kv_cache_update
↓
FlashAttentionImpl
同样 slots=[16...46]
```

所以不是：

```text
ModelRunner 算了一份 slot mapping
Attention 又自己重新算一份
```

而是：

> **ModelRunner 产生的 exact write address 被一路传到实际 Attention Backend write path。**

---

# 40. `[KV_LAYOUT]` 对应 `FlashAttentionImpl.do_kv_cache_update()` 内部

日志：

```text
[ATTN][KV_LAYOUT]
kv_shape=(8546,8,16,256)
kv_stride=(32768,256,2048,1)
key_cache_shape=(8546,16,8,128)
value_cache_shape=(8546,16,8,128)
```

对应源码：

```python
key_cache, value_cache = (
    kv_cache.transpose(1,2)
    .split(self.head_size, dim=-1)
)
```

所以这行 trace 的位置已经进一步深入到 backend-specific data layout。

调用链：

```text
unified_kv_cache_update()
↓
FlashAttentionImpl.do_kv_cache_update()
↓
kv_cache.transpose(1,2)
↓
split(K,V)
↓
KV_LAYOUT trace
↓
reshape_and_cache_flash(...)
```

---

# 41. 真正 KV write 最终是谁做

`FlashAttentionImpl.do_kv_cache_update()` 最终：

```python
reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    ...
)
```

再向下是底层 op。

所以完整 write path：

```text
Qwen Attention
↓
Attention.forward
↓
unified_kv_cache_update
↓
get_attention_context
↓
FlashAttentionImpl.do_kv_cache_update
↓
reshape_and_cache_flash
↓
底层 CUDA cache op
↓
HBM KV Cache
```

E6 没有 trace CUDA instruction，但 Python/backend call chain 已闭环。

---

# 42. `[ATTN][KV_READ]` 对应哪个函数

日志：

```text
[ATTN][KV_READ]
impl=FlashAttentionImpl
query_shape=(31,16,128)
block_table0=[1,2,0,...]
seq_lens=[31]
query_start_loc=[0,31]
max_query_len=31
max_seq_len=31
```

Trace 放在：

```text
vllm/model_executor/layers/attention/attention.py
unified_attention_with_output()
```

调用链：

```text
Attention.forward()
↓
unified_attention_with_output(...)
↓
get_attention_context(layer_name)
↓
得到：
- attn_metadata
- Attention layer
- kv_cache
↓
self.impl.forward(...)
↓
FlashAttentionImpl.forward()
```

所以：

```text
[KV_READ]
```

不是 kernel 执行后的结果，而是：

> **进入具体 Attention Backend forward 之前，已经准备好的 paged-read metadata 快照。**

---

# 43. 为什么 WRITE 用 `slot_mapping`，READ 用 `block_table`

这是 E6 最应该形成的长期认知。

WRITE 当前只需要回答：

> 当前这31个新 K/V 各自写到哪里？

所以：

```text
slot_mapping[token]
→ exact physical slot
```

READ 需要回答：

> 这个 Request 的整个历史 sequence 分散在哪些 physical pages？有效历史长度是多少？

所以：

```text
block_table
+
seq_lens
+
query metadata
```

例如 Prefill：

```text
WRITE:
slots 16...46

READ:
block_table=[1,2]
seq_len=31
```

Decode：

```text
WRITE:
slot=[47]

READ:
block_table=[1,2]
seq_len=32
```

这就是：

```text
WRITE = token-level address
READ  = request-level paged address space
```

---

# 44. `EXEC_SUBMIT_RETURN`：为什么 `future_done=True` 不代表整个 Call 完成

Call0 Attention 结束后：

```text
[CORE][EXEC_SUBMIT_RETURN]
future_type=Future
future_done=True
```

这回到了：

```text
EngineCore.step_with_batch_queue()
```

说明：

```text
self.model_executor.execute_model(...)
```

已经返回 host/runtime Future。

但不要推导成：

```text
GPU 所有硬件工作已经全局 synchronize 完成
```

它只说明当前 Executor execute-side Future 在这个 abstraction 上已 ready。

MRV2 sampling 还有单独路径。

---

# 45. `SAMPLE_SUBMIT_BEGIN / MRV2 SAMPLE_BEGIN`

日志：

```text
[CORE][SAMPLE_SUBMIT_BEGIN]
↓
[MRV2][SAMPLE_BEGIN]
↓
[CORE][SAMPLE_SUBMIT_RETURN]
future_type=AsyncOutputFuture
future_done=False
```

对应：

```text
EngineCore.step_with_batch_queue()
↓
model_executor / worker 的 sample_tokens 路径
↓
GPUModelRunnerV2.sample_tokens()
```

当前 V2 Runtime 把：

```text
execute_model
```

和：

```text
sample_tokens
```

拆成两个阶段。

所以不能理解成：

```text
execute_model()
= forward + sampling + Scheduler result update 全部完成
```

更接近：

```text
execute side
+
sample side
+
later reconciliation
```

---

# 46. `QUEUE_PUSH`：batch_queue 存的不是 Request

Call0：

```text
[QUEUE_PUSH]
queue_len=1
queue_capacity=2
scheduled=31
```

这里 push 的概念不是：

```text
Request R 等待执行
```

R 已经 submit 了。

更准确 queue item 是 outstanding batch transaction，包含类似：

```text
SchedulerOutput
+
execution future / sample future
+
用于之后 reconcile 的 batch context
```

所以：

```text
batch_queue
= 已提交但尚未完成 Scheduler reconciliation 的 batch transactions
```

不是 waiting request queue。

---

# 47. `YIELD_WITHOUT_CONSUME`：为什么 Call0 没有立刻 `update_from_output()`

```text
[YIELD_WITHOUT_CONSUME]
queue_len=1
reason=queue_not_full
```

当前 queue capacity=2。

Call0 submit 后：

```text
queue=[Call0]
```

还没满。

Async scheduling 的目的就是允许：

```text
Call0 已 submit
↓
暂不 consume Call0 result
↓
进入 Call1
↓
继续 schedule / submit 下一份 work
```

因此 Call0 结束点不是：

```text
Scheduler.update_from_output(Call0)
```

而是：

```text
transaction outstanding in batch_queue
```

---


## 47.1 为什么 CPU/GPU 明明异步，E6 日志却看起来“先把 GPU 全走完，再进入下一轮 CPU”

这是 E6 最容易产生错误直觉的一点。

看到：

```text
[MRV2][EXECUTE_BEGIN]
[ADDR]...
[KV_WRITE]
[KV_LAYOUT]
[KV_READ]
[EXEC_SUBMIT_RETURN]
[SAMPLE_SUBMIT_BEGIN]
[SAMPLE_BEGIN]
[SAMPLE_SUBMIT_RETURN]
[QUEUE_PUSH]
[YIELD_WITHOUT_CONSUME]
```

很容易脑补成：

```text
CPU 调用 GPU
↓
GPU 已完整计算结束
↓
CPU 才继续
```

**E6 Trace 并不能支持这个结论。**

### 47.1.1 第一原则：这些 `[BRIDGE]` 基本都是 Host/Python Trace

`[ADDR]`、`[ATTN]`、`[MRV2]` 是 Python / host-side 代码打印的语义日志。

它们记录的是：

```text
CPU 当前执行到了哪个 Python/C++ API 调用阶段
```

不是：

```text
GPU 某个 kernel 在这一纳秒已经物理执行完成
```

CUDA 常见执行模型更接近：

```text
CPU timeline

prepare metadata
↓
launch kernel A
↓
Python 继续执行
↓
launch kernel B
↓
Python 继续执行


GPU timeline

      kernel A ───────
               kernel B ───────
```

因此：

```text
Host 控制流有顺序
```

和：

```text
Host 与 GPU 可以异步推进
```

完全不矛盾。

### 47.1.2 为什么 `[KV_WRITE] → [KV_READ]` 也会严格按顺序出现

因为这些 trace 记录的是 API / dispatch 顺序。

而且 KV write 和后续 attention read 本身存在数据依赖：

```text
当前 token 的 K/V
先写入 cache
↓
Attention read
才能看到完整历史
```

即使 GPU 是异步执行，同一个依赖链也不能被任意乱序。

所以：

```text
日志顺序稳定
```

不等于：

```text
CPU 每一步都 cudaDeviceSynchronize 等 GPU 完成
```

### 47.1.3 E6 真正证明 async scheduling 的地方不是日志交叉，而是“延后 reconcile”

Call0：

```text
QUEUE_PUSH queue_len=1
YIELD_WITHOUT_CONSUME
```

然后没有出现：

```text
WAIT_RESULT
RECONCILE
TOKEN_APPEND
```

就直接进入：

```text
Call1
SCHEDULE_BEGIN
```

所以 E6 直接证明的是：

```text
Call0 已提交
↓
Call0 result 尚未 Scheduler.update_from_output()
↓
EngineCore 已经允许 Call1 schedule / submit
```

这就是当前 `step_with_batch_queue()` 最重要的 async 语义：

> **execution submission 与 result consumption/reconciliation 解耦。**

### 47.1.4 `future_done=True` 为什么仍不能写成“GPU 已经全部做完”

E6：

```text
EXEC_SUBMIT_RETURN
future_type=Future
future_done=True
```

随后：

```text
SAMPLE_SUBMIT_RETURN
future_type=AsyncOutputFuture
future_done=False
```

需要严格按照抽象层解释：

```text
execute_model 这一层 Future 已处于完成态
```

不等价于：

```text
GPU 上与这一 transaction 相关的所有 kernel 都已经物理完成，
并且采样结果已经被 Scheduler 消费
```

后续真正被 queue 延迟消费的 output/result 仍然没有 settle。

### 47.1.5 `batch_queue` 存的是 batch transaction，不是某个 Request 的“下一 token step”

E6 只有一个用户 Request，并且：

```text
max_num_seqs=1
```

所以 Call0 / Call1 / Call2 看起来很像：

```text
同一个 Request 的 Prefill step
↓
Decode step
↓
Decode step
```

但 EngineCore `call=N` 的本质是：

```text
一次 runtime batch transaction / EngineCore step invocation
```

如果同时有 A/B/C：

```text
Call N
SchedulerOutput:
A → 1 token
B → 128 tokens
C → 1 token

total_scheduled=130
```

仍然只是一整个 Call / batch transaction。

所以：

```text
Call ID != Request step ID
batch_queue != per-request decode queue
```

### 47.1.6 两条时间轴：应该怎样脑补

不要画成：

```text
Schedule A
↓
GPU A 完全结束
↓
Schedule B
```

更合适的概念图：

```text
EngineCore / CPU host timeline

Schedule A
↓
Submit Execute A
↓
Submit Sample A
↓
Queue A
↓
不消费 A
↓
Schedule B
↓
Submit Execute B
↓
Submit Sample B
↓
Queue B
↓
Queue 满
↓
Consume / Reconcile A


GPU timeline（概念）

        A kernels ─────────────
                 B dependent work ─────────
```

这里必须加一句证据边界：

> **E6 Python Trace 只能证明 host/runtime 的 ahead-of-reconcile 行为，不能测量 A/B GPU kernel 实际 overlap 了多少。**

### 47.1.7 A100 + 小模型是不是可能 GPU 已经非常快地完成了？

可能。

当前：

```text
Qwen3-0.6B
31-token prompt
A100
```

workload 很小，GPU 可能在 CPU 下一轮做很多工作之前就已经完成大量 GPU work。

但这属于：

```text
合理可能性
```

不是 E6 Trace 可以直接证明的事实。

要回答：

```text
Call1 Scheduler.schedule() 执行时，Call0 的哪些 GPU kernels 仍在跑？
CPU/GPU overlap 到底多少？
```

需要 Nsight Systems 同时观察：

```text
CPU thread timeline
CUDA API timeline
GPU kernel timeline
```

因此：

```text
BRIDGE Trace
= semantic / state-machine timeline

Nsight Systems
= hardware execution timeline
```

两者不能互相替代。


# 48. Call0 整条链一次完整对齐

现在把用户贴的 Call0 压成一张函数链：

```text
EngineCoreProc.run_busy_loop()
│
└── self.step_fn()
    │
    └── EngineCore.step_with_batch_queue()
        │
        ├── [CALL_BEGIN]
        │
        ├── [SCHEDULE_BEGIN]
        │   └── Scheduler.schedule()
        │       ├── [QUEUE_SNAPSHOT]
        │       ├── prefix lookup
        │       ├── KVCacheManager.allocate_slots()
        │       │   └── BlockPool → [1,2]
        │       ├── [BEFORE_COMMIT]
        │       ├── _update_after_schedule()
        │       ├── [AFTER_COMMIT]
        │       └── return SchedulerOutput
        │
        ├── [SCHEDULE_END]
        │
        ├── [EXEC_SUBMIT_BEGIN]
        │   └── UniProcExecutor.execute_model()
        │       └── collective_rpc()
        │           └── WorkerWrapperBase
        │               └── CUDA Worker.execute_model()
        │                   └── GPUModelRunnerV2.execute_model()
        │                       ├── [MRV2 EXECUTE_BEGIN]
        │                       ├── persistent state sync
        │                       ├── prepare_inputs()
        │                       ├── prepare_attn()
        │                       │   ├── gather_block_tables()
        │                       │   │   └── [ADDR BLOCK_TABLE]=[1,2]
        │                       │   └── compute_slot_mappings()
        │                       │       └── [ADDR MAPPING] 0..30
        │                       ├── model_state.prepare_attn()
        │                       └── model forward
        │                           └── Attention layer0
        │                               ├── unified_kv_cache_update()
        │                               │   ├── [KV_WRITE]
        │                               │   └── FlashAttentionImpl.do_kv_cache_update()
        │                               │       ├── [KV_LAYOUT]
        │                               │       └── reshape_and_cache_flash()
        │                               │
        │                               └── unified_attention_with_output()
        │                                   ├── [KV_READ]
        │                                   └── FlashAttentionImpl.forward()
        │
        ├── [EXEC_SUBMIT_RETURN]
        │
        ├── [SAMPLE_SUBMIT_BEGIN]
        │   └── GPUModelRunnerV2.sample_tokens()
        │       └── [MRV2 SAMPLE_BEGIN]
        │
        ├── [SAMPLE_SUBMIT_RETURN]
        │
        ├── [QUEUE_PUSH]
        │
        └── [YIELD_WITHOUT_CONSUME]
```

**这张图就是以后读任何一轮 E6 runtime log 的主索引。**

---

# 49. Call1：为什么还没有 consume Call0，就已经 schedule position31

Call1 开头：

```text
[CALL_BEGIN]
call=1
queue_len=1
scheduler_has_requests=True
```

注意 Call0 仍在 batch_queue。

Scheduler：

```text
[BEFORE_COMMIT]
num_tokens=31
placeholders=1
num_computed=31
num_in_flight=31
pending=1
scheduled=1
new_block_ids=None
```

为什么 pending=1？

因为 async placeholder 已经代表：

```text
Call0 会产生一个 output position
```

即使 Call0 的 sampled token 还没有通过 `update_from_output()` append 到 Request，Scheduler 仍能提前提交下一份 decode work。

这是 schedule-time logical frontier 和 result-time real token append 解耦的体现。

---

# 50. Call1 地址链：position31 仍在 physical block2

ModelRunner：

```text
[BLOCK_TABLE]
[1,2]

[MAPPING]
position=31
logical_block=1
physical_block=2
offset=15
slot=47
```

为什么 Scheduler：

```text
new_block_ids=None
```

因为 physical block2 的 slot 范围：

```text
32...47
```

Call0 只写到了：

```text
position30 → slot46
```

所以：

```text
position31 → slot47
```

仍然可以使用原 block2。

Attention WRITE：

```text
slots=[47]
```

READ：

```text
block_table=[1,2]
seq_lens=[32]
```

又一次把 control/data 链完整对齐。

---

# 51. Call1 queue 满了，所以开始 consume Call0

Call1 submit 后：

```text
queue=[Call0, Call1]
capacity=2
```

日志：

```text
[QUEUE_CONSUME_BEGIN]
↓
[QUEUE_POP]
scheduled=31
```

注意 pop 的 old batch 是：

```text
Call0 / 31-token Prefill transaction
```

不是刚刚提交的 Call1。

然后：

```text
[WAIT_RESULT_BEGIN]
↓
future.result()
↓
[WAIT_RESULT_END]
model_output_type=ModelRunnerOutput
```

这才开始把 Call0 的 result 交回 Scheduler。

---

# 52. `RECONCILE_BEGIN` 对应 `Scheduler.update_from_output()`

```text
[RECONCILE_BEGIN]
scheduled={'0-a2c39288':31}
```

函数层：

```text
EngineCore.step_with_batch_queue()
↓
拿到 oldest ModelRunnerOutput
↓
self.scheduler.update_from_output(
    old_scheduler_output,
    model_output
)
```

进入 Scheduler result-time reconciliation。

这里和 `Scheduler.schedule()` 是同一个 stateful controller 的两端：

```text
schedule()
= 决策 / commit

update_from_output()
= 执行结果 settle / token append / lifecycle update
```

---

# 53. `[SCHED][SETTLE]` 和 `[TOKEN_APPEND]` 各自对应什么

Call1 consume Call0：

```text
[SETTLE]
settled=31
in_flight_before=32
in_flight_after=1
num_computed=32
num_tokens=31
```

为什么 before=32？

因为：

```text
Call0 in-flight 31
+
Call1 已提前 schedule 1
=
32
```

settle Call0：

```text
32 - 31 = 1
```

所以还有 Call1 outstanding。

随后：

```text
[TOKEN_APPEND]
num_tokens_after=32
num_output_tokens=1
```

这才是：

> Call0 sample 出来的第一个真实 output token 被正式 append 到 Request token sequence。

因此：

```text
placeholder
≠ real appended output token
```

前者服务 async ahead scheduling；后者在 result reconciliation 才正式进入 Request。

---

# 54. Call2：position32 为什么触发 `new_block_ids=[3]`

Call0 result append 后：

```text
num_tokens=32
```

positions 0...31 正好填满：

```text
block1: 0...15
block2: 16...31
```

下一份 work 是 position32。

Scheduler：

```text
[BEFORE_COMMIT]
num_tokens=32
scheduled=1
new_block_ids=([3],)
```

这里 control plane 完成：

```text
logical block2
→ allocate physical block3
```

ModelRunner：

```text
block_table=[1,2,3]
```

Address：

```text
position32
logical_block=2
physical_block=3
offset=0
expected_slot=48
actual_slot=48
```

Attention：

```text
WRITE slots=[48]
READ block_table=[1,2,3], seq_len=33
```

这是整个 E6 最强的一组跨层证据：

```text
Scheduler new block3
↓
ModelRunner block_table 第三项=3
↓
position32 映射 physical3
↓
slot48
↓
FlashAttention write slot48
↓
FlashAttention read page table包含3
```

---

# 55. 为什么这条 position32 链特别重要

因为它不是“两个连续 block ID 恰好看起来合理”。

它同时证明了三个层次：

```text
Control Plane
Scheduler / KV manager
决定新的 physical block ownership

Execution Preparation
ModelRunner 把 ownership 翻译为 block_table / slot

Data Plane
Attention Backend 真正消费这个地址写/读 KV
```

所以 E6 第一次真正把：

```text
Scheduler
→ KV
→ ModelRunner
→ Attention
```

从静态调用链变成了一条真实 runtime transaction。

---

# 56. 最后几个 `scheduled=0` 不是又跑了模型业务

Request 最后一个 output token sample 完以后，还会看到：

```text
SCHEDULE_END total_scheduled=0
MRV2 EXECUTE_BEGIN total_scheduled=0
QUEUE_PUSH
QUEUE_POP
RECONCILE scheduled={}
```

这是 async pipeline drain。

原因：

```text
Request 生命周期结束
≠ batch_queue 已经完全为空
```

队列中还可能残留：

```text
empty transaction
last outstanding future
```

所以最后：

```text
CALL_BEGIN scheduler_has_requests=False
```

仍可能只负责 drain queue。

最终：

```text
[shutdown] exiting busy loop
```

才表示 EngineCoreProc 的长期 busy loop 退出。

---

# 57. Shutdown：必须区分 Request finish、async queue drain、Process shutdown 三个阶段

E6 尾部很容易被压成一句：

```text
请求结束 → shutdown
```

但真实生命周期至少有三个不同层级。

## 57.1 第一层：Request finished

E6 最后一次有效 reconcile 后：

```text
TOKEN_APPEND
num_output_tokens=4
```

随后 Request 对应 KV blocks 被 free。

这只说明：

```text
用户 Request lifecycle 已完成
```

并不说明：

```text
batch_queue 已空
EngineCore busy loop 已退出
Linux EngineCore Process 已结束
```

## 57.2 第二层：async batch_queue drain

因此后面仍然出现：

```text
SCHEDULE_END total_scheduled=0
MRV2 EXECUTE_BEGIN total_scheduled=0
QUEUE_PUSH
QUEUE_POP
RECONCILE scheduled={}
```

这些 `scheduled=0` 不是新的业务 token forward。

它们属于：

```text
outstanding transaction / future / empty batch bookkeeping
```

沿统一 async pipeline 继续被消费干净。

所以必须记住：

```text
Request finished
!=
batch_queue empty
```

只有当 Scheduler 已无 Request 且 outstanding queue 也 drain 完，Runtime 才真正空闲。

## 57.3 第三层：EngineCore Process 收到 shutdown signal

E6：

```text
[shutdown] EngineCore: trigger received signal=SIGTERM
[shutdown] EngineCore: start mode=abort timeout=0s
[shutdown] EngineCore: request processing complete; starting resource teardown
[shutdown] EngineCore: exiting busy loop
```

这些日志说明：

```text
EngineCoreProc 的 process-level shutdown state machine 已启动
```

它已经不再是“某个 Request 如何 finish”的问题，而是：

```text
整个 EngineCore Linux Process 生命周期如何结束
```

需要注意证据边界：

> E6 日志直接证明 EngineCore Process **收到了 SIGTERM**；仅靠这几行日志不能精确证明“是哪一个具体函数/哪个父侧对象发送了该 SIGTERM”。

结合整体架构可以确定 parent-side `CoreEngineProcManager` 负责 EngineCore child process 生命周期与 shutdown 管理；但如果以后要精确追“E6 这一次 SIGTERM 的发送调用点”，应单独对 manager / client cleanup path 加 trace，而不是只从 child 侧日志反推。

## 57.4 `mode=abort` 不等于“用户请求异常失败”

当前日志里：

```text
start mode=abort timeout=0s
```

发生时用户 Request 已经正常得到 4 个 output token，并且 queue 正在/已经完成 drain。

因此这里的 `abort` 描述的是：

```text
EngineCore process shutdown strategy / mode
```

不能反向解释成：

```text
前面的 inference request 被 abort 了
```

## 57.5 最终生命周期闭环

```text
Frontend / parent-side resources
↓
启动 EngineCore child process
↓
════════ OS Process Boundary ════════
↓
EngineCoreProc.run_engine_core()
↓
EngineCoreProc(...)
↓
EngineCore.__init__() + Process/IPC wiring
↓
run_busy_loop()
↓
反复：
    ingress
    schedule
    execute
    queue
    reconcile
↓
Request finished
↓
async queue drain
↓
SIGTERM / shutdown state
↓
resource teardown
↓
run_busy_loop() exit
↓
run_engine_core() cleanup / child process exit
```

这样整个系统才真正从：

```text
proc.start()
```

闭环到：

```text
child process exit
```

而不只是从：

```text
Prompt
```

走到：

```text
最后一个 token
```

---

# 58. 一张“Trace Tag → 文件 → 函数 → 阶段”的速查表

| Trace / Log | Linux Context | 文件 / 函数 | 阶段 | 它证明什么 |
|---|---|---|---|---|
| `Initializing a V1 LLM engine` | EngineCore Process | `core.py / EngineCore.__init__` 附近 | Init | Core Runtime 开始构建 |
| `world_size/rank` | EngineCore Process | distributed / Worker init | Init | rank/device runtime 建立 |
| `Using V2 Model Runner` | EngineCore Process | `gpu_worker.py / Worker.init_device` | Init | 当前 active ModelRunner = V2 |
| `Loading model from scratch` | EngineCore Process | MRV2 load_model | Init | 模型权重生命周期 |
| `Using FLASH_ATTN` | EngineCore Process | backend selector | Init | 默认 backend 实际选择 |
| `_dummy_req_0` `MRV2 EXECUTE_BEGIN` | EngineCore Process | Worker/MRV2 profile path | Init/Profile | synthetic memory profiling，不经过 runtime Scheduler |
| `Available KV cache memory` | EngineCore Process | Worker + `_initialize_kv_caches` | Init | profile 后 KV budget |
| `GPU KV cache size` | EngineCore Process | KV cache config | Init | token capacity |
| `_warmup_0_` | EngineCore Process | Worker/MRV2 warmup | Init/Warmup | KV 已创建后的热身 |
| `[CORE][INIT]` | EngineCore Process | `EngineCore.__init__` | Init | batch queue / step_fn 已固定 |
| `[CLIENT_SEND]` | Frontend Process | `SyncMPClient.add_request()` | Ingress | Frontend SEND |
| `[IPC_RECV]` | EngineCore input thread | `process_input_sockets()` | Ingress | ZMQ 已收到 |
| `[IPC_ENQUEUE]` | EngineCore input thread | `process_input_sockets()` | Ingress | 进入 local input_queue |
| `[CORE_DEQUEUE]` | EngineCore busy loop | `_handle_client_request()` | Ingress | busy loop 开始处理 |
| `[SCHED_ADD_DONE]` | EngineCore busy loop | `add_request → Scheduler.add_request` | Ingress | Scheduler ownership 建立 |
| `[CALL_BEGIN]` | EngineCore busy loop | `step_with_batch_queue()` | Runtime Orchestration | 本轮 Core transaction 开始 |
| `[SCHEDULE_BEGIN]` | EngineCore busy loop | `step_with_batch_queue()` | Control Plane entrance | 即将调用 `Scheduler.schedule()` |
| `[QUEUE_SNAPSHOT]` | EngineCore Process | `Scheduler.schedule()` | Scheduler | waiting/running/known state |
| `[PREFIX][LOOKUP]` | EngineCore Process | Scheduler waiting / KV lookup path | Scheduler/KV | cached computed prefix |
| `[BEFORE_COMMIT]` | EngineCore Process | `Scheduler.schedule()` | Scheduler/KV | 本轮 token + block 决策已形成 |
| `[AFTER_COMMIT]` | EngineCore Process | `_update_after_schedule()` 后 | Scheduler | logical commit / in-flight |
| `[SCHEDULE_END]` | EngineCore busy loop | `step_with_batch_queue()` | Protocol boundary | SchedulerOutput 已返回 |
| `[EXEC_SUBMIT_BEGIN]` | EngineCore Process | `step_with_batch_queue()` | Execution bridge | 调 Executor |
| `[MRV2][EXECUTE_BEGIN]` | EngineCore Process | `GPUModelRunnerV2.execute_model()` | Execution | SchedulerOutput 到达 MRV2 |
| `[ADDR][BATCH]` | EngineCore Process | `GPUModelRunner.prepare_attn()` | Addressing | 当前 token/addressing batch |
| `[ADDR][BLOCK_TABLE]` | EngineCore Process | `gather_block_tables()` 后 | Addressing | logical block→physical block |
| `[ADDR][MAPPING]` | EngineCore Process | `compute_slot_mappings()` 后 | Addressing | position→physical slot |
| `[ATTN][KV_WRITE]` | EngineCore Process | `unified_kv_cache_update()` | Data Plane | K/V + slot_mapping 进入 backend write |
| `[ATTN][KV_LAYOUT]` | EngineCore Process | `FlashAttentionImpl.do_kv_cache_update()` | Backend Data Layout | packed KV → K/V views |
| `[ATTN][KV_READ]` | EngineCore Process | `unified_attention_with_output()` | Data Plane | Q + block_table + seq metadata 进入 backend read |
| `[EXEC_SUBMIT_RETURN]` | EngineCore Process | `step_with_batch_queue()` | Execution return | execute-side Future 返回 |
| `[SAMPLE_BEGIN]` | EngineCore Process | MRV2 `sample_tokens()` | Sampling | sampling stage |
| `[QUEUE_PUSH]` | EngineCore Process | `step_with_batch_queue()` | Async Pipeline | transaction outstanding |
| `[YIELD_WITHOUT_CONSUME]` | EngineCore Process | `step_with_batch_queue()` | Async Pipeline | queue 未满，允许继续 ahead |
| `[QUEUE_POP]` | EngineCore Process | `step_with_batch_queue()` | Async Pipeline | 取 oldest transaction |
| `[WAIT_RESULT_*]` | EngineCore Process | Future `.result()` | Async Pipeline | 等待旧 batch result |
| `[RECONCILE_BEGIN]` | EngineCore Process | `Scheduler.update_from_output()` 前 | Reconciliation | result 回控制面 |
| `[SETTLE]` | EngineCore Process | `Scheduler.update_from_output()` | Reconciliation | in-flight work settle |
| `[TOKEN_APPEND]` | EngineCore Process | `Scheduler.update_from_output()` / Request append | Reconciliation | sampled token 真正进入 Request |
| `[RECONCILE_END]` | EngineCore Process | update 完成 | Reconciliation | state_t → state_t+1 |

---

# 59. 以后读日志的正确方法：先判断“阶段”，不要先盯变量

拿到一行：

```text
[BRIDGE][ADDR][MAPPING]
```

不要一上来问：

```text
为什么 slot=48？
```

先问：

```text
1. 这是 Init 还是 Runtime？
→ Runtime

2. Control Plane 还是 Data Plane？
→ 中间的 Execution Preparation / Addressing

3. 哪个 Object？
→ GPUModelRunnerV2

4. 哪个函数？
→ prepare_attn() 内 compute_slot_mappings() 之后

5. 上游是谁？
→ SchedulerOutput → MRV2 persistent state → InputBatch

6. 下游是谁？
→ slot_mappings_by_layer → Attention forward_context → KV write
```

回答完这6个问题，`slot=48` 才有上下文。

---

# 60. 三个“世界”一定要分开

E6 现在可以真正用动态日志区分：

## 60.1 Control Plane

```text
Scheduler
KVCacheManager
BlockPool
```

日志代表：

```text
scheduled=31
new_block_ids=[1,2]
```

回答：

> 这一轮算什么？Request 拥有哪些 physical pages？

---

## 60.2 Execution Preparation / Addressing

```text
MRV2
prepare_inputs
prepare_attn
block_table
slot_mapping
```

日志代表：

```text
block_table=[1,2]
position16 → slot32
```

回答：

> 已经决定好的计划，怎样变成 GPU metadata / address？

---

## 60.3 Data Plane

```text
Model
Attention Backend
KV Tensor
FlashAttention / CUDA op
```

日志代表：

```text
KV_WRITE slots=[...]
KV_READ block_table=[...]
```

回答：

> GPU 数据实际按什么 layout 写/读？

---

# 61. 一张真正完整的 E6 日志驱动调用链

```text
Frontend Process
│
│ LLM.generate()
│ SyncMPClient.add_request()
│ [CLIENT_SEND]
│
└──────────── ZMQ ───────────────────────────────────┐
                                                    ▼
EngineCore Process
│
├── input thread
│   └── EngineCoreProc.process_input_sockets()
│       ├── [IPC_RECV]
│       └── [IPC_ENQUEUE]
│
└── busy loop
    └── EngineCoreProc.run_busy_loop()
        │
        ├── drain input_queue
        │   └── _handle_client_request()
        │       ├── [CORE_DEQUEUE]
        │       └── Scheduler.add_request()
        │           └── [SCHED_ADD_DONE]
        │
        └── self.step_fn()
            └── EngineCore.step_with_batch_queue()
                │
                ├── [CALL_BEGIN]
                │
                ├── Scheduler.schedule()
                │   ├── [QUEUE_SNAPSHOT]
                │   ├── Prefix lookup
                │   ├── KVCacheManager.allocate_slots()
                │   │   └── Coordinator
                │   │       └── SingleTypeKVCacheManager
                │   │           └── BlockPool
                │   │               └── physical block IDs
                │   ├── [BEFORE_COMMIT]
                │   ├── _update_after_schedule()
                │   ├── [AFTER_COMMIT]
                │   └── SchedulerOutput
                │
                ├── UniProcExecutor.execute_model()
                │   └── collective_rpc()
                │       └── WorkerWrapperBase
                │           └── CUDA Worker.execute_model()
                │               └── GPUModelRunnerV2.execute_model()
                │                   ├── [MRV2 EXECUTE_BEGIN]
                │                   ├── state sync
                │                   ├── prepare_inputs()
                │                   ├── prepare_attn()
                │                   │   ├── gather_block_tables()
                │                   │   │   └── [ADDR BLOCK_TABLE]
                │                   │   └── compute_slot_mappings()
                │                   │       └── [ADDR MAPPING]
                │                   ├── model_state.prepare_attn()
                │                   └── model forward
                │                       └── Attention.forward()
                │                           │
                │                           ├── WRITE
                │                           │   └── unified_kv_cache_update()
                │                           │       ├── [KV_WRITE]
                │                           │       └── FlashAttentionImpl.do_kv_cache_update()
                │                           │           ├── [KV_LAYOUT]
                │                           │           └── reshape_and_cache_flash()
                │                           │
                │                           └── READ
                │                               └── unified_attention_with_output()
                │                                   ├── [KV_READ]
                │                                   └── FlashAttentionImpl.forward()
                │
                ├── sample_tokens()
                │   └── [MRV2 SAMPLE_BEGIN]
                │
                ├── batch_queue.push()
                │
                ├── 如果未满：继续下一 Call
                │
                └── 如果需要 consume old batch
                    ├── queue.pop()
                    ├── Future.result()
                    └── Scheduler.update_from_output()
                        ├── [SETTLE]
                        ├── [TOKEN_APPEND]
                        └── Request / Scheduler next state
```

---

# 62. 最后用 position32 作为整条架构的“锚点”

如果以后忘了所有类名，只记这一个 E6 事件即可恢复整条主线。

```text
Request 已知 32 tokens
positions0...31 已占满两个 block
↓
Scheduler.schedule()
↓
下一 token position32 需要 KV capacity
↓
KVCacheManager / BlockPool
分配 physical block3
↓
Scheduler log:
new_block_ids=[3]
↓
SchedulerOutput
↓
Executor / Worker
↓
MRV2 persistent block state 更新
↓
prepare_attn()
↓
block_table=[1,2,3]
↓
compute_slot_mappings()
↓
position32 //16 = logical block2
block_table[2]=physical3
offset=0
slot=3*16=48
↓
ADDR log:
actual_slot=48
↓
Attention.forward()
↓
unified_kv_cache_update()
↓
KV_WRITE slots=[48]
↓
reshape_and_cache_flash()
↓
BF16 KV Cache physical block3 / offset0
↓
unified_attention_with_output()
↓
FlashAttentionImpl.forward()
↓
KV_READ:
block_table=[1,2,3]
seq_len=33
↓
Attention 看到完整历史0...32
↓
sample token
↓
batch_queue
↓
Scheduler.update_from_output()
↓
Request state 进入下一轮
```

这一条就是：

> **从 Scheduler 的抽象 token decision，一直到 GPU HBM 中某个具体 KV slot，再回到 Scheduler state machine 的完整 runtime 闭环。**

---

# 63. 以后每读一个函数，只需要问这 7 个问题

建议把下面作为后续 vLLM 源码阅读固定模板：

```text
1. 它运行在哪个 Linux Process？

2. 它属于哪个长期 Object？
   EngineCoreProc / Scheduler / Executor / Worker / ModelRunner / Attention？

3. 它属于哪一阶段？
   Initialization / Ingress / Control / Execution Prep / Data Plane / Reconcile？

4. 谁调用它？

5. 它调用谁？

6. 它消费的输入是谁生产的？

7. 它产生的输出下一层怎么用？
```

例如：

```text
compute_slot_mappings()

Process：EngineCore
Object：GPUModelRunner / BlockTables
Phase：Execution Preparation / Addressing
Caller：GPUModelRunner.prepare_attn()
Input producer：Scheduler/KV ownership → persistent block tables + InputBatch.positions
Output：slot_mapping
Consumer：Attention unified_kv_cache_update → Backend KV write
```

这样就不会再出现“函数本身看懂了，但不知道它处于整条链的哪里”的问题。

---

# 64. 30 秒恢复版

```text
初始化：
LLM
→ SyncMPClient
→ CoreEngineProcManager
→ proc.start
→ EngineCoreProc.run_engine_core
→ EngineCoreProc/EngineCore
→ UniProcExecutor
→ Worker
→ V2 ModelRunner
→ model load
→ profile
→ KV cache init
→ warmup
→ Scheduler
→ run_busy_loop

请求进入：
CLIENT_SEND
→ IPC_RECV
→ IPC_ENQUEUE
→ CORE_DEQUEUE
→ Scheduler.add_request

一轮 Runtime：
run_busy_loop
→ step_with_batch_queue
→ Scheduler.schedule
→ KVCacheManager / BlockPool
→ SchedulerOutput
→ Executor
→ Worker
→ MRV2
→ prepare_inputs
→ prepare_attn
→ block_table + slot_mapping
→ Attention
→ KV write / paged read
→ sample
→ batch_queue
→ update_from_output
→ 下一轮
```

---

# 65. 最终一句话

以前的认知是：

```text
“我学过 EngineCore、Scheduler、ModelRunner、Attention。”
```

E6 之后应该升级成：

```text
我可以拿一条真实 Runtime log，
先判断它属于哪个 Process 和阶段，
再定位到对应函数，
沿 caller/callee 把它挂回：

Frontend
→ EngineCoreProc busy loop
→ Scheduler/KV control plane
→ SchedulerOutput
→ Executor/Worker
→ MRV2 addressing
→ Attention data plane
→ batch_queue
→ Scheduler reconciliation

这条完整状态机。
```

这才是后面继续读 Triton Attention、INT4 KV、LMCache/offload 时真正需要的总体认知。
