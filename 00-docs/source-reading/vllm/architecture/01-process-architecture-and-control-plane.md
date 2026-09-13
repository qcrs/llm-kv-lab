# vLLM V1 进程架构与控制面追踪

> 环境：vLLM v0.26.0
> Commit：`568afb3a13806beb53bb2e6bd518269357b237c0`
> 模型：`/data/models/Qwen3-0.6B`
> 当前实验：offline `LLM(...)`，TP=1，PP=1，DP=1
> 本节目标：理解 **Frontend Python 为什么会创建独立的 EngineCore Process，以及这一进程边界在源码中如何形成。**

---

# 1. 为什么从这条链路开始

在正式阅读 Scheduler、KVCacheManager、ModelRunner 之前，首先需要回答一个更基础的问题：

> **vLLM 中这些对象到底运行在哪个 Process？谁创建谁？**

如果进程边界没有理清，后面很容易把：

```text
Python Object
Process
Thread
Worker
Executor
```

混在一起。

例如：

```python
self.scheduler = Scheduler(...)
```

只是创建一个 Python Object；

而：

```python
proc.start()
```

才可能真正创建新的 Linux Process。

因此第一阶段不直接读 Scheduler，而是从我们实际运行的入口：

```python
llm = LLM(...)
```

开始，回答：

> **为什么执行这一行之后，Linux 中会出现一个独立的 `VLLM::EngineCore` Process？**

---

# 2. 先从运行现象建立问题

运行：

```bash
./04-experiments/vllm-bridge/raw/e0-smoke/commands.sh
```

通过：

```bash
ps -eo pid,ppid,stat,cmd --forest
```

观察到：

```text
python smoke_06b.py
PID 3907513
    │
    └── VLLM::EngineCore
        PID 3908079
        PPID 3907513
```

进一步：

```bash
nvidia-smi \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv
```

观察到：

```text
3908079, VLLM::EngineCore, 75170 MiB
```

因此首先得到动态事实：

```text
Frontend Python Process
PID 3907513
        │
        │ 创建子进程
        ▼
EngineCore Process
PID 3908079
        │
        └── 持有主要 GPU / CUDA / KV Cache 资源
```

于是产生本次源码追踪问题：

```text
LLM(...)
↓
源码到底经过什么路径
↓
最终触发 Linux 创建 EngineCore Process？
```

---

# 3. 源码分析方法

这一轮没有从 `core.py` 第一行开始读，而采用：

```text
运行现象
↓
提出一个具体问题
↓
定位入口 Symbol
↓
寻找对象创建点
↓
沿调用关系向下一层
↓
遇到配置分支，追实际运行值
↓
遇到 Process 边界，确认真正的 start()
↓
再与 ps / nvidia-smi 动态结果对应
```

核心原则：

> **每一步的下一跳，都应该由当前代码暴露出来，而不是随便选择一个看起来重要的文件。**

例如：

```python
self.llm_engine = LLMEngine.from_engine_args(...)
```

自然产生：

> `from_engine_args()` 做了什么？

又例如：

```python
return SyncMPClient(...)
```

自然产生：

> `SyncMPClient.__init__()` 做了什么？

不是看到 `Scheduler` 就顺手跳进去。

---

# 4. 第一步：从用户入口 `LLM(...)` 开始

我们的代码：

```python
llm = LLM(...)
```

首先定位：

```bash
rg -n "^class LLM" vllm/entrypoints
```

找到：

```text
vllm/entrypoints/llm.py
```

进入 `LLM.__init__()` 后，不逐行阅读所有参数，而寻找它持有的核心 Engine Object。

最终找到：

```python
self.llm_engine = LLMEngine.from_engine_args(
    engine_args=engine_args,
    usage_context=UsageContext.LLM_CLASS,
)
```

因此第一跳确定：

```text
LLM
│
│ 创建 / 持有
▼
LLMEngine
```

准确路径：

```text
smoke_06b.py
↓
LLM(...)
↓
LLM.__init__()
↓
LLMEngine.from_engine_args(...)
```

---

# 5. 第二步：追 `multiprocess_mode` 为什么实际为 True

进入：

```python
LLMEngine.from_engine_args(...)
```

看到：

```python
@classmethod
def from_engine_args(
    cls,
    engine_args: EngineArgs,
    ...
    enable_multiprocessing: bool = False,
) -> "LLMEngine":

    vllm_config = engine_args.create_engine_config(usage_context)
    executor_class = Executor.get_class(vllm_config)

    if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
        enable_multiprocessing = True

    return cls(
        ...
        multiprocess_mode=enable_multiprocessing,
    )
```

这里第一次容易产生误解：

```python
enable_multiprocessing: bool = False
```

只是函数的**默认参数**。

实际运行继续检查：

```python
envs.VLLM_ENABLE_V1_MULTIPROCESSING
```

当前版本：

```python
VLLM_ENABLE_V1_MULTIPROCESSING = True
```

所以运行时：

```text
enable_multiprocessing
初始 False
↓
VLLM_ENABLE_V1_MULTIPROCESSING == True
↓
enable_multiprocessing = True
↓
LLMEngine(
    multiprocess_mode=True
)
```

这一步体现了一种重要源码分析方法：

## 参数追踪

看到：

```python
multiprocess_mode=multiprocess_mode
```

不能只看当前位置，而应该向上追：

```text
当前变量
↓
谁传进来的
↓
调用方变量来自哪里
↓
最终配置源头
```

以后追：

```text
block_size
gpu_memory_utilization
max_num_batched_tokens
enable_prefix_caching
kv_cache_dtype
```

也是同样的方法。

---

# 6. 第三步：为什么选择 `SyncMPClient`

`LLMEngine.__init__()` 中：

```python
self.engine_core = EngineCoreClient.make_client(
    multiprocess_mode=multiprocess_mode,
    asyncio_mode=False,
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=self.log_stats,
)
```

经过前面的参数追踪，当前实际等价于：

```python
EngineCoreClient.make_client(
    multiprocess_mode=True,
    asyncio_mode=False,
    ...
)
```

`make_client()`：

```python
if asyncio_mode and not multiprocess_mode:
    raise NotImplementedError(...)

if multiprocess_mode and asyncio_mode:
    return EngineCoreClient.make_async_mp_client(...)

if multiprocess_mode and not asyncio_mode:
    return SyncMPClient(...)

return InprocClient(...)
```

因此：

```text
multiprocess_mode = True
asyncio_mode      = False
```

进入：

```python
SyncMPClient(...)
```

---

# 7. `multiprocess_mode` 和 `asyncio_mode` 是两个不同维度

这里必须避免一个常见混淆。

## multiprocess_mode

回答：

> **EngineCore 是否运行在独立 Process 中？**

```text
False
→ Frontend / EngineCore 同进程

True
→ EngineCore 独立后台进程
```

## asyncio_mode

回答：

> **Frontend Client 是否使用 asyncio 风格处理请求/响应？**

```text
False
→ 同步 Client

True
→ asyncio Client
```

因此当前：

```text
offline LLM
+
独立 EngineCore

=
multiprocess_mode=True
asyncio_mode=False
=
SyncMPClient
```

可以总结为：

| multiprocessing | asyncio | Client          |
| --------------- | ------- | --------------- |
| False           | False   | `InprocClient`  |
| True            | False   | `SyncMPClient`  |
| True            | True    | `AsyncMPClient` |
| False           | True    | 当前不支持           |

注意：

```text
asyncio_mode=False
```

和日志中的：

```text
Asynchronous scheduling is enabled
```

不是一个概念。

前者是 Frontend Client / asyncio 模型；

后者属于 Engine 内部调度机制，后续单独研究。

---

# 8. 第四步：`SyncMPClient` 自己并不直接启动 EngineCore

类关系：

```python
class SyncMPClient(MPClient):
```

初始化第一件事情：

```python
super().__init__(
    asyncio_mode=False,
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=log_stats,
)
```

因此：

```text
SyncMPClient.__init__()
↓
MPClient.__init__()
```

`SyncMPClient` 自己后面的重点主要是创建一个后台 Thread：

```python
self.output_queue_thread = Thread(
    target=process_outputs_socket,
    name="EngineCoreOutputQueueThread",
    daemon=True,
)
self.output_queue_thread.start()
```

这个是：

```text
Thread
```

而不是：

```text
EngineCore Process
```

主要作用可以暂时理解成：

```text
EngineCore Output
↓
ZMQ output socket
↓
后台线程读取
↓
outputs_queue
↓
同步 LLM Frontend 获取结果
```

因此真正的 EngineCore Process 创建逻辑要继续进入：

```text
MPClient.__init__()
```

---

# 9. 第五步：MPClient 决定谁管理 EngineCore

`MPClient.__init__()` 中有一个关键分支：

```python
if client_addresses:
    # Engines are managed externally to this client.
    ...
else:
    # Engines are managed by this client.
```

当前 `SyncMPClient` 调用父类时并没有提供：

```python
client_addresses
```

其默认：

```python
client_addresses = None
```

所以当前进入：

```python
else:
    # Engines are managed by this client.
```

随后：

```python
with launch_core_engines(
    vllm_config,
    executor_class,
    log_stats,
    addresses,
) as (
    engine_manager,
    coordinator,
    addresses,
    tensor_queue,
):
    self.resources.coordinator = coordinator
    self.resources.engine_manager = engine_manager
```

因此链路继续：

```text
SyncMPClient
↓
MPClient
↓
launch_core_engines(...)
```

---

# 10. `launch_core_engines()` 应该怎么读

这个函数较长，而且包含：

```text
DP
Ray
Coordinator
Multi-node
ZMQ
Multimodal IPC
Elastic EP
```

如果逐行读，非常容易迷路。

因此第一遍只提取控制流骨架：

```text
launch_core_engines()
│
├── 读取 parallel config
│
├── 可选创建 tensor IPC queue
│
├── 可选启动 DP Coordinator
│
├── Ray backend
│   └── CoreEngineActorManager
│
├── 确定 EngineCore handshake 对象
│
├── 创建 handshake 地址
│
└── 本地 Engine
    └── CoreEngineProcManager
```

当前实验：

```text
DP=1
TP=1
PP=1
非 Ray
本地 offline LLM
```

因此当前重点可以直接收敛到：

```python
with zmq_socket_ctx(
    local_handshake_address,
    zmq.ROUTER,
    bind=True,
) as handshake_socket:

    if local_engine_count:
        local_engine_manager = CoreEngineProcManager(...)
```

源码注释也直接写：

```python
# Start local engines.
```

因此：

```text
launch_core_engines
↓
CoreEngineProcManager
```

成为下一跳。

---

# 11. ZMQ handshake socket 不是创建 Process 的地方

这里：

```python
with zmq_socket_ctx(...) as handshake_socket:
```

负责的是：

```text
进程间通信 / 启动握手
```

不是创建 Linux Process。

可以暂时理解为：

```text
Frontend
↓
启动 EngineCore
↓
EngineCore 初始化 CUDA / Model / KV Cache / IPC
↓
EngineCore 发 ready
↓
Frontend handshake socket 收到
↓
确认启动完成
```

因此：

```text
zmq_socket_ctx
→ 通信设施

CoreEngineProcManager
→ Process 生命周期管理
```

当前追 Process 时，重点是后者。

---

# 12. `@contextmanager + yield` 在这里做什么

`launch_core_engines()`：

```python
@contextlib.contextmanager
def launch_core_engines(...):
```

内部：

```python
yield local_engine_manager, coordinator, addresses, tensor_queue

wait_for_engine_startup(...)
```

调用方：

```python
with launch_core_engines(...) as (
    engine_manager,
    coordinator,
    addresses,
    tensor_queue,
):
    self.resources.coordinator = coordinator
    self.resources.engine_manager = engine_manager
```

执行顺序不是：

```text
launch_core_engines 全部执行完
↓
返回
```

而是：

```text
进入 launch_core_engines
↓
创建资源
↓
执行到 yield
↓
launch_core_engines 暂停
↓
把 manager 等对象交给外部 with
↓
MPClient 保存资源引用
↓
with block 结束
↓
回到 launch_core_engines
↓
从 yield 后继续
↓
wait_for_engine_startup(...)
```

即：

```text
创建资源
↓
交出资源
↓
调用方登记 ownership
↓
再继续启动检查 / handshake
```

---

# 13. 为什么要先把资源交给 `MPClient`

调用方在 `yield` 阶段立即：

```python
self.resources.engine_manager = engine_manager
```

这样 `MPClient.resources` 尽早知道：

```text
“这些 EngineCore Process 是我需要负责管理和清理的资源。”
```

如果采用：

```python
manager = CoreEngineProcManager(...)
wait_for_engine_startup(...)
return manager
```

假设：

```text
Process 已经创建
↓
EngineCore 初始化失败
↓
wait_for_engine_startup 抛异常
↓
函数还没有 return manager
```

调用方可能还没有拿到 manager。

这样异常路径上的 Process / Socket 等资源清理会更复杂。

当前代码采取：

```text
资源一创建
↓
先把 ownership / 引用登记给统一 Resource Manager
↓
再执行可能失败的 startup / handshake
```

这是一种典型的资源生命周期管理思路。

需要注意：

> 从当前代码可以确定执行顺序确实是“先保存 manager，再继续 startup wait”；“主要为了异常清理”是根据 `BackgroundResources`、finalizer 和 shutdown 结构做出的合理设计推断。

---

# 14. 第六步：CoreEngineProcManager 构造 Process

`CoreEngineProcManager` 的职责已经直接写在 docstring：

```text
creation
readiness
shutdown
```

初始化：

```python
context = get_mp_context()
```

获得 multiprocessing context。

然后：

```python
self.processes: list[BaseProcess] = []
```

Manager 保存它负责管理的所有 Process。

对于每个 local Engine：

```python
for index in range(local_engine_count):
```

创建：

```python
self.processes.append(
    context.Process(
        target=EngineCoreProc.run_engine_core,
        name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
        kwargs=common_kwargs
        | {
            "dp_rank": global_index,
            "local_dp_rank": local_index,
        },
    )
)
```

这里有三个重要信息。

---

## 14.1 `context.Process(...)`

创建的是：

```text
Python multiprocessing Process Object
```

此时还不能说 Linux 子进程已经启动。

只是定义：

```text
未来启动什么 Process
运行什么函数
叫什么名字
传什么参数
```

---

## 14.2 target

```python
target=EngineCoreProc.run_engine_core
```

表示这个 Process 真正启动后，新 Process 的入口是：

```python
EngineCoreProc.run_engine_core(...)
```

即：

```text
EngineCore Process
↓
run_engine_core()
```

---

## 14.3 name

当前：

```text
DP = 1
```

所以：

```python
is_dp = False
```

Process name：

```python
name="EngineCore"
```

这与 Linux 中看到的：

```text
VLLM::EngineCore
```

存在对应关系。

`VLLM::` 前缀具体在哪里设置，当前阶段不继续追。

---

# 15. Process Object 和 Linux Process 不是一回事

这是这一轮最重要的 Python/Linux 知识之一。

下面：

```python
p = context.Process(
    target=foo,
)
```

只是创建：

```text
Process Object
```

真正：

```python
p.start()
```

之后才会：

```text
multiprocessing
↓
创建新的 OS Process
↓
分配新的 PID
↓
新 Process 执行 foo()
```

所以必须继续找：

```text
self.processes 在哪里 start？
```

使用：

```bash
rg -n "self\.processes|proc\.start|process\.start" \
  vllm/v1/engine/utils.py
```

最终定位：

```python
proc.start()
```

---

# 16. 真正创建 Linux EngineCore Process

实际代码骨架：

```python
for proc, local_dp_rank in zip(self.processes, local_dp_ranks):

    # 特定 DP/backend 下准备 GPU mapping
    ...

    with numa_utils.configure_subprocess(
        vllm_config,
        local_rank=0,
        dp_local_rank=local_dp_rank,
        process_kind="EngineCore",
    ):
        proc.start()
```

真正关键：

```python
proc.start()
```

此时发生：

```text
Frontend Python Process
↓
multiprocessing start/spawn
↓
Linux 创建新 Process / 新 PID
↓
EngineCoreProc.run_engine_core(...)
```

因此这一行和动态实验完全对应：

```text
Python 源码：

proc.start()

        ↓

Linux：

python smoke_06b.py
PID 3907513
    │
    └── VLLM::EngineCore
        PID 3908079
```

---

# 17. 为什么 `start()` 前还要修改配置

代码注释：

```python
# Mutating the config before each proc.start() works because the spawn method
# pickles process args at start() time, sequentially per rank.
```

关键是：

```text
Process(...)
```

阶段还没有把最终运行参数完全发送给新 Process。

在：

```python
proc.start()
```

时，spawn 才会序列化/pickle 对应参数。

所以 vLLM 可以：

```text
Process Object 已创建
↓
针对当前 DP rank 修改 GPU / rank 配置
↓
proc.start()
↓
当前配置被 pickle 给该子进程

下一 DP rank
↓
修改成另一组配置
↓
再 proc.start()
```

因此能够逐 rank 启动不同 EngineCore。

当前：

```text
DP=1
```

这些复杂 DP 映射逻辑不是主线，可以先跳过。

---

# 18. `configure_subprocess()` 的当前理解

```python
with numa_utils.configure_subprocess(...):
    proc.start()
```

当前不深入 NUMA。

只需要理解：

> 在创建 EngineCore Process 前，为这个子进程临时准备 rank / GPU / NUMA 等运行环境，然后调用 `start()`。

即：

```text
配置子进程运行环境
↓
proc.start()
↓
子进程启动
↓
父进程恢复自己的环境
```

---

# 19. EngineCore 启动失败怎么处理

启动过程：

```python
try:
    for ...:
        ...
        proc.start()
finally:
    if self.finished_procs():
        self.shutdown()
```

这里的目标是避免部分 Process 启动成功、部分失败后留下残留资源。

例如：

```text
P0 start 成功
P1 start 成功
P2 初始化失败
```

如果不统一处理：

```text
P0 / P1 可能继续存在
GPU memory 没释放
socket 没释放
```

因此异常路径调用：

```python
self.shutdown()
```

这和前面：

```text
BackgroundResources
weakref.finalize
CoreEngineProcManager
```

共同体现 vLLM 对后台 Process 生命周期的管理。

---

# 20. 第一条源码链完整闭环

最终，从用户 API 到 Linux EngineCore Process：

```text
smoke_06b.py

LLM(...)
│
▼
LLM.__init__()
│
│ self.llm_engine =
│ LLMEngine.from_engine_args(...)
▼
LLMEngine.from_engine_args()
│
│ VLLM_ENABLE_V1_MULTIPROCESSING=True
▼
enable_multiprocessing=True
│
▼
LLMEngine.__init__(
    multiprocess_mode=True
)
│
▼
EngineCoreClient.make_client(
    multiprocess_mode=True,
    asyncio_mode=False
)
│
▼
SyncMPClient
│
│ super().__init__()
▼
MPClient.__init__()
│
│ client_addresses=None
▼
launch_core_engines(...)
│
▼
CoreEngineProcManager(...)
│
▼
context.Process(
    target=EngineCoreProc.run_engine_core,
    name="EngineCore",
)
│
│ Process Object
▼
self.processes[]
│
▼
configure_subprocess(...)
│
▼
proc.start()
│
│ OS 创建真正的新 Process
▼
VLLM::EngineCore
│
▼
EngineCoreProc.run_engine_core(...)
```

---

# 21. 这条链路最终说明了什么

## 21.1 `LLM` 并不直接执行核心推理

用户：

```python
LLM(...)
```

首先处于 Frontend Process。

它负责：

```text
用户接口
EngineArgs
Input / Output Processor
EngineCore Client
```

而真正核心 Engine 被拆到了后台 Process。

---

## 21.2 Frontend 与 EngineCore 是独立 Process

动态：

```text
Frontend PID 3907513

EngineCore PID 3908079
PPID       3907513
```

静态源码：

```python
context.Process(...)
proc.start()
```

两者完全对应。

---

## 21.3 EngineCoreClient 是进程边界前的抽象层

结构：

```text
LLMEngine
↓
EngineCoreClient
```

Client 根据：

```text
multiprocess_mode
asyncio_mode
```

选择：

```text
InprocClient
SyncMPClient
AsyncMPClient
```

因此 Frontend API 与 EngineCore 执行方式被解耦。

---

## 21.4 当前 offline LLM 使用 SyncMPClient

当前实际：

```text
multiprocess_mode=True
asyncio_mode=False
```

所以：

```text
offline LLM
↓
同步 Frontend Client
+
独立 EngineCore Process
↓
SyncMPClient
```

---

## 21.5 MPClient 通过 ZMQ 与 EngineCore 通信

当前已经看到：

```text
input socket
output socket
handshake socket
```

可以先抽象：

```text
Frontend Process
    │
    │ EngineCoreRequest
    │ ZMQ
    ▼
EngineCore Process
    │
    │ EngineCoreOutputs
    │ ZMQ
    ▼
Frontend Process
```

ZMQ 协议和具体消息格式暂时不深入。

---

## 21.6 Process 生命周期被统一管理

不是简单：

```text
Process.start()
然后不管
```

而是：

```text
CoreEngineProcManager
├── creation
├── readiness
└── shutdown
```

再配合：

```text
BackgroundResources
weakref.finalize
shutdown()
```

形成：

```text
创建
↓
登记 ownership
↓
启动
↓
等待 ready
↓
运行
↓
异常/正常 shutdown
```

这是生产级 inference runtime 相比 nano-vLLM 更明显的一层工程复杂度。

---

# 22. 当前架构图

现阶段能够确认：

```text
┌─────────────────────────────────────────┐
│ Frontend Python Process                 │
│                                         │
│ LLM                                     │
│   │                                     │
│   └── LLMEngine                         │
│         │                               │
│         ├── InputProcessor              │
│         ├── OutputProcessor             │
│         │                               │
│         └── SyncMPClient                │
│               │                         │
│               ├── ZMQ sockets           │
│               ├── Output Thread         │
│               └── Engine Manager        │
└─────────────────┬───────────────────────┘
                  │
                  │ multiprocessing
                  │ ZMQ IPC
                  ▼
┌─────────────────────────────────────────┐
│ EngineCore Process                      │
│                                         │
│ PID != Frontend PID                     │
│                                         │
│ Entry:                                  │
│ EngineCoreProc.run_engine_core()        │
│                                         │
│ GPU / CUDA / KV Cache                  │
│ primarily owned here                    │
└─────────────────────────────────────────┘
```

当前尚未展开 EngineCore 内部：

```text
Scheduler
KVCacheManager
Executor
GPUWorker
ModelRunner
```

这是下一阶段。

---

# 23. 本轮源码追踪方法总结

这一轮真正需要复用的能力不是记住文件名，而是：

## Step 1：从真实运行入口开始

```text
LLM(...)
```

不要从内部模块随便挑一个类开始。

## Step 2：每轮只有一个问题

本轮：

```text
为什么出现独立 EngineCore Process？
```

## Step 3：看对象创建点

例如：

```python
self.llm_engine = ...
self.engine_core = ...
```

## Step 4：遇到配置分支追实际值

例如：

```text
multiprocess_mode
↓
enable_multiprocessing
↓
VLLM_ENABLE_V1_MULTIPROCESSING
↓
True
```

## Step 5：区分 Object / Thread / Process

```python
Thread(...)
```

不是 Linux EngineCore Process。

```python
Process(...)
```

只是 Process Object。

```python
proc.start()
```

才真正创建 OS Process。

## Step 6：大函数先看控制流骨架

`launch_core_engines()` 不逐行读：

```text
Ray？
DP Coordinator？
Local Engine？
```

根据当前实验配置删除无关分支。

## Step 7：静态源码必须与动态证据对应

源码：

```python
proc.start()
```

动态：

```text
Frontend PID
└── EngineCore PID
```

只有两边对应，结论才足够扎实。

---

# 24. 当前阶段完成状态

已确认：

```text
✓ LLM → LLMEngine
✓ multiprocessing 配置来源
✓ SyncMPClient 选择逻辑
✓ SyncMPClient → MPClient
✓ MPClient → launch_core_engines
✓ launch_core_engines → CoreEngineProcManager
✓ Process Object 创建位置
✓ proc.start() 真正启动位置
✓ EngineCore target = EngineCoreProc.run_engine_core
✓ Linux PID/PPID 与源码对应
✓ EngineCore 为主要 GPU memory owner
```

因此：

```text
VB-01-001 Part 1
Frontend → EngineCore Process
PASS
```

---

# 25. 下一阶段问题

下一阶段不再继续研究“谁创建 Process”。

新的问题是：

> **EngineCore Process 进入 `EngineCoreProc.run_engine_core()` 后，到底创建了哪些核心对象？**

重点将开始进入：

```text
EngineCoreProc
↓
EngineCore
↓
Scheduler
↓
KVCacheManager
↓
Executor
↓
GPUWorker / ModelRunner
```

下一轮仍然保持同样方法：

```text
先提出一个问题
↓
只定位一个入口
↓
看一小段源码
↓
确认对象 ownership
↓
再决定下一跳
```

而不是直接开始通读 `core.py`。

# 第二部分：从源码实现抽象 Runtime 设计原则

对，这个缺口很重要。

你现在这篇笔记主要回答了：

> **vLLM 是怎么做的？**

但对于真正做 AI Infra / 推理框架的人，还应该多两层：

> **为什么要这么做？解决了什么系统问题？**
> **这种设计能抽象成什么通用方法，以后看到 SGLang、LMCache、Ray、Serving Runtime 时也能复用？**

建议在现有 `02-process-and-control-plane.md` 里，**第 21 节“这条链路说明了什么”之后**，增加下面这一大节。

---

# 22. 为什么 vLLM 要把 Frontend 和 EngineCore 拆开

这一节不再描述源码，而是尝试从系统设计角度理解：

```text
LLM
↓
EngineCoreClient
↓
独立 EngineCore Process
```

背后到底解决什么问题。

首先要明确，如果是一个非常简单的推理程序，其实完全可以设计成：

```text
Python 用户程序
│
├── Tokenizer
├── Scheduler
├── KVCacheManager
├── ModelRunner
└── CUDA
```

也就是所有组件都在一个 Python Process：

```text
┌─────────────────────────────┐
│ Single Process              │
│                             │
│ API                         │
│ Scheduler                   │
│ KV Manager                  │
│ ModelRunner                 │
│ CUDA                        │
└─────────────────────────────┘
```

nano-vLLM 更接近这种结构。

这种设计的优点非常明显：

```text
简单
调用链短
容易 Debug
不需要 IPC
不需要序列化
不需要管理子进程
```

如果目标只是：

```text
理解推理流程
研究一个 Scheduler
实现一个 KV 算法
```

这种结构甚至非常好。

但生产 Serving Runtime 面临的问题不同。

---

# 23. 问题一：Frontend 和推理 Runtime 的生命周期不同

用户入口可能有很多种：

```text
Offline Python
OpenAI HTTP Server
gRPC Server
AsyncLLM
多 API Server
```

例如：

```python
llm.generate(...)
```

和：

```text
POST /v1/chat/completions
```

本质上都是：

```text
Request
↓
模型推理
↓
Output
```

但它们的 Frontend 完全不同。

如果把 Engine 和 Frontend 强绑定：

```text
OfflineLLMEngine
OpenAIServerEngine
AsyncEngine
GrpcEngine
```

容易产生大量重复实现。

更合理的抽象是：

```text
不同 Frontend
      │
      ▼
统一 Request / Client 边界
      │
      ▼
EngineCore
```

也就是：

```text
                ┌── Offline LLM
                │
                ├── AsyncLLM
                │
Frontend ───────┼── OpenAI Server
                │
                └── gRPC
                     │
                     ▼
              EngineCoreClient
                     │
                     ▼
                EngineCore
```

这样 EngineCore 不需要关心：

```text
请求来自 Python
还是 HTTP
还是 async server
```

它只关心：

```text
EngineCoreRequest
↓
Schedule
↓
Execute
↓
EngineCoreOutputs
```

---

# 24. 抽象一：Control Plane / Execution Plane 解耦

可以把当前架构粗略理解为两层。

Frontend 更接近：

```text
Control / Service Plane
```

负责：

```text
用户 API
请求接收
输入转换
输出转换
async / sync
连接管理
```

EngineCore 更接近：

```text
Inference Runtime / Execution Control Plane
```

负责：

```text
Scheduler
KV Cache 生命周期
Executor
GPU Worker
ModelRunner
```

因此形成：

```text
┌───────────────────────┐
│ Service / Frontend    │
│                       │
│ LLM / HTTP / Async    │
│ InputProcessor        │
│ OutputProcessor       │
└──────────┬────────────┘
           │
           │ Request / Output
           │
           ▼
┌───────────────────────┐
│ Engine Runtime        │
│                       │
│ Scheduler             │
│ KVCacheManager        │
│ Executor              │
│ GPU Worker            │
│ ModelRunner           │
└───────────────────────┘
```

这里最重要的设计思想是：

> **让用户服务形态和底层推理 Runtime 可以独立演化。**

---

# 25. 问题二：不同 Frontend 需要不同并发模型

这也是：

```python
EngineCoreClient.make_client(...)
```

存在的原因之一。

我们已经看到三种 Client：

```text
InprocClient
SyncMPClient
AsyncMPClient
```

它们面对的是不同部署模式。

例如 Offline：

```python
outputs = llm.generate(...)
```

用户期待同步：

```text
调用
↓
等待
↓
返回结果
```

所以：

```text
SyncMPClient
```

合适。

在线 API Server 则可能同时处理：

```text
Request A
Request B
Request C
Request D
...
```

Frontend 不希望：

```text
处理 A
↓
阻塞
↓
A 完成
↓
才能处理 B
```

所以需要 asyncio 风格：

```text
AsyncMPClient
```

但是无论 Frontend 是：

```text
sync
async
```

后面的核心 Runtime 仍然可以围绕同一个：

```text
EngineCore
```

构建。

因此这里使用了一种很典型的设计：

> **稳定核心抽象 + 可替换 Adapter/Client。**

---

# 26. 抽象二：接口稳定，通信实现可替换

可以把：

```text
EngineCoreClient
```

理解为一个抽象边界。

上层只关心：

```text
send request
receive output
abort request
utility call
```

至于底层到底是：

```text
同进程直接函数调用
ZMQ + Process
async ZMQ
```

由具体 Client 实现。

结构类似：

```text
                 EngineCoreClient
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    InprocClient   SyncMPClient   AsyncMPClient
          │             │             │
       direct        ZMQ sync      ZMQ async
          │             │             │
          └─────────────┴─────────────┘
                        │
                     EngineCore
```

这是一种非常通用的系统设计：

> **上层依赖抽象接口，而不是依赖具体通信机制。**

这样以后底层通信机制发生变化，上层 `LLM` API 不一定需要重写。

---

# 27. 问题三：推理 Runtime 是长生命周期、有状态的系统

EngineCore 并不是：

```text
收到一个请求
↓
创建模型
↓
推理
↓
销毁
```

而是一个典型的长期运行 Runtime：

```text
启动
↓
加载模型
↓
预分配 KV Cache
↓
建立 Scheduler
↓
等待 Request
↓
不断 schedule / execute
↓
Request finish
↓
继续等待下一个 Request
```

其中存在大量长期状态：

```text
Model Weight
KV Cache Pool
Scheduler Queue
Request State
BlockPool
CUDA Context
Worker
```

所以 EngineCore 更像：

> **一个持续运行的推理服务内核。**

这也是为什么：

```python
target=EngineCoreProc.run_engine_core
```

最终很可能进入一个：

```text
busy loop
```

而不是执行一次函数就退出。

---

# 28. 抽象三：Stateful Runtime 独立持有状态

这是 AI Serving 中非常重要的一种设计。

Frontend 可以相对“轻”：

```text
HTTP Connection
Request parsing
Tokenization
Output formatting
```

而 Runtime 持有“重状态”：

```text
Model
KV Cache
Scheduler
CUDA Context
```

于是：

```text
Frontend
    │
    │ Request
    ▼
Stateful Engine Runtime
    │
    │ Output
    ▼
Frontend
```

这样有一个很重要的性质：

> **核心推理状态不会随着某个单独 Request 的结束而销毁。**

这就是 serving 和普通 Python function 的根本区别之一。

---

# 29. 问题四：为什么使用独立 Process，而不只是 Object？

如果只是代码组织，完全可以：

```python
self.engine_core = EngineCore(...)
```

所有东西仍然在同一个 Process。

但 vLLM 当前选择：

```text
Frontend Process
        │
        │ IPC
        ▼
EngineCore Process
```

Process 隔离带来几个系统属性。

---

## 29.1 故障边界更明确

如果 EngineCore：

```text
CUDA error
Worker crash
模型初始化失败
```

它属于独立后台 Process。

Frontend 可以：

```text
检测死亡
做 cleanup
报告 EngineDead
```

代码里我们也已经看到：

```text
monitor
finished_procs()
shutdown()
ENGINE_CORE_DEAD
```

这种设计倾向。

相比全部塞进一个 Process，故障生命周期更加明确。

---

## 29.2 资源 ownership 更清晰

我们动态观察到：

```text
EngineCore PID
↓
持有约 75GB GPU Memory
```

所以可以形成清楚的 ownership：

```text
Frontend Process
→ 用户/API相关资源

EngineCore Process
→ CUDA
→ Model
→ KV Cache
→ Executor
```

当 EngineCore Process 生命周期结束时，可以集中释放这一批 Runtime 资源。

---

## 29.3 更容易扩展多 Process / 多 Engine

现在我们只跑：

```text
DP=1
```

所以只有：

```text
EngineCore
```

但源码已经明显考虑：

```text
DP rank 0
DP rank 1
DP rank 2
...
```

因此：

```python
self.processes: list[BaseProcess]
```

而不是：

```python
self.process: Process
```

这说明设计本身就考虑：

```text
一个 Manager
↓
管理多个 EngineCore Process
```

于是可以进一步扩展到：

```text
多 DP Engine
多节点
Ray Actor
External LB
```

当前 smoke 虽然没有走这些路径，但架构已经为它们留出了空间。

---

# 30. 抽象四：Manager 管理“多个同构 Runtime 实例”

这是另一种常见模式：

```text
Manager
│
├── Runtime 0
├── Runtime 1
├── Runtime 2
└── Runtime N
```

vLLM 当前是：

```text
CoreEngineProcManager
│
├── EngineCore DP0
├── EngineCore DP1
└── ...
```

Manager 不负责真正执行模型；

它负责的是：

```text
creation
start
readiness
failure detection
shutdown
```

因此：

> **执行逻辑和生命周期管理分离。**

这是非常成熟的 runtime 设计思想。

---

# 31. 为什么 `CoreEngineProcManager` 不直接做推理

如果把：

```text
启动 Process
监控 Process
Scheduler
KV
GPU Execution
```

全部放进一个 class，会变成一个巨型对象。

当前设计则分：

```text
CoreEngineProcManager
↓
管理 EngineCore Process 生命周期

EngineCoreProc / EngineCore
↓
管理推理运行逻辑
```

可以抽象为：

```text
Lifecycle Manager
        │
        ▼
Runtime Instance
```

这也是后面理解 Worker Manager、Executor 等组件时值得观察的模式。

---

# 32. 问题五：为什么要有 handshake / ready，而不是 `start()` 后立即使用

这一点也非常典型。

执行：

```python
proc.start()
```

只说明：

> OS Process 已经创建。

不代表：

```text
模型加载完成
CUDA 初始化完成
KV Cache 创建完成
ZMQ socket ready
Runtime 可以处理 Request
```

所以必须区分：

```text
Process Alive
```

和：

```text
Service Ready
```

两种状态。

例如：

```text
proc.start()
↓
PID 已经出现
↓
EngineCore 初始化
↓
load model
↓
create KV cache
↓
setup communication
↓
READY
```

这就是为什么存在：

```text
handshake
wait_for_engine_startup()
```

---

# 33. 抽象五：Liveness ≠ Readiness

这是服务系统中非常通用的一条原则：

```text
Process 存活
≠
服务可用
```

例如 Kubernetes 里也会区分类似：

```text
liveness
readiness
```

vLLM 当前这一套 startup handshake 本质上也是类似思想：

```text
process start
↓
liveness
↓
runtime initialization
↓
handshake success
↓
readiness
```

所以不能：

```text
proc.start()
↓
立即发送推理请求
```

必须等 EngineCore ready。

---

# 34. 为什么要先登记资源 ownership，再等待 ready

我们刚才特别讨论了：

```python
yield engine_manager
```

之后：

```python
self.resources.engine_manager = engine_manager
```

然后才：

```python
wait_for_engine_startup(...)
```

这个顺序体现的不是 vLLM 特有技巧，而是一种通用资源管理原则：

> **Acquire → Register Ownership → Perform Fallible Initialization**

即：

```text
创建资源
↓
立即登记资源归属
↓
再执行可能失败的操作
```

而不是：

```text
创建资源
↓
执行大量可能失败的初始化
↓
全部成功后才登记
```

为什么？

假设：

```text
Process 已创建
↓
GPU 初始化失败
↓
Exception
```

如果资源已经登记：

```text
Resource Manager
↓
知道该 terminate 谁
↓
知道该 close 哪些 socket
```

否则容易产生：

```text
orphan process
GPU memory leak
socket leak
```

---

# 35. 抽象六：RAII / Structured Resource Management 思想

虽然 Python 不完全等于 C++ RAII，但这里的：

```text
contextmanager
weakref.finalize
BackgroundResources
CoreEngineProcManager
shutdown
```

体现的是类似思想：

> **资源生命周期必须和某个明确的管理对象绑定。**

即：

```text
谁创建
谁登记
谁监控
谁释放
```

而不是：

```text
很多地方都可以创建
但没人明确负责 cleanup
```

这种设计对于：

```text
GPU memory
Process
Socket
Shared Memory
CUDA Context
```

尤其重要。

---

# 36. 为什么 Process 创建和 Process 启动分成两步

源码：

```python
context.Process(...)
```

然后：

```python
proc.start()
```

不是多余。

分两步后，系统可以：

```text
先定义：
这个 Process 要运行什么
传什么参数
叫什么名字

↓

然后在真正 start 前：
为当前 DP rank 修改配置
设置设备映射
设置 NUMA
设置环境变量

↓

最后：
proc.start()
```

也就是说：

> **Description 与 Activation 分离。**

先描述资源：

```text
Process Specification
```

再准备环境：

```text
Environment Configuration
```

最后激活：

```text
start
```

---

# 37. 抽象七：Configure Before Activate

这也是通用系统设计方法：

```text
Construct
↓
Configure
↓
Activate
```

而不是：

```text
Construct + Immediately Run
↓
然后再想办法修改
```

类似的设计在很多地方都存在：

```text
创建 CUDA Graph
↓
配置
↓
launch

创建线程
↓
设置参数
↓
start

创建网络服务
↓
bind
↓
listen
```

vLLM EngineCore Process 当前也是：

```text
Process(...)
↓
配置 GPU / NUMA / rank
↓
start()
```

---

# 38. 整条链路可以抽象成一个通用 Runtime 启动模型

把 vLLM 的具体类名去掉，得到：

```text
User API
↓
Frontend Object
↓
Runtime Client Abstraction
↓
选择通信模式
↓
Runtime Manager
↓
创建 Runtime Process Specification
↓
登记 Resource Ownership
↓
配置 Runtime Environment
↓
Start Process
↓
Handshake
↓
Ready
↓
进入长期 Runtime Loop
```

这是比记：

```text
LLM → SyncMPClient → CoreEngineProcManager
```

更重要的东西。

因为未来你看 SGLang、Ray、LMCache、数据库、分布式 Serving Runtime 时，都可能看到类似结构。

---

# 39. 从 AI Infra 角度，这条链真正告诉我们的东西

这条源码不是单纯教我们：

> “vLLM 用 multiprocessing。”

真正重要的是它体现了一个生产推理 Runtime 的几个核心设计要求：

```text
1. API 与 Runtime 解耦
2. 同步/异步 Frontend 与核心 Engine 解耦
3. Stateful Runtime 独立持有 GPU/KV 状态
4. Runtime Process 生命周期统一管理
5. Process Alive 和 Engine Ready 分离
6. 资源创建后尽早登记 ownership
7. 配置和启动分离
8. 多实例/多 DP 扩展由 Manager 统一管理
9. 异常路径必须和正常路径一样重视 cleanup
```

这就是 nano-vLLM 到 vLLM 一个非常明显的工程化差别：

```text
nano-vLLM：
重点是“推理算法怎么跑起来”

vLLM：
除了算法，还必须解决
“这个 Runtime 怎么长期、稳定、可扩展地活着”
```

---

# 40. 对我们 QCache 项目的启示

这部分对后面的 QCache 很重要。

以后实现 Quantized KV Cache 时，不应该只问：

```text
INT4 quant kernel 写在哪里？
KV block 如何压缩？
```

还必须问：

```text
这个状态由谁持有？

生命周期跟 Request 还是 EngineCore？

谁负责创建低比特 KV Pool？

谁负责释放？

如果初始化失败怎么办？

配置从 Frontend 如何传到 EngineCore？

统计信息怎么跨 Process 返回？

如果未来 TP/DP 扩展，状态是 per-rank 还是 shared？
```

因此这次进程链学习，实际上是在提前训练：

> **“一个优化模块应该插在哪里，并由谁拥有”**

这种 production-runtime 思维。

这比单纯能写一个 Triton kernel 更接近推理框架工程。

---

我建议把这部分放进 `02-process-and-control-plane.md`，标题直接叫：

```text
第二部分：从源码实现抽象 Runtime 设计原则
```

这样整篇笔记形成三层：

```text
第一层：怎么追出来
→ 源码分析方法

第二层：vLLM 实际怎么做
→ Frontend → EngineCore 调用链

第三层：为什么这么做
→ 解耦、ownership、readiness、lifecycle、failure handling
```

这三层齐了，才是一篇真正有价值的源码学习笔记。
