下面这版可以直接放进你的 `00-docs/source-reading/vllm/architecture/01-process-architecture-and-control-plane.md`，作为 **Executor / EngineCore Runtime** 这一节。重点不是简单记录“UniProcExecutor 是什么”，而是保留我们实际追源码时解决问题的过程和设计理解。

# vLLM v0.26.0 定向源码学习：EngineCore → Executor 执行层分析

> 固定版本：vLLM v0.26.0
> commit：`568afb3a13806beb53bb2e6bd518269357b237c0`
> 当前实验配置：
>
> * Offline `LLM`
> * TP = 1
> * PP = 1
> * DP = 1
> * V2 Model Runner
> * A100 80GB
> * `VLLM_ENABLE_V1_MULTIPROCESSING=True`

---

## 1. 本阶段解决的问题

前面已经确认：

```text
Frontend Linux Process
        │
        │ ZMQ
        ▼
EngineCore Linux Process
        │
        └── EngineCoreProc Python Object
                IS-A
             EngineCore
```

进入 `EngineCore.__init__()` 后，第一个核心对象是：

```python
self.model_executor = executor_class(vllm_config)
```

因此这一阶段真正需要回答：

```text
1. executor_class 当前到底是什么？

2. executor_class 是怎么根据当前配置选出来的？

3. self.model_executor 是：
   - Linux Process？
   - Thread？
   - Python Object？

4. 为什么 EngineCore 不直接调用 Worker，
   而要增加 Executor 这一层？

5. Executor 是怎么被 EngineCore Runtime 驱动的？

6. 当前实际使用普通 step()
   还是 step_with_batch_queue()？

7. PP、batch queue、async scheduling
   三者之间到底是什么关系？
```

---

# 2. 从 `executor_class` 开始追

EngineCore：

```python
self.model_executor = executor_class(vllm_config)
```

一开始不能看到 `Executor` 就猜：

```text
Executor = 一个进程
Executor = Worker Pool
Executor = GPU Worker
```

第一件事应该是追：

> `executor_class` 是从哪里来的？

搜索：

```bash
rg -n "executor_class" vllm/v1/engine vllm/v1/executor
```

得到关键链：

```text
LLMEngine
↓
Executor.get_class(vllm_config)
↓
executor_class
↓
一路作为 Python Class 传递
↓
EngineCore.__init__()
↓
executor_class(vllm_config)
```

其中最关键的是：

```python
executor_class = Executor.get_class(vllm_config)
```

---

# 3. `Executor.get_class()`：选择 Executor implementation

核心源码逻辑：

```python
parallel_config = vllm_config.parallel_config

distributed_executor_backend = (
    parallel_config.distributed_executor_backend
)

if distributed_executor_backend == "ray":
    executor_class = ...

elif distributed_executor_backend == "mp":
    executor_class = MultiprocExecutor

elif distributed_executor_backend == "uni":
    executor_class = UniProcExecutor

elif distributed_executor_backend == "external_launcher":
    executor_class = ExecutorWithExternalLauncher

...
```

因此：

```text
Executor.get_class()
```

不是创建 Executor Object。

它只是：

> 根据配置选择一个具体的 Executor implementation **Class**。

例如：

```python
executor_class = UniProcExecutor
```

此时：

```text
executor_class
= Python Class
≠ Python Object
≠ Linux Process
≠ Thread
```

直到：

```python
self.model_executor = executor_class(vllm_config)
```

才真正相当于：

```python
self.model_executor = UniProcExecutor(vllm_config)
```

这里才创建 Python Object。

---

# 4. 当前为什么选中 `UniProcExecutor`

接下来问题变成：

> 当前 `distributed_executor_backend` 是什么？

关键配置在：

```text
vllm/config/parallel.py
```

核心逻辑：

```python
if (
    self.distributed_executor_backend is None
    and self.world_size_across_dp > 1
):
    ...
    self.distributed_executor_backend = backend

if (
    self.distributed_executor_backend is None
    and self.world_size == 1
):
    self.distributed_executor_backend = "uni"
```

我们当前：

```text
TP = 1
PP = 1
DP = 1
```

因此当前没有需要 distributed executor topology 的多 rank execution。

最终：

```text
distributed_executor_backend
        ↓
       "uni"
```

于是：

```text
"uni"
↓
Executor.get_class()
↓
UniProcExecutor
```

因此当前已经可以确定：

```python
executor_class = UniProcExecutor
```

最终：

```python
self.model_executor = UniProcExecutor(vllm_config)
```

---

# 5. 一个非常重要的区分：`UniProc` ≠ 整个 vLLM 单进程

这是这一阶段最容易误解的地方。

前面已经通过 PID 实验确认：

```text
Frontend Linux Process
        │
        │ ZMQ
        ▼
EngineCore Linux Process
```

但是现在：

```text
distributed_executor_backend = "uni"
```

两者完全不冲突。

因为它们解决的是**两个不同维度的问题**。

---

## 5.1 Frontend / EngineCore Process Boundary

由：

```text
VLLM_ENABLE_V1_MULTIPROCESSING
```

控制。

当前：

```text
Frontend Process
        │
       ZMQ
        ▼
EngineCore Process
```

这是：

> Frontend 和 Core Runtime 是否处于独立 Linux Process。

---

## 5.2 Executor Backend

由：

```text
distributed_executor_backend
```

控制。

它回答：

> EngineCore 内部的 model execution 采用什么 execution topology。

可能包括：

```text
uni
mp
ray
external launcher
...
```

所以当前结构是：

```text
Frontend Linux Process

        │ ZMQ

EngineCore Linux Process
│
├── EngineCoreProc object
│
├── Scheduler object
│
└── UniProcExecutor object
```

因此：

> `UniProcExecutor` 的 UniProc 不能理解成“整个 vLLM 只有一个 Linux Process”。

当前更准确的含义是：

> EngineCore 内部 model execution 不再通过 distributed/multiprocess Executor topology 展开。

---

# 6. `Executor` 与 `UniProcExecutor` 的继承关系

源码：

```python
class UniProcExecutor(Executor):
    ...
```

但是：

```text
UniProcExecutor
```

没有自己的：

```python
__init__()
```

最开始容易产生疑问：

> 那它怎么初始化？

Python 的规则是：

> 如果子类没有定义 `__init__()`，实例化子类时会沿 MRO 使用父类 `__init__()`。

因此：

```python
UniProcExecutor(vllm_config)
```

实际首先进入：

```python
Executor.__init__(...)
```

---

# 7. `Executor.__init__()`：Template Method

父类：

```python
def __init__(self, vllm_config):
    self.vllm_config = vllm_config
    self.model_config = ...
    self.cache_config = ...
    self.parallel_config = ...
    self.scheduler_config = ...
    self.device_config = ...
    ...

    self._init_executor()

    self.is_sleeping = False
    self.sleeping_tags = set()
    self.kv_output_aggregator = None
```

关键在：

```python
self._init_executor()
```

虽然代码写在：

```text
Executor.__init__()
```

里面，但是 `self` 实际是：

```text
UniProcExecutor Object
```

所以 Python 动态派发会调用：

```python
UniProcExecutor._init_executor()
```

整个流程：

```text
UniProcExecutor(vllm_config)
        ↓
子类没有 __init__
        ↓
Executor.__init__()
        ↓
初始化公共 config/state
        ↓
self._init_executor()
        ↓ 动态派发
UniProcExecutor._init_executor()
        ↓
回到 Executor.__init__()
        ↓
初始化剩余公共 runtime state
```

---

# 8. 为什么不让每个子类自己写 `__init__ + super().__init__()`？

当然可以写成：

```python
class UniProcExecutor(Executor):
    def __init__(self, config):
        super().__init__(config)
        ...
```

但是 vLLM 当前选择：

```text
父类 __init__
+
子类 _init_executor()
```

本质上是经典的：

> **Template Method**

父类控制生命周期骨架：

```text
公共配置初始化
↓
backend-specific initialization
↓
公共 runtime state 初始化
```

子类只负责：

```text
backend-specific initialization
```

---

## 8.1 好处一：公共初始化逻辑只有一份

所有 Executor 都需要：

```text
vllm_config
model_config
cache_config
parallel_config
scheduler_config
device_config
...
```

不需要每个 Executor 重复。

---

## 8.2 好处二：父类掌握初始化顺序

如果每个子类都自己：

```python
super().__init__()
```

可能出现：

```text
Subclass A：
先 backend init
再 super

Subclass B：
先 super
再 backend init

Subclass C：
忘记 super
```

生命周期可能逐渐失控。

现在则由：

```text
Executor.__init__()
```

统一约束：

```text
哪些状态必须先 ready
↓
什么时候初始化 backend
↓
什么时候建立其余公共 runtime state
```

---

## 8.3 好处三：Executor implementation 只实现差异

例如概念上：

```text
Executor.__init__()
        │
        └── _init_executor()
                 │
        ┌────────┼───────────┐
        ▼        ▼           ▼
      Uni      MP           Ray
```

不同 backend 只关心：

```text
我应该如何建立自己的 execution environment？
```

而不需要重新实现整个 Executor 生命周期。

---

# 9. `UniProcExecutor._init_executor()` 到底做了什么

核心：

```python
def _init_executor(self):
    self.driver_worker = WorkerWrapperBase(rpc_rank=0)

    distributed_init_method, rank, local_rank = (
        self._distributed_args()
    )

    kwargs = dict(
        vllm_config=self.vllm_config,
        local_rank=local_rank,
        rank=rank,
        distributed_init_method=distributed_init_method,
        is_driver_worker=True,
        shared_worker_lock=Lock(),
    )

    self.driver_worker.init_worker(all_kwargs=[kwargs])
    self.driver_worker.init_device()
    self.driver_worker.load_model()
```

第一遍不要进入 Worker 内部。

只抽取 control-flow skeleton：

```text
UniProcExecutor._init_executor()
↓
创建 WorkerWrapperBase Python Object
↓
准备 rank / local_rank / distributed_init_method
↓
init_worker()
↓
init_device()
↓
load_model()
↓
execution environment ready
```

当前对象所有权：

```text
EngineCoreProc Object
│
└── model_executor
      ↓
   UniProcExecutor Object
      │
      └── driver_worker
            ↓
         WorkerWrapperBase Object
```

注意目前这里：

```text
没有看到 multiprocessing.Process()
没有看到 proc.start()
没有看到 Thread()
没有看到 Ray Actor
```

所以不能把：

```text
driver_worker
```

画成另一个 Linux Process。

---

# 10. 一个新的疑问：UniProcExecutor 中只有方法定义，谁调用它？

这是阅读 class 时很容易产生的问题。

看到：

```python
def execute_model(...)
def collective_rpc(...)
def shutdown(...)
```

容易觉得：

> 这里只有函数定义，没有 main loop，那它什么时候运行？

答案：

> `UniProcExecutor` 本身不是主动驱动者，而是被 EngineCore 调用的服务对象。

应该去找：

```text
caller
```

而不是继续盯着 class 定义。

搜索：

```bash
rg -n "model_executor\.execute_model" vllm/v1/engine/core.py
```

找到：

```python
future = self.model_executor.execute_model(
    scheduler_output,
    non_block=True,
)
```

所以真正的调用关系是：

```text
EngineCore
↓
Scheduler.schedule()
↓
SchedulerOutput
↓
model_executor.execute_model()
↓
UniProcExecutor.execute_model()
```

---

# 11. `UniProcExecutor.execute_model()` 做了什么

源码：

```python
def execute_model(
    self,
    scheduler_output,
    non_block=False,
):
    output = self.collective_rpc(
        "execute_model",
        args=(scheduler_output,),
        non_block=non_block,
        single_value=True,
    )
    return output
```

继续：

```python
collective_rpc(...)
```

UniProc 里面核心是：

```python
run_method(
    self.driver_worker,
    method,
    args,
    kwargs,
)
```

因此当前可以抽象成：

```text
EngineCore
↓
UniProcExecutor.execute_model()
↓
collective_rpc("execute_model")
↓
run_method(driver_worker, ...)
↓
driver_worker.execute_model(...)
```

---

# 12. 为什么 UniProc 下还叫 `collective_rpc`

从当前实现看，UniProc 下：

```text
collective_rpc
```

并不是真的经过网络 RPC。

基本是：

```text
对本地唯一 driver_worker
进行 Python method dispatch
```

但 Executor 上层仍然保留统一接口。

可以概念化为：

```text
                    Executor
                       │
                collective_rpc
                       │
        ┌──────────────┼─────────────┐
        ▼              ▼             ▼
    UniProc           MP             Ray
        │              │             │
本地调用一个      多 Worker       Ray Actor
Worker           Process RPC      调用
```

因此 EngineCore 不需要关心：

```text
底下到底几个 Worker
Worker 是否跨进程
Worker 是否是 Ray Actor
```

EngineCore 只写：

```python
model_executor.execute_model(...)
```

---

# 13. 为什么 EngineCore 需要 Executor 层

如果没有 Executor：

```text
EngineCore
↓
需要自己知道
↓
当前是 Uni / MP / Ray？
↓
怎么创建 Worker？
怎么发 RPC？
几个 Worker？
怎么收结果？
怎么 shutdown？
```

EngineCore 会同时承担：

```text
调度逻辑
+
Runtime orchestration
+
Worker topology
+
Process management
+
Distributed backend
+
RPC
```

耦合非常重。

有 Executor 以后：

```text
EngineCore
↓
Executor abstraction
↓
具体 backend
```

分工可以总结成：

### Scheduler

回答：

> **WHAT to run**

例如：

```text
本轮有哪些 request
每个 request 算多少 token
需要多少 KV resource
```

---

### EngineCore

回答：

> **WHEN / runtime orchestration**

负责：

```text
schedule
↓
submit execution
↓
管理 in-flight work
↓
获得 result
↓
update scheduler
```

---

### Executor

回答：

> **HOW / WHERE to execute**

负责把：

```text
SchedulerOutput
```

落实到：

```text
UniProc
Multiproc
Ray
...
```

具体 execution topology。

---

### Worker

更下面才是真正：

> 执行 Worker 级模型工作。

当前还没有深入。

因此现阶段不要写成：

```text
Executor = GPU 模型执行本身
```

更准确：

> **Executor 是 EngineCore 与具体 Worker execution topology 之间的 abstraction / adapter。**

---

# 14. EngineCore 有两个 `execute_model()` 调用点

搜索得到：

```text
step()
step_with_batch_queue()
```

这两个不是两个 Executor。

而是：

> EngineCore 的两种 Runtime Step 模式。

---

# 15. 普通 `step()` 模式

核心：

```python
scheduler_output = self.scheduler.schedule(...)

future = self.model_executor.execute_model(
    scheduler_output,
    non_block=True,
)

model_output = future.result()

...

engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output,
    model_output,
)
```

可以缩成：

```text
Scheduler.schedule()
↓
SchedulerOutput
↓
Executor.execute_model()
↓
等待 future.result()
↓
ModelRunnerOutput
↓
Scheduler.update_from_output()
```

控制流：

```text
Schedule A
↓
Execute A
↓
Wait A
↓
Update A
↓
Schedule B
```

虽然调用：

```python
non_block=True
```

但 EngineCore 随后马上：

```python
future.result()
```

所以从 EngineCore Runtime 角度看，仍然属于：

> submit 后立即消费结果。

---

# 16. `step_with_batch_queue()` 模式

这个函数多了一层：

```text
batch_queue
```

主干：

```python
scheduler_output = self.scheduler.schedule(...)

exec_future = self.model_executor.execute_model(
    scheduler_output,
    non_block=True,
)

batch_queue.appendleft(
    (future, scheduler_output, exec_future)
)
```

如果 queue 还有空间：

```text
不立即等待当前 result
↓
可以继续下一轮 schedule / submit
```

后面需要结果时：

```python
future, scheduler_output, exec_model_fut = (
    batch_queue.pop()
)
```

于是形成：

```text
Schedule A
↓
Submit A
↓
Future A 入 queue
↓
Schedule B
↓
Submit B
↓
Future B 入 queue
↓
之后再消费 A Result
```

核心设计：

> **Execution submission 与 result consumption 解耦。**

---

# 17. `non_block=True` 不等于创建 Thread / Process

这是一个重要误区。

看到：

```python
execute_model(..., non_block=True)
```

不能直接理解成：

```text
创建后台 Python Thread
```

也不能理解成：

```text
创建新的 Linux Process
```

UniProc `collective_rpc()` 中仍然：

```python
result = run_method(
    self.driver_worker,
    method,
    args,
    kwargs,
)
```

如果底层返回：

```text
AsyncModelRunnerOutput
```

才包装成异步 Future。

否则也可能只是构造一个已经完成的 `Future`。

因此：

```text
non-blocking API
≠ 新 Thread
≠ 新 Process
```

它表达的是：

> 上层 Runtime 可以用 Future-like abstraction 延后消费 execution result。

真正的异步执行机制最终怎么做到，需要后面进入 Worker / V2 Model Runner 才能确认。

---

# 18. 当前到底走 `step()` 还是 `step_with_batch_queue()`

EngineCore 初始化：

```python
self.batch_queue_size = (
    vllm_config.max_concurrent_batches
)

if self.batch_queue_size > 1:
    self.batch_queue = deque(
        maxlen=self.batch_queue_size
    )

self.step_fn = (
    self.step
    if self.batch_queue is None
    else self.step_with_batch_queue
)
```

所以：

```text
max_concurrent_batches <= 1
↓
step()

max_concurrent_batches > 1
↓
step_with_batch_queue()
```

---

# 19. `max_concurrent_batches` 又是怎么来的

源码：

```python
@property
def max_concurrent_batches(self) -> int:
    pp_size = (
        self.parallel_config.pipeline_parallel_size
    )

    if self.scheduler_config.async_scheduling:
        if self.use_v2_model_runner:
            return pp_size + 1

        if pp_size <= 1:
            return 2

    return pp_size
```

这时出现两个新的概念：

```text
PP
Async Scheduling
```

这里一开始很容易混。

---

# 20. PP 是什么

PP：

> Pipeline Parallelism，流水线并行。

它不是：

```text
允许多个 batch
```

PP 的本体是：

> 把模型不同层切成多个 Pipeline Stage。

例如：

```text
PP = 2

GPU0 / Stage0
Layer 0 ~ 15
       │
       │ hidden states
       ▼
GPU1 / Stage1
Layer 16 ~ 31
```

一个 batch：

```text
Input
↓
Stage 0
↓
发送 activation
↓
Stage 1
↓
Output
```

---

# 21. 为什么 PP 又需要多个 batch

如果：

```text
PP = 2
```

但系统永远只允许一个 batch：

```text
时间 T1

Stage0 : Batch A
Stage1 : 空闲
```

然后：

```text
时间 T2

Stage0 : 空闲
Stage1 : Batch A
```

两个 GPU 总有一个闲着。

真正希望：

```text
             T1       T2       T3

Stage0       A        B        C
Stage1       -        A        B
```

这样 T2：

```text
Stage0 正在算 Batch B
Stage1 正在算 Batch A
```

两个 stage 同时工作。

所以：

> PP 是“把模型按层切成多个 stage”。

而：

> 多个 concurrent batches 是“把这些 pipeline stage 填起来”的手段。

不能说：

```text
PP = 支持多个 batch
```

而应该说：

```text
PP
↓
产生多个 pipeline stages
↓
为了减少 pipeline bubble
↓
需要多个 batch 同时 in-flight
```

---

# 22. 当前为什么不是 PP 导致 batch_queue=2

当前：

```text
PP = 1
```

所以：

```text
模型只有一个 Pipeline Stage
```

不存在真正的 Pipeline Parallel。

如果：

```text
async_scheduling = False
```

则：

```python
max_concurrent_batches = pp_size = 1
```

最终：

```text
batch_queue = None
↓
step()
```

但我们当前实际还有：

```text
async_scheduling = True
```

因此：

```python
max_concurrent_batches
= pp_size + 1
= 1 + 1
= 2
```

所以当前：

```text
batch_queue_size = 2
↓
batch_queue enabled
↓
step_fn = step_with_batch_queue
```

因此必须记住：

> **当前出现 2 个 concurrent batches 不是因为 PP。**

而是：

```text
PP 基础需求：1
+
V2 Async Scheduling 额外需求：1
=
2
```

---

# 23. 当前 async scheduling 为什么是 True

SchedulerConfig 默认：

```python
async_scheduling: bool | None = None
```

在 `VllmConfig.__post_init__()` 中，会检查：

```text
Executor 是否支持 async scheduling
是否有 incompatible feature
模型类型
Spec Decode 配置
ROCm 特殊情况
...
```

我们当前：

```text
UniProcExecutor
```

实现：

```python
@classmethod
def supports_async_scheduling(cls) -> bool:
    return True
```

因此普通 Qwen3 generation + CUDA + 当前配置没有命中 incompatible 分支时：

```text
async_scheduling:
None
↓
自动设置
True
```

相关初始化逻辑也明确会最终打印 asynchronous scheduling 是否启用。

所以当前链：

```text
SchedulerConfig.async_scheduling
初始 None
↓
VllmConfig.__post_init__()
↓
UniProcExecutor supports async
↓
无 incompatible feature
↓
async_scheduling = True
```

---

# 24. 什么是 V2 Async Scheduling

这里最容易和“EngineCore Process 启动”混淆。

它**不是启动机制**。

启动阶段只是：

```text
检查是否支持
↓
async_scheduling=True
↓
配置 batch queue
↓
选择 step_with_batch_queue()
```

真正 Async Scheduling 发生在**推理 Runtime**。

---

## 普通同步 Runtime

```text
Schedule A
↓
Execute A
↓
Wait A
↓
Consume A output
↓
Update Scheduler
↓
Schedule B
```

必须消费完上一轮 execution result，才能继续推进。

---

## Async Scheduling

目标变成：

```text
Schedule A
↓
Submit A
↓
A execution in-flight
│
├──────────────┐
│              ↓
│         Schedule B
│              ↓
│         Submit B
│
↓
之后再 Consume A result
```

所以核心不是：

> “GPU 本来就是异步的。”

而是：

> **Scheduler submission 与之前 Model Execution Result 的 consumption 不再严格串行。**

因此叫：

```text
Async Scheduling
```

---

# 25. 为什么 V2 Async Scheduling 要 `pp_size + 1`

源码：

```python
if async_scheduling:
    if use_v2_model_runner:
        return pp_size + 1
```

理解成两个来源：

```text
pp_size
↓
满足 Pipeline 本身需要的 in-flight batches

+1
↓
给 Async Scheduling 留出额外一个 batch
```

例如当前：

```text
PP=1

一个 batch：
execution pipeline 中

额外一个 batch：
Scheduler 可以继续准备 / 提交
```

因此：

```text
max_concurrent_batches = 2
```

如果概念上：

```text
PP=2
async=True
```

则：

```text
max_concurrent_batches = 3
```

其中：

```text
2
→ Pipeline stages 的基本填充需求

+1
→ Async scheduling 的额外 overlap 空间
```

这里目前只是从配置和 EngineCore Runtime 层理解。

真正 ModelRunner 如何提供异步 execution，还没有追。

---

# 26. 当前完整 Runtime 路径

到这里当前 smoke 已经可以写成：

```text
Frontend Linux Process
        │
        │ ZMQ
        ▼
EngineCore Linux Process
        │
        ▼
EngineCoreProc Object
        │
        │ IS-A EngineCore
        │
        ├── Scheduler Object
        │
        └── UniProcExecutor Object
                │
                └── WorkerWrapperBase Object
```

初始化：

```text
ParallelConfig
↓
world_size = 1
↓
distributed_executor_backend = "uni"
↓
Executor.get_class()
↓
UniProcExecutor Class
↓
EngineCore.__init__()
↓
UniProcExecutor(vllm_config)
↓
Executor.__init__()
↓
UniProcExecutor._init_executor()
↓
Worker initialization
↓
device initialization
↓
model loading
```

同时：

```text
async_scheduling:
None → True
↓
PP = 1
↓
max_concurrent_batches = 2
↓
batch_queue enabled
↓
step_fn = step_with_batch_queue
```

真正推理：

```text
EngineCoreProc.run_busy_loop()
↓
EngineCore.step_fn()
↓
step_with_batch_queue()
↓
Scheduler.schedule()
↓
SchedulerOutput
↓
UniProcExecutor.execute_model()
↓
collective_rpc("execute_model")
↓
run_method(driver_worker, ...)
↓
Worker execution
↓
Future / Async result
↓
batch_queue
↓
ModelRunnerOutput
↓
Scheduler.update_from_output()
```

---

# 27. 当前几个对象分别是什么

| 名称                 | 类型                       | 是否 Linux Process | 当前作用                           |
| ------------------ | ------------------------ | ---------------: | ------------------------------ |
| Frontend           | Runtime                  |                是 | User API / request frontend    |
| EngineCore Process | OS Process               |                是 | Core Runtime 容器                |
| EngineCoreProc     | Python Object            |                否 | Process/ZMQ-enabled EngineCore |
| EngineCore         | Parent Class能力           |                否 | inference runtime core         |
| Scheduler          | Python Object            |                否 | 决定本轮算什么                        |
| UniProcExecutor    | Python Object            |                否 | execution backend adapter      |
| WorkerWrapperBase  | Python Object            |         否，目前源码如此 | Worker wrapper                 |
| batch_queue        | Python deque             |                否 | 保存 in-flight batch/Future      |
| Future             | Python async abstraction |                否 | 延迟消费 execution result          |

---

# 28. 这条源码分析过程中遇到的几个典型问题

## 问题 1：看到 Class 就容易当成 Runtime Object

例如：

```text
EngineCore
EngineCoreProc
Executor
UniProcExecutor
```

必须始终问：

```text
这是 Class？
还是 Object？
在哪里 new？
谁 owns？
```

---

## 问题 2：看到 `Process` / `Proc` 名字容易当成 Linux Process

必须找：

```python
multiprocessing.Process(...)
proc.start()
```

这种真正 OS boundary 的证据。

Python class 名字不能作为证据。

---

## 问题 3：看到 `UniProc` 容易理解成整个系统单进程

错误。

应该区分：

```text
Frontend/Core Process topology
```

和：

```text
Executor worker topology
```

这是两个独立维度。

---

## 问题 4：看到方法定义不知道什么时候执行

例如：

```python
UniProcExecutor.execute_model()
```

class 本身不会主动运行。

应该搜索：

```text
caller
```

也就是：

```bash
rg -n "model_executor\.execute_model"
```

---

## 问题 5：看到 `non_block=True` 就认为创建后台线程

错误。

应该继续确认：

```text
Thread?
Process?
CUDA async?
Future?
AsyncModelRunnerOutput?
```

当前只证明：

```text
Executor 暴露 Future-like non-blocking interface
```

不能进一步声称用了新线程。

---

## 问题 6：把 PP 和多个 Batch 混成一个概念

PP：

```text
模型按层切 stage
```

Concurrent batches：

```text
让不同 stage 同时有工作
```

关系是：

```text
PP 产生 pipeline
↓
多个 batch 填 pipeline
```

不是：

```text
PP = 多 batch
```

---

## 问题 7：把 Async Scheduling 当成启动机制

错误。

启动阶段只决定：

```text
enable or disable
```

真正 async scheduling 是：

```text
runtime schedule / execute / consume
```

之间的 overlap。

---

# 29. 这条链体现出的通用 Runtime 设计原则

## 29.1 Configuration → Implementation Selection

```text
配置
↓
选择具体 Class
↓
统一 Interface
```

例如：

```text
distributed_executor_backend
↓
Executor.get_class()
↓
Uni / MP / Ray
```

属于：

> backend selection / factory-like dispatch。

---

## 29.2 Template Method

```text
Executor.__init__()
↓
公共初始化
↓
_init_executor()
↓
backend-specific implementation
↓
公共 Runtime state
```

属于：

> 父类规定生命周期，子类实现变化点。

---

## 29.3 Runtime Orchestrator 与 Execution Backend 分离

```text
EngineCore
↓
Executor
↓
Worker
```

分别解决：

```text
EngineCore：
Runtime orchestration

Executor：
Execution topology abstraction

Worker：
Execution implementation
```

这样部署拓扑变化不会直接污染 Scheduler / EngineCore。

---

## 29.4 Submission 与 Completion 解耦

普通：

```text
submit
↓
wait
↓
consume
```

Async：

```text
submit A
↓
submit B
↓
consume A
```

通过：

```text
Future
+
batch_queue
```

表达 in-flight execution。

这是生产 Runtime 中非常常见的异步流水设计。

---

# 30. 当前阶段可以形成的最终认识

目前最准确的一句话是：

> **EngineCore 是 inference runtime 的驱动者，Scheduler 决定本轮执行什么，Executor 将 SchedulerOutput 映射到底层 execution topology。当前 TP=1 / PP=1 / DP=1 下选择 UniProcExecutor；UniProcExecutor 本身是 EngineCore Process 中的 Python Object，通过本地 driver worker 执行模型。由于 V2 Model Runner 对 async scheduling 的支持，当前配置默认启用 async scheduling，使 `max_concurrent_batches=2`，EngineCore 使用 `step_with_batch_queue()` 将 model execution submission 与 execution result consumption 解耦。**

---

# 31. 当前还没有回答的问题

以下内容**故意暂时不展开**：

```text
WorkerWrapperBase 为什么存在？

init_worker() 到底创建哪个真实 Worker？

Worker 是不是 GPUWorker？

GPU device / CUDA context 到底在哪一步建立？

Worker 如何创建 V2 Model Runner？

V2 Model Runner 为什么能返回 AsyncModelRunnerOutput？

ModelRunner 到底如何真正 launch GPU kernels？

KV Cache Tensor 最终由谁 allocate？
```

这些属于下一条：

```text
Executor
↓
WorkerWrapperBase
↓
Real Worker
```

当前不要同时进入：

```text
Scheduler
KVCacheManager
BlockPool
Attention
```

否则又会把 execution 线和 KV 线混在一起。

---

# 32. 当前源码链总结

最终可以压缩成：

```text
ParallelConfig
│
│ world_size=1
▼
distributed_executor_backend="uni"
│
▼
Executor.get_class()
│
▼
UniProcExecutor Class
│
▼
EngineCore.__init__()
│
▼
UniProcExecutor Object
│
│ Executor.__init__()
│ └─ _init_executor()
│       ↓
│   WorkerWrapperBase
│   init_worker
│   init_device
│   load_model
│
├───────────────────────────────┐
│                               │
│ Runtime                       │
▼                               │
step_with_batch_queue()         │
│                               │
├─ Scheduler.schedule()         │
│       ↓                       │
│  SchedulerOutput              │
│                               │
├─ Executor.execute_model() ────┘
│       ↓
│  collective_rpc
│       ↓
│  driver_worker
│
├─ Future / batch_queue
│
├─ ModelRunnerOutput
│
└─ Scheduler.update_from_output()
```

这条链到这里可以先视为：

> **EngineCore → Executor abstraction：完成。**

下一阶段再从：

```text
UniProcExecutor
↓
WorkerWrapperBase
```

开始，不提前进入 ModelRunner。
