# vLLM v0.26.0 源码学习笔记：EngineCore → Executor → Worker 执行链

> 版本：vLLM v0.26.0  
> Commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> 当前实验环境：Offline `LLM`，TP=1，PP=1，DP=1，V2 Model Runner，A100 80GB  
> 当前目标：不进入 Scheduler/KVCacheManager/Attention 内部，先把 **EngineCore 如何把 Scheduler 决策交给实际 GPU 执行层** 这条运行时链路彻底梳理清楚。

---

# 1. 为什么这一阶段从 `execute_model()` 开始

上一阶段已经确认了 vLLM V1 的外层 Runtime 结构：

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

并且已经知道 `EngineCoreProc` 负责：

- ZMQ / IPC；
- 输入输出线程；
- EngineCore Process 生命周期；
- signal / shutdown；
- `run_busy_loop()`。

而真正的 inference runtime 主体仍然来自父类 `EngineCore`。

进入 `EngineCore.__init__()` 后，我们看到：

```python
self.model_executor = executor_class(vllm_config)
kv_cache_config = self._initialize_kv_caches(vllm_config)
self.scheduler = Scheduler(...)
```

此时最自然的问题不是立刻进入 KV Cache，而是：

> **Scheduler 做出的调度决策，最终到底是怎么被执行到 GPU 上的？**

因此我们选择沿：

```text
EngineCore
↓
model_executor.execute_model(...)
```

这条链向下追。

这一阶段的核心问题是：

```text
1. executor_class 当前到底是谁？
2. Executor 是 Process、Thread，还是普通 Python Object？
3. 为什么 EngineCore 不直接调用 Worker / ModelRunner？
4. 当前 TP=1 / PP=1 / DP=1 为什么选择 UniProcExecutor？
5. execute_model() 到底是谁调用、怎么调用？
6. 当前 EngineCore 用 step() 还是 step_with_batch_queue()？
7. async scheduling 到底是什么？
8. WorkerWrapperBase 为什么存在？
9. Worker 是不是只有 TP 场景才需要？
10. 当前 CUDA Worker 是在哪里被选择出来的？
11. Worker 和 ModelRunner 又是什么关系？
```

这份笔记按照这条真实追踪过程组织。

---

# 2. 先从 Runtime 的 `execute_model()` 看起

在 `EngineCore` 中搜索：

```bash
rg -n "model_executor\.execute_model" vllm/v1/engine/core.py
```

可以找到两处主要调用：

```python
self.model_executor.execute_model(
    scheduler_output,
    non_block=True,
)
```

分别位于：

- `step()`
- `step_with_batch_queue()`

因此第一层 Runtime 主干已经很明确：

```text
EngineCore
↓
Scheduler.schedule()
↓
SchedulerOutput
↓
model_executor.execute_model()
↓
执行层
```

这里先得到一个非常重要的职责划分：

```text
Scheduler
= 决定 WHAT TO RUN

Executor
= 决定 HOW / WHERE TO EXECUTE

EngineCore
= 驱动整个 runtime，组织 schedule → execute → consume → update
```

也就是说：

> Scheduler 不直接执行 GPU forward；Executor 也不重新做 request-level scheduling。

Scheduler 输出的是 **执行计划**，Executor 负责把执行计划落实到底层 execution topology。

---

# 3. `executor_class` 到底是谁

在 `EngineCore.__init__()` 里：

```python
self.model_executor = executor_class(vllm_config)
```

这里的 `executor_class` 一开始只是一个 Python Class，不是 Object，更不是 Process。

继续追：

```bash
rg -n "executor_class" vllm/v1/engine vllm/v1/executor
```

找到：

```python
executor_class = Executor.get_class(vllm_config)
```

`Executor.get_class()` 的核心逻辑是：

```python
parallel_config = vllm_config.parallel_config
distributed_executor_backend = parallel_config.distributed_executor_backend

if distributed_executor_backend == "ray":
    ...
elif distributed_executor_backend == "mp":
    executor_class = MultiprocExecutor
elif distributed_executor_backend == "uni":
    executor_class = UniProcExecutor
elif distributed_executor_backend == "external_launcher":
    ...
```

所以：

```text
distributed_executor_backend
↓
Executor.get_class()
↓
具体 Executor Class
```

这里只做 **Class Selection**。

真正 Object creation 发生在：

```python
self.model_executor = executor_class(vllm_config)
```

---

# 4. 当前为什么是 `UniProcExecutor`

继续追：

```text
parallel_config.distributed_executor_backend
```

在 `ParallelConfig` 中，如果用户没有显式指定 backend：

```python
if (
    self.distributed_executor_backend is None
    and self.world_size == 1
):
    self.distributed_executor_backend = "uni"
```

当前：

```text
TP = 1
PP = 1
DP = 1
```

因此：

```text
world_size = 1
↓
distributed_executor_backend = "uni"
↓
Executor.get_class()
↓
UniProcExecutor
```

所以最终：

```python
self.model_executor = UniProcExecutor(vllm_config)
```

---

# 5. 一个非常重要的误区：`UniProcExecutor` ≠ 整个 vLLM 单进程

这是追源码过程中第一个容易混乱的地方。

我们前面已经通过 PID 和 `nvidia-smi` 实验确认：

```text
Frontend Linux Process
        │
        │ ZMQ
        ▼
EngineCore Linux Process
```

而现在：

```text
distributed_executor_backend = "uni"
```

两者完全不冲突。

它们属于两个不同维度：

## 5.1 Frontend / EngineCore Process Boundary

由 V1 multiprocessing 决定：

```text
Frontend
↓ ZMQ
EngineCore Process
```

这是 **API frontend 和 Core Runtime 是否隔离成不同 Linux Process**。

## 5.2 Executor topology

由：

```text
distributed_executor_backend
```

决定。

它回答的是：

> EngineCore 内部的模型执行 backend 如何组织 Worker。

因此当前结构是：

```text
Frontend Linux Process
        │
        │ ZMQ
        ▼
EngineCore Linux Process
        │
        └── EngineCoreProc Object
                │
                └── UniProcExecutor Object
```

`UniProc` 只意味着：

> 当前 Executor 层没有再展开成 Multiproc / Ray 的 Worker topology。

不能理解成：

> 整个 vLLM 只有一个 Linux Process。

---

# 6. `Executor.__init__()` 与 Template Method

`UniProcExecutor` 自己没有定义 `__init__()`。

所以：

```python
UniProcExecutor(vllm_config)
```

会进入父类：

```python
Executor.__init__()
```

核心：

```python
def __init__(self, vllm_config):
    self.vllm_config = ...
    self.model_config = ...
    self.cache_config = ...
    ...

    self._init_executor()

    self.is_sleeping = False
    ...
```

这里：

```python
self._init_executor()
```

会动态派发到：

```python
UniProcExecutor._init_executor()
```

因此链路：

```text
UniProcExecutor(vllm_config)
↓
Executor.__init__()
↓
初始化公共 Executor state
↓
self._init_executor()
↓ 动态派发
UniProcExecutor._init_executor()
↓
回到 Executor.__init__()
↓
继续公共 runtime state 初始化
```

这是一种典型的 **Template Method**：

```text
父类
= 固定生命周期骨架

子类
= 只实现 backend-specific variation point
```

为什么不让每个子类自己写 `__init__ + super()`？

因为父类希望统一控制：

```text
公共配置初始化
↓
backend init
↓
公共 runtime state
```

减少不同 backend 初始化顺序失控的风险。

---

# 7. `UniProcExecutor._init_executor()` 做什么

当前核心代码：

```python
self.driver_worker = WorkerWrapperBase(rpc_rank=0)

distributed_init_method, rank, local_rank = self._distributed_args()

kwargs = dict(
    vllm_config=self.vllm_config,
    local_rank=local_rank,
    rank=rank,
    distributed_init_method=distributed_init_method,
    is_driver_worker=True,
    ...
)

self.driver_worker.init_worker(all_kwargs=[kwargs])
self.driver_worker.init_device()
self.driver_worker.load_model()
```

第一遍只抽控制流：

```text
UniProcExecutor._init_executor()
↓
创建 WorkerWrapperBase
↓
准备 rank / local_rank / distributed init 参数
↓
init_worker()
↓
init_device()
↓
load_model()
```

此时还不能把：

```text
driver_worker
```

直接理解成 GPU Worker，因为这里创建的是：

```python
WorkerWrapperBase(...)
```

---

# 8. Executor 到底干什么

经过这一轮追踪，可以更准确地区分：

## Scheduler

负责：

```text
WHAT TO RUN
```

例如：

- 哪些 request；
- 每个 request 本轮算多少 token；
- request 是否 prefill / decode；
- resource budget；
- KV resource 是否可用。

输出：

```text
SchedulerOutput
```

## EngineCore

负责：

```text
Runtime orchestration
```

组织：

```text
schedule
↓
submit execution
↓
管理 in-flight batch
↓
消费 ModelRunnerOutput
↓
Scheduler.update_from_output()
```

## Executor

负责：

```text
HOW / WHERE TO EXECUTE
```

也就是：

- 当前 execution topology 是 UniProc / MP / Ray？
- 有几个 Worker？
- Worker 如何调用？
- RPC / collective RPC 如何处理？
- shutdown 如何处理？
- execution result 怎么返回？

所以：

> **Executor 是 execution topology abstraction。**

不是 GPU forward 本身。

---

# 9. Runtime 中 `UniProcExecutor.execute_model()` 的调用链

`UniProcExecutor.execute_model()`：

```python
output = self.collective_rpc(
    "execute_model",
    args=(scheduler_output,),
    non_block=non_block,
    single_value=True,
)
```

UniProc 的 `collective_rpc()` 最终调用：

```python
run_method(self.driver_worker, method, args, kwargs)
```

所以：

```text
EngineCore
↓
model_executor.execute_model()
↓
UniProcExecutor.execute_model()
↓
collective_rpc("execute_model")
↓
run_method(driver_worker, ...)
```

此时还要继续看 `driver_worker` 是什么。

---

# 10. `step()` 和 `step_with_batch_queue()`

EngineCore 有两个 runtime step path。

## 10.1 普通 `step()`

核心：

```text
Scheduler.schedule()
↓
SchedulerOutput
↓
Executor.execute_model()
↓
future.result()
↓
ModelRunnerOutput
↓
Scheduler.update_from_output()
```

也就是：

```text
Schedule A
↓
Execute A
↓
Wait A
↓
Consume A
↓
Update A
↓
Schedule B
```

---

## 10.2 `step_with_batch_queue()`

这里多了：

```text
batch_queue
```

逻辑是：

```text
Schedule A
↓
Submit A
↓
Future A 入队
↓
如果 queue 有空间
继续 Schedule B
↓
Submit B
↓
之后再消费 A 的结果
```

所以核心设计是：

> **execution submission 与 result consumption 解耦。**

---

# 11. 当前为什么使用 `step_with_batch_queue()`

EngineCore 初始化：

```python
self.batch_queue_size = vllm_config.max_concurrent_batches
```

如果：

```python
self.batch_queue_size > 1
```

则：

```python
self.batch_queue = deque(maxlen=self.batch_queue_size)
```

最终：

```python
self.step_fn = (
    self.step
    if self.batch_queue is None
    else self.step_with_batch_queue
)
```

因此关键是：

```text
max_concurrent_batches
```

---

# 12. `PP` 与 `max_concurrent_batches`

源码：

```python
@property
def max_concurrent_batches(self) -> int:
    pp_size = self.parallel_config.pipeline_parallel_size

    if self.scheduler_config.async_scheduling:
        if self.use_v2_model_runner:
            return pp_size + 1
        if pp_size <= 1:
            return 2

    return pp_size
```

这里一度把 PP 和“多个 batch”混在了一起，需要明确区分。

## 12.1 PP 是什么

PP = Pipeline Parallelism。

它的本体是：

> 把模型不同层切成多个 Pipeline Stage。

例如：

```text
PP=2

GPU0 / Stage0
Layer 0 ~ 15
        │
        ▼
GPU1 / Stage1
Layer 16 ~ 31
```

## 12.2 为什么 PP 需要多个 batch

如果一次只允许一个 batch：

```text
T1:
Stage0 = A
Stage1 = idle

T2:
Stage0 = idle
Stage1 = A
```

会产生 pipeline bubble。

更理想：

```text
        T1   T2   T3
Stage0   A    B    C
Stage1   -    A    B
```

所以：

> PP 是“把模型按层切成多个 stage”。

而：

> 多个 concurrent batch 是“把 pipeline stages 填满”的手段。

不能说：

```text
PP = 支持多个 batch
```

---

# 13. 当前为什么 `max_concurrent_batches = 2`

当前：

```text
PP = 1
V2 Model Runner = True
```

继续追：

```python
async_scheduling: bool | None = None
```

在 `VllmConfig.__post_init__()` 中，如果 Executor 支持 async scheduling，并且没有 incompatible feature：

```python
self.scheduler_config.async_scheduling = True
```

当前 `UniProcExecutor` 支持 async scheduling，所以最终：

```text
async_scheduling:
None
↓
True
```

因此：

```python
max_concurrent_batches
= pp_size + 1
= 1 + 1
= 2
```

最终：

```text
batch_queue_size = 2
↓
batch_queue enabled
↓
step_fn = step_with_batch_queue
```

注意：

> 当前出现两个 concurrent batch **不是因为 PP**。

而是：

```text
PP 基础需求 = 1
+
V2 Async Scheduling 额外 +1
=
2
```

---

# 14. V2 Async Scheduling 到底是什么

这里另一个误区是把它理解成“启动机制”。

不是。

初始化阶段只是：

```text
检查 async 是否支持
↓
async_scheduling=True
↓
配置 batch queue
↓
选择 step_with_batch_queue
```

真正 async scheduling 发生在 runtime：

普通模式：

```text
Schedule A
↓
Execute A
↓
Wait A
↓
Consume A
↓
Schedule B
```

Async scheduling：

```text
Schedule A
↓
Submit A
↓
A execution in-flight
│
├── 可以继续 Schedule B
│
└── Submit B
↓
之后再 Consume A
```

核心不是“CUDA 本身异步”，而是：

> **Scheduler submission 与之前 execution result 的 consumption 不再严格串行。**

---

# 15. `non_block=True` 不等于创建 Thread / Process

看到：

```python
execute_model(..., non_block=True)
```

不能推断：

```text
创建后台 Python Thread
```

也不能推断：

```text
创建新的 Linux Process
```

在 UniProc 路径中仍然：

```python
run_method(self.driver_worker, ...)
```

是否真正异步，要继续看更下层返回的 `AsyncModelRunnerOutput` 等机制。

因此：

```text
non-blocking interface
≠
Thread
≠
Process
```

它表达的是：

> 上层可以用 Future-like abstraction 延后消费结果。

---

# 16. `WorkerWrapperBase` 是什么

继续向下看：

```python
class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine.
    It is responsible for lazily initializing the worker and handling
    the worker's lifecycle.
    """
```

核心字段：

```python
self.worker: WorkerBase
self.vllm_config: VllmConfig
```

真正创建 Worker 的地方：

```python
self.worker = worker_class(**kwargs)
```

所以：

```text
WorkerWrapperBase
≠
Actual Worker
```

而是：

> **Worker lifecycle wrapper + proxy + lazy initializer。**

对象关系：

```text
UniProcExecutor
HAS-A
WorkerWrapperBase
    │
    HAS-A
    Actual Worker
```

---

# 17. 为什么 WorkerWrapper 使用 HAS-A，而不是继承 Worker

这是追踪过程中一个重要疑问。

如果：

```python
class WorkerWrapperBase(WorkerBase):
```

语义会是：

> Wrapper IS-A Worker。

但实际源码是：

```python
self.worker = worker_class(...)
```

说明它真正表达的是：

> Wrapper OWNS / MANAGES a Worker。

Wrapper 的职责：

```text
lazy initialization
lifecycle management
environment setup
method forwarding
proxy
```

Actual Worker 的职责：

```text
device / rank execution
model runner
model loading
GPU execution
```

所以应该用 composition：

```text
Wrapper HAS-A Worker
```

而不是 inheritance。

---

# 18. `WorkerWrapperBase.init_worker()` 的真正流程

核心：

```python
kwargs = all_kwargs[self.rpc_rank]

vllm_config = kwargs.get("vllm_config")
self.vllm_config = vllm_config

parallel_config = vllm_config.parallel_config

worker_class = resolve_obj_by_qualname(
    parallel_config.worker_cls
)

with set_current_vllm_config(self.vllm_config):
    self.worker = worker_class(**kwargs)
```

第一遍可以压缩成：

```text
init_worker()
↓
拿到当前 rank 的 kwargs
↓
保存 vllm_config
↓
读取 parallel_config.worker_cls
↓
字符串 → Python Worker Class
↓
self.worker = worker_class(...)
↓
Actual Worker Object 创建完成
```

---

# 19. `resolve_obj_by_qualname()` 是什么

源码：

```python
def resolve_obj_by_qualname(qualname: str) -> Any:
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
```

例如：

```text
"vllm.v1.worker.gpu_worker.Worker"
```

先：

```text
module_name =
"vllm.v1.worker.gpu_worker"

obj_name =
"Worker"
```

然后：

```python
importlib.import_module(module_name)
```

动态 import module。

再：

```python
getattr(module, "Worker")
```

拿到：

```text
Worker Class
```

所以：

```text
string
↓
dynamic import
↓
Python Class
```

注意仍然不是 Object。

直到：

```python
worker_class(**kwargs)
```

才创建 Object。

---

# 20. `with set_current_vllm_config(...)` 是什么

`with` 是 Python context manager 语法。

核心语义：

```text
进入某个临时上下文
↓
执行代码块
↓
无论正常结束还是异常
↓
退出 / 恢复上下文
```

这里：

```python
with set_current_vllm_config(self.vllm_config):
    self.worker = worker_class(**kwargs)
```

可以理解成：

```text
进入：
当前 vLLM config = self.vllm_config
↓
创建 Worker
↓
Worker 初始化内部可以通过 context 读取当前 config
↓
退出：
恢复之前的 config context
```

目的之一是避免深层调用都显式传递：

```text
foo(config)
→ bar(config)
→ baz(config)
```

---

# 21. WorkerWrapper 的 Proxy 行为

Wrapper 明确定义：

```python
def init_device(self):
    self.worker.init_device()
```

以及：

```python
def execute_model(self, scheduler_output):
    return self.worker.execute_model(scheduler_output)
```

另外：

```python
def __getattr__(self, attr: str):
    return getattr(self.worker, attr)
```

因此例如：

```python
driver_worker.load_model()
```

如果 Wrapper 自己没有 `load_model()`：

```text
WorkerWrapperBase
找不到 load_model
↓
__getattr__("load_model")
↓
getattr(self.worker, "load_model")
↓
ActualWorker.load_model()
```

所以它确实是一个 Proxy。

---

# 22. 当前 Runtime execute 链已经扩展到 Worker

现在：

```text
EngineCore
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
run_method(WorkerWrapperBase, ...)
↓
WorkerWrapperBase.execute_model()
↓
self.worker.execute_model()
↓
Actual Worker
```

这里开始正式进入 Worker 层。

---

# 23. Worker 不是 TP 专属

这是追踪过程中另一个重要误区。

一开始容易把 Worker 理解成：

> TP 场景下，每张 GPU 一个 Worker。

这只说对了一部分。

更准确：

> **Worker 是一个 execution rank 的基本软件实体。**

即使：

```text
TP=1
PP=1
DP=1
```

也必须有：

```text
Worker rank0
```

因为总要有人负责：

```text
device 初始化
rank state
distributed environment
ModelRunner
模型加载
GPU memory
execute_model
```

TP>1 时，只是 Worker 数量和它们之间的 distributed relationship 发生变化。

例如概念上：

```text
TP=1:
Executor
└── Worker rank0

TP=2:
Executor
├── Worker rank0 → GPU0
└── Worker rank1 → GPU1
```

---

# 24. Executor 和 Worker 都“适配”，但适配维度不同

这是整个 architecture 最重要的抽象之一。

## Executor

适配：

```text
execution / deployment topology
```

例如：

```text
UniProc
Multiproc
Ray
External Launcher
```

## Worker

适配：

```text
device / platform execution implementation
```

例如：

```text
CUDA Worker
CPU Worker
XPU Worker
```

因此：

```text
EngineCore
    │
    ▼
Executor
    │
    │ execution topology
    ▼
WorkerWrapper
    │
    │ lifecycle / proxy / lazy init
    ▼
Worker
    │
    │ platform / device execution
    ▼
ModelRunner
```

---

# 25. 为什么 `Worker` 又继承 `WorkerBase`

CUDA Worker：

```python
class Worker(WorkerBase):
```

这里和 WorkerWrapper 的关系完全不同。

`WorkerWrapper HAS-A Worker`，因为 Wrapper 不是 Worker 本体。

而：

```text
CUDA Worker IS-A WorkerBase
```

表示：

> CUDA Worker 是 Worker 抽象的一种具体实现。

结构：

```text
                 WorkerBase
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   CUDA Worker   CPUWorker    XPUWorker
```

这样上层可以统一调用：

```text
init_device()
load_model()
execute_model()
initialize_from_config()
shutdown()
```

而具体 CUDA / CPU / XPU 实现由不同子类处理。

---

# 26. 当前 CUDA Worker 是在哪里选择出来的

`ParallelConfig` 初始：

```python
worker_cls: str = "auto"
```

搜索发现：

```text
CUDA:
vllm/platforms/cuda.py

CPU:
vllm/platforms/cpu.py

XPU:
vllm/platforms/xpu.py
```

CUDA 中：

```python
@classmethod
def check_and_update_config(cls, vllm_config):
    parallel_config = vllm_config.parallel_config

    if parallel_config.worker_cls == "auto":
        parallel_config.worker_cls = (
            "vllm.v1.worker.gpu_worker.Worker"
        )
```

---

# 27. `worker_cls="auto"` 是什么时候被解析的

关键是在：

```text
VllmConfig.__post_init__()
```

中：

```python
current_platform.check_and_update_config(self)
```

因此：

```text
ParallelConfig
worker_cls="auto"
↓
VllmConfig.__post_init__()
↓
current_platform.check_and_update_config(self)
↓
当前平台 = CUDA
↓
CudaPlatform.check_and_update_config()
↓
worker_cls =
"vllm.v1.worker.gpu_worker.Worker"
```

到 WorkerWrapper `init_worker()` 时，`worker_cls` 已经是 resolved config。

---

# 28. `__post_init__()` 的含义

`__post_init__()` 可以理解成：

> 基础字段初始化之后，再做一次“配置收敛 / 校验 / 自动推导”。

概念：

```text
Config Object 创建
↓
__init__()
↓
字段赋值
↓
__post_init__()
↓
cross-config validation
platform-specific update
derived configuration
automatic feature selection
↓
Resolved Config
```

在 vLLM 中已经看到很多例子：

```text
worker_cls = "auto"
↓
根据平台变成 CUDA Worker

distributed_executor_backend = None
↓
根据 world_size 变成 "uni"

async_scheduling = None
↓
根据 compatibility 变成 True / False
```

所以 vLLM Config 不是纯粹的“参数容器”，更准确地说是：

```text
用户输入
+
默认值
+
平台推导
+
跨配置校验
+
runtime policy selection
```

---

# 29. 为什么 Platform 修改 Config，而不是 WorkerWrapper 判断 CUDA

如果 WorkerWrapper 自己写：

```python
if cuda:
    worker = GPUWorker()
elif cpu:
    worker = CPUWorker()
elif xpu:
    worker = XPUWorker()
```

那么 Wrapper 会知道所有平台。

现在 vLLM 的设计是：

```text
Platform
↓
负责回答：
“这个平台应该使用哪个 Worker”
↓
写回 resolved worker_cls

WorkerWrapper
↓
只负责：
“给我一个 class name，我负责创建它”
```

因此：

```text
Platform = policy / platform selection
WorkerWrapper = generic lifecycle mechanism
```

职责更清晰。

---

# 30. GPU Worker 初始化：`WorkerBase.__init__()`

CUDA Worker：

```python
class Worker(WorkerBase):
```

先：

```python
super().__init__(...)
```

WorkerBase 主要保存公共 Worker state：

```python
self.vllm_config
self.model_config
self.cache_config
self.parallel_config
self.scheduler_config
self.device_config
...
```

以及：

```python
self.local_rank
self.rank
self.distributed_init_method
self.is_driver_worker
```

非常关键：

```python
self.device = None
self.model_runner = None
```

说明刚创建 Worker 时：

```text
Worker Object 已存在
但 GPU device 还没真正建立
ModelRunner 也还没创建
```

因此 Worker 生命周期是分阶段的。

---

# 31. Worker 生命周期

当前已经可以明确：

```text
WorkerWrapperBase()
↓
只有 Wrapper

WorkerWrapperBase.init_worker()
↓
创建 Actual Worker
↓
Worker.__init__()
↓
保存配置 / rank / state
↓
device = None
model_runner = None

Worker.init_device()
↓
建立 CUDA / distributed environment
↓
创建 GPUModelRunnerV2

Worker.load_model()
↓
self.model_runner.load_model()
↓
模型权重真正加载
```

这就是 production runtime 中典型的 staged initialization。

---

# 32. `Worker.init_device()` 第一遍骨架

当前 CUDA Worker 的 `init_device()` 很长，但第一遍只提取 control-flow：

```text
Worker.init_device()
│
├─ 检查 device_type == cuda
├─ 根据 DP/TP/PP 修正 local_rank
├─ logical GPU → visible GPU
├─ self.device = cuda:X
├─ set current device
├─ 检查 dtype
├─ 初始化 distributed environment
├─ 设置随机种子
├─ 清缓存
├─ MemorySnapshot
├─ request_memory()
├─ 初始化 workspace manager
└─ 创建 GPUModelRunner
```

---

# 33. Worker 在哪里真正绑定 GPU

核心：

```python
visible_device_index = (
    current_platform.logical_device_id_to_visible_device_id(
        self.local_rank
    )
)

self.device = torch.device(
    f"cuda:{visible_device_index}"
)

torch.accelerator.set_device_index(self.device)
```

所以：

```text
rank / local_rank
↓
logical GPU id
↓
visible GPU id
↓
self.device
↓
当前 Worker 与某张 GPU 建立对应关系
```

当前 TP=1 / PP=1 / DP=1，通常就是 rank0 → cuda:0，但还要考虑 `CUDA_VISIBLE_DEVICES` 映射。

---

# 34. 为什么 distributed init 在 memory snapshot 前

源码明确：

```python
# Initialize the distributed environment BEFORE taking
# memory snapshot
# This ensures NCCL buffers are allocated before we measure
# available memory
```

这是一个非常典型的 production-runtime 细节。

如果：

```text
先测剩余 GPU memory
↓
再初始化 NCCL
↓
NCCL 又占一部分显存
```

那么用于 KV Cache 的可用显存预算会被高估。

所以：

```text
CUDA device ready
↓
distributed / NCCL buffers ready
↓
MemorySnapshot
↓
得到更真实的可用显存
```

这个节点之后进入 KV Cache 初始化时会非常重要。

---

# 35. Worker 在哪里创建 ModelRunner

当前：

```python
if self.use_v2_model_runner:
    from vllm.v1.worker.gpu.model_runner import (
        GPUModelRunner as GPUModelRunnerV2,
    )

    self.model_runner = GPUModelRunnerV2(
        self.vllm_config,
        self.device,
    )
```

因此：

```text
Worker
HAS-A
GPUModelRunnerV2
```

而不是：

```text
Worker IS-A ModelRunner
```

因为：

```text
Worker
= execution rank / device lifecycle / resource container

ModelRunner
= 在这个 Worker/device 上准备输入并执行模型
```

---

# 36. 为什么看到 Worker 创建 ModelRunner，感觉 Scheduler 被“跳过”了

这是一个非常重要的阅读误区：

> 把“初始化链”与“Runtime execution 链”混在一起。

## 初始化链

```text
EngineCore.__init__()
│
├── 创建 Executor
│     └── WorkerWrapper
│           └── Worker
│                 └── init_device()
│                       └── 创建 ModelRunner
│
└── 创建 Scheduler
```

这里只是在搭建对象图。

并没有开始执行 request。

## Runtime 链

真正运行 request 时：

```text
EngineCore.step_fn()
↓
Scheduler.schedule()
↓
SchedulerOutput
↓
Executor.execute_model()
↓
Worker.execute_model()
↓
ModelRunner.execute_model()
↓
GPU
↓
ModelRunnerOutput
↓
Scheduler.update_from_output()
```

因此：

> Scheduler 没有被跳过。

我们之前只是沿 **初始化路径** 先看到了 ModelRunner 被创建。

---

# 37. Worker 与 Scheduler / ModelRunner 的职责关系

当前最准确的整体结构：

```text
EngineCore
├── Scheduler
│      │
│      └── WHAT TO RUN
│
└── Executor
       │
       └── HOW / WHERE TO EXECUTE
              │
              ▼
          WorkerWrapper
              │
              ▼
            Worker
              │
              └── execution rank / device lifecycle
                     │
                     ▼
                 ModelRunner
                     │
                     └── prepare tensors / model forward / GPU execution
```

Scheduler 输出的：

```text
SchedulerOutput
```

沿：

```text
Executor
→ Worker
→ ModelRunner
```

被真正执行。

---

# 38. 当前完整源码链总结

到目前为止，初始化与 Runtime 两条链必须分开记。

## 38.1 初始化链

```text
ParallelConfig
│
├─ distributed_executor_backend=None
├─ worker_cls="auto"
│
▼
VllmConfig.__post_init__()
│
├─ current_platform.check_and_update_config()
│     │
│     └─ CUDA:
│        worker_cls =
│        "vllm.v1.worker.gpu_worker.Worker"
│
├─ distributed_executor_backend
│     ↓
│    "uni"
│
└─ async_scheduling
      ↓
     True
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
└─ Executor.__init__()
      │
      └─ self._init_executor()
            ↓
      UniProcExecutor._init_executor()
            │
            ├─ WorkerWrapperBase()
            │
            ├─ init_worker()
            │    │
            │    ├─ resolve worker_cls
            │    └─ self.worker = CUDA Worker(...)
            │
            ├─ init_device()
            │    │
            │    ├─ bind CUDA device
            │    ├─ distributed init
            │    ├─ memory snapshot
            │    └─ GPUModelRunnerV2(...)
            │
            └─ load_model()
                 │
                 └─ model_runner.load_model()
```

---

## 38.2 Runtime execute 链

```text
EngineCore.run_busy_loop()
│
▼
step_fn
│
│ 当前：
│ step_with_batch_queue()
│
▼
Scheduler.schedule()
│
▼
SchedulerOutput
│
▼
Executor.execute_model()
│
▼
UniProcExecutor.collective_rpc()
│
▼
run_method(driver_worker, "execute_model")
│
▼
WorkerWrapperBase.execute_model()
│
▼
Actual CUDA Worker.execute_model()
│
▼
GPUModelRunnerV2.execute_model()
│
▼
GPU execution
│
▼
Future / AsyncModelRunnerOutput
│
▼
batch_queue
│
▼
ModelRunnerOutput
│
▼
Scheduler.update_from_output()
```

> 注意：最后 `Worker.execute_model() → GPUModelRunnerV2.execute_model()` 这一小段还没有正式逐行验证，下一阶段需要用实际源码补证据。

---

# 39. 当前各层作用总结

| 层 | 当前实际对象/实现 | 核心职责 |
|---|---|---|
| Frontend | LLM / LLMEngine 所在进程 | 用户 API、请求入口 |
| EngineCore Process | `VLLM::EngineCore` | inference runtime OS 容器 |
| EngineCoreProc | Python Object | IPC、线程、signal、lifecycle |
| EngineCore | Parent runtime capability | runtime 主循环与 orchestration |
| Scheduler | `Scheduler` | 决定本轮算哪些 request / token |
| Executor | `UniProcExecutor` | execution topology abstraction |
| WorkerWrapper | `WorkerWrapperBase` | lazy init、proxy、lifecycle |
| Worker | `gpu_worker.Worker` | execution rank、device/distributed environment |
| ModelRunner | `GPUModelRunnerV2` | 模型输入准备与 GPU 模型执行 |

---

# 40. 当前 Process / Object 边界

当前 TP=1 / PP=1 / DP=1：

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

到目前看到的 UniProc 路径中：

```text
WorkerWrapperBase()
Worker(...)
GPUModelRunnerV2(...)
```

都是 Python Object creation。

没有看到新的：

```text
multiprocessing.Process()
proc.start()
Thread(...)
```

因此当前不能把这些对象画成新的 Linux Process。

---

# 41. 这一阶段遇到的主要问题与源码阅读经验

## 41.1 看到 `Proc` / `Process` 名字不能直接认为是 Linux Process

判断标准必须找：

```python
multiprocessing.Process(...)
proc.start()
```

而不是看 class 名。

---

## 41.2 看到 `Class` 先问：什么时候实例化

例如：

```text
executor_class
worker_class
```

都只是 Python Class。

真正 Object creation：

```python
executor_class(vllm_config)
worker_class(**kwargs)
```

---

## 41.3 看到一个 class 只有方法定义，不知道什么时候运行

不要继续盯 class。

应该找 caller：

```bash
rg -n "model_executor\.execute_model"
```

从 caller 还原 runtime data path。

---

## 41.4 初始化链和 Runtime 链必须分开

初始化：

```text
create object
init device
load model
```

Runtime：

```text
schedule
execute
consume
update
```

混在一起看，会误以为 ModelRunner 绕过了 Scheduler。

---

## 41.5 `HAS-A` 与 `IS-A` 必须结合语义判断

```text
EngineCoreProc IS-A EngineCore
CUDA Worker IS-A WorkerBase
```

表示能力扩展 / polymorphism。

而：

```text
Executor HAS-A WorkerWrapper
WorkerWrapper HAS-A Worker
Worker HAS-A ModelRunner
```

表示 ownership / composition。

---

## 41.6 Executor 与 Worker 都是 abstraction，但维度不同

```text
Executor
= topology abstraction

Worker
= platform / execution-rank abstraction
```

这是理解 production vLLM 的关键。

---

## 41.7 Config 不是静态参数表

像：

```text
worker_cls="auto"
async_scheduling=None
distributed_executor_backend=None
```

都会在初始化阶段被 resolve。

所以 Config 更像：

```text
Input Config
↓
Validation / Platform / Policy resolution
↓
Resolved Runtime Config
```

---

# 42. 为什么 vLLM 要这样设计

这条链最重要的不是记住类名，而是理解 production inference runtime 为什么分这么多层。

如果 EngineCore 直接调用 GPUModelRunner：

```text
EngineCore
↓
GPUModelRunner
```

那么 EngineCore 就要同时知道：

```text
CUDA / CPU / XPU
UniProc / MP / Ray
TP / PP / DP
rank / local_rank
Process topology
RPC
GPU lifecycle
ModelRunner lifecycle
```

这会导致严重耦合。

vLLM 把不同变化维度拆开：

```text
EngineCore
= orchestration

Scheduler
= scheduling policy

Executor
= deployment / execution topology

WorkerWrapper
= lifecycle / proxy / lazy init

Worker
= rank + platform execution

ModelRunner
= model execution
```

因此可以独立演化：

```text
换 Executor backend
不必改 Scheduler

换 CUDA / CPU Worker
不必改 Executor API

换 ModelRunner V1 / V2
不必改 EngineCore runtime contract
```

这就是 production runtime 中很典型的：

> **把不同变化维度正交拆分。**

---

# 43. 当前阶段结论

目前已经可以形成一个稳定认识：

> **EngineCore 是 inference runtime 的驱动者，Scheduler 在 EngineCore 中负责产生 `SchedulerOutput`，Executor 负责把这个执行计划映射到具体 execution topology。当前 TP=1 / PP=1 / DP=1 选择 `UniProcExecutor`。UniProcExecutor 持有 `WorkerWrapperBase`，Wrapper 负责延迟创建和代理 Actual Worker；CUDA 平台在 `VllmConfig.__post_init__()` 阶段把 `worker_cls="auto"` resolve 为 `vllm.v1.worker.gpu_worker.Worker`。CUDA Worker 继承 `WorkerBase`，负责当前 rank 的 device / distributed / memory / model-runner 生命周期，并在 `init_device()` 中创建 `GPUModelRunnerV2`。真正 Runtime 时，Scheduler 的决策仍然通过 `Executor → Worker → ModelRunner` 被执行，没有绕过 Scheduler。**

---

# 44. 下一阶段从哪里继续

下一阶段不再继续初始化链，而重新回到：

```text
Runtime execute chain
```

唯一主问题：

> **`gpu_worker.Worker.execute_model()` 到底怎样把 `SchedulerOutput` 交给 `GPUModelRunnerV2`，中间又做了哪些 Worker-level 处理？**

第一步应该只看：

```text
gpu_worker.Worker.execute_model()
```

第一遍只关心：

```text
1. 输入是不是 SchedulerOutput？
2. Worker 在交给 ModelRunner 前做了什么？
3. 最终调用哪个 model_runner method？
4. 返回 ModelRunnerOutput 还是 AsyncModelRunnerOutput？
5. 有没有 PP / distributed communication branch？
```

然后才进入：

```text
GPUModelRunnerV2.execute_model()
```

此时新的真实问题应该是：

> **SchedulerOutput 如何被转换成 GPU ModelRunner 真正需要的 tensor / metadata，并最终驱动一次 model forward？**

这将正式从：

```text
Runtime control plane
```

进入：

```text
model execution data path
```

但仍然暂时不展开 KVCacheManager / BlockPool / Attention 内部，先把 `SchedulerOutput → Worker → ModelRunner` 这一条 execution bridge 跑通。
