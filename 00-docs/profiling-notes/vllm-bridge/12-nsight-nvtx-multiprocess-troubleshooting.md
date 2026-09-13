# 12-vLLM：Nsight / NVTX / 多进程 Profiling 工程排错手册

> 这份文档不讨论“模型哪里慢”，而专门记录 E6.5/E6.6 为了**让 profiler 真正采到可信数据**经历的工程问题。  
> 核心原则：**profiling 工具链本身也需要验证；程序跑通、report 生成、GUI 能打开，都不等于 tracing 有效。**

---

# 1. 为什么这部分应该和性能报告分开

性能报告关心：

```text
Scheduler 多久？
MRV2 多久？
GPU active 多久？
有没有 overlap？
CUDA Graph 提升多少？
```

但在此之前，我们实际遇到：

```text
Python NVTX 自己 segfault
report 生成但没有 CUDA/NVTX payload
旧 Nsight 跟不进 vLLM 子进程
2024 版本默认 multiprocessing 下仍可能缺数据
shared GPU 产生假 overlap
CPU sampling 能力被系统禁用
```

如果把这些全部塞进主性能报告，会打断主线；但如果不记录，下次会再次踩坑。

---

# 2. Profiler 成功必须分三层判断

错误判断：

```text
程序 return_code=0
+
生成 xxx.nsys-rep
=
profiling 成功
```

不够。

应该检查：

## Level 1：目标程序是否正常结束

```text
return code
segfault
exception
```

## Level 2：report 是否生成

```text
xxx.nsys-rep
```

## Level 3：核心 tracing payload 是否真的存在

导出 SQLite 后至少检查：

```text
NVTX_EVENTS
CUPTI_ACTIVITY_KIND_RUNTIME
CUPTI_ACTIVITY_KIND_KERNEL
```

本轮曾真实遇到：

```text
.nsys-rep 存在
SQLite 也存在

但只有：
ANALYSIS_DETAILS
PROCESSES
TARGET_INFO_*
StringIds
ENUM_*

没有：
NVTX_EVENTS
RUNTIME
KERNEL
```

这叫：

```text
report shell 有了
核心 payload 没进去
```

所以正式脚本后面都增加 table row-count validation。

---

# 3. 第一个坑：Python `nvtx.annotate(..., domain=...)` 在完整 vLLM 下 segfault

最早 instrumentation 使用 Python `nvtx` package 的 custom domain。

完整 vLLM + Nsight capture 初始化阶段出现 native crash，stack 进入：

```text
nvtxExtInitOnce_v3
nvtxExtPayloadInitOnce_v3
nvtxDomainIsEnabled
src/nvtx/_lib/lib.c
DomainHandle.__new__
```

重要现象：

```text
不是 Python exception
而是 native segfault
```

而且 crash 发生在正式用户 request 之前的 engine init / warmup 周期。

---

## 3.1 为什么“还没正式请求”也会进入我们标的 ModelRunner range

我们最开始想当然认为：

```text
GPUWorker.execute_model()
只在真实 generate 时运行
```

但 production inference runtime 初始化可能执行：

```text
profile run
warmup
dummy run
memory profiling
backend init
```

这些路径也可能进入 ModelRunner/Worker execution path。

因此 instrumentation 的生命周期必须考虑：

```text
初始化
warmup
正式推理
shutdown
```

而不是只考虑用户 request。

这也是后面 phase instrumentation 增加：

```text
not dummy_run
```

条件的原因。

---

# 4. 为什么不能简单得出“NVTX API 不兼容”

做了最小 probe：

```text
Python nvtx 0.2.15 annotate
```

在小程序 + 旧 Nsight 下可以运行。

`torch.cuda.nvtx.range` 也可以运行。

所以证据更接近：

> **不是 NVTX 功能整体不可用，而是完整 vLLM 多进程/初始化上下文中触发 Python nvtx DomainHandle 路径时出现问题。**

注意这是边界性描述，不应过度声称已经定位到 Python nvtx 或 Nsight 的某个确定 bug。

---

# 5. 修复策略：改用 `torch.cuda.nvtx.range`

最终将我们的 marker 统一改成：

```python
with torch.cuda.nvtx.range("BRIDGE::MRV2_EXECUTE"):
    ...
```

Core：

```text
BRIDGE::SCHEDULE::<id>
BRIDGE::EXEC_SUBMIT::<id>
BRIDGE::SAMPLE_SUBMIT::<id>
```

Worker：

```text
BRIDGE::MRV2_EXECUTE
BRIDGE::MRV2_SAMPLE
```

同时：

```text
不使用 custom NVTX domain
```

这样避免了之前 Python `nvtx` DomainHandle 路径。

---

# 6. 第二个坑：旧 Nsight 2023 能生成 report，但完整 vLLM 里没有核心 trace

旧版：

```text
/usr/local/cuda-12.1/bin/nsys
2023.1.2
```

某次 P2B：

```text
return_code=0
report exists
```

但导出/统计后：

```text
NVTX_EVENTS                        missing
CUPTI_ACTIVITY_KIND_RUNTIME        missing
CUPTI_ACTIVITY_KIND_KERNEL         missing
```

因此：

> **GUI/report 文件存在不代表 profiler 真正跟到了执行 CUDA 的进程。**

---

# 7. 为什么多进程 vLLM 对 profiler 特别敏感

先不要把问题理解成：

```text
“Nsight 为什么抓不到 vLLM？”
```

更准确的问题是：

> **真正执行 CUDA 的进程是谁？这个进程是如何被创建出来的？Profiler 是否成功覆盖到了这个 child process？**

当前离线 vLLM 路径不是：

```text
python benchmark.py
↓
所有 vLLM 代码
↓
GPU
```

而更接近：

```text
Frontend / Benchmark Process
PID A
│
├─ LLM
├─ LLMEngine
├─ SyncMPClient
│
│ IPC
│
└───────────────────────────────┐
                                ▼
                        EngineCore Process
                        PID B
                        │
                        ├─ EngineCore
                        ├─ Scheduler
                        ├─ UniProcExecutor
                        ├─ Worker
                        ├─ GPUModelRunner
                        │
                        ├─ CUDA Runtime
                        └─ GPU
```

因此我们在 Nsight 中真正想观察的：

```text
BRIDGE::MRV2_EXECUTE
cudaLaunchKernel
FlashAttention kernel
GEMM kernel
KV write kernel
```

主要发生在：

```text
EngineCore child process
```

而不是最外层 benchmark Python process。

这意味着 profiler 不能只做到：

```text
跟踪 parent
```

还必须做到：

```text
parent
↓
创建 child
↓
profiler 正确覆盖 child
↓
child 初始化 CUDA / NVTX / CUPTI
↓
child 提交 kernel
↓
Nsight 收到完整 activity
```

因此：

```text
process creation method
```

本身就是 profiling 环境的一部分。

---

# 8. `fork` 和 `spawn` 到底是什么

Python multiprocessing 在 Linux 上可以用不同方式创建 child process。

本实验最重要的是：

```text
fork
spawn
```

这两个方式的本质完全不同。

---

## 8.1 `fork`：复制父进程当前状态

可以先把一个运行中的 Python 进程想象成：

```text
Parent Process
│
├─ Python Interpreter
├─ imported Python modules
├─ PyTorch
├─ vLLM
├─ loaded native .so
├─ CUDA Runtime state
├─ CUDA Driver state
├─ NCCL / NVML state
├─ NVTX state
├─ CUPTI / profiler state
├─ background threads
├─ mutex / lock
└─ global variables
```

执行：

```text
fork()
```

以后，从逻辑上可以近似理解为：

```text
                  fork()
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
Parent Process             Child Process
-------------              -------------
Python state               inherited Python state
loaded .so                 inherited loaded .so state
global variables           inherited variables
native runtime state       inherited runtime snapshot
```

Linux 实际上通常通过：

```text
copy-on-write
```

避免立即复制全部物理内存，因此 fork 很快。

但是关键不在于“内存有没有立刻复制”，而在于：

> **child 继承的是父进程 fork 那一刻已经存在的进程状态。**

对于普通 Python 数据：

```python
x = [1, 2, 3]
```

这通常没有问题。

但对于：

```text
CUDA
CUPTI
NVTX
pthread
mutex
background thread
driver handle
profiler callback
```

就没这么简单。

---

## 8.2 `spawn`：启动一个新的 Python 进程

`spawn` 不是复制当前已经运行了一半的 Python runtime。

更接近：

```text
Parent Process
│
│ 我要创建一个 worker
│
└──────── spawn ──────────────┐
                              ▼
                      New Python Process
                      │
                      ├─ 启动新的 Python interpreter
                      ├─ 重新 import module
                      ├─ 根据参数恢复目标对象
                      ├─ 初始化 PyTorch
                      ├─ 初始化 native libraries
                      ├─ 初始化 CUDA
                      └─ 执行 child workload
```

所以 spawn 的核心特征是：

> **child 从一个新的进程运行时开始初始化，而不是继承 parent 已经初始化了一半的 native runtime 快照。**

代价是：

```text
启动更慢
需要重新 import
某些 Python 对象需要 pickle / serialization
```

但它的状态边界更清晰。

---

# 9. 为什么 `fork` 对 CUDA / profiler 特别危险

这里不能简单说：

```text
fork 错
spawn 对
```

更准确是：

```text
CPU-only 简单 multiprocessing
→ fork 通常很好

CUDA / accelerator / profiler runtime
→ fork 后继承复杂 native state，风险显著更高
```

---

## 9.1 先看最简单的“线程 + 锁”例子

假设 parent 有两个线程：

```text
Parent

Thread A
Thread B
```

Thread B 当前持有：

```text
mutex X = locked
```

此时 Thread A 调用：

```text
fork()
```

fork 后 child 通常只有调用 fork 的那个线程继续存在。

可能形成：

```text
Parent

Thread A
Thread B ───── owns mutex X


fork()
        ↓

Child

Thread A

mutex X = locked
但原来拥有 mutex X 的 Thread B 不存在了
```

child 如果后续执行：

```text
lock(mutex X)
```

可能永远等不到这个锁被释放。

这只是最简单的例子。

真实的 CUDA / CUPTI / profiler runtime 内部状态要复杂得多。

---

## 9.2 CUDA 并不是普通 Python 对象

CUDA 在 host process 中维护大量状态，例如：

```text
CUDA Runtime
CUDA Driver
CUDA Context
Streams
Events
Allocator
Library handles
Internal synchronization state
Background/runtime state
```

所以：

```text
父进程初始化 CUDA
↓
fork
↓
child 继承 parent 的 host-side CUDA state snapshot
```

并不等价于：

```text
child 自己重新建立了一套干净的 CUDA runtime
```

这也是为什么 vLLM v0.26 自己会检查：

```text
cuda_is_initialized()
```

如果 CUDA 已经初始化，就将：

```text
VLLM_WORKER_MULTIPROC_METHOD
```

强制改为：

```text
spawn
```

也就是说：

> **vLLM 本身就明确把“CUDA 已经初始化后再 fork”视为需要避免的情况。**

---

# 10. 为什么 Nsight / CUPTI 也会受到 fork 影响

Nsight Systems 并不是站在进程外面“只看 GPU”。

为了采集：

```text
CUDA Runtime API
GPU kernel
NVTX
```

它需要和目标进程中的 profiling infrastructure 建立关系。

粗略可以理解成：

```text
Nsight Systems
      │
      │ instrumentation / tracing
      ▼
Target Process
│
├─ NVTX
├─ CUDA Runtime
├─ CUDA Driver
├─ CUPTI
│
└─ GPU
```

其中：

```text
CUPTI_ACTIVITY_KIND_RUNTIME
CUPTI_ACTIVITY_KIND_KERNEL
```

就是 CUDA profiling activity 的关键来源。

---

## 10.1 如果 parent 已经有 profiler/native state，然后再 fork

假设：

```text
nsys profile python workload.py
```

先启动 parent：

```text
Parent
│
├─ Python
├─ PyTorch
├─ loaded native libraries
├─ NVTX state
├─ CUPTI/profiler state
└─ CUDA-related state
```

然后 vLLM 创建 EngineCore：

```text
fork()
```

概念上可能出现：

```text
                      fork
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          Parent               Child
          ------               -----
CUPTI/profiler state        inherited snapshot
NVTX state                  inherited snapshot
native locks                inherited snapshot
runtime buffers             inherited snapshot
background threads          not equivalently recreated
```

这里最危险的地方是：

> child 看起来继承了 profiler/native runtime 的一部分状态，但这不等于 child 对这些 runtime 做了一次完整、干净、受支持的重新初始化。

这可能导致一种非常迷惑的情况：

```text
程序能跑                ✓
GPU 也真的在计算        ✓
.nsys-rep 能生成         ✓
SQLite 能 export         ✓

但：

NVTX_EVENTS              ✗
CUDA Runtime activity    ✗
GPU Kernel activity      ✗
```

这正是我们 P2D 的症状。

---

# 19. 为什么“report 能生成”仍然不能说明 tracing 成功

这一点需要单独强调。

`.nsys-rep` 只是一次 profiling session 的 report container。

生成：

```text
xxx.nsys-rep
```

只能说明：

```text
Nsight session 启动了
↓
session 最终结束
↓
report 文件被写出
```

但不能证明：

```text
真正执行 GPUModelRunner 的 EngineCore child
↓
NVTX 被采到
↓
CUDA Runtime API 被采到
↓
GPU kernels 被采到
```

因此 P2D：

```text
report exists
sqlite exists
```

看起来像：

```text
profiling succeeded
```

但实际检查 SQLite：

```text
NVTX_EVENTS
CUPTI_ACTIVITY_KIND_RUNTIME
CUPTI_ACTIVITY_KIND_KERNEL
```

发现核心表缺失或没有有效 activity。

于是我们才得到：

```text
report shell exists
≠
CUDA/NVTX tracing succeeded
```

这也是后续所有正式脚本都要验证 payload row count 的原因。

---

# 20. P2D → P2E：为什么改成 `spawn` 后 tracing 成功

P2D：

```text
Nsight Systems 2024.7.1
+
full vLLM
+
默认 multiprocessing startup
```

程序和 report 都能生成，但关键 tracing payload 不完整。

随后固定：

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

并实际验证：

```text
get_mp_context().get_start_method() = spawn
```

这时 child process 的生命周期更接近：

```text
Parent
│
└──── spawn ────────────────────────────┐
                                        ▼
                              Fresh EngineCore Process
                              │
                              ├─ start Python interpreter
                              ├─ import torch
                              ├─ import vLLM
                              ├─ load native .so
                              ├─ initialize runtime state
                              ├─ initialize CUDA
                              ├─ initialize NVTX/CUPTI path
                              ├─ create GPUModelRunner
                              └─ submit CUDA work
```

也就是说：

```text
CUDA / NVTX / profiler-related runtime
```

在 child 自己的生命周期里完成初始化，而不是继承 parent 的复杂 native state 快照。

P2E 最终成功得到：

```text
NVTX_EVENTS                     40
CUPTI_ACTIVITY_KIND_RUNTIME     7148
CUPTI_ACTIVITY_KIND_KERNEL      2973
```

这三个结果分别说明：

```text
NVTX_EVENTS
↓
业务语义 range 成功采集

CUPTI_ACTIVITY_KIND_RUNTIME
↓
CPU 侧 CUDA Runtime 调用成功采集

CUPTI_ACTIVITY_KIND_KERNEL
↓
GPU kernel activity 成功采集
```

因此完整 tracing chain 才真正成立：

```text
Business semantics
      NVTX
       ↓
CPU CUDA API
   CUPTI Runtime
       ↓
 correlationId
       ↓
GPU execution
   CUPTI Kernel
```

---

# 21. 这组实验到底证明了什么，没证明什么

这里必须保持证据边界。

我们能够证明：

```text
P2D
默认 process startup
→ full vLLM tracing payload 缺失

P2E
固定 spawn
→ NVTX / CUDA Runtime / Kernel activity 全部出现
```

因此：

> **multiprocessing start method 是本实验中影响 full-vLLM Nsight tracing 是否成功的关键变量。**

但我们不能进一步断言：

```text
一定是 CUPTI 某个 mutex
一定是 NVTX 某个 global state
一定是 CUDA driver handle
一定是 Nsight 某个内部 callback
```

因为我们没有 NVIDIA profiler 内部源码或更深 native-level 证据。

所以最严谨的表述是：

> `fork` 会继承父进程当前的 native runtime 状态，而 CUDA、CUPTI、NVTX、线程、锁以及 profiler instrumentation 都包含复杂的进程局部状态。在本实验的 full-vLLM multiprocessing 场景中，默认 startup 未能形成有效 child-process CUDA/NVTX tracing；切换为 `spawn` 后，child 从新的 Python/runtime 状态初始化，Nsight 成功获取完整 NVTX、CUDA Runtime 与 GPU Kernel activity。实验能够确认 process start method 与 tracing failure 存在因果关联，但不能据此进一步断言具体是哪一个 Nsight/CUPTI 内部状态在 fork 后失效。

---

# 22. 为什么 vLLM 默认仍可能使用 `fork`

`fork` 本身不是错误设计。

它有明显优势：

```text
创建进程快
Linux 上成熟
不需要从头启动 Python interpreter
很多普通 multiprocessing workload 可正常使用
```

所以 vLLM 的 multiprocessing helper 默认可以选择：

```text
fork
```

但是 vLLM 同时实现 `_maybe_force_spawn()`。

当检测到：

```text
CUDA already initialized
XPU already initialized
Ray actor
NUMA binding
WSL 等
```

会切换：

```text
spawn
```

这说明 vLLM 的设计不是：

```text
fork 永远安全
```

而是：

```text
默认优先低启动成本
+
遇到已知高风险 runtime 条件时强制 spawn
```

---

# 23. 为什么 profiling 实验要显式固定 `spawn`

即使 vLLM 有自动判断逻辑，正式 profiling 仍然应该显式固定：

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

原因不是为了更快，而是为了：

```text
实验可重复性
```

如果不固定，可能出现：

```text
Run A:
创建 child 之前 CUDA 已初始化
→ vLLM 自动 spawn

Run B:
创建 child 之前 CUDA 尚未初始化
→ 仍然 fork
```

然后两次 Nsight trace 的 process creation path 不一样。

这样：

```text
Run A vs Run B
```

就多了一个隐含变量。

正式 benchmark/profiling 的原则应该是：

```text
process start method
=
controlled variable
```

所以我们明确固定：

```text
Nsight Systems 2024.7.1
+
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

---

# 24. `spawn` 在这里不是性能优化项

这点必须和 CUDA Graph 之类的性能优化区分。

我们使用 spawn 的目的：

```text
让 profiler 稳定覆盖真正执行 CUDA 的 child process
```

不是：

```text
让 decode 更快
```

所以：

```text
spawn
```

属于：

```text
profiling / multiprocessing engineering configuration
```

而不是：

```text
model execution optimization
```

不要写成：

```text
“spawn 提升了 vLLM 性能”
```

本实验没有证明这一点。

更准确是：

> **spawn 提高了本实验中 child-process CUDA/NVTX tracing 的可观察性与稳定性。**

---

# 25. 一张图记住 fork / spawn 与 profiler 的区别

## Fork 路径

```text
Nsight
  │
  ▼
Parent Python
  │
  ├─ PyTorch
  ├─ native .so
  ├─ profiler/CUPTI/NVTX state
  ├─ CUDA-related state
  │
  └──────── fork ──────────────────┐
                                   ▼
                           EngineCore Child
                           │
                           ├─ inherited native state
                           ├─ GPUModelRunner
                           └─ CUDA
                               ↓
                              GPU

风险：
复杂 native/profiler/runtime state
是“继承快照”

而不是
child 自己从头干净初始化
```

## Spawn 路径

```text
Nsight
  │
  ▼
Parent Python
  │
  └──────── spawn ─────────────────┐
                                   ▼
                           Fresh Python Process
                           │
                           ├─ import torch
                           ├─ import vLLM
                           ├─ load native .so
                           ├─ init runtime
                           ├─ init CUDA
                           ├─ init profiler path
                           ├─ EngineCore
                           ├─ GPUModelRunner
                           └─ GPU
```

最重要的认知：

```text
fork
=
继承 parent 当前 runtime snapshot

spawn
=
child 建立新的 runtime
```

对于普通 Python multiprocessing，两者都可能合理。

对于：

```text
CUDA
+
复杂 native libraries
+
Nsight/CUPTI profiling
```

spawn 通常提供更清晰、更可控的 process initialization boundary。

---

# 26. 本实验最终对 multiprocessing 的结论

最终固定结论不要写成：

```text
多进程必须 spawn
```

而应该写成：

> **vLLM 是多进程 runtime，真正执行 GPUModelRunner/CUDA 的 EngineCore 位于 child process。Nsight 要获得有效 tracing，必须正确覆盖这个 child。`fork` 通过继承父进程状态创建 child，而 `spawn` 创建新的 Python 进程并重新初始化 runtime。CUDA、CUPTI、NVTX 和 profiler instrumentation 都包含复杂的进程局部 native 状态，因此 process creation method 会直接影响 profiler 能否正确观测 child-process GPU activity。我们的 P2D/P2E controlled comparison 表明，在当前 full-vLLM + Nsight 2024 环境中，默认 startup 产生 report 但缺少关键 payload；固定 spawn 后 NVTX、CUDA Runtime 和 Kernel activity 均成功采集。因此后续 profiling 将 `VLLM_WORKER_MULTIPROC_METHOD=spawn` 固定为实验工程条件。**

这也是为什么：

```text
program runs successfully
```

和：

```text
profiler observes the correct process successfully
```

是两个完全不同的问题。

---

# 19. `.nsys-rep`、SQLite、Stats 的正确角色

## `.nsys-rep`

主要给 GUI：

```text
Timeline
Events
Correlation arrows
Processes/Threads
CUDA HW streams
```

## SQLite

主要给精确、可复现的定量分析：

```sql
SELECT start,end,text,globalTid
FROM NVTX_EVENTS;
```

以及：

```text
按 correlationId 找 kernel
计算 interval overlap
算 median / p95
统计每 step kernel count
```

## `nsys stats`

用于快速 summary：

```text
NVTX summary
CUDA API summary
kernel summary
MemOps summary
```

但是默认全局，不应自动代表 steady decode。

---

# 20. 采集命令为什么第一轮只用 `cuda,nvtx`

第一轮正式低开销 capture：

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  ...
```

原因：

```text
先回答 CPU submission / GPU execution 基本关系
```

不一开始开启：

```text
CPU sampling
context switch
GPU metrics
```

因为采集越重：

```text
profiler overhead ↑
数据量 ↑
系统行为扰动 ↑
分析复杂度 ↑
```

正确策略是 progressive profiling。

---

# 21. CPU sampling 为什么后来失败，失败后还能相信什么

E6.6-D 开：

```text
CPU sampling / context switch / CUDA backtrace
```

Nsight 明确 warning：

```text
CPU IP/backtrace sampling not supported, disabling.
CPU context switch tracing not supported, disabling.
CUDA backtraces will not be collected because CPU sampling is disabled.
```

所以不能继续声称：

```text
“CPU sampling 已定位 Python hotspot”
```

但是同一 report 中：

```text
NVTX
CUDA Runtime
GPU Kernel
```

仍然有效。

原则：

> **Profiler 报告不是“全有或全无”；每种 tracing capability 都有自己的证据边界。**

---

# 22. Shared GPU：为什么性能实验前必须做 idle guard

早期 GPU0/GPU2 有外部 process，util 曾达到：

```text
几十% 到 70%+
```

这会导致：

```text
我们的 kernel launch 后排队
↓
GPU start time 被推迟
↓
kernel 跨到下一 CPU step
↓
看起来像 async overlap
```

这种 overlap 可能只是资源竞争造成。

所以后面正式脚本使用：

```text
连续 8 次采样 selected physical GPU utilization
max <= 10%
```

否则自动 abort。

真正 clean GPU1 后：

```text
8/8 samples = 0%
```

才用于正式 cross-step overlap 和 CUDA Graph A/B。

---

# 23. 为什么不能“为了跑通”把 idle threshold 调低

如果 GPU 已经 40~70% 外部负载，把 threshold 改成 80% 只是让脚本继续运行，不会让数据变可信。

性能 profiling 的原则：

```text
环境不满足实验前提
→ 终止实验
```

而不是：

```text
为了生成一个数字
→ 修改验收门槛
```

这也是本轮最重要的实验纪律之一。

---

# 24. NVTX marker 应该怎么放：语义层级优先于函数数量

错误做法：

```text
每个 Python 函数都加 range
每一层都加几十个 marker
```

问题：

```text
可读性下降
instrumentation overhead 增大
timeline 变成彩条海洋
```

推荐层次：

```text
Level 1
Scheduler / Execute / Sample

Level 2
ModelRunner coarse phases

Level 3
只有 Level 2 最大块仍不清楚时
才拆一层 model / operator
```

本轮就是：

```text
MRV2 ≈22ms
↓
6 个 coarse phase
↓
MODEL_FORWARD ≈21ms
↓
直接做 CUDA Graph intervention
```

因为 intervention 已经验证根因，所以没有继续无意义拆 28 层。

---

# 25. Phase instrumentation 为什么独立一个环境变量

最终：

```text
QCACHE_BRIDGE_TRACE=0
QCACHE_BRIDGE_NVTX=1
QCACHE_BRIDGE_PHASE_NVTX=1
```

三者职责：

```text
TRACE
= debug print / semantic trace

NVTX
= 顶层 Core / Worker business ranges

PHASE_NVTX
= 更细 ModelRunner phase ranges
```

这样可以按实验需要控制扰动。

---

# 26. “报告有数据”之后还要做 schema 检查

不要假设不同 Nsight 版本导出的 SQLite schema 完全一致。

先：

```sql
.schema NVTX_EVENTS
.schema CUPTI_ACTIVITY_KIND_RUNTIME
.schema CUPTI_ACTIVITY_KIND_KERNEL
```

再写分析代码。

例如 NVTX 文本有的报告直接在：

```text
NVTX_EVENTS.text
```

有的分析模式可能需要：

```text
textId → StringIds
```

所以脚本最好先 inspect schema，再写硬编码 SQL。

---

# 27. Correlation ID 的正确使用

目标：

```text
业务 NVTX
→ CPU CUDA API
→ GPU event
```

基本链：

```text
找到 MRV2_EXECUTE start/end
↓
筛选同 process 的 CUDA Runtime API
↓
拿 correlationId
↓
匹配 GPU kernel correlationId
```

不能只根据：

```text
kernel 名字接近
时间看起来接近
```

就认定属于同一次 host call。

---

# 28. 常见 profiler 误读清单

## 误读 1

```text
有 .nsys-rep
→ capture 成功
```

错。必须验证 payload table。

## 误读 2

```text
NVTX raw range = GPU active
```

错。NVTX 首先是 CPU host semantic range。

## 误读 3

```text
GPU projection = GPU active sum
```

错。projection/envelope 可以包含 event 间 gap。

## 误读 4

```text
cudaLaunchKernel duration = kernel duration
```

错。前者 CPU API，后者 GPU execution。

## 误读 5

```text
全局 memcpy 很大
→ decode PCIe bottleneck
```

错。可能是模型加载。

## 误读 6

```text
看到 overlap
→ async pipeline 成功
```

错。共享 GPU contention 可以制造假 overlap。

## 误读 7

```text
CPU sampling warning
→ 整个 report 都废了
```

错。应按 tracing capability 分层判断。

---

# 29. 推荐的 vLLM Nsight 采集 SOP

```text
1. 固定 repo / model / Python / torch / GPU / nsys 版本

2. 确认 multiprocessing method
   profiler 环境固定 spawn

3. 先跑不带 profiler 的 smoke

4. 加稀疏 NVTX

5. 第一轮只采 cuda,nvtx
   sample=none
   cpuctxsw=none

6. capture 后不要立刻分析
   先 export sqlite

7. 验证：
   NVTX_EVENTS > 0
   RUNTIME > 0
   KERNEL > 0

8. 检查 selected GPU 采集前后是否 idle

9. 再做业务 window / correlation / overlap

10. 只有 host gap 仍无法解释时
    再尝试 CPU sampling/osrt/context switch

11. 只有最后锁到 GPU utilization/HBM 时
    再加 GPU metrics / NCU
```

---

# 30. 本轮最终固定的 profiler 配置原则

```text
Nsight Systems:
2024.7.1

Multiprocessing:
spawn

Business trace:
torch.cuda.nvtx.range

Text logging:
formal profile 关闭

First-pass tracing:
cuda,nvtx

CPU sampling:
当前机器不作为正式依赖

GPU timing:
必须 idle guard

Analysis:
GUI 用于定位
SQLite 用于精确统计
Stats 用于交叉验证
```

---

# 31. 一页恢复版

```text
Python nvtx custom Domain 在完整 vLLM + profiler init 下 segfault
↓
换 torch.cuda.nvtx.range
↓
旧 nsys 2023 能生成 report，但可能没有 CUDA/NVTX payload
↓
升级 nsys 2024
↓
小 mp probe 成功，但完整 vLLM 默认 mp 仍可能空壳
↓
固定 VLLM_WORKER_MULTIPROC_METHOD=spawn
↓
SQLite 出现：
NVTX_EVENTS
RUNTIME
KERNEL
↓
正式 tracing chain 建立
↓
发现 shared GPU 会制造 queue delay / 假 overlap
↓
加入 8-sample idle guard
↓
CPU sampling 在当前主机 unsupported
↓
只使用它真正采到的 NVTX/CUDA/kernel 证据
```

这套工程排错本身也是 profiling 能力的一部分：**先证明 profiler 在观察正确的 process 和正确的数据，再讨论性能结论。**
