# Nsight Systems：CLI、SQLite、GUI 三种分析方式的选择与分工手册

> 适用场景：vLLM / SGLang / LMCache / KV Cache / Triton / LLM Inference Profiling  
> 核心目标：明确 **什么时候用命令行、什么时候用 SQLite、什么时候必须打开 Nsight GUI**，避免每次拿到 `.nsys-rep` 后不知道从哪里开始。

---

# 0. 先给结论

Nsight Systems 的三种分析方式不是替代关系：

```text
CLI
=
快速体检 / 快速筛查 / 自动化入口

SQLite
=
精确定量 / NVTX scoped analysis / 批量实验 / 统计证据

GUI
=
时间线理解 / 因果关系 / overlap / queue / stream / 多进程拓扑
```

最推荐的工作流不是：

```text
CLI → SQLite → GUI
```

机械地全部走一遍。

而是：

```text
采集 .nsys-rep
      ↓
先问：我现在想解决什么问题？
      ↓
┌─────────────────────────────────────┐
│ “谁最多？有没有明显异常？”          │
│ → CLI                               │
│                                     │
│ “准确多少？100个step是不是都这样？”│
│ → SQLite                            │
│                                     │
│ “为什么会这样？谁在等谁？”          │
│ → GUI                               │
└─────────────────────────────────────┘
```

一句话记：

> **CLI 找异常，GUI 找机制，SQLite 做证据。**

如果最终问题已经收敛到：

```text
“这个 kernel 本身为什么慢？”
```

再进入：

```text
NCU
```

---

# 1. 三种方式本质上分别在回答什么问题

---

## 1.1 CLI：回答“总体上发生了什么”

CLI 这里主要指：

```text
nsys stats
nsys export
以及围绕它写的小型 shell/python 脚本
```

它最擅长回答：

```text
这份 report 有没有 CUDA 数据？
哪个 CUDA API 调用最多？
哪个 kernel 总时间最多？
Memcpy 多不多？
有没有 cudaGraphLaunch？
NVTX 哪些 range 总时间比较大？
```

典型问题：

```text
cudaLaunchKernel 一共有多少次？
Top 10 kernels 是谁？
H2D / D2H 是否很多？
GraphLaunch 有没有出现？
```

这些问题不需要理解完整时间线。

所以：

> CLI 是 **快速筛查工具**。

---

## 1.2 SQLite：回答“在我指定的业务范围里，精确是多少”

SQLite 最适合回答：

```text
只看 steady decode Call50
MRV2_EXECUTE median 到底是多少？

只看 MODEL_FORWARD
kernel count 是多少？
kernel active union 是多少？
CUDA Runtime API sum 是多少？

launch cadence median 是多少？
inter-kernel gap median 是多少？
p95 是多少？

Eager vs Graph
每个 steady step 有多少 LaunchKernel？
有多少 GraphLaunch？
```

它擅长：

```text
精确统计
业务范围过滤
批量计算
重复实验比较
自动化
```

所以：

> SQLite 是 **实验量化工具**。

---

## 1.3 GUI：回答“这些事件在时间上是什么关系”

GUI 最适合回答：

```text
CPU launch K2 的时候
K1 已经执行完了吗？

K2 是 Host 迟提交
还是已经在 GPU queue 里等待？

GraphLaunch 为什么出现在另一个 Graph 执行中间？

MRV2_EXECUTE 已经结束
为什么 GPU Graph 还在继续？

CPU Schedule(N+1)
有没有和 GPU Model(N) overlap？

哪个 PID / Thread 真正在提交 CUDA？

不同 Stream 是串行还是并行？
```

这些问题本质是：

```text
时序
拓扑
依赖
因果关系
```

所以：

> GUI 是 **机制理解工具**。

---

# 2. 三种方式的优缺点总表

| 维度 | CLI | SQLite | GUI |
|---|---|---|---|
| 上手速度 | 最快 | 中等 | 中等 |
| 是否适合第一次快速看 report | 很适合 | 一般 | 适合 |
| 是否适合批量实验 | 很适合 | **最适合** | 不适合 |
| 是否适合精确统计 | 一般 | **最强** | 较弱 |
| 是否适合看 overlap | 很弱 | 可以算 | **最直观** |
| 是否适合看多 Stream | 很弱 | 可以 | **最直观** |
| 是否适合看多进程/线程拓扑 | 较弱 | 可以查 | **最好** |
| 是否适合看 Host→GPU 因果链 | 一般 | 很强 | **最好理解** |
| 是否容易自动化 | **非常适合** | **非常适合** | 很差 |
| 是否容易做 regression | 很适合 | **最好** | 很差 |
| 是否适合发现“未知机制” | 一般 | 较弱 | **最好** |
| 是否适合正式实验报告数字 | 一般 | **最好** | 只适合截图/机制证据 |
| 是否适合 50+ reports | 可以 | **必须优先** | 基本不可行 |
| 学习价值 | 建立统计感 | 建立数据模型 | 建立时间线直觉 |

---

# 3. CLI：什么时候它已经“够用了”

---

## 3.1 Capture 健康检查

这是 CLI 最应该做的第一件事。

比如你刚采完：

```text
eager.nsys-rep
```

第一件事情不是开 GUI。

而是确认：

```text
有没有 NVTX
有没有 CUDA Runtime
有没有 Kernel
有没有 Memcpy
有没有 Graph
```

如果连核心 payload 都没有：

```text
GUI 没意义
SQLite 性能分析也没意义
```

先修采集。

这正是我们之前遇到过的问题：

```text
report 能生成
≠
profiling 成功
```

所以：

> **Capture validation 应该优先 CLI / SQLite row count。**

---

## 3.2 快速看 Top CUDA API

例如：

```text
cudaLaunchKernel
cudaMemcpyAsync
cudaEventSynchronize
cudaGraphLaunch
```

你只想知道：

```text
谁调用得最多？
谁总时间最高？
```

CLI 就够。

这种时候没必要开 GUI。

---

## 3.3 快速看 Top Kernels

例如：

```text
FlashAttention
GEMM
RMSNorm
RoPE
reshape_and_cache_flash
```

如果你只是做：

```text
Top N 排名
```

CLI 足够。

如果发现：

```text
某个 kernel 占 GPU 时间 40%
```

才值得进一步：

```text
SQLite 精确限定 steady range
或
NCU 深钻
```

---

## 3.4 快速 A/B sanity check

例如：

```text
Eager
vs
CUDA Graph
```

只想确认：

```text
LaunchKernel 是否明显减少？
GraphLaunch 是否出现？
Kernel count 是否变化？
```

CLI 很适合。

这类属于：

```text
先看方向对不对
```

不是最终机制分析。

---

# 4. CLI 的局限

CLI 最大的问题：

> **丢失时间关系。**

例如：

```text
cudaLaunchKernel median = 6us
kernel median = 3us
```

CLI 可以告诉你两个数字。

但它不能直接告诉你：

```text
K1结束之后
Host多久才launch K2？
```

也很难直观回答：

```text
GraphLaunch 已经发生
为什么 GPU Graph 1ms 后才开始？
```

所以：

```text
“多少”
CLI很好

“为什么”
CLI通常不够
```

---

# 5. SQLite：为什么以后做 AI Infra Profiling 必须会

如果未来只做单次 GUI 分析，SQLite 似乎很麻烦。

但真正做性能研究，你会遇到：

```text
batch=1/4/8/16
async on/off
eager/graph
不同block_size
不同KV策略
不同模型
不同并发
7 repeats
```

很快就是：

```text
几十个 report
```

你不可能逐个 GUI 点。

所以：

> **SQLite 是从“学习 profiling”走向“工程化 profiling”的关键。**

---

# 6. SQLite 最强的能力：NVTX scoped analysis

全局统计最大的风险是：

```text
把启动 / warmup / prefill / decode / shutdown 混在一起
```

例如：

```text
cudaMemcpyAsync 很多
```

可能只是模型加载。

你真正想研究的是：

```text
steady decode
```

SQLite 可以：

```text
先找到 BRIDGE::MRV2_EXECUTE
或 MODEL_FORWARD 的 start/end
```

再只统计：

```text
这个 NVTX range 内的 CUDA Runtime
这个 range 内的 kernels
这个 range 内的 memcpy
```

这就是：

# **NVTX scoped analysis**

对于推理框架非常重要。

---

# 7. SQLite 特别适合算哪些东西

---

## 7.1 Steady-state median / p95

例如：

```text
MRV2_EXECUTE
Call2~66
```

你不应该手工看：

```text
21.3
21.4
21.5
...
```

而应该统计：

```text
median
p95
min
max
```

正式报告数字应该来自脚本。

---

## 7.2 Kernel count / step

例如：

```text
Eager:
~345 kernels / step

Batch16:
~485 kernels / step
```

这种统计 GUI 不适合做。

---

## 7.3 GPU active union

这是很重要的高级点。

如果存在：

```text
Stream A kernel
和
Stream B kernel overlap
```

简单：

```text
sum(kernel duration)
```

会重复计算 overlap。

所以应该算：

```text
所有 GPU interval 的 union
```

这类精确统计非常适合 Python + SQLite。

---

## 7.4 Launch cadence

例如：

```text
launch1.start
launch2.start
launch3.start
```

计算：

```text
launch[i+1].start - launch[i].start
```

得到：

```text
median launch cadence
p95
```

这能量化：

```text
Host 每多久产生一个新的 GPU work
```

---

## 7.5 Inter-kernel gap

例如：

```text
kernel[i].end
→
kernel[i+1].start
```

统计：

```text
median gap
p95 gap
```

这是我们 Eager 分析里非常关键的指标。

---

## 7.6 Correlation ID 批量关联

GUI 手工做：

```text
CPU launch
corr=303080
↓
GPU kernel
corr=303080
```

SQLite 可以一次做几百次：

```text
Runtime.correlationId
JOIN
Kernel.correlationId
```

然后统计：

```text
launch→GPU latency
```

这就是：

> GUI 理解一次，SQLite 统计全部。

---

# 8. SQLite 的局限

SQLite 最大的问题：

> **它要求你已经知道你想问什么。**

例如你写 SQL：

```text
计算 inter-kernel gap
```

它会给你：

```text
42us
```

但是：

```text
42us 为什么存在？
```

SQL 不会自动告诉你。

你还要知道：

```text
Host launch 在哪里
stream dependency 在哪里
是不是有别的 GPU work
是不是 multi-stream
```

如果一开始连机制都不知道，SQLite 很容易变成：

```text
有很多数字
但不知道这些数字是什么意思
```

所以：

> SQLite 最适合 **验证假设**，不一定最适合 **发现未知机制**。

---

# 9. GUI：什么时候必须打开

GUI 不应该每个实验都开。

但遇到下面这些问题时，非常值得。

---

## 9.1 多进程 / 多线程拓扑

例如：

```text
Frontend PID
EngineCore PID
Worker
哪个线程在提交 CUDA？
```

GUI 一眼就能看清。

纯 SQLite 当然也能查 globalTid/globalPid。

但是学习和排错阶段 GUI 更高效。

---

## 9.2 CPU / GPU overlap

例如：

```text
GPU Model(N)
```

是否和：

```text
CPU Schedule(N+1)
```

overlap？

这种问题 GUI 是第一选择。

因为它本质是：

```text
二维时间关系
```

---

## 9.3 多 Stream

例如：

```text
Compute Stream
Memcpy Stream
Graph Stream
```

是否：

```text
并行
串行
互相阻塞
```

GUI 非常强。

---

## 9.4 Gap / Bubble

看到：

```text
GPU:
K1
   gap
      K2
```

你真正关心：

```text
为什么 gap？
```

这时 GUI 非常有价值。

---

## 9.5 CUDA Graph

Graph 是特别适合 GUI 的对象。

因为你需要理解：

```text
Host GraphLaunch
GPU GraphExec
Stream queue
前后 Graph
异步 enqueue
```

只看 SQL 很难形成正确心智模型。

---

## 9.6 第一次研究陌生框架

例如第一次：

```text
SGLang
TensorRT-LLM
LMCache
```

建议至少：

```text
打开一次 GUI
```

先建立：

```text
process topology
thread topology
NVTX hierarchy
GPU streams
```

之后再自动化。

---

# 10. GUI 的局限

GUI 最大问题：

```text
不可批量
不可重复
容易肉眼误差
不适合精确统计
```

例如：

```text
“这个 gap 看起来大概 30us”
```

学习阶段没问题。

正式报告：

```text
median gap = 42.74us
```

应该来自 SQLite。

所以：

> GUI 是用来理解机制的，不是拿来给几十个实验做统计的。

---

# 11. 三种方式不是线性流程，而是循环

最成熟的 profiling workflow 是：

```text
CLI
↓
快速发现异常
↓
GUI
↓
建立机制假设
↓
SQLite
↓
批量验证
↓
发现异常样本
↓
GUI
↓
再次解释
```

例如我们这次：

```text
SQLite / Stats：
MODEL_FORWARD 很长
kernel 很短
launch很多
      ↓

GUI：
看见 K1结束后 Host 很晚才launch K2
      ↓

形成：
Host/framework cadence hypothesis
      ↓

CUDA Graph A/B
      ↓

SQLite：
MRV2 21ms → 1ms
TPS 7.9×
      ↓

GUI：
理解 GraphLaunch / GraphExec / Stream queue
```

这就是正确分工。

---

# 12. 什么时候“CLI 就够了”

下面这些场景一般不用 GUI：

```text
capture 是否成功
Top APIs
Top Kernels
Memcpy 总量
Graph 是否出现
简单 smoke A/B
自动 regression
CI 性能检查
```

例如：

```text
今天 commit 后
cudaLaunchKernel count 从 300 → 600
```

CLI/SQLite 就足够触发报警。

---

# 13. 什么时候“必须 SQLite”

下面这些场景应该优先 SQLite：

```text
正式 benchmark
多 repeats
多个 batch size
多个模型
多个配置
median/p95
NVTX scoped stats
kernel union
gap distribution
launch cadence
correlation 批量关联
```

如果你需要：

```text
“100个 step 是否都这样？”
```

基本就是 SQLite。

---

# 14. 什么时候“GUI 非常值得”

下面这些问题建议直接 GUI：

```text
为什么这里有空洞？
为什么没有 overlap？
为什么 Graph 在这里开始？
为什么 CPU 已经返回但 GPU 还在跑？
哪个线程在 launch？
哪个 stream 在阻塞？
为什么 Memcpy 没有和 compute overlap？
```

它们共同特征：

> 问题核心是“时间关系”和“执行拓扑”。

---

# 15. 一张工具选择决策树

```text
拿到 .nsys-rep
      │
      ▼
Capture 数据完整吗？
      │
      ├─ NO
      │   → CLI / row count
      │   → 修采集
      │
      ▼
     YES
      │
      ▼
只是想快速知道谁最多？
      │
      ├─ YES
      │   → CLI
      │
      ▼
     NO
      │
      ▼
需要精确统计/多次实验比较？
      │
      ├─ YES
      │   → SQLite
      │
      ▼
     NO / 还不知道为什么
      │
      ▼
问题是否涉及：
overlap / gap / stream / thread /
Graph / queue / process topology？
      │
      ├─ YES
      │   → GUI
      │
      ▼
问题是否已经收敛到单个 kernel？
      │
      ├─ YES
      │   → NCU
      │
      └─ NO
          → GUI建立机制
          → SQLite验证
```

---

# 16. 实际案例：Eager Model Forward 为什么慢

---

## 第一步 CLI / Stats

发现：

```text
cudaLaunchKernel 非常多
kernel 都很短
```

得到：

```text
可能存在 submission overhead
```

但还不能定根因。

---

## 第二步 SQLite

得到：

```text
MODEL_FORWARD ≈20ms
CUDA API ≈几ms
GPU active ≈2ms
kernel median ≈3us
launch cadence ≈几十us
```

问题变成：

```text
为什么 kernel 间有这么大 gap？
```

---

## 第三步 GUI

看：

```text
K1 end
↓
Host晚一段时间
↓
launch K2
↓
K2
```

于是：

```text
Host/framework cadence
```

成为强假设。

---

## 第四步 Controlled A/B

打开 CUDA Graph。

---

## 第五步 SQLite

量化：

```text
MRV2:
21ms → 1ms

TPS:
50 → 401
```

---

## 第六步 GUI

看到：

```text
cudaGraphLaunch
↓
GraphExec
```

并理解：

```text
CPU提前enqueue
GPU按stream继续执行
```

最终闭环。

---

# 17. 实际案例：Async KV Offload 是否真正 overlap

这是未来你 Project-2 很可能遇到的。

问题：

```text
D2H
H2D
Compute
```

是否 overlap。

---

## 第一优先 GUI

因为你首先要看：

```text
Memcpy Stream
Compute Stream
```

在时间线上：

```text
是否真正重叠
```

这是一个典型：

```text
“什么时候”
```

问题。

---

## 第二步 SQLite

在机制确认后，计算：

```text
copy duration
compute duration
overlap union
overlap ratio
```

对 100 个请求统计。

---

## CLI

主要做：

```text
总 Memcpy count
总 bytes
Top APIs
```

作为筛查。

---

# 18. 实际案例：KV INT4 kernel 很慢

如果 Nsight 已经告诉你：

```text
dequant kernel
占 GPU 时间大头
```

这时候：

```text
GUI价值开始下降
```

因为问题已经变成：

```text
这个 kernel 内部为什么慢？
```

此时：

```text
NCU
```

看：

```text
memory throughput
SM utilization
warp stall
occupancy
instruction mix
```

---

# 19. 建议你以后固定一个自动化 Profiling Pipeline

目录可以固定成：

```text
04-experiments/vllm-bridge/raw/...
    xxx.nsys-rep
    xxx.sqlite
    xxx.log
    summary.txt
    summary.json
```

采集之后自动：

```text
nsys profile
↓
nsys export sqlite
↓
Python analyzer
↓
summary.json
```

自动输出：

```text
Capture Health
NVTX Counts
MRV2 median/p95
CUDA Runtime
Kernel Count
Kernel Active Union
Launch Count
Launch Cadence
Kernel Gap
GraphLaunch
Graph Duration
Memcpy
```

然后：

```text
summary正常
→ 不开GUI

summary异常
→ GUI
```

这才是长期工程化路线。

---

# 20. 对你未来的推荐使用比例

不是硬标准，只是工作习惯：

```text
SQLite / 自动 analyzer      ~60%
CLI / Stats                ~25%
GUI 机制分析                ~10%
NCU Kernel 深钻             ~5%
```

在学习阶段可以：

```text
GUI 多一些
```

因为现在你正在建立：

```text
CPU
CUDA API
Stream
GPU
Graph
```

这些时间线直觉。

以后越熟练：

```text
越应该脚本化
```

---

# 21. SQLite 需要学到什么程度

不用成为数据库工程师。

够用的 SQL：

```text
SELECT
WHERE
JOIN
GROUP BY
ORDER BY
COUNT
AVG
MIN
MAX
```

更重要的是理解 Nsight 数据模型：

```text
NVTX_EVENTS
CUPTI_ACTIVITY_KIND_RUNTIME
CUPTI_ACTIVITY_KIND_KERNEL
StringIds
Memcpy
Memset
Synchronization
```

以及常见字段：

```text
start
end
correlationId
globalTid
globalPid
streamId
```

真正重要的是：

> **知道怎样把 profiler 问题翻译成数据查询。**

---

# 22. 建议长期维护自己的 `nsys_analyze.py`

以后建议形成：

```text
tools/nsys_analyze.py
```

输入：

```text
xxx.sqlite
```

输出：

```text
=== Capture Health ===
NVTX rows
Runtime rows
Kernel rows

=== Steady Decode ===
num_steps
MRV2 median / p95
MODEL_FORWARD median

=== Host Submission ===
Launch count / step
Launch median
Launch cadence
Host gap

=== GPU ===
Kernel count
Kernel median
GPU active union

=== Graph ===
Graph launch count
Graph launch median
Graph duration

=== Memory ===
H2D / D2H count
bytes
duration
```

长期会比每次 GUI 手工分析高效得多。

---

# 23. CLI / SQLite / GUI 的常见误用

---

## 误用 1：只看 CLI 就开始解释因果

错误：

```text
cudaLaunchKernel 很多
→ 所以 launch 是瓶颈
```

正确：

```text
cudaLaunchKernel很多
→ 值得进一步看 Host/GPU cadence
```

---

## 误用 2：只看 SQLite 数字，不看时间线

错误：

```text
inter-kernel gap 42us
→ GPU一定在等Host
```

正确：

```text
还要看 Host launch 是什么时候发生
```

---

## 误用 3：用 GUI 肉眼做正式统计

错误：

```text
“看起来大概40us”
```

写进正式报告。

正确：

```text
GUI建立机制
SQLite给median/p95
```

---

## 误用 4：所有实验都开 GUI

这样会：

```text
慢
不可重复
难比较
```

GUI 应该：

```text
低频
高价值
```

---

## 误用 5：已经定位到 kernel 还停留在 Nsight Systems

如果问题已经是：

```text
这个 Triton kernel 为什么慢？
```

就应该：

```text
NCU
```

而不是继续在 Systems 上看时间线。

---

# 24. 最终选择原则

可以固定记成：

```text
“谁最多？”
→ CLI

“准确多少？”
→ SQLite

“为什么这样？”
→ GUI

“很多次是不是都这样？”
→ SQLite

“CPU和GPU谁在等谁？”
→ GUI

“多个stream有没有overlap？”
→ GUI

“这个kernel内部为什么慢？”
→ NCU
```

最后压缩成一句：

> **CLI 找异常，GUI 找机制，SQLite 做证据，NCU 钻 kernel。**

---

# 25. 最推荐的完整 AI Infra Profiling 工作流

```text
问题
↓
定义 workload / baseline
↓
Nsight capture
↓
CLI Capture Health
↓
CLI 快速 triage
↓
发现问题
↓
GUI 建立时序/拓扑机制
↓
形成 hypothesis
↓
SQLite 写成可重复统计
↓
批量 A/B
↓
Controlled intervention
↓
SQLite 验证效果
↓
GUI 抽查机制是否真的改变
↓
如果落到单 kernel
↓
NCU
↓
优化
↓
重新回到 Nsight Systems 看系统级瓶颈是否迁移
```

这套流程非常适合：

```text
vLLM
SGLang
LMCache
KV Cache
Triton
CUDA Graph
PD / Async Offload
```

以后换框架也可以继续复用。

---

# 26. 本文最终结论

不要再把：

```text
CLI
SQLite
GUI
```

看成三个“谁更高级”的工具。

它们实际上是三个不同的问题空间：

```text
CLI
=
快速全局摘要

SQLite
=
精确定量与自动化

GUI
=
执行关系与机制理解
```

成熟的性能工程不是一直用 GUI，也不是只会 SQL，而是：

> **先明确当前问题属于“筛查、定量、还是机制理解”，再选择成本最低、信息密度最高的工具。**

对于长期 AI Infra 项目，最终应该越来越多依赖：

```text
SQLite + 自动 analyzer
```

而把：

```text
Nsight GUI
```

保留给真正需要理解：

```text
overlap
gap
stream
queue
async
graph
多进程
```

这些复杂执行关系的场景。
