# 11-vLLM E6.5 / E6.6：从 Runtime 语义追踪到 Nsight 瓶颈定位，再到 CUDA Graph 闭环验证

> 版本：vLLM v0.26.0 / V1 Engine / V2 Model Runner  
> GPU：NVIDIA A100 80GB PCIe  
> 模型：`/data/models/Qwen3-0.6B`  
> 目的：把前面零散的 E6、E6.5、E6.6 实验重新整理成一条完整的性能分析链：**先知道 Runtime 在做什么，再知道 CPU/GPU 在什么时候做，最后用 controlled A/B 验证瓶颈原因和优化机制。**

---

# 0. 这份报告解决什么问题

前面的实验很多，最容易出现的问题不是“没做实验”，而是：

```text
日志很多
Nsight 表很多
kernel 名很多
SQLite 字段很多

但是不知道：
这些证据分别回答什么问题？
为什么要做下一步？
一个结论到底是怎么从上一步推出来的？
```

所以本文不按“脚本编号”机械罗列，而按真实的性能工程流程重新组织：

```text
E6：Runtime 语义链
↓
发现 async scheduling 只能从 host 语义上理解
↓
E6.5：引入 Nsight，把 Host 语义和 GPU hardware timeline 对齐
↓
先验证 async overlap 是否真的发生
↓
发现没有稳定 overlap，同时 MRV2_EXECUTE 很长
↓
分析 MRV2_EXECUTE 内 CUDA API / kernel / host gap
↓
E6.6：逐层拆 ModelRunner
↓
定位到 MODEL_FORWARD ≈ 95%
↓
提出假设：eager per-op dispatch / submission 是核心瓶颈
↓
CUDA Graph controlled A/B
↓
LaunchKernel 345 → 6，GraphLaunch 0 → 1
MRV2_EXECUTE 21.534ms → 1.017ms
TPS 50.58 → 401.36
↓
瓶颈原因得到闭环验证
```

这条链是本文最重要的内容。

---

# 1. 最终结论先放在前面

## 1.1 当前 workload 的主瓶颈是什么

对于：

```text
Qwen3-0.6B
A100 80GB
batch=1 decode
enforce_eager=True
CompilationMode.NONE
CUDAGraphMode.NONE
FlashAttention 2
```

最终实验证明：

> **主瓶颈不是 Scheduler，不是 Paged-KV metadata，也不是 FlashAttention kernel 本身，而是 eager model-forward 的 host/framework submission path。**

最关键的证据：

```text
MRV2_EXECUTE          ≈ 22 ms
MODEL_FORWARD         ≈ 21 ms
MODEL_FORWARD占比     ≈ 94.5%~95.5%

GPU active            ≈ 1.9 ms（batch1）
CUDA Runtime API      ≈ 2.8 ms
kernel median         ≈ 3.4 us
kernel count          ≈ 345 / decode execute
```

也就是说：

```text
不是某个 GPU kernel 跑了 20ms

而是：

Python / framework / dispatcher / host wrapper
        ↓
    launch tiny kernel
        ↓
Python / framework / dispatcher / host wrapper
        ↓
    launch tiny kernel
        ↓
    ……重复数百次
```

---

## 1.2 CUDA Graph 是否真的解决了这个问题

是，而且我们做了**干净的 controlled A/B**：

```text
A：CompilationMode.NONE + CUDAGraphMode.NONE
B：CompilationMode.NONE + CUDAGraphMode.FULL
```

也就是说，A/B 两边都没有 `torch.compile` / `VLLM_COMPILE`，唯一主要变量就是 CUDA Graph。

结果：

| 指标 | Eager | Pure CUDA Graph FULL | 变化 |
|---|---:|---:|---:|
| MRV2_EXECUTE median | 21.534 ms | 1.017 ms | 约 21.18× host speedup |
| CUDA API time | 2.735 ms | 0.148 ms | 大幅下降 |
| LaunchKernel / execute | 345 | 6 | 大幅减少 |
| GraphLaunch / execute | 0 | 1 | 出现 graph replay |
| 64-token median elapsed | 1265.423 ms | 159.457 ms | -87.4% |
| throughput | 50.58 tok/s | 401.36 tok/s | **7.936×** |

再使用 vLLM 默认优化路径：

```text
VLLM_COMPILE + FULL_AND_PIECEWISE
```

得到：

```text
138.398 ms
462.43 tok/s
9.143× eager
```

因此可以非常明确地说：

> **CUDA Graph 不是“理论上可能有帮助”，而是被当前实验直接验证为 eager submission bottleneck 的主要解决手段。**

---

# 2. 实验环境与证据边界

## 2.1 固定环境

```text
repo:
~/learning/llm-kv-lab/third_party/vllm

vLLM:
v0.26.0
commit:
568afb3a13806beb53bb2e6bd518269357b237c0

Python:
3.10.20

PyTorch:
2.11.0+cu129

Torch CUDA runtime:
12.9

GPU driver:
565.57.01

GPU:
A100 80GB PCIe

Model:
/data/models/Qwen3-0.6B

V1 Engine
V2 Model Runner
TP=1
PP=1
DP=1
```

正式 Nsight 捕获最终使用：

```text
/opt/nvidia/nsight-systems/2024.7.1/bin/nsys
Nsight Systems 2024.7.1.84
```

系统同时存在旧版 Nsight 2023，但后面会解释为什么最终 profile 固定到 2024.7.1。

---

## 2.2 E6 基线 workload

最早的 E6 workload：

```text
script:
04-experiments/vllm-bridge/scripts/e6_paged_kv_attention.py

prompt_tokens = 31
block_size = 16
max_tokens = 4
max_num_seqs = 1
prefix cache = OFF
dtype = BF16
enforce_eager = True
attention backend = FlashAttention 2
```

这是一个故意很小的 workload。

目的不是追求 production throughput，而是让：

```text
Scheduler
block allocation
block_table
slot_mapping
KV write
KV read
Decode 下一 block 边界
```

都能被清楚观察。

---

## 2.3 本文对证据分三层

后面所有结论必须区分：

### A. Runtime / semantic evidence

来自：

```text
[BRIDGE][CORE]
[BRIDGE][SCHED]
[BRIDGE][MRV2]
[BRIDGE][ADDR]
[BRIDGE][ATTN]
```

它回答：

> 代码逻辑走到了哪里、谁调用谁、状态怎么变化。

它**不能**直接证明 GPU 某个 kernel 此时正在跑。

---

### B. Nsight host/hardware evidence

来自：

```text
NVTX
CUDA Runtime API
CUDA Kernel
Memcpy / Memset
Correlation ID
```

它回答：

> CPU 在什么时候调用 CUDA，GPU 在什么时候真正执行 kernel。

---

### C. Controlled intervention evidence

例如：

```text
只改变 CUDAGraphMode.NONE → FULL
CompilationMode 保持 NONE
```

如果性能和 launch 行为随之发生预期变化，才可以把“瓶颈原因”升级为强结论。

这是本实验最后能闭环的关键。

---

# 3. E6 第一阶段：先把 Runtime 主链跑通

在做 Nsight 之前，我们先解决一个更基础的问题：

> 一条真实 request，从 Frontend 到 Scheduler，再到 ModelRunner、Paged KV、Attention Backend，到底是怎么流动的？

完整链：

```text
Frontend Process
│
│ LLM.generate()
│ SyncMPClient.add_request()
│
└── ZMQ
    ↓
EngineCore Process
│
├── EngineCoreProc input thread
│   ├── IPC_RECV
│   └── IPC_ENQUEUE
│
└── busy loop
    ↓
EngineCore.step_with_batch_queue()
    ↓
Scheduler.schedule()
    ↓
KVCacheManager / BlockPool
    ↓
SchedulerOutput
    ↓
UniProcExecutor
    ↓
WorkerWrapperBase
    ↓
CUDA Worker
    ↓
V2 GPUModelRunner.execute_model()
    ↓
prepare_inputs()
    ↓
prepare_attn()
    ├── block_table
    └── slot_mapping
    ↓
Qwen3 model forward
    ↓
Attention
    ├── KV WRITE
    └── KV READ
    ↓
sample_tokens()
    ↓
batch_queue
    ↓
Scheduler.update_from_output()
```

这个阶段的价值不是性能，而是建立**语义坐标系**。

后面 Nsight 看到一段 CUDA kernel 时，我们必须知道它属于：

```text
Scheduler？
ModelRunner preparation？
Attention？
Sampling？
```

否则 timeline 再漂亮也没有业务含义。

---

# 4. E6 的 Paged KV 数据链验证

## 4.1 block_table 与 slot_mapping 的职责不同

必须长期记住：

```text
slot_mapping
= WRITE address
= 当前 token 的 K/V 应该写到哪个 physical KV slot

block_table
= READ mapping
= 当前 request 的 logical KV blocks 对应哪些 physical blocks
```

---

## 4.2 slot 计算公式

当前：

```text
block_size = 16
```

对于一个 token position：

```text
logical_block = position // block_size
offset        = position % block_size
physical_block = block_table[logical_block]
slot           = physical_block * block_size + offset
```

例如：

```text
block_table = [1, 2, 3]
position = 32

logical_block = 32 // 16 = 2
physical_block = block_table[2] = 3
offset = 32 % 16 = 0

slot = 3 * 16 + 0 = 48
```

E6 实测：

```text
expected_slot = 48
actual_slot   = 48
match         = True
```

---

## 4.3 position32 为什么是非常好的系统锚点

31-token prompt 最开始使用两个 physical blocks。

当继续 decode 到 position32：

```text
positions 0...31
已经填满两个 16-token blocks
↓
Scheduler 下一轮发现需要新 KV capacity
↓
BlockPool 分配 physical block3
↓
SchedulerOutput new_block_ids=[3]
↓
ModelRunner block_table=[1,2,3]
↓
position32 → logical block2
↓
block_table[2] = physical3
↓
slot = 48
↓
Attention KV_WRITE slot=[48]
↓
reshape_and_cache_flash 写入实际 KV Cache
↓
Attention KV_READ block_table=[1,2,3]
```

这条链第一次动态串起：

```text
Scheduler control plane
→ physical block ownership
→ ModelRunner addressing
→ exact KV slot
→ Attention Backend
→ GPU KV data plane
```

这也是后面做 KV 优化时最重要的基础。

---

# 5. E6 暴露出的第一个性能问题：Async Scheduling 到底“异步”在哪里

E6 runtime trace 里出现了：

```text
Call0
QUEUE_PUSH
YIELD_WITHOUT_CONSUME

然后没有先：
WAIT_RESULT
RECONCILE
TOKEN_APPEND

而是直接：
Call1
SCHEDULE_BEGIN
```

因此 E6 可以直接证明：

```text
Call0 execution 已提交
↓
Call0 result 尚未 reconcile 到 Scheduler
↓
EngineCore 已允许 Call1 schedule / submit
```

这说明：

> **Async Scheduling 的 host/runtime 语义确实存在：submission 和 result reconciliation 被解耦。**

但这里出现一个重要误区。

---

## 5.1 为什么 E6 的 print 日志不能证明 GPU overlap

`[BRIDGE]` 日志主要是 Python/host trace。

例如：

```text
[MRV2 EXECUTE_BEGIN]
[ADDR]
[KV_WRITE]
[KV_READ]
```

它记录的是：

```text
CPU 已经执行到某个 API / Python 位置
```

不是：

```text
GPU 对应 kernel 已经物理执行完毕
```

CUDA 的常见关系是：

```text
CPU:
launch K1
↓
继续准备 K2
↓
launch K2

GPU:
      K1 ─────
           K2 ─────
```

所以：

```text
Host 日志顺序
≠
GPU 完成顺序的直接证据
```

这就是为什么 E6.5 必须引入 Nsight Systems。

---

# 6. E6.5 的核心方法：建立三层时间线

后面所有 Nsight 分析都围绕这三层：

```text
业务语义：NVTX
    ↓
CPU submission：CUDA Runtime API
    ↓
GPU execution：CUDA Kernel
```

具体：

```text
BRIDGE::MRV2_EXECUTE
       │
       ├── cudaLaunchKernel(...)
       │       correlationId = X
       │
       └── GPU kernel
               correlationId = X
```

这样我们才能回答：

> 某个 `MRV2_EXECUTE` 内到底 launch 了哪些 kernel？

以及：

> Call N 的 GPU kernel 是否和 Call N+1 的 Scheduler / ModelRunner 重叠？

---

# 7. 为什么选择 NVTX，而不是继续 print

正式 profile 时：

```text
QCACHE_BRIDGE_TRACE=0
QCACHE_BRIDGE_NVTX=1
```

原因：

```text
print / logging
可能引入 stdout lock、format、IO、进程调度扰动

NVTX
只给 timeline 打语义 range，开销更适合 profiler
```

最终 coarse NVTX：

### EngineCore

```text
BRIDGE::SCHEDULE::<call_id>
BRIDGE::EXEC_SUBMIT::<call_id>
BRIDGE::SAMPLE_SUBMIT::<call_id>
```

### Worker / ModelRunner bridge

```text
BRIDGE::MRV2_EXECUTE
BRIDGE::MRV2_SAMPLE
```

后续 Phase 拆分又加入：

```text
BRIDGE::MRV2_PHASE::BATCH_DISPATCH
BRIDGE::MRV2_PHASE::PREPARE_INPUTS
BRIDGE::MRV2_PHASE::PREPARE_ATTN_ADDR
BRIDGE::MRV2_PHASE::PREPARE_ATTN_META
BRIDGE::MRV2_PHASE::MODEL_STATE_INPUTS
BRIDGE::MRV2_PHASE::MODEL_FORWARD
```

---

# 8. Nsight 捕获命令为什么只开这些选项

正式第一轮使用的核心思路：

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  ...
```

这里是故意“少采”。

## `--trace=cuda,nvtx`

只需要：

```text
NVTX：业务阶段
CUDA：Runtime API + Kernel + Memcpy 等 CUDA activity
```

## `--sample=none`

一开始不做 CPU sampling，因为：

```text
采样开销更高
数据更多
问题还没有缩小
```

正确的 profiler 方法是：

```text
先粗分
↓
确定最大区域
↓
再深入
```

而不是第一次就把所有功能打开。

---

# 9. Profiling 基础设施本身也经历了一轮排错

这一段很重要，因为真实 AI Infra profiling 经常不是“运行 nsys 就成功”。

---

## 9.1 第一版 Python `nvtx.annotate(domain=...)` 出现崩溃

最开始使用 Python `nvtx` package 和 custom domain。

完整 vLLM multiprocessing/init profile 下发生 segfault，堆栈落在 NVTX3 domain 初始化路径。

但最小 probe 中：

```text
Python nvtx annotate 可以工作
torch.cuda.nvtx 也可以工作
```

所以不能简单下结论：

```text
“NVTX 与环境不兼容”
```

更准确的处理是：

> full vLLM multiprocessing/init + Python NVTX Domain 这条组合路径存在问题。

因此把自己的 annotation 改成：

```python
with torch.cuda.nvtx.range("BRIDGE::..."):
    ...
```

并取消 custom domain。

这一步后 segfault 消失。

---

## 9.2 Nsight 2023 能生成 report，但 full vLLM payload 不完整

系统 CUDA 12.1 自带：

```text
Nsight Systems 2023.1.2
```

在某些 full vLLM capture 中：

```text
report 文件生成成功
```

但导出后缺少实际：

```text
NVTX_EVENTS
CUDA Runtime
CUDA Kernel
```

这说明：

> **“有 `.nsys-rep` 文件”不代表 capture 真正成功。**

必须检查导出表。

---

## 9.3 换 Nsight 2024 后，最小 spawn probe 正常

使用：

```text
/opt/nvidia/nsight-systems/2024.7.1/bin/nsys
```

最小 multiprocessing + CUDA + NVTX probe 可以得到：

```text
NVTX
Runtime
Kernel
Memcpy
Memset
Synchronization
```

因此 2024 能力本身正常。

---

## 9.4 但 full vLLM 默认 multiprocessing 仍然可能没有 tracing payload

P2D：

```text
Nsight 2024
full vLLM
默认 multiprocessing startup
```

report 能生成，但仍缺关键 tracing data。

随后固定：

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

并验证：

```text
get_mp_context().get_start_method() = spawn
```

P2E 成功得到：

```text
NVTX_EVENTS                     40
CUPTI_ACTIVITY_KIND_RUNTIME     7148
CUPTI_ACTIVITY_KIND_KERNEL      2973
```

因此最终正式 profiling 环境固定：

```text
Nsight Systems 2024.7.1
+
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

这个经验很重要：

> profiler 是否能正确跟踪多进程 runtime，与 process creation method 有关；在多进程框架里不能只检查模型本身，也要检查 profiler 能否正确覆盖 child process。

---

# 10. 为什么必须做 GPU idle guard

早期 GPU0 / GPU2 上存在其他进程。

这会产生一个很危险的现象：

```text
我们 launch 的 kernel
↓
因为别人占 GPU 而延迟进入 GPU
↓
kernel 被排队
↓
它可能“拖”到下一轮 CPU phase
↓
看起来好像发生了 cross-step overlap
```

但这个 overlap 不是我们的 runtime pipeline 主动制造的，而是外部 contention 造成的 queue delay。

所以后来 formal script 加入：

```text
连续 8 次 nvidia-smi GPU utilization sample

max util > 10%
→ abort
```

最终最关键实验在 GPU1 上：

```text
8/8 samples = 0%
实验后 GPU util = 0%
```

虽然 GPU 上仍有少量其他进程占显存，但采样期间没有观察到 compute activity。

所以后面的最终 timing / overlap 证据可信度显著高于早期 shared-GPU 结果。

---

# 11. SQLite 到底怎么分析：这是前面最容易看不懂的部分

Nsight GUI 很适合人看，但自动分析更适合导出 SQLite。

核心表：

```text
NVTX_EVENTS
CUPTI_ACTIVITY_KIND_RUNTIME
CUPTI_ACTIVITY_KIND_KERNEL
StringIds
```

---

## 11.1 NVTX_EVENTS：业务阶段在哪里

例如：

```text
BRIDGE::MRV2_EXECUTE
start = S
end   = E
```

这给出一个 host-side business interval：

```text
[S, E]
```

---

## 11.2 Runtime：CPU 在这个区间调用了哪些 CUDA API

例如：

```text
cudaLaunchKernel
cudaMemcpyAsync
cuLaunchKernelEx
cudaStreamSynchronize
...
```

每条 Runtime activity 有：

```text
start
end
correlationId
thread/process identity
nameId
```

`nameId` 再通过 `StringIds` 转成函数名。

---

## 11.3 Kernel：真正 GPU execution

Kernel activity 有：

```text
start
end
correlationId
streamId
process identity
```

如果 Runtime API 与 Kernel 的：

```text
correlationId 相同
```

就可以建立：

```text
CPU API submit
      ↓
对应 GPU kernel
```

这就是：

```text
NVTX
→ CUDA Runtime
→ correlationId
→ GPU Kernel
```

---

## 11.4 为什么不能只看 kernel 名

假设看到：

```text
flash_fwd_splitkv_kernel
```

单看名字只能知道：

```text
GPU 跑了一个 FlashAttention kernel
```

但不知道它属于：

```text
Call0？
Call1？
Prefill？
Decode？
MRV2_EXECUTE？
Sampling？
```

必须先通过 host range / process / correlation 把它归属到业务阶段。

---

# 12. Async overlap 实验到底怎么判定

问题：

> Call N 的 GPU execution 是否与 Call N+1 的 CPU Scheduler / ModelRunner 重叠？

不是看两行日志有没有交叉，而是做 interval overlap test。

对于：

```text
GPU kernel interval = [Ks, Ke]
Host phase interval = [Hs, He]
```

存在 overlap 当且仅当：

```text
Ks < He
且
Ke > Hs
```

等价于：

```text
max(Ks, Hs) < min(Ke, He)
```

我们分别检查：

```text
prior Call GPU
vs
next SCHEDULE

prior Call GPU
vs
next EXEC_SUBMIT

prior Call GPU
vs
next MRV2_EXECUTE

prior Call GPU
vs
next MRV2_SAMPLE
```

这比“看 timeline 感觉好像重叠”严格得多。

---

# 13. E6.5：第一轮 clean overlap 结果

clean rerun 中：

```text
Call0 MRV2_EXECUTE ≈ 50.95 ms（包含首轮特殊开销）
Call0 execute kernels = 406
Call0 sample kernels  = 10
```

在 `SCHEDULE::1` 开始时：

```text
Call0 active kernels = 0
```

而且：

```text
Call0 kernels overlap EXEC_SUBMIT::1 = 0
Call0 kernels overlap MRV2_EXECUTE::1 = 0
Call0 kernels overlap MRV2_SAMPLE::1 = 0
```

最后一个 Call0 kernel 在 `SCHEDULE::1` 前已经结束。

这说明：

> 在该小模型、TP1、eager workload 中，虽然 async scheduling 在 host state machine 层允许 ahead-of-reconcile，但没有观察到上一 step GPU 与下一 step ModelRunner 的实际 hardware overlap。

---

# 14. 为什么后来又做 batch16 overlap

一个合理假设是：

```text
batch1 GPU work 太短
↓
CPU 下一 step 还没开始
GPU 已经做完
↓
所以看不到 overlap
```

那么：

```text
batch ↑
↓
每 step GPU work ↑
↓
可能给 CPU N+1 留出 overlap window
```

因此 E6.6-C 专门测试：

```text
batch=16
async_scheduling=True
max_concurrent_batches=2
CUDA Graph OFF
enforce_eager=True
GPU1 clean
```

最终 steady decode：

```text
transitions = 33

prior GPU overlap next SCHEDULE      = 0
prior GPU overlap next EXEC_SUBMIT   = 0
prior GPU overlap next MRV2_EXECUTE  = 0
```

而每个 transition 之间仍有约：

```text
200~500 us 正 gap
```

因此：

> **不能把之前 batch 增大时看到的 async throughput 提升归因于“GPU N 与 CPU N+1 跨 step overlap”。**

这一点非常重要，因为它修正了一个看起来很合理、但被硬件证据否定的解释。

---

# 15. Async Scheduling 到底应该怎么理解

现在可以准确地分成两件事。

## 软件能力

```text
async ON
max_concurrent_batches = 2
```

说明 runtime 允许：

```text
batch transaction outstanding
result 延迟 reconcile
```

这个能力是真实存在的。

## 实际 hardware overlap

却取决于：

```text
CPU 每 step 多快
GPU 每 step 多久
是否有 synchronization/dependency
execution mode
batch size
GPU contention
```

所以：

```text
max_concurrent_batches=2
```

绝对不等价于：

```text
GPU(N) 一定 overlap CPU(N+1)
```

这是本轮 async 实验最重要的结论。

---

# 16. 接下来为什么从“overlap”转向“MRV2_EXECUTE 为什么这么长”

在 clean decode 中，一个稳定现象越来越明显：

```text
SCHEDULE 很短
MRV2_EXECUTE 很长
GPU kernels 很短
```

所以分析问题从：

```text
“async 有没有 overlap？”
```

转成：

```text
“为什么一次 ModelRunner execute 要 20 多 ms？”
```

这就是 E6.6 的真正主线。

---

# 17. 第一层时间拆解：MRV2 wall、CUDA Runtime、GPU active

clean batch1 最终数据：

```text
MRV2_EXECUTE median      22.284 ms
CUDA Runtime API          2.769 ms
GPU kernel active union   1.865 ms
kernel count                345
median kernel             3.360 us
```

这里必须理解几个概念。

---

## 17.1 MRV2 wall

```text
BRIDGE::MRV2_EXECUTE start → end
```

是 CPU host function 的 wall time。

它里面可能包含：

```text
Python
C++
PyTorch dispatcher
CUDA API
GPU synchronization wait
metadata preparation
operator wrapper
```

---

## 17.2 CUDA Runtime API time

例如：

```text
cudaLaunchKernel 7us
```

意思是：

> CPU 在 CUDA Runtime 的这个 API 调用里花了约 7us。

不是：

> GPU kernel 执行了 7us。

---

## 17.3 GPU active union

将属于该 MRV2 execute 的 GPU kernel intervals 做时间 union。

为什么不是简单 sum？

因为不同 stream 有可能重叠。

union 更接近：

> 在这段时间里，GPU 至少有一个归属于该 execute 的 kernel active 了多久。

---

## 17.4 这些时间不能简单相加减

不能写：

```text
host wall - CUDA API - GPU = Python time
```

因为：

```text
CPU 与 GPU 可以 overlap
Runtime API 与 GPU execution 不是串行总和
```

所以这里的正确用法是判断量级，而不是做错误的加法分摊。

---

# 18. 小 kernel + 大 host gap：host submission bottleneck 的早期信号

前面 stable decode 还测到了非常关键的一组 cadence 数据：

```text
launch API median        ≈ 7.02 us
GPU kernel median        ≈ 3.30 us
launch start cadence     ≈ 50.48 us
end-to-next-start gap    ≈ 42.74 us
launch→GPU queue delay   ≈ 1.18 us median
```

这组数据怎么理解？

假设：

```text
launch K1 start
↓ 7us
launch K1 return
↓ 约43us
launch K2 start
```

而 GPU kernel 只有：

```text
≈3us
```

说明大部分 iteration 时间并不是：

```text
GPU kernel 在算
```

也不是：

```text
CPU 一直卡在 cudaLaunchKernel API 里面
```

而是发生在：

```text
两次 CUDA API 之间的 host/framework path
```

所以当时我们把问题描述为：

> **host/framework submission bound**

但这一阶段仍然只是强信号，还没有定位到 ModelRunner 内部哪个 phase。

---

# 19. 为什么全局 `nsys stats` 不能直接回答 MRV2 内部瓶颈

E6.6-D 曾尝试 CPU/Python sampling，但环境提示：

```text
CPU IP/backtrace sampling not supported, disabling
CPU context switch tracing not supported, disabling
CUDA backtraces will not be collected because CPU sampling is disabled
```

因此：

```text
Python function hotspot
C++ call stack
```

没有拿到。

同时 `nsys stats` 中出现：

```text
cudaMemGetInfo
cuLibraryLoadData
cudaMalloc
cudaFree
大 H2D copy
```

这些明显包含：

```text
初始化
模型加载
warmup
memory profiling
正式 inference
shutdown
```

所以不能拿整个 report 的：

```text
CUDA API Summary
GPU MemOps Summary
```

直接说：

> “decode 的 22ms 主要就是 cudaLaunchKernel”。

这是 profiling 中很常见的错误。

正确做法：

> **必须把统计限制到业务 NVTX range 内。**

---

# 20. E6.6-E：对 GPUModelRunner.execute_model 做 Phase Decomposition

CPU sampling 不可用后，没有继续硬撞 sampling，而是改用更稳定的 source-level NVTX narrowing。

在 `GPUModelRunner.execute_model()` 中插入：

```text
BATCH_DISPATCH
PREPARE_INPUTS
PREPARE_ATTN_ADDR
PREPARE_ATTN_META
MODEL_STATE_INPUTS
MODEL_FORWARD
```

目的：

```text
先回答 22ms 在哪一个“大块”
```

而不是一开始就把 28 层 Transformer 全部打满 NVTX。

这就是 top-down profiling。

---

# 21. Phase Decomposition 最终结果

## Batch=1

```text
MRV2_EXECUTE       22.284 ms

BATCH_DISPATCH      0.0221 ms
PREPARE_INPUTS      0.5780 ms
PREPARE_ATTN_ADDR   0.1329 ms
PREPARE_ATTN_META   0.1130 ms
MODEL_STATE_INPUTS  0.0031 ms
MODEL_FORWARD      21.0496 ms
```

`MODEL_FORWARD` 占：

```text
21.0496 / 22.284 ≈ 94.46%
```

未归因只剩：

```text
≈0.385 ms
```

---

## Batch=16

```text
MRV2_EXECUTE       23.170 ms

BATCH_DISPATCH      0.0197 ms
PREPARE_INPUTS      0.4841 ms
PREPARE_ATTN_ADDR   0.1135 ms
PREPARE_ATTN_META   0.1032 ms
MODEL_STATE_INPUTS  0.0027 ms
MODEL_FORWARD      22.1161 ms
```

`MODEL_FORWARD` 占：

```text
22.1161 / 23.170 ≈ 95.45%
```

未归因：

```text
≈0.331 ms
```

---

# 22. 这一步排除了什么

从这一刻开始，可以正式排除以下部分是 20ms 级主瓶颈：

```text
Scheduler                  ❌
Batch dispatch             ❌
prepare_inputs              ❌
block_table / slot_mapping  ❌
attention metadata          ❌
```

真正的大头被锁定到：

```text
GPUModelRunner.execute_model()
        ↓
MODEL_FORWARD
        ↓
self.model(**model_inputs)
```

这一步是整个分析的关键 narrowing。

---

# 23. Batch1 vs Batch16：为什么 16 倍 token，GPU 时间只涨一点

clean Nsight 对比：

| 指标 | Batch1 | Batch16 | 比例 |
|---|---:|---:|---:|
| MRV2_EXECUTE | 22.284 ms | 23.170 ms | 1.040× |
| CUDA API | 2.769 ms | 3.205 ms | 1.157× |
| GPU active | 1.865 ms | 2.634 ms | 1.412× |
| kernel count | 345 | 485 | 1.406× |
| kernel median | 3.360 us | 3.520 us | 1.048× |

但每个 decode step 处理 token 数：

```text
1 → 16
= 16×
```

---

## 23.1 为什么 kernel latency 不会 ×16

batch1 的 Linear 更接近：

```text
[1, H] × [H, ...]
```

GPU 并行度很低。

batch16：

```text
[16, H] × [H, ...]
```

可以让更多 thread blocks / warps 同时工作。

因此：

```text
计算量 ↑很多
```

可以部分转换成：

```text
GPU parallelism ↑
```

而不是完全转换成：

```text
kernel latency ↑16×
```

---

## 23.2 kernel count 为什么也没有 ×16

batch16 不是：

```text
执行16次模型
```

而是：

```text
同一次 model forward
同一套 operator DAG
一次处理更多 requests/tokens
```

所以：

```text
345 → 485
```

而不是：

```text
345 × 16
```

---

## 23.3 batching 的真正收益

当前 eager 模式有一个约 20ms 的固定-ish model-forward host cost。

batch1：

```text
1 token 承担一次这套 host cost
```

batch16：

```text
16 tokens 一起摊薄这套 host cost
```

所以 batching 对 small-model eager decode 非常有效。

这与 CUDA Graph 的优化方向不同但互补：

```text
Batching：
把固定成本摊给更多 token

CUDA Graph：
把固定成本本身大幅消掉
```

---

# 24. 到这里为什么提出 CUDA Graph 假设

现在已经有四条证据：

```text
1. MODEL_FORWARD ≈ 21ms，占 MRV2 ≈95%

2. GPU active 只有 ≈1.9ms

3. 每步约 345 个 kernels

4. median kernel 只有 ≈3.4us
```

而当前配置明确是：

```text
enforce_eager=True
CompilationMode.NONE
CUDAGraphMode.NONE
```

于是最自然的假设：

> 当前最大的 cost 来自 eager 模式下数百个 operator 的重复 dispatch / host wrapper / CUDA submission。

CUDA Graph 正是针对这种 pattern：

```text
Eager：
CPU 每次都逐 op launch

CUDA Graph：
第一次 capture
后续一次 graph replay
```

但注意：

> 在做 A/B 前，这仍然只是 hypothesis。

---

# 25. 先把 Eager、CUDA Graph、FlashAttention、Compile 分层

这几个概念不能混。

```text
Runtime scheduling
Async / continuous batching
        ↓
Model execution / compilation
Eager / torch.compile / VLLM_COMPILE
        ↓
CUDA launch replay
CUDAGraph NONE / PIECEWISE / FULL
        ↓
Operator backend
FlashAttention / FlashInfer / custom CUDA / Triton
        ↓
GPU kernel
```

---

## 25.1 FlashAttention 是 Attention 算子实现

当前即使：

```text
enforce_eager=True
```

仍然：

```text
Using FLASH_ATTN
Using FlashAttention version 2
```

完全不冲突。

因为：

```text
Eager
= 整个模型的算子怎么被 host 逐个执行/提交

FlashAttention
= Attention 这个 operator 本身怎么计算
```

---

## 25.2 CompilationMode

vLLM 0.26 有：

```text
NONE
STOCK_TORCH_COMPILE
DYNAMO_TRACE_ONCE
VLLM_COMPILE
```

其中：

```text
NONE
= fully eager PyTorch
```

---

## 25.3 CUDAGraphMode

有：

```text
NONE
PIECEWISE
FULL
FULL_DECODE_ONLY
FULL_AND_PIECEWISE
```

CUDA Graph 逻辑与 compilation 基本是正交维度。

所以可以专门构造：

```text
Compilation NONE
+
CUDA Graph FULL
```

来隔离 CUDA Graph 的效果。

这正是后面 A/B 的设计。

---

# 26. E6.6-F：CUDA Graph Controlled A/B/C 设计

## Case A：Eager baseline

```text
CompilationMode.NONE
CUDAGraphMode.NONE
```

回答：

> 当前 fully eager 路径有多慢？

---

## Case B：Pure CUDA Graph

```text
CompilationMode.NONE
CUDAGraphMode.FULL
```

回答：

> 不引入 compile，只打开 CUDA Graph，能消掉多少 overhead？

A→B 是最关键的机制实验。

---

## Case C：vLLM default optimized path

```text
VLLM_COMPILE
+
FULL_AND_PIECEWISE
```

回答：

> production-style 默认优化路径相对 eager 能到哪里？

但 C 不能单独用来归因 CUDA Graph，因为它同时改变 compilation。

---

# 27. E6.6-F 性能实验怎么测

固定：

```text
batch_size = 1
max_tokens = 64
warmup = 2
repeats = 7
async_scheduling = ON
```

每个 case：

```text
先 warmup
↓
重复 7 次
↓
记录 elapsed / output tokens / TPS
↓
取 median
```

为什么用 median：

```text
compile/warmup/cold path
OS scheduling
偶发 runtime jitter
```

都可能产生 outlier。

median 比 mean 更适合这一轮 steady latency 对比。

---

# 28. E6.6-F 性能结果

| Case | Compile | CUDA Graph | Median elapsed | TPS | vs Eager |
|---|---|---|---:|---:|---:|
| eager | NONE | NONE | 1265.423 ms | 50.58 | 1.000× |
| cg_full | NONE | FULL | 159.457 ms | 401.36 | **7.936×** |
| default | VLLM_COMPILE | FULL_AND_PIECEWISE | 138.398 ms | 462.43 | **9.143×** |

A→B：

```text
1265.423ms → 159.457ms
latency下降 ≈ 87.4%
TPS 50.58 → 401.36
≈ 7.94×
```

B→C：

```text
401.36 → 462.43 tok/s
≈ +15.2%
```

说明：

```text
CUDA Graph 解决了最大的一块 host submission overhead

VLLM_COMPILE / Inductor / fusion / specialization
在此基础上继续优化
```

---

# 29. 仅看 TPS 还不够，所以再做 Nsight mechanism A/B

性能快了 7.9×，但仍需要回答：

> 到底是不是因为 graph replay 消掉了 per-op launch？

于是 Nsight A/B 只比较：

```text
eager
vs
cg_full
```

并查看：

```text
MRV2_EXECUTE wall
CUDA API time
LaunchKernel count
GraphLaunch count
```

---

# 30. CUDA Graph mechanism 最终证据

结果：

| Case | MRV2 ms | CUDA API ms | LaunchKernel | GraphLaunch | Kernels start inside host MRV2 |
|---|---:|---:|---:|---:|---:|
| eager | 21.534 | 2.735 | 345 | 0 | 345 |
| cg_full | 1.017 | 0.148 | 6 | 1 | 0 |

Host side：

```text
21.534 ms → 1.017 ms
= 21.179×
```

这组数据是整个实验最重要的机制闭环。

---

# 31. 为什么 `Kernels in host = 0` 不表示 CUDA Graph 没有 kernel

这是必须解释清楚的地方。

`Kernels in host` 的定义只是：

```text
GPU kernel start timestamp
是否落在 MRV2_EXECUTE host NVTX range 内
```

CUDA Graph replay 是异步的：

```text
CPU:
MRV2_EXECUTE
  ↓
cudaGraphLaunch
  ↓
很快返回
MRV2_EXECUTE end

GPU:
           graph replay
           K1 K2 K3 ... Kn
```

所以：

```text
MRV2 host range 已结束
```

完全可能同时：

```text
GPU 正在继续跑 graph 里的 kernels
```

因此：

```text
0 kernels in host range
≠
0 GPU kernels executed
```

反而说明 host 已经不需要陪着 GPU 一个个 launch 了。

---

# 32. 为什么 MRV2 host 快 21×，端到端只快 7.94×

这是典型的 bottleneck migration。

Eager：

```text
host submission
≈ critical path
```

CUDA Graph 后：

```text
host submission 被大幅压缩
↓
GPU graph replay
sampling
scheduler / reconcile
其他 runtime coordination
```

开始成为新的 critical path。

所以不能期待：

```text
MRV2 21× faster
→ E2E 一定 21× faster
```

实际：

```text
Host MRV2：21.18×
E2E TPS：7.94×
```

这是完全合理的。

---

# 33. Controlled A/B 最终证明了什么

因为 A/B：

```text
CompilationMode 都是 NONE
模型一样
FlashAttention backend 一样
batch 一样
workload 一样
```

主要改变：

```text
CUDA Graph NONE → FULL
```

然后同时观察到：

```text
LaunchKernel 345 → 6
GraphLaunch 0 → 1
MRV2 host 21.534 → 1.017 ms
TPS 50.58 → 401.36
```

所以现在可以把之前的 hypothesis 升级为强结论：

> **当前 eager 小模型 decode 的主要性能损失来自 per-op host/framework dispatch and submission；CUDA Graph replay 大幅减少了这条路径。**

这比“看见很多 tiny kernels，所以猜 launch-bound”更强。

---

# 34. 为什么不需要继续拆 28 层 Transformer

在 Phase decomposition 后，本来还可以继续：

```text
MODEL_FORWARD
├── embedding
├── layer0
├── layer1
...
├── layer27
└── final norm
```

甚至：

```text
layer
├── Attention
├── MLP
└── Norm
```

但当前问题是：

> 21ms 为什么存在？

controlled CUDA Graph A/B 已经直接把：

```text
21.534ms → 1.017ms
```

并让 launch behavior 按预测发生变化。

因此继续拆层已经变成另一个研究问题：

```text
kernel-level / operator-level optimization
```

而不是当前 host bottleneck diagnosis 的必要条件。

这时候应该收尾，而不是为了“多分析”继续打点。

---

# 35. 这次实验对 FlashAttention 的最终判断

不能写：

```text
FlashAttention 是瓶颈
```

因为：

```text
Eager → CUDA Graph
Attention backend 没换
仍然是 FA2
```

但性能却发生数量级变化。

所以当前主要问题在 FA2 上层：

```text
Host execution / submission orchestration
```

FlashAttention 解决的是：

```text
Attention operator 本身怎样高效地读 KV / 算 Attention
```

CUDA Graph 解决的是：

```text
整串 operators / kernels 怎样低开销地反复提交
```

二者是正交优化层。

---

# 36. 这次实验对 Async Scheduling 的最终判断

当前可以写：

```text
async_scheduling=True
max_concurrent_batches=2
```

证明 runtime 有额外 in-flight transaction capacity。

但在：

```text
TP1
Qwen3-0.6B
Eager
batch1 / batch16
```

clean GPU 上没有观察到：

```text
prior-step GPU
overlap
next-step MRV2 execute
```

而现在又发现 eager CPU 每 step 本身需要约 21ms host model-forward，而 GPU active 只有约 2ms。

所以一个很合理的系统解释是：

```text
CPU submission path 太慢
GPU 很快完成 tiny kernels
CPU 无法成为“提前准备下一步去隐藏 GPU”的一方
```

但注意：

> 这个结论不能推广到 CUDA Graph / production optimized path。

因为 CUDA Graph 把 host MRV2 压到了约 1ms，CPU/GPU balance 已经发生巨大变化。

如果未来专门研究 async scheduling，应在 optimized execution 下重新实验。

---

# 37. Nsight GUI 以后应该怎么看

拿到新的 trace，建议固定按以下顺序。

## Step 1：先找 NVTX

先确定：

```text
SCHEDULE
MRV2_EXECUTE
MRV2_SAMPLE
```

不要先看 kernel 名。

---

## Step 2：展开 CUDA API

看：

```text
MRV2_EXECUTE 内
cudaLaunchKernel 是否密集
是否有 memcpy
是否有 synchronize
是否有 graph launch
```

---

## Step 3：再看 GPU stream

看：

```text
kernel 是否很短
kernel 之间 gap 多大
是否 GPU 连续 busy
是否存在 queue delay
```

---

## Step 4：通过 correlation 对齐

不要因为：

```text
某 kernel 看起来在 MRV2 附近
```

就直接归因。

需要：

```text
业务 range
→ Runtime API
→ correlationId
→ GPU kernel
```

---

## Step 5：如果发现大 gap，再问 gap 属于谁

例如本次：

```text
MRV2 22ms
GPU 2ms
CUDA API 3ms
```

说明问题应该继续向 host/framework narrowing，而不是立刻跑 NCU。

---

# 38. 为什么这一阶段不需要 Nsight Compute / NCU

Nsight Systems 解决的是：

```text
系统时间线
CPU/GPU overlap
launch behavior
kernel gap
线程 / process
业务 phase
```

Nsight Compute 解决的是：

```text
单个 kernel 内部
SM utilization
occupancy
memory throughput
warp stall
instruction mix
```

本次问题已经证明：

```text
单个 kernel 只有几微秒
系统的大头在 host submission
```

所以一开始去 NCU 深挖 FlashAttention kernel：

```text
方向错误
```

正确顺序是：

```text
nsys 找系统瓶颈
↓
如果最后锁到某个 kernel
↓
再用 ncu
```

这也是这次实验最重要的性能工程方法论之一。

---

# 39. 本次踩过的几个 profiling 误区

## 误区 1：有 report 文件 = capture 成功

错误。

必须检查：

```text
NVTX_EVENTS
CUPTI_ACTIVITY_KIND_RUNTIME
CUPTI_ACTIVITY_KIND_KERNEL
```

是否真正有数据。

---

## 误区 2：Host trace 顺序 = GPU 执行顺序

错误。

BRIDGE print 只能描述 host semantics。

---

## 误区 3：`non_block=True` = execute_model 立刻返回

错误。

它是 Executor/Future API 语义。

在 TP1 UniProc V2 下，实际 host execute path 仍可能同步运行很久。

---

## 误区 4：`future_done=True` = GPU 全部完成

错误。

Future 完成状态属于某一抽象层，不等价于所有 GPU work 和 downstream reconciliation 已完成。

---

## 误区 5：看到 next-step overlap 就说明 async pipeline 很好

不一定。

shared GPU contention 会让 kernel 被排队，从而产生假 overlap。

---

## 误区 6：CUDA API Summary 里 launch 占比高 = decode 就 launch-bound

不够。

全局 summary 混合了 init / warmup / inference。

必须限制到业务 NVTX window。

---

## 误区 7：GPU kernel 很短 = GPU 没问题

不一定。

短 kernel 可能意味着：

```text
launch-bound
低 occupancy
小 batch
memory latency
```

必须结合 system timeline。

本次通过 controlled CUDA Graph A/B 才最终确认是 host submission bottleneck。

---

# 40. 本次形成的可复用性能分析工作流

以后分析 vLLM / SGLang / nano-vLLM 都可以直接复用。

```text
Step 1
定义一个明确问题
例如：decode 为什么 20ms？

↓

Step 2
先画 Runtime call chain
知道 function / object / process 边界

↓

Step 3
加最小语义 instrumentation
NVTX，不要先把所有函数打满

↓

Step 4
nsys 采：
NVTX + CUDA Runtime + GPU Kernel

↓

Step 5
先看系统时间线
CPU wall / GPU active / kernel count / gap

↓

Step 6
做 coarse phase decomposition
找最大 phase

↓

Step 7
只继续拆最大 phase

↓

Step 8
形成 bottleneck hypothesis

↓

Step 9
设计 controlled intervention
只改一个主要变量

↓

Step 10
同时检查：
性能变化 + mechanism变化

↓

Step 11
如果 hypothesis 被验证，停止继续无意义拆分

↓

Step 12
如果最后锁到具体 kernel，再上 NCU
```

---

# 41. 这次实验如何体现“不是看图猜瓶颈”

整个过程其实经历了多次假设修正。

### 最早可能的解释

```text
Async queue=2
所以可能有 GPU(N) / CPU(N+1) overlap
```

硬件 trace：

```text
没有稳定 overlap
```

假设被否。

---

### 后来可能的解释

```text
MRV2 22ms
是不是 Scheduler / prepare_attn 很重？
```

Phase NVTX：

```text
Scheduler ~0.1ms
prepare_inputs ~0.6ms
attn addr/meta ~0.1ms
MODEL_FORWARD ~21ms
```

假设被进一步收紧。

---

### 再后来

```text
MODEL_FORWARD 21ms
是不是 GPU kernel 本身慢？
```

Nsight：

```text
GPU active ~1.9ms
median kernel ~3.4us
345 tiny kernels
```

不支持“一个大 kernel 很慢”。

---

### 最后假设

```text
Eager per-op host/framework submission overhead
```

Controlled A/B：

```text
LaunchKernel 345 → 6
GraphLaunch 0 → 1
MRV2 21.534 → 1.017ms
TPS 7.94×
```

假设被验证。

这才是完整的 performance diagnosis。

---

# 42. 实验脚本 / 产物索引

## E6 Runtime / Paged KV

```text
04-experiments/vllm-bridge/scripts/e6_paged_kv_attention.py
```

主链笔记：

```text
10-vLLM-E6日志驱动的完整Runtime调用链总桥接-v2.md
```

---

## E6.5 Nsight capture

关键过程脚本包括：

```text
04-experiments/vllm-bridge/raw/e6.5-nsys/e6.5-p2b_run.sh
04-experiments/vllm-bridge/raw/e6.5-nsys/e6.5-p2d_capture_2024.sh
04-experiments/vllm-bridge/raw/e6.5-nsys/e6.5-p2e_force_spawn_capture.sh
04-experiments/vllm-bridge/raw/e6.5-nsys/e6.5-clean-rerun.sh
```

成功的 tracing chain：

```text
NVTX
→ CUDA Runtime
→ GPU Kernel
```

---

## E6.6 Async / Host Root Cause

```text
04-experiments/vllm-bridge/raw/e6.6-nsys/e6.6-final-two.sh
```

用于：

```text
batch16 async overlap
host root-cause 尝试
```

---

## E6.6 Phase patch

```text
04-experiments/vllm-bridge/raw/e6.6-nsys/e6.6-d_patch_phase_nvtx.py
```

插入：

```text
BATCH_DISPATCH
PREPARE_INPUTS
PREPARE_ATTN_ADDR
PREPARE_ATTN_META
MODEL_STATE_INPUTS
MODEL_FORWARD
```

---

## E6.6 Batch compare

```text
04-experiments/vllm-bridge/raw/e6.6-nsys/e6.6-e_phase_batch_compare.sh
```

输出：

```text
04-experiments/vllm-bridge/raw/e6.6-nsys/e-phase-batch-compare/
├── e6.6-e-phase-batch-compare.log
├── phase-summary.txt
├── phase-summary.tsv
├── batch1-async-on.nsys-rep
├── batch1-async-on.sqlite
├── batch16-async-on.nsys-rep
└── batch16-async-on.sqlite
```

---

## E6.6 CUDA Graph A/B/C

```text
04-experiments/vllm-bridge/raw/e6.6-nsys/e6.6-f_cudagraph_ab.sh
```

关键结果：

```text
perf/summary.txt
nsys/summary.txt
```

---

# 43. 最终数据总表

## Eager phase decomposition

| 指标 | Batch1 | Batch16 |
|---|---:|---:|
| MRV2_EXECUTE | 22.284 ms | 23.170 ms |
| CUDA API | 2.769 ms | 3.205 ms |
| GPU active | 1.865 ms | 2.634 ms |
| kernel count | 345 | 485 |
| median kernel | 3.360 us | 3.520 us |
| MODEL_FORWARD | 21.0496 ms | 22.1161 ms |
| MODEL_FORWARD / MRV2 | 94.46% | 95.45% |
| PREPARE_INPUTS | 0.5780 ms | 0.4841 ms |
| PREPARE_ATTN_ADDR | 0.1329 ms | 0.1135 ms |
| PREPARE_ATTN_META | 0.1130 ms | 0.1032 ms |
| unattributed | 0.385 ms | 0.331 ms |

---

## CUDA Graph performance

| Case | Compile | CUDA Graph | Median elapsed | TPS | Speedup |
|---|---|---|---:|---:|---:|
| eager | NONE | NONE | 1265.423 ms | 50.58 | 1.000× |
| cg_full | NONE | FULL | 159.457 ms | 401.36 | 7.936× |
| default | VLLM_COMPILE | FULL_AND_PIECEWISE | 138.398 ms | 462.43 | 9.143× |

---

## CUDA Graph mechanism

| Case | MRV2 | CUDA API | LaunchKernel | GraphLaunch |
|---|---:|---:|---:|---:|
| eager | 21.534 ms | 2.735 ms | 345 | 0 |
| cg_full | 1.017 ms | 0.148 ms | 6 | 1 |

---

# 44. 最终瓶颈模型

## Eager batch1

```text
一个 decode iteration

Scheduler
~0.1ms
   │
   ▼
ModelRunner prepare
<1ms
   │
   ▼
MODEL_FORWARD
~21ms
   │
   ├── Python / module traversal
   ├── PyTorch dispatcher
   ├── C++ / custom op wrappers
   ├── per-op host preparation
   ├── ~345 CUDA kernel launches
   └── GPU active ~1.9ms
   │
   ▼
Sampling / reconcile
```

这是一种典型的：

> **host/framework submission dominated small-model eager decode**

---

## CUDA Graph 后

```text
MRV2_EXECUTE
~1ms
   │
   ├── 少量 host work
   └── 1 x GraphLaunch
         │
         ▼
GPU replay
K1 → K2 → K3 → ... → Kn
```

瓶颈从：

```text
逐 op host submission
```

迁移到：

```text
GPU graph execution
sampling
runtime coordination
其他剩余路径
```

---

# 45. 对以后 KV / Triton 项目的意义

这次实验最重要的不是“CUDA Graph 7.9×”。

真正重要的是你已经有了一套判断“该优化哪里”的方法。

以后做 Quantized Paged KV Cache 时：

如果看到：

```text
KV dequant kernel 500us
占 step 大头
```

才值得：

```text
NCU
Triton kernel optimization
memory access / occupancy / HBM 分析
```

如果看到：

```text
kernel 都很短
CPU gap 很大
```

则应该考虑：

```text
fusion
CUDA Graph
compile
减少 dispatch
```

如果看到：

```text
GPU 持续 busy
Attention / KV gather 占大头
```

再进一步考虑：

```text
KV layout
量化
HBM traffic
paged gather/dequant fusion
```

也就是说：

> **优化方案必须从 profiler 证据出发，而不是因为“某技术很热门”就先写某个 kernel。**

---

# 46. 30 秒恢复版

如果以后忘了整个实验，只记下面这条：

```text
E6
先用 BRIDGE trace 把：
Scheduler → KV blocks → block_table → slot_mapping → FA2 KV write/read
跑通

↓

发现 async host trace 不能证明 GPU overlap

↓

E6.5
用 Nsight 建：
NVTX → CUDA Runtime → correlationId → GPU kernel

↓

clean GPU 实测：
async queue=2
但没有稳定 prior-GPU / next-MRV2 overlap

↓

同时发现：
MRV2 ≈22ms
GPU active ≈2ms
345 tiny kernels

↓

E6.6 phase decomposition：
MODEL_FORWARD ≈21ms ≈95%

↓

假设：
eager per-op host/framework submission bottleneck

↓

Controlled CUDA Graph A/B：
Compilation NONE 保持不变
CUDAGraph NONE → FULL

↓

LaunchKernel 345 → 6
GraphLaunch 0 → 1
MRV2 21.534 → 1.017ms
TPS 50.58 → 401.36

↓

vLLM default：
VLLM_COMPILE + FULL_AND_PIECEWISE
462.43 tok/s

↓

结论：
当前小模型 eager decode 的主要瓶颈
是 host/framework submission，
不是 Scheduler，也不是 FA2 kernel 本身。
```

---

# 47. 最终实验结论

本阶段从 Runtime semantic trace 出发，先验证了 vLLM V1 / V2 Model Runner 下 Scheduler、Paged KV addressing、FlashAttention KV write/read 与 async batch queue 的完整执行链。随后使用 Nsight Systems 将业务 NVTX、CUDA Runtime API 和 GPU Kernel 通过 process identity 与 correlation ID 对齐，从而把“源码调用顺序”升级为“CPU/GPU hardware timeline”。

实验首先证明：`async_scheduling=True` 与 `max_concurrent_batches=2` 表示 runtime 具备额外 in-flight batch transaction 能力，但在当前 Qwen3-0.6B / A100 / TP1 / eager workload 下，即使 batch=16，也没有观察到 prior-step GPU execution 与 next-step ModelRunner execute 的稳定跨 step overlap。因此不能把 async throughput 变化简单归因于该 overlap 机制。

随后对 steady decode 进行时间拆解发现，batch1 的 `MRV2_EXECUTE` median 约为 22.284ms，而 GPU active union 只有约1.865ms；每次 execute 约产生345个 GPU kernels，median kernel latency 只有约3.36us。进一步通过 source-level NVTX 对 `GPUModelRunner.execute_model()` 做 coarse decomposition 后，`MODEL_FORWARD` 约21.050ms，占整个 MRV2 wall 的94.46%；batch16 下也达到95.45%。因此 Scheduler、input preparation、Paged-KV addressing 和 attention metadata 均不是20ms级主瓶颈，主要开销集中于 `self.model(**model_inputs)` 的 eager model-forward path。

最终通过保持 `CompilationMode.NONE` 不变、仅将 `CUDAGraphMode.NONE` 切换为 `FULL` 进行 controlled A/B，`MRV2_EXECUTE` 从21.534ms降至1.017ms，per-execute `LaunchKernel` 从约345次降至6次，并出现1次 `GraphLaunch`；端到端吞吐从50.58 tok/s提升到401.36 tok/s，约7.94×。进一步使用 vLLM 默认 `VLLM_COMPILE + FULL_AND_PIECEWISE` 后达到462.43 tok/s，约为 eager 的9.14×。

因此，本阶段最终形成的性能诊断是：

> **在 Qwen3-0.6B / A100 / batch1 的 eager decode 路径中，主要瓶颈是数百个 tiny GPU kernels 背后的 per-operator host/framework dispatch and submission，而不是 Scheduler、Paged-KV metadata 或 FlashAttention kernel 本身。CUDA Graph 通过 capture/replay 大幅减少重复 kernel submission，直接消除了这条主要 host bottleneck；VLLM_COMPILE 则在此基础上进一步通过编译、fusion 与 specialization 优化执行图。**

至此，E6.5 / E6.6 的“Nsight 系统级瓶颈定位”阶段可以正式收尾。
