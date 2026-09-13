# P1-M1-T1-S1：MRV2 `num_computed_tokens` 与 `effective_kv_len` 状态生命周期详解

> Project 1 — Physical KV Cache Reclamation for vLLM  
> Slice：`P1-M1-T1-S1 — MRV2 Effective-KV Persistent State Plumbing`  
> 目标：在进入 S2（logical position / physical cache position 拆分）之前，把 MRV2 的 logical progress 状态体系、CPU/GPU 双时钟、`RequestState` 生命周期和 S1 新增的 physical progress state 彻底梳理清楚。

> v2 补充：加入 Scheduler/Worker/GPU 三执行实体、`update_from_output()` 闭环、Worker CPU `_np` 的真实位置、RPC FIFO / GPU stream ordering，以及两个 in-flight batch 下 `_np` 可以领先 GPU 的完整时序分析。

---

## 0. 本笔记解决什么问题

M1 的核心不是“加一个变量”，而是为 P1 建立下面这个新的 runtime invariant：

```text
原生 vLLM 常规路径：
logical progress == physical KV progress

P1 reclaim 之后：
logical progress != physical KV progress
```

因此我们必须先理解原生 MRV2 里的 `num_computed_tokens`：

1. 谁拥有它？
2. Scheduler 侧和 Worker 侧各自有哪些版本？
3. Scheduler 为什么会“提前”推进？
4. Worker 为什么又有 CPU `_np` 和 GPU persistent state？
5. `InputBatch` 在哪里创建、谁调用它、它是不是 owner？
6. forward 前谁读取 `num_computed_tokens`？
7. forward 后谁真正推进 GPU execution state？
8. `post_update()` 和 `model_state.postprocess_state()` 有什么区别？
9. request remove / req_idx reuse 时为什么不需要清零？
10. 为什么 P1 的 `effective_kv_len` 应该跟 `post_update` 的 delta 走，而不能直接从 Scheduler logical state 同步？

本笔记把这些问题组织成一条完整链路，而不是零散源码摘录。

---

# 1. 先建立最重要的心智模型：这里其实有“三个状态域”

对于同一个 request，至少要区分：

```text
┌──────────────────────────────────────────────────────┐
│ 1. Scheduler CPU control-plane state                 │
│                                                      │
│ Request.num_computed_tokens                          │
│                                                      │
│ 含义：Scheduler 对 logical progress 的控制面视图      │
│ 特点：schedule 后可以 optimistic advance             │
└──────────────────────────────────────────────────────┘
                         │
                         │ SchedulerOutput
                         ▼
┌──────────────────────────────────────────────────────┐
│ 2. Worker CPU mirror                                 │
│                                                      │
│ RequestState.num_computed_tokens_np                  │
│                                                      │
│ 含义：Worker CPU 侧 logical progress mirror           │
│ 用途：CPU batch preparation / upper-bound bookkeeping│
│ 特点：不是 GPU→CPU copy                              │
└──────────────────────────────────────────────────────┘
                         │
                         │ 与 GPU state 语义同属 logical
                         │ 但更新时序不同
                         ▼
┌──────────────────────────────────────────────────────┐
│ 3. Worker GPU execution state                        │
│                                                      │
│ RequestState.num_computed_tokens.gpu                 │
│                                                      │
│ 含义：GPU execution-side persistent logical progress │
│ forward 前读取；post_update 后按 actual delta 推进    │
└──────────────────────────────────────────────────────┘
```

P1 S1 新增第四个、但语义不同的状态：

```text
┌──────────────────────────────────────────────────────┐
│ Worker GPU physical-KV execution state               │
│                                                      │
│ RequestState.effective_kv_len.gpu                    │
│                                                      │
│ 含义：当前仍有效、可被后续 attention/append 使用的     │
│       physical KV token 数                           │
│                                                      │
│ 普通执行：+ actual computed_delta                    │
│ reclaim：未来允许独立下降                            │
└──────────────────────────────────────────────────────┘
```

最关键的区别：

```text
num_computed_tokens
    = logical / model progression

effective_kv_len
    = physical KV occupancy / execution view
```

S1 的任务只是让第四个状态“活起来”，还没有让它成为 slot mapping 或 attention consumer。

---

# 2. 固定源码基线与本 Slice 范围

本阶段阅读和实现基于：

```text
vLLM: v0.26.0 pinned source
commit: 568afb3a13806beb53bb2e6bd518269357b237c0
Model Runner: MRV2
Core scope:
  TP=1
  PP=1
  DP=1
  speculative decoding OFF
  async scheduling OFF
  CUDA Graph OFF（Core 阶段）
```

S1 runtime patch 只涉及：

```text
vllm/v1/worker/gpu/states.py
vllm/v1/worker/gpu/input_batch.py
vllm/v1/worker/gpu/model_runner.py
```

测试：

```text
tests/v1/worker/test_gpu_effective_kv_state.py
```

S1 明确不修改：

```text
Scheduler
KVCacheManager
BlockPool
BlockTables / slot mapping
prepare_pos_seq_lens semantic
attention metadata
FlashAttention backend
reclaim policy / reclaim transaction
worker→scheduler ACK
```

这很重要：S1 是 **state foundation**，不是 reclamation 功能完成。


> **关于本笔记中的 async 章节**  
> 当前 P1 Core 配置仍然把 async scheduling 关闭；但为了理解 vLLM 为什么同时维护 Scheduler CPU、Worker CPU `_np`、Worker GPU persistent state，本笔记会继续分析 upstream MRV2 的 async / multi-in-flight 机制。  
> 这些内容属于 **源码语义学习和 V3 hardening 前置知识**，不是说 S1 已经支持 async reclamation。

---

# 3. 源码地图：每个对象在哪个文件、属于谁

| 层级 | 文件 | 类 / 函数 | 角色 |
|---|---|---|---|
| Scheduler request truth | `vllm/v1/request.py` | `Request` | Scheduler 侧 request logical state |
| Scheduling | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule()` | 决定本轮每个 request 计算多少 token |
| Scheduler optimistic advance | 同上 | `Scheduler._update_after_schedule()` | 在 schedule 后先推进 Scheduler logical progress |
| Scheduler result correction | 同上 | `Scheduler.update_from_output()` | 根据实际 output / spec rejection 修正 control-plane state |
| Scheduler→Worker transport | `vllm/v1/core/sched/output.py` | `NewRequestData` | 新 request 的完整/初始 worker 数据 |
| Scheduler→Worker transport | 同上 | `CachedRequestData` | 已存在 request 的增量数据 |
| Scheduler→Worker transport | 同上 | `SchedulerOutput` | 每个 step 的总调度输出 |
| Worker persistent state | `vllm/v1/worker/gpu/states.py` | `RequestState` | MRV2 Worker request-lifetime state owner |
| CPU→GPU state plumbing | `vllm/v1/worker/gpu/buffer_utils.py` | `StagedWriteTensor` | GPU persistent tensor + CPU staged sparse writes |
| CPU source-of-truth / UVA | 同上 | `UvaBackedTensor` | CPU source-of-truth + UVA view，语义不同于 StagedWriteTensor |
| Worker orchestration | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner` | add/update/prepare/forward/sample/postprocess 主链 |
| Per-step batch | `vllm/v1/worker/gpu/input_batch.py` | `InputBatch` | 一轮 execution 的 ephemeral batch view，不是 request owner |
| Input preparation kernel | 同上 | `prepare_prefill_inputs()` | 根据 logical progress 取本轮 prefill token |
| Position/seq kernel | 同上 | `prepare_pos_seq_lens()` | 根据 `num_computed_tokens.gpu` 生成 positions / seq_lens |
| Post-step state kernel | 同上 | `post_update()` / `_post_update_kernel` | GPU 侧本轮通用 request state commit |
| Attention/block addressing | `vllm/v1/worker/gpu/block_table.py` | `BlockTables.compute_slot_mappings()` | 当前用 `input_batch.positions` 做 KV physical slot addressing |
| Model-specific state hook | `vllm/v1/worker/gpu/model_states/interface.py` | `ModelState.postprocess_state()` | 特殊模型在通用 post_update 后额外处理模型状态 |
| Mamba override | `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | `MambaHybridModelState.postprocess_state()` | Mamba recurrent/accepted-token state 收尾 |

这张表建议作为后续 M1/M2 源码阅读的导航图。

---

# 4. `num_computed_tokens` 到底表示什么

Scheduler 源码对它的定义很关键：Scheduler 没有简单地把 request 分成传统“prefill request / decode request”两个永久类别，而是统一考虑：

```text
request 当前已经 computed 多少 token？
request 当前总共有多少 token 需要追上？
```

概念上：

```text
num_new_tokens
≈ target_token_count - num_computed_tokens
```

所以 `num_computed_tokens` 是整个 unified scheduling 模型里的“进度坐标”。

例如：

```text
prompt = 128 tokens
num_computed_tokens = 64

→ 还处于 prefill progress
```

再比如：

```text
prompt = 128
已经生成 10 个 output token
num_computed_tokens = 137

→ 模型已经计算到 logical position 137
```

注意它描述的是：

> **模型逻辑上已经处理过多少 token。**

它不是：

> “当前还保留了多少 KV token”。

原生系统中这两个数字通常相等，所以这个差异长期没有暴露；P1 reclaim 恰好要打破这个等式。

---

# 5. Scheduler CPU 侧：为什么你以前总看到它“提前更新”

这是这轮学习里最重要的源码澄清之一。

## 5.1 `Scheduler.schedule()` 先按当前 logical state 构造本轮输出

文件：

```text
vllm/v1/core/sched/scheduler.py
```

类：

```text
Scheduler
```

在 `schedule()` 中，Scheduler 根据当前：

```text
Request.num_computed_tokens = L
```

计算本轮：

```text
num_scheduled_tokens = q
```

然后构造：

```text
NewRequestData
或
CachedRequestData
```

其中携带的 `num_computed_tokens` 仍然是本轮执行**开始位置**附近的 logical state。

关键顺序是：

```text
Scheduler.schedule()
    │
    ├─ 根据 Request.num_computed_tokens=L 决定 q
    │
    ├─ _make_cached_request_data() / NewRequestData.from_request()
    │      └─ 把 L 写进 SchedulerOutput payload
    │
    ├─ 构造 SchedulerOutput
    │
    └─ _update_after_schedule(scheduler_output)
           └─ Scheduler 内部 Request.num_computed_tokens += q
```

也就是说：

```text
SchedulerOutput 发给 Worker 的本轮 start state：L

Scheduler 自己返回 schedule() 前：
内部 logical state 已 optimistic 变成 L+q
```

这就是你以前看到“为什么 Scheduler 好像执行前就把 num_computed 更新了”的根源。

---

## 5.2 `_update_after_schedule()` 为什么要这么做

源码注释直接说明了三个目的：

1. 当前 `SchedulerOutput` 仍需要原来的 scheduled/start 信息来准备本轮输入；
2. Scheduler 先推进 logical progress 后，可以在 async / pipeline 场景更快地继续 schedule 后续 step；
3. 如果 speculative token 后来被 reject，再在 `update_from_output()` 中回调修正。

核心概念：

```text
Scheduler 的 Request.num_computed_tokens
不是“GPU 已经物理完成到这里”的硬同步计数器。

它可以是：
control-plane optimistic logical progress。
```

举例：

```text
本轮开始：
Scheduler Request logical = 128

schedule 1 token：
SchedulerOutput 携带 start logical = 128
num_scheduled_tokens = 1

随后 _update_after_schedule：
Scheduler internal logical = 129

此时 GPU 甚至可能还没执行这个 token。
```

在同步 Core 下，这个窗口通常很短；在 async scheduling 下，这个差异非常重要。

---

## 5.3 `update_from_output()`：为什么还要“修正”

文件仍然是：

```text
vllm/v1/core/sched/scheduler.py
```

函数：

```text
Scheduler.update_from_output()
```

如果 speculative decoding 中：

```text
scheduled draft = 4
accepted = 2
rejected = 2
```

Scheduler 在 schedule 时可能已经 optimistic 按 scheduled count 推进；output 回来后会：

```text
Request.num_computed_tokens -= num_rejected
```

因此 Scheduler CPU 侧是：

```text
schedule-time optimistic advance
          +
output-time correction
```

而不是简单等待 GPU 完成后再同步一个数字回来。

这也是理解 `_np` 的关键背景。

---

# 6. SchedulerOutput：logical state 怎么送进 Worker

文件：

```text
vllm/v1/core/sched/output.py
```

## 6.1 新 request：`NewRequestData`

`NewRequestData` 里包含：

```text
req_id
prompt_token_ids
prefill_token_ids
block_ids
num_computed_tokens
sampling_params
...
```

`NewRequestData.from_request()` 直接从 Scheduler-side `Request` 取：

```text
request.num_computed_tokens
```

于是链路是：

```text
Scheduler Request.num_computed_tokens
            ↓
NewRequestData.num_computed_tokens
            ↓
GPUModelRunner.add_requests()
            ↓
RequestState.add_request()
```

---

## 6.2 已缓存 request：`CachedRequestData`

对于已经存在于 Worker 的 request，没必要每轮重发所有 prompt/token state。

所以 Scheduler 使用：

```text
CachedRequestData
```

其中包含：

```text
req_ids
new_block_ids
num_computed_tokens
...
```

然后 Worker：

```text
GPUModelRunner.update_requests()
```

读取这份 payload。

这是典型的：

```text
persistent worker cache
+
per-step diff transport
```

---

# 7. Worker 侧真正的 persistent owner：`RequestState`

文件：

```text
vllm/v1/worker/gpu/states.py
```

类：

```python
class RequestState:
```

它负责：

```text
req_id ↔ req_idx 映射
all_token_ids
prompt_len / prefill_len / total_len
num_computed_tokens
last_sampled_tokens
...
```

S1 后又增加：

```text
effective_kv_len
```

关键点：

> `RequestState` 是 request-lifetime state；`InputBatch` 只是 step-lifetime view。

不要把这两个混在一起。

---

# 8. 为什么 `num_computed_tokens` 有 GPU 和 NumPy 两套状态

当前：

```python
self.num_computed_tokens = StagedWriteTensor(...)
self.num_computed_tokens_np = np.zeros(...)
```

它们不是简单的“GPU tensor + 同步备份”。

## 8.1 `num_computed_tokens.gpu`

语义：

```text
Worker GPU execution-side persistent logical progress
```

主要特点：

- 长期驻留 GPU；
- forward/input preparation kernel 可以直接读取；
- `post_update` 可以在 GPU 上直接推进；
- 不需要每个 step CPU→GPU 重写整个 state。

---

## 8.2 `num_computed_tokens_np`

语义：

```text
Worker CPU-side optimistic logical mirror / upper bound
```

重要纠错：

```text
错误理解：
num_computed_tokens_np = GPU → CPU async copy

正确：
Scheduler CPU logical state
        ↓ SchedulerOutput
Worker.update_requests()
        ↓
num_computed_tokens_np
```

也就是说：

> `_np` 主要不是从 GPU 抄回来，而是 Worker CPU 从 Scheduler control-plane payload 获得。

为什么这么设计？

因为 CPU batch preparation 也需要 sequence progress，但如果每轮都：

```text
GPU num_computed_tokens
        ↓ D2H
CPU wait
```

CPU 就会产生对 GPU completion 的依赖，破坏 CPU/GPU overlap。

于是系统选择：

```text
Scheduler 自己维护 logical progress
        ↓
把 logical state 给 Worker CPU
        ↓
CPU 不需要等 GPU D2H
```

---


# 8A. 必须先拆清楚：Worker 不是 GPU，Worker 自己也有 CPU

这一点是本轮学习中最容易漏掉的系统结构。

很多时候我们口头说：

```text
Scheduler
    ↓
Worker
    ↓
GPU
```

容易让人下意识把：

```text
Worker == GPU
```

这是错误的。

更准确的执行实体是：

```text
┌──────────────────────────────────────────────┐
│ ① EngineCore / Scheduler CPU Process         │
│                                              │
│ Scheduler.schedule()                         │
│ Scheduler._update_after_schedule()           │
│ Scheduler.update_from_output()               │
└──────────────────────┬───────────────────────┘
                       │
                       │ SchedulerOutput / RPC
                       ▼
┌──────────────────────────────────────────────┐
│ ② Worker CPU Process                        │
│                                              │
│ GPUModelRunner                               │
│ RequestState                                 │
│ num_computed_tokens_np   ← NumPy / CPU       │
│ update_requests()                            │
│ prepare_inputs()                             │
│ Python dict / metadata / kernel launch       │
└──────────────────────┬───────────────────────┘
                       │
                       │ enqueue CUDA/Triton work
                       ▼
┌──────────────────────────────────────────────┐
│ ③ GPU                                       │
│                                              │
│ num_computed_tokens.gpu                     │
│ effective_kv_len.gpu                        │
│ prepare_pos_seq_lens kernel                 │
│ model forward                               │
│ sampling / post_update kernel               │
└──────────────────────────────────────────────┘
```

所以：

```text
num_computed_tokens_np
```

不是 Scheduler 进程里的变量。

它是：

> **Worker CPU 进程里的 NumPy mirror。**

而：

```text
num_computed_tokens.gpu
```

才是同一个 Worker 所控制的 GPU 上的 persistent execution state。

---

# 8B. 实际上存在“两层异步”

## 8B.1 Scheduler CPU ↔ Worker CPU

EngineCore 可以通过 non-blocking executor/RPC：

```text
schedule Batch1
    ↓
提交给 Worker

不一定立刻等最终结果
    ↓
schedule Batch2
    ↓
继续提交给 Worker
```

在 batch-queue / async scheduling 路径中，EngineCore 明确允许先填充多个 in-flight batch，再等待较早 batch 的结果。

所以：

```text
Scheduler CPU
```

可以领先：

```text
Worker CPU
```

---

## 8B.2 Worker CPU ↔ GPU

Worker CPU 运行：

```python
execute_model(...)
sample_tokens(...)
```

时，大量工作只是：

```text
launch / enqueue CUDA or Triton kernels
```

CPU 提交 kernel 后通常不需要等待 GPU 真正执行完成。

所以又可能出现：

```text
Worker CPU submission progress
>
GPU execution progress
```

因此 async 情况下应该把系统想成三个时钟：

```text
Scheduler CPU logical/control clock

        可以领先

Worker CPU submission / optimistic mirror clock

        可以领先

GPU execution / committed state clock
```

不是说三个值永远严格满足 `>`，而是它们**允许处在不同进度点**。

---

# 8C. “Scheduler 和 ModelRunner 是循环的，那每轮到底谁在自增？”

这个问题需要拆成三份状态分别回答。

| 状态 | 所在域 | 下一步怎么得到 |
|---|---|---|
| `Request.num_computed_tokens` | Scheduler CPU | `_update_after_schedule()` 按 scheduled token optimistic 自增；`update_from_output()` 再修正 |
| `RequestState.num_computed_tokens_np` | Worker CPU | 不靠自己长期 `+=`；每轮 cached request 主要由 `SchedulerOutput` 刷新 |
| `RequestState.num_computed_tokens.gpu` | Worker GPU | persistent；不由每轮 `update_requests()` 重写，而由 `post_update()` 按 actual delta 自增 |

所以“ModelRunner 每轮输入来自 SchedulerOutput，还是自己自增？”不能只回答一个“是/否”。

正确答案是：

```text
Worker CPU control view：
来自 SchedulerOutput

Worker GPU execution progress：
persistent，自己通过 post_update 自增

Scheduler logical control state：
自己通过 _update_after_schedule optimistic 自增
并在 update_from_output 修正
```

---

# 8D. 为什么 SchedulerOutput 里是旧的 L，而 Scheduler 自己已经变成 L+q

假设 Step N 开始：

```text
Scheduler Request.num_computed_tokens = 128
```

本轮决定：

```text
q = 1
```

Scheduler 的顺序是：

```text
1. 构造本轮 SchedulerOutput
   start logical = 128
   num_scheduled_tokens = 1

2. 再执行 _update_after_schedule()

3. Scheduler 内部：
   128 → 129
```

于是同一时刻可以出现：

```text
SchedulerOutput snapshot = 128
Scheduler current state   = 129
```

这不是矛盾。

SchedulerOutput 描述：

> **“这一轮从哪里开始执行”**

Scheduler internal state 描述：

> **“从 control plane 看，我已经把这一轮 token 放进 in-flight 了”**

下一轮 Scheduler 才会基于 129 构造新的 start snapshot。

---

# 8E. `update_from_output()` 在整个闭环的哪里

文件：

```text
vllm/v1/core/sched/scheduler.py
```

类：

```text
Scheduler
```

函数：

```python
update_from_output(
    scheduler_output,
    model_runner_output,
)
```

调用位置在：

```text
EngineCore.step()
```

概念链路：

```text
Scheduler.schedule()
        ↓
SchedulerOutput
        ↓
ModelExecutor / Worker
        ↓
execute_model()
        ↓
sample_tokens()
        ↓
ModelRunnerOutput
        ↓
Scheduler.update_from_output()
        ↓
下一轮 Scheduler.schedule()
```

因此它不是：

```text
GPU num_computed_tokens
    ↓ D2H
Scheduler.num_computed_tokens = GPU value
```

而是：

```text
Scheduler 已经知道自己 schedule 了什么
+
Worker 返回真正产生/接受/拒绝了什么
        ↓
Scheduler 用 execution result 修正自己的 optimistic state
```

例如 speculative decode：

```text
start = 128
scheduled = 4

Scheduler optimistic:
128 → 132

实际 reject 2

Worker GPU actual:
128 → 130

Scheduler.update_from_output():
132 - 2 → 130
```

最终 control-plane state 与 execution semantics 重新收敛。

---

# 8F. 最关键的问题：Batch1 GPU 还没执行完，Batch2 会不会先进入 Worker 更新 `_np`？

**会。**

而且这正是 `_np` 被叫做 optimistic mirror 的一个最直观例子。

假设同一个 request A：

```text
初始：

Scheduler CPU        = 128
Worker CPU _np       = 128
Worker GPU persistent= 128
```

## Batch1

Scheduler：

```text
Batch1 start = 128
q = 1

SchedulerOutput(B1):
num_computed_tokens = 128

随后 Scheduler optimistic:
128 → 129
```

Worker CPU 收到 Batch1：

```text
update_requests(B1)
→ _np = 128
```

然后 Worker CPU 提交：

```text
B1 prepare
B1 forward
B1 sample
B1 post_update
```

但是：

> **“已经提交 B1 post_update” 不等于 “GPU 已经执行完 B1 post_update”。**

GPU 此刻可能还正在：

```text
B1 forward
```

---

## Batch2 此时仍然可以被 Scheduler 构造

Scheduler 当前已经是：

```text
129
```

于是 Batch2：

```text
SchedulerOutput(B2):
start logical = 129
q = 1
```

Scheduler 随后又 optimistic：

```text
129 → 130
```

Batch2 进入 Worker CPU 后：

```text
update_requests(B2)
→ num_computed_tokens_np = 129
```

此时完全可能出现：

```text
Scheduler CPU         = 130
Worker CPU _np        = 129
Worker GPU physical execution 仍可能 = 128
```

这个状态是**合法的**。

因为 `_np=129` 的含义不是：

> “GPU 现在已经执行完成 129 个 token。”

而是：

> “对于 Batch2 的 CPU-side logical batch view，它应该从 logical 129 开始。”

---

# 8G. Batch2 已经把 `_np` 更新成 129，为什么 GPU 不会错误地读到旧 128？

因为：

> **Batch2 到达 Worker CPU，不等于 Batch2 已经在 GPU 上开始执行。**

这是本轮最重要的一句话。

Worker CPU 处理的是 Python / metadata / kernel submission。

GPU 实际执行的是已经排进 CUDA stream 的工作。

正常依赖顺序可以抽象成：

```text
Worker CPU submission order

B1 execute_model()
        ↓
B1 sample_tokens()
        ↓
B1 post_update kernel 已 enqueue
        ↓
B2 execute_model()
        ↓
B2 prepare_pos_seq_lens kernel enqueue
```

对应 GPU stream：

```text
GPU execution order

B1 prepare
    ↓
B1 forward
    ↓
B1 sample
    ↓
B1 post_update
    │
    │ num_computed_tokens.gpu:
    │ 128 → 129
    ▼
B2 prepare_pos_seq_lens
    │
    │ read num_computed_tokens.gpu
    ▼
   129
    ↓
B2 forward
```

因此，在 Worker CPU 正在处理 Batch2 时，GPU tensor **此刻物理上**可能仍然是 128。

但真正等 GPU 执行到：

```text
B2 prepare_pos_seq_lens
```

之前，它会先执行：

```text
B1 post_update
```

于是 B2 的 GPU-side read 得到 129。

correctness 依赖的是：

```text
B1 post_update
happens-before
B2 GPU state read
```

而不是：

```text
CPU 必须等待 B1 GPU 完全执行结束
再允许 Batch2 进入 Worker
```

---

# 8H. 为什么 Worker 侧不会把 B1/B2 的调用顺序乱掉

在 multiprocess executor 中，EngineCore/Executor 把 RPC 按顺序 enqueue 给 Worker。

Worker process 有一个主 busy loop：

```text
dequeue RPC
    ↓
调用对应 worker method
    ↓
处理下一个 RPC
```

所以普通 generation path 可以理解成：

```text
Worker CPU method order

execute_model(B1)
        ↓
sample_tokens(B1)
        ↓
execute_model(B2)
        ↓
sample_tokens(B2)
```

Worker CPU 方法本身顺序执行，但其中 launch 的 GPU work 是异步的。

因此要区分：

```text
Worker CPU method serial ordering
```

和：

```text
GPU kernel asynchronous execution
```

两者同时成立，并不矛盾。

---

# 8I. 用一张并排时序图看两个 in-flight batch

```text
时间向下
────────────────────────────────────────────────────────────────────

Scheduler / EngineCore CPU      Worker CPU                 GPU

state=128                      _np=128                    gpu=128

schedule B1
snapshot(B1)=128
optimistic 128→129
        │
        ├──── RPC B1 ──────────► update_requests(B1)
                                 _np=128
                                 launch B1 prepare ───────► B1 prepare
                                 launch B1 forward ───────► B1 forward
                                 launch B1 sample ────────► 排队
                                 launch B1 post ──────────► 排队

schedule B2
snapshot(B2)=129
optimistic 129→130
        │
        ├──── RPC B2 ──────────► update_requests(B2)
                                 _np=129
                                 launch B2 prepare ───────► 排在 B1 post 后
                                 launch B2 forward ───────► 继续排队

                                                           B1 sample
                                                           ↓
                                                           B1 post_update
                                                           gpu:128→129
                                                           ↓
                                                           B2 prepare
                                                           read gpu=129
                                                           ↓
                                                           B2 forward
                                                           ↓
                                                           B2 post_update
                                                           gpu:129→130
```

所以某个中间瞬间：

```text
Scheduler = 130
Worker _np = 129
GPU = 128
```

完全正常。

最终随着 GPU执行和 output correction：

```text
Scheduler logical
Worker CPU mirror
Worker GPU execution state
```

会在相应 reconciliation boundary 上重新靠拢。

---

# 8J. 为什么 `_np` 可以领先，但不能被当成 exact GPU truth

`_np` 适合：

```text
CPU batch preparation
logical upper bound
prefill bookkeeping
CPU metadata
```

例如：

```text
seq_lens_cpu_upper_bound
=
num_computed_tokens_np
+
num_scheduled_tokens
```

它本来就是 upper-bound/control-plane 语义。

但如果某个操作需要：

```text
GPU 当前实际已经 commit 到哪里？
```

就不能简单拿 `_np` 当事实。

例如 execution-sensitive 的：

```text
prepare_pos_seq_lens
```

直接读取：

```text
num_computed_tokens.gpu
```

这就是为什么 Worker 同时保留 CPU mirror 与 GPU persistent state。

---

# 8K. 三份 state 的最终精确定义

| State | Owner / Memory | 语义 | 谁更新 | 可以暂时领先谁 |
|---|---|---|---|---|
| `Request.num_computed_tokens` | Scheduler CPU | control-plane logical progress，允许含 in-flight optimistic progress | `_update_after_schedule()`；`update_from_output()` 修正 | 可以领先 Worker/GPU |
| `RequestState.num_computed_tokens_np` | Worker CPU / NumPy | 当前 SchedulerOutput 对本 batch 的 logical CPU view / upper-bound basis | `update_requests()` | 可以领先 GPU actual |
| `RequestState.num_computed_tokens.gpu` | Worker GPU | persistent execution-side logical progress | `post_update()` actual delta | 不应靠 `_np` 强行覆盖 |

所以：

```text
三个 state 某一时刻数值不同
```

本身不是 bug。

真正需要检查的是：

```text
它们各自的 consumer 是否只使用了符合自己语义的 state
+
跨域依赖是否有明确 ordering/reconciliation
```

---

# 8L. 这件事为什么直接影响 P1 的 `effective_kv_len`

Logical progress 可以允许：

```text
Scheduler optimistic
Worker CPU optimistic mirror
Worker GPU actual execution
```

因为最终只是 logical progress reconciliation。

但 physical KV state 涉及真正的：

```text
block-table view
physical slot
GPU memory lifetime
BlockPool.free
cross-request reuse
```

如果 physical state 也随便增加一个 CPU optimistic mirror，就必须马上回答：

```text
CPU 说 physical=64 时
GPU 是否真的已经切换到64？

旧 block 是否还有 in-flight batch 引用？

什么时候可以 free？

另一个 request 什么时候可以 reuse？
```

这些问题比 logical scalar 更危险。

所以 S1 当前只定义：

```text
effective_kv_len.gpu
```

并把 async physical reclamation 留到后续 hardening，是有明确系统原因的。

对于 P1 Core，我们当前仍然保持：

```text
async OFF
```

这样 M2/M3 先把：

```text
physical view commit
→ ACK
→ scheduler ownership update
→ safe free
```

在单 in-flight 场景证明正确，再讨论多 batch lifetime fence。

---

# 8M. 本轮学习问题的最终闭环

我们实际经历了几次很典型的误解：

```text
Q1:
num_computed_tokens_np 是不是 GPU→CPU copy？

A:
不是。主要来自 SchedulerOutput，
存在于 Worker CPU。


Q2:
Scheduler 和 ModelRunner 每轮循环，
那 num_computed 到底是 Scheduler 给的还是自己加的？

A:
Scheduler CPU 自己 optimistic 加；
Worker CPU mirror 每轮从 SchedulerOutput 刷；
Worker GPU persistent state 由 post_update 自己加。


Q3:
update_from_output 在哪里？

A:
Scheduler.update_from_output()，
位于 Worker/ModelRunner output 回到 Scheduler 的闭环处，
负责依据实际 execution result 修正 optimistic state。


Q4:
Worker 不是 GPU 吗？为什么还有 np？

A:
Worker 是 CPU process + GPU execution orchestrator。
_np 是 Worker CPU 内存。


Q5:
Batch1 GPU 没执行完，
Batch2 会不会进入 Worker 把 _np 更新成129？

A:
会，而且允许。


Q6:
那 Batch2 不会在 GPU 上读到旧128吗？

A:
Batch2 的 CPU-side _np 可以先变129；
但真正 GPU-side prepare kernel
排在 B1 post_update 后，
所以 GPU执行到 B2 时读到129。
```

如果这六个问题都能自己解释清楚，就说明 MRV2 `num_computed_tokens` 的 CPU/GPU/async 生命周期已经真正掌握，而不是只记住函数名。

---


# 9. `StagedWriteTensor` 到底是什么

文件：

```text
vllm/v1/worker/gpu/buffer_utils.py
```

类：

```text
StagedWriteTensor
```

## 9.1 它不是“CPU/GPU 两份 Tensor”

更准确的结构：

```text
             CPU staging lists
       indices / starts / contents
                  │
                  │ apply_write()
                  ▼
        _apply_write_kernel (Triton)
                  │
                  ▼
       persistent GPU tensor (.gpu)
```

例如：

```python
stage_write_elem(3, 100)
stage_write_elem(7, 240)
stage_write_elem(9, 56)
```

此时只是 CPU 记录：

```text
indices  = [3, 7, 9]
contents = [100, 240, 56]
```

调用：

```python
apply_write()
```

才批量用 GPU kernel 写入对应 row。

所以你的早期理解：

> “很多小写入 launch 太贵，所以攒起来一起写。”

方向是对的，但更完整的说法是：

> **把零碎 CPU control-plane mutation staging 成 sparse diffs，再批量 apply 到 persistent GPU state。**

它既减少 tiny write/launch，也避免每轮重写整个 tensor。

---

## 9.2 为什么 `UvaBackedTensor` 又不同

同文件还有：

```text
UvaBackedTensor
```

它的语义是：

```text
CPU tensor = source of truth
GPU 通过 UVA view 访问
```

而 `StagedWriteTensor` 更像：

```text
GPU tensor = persistent execution storage
CPU 只负责 staged mutation metadata
```

所以看到 `.np`、`.gpu`、UVA 时不能只按“两个副本”理解，必须先问：

> **source of truth 是谁？更新边界是什么？consumer 在哪？**

---

# 10. 新 request 初始化：两条 logical state 怎么同时出生

文件：

```text
vllm/v1/worker/gpu/model_runner.py
```

类：

```text
GPUModelRunner
```

函数：

```text
add_requests()
```

链路：

```text
Scheduler
  ↓
NewRequestData.num_computed_tokens = L
  ↓
GPUModelRunner.add_requests()
  ↓
RequestState.add_request(..., num_computed_tokens=L)
```

`RequestState.add_request()` 中：

```text
CPU mirror：
num_computed_tokens_np[req_idx] = L

GPU persistent state：
num_computed_tokens.stage_write_elem(req_idx, L)
```

然后：

```text
GPUModelRunner.add_requests()
        ↓
RequestState.apply_staged_writes()
        ↓
StagedWriteTensor.apply_write()
        ↓
GPU num_computed_tokens[req_idx] = L
```

所以初始化完成后：

```text
Worker CPU logical = L
Worker GPU logical = L
```

S1 新增：

```text
Worker GPU physical effective = L
```

为什么初值也设为 L？

因为 request 刚进入 Worker 时，S1 还没有执行任何 P1 reclaim：

```text
initial logical == initial physical
```

---

# 11. 已存在 request：`update_requests()` 到底更新什么

文件：

```text
vllm/v1/worker/gpu/model_runner.py
```

类：

```text
GPUModelRunner
```

函数：

```python
def update_requests(self, scheduler_output):
```

关键逻辑：

```text
scheduled_cached_reqs.num_computed_tokens
        ↓
num_computed_tokens_np[req_index] = num_computed_tokens
```

注意：

```text
update_requests()
没有重新 stage_write num_computed_tokens.gpu
```

原因是：

```text
GPU persistent logical state
已经在上一轮 post_update 中自己推进
```

所以 cached request 的普通链路是：

```text
GPU logical：自己持续推进
CPU logical mirror：每轮由 SchedulerOutput 刷新
```

这避免了：

```text
Scheduler CPU → Worker CPU → 每轮再 H2D 重写同一个 GPU logical counter
```

---

# 12. `num_computed_prefill_tokens` 又是什么

`update_requests()` 还会：

```python
np.minimum(
    num_computed_tokens_np,
    prefill_len.np,
    out=num_computed_prefill_tokens,
)
```

它服务的是 CPU 侧 prefill bookkeeping。

概念：

```text
num_computed_prefill_tokens
= min(logical computed progress, prefill_len)
```

然后 `prepare_inputs()` 中：

```text
is_prefilling_np
=
num_computed_prefill_tokens_np < prefill_len_np
```

所以 CPU mirror 有明确用途，不是为了“备份 GPU tensor”。

---

# 13. `InputBatch` 是在哪里创建的？谁调用？

这是之前笔记没有体系化说明的地方。

## 13.1 `InputBatch` 定义

文件：

```text
vllm/v1/worker/gpu/input_batch.py
```

对象：

```python
@dataclass
class InputBatch:
```

它保存本轮 batch 的 execution view，例如：

```text
req_ids
idx_mapping / idx_mapping_np
num_scheduled_tokens
query_start_loc / query_start_loc_np
seq_lens
num_computed_tokens_np
positions
input_ids
logits_indices
...
```

它不是 persistent request owner。

生命周期更像：

```text
每个 execute_model step
       ↓
构造 InputBatch
       ↓
forward/sample 使用
       ↓
step 结束后丢弃
```

---

## 13.2 谁创建 `InputBatch`

`GPUModelRunner.prepare_inputs()` 最后：

```python
input_batch = InputBatch(...)
return input_batch
```

---

## 13.3 谁调用 `prepare_inputs()`

在：

```text
GPUModelRunner.execute_model()
```

真实 batch 路径：

```text
execute_model(scheduler_output)
       │
       ├─ update_pp_decode_requests()
       ├─ finish_requests()
       ├─ free_states()
       ├─ add_requests()
       ├─ update_requests()
       ├─ block_tables.apply_staged_writes()
       │
       ├─ prepare_inputs(scheduler_output, batch_desc)
       │       └─ 生成 InputBatch
       │
       ├─ prepare_attn(input_batch)
       │
       ├─ model_state.preprocess_state(...)
       │
       ├─ model_state.prepare_attn(...)
       │
       └─ model forward
```

因此：

> `InputBatch` 是 `GPUModelRunner.execute_model()` 为“这一轮 GPU execution”构造出来的临时数据视图。

---

# 14. `prepare_inputs()` 中 CPU 和 GPU 两条 `num_computed` 路径第一次明显分开

文件：

```text
vllm/v1/worker/gpu/model_runner.py
```

函数：

```text
GPUModelRunner.prepare_inputs()
```

## 14.1 GPU logical state 用来做真正的 execution input

例如 prefill：

```python
prepare_prefill_inputs(
    ...,
    self.req_states.num_computed_tokens.gpu,
)
```

然后 position / seq len：

```python
prepare_pos_seq_lens(
    idx_mapping,
    query_start_loc,
    self.req_states.num_computed_tokens.gpu,
    positions,
    seq_lens,
)
```

也就是：

```text
num_computed_tokens.gpu
        ↓
prepare_pos_seq_lens
        ├─ logical positions
        └─ seq_lens
```

---

## 14.2 CPU `_np` 用来算 CPU upper bound

同一个 `prepare_inputs()` 里：

```text
num_computed_tokens_np
        +
num_scheduled_tokens
        ↓
seq_lens_cpu_upper_bound_np
```

所以同一轮里面同时存在：

```text
GPU exact/current execution state
和
CPU scheduling-oriented upper bound
```

这正是为什么不能把两者理解为“强同步双副本”。

---

# 15. `prepare_pos_seq_lens()`：原始 coupling 是怎么形成的

文件：

```text
vllm/v1/worker/gpu/input_batch.py
```

函数：

```text
prepare_pos_seq_lens()
```

Triton kernel 的核心逻辑：

```text
num_computed = num_computed_tokens[req_state_idx]
query_len = query_end - query_start

seq_len = num_computed + query_len

每个 query token：
pos = num_computed + local_offset
```

于是：

```text
num_computed_tokens.gpu
        ↓
positions
        ↓
model_inputs["positions"]
        ↓
RoPE / model logical semantics
```

目前这部分对于 P1 reclaim 后仍然应该保持 logical。

例如：

```text
logical_num_computed = 128
physical effective KV = 64

下一 token 的模型 position：128
绝不能改成 64
```

---

# 16. 真正的问题：同一个 `positions` 又被拿去做 physical slot mapping

文件：

```text
vllm/v1/worker/gpu/model_runner.py
```

函数：

```text
GPUModelRunner.prepare_attn()
```

当前：

```python
slot_mappings = self.block_tables.compute_slot_mappings(
    input_batch.idx_mapping,
    input_batch.query_start_loc,
    input_batch.positions,
    ...
)
```

也就是：

```text
logical num_computed
      ↓
logical positions
      ├────────────→ model/RoPE
      │
      └────────────→ physical KV slot mapping
```

为什么原生 vLLM 没问题？

因为长期满足：

```text
logical position == physical KV append position
```

为什么 P1 会出问题？

reclaim 后：

```text
logical = 128
physical = 64
```

下一 token：

```text
model position 应该 = 128
KV physical append position 应该 = 64
```

于是一个 `positions` 已经无法同时承担两种语义。

这就是 S2 必须解决的 coupling。

---

# 17. forward 之后：真正的 GPU progress 在哪里 commit

MRV2 当前主链不是简单：

```text
execute_model() 一口气 forward+sample+post
```

而是：

```text
GPUModelRunner.execute_model()
        ↓
prepare inputs / attention
        ↓
model forward
        ↓
保存 ExecuteModelState
        ↓
return

GPUModelRunner.sample_tokens()
        ↓
sample()
        ↓
构造 AsyncOutput
        ↓
postprocess_sampled()
        ↓
post_update()
```

这点非常重要：

> `execute_model()` 负责 model forward；真正 sample 和 post-step request state commit 在 `sample_tokens()` 这条后续链路中完成。

---

# 18. `post_update()` 到底干什么

文件：

```text
vllm/v1/worker/gpu/input_batch.py
```

函数：

```text
post_update()
```

底层 Triton：

```text
_post_update_kernel
```

它是通用 request runtime bookkeeping，不是某个具体模型专属逻辑。

它会处理：

```text
last_sampled_tokens
all_token_ids append
total_len
output_bin_counts
num_computed_tokens
```

其中最关键：

```python
computed_delta = query_len - num_rejected
```

然后：

```text
num_computed_tokens += computed_delta
```

S1 后变成：

```text
                 computed_delta
                  /          \
                 /            \
                ▼              ▼
num_computed_tokens      effective_kv_len
     += delta                += delta
```

注意：

```text
共享的是 delta
不是 absolute value
```

---

# 19. 为什么 `computed_delta` 比 `query_len` 更准确

正常 decode：

```text
query_len = 1
num_rejected = 0
computed_delta = 1
```

spec decode 假设：

```text
query_len = 4
rejected = 2
computed_delta = 2
```

那么真正 commit 的 logical progress 是：

```text
+2
```

对 physical KV valid progress 来说，普通执行也应该：

```text
+2
```

所以 `post_update` 是一个天然的 execution commit boundary。

当前 P1 Core 虽然 spec OFF，但沿用 `computed_delta` 的语义不会破坏原有机制。

---

# 20. `post_update()` 和 `model_state.postprocess_state()` 的区别

在 `GPUModelRunner.postprocess_sampled()` 中：

```text
post_update(...)

然后：

model_state.postprocess_state(...)
```

这两者不是重复。

## 20.1 `post_update()`

层级：

```text
通用 Request runtime state
```

它不关心当前模型是：

```text
Qwen / Llama / Mamba / DeepSeek
```

---

## 20.2 `ModelState.postprocess_state()`

文件：

```text
vllm/v1/worker/gpu/model_states/interface.py
```

基类默认：

```text
no-op
```

它是一个 **model-specific hook**。

Mamba 在：

```text
vllm/v1/worker/gpu/model_states/mamba_hybrid.py
```

override 它，用于：

```text
num_accepted_tokens_gpu
Mamba recurrent / align state
```

而且 Mamba 源码明确依赖：

```text
num_computed_tokens 已经是 post-step advanced count
```

因此顺序是：

```text
sample
  ↓
post_update
  ↓
通用 request logical state 已经 commit
  ↓
model_state.postprocess_state
  ↓
模型专属状态依据新 logical progress 收尾
```

S1 只把 `effective_kv_len` 接入前者是合理的，因为它是通用 physical KV execution state，不是某个模型专属 state。

---

# 21. 一个完整 Step：从 `num_computed_tokens` 视角追一次

假设：

```text
Step N 开始 logical L = 128
本轮 q = 1
spec OFF
```

## 21.1 Scheduler CPU

```text
Request.num_computed_tokens = 128
        ↓
schedule()
        ↓
决定 num_scheduled_tokens=1
        ↓
SchedulerOutput 中携带本轮 start logical=128
        ↓
_update_after_schedule()
        ↓
Scheduler internal Request.num_computed_tokens = 129
```

注意这时 Scheduler 已经 optimistic 129，但 GPU 可以仍然是 128。

---

## 21.2 Worker 接收本轮 SchedulerOutput

cached request：

```text
GPUModelRunner.update_requests()
        ↓
num_computed_tokens_np = 128
```

GPU persistent state：

```text
num_computed_tokens.gpu = 128
```

没有被 `update_requests()` 重新覆盖。

---

## 21.3 Worker `prepare_inputs()`

GPU 路：

```text
num_computed_tokens.gpu=128
        ↓
prepare_pos_seq_lens
        ↓
positions=[128]
seq_len=129
```

CPU 路：

```text
num_computed_tokens_np=128
+
num_scheduled_tokens=1
        ↓
seq_lens_cpu_upper_bound=129
```

---

## 21.4 forward / sample / post

```text
model forward
   ↓
sample_tokens
   ↓
num_rejected=0
   ↓
post_update
   ↓
computed_delta=1
```

GPU：

```text
num_computed_tokens.gpu:
128 → 129
```

---

## 21.5 Scheduler output processing

如果没有 reject：

```text
Scheduler optimistic 129
保持 129
```

如果有 spec reject，则：

```text
update_from_output
↓
Scheduler logical 回调修正
```

---

# 22. 同一个 Step：从 S1 `effective_kv_len` 视角再追一次

S1 baseline 下：

```text
logical=128
physical=128
```

Scheduler 不携带：

```text
effective_kv_len
```

Worker 自己维护：

```text
RequestState.effective_kv_len.gpu=128
```

forward 后：

```text
computed_delta=1
```

于是：

```text
logical GPU:
128 → 129

physical GPU:
128 → 129
```

两者暂时一样，但只是因为还没 reclaim。

---

# 23. reclaim 后为什么绝不能再从 Scheduler logical absolute value 同步 physical

未来假设：

```text
logical = 128
physical = 64
```

这个状态意味着：

```text
模型历史上已经计算过 128 token
但 physical KV 中只保留 64 个有效 token
```

Scheduler 继续知道的是：

```text
Request.num_computed_tokens = 128
```

如果在 Worker `update_requests()` 中写：

```text
effective_kv_len = scheduler.num_computed_tokens
```

会发生：

```text
64 → 128
```

直接把 reclaim state 抹掉。

所以 physical state 只能：

```text
普通 execution：按 actual delta 增加
reclaim commit：按 physical mutation 独立下降
```

不能：

```text
每轮 physical = logical absolute value
```

这一句话是 S1 最重要的设计结论。

---

# 24. 为什么 S1 暂时没有 `effective_kv_len_np`

`num_computed_tokens_np` 有明确 CPU consumer：

```text
prefill bookkeeping
seq_lens_cpu_upper_bound
InputBatch CPU metadata
```

但 S1 的 physical state 当前真正面向：

```text
future cache_positions
future slot_mapping
future attention physical/effective metadata
```

这些主要是 GPU execution/data-plane consumer。

如果现在提前加入：

```text
effective_kv_len_np
```

马上会引入没有答案的问题：

```text
它是谁的 truth？
Worker committed physical value？
还是 Scheduler optimistic physical upper bound？
什么时候更新？
reclaim commit 前还是后？
ACK 前还是后？
```

当前没有 CPU consumer，就不应该提前制造第二份 physical state。

因此 S1 决策：

```text
RequestState.effective_kv_len
= GPU persistent StagedWriteTensor only
```

---

# 25. S1 实际修改了什么

## 25.1 `states.py`

新增：

```text
RequestState.effective_kv_len
```

类型：

```text
StagedWriteTensor[int32]
```

生命周期：

```text
add_request
  ↓
initial effective_kv_len = initial num_computed_tokens
  ↓
apply_staged_writes
  ↓
GPU persistent state ready
```

没有加入：

```text
effective_kv_len_np
```

---

## 25.2 `input_batch.py`

给：

```text
post_update / _post_update_kernel
```

增加 physical state pointer。

原来：

```text
computed_delta
   ↓
logical += delta
```

现在：

```text
computed_delta
   ├─ logical += delta
   └─ physical += delta
```

---

## 25.3 `model_runner.py`

`GPUModelRunner.postprocess_sampled()` 传入：

```text
self.req_states.effective_kv_len.gpu
```

但没有新增 consumer。

所以 S1 后：

```text
effective_kv_len
只“存在 + 正确推进”
还不参与 execution addressing
```

---

# 26. 为什么 remove 时不需要清零

`RequestState.remove_request()` 的核心是：

```text
删除 req_id ↔ req_idx mapping
把 req_idx 放回 free_indices
```

它不会保证：

```text
num_computed_tokens[req_idx] = 0
total_len[req_idx] = 0
all_token_ids row = 0
```

正确 invariant 是：

```text
旧 row 可以 stale
但新 request 拿到 req_idx 时
add_request 必须完整覆盖所有之后会读取的 state
```

S1 的 `effective_kv_len` 遵循同样规则。

例子：

```text
Request A
idx=0
physical=64

remove A
physical[0] 仍可能是 64

Request B reuse idx=0
initial num_computed=37

add_request + apply
physical[0] 必须覆盖为 37
```

这种设计比 remove 时到处 memset 更符合现有 MRV2 row-reuse 模型。

---

# 27. S1 测试文件整体在模拟什么

文件：

```text
tests/v1/worker/test_gpu_effective_kv_state.py
```

它不是 E2E，不跑：

```text
Scheduler → ModelRunner → Transformer → Attention → Sample
```

它只构造：

```text
RequestState
+
真实 post_update kernel
```

目的是精准验证 S1 state lifecycle。

---

# 28. `make_request_state()` 怎么理解

```python
def make_request_state() -> RequestState:
    return RequestState(
        max_num_reqs=1,
        max_model_len=256,
        max_num_batched_tokens=8,
        num_speculative_steps=0,
        vocab_size=128,
        device=torch.device("cuda"),
    )
```

它创建一个最小 MRV2 persistent state container。

为什么：

```text
max_num_reqs=1
```

因为：

- T1-T3 不需要 batch complexity；
- T4 需要确定地证明 A remove 后 B 一定复用同一个 slot；
- 只有一个 slot 时 reuse 是 deterministic 的。

---

# 29. 测试 helper `add_request()` 怎么理解

测试 helper：

```text
state.add_request(...)
        ↓
state.apply_staged_writes()
        ↓
torch.accelerator.synchronize()
```

它模拟：

```text
NewRequestData
   ↓
RequestState.add_request
   ↓
CPU staged writes
   ↓
apply_staged_writes
   ↓
GPU persistent state initialized
```

为什么要 `synchronize()`？

因为测试马上会 `.item()` 读 GPU 结果。

它是：

```text
测试 deterministic boundary
```

不是说 production 每个 request add 后都应该人为全局 synchronize。

---

# 30. `advance_one_token()` 是测试里最核心的 helper

它直接调用 production：

```text
post_update()
```

构造：

```text
idx_mapping=[req_idx]
query_start_loc=[0,1]
num_rejected=[0]
num_sampled=[1]
sampled_tokens=[[7]]
```

含义：

```text
这个 batch 只有 1 个 request
它本轮 query range = [0,1)
所以 query_len = 1
没有 reject
```

因此：

```text
computed_delta
= query_len - num_rejected
= 1 - 0
= 1
```

它本质上是在说：

> “不用真的跑 Transformer，假设本轮 forward 已经正确完成 1 token，现在直接测试 post-step commit 是否正确。”

`sampled_tokens=[[7]]` 只是为了给原生 `post_update()` 合法的 sample 输入；S1 并不重点验证 token 7 本身。

---

# 31. T1 — Initialization

输入：

```text
initial num_computed = 37
```

断言：

```text
logical.gpu   = 37
physical.gpu  = 37
```

证明：

```text
state owner 正确
add_request 初始化正确
StagedWriteTensor apply 正确
physical 初值不是错误地固定为 0
```

---

# 32. T2 — Baseline Advancement

初始：

```text
logical = 128
physical = 128
```

调用一次 `advance_one_token()`：

```text
computed_delta=1
```

预期：

```text
logical = 129
physical = 129
```

它证明：

> 在 feature 还没有 reclaim 时，新 state 不破坏原生 `logical == physical` baseline。

但仅有 T2 还不够。

---

# 33. T3 — Independent Divergence Advancement：S1 最重要的 oracle

测试先直接：

```text
logical = 128
physical = 64
```

这里：

```python
state.effective_kv_len.gpu[req_idx] = 64
```

只是 test fixture，**不是未来 reclaim 的实现方法**。

然后执行一次 normal delta：

```text
+1
```

必须得到：

```text
logical:
128 → 129

physical:
64 → 65
```

它证明：

```text
effective_kv_len
和
num_computed_tokens
```

是独立 storage。

为什么这个 test 比 T2 更重要？

假设错误实现：

```text
每轮 effective = logical
```

那么 T2：

```text
128/128 → 129/129
```

仍然会 PASS。

但 T3 会变成：

```text
128/64 → 129/129
```

立即失败。

所以 T3 真正钉死：

> **两者共享 execution delta，不共享 absolute value。**

---

# 34. T4 — Request Slot Reuse

过程：

```text
Request A
idx=0
logical=128
physical 人为设成64
        ↓
remove A
        ↓
idx=0 回 free_indices
        ↓
Request B
initial=37
        ↓
reuse idx=0
        ↓
add_request + apply
```

必须：

```text
B logical=37
B physical=37
```

它证明：

```text
remove 不清零没有问题
前提是 add_request 完整覆盖 reused row
```

这是 persistent slot-array 系统非常典型的 stale-state correctness test。

---

# 35. S1 四个测试合起来验证的是一个完整生命周期

```text
T1
Birth / Initialization
      ↓
T2
Normal Growth
      ↓
T3
Independent Divergence + Growth
      ↓
T4
Death / Slot Reuse
```

所以 S1 虽然 runtime 只增加十几行左右，但 tests 的语义密度非常高。

---

# 36. 这轮学习中最容易出现的错误理解

## 误区 1：`num_computed_tokens_np` 是 GPU→CPU async copy

错误。

正确：

```text
Scheduler logical state
→ SchedulerOutput
→ Worker update_requests
→ num_computed_tokens_np
```

GPU 和 CPU 不是靠每轮互相 copy 来保持 logical 语义一致。

---

## 误区 2：Scheduler 的 `num_computed_tokens` 永远表示 GPU 已经完成的 exact count

不准确。

Scheduler 会在：

```text
_update_after_schedule()
```

按 scheduled tokens **optimistically advance**。

所以尤其 async 下：

```text
Scheduler logical progress
可能领先 GPU execution progress
```

---

## 误区 3：`update_requests()` 在推进 Worker GPU state

不是。

cached request 普通路径中它主要：

```text
更新 Worker CPU `_np`
追加 new block IDs
维护 CPU prefill bookkeeping
```

GPU `num_computed_tokens.gpu` 是 persistent 的，正常由 `post_update` 自己推进。

---

## 误区 4：`StagedWriteTensor` 是 CPU/GPU 双副本

不是。

更准确：

```text
GPU persistent tensor
+
CPU sparse write staging queue
```

---

## 误区 5：既然 logical 和 physical baseline 一样，就可以每轮同步绝对值

不可以。

P1 的核心就是未来允许：

```text
logical=128
physical=64
```

因此只能共享：

```text
normal execution delta
```

不能共享：

```text
absolute state
```

---

## 误区 6：S1 4 tests PASS 就表示 reclaim 能工作

完全不是。

现在没有任何测试覆盖：

```text
physical cache_positions
slot mapping
shortened BlockTable
attention effective length
BlockPool.free
cross-request physical block reuse
```

S1 PASS 只表示：

```text
physical progress state lifecycle 正确
```

---

## 误区 7：T3 里直接写 `effective_kv_len.gpu=64` 就是未来 reclaim 方法

不是。

这是测试 fixture，只负责构造：

```text
logical != physical
```

真实 reclaim 还必须解决：

```text
BlockTable commit
physical state commit
worker ACK
scheduler canonical ownership
safe free
```

---

# 37. 从“两只时钟”升级到“三个执行进度”

一开始用“两只时钟”理解已经比“CPU/GPU 必须每时每刻一致”更接近真实系统：

```text
Scheduler logical clock
vs
Worker GPU execution clock
```

但理解 multi-in-flight batch 后，更准确的是三个 execution progress：

```text
① Scheduler CPU control progress
② Worker CPU submission / `_np` progress
③ GPU committed execution progress
```

例如 Batch1 尚未在 GPU 执行完、Batch2 已进入 Worker CPU 时：

```text
Scheduler CPU = 130
Worker CPU np = 129
Worker GPU    = 128
```

这三个值同时存在是合法的。

它们分别表示：

```text
130:
Scheduler 已经把 B1、B2 都纳入 in-flight logical progress

129:
Worker CPU 正在按 Batch2 的 start logical view 准备 metadata

128:
GPU 此刻物理执行还没有完成 B1 post_update
```

真正 correctness 约束不是：

```text
三个 scalar 必须每时每刻相等
```

而是：

```text
1. 每个 state 的 owner 和语义明确；
2. CPU optimistic state 不冒充 exact GPU state；
3. Worker RPC/submission 保持正确先后关系；
4. GPU dependency 通过 stream/event ordering 保证；
5. execution output 最终通过 update_from_output reconciliation。
```

因此 Scheduler/Worker/GPU 的设计本质是：

```text
允许 transient divergence
+
在明确边界保证 semantic convergence
```

而不是靠每轮 D2H/H2D 强制 lock-step。

---

# 38. S1 后，logical 和 physical 两条状态链

## Logical chain

```text
Scheduler Request.num_computed_tokens
        │
        ├─ schedule-time optimistic advance
        │
        └─ SchedulerOutput start-state transport
                    ↓
          Worker num_computed_tokens_np
                    │
                    └─ CPU batch metadata

Worker num_computed_tokens.gpu
        │
        ├─ prepare_prefill_inputs
        ├─ prepare_pos_seq_lens
        ├─ model logical positions
        └─ post_update + actual delta
```

## Physical chain（S1）

```text
RequestState.effective_kv_len.gpu
        │
        ├─ add_request: initialize from initial logical
        │
        ├─ normal post_update: + actual delta
        │
        └─ future reclaim commit: independently shrink

目前没有 consumer
```

---

# 39. 为什么 P1 的设计最终必须至少有“三种 truth”

到后续 M2/M3，还会形成：

```text
1. Logical/model truth
   num_computed_tokens

2. Worker physical execution truth
   effective_kv_len
   + active BlockTable view

3. Scheduler allocator ownership truth
   req_to_blocks / BlockPool ownership
```

这三者不能互相偷换。

例如未来：

```text
logical = 128
physical effective = 64
scheduler owns 4 retained blocks
```

这三个数字/结构分别回答不同问题。

---

# 40. S1 Web Review 的结论

S1 已完成：

```text
✓ effective_kv_len 是独立 GPU persistent state
✓ 新 request 初始化正确
✓ normal execution 共享 actual computed_delta
✓ 128/64 → 129/65 oracle 成立
✓ remove/reuse row overwrite 成立
✓ 没有误加 CPU mirror
✓ 没有提前改 attention/slot/scheduler
```

所以：

```text
P1-M1-T1-S1
WEB REVIEW: PASS
```

但 Parent Task / M1 Gate 还不能 PASS。

---

# 41. 为什么下一步自然进入 S2

S1 之后我们已经能够稳定保存：

```text
logical = 128
physical = 64
```

但 execution consumer 仍然是原生逻辑：

```text
num_computed_tokens.gpu
        ↓
prepare_pos_seq_lens
        ↓
positions = logical positions
        │
        ├─ model/RoPE             ← 正确
        │
        └─ compute_slot_mappings  ← reclaim 后错误
```

所以 S2 的核心问题已经非常明确：

> **如何在 MRV2 中把 logical model position 和 physical KV cache position 拆开？**

目标结构：

```text
num_computed_tokens.gpu
        ↓
logical_positions
        ↓
model / RoPE


effective_kv_len.gpu
        ↓
cache_positions
        ↓
compute_slot_mappings
```

此外还必须继续审计：

```text
attention seq_lens / effective KV length
```

因为不能粗暴地把所有 `seq_lens` 都从 logical 改成 physical；必须逐 consumer 判断语义。

这就是 M1-T1 下一 Slice 真正开始改变 execution semantics 的地方。

---

# 42. 后续读一个 persistent state 时统一使用这套问题

以后不论读：

```text
effective_kv_len
BlockTables.num_blocks
slot mapping state
ACK state
allocator ownership
```

统一问：

```text
1. Owner 是谁？
2. Source of truth 是谁？
3. 生命周期多长？request / step / kernel？
4. 谁初始化？
5. 谁在 CPU 读？
6. 谁在 GPU 读？
7. 谁推进？按 absolute value 还是 delta？
8. commit boundary 在哪里？
9. async 时谁可能领先？
10. remove/reuse 如何避免 stale state？
11. 有没有第二份 mirror？为什么需要？
12. 它影响的是 logical semantic 还是 physical semantic？
```

这是比单纯 `rg` 函数名更重要的源码阅读方法。

---

# 43. 建议保留的源码定位命令

为了避免后续版本/patch 后行号漂移，建议笔记主要记录 **file + class/function**，需要复查时使用：

```bash
cd /home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim

# Scheduler logical state
rg -n -C 6 \
'num_computed_tokens|_update_after_schedule|update_from_output|_make_cached_request_data' \
vllm/v1/core/sched/scheduler.py

# SchedulerOutput transport
rg -n -C 5 \
'class NewRequestData|class CachedRequestData|class SchedulerOutput|num_computed_tokens' \
vllm/v1/core/sched/output.py

# Worker persistent state
rg -n -C 5 \
'class RequestState|num_computed_tokens|effective_kv_len|apply_staged_writes|remove_request' \
vllm/v1/worker/gpu/states.py

# StagedWriteTensor
rg -n -C 6 \
'class StagedWriteTensor|stage_write_elem|apply_write|_apply_write_kernel|class UvaBackedTensor' \
vllm/v1/worker/gpu/buffer_utils.py

# Worker orchestration
rg -n -C 6 \
'def add_requests|def update_requests|def prepare_inputs|def prepare_attn|def execute_model|def sample_tokens|def postprocess_sampled' \
vllm/v1/worker/gpu/model_runner.py

# InputBatch / positions / post-update
rg -n -C 6 \
'class InputBatch|prepare_prefill_inputs|prepare_pos_seq_lens|post_update|computed_delta' \
vllm/v1/worker/gpu/input_batch.py

# physical slot consumer
rg -n -C 6 \
'compute_slot_mappings' \
vllm/v1/worker/gpu/block_table.py \
vllm/v1/worker/gpu/model_runner.py
```

---

# 44. 最终压缩：只记住这一张图

```text
                    ┌──────────────────────────┐
                    │ Scheduler::Request       │
                    │ logical progress         │
                    │ num_computed_tokens      │
                    └────────────┬─────────────┘
                                 │
                schedule()       │ SchedulerOutput carries start L
                optimistic +q    ▼
                    ┌──────────────────────────┐
                    │ GPUModelRunner           │
                    │ add/update_requests      │
                    └────────────┬─────────────┘
                                 │
                ┌────────────────┴─────────────────┐
                ▼                                  ▼
 num_computed_tokens_np                RequestState GPU persistent
 CPU logical mirror                    ┌──────────────────────────┐
                                      │ num_computed_tokens.gpu  │
                                      │ logical                  │
                                      ├──────────────────────────┤
                                      │ effective_kv_len.gpu     │
                                      │ physical                 │
                                      └────────────┬─────────────┘
                                                   │
                                                   ▼
                                           prepare_inputs
                                                   │
                         logical ──────────────────┤
                                                   ▼
                                             positions
                                                   │
                                  ┌────────────────┴───────────────┐
                                  ▼                                ▼
                              model/RoPE                    slot_mapping
                              logical ✅                    当前耦合点 ⚠

                                                   ↓ forward
                                                   ↓ sample
                                                   ↓ post_update

                                      logical   += actual_delta
                                      physical  += actual_delta

                                  future reclaim:
                                      logical unchanged
                                      physical independently shrink
```

一句话总结：

> **MRV2 的 `num_computed_tokens` 本质上是一个跨 Scheduler CPU、Worker CPU mirror、Worker GPU execution 三个时序域传播的 logical progress；P1 S1 新增 `effective_kv_len`，不是复制这套 logical state，而是在 Worker GPU execution domain 建立一条独立 physical-KV progress state。S1 只建立生命周期；S2 才开始让 physical state真正接管 cache addressing。**

---

# 45. 当前阶段状态与下一阅读入口

当前：

```text
M0: source seam / oracle closed
M1-T1-S1: effective physical state lifecycle PASS
```

下一阅读入口：

```text
M1-T1-S2
Logical Position / Physical Cache Position Consumer Split
```

建议下一步从下面这条现有链路开始逐 consumer 拆：

```text
GPUModelRunner.prepare_inputs()
    ↓
prepare_pos_seq_lens()
    ↓
InputBatch.positions
    ├─ model_inputs["positions"]
    └─ BlockTables.compute_slot_mappings()
```

先回答：

```text
哪些 consumer 必须继续使用 logical positions？
哪些 consumer 必须改为 physical cache_positions？
attention 的 seq_lens 中哪些是 model/logical semantic，哪些是 KV-effective semantic？
```

在这些问题冻结之前，不应该直接修改 S2 代码。
