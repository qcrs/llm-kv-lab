E1 可以收尾了。**没有必要再扩 E1 instrumentation。**

唯一还没动态钉死的是：

```text
Call N sample 得到的 token
        ↓
如何在 Scheduler 尚未 reconcile 时
        ↓
成为 Call N+1 的 MRV2 输入
```

这个问题更属于后面的 **ModelRunner execution-state / input preparation 实验**，不是 E1 Scheduler baseline 的验收项。继续塞进 E1 会让边界失控。

下面这份可以直接作为 E1 实验报告/学习笔记。

---

# E1：单请求 Prefill → Decode 与 Async Scheduling 动态链路实验

## 1. 实验定位

本实验是 vLLM 0.26 动态桥接实验的第一个正式实验。

E0 只负责证明：

```text
环境能够启动
模型能够加载
请求能够完成
基础配置正确
```

E1 则开始回答真正的 Runtime 问题：

> 一个真实请求进入 vLLM 后，Scheduler 如何从 Prefill 过渡到 Decode？SchedulerOutput 如何进入 ModelRunner？异步调度下为什么上一个请求结果还没有回写，Scheduler 就可以继续产生下一批工作？

因此 E1 的重点不是性能，而是：

```text
Runtime control flow
Scheduler state transition
Scheduler ↔ ModelRunner contract
Async batch pipeline
Request token progress
KV block allocation
```

---

# 2. 实验环境

模型：

```text
/data/models/Qwen3-0.6B
```

关键配置：

```python
LLM(
    model="/data/models/Qwen3-0.6B",
    dtype="bfloat16",
    tensor_parallel_size=1,

    block_size=16,
    max_model_len=2048,
    max_num_batched_tokens=256,
    max_num_seqs=4,

    gpu_memory_utilization=0.2,
    enforce_eager=True,
    enable_prefix_caching=False,
)
```

生成配置：

```python
SamplingParams(
    temperature=0.0,
    max_tokens=3,
    ignore_eos=True,
)
```

启动环境：

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=1
export QCACHE_BRIDGE_TRACE=1
```

运行结果确认：

```text
V1 LLM Engine
V2 Model Runner
TP=1
PP=1
DP=1
FLASH_ATTN backend
FlashAttention 2
Async scheduling enabled
Prefix Cache disabled
Chunked Prefill enabled
```

同时 Runtime 明确打印：

```text
async_scheduling=True
use_v2_model_runner=True
pp_size=1
max_concurrent_batches=2
batch_queue_size=2
batch_queue_enabled=True
step_fn=step_with_batch_queue
```

因此当前真正执行的不是普通：

```text
EngineCore.step()
```

而是：

```text
EngineCore.step_with_batch_queue()
```

。

---

# 3. E1 输入与实际 Token 数

输入 Prompt：

```text
Large language model inference systems use KV cache to avoid
recomputing previous keys and values during autoregressive decoding.
Explain briefly why paged KV cache is useful for serving multiple
requests efficiently.
```

Tokenizer 实际得到：

```text
prompt_tokens = 38
```

最终生成：

```text
output_token_ids = [3555, 374, 279]
num_output_tokens = 3
```

对应文本：

```text
' What is the'
```

。

因此本实验模型计算可以抽象为：

```text
Prefill:
P0 ... P37
      ↓
sample O0

Decode:
O0
 ↓
sample O1

Decode:
O1
 ↓
sample O2

达到 max_tokens=3
结束
```

注意最后的 O2 不需要再作为 Decode input 执行一次 forward。

---

# 4. 为什么给源码加入 Trace

静态阅读只能看到：

```python
scheduler.schedule()

model_executor.execute_model(...)

scheduler.update_from_output(...)
```

但静态代码很难回答以下问题：

```text
schedule() 和 update_from_output() 中间到底隔了多久？

SchedulerOutput 到底什么时候进入 ModelRunner？

ModelRunner 什么时候 sample？

Future 什么时候进入 batch queue？

什么时候真正调用 future.result()？

为什么 num_computed_tokens 已经变了，
num_tokens 却还没有变？

为什么 num_in_flight_tokens 会是 38？

为什么 queue 里明明还有 batch，
请求已经完成了？
```

因此本实验采用最小侵入式 instrumentation：

```text
不修改调度逻辑
不修改 KV 分配逻辑
不修改 ModelRunner 计算逻辑

只打印 Runtime state
```

所有 trace 受：

```bash
QCACHE_BRIDGE_TRACE=1
```

控制。

关闭：

```bash
QCACHE_BRIDGE_TRACE=0
```

即可恢复无实验输出状态。

---

# 5. Trace 加入的位置

本实验主要修改三个层次。

## 5.1 EngineCore

文件：

```text
vllm/v1/engine/core.py
```

观察：

```text
Runtime path selection
step_with_batch_queue 调用
Scheduler 调用边界
execute_model submit
sample_tokens submit
batch_queue push/pop
Future wait
Scheduler reconcile
```

主要 Trace：

```text
[CORE][INIT]

[CORE][CALL_BEGIN]

[CORE][SCHEDULE_BEGIN]
[CORE][SCHEDULE_END]

[CORE][EXEC_SUBMIT_BEGIN]
[CORE][EXEC_SUBMIT_RETURN]

[CORE][SAMPLE_SUBMIT_BEGIN]
[CORE][SAMPLE_SUBMIT_RETURN]

[CORE][QUEUE_PUSH]
[CORE][YIELD_WITHOUT_CONSUME]

[CORE][QUEUE_CONSUME_BEGIN]
[CORE][QUEUE_POP]

[CORE][WAIT_RESULT_BEGIN]
[CORE][WAIT_RESULT_END]

[CORE][RECONCILE_BEGIN]
[CORE][RECONCILE_END]
```

---

# 6. Scheduler Trace

文件：

```text
vllm/v1/core/sched/scheduler.py
```

观察两个完全不同阶段：

```text
Schedule-time Commit
Result-time Reconciliation
```

核心 Trace：

```text
[SCHED][BEFORE_COMMIT]

[SCHED][AFTER_COMMIT]

[SCHED][SETTLE]

[SCHED][TOKEN_APPEND]
```

这四个位置实际上是本实验最有价值的 instrumentation。

它们让 Scheduler state 从过去的一个“黑盒数字”变成：

```text
Decision
↓
Commit
↓
Execution
↓
Result
↓
Reconcile
```

---

# 7. ModelRunner Trace

文件：

```text
vllm/v1/worker/gpu/model_runner.py
```

加入：

```text
[MRV2][EXECUTE_BEGIN]

[MRV2][SAMPLE_BEGIN]
```

目的不是分析 Tensor。

而是验证：

```text
SchedulerOutput
↓
Executor
↓
Worker
↓
V2 GPUModelRunner
```

真实发生。

---

# 8. 初始化阶段出现 `_dummy_req_*` 是什么

在真实用户请求之前，日志首先出现：

```text
[MRV2][EXECUTE_BEGIN]
total_scheduled=256

per_req={
    '_dummy_req_0': 64,
    '_dummy_req_1': 64,
    '_dummy_req_2': 64,
    '_dummy_req_3': 64
}
```

这里：

```text
4 requests × 64 tokens
=
256 tokens
```

它不是用户请求。

这是 vLLM 初始化/profile 阶段构造的 synthetic workload。

它说明：

> `GPUModelRunner.execute_model()` 是一个通用 execution data plane，不仅 serving request 会走这里，初始化/profile/warmup 同样会复用这条路径。

随后又出现：

```text
_warmup_0_
_warmup_1_
_warmup_2_
_warmup_3_
```

以及：

```text
total_scheduled=8
total_scheduled=4
total_scheduled=0
```

这些属于 warmup。

因此以后分析 ModelRunner trace 必须根据：

```text
per_req
```

区分：

```text
_dummy_req_*
→ profile/init

_warmup_*
→ warmup

0-xxxx
→ 真实请求

{}
→ empty pipeline work
```

。

---

# 9. E1 真正运行开始

真实 Request ID：

```text
0-b7d8f1ea
```

真正 runtime 开始：

```text
[CORE][CALL_BEGIN]
call=0
path=step_with_batch_queue
queue_len=0
queue_capacity=2
scheduler_has_requests=True
```

这说明：

```text
run_busy_loop
↓
_process_engine_step
↓
self.step_fn
↓
step_with_batch_queue()
```

已经进入第一次 pipeline pump。

这里的：

```text
call=0
```

不是 GPU step。

不是 token position。

不是完整 inference iteration。

只是：

> `step_with_batch_queue()` 被调用的次数。

---

# 10. Call 0：完整 Prefill

Scheduler 开始：

```text
[CORE][SCHEDULE_BEGIN]
```

随后得到：

```text
budget_start=256
budget_remaining=218
```

所以：

```text
256 - 218 = 38
```

正好调度整个 Prompt。

---

## 10.1 BEFORE_COMMIT

```text
num_tokens=38
num_tokens_with_spec=38
placeholders=0

num_computed=0
num_in_flight=0

pending=38
scheduled=38

new_block_ids=([1,2,3],)
```

这一时刻：

```text
真实 Request：
P0 ... P37

known frontier = 38
computed frontier = 0
```

因此：

```text
pending = 38
```

Scheduler 决定：

```text
scheduled = 38
```

---

# 11. Prefill KV Block 分配

当前：

```text
block_size=16
```

Prompt：

```text
38 tokens
```

所以需要：

```text
ceil(38 / 16)
=
3 blocks
```

Runtime 实际：

```text
new_block_ids=([1,2,3],)
```

所以：

```text
logical block 0
positions 0~15
→ physical block 1

logical block 1
positions 16~31
→ physical block 2

logical block 2
positions 32~47
→ physical block 3
```

第三块当前只占：

```text
positions 32~37
```

所以仍有：

```text
10 slots
```

可以继续承载后续 Decode token。

这也解释为什么之后：

```text
new_block_ids=None
```

。



---

# 12. `_update_after_schedule()`：Schedule-time Commit

随后：

```text
AFTER_COMMIT:

num_tokens=38
placeholders=1
num_computed=38
num_in_flight=38
```

对比：

```text
BEFORE                     AFTER

num_tokens       38          38
placeholder       0           1
computed          0          38
in_flight         0          38
```

这里第一次动态证明：

```text
_update_after_schedule()
```

不是 GPU result update。

而是：

> **Scheduler 对刚刚产生的调度计划进行 schedule-time commit。**

核心含义：

```text
computed += 38
```

表示：

> P0~P37 已经被 Scheduler 认领/发出，下一次 schedule 不能重复调度。

以及：

```text
in_flight += 38
```

表示：

> 这 38 个 scheduled work 尚未通过 `update_from_output()` 完成 result reconciliation。

因此：

```text
num_computed_tokens
```

不能简单理解为：

> GPU 已经确认计算完成多少 token。

更准确是：

> Scheduler logical computation frontier。

---

# 13. Placeholder 的意义

Commit 后：

```text
num_tokens=38
placeholder=1
```

此时真实 sampled token O0 尚未 append。

但 Scheduler 已经知道：

```text
当前 batch 将产生一个 output position
```

因此先创建一个逻辑占位：

```text
P0 ... P37 [?]
             ↑
        placeholder
```

这个 `?` 尚不知道真实 token ID。

---

# 14. SchedulerOutput 进入 ModelRunner

Scheduler 返回：

```text
total_scheduled=38
per_req={
    '0-b7d8f1ea':38
}
```

然后：

```text
EXEC_SUBMIT_BEGIN
↓
MRV2 EXECUTE_BEGIN
```

两边都打印：

```text
38
```

说明：

```text
SchedulerOutput
```

真正一路传入：

```text
EngineCore
↓
Executor
↓
Worker
↓
GPUModelRunner
```

也就是：

```text
Control Plane
↓
Execution Contract
↓
Data Plane
```

这条边界被动态验证。

---

# 15. execute 与 sample

Call 0：

```text
EXEC_SUBMIT_RETURN
future_type=Future
future_done=True
```

然后：

```text
SAMPLE_SUBMIT_BEGIN
↓
MRV2 SAMPLE_BEGIN
↓
SAMPLE_SUBMIT_RETURN

future_type=AsyncOutputFuture
future_done=False
```

这说明：

```text
execute_model()
```

和：

```text
sample_tokens()
```

是两个 execution stages。

但这里需要特别严谨：

```text
execute Future done=True
```

不能直接等价于：

> GPU 所有 kernel 已经物理执行完毕。

它只能说明：

> execute-side host/runtime Future 已经 ready，CPU 可以继续推进。

而：

```text
AsyncOutputFuture done=False
```

明确说明：

> 最终 sampled ModelRunnerOutput 此刻还没有被 CPU consumption side 获取。

---

# 16. batch_queue 的真实含义

随后：

```text
QUEUE_PUSH

queue_len=1
queue_capacity=2
scheduled=38
```

Queue 中不是：

> 等待执行的任务。

而是：

> **已经 submit，但其 result 尚未完成 Scheduler reconciliation 的 outstanding batch transaction。**

可以抽象：

```text
A = {
    SchedulerOutput,
    result Future,
    execution state
}
```

所以：

```text
queue=[A]
```

不表示：

```text
A 还没执行
```

而表示：

```text
A 已经进入 execution/result pipeline，
但 CPU 还没有 consume/reconcile A。
```

---

# 17. Async Scheduling 的关键行为

Call 0：

```text
queue_len=1
capacity=2
```

因此：

```text
YIELD_WITHOUT_CONSUME
reason=queue_not_full
```

也就是：

```text
不调用 A.result()
不调用 update_from_output(A)
```

直接 return。

随后 busy loop 立刻调用：

```text
step_with_batch_queue()
```

进入 Call 1。

这就是 Async Scheduling 最核心的动态证据。

---

# 18. Call 1：为什么仍然是 38

Call 1：

```text
num_tokens=38
placeholders=1

num_computed=38
num_in_flight=38

pending=1
scheduled=1
```

这里非常关键。

### `num_tokens=38`

因为：

```text
O0 尚未经过 update_from_output()
```

所以 Request 仍然只有 38 个真实 token。

---

### `num_computed=38`

因为 Prefill 的 38 tokens 已经被 schedule。

---

### `num_in_flight=38`

因为 Batch A 尚未 reconcile。

---

### `placeholder=1`

虽然真实 O0 还没 append，但 Scheduler 已经预留下一 output position。

因此：

```text
target scheduling frontier
=
38 + 1
=
39
```

而：

```text
computed frontier
=
38
```

所以：

```text
pending
=
39 - 38
=
1
```

于是 Scheduler 能继续：

```text
scheduled=1
```

这就是 AsyncScheduler 存在的根本意义。

---

# 19. Call 1 Commit

Schedule 一个 token 后：

```text
placeholders:
1 → 2

computed:
38 → 39

in_flight:
38 → 39
```

此时 Scheduler 的账本里有两笔 outstanding work：

```text
A = 38
B = 1
```

所以：

```text
in_flight=39
```

。

---

# 20. 第二个 Batch 入队

B 也经过：

```text
execute_model
↓
sample_tokens
↓
AsyncOutputFuture
```

然后：

```text
QUEUE_PUSH
queue_len=2
queue_capacity=2
```

现在：

```text
[A,B]
```

队列满。

因此开始消费 oldest。

---

# 21. FIFO 消费 Batch A

日志：

```text
QUEUE_POP
queue_len_after=1
scheduled=38
```

因为：

```text
A scheduled=38
B scheduled=1
```

所以被 pop 的明确是：

```text
A
```

此时 queue：

```text
[B]
```

这证明 `batch_queue` 是一个 FIFO outstanding transaction pipeline。

---

# 22. `future.result()` 到底是什么

随后：

```text
WAIT_RESULT_BEGIN
↓
WAIT_RESULT_END
model_output_type=ModelRunnerOutput
```

这里才是真正：

```python
future.result()
```

也就是：

> CPU result-consumption side 现在真正需要 A 的结果。

如果 Future 已经 ready：

```text
直接返回
```

如果尚未 ready：

```text
CPU 在这里等待
```

当前 trace 没有精确时间戳，因此不能判断实际阻塞多久。

但控制流已经明确。

---

# 23. `update_from_output()`：Result-time Reconciliation

拿到：

```text
ModelRunnerOutput A
```

之后：

```text
RECONCILE_BEGIN
```

进入：

```text
Scheduler.update_from_output()
```

这与之前：

```text
_update_after_schedule()
```

职责完全不同。

可以正式定义：

```text
_update_after_schedule()
=
Schedule-time Commit

update_from_output()
=
Result-time Reconciliation
```

---

# 24. SETTLE：为什么 39 → 1

真实：

```text
settled=38

in_flight_before=39
in_flight_after=1
```

因为：

```text
Outstanding:

A = 38
B = 1

total = 39
```

现在 A 返回：

```text
39 - 38 = 1
```

所以：

```text
remaining outstanding
=
B
```

这直接证明：

> `num_in_flight_tokens` 不是 GPU 此刻正在执行几个 token，而是 Scheduler 尚未完成 result reconciliation 的 scheduled work 数量。

---

# 25. sampled token 什么时候真正进入 Request

紧接：

```text
TOKEN_APPEND

num_tokens_after=39
num_output_tokens=1
num_computed=39
num_in_flight=1
```

这里终于：

```text
num_tokens:
38 → 39
```

也就是说 O0 真正进入 Request 的时间点是：

```text
ModelRunnerOutput
↓
Scheduler.update_from_output
↓
_update_request_with_output
↓
append output token
```

不是：

```text
sample_tokens() 被调用
```

的时候。

---

# 26. Call 2：进入稳定 Pipeline

此时：

```text
queue=[B]
```

Request：

```text
tokens=39
placeholder=1
computed=39
inflight=1
```

Scheduler 又 schedule：

```text
C=1
```

commit：

```text
computed:
39 → 40

in_flight:
1 → 2
```

queue：

```text
[B,C]
```

满后：

```text
consume B
```

于是：

```text
in_flight:
2 → 1

num_tokens:
39 → 40
```

得到第二个 output token O1。

这时 Pipeline 已进入稳态：

```text
保留一个 outstanding batch
↓
加入一个新 batch
↓
queue 满
↓
消费 oldest
↓
重新留下一个 outstanding batch
```

即：

```text
[A]
→ [A,B]
→ [B]

→ [B,C]
→ [C]

→ ...
```

---

# 27. Call 3 为什么没有 scheduled token

Call 3：

```text
total_scheduled=0
per_req={}
```

原因：

```text
已经 materialize:
O0
O1

C outstanding:
会产生 O2
```

而：

```text
max_tokens=3
```

所以 C 结果一回来即完成请求。

没有必要再执行：

```text
forward(O2)
```

因此：

```text
scheduled=0
```

。

---

# 28. 为什么最终 `num_tokens=41`，`num_computed=40`

C result 回来：

```text
TOKEN_APPEND

num_tokens_after=41
num_output_tokens=3
num_computed=40
```

此时：

```text
38 Prompt
+
3 Output
=
41 tokens
```

但最后：

```text
O2
```

只被 sample 出来。

没有再作为输入执行一次 Transformer forward。

所以：

```text
num_tokens = 41
num_computed = 40
```

这是一条非常好的动态证据：

> `num_tokens` 和 `num_computed_tokens` 从来不是同一个 frontier。

---

# 29. 为什么还有 Call 4 和 Call 5

请求生成完成不等于：

```text
batch_queue 已经清空
```

Call 3 之后仍存在 pipeline entry。

因此 Runtime 需要：

```text
drain outstanding queue
```

Call 4：

```text
scheduled=0
```

仍通过 queue lifecycle 消费一个 empty transaction。

Call 5：

```text
scheduler_has_requests=False
queue_len=1
```

此时已经：

```text
没有新 Request work
```

但 queue 仍不为空。

因此 Call 5 不再 schedule 新 batch，而是直接：

```text
QUEUE_POP
↓
WAIT_RESULT
↓
RECONCILE {}
```

最终：

```text
queue=[]
```

才进行 teardown。

因此必须区分：

```text
Request finished
```

和：

```text
Pipeline drained
```

两个事件。

---

# 30. batch_queue_size=2 的运行模型

本次实验最终验证出的稳态模型：

```text
初始：

queue=[]


产生 A：

[A]

queue 未满
→ 不消费


产生 B：

[A,B]

queue 满
→ consume A

[B]


产生 C：

[B,C]

queue 满
→ consume B

[C]


之后：

[C, empty]

consume C

[empty]


最终继续 drain：

[]
```

所以：

```text
max_concurrent_batches=2
```

不是两个 batch 同时在 GPU 上做完整 Transformer forward 的意思。

更准确是：

> Runtime 最多允许两个 batch 处于 outstanding execution/result pipeline 中。

---

# 31. Async Scheduling 最终心智模型

传统同步：

```text
CPU Schedule A
↓
Submit A
↓
GPU A
↓
Sample A
↓
CPU Wait A
↓
Update A

然后才能：

CPU Schedule B
```

当前 Async Scheduling：

```text
CPU Schedule A
↓
Submit A
↓
Sample-call A
↓
Future A
        │
        │ result pipeline
        ▼
CPU 不等待 A

CPU Schedule B
↓
Submit B
↓
Sample-call B
↓
Future B

queue=[A,B]

然后：

Consume A
↓
Update A

继续：

Schedule C
↓
...
```

它优化的核心不是破坏 token dependency。

而是：

> **消除每个 batch 后 CPU scheduling / GPU execution / CPU result consumption 之间的强制全局 barrier。**

---

# 32. Scheduler 与 GPUModelRunner 的解耦

本实验还验证了一件很重要的架构事实。

不是：

```text
Scheduler
直接调用
GPUModelRunner
```

真正是：

```text
                     EngineCore
                         │
        ┌────────────────┴──────────────┐
        │                               │
        ▼                               ▼
    Scheduler                       batch_queue
        │                               │
        │ SchedulerOutput               │
        ▼                               │
     Executor                           │
        ▼                               │
      Worker                            │
        ▼                               │
 GPUModelRunner                         │
        │                               │
        └────── ModelRunnerOutput ──────┘
                         │
                         ▼
              Scheduler.update_from_output
```

其中：

```text
SchedulerOutput
```

是：

> Scheduler → Execution Plane 的单步执行契约。

而：

```text
ModelRunnerOutput
```

是：

> Execution Plane → Scheduler 的结果契约。

所以：

```text
Scheduler = WHAT TO RUN

ModelRunner = RUN THE WORK

EngineCore = ORCHESTRATE THE PIPELINE
```

---

# 33. E1 对 KV Cache 学习的直接价值

虽然 E1 重点是 Scheduler，但已经得到几条 KV 重要证据。

第一：

```text
38 tokens
block_size=16
↓
3 physical blocks
```

真实：

```text
[1,2,3]
```

。

第二：

后续 Decode：

```text
scheduled=1
new_block_ids=None
```

说明：

> 每生成一个 token 不代表必须分配一个新 KV block。

因为：

```text
physical block 3
```

还有剩余 slots。

第三：

这进一步说明：

```text
Token Scheduling
```

和：

```text
KV Block Allocation
```

是相关但不一一对应的两个决策层。

这会直接进入后面的 KV block lifecycle 实验。

---

# 34. E1 最终验收结论

E1 可以判定为 **PASS**。

已经动态验证：

```text
[PASS] 单请求完整 Prefill

[PASS] Prefill 后连续 Decode

[PASS] token budget 生效

[PASS] num_tokens / num_computed_tokens 区分

[PASS] num_in_flight_tokens 动态语义

[PASS] output placeholder 动态语义

[PASS] _update_after_schedule 的 schedule-time commit

[PASS] update_from_output 的 result-time reconciliation

[PASS] SchedulerOutput → MRV2 数据面

[PASS] execute_model / sample_tokens 分阶段

[PASS] AsyncOutputFuture

[PASS] step_with_batch_queue 是当前真实路径

[PASS] batch_queue_size=2

[PASS] outstanding batch FIFO

[PASS] queue 满后消费 oldest

[PASS] Request finished 与 pipeline drained 区分

[PASS] Prompt 38 / block_size 16 → 3 KV blocks

[PASS] Decode reuse partial block

[PASS] 初始化 dummy/warmup workload 与真实 request 区分
```

真实实验数据和完整调用顺序均由本次 trace 直接支持。

---

# 35. E1 后不继续扩 Trace 的原因

E1 还可以继续追一个问题：

```text
Call 0 sample O0
↓
O0 尚未被 CPU Scheduler reconcile
↓
Call 1 MRV2 如何获得真实 O0 token_id
作为下一次 input
```

但这个已经从：

```text
Scheduler Runtime
```

进入：

```text
MRV2 execution state
input preparation
sampled-token handoff
```

如果继续塞进 E1，会破坏实验边界。

因此建议记录为：

```text
Open Question / Later Experiment

MRV2 async sampled-token handoff:
sampled O_t
→ execution-side state
→ next-step input_ids
```

放到后续：

```text
SchedulerOutput → ModelRunner
Persistent RequestState
InputBatch
prepare_inputs
```

的数据面实验中再验证。

---

# 36. E1 一句话总结

E1 最重要的结论不是“vLLM 有 Prefill 和 Decode”，而是：

> **vLLM V1 + MRV2 在当前配置下通过 `step_with_batch_queue()` 将 Scheduler decision、execution submission、sampling/result readiness 和 Scheduler reconciliation 做成流水线；Scheduler 使用 optimistic `num_computed_tokens`、`num_in_flight_tokens` 与 output placeholder，在上一个 batch 尚未完成 CPU-side result reconciliation 时即可继续规划下一个 batch，而 Scheduler 与 GPUModelRunner 通过 SchedulerOutput / ModelRunnerOutput 解耦。**

这就是后面继续读 Chunked Prefill、KV Block、slot mapping、Paged KV 数据面的 Runtime 基础。
