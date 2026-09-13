# E3-vLLM-Continuous-Batching并发调度与Ingress链路动态实验

> 实验状态：**PASS / 正式收尾**  
> 固定版本：vLLM v0.26.0  
> commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> 实验日期：2026-08-19  
> 实验目录：`04-experiments/vllm-bridge/raw/e3-concurrency/`  
> 实验脚本：`04-experiments/vllm-bridge/scripts/e3_concurrency.py`  
> Runtime：Offline LLM / V1 Engine / V2 Model Runner / TP=1 / PP=1 / DP=1 / Async Scheduling / FlashAttention 2

---

# 1. 实验定位

E1 已经回答：

```text
单请求
完整 Prefill
↓
Decode
```

E2 已经回答：

```text
单请求
长 Prompt
↓
Chunked Prefill
↓
Decode
```

E3 第一次进入真正的多请求场景，目标是回答：

```text
多个 Request 如何共享同一个 Scheduler step token budget？

一个 SchedulerOutput 能否同时包含多个 Request？

短请求的 Decode
能否和长请求的 Chunked Prefill
同时出现在一个 execution batch 中？

waiting / running
如何在 continuous batching 中动态变化？

Frontend 一次 generate([A,B,C])
是否意味着 Scheduler 第一个 step 就同时看到 A/B/C？
```

三周桥接手册对 E3 的原始要求是：

```text
3 requests
prompt lengths ≈ 32 / 160 / 320
max_num_batched_tokens=128
max_num_seqs=4
max_tokens=4
```

并要求至少出现：

```text
一个 step 同时包含两个 request

且：

长 Prompt Prefill chunk
与其他 request 的 work
发生交错
```

本实验实际不仅满足该 DoD，还进一步动态补证了：

```text
Frontend
↓
SyncMPClient
↓
ZMQ
↓
EngineCoreProc input thread
↓
local input_queue
↓
EngineCore busy loop
↓
Scheduler.add_request()
```

这条请求 ingress 主链。

---

# 2. 固定环境

```text
vLLM:
v0.26.0

commit:
568afb3a13806beb53bb2e6bd518269357b237c0

repo:
~/learning/llm-kv-lab/third_party/vllm

env:
source ./activate

Python:
qcache

GPU:
A100 80GB PCIe

model:
/data/models/Qwen3-0.6B

TP=1
PP=1
DP=1

Frontend:
Offline LLM

Engine:
V1 Engine

Model Runner:
V2 Model Runner
```

实验配置：

```text
dtype=bfloat16
block_size=16
max_model_len=2048
max_num_batched_tokens=128
max_num_seqs=4
gpu_memory_utilization=0.2
enforce_eager=True
enable_prefix_caching=False
max_tokens=4
ignore_eos=True
```

Instrumentation：

```bash
export QCACHE_BRIDGE_TRACE=1
```

注意：

```text
QCACHE_BRIDGE_TRACE
```

是本学习分支自定义 trace 开关，不是 vLLM 官方变量。

---

# 3. 实验输入

实际 tokenizer 长度：

```text
A = 45 tokens
B = 171 tokens
C = 325 tokens
```

因此：

```text
A < 128

B > 128

C >> 128
```

这三个 Request 刻意代表三类 workload：

```text
A:
短 Prompt
很快进入 Decode

B:
中等 Prompt
需要两次 Prefill

C:
长 Prompt
需要多次 Chunked Prefill
```

这样才有机会在同一个 Scheduler step 中形成真正的 mixed workload。

---

# 4. E3 原始实验脚本的核心设计

核心调用：

```python
outputs = llm.generate(
    [prompt_a, prompt_b, prompt_c],
    sampling_params,
    use_tqdm=False,
)
```

表面上是：

```text
一次 Python generate()
传入三个 prompts
```

实验开始时我们暂时不假设：

```text
Scheduler 第一个 step
一定同时拥有 A/B/C
```

后续 ingress trace 正是为验证这一点而补充。

---

# 5. E3 第一阶段：复用已有 Scheduler / EngineCore / MRV2 trace

E3 初始没有修改源码。

继续复用 E1/E2 已有 instrumentation。

---

## 5.1 EngineCore trace

文件：

```text
vllm/v1/engine/core.py
```

事件：

```text
[BRIDGE][CORE][CALL_BEGIN]
[BRIDGE][CORE][SCHEDULE_BEGIN]
[BRIDGE][CORE][SCHEDULE_END]

[BRIDGE][CORE][EXEC_SUBMIT_BEGIN]
[BRIDGE][CORE][EXEC_SUBMIT_RETURN]

[BRIDGE][CORE][SAMPLE_SUBMIT_BEGIN]
[BRIDGE][CORE][SAMPLE_SUBMIT_RETURN]

[BRIDGE][CORE][QUEUE_PUSH]
[BRIDGE][CORE][QUEUE_POP]

[BRIDGE][CORE][RECONCILE_BEGIN]
[BRIDGE][CORE][RECONCILE_END]
```

用于观察：

```text
SchedulerOutput
↓
execution submit
↓
batch_queue
↓
result consume
↓
Scheduler reconciliation
```

---

## 5.2 Scheduler trace

文件：

```text
vllm/v1/core/sched/scheduler.py
```

已有：

```text
[BRIDGE][SCHED][step=N]

[BRIDGE][SCHED][BEFORE_COMMIT]

[BRIDGE][SCHED][AFTER_COMMIT]

[BRIDGE][SCHED][SETTLE]

[BRIDGE][SCHED][TOKEN_APPEND]
```

核心字段：

```text
kind
num_tokens
num_computed
num_in_flight
placeholders
pending
scheduled
new_block_ids
budget_start
budget_remaining
```

---

## 5.3 MRV2 trace

文件：

```text
vllm/v1/worker/gpu/model_runner.py
```

已有：

```text
[BRIDGE][MRV2][EXECUTE_BEGIN]
[BRIDGE][MRV2][SAMPLE_BEGIN]
```

最关键字段：

```text
total_scheduled
per_req
```

它用于验证：

```text
Scheduler 决定的 multi-request SchedulerOutput
```

是否真实进入：

```text
MRV2 execution batch
```

---

# 6. 第一阶段真实结果：Continuous Batching 已经成立

实验的核心 SchedulerOutput 可以压缩成：

| Call | A | B | C | total |
|---:|---|---|---|---:|
| 0 | Prefill 45 | — | — | 45 |
| 1 | Decode-like 1 | Prefill 127 | — | 128 |
| 2 | Decode 1 | Final Prefill 44 | Prefill 83 | 128 |
| 3 | Decode 1 | Decode 1 | Prefill 126 | 128 |
| 4 | — | Decode 1 | Final Prefill 116 | 117 |
| 5 | — | Decode 1 | Decode 1 | 2 |
| 6 | — | — | Decode 1 | 1 |
| 7 | — | — | Decode 1 | 1 |
| 8 | — | — | — | 0 / drain |

Batch membership：

```text
Call 0:
[A]

Call 1:
[A,B]

Call 2:
[A,B,C]

Call 3:
[A,B,C]

Call 4:
[B,C]

Call 5:
[B,C]

Call 6:
[C]

Call 7:
[C]
```

这就是 Continuous Batching 最直观的动态表现：

```text
batch membership
不是固定的

而是每个 Scheduler step
根据当前 active request state
重新构造 current-step execution view
```

---

# 7. Call 1：第一次证明 Global Token Budget

Call 1：

```text
budget_start=128
```

A：

```text
pending=1
scheduled=1
```

剩：

```text
127
```

B：

```text
pending=171
scheduled=127
```

所以：

```text
A 1
+
B 127
=
128
```

真实 SchedulerOutput：

```text
per_req={
    A: 1,
    B: 127
}
```

MRV2 也收到：

```text
total_scheduled=128

per_req={
    A: 1,
    B: 127
}
```

因此动态证明：

> `max_num_batched_tokens=128` 是 **整个 Scheduler step 的 global compute token budget**，不是每个 Request 各有 128。

---

# 8. Call 2：整个 E3 最关键的一轮

Call 2：

```text
budget=128
```

此时：

```text
A:
已经进入 Decode
pending=1

B:
num_computed=127
num_tokens=171
仍有 44 Prompt tokens

C:
尚未计算
pending=325
```

Scheduler 分配：

```text
A → 1

剩：
127

B → 44

剩：
83

C → 83

最终：
1 + 44 + 83 = 128
```

因此同一个 SchedulerOutput 中出现：

```text
A:
Decode

B:
Final Prefill chunk

C:
New / Chunked Prefill
```

这是 E3 最重要的动态证据：

> **Prefill / Chunked Prefill / Decode 可以在同一个 SchedulerOutput 中混合存在。**

Scheduler 上层不需要：

```text
独立 Prefill scheduler
+
独立 Decode scheduler
```

它使用的统一抽象更接近：

```text
每个 request 当前还有多少 token work pending
↓
根据 global token budget
决定本轮每个 request 的 num_scheduled_tokens
```

---

# 9. Call 3：Decode + Decode + Chunked Prefill

Call 3：

```text
A → 1
B → 1
C → 126
```

刚好：

```text
1 + 1 + 126 = 128
```

这轮代表：

```text
A Decode
+
B Decode
+
C Chunked Prefill
```

同一个 MRV2 batch。

它是 production mixed workload 的直接动态样本。

---

# 10. C 的 Chunked Prefill 为什么不是固定 128

单请求 E2 中：

```text
max_num_batched_tokens=64

只有一个 request
```

因此 chunk：

```text
64
64
64
64
64
45
```

E3 中 C 实际 Prefill：

```text
83
126
116
```

验证：

```text
83 + 126 + 116
=
325
```

原因不是 C 自己有固定 chunk size，而是：

```text
C 本轮能得到多少 token

=
当前全局 token budget
-
前面其他 running request 已占用 token work
```

Call 2：

```text
128
- A 1
- B 44
=
83
```

Call 3：

```text
128
- A 1
- B 1
=
126
```

Call 4：

```text
C 只剩 116 Prompt tokens
```

所以：

```text
B 1
+
C 116
=
117

budget_remaining=11
```

因此：

> **多请求场景中的 Chunked Prefill chunk size 是动态结果，不是固定常数。**

它取决于：

```text
request pending work
+
global token budget
+
running order
+
其他 request 本轮 work
```

---

# 11. KV block 分配结果

虽然 E3 不是 KV pressure 实验，但 block growth 仍然可以直接验证。

当前：

```text
block_size=16
```

---

## 11.1 A

Prompt：

```text
45
```

理论：

```text
ceil(45/16)=3 blocks
```

真实：

```text
[1,2,3]
```

---

## 11.2 B

Prompt：

```text
171
```

理论：

```text
ceil(171/16)=11 blocks
```

实际：

```text
Call 1:
[4..11]
= 8 blocks

Call 2:
[12,13,14]
= 3 blocks

total:
11 blocks
```

完全一致。

---

## 11.3 C

Prompt：

```text
325
```

理论：

```text
ceil(325/16)=21 blocks
```

实际：

```text
Call 2:
[15..20]
= 6

Call 3:
[21..28]
= 8

Call 4:
[29..35]
= 7

total:
6+8+7
=
21
```

完全一致。

因此 E3 进一步证明：

```text
多个 request
共享 global compute budget

但各自拥有独立：

logical computation frontier
KV coverage
physical block ownership
```

---

# 12. Async batch_queue 在多请求场景中的真正含义

E1/E2 中 queue item 基本只含一个 Request，很容易误解：

```text
batch_queue
=
request queue
```

E3 纠正了这一点。

例如 Call 2 产生：

```text
SchedulerOutput #2

{
    A: 1,
    B: 44,
    C: 83
}
```

整个 SchedulerOutput 作为一个 batch transaction 进入：

```text
batch_queue
```

Call 3 又产生：

```text
{
    A: 1,
    B: 1,
    C: 126
}
```

所以 queue 更准确表示：

```text
[
    outstanding batch transaction #2,
    outstanding batch transaction #3
]
```

随后：

```text
pop batch transaction
↓
ModelRunnerOutput
↓
Scheduler.update_from_output()
↓
分别 SETTLE A/B/C
```

因此：

> **batch_queue 保存的是已经 submit、尚未完成 Scheduler reconciliation 的 SchedulerOutput-level batch transactions。**

一个 transaction 可以同时包含多个 Request。

---

# 13. 第一阶段发现的唯一疑问：Call 0 为什么只调度 A

最初 Call 0：

```text
budget_start=128
A scheduled=45
budget_remaining=83
```

自然产生疑问：

```text
为什么剩余 83 没给 B/C？
```

已有 trace 当时只能看到：

```text
Call 0 SchedulerOutput 只有 A
```

但不能区分：

```text
情况 1：
B/C 尚未进入 Scheduler

情况 2：
B/C 已经 waiting，
但 waiting admission 没有处理它们
```

所以增加了第一组补证 trace。

---

# 14. 补证 Trace A：`QUEUE_SNAPSHOT`

文件：

```text
vllm/v1/core/sched/scheduler.py
```

位置：

```text
Scheduler.schedule()

在：
# First, schedule the RUNNING requests.

之前
```

即：

```text
本 Scheduler step
还没有修改 running / waiting 之前
```

增加：

```python
if os.getenv("QCACHE_BRIDGE_TRACE", "0") == "1":
    snapshot_step = getattr(self, "_bridge_trace_step", 0)

    print(
        f"[BRIDGE][SCHED][QUEUE_SNAPSHOT] "
        f"sched_call={snapshot_step} "
        f"known_requests={list(self.requests.keys())} "
        f"running={[req.request_id for req in self.running]} "
        f"waiting={[req.request_id for req in self.waiting]}",
        flush=True,
    )
```

---

## 14.1 为什么放在这里

必须发生在：

```text
running scheduling loop
waiting admission loop
```

之前。

否则 snapshot 可能已经被本 step 的调度逻辑修改过。

它要回答的是：

```text
这个 Scheduler step
真正开始时

Scheduler 当前知道哪些 request？

谁在 running？

谁在 waiting？
```

---

# 15. QUEUE_SNAPSHOT 的真实结果

Call 0：

```text
known=[A]
running=[]
waiting=[A]
```

直接证明：

```text
Call 0 开始时
Scheduler 根本不知道 B/C
```

所以：

```text
budget_remaining=83
```

不是 Scheduler 不愿意给 B/C，而是：

```text
没有 B/C 可调度
```

---

## Call 1

```text
known=[A,B,C]
running=[A]
waiting=[B,C]
```

然后：

```text
先 running A → 1
剩 127

再 waiting B → 127
剩 0

C 继续 waiting
```

这是真实的：

```text
running first
↓
waiting admission
```

动态证据。

---

## Call 2

```text
running=[A,B]
waiting=[C]
```

说明 B 在上一轮已经：

```text
waiting → running
```

本轮：

```text
A → 1
B → 44
剩 83
C waiting → 83
```

于是 C admission。

---

## Call 3

```text
running=[A,B,C]
waiting=[]
```

说明 C 已经：

```text
waiting → running
```

因此 E3 动态观察到了完整 request lifecycle：

```text
not yet visible
↓
known / waiting
↓
admission
↓
running
↓
finish
↓
remove
```

---

# 16. 继续追问：为什么 `generate([A,B,C])`，Call 0 却只知道 A？

QUEUE_SNAPSHOT 已证明：

```text
Call 0 Scheduler 只知道 A
```

但仍然没有回答：

```text
B/C 到底卡在哪里？
```

所以又增加了 ingress trace。

---

# 17. Ingress 补证的目标

要动态打通：

```text
Frontend
↓
SyncMPClient
↓
ZMQ
↓
EngineCoreProc input thread
↓
input_queue
↓
busy loop
↓
Scheduler.add_request()
```

因此增加三个 instrumentation 位置。

---

# 18. 补证 Trace B1：Frontend `CLIENT_SEND`

文件：

```text
vllm/v1/engine/core_client.py
```

位置：

```text
SyncMPClient.add_request()
```

不是：

```text
abstract add_request
InprocClient.add_request
```

而是 multiprocess runtime 当前真实路径中的：

```python
def add_request(self, request: EngineCoreRequest) -> None:
    if self.is_dp:
        self.engines_running = True

    ...

    self._send_input(EngineCoreRequestType.ADD, request)
```

增加：

```python
if os.getenv("QCACHE_BRIDGE_TRACE", "0") == "1":
    print(
        f"[BRIDGE][INGRESS][CLIENT_SEND] "
        f"ts_ns={time.monotonic_ns()} "
        f"pid={os.getpid()} "
        f"req={request.request_id}",
        flush=True,
    )
```

---

## 18.1 这个 trace 回答什么

它定义：

```text
Frontend 真正把 Request
交给 EngineCore transport
```

的时间点。

因此可以判断：

```text
A/B/C 是否在 Call 0 之前
都已经 SEND
```

---

# 19. 补证 Trace B2：`IPC_RECV / IPC_ENQUEUE`

文件：

```text
vllm/v1/engine/core.py
```

位置：

```text
EngineCoreProc.process_input_sockets()
```

真实链：

```python
type_frame, *data_frames = input_socket.recv_multipart(...)

if request_type == ADD:
    req = add_request_decoder.decode(...)
    request = self.preprocess_add_request(req)

...

self.input_queue.put_nowait((request_type, request))
```

在：

```text
input_queue.put_nowait()
```

前后增加：

```text
[BRIDGE][INGRESS][IPC_RECV]

[BRIDGE][INGRESS][IPC_ENQUEUE]
```

记录：

```text
timestamp
pid
input thread
request_id
input_queue size
```

---

## 19.1 这个 trace 回答什么

区分：

```text
Request 已经通过 ZMQ 到 EngineCoreProc
```

和：

```text
Request 已经进入 EngineCore local input_queue
```

两个阶段。

---

# 20. 补证 Trace B3：`CORE_DEQUEUE / SCHED_ADD_DONE`

文件：

```text
vllm/v1/engine/core.py
```

位置：

```text
EngineCoreProc._handle_client_request()
```

真实 ADD 分支：

```python
elif request_type == EngineCoreRequestType.ADD:
    req, request_wave = request

    if self._reject_add_in_shutdown(req):
        return

    self.add_request(req, request_wave)
```

在：

```text
self.add_request()
```

前后增加：

```text
[BRIDGE][INGRESS][CORE_DEQUEUE]

[BRIDGE][INGRESS][SCHED_ADD_DONE]
```

其中：

```text
CORE_DEQUEUE
```

表示：

```text
busy loop
已经从 local input_queue
开始处理该 request
```

而：

```text
SCHED_ADD_DONE
```

表示：

```text
Scheduler.add_request()
已经完成

Request 正式进入：
Scheduler.requests / waiting
```

---

# 21. 最终 Ingress trace 形成的完整证据链

```text
CLIENT_SEND
↓
IPC_RECV
↓
IPC_ENQUEUE
↓
CORE_DEQUEUE
↓
SCHED_ADD_DONE
↓
CALL_BEGIN
↓
QUEUE_SNAPSHOT
```

这条链第一次把：

```text
Frontend request ingress
```

和：

```text
Scheduler visibility
```

动态连接起来。

---

# 22. 最终时间轴：为什么 Call 0 只有 A

最终 trace：

```text
A CLIENT_SEND
ts = 12617784397711815
```

之后：

```text
A IPC_RECV
+0.465 ms

A IPC_ENQUEUE
+0.536 ms

A CORE_DEQUEUE
+0.600 ms

A SCHED_ADD_DONE
+0.653 ms
```

也就是说：

```text
A 从 Frontend SEND
到 Scheduler 可见
只需要约 0.653 ms
```

然后 EngineCore 立即：

```text
Call 0
```

此时 snapshot：

```text
known=[A]
waiting=[A]
```

---

# 23. 最关键证据：B/C 是 Call 0 开始后才被 Frontend SEND

B：

```text
CLIENT_SEND B
约在 A SEND 后 +2.077 ms
```

C：

```text
CLIENT_SEND C
约在 A SEND 后 +4.389 ms
```

而 A：

```text
约 +0.653 ms
已经进入 Scheduler
```

并已经进入：

```text
Call 0
```

所以真实顺序是：

```text
Frontend SEND A
↓
A 到 EngineCore
↓
Scheduler.add_request(A)
↓
Call 0 开始
↓
schedule A
↓
MRV2 execute A

此后才：

Frontend SEND B
Frontend SEND C
```

因此：

> **Call 0 只有 A 的根因根本不是 Scheduler policy，而是 B/C 在该 schedule step 开始时尚未由 Frontend 发出。**

---

# 24. `generate([A,B,C])` 不等于 atomic batch ingress

这是 E3 后续分析得到的一个非常重要的系统认知。

Python 层：

```python
llm.generate(
    [A, B, C]
)
```

看起来是一次 API call。

但 Runtime 实际不是：

```text
generate([A,B,C])
↓
atomic send [A,B,C]
↓
Scheduler waiting=[A,B,C]
```

而表现为：

```text
Frontend 逐 request 处理

A
↓
CLIENT_SEND A

随后 B
↓
CLIENT_SEND B

随后 C
↓
CLIENT_SEND C
```

所以：

> **Python API-level batch boundary 不等于 EngineCore/Scheduler-level admission batch boundary。**

这也是 continuous batching 的重要基础。

---

# 25. 为什么 A 很快，而 B/C 到 Scheduler 晚很多

最初容易误读成：

```text
A:
0.65

B/C:
约 40

所以后续 Request 越来越慢
```

实际不是。

单位首先是：

```text
A ≈ 0.653 ms
```

不是秒。

更关键的是 B/C 的延迟主要不是 IPC。

---

## 25.1 B

Frontend send B 后：

```text
IPC_ENQUEUE B
约 +1.04 ms
```

说明：

```text
Frontend
→ ZMQ
→ input thread
→ input_queue
```

并不慢。

但：

```text
B IPC_ENQUEUE
→ CORE_DEQUEUE
≈ 38.55 ms
```

---

## 25.2 C

同样：

```text
C IPC_ENQUEUE
→ CORE_DEQUEUE
≈ 36.81 ms
```

---

# 26. 为什么 B/C 在 input_queue 中等待约 37～39ms

因为 EngineCoreProc 中存在至少两个不同 execution contexts：

```text
EngineCoreProc
│
├── input thread
│
│     process_input_sockets()
│
│     ZMQ recv
│     preprocess
│     input_queue.put()
│
└── busy-loop thread
      drain input_queue
      Scheduler
      execute
      sample
      batch_queue
      reconcile
```

当 B/C 到来时：

```text
input thread
```

仍然可以继续：

```text
recv B
recv C
enqueue B
enqueue C
```

但 busy-loop 此时正在执行：

```text
Call 0
```

因此：

```text
B/C 已经到 EngineCoreProc
并不等于
B/C 已经对 Scheduler 可见
```

它们先停在：

```text
local input_queue
```

直到 busy-loop 完成 Call 0 当前工作并回来处理 input queue。

---

# 27. E3 暴露出的两级 Request Ingress 状态

可以第一次精确区分：

## Level 1：EngineCoreProc 已经收到

```text
IPC_RECV
IPC_ENQUEUE
```

此时：

```text
Request 已经到 EngineCore process
```

但 Scheduler 可能还不知道。

---

## Level 2：Scheduler ownership 建立

```text
CORE_DEQUEUE
↓
SCHED_ADD_DONE
```

此时才真正：

```text
Scheduler.requests contains request
waiting contains request
```

所以：

```text
Request arrival
```

和：

```text
Scheduler visibility
```

是两个不同的状态边界。

---

# 28. 为什么 vLLM 不等待 B/C 一起到再跑 A

如果系统采用：

```text
batch collection barrier
```

就会变成：

```text
A arrives
↓
等待 B
↓
等待 C
↓
一起 schedule
```

这会人为增加 A 的等待时间。

当前 Runtime 的行为是：

```text
只要已经有可执行 work
↓
立即 schedule

与此同时：
新的 request 可以继续 ingress

下一 step：
动态并入 current batch
```

因此：

```text
Call 0:
[A]

Call 1:
[A,B]

Call 2:
[A,B,C]
```

不是异常，而与 continuous batching 的设计目标一致。

> **Batch 是 step-level dynamic execution view，而不是 API-call-level static group。**

---

# 29. E3 的完整系统图

```text
Frontend Process
────────────────────────────────────

generate([A,B,C])

    │
    ├─ process/send A
    │
    │   CLIENT_SEND A
    │        │
    │        ▼
    │      ZMQ
    │
    │
    │                EngineCoreProc
    │                ─────────────────────────
    │
    │                input thread
    │                IPC_RECV A
    │                    ↓
    │                IPC_ENQUEUE A
    │                    ↓
    │                input_queue=[A]
    │
    │                busy loop
    │                CORE_DEQUEUE A
    │                    ↓
    │                Scheduler.add_request(A)
    │                    ↓
    │                waiting=[A]
    │                    ↓
    │                Call 0
    │                    ↓
    │                Prefill A=45
    │                    ↓
    │                MRV2 execute A
    │
    │
    ├─ process/send B
    │   CLIENT_SEND B
    │        ↓
    │                IPC_RECV B
    │                IPC_ENQUEUE B
    │                input_queue=[B]
    │
    ├─ process/send C
    │   CLIENT_SEND C
    │        ↓
    │                IPC_RECV C
    │                IPC_ENQUEUE C
    │                input_queue=[B,C]
    │
    │                busy loop 此时仍在 Call 0
    │
    │                    ...
    │
    │                Call 0 submit 完成
    │                    ↓
    │                CORE_DEQUEUE B
    │                    ↓
    │                Scheduler.add_request(B)
    │                    ↓
    │                CORE_DEQUEUE C
    │                    ↓
    │                Scheduler.add_request(C)
    │
    │                running=[A]
    │                waiting=[B,C]
    │
    │                    ↓
    │
    │                Call 1
    │
    │                A=1
    │                B=127
    │
    │                    ↓
    │
    │                Call 2
    │                A=1
    │                B=44
    │                C=83
    ▼
```

---

# 30. E3 最终得到的核心结论

## 结论 1：Token budget 是 Scheduler-step global budget

```text
Σ per-request scheduled tokens
<= max_num_batched_tokens
```

---

## 结论 2：Scheduler 先推进 running，再用剩余 budget admission waiting

动态证据：

```text
Call 2:

running:
A=1
B=44

剩 83

waiting:
C=83
```

---

## 结论 3：一个 SchedulerOutput 可以包含多个 Request

例如：

```text
{
    A:1,
    B:44,
    C:83
}
```

---

## 结论 4：同一个 SchedulerOutput 可以混合 Prefill / Decode

Call 2：

```text
A Decode
B Final Prefill
C New/Chunked Prefill
```

Call 3：

```text
A Decode
B Decode
C Chunked Prefill
```

---

## 结论 5：`RUNNING` 不是 Decode 标志

C 在：

```text
running
```

时仍然：

```text
pending=242
scheduled=126
```

显然仍在 Prefill。

---

## 结论 6：Continuous Batching 的 batch membership 是动态的

```text
[A]
→
[A,B]
→
[A,B,C]
→
[B,C]
→
[C]
```

---

## 结论 7：Request 有独立 logical / KV frontier，但共享 compute budget

```text
Global:
token budget

Per-request:
num_computed_tokens
num_in_flight_tokens
KV blocks
output progress
```

---

## 结论 8：batch_queue 保存 batch transaction，不是 request

一个 queue item 可以是：

```text
SchedulerOutput {
    A,
    B,
    C
}
```

---

## 结论 9：Frontend API batch 不等于 Scheduler admission batch

```text
generate([A,B,C])
```

不保证：

```text
Call 0:
waiting=[A,B,C]
```

真实：

```text
Call 0:
known=[A]

Call 1:
known=[A,B,C]
```

---

## 结论 10：Request 到 EngineCoreProc 与 Request 对 Scheduler 可见不是同一个时刻

```text
IPC_ENQUEUE
≠
SCHED_ADD_DONE
```

中间还有：

```text
busy-loop drain input_queue
```

---

## 结论 11：Input thread 与 Engine busy loop 解耦

因此 Call 0 model work 期间：

```text
input thread
仍然可以接收并 enqueue B/C
```

但 Scheduler ownership 要等 busy-loop 回来。

---

# 31. 本实验实际增加的 instrumentation 汇总

| 文件 | 位置 | Trace | 目的 |
|---|---|---|---|
| `scheduler.py` | `schedule()` running loop 前 | `QUEUE_SNAPSHOT` | 看 step 开始时 known/running/waiting |
| `core_client.py` | `SyncMPClient.add_request()` | `CLIENT_SEND` | 看 Frontend 真正 SEND 请求的时刻 |
| `core.py` | `process_input_sockets()` | `IPC_RECV` | 看 EngineCore input thread 收到请求 |
| `core.py` | `process_input_sockets()` | `IPC_ENQUEUE` | 看请求进入 local input_queue |
| `core.py` | `_handle_client_request(ADD)` | `CORE_DEQUEUE` | 看 busy-loop 开始消费该请求 |
| `core.py` | `_handle_client_request(ADD)` | `SCHED_ADD_DONE` | 看请求正式进入 Scheduler ownership |

所有 trace：

```text
只读
只打印
QCACHE_BRIDGE_TRACE gate
```

没有修改：

```text
Scheduler policy
token budget
waiting/running 操作
KV allocation
ModelRunner
GPU kernel
```

---

# 32. E1 → E2 → E3 的认知递进

## E1

```text
一个 Request
完整 Prefill
↓
Decode
```

重点建立：

```text
schedule-time commit
in-flight
async batch_queue
reconcile
placeholder
```

---

## E2

```text
一个长 Request
↓
Chunked Prefill
↓
Decode
```

重点建立：

```text
token budget
logical frontier
KV block growth
RUNNING != Decode
```

---

## E3

```text
多个 Request
↓
共享 global token budget
↓
不同 execution phase 混合
↓
dynamic batch membership
```

进一步建立：

```text
waiting / running admission
multi-request SchedulerOutput
continuous batching
request ingress
input thread / busy-loop decoupling
Scheduler visibility boundary
```

因此三轮联合形成：

```text
Request lifecycle
+
Scheduler token abstraction
+
KV allocation
+
Async execution pipeline
+
Continuous batching
+
Ingress path
```

的一条完整控制面动态主链。

---

# 33. E3 PASS 判定

原实验要求：

```text
3 requests
约 32 / 160 / 320 tokens
budget=128

至少一个 step 包含多个 requests

长 Prompt Prefill
与其他 request work 交错
```

实际：

```text
45 / 171 / 325

Call 1:
A + B

Call 2:
A + B + C

Call 3:
A + B + C
```

且：

```text
Call 2:
Decode + Prefill + Prefill

Call 3:
Decode + Decode + Chunked Prefill
```

完全满足。

此外补证 trace 还解释了 Call 0 的 request arrival 问题，没有留下无法解释的字段。

因此：

```text
E3 = PASS
```

无需继续增加新的 E3 trace。

---

# 34. 当前不继续追的问题

以下问题不属于 E3 blocker：

```text
Frontend 每个 request SEND 之间约 2ms
具体花在哪个 frontend 函数

Input queue 的更细粒度 polling / scheduling timing

ZMQ 内部实现

OS thread scheduling

GPU kernel execution overlap
```

如果以后专门研究 request ingress latency，再继续：

```text
LLM.generate
↓
_add_completion_requests
↓
LLMEngine.add_request
↓
SyncMPClient.add_request
```

向前细化即可。

当前 E3 的目标已经全部完成。

---

# 35. 一句话收尾

E3 最核心的结论不是：

> “三个 Request 一起跑了。”

而是：

> **vLLM V1 Scheduler 每个 step 用一个 global token budget 推进多个独立 Request frontier，并允许 Decode、Final Prefill、Chunked Prefill 同时出现在一个 SchedulerOutput；新的 Request 通过独立 ingress path 持续加入 Scheduler，因此 batch membership 是 step-level 动态变化的，这正是 Continuous Batching 的控制面本质。**

