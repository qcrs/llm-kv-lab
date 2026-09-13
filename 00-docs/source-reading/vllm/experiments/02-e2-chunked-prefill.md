# E2-vLLM-Chunked-Prefill动态调度与KV增长实验

> 实验状态：**PASS / 正式收尾**  
> 固定版本：vLLM v0.26.0  
> commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> 实验日期：2026-08-19  
> 实验目录：`04-experiments/vllm-bridge/raw/e2-chunked-prefill/`  
> 原始日志：`04-experiments/vllm-bridge/raw/e2-chunked-prefill/startup.log`  
> 实验脚本：`04-experiments/vllm-bridge/scripts/e2_chunked_prefill.py`  
> 当前主路径：Offline LLM / V1 Engine / V2 Model Runner / TP=1 / PP=1 / DP=1 / Async Scheduling / FlashAttention 2

---

# 1. 实验目的

E1 已经动态确认了单请求在 Prompt 可以一次放入 token budget 时的完整运行链：

```text
完整 Prefill
↓
sampling
↓
Decode
↓
Async batch queue
↓
Scheduler result reconciliation
```

E2 不再重复验证 E1，而是只引入一个新的核心条件：

> **Prompt 的待计算 token 数大于单个 Scheduler step 可使用的 token budget。**

实验要动态确认：

```text
长 Prompt
↓
无法在一个 Scheduler step 完成
↓
Scheduler 将 Prefill 拆成多个 chunk
↓
num_computed_tokens 逐 chunk 推进
↓
KV physical blocks 随 logical frontier 扩张
↓
最后一个 Prefill chunk 完成
↓
进入 Decode
```

同时验证一个非常重要的 vLLM V1 Scheduler 心智模型：

> **Prefill、Chunked Prefill 和 Decode 在 Scheduler 层并不是三套完全独立的调度流程，而是统一表达为“这个 request 本轮需要执行多少 token”。**

---

# 2. 为什么需要 E2

如果只做 E1，很容易形成一个过于简单的理解：

```text
NEW = Prefill
RUNNING = Decode
```

或者：

```text
Prefill = 一次性把整个 Prompt 算完
```

生产 Serving Runtime 中并不是这样。

当 Prompt 很长、当前 step 的 token budget 较小时，一个 request 会经历：

```text
NEW
↓
第一次 Prefill chunk

RUNNING
↓
第二个 Prefill chunk

RUNNING
↓
第三个 Prefill chunk

...

RUNNING
↓
Decode
```

因此 E2 的价值是动态证明两个事实：

```text
Request lifecycle state
和
Prefill / Decode workload phase
不是同一维度
```

以及：

```text
Chunked Prefill
可以由：
pending token work
+
step token budget
自然产生
```

这对后续理解 Scheduler、KV admission、连续批处理以及 QCache 的 block lifecycle 都是基础。

---

# 3. 固定环境

本实验继续沿用 E1 的固定环境，不重新改变无关变量。

```text
vLLM version: v0.26.0
commit: 568afb3a13806beb53bb2e6bd518269357b237c0
repo: ~/learning/llm-kv-lab/third_party/vllm
Python env: qcache
GPU: A100 80GB PCIe
model: /data/models/Qwen3-0.6B
Frontend: Offline LLM
Engine: V1 Engine
Model Runner: V2 Model Runner
TP=1
PP=1
DP=1
dtype=BF16
block_size=16
max_model_len=2048
max_num_seqs=4
gpu_memory_utilization=0.2
enforce_eager=True
enable_prefix_caching=False
max_tokens=3
ignore_eos=True
```

启动环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=1
export QCACHE_BRIDGE_TRACE=1
```

其中 `QCACHE_BRIDGE_TRACE` 是本学习分支自行定义的 instrumentation 开关，不是 vLLM 官方环境变量。

---

# 4. 实验变量

E2 尽可能保持 E1 不变，只操纵与 Chunked Prefill 直接相关的变量。

## E1

```text
Prompt=38 tokens
max_num_batched_tokens=256
```

因此整个 Prompt 可以一次 schedule。

## E2

真实 Prompt：

```text
365 tokens
```

Scheduler token budget：

```text
max_num_batched_tokens=64
```

因此：

```text
365 > 64
```

Prompt 不可能在一个 Scheduler step 完成。

理论上至少需要：

```text
ceil(365 / 64) = 6
```

个 Prefill scheduling steps。

理论 chunk：

```text
64
64
64
64
64
45
```

E2 的核心就是验证 Runtime 是否真实出现这一行为。

---

# 5. 实验脚本

实验脚本：

```text
04-experiments/vllm-bridge/scripts/e2_chunked_prefill.py
```

核心逻辑分为三部分。

## 5.1 构造长 Prompt

通过 Qwen3 tokenizer 构造约 350 token 的文本，最终真实 tokenizer 结果为：

```text
constructed_prompt_tokens_no_special=365
```

vLLM 最终返回：

```text
prompt_tokens=365
```

因此后续分析统一使用 `P=365` 作为真实 Prompt 长度。

## 5.2 强制 token budget

关键配置：

```python
max_num_batched_tokens=64
```

它把单个 Scheduler step 的总 token compute quota 限制为 64。

## 5.3 保留 3 个 output tokens

```python
SamplingParams(
    temperature=0.0,
    max_tokens=3,
    ignore_eos=True,
)
```

这样在 Chunked Prefill 完成后仍能看到 Prefill → Decode 的转换，并继续验证 E1 已建立的 output placeholder / async reconciliation 语义。

---

# 6. Instrumentation

E2 没有重新大规模增加 instrumentation，而是复用 E1 已经建立的 trace。

这是刻意的实验设计：

> **E2 的目标是改变 workload，不是改变观测系统。**

## 6.1 EngineCore trace

文件：

```text
vllm/v1/engine/core.py
```

核心事件：

```text
[BRIDGE][CORE][INIT]
[BRIDGE][CORE][CALL_BEGIN]
[BRIDGE][CORE][SCHEDULE_BEGIN]
[BRIDGE][CORE][SCHEDULE_END]
[BRIDGE][CORE][EXEC_SUBMIT_BEGIN]
[BRIDGE][CORE][EXEC_SUBMIT_RETURN]
[BRIDGE][CORE][SAMPLE_SUBMIT_BEGIN]
[BRIDGE][CORE][SAMPLE_SUBMIT_RETURN]
[BRIDGE][CORE][QUEUE_PUSH]
[BRIDGE][CORE][YIELD_WITHOUT_CONSUME]
[BRIDGE][CORE][QUEUE_CONSUME_BEGIN]
[BRIDGE][CORE][QUEUE_POP]
[BRIDGE][CORE][WAIT_RESULT_BEGIN]
[BRIDGE][CORE][WAIT_RESULT_END]
[BRIDGE][CORE][RECONCILE_BEGIN]
[BRIDGE][CORE][RECONCILE_END]
```

作用：观察 schedule → execute submit → sample submit → queue → result consume → Scheduler reconciliation。

## 6.2 Scheduler trace

文件：

```text
vllm/v1/core/sched/scheduler.py
```

事件：

```text
[BRIDGE][SCHED][step=N]
[BRIDGE][SCHED][BEFORE_COMMIT]
[BRIDGE][SCHED][AFTER_COMMIT]
[BRIDGE][SCHED][SETTLE]
[BRIDGE][SCHED][TOKEN_APPEND]
```

E2 中尤其重要的是：

```text
budget_start
budget_remaining
num_tokens
num_tokens_with_spec
placeholders
num_computed
num_in_flight
pending
scheduled
new_block_ids
```

## 6.3 MRV2 trace

文件：

```text
vllm/v1/worker/gpu/model_runner.py
```

事件：

```text
[BRIDGE][MRV2][EXECUTE_BEGIN]
[BRIDGE][MRV2][SAMPLE_BEGIN]
```

作用：确认 SchedulerOutput 中的 `total_scheduled/per_req` 确实进入当前 V2 ModelRunner execution path。

---

# 7. Trace 字段含义

## `num_tokens`

Request 当前真正拥有的 token 数：

```text
Prompt + 已经正式 append 的 output token
```

E2 Prefill 阶段一直是 365，直到第一个 sampled output 正式进入 Request。

## `num_computed`

Scheduler logical computation frontier。

它表示 Scheduler 已经把多少 token work 视为“已经发出去，不应该再次重复 schedule”。Async Scheduling 下，它可以领先于 result reconciliation。

因此不能解释成“GPU 已经确认完成多少 token”。

## `num_in_flight`

已经 schedule/commit，但对应 SchedulerOutput 还没有完成 `update_from_output()` reconciliation 的 token 数。

它描述的是 outstanding Scheduler transaction work，而不是 GPU 当前物理上正在运行的 token 数。

## `pending`

当前 Scheduler 仍然可以继续安排的 logical token work。

当前普通 async 路径中可以用下面的心智模型理解：

```text
pending ≈ num_tokens + num_output_placeholders - num_computed_tokens
```

## `scheduled`

当前这一个 Scheduler step 最终安排给该 Request 的 token 数。这是单步决策，不是 Request 长期状态。

## `placeholders`

Async Scheduling 为尚未正式 append、但 Scheduler 已经知道未来会存在的 output position 保留的逻辑位置。

它允许 sample result 尚未 reconcile 时，Scheduler 仍然提前构造下一份 Decode work。

## `new_block_ids`

当前 Scheduler step 为该 Request 新增的 physical KV block IDs，表示本轮 KV coverage 扩张所产生的 physical block delta。

## `budget_start / budget_remaining`

当前 Scheduler step 开始时的 token compute quota，以及完成本轮 scheduling 后剩余的 quota。

E2 中 `budget_start=64`。完整 64-token Prefill chunk 会把预算耗尽；最后一个 45-token chunk 后 `budget_remaining=19`。

---

# 8. 真实运行配置确认

启动日志明确确认：

```text
Chunked prefill is enabled with max_num_batched_tokens=64.
Asynchronous scheduling is enabled.
```

EngineCore trace：

```text
async_scheduling=True
use_v2_model_runner=True
pp_size=1
max_concurrent_batches=2
batch_queue_size=2
batch_queue_enabled=True
step_fn=step_with_batch_queue
```

因此 E2 的真实 Runtime 仍然是：

```text
V1 Engine
+
V2 Model Runner
+
Async Scheduling
+
EngineCore.step_with_batch_queue()
```

Attention 启动日志确认：

```text
FLASH_ATTN
FlashAttention version 2
```

GPU memory 配置：

```text
gpu_memory_utilization=0.2
Available KV cache memory=14.61 GiB
GPU KV cache size=136,768 tokens
```

当前实验只有单请求 365-token Prompt，因此不存在 KV pressure。

这一点非常重要：

> E2 观察到的 chunking 来自 token budget，而不是 KV 不足或 preemption。

---

# 9. 真实运行结果

最终：

```text
prompt_tokens=365
output_token_ids=[20286, 4128, 1614]
num_output_tokens=3
output_text=' Large language model'
```

因此最终 known token frontier：

```text
365 Prompt + 3 output = 368
```

---

# 10. Scheduler 状态变化总表

| Call | 阶段 | kind | num_tokens(before) | placeholders(before) | computed(before) | in_flight(before) | pending | scheduled | computed(after) | in_flight(after) | new_block_ids |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | Prefill chunk 1 | NEW | 365 | 0 | 0 | 0 | 365 | 64 | 64 | 64 | 1-4 |
| 1 | Prefill chunk 2 | RUNNING | 365 | 0 | 64 | 64 | 301 | 64 | 128 | 128 | 5-8 |
| 2 | Prefill chunk 3 | RUNNING | 365 | 0 | 128 | 64 | 237 | 64 | 192 | 128 | 9-12 |
| 3 | Prefill chunk 4 | RUNNING | 365 | 0 | 192 | 64 | 173 | 64 | 256 | 128 | 13-16 |
| 4 | Prefill chunk 5 | RUNNING | 365 | 0 | 256 | 64 | 109 | 64 | 320 | 128 | 17-20 |
| 5 | Prefill chunk 6 | RUNNING | 365 | 0 | 320 | 64 | 45 | 45 | 365 | 109 | 21-23 |
| 6 | Decode | RUNNING | 365 | 1 | 365 | 45 | 1 | 1 | 366 | 46 | None |
| 7 | Decode | RUNNING | 366 | 1 | 366 | 1 | 1 | 1 | 367 | 2 | None |
| 8 | drain/result consume | — | — | — | — | — | — | 0 | — | — | — |

这里最重要的是看出三个 frontier：

```text
known token frontier          = num_tokens
logical computation frontier  = num_computed_tokens
unreconciled work frontier    = num_in_flight_tokens
```

三者在 async runtime 中不是同一个值。

---

# 11. 按调用时间顺序分析

## 11.1 Call 0：第一个 Prefill chunk

初始：

```text
num_tokens=365
num_computed=0
num_in_flight=0
placeholders=0
pending=365
token_budget=64
```

因此：

```text
scheduled=min(365,64)=64
```

真实：

```text
kind=NEW
scheduled=64
new_block_ids=[1,2,3,4]
```

commit 后：

```text
num_computed: 0 → 64
num_in_flight: 0 → 64
```

结果进入 batch queue：

```text
queue=[A]
```

因为 queue 尚未满，所以 EngineCore 没有立即 reconcile A。

## 11.2 Call 1：Request 已经 RUNNING，但仍然是 Prefill

Call 1：

```text
kind=RUNNING
num_tokens=365
num_computed=64
num_in_flight=64
pending=301
scheduled=64
```

虽然 `kind=RUNNING`，但 Prompt 只推进到 64/365，显然仍然处于 Chunked Prefill。

因此动态证明：

```text
RUNNING ≠ Decode
```

更准确：

```text
NEW / RUNNING = Request lifecycle / scheduling state
Prefill / Decode = 当前 token execution workload
```

Call 1 commit：

```text
num_computed: 64 → 128
num_in_flight: 64 → 128
```

此时 A=64 outstanding，B=64 outstanding。

queue 达到 capacity 后：

```text
[A,B]
↓
pop A
↓
A.result()
↓
Scheduler.update_from_output(A)
```

SETTLE：

```text
in_flight: 128 → 64
```

留下 B outstanding。

## 11.3 Call 2～4：稳定 Chunked Prefill pipeline

之后每一轮都形成：

```text
上一 chunk outstanding
+
schedule 下一 chunk
↓
queue 满
↓
consume oldest chunk
```

logical computation frontier：

```text
Call 2: 128 → 192
Call 3: 192 → 256
Call 4: 256 → 320
```

每轮 `scheduled=64`。

这说明 `schedule chunk N+1` 和 `reconcile chunk N` 可以重叠推进。

## 11.4 Call 5：最后一个 Prefill chunk

Call 5 BEFORE：

```text
num_tokens=365
num_computed=320
placeholders=0
pending=45
```

虽然 `token_budget=64`，但只剩 45 个 Prompt tokens，因此：

```text
scheduled=45
budget_remaining=19
```

commit：

```text
num_computed: 320 → 365
placeholders: 0 → 1
```

这是关键边界：logical computation frontier 已覆盖完整 Prompt，系统开始为未来 sampled token 建立 placeholder。

## 11.5 Call 6：最后 Prefill result 尚未 reconcile，就提前 schedule Decode

Call 6 BEFORE：

```text
num_tokens=365
placeholders=1
num_computed=365
num_in_flight=45
```

此时 O0 还没有 `TOKEN_APPEND`，但：

```text
pending=365+1-365=1
```

所以 Scheduler 已经可以：

```text
scheduled=1
```

随后 queue：

```text
[FinalPrefill45, Decode1]
```

EngineCore pop 最老的 `FinalPrefill45`。

SETTLE：

```text
in_flight: 46 → 1
```

然后真正：

```text
TOKEN_APPEND
num_tokens: 365 → 366
num_output_tokens: 0 → 1
```

也就是 O0 此刻才正式进入 Request。

## 11.6 Call 7：第二个 Decode work

```text
num_tokens=366
num_computed=366
placeholders=1
num_in_flight=1
pending=1
scheduled=1
```

commit：

```text
num_computed: 366 → 367
```

随后 consume Call 6 result：

```text
TOKEN_APPEND
num_tokens: 366 → 367
num_output_tokens: 1 → 2
```

## 11.7 Call 8：不再提交 model work，只消费最后结果

Call 8：

```text
budget_start=64
budget_remaining=64
total_scheduled=0
```

这不表示异常。

此时已经没有新的 model work 需要提交，但 queue 中还有之前的 outstanding result，因此 Call 8 主要用于 drain。

消费 Call 7 后：

```text
TOKEN_APPEND
num_tokens: 367 → 368
num_output_tokens: 2 → 3
num_computed: 367
```

最终：

```text
num_tokens=368
num_computed_tokens=367
```

---

# 12. 为什么生成 3 个 output token，却只有两个 `scheduled=1`

最终输出 O0、O1、O2，但 forward 关系是：

```text
最后一个 Prompt token forward
↓
sample O0

forward(O0)
↓
sample O1

forward(O1)
↓
sample O2
```

因此 3 个 sampled output tokens 不要求 3 次 output-token Decode forward，只需要 2 次 Decode input forward。

最后一个 O2 已经是最终结果，不需要再 `forward(O2)`。

所以：

```text
num_tokens=365+3=368
num_computed=365+2=367
```

与 E1 的最终 frontier 语义完全一致。

---

# 13. KV Block 变化

当前：

```text
block_size=16
```

365 个 Prompt token 最终需要：

```text
ceil(365/16)=23
```

个 physical blocks。

真实 allocation：

```text
Call 0: [1,2,3,4]
Call 1: [5,6,7,8]
Call 2: [9,10,11,12]
Call 3: [13,14,15,16]
Call 4: [17,18,19,20]
Call 5: [21,22,23]
```

总计 23 blocks，与理论完全一致。

## 13.1 为什么前五个 chunk 每次 4 blocks

完整 chunk：

```text
64 tokens / block_size 16 = 4 blocks
```

实际也确实每轮新增 4 个。

## 13.2 为什么最后 45 tokens 新增 3 blocks

```text
ceil(45/16)=3
```

真实 `[21,22,23]`。

## 13.3 Decode 为什么没有新 block

23 个 block 能覆盖：

```text
23 × 16 = 368 slots
```

365-token Prompt 后最后一个 block 仍有 3 个 slot 可用，因此后面的两个 `scheduled=1` 都看到 `new_block_ids=None`。

这不是 Decode 不需要 KV，而是当前 Request 已经持有的最后一个 physical block 还有可写 slot。

---

# 14. E2 的完整控制面链路

```text
Request
num_tokens=365
num_computed=0
        │
        ▼
Scheduler.schedule()
        │
        │ pending work=365
        │ token budget=64
        ▼
scheduled=64
        │
        ▼
KVCacheManager.allocate_slots()
        │
        ▼
physical blocks [1,2,3,4]
        │
        ▼
SchedulerOutput
        │
        ▼
Executor / Worker / MRV2
        │
        ▼
64-token model work
        │
        ▼
AsyncOutputFuture
        │
        ▼
batch_queue
        │
        │ Scheduler 已提前推进 logical frontier
        ▼
下一次 schedule()
        │
        ▼
再安排下一 chunk
        │
        ▼
旧 batch result reconcile
        │
        ▼
循环推进
```

直到：

```text
computed:
0→64→128→192→256→320→365
```

然后：

```text
Prompt logical frontier complete
↓
placeholder=1
↓
scheduled=1
↓
Decode
```

---

# 15. Async pipeline 与 E2 的关系

E2 不是重新研究 async scheduling。

E1 已经证明 batch queue 的基本 pipeline。E2 的新增价值是证明：

> **同样的 async transaction pipeline 不只用于 Decode，也同样包裹 Chunked Prefill work。**

因此连续 Prefill chunks 可以表现为：

```text
schedule chunk N+1
```

发生在：

```text
chunk N result reconciliation
```

之前。

Scheduler 依赖 schedule-time commit + `num_in_flight` tracking 避免重复 schedule 已发出的 Prefill positions。

---

# 16. 静态源码认知与动态证据对应

## 16.1 Token budget 裁剪

静态心智模型：

```text
pending work
↓
min(pending, token_budget)
↓
num_scheduled_tokens
```

动态：

```text
pending=365 budget=64 scheduled=64
pending=301 budget=64 scheduled=64
...
pending=45  budget=64 scheduled=45
```

完全吻合。

## 16.2 `RUNNING` 不是 Decode

静态理解：

```text
waiting/running = request lifecycle
Prefill/Decode  = execution progress
```

动态 Call 1～5 全部 `kind=RUNNING`，但仍处于 Prompt 计算阶段。

## 16.3 Scheduler logical frontier

静态：schedule-time commit 会提前推进 `num_computed_tokens`。

动态 Call 1 BEFORE：

```text
computed=64
in_flight=64
```

说明前 64 token 尚未 reconcile，但 Scheduler 已把 logical frontier 推到 64。

## 16.4 KV coverage

静态：target token coverage → 需要多少 physical blocks → allocate missing blocks。

动态：365 tokens → 23 physical blocks，并逐 chunk 增长。

## 16.5 Result reconciliation

静态：

```text
update_from_output()
↓
num_in_flight -= settled batch tokens
```

动态典型：

```text
in_flight_before=128
settled=64
in_flight_after=64
```

吻合。

---

# 17. E2 得到的核心结论

1. **Chunked Prefill 本质上是 token work 被 step budget 截断。** 当前 365-token Prompt 在 64-token budget 下形成 `64×5+45`。
2. **V1 Scheduler 用统一 token progress 表达 Prefill / Chunked Prefill / Decode。**
3. **RUNNING 是生命周期状态，不是 Decode 标记。** Call 1～5 都是 RUNNING，但仍在 Prefill。
4. **logical computation frontier 可以领先 result reconciliation。**
5. **`num_in_flight_tokens` 描述 unreconciled scheduled work，不是 GPU utilization。**
6. **KV physical block ownership 随 logical frontier 扩张。** 365 tokens 对应 23 blocks。
7. **最后 Prefill chunk 是 Prefill→generation 的关键边界。** Call 5 中 computed 到 365，同时 placeholder 0→1。
8. **Async placeholder 允许 Prefill result 未 reconcile 时提前安排 Decode。**

---

# 18. 与 E1 的衔接

E1：

```text
Prompt=38
budget=256
```

所以 Prefill 一次完成。

E2：

```text
Prompt=365
budget=64
```

所以 Prefill 分六次完成。

但二者使用的是同一套核心状态语义：

```text
pending
↓
scheduled
↓
schedule-time commit
↓
in-flight
↓
execute/sample submit
↓
queue
↓
reconcile
↓
token append
```

因此 E1 + E2 联合证明：

> **V1 Scheduler 统一的是 token scheduling abstraction；Prefill/Decode 的 workload 差异仍然存在，并在 execution metadata / query length / sequence length / kernel workload 中继续体现。**

---

# 19. 是否还需要追加 E2 实验

结论：

> **不需要。E2 Runtime 行为实验正式停止。**

当前已经满足 E2 的所有核心判定条件：

```text
1. Prompt >= 256
   实际 365

2. max_num_batched_tokens=64
   已确认

3. 至少 4 个 Prefill Scheduler step
   实际 6 个

4. 每个 Prefill scheduled <= 64
   实际 64,64,64,64,64,45

5. num_computed_tokens 单调推进
   0→64→128→192→256→320→365

6. 最后 Prompt chunk 后进入 Decode
   已确认

7. NEW first chunk / RUNNING subsequent chunks
   已确认

8. RUNNING ≠ Decode
   已动态证明

9. KV block 随 frontier 增长
   1~23，与 ceil(365/16)=23 一致

10. Async commit / in-flight / reconcile
    已解释且无阻断未知字段
```

## 19.1 为什么不补 `free block count`

早期 Scheduler 动态实验计划中曾列出 `free block count`，但当前 E2 不需要为了这一字段重新增加 instrumentation。

原因：

```text
E2 研究：token-budget-induced chunking
E4 研究：KV pressure / admission / preemption
```

当前日志已经确认 KV cache capacity=136,768 tokens，而 workload 只有 365 Prompt tokens，所以 E2 不存在 KV capacity ambiguity。

`new_block_ids` 已经足够证明 logical frontier → physical block growth。free block count 留到 KV-pressure 实验才真正有解释价值。

## 19.2 原手册的 JSONL / scheduler-steps.csv

原三周手册把 E1/E2/E3 JSONL 与 `scheduler-steps.csv` 列为整个 Scheduler 动态实验组的产物。

当前 E2 已有完整 raw trace，因此：

> **这属于实验产物整理，不是新的 E2 Runtime 实验。**

建议等 E3 完成后统一把 E1/E2/E3 抽取成：

```text
04-experiments/vllm-bridge/processed/scheduler-steps.csv
```

避免现在为了格式转换打断实验主线。

---

# 20. 对后续实验意味着什么

E2 已经完成单请求维度上的：

```text
long pending work
+
token budget
+
KV growth
+
async progress
```

动态验证。

它没有回答、也不应该回答：

```text
多个 request 如何共享一个 token budget
多个 request 如何同时出现在 SchedulerOutput
一个 request 的 chunked prefill 如何与另一个 request 的 work 交错
KV 压力下如何 admission/preempt
Prefix Cache 如何改变 computed frontier
```

这些属于后续独立实验。

因此当前单请求 Scheduler 主线已经稳定：

```text
E1: 完整 Prefill → Decode
E2: Chunked Prefill → Decode
```

不需要继续对 E2 增加边界 case。

---

# 21. 仍未解决的问题

以下问题明确不是 E2 blocker。

## 21.1 Sampled-token handoff

Async runtime 中，当 Call N sample 得到 `O_t`，但 Scheduler 尚未 `update_from_output()/append O_t` 时，Call N+1 MRV2 如何取得真实 `O_t` 作为下一次 input token？

后续进入：

```text
MRV2 RequestState
model_state
InputBatch
prepare_inputs
sampled-token handoff
```

时再动态验证。

## 21.2 Token → physical slot 数值验证

当前已经证明 logical frontier → block IDs，但还没有逐 token 验证：

```text
position
→ logical block
→ physical block
→ slot_mapping
```

这一项属于后续 ModelRunner / Attention metadata 实验。

## 21.3 KV pressure

E2 没有研究 free blocks 不足、allocate failure、preemption、watermark、admission。这是独立 KV-pressure 实验。

## 21.4 Prefix reuse

当前 `enable_prefix_caching=False`，所以 E2 没有研究 prefix hit 如何改变 `num_computed_tokens/num_new_tokens/block ownership`。这属于 Prefix Cache 专门实验。

---

# 22. E2 最终结论图

```text
Prompt = 365
max_num_batched_tokens = 64
block_size = 16

Call 0
NEW
Prefill 64
blocks 1~4
0 → 64
   ↓
Call 1
RUNNING
Prefill 64
blocks 5~8
64 → 128
   ↓
Call 2
RUNNING
Prefill 64
blocks 9~12
128 → 192
   ↓
Call 3
RUNNING
Prefill 64
blocks 13~16
192 → 256
   ↓
Call 4
RUNNING
Prefill 64
blocks 17~20
256 → 320
   ↓
Call 5
RUNNING
Prefill 45
blocks 21~23
320 → 365
placeholder 0 → 1
   │
   │ Prompt complete
   ▼
Call 6
RUNNING
Decode 1
no new block
365 → 366
同时 consume Call 5 / append O0
   ↓
Call 7
RUNNING
Decode 1
no new block
366 → 367
同时 consume Call 6 / append O1
   ↓
Call 8
scheduled=0
consume Call 7 / append O2
   ↓
Final

num_tokens=365+3=368
num_computed_tokens=365+2=367
```

---

# 23. PASS 判定

E2：

```text
Forced Chunked Prefill
+
Scheduler logical frontier
+
KV physical block growth
+
Prefill→Decode transition
```

已全部取得动态证据。

最终状态：

```text
E2 = PASS
```

无需继续添加 trace，也无需增加新的 E2 case。

后续如果需要快速恢复 E2，只需要记住：

```text
365 = 64×5 + 45

computed:
0→64→128→192→256→320→365

KV blocks:
0→4→8→12→16→20→23

RUNNING ≠ Decode

最后 Prefill chunk：
computed reaches prompt frontier
+
placeholder appears
↓
generation pipeline begins
```

这就是 E2 的全部核心价值。
