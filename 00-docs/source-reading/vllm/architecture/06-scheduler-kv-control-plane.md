# vLLM 源码桥接笔记：Scheduler 与 KV 控制面主链

> 固定版本：vLLM v0.26.0  
> commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> 当前配置：TP=1 / PP=1 / DP=1 / V2 Model Runner / Offline LLM  
> 模型：`/data/models/Qwen3-0.6B`  
> 目标方向：LLM inference / KV Cache  
> 本文用途：**直接承接 `03-vLLM-ModelRunner执行桥与后续追踪路线.md` 第 77 节。** 前一篇已经提前进入 `GPUModelRunnerV2`，确认了执行侧的形状；从这里开始暂停继续向下钻 ModelRunner，回到 Scheduler / KV 控制面，从头补齐真正决定 `block_table` 上游状态的生产级链路。


> **MRV1 / MRV2 路径校正（重要）**  
> 当前固定的是 **vLLM V1 Engine + V2 Model Runner（MRV2）**。  
> 两个“V1/V2”不是同一层级：
>
> ```text
> vLLM V1
> = Engine / Scheduler / KV control plane 这一代整体架构
>
> Model Runner V1 / V2
> = vLLM V1 Engine 内部 execution-side 的两套实现
> ```
>
> 当前源码中必须严格区分：
>
> ```text
> MRV1:
> vllm/v1/worker/gpu_model_runner.py
> → CachedRequestState
> → persistent InputBatch 与 step input 强耦合
>
> MRV2:
> vllm/v1/worker/gpu/model_runner.py
> → req_states / RequestState
> → model_state
> → persistent request state 与 per-step InputBatch 分离
> ```
>
> 因此本文此前凡是把：
>
> ```text
> MRV2 GPUModelRunner
> CachedRequestState
> self.requests
> persistent self.input_batch
> ```
>
> 当成当前 MRV2 主路径的地方，均以本校正版后文为准。  
> Scheduler / KVCacheManager / BlockPool 主链不因此推翻；错误主要集中在 `SchedulerOutput` 之后的 execution-side 路径。

---

# 78. 为什么从 ModelRunner 暂停，重新回到 Scheduler / KV 控制面

前一阶段已经确认真实的高层运行链：

```text
EngineCore
↓
Executor
↓
Worker
↓
Model Runner
```

但这里必须做一次版本校正。

当前固定配置是：

```text
vLLM V1 Engine
+
V2 Model Runner（MRV2）
```

因此当前真正需要继续追的 execution-side 实现是：

```text
vllm/v1/worker/gpu/model_runner.py
```

而不是：

```text
vllm/v1/worker/gpu_model_runner.py
```

后者是 Model Runner V1（MRV1）的实现。

此前从 MRV1 看到的：

```text
CachedRequestState
persistent self.input_batch
_update_states()
```

虽然帮助我们理解了“为什么 serving runtime 需要保存跨 step 状态”，但不能继续当成当前 MRV2 的真实主路径。

MRV2 的核心状态模型已经变成：

```text
SchedulerOutput
↓
MRV2 GPUModelRunner
↓
finish_requests()
add_requests()
update continuing requests
↓
req_states / RequestState
+
model_state
↓
per-step InputBatch / GPU input metadata
↓
model forward
```

官方 MRV2 设计的核心变化之一，就是：

```text
MRV1:
persistent state
和
per-step model input
耦合在 Persistent InputBatch 中

MRV2:
persistent request state
和
per-step InputBatch
分离
```

因此我们暂停 execution side、回到 Scheduler/KV 控制面的理由仍然成立：

```text
block IDs 谁分配？
本轮为什么算 N tokens？
preemption 为什么发生？
SchedulerOutput 怎么形成？
```

这些仍然属于 Scheduler / KV control plane。

后续正确主链改为：

```text
Request progress
↓
Scheduler.schedule()
↓
num_scheduled_tokens / token budget
↓
KVCacheManager.allocate_slots()
↓
Coordinator
↓
SingleTypeKVCacheManager
↓
BlockPool
↓
physical block ID
↓
SchedulerOutput
↓
MRV2 GPUModelRunner
   vllm/v1/worker/gpu/model_runner.py
↓
req_states / model_state
↓
per-step input metadata / block table / slot mapping
↓
Attention Backend
```

本文控制面部分继续有效；execution-side 从这里开始以 MRV2 实现重新追踪。

---

# 79. 先重新确认 EngineCore 的主循环：Scheduler 是有状态控制器

`EngineCore.step()` 的核心主循环已经确认：

```python
if not self.scheduler.has_requests():
    return {}, False

scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)
...
model_output = future.result()
...
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output,
    model_output,
)
```

第一遍不要陷入 async / batch queue 分支，只看闭环：

```text
Scheduler current state
        ↓
schedule()
        ↓
SchedulerOutput
        ↓
Executor / Worker / ModelRunner
        ↓
ModelRunnerOutput
        ↓
Scheduler.update_from_output()
        ↓
Scheduler next state
```

因此 Scheduler 不是：

```text
输入 Request → 一次性返回一个 batch
```

而是一个持续存在的 **stateful controller**：

```text
State_t
↓
Decision_t
↓
Execution_t
↓
Result_t
↓
State_t+1
```

这件事是理解后面所有变量的基础。

例如：

```text
num_computed_tokens
```

不是一个临时计算值，而是 Request 在多轮 Scheduler step 中不断演进的执行进度状态；

而：

```text
num_scheduled_tokens
```

描述的是某一个 step 的本轮决策。

所以必须区分：

```text
长期 Request state
vs
单步 Scheduling decision
```

---

# 80. Scheduler 的对象所有权：它到底持有什么

从 `Scheduler.__init__()` 可以把当前主线状态压缩为四类。

## 80.1 Request lifecycle state

```python
self.requests: dict[str, Request] = {}
self.waiting = create_request_queue(self.policy)
self.running: list[Request] = []
self.finished_req_ids: set[str] = set()
self.reset_preempted_req_ids: set[str] = set()
```

因此 Scheduler 明确 owns：

```text
requests
waiting
running
finished/preempted bookkeeping
```

第一层可以理解为：

```text
Scheduler
├── requests: request_id → Request
├── waiting: 待 admission / 待重新调度
└── running: 已进入 active runtime 的 Requests
```

## 80.2 Scheduling constraints

```python
self.max_num_running_reqs = self.scheduler_config.max_num_seqs

self.max_num_scheduled_tokens = (
    self.scheduler_config.max_num_scheduled_tokens
    if self.scheduler_config.max_num_scheduled_tokens is not None
    else self.scheduler_config.max_num_batched_tokens
)

self.max_model_len = vllm_config.model_config.max_model_len
```

它们分别约束：

```text
max_num_running_reqs
→ active request 数量上限

max_num_scheduled_tokens
→ 一个 Scheduler step 的 token compute 总预算

max_model_len
→ 单个 Request 的模型长度边界
```

## 80.3 KV resource state

Scheduler 内直接创建并持有：

```python
self.kv_cache_manager = KVCacheManager(...)
```

因此关系是：

```text
Scheduler HAS-A KVCacheManager
```

不是 Scheduler 自己直接维护 free block queue，也不是 ModelRunner 自己决定 block allocation。

## 80.4 Feature-specific state

`Scheduler.__init__()` 还包含：

```text
KV Connector
Encoder / MultiModal
Speculative Decode
Mamba
DP/PP async bookkeeping
Structured Output
metrics
```

当前全部暂时标记：

```text
OUT-OF-SCOPE
```

第一遍只保留：

```text
requests / waiting / running
max_num_running_reqs
max_num_scheduled_tokens
KVCacheManager
```

---

# 81. `waiting / running` 与 `Prefill / Decode` 不是同一个维度

这是 Scheduler 第一处必须钉死的概念。

可以存在：

```text
A：正在 Decode
B：长 Prompt，正在进行 Chunked Prefill
C：刚刚到达
```

Scheduler state 可能是：

```text
running:
    A
    B

waiting:
    C
```

因此：

```text
running ≠ Decode
waiting ≠ Prefill
```

更准确地说：

```text
waiting / running
= Request lifecycle / scheduling state

Prefill / Decode
= Request 当前 token execution progress 所体现的计算行为
```

B 即使仍未完成 Prompt Prefill，只要它已经进入 active runtime，就可以属于 `running`。

这也是为什么读 V1 Scheduler 时不能沿用“一个 Prefill queue + 一个 Decode queue”的旧心智模型。

---

# 82. Request 的核心状态：已知 token 与已计算 token 必须分开

`Request` 初始化中：

```python
self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(...)
self._output_token_ids: list[int] = []
self._all_token_ids = prompt_token_ids.copy()
self.spec_token_ids: list[int] = []
self.num_computed_tokens = 0
```

并有：

```python
@property
def num_tokens(self) -> int:
    return len(self._all_token_ids)

@property
def num_tokens_with_spec(self) -> int:
    return len(self._all_token_ids) + len(self.spec_token_ids)

@property
def num_output_tokens(self) -> int:
    return len(self._output_token_ids)
```

因此几个量要严格区分。

## 82.1 `num_prompt_tokens`

```text
原始输入 Prompt 长度
```

例如 Prompt 1000 token：

```text
num_prompt_tokens = 1000
```

## 82.2 `num_output_tokens`

```text
当前已经 sample 出来的 output token 数量
```

它不是模型已经 forward 了多少 token。

## 82.3 `num_tokens`

```text
当前 Request 已经拥有 / 知道的真实 token sequence 长度
≈ Prompt + 已生成 output
```

如果 Prompt=100，已经生成 20 个 token：

```text
num_tokens = 120
```

## 82.4 `num_tokens_with_spec`

```text
num_tokens + speculative draft tokens
```

当前不研究 speculative decode，因此普通路径中：

```text
num_tokens_with_spec ≈ num_tokens
```

## 82.5 `num_computed_tokens`

这是 Scheduler 最核心的 execution progress：

> 当前 Request 有多少 token position 已经被模型计算过。

例如：

```text
Prompt 总长 = 1000
已完成一个 256-token chunk

num_tokens          = 1000
num_computed_tokens = 256
```

意味着：

```text
已知完整 1000 token
但模型只 forward 到 position 255
```

剩余 work：

```text
1000 - 256 = 744 tokens
```

注意 async / PP 场景中源码注释指出 `num_computed_tokens` 可能 optimistic 地包含 in-flight token；当前 PP=1 普通主路径先不展开。

---

# 83. 为什么 V1 Scheduler 可以统一 Prefill 和 Decode

Scheduler 源码自己的设计注释非常重要：

```text
There's no "decoding phase" nor "prefill phase" in the scheduler.
Each request just has num_computed_tokens and num_tokens_with_spec.
```

它真正做的是：

```text
让 num_computed_tokens
逐步追上
当前需要计算的 token sequence 长度
```

## 83.1 Prefill 例子

```text
num_tokens          = 1000
num_computed_tokens = 400
```

那么普通路径下 pending work：

```text
600 tokens
```

## 83.2 Decode 例子

假设 Prompt 已完成，并刚 sample 出 `O0`：

```text
Request 已经知道 O0
→ num_tokens + 1

但 O0 本身还没有作为下一次模型输入 forward
→ num_computed_tokens 尚未 +1
```

因此典型 Decode 状态：

```text
num_tokens          = 501
num_computed_tokens = 500
```

pending work：

```text
1 token
```

于是 Scheduler 第一层看到的是：

```text
A：pending work = 1
B：pending work = 600
```

而不是先把它们拆成两套完全独立调度器。

这正是 V1 Scheduler 很重要的一层统一抽象：

```text
Request progress
↓
本轮还可以继续推进多少 token
```

---

# 84. `schedule()` 的第一层骨架：先 Running，再 Waiting

`schedule()` 初始化：

```python
scheduled_new_reqs = []
scheduled_resumed_reqs = []
scheduled_running_reqs = []
preempted_reqs = []

req_to_new_blocks = {}
num_scheduled_tokens = {}
token_budget = self.max_num_scheduled_tokens
```

然后：

```python
# First, schedule the RUNNING requests.
while req_index < len(self.running) and token_budget > 0:
    ...
```

之后才进入 waiting flow。

因此主结构是：

```text
Step 开始
↓
初始化 step-level budget / output accumulators
↓
优先推进 RUNNING requests
↓
再尝试 admission WAITING / PREEMPTED requests
↓
构造 SchedulerOutput
```

再次强调：

```text
Running first
≠ Decode first
```

因为 running 内既可能有 Decode，也可能有 in-progress Chunked Prefill。

---

# 85. `token_budget`：一个 Scheduler step 的全局 compute quota

源码：

```python
token_budget = self.max_num_scheduled_tokens
```

这是 **整个 step 共享** 的 token 预算，而不是某个 request 的预算。

例如：

```text
max_num_scheduled_tokens = 256
```

那么本轮所有 request 必须满足：

```text
Σ num_scheduled_tokens[request] <= 256
```

假设：

```text
A 用 1 token
B 用 200 token
```

则：

```text
token_budget:
256 → 255 → 55
```

Scheduler 成功 commit 某个 request 后：

```python
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
```

因此 `token_budget` 可以理解为：

> **当前 Scheduler step 尚未分配出去的 compute token quota。**

---

# 86. `num_new_tokens`：从 Request progress 得到本轮 candidate work

这里需要区分 running flow 和 waiting flow 的真实源码。

## 86.1 Running flow

源码：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

普通非 speculative / 非 async 主路径可以约化为：

```text
num_new_tokens
≈ request.num_tokens - request.num_computed_tokens
```

然后会经过：

```python
num_new_tokens = min(num_new_tokens, token_budget)
```

以及 max model length、long prefill threshold 等其他约束。

## 86.2 Waiting / resumed flow

在当前源码另一段中，普通路径使用局部：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
```

并根据 speculative / prefix / external KV 等情况继续修正。

因此第一层不要死记某一个单行公式，而应该理解：

```text
Request 当前需要追赶的 token frontier
-
当前有效 computed progress
↓
得到 candidate work
↓
再被本 step 各类 constraints 裁剪
```

普通文本、无 spec / connector 时可稳定使用：

```text
pending work ≈ num_tokens - num_computed_tokens
```

---

# 87. Chunked Prefill 为什么可以自然从 token budget 产生

假设：

```text
Prompt = 1000 tokens
num_computed_tokens = 0
token_budget = 256
```

则：

```text
pending = 1000
```

但：

```python
num_new_tokens = min(1000, 256)
```

本轮只能安排：

```text
256 tokens
```

下一轮：

```text
num_computed_tokens = 256
pending = 744
```

再被 budget 裁成：

```text
256
```

于是自然形成：

```text
Step 1: 256
Step 2: 256
Step 3: 256
Step 4: 232
```

因此在 V1 Scheduler 这个统一抽象下：

> **Chunked Prefill 可以理解为“大量 pending token work 被单 step token budget 分块推进”。**

实际源码还会受到 `long_prefill_token_threshold`、encoder budget、max model length 等约束，但核心机制已经成立。

---

# 88. Scheduler 实际上同时受两类资源约束

仅仅满足：

```text
token_budget
```

还不够。

因为每计算一个新的 token position，Attention 都会产生新的 K/V 状态，因此还必须有 KV storage capacity。

所以真正的 Scheduler 不是单纯的 queue：

```text
              Request progress
                    │
                    ▼
             remaining work
                    │
                    ▼
                Scheduler
               /         \
              /           \
     Compute Budget      KV Budget
       token_budget      KV blocks
              \           /
               \         /
                ▼       ▼
             本轮执行计划
```

可以把它理解为：

> **在 compute token quota 与 KV memory capacity 的联合约束下，为 Request 的 token progress 分配本轮 execution quota。**

这也是 production LLM Scheduler 与普通任务队列的根本区别之一。

---

# 89. Scheduler 为什么持有 KVCacheManager

Scheduler 创建：

```python
self.kv_cache_manager = KVCacheManager(...)
```

因此每当 Scheduler 决定：

```text
A 本轮再计算 N tokens
```

马上需要问：

```text
这些 token 对应的 KV slots 是否可用？
还需要多少 blocks？
如果 KV 不够，能否 schedule？
```

所以主调用是：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

这一接口非常重要，因为它说明 Scheduler 没有传：

```text
CUDA 地址
Tensor pointer
GPU storage offset
```

而是传：

```text
Request
+
本轮准备计算的 token 数
```

因此 `KVCacheManager` 是 **Request-oriented KV resource interface**，不是 GPU Tensor allocator。

---

# 90. KV allocation 失败为什么会触发 Scheduler preemption

Scheduler 中：

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(...)

    if new_blocks is not None:
        break

    # preempt another request
    ...
```

所以：

```text
allocate_slots() success
→ 返回 KVCacheBlocks

allocate_slots() failure
→ None
```

但 `None` 不代表 Request 直接失败。

它意味着：

> **当前 active working set 下 KV capacity 不足。**

这时由 Scheduler 根据 policy 决定 preempt 谁：

```text
KVCacheManager：
“资源不够。”

Scheduler：
“那根据 scheduling policy，我决定让谁退出 running。”
```

这个职责边界很关键：

```text
KVCacheManager
= resource feasibility / KV lifecycle

Scheduler
= scheduling policy / victim selection
```

KVCacheManager 不应该知道 request priority、FCFS、公平性等 scheduler policy；Scheduler 也不应该自己维护 free block linked list、prefix hash、refcount。

---

# 91. Preemption 为什么还要恢复 token budget

Scheduler 可能已经在本 step 暂时安排了 Request B：

```text
B → 100 tokens
```

后来为另一个 Request 分 KV 时发现容量不足，需要 preempt B。

源码会撤销 B 本轮已经形成的 tentative scheduling decision，例如：

```python
scheduled_running_reqs.remove(preempted_req)
token_budget += num_scheduled_tokens.pop(preempted_req_id)
req_to_new_blocks.pop(preempted_req_id)
```

因此 `schedule()` 不是：

```text
做一个决定 → 永远不能修改
```

而可能是：

```text
Tentative scheduling
↓
发现 KV conflict
↓
rollback 某些本轮决定
↓
恢复 compute quota / block delta bookkeeping
↓
重新尝试 allocation
```

这说明 Scheduler 实际管理的是一个联合资源状态：

```text
Compute quota
+
KV capacity
```

preemption 不只释放 KV 资源，也必须撤销对应的本 step compute quota。

---

# 92. KVCacheManager 内部为什么又有 Coordinator

`KVCacheManager.__init__()`：

```python
self.coordinator = get_kv_cache_coordinator(...)
self.block_pool = self.coordinator.block_pool
```

所以这里是明确的组合关系：

```text
KVCacheManager HAS-A Coordinator
```

不是：

```text
Coordinator IS-A KVCacheManager
```

也不要把 `Coordinator` 理解成又一个重复的 KVCacheManager。

当前从源码可确认：Coordinator 持有 / 组织：

```text
single_type_managers
shared block_pool
```

其一项典型职责：

```python
for i, manager in enumerate(self.single_type_managers):
    num_blocks_to_allocate += manager.get_num_blocks_to_allocate(...)
```

因此第一层可以理解为：

```text
KVCacheManager
= 面向 Scheduler 的统一 façade / request-level KV interface

Coordinator
= 在多个具体 KV cache manager / group 之间做协调与汇总

SingleTypeKVCacheManager
= 某一种具体 KV cache 类型的 block 规则

BlockPool
= 全局 KVCacheBlock 资源池
```

架构图：

```text
Scheduler
    │
    ▼
KVCacheManager
    │ HAS-A
    ▼
KVCacheCoordinator
    │
    ├── SingleTypeManager A
    ├── SingleTypeManager B
    └── ...
    │
    └── shared BlockPool
```

当前还没有追 `get_kv_cache_coordinator()` 对当前 Qwen3 配置具体选择哪个 coordinator / concrete manager，因此不要擅自假定具体 subclass 数量和组合。

---

# 93. `allocate_slots()` 真正解决的是什么问题

函数入口：

```python
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    ...
) -> KVCacheBlocks | None:
```

当前主路径去掉 Prefix Cache hit、External KV、Spec Decode、Encoder、Mamba，可以先压缩成一句话：

> **Request A 已经拥有一部分 KV blocks；Scheduler 本轮准备再计算 N token。`allocate_slots()` 负责保证本轮结束后 A 的 KV slot coverage 足够，并在容量不足时返回 `None`。**

关键不是简单：

```text
N token → 申请 N 个 block
```

而是：

```text
当前已有 KV coverage
+
本轮新增计算量
↓
本轮结束后的目标 token coverage
↓
结合已有 blocks
↓
计算真正还缺多少 physical blocks
```

---

# 94. `num_new_tokens` 与 `num_tokens_need_slot`：Delta 与 Target 必须分开

这是这条链里最容易混的变量之一。

普通路径：

```python
num_tokens_main_model = total_computed_tokens + num_new_tokens

num_tokens_need_slot = min(
    num_tokens_main_model + num_lookahead_tokens,
    self.max_model_len,
)
```

当前没有 lookahead，因此：

```text
num_tokens_need_slot
≈ computed_tokens + num_new_tokens
```

## 94.1 `num_new_tokens`

表示：

```text
本 step 准备新增计算多少 token
```

它是 delta：

```text
+1
+20
+255
```

## 94.2 `num_tokens_need_slot`

表示：

> **执行完这一轮之后，整个 Request 的 KV slots 需要覆盖到多少 token positions。**

它是 target / endpoint，而不是 delta。

例如：

```text
当前 computed = 20
本轮 new = 10

num_tokens_need_slot = 30
```

含义：

```text
本轮之后必须能够为 token position [0, 30) 提供合法 KV slot mapping
```

这就是为什么 `get_num_blocks_to_allocate()` 接收到的 `num_tokens` 注释明确写：

```text
The total number of tokens that need a slot
(including tokens that are already allocated).
```

---

# 95. 为什么不能只把 `num_new_tokens` 传给 block manager

假设：

```text
block_size = 16
当前 computed = 20
当前已经持有 2 blocks
```

这两个 blocks 已经可以覆盖：

```text
[0, 32)
```

其中第二个 block：

```text
16~19 已使用
20~31 仍有空余 slots
```

如果本轮：

```text
num_new_tokens = 5
```

若孤立地做：

```text
ceil(5 / 16) = 1 block
```

会错误地多申请一个 block。

正确做法是先计算目标：

```text
20 + 5 = 25 tokens
```

目标 `[0, 25)` 已经被现有两个 blocks `[0, 32)` 覆盖，因此：

```text
新增 block = 0
```

所以 API 使用：

```text
target total token coverage
```

而不是：

```text
本轮 delta token count
```

这是 Paged KV allocation 能正确利用 partial block 剩余 slot 的关键。

---

# 96. Coordinator 的 `get_num_blocks_to_allocate()`：分发并汇总需求

Coordinator 层代码：

```python
num_blocks_to_allocate = 0

for i, manager in enumerate(self.single_type_managers):
    ...
    num_blocks_to_allocate += manager.get_num_blocks_to_allocate(...)

return num_blocks_to_allocate
```

这说明 Coordinator 自己主要不是去做具体：

```text
ceil(tokens / block_size) - current_blocks
```

而是：

```text
一次 Request KV demand
↓
分发给多个 concrete manager
↓
每个 manager 根据自己的 cache type 规则计算 block requirement
↓
Coordinator 汇总总需求
```

例如概念上：

```text
Manager A → 需要 10 blocks
Manager B → 需要 4 blocks

Coordinator → total = 14 blocks
```

之所以需要这一层，是因为 production vLLM 不能让 Scheduler 假定整个模型永远只有一种 KV cache layout / spec。

这样可以把：

```text
Full Attention
Sliding Window
Cross Attention
Hybrid / other KV specs
```

等具体规则隔离在更低层，不污染 Scheduler。

---

# 97. SingleTypeKVCacheManager：真正把 token target 转成 block requirement

具体 manager 的代码核心：

```python
num_required_blocks = cdiv(num_tokens, self.block_size)
num_req_blocks = len(self.req_to_blocks.get(request_id, ()))
```

普通 Full Attention、无 Prefix Cache 特殊状态时，可以约化为：

```text
num_required_blocks
= ceil(target_tokens / block_size)

num_new_blocks
= max(num_required_blocks - current_request_blocks, 0)
```

例如：

```text
block_size = 16
目标 token coverage = 40
```

则：

```text
num_required_blocks
= ceil(40 / 16)
= 3
```

如果 Request 当前已经持有：

```text
2 blocks
```

那么：

```text
还缺 1 block
```

因此这一步完成了真正的空间转换：

```text
token position space
↓
logical block capacity requirement
```

---

# 98. `req_to_blocks`：Request ownership 与 BlockPool 必须分开

具体 manager 中：

```python
num_req_blocks = len(self.req_to_blocks.get(request_id, ()))
```

以及真正 allocation 时：

```python
req_blocks = self.req_to_blocks[request_id]
...
req_blocks.extend(new_blocks)
```

所以：

```text
req_to_blocks
= Request → 当前持有的 KVCacheBlock sequence
```

例如：

```text
A → [block7, block18, block25]
B → [block3, block9]
```

这和 `BlockPool.blocks` 是两个完全不同的视角。

```text
BlockPool.blocks
= 系统所有 physical block metadata 的 universe

req_to_blocks[A]
= A 的 logical block sequence 对应哪些 physical blocks
```

例如：

```text
A logical block 0 → physical block 7
A logical block 1 → physical block 18
A logical block 2 → physical block 25
```

这已经非常接近后面 ModelRunner `block_table` 的上游语义，但目前不能直接说 `req_to_blocks` 就等于最终 GPU block table；中间还要经过 `KVCacheBlocks → SchedulerOutput → ModelRunner persistent state`。

---

# 99. `get_num_blocks_to_allocate()` 与 `allocate_new_blocks()` 不是同一件事

这两个函数看起来很像，但职责不同。

## 99.1 `get_num_blocks_to_allocate()`

回答：

```text
“还需要几个 block？”
```

例如：

```text
return 16
```

它本质是 requirement prediction / dry-run，用于上层做 capacity feasibility check。

## 99.2 `allocate_new_blocks()`

回答：

```text
“现在真正把这 16 个 block 拿出来，并登记到这个 Request。”
```

普通路径：

```python
req_blocks = self.req_to_blocks[request_id]

num_required_blocks = cdiv(num_tokens, self.block_size)
num_new_blocks = num_required_blocks - len(req_blocks)

new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
req_blocks.extend(new_blocks)

return new_blocks
```

因此整体设计是：

```text
predict requirement
↓
check capacity
↓
commit actual allocation
```

而不是：

```text
先拿一部分
↓
拿到一半发现不够
↓
再复杂 rollback 半分配状态
```

这也是 Scheduler 能干净地处理 `allocate_slots() → None → preemption` 的基础。

---

# 100. BlockPool 是什么：全局 physical block 资源底座

`BlockPool.__init__()`：

```python
self.blocks: list[KVCacheBlock] = [
    KVCacheBlock(idx) for idx in range(num_gpu_blocks)
]
```

假设：

```text
num_gpu_blocks = 1000
```

初始化时已经创建固定数量的 block metadata：

```text
KVCacheBlock(0)
KVCacheBlock(1)
...
KVCacheBlock(999)
```

这说明运行时不是不停：

```text
new KVCacheBlock()
```

而是：

```text
固定 pool
↓
block metadata 在 free / active / cached / evictable 等状态间流转
```

因此：

> **BlockPool 是全局 KVCacheBlock resource pool。**

它管理的是 block resource，而不是 Request scheduling policy。

---

# 101. `KVCacheBlock(block_id)` 仍然不是 CUDA pointer

目前看到：

```text
KVCacheBlock(25)
```

只能说明控制面有一个 physical block 编号：

```text
block_id = 25
```

不要把它理解成：

```text
GPU address = 0x...
```

也不要理解成 `KVCacheBlock` Python 对象内部直接存整块 K/V Tensor。

当前它更接近 block metadata：

```text
KVCacheBlock
├── block_id
├── ref_cnt
├── block_hash
├── free queue / cache metadata
└── null / other bookkeeping
```

真正 GPU KV storage 仍然要在执行侧通过：

```text
block_id
↓
block_table
↓
slot_mapping / Attention Backend
↓
实际 KV Tensor storage
```

连接。

这是 control plane / data plane 的关键边界。

---

# 102. BlockPool 的核心状态：`free_block_queue`

初始化：

```python
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
```

可以第一层理解：

```text
所有可被 allocator 取得的 blocks
按照 allocation / eviction priority
组织在一个 queue 中
```

注意：

```text
free ≠ GPU KV 内容一定为空
```

Prefix Cache 开启时可能存在：

```text
ref_cnt = 0
block_hash != None
```

这种 block：

```text
当前没有 active Request 强引用
但仍保留有价值的 Prefix Cache KV
```

它仍然可以进入 free/eviction queue，因为内存紧张时 allocator 可以 eviction 并重新使用它。

所以 free queue 更准确的含义是：

> **当前 allocator 可以取得 / 驱逐后取得的 block 集合。**

---

# 103. `BlockPool.get_new_blocks()`：真正选择具体 physical block ID

代码：

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(...)

    ret = self.free_block_queue.popleft_n(num_blocks)

    for block in ret:
        if self.enable_caching:
            self._maybe_evict_cached_block(block)
        assert block.ref_cnt == 0
        block.ref_cnt += 1

    return ret
```

因此它真正做：

```text
需要 N blocks
↓
free_block_queue.popleft_n(N)
↓
得到具体 KVCacheBlock objects / block IDs
↓
必要时清理旧 Prefix Cache 身份
↓
ref_cnt: 0 → 1
↓
返回
```

例如 free queue：

```text
[7] [18] [25] [40] ...
```

调用：

```text
get_new_blocks(3)
```

得到：

```text
[KVCacheBlock(7), KVCacheBlock(18), KVCacheBlock(25)]
```

所以职责明确：

```text
get_num_blocks_to_allocate()
→ 算“数量”

BlockPool.get_new_blocks()
→ 选择“具体哪几个 physical blocks”
```

block ID 不是 Scheduler 算出来的。

---

# 104. 为什么 `get_new_blocks()` 还要 `_maybe_evict_cached_block()`

因为 free queue 中可能存在仍带 Prefix Cache hash 的 block。

例如：

```text
block18
ref_cnt = 0
block_hash = ABC
```

含义：

```text
当前没有 active Request 使用它
但 hash ABC → block18 的 Prefix Cache mapping 仍然存在
```

如果新的 Request 真正需要一个“新可写 block”，并从 free queue 取到了 block18：

```text
旧 cache identity 必须先失效
```

否则新的 KV 覆盖 block18 后，旧 hash 还指向它，会产生错误 cache hit。

因此：

```text
popleft block18
↓
_maybe_evict_cached_block(block18)
↓
清除旧 Prefix Cache mapping
↓
ref_cnt 0 → 1
↓
重新作为 active block 使用
```

这再次说明：

```text
free
≠
无数据 / 无 cache identity
```

---

# 105. Prefix Cache lookup 与 New Block Allocation 是两条不同路径

`BlockPool` 还提供：

```python
get_cached_block(block_hash, kv_cache_group_ids)
```

所以必须区分：

## 105.1 Cache hit 路径

```text
block hash
↓
get_cached_block()
↓
找到已有 KVCacheBlock
↓
touch / 增加 active reference
↓
直接复用已有 KV
```

## 105.2 New allocation 路径

```text
真正还缺 N blocks
↓
get_new_blocks(N)
↓
从 free queue 取得 blocks
↓
必要时 eviction 旧 cache identity
↓
ref_cnt 0 → 1
↓
后续写入新 KV
```

`get_new_blocks()` 的源码注释明确指出：

```text
we do not check block cache in this function
```

也就是说 cache lookup 是上层已经做出的另外一条决策；`get_new_blocks()` 只负责：

> **现在明确需要“新的可写 block”，从 resource pool 给我 N 个。**

---

# 106. `ref_cnt`：Active ownership 与 Prefix Cache residency 必须分开

`BlockPool.free_blocks()`：

```python
for block in ordered_blocks:
    block.ref_cnt -= 1
    if block.ref_cnt == 0 and not block.is_null:
        ...
```

`ref_cnt` 可以第一层理解为：

```text
当前有多少 active ownership/reference 正在使用这个 block
```

例如共享 Prefix：

```text
Request A ─┐
           ├── block17
Request B ─┘
```

可能：

```text
block17.ref_cnt = 2
```

A release：

```text
2 → 1
```

仍不能作为普通 free resource 处理。

B 也 release：

```text
1 → 0
```

此时才可以重新进入 free / eviction queue。

所以：

```text
Request release
↓
ref_cnt--
↓
ref_cnt == 0 ?
├── No → 仍 active
└── Yes → 可进入 free/evictable pool
```

---

# 107. 为什么 `free_blocks()` 把有 hash / 无 hash 的 block 放在不同位置

源码：

```python
if block.block_hash is None:
    blocks_without_hash.append(block)
else:
    blocks_with_hash.append(block)

self.free_block_queue.prepend_n(blocks_without_hash)
self.free_block_queue.append_n(blocks_with_hash)
```

而 allocation 使用：

```python
popleft_n(...)
```

因此效果是：

```text
queue head                                      queue tail
│                                                   │
无 hash、无 Prefix Cache 价值        有 hash、仍有 cache 复用价值
│                                                   │
优先被下一次 allocation 复用          尽量晚被 eviction / reuse
```

这说明 `free_block_queue` 同时承担：

```text
allocation order
+
Prefix Cache eviction priority
```

它不是普通 `list[int]`。

---

# 108. `free` 与 `evict` 不是一个动作

`BlockPool` 同时有：

```python
free_blocks(...)
evict_blocks(block_ids)
```

必须严格区分。

## 108.1 `free`

描述：

```text
某个 active Request 不再引用 block
```

核心影响：

```text
ref_cnt
```

## 108.2 `evict`

描述：

```text
这个 block 不再作为 Prefix Cache entry 保留
```

核心影响：

```text
hash / cache mapping
```

例如：

```text
block17
ref_cnt = 1
block_hash = ABC
```

可以使：

```text
ABC → block17
```

这个 Prefix Cache mapping 失效；但 Request 仍在 active 使用 block17，因此不能把 physical block17 立即重新分配给其他 Request。

因此：

```text
Prefix Cache eviction
≠
Active KV block release
```

这对后续做 KV eviction / compression 项目非常重要。

---

# 109. BlockPool 第一层生命周期状态机

当前可以把 BlockPool 的核心生命周期画成：

```text
                     ┌─────────────────┐
                     │ reusable block  │
                     │ ref_cnt = 0     │
                     └────────┬────────┘
                              │ get_new_blocks()
                              │ popleft
                              ▼
                     ┌─────────────────┐
                     │ active block    │
                     │ ref_cnt > 0     │
                     └────────┬────────┘
                              │ request release
                              ▼
                          ref_cnt--
                              │
                    ┌─────────┴─────────┐
                    │                   │
              ref_cnt > 0         ref_cnt == 0
                    │                   │
                    │                   ▼
                    │          ┌─────────────────┐
                    │          │ reusable /      │
                    │          │ evictable       │
                    │          └────────┬────────┘
                    │                   │
                    │        ┌──────────┴──────────┐
                    │        │                     │
                    │   hash=None             hash exists
                    │        │                     │
                    │        ▼                     ▼
                    │   queue front          queue tail
                    │   优先复用              尽量保留 cache
                    │
                    └── 仍由 active owner 使用
```

这已经足够完成 BlockPool 第一遍理解；暂时不需要继续深入双向链表实现、partial cache hit、CoW 等细节。

---

# 110. 真正 allocation 后，谁把 block 登记到 Request 名下

`BlockPool.get_new_blocks()` 参数中没有 `request_id`：

```python
get_new_blocks(num_blocks)
```

这直接证明：

> BlockPool 根本不知道这些 block 属于哪个 Request。

真正 ownership 更新发生在 `SingleTypeKVCacheManager.allocate_new_blocks()`：

```python
req_blocks = self.req_to_blocks[request_id]

num_required_blocks = cdiv(num_tokens, self.block_size)
num_new_blocks = num_required_blocks - len(req_blocks)

new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
req_blocks.extend(new_blocks)

return new_blocks
```

例如：

```text
req_to_blocks[A] = [7, 18]
```

本轮还缺 1 block：

```text
BlockPool.get_new_blocks(1)
→ [block25]
```

然后：

```text
req_to_blocks[A].extend([block25])
```

最终：

```text
req_to_blocks[A] = [7, 18, 25]
```

因此职责边界非常明确：

```text
BlockPool
= 资源层：给我哪个 block

SingleTypeKVCacheManager
= ownership / mapping 层：这个 block 属于哪个 Request logical sequence
```

---

# 111. `allocate_slots()` 后半段：先 feasibility check，再 commit

上层 `KVCacheManager.allocate_slots()`：

```python
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks

if required_blocks > available_blocks:
    return None
```

如果资源够，才：

```python
new_blocks = self.coordinator.allocate_new_blocks(
    request.request_id,
    num_tokens_need_slot,
    num_tokens_main_model,
    num_encoder_tokens,
)
```

因此主过程是：

```text
① 计算 target token coverage
↓
② predict block requirement
↓
③ check global free capacity
↓
├── insufficient → return None
│                    ↓
│                 Scheduler preempt
│
└── enough
     ↓
④ actual allocate_new_blocks()
     ↓
⑤ BlockPool.get_new_blocks()
     ↓
⑥ req_to_blocks update
```

这是非常清晰的：

```text
feasibility check
→ commit
```

而不是先产生半完成 allocation 再复杂回滚。

---

# 112. Watermark 的意义：避免过度 admission 引发频繁 preemption

`KVCacheManager.__init__()`：

```python
self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
```

`allocate_slots()` 对 waiting / preempted admission 场景可能加入：

```text
required_blocks
=
num_blocks_to_allocate + watermark_blocks
```

它不是简单表示“这些 blocks 永远不能用”，而是一种 admission headroom。

例如：

```text
free = 100 blocks
waiting request 需要 95 blocks
watermark = 10 blocks
```

虽然：

```text
95 <= 100
```

但：

```text
95 + 10 > 100
```

系统可能拒绝当前 admission，以免刚放进来就把 active request 后续增长空间吃光，造成：

```text
admit
→ 立刻资源不足
→ preempt
→ resume
→ 再不足
```

即减少 scheduling thrashing。

---

# 113. `full_sequence_must_fit`：Chunked Prefill 下更保守的 admission gate

普通 Chunked Prefill 可能只检查当前 chunk：

```text
Prompt = 4096
当前 chunk = 256
```

如果只看 256 token：

```text
当前能放下
→ admission
```

但可能整个 sequence 最终根本无法在当前资源模型下满足。

`full_sequence_must_fit=True` 会提前用：

```text
full_num_tokens
```

做一次完整 sequence requirement check。

它解决的是：

```text
小 chunk 暂时能放下
≠
整个 request 资源上可持续推进
```

当前只需要理解为一种更严格的 admission gate，不继续深入 SWA / recycling-aware cap。

---

# 114. `allocate_slots()` 中 Prefix Cache commit 与 block allocation 不是同一件事

真正 allocation 已经发生在：

```python
self.coordinator.allocate_new_blocks(...)
```

之后，如果 caching enable：

```python
num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,
    request.num_tokens,
)

self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

因此必须区分：

```text
allocate block
≠
cache block
```

第一步回答：

```text
Request 需要新的 physical KV capacity，给它哪些 blocks？
```

第二步回答：

```text
这些已经产生的 KV 中，哪些 finalized prefix 可以注册为 Prefix Cache，以供未来 Request 复用？
```

当前主线先停在 allocation；Prefix Cache 的 hash / full block / partial block / CoW 以后单独补。

---

# 115. `new_blocks` 与 `req_to_blocks`：Delta 与 Full State 再次分开

`SingleTypeKVCacheManager.allocate_new_blocks()` 返回：

```text
new_blocks
```

这是：

> **本次 allocation 新增的 block delta。**

例如原来：

```text
req_to_blocks[A] = [7,18]
```

本轮新增：

```text
new_blocks = [25]
```

allocation 后完整 state：

```text
req_to_blocks[A] = [7,18,25]
```

因此：

```text
new_blocks
= step-level delta

req_to_blocks[A]
= persistent full ownership state
```

这和 execution side 的“长期状态 + step delta”思想一致：

```text
控制面保留 authoritative persistent state
↓
每个 step 只传播必要增量
↓
MRV2 在 req_states / model_state 中维护 execution-side persistent state
```

后面应从 `SchedulerOutput → MRV2 GPUModelRunner` 重新接上，而不是进入 MRV1 的 `_update_states()`。

---

# 116. 当前整个 Scheduler → KV block allocation 主链

现在可以把从 Request progress 到 physical block ID 的完整控制链画出来。

假设：

```text
block_size = 16
Request B:
num_tokens = 1000
num_computed_tokens = 400

本 step token_budget = 256
```

Scheduler：

```text
pending work
≈ 1000 - 400
= 600
```

如果 A Decode 先占 1 token，则 B 可能得到：

```text
num_new_tokens = 255
```

进入 KV allocation：

```text
Request B
num_computed_tokens = 400
        │
        │ Scheduler 本轮安排 255
        ▼
num_tokens_need_slot
= 400 + 255
= 655
        │
        ▼
Coordinator.get_num_blocks_to_allocate()
        │
        ▼
SingleTypeKVCacheManager
        │
block_size = 16
required_blocks = ceil(655 / 16) = 41
        │
假设当前 B 已经持有 25 blocks
        │
new block demand = 41 - 25 = 16
        ▼
KVCacheManager capacity check
        │
        ├── free capacity < 16
        │      ↓
        │    return None
        │      ↓
        │    Scheduler preempt / retry
        │
        └── free capacity >= 16
               ↓
        Coordinator.allocate_new_blocks()
               ↓
        SingleTypeKVCacheManager
               ↓
        BlockPool.get_new_blocks(16)
               ↓
        free_block_queue.popleft_n(16)
               ↓
        concrete KVCacheBlock IDs
               ↓
        ref_cnt: 0 → 1
               ↓
        req_to_blocks[B].extend(new_blocks)
               ↓
        返回本轮 new block delta
```

这条链说明的不是“vLLM 如何申请 Python 对象”，而是：

> **Scheduler 的 token-level execution decision 如何被转换成 request-level KV capacity demand，再转换成 block-level physical resource allocation。**

---

# 117. A / B / C 三请求例子：Scheduler 与 KV resource 如何协同

假设：

```text
max_num_scheduled_tokens = 256
block_size = 16
```

当前：

```text
running:
A：Decode
B：Chunked Prefill

waiting:
C：new request
```

Request progress：

```text
A:
num_tokens = 501
num_computed_tokens = 500
pending ≈ 1

B:
num_tokens = 1000
num_computed_tokens = 400
pending ≈ 600

C:
num_tokens = 200
num_computed_tokens = 0
pending ≈ 200
```

Step 开始：

```text
token_budget = 256
```

## 117.1 先处理 A

```text
A pending = 1
→ num_new_tokens = 1
```

如果 KV coverage 已经在现有 partial block 内：

```text
new block demand = 0
```

commit：

```text
A scheduled = 1
token_budget = 255
```

## 117.2 再处理 B

```text
B pending = 600
min(600, 255)
→ candidate = 255
```

KV 目标：

```text
400 + 255 = 655 tokens
```

block requirement：

```text
ceil(655 / 16) = 41 blocks
```

如果 B 已有 25：

```text
还缺 16
```

如果 BlockPool 可满足：

```text
B scheduled = 255
token_budget = 0
```

## 117.3 C 本轮不再 admission

因为：

```text
token_budget = 0
```

所以 C 保持 waiting。

这一例子把三个概念连接起来：

```text
Request progress
↓
compute token budget
↓
KV block capacity
↓
本 step execution plan
```

---

# 118. 为什么这些层不能合并成一个“大 KV Manager”

现在可以从架构变化维度解释分层原因。

## 118.1 Scheduler：Policy 变化维度

它关心：

```text
谁先运行
谁继续 running
谁 admission
谁被 preempt
本轮每个 Request 给多少 token quota
```

这些属于 scheduling policy。

## 118.2 KVCacheManager：Request KV lifecycle 变化维度

它关心：

```text
这个 Request 本轮需要多少 KV slot coverage
Prefix / external computed state
是否能满足 allocation
何时 free request KV
```

## 118.3 Coordinator：Heterogeneous KV layout/spec 变化维度

它关心：

```text
多个 concrete KV manager 如何统一协调
一次 request demand 如何分发和汇总
共享 BlockPool 如何被多个 manager 使用
```

## 118.4 SingleTypeKVCacheManager：Concrete mapping rule 变化维度

它关心：

```text
token coverage
→ required blocks
Request 当前拥有哪些 blocks
某一种 cache spec 下哪些 block 仍有效
```

## 118.5 BlockPool：Global physical resource 变化维度

它关心：

```text
系统有哪些 KVCacheBlock
哪些可分配
具体拿哪几个
ref_cnt
Prefix Cache hash / eviction priority
free / reuse
```

因此整体是：

```text
Policy
↓
Request-level resource semantics
↓
multi-type coordination
↓
concrete mapping rule
↓
global block resource pool
```

每层的变化原因不同，所以拆开比全部塞进 Scheduler 或 KVCacheManager 更稳定。

---

# 119. 这条链路真正说明了什么

## 119.1 vLLM Scheduler 调度的是 token progress，不只是 Request 顺序

Scheduler 不只是：

```text
A → B → C
```

而是：

```text
A 本 step = 1 token
B 本 step = 255 tokens
C 本 step = 0
```

所以核心产物是 per-request execution quota，而不仅是 request ordering。

## 119.2 Compute scheduling 与 KV memory scheduling 强耦合

Scheduler 决定 N token 后，必须同步验证：

```text
这些 token 产生的 KV 是否有位置可写
```

因此：

```text
compute budget
+
KV budget
```

共同决定 request 能否真正执行。

## 119.3 block allocation 是控制面 metadata 操作，不是 GPU KV 写入

当前整条：

```text
KVCacheManager
Coordinator
SingleTypeKVCacheManager
BlockPool
KVCacheBlock(block_id)
```

仍主要是在 CPU control plane 管理：

```text
logical ownership
physical block ID
resource availability
refcount / cache metadata
```

真正 GPU K/V Tensor 写入还没发生。

因此：

```text
allocate block ID
≠
已经写入 KV Tensor
```

## 119.4 Request logical block sequence 与全局 BlockPool 是两种状态

```text
req_to_blocks[A]
= A 的 logical → physical mapping

BlockPool
= 全局 physical block resource universe
```

这就是之后 `block_table` 能成立的基础。

## 119.5 Prefix Cache eviction 与 Active KV free 是不同生命周期

Block 可以：

```text
ref_cnt = 0
但仍保留 hash / KV cache value
```

也可以：

```text
cache identity 被 evict
但 ref_cnt > 0，仍被 active Request 使用
```

因此不能把所有“evict / free / reuse”混成一个概念。

## 119.6 Production vLLM 大量使用 persistent state + delta propagation

目前已经看到：

```text
Scheduler persistent state:
requests / running / waiting / req_to_blocks

step delta:
num_scheduled_tokens / new_blocks / finished etc.
```

execution side 同样需要：

```text
persistent request state
+
SchedulerOutput step delta
```

这个总体思想和控制面是上下呼应的。

但实现必须区分：

```text
MRV1 → CachedRequestState + persistent InputBatch
MRV2 → req_states / RequestState + per-step input preparation
```

当前主路径以后者为准。

---

# 120. 对后续 QCache / KV 项目的直接启示

当前还没有开始修改 vLLM，但这条控制链已经给出几个非常重要的工程边界。

## 120.1 不要把 Scheduler metadata 与 GPU physical storage 混在一起

如果后续 QCache 引入：

```text
RECENT_BF16
HISTORY_INT4
EVICTED
```

需要明确：

```text
哪些状态是 Scheduler / KV manager 的逻辑状态
哪些状态属于 Worker / GPU physical pool
```

否则很容易出现控制面认为 block 存在、GPU storage 已经迁移/压缩但 metadata 没同步的问题。

## 120.2 资源不足后的“牺牲谁”应属于 Scheduler policy

KV allocator 应告诉上层：

```text
能不能满足 / 需要多少资源
```

而不是自己决定 request priority / victim。

未来任何 compression / migration policy 如果会改变 admission / preemption，必须尊重这个职责边界。

## 120.3 block ID 是稳定的控制面连接点

当前已经看到：

```text
Request
→ req_to_blocks
→ physical block_id
```

后续还会连接：

```text
block_id
→ SchedulerOutput
→ ModelRunner block_table
→ slot_mapping
→ GPU KV storage
```

因此 QCache 需要重点保护的是：

```text
logical block identity
与
physical storage format/location
之间的映射一致性
```

而不是简单“把一个 Tensor 换成 INT4”。

---

# 121. 当前已经源码确认的核心责任表

| 层 / 对象 | 当前确认的核心职责 | 当前关键状态 / 接口 | 不应该负责什么 |
|---|---|---|---|
| `EngineCore` | 驱动 schedule → execute → update 闭环 | `scheduler`, `model_executor` | 具体 block allocation |
| `Scheduler` | Request lifecycle + per-step token scheduling + preemption policy | `waiting`, `running`, `token_budget`, `num_scheduled_tokens` | free queue / CUDA KV Tensor |
| `Request` | 保存长期 token / execution progress | `num_tokens`, `num_tokens_with_spec`, `num_computed_tokens` | 全局 KV block pool |
| `KVCacheManager` | 面向 Scheduler 的 Request-level KV resource interface | `allocate_slots()`, `free()`, `coordinator` | request priority policy |
| `Coordinator` | 协调多个 concrete KV managers 并汇总需求 | `single_type_managers`, `block_pool` | Scheduler policy |
| `SingleTypeKVCacheManager` | 某类 KV cache 的 token→block 规则和 Request ownership | `req_to_blocks`, `get_num_blocks_to_allocate()`, `allocate_new_blocks()` | 全局调度顺序 |
| `BlockPool` | 全局 KVCacheBlock resource pool | `blocks`, `free_block_queue`, `get_new_blocks()`, `free_blocks()` | Request token budget |
| `KVCacheBlock` | physical block metadata identity | `block_id`, `ref_cnt`, `block_hash` 等 | 直接等价 CUDA pointer |
| `MRV2 GPUModelRunner` | 把 SchedulerOutput 翻译为 GPU execution state | `req_states` / `RequestState`, `model_state`, per-step input metadata | Scheduler block admission policy |

---

# 122. 当前最容易混淆的变量表

| 变量 | 层级 | 含义 | 是否 persistent |
|---|---|---|---|
| `num_prompt_tokens` | Request | 原始 Prompt 长度 | 是 |
| `num_output_tokens` | Request | 已 sample 的 output token 数 | 是 |
| `num_tokens` | Request | 当前已知真实 token sequence 长度 | 是 |
| `num_tokens_with_spec` | Request | `num_tokens + draft tokens` | 是/动态 |
| `num_computed_tokens` | Request | 当前模型计算进度；async 下可 optimistic | 是 |
| `num_new_tokens` | Scheduler step | 当前 Request 本 step candidate/scheduled work，经过多层约束裁剪 | 否 |
| `token_budget` | Scheduler step | 整个 step 剩余 compute token quota | 否 |
| `num_scheduled_tokens[req]` | SchedulerOutput 前置状态 | 本 step 最终给该 Request 安排多少 token | 否 |
| `num_tokens_need_slot` | KV allocation | 本 step 后 Request 需要覆盖的总 KV token endpoint | 否 |
| `num_required_blocks` | concrete KV manager | 为 target coverage 总共需要的 block 数 | 否 |
| `num_req_blocks` | concrete KV manager | Request 当前已经持有的 block 数 | 来自 persistent mapping |
| `num_new_blocks` | concrete KV manager | 这次还需要增加多少 block | 否 |
| `req_to_blocks[req]` | concrete KV manager | Request 当前完整 block ownership / mapping | 是 |
| `new_blocks` | allocation result | 本次实际新增的 block delta | 否 |
| `block_id` | BlockPool / control plane | physical KV block 编号 | block metadata persistent |

---

# 123. 当前最容易混淆的概念对照

## 123.1 `running` vs `decode`

```text
running = request lifecycle state
decode  = token execution behavior
```

## 123.2 `num_new_tokens` vs `num_tokens_need_slot`

```text
num_new_tokens
= 本 step delta

num_tokens_need_slot
= 本 step 结束后的 total KV coverage target
```

## 123.3 `num_required_blocks` vs `num_new_blocks`

```text
num_required_blocks
= target coverage 总共需要多少 blocks

num_new_blocks
= 减去 Request 已有 blocks 后，本次还缺多少
```

## 123.4 `get_num_blocks_to_allocate()` vs `allocate_new_blocks()`

```text
get_num_blocks_to_allocate()
= predict / dry-run

allocate_new_blocks()
= actual commit
```

## 123.5 `req_to_blocks` vs `BlockPool.blocks`

```text
req_to_blocks
= per-request ownership / logical mapping

BlockPool.blocks
= global physical block resource universe
```

## 123.6 `free` vs `evict`

```text
free
= active ownership release / ref_cnt lifecycle

evict
= Prefix Cache identity/hash eviction
```

## 123.7 `KVCacheBlock(block_id)` vs GPU address

```text
block_id
= control-plane physical block identifier

CUDA pointer / KV Tensor storage
= execution/data-plane object
```

两者之间还隔着 ModelRunner / block_table / slot_mapping。

---

# 124. 当前静态主链总结

到目前为止，源码可以压缩成：

```text
EngineCore.step()
│
▼
Scheduler.schedule()
│
├── Request progress
│   ├── num_tokens / num_tokens_with_spec
│   └── num_computed_tokens
│
├── Compute constraints
│   ├── max_num_running_reqs
│   └── token_budget
│
├── num_new_tokens
│
▼
KVCacheManager.allocate_slots(request, num_new_tokens)
│
├── 计算 total / target token coverage
│   └── num_tokens_need_slot
│
├── Coordinator.get_num_blocks_to_allocate()
│       │
│       └── SingleTypeKVCacheManager
│             ├── num_required_blocks
│             ├── req_to_blocks
│             └── num_new_blocks
│
├── BlockPool capacity check
│
├── insufficient
│       ↓
│    return None
│       ↓
│    Scheduler preemption / retry
│
└── sufficient
        ↓
    Coordinator.allocate_new_blocks()
        ↓
    SingleTypeKVCacheManager.allocate_new_blocks()
        ↓
    BlockPool.get_new_blocks(N)
        ↓
    free_block_queue.popleft_n(N)
        ↓
    KVCacheBlock(block_id)
        ↓
    ref_cnt 0 → 1
        ↓
    req_to_blocks[request_id].extend(new_blocks)
        ↓
    return allocation delta
```

这已经把：

```text
token demand
→ KV capacity demand
→ logical block requirement
→ physical block allocation
→ Request ownership
```

完整闭环。

---

# 125. 当前阶段完成度

相对于 `02-vLLM三周定向桥接学习手册` 的路线，目前已经实际完成 / 建立了：

```text
① Scheduler overall mental model               已完成第一层
② Request / waiting / running                  已完成第一层
③ num_computed_tokens / token budget           已完成核心语义
④ Scheduler.schedule()                         已完成主路径第一层
⑤ KVCacheManager.allocate_slots()              已完成普通主路径第一层
⑥ Coordinator / SingleType manager             已建立职责边界
⑦ BlockPool / block allocation lifecycle       已完成第一层
```

其中 Scheduler 的复杂 feature branches 仍未深入：

```text
Speculative Decode
KV Connector
Encoder / Cross Attention
Mamba / Sliding Window 特殊规则
async / PP in-flight bookkeeping
Prefix Cache partial hit / CoW
```

这些不是当前主线阻塞项。

---

# 126. 当前停点：下一次从哪里继续

到这里，Scheduler / KV allocation 主链仍然成立：

```text
allocate_slots()
↓
Coordinator.allocate_new_blocks()
↓
SingleTypeKVCacheManager
↓
BlockPool
↓
KVCacheBlocks / physical block IDs
```

接下来需要完成：

```text
KVCacheBlocks
↓
Scheduler 内 req_to_new_blocks
↓
SchedulerOutput construction
↓
new / resumed / running request data contract
↓
Executor / Worker
↓
MRV2 GPUModelRunner
   vllm/v1/worker/gpu/model_runner.py
↓
req_states / model_state
↓
per-step InputBatch / block table / slot mapping
```

这里必须删除旧停点中的：

```text
MRV2 GPUModelRunner
↓
CachedRequestState.block_ids
↓
persistent input_batch.block_table
```

因为它属于 MRV1 implementation，不是当前 MRV2 主路径。

下一阶段第一个 execution-side 问题应改成：

> **MRV2 如何消费 SchedulerOutput：finished/preempted request 如何 remove，new/resumed 如何 add，continuing request 如何更新 `req_states`，block IDs 又如何进入 MRV2 的 persistent block state 与 per-step input metadata？**

---

# 127. 当前阶段一句话总结

前一阶段从执行侧看到了：

```text
ModelRunner 需要 block_table / slot_mapping
```

这一阶段终于从控制面解释了这些东西为什么存在：

```text
Request token progress
↓
Scheduler 分配本 step token quota
↓
KVCacheManager 将 compute demand 转成 KV capacity demand
↓
Coordinator / SingleTypeManager 将 token target 转成 block requirement
↓
BlockPool 选择具体 physical block IDs
↓
per-request ownership 被更新
↓
下一步通过 SchedulerOutput 把 block delta 交给 ModelRunner
```

最核心的架构认知是：

> **vLLM 的 Paged KV Cache 不是一个孤立的 GPU 内存结构，而是一条贯穿 Scheduler policy、Request progress、KV resource admission、logical-to-physical block mapping 和 execution-side block table 的完整控制链。**

只有把这条链打通，后面再谈 KV 压缩、迁移、offload、eviction 或 Attention backend 修改，才知道“状态应该放在哪里、谁负责决策、谁负责实际执行、结果又必须反馈给谁”。

---

# 128. 从 KV allocation 回到 SchedulerOutput：为什么这里必须重新“向上收”

前面我们已经把 KV allocation 的控制链一路追到了最底层：

```text
Scheduler
↓
KVCacheManager.allocate_slots()
↓
Coordinator
↓
SingleTypeKVCacheManager
↓
BlockPool.get_new_blocks()
↓
具体 KVCacheBlock(block_id)
↓
req_to_blocks[request_id]
```

到这里，一个非常重要的问题已经解决：

> Scheduler 决定“Request A 本轮再计算 N 个 token”之后，系统如何把这个 compute demand 转换为 KV block demand，并最终得到具体的 physical block IDs？

但这还只是 **Scheduler / KV control plane 内部状态**。

GPUModelRunner 并不会直接读取：

```text
KVCacheManager
Coordinator
SingleTypeKVCacheManager
BlockPool
req_to_blocks
```

因此还必须存在一个明确的边界，把 Scheduler 本轮做出的决策转换成执行侧能够消费的数据。

这个边界就是：

```text
SchedulerOutput
```

所以接下来的核心问题变成：

```text
Scheduler 内部状态
        ↓
如何编码
        ↓
SchedulerOutput
        ↓
如何被 GPUModelRunnerV2 消费
```

这也是为什么在 BlockPool 第一层看完之后，不应该继续钻 Prefix Cache hash、free queue 双向链表或 CoW，而应该返回 Scheduler：

> **我们当前主线的目标不是把每个 KV 子模块看完，而是先把“控制面决策如何抵达执行面”闭环。**

---

# 129. SchedulerOutput 的定位：Control Plane → Execution Plane 的 step contract

在 `Scheduler.schedule()` 末尾，可以看到：

```python
scheduler_output = SchedulerOutput(
    scheduled_new_reqs=new_reqs_data,
    scheduled_cached_reqs=cached_reqs_data,
    num_scheduled_tokens=num_scheduled_tokens,
    total_num_scheduled_tokens=total_num_scheduled_tokens,
    ...
    preempted_req_ids=self.reset_preempted_req_ids,
    finished_req_ids=self.finished_req_ids,
    new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
    ...
)
```

第一遍不要被其它 feature 字段干扰。当前 TP=1 / PP=1 / DP=1 / V2 ModelRunner / 普通 decoder-only 主路径下，可以把它压缩成：

```text
SchedulerOutput
│
├── Request state synchronization
│   ├── scheduled_new_reqs
│   └── scheduled_cached_reqs
│
├── Execution quota
│   ├── num_scheduled_tokens
│   └── total_num_scheduled_tokens
│
├── Lifecycle synchronization
│   ├── preempted_req_ids
│   └── finished_req_ids
│
└── KV maintenance
    └── new_block_ids_to_zero
```

因此，`SchedulerOutput` 不是单纯的：

```text
“本轮算哪些 token”
```

而是一次完整的：

```text
Scheduler state synchronization
+
execution plan
+
request lifecycle changes
+
KV maintenance instructions
```

可以把它定义成：

> **SchedulerOutput 是 Scheduler 每一个 step 向 Execution Plane 发出的执行契约。它告诉 Worker / ModelRunner：本轮有哪些 request 需要初始化、哪些 request 只需要增量更新、每个 request 算多少 token、哪些 request 已结束或被抢占，以及有哪些 KV 维护动作需要执行。**

这就是 Scheduler 和 ModelRunner 之间真正的架构边界。

---

# 130. 为什么 ModelRunner 不应该直接读取 Scheduler 内部状态

一种看似简单但架构上很差的设计是：

```text
GPUModelRunner
↓
直接读取 Scheduler.requests
直接读取 KVCacheManager.req_to_blocks
直接读取 BlockPool
```

这样会导致：

```text
Scheduling policy
KV resource ownership
GPU execution runtime
```

强耦合在一起。

vLLM 当前看到的设计则是：

```text
Scheduler / KV Control Plane
        │
        │ SchedulerOutput
        ▼
Executor
        ▼
Worker
        ▼
GPUModelRunnerV2
```

因此：

```text
Scheduler
负责 authoritative control state

ModelRunner
负责 execution-side runtime state

SchedulerOutput
负责两者之间的 state synchronization
```

这种设计的重要意义是：

1. Scheduler 不需要知道 GPUModelRunner 内部具体如何组织 tensor；
2. ModelRunner 不需要理解 Scheduler 的 waiting/running/policy/preemption 实现；
3. Executor / Worker 可以作为中间执行层；
4. 后续 TP / PP / async 等模式仍然围绕同一个 execution contract 扩展。

---

# 131. Scheduler 内部三类 Request，为什么输出时只剩 new / cached 两类

在 Scheduler 内部，我们已经看到三类本轮被调度的 request：

```text
scheduled_new_reqs
scheduled_resumed_reqs
scheduled_running_reqs
```

它们分别表示：

```text
scheduled_new_reqs
= 第一次从 WAITING admission 到 RUNNING

scheduled_resumed_reqs
= 之前被 preempt，当前重新进入 RUNNING

scheduled_running_reqs
= 原本已经 RUNNING，本轮继续推进
```

但到了 `SchedulerOutput`，字段不是这三个，而是：

```text
scheduled_new_reqs
scheduled_cached_reqs
```

这不是简单重命名，而是在做一次 **Scheduler state → Execution state 的语义转换**。

---

# 校正说明：关于 V2 resumed request 的 producer / consumer 语义

本节根据 vLLM v0.26.0 当前固定源码重新校正。

此前误把 MRV1 `vllm/v1/worker/gpu_model_runner.py::_update_states()` 当成 MRV2 主路径时，看到：

```text
CachedRequestData.resumed_req_ids
+
resumed 时 block_ids replace
```

容易推断：

```text
V2 resumed request → CachedRequestData
```

但 Scheduler producer 端的真实 `schedule()` 明确显示，标准 V2 路径会先：

```python
scheduled_new_reqs.extend(scheduled_resumed_reqs)
scheduled_resumed_reqs.clear()
```

因此普通 preemption-resume 在当前 V2 producer 主路径中实际进入：

```text
NewRequestData
```

而不是标准 `CachedRequestData.resumed_req_ids` 路径。

所以后续阅读必须坚持：

```text
producer 决定当前主路径实际发送什么
consumer 决定接口能够消费什么
```

不能仅凭 consumer 中存在某个兼容/通用分支，就反推当前 producer 一定会生成该分支的数据。

---

# 132. V2 Model Runner 下的关键特殊处理：Scheduler 内部 resumed → `NewRequestData`

Scheduler 内部本轮 request 先分：

```text
scheduled_new_reqs
scheduled_resumed_reqs
scheduled_running_reqs
```

Waiting / Preempted request 被 admission 时：

```python
if request.status == RequestStatus.WAITING:
    scheduled_new_reqs.append(request)
elif request.status == RequestStatus.PREEMPTED:
    scheduled_resumed_reqs.append(request)
```

因此 Scheduler lifecycle 语义是：

```text
WAITING    → new
PREEMPTED  → resumed
RUNNING    → continuing
```

但 V2 output packaging 会：

```python
if self.use_v2_model_runner:
    scheduled_new_reqs.extend(scheduled_resumed_reqs)
    scheduled_resumed_reqs.clear()
```

所以当前 MRV2 producer contract 是：

```text
new + resumed
→ NewRequestData

continuing running
→ CachedRequestData
```

这不是“Scheduler 也升级成另一套 V2 Scheduler”，而是：

> **同一个 Scheduler 根据 execution-side runner 的能力，选择不同的 state synchronization contract。**

因此 `use_v2_model_runner` 是 Scheduler 与两套 Model Runner 之间的 compatibility/adaptation point。

---

# 133. MRV2 为什么可以把 resumed 当作 fresh execution state

现在 producer 和 consumer 两侧已经都能确认。

Scheduler preemption：

```python
self._free_request_blocks(request)
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
self.waiting.prepend_request(request)
self.reset_preempted_req_ids.add(request.request_id)
```

所以 Scheduler 侧：

```text
Request identity / token history
仍存在

但：
KV ownership 被释放
computed frontier 被 reset
status → PREEMPTED
```

SchedulerOutput 随后带上：

```text
preempted_req_ids
```

MRV2 execution side 的真实实现位于：

```text
vllm/v1/worker/gpu/model_runner.py
```

其中：

```python
def finish_requests(self, scheduler_output):
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids
    if preempted_req_ids:
        finished_req_ids = finished_req_ids.union(preempted_req_ids)
    for req_id in finished_req_ids:
        self._remove_request(req_id)
```

所以 MRV2 明确采用：

```text
preemption
≈ execution-side completion
↓
_remove_request()
↓
释放 MRV2 request slot/state
```

后续 Scheduler 再 resume A：

```text
PREEMPTED
↓
scheduled_resumed_reqs
↓
V2 merge
↓
NewRequestData
```

MRV2 `add_requests()` 再把它作为 fresh execution state 加回来。

因此现在可以准确写：

> **Scheduler Request lifetime 可以连续，但 MRV2 execution-side active-state lifetime 在 preemption 处结束；resume 时重新 add。**

这也是 MRV2 官方设计文档明确强调的原则：

```text
Treat preemption as completion.
On resume, re-add request data as fresh state.
```

---

# 134. `NewRequestData`：MRV2 的 add / re-add payload

V2 Scheduler producer：

```python
NewRequestData.from_request(
    req,
    req_to_new_blocks[req.request_id].get_block_ids(),
    req._all_token_ids,
)
```

因此：

```text
BlockPool / KV manager
↓
block state
↓
req_to_new_blocks
↓
NewRequestData
```

在当前 MRV2 路径中它服务两类 request：

```text
真正 new
+
preempt 后 resumed
```

而 MRV2 `add_requests()` 会对 `scheduled_new_reqs` 重新建立 request execution state，例如：

```text
req_states.add_request(...)
```

所以现在可以比上一版更明确地定义：

> **`NewRequestData` 是 MRV2 对“需要 add/re-add execution-side state”的 request 使用的较完整同步 payload。**

这里的 `new` 是 execution-state 建立语义，不等价于用户层 Request identity 一定第一次出现。

---

# 135. MRV2 不再依赖 `CachedRequestState` 作为第二份 request backup

此前读 MRV1：

```text
vllm/v1/worker/gpu_model_runner.py
```

会看到：

```text
CachedRequestState
+
persistent InputBatch
```

这套设计的背景是：MRV1 把 persistent state tensors 直接当作 model/sampler inputs，因此 request row 的重排、退出、恢复会使状态管理复杂，还需要 `CachedRequestState` 作为额外 backup。

MRV2 重构后，persistent request state 独立维护。

当前源码已经能看到：

```text
self.req_states
self.model_state
```

官方设计进一步说明：

```text
每个 active request 分配稳定 row
直到 finish / preemption

persistent request state
与
per-step input tensors
分离
```

因此：

```text
MRV1:
CachedRequestState + persistent InputBatch

MRV2:
RequestState / req_states + model_state
→ gather / prepare current-step inputs
```

这正是两套 Model Runner 最重要的结构差异之一。

---

# 136. `req_to_new_blocks` 对 resumed 的含义：当前应理解为 re-add 时需要同步的 block state

Preemption 时 Scheduler：

```text
free old KV ownership
num_computed_tokens → 0
```

Resume 时 Request 重新经过 waiting admission / KV allocation。

随后：

```text
resumed
→ NewRequestData
```

因此对于 MRV2，不需要再维护 MRV1 那种：

```text
旧 CachedRequestState.block_ids
+
append / replace
```

心智模型。

更准确的是：

```text
old MRV2 execution state
已在 preemption 时 remove

resume
↓
Scheduler 形成新的 block state
↓
NewRequestData
↓
MRV2 re-add request state
```

至于 prefix hit、partial reuse、KV connector 等情况下 `NewRequestData.block_ids` 的精确组成，后续需要继续沿 MRV2 block-state 更新源码确认；当前不把它简单等同于“全部 newly allocated blocks”。

---

# 137. `CachedRequestData`：MRV2 continuing request 的增量同步包

V2 构造输出前已经：

```python
scheduled_new_reqs.extend(scheduled_resumed_reqs)
scheduled_resumed_reqs.clear()
```

因此标准 MRV2 主路径：

```text
NewRequestData
→ new + resumed

CachedRequestData
→ continuing RUNNING requests
```

这里的 `cached` 仍然不是 Prefix Cache。

它表达的是：

> **MRV2 已经保留该 active request 的 execution-side persistent state，本 step 只需同步必要 delta。**

因此整个 contract 可以压缩成：

```text
add / re-add
→ richer NewRequestData

continue
→ CachedRequestData delta

finish / preempt
→ remove execution-side state
```

---

# 138. MRV2 的 persistent state 与 per-step InputBatch：这里必须与 MRV1 区分

MRV1 的 `InputBatch` 本身承担 persistent batch state：

```text
CachedRequestState
+
persistent InputBatch
+
复杂 row reorder
```

MRV2 则重新设计为：

```text
Persistent Request State
(req_states / RequestState)
        │
        │ stable request row
        ▼
per-step current request ordering
        │
        ▼
InputBatch / input metadata for this step
        │
        ▼
GPU model forward
```

所以用户之前的直觉：

> “最终 forward 还是要把当前 step 的 token 拼起来；为什么需要中间状态？”

在 MRV2 中答案更清楚：

```text
req_states
= 跨 step 保存、增量更新的 authoritative execution-side request state

InputBatch
= 本 step 从 persistent state gather/prepare 出来的 execution representation
```

也就是说，MRV2 **仍然需要 batch representation**，但不再把“跨 step persistent state”和“本 step forward input”绑成同一个对象。

这正是 MRV2 相比 MRV1 的核心改进。

---

# 139. `new_block_ids`：正式证明 continuing request 只同步 block delta

源码：

```python
new_block_ids.append(
    req_to_new_blocks[req_id].get_block_ids(allow_none=True)
)
```

假设 ModelRunner 当前本地已经有：

```text
A.block_ids = [7, 18]
```

本轮 Scheduler 继续推进 A，KV allocation 跨过一个 block boundary，新分配：

```text
block25
```

则：

```text
req_to_new_blocks[A]
= [25]
```

SchedulerOutput 中：

```text
CachedRequestData.new_block_ids[A]
= [25]
```

Execution side 不需要再收到：

```text
[7, 18, 25]
```

只需要：

```text
已有 [7,18]
+
新增 [25]
↓
[7,18,25]
```

因此这一条已经把 vLLM MRV1-style Persistent Batch 的一个核心思想证明出来：

> **长期状态保留在 ModelRunner，本轮 SchedulerOutput 主要发送变化量。**

---

# 140. Decode 为什么很多 step 根本没有 new block IDs

假设：

```text
block_size = 16
```

Request A 当前已经有 2 blocks：

```text
[block7, block18]
```

它们覆盖：

```text
logical positions [0, 32)
```

如果当前 Decode 只是：

```text
position 20 → 21
```

或者：

```text
position 30 → 31
```

都不需要新增 physical block。

因此该 step 可能是：

```text
num_scheduled_tokens[A] = 1
new_block_ids[A] = None
```

只有当目标 coverage 跨过 block boundary，例如：

```text
32 → 33
```

才需要：

```text
new_block_ids[A] = [new physical block ID]
```

所以：

```text
本轮执行了 token
≠
本轮一定新增 KV block
```

这也是为什么 `num_scheduled_tokens` 和 `new_block_ids` 是两个独立字段。

---

# 141. `num_computed_tokens` 为什么还要同步给 ModelRunner

源码：

```python
num_computed_tokens.append(req.num_computed_tokens)
```

Scheduler 是 request progress 的 control-plane owner。

虽然 ModelRunner 也维护 execution-side cached state，但它不应该自己推导：

```text
Scheduler 现在逻辑上认为 A 到底 computed 到哪里？
```

因此 SchedulerOutput 还会同步当前的 progress state：

```text
Scheduler Request.num_computed_tokens
        ↓
CachedRequestData
        ↓
ModelRunner local state
```

这里要特别谨慎：

```text
CachedRequestData.num_computed_tokens
```

表示的是 **SchedulerOutput 构造时 Scheduler 侧 Request 当前的 computed progress**。

不要在还没看 `_update_after_schedule()` 前就简单写成：

```text
“本轮 GPU 执行结束后的 computed token 数”
```

因为 async / optimistic scheduling 等模式可能改变这个语义。

当前主路径只固定：

> **它用于把 Scheduler authoritative progress 同步给 execution-side persistent state。**

---

# 142. `num_output_tokens` 为什么也属于同步状态

源码：

```python
num_output_tokens.append(
    req.num_output_tokens + req.num_output_placeholders
)
```

普通同步路径下可以先近似理解为：

```text
num_output_tokens ≈ req.num_output_tokens
```

它描述 Scheduler 当前已经记录的 output progress。

对于 ModelRunner 来说，persistent runtime state 不只有：

```text
block IDs
```

还需要知道 request 当前 sequence/runtime 进度，例如：

```text
Prompt 已有多少 token
已经生成多少 output token
下一步 position 在哪里
sampling/runtime state 如何推进
```

因此 CachedRequestData 同步的是：

```text
KV state delta
+
logical request progress
```

而不是纯粹的 block update packet。

---

# 143. 为什么 V2 continuing request 不需要每轮发送完整 token history

在 `_make_cached_request_data()` 中：

```python
if not self.use_v2_model_runner:
    ...
    all_token_ids[req_id] = req.all_token_ids.copy()
```

当前 V2 路径不走这段。

所以：

```text
NewRequestData
→ V2 显式传 req._all_token_ids

CachedRequestData
→ 不再重复发送完整 all_token_ids
```

这说明 V2 的 ModelRunner persistent request state 承担了更强的本地缓存职责。

因此整个数据同步模式非常清楚：

```text
第一次出现：full initialization snapshot

后续 continuing steps：incremental update
```

这正是高频 serving step 中减少重复 state reconstruction / data movement 的关键思路之一。

---

# 144. 为什么 PP 路径可能需要额外 new_token_ids，而当前 PP=1 不需要

源码：

```python
if self.use_pp and not self.scheduler_config.async_scheduling:
    ...
    new_token_ids.append(token_ids)
```

注释说明：PP 场景下 first-stage worker 与 last-stage worker 之间没有普通单进程路径下那样直接共享 sampled token state，因此 Scheduler 可能需要把 sampled token 回传。

当前：

```text
PP = 1
self.use_pp = False
```

所以这部分不进入当前主链。

重要的是理解为什么普通路径可以省掉它：

```text
GPUModelRunner
↓
sample token O0
↓
ModelRunner 自己已经缓存 O0
↓
下一 step 不需要 Scheduler 再把 O0 原样传回来
```

否则会形成无意义的：

```text
ModelRunner → Scheduler → ModelRunner
```

重复数据移动。

这进一步说明 MRV1-style Persistent Batch 的设计目标：

> **能在 execution side 保留的稳定状态，就不要每一步经 Scheduler 重传。**

---

# 145. `req_ids` + 多个 list：CachedRequestData 的 batch-oriented 表示

源码不是为每个 request 创建一个独立小对象，而是：

```text
req_ids[]
new_block_ids[]
num_computed_tokens[]
num_output_tokens[]
```

例如：

```text
req_ids:
[A, B]

new_block_ids:
[None, [25, 31]]

num_computed_tokens:
[500, 400]

num_output_tokens:
[32, 0]
```

这些字段通过相同 index 对齐：

```text
index 0 → Request A
index 1 → Request B
```

因此：

```text
CachedRequestData
```

更像一个 batch-oriented state update structure。

这和后续 GPUModelRunner 面向 batch 做增量更新是自然衔接的。

---

# 146. `num_scheduled_tokens`：和 CachedRequestData 不同，它是本 step 的 execution quota

前面 Scheduler 决策阶段已经有：

```python
num_scheduled_tokens[request_id] = num_new_tokens
```

最终直接进入：

```python
SchedulerOutput(
    num_scheduled_tokens=num_scheduled_tokens,
    total_num_scheduled_tokens=total_num_scheduled_tokens,
)
```

例如：

```text
A Decode:
num_scheduled_tokens[A] = 1

B Chunked Prefill:
num_scheduled_tokens[B] = 255
```

那么：

```text
total_num_scheduled_tokens = 256
```

这和：

```text
new_block_ids
```

不是一回事。

分别回答：

```text
num_scheduled_tokens
= 本轮到底计算多少 token

new_block_ids
= 为支撑本轮及当前 target coverage，本轮新分配了哪些 block
```

例如：

```text
A:
num_scheduled_tokens = 1
new_block_ids = None
```

完全合法。

---

# 147. `total_num_scheduled_tokens` 为什么执行侧也需要

`num_scheduled_tokens` 是 per-request quota：

```text
A → 1
B → 255
```

而：

```text
total_num_scheduled_tokens = 256
```

是整个 execution batch 的 flattened token 数规模。

后续 ModelRunner 准备输入时，会涉及：

```text
input_ids
positions
slot_mapping
attention metadata
```

这些 tensor / metadata 的尺寸都和本 step 总 token 数直接相关。

因此：

```text
per-request scheduling shape
+
global execution batch shape
```

都需要明确表达。

---

# 148. `finished_req_ids`：MRV2 如何同步 Request 生命周期结束

SchedulerOutput：

```text
finished_req_ids
```

MRV2 consumer：

```python
finish_requests(scheduler_output)
```

会最终：

```text
finished request
↓
_remove_request(req_id)
↓
model_state.remove_request(...)
req_states.remove_request(...)
以及相关 encoder / prompt-logprob / LoRA state cleanup
```

因此 Scheduler 与 MRV2 各自维护长期状态，但生命周期变化通过 SchedulerOutput 显式同步：

```text
Scheduler authoritative lifecycle
↓
finished_req_ids
↓
MRV2 execution state cleanup
```

---

# 149. `preempted_req_ids`：MRV2 把 preemption 当作 execution-side completion

Scheduler preempt：

```text
KV ownership free
num_computed_tokens → 0
status → PREEMPTED
重新进入 waiting
```

同时：

```text
reset_preempted_req_ids
↓
SchedulerOutput.preempted_req_ids
```

MRV2：

```python
finished_req_ids = scheduler_output.finished_req_ids
preempted_req_ids = scheduler_output.preempted_req_ids

if preempted_req_ids:
    finished_req_ids = finished_req_ids.union(preempted_req_ids)

for req_id in finished_req_ids:
    self._remove_request(req_id)
```

所以在 MRV2 execution side：

```text
finish
和
preempt
```

都会结束当前 active request slot/state。

区别在 Scheduler：

```text
finished
→ Request 生命周期结束

preempted
→ Scheduler Request 仍存在，将来可能 resume
```

Resume 后：

```text
NewRequestData
↓
MRV2 add/re-add
```

因此要严格区分：

```text
Scheduler Request lifetime
vs
MRV2 active execution-state lifetime
```

---

# 150. `new_block_ids_to_zero`：Block allocation 之后还有 execution-side maintenance

之前在 SingleTypeKVCacheManager 中已经看到：

```python
self.new_block_ids.extend(b.block_id for b in new_blocks)
```

以及：

```text
Whether this manager's new blocks are zeroed by the worker.
```

现在这些 IDs 又通过：

```python
new_block_ids_to_zero=self._get_new_block_ids_to_zero()
```

进入 SchedulerOutput。

因此链条是：

```text
BlockPool allocate new physical blocks
↓
SingleTypeManager 记录 new_block_ids
↓
Scheduler 收集需要 zero 的 IDs
↓
SchedulerOutput
↓
Worker / ModelRunner 做 execution-side KV maintenance
```

说明 `SchedulerOutput` 还负责：

```text
KV allocation result
→ GPU execution-side maintenance instruction
```

当前不用深入 zeroing 原因，只保留这个架构认知。

---

# 151. 一次完整 A/B/C step：从 Scheduler 状态到 SchedulerOutput

假设当前：

```text
running:
A = Decode
B = Chunked Prefill

waiting:
C = New Request
```

并假设：

```text
token_budget = 256
```

状态：

```text
A:
num_tokens = 501
num_computed_tokens = 500
pending = 1
ModelRunner 已缓存 block IDs [7,18]

B:
num_tokens = 1000
num_computed_tokens = 400
pending = 600
ModelRunner 已缓存 block IDs [...]

C:
第一次 admission
ModelRunner 还没有它的 CachedRequestState
```

Scheduler 先处理 running：

```text
A → schedule 1
B → schedule 255
```

假设：

```text
A 没跨 block boundary
→ new_block_ids[A] = None

B 新增 [25,31]
→ new_block_ids[B] = [25,31]
```

由于 token budget 已耗尽：

```text
C 本轮仍 waiting
```

那么 SchedulerOutput 核心概念可以近似为：

```text
scheduled_new_reqs:
[]

scheduled_cached_reqs:
A, B 的增量状态

num_scheduled_tokens:
A → 1
B → 255

new_block_ids:
A → None
B → [25,31]
```

下一 step 如果 C 被 admission：

```text
C → NewRequestData
```

ModelRunner 才会首次建立 C 的 execution-side state。

---

# 152. Continuous Batching 在 MRV2 中真正如何落地

Continuous batching 仍然表示：

```text
Step N:
A B C

Step N+1:
A B D
```

但 MRV2 不再采用 MRV1 那种：

```text
persistent InputBatch 本身不断 remove/add/reorder
```

而是维护稳定的 persistent request slots：

```text
req_states

slot 7  → A
slot 19 → B
slot 31 → C
```

只要 Request active，它的 slot/index 保持稳定。

下一 step 当前要执行：

```text
A B D
```

execution side 再根据本轮 request ordering 从 persistent state gather / prepare 出本 step input representation。

所以 MRV2 的优化仍然利用：

```text
跨 step 状态不从零重建
```

但实现方式从：

```text
MRV1:
persistent batch layout

变成：

MRV2:
persistent request-state table
+
per-step gathered input batch
```

这比 MRV1 更适合 async scheduling、request reorder 和 GPU-native input preparation。

---

# 153. Scheduler 与 MRV2 各自缓存什么

## Scheduler authoritative state

```text
Request objects
waiting / running
num_computed_tokens
num_output_tokens
KVCacheManager
req_to_blocks
BlockPool ownership
```

回答：

> 系统逻辑上谁该运行、运行多少、资源如何分配。

## MRV2 execution-side persistent state

当前源码已经看到：

```text
req_states / RequestState
model_state
```

以及其它 execution-specific state。

它回答：

> 为了后续 GPU execution，这个 active request 已经有哪些可复用的运行状态。

MRV2 的关键变化是：

```text
persistent request state
≠
本 step InputBatch
```

每 step input 是从 persistent state 根据当前 execution ordering 准备出来的。

因此不要再使用 MRV1 的：

```text
CachedRequestState
+
persistent input_batch
```

作为当前 MRV2 心智模型。

---

# 154. block ID 的控制面传递链已建立；MRV2 execution-side 落点重新定位

控制面链仍然成立：

```text
Request token progress
↓
Scheduler.schedule()
↓
KVCacheManager.allocate_slots()
↓
SingleTypeKVCacheManager
↓
BlockPool.get_new_blocks()
↓
physical block IDs
↓
req_to_blocks / req_to_new_blocks
↓
NewRequestData / CachedRequestData
↓
SchedulerOutput
↓
Executor / Worker
↓
MRV2 GPUModelRunner
```

从这里开始，旧笔记中的：

```text
CachedRequestState.block_ids
↓
persistent input_batch.block_table
```

属于 MRV1，不能继续沿用。

当前 MRV2 下一步真正要确认的是：

```text
SchedulerOutput block IDs
↓
MRV2 哪个 persistent block-state structure？
↓
本 step block table 如何 gather / materialize？
↓
slot_mapping 如何在 GPU 上形成？
↓
Attention Backend
```

官方 MRV2 设计说明 block table persistent state 会与 per-step input 解耦，并在 GPU 侧进行增量更新 / gather；但我们接下来仍以当前 v0.26.0 真实源码为准逐层确认。

---

# 155. 这条链为什么对后续 KV 项目非常重要

如果未来要修改：

```text
KV compression
KV quantization
KV offload
KV migration
importance eviction
mixed-precision KV
```

必须先明确你修改的是哪一层。

例如：

## 如果修改“谁应该保留 / 谁应该被驱逐”

更接近：

```text
Scheduler / KVCacheManager policy
```

## 如果修改“Request 需要哪些 logical blocks”

更接近：

```text
SingleTypeKVCacheManager / coordinator
```

## 如果修改“physical block 如何分配 / free / eviction”

更接近：

```text
BlockPool
```

## 如果修改“block 内真实数据格式”

例如 BF16 → INT4：

```text
GPU KV storage / Attention Backend / ModelRunner
```

## 如果修改“Scheduler 分出的 block metadata 如何到 GPU”

则涉及：

```text
SchedulerOutput
→ ModelRunner block_table / slot_mapping
```

因此现在这条源码链最大的意义不是记函数名，而是建立：

> **KV 系统中的 Policy、Resource Management、Metadata Synchronization、GPU Data Layout 是四个不同层次。**

未来做 QCache 时才能避免把策略、allocator 和 Attention kernel 混在同一个地方。

---

# 156. 当前阶段重新形成的完整架构图（MRV2 校正版）

```text
                           EngineCore
                               │
                               ▼
                           Scheduler
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       Request State      Compute Budget      KV Resource
       waiting/running    token_budget        KVCacheManager
             │                 │                 │
             └────────┬────────┘                 │
                      ▼                          │
               num_new_tokens                   │
                      │                          │
                      └─────────────┬────────────┘
                                    ▼
                         KVCacheManager.allocate_slots
                                    │
                                    ▼
                              Coordinator
                                    │
                                    ▼
                         SingleTypeKVCacheManager
                                    │
                                    ▼
                                BlockPool
                                    │
                         physical KV block IDs
                                    │
                                    ▼
                         req_to_blocks / delta
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
                ▼                                       ▼
         NewRequestData                         CachedRequestData
          new + resumed                       continuing running
                │                                       │
                └───────────────────┬───────────────────┘
                                    ▼
                            SchedulerOutput
                                    │
                                    ▼
                                Executor
                                    │
                                    ▼
                                  Worker
                                    │
                                    ▼
                         MRV2 GPUModelRunner
                  vllm/v1/worker/gpu/model_runner.py
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
             req_states / RequestState          model_state
          persistent request state        model-specific state
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                         per-step input preparation
                                    │
                                    ▼
                          block table / slot mapping
                         【下一阶段源码确认】
                                    │
                                    ▼
                           Attention Backend
```

---

# 157. 当前容易混淆的概念再次统一

## `req_to_blocks`

```text
SingleTypeKVCacheManager 内部
Request → 当前完整 block ownership
```

例如：

```text
A → [7,18,25]
```

## `req_to_new_blocks`

```text
Scheduler 当前 step
Request → 本次 allocation 返回的 block delta / initialization block data
```

## `new_block_ids`

```text
经过 .get_block_ids()
进入 NewRequestData / CachedRequestData 的 block ID payload
```

## `BlockPool.blocks`

```text
整个系统所有 physical KVCacheBlock metadata objects
```

## `block_table`

```text
ModelRunner / execution side 使用的 logical block → physical block ID 映射表示
```

当前还没正式读它的实现。

---

# 158. 四组容易混淆的“V1/V2/new/cached”语义

## 1. vLLM V1

```text
Engine architecture generation
```

当前 Scheduler、EngineCore、KV manager 都位于：

```text
vllm/v1/...
```

## 2. Model Runner V1 / V2

这是 **vLLM V1 Engine 内部 execution implementation 的两代实现**：

```text
MRV1:
vllm/v1/worker/gpu_model_runner.py

MRV2:
vllm/v1/worker/gpu/model_runner.py
```

不能把：

```text
vLLM V1
```

误解成：

```text
只能使用 Model Runner V1
```

## 3. Scheduler lifecycle：new / resumed / running

```text
WAITING first admission → new
PREEMPTED re-admission  → resumed
RUNNING continuation    → running
```

## 4. V2 transport：NewRequestData / CachedRequestData

```text
new + resumed
→ NewRequestData

continuing
→ CachedRequestData
```

## 5. Prefix Cache

这是 KV reuse mechanism，与上面两组分类无关。

因此最重要的是：

```text
vLLM V1 / MRV1 / MRV2
是架构层级命名

new / resumed / running
是 Scheduler lifecycle

NewRequestData / CachedRequestData
是 Scheduler → ModelRunner synchronization contract

Prefix Cache
是 KV reuse
```

---

# 159. 当前阅读完成度更新

截至这里，Scheduler/KV 控制面已经完成：

```text
① Scheduler owner/state                         已完成
② Request waiting/running                       已完成
③ Request token progress                        已完成
④ token_budget / num_new_tokens                 已完成
⑤ running/waiting scheduling 第一层             已完成
⑥ KVCacheManager.allocate_slots                 已完成普通主路径
⑦ Coordinator / SingleType manager              已完成核心职责
⑧ BlockPool allocation/free 第一层              已完成
⑨ concrete block ID → req_to_blocks             已完成
⑩ KVCacheBlocks → req_to_new_blocks             已完成第一层
⑪ New / resumed / running → V2 RequestData       已完成 producer 主路径
⑫ SchedulerOutput contract                      已完成第一层
```

当前还没有深入：

```text
Prefix Cache hash lifecycle
partial hit / CoW
Sliding Window / Mamba
KV Connector
PP / async 特殊路径
Speculative Decode
```

这些均不阻塞当前主链。

---

# 160. 新的准确停点：正式进入 MRV2，而不是 MRV1 `_update_states()`

控制面已经确认：

```text
Scheduler internal:
new / resumed / running

MRV2 SchedulerOutput:
(new + resumed) → NewRequestData
running         → CachedRequestData

preempted_req_ids
→ MRV2 finish_requests()
→ _remove_request()
```

下一步必须进入：

```text
vllm/v1/worker/gpu/model_runner.py
```

而不是：

```text
vllm/v1/worker/gpu_model_runner.py
```

下一阶段先建立 MRV2 execution skeleton：

```text
SchedulerOutput
↓
finish_requests()
↓
add_requests()
↓
update continuing request state
↓
req_states / model_state
↓
prepare current-step inputs
↓
model forward
↓
sample
```

然后再回答 KV 主线问题：

```text
1. NewRequestData 的 block IDs 写入 MRV2 哪个 persistent structure？

2. CachedRequestData.new_block_ids 如何增量更新 persistent block state？

3. req_states 的 stable request index 如何和本 step request ordering 对应？

4. MRV2 的 per-step InputBatch 到底是什么？

5. persistent block state 如何 gather 成本 step block table？

6. slot_mapping 如何在 GPU 侧生成？

7. 最终如何进入 Attention Backend / KV write / paged KV read？
```

这才是当前固定 MRV2 路径的正确下一阶段。

---

# 161. 当前阶段一句话总结（MRV2 校正版）

当前已经闭环到：

```text
physical block IDs
↓
req_to_blocks / req_to_new_blocks
↓
NewRequestData / CachedRequestData
↓
SchedulerOutput
↓
MRV2 execution plane
```

并且已经确认一个非常重要的版本边界：

```text
Scheduler / KV control plane
并没有因为 MRV2 被整体重写

真正发生 ground-up redesign 的是：
Model Runner execution side
```

因此当前应该把系统理解成：

> **同一套 vLLM V1 Scheduler / KV control plane，通过 `SchedulerOutput` 同时兼容不同 Model Runner execution implementations；MRV2 重构的是 execution-side persistent state、input preparation、async execution 和 sampling，而不是重新发明一套 scheduling policy。**

下一步只需沿 MRV2 `vllm/v1/worker/gpu/model_runner.py` 继续追 block metadata 如何落到 GPU input。

---

# 162. 为什么在正式进入 MRV2 execution side 前，还需要补一次 EngineCore / Scheduler 闭环

前面已经把控制面一路追到：

```text
Request progress
↓
Scheduler.schedule()
↓
KV allocation
↓
SchedulerOutput
↓
Execution Plane
```

如果现在立刻进入 execution side，正确入口应该是：

```text
vllm/v1/worker/gpu/model_runner.py
```

而不是 MRV1 的：

```text
vllm/v1/worker/gpu_model_runner.py::_update_states()
```

控制面闭环补齐后再进入 MRV2，会更容易理解每个字段的时间语义。

但是还有一个非常重要的问题没有真正闭环：

> **Scheduler 发出一次 SchedulerOutput 以后，GPU 的执行结果到底怎样重新变成下一轮 Scheduler 的 Request state？**

也就是说，我们虽然已经知道：

```text
State_t
↓
schedule()
↓
SchedulerOutput
```

却还没有真正看清：

```text
SchedulerOutput
↓
ModelRunnerOutput
↓
State_t+1
```

这会直接影响几个已经反复出现的核心变量：

```text
num_tokens
num_computed_tokens
num_in_flight_tokens
_output_token_ids
_all_token_ids
Request.status
```

如果这一层不补齐，后面进入 ModelRunner 时容易形成一个错误心智模型：

```text
Scheduler 决定要算 N token
↓
GPU 算完
↓
num_computed_tokens += N
```

真实源码并不是这么简单。

vLLM 为了支持 production runtime 中的：

```text
async scheduling
multiple in-flight batches
pipeline overlap
speculative correction
```

采用的是：

```text
schedule-time optimistic progress
+
result-time reconciliation / correction
```

因此这一小段不是重新深入 EngineCore，而是为了把整个 Scheduler state machine 的闭环真正钉死。

---

# 163. EngineCore 的定位：它不是 Scheduler，也不是 GPU execution implementation

当前已经确认 `EngineCore.step()` 第一层可以压缩为：

```python
scheduler_output = self.scheduler.schedule(...)

future = self.model_executor.execute_model(
    scheduler_output,
    non_block=True,
)

model_output = future.result()

engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output,
    model_output,
)
```

因此 EngineCore 的主要职责不是：

```text
决定谁运行
分配 KV block
准备 GPU tensor
执行 Attention
```

而是把几个长期存在的 subsystem 串成闭环：

```text
Scheduler
= decide / control

Executor + Worker + ModelRunner
= execute

Scheduler.update_from_output()
= consume result / reconcile state

EngineCore
= orchestration loop
```

可以把它理解为：

```text
                 EngineCore
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
      Scheduler   Executor   Scheduler
       decide      execute    reconcile
```

所以当前不需要继续深挖 EngineCore 的：

```text
batch_queue
PP pipeline filling
async queue overlap
future management
```

这些属于更高阶的并发执行优化。

当前只需要理解：

> **EngineCore 负责把一次“决策 → 执行 → 结果回收”循环不断驱动下去。**

---

# 164. Scheduler 一个 step 其实有两个不同的状态推进阶段

看完真实源码后，Scheduler 的一次 step 不能再简单理解成：

```text
schedule()
↓
execute()
↓
update()
```

更精确应该分成：

```text
State_t
│
▼
schedule()
│
├── 选择 Request
├── 决定 num_scheduled_tokens
├── 分配 / 预留 KV resources
└── 构造 SchedulerOutput
│
▼
_update_after_schedule()
│
├── optimistic advance num_computed_tokens
└── increase num_in_flight_tokens
│
▼
Execution Plane
│
├── forward
└── sample
│
▼
ModelRunnerOutput
│
▼
update_from_output()
│
├── decrease num_in_flight_tokens
├── consume sampled token
├── correction / rollback if needed
├── check stop
└── update lifecycle
│
▼
State_t+1
```

这两个状态推进阶段分别解决不同问题：

## Schedule-time state advancement

解决：

```text
“这些 work 我已经发出去了，下一轮不能重复 schedule。”
```

## Result-time reconciliation

解决：

```text
“GPU 实际返回了什么？
哪些 token 被接受？
生成了什么 output？
Request 是否结束？”
```

这就是 production Scheduler 与简单同步 for-loop 的重要区别。

---

# 165. `update_from_output()` 为什么同时需要 SchedulerOutput 和 ModelRunnerOutput

真实函数入口：

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
```

它同时接收两份信息：

```text
SchedulerOutput
= 本轮原计划是什么

ModelRunnerOutput
= 本轮真实执行结果是什么
```

例如 Scheduler 之前决定：

```text
A → schedule 1 token
B → schedule 255 tokens
```

这属于：

```python
scheduler_output.num_scheduled_tokens
```

而 GPU / ModelRunner 最后可能返回：

```text
A → sampled token O17
B → []，因为仍处于 Prefill chunk
```

这属于：

```python
model_runner_output.sampled_token_ids
```

所以 `update_from_output()` 本质是一次：

```text
Plan_t
+
Result_t
↓
reconcile
↓
State_t+1
```

这也是为什么只有 `ModelRunnerOutput` 不够：

ModelRunner 只知道自己产生了什么结果，但 Scheduler 还必须知道：

```text
本轮当初给这个 request schedule 了多少 token？
哪些 token 当前属于 in-flight？
哪些 request 是这次真正执行过的？
```

---

# 166. `update_from_output()` 只遍历本轮真正被 schedule 的 Request

源码核心：

```python
num_scheduled_tokens = scheduler_output.num_scheduled_tokens

for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
    ...
```

这里没有：

```text
遍历整个 Scheduler.requests
遍历所有 waiting
遍历所有 running
```

而是只处理：

> **本 step 真正进入 SchedulerOutput、并实际被送去执行的 request。**

例如：

```text
running: A, B
waiting: C

本轮：
A → 1
B → 255
C → 0
```

那么 result reconciliation 只需要：

```text
A
B
```

C 本轮没有执行，自然没有新的 ModelRunner result 需要吸收。

这体现 SchedulerOutput 的另一个作用：

```text
它不仅是执行计划，
也是 result reconciliation 的 step-level context。
```

---

# 167. 为什么 ModelRunnerOutput 还需要 `req_id_to_index`

源码：

```python
req_index = model_runner_output.req_id_to_index[req_id]

generated_token_ids = (
    sampled_token_ids[req_index]
    if sampled_token_ids
    else []
)
```

这里不能假定：

```text
Scheduler dict 的顺序
=
ModelRunner input_batch 的顺序
```

例如 ModelRunner 当前 batch 可能是：

```text
[B, A]
```

而 Scheduler 遍历顺序可能是：

```text
[A, B]
```

因此必须显式建立：

```text
request_id
↓
req_id_to_index
↓
ModelRunner batch index
↓
该 Request 对应的 sampled result
```

所以 `request_id` 在 Scheduler ↔ Execution Plane 之间承担的是稳定 identity，而不是依赖某个容器的偶然 index。

这也是大规模 continuous batching 中很常见的设计：

```text
logical identity stable
batch position dynamic
```

---

# 168. `_update_after_schedule()`：真实源码证明 `num_computed_tokens` 是先乐观推进的

源码：

```python
def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens

    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        request = self.requests[req_id]

        request.num_computed_tokens += num_scheduled_token
        request.num_in_flight_tokens += num_scheduled_token
```

这里直接证明：

```text
num_computed_tokens
```

不是等到：

```text
GPU 完成
→ update_from_output()
```

以后才增加。

而是在：

```text
SchedulerOutput 已经构造完成
↓
真正结果回来之前
```

就执行：

```text
num_computed_tokens += num_scheduled_token
```

因此之前第一层把它解释成：

```text
“GPU 已经完成计算的 token 数”
```

对于同步直觉是有帮助的，但对 production runtime 来说不够精确。

更准确应该理解成：

> **Scheduler 当前逻辑上已经推进 / 已经承诺出去的 computation frontier。**

特别在 async 场景中，其中一部分可能仍然只是 in-flight。

---

# 169. 为什么必须提前推进 `num_computed_tokens`

源码注释直接说明：

```text
Advance the number of computed tokens here allowing us to
schedule the prefill request again immediately in the next
scheduling step.
```

假设：

```text
Request A
Prompt = 1000
num_computed_tokens = 0
```

本轮 Scheduler：

```text
schedule 256 tokens
```

如果 Scheduler 发出这 256 token 后，仍然保持：

```text
num_computed_tokens = 0
```

而 async scheduling 已经允许 Scheduler 准备下一 batch，那么下一轮又会看到：

```text
pending
= 1000 - 0
= 1000
```

它就可能把：

```text
position [0, 256)
```

重复 schedule 一遍。

正确做法是本轮一旦 work 被发出：

```text
num_computed_tokens:
0 → 256
```

那么下一轮立即看到：

```text
pending
= 1000 - 256
= 744
```

于是可以继续安排：

```text
position [256, 512)
```

因此这里的核心动机不是“假装 GPU 已经完成”，而是：

> **Scheduler 必须把已经发出去的 work 从未来可调度 work 中排除。**

这是允许 scheduling 与 execution overlap 的基础。

---

# 170. `num_in_flight_tokens`：把 optimistic progress 中尚未回收结果的部分标出来

和 `num_computed_tokens` 同时更新：

```python
request.num_computed_tokens += num_scheduled_token
request.num_in_flight_tokens += num_scheduled_token
```

假设：

```text
Before:
num_computed_tokens = 256
num_in_flight_tokens = 0
```

本轮又 schedule：

```text
256 tokens
```

之后：

```text
num_computed_tokens = 512
num_in_flight_tokens = 256
```

可以解释为：

```text
Scheduler logical frontier:
已经推进到 512

其中最后 256：
已经 schedule / 发出
但 update_from_output() 尚未处理结果
```

等结果回来，`update_from_output()` 中：

```python
request.num_in_flight_tokens -= num_tokens_scheduled
```

于是：

```text
num_computed_tokens = 512
num_in_flight_tokens = 0
```

因此第一层可以把两者理解为：

```text
num_computed_tokens
= Scheduler logical processed / committed frontier

num_in_flight_tokens
= 这个 frontier 中尚未完成 result reconciliation 的 token 数
```

在当前 PP=1、普通同步主路径里，in-flight 状态通常存在时间很短；但字段存在是为了同一套 Scheduler state machine 能支持更复杂的 production execution overlap。

---

# 171. 为什么不是等 GPU 完成以后再统一推进状态

一种更简单的同步设计可以是：

```text
schedule N tokens
↓
等待 GPU 完成
↓
num_computed_tokens += N
↓
下一次 schedule
```

它的优点是概念简单，但代价是 Scheduler 与 GPU execution 严格串行：

```text
Schedule
   ↓ wait
GPU Execute
   ↓ wait
Update
   ↓
Next Schedule
```

production serving 为了提高利用率，希望逐步支持：

```text
Schedule batch N+1
       ↕ overlap
Execute batch N
```

甚至 PP / async 环境中可以存在多个未完成 batch。

这要求 Scheduler 的状态能够表示：

```text
“这部分 work 已经被我发出去，不能重复安排；
但它的结果还没有回来。”
```

所以需要：

```text
optimistic num_computed_tokens
+
num_in_flight_tokens
+
result-time correction
```

这套 bookkeeping。

---

# 172. 为什么 `_update_after_schedule()` 必须发生在 SchedulerOutput 构造之后

源码注释第一点：

```text
The scheduler_output of the current step has to include the
original number of scheduled tokens to determine input IDs.
```

这里本质是在保护 **本 step 的 execution context**。

假设：

```text
Before step:
num_computed_tokens = 400

本轮 scheduled = 100
```

那么 ModelRunner 本轮真正要处理的 token range 是：

```text
[400, 500)
```

Scheduler 先基于旧 state：

```text
computed = 400
```

构造本次 SchedulerOutput / RequestData，描述：

```text
“从当前 frontier 开始，本轮推进 100。”
```

之后 Scheduler 自己再：

```text
400 → 500
```

进入下一逻辑状态。

所以顺序是：

```text
State_t
↓
构造 transition description（SchedulerOutput）
↓
advance Scheduler internal state
↓
State_t+1 logical frontier
```

这比简单记住函数调用顺序更重要：

> **SchedulerOutput 描述的是本次 state transition，而 Scheduler 内部随后提前进入 transition 之后的逻辑 frontier。**

---

# 173. `is_prefill_chunk` 也是从 token frontier 自然推导出来的

源码：

```python
request.is_prefill_chunk = request.num_computed_tokens < (
    request.num_tokens + request.num_output_placeholders
)
```

普通同步、无 async placeholder 主路径可以近似成：

```text
is_prefill_chunk
≈ num_computed_tokens < num_tokens
```

例如：

```text
Prompt = 1000
computed = 256
```

则：

```text
256 < 1000
→ True
```

说明这个 Request 虽然已经：

```text
status = RUNNING
```

但仍然处于未完成的 Prefill chunk。

当：

```text
computed = 1000
```

则：

```text
1000 < 1000
→ False
```

这再次验证前面的核心认知：

```text
waiting / running
= lifecycle dimension

Prefill / Decode-like behavior
= token frontier relationship
```

V1 Scheduler 不需要一个额外独立的：

```text
“现在切换到 Decode phase”
```

状态机才能理解主路径。

---

# 174. `update_from_output()` 的第一件核心工作：让 in-flight work 完成 result reconciliation

源码：

```python
request = self.requests.get(req_id)

if request is not None:
    request.num_in_flight_tokens -= num_tokens_scheduled
```

所以：

```text
schedule time:
num_in_flight += N

result time:
num_in_flight -= N
```

形成：

```text
work issued
↓
in-flight
↓
result consumed
↓
not in-flight
```

注意这里没有普通路径的：

```python
request.num_computed_tokens += num_tokens_scheduled
```

因为这个 advance 已经发生在：

```python
_update_after_schedule()
```

中。

这就是为什么 `update_from_output()` 更准确叫：

```text
result reconciliation
```

而不是：

```text
“执行完成以后才第一次更新 Scheduler progress”
```

---

# 175. 为什么 `update_from_output()` 还可能修正 `num_computed_tokens`

源码在 speculative decode branch 中可以看到：

```python
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected
```

源码注释也明确说明：

```text
If some tokens are rejected later,
the number of computed tokens will be adjusted in update_from_output.
```

因此状态模型是：

```text
先 optimistic commit
↓
真实执行结果返回
↓
如果全部有效
    不需要改变 computed frontier
↓
如果有 reject / failure
    correction / rollback
```

这是一种非常典型的 optimistic execution bookkeeping：

```text
fast common path
+
exception correction
```

当前不深入 speculative decode，但这个 branch 很重要，因为它证明 `update_from_output()` 的系统角色：

> **它负责把 Scheduler 的逻辑预测状态与 Execution Plane 的真实结果重新对齐。**

---

# 176. `generated_token_ids`：Execution Plane 怎样把新 token 返回 Scheduler

源码：

```python
req_index = model_runner_output.req_id_to_index[req_id]

generated_token_ids = (
    sampled_token_ids[req_index]
    if sampled_token_ids
    else []
)
```

于是数据链是：

```text
GPU / ModelRunner
↓
logits
↓
sampling
↓
sampled_token_ids
↓
ModelRunnerOutput
↓
req_id_to_index
↓
generated_token_ids for Request A
↓
Scheduler.update_from_output()
```

Scheduler 本身不负责做模型 sampling。

它只负责：

```text
接收 sampling result
↓
写回 Request sequence state
↓
决定 Request 是否继续
```

所以：

```text
ModelRunner
= token producer

Scheduler / Request
= token sequence + lifecycle owner
```

---

# 177. Prefill 为什么通常没有 `new_token_ids`

`_update_request_with_output()` 上方源码注释：

```text
if a request is still being prefilled,
we expect the model runner to return empty token ids for the request.
```

因此一个仍在 Chunked Prefill 的 Request：

```text
Scheduler:
本轮推进 256 prompt tokens
↓
ModelRunner:
forward 256 positions
↓
generated_token_ids = []
```

它的主要 progress 已经通过：

```text
_update_after_schedule()
↓
num_computed_tokens += 256
```

推进。

而在生成阶段，ModelRunner 才产生：

```text
sampled token
```

并进入后面的：

```text
_update_request_with_output()
```

因此必须区分两种变化：

```text
Computation progress
= computed frontier 向前推进

Sequence growth
= sample 出新的 output token，known-token frontier 向前推进
```

这两个 cursor 的交替推进，正是后面统一 Prefill / Decode 的关键。

---

# 178. `_update_request_with_output()`：把真实 sampled token 写回 Request

真实源码非常短：

```python
def _update_request_with_output(
    self, request: Request, new_token_ids: list[int]
) -> tuple[list[int], bool]:
    stopped = False

    for num_new, output_token_id in enumerate(new_token_ids, 1):
        request.append_output_token_ids(output_token_id)

        stopped = check_stop(request, self.max_model_len)
        if stopped:
            del new_token_ids[num_new:]
            break

    return new_token_ids, stopped
```

它只做两件核心事情：

```text
1. append generated token into Request
2. after each append, check whether Request should stop
```

所以整个 result-time sequence update 可以压缩成：

```text
ModelRunner sample token O0
↓
_update_request_with_output()
↓
append_output_token_ids(O0)
↓
Request sequence grows
↓
check_stop()
↓
continue / finished
```

---

# 179. `append_output_token_ids()`：一个 generated token 同时进入两份 Request state

源码：

```python
def append_output_token_ids(
    self,
    token_ids: int | list[int],
) -> None:
    if isinstance(token_ids, int):
        self._output_token_ids.append(token_ids)
        self._all_token_ids.append(token_ids)
    else:
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)
```

因此一个 generated token 有两个身份：

```text
_output_token_ids
= 这是模型生成的输出

_all_token_ids
= 它也已经成为 Request 当前完整 token sequence 的一部分
```

例如初始：

```text
Prompt:
[P0, P1, P2]

_output_token_ids = []
_all_token_ids    = [P0, P1, P2]
```

ModelRunner sample：

```text
O0
```

append 后：

```text
_output_token_ids
[] → [O0]

_all_token_ids
[P0,P1,P2]
→ [P0,P1,P2,O0]
```

而：

```python
@property
def num_tokens(self):
    return len(self._all_token_ids)
```

因此不需要额外：

```python
num_tokens += 1
```

`_all_token_ids` 长度变化以后：

```text
num_tokens
3 → 4
```

自然发生。

---

# 180. Decode 最容易混淆的一点：生成一个 token，不等于这个 token 已经被 forward

这是理解 autoregressive inference scheduler 最重要的细节之一。

假设 Prompt：

```text
[P0 ... P99]
```

Prefill 完成后：

```text
num_tokens = 100
num_computed_tokens = 100
```

最后一个 prompt position 的 logits 被用于 sample：

```text
O0
```

此时经过：

```text
append_output_token_ids(O0)
```

Request 变成：

```text
_all_token_ids:
[P0 ... P99, O0]

num_tokens = 101
num_computed_tokens = 100
```

为什么 `num_computed_tokens` 仍然只有 100？

因为：

```text
O0 已经“生成”出来
```

但：

```text
O0 自己还没有作为下一次模型输入被 forward
```

所以 Scheduler 下一轮看到：

```text
pending
= num_tokens - num_computed_tokens
= 101 - 100
= 1
```

这一个 pending token 就是：

```text
O0
```

下一轮 Scheduler 安排：

```text
forward(O0)
```

ModelRunner 使用 O0 计算新的 logits，再 sample：

```text
O1
```

所以严格区分：

```text
sample O0
≠
compute O0
```

前者让：

```text
known-token frontier +1
```

后者让：

```text
computed frontier +1
```

---

# 181. 两个 cursor：这是理解 V1 Scheduler 最重要的心智模型

现在可以把 Request 的执行状态压缩成两个 cursor。

## Cursor 1：Known-token frontier

由：

```text
num_tokens = len(_all_token_ids)
```

表示：

> Request 当前已经知道多少真实 token。

这个 frontier 会因为：

```text
Prompt 初始化
+
ModelRunner 新 sample output
```

而增长。

## Cursor 2：Computed frontier

由：

```text
num_computed_tokens
```

表示：

> Scheduler 已经安排模型计算到哪个 token position。

它通过：

```text
schedule()
↓
_update_after_schedule()
```

向前追赶 known-token frontier。

所以整个 V1 Scheduler 可以抽象成：

```text
known-token frontier
        ↑
        │ Scheduler 要追赶的目标
        │
computed frontier
```

Scheduler 的核心工作之一就是：

> **不断让 computed frontier 追赶当前 known-token frontier。**

---

# 182. 为什么 Prefill 和 Decode 能被同一个公式统一

## Prefill

假设：

```text
Prompt = 1000 tokens
num_tokens = 1000
num_computed_tokens = 256
```

两个 cursor：

```text
Known frontier:
P0 ................................ P999
                                      ↑ 1000

Computed frontier:
P0 ........ P255
              ↑ 256
```

距离：

```text
1000 - 256 = 744
```

Scheduler 可以从中切一个 chunk：

```text
min(744, token_budget)
```

继续追赶。

## Decode

假设 Prompt 已完成，并刚 sample O0：

```text
num_tokens = 1001
num_computed_tokens = 1000
```

两个 cursor：

```text
Known frontier:
P0 ........ P999 O0
                  ↑ 1001

Computed frontier:
P0 ........ P999
              ↑ 1000
```

距离：

```text
1001 - 1000 = 1
```

Scheduler 仍然只是做：

```text
让 computed frontier 追上 known frontier
```

因此：

```text
Prefill
= known frontier 领先很多

Decode
= known frontier 通常领先 1
```

算法抽象是一致的。

这比死记：

```text
“V1 没有 Prefill phase / Decode phase”
```

更重要，因为这里已经知道它为什么可以这样设计。

---

# 183. Decode 的完整稳态循环

假设 Prompt 长度为 100，Prefill 已完成。

## Round 0：sample O0

```text
computed = 100
known    = 100
↓
ModelRunner 根据最后 logits sample O0
↓
append O0
↓
known = 101
computed = 100
```

出现：

```text
1 token pending
```

## Round 1：compute O0

Scheduler：

```text
schedule 1
```

`_update_after_schedule()`：

```text
computed:
100 → 101
```

ModelRunner：

```text
forward(O0)
↓
sample O1
```

append O1：

```text
known:
101 → 102
```

于是：

```text
known = 102
computed = 101
```

再次产生 1 token pending。

## Round 2：compute O1

重复：

```text
schedule O1
↓
computed +1
↓
forward O1
↓
sample O2
↓
known +1
```

所以正常 Decode 稳态表现为：

```text
sample
→ known frontier +1
→ Scheduler sees 1 pending token
→ compute frontier +1
→ sample next token
```

这是整个 autoregressive serving closed loop。

---

# 184. 为什么 `check_stop()` 必须发生在 append 之后

源码顺序：

```python
request.append_output_token_ids(output_token_id)

stopped = check_stop(request, self.max_model_len)
```

不能反过来。

因为 stop condition 可能依赖：

```text
刚生成的 token 是否 EOS / stop token
当前 output length
当前 total sequence length
sampling_params 中的 stopping policy
```

所以逻辑必须是：

```text
先接受新 token 进入 Request state
↓
再判断“生成到这里以后是否应该停止”
```

而不是：

```text
还没有把 token 放进 Request
↓
就先判断生成这个 token 后是否结束
```

这也是 lifecycle policy 属于 Scheduler / Request control plane 的体现。

---

# 185. 为什么一次可能处理多个 `new_token_ids`

普通单-token Decode 中经常是：

```text
new_token_ids = [O0]
```

但源码使用：

```python
for num_new, output_token_id in enumerate(new_token_ids, 1):
```

说明系统允许一次 result 中出现多个 token，例如 speculative decode 场景。

假设：

```text
new_token_ids = [31, 52, EOS, 77, 91]
```

Scheduler 必须逐个：

```text
append 31 → check stop
append 52 → check stop
append EOS → stop
```

后面的：

```text
77
91
```

不能再作为有效用户输出。

因此源码：

```python
if stopped:
    del new_token_ids[num_new:]
    break
```

会把返回 token 截断到：

```text
[31, 52, EOS]
```

当前不深入 speculative decode，但这里再次证明：

> **GPU / ModelRunner 负责产生 candidate output；Scheduler / Request lifecycle 决定哪些 output 最终被接受并使 Request 继续或终止。**

---

# 186. 为什么 stop / finish policy 不应该放在 ModelRunner

ModelRunner 最适合负责：

```text
input preparation
model forward
logits
sampling
GPU execution state
```

而 Request 是否应该停止通常依赖：

```text
EOS
max_tokens
max_model_len
stop token / stop string
sampling parameters
request-specific policy
```

这些都是 serving / request lifecycle 语义。

如果让 ModelRunner 自己决定：

```text
“这个 request finished”
```

就会把：

```text
GPU execution implementation
```

和：

```text
serving lifecycle policy
```

耦合起来。

因此当前职责可以压缩成：

```text
ModelRunner
= 产生 token

Request
= 保存 token sequence / parameters

check_stop()
= 判断 generation lifecycle

Scheduler
= authoritative request lifecycle owner
```

这和前面：

```text
BlockPool 不负责决定 preempt 谁
```

是同一种架构原则：

> **底层执行 / 资源模块提供事实，上层 control plane 决定 policy。**

---

# 187. `finished_req_ids = set()` 为什么不能直接 `.clear()`

`_update_after_schedule()` 结尾：

```python
self.finished_req_ids = set()
self.reset_preempted_req_ids = set()
```

源码特别说明不能简单：

```python
self.finished_req_ids.clear()
```

原因是 SchedulerOutput 已经持有旧 set object。

例如：

```text
old_set = {A, B}

SchedulerOutput.finished_req_ids
        │
        └────→ old_set

Scheduler.finished_req_ids
        │
        └────→ old_set
```

如果执行：

```python
old_set.clear()
```

那么：

```text
SchedulerOutput 里的 finished_req_ids
```

也会一起变空，因为两边引用同一个 Python object。

正确做法：

```python
self.finished_req_ids = set()
```

是：

```text
SchedulerOutput
→ 继续持有 old_set

Scheduler
→ 换到一个新的 empty set
```

这个细节很小，但很能体现读大项目源码时要区分：

```text
变量名
vs
Python object identity / reference ownership
```

---

# 188. 现在 EngineCore / Scheduler 的真实闭环已经可以完整画出

```text
                         Request State_t
                 ┌──────────────────────────┐
                 │ num_tokens               │
                 │ num_computed_tokens      │
                 │ num_in_flight_tokens     │
                 │ _output_token_ids        │
                 │ _all_token_ids           │
                 │ status                   │
                 └─────────────┬────────────┘
                               │
                               ▼
                       Scheduler.schedule()
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
           choose request   token quota   KV allocation
                 │             │             │
                 └─────────────┴─────────────┘
                               │
                               ▼
                        SchedulerOutput
                               │
                               ▼
                    _update_after_schedule()
                               │
                 num_computed_tokens += N
                 num_in_flight_tokens += N
                               │
                               ▼
                    Executor / Worker / ModelRunner
                               │
                       forward + sampling
                               │
                               ▼
                       ModelRunnerOutput
                               │
                               ▼
                    Scheduler.update_from_output()
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       in_flight -= N    append sampled token   correction
                              │                  if needed
                              ▼
                     _all_token_ids grows
                              │
                              ▼
                         check_stop()
                              │
                              ▼
                         Request status
                              │
                              ▼
                         State_t+1
                              │
                              └────────────→ next schedule()
```

这已经是当前主路径最重要的 runtime state machine。

---

# 189. 把 SchedulerOutput 再放回这个闭环中看，它的定位更加清楚

前面把 SchedulerOutput 定义成：

```text
Control Plane → Execution Plane 的 step contract
```

现在可以再补一层：

它同时还是：

> **Scheduler 用来描述“这次 state transition 到底发出了哪些 work”的不可缺少上下文。**

因为 result 回来以后：

```python
update_from_output(
    scheduler_output,
    model_runner_output,
)
```

还必须利用原来的：

```text
num_scheduled_tokens
request IDs
spec scheduling info
```

来解释 ModelRunnerOutput。

所以 SchedulerOutput 同时连接：

```text
             Scheduler State_t
                    │
                    ▼
               SchedulerOutput
                 /          \
                /            \
               ▼              ▼
        Execution Plane   Result Reconcile
               │              ▲
               ▼              │
        ModelRunnerOutput ─────┘
```

这让它不只是一个“下发 packet”，而是整个 step 的 transaction / transition context。

---

# 190. 当前可以如何更精确地定义几个核心变量

## `num_tokens`

```text
Request 当前已经知道的真实 token sequence 长度
= len(_all_token_ids)
```

由：

```text
Prompt initialization
+
accepted sampled output
```

增长。

## `num_computed_tokens`

第一层精确定义：

> **Scheduler 当前逻辑上已经推进到的 computation frontier。**

在普通同步情况下，它很接近已经执行完成的 token 数；在 async / overlapping 场景中，它可能包含已经 schedule 但尚未完成 result reconciliation 的 token。

## `num_in_flight_tokens`

```text
num_computed_tokens frontier 中
已经 schedule / 发出
但结果尚未被 update_from_output() 消费的 token 数
```

## `num_scheduled_tokens`

```text
某个具体 Scheduler step
给 Request 分配的 computation delta
```

## `_all_token_ids`

```text
Prompt + accepted generated outputs
```

它决定 `num_tokens`。

## `_output_token_ids`

```text
只记录模型真正生成并被 Request 接受的 output token
```

这些变量分别属于不同时间尺度：

```text
persistent request state
vs
step-level transition state
```

不能互相替代。

---

# 191. 一个完整例子：Chunked Prefill 从 State_t 到 State_t+1

假设：

```text
Prompt = 1000
num_tokens = 1000
num_computed_tokens = 256
num_in_flight_tokens = 0
```

本轮：

```text
token_budget 足够给 A 256
```

## Schedule

```text
pending = 1000 - 256 = 744
↓
num_scheduled_tokens[A] = 256
```

KV allocation 保证：

```text
本轮后 target coverage = 512
```

SchedulerOutput 被构造。

## `_update_after_schedule()`

```text
num_computed_tokens:
256 → 512

num_in_flight_tokens:
0 → 256
```

## ModelRunner execution

```text
forward positions [256,512)
```

由于仍然在 Prefill：

```text
generated_token_ids = []
```

## `update_from_output()`

```text
num_in_flight_tokens:
256 → 0
```

普通路径没有 reject：

```text
num_computed_tokens 继续保持 512
```

Request 状态变成：

```text
num_tokens = 1000
num_computed_tokens = 512
```

下一轮：

```text
pending = 488
```

继续追赶 known frontier。

---

# 192. 一个完整例子：Decode 从 State_t 到 State_t+1

假设 Prompt 已完成，并且 Request 当前：

```text
_all_token_ids = [P0 ... P99, O0]

num_tokens = 101
num_computed_tokens = 100
num_in_flight_tokens = 0
```

这里 `O0` 已经 sample 出来，但还没有作为 input forward。

## Schedule

```text
pending = 101 - 100 = 1
↓
num_scheduled_tokens[A] = 1
```

KV 只在跨 block boundary 时才新增 physical block。

## `_update_after_schedule()`

```text
num_computed_tokens:
100 → 101

num_in_flight_tokens:
0 → 1
```

## ModelRunner

```text
forward(O0)
↓
logits
↓
sample O1
```

ModelRunnerOutput：

```text
generated_token_ids = [O1]
```

## `update_from_output()`

先：

```text
num_in_flight_tokens:
1 → 0
```

然后：

```text
append_output_token_ids(O1)
```

得到：

```text
_output_token_ids:
[..., O0] → [..., O0, O1]

_all_token_ids:
[P0...P99,O0]
→ [P0...P99,O0,O1]

num_tokens:
101 → 102
```

而：

```text
num_computed_tokens = 101
```

所以新的状态：

```text
num_tokens = 102
num_computed_tokens = 101
```

下一轮又得到：

```text
pending = 1
```

这就是稳定 Decode loop。

---

# 193. 这段源码真正体现的架构动机

## 193.1 Scheduler 不等待 GPU 才能推进逻辑 frontier

为了允许 overlap：

```text
work issued
→ logical progress advanced
→ next work can be planned
```

而不是 GPU 完成以后才允许下一次 scheduling。

## 193.2 “计划状态”和“真实结果”是两种不同事实

SchedulerOutput 记录：

```text
我计划 / 发出了什么
```

ModelRunnerOutput 记录：

```text
执行面实际产生了什么
```

`update_from_output()` 把二者 reconcile。

## 193.3 Request sequence growth 与 computation progress 分离

```text
sample output
→ num_tokens grows

schedule/compute token
→ num_computed_tokens grows
```

两者交替前进，形成 autoregressive loop。

## 193.4 V1 的 Prefill / Decode 统一不是“没有区别”

两者当然在 execution behavior 上不同：

```text
Prefill → 一次处理大量已知 tokens
Decode  → 通常一次处理 1 个 newly known token
```

真正统一的是 Scheduler 的 progress abstraction：

```text
computed frontier
追赶
known-token frontier
```

---

# 194. 对后续 MRV2 阅读有什么直接帮助

现在回到 MRV2：

```text
vllm/v1/worker/gpu/model_runner.py
```

时，我们已经知道 SchedulerOutput 中的状态处在什么时间点。

特别是：

```text
SchedulerOutput 构造
发生在
_update_after_schedule() optimistic advance 之前
```

因此 MRV2 收到的 RequestData 描述的是本次 state transition 所需的输入状态 / delta，而 Scheduler 内部随后已经提前推进自己的 logical frontier。

更重要的是，现在版本边界已经清楚：

```text
不能再沿 MRV1：
CachedRequestState
→ persistent InputBatch
→ _prepare_inputs()

而应该沿 MRV2：
finish/remove
→ add/update req_states
→ model_state
→ per-step input preparation
```

接下来读：

```text
NewRequestData
CachedRequestData
num_scheduled_tokens
```

时始终问：

```text
它更新哪个 persistent RequestState？
这个状态在 CPU/UVA/GPU 哪一侧？
本 step 又如何从 persistent state gather 出实际 input？
```

这三个问题比继续记 MRV1 的对象名更重要。

---

# 195. EngineCore / Scheduler 闭环到这里应该停止继续深挖

目前已经足够回答：

```text
为什么 EngineCore 需要 schedule → execute → update 闭环？

为什么 Scheduler 不能只 schedule 一次然后忘掉结果？

为什么 num_computed_tokens 会在 GPU 结果回来前推进？

num_in_flight_tokens 为什么存在？

sampled token 怎样进入 Request？

num_tokens 为什么会增加？

为什么 Decode 每轮又产生新的 1-token pending work？

为什么 Prefill / Decode 可以共享同一个 Scheduler progress model？

Request finished 为什么属于 result-time lifecycle decision？
```

当前不继续深入：

```text
Spec Decode accept/reject 细节
KVConnector invalid block recovery
routed experts
structured output grammar
encoder cache release
PP / async 多 frame corner cases
```

因为它们都不是下一步进入 MRV2 GPUModelRunner 的阻塞项。

---

# 196. 更新后的完整主链：Control Plane 已经真正闭环

现在整个控制面不再只是单向：

```text
Request
→ Scheduler
→ KV
→ SchedulerOutput
→ ModelRunner
```

而是完整循环：

```text
Request State_t
│
├── known token frontier
│   └── num_tokens
│
├── computed frontier
│   └── num_computed_tokens
│
└── lifecycle
    └── status
        │
        ▼
Scheduler.schedule()
        │
        ├── select requests
        ├── compute token quota
        ├── KV feasibility / allocation
        └── construct SchedulerOutput
        │
        ▼
_update_after_schedule()
        │
        ├── computed frontier optimistic advance
        └── in-flight bookkeeping
        │
        ▼
SchedulerOutput
        │
        ▼
Executor
        │
        ▼
Worker
        │
        ▼
GPUModelRunnerV2
        │
        ├── prepare execution state
        ├── forward
        └── sample
        │
        ▼
ModelRunnerOutput
        │
        ▼
Scheduler.update_from_output()
        │
        ├── reconcile in-flight work
        ├── correction if needed
        ├── append accepted outputs
        ├── check stop
        └── update request lifecycle
        │
        ▼
Request State_t+1
        │
        └──────────────→ next Scheduler step
```

这就是当前应该保留的 production inference runtime 心智模型。

---

# 197. 与前面 KV 控制链合并后的完整全景

现在可以把两条之前分开的链真正合起来：

## Scheduling / KV allocation path

```text
Request progress
↓
Scheduler.schedule()
↓
num_scheduled_tokens
↓
KVCacheManager.allocate_slots()
↓
Coordinator
↓
SingleTypeKVCacheManager
↓
BlockPool
↓
physical block IDs
↓
SchedulerOutput
```

## Execution / result path

```text
SchedulerOutput
↓
GPUModelRunnerV2
↓
forward / sample
↓
ModelRunnerOutput
↓
Scheduler.update_from_output()
↓
append output token / stop / correction
↓
Request next state
```

合并后：

```text
                     Request State_t
                           │
                           ▼
                    Scheduler.schedule
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
      Compute quota                  KV capacity
            │                             │
            │                      KVCacheManager
            │                             │
            │                         Coordinator
            │                             │
            │                  SingleTypeKVCacheManager
            │                             │
            │                         BlockPool
            │                             │
            └──────────────┬──────────────┘
                           ▼
                    SchedulerOutput
                           │
                           ▼
                   GPUModelRunnerV2
                           │
                  forward + sampling
                           │
                           ▼
                    ModelRunnerOutput
                           │
                           ▼
                Scheduler.update_from_output
                           │
                           ▼
                     Request State_t+1
                           │
                           └──────→ next step
```

到这里，Scheduler / KV control plane 的主链才算真正闭环，而不是只完成了“向 GPU 发任务”的半条链。

---

# 198. 当前最值得记住的 8 个结论

## 1

```text
EngineCore
= orchestrator
```

负责驱动：

```text
schedule → execute → reconcile
```

## 2

```text
Scheduler
= stateful controller
```

不是一次性 batch builder。

## 3

```text
SchedulerOutput
= 当前 step 的 execution contract + transition context
```

既用于下发 execution，也用于 result reconcile。

## 4

```text
num_computed_tokens
```

在 production 语义上是 Scheduler 的 logical computation frontier，不严格等于“已经完成 result 回收的 GPU token 数”。

## 5

```text
num_in_flight_tokens
```

标记 logical frontier 中尚未完成 result reconciliation 的部分。

## 6

```text
num_tokens
```

由 `_all_token_ids` 的长度决定；sampled output append 后 known-token frontier 自动增长。

## 7

Prefill / Decode 在 Scheduler 层能够统一，是因为两者都可以抽象成：

```text
computed frontier
追赶
known-token frontier
```

区别只是 frontier distance 通常分别是：

```text
Prefill → 很大
Decode  → 约 1
```

## 8

```text
ModelRunner 产生 token
Scheduler / Request 决定 token 如何改变 lifecycle
```

这体现 control plane 与 execution plane 的职责边界。

---

# 199. 更新后的准确停点：进入 MRV2 execution main path

控制面已经完整回答：

```text
Request pending work
↓
Scheduler token quota
↓
KV allocation
↓
SchedulerOutput
↓
optimistic progress
↓
ModelRunnerOutput
↓
Scheduler reconcile
```

同时版本关系已经校正：

```text
vLLM V1 Engine
├── shared Scheduler / KV control plane
├── Model Runner V1
└── Model Runner V2（当前固定路径）
```

所以下一步固定进入：

```text
vllm/v1/worker/gpu/model_runner.py
```

第一轮不要钻 block table，先建立：

```text
execute_model()
│
├── finish_requests()
├── add_requests()
├── update continuing requests
├── req_states / model_state
├── input preparation
├── forward
└── sampling
```

然后才继续：

```text
SchedulerOutput block IDs
↓
MRV2 persistent block state
↓
per-step block table
↓
slot_mapping
↓
Attention metadata/backend
↓
KV write / paged KV read
```

这里不再回到 MRV1 的 `_update_states()`，除非后续专门做 MRV1/MRV2 对比。

---

# 200. 本阶段最终一句话总结（MRV2 路径校正版）

Scheduler / KV 控制面主链仍然成立：

```text
Scheduler 决定 token work
→ KVCacheManager / BlockPool 提供 physical block resources
→ SchedulerOutput 把 execution contract 交给 Worker
→ ModelRunner 执行
→ ModelRunnerOutput 返回结果
→ Scheduler reconcile
```

这次版本校正增加了一个必须长期保留的认知：

> **vLLM V1 是 Engine / Scheduler 这一代整体架构；Model Runner V1 与 Model Runner V2 是这个 V1 Engine 内部 execution side 的两套实现。MRV2 是针对旧 Model Runner 技术债的重新实现，而不是 Scheduler V2。**

MRV1 的核心状态形状：

```text
CachedRequestState
+
persistent InputBatch
```

不应再作为当前路径。

当前 MRV2 应改成：

```text
SchedulerOutput
↓
finish/remove + add/update
↓
req_states / RequestState
+
model_state
↓
per-step input preparation
↓
block table / slot mapping
↓
Attention Backend
```

因此下一阶段真正的问题不是“旧 `_update_states()` 怎么维护 persistent batch”，而是：

> **MRV2 如何把 SchedulerOutput 的 request/token/block delta 写入稳定的 persistent request state，并在 GPU 上高效地构造当前 step 的 block table、slot mapping 和 Attention 输入。**

---

