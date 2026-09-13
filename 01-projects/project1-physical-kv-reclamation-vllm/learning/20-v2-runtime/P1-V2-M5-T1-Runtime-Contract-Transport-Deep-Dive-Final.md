# P1 V2 — M5-T1 Runtime Contract & Transport Plumbing
## 结项学习笔记 / 链路冻结 / 设计决策总结

**Project:** P1 — Physical KV Cache Reclamation for vLLM  
**Phase:** V2 — Token-Level Physical KV Compaction  
**Slice:** M5-T1 — Compaction Plan / Result Contract + Transport Plumbing  
**Status:** `PASS / CLOSED / FROZEN`

---

# 1. 这一轮到底解决了什么

M4 已经解决的是数据面：

```text
Paged KV
    ↓
Gather
    ↓
Independent Scratch
    ↓
Writeback
    ↓
Compacted Physical Prefix
```

也就是：**KV payload 怎么搬。**

但 M4 没有回答：

```text
谁告诉 Worker 要 compact？
这个 plan 怎么进入 Worker？
Worker 做完以后怎么把结果告诉 Scheduler？
Scheduler 在哪里看到这个结果？
```

M5-T1 解决的就是这四个问题。

所以这一轮不是数据面，也不是 ownership commit，而是建立：

```text
Scheduler
    ↓
CompactionPlanData
    ↓
SchedulerOutput
    ↓
GPUModelRunner

以及：

GPUModelRunner / Worker
    ↓
CompactionResultData
    ↓
ModelRunnerOutput
    ↓
Scheduler.update_from_output()
```

一句话：

> **M5-T1 建的是“去程 Plan + 回程 Result”的 runtime transport contract。**

---

# 2. M5 总链路

最终 M5 的完整目标仍然是：

```text
Explicit Compaction Plan
        ↓
Scheduler-side binding
        ↓
SchedulerOutput
        ↓
GPUModelRunner.execute_model()
        ↓
model.forward()
        ↓
POST_FORWARD_PRE_EXECUTE_STATE
        ↓
M4 compact_paged_kv_triton_2d()
        ↓
Worker BlockTable / E commit
        ↓
CompactionResultData
        ↓
ModelRunnerOutput
        ↓
Scheduler.update_from_output()
        ↓
Scheduler canonical reconcile
        ↓
request.effective_kv_len commit
        ↓
deferred free / step fence
        ↓
BlockPool reuse
```

M5-T1 只完成：

```text
Plan contract
Transport to Worker
Result contract
Transport back to Scheduler
```

还没有：

```text
真实 compaction
Worker state mutation
Scheduler canonical mutation
free
reuse
```

---

# 3. 为什么必须拆成 Plan 和 Result

M5 transaction 是双向的。

## 去程：Plan

```text
Scheduler
    ↓
Worker
```

语义：

> “请你基于这份 source state，执行这次 token-level compaction。”

对象：

```python
CompactionPlanData
```

## 回程：Result

```text
Worker
    ↓
Scheduler
```

语义：

> “我已经基于那份 source state 做完了，现在 physical shape 变成这样。”

对象：

```python
CompactionResultData
```

两者不能硬塞进同一个类，因为生命周期不同：

```text
Plan
= intent / command-like message

Result
= completed fact
```

如果混在一起，会模糊：

```text
new_E 是计划值还是执行结果？
keep 是 Scheduler 要求还是 Worker 实际结果？
source E 是 expected 还是 actual？
```

所以分开是正确的 transaction contract。

---

# 4. `@dataclass(frozen=True)` 的意义

两个 contract 都使用：

```python
@dataclass(frozen=True)
```

`@dataclass` 会自动生成：

```text
__init__
__repr__
__eq__
```

适合“只负责装数据”的结构。

例如：

```python
plan = CompactionPlanData(
    request_id="A",
    keep_member_indices=[0, 2, 5],
    expected_source_effective_kv_len=10,
    expected_source_num_blocks=3,
)
```

可以直接：

```python
plan.request_id
plan.keep_member_indices
```

`frozen=True` 表示字段创建后不可重新赋值：

```python
plan.expected_source_effective_kv_len = 20
```

会失败。

设计意图：

```text
Plan / Result
应该是 transaction message
而不是可随意修改的 runtime state
```

但注意：

```python
keep_member_indices: list[int]
```

list 本身仍是 mutable，所以 `frozen=True` 不是深度 immutable。后续如需更严格，可考虑 `tuple[int, ...]`，但当前 T1 不是 blocker。

---

# 5. `CompactionPlanData`

定义：

```python
@dataclass(frozen=True)
class CompactionPlanData:
    request_id: str
    keep_member_indices: list[int]
    expected_source_effective_kv_len: int
    expected_source_num_blocks: int
    step_seq: int | None = None
```

它可以理解成：

```text
Scheduler 发给 Worker 的施工单
```

## 5.1 `request_id`

说明这份 plan 属于哪个 request。

因为一轮可以同时调度多个 request，必须避免 request 串线。

## 5.2 `keep_member_indices`

这是 V2 compaction 的核心输入。

语义：

```text
当前 physical member sequence 中
哪些 member 要保留
```

例如：

```text
m0 m1 m2 m3 m4 m5 m6 m7 m8 m9
```

plan：

```text
[0,2,5,8,9]
```

表示保留：

```text
m0 m2 m5 m8 m9
```

它不是 logical token id、block id、slot id。

## 5.3 `expected_source_effective_kv_len`

这是 source-state fence 的第一部分。

表示：

```text
这份 plan 是针对多长的 post-forward physical member sequence 生成的
```

例如：

```text
forward-entry E = 32
current q = 16
post-forward E_source = 48
```

则：

```python
expected_source_effective_kv_len = 48
```

非常重要：这是 **POST-FORWARD E**，不是 forward-entry E。

## 5.4 `expected_source_num_blocks`

这是 source-state fence 的第二部分。

表示 Worker 执行 compaction 前，当前 request 的 physical block row 应该有多少页。

例如：

```text
E_source = 10
block_size = 4
```

则：

```text
expected_source_num_blocks = 3
```

两者分别检查：

```text
E_source
= member/token extent

num_blocks
= physical page-row shape
```

## 5.5 `step_seq`

可选 step identity，用来表达这份 plan 属于哪个 scheduler/execution step。

当前 T1 只是 transport capability；真正是否 enforce，留给 T2。

---

# 6. `CompactionResultData`

定义：

```python
@dataclass(frozen=True)
class CompactionResultData:
    request_id: str
    expected_source_effective_kv_len: int
    expected_source_num_blocks: int
    new_effective_kv_len: int
    new_num_blocks: int
    step_seq: int | None = None
```

表示：

```text
Worker 已完成 post-forward physical compaction
现在告诉 Scheduler：
我是基于什么 source 做的
以及最终 physical shape 是什么
```

## 6.1 为什么 Result 里还要带 expected old state

这是为了后续 transaction validation。

Scheduler 可以做：

```text
if canonical source still matches expected:
    commit new state
else:
    fail closed
```

所以 Result 同时表达：

```text
我基于哪份旧状态做的
+
我做完以后变成什么
```

## 6.2 `new_effective_kv_len`

就是：

```text
K = len(keep_member_indices)
```

表示 compact 后真正有效的 physical KV member 数。

它不是 `num_computed_tokens`，logical progress 不变。

## 6.3 `new_num_blocks`

表示 compact 后还需要多少 physical pages：

```text
ceil(new_effective_kv_len / block_size)
```

例如：

```text
K = 5
block_size = 4
new_num_blocks = 2
```

后续 Scheduler 用 canonical prefix 进行 reconcile。

---

# 7. CHG-02：`SchedulerOutput.compaction_plans`

新增：

```python
compaction_plans: dict[str, CompactionPlanData] = field(
    default_factory=dict
)
```

这里的 `str` 就是 `request_id`。

例如：

```python
{
    "A": CompactionPlanData(...),
    "C": CompactionPlanData(...),
}
```

即：

```text
request A → Plan A
request C → Plan C
```

## `field(default_factory=dict)`

表示：

```text
如果构造 SchedulerOutput 时没传 compaction_plans
自动创建一个新的空 dict
```

正常路径就是：

```python
compaction_plans = {}
```

这样旧流程不受影响。

不能直接写：

```python
compaction_plans = {}
```

因为 mutable default 可能让多个实例共享同一个 dict。

---

# 8. CHG-03：Plan 如何进入 Scheduler 并发给 Worker

完整链路：

```text
External Explicit Plan Producer
        ↓
Scheduler._set_prepared_compaction_plan()
        ↓
Scheduler._prepared_compaction_plans
        ↓
Scheduler.schedule()
        ↓
request 本轮真的被 schedule
        ↓
pop 对应 plan
        ↓
SchedulerOutput.compaction_plans
```

## `_set_prepared_compaction_plan()`

这是 plan ingress。

它不负责 importance、top-k、retention policy，只接收已经准备好的 plan。

## `_prepared_compaction_plans`

Scheduler-side pending plan buffer。

plan 先暂存在这里，等待 request 真正进入当前 schedule。

## 为什么 schedule-time materialization

有 plan 不代表这个 request 当前 step 一定被调度。

所以：

```text
只有 request 本轮真正 scheduled
才把 plan 装入 SchedulerOutput
```

## 为什么 `pop`

表示 plan 是 one-shot。

当前 step 发送以后从 pending map 删除，不会下一轮重复发送。

---

# 9. CHG-04：为什么 `ModelRunnerOutput` 加 `compaction_results`

新增：

```python
compaction_results: list[CompactionResultData] = field(
    default_factory=list
)
```

`ModelRunnerOutput` 是 Worker→Scheduler 的 per-step result carrier。

链路：

```text
GPUModelRunner / model executor
        ↓
ModelRunnerOutput
        ↓
EngineCore
        ↓
Scheduler.update_from_output()
```

所以它天然适合携带 `CompactionResultData`。

## 为什么 Result 用 list

一轮可以：

```text
A 做 compaction
B 不做
C 做 compaction
```

于是：

```python
[result_A, result_C]
```

每个 result 内已经有 `request_id`。

## Plan 用 dict，Result 用 list

Plan：

```text
执行前经常需要按 request_id 查
```

所以用 dict。

Result：

```text
执行后只需把本轮产生的若干结果带回来
```

所以 list 足够。

这是 implementation decision，不是 architecture 必须。

---

# 10. CHG-05：Plan 到 Worker 后发生什么

新增：

```python
@staticmethod
def extract_compaction_plans(...)
```

它只做：

```text
读取
+
structural validation
```

不执行 compaction。

## 为什么 `@staticmethod`

因为它只依赖 `scheduler_output`，不依赖 `self`。

## 它验证什么

1. `compaction_plans` 必须是 dict  
2. 每个 value 必须是 `CompactionPlanData`  
3. dict key 的 request_id 必须等于 `plan.request_id`

例如：

合法：

```python
{
    "A": CompactionPlanData(request_id="A", ...)
}
```

非法：

```python
{
    "A": CompactionPlanData(request_id="B", ...)
}
```

这样可以避免 request cross-wire。

---

# 11. `execute_model()` 为什么一进入就 extract plan

当前：

```python
compaction_plans = self.extract_compaction_plans(
    scheduler_output
)
```

表示：

```text
Worker 一进入当前 execution step
就先把本轮对应的 plan 读出来并固定下来
```

此时还没有：

```text
M4 invocation
KV mutation
E mutation
BlockTable mutation
```

只是 read / validate / keep local reference。

---

# 12. `ExecuteModelState` 是什么

`ExecuteModelState` 是 GPUModelRunner 内部的 per-step execution state。

它不是 SchedulerOutput，也不是 ModelRunnerOutput。

三者关系：

```text
Scheduler
    │
    │ SchedulerOutput
    ▼
Worker

execute_model()
    ↓
ExecuteModelState
    ↓
sample_tokens()

Worker
    │
    │ ModelRunnerOutput
    ▼
Scheduler
```

当前 `ExecuteModelState.compaction_plans` 仍然是：

```python
dict[str, CompactionPlanData]
```

因为从头到尾就是同一个 mapping：

```text
SchedulerOutput.compaction_plans
        ↓
local compaction_plans
        ↓
ExecuteModelState.compaction_plans
```

---

# 13. ExecuteModelState 的生命周期位置

当前大致顺序：

```text
execute_model()
    ↓
extract plan
    ↓
prepare inputs
    ↓
prepare attention
    ↓
model.forward()
    ↓
ExecuteModelState(...)
    ↓
execute_model() return
    ↓
sample_tokens()
```

所以 `ExecuteModelState` 是 forward 后发布的内部 state。

但这里有一个 T2 必须保持的关键约束：

```text
forward
    ↓
compaction
    ↓
Worker state commit
    ↓
CompactionResultData
    ↓
ExecuteModelState publication
```

T2 不能变成：

```text
forward
    ↓
先发布 ExecuteModelState
    ↓
再做 compaction
```

更合理应是：

```python
compaction_plans = self.extract_compaction_plans(...)

...

model_output = self.model(...)

# T2 在这里消费 compaction_plans
run_v2_compaction(compaction_plans)

# 成功以后
self.execute_model_state = ExecuteModelState(...)
```

后续还可以重新评估：如果 plan 已经在 post-forward hook 消费完，`ExecuteModelState` 是否还需要长期保存 `compaction_plans`。

---

# 14. CHG-06：Result 为什么接到 `Scheduler.update_from_output()`

真实回程：

```text
Worker
    ↓
ModelRunnerOutput
    ↓
EngineCore
    ↓
Scheduler.update_from_output()
```

所以 `update_from_output()` 就是 Scheduler 消费当前 Worker execution result 的入口。

不是“下一轮才收到”。

更准确是：

```text
当前 step execution 完成
    ↓
Scheduler update current state
    ↓
然后才进入下一次 schedule
```

---

# 15. `_validate_compaction_results()` 当前做什么

当前只检查：

```text
同一个 request 在当前 result list 中不能出现两次
```

例如：

```python
[
    CompactionResultData(request_id="A", ...),
    CompactionResultData(request_id="A", ...),
]
```

会 fail closed。

当前 T1 不检查：

```text
actual source E
canonical block row
new_num_blocks consistency
effective_kv_len commit
ownership reconcile
```

这些属于后续 T2/T3。

---

# 16. 为什么当前不应该做更多 validation

如果现在就开始检查：

```text
canonical row
E commit
removed blocks
```

就已经进入 M5-T3。

所以 T1 只做 carrier-level structural validation 是正确的边界。

---

# 17. `update_from_output()` 附近为什么有 deferred free

当前已有：

```text
processed_step_seq
deferred_frees
```

大致链：

```text
ModelRunnerOutput returned
    ↓
Scheduler.update_from_output()
    ↓
GPU step completion boundary
    ↓
processed_step_seq++
    ↓
drain deferred frees
```

这是 V1 已经建立的 safe-free lifetime seam。

因此未来 V2 trailing removed blocks 也应优先复用这条 lifetime system，而不是重新造新的 CUDA event/free manager。

---

# 18. M5-T1 最终双向链路

## 去程

```text
External Plan Producer
        ↓
CompactionPlanData
        ↓
Scheduler._set_prepared_compaction_plan()
        ↓
_prepared_compaction_plans
        ↓
Scheduler.schedule()
        ↓
SchedulerOutput.compaction_plans
        ↓
GPUModelRunner.execute_model()
        ↓
extract_compaction_plans()
        ↓
local compaction_plans
```

## 回程

```text
Worker
        ↓
CompactionResultData
        ↓
ModelRunnerOutput.compaction_results
        ↓
EngineCore
        ↓
Scheduler.update_from_output()
        ↓
_validate_compaction_results()
```

当前只是 transport，没有 runtime commit。

---

# 19. M5-T1 重要设计决策

## Decision 1 — V1 / V2 contract 分离

```text
ReclaimTransitionData
= V1-only
```

不复用到 V2。

## Decision 2 — Plan / Result 分离

因为方向、生命周期、E 语义不同。

## Decision 3 — Plan source E 是 POST-FORWARD E

不是 forward-entry E。

## Decision 4 — V2 Core policy-neutral

Scheduler 不计算 importance，只 transport external explicit plan。

## Decision 5 — Plan 使用 request-indexed dict

便于按 request 查找。

## Decision 6 — Result 不携带 allocator-authoritative freed IDs

Worker 只报告 physical shape。

Scheduler 后续自己从 canonical row 推 removed suffix。

## Decision 7 — Worker / Scheduler authority 不变

```text
Worker
= execution truth

Scheduler
= ownership truth
```

## Decision 8 — T1 只建立 transport

不提前进入：

```text
KV mutation
BlockTable mutation
E commit
Scheduler reconcile
free
```

---

# 20. T1 Freeze

当前冻结为：

```text
CompactionPlanData
= APPROVED IMPLEMENTATION BASELINE

CompactionResultData
= APPROVED IMPLEMENTATION BASELINE

SchedulerOutput.compaction_plans
= APPROVED

ModelRunnerOutput.compaction_results
= APPROVED

prepared plan materialization
= APPROVED

Worker plan extraction
= APPROVED

Scheduler result receiver seam
= APPROVED
```

---

# 21. 下一轮仍需解决

留给 M5-T2 / M5-T3：

```text
actual source-E validation
actual source block-row validation
step_seq 是否真正 enforce
keep list → CUDA tensor materialization
all-layer M4 invocation
Worker BlockTable prefix commit
Worker effective_kv_len = K
CompactionResultData real publication
Scheduler canonical prefix reconcile
removed suffix → deferred free
BlockPool reuse
```

---

# 22. M5-T1 Closure Verdict

当前证据：

```text
T1 transport tests
= PASS

M4 regression
= PASS

V1 reclaim regression
= PASS

KV payload mutation
= NO

Worker row mutation
= NO

Worker E mutation
= NO

Scheduler ownership mutation
= NO

BlockPool free
= NO
```

因此：

```text
P1-M5-T1
=
PASS / CLOSED / FROZEN
```

---

# 23. 下一轮进入点

下一轮：

# P1-M5-T2 — Worker Post-Forward Compaction Transaction

真正开始第一次 runtime physical mutation。

核心目标：

```text
SchedulerOutput.compaction_plans
        ↓
Worker source-state validation
        ↓
model.forward()
        ↓
POST_FORWARD
        ↓
all-layer compact_paged_kv_triton_2d()
        ↓
Worker BlockTable prefix commit
        ↓
Worker effective_kv_len = K
        ↓
CompactionResultData
        ↓
ExecuteModelState publication
```

建议继续拆：

```text
M5-T2A
Source Fence + Execution Preparation

M5-T2B
Real Post-Forward Compaction + Worker State Commit
```

原因：

```text
T2 是第一次 destructive KV payload mutation
```

应把 validation 和真实 mutation/commit 分开 review。

---

# 24. 最终一句话记忆

> **M5-T1 用 `CompactionPlanData` 告诉 Worker“基于哪份 post-forward physical source、保留哪些 members”；再用 `CompactionResultData` 告诉 Scheduler“我基于那份 source 做完以后，physical E 和 block count 变成了什么”。这一轮只把这两个消息正确送到两端，不执行任何 physical mutation。**
