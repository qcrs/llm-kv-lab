# P1 V2 Token-Level KV Compaction：从 EngineCore 到 Scheduler Reconciliation 的完整源码链路

> **Project**: Physical KV Cache Reclamation for vLLM 0.26.0  
> **Scope**: P1 V2 Token-Level Compaction，覆盖 M5-T1 → T2A → T2B Harden → T3 Harden 后的最终架构  
> **Baseline / Worktree HEAD in evidence**: `1f04cdca301fa72d20046a791672cf265bf9df09`（相关改动仍为 dirty worktree，未形成 commit）  
> **MVP**: MRV2、single KV group、prefix caching OFF、spec decode OFF、async scheduling OFF、PP=1、CUDA Graph OFF  
> **本文目标**: 不按 T1/T2A/T2B/T3 的研发顺序复述，而是从一次真实 `EngineCore.step()` 出发，把 V2 插入 vLLM 的每个位置、数据结构、函数和状态变化重新串成一条可读源码主链。

---

## 1. 先建立总图：V2 不是一个 Kernel，而是一条跨层 physical-state transition

V2 最容易被误解成：

```text
选一些 Token
→ 删除它们的 KV
→ 把剩余 KV 搬紧
→ 结束
```

但在 vLLM 中，仅搬动 KV payload 不够。V2 实际上必须同时维护四类状态：

```text
1. Scheduler control / ownership state
   - Request.effective_kv_len
   - req_to_blocks
   - BlockPool ownership / release lifecycle

2. Scheduler ↔ Worker transport state
   - CompactionPlanData
   - CompactionResultData

3. Worker execution metadata
   - effective_kv_len
   - BlockTables
   - cache_positions
   - effective_kv_seq_lens

4. GPU KV payload
   - 每层真正的 K/V tensor
```

因此 V2 的本质不是“删除 Token”，而是：

```text
post-forward physical source
        E_source
           │
           ▼
选择 retained physical members
           │
           ▼
GPU payload compact 到 dense prefix [0, K)
           │
           ├── Worker BlockTable → compact prefix
           ├── Worker physical E → K
           │
           ▼
Worker Result
           │
           ▼
Scheduler canonical ownership reconcile
           │
           ├── req_to_blocks → compact prefix
           ├── Scheduler physical E → K
           └── removed suffix → deferred release
```

最终必须重新建立：

```text
Worker physical view
        ==
Scheduler canonical physical view
```

这也是为什么 V2 会跨 `Scheduler → Worker → Scheduler` 一整圈，而不是只新增一个 Triton Kernel。

---

# 2. 原 vLLM 一次 step 的外层骨架

当前同步 MVP 下，最适合建立认知的入口是：

```python
EngineCore.step()
```

从架构上可以先记成：

```text
EngineCore.step()
│
├── 1. Scheduler.schedule()
│      └── SchedulerOutput
│
├── 2. model_executor.execute_model(SchedulerOutput)
│      └── Worker
│           └── GPUModelRunner.execute_model()
│                └── model.forward()
│
├── 3. model_executor.sample_tokens(...)
│      └── Worker
│           └── GPUModelRunner.sample_tokens()
│                └── ModelRunnerOutput
│
└── 4. Scheduler.update_from_output(
          scheduler_output,
          model_runner_output,
       )
```

这里要特别纠正一个容易产生的错误认知：

```text
ModelRunnerOutput → EngineCore → Scheduler.update_from_output()
```

不能把 `EngineCore` 理解成一个“返回结果经过的中转节点”。

更准确的是：

```text
EngineCore.step()
就是这一整个 iteration 的 orchestrator。

它从一开始就在外面：

schedule
→ execute
→ sample
→ update
```

`model_executor` 则是 EngineCore 与 Worker 之间的执行抽象层；Scheduler 并不直接调用 `GPUModelRunner`。

源码导航：

```text
vllm/v1/engine/core.py
└── EngineCore.step()

vllm/v1/core/sched/scheduler.py
├── Scheduler.schedule()
└── Scheduler.update_from_output()

vllm/v1/worker/gpu/model_runner.py
├── GPUModelRunner.execute_model()
└── GPUModelRunner.sample_tokens()
```

V2 没有重写这条主链，而是在原链路的几个明确 seam 上插入自己的 contract、validation、execution 和 reconciliation。

---

# 3. V2 插入后的完整主调用链

先看最终图，再逐层拆。

```text
EngineCore.step()
│
├───────────────────────────────────────────────────────────
│  A. Scheduler / Control Plane
│
├── Scheduler.schedule()
│      │
│      ├── allocate_slots()               [共享 physical allocation]
│      │
│      ├── [V2] 从 _prepared_compaction_plans
│      │         取出 CompactionPlanData
│      │
│      ├── [V2] V1/V2 same-step XOR
│      │
│      └── SchedulerOutput
│             └── compaction_plans
│
├───────────────────────────────────────────────────────────
│  B. Scheduler → Worker transport
│
├── model_executor.execute_model(SchedulerOutput)
│      ↓
│    Worker
│      ↓
│    GPUModelRunner.execute_model()
│      │
│      ├── [V2] extract_compaction_plans()
│      │
│      ├── prepare InputBatch / attention metadata
│      │
│      ├── model.forward()
│      │      └── 当前 q 的 KV 已写入
│      │
│      ├── [V2/T2A] _prepare_v2_compactions()
│      │      └── _PreparedCompaction
│      │
│      ├── [V2/T2B] _execute_v2_compactions()
│      │      │
│      │      ├── compact_paged_kv_triton_2d()
│      │      ├── stage compact 后 BlockTable
│      │      ├── effective_kv_len_override = K
│      │      └── CompactionResultData
│      │
│      └── ExecuteModelState
│             ├── effective_kv_len_override_valid/value
│             └── compaction_results
│
├───────────────────────────────────────────────────────────
│  C. Worker postprocess / output publication
│
├── model_executor.sample_tokens(...)
│      ↓
│    GPUModelRunner.sample_tokens()
│      │
│      ├── consume ExecuteModelState
│      ├── sampling
│      ├── post_update()
│      │      ├── normal: E += delta
│      │      └── V2:     E = K
│      │
│      └── ModelRunnerOutput
│             └── compaction_results
│
├───────────────────────────────────────────────────────────
│  D. Worker → Scheduler reconciliation
│
└── Scheduler.update_from_output(...)
       │
       ├── [V2] _validate_compaction_results()
       │
       ├── [V2] _prepare_compaction_reconciliations()
       │      └── _PreparedCompactionReconciliation
       │
       ├── [V2] _commit_compaction_reconciliations()
       │      │
       │      ├── KVCacheManager.reconcile_compacted_blocks()
       │      │      ↓
       │      ├── KVCacheCoordinator.reconcile_compacted_blocks()
       │      │      ↓
       │      └── SingleTypeKVCacheManager.reconcile_compacted_blocks()
       │
       ├── Scheduler request.effective_kv_len = K
       ├── removed suffix → deferred_frees
       │
       └── V2 request 跳过 normal additive E path
```

接下来从上往下解释每个新增/修改点。

---

# 4. 第一层：Scheduler 如何得到一个 V2 Plan

## 4.1 当前没有实现 retention policy

首先要明确当前项目边界。

V2 当前实现的是：

```text
physical compaction mechanism
```

而不是：

```text
完整 retention / importance policy
```

因此目前不会在 `Scheduler.schedule()` 内写：

```text
观察注意力
→ 计算 importance
→ 自己决定 keep_member_indices
```

而是提供一个显式注入口：

```python
Scheduler._set_prepared_compaction_plan(...)
```

位置：

```text
vllm/v1/core/sched/scheduler.py
```

作用：

```text
external explicit plan producer
        ↓
_set_prepared_compaction_plan()
        ↓
self._prepared_compaction_plans[request_id]
```

它构造的是：

```python
CompactionPlanData(
    request_id=request_id,
    keep_member_indices=keep_member_indices,
    expected_source_effective_kv_len=...,
    expected_source_num_blocks=...,
    step_seq=...,
)
```

为什么这样放？

因为当前 Slice 要冻结的是“执行机制”，不是 retention policy。这样将来可以把不同 producer 接进来，而无需修改 V2 data plane。

---

## 4.2 `CompactionPlanData` 声明在哪里

位置：

```text
vllm/v1/core/sched/output.py
```

最终 T1 记录中的 anchor：

```text
CompactionPlanData   L120-L135
CompactionResultData L137-L151
```

它们不是 `Scheduler` 内部 private class，而是跨 Scheduler / Worker 的 transport contract。

```python
@dataclass(frozen=True)
class CompactionPlanData:
    request_id: str
    keep_member_indices: list[int]
    expected_source_effective_kv_len: int
    expected_source_num_blocks: int
    step_seq: int | None
```

它回答的是：

```text
Scheduler / producer：
“Worker，这一步你应该对哪版 source 做什么？”
```

字段重点：

```text
request_id
= 哪个 request

keep_member_indices
= post-forward physical member sequence 中保留哪些 member

expected_source_effective_kv_len
= Worker 做 compact 前应该看到的 post-forward source E

expected_source_num_blocks
= 该 source 应该占多少 active blocks

step_seq
= 可选 step identity fence
```

这里最重要的是：

```text
Plan.expected_source_effective_kv_len
是 POST-FORWARD source extent

不是 forward-entry E
```

例如：

```text
step start E = 20
q = 1

Plan expected_source_effective_kv_len = 21
```

因为 V2 是先 forward，再 compact。

---

# 5. `Scheduler.schedule()`：Plan 什么时候真正属于当前 step

位置：

```text
vllm/v1/core/sched/scheduler.py
└── Scheduler.schedule()
```

V2 增加/修改的相关状态：

```python
self._prepared_compaction_plans
```

以及当前 `schedule()` 的局部：

```python
compaction_plans: dict[str, CompactionPlanData]
```

二者不是同一个生命周期。

```text
_prepared_compaction_plans
= 已准备，但还没有绑定具体 SchedulerOutput 的 pending Plan

compaction_plans
= 当前这一次 schedule() 真正准备发给 Worker 的 Plan
```

在 request 真正 allocation 成功、进入 current scheduled set 后：

```python
compaction_plan = self._prepared_compaction_plans.pop(request_id, None)

if compaction_plan is not None:
    ...
    compaction_plans[request_id] = compaction_plan
```

然后最终进入：

```python
SchedulerOutput(
    ...,
    compaction_plans=compaction_plans,
)
```

为什么不能 prepared 后直接发 Worker？

因为 request 可能最终没有被当前 step 调度，或者在 same-step allocation / preemption 中又被撤掉。

因此正确语义是：

```text
Plan ready
   ≠
Plan belongs to this executed step
```

必须等 Scheduler admission 成功后才绑定。

---

## 5.1 same-step preemption hardening

T3 HARDEN-01 补了：

```python
compaction_plans.pop(preempted_req_id, None)
```

位置：

```text
Scheduler.schedule() 的 same-step preemption rollback
```

原因：某 request 可能已经暂时加入本轮 scheduled set，也已经绑定了 V2 Plan，但为了给更高优先级 request 腾 block，又在同一个 `schedule()` 中被 preempt。

此时要一起 rollback：

```text
scheduled tokens
new blocks
V1 transition
V2 compaction_plan
spec tokens
encoder inputs
```

否则 stale V2 Plan 会继续进入 `SchedulerOutput`。

---

## 5.2 V1/V2 XOR 在这里检查

同一个 request、同一个 step 当前禁止：

```text
V1 pre-forward reclaim
+
V2 post-forward compaction
```

对应：

```python
if compaction_plan is not None:
    if tentative_reclaim_transition is not None:
        raise ValueError(
            "V1 reclaim and V2 compaction cannot coexist in one step"
        )
```

为什么放在 `schedule()`？

因为这里是当前 step command assembly 的地方；一旦两个 command 同时发出去，Worker 端 source semantics 会变得不明确。

两者的 timeline 本来就不同：

```text
V1:
old E
→ pre-forward reclaim R
→ forward q
→ final R + q

V2:
old E
→ forward q
→ source E
→ post-forward compact
→ final K
```

---

# 6. `allocate_slots()`：V2 没有另造 allocator

在 `Scheduler.schedule()` 内，正常还会调用：

```python
self.kv_cache_manager.allocate_slots(...)
```

这里不是 V2 新函数。

V2 复用了 P1 V1 已建立的 physical-E-aware allocation：

```python
allocation_base = (
    request.effective_kv_len
    if request.effective_kv_len is not None
    else total_computed_tokens
)
```

因此它现在已经是 V1/V2 共享基础设施：

```text
如果从未发生 logical / physical divergence：
allocation_base = logical progress

如果之前 V1 reclaim 后 E < L：
allocation_base = E

如果之前 V2 compact 后 E = K < L：
allocation_base = K
```

为什么当前 step 的 V2 不能提前按未来 K allocation？

因为当前 step V2 顺序是：

```text
allocate 当前 forward source capacity
→ forward 把 q 写入 KV
→ 得到 source_E = old_E + q
→ 再 compact 到 K
```

所以当前 `allocate_slots()` 必须先保证 `old_E + q` 能合法存在。

V2 新 K 的 allocation benefit 是在下一 step 才体现：

```text
Step N:
E_old → forward → source_E → compact → K
Scheduler commit E=K

Step N+1:
allocate_slots()
→ allocation_base = K
```

---

# 7. `SchedulerOutput.compaction_plans`：跨到 Worker

位置：

```text
vllm/v1/core/sched/output.py
└── SchedulerOutput
```

新增：

```python
compaction_plans: dict[str, CompactionPlanData] = field(default_factory=dict)
```

它是 Scheduler→Worker carrier。

因此去程是：

```text
CompactionPlanData
        ↓
SchedulerOutput.compaction_plans
        ↓
model_executor.execute_model()
        ↓
Worker
        ↓
GPUModelRunner.execute_model()
```

为什么用现有 `SchedulerOutput`？

因为这本来就是 vLLM Scheduler→Worker 的 per-step transport boundary。没有必要为了 V2 新建 RPC/channel。

---

# 8. Worker 入口：`extract_compaction_plans()`

文件：

```text
vllm/v1/worker/gpu/model_runner.py
```

新增：

```python
GPUModelRunner.extract_compaction_plans(
    scheduler_output: SchedulerOutput,
) -> dict[str, CompactionPlanData]
```

放置位置：

```text
GPUModelRunner.execute_model() 入口阶段
```

作用仅是 structural extraction：

```text
SchedulerOutput.compaction_plans
        ↓
检查 mapping 类型
检查 value 是 CompactionPlanData
检查 dict key == plan.request_id
        ↓
返回 plans
```

它不会：

```text
调用 M4
修改 KV
修改 BlockTable
修改 E
```

为什么需要单独 helper？

因为 transport contract validation 与 destructive physical transition 应该分开；这里仅建立 Worker 对 Plan carrier 的消费入口。

---

# 9. Worker 正常 input / forward：V2 此时还不执行

`GPUModelRunner.execute_model()` 随后仍然走原正常路径：

```text
update/add request state
        ↓
prepare InputBatch
        ↓
positions
cache_positions
effective_kv_seq_lens
        ↓
attention metadata
        ↓
model.forward()
```

这里必须理解三个坐标：

```text
num_computed_tokens / positions
= logical model progress

effective_kv_len / cache_positions
= physical storage frontier

effective_kv_seq_lens
= 当前 forward attention 实际看到的 physical KV extent
```

假设之前 V2 已经压缩过：

```text
logical L = 100
physical E = 20
q = 1
```

那么当前 step：

```text
logical position = 100
physical cache_position = 20
```

forward 后：

```text
post-forward physical source_E = 21
```

注意：这解释了为什么 V2 也必须使用 `effective_kv_len`。V2 不是 compact 完以后才需要 E；下一轮从哪里写 KV、Attention 读多少 KV、Scheduler 分多少 physical capacity，都依赖这个统一 physical frontier。

---

# 10. 为什么 V2 插在 `model.forward()` 之后

最终 seam：

```text
GPUModelRunner.execute_model()

model.forward()
        ↓
[V2 T2A/T2B]
        ↓
ExecuteModelState publication
```

原因非常直接：

```text
V2 要 compact 的 source
= 当前 forward 已经写入 q 之后的完整 physical source
```

例如：

```text
E_before = 20
q = 1

forward 前 source = 20
forward 后 source = 21

V2 keep_member_indices
索引的是这 21 个 physical members
```

如果放在 forward 前，就变成 V1 的语义，而不是 V2。

---

# 11. T2A：`_PreparedCompaction` + `_prepare_v2_compactions()`

## 11.1 `_PreparedCompaction`

声明位置：

```text
vllm/v1/worker/gpu/model_runner.py
```

最终 Harden 记录 anchor：

```text
_PreparedCompaction L129-L140
```

它是 Worker private immutable descriptor，不跨 Scheduler/Worker boundary。

可以把它理解为：

```text
CompactionPlanData
= Scheduler 发来的“命令”

_PreparedCompaction
= Worker 结合真实 runtime source 审核后的“可执行描述”
```

它保存的核心信息包括：

```text
request identity
req_state_idx
batch_idx
source effective E
source block row
block_size
keep set
K = len(keep)
new_num_blocks
step_seq
```

最终 hardened 版本中，它只活在 `execute_model()` 内，不再穿过 `ExecuteModelState`。

---

## 11.2 `_prepare_v2_compactions()`

位置：

```text
vllm/v1/worker/gpu/model_runner.py
GPUModelRunner._prepare_v2_compactions()

final harden anchor: L1812-L1978
```

插入：

```text
model.forward()
        ↓
_prepare_v2_compactions()
        ↓
_execute_v2_compactions()
```

它不修改 runtime state，只做 READ + VALIDATE + DERIVE。

读取的真实 source 包括：

```text
InputBatch.req_ids[batch_idx]
InputBatch.idx_mapping_np[batch_idx]
InputBatch.effective_kv_seq_lens[batch_idx]
BlockTables.num_blocks.np[0, req_state_idx]
BlockTables active prefix
self.kv_caches
```

为什么 `source_E` 从 `effective_kv_seq_lens` 读，而不是 persistent `req_states.effective_kv_len`？

因为当前 q 已经 forward：

```text
persistent E 仍代表 step-start / committed base

post-forward effective_kv_seq_lens
才代表真正要被 compact 的 source extent
```

它验证：

```text
Plan request identity
request 是否在 current batch
batch_idx ↔ req_state_idx
Plan source_E == Worker actual source_E
Plan source_num_blocks == active row shape
keep 非空、整数、非负、严格递增、in-range
K = len(keep)
new_num_blocks = ceil(K / block_size)
所有 KV layer shape / block address / device 合法
```

全部通过后才返回：

```python
_PreparedCompaction
```

为什么要这一层，而不直接让 `_execute_v2_compactions()` 解读 Plan？

因为 destructive mutation 之前先把所有可确定错误 fail closed；同时避免 batch index / persistent index / stale source 混淆。

---

# 12. T2B：`_execute_v2_compactions()` 真正改变 Worker physical state

位置：

```text
vllm/v1/worker/gpu/model_runner.py
GPUModelRunner._execute_v2_compactions()

final harden anchor: L1981-L2042
```

输入：

```text
_PreparedCompaction
```

输出三类东西：

```text
effective_kv_len_override_valid

effective_kv_len_override

list[CompactionResultData]
```

它最终被硬化成两个 phase。

---

## 12.1 Phase 1：所有 request × 所有 layer 先做 payload compaction

核心调用：

```python
compact_paged_kv_triton_2d(...)
```

位置：

```text
vllm/v1/worker/gpu/kv_compaction.py
```

M4 primitive 的职责只有 physical KV tensor：

```text
paged source
   ↓
gather retained members
   ↓
independent scratch
   ↓
writeback / scatter
   ↓
dense paged prefix [0, K)
```

例如：

```text
block_size = 4
source_E = 10
row = [B7, B2, B11]

source members:
[m0 m1 m2 m3 | m4 m5 m6 m7 | m8 m9]

keep = [0, 2, 5, 8, 9]
K = 5

compact 后：
B7 = [m0, m2, m5, m8]
B2 = [m9, 0, 0, 0]
B11 = 不再属于最终 active prefix
```

为什么要 scratch，而不能 arbitrary paged→paged 原地搬？

因为 source/destination 可能重叠；独立 scratch 消除 overwrite-before-read hazard。

---

## 12.2 T2B Harden：先完成所有 payload，再提交 metadata

旧风险：

```text
request A payload成功
→ A metadata 已 stage

request B 某 layer payload失败
→ 整个 step fail

结果 A metadata 已变化，B 没完成
```

最终 hardening 改成：

```text
Phase 1:
A layer0..N payload
B layer0..N payload
C layer0..N payload
全部成功

        ↓

Phase 2:
A/B/C metadata + result publication
```

注意：当前 MVP 仍然没有 payload rollback。若更后面的 layer/runtime operation fail，前面已写的 payload 可能已经改变；但至少 metadata/result 不会提前发布成功。

---

## 12.3 Phase 2：Worker BlockTable 更新

所有 payload 成功后：

```python
self.block_tables.append_block_ids(
    req_index=...,
    new_block_ids=...,
    overwrite=True,
)
```

语义：

```text
Worker active row
[B7, B2, B11]
        ↓
[B7, B2]
```

这里 stage 的是 Worker execution-addressing next-state，不是 Scheduler canonical ownership。

这一点必须区分：

```text
Worker BlockTable
≠
Scheduler req_to_blocks authority
```

因此 T2B 做完后仍需要 T3。

---

# 13. 为什么 T2B 还要产生 `effective_kv_len_override = K`

假设：

```text
E_before = 20
q = 1
```

forward 后：

```text
source_E = 21
```

V2 compact 后：

```text
K = 8
```

最终 physical E 必须是：

```text
E = 8
```

而不能让正常 post-update 再：

```text
E += q
```

否则会变成：

```text
K + q = 9
```

这就是 double fold-in。

所以 `_execute_v2_compactions()` 产生：

```text
effective_kv_len_override_valid

effective_kv_len_override = K
```

它回答的是：

```text
“这个 request 当前 step 的 physical E
不是 normal additive update，
而是已经被 post-forward transition 重置为绝对值 K。”
```

---

# 14. `CompactionResultData`：Worker 的执行回执

声明位置：

```text
vllm/v1/core/sched/output.py
```

与 `CompactionPlanData` 放在一起，因为二者都是跨组件 transport contract。

核心字段：

```python
@dataclass(frozen=True)
class CompactionResultData:
    request_id: str
    expected_source_effective_kv_len: int
    expected_source_num_blocks: int
    new_effective_kv_len: int
    new_num_blocks: int
    step_seq: int | None
```

Plan 与 Result 的区别：

```text
CompactionPlanData
= command / intent
= “你应该怎么做”

核心独有字段：
keep_member_indices
```

```text
CompactionResultData
= completion receipt
= “我实际上已经做完，结果是什么”

核心结果字段：
new_effective_kv_len
new_num_blocks
```

Result 为什么还重复 source fence？

因为 Scheduler回来后必须验证：

```text
Worker 这张 receipt
确实对应我发出的那一版 source
```

例如：

```text
Plan:
source_E = 21
source_blocks = 3
step = 17

Result:
source_E = 21
source_blocks = 3
step = 17
new_E = 8
new_blocks = 1
```

Scheduler不能只看到：

```text
new_E = 8
```

就接受；如果 Result其实对应 stale source_E=37，那么 K=8 即使数值合法，也不能作用到当前 canonical state。

---

# 15. `ExecuteModelState`：为什么 Result 不能直接回 Scheduler

这里是整个 V2 transport 中最容易误解的一层。

`_execute_v2_compactions()` 发生在：

```text
GPUModelRunner.execute_model()
```

但是当前 generate path 真正构造 `ModelRunnerOutput` 的位置在后面的：

```text
GPUModelRunner.sample_tokens()
```

因此时序是：

```text
execute_model()
│
├── model.forward()
├── V2 compact
├── 产生 CompactionResultData
│
└── return None

        ↓

sample_tokens()
│
└── 构造 ModelRunnerOutput
```

所以 `CompactionResultData` 产生时，`ModelRunnerOutput` 还不存在。

vLLM 原本已经有一个 execute→sample 的内部 per-step bridge：

```python
ExecuteModelState
```

V2 没有新造特殊 channel，而是扩展它：

```text
ExecuteModelState
├── ... upstream state
├── effective_kv_len_override_valid
├── effective_kv_len_override
└── compaction_results
```

最终 Harden 后：

```text
_PreparedCompaction
不会进入 ExecuteModelState

它在 _execute_v2_compactions() 消费后生命周期结束。
```

真正穿过 ExecuteModelState 的只有：

```text
1. E override
2. CompactionResultData list
```

---

# 16. `sample_tokens()`：消费 state，并发布最终 Worker output

位置：

```text
vllm/v1/worker/gpu/model_runner.py
└── GPUModelRunner.sample_tokens()
```

一开始从：

```python
self.execute_model_state
```

取出：

```text
input_batch
hidden_states
attn metadata
E override
compaction_results
...
```

然后 `self.execute_model_state = None`，说明这是本 step 的一次性交接状态。

V2 在 sample 阶段不再执行 payload compact。

它主要做两件事情：

```text
A. E override 参与 post-update

B. compaction_results 被塞进最终 ModelRunnerOutput
```

---

# 17. `input_batch.py::post_update()`：Worker physical E 真正 fold-in

文件：

```text
vllm/v1/worker/gpu/input_batch.py
```

相关：

```text
_post_update_kernel()
post_update()
```

T2B 对这里的核心修改是增加：

```text
effective_kv_len_override_valid

effective_kv_len_override
```

最终状态机：

```text
computed_delta = query_len - num_rejected

logical:
num_computed_tokens += computed_delta

physical:
if V2 override valid:
    effective_kv_len = K
else:
    effective_kv_len += computed_delta
```

所以：

```text
Normal:
E_t → E_t + q

V1:
前面已经 pre-forward reclaim 到 R
最终 → R + q

V2:
forward source E_t + q
→ compact 到 K
→ absolute E = K
```

这是 V2 与 normal/V1 最大的状态语义差异之一。

---

# 18. `ModelRunnerOutput.compaction_results`：真正跨回 Scheduler

文件：

```text
vllm/v1/outputs.py
└── ModelRunnerOutput
```

新增：

```python
compaction_results: list[CompactionResultData] = field(default_factory=list)
```

这里不存在一种新的 `Results` 类型转换。

只是：

```text
单 request:
CompactionResultData(A)

多个 request:
[
  Result(A),
  Result(C),
]

        ↓
ModelRunnerOutput.compaction_results
```

为什么 `Plan` 用 dict、`Result` 用 list？

```text
Plan:
SchedulerOutput 本身大量按 request-id 组织 per-request payload
→ dict[str, CompactionPlanData]

Result:
ModelRunnerOutput 是整个 step 的 serialized result carrier
→ list[CompactionResultData]
→ Scheduler 再按 request_id admission
```

真正跨 Scheduler / Worker 边界的 V2 contract只有：

```text
CompactionPlanData       Scheduler → Worker
CompactionResultData     Worker → Scheduler
```

其他 `_Prepared...` 都是组件内部事务对象。

---

# 19. 回到 `EngineCore.step()`：同一 step 内交回 Scheduler

Worker 完成 `sample_tokens()` 后得到：

```text
ModelRunnerOutput
```

然后仍然在同一个外层 `EngineCore.step()` 中：

```python
self.scheduler.update_from_output(
    scheduler_output,
    model_runner_output,
)
```

这里有两份关键输入：

```text
scheduler_output
= 我这一轮 Scheduler 当初发了什么

model_runner_output
= Worker 对这一轮实际返回了什么
```

因此 V2 Scheduler reconciliation 正好可以做：

```text
Plan
⇅
Result
+
Scheduler current canonical state
```

三方对账。

---

# 20. T3 第一层：`_validate_compaction_results()`

文件：

```text
vllm/v1/core/sched/scheduler.py
```

这个函数最早在 T1 transport 阶段就建立了。

职责很轻：

```text
ModelRunnerOutput.compaction_results
        ↓
做 carrier-level structural validation
例如 duplicate request ID
```

它不是 T3 的完整 canonical validation。

可以理解成：

```text
_validate_compaction_results()
= “这包 Result 至少长得像一个合法 carrier 吗？”
```

后面的 `_prepare_compaction_reconciliations()` 才是 Scheduler state-level admission。

---

# 21. `_PreparedCompactionReconciliation`：Scheduler 私有 commit descriptor

声明位置：

```text
vllm/v1/core/sched/scheduler.py
在 Scheduler class 前面的 private frozen dataclass
```

最终记录：

```python
@dataclass(frozen=True)
class _PreparedCompactionReconciliation:
    request_id: str
    expected_source_num_blocks: int
    new_effective_kv_len: int
    new_num_blocks: int
    step_seq: int | None
```

它不是 Result 的“无意义包装”。

两者权限不同：

```text
CompactionResultData
= Worker 外部 receipt
= “Worker 说它做完了”

_PreparedCompactionReconciliation
= Scheduler 内部 approved commit intent
= “Scheduler 已经结合 canonical state 审核通过，可以提交”
```

从架构上是一次权限升级：

```text
untrusted / external receipt
        ↓ Scheduler validation
trusted internal commit descriptor
```

---

# 22. `_prepare_compaction_reconciliations()`：所有 Result 先验证，不修改状态

位置：

```text
vllm/v1/core/sched/scheduler.py
final T3 anchor around L1716
```

接口：

```python
_prepare_compaction_reconciliations(
    scheduler_output,
    results,
)
```

返回：

```text
dict[str, _PreparedCompactionReconciliation]
stale request set
```

这一步是：

```text
READ + VALIDATE
```

不是：

```text
WRITE / COMMIT
```

它主要验证四组东西。

### 22.1 Plan ↔ Result identity

```text
Result 必须有 matching Plan
request_id match
step_seq match
source fence match
```

T3 HARDEN-01 又补上反方向：

```text
active current-step Plan
必须有 Result
```

于是 current active request 上接近：

```text
Plan ⇔ Result
```

防止：

```text
Scheduler发了 V2 Plan
但 Worker没 Result
→ request 后面误走 normal additive E path
```

### 22.2 Lifecycle

```text
request 不存在 / 已 finished
→ stale receipt
→ 不再 canonical apply
```

### 22.3 重建当前 Worker 应该 compact 的 source_E

Scheduler schedule 之前的 physical base是：

```python
source_base = (
    request.effective_kv_len
    if request.effective_kv_len is not None
    else request.num_computed_tokens - q
)
```

然后：

```python
source_e = source_base + q
```

这里要牢记：

```text
effective_kv_len is None
不代表 physical E=0
```

而是表示尚未显式 materialize logical/physical divergence，此时 physical base隐式等于 pre-step logical progress。

为什么要 `+q`？

因为 Result对应的是：

```text
post-forward
pre-compaction
```

source。

### 22.4 Geometry / canonical row fence

验证：

```text
new_E > 0
new_E <= source_E
source_num_blocks == ceil(source_E / block_size)    [当前 P1 MVP invariant]
new_num_blocks == ceil(new_E / block_size)
0 < new_num_blocks <= source_num_blocks
Scheduler canonical row 当前长度 == expected source blocks
```

全部通过才构造：

```python
_PreparedCompactionReconciliation(...)
```

---

# 23. 为什么 prepare / commit 必须分开

不能简单写：

```python
for result in results:
    validate(result)
    commit(result)
```

因为假设：

```text
Result A valid
Result B invalid
```

那么：

```text
A validate
A commit

B validate
FAIL
```

Scheduler 会变成：

```text
A 已提交
B 未提交
```

形成 half-commit。

所以现在结构是：

```text
Phase 1:
Result A → Prepared A
Result B → Prepared B
Result C → Prepared C

所有 validation 全 PASS
        ↓
Phase 2:
统一 commit A/B/C
```

这就是 `_PreparedCompactionReconciliation` 存在的核心原因。

---

# 24. `_commit_compaction_reconciliations()`：Scheduler canonical commit

位置：

```text
vllm/v1/core/sched/scheduler.py
final T3 anchor around L1775
```

核心：

```python
for item in prepared.values():
    removed = self.kv_cache_manager.reconcile_compacted_blocks(
        item.request_id,
        item.expected_source_num_blocks,
        item.new_num_blocks,
    )

    request.effective_kv_len = item.new_effective_kv_len

    if removed:
        self.deferred_frees.append(
            (self.sched_step_seq + 1, removed)
        )
```

这里第一次真正修改 Scheduler canonical V2 state。

做三件事：

```text
1. canonical ownership row truncate
2. Scheduler persistent physical E = K
3. removed suffix → deferred release lifecycle
```

---

# 25. 为什么 `reconcile_compacted_blocks()` 分三层

调用：

```text
Scheduler
  ↓
KVCacheManager.reconcile_compacted_blocks()
  ↓
KVCacheCoordinator.reconcile_compacted_blocks()
  ↓
SingleTypeKVCacheManager.reconcile_compacted_blocks()
```

对应文件：

```text
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/single_type_kv_cache_manager.py
```

职责不同。

### `KVCacheManager`

```text
Scheduler-facing facade
```

Scheduler 不应该直接摸 coordinator / single-type ownership internals。

### `KVCacheCoordinator`

当前 MVP：

```text
检查 exactly one KV cache group
→ forwarding
```

因为当前 `CompactionPlanData/ResultData` 只有一个 source/new block count，没有 multi-group per-group semantics。

### `SingleTypeKVCacheManager`

这里才真正持有：

```python
req_to_blocks
```

所以真正 ownership mutation发生这里。

例如：

```text
current canonical row:
[B7, B2, B11]

new_num_blocks = 2

retained:
[B7, B2]

removed:
[B11]
```

最终：

```python
self.req_to_blocks[request_id] = retained
return removed
```

注意：

```text
这里 detach ownership
但不立即 free
```

这保持了：

```text
ownership detach
≠
allocator reuse
```

的生命周期分离。

---

# 26. Scheduler `effective_kv_len = K` 为什么仍然必要

Worker已经：

```text
Worker E = K
Worker BlockTable = compact prefix
```

但 Scheduler仍是 allocator / canonical ownership authority。

如果 Scheduler不提交：

```python
request.effective_kv_len = K
```

下一 step `allocate_slots()` 仍然可能从错误 physical base进行 accounting。

所以 T3结束后必须建立：

```text
Worker physical E    = K
Scheduler physical E = K
```

然后下一 step：

```text
Scheduler.allocate_slots()
→ allocation_base = K

Worker input prepare
→ cache_position starts from K

Attention
→ physical visibility继续从新的 E 构建
```

这才闭环。

---

# 27. `update_from_output()` 后面的 V2 `pass` 是什么含义

T3 canonical commit发生在 `update_from_output()` 很靠前的位置：

```text
_validate_compaction_results()
_prepare_compaction_reconciliations()
_commit_compaction_reconciliations()
```

此时 Scheduler E 已经：

```text
E = K
```

所以后面原来的 per-request bookkeeping再处理 physical E时：

```python
if req_id in prepared_compactions:
    pass
else:
    ...
```

V2必须什么都不做。

否则会：

```text
E = K
然后 normal += q
→ K + q
```

这和 Worker post-update的 override理由完全一致：

```text
Worker side 防一次 double fold-in
Scheduler side 也防一次 double fold-in
```

因为 Worker和Scheduler各自维护一份 physical state。

---

# 28. removed block 与 `sched_step_seq + 1`

当前 T3：

```python
self.deferred_frees.append(
    (self.sched_step_seq + 1, removed)
)
```

要准确理解：

```text
T3 canonical reconcile
= request 不再拥有 removed blocks

T4 / deferred-free lifecycle
= removed blocks什么时候重新进入 allocator pool 可复用
```

当前 `+1` 的直接作用是：

```text
本次 update_from_output() 内新 detach 的 block
不会在同一次 drain pass 里立刻变成可复用 block
```

不要过度解释成：

```text
当前 GPU 一定还在读这些 block
```

当前同步 MVP 在 `update_from_output()` 的相关 fence处已有“GPU writes completed”的语义。这里的 `+1` 更准确地说是一个保守的 ownership-detach → allocator-reuse boundary；其必要性和最优 timeliness 属于后续 T4 生命周期审计。

---

# 29. 四个数据类：不要再混

整个 V2 最重要的数据对象只有四类语义。

| 对象 | 声明位置 | 谁创建 | 是否跨组件 | 回答的问题 |
|---|---|---|---|---|
| `CompactionPlanData` | `core/sched/output.py` | Scheduler / explicit producer | 是，S→W | **我要你做什么？** |
| `_PreparedCompaction` | `worker/gpu/model_runner.py` | Worker | 否 | **这个 Plan 对当前真实 source 能不能执行？具体参数是什么？** |
| `CompactionResultData` | `core/sched/output.py` | Worker | 是，W→S | **我实际做完后结果是什么？** |
| `_PreparedCompactionReconciliation` | `core/sched/scheduler.py` | Scheduler | 否 | **这个 Result 对当前 canonical state 能不能提交？** |

可以直接背：

```text
Intent
  CompactionPlanData
        ↓
Worker Validation
  _PreparedCompaction
        ↓
Execution Receipt
  CompactionResultData
        ↓
Scheduler Validation
  _PreparedCompactionReconciliation
        ↓
Canonical Commit
```

其中真正跨 Scheduler/Worker boundary 的只有：

```text
CompactionPlanData
CompactionResultData
```

两个带 `_Prepared...` 的对象都是组件内部为了实现：

```text
validate first
mutate later
```

而存在的事务 descriptor。

---

# 30. 一次完整 V2 step：带数字走到底

假设：

```text
logical L = 100
Scheduler committed physical E = 20
q = 1
block_size = 8
canonical row = [B7, B2, B11]
```

外部 producer 决定：

```text
post-forward source_E = 21
keep_member_indices = [0,2,4,7,10,15,19,20]
K = 8
```

## A. Plan 准备

```python
_set_prepared_compaction_plan(...)
```

得到：

```text
CompactionPlanData
request=A
keep=[...]
source_E=21
source_blocks=3
step=17
```

进入：

```text
_prepared_compaction_plans[A]
```

---

## B. Scheduler.schedule()

先正常 allocation：

```text
allocation_base = E = 20
target = 20 + q = 21
```

request admission成功后：

```text
_prepared_compaction_plans[A]
        ↓ pop
compaction_plans[A]
        ↓
SchedulerOutput.compaction_plans
```

---

## C. Worker execute_model()

```text
extract_compaction_plans()
        ↓
prepare input
```

坐标：

```text
logical position = 100
physical cache_position = 20
```

forward：

```text
E 20 + q1
→ post-forward source_E = 21
```

---

## D. Worker T2A

```python
_prepare_v2_compactions()
```

验证：

```text
Plan source_E 21 == actual 21
Plan source blocks 3 == actual 3
keep合法
layer structure合法
```

得到：

```text
_PreparedCompaction
source_E=21
block_ids=(B7,B2,B11)
K=8
new_num_blocks=1
```

---

## E. Worker T2B

```python
_execute_v2_compactions()
```

所有 layer：

```text
21 physical members
→ M4 compact
→ retain 8
```

Worker metadata：

```text
BlockTable:
[B7,B2,B11]
→ [B7]

physical E override:
K = 8
```

生成：

```text
CompactionResultData
source_E=21
source_blocks=3
new_E=8
new_blocks=1
step=17
```

---

## F. ExecuteModelState → sample_tokens()

```text
ExecuteModelState
├── effective_kv_len_override = 8
└── compaction_results=[Result(A)]
```

sample/post-update：

```text
logical num_computed_tokens
100 → 101

physical E
不是 20→21
而是 absolute → 8
```

最终：

```text
ModelRunnerOutput.compaction_results
= [Result(A)]
```

---

## G. Scheduler.update_from_output()

Scheduler重建它认为的 source：

```text
source_base = committed E_before = 20
source_e = source_base + q = 21
```

然后：

```text
Plan source=21
Result source=21
Scheduler reconstructed source=21
canonical row blocks=3

全部 match
```

构造：

```text
_PreparedCompactionReconciliation
request=A
expected_source_blocks=3
new_E=8
new_blocks=1
step=17
```

---

## H. Canonical commit

```text
req_to_blocks:
[B7,B2,B11]
→ [B7]

Scheduler E:
20 → 8

removed:
[B2,B11]
→ deferred_frees
```

至此：

```text
logical progress = 101
Worker physical E = 8
Scheduler physical E = 8
Worker active row = [B7]
Scheduler canonical row = [B7]
```

下一 step：

```text
allocate_slots(): base = 8
Worker cache_position starts from 8
```

闭环完成。

---

# 31. V1 / Normal / V2 放到同一条时间轴

| Path | Forward 前 | Forward | Forward 后 | 最终 E |
|---|---|---|---|---|
| Normal | E | `E + q` | 无 transition | `E + q` |
| V1 | `E → R` pre-forward reclaim | `R + q` | 无 post-forward compact | `R + q` |
| V2 | E | `E + q = source_E` | `source_E → K` compact | `K` |

因此：

```text
V1 command 的核心对象
= pre-forward block-level transition

V2 command 的核心对象
= post-forward member-level Plan
```

这也是为什么 V1 在 Scheduler 要 materialize retained physical block IDs，而 V2 不在 schedule 阶段 materialize具体 token→slot搬运；V2 要等 forward 后结合 Worker真实 source进行解析。

---

# 32. 最终源码导航表

下面这张表适合作为后续直接读代码的索引。

| 顺序 | 层 | 文件 | Symbol | ADD / EXTEND | 作用 |
|---:|---|---|---|---|---|
| 1 | Transport contract | `vllm/v1/core/sched/output.py` | `CompactionPlanData` | ADD | Scheduler→Worker V2 command |
| 2 | Transport contract | `vllm/v1/core/sched/output.py` | `CompactionResultData` | ADD | Worker→Scheduler V2 receipt |
| 3 | Scheduler state | `vllm/v1/core/sched/scheduler.py` | `_prepared_compaction_plans` | ADD | pending、尚未绑定 step 的 Plan |
| 4 | Scheduler ingress | `scheduler.py` | `_set_prepared_compaction_plan()` | ADD | explicit Plan 注入口 |
| 5 | Scheduler step | `scheduler.py` | `schedule()` | EXTEND | request admission 后绑定 Plan；V1/V2 XOR；preemption rollback |
| 6 | Transport | `core/sched/output.py` | `SchedulerOutput.compaction_plans` | EXTEND | Plan 的 Scheduler→Worker carrier |
| 7 | Worker ingress | `worker/gpu/model_runner.py` | `extract_compaction_plans()` | ADD | 读取/结构校验 Plan carrier |
| 8 | Worker private | `model_runner.py` | `_PreparedCompaction` | ADD | source-fence 后的可执行描述 |
| 9 | Worker post-forward | `model_runner.py` | `_prepare_v2_compactions()` | ADD | 读取真实 post-forward source并 validate |
| 10 | Worker post-forward | `model_runner.py` | `_execute_v2_compactions()` | ADD | payload + Worker metadata + Result |
| 11 | GPU data plane | `worker/gpu/kv_compaction.py` | `compact_paged_kv_triton_2d()` | M4 ADD | 真正搬 K/V payload |
| 12 | Worker addressing | `model_runner.py` / BlockTables | `append_block_ids(... overwrite=True)` | USE/EXTEND seam | stage compact 后 active row |
| 13 | Worker bridge | `model_runner.py` | `ExecuteModelState` | EXTEND | execute→sample 携带 E override + Results |
| 14 | Worker state | `worker/gpu/input_batch.py` | `post_update()` / `_post_update_kernel()` | EXTEND | normal `E+=delta`; V2 `E=K` |
| 15 | Return transport | `vllm/v1/outputs.py` | `ModelRunnerOutput.compaction_results` | EXTEND | Results 的 Worker→Scheduler carrier |
| 16 | Scheduler receipt | `scheduler.py` | `_validate_compaction_results()` | ADD | carrier-level basic validation |
| 17 | Scheduler private | `scheduler.py` | `_PreparedCompactionReconciliation` | ADD | validated canonical commit descriptor |
| 18 | Scheduler validation | `scheduler.py` | `_prepare_compaction_reconciliations()` | ADD | Plan/Result/current-state 全量 admission |
| 19 | Scheduler commit | `scheduler.py` | `_commit_compaction_reconciliations()` | ADD | canonical ownership + E + release handoff |
| 20 | Ownership facade | `core/kv_cache_manager.py` | `reconcile_compacted_blocks()` | ADD | Scheduler-facing forwarding API |
| 21 | Ownership topology | `core/kv_cache_coordinator.py` | `reconcile_compacted_blocks()` | ADD | single-group guard + forwarding |
| 22 | Ownership authority | `core/single_type_kv_cache_manager.py` | `reconcile_compacted_blocks()` | ADD | 真正 truncate `req_to_blocks`，返回 suffix |
| 23 | Lifecycle | `scheduler.py` | `deferred_frees` | REUSE / EXTEND flow | ownership detach 后等待 allocator release |

---

# 33. 当前最终版本的关键 line anchors

> 行号来自当前 implementation evidence；工作树尚未 commit，后续继续修改时 line number 可能漂移，因此优先以 symbol 搜索为准。

```text
vllm/v1/core/sched/output.py
  CompactionPlanData        ~ L120-L135
  CompactionResultData      ~ L137-L151

vllm/v1/worker/gpu/model_runner.py
  _PreparedCompaction       ~ L129-L140
  _prepare_v2_compactions   ~ L1812-L1978
  _execute_v2_compactions   ~ L1981-L2042
  execute_model T2B seam    ~ L2265-L2289
  ExecuteModelState         ~ L2550-L2563

vllm/v1/core/sched/scheduler.py
  _PreparedCompactionReconciliation ~ L85-L90
  schedule V1/V2 XOR                ~ L660
  _prepare_compaction_reconciliations ~ L1716
  _commit_compaction_reconciliations  ~ L1775
  update_from_output                  ~ L1791

vllm/v1/core/kv_cache_coordinator.py
  reconcile_compacted_blocks ~ L317

vllm/v1/core/kv_cache_manager.py
  reconcile_compacted_blocks ~ L543

vllm/v1/core/single_type_kv_cache_manager.py
  reconcile_compacted_blocks ~ L573
```

建议实际读代码时使用：

```bash
rg -n "CompactionPlanData|CompactionResultData|_PreparedCompaction|_prepare_v2_compactions|_execute_v2_compactions" vllm/v1

rg -n "_PreparedCompactionReconciliation|_prepare_compaction_reconciliations|_commit_compaction_reconciliations|reconcile_compacted_blocks" vllm/v1
```

不要依赖固定行号。

---

# 34. 状态与 Authority：最后再统一一次

整个设计最重要的不是函数数量，而是谁有权改什么。

| State | Authority / Owner | V2 在哪里改 |
|---|---|---|
| logical `num_computed_tokens` | normal request/model progress | normal post-update；V2 不把 logical history 压成 K |
| Scheduler `request.effective_kv_len` | Scheduler committed physical frontier | T3 `_commit_compaction_reconciliations()` → `E=K` |
| Worker persistent `effective_kv_len` | Worker execution physical frontier | sample/post_update → override `E=K` |
| Worker `effective_kv_seq_lens` | current-forward physical visibility | input preparation；T2A用它读取 post-forward source E |
| Worker BlockTable | execution addressing view | T2B stage compact prefix |
| Scheduler `req_to_blocks` | canonical allocator ownership | T3 `SingleTypeKVCacheManager.reconcile_compacted_blocks()` |
| GPU K/V tensors | Worker/GPU data plane | M4 `compact_paged_kv_triton_2d()` |
| removed block reuse | Scheduler allocator lifecycle | T3 detach → `deferred_frees`；更完整 closure 属 T4 |

一句话：

```text
Worker Result 不是 ownership authority。

Worker 只报告：
“我的 physical payload / view 已经 compact 到 K。”

Scheduler 才决定：
“canonical ownership 可以从 old row 提交成 compact prefix，并把 suffix 交给 release lifecycle。”
```

---

# 35. 为什么这个架构看起来“层很多”，但其实边界很干净

整个 V2 可以压缩成六个动词：

```text
COMMAND
  CompactionPlanData
        ↓
VALIDATE
  _PreparedCompaction
        ↓
EXECUTE
  M4 + Worker metadata
        ↓
RECEIPT
  CompactionResultData
        ↓
VALIDATE
  _PreparedCompactionReconciliation
        ↓
COMMIT
  Scheduler canonical ownership / E
```

也就是：

```text
命令
→ Worker验证
→ 执行
→ 回执
→ Scheduler验证
→ 提交
```

从这个角度看，并不是“同一份数据被包装很多次”，而是每经过一个 authority boundary，语义发生一次升级：

```text
Plan
= 想做

PreparedCompaction
= Worker确认能做

Result
= Worker确认做完

PreparedReconciliation
= Scheduler确认这个结果仍然能作用到当前 canonical state

Commit
= 正式改变 allocator ownership / committed physical frontier
```

这是整个 V2 设计最值得保留的架构认知。

---

# 36. 推荐后续源码阅读顺序

不要再按 M5-T1 → T2A → T2B → T3 的研发历史顺序读。

建议严格沿 runtime 主链：

```text
1. vllm/v1/engine/core.py
   EngineCore.step()

2. vllm/v1/core/sched/scheduler.py
   Scheduler.schedule()

3. vllm/v1/core/sched/output.py
   CompactionPlanData
   SchedulerOutput.compaction_plans

4. vllm/v1/worker/gpu/model_runner.py
   extract_compaction_plans()
   execute_model()

5. 找 model.forward() seam

6. _prepare_v2_compactions()
   _PreparedCompaction

7. _execute_v2_compactions()

8. vllm/v1/worker/gpu/kv_compaction.py
   compact_paged_kv_triton_2d()

9. 回 model_runner.py
   ExecuteModelState
   sample_tokens()

10. vllm/v1/worker/gpu/input_batch.py
    post_update()

11. vllm/v1/outputs.py
    ModelRunnerOutput.compaction_results

12. 回 scheduler.py
    update_from_output()
    _prepare_compaction_reconciliations()
    _commit_compaction_reconciliations()

13. 三层 KV ownership API
    KVCacheManager
    → KVCacheCoordinator
    → SingleTypeKVCacheManager

14. 最后再看 deferred_frees / T4 lifecycle
```

每读一个函数只问三个问题：

```text
Q1. 我现在位于 EngineCore step 的哪个阶段？

Q2. 这个函数当前操作的是：
    intent / Worker view / GPU payload / receipt / canonical ownership
    中的哪一种？

Q3. 谁是这份状态的 authority？
```

只要这三个问题始终回答得出来，V2 链路就不会再散。

---

# 37. Evidence / Source Basis

本文基于当前项目中的以下实现记录重新综合，而不是复述单个 Slice：

```text
P1-M5-T1-SOURCE-MAP.md
P1-M5-T1-CODE-CHANGE-RECORD.md
P1-M5-T2A-CODE-CHANGE-RECORD.md
P1-M5-T2A-IMPLEMENTATION.log
P1-M5-T2B-FINAL-DESIGN-WALKTHROUGH.log
P1-M5-T2B-HARDEN-01-CODE-CHANGE-RECORD.md
P1-M5-T2B-HARDEN-01.log
P1-M5-T3-IMPLEMENT-01-CODE-CHANGE-RECORD.md
P1-M5-T3-IMPLEMENT-01.log
P1-M5-T3-HARDEN-01-CODE-CHANGE-RECORD.md
P1-M5-T3-HARDEN-01.log
P1-M5-T3-ACCEPTANCE-AUDIT.log
```

本文以 T2B Harden / T3 Harden 后的最终结构为准；历史 T2A 曾让 `_PreparedCompaction` 跨 `ExecuteModelState`，该 carrier 在最终 T2B Harden 中已经删除，因此不再作为当前架构描述。
