# P1 M5-T3 — Scheduler Canonical Reconciliation
## Source Audit → Problem Model → Frozen Design Contract

> Project: Physical KV Cache Reclamation for vLLM  
> Upstream: vLLM v0.26.0  
> Branch: `p1/v2-token-compaction-v026`  
> Audited baseline HEAD: `1f04cdca301fa72d20046a791672cf265bf9df09`  
> Previous state: M5-T2A CLOSED / M5-T2B HARDEN-01 PASS  
> This document freezes M5-T3 design only. No implementation is included.

---

# 0. Executive Decision

M5-T3 的职责是：

> 把 Worker 已经完成的 V2 post-forward physical KV compaction，安全地折回 Scheduler 的 canonical physical state。

T2B 结束后，Worker 已经是：

```text
KV payload            = compacted
Worker BlockTable     = retained physical prefix
Worker effective_kv_len = K
```

但 Scheduler 仍可能保持：

```text
request.effective_kv_len = pre-compaction physical frontier
req_to_blocks             = full post-allocation source row
```

所以 T3 必须完成：

```text
CompactionPlanData
+ CompactionResultData
+ Scheduler current canonical state
        ↓
all-result validation / prepare
        ↓
canonical req_to_blocks truncate
        ↓
Scheduler effective_kv_len = K      # absolute
        ↓
removed suffix detached
        ↓
Scheduler-controlled deferred release lifecycle
```

T3 不重新执行 payload compaction，不修改 Worker，不实现 retention policy，也不扩展到 PP>1 / multi-KV-group / spec decode / prefix caching / CUDA Graph。

---

# 1. 我们关心的主链路

```text
prepared V2 decision
    ↓
Scheduler._prepared_compaction_plans
    ↓
Scheduler.schedule()
    ↓
allocate_slots()
    ↓
Scheduler canonical req_to_blocks grows for current q
    ↓
SchedulerOutput.compaction_plans
    ↓
Worker extract_compaction_plans()
    ↓
model.forward()
    ↓
T2A _prepare_v2_compactions()
    ↓
_PreparedCompaction
    ↓
T2B _execute_v2_compactions()
    ↓
all requests × all layers payload compaction
    ↓
Worker BlockTable retained-prefix stage
    ↓
Worker effective_kv_len absolute override
    ↓
CompactionResultData
    ↓
ModelRunnerOutput
    ↓
Scheduler.update_from_output()
    ↓
M5-T3 canonical reconciliation  ← CURRENT WORK
```

这条链里有两个 authority：

```text
Worker:
execution truth

Scheduler:
allocator / ownership truth
```

T2B 只把 Worker truth 改正确；T3 才把 Scheduler truth 对齐。

---

# 2. State Ownership Model

## 2.1 Scheduler owns canonical ownership

核心 authority：

```text
SingleTypeKVCacheManager.req_to_blocks
```

它决定：

- request 当前 canonical 拥有哪些 KVCacheBlock；
- 后续 `get_num_blocks_to_allocate()` / `allocate_new_blocks()` 如何算新增 block；
- finish / preemption 时哪些 block 属于 request；
- block 最终何时能进入 BlockPool reuse lifecycle。

因此 Worker 不能 authoritative 地告诉 Scheduler “free B11”。

---

## 2.2 Worker owns execution view

Worker 侧主要 physical state：

```text
KV payload
BlockTable execution-addressing view
persistent effective_kv_len
```

T2B 已经让三者在 Worker 侧收敛。

---

## 2.3 CompactionResultData is a receipt

Result 当前只报告：

```text
request_id
expected_source_effective_kv_len
expected_source_num_blocks
new_effective_kv_len
new_num_blocks
step_seq
```

它不返回：

```text
freed_block_ids
```

这是刻意的 design decision。

Scheduler 应从自己的 canonical row 推导：

```text
retained prefix
removed suffix
```

---

# 3. 为什么 T3 是 correctness requirement

假设 T2B 后：

```text
Worker:
E = 20
active blocks = [B7,B2]
```

Scheduler 仍然：

```text
E = 32
req_to_blocks = [B7,B2,B11]
```

下一轮 allocation 同时依赖：

```text
request.effective_kv_len
len(req_to_blocks)
```

只改其中一个都会造成下一步 physical demand 与 already-owned capacity 不一致。

所以 T3 必须一起 reconcile：

```text
Scheduler E
+
Scheduler req_to_blocks
```

---

# 4. V1 和 V2 最终都在哪里改 Scheduler

两者最终都回到：

```text
Scheduler.update_from_output()
```

附近做 canonical reconciliation。

真正差别不是：

```text
V1 更新 Scheduler
V2 不更新 Scheduler
```

而是：

> V1 reclaim 发生在 forward 前，V2 compaction 发生在 forward 后。这导致 same-step newly allocated block 在两条路径中的语义完全不同。

---

# 5. V1 Timeline

设：

```text
block_size = 4
step start E = 8
old row = [B0,B1]
q = 2
```

Scheduler 为 q allocate：

```text
B2
```

暂时 canonical：

```text
[B0,B1,B2]
```

V1 reclaim 在 forward 前发生。

假设保留：

```text
old retained = [B0]
R = 4
```

此时 q 的 KV 尚未生成，因此：

```text
B2 = upcoming forward write capacity
```

Worker forward 前构造：

```text
[B0] + [B2]
=
[B0,B2]
```

forward 后：

```text
E_next = R + q = 6
```

所以 Scheduler update 时要把 current row 分成：

```text
old region:
[B0,B1]

same-step new region:
[B2]
```

最终：

```text
retained old
+
all same-step new
=
[B0,B2]
```

所以 V1 semantic model 是：

```text
select subset from PRE-STEP old row
+
preserve all current-step new blocks
```

---

# 6. V2 Timeline

相同初始条件：

```text
E = 8
old row = [B0,B1]
q = 2
allocate B2
```

Scheduler canonical：

```text
[B0,B1,B2]
```

V2 不在 forward 前 compact，而是先正常 forward：

```text
q KV written into B2
```

post-forward source：

```text
old KV + current-step new KV
=
whole source row [B0,B1,B2]
```

若 T2B：

```text
source E = 10
K = 5
new_num_blocks = 2
```

M4 可以把 B2 中仍需保留的 member gather 出来，再写回 physical prefix。

所以 B2 此时已经不再是 “未来必须保留的 write capacity”，只是完整 source storage 的一个 page。

最终 Worker：

```text
[B0,B1]
```

Scheduler reconciliation 应：

```text
current complete source row:
[B0,B1,B2]

↓ new_num_blocks = 2

retained = [B0,B1]
removed  = [B2]
```

即：

```python
retained_blocks = current_blocks[:result.new_num_blocks]
removed_blocks = current_blocks[result.new_num_blocks:]
```

---

# 7. Frozen Decision — V1 helper cannot be reused directly

**DO NOT directly reuse V1 `reconcile_reclaimed_blocks()` for V2.**

V1：

```text
retained subset of PRE-STEP old row
+
all same-step newly allocated blocks
```

V2：

```text
truncate COMPLETE post-allocation/post-forward source row
to prefix [0,new_num_blocks)
```

另外现有 V1 helper 还把：

```text
canonical reconciliation
+
BlockPool free
```

绑定在一起。

V2 T3 要把：

```text
canonical detach
```

与：

```text
safe release / reuse
```

解耦。

---

# 8. Scheduler Logical vs Physical Progress

Scheduler `_update_after_schedule()` 会 optimistic 推进：

```text
request.num_computed_tokens += num_scheduled_tokens
request.num_in_flight_tokens += num_scheduled_tokens
```

但不会同步推进：

```text
request.effective_kv_len
```

所以进入 `update_from_output()` 时可以是：

```text
num_computed_tokens = 48
effective_kv_len    = 32
```

而 Worker Result：

```text
expected_source_effective_kv_len = 48
```

这是正确的。

---

# 9. Scheduler 如何重建 V2 source E

不能简单验证：

```python
request.effective_kv_len == result.expected_source_effective_kv_len
```

当前 MVP（spec decode OFF）：

```python
q = scheduler_output.num_scheduled_tokens[req_id]

source_base = (
    request.effective_kv_len
    if request.effective_kv_len is not None
    else request.num_computed_tokens - q
)

scheduler_expected_source_E = source_base + q
```

必须满足：

```text
scheduler_expected_source_E
==
plan.expected_source_effective_kv_len
==
result.expected_source_effective_kv_len
```

---

# 10. V2 Scheduler E 必须 absolute commit

V1：

```text
E_next = R + q
```

因为 V1 reclaim source 不包含本轮 q。

V2：

```text
source = old physical + q
↓
post-forward compact
↓
K
```

所以：

```text
E_next = K
```

绝不能：

```text
K + q
```

这和 T2B Worker 的 absolute E override 是同一个原则。

---

# 11. T3 不能改 logical num_computed_tokens

M5-T3 不能：

```python
request.num_computed_tokens = result.new_effective_kv_len
```

合法状态就是：

```text
logical num_computed_tokens = 48
physical effective_kv_len   = 20
```

这是 P1 logical / physical divergence 的核心。

---

# 12. Request Lifecycle

## Already aborted / already finished

现有 `update_from_output()` 允许：

```text
request 在 GPU 执行期间被 abort
旧 in-flight output 后来返回
```

这种 output 应视为 stale completed output。

因此：

```text
Result identity 可以合法
但 request 已死亡
→ 不做 canonical apply
```

`Result valid` 与 `Result must be applied` 必须分开。

## Same-step EOS

若 request 进入 `update_from_output()` 时仍 active：

```text
先 T3 reconcile
再处理 sampled token / stop
最后 finish/free
```

是合理顺序。

## Preemption

`_preempt_request()` 会 reset：

```text
num_computed_tokens = 0
effective_kv_len = None
```

旧 Result 不能再把 K 写回去。

---

# 13. New Finding F1 — same-step preemption leaves local compaction plan

source audit 发现：

A 已经先被 schedule：

```text
_prepared_compaction_plans[A]
↓ pop
local compaction_plans[A]
```

随后同一个 `schedule()` 内 A 因 allocation pressure 被 preempt。

当前 rollback 会清：

```text
num_scheduled_tokens[A]
req_to_new_blocks[A]
req_to_reclaim_transition[A]
scheduled_spec_decode_tokens[A]
...
```

但 local：

```text
compaction_plans[A]
```

可能没有被清。

后果：

```text
SchedulerOutput.compaction_plans contains A
but A is not in actual Worker batch
```

Worker T2A 会报 request not in current batch。

---

# 14. Frozen Decision F1

在同一步 scheduled-request rollback path 增加：

```python
compaction_plans.pop(preempted_req_id, None)
```

Mandatory regression：

```text
A has V2 plan and is scheduled first
B causes A same-step preemption

EXPECT:
A absent from num_scheduled_tokens
A absent from final scheduled batch
A absent from SchedulerOutput.compaction_plans
```

---

# 15. New Finding F2 — V1 / V2 same-request same-step coexistence is not fenced

当前：

```text
_prepared_reclaim_plans
_prepared_compaction_plans
```

是独立 ingress。

理论上同一 request 同 step 可以同时：

```text
V1 pre-forward reclaim
+
V2 post-forward compaction
```

当前 source-E / ownership contracts 并未定义这种组合。

---

# 16. Frozen Decision F2

P1 MVP：

```text
same request / same scheduler step:

V1 ReclaimTransition
XOR
V2 CompactionPlan
```

不得共存。

必须在 schedule-time materialization point 有 final correctness fence：

```python
if tentative_reclaim_transition is not None and compaction_plan is not None:
    raise ValueError(...)
```

setter early guard 可选，但 schedule-time fence 必须存在。

---

# 17. T3 Transaction Model

冻结成：

```text
PHASE 0 — Result Admission
PHASE 1 — Prepare / Validate
PHASE 2 — Canonical Commit
```

禁止：

```text
validate A
→ mutate A
→ validate B
→ B fails
```

必须：

```text
validate ALL active Results
→ all pass
→ commit ALL
```

这是 Scheduler metadata transaction atomicity。

---

# 18. Phase 0 — Result Admission

Inputs：

```text
scheduler_output.compaction_plans
model_runner_output.compaction_results
```

至少验证：

```text
unique Result request IDs
every Result corresponds to current-step Plan
request_id matches
expected source E matches
expected source block count matches
step_seq matches
```

不允许 unsolicited Result。

已经 abort / finished 的 request 不自动算 transaction identity error；identity validation 与 lifecycle apply decision分开。

---

# 19. Scheduler-side Prepared Reconciliation

建议 temporary descriptor：

```python
@dataclass(frozen=True)
class _PreparedCompactionReconciliation:
    request_id: str
    source_effective_kv_len: int
    source_num_blocks: int
    new_effective_kv_len: int
    new_num_blocks: int
    step_seq: int | None
```

如果为了保证 all-validation-before-commit 更干净，可以再保存：

```text
canonical snapshot
retained blocks
removed blocks
```

Worker `_PreparedCompaction`：

```text
how to execute payload movement
```

Scheduler `_PreparedCompactionReconciliation`：

```text
how to commit canonical ownership
```

二者职责不同。

---

# 20. Phase 1 Validation Contract

对于仍 active 的 Result：

## Transaction identity

```text
result.request_id == plan.request_id
result.expected_source_E == plan.expected_source_E
result.expected_source_num_blocks == plan.expected_source_num_blocks
result.step_seq == plan.step_seq
```

## Current scheduled state

```text
req_id in scheduler_output.num_scheduled_tokens
q > 0
request exists
request not already finished
```

## Reconstructed source

```text
source_base =
    effective_kv_len
    if E exists
    else num_computed_tokens - q

expected_source_E =
    source_base + q
```

必须：

```text
expected_source_E == result.expected_source_E
```

## Canonical ownership

```text
current_blocks = canonical req_to_blocks[req_id]

len(current_blocks)
==
result.expected_source_num_blocks
```

Current P1 MVP exact-page contract：

```text
source_num_blocks
==
cdiv(source_E, block_size)
```

## Destination

```text
0 < new_effective_kv_len <= source_effective_kv_len

new_num_blocks
==
cdiv(new_effective_kv_len, block_size)

0 < new_num_blocks <= source_num_blocks
```

## Derivation

```python
retained_blocks = current_blocks[:new_num_blocks]
removed_blocks = current_blocks[new_num_blocks:]
```

Phase 1 禁止任何 runtime mutation。

---

# 21. Keep-all / same-page-count transition is valid

例如：

```text
source E = 32
B = 16
source pages = 2

K = 25
new pages = 2
```

token-level payload 已经从 32 压成 25，但 page count 仍为 2。

因此：

```text
new_num_blocks == source_num_blocks
removed_blocks = []
```

完全合法。

不能要求：

```text
new_num_blocks < source_num_blocks
```

---

# 22. Phase 2 — Canonical Commit

所有 active Result 验证通过后才：

```text
req_to_blocks = retained prefix
request.effective_kv_len = K
num_cached_block compatibility bookkeeping
removed suffix -> Scheduler-owned release lifecycle
```

`request.num_computed_tokens` 不改。

---

# 23. V2-specific ownership helper

建议独立 API，例如：

```python
reconcile_compacted_blocks(
    request_id: str,
    expected_source_num_blocks: int,
    new_num_blocks: int,
) -> list[KVCacheBlock]
```

SingleType semantics：

```text
validate current row exists
validate len(current row) == expected source blocks
validate 0 < new_num_blocks <= expected_source_num_blocks

retained = current_blocks[:new_num_blocks]
removed  = current_blocks[new_num_blocks:]

req_to_blocks[request_id] = retained

if num_cached_block entry exists:
    bound it to len(retained)

return removed
```

MUST NOT：

```text
accept Worker freed IDs
call block_pool.free_blocks()
modify request.effective_kv_len
modify Worker
modify payload
```

---

# 24. Removed suffix ownership

不能：

```text
truncate req_to_blocks
然后丢失 removed block references
```

否则 block 会脱离 request ownership、又未归还 BlockPool，形成 ownership leak。

所以 T3 canonical detach 必须同时把 removed suffix 交给明确 Scheduler-controlled release owner。

---

# 25. Deferred release

Scheduler 已有：

```text
sched_step_seq
processed_step_seq
deferred_frees
```

优先复用已有 infrastructure。

但：

```text
CompactionResultData.step_seq
```

主要是 transaction correlation，不自动等同于 Scheduler free fence。

冻结：

```text
transaction identity
!=
release fence authority
```

release fence 应由 Scheduler 自己的 execution lifecycle 决定。

保守地多延迟一个 drain 周期是可接受的 MVP tradeoff。

---

# 26. Scheduler E branch after T3

现有结构大致：

```python
if reclaim_state is not None:
    E = V1_R + q
elif request.effective_kv_len is not None:
    E += q
```

T3 后语义应变为：

```python
if req_id in compacted_req_ids:
    # T3 already committed E=K absolute
    pass
elif reclaim_state is not None:
    # existing V1 path
    E = R + q
elif request.effective_kv_len is not None:
    # normal physical-diverged path
    E += q
```

由于 V1 XOR V2，不应出现双重 transition。

---

# 27. Canonical E2E Example

Start：

```text
B = 16
logical L = 32
physical E = 32
row = [B7,B2]
```

Current q：

```text
q = 16
allocate B11
```

After scheduling：

```text
num_computed_tokens = 48
effective_kv_len = 32
row = [B7,B2,B11]
```

Plan：

```text
expected_source_E = 48
expected_source_blocks = 3
```

Worker T2B：

```text
[B7,B2,B11], E=48
↓ compact
K=20
new_num_blocks=2
```

Worker：

```text
BlockTable=[B7,B2]
E=20
```

Result：

```text
source_E=48
source_blocks=3
new_E=20
new_blocks=2
```

T3：

```text
source_base = 32
expected source = 32 + 16 = 48

current canonical len = 3

retained=[B7,B2]
removed=[B11]
```

Commit：

```text
req_to_blocks:
[B7,B2,B11] -> [B7,B2]

Scheduler E:
32 -> 20

logical num_computed_tokens:
48 unchanged

removed B11:
Scheduler deferred release lifecycle
```

Final：

```text
logical progress = 48
physical retained = 20
```

---

# 28. Required Regression Matrix

```text
T3-01 canonical success
T3-02 Scheduler E absolute K
T3-03 E never becomes K+q
T3-04 logical num_computed_tokens unchanged
T3-05 canonical row truncates to prefix
T3-06 removed suffix has explicit release owner
T3-07 same page count allowed
T3-08 Plan/Result source-E mismatch fails pre-mutation
T3-09 Plan/Result source-block mismatch fails pre-mutation
T3-10 step_seq mismatch fails pre-mutation
T3-11 canonical row count mismatch fails pre-mutation
T3-12 multiple Results: later invalid -> no earlier commit
T3-13 stale aborted/finished Result skipped
T3-14 V1 path unchanged
T3-15 normal additive E path unchanged
T3-16 same-step preemption removes materialized V2 plan
T3-17 V1 + V2 same request/same step rejected
T3-18 no immediate unsafe BlockPool reuse
```

---

# 29. Expected Touch Surface

Production：

```text
vllm/v1/core/sched/scheduler.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/single_type_kv_cache_manager.py
```

Tests：

```text
tests/v1/core/test_compaction_reconciliation.py
tests/v1/core/test_reclaim_transport.py
tests/v1/core/test_reclaim_ownership.py
```

T3 原则上不应修改：

```text
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/input_batch.py
vllm/v1/worker/gpu/kv_compaction.py
```

---

# 30. Non-goals

M5-T3 不做：

```text
Worker payload mutation
M4 redesign
retention policy
K=0
multi-KV-group
prefix-cache support
spec-decode semantics
PP>1
CUDA Graph support
performance optimization
V1+V2 generalized composition
```

---

# 31. Implementation Traceability Contract

每一处实际改动必须归类为：

```text
ADD
DELETE
REPLACE
EXTEND
MOVE
TEST_ONLY
EVIDENCE_ONLY
```

每个 CHG 必须记录：

```text
Change ID
File
Change type
Symbol
Old line range
Final line range

Previous code / responsibility
Exact operation
New code / responsibility
Why required
Requirement mapping
Classification
Runtime mutation introduced?
Compatibility impact
Tests covering it
```

尤其禁止只写：

```text
modified scheduler.py
```

必须能回答：

```text
原来哪段 branch 是什么
删了什么
加了什么
替换成什么
最终在哪一行
为什么改
```

---

# 32. Frozen Status

```text
M5-T3 Source Audit
= CLOSED

M5-T3 Lifecycle Audit
= CLOSED

V1 helper reuse
= NO / FROZEN

Scheduler E
= absolute K / FROZEN

Multi-result transaction
= all validate before commit / FROZEN

Removed suffix
= Scheduler-controlled deferred release / FROZEN

F1 same-step stale V2 plan
= MUST FIX

F2 V1/V2 same-step coexistence
= MUST REJECT

M5-T3
= READY FOR IMPLEMENTATION SLICE
```
