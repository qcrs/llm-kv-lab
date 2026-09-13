# P1 V2 — Token-Level Physical KV Compaction Runtime Architecture

## Approved Design v1

**Project:** P1 — Physical KV Cache Reclamation for vLLM  
**Phase:** V2 — Token-Level Physical KV Compaction  
**Status:** `APPROVED_DESIGN / V2 CORE RUNTIME ARCHITECTURE FROZEN`  
**Date:** 2026-09-07  
**Design Authority:** ChatGPT Web / Project Architect  
**Implementation Agent:** Codex CLI（仅在后续批准 Slice 后执行）

---

# 0. 文档目的

本文正式冻结 P1 V2 Token-Level Physical KV Compaction 的核心 runtime architecture。

它不是实现说明书，也不是 retention policy 设计文档。本文要回答并固定的是以下系统问题：

1. **V2 compaction 应该插入 vLLM runtime 的哪个准确位置？**
2. **`keep_member_indices` 由谁产生、什么时候绑定到请求状态、如何传到 Worker？**
3. **Worker 完成 KV payload compaction 后，Worker state、Scheduler canonical ownership、`effective_kv_len` 和 trailing block lifetime 应如何提交？**
4. **V1 哪些机制继续复用，哪些必须保持 V1-only，哪些是 V2 新能力？**
5. **后续 PyTorch Oracle / Triton Gather / Scatter / Runtime Integration 应围绕什么明确 contract 实现？**

本文基于：

- upstream vLLM `v0.26.0`
- upstream commit `568afb3a13806beb53bb2e6bd518269357b237c0`
- P1 V1 accepted implementation `cd444b72ceecb2496bf015baf8c18bf883151fa1`
- 当前 V2 worktree observed HEAD `064f4efe082bc23087043381a4a401c170997f5c`
- Tangram audited commit `8e1cbfa5cf82acc67fe1880df0c632f2a6485e35`
- `P1-V2-SOURCE-WALKTHROUGH`
- `P1-V2-RUNTIME-SEAM-WALKTHROUGH.log`
- P1 V1/Core accepted gate evidence

本文不是对 Tangram 的移植方案。Tangram 只作为 **semantic / runtime placement reference**。

---

# 1. 当前 Scope

本文所有 `APPROVED_DESIGN` 结论严格限定于当前 P1 V2 MVP：

```text
vLLM                   = v0.26.0
Model Runner            = MRV2
GPU                     = A100 80GB
dtype                   = BF16
Attention backend       = FA2
Execution               = eager
TP / PP / DP            = 1 / 1 / 1
DCP / PCP               = 1 / 1
KV cache group          = single group
block_size              = 16
prefix caching          = OFF
speculative decoding    = OFF
async scheduling        = OFF
CUDA Graph              = OFF
KV connector/offload    = OFF
```

当前设计不宣称支持：

```text
PP > 1
TP > 1
multi KV group
hybrid / ragged KV
prefix sharing
spec decode
async scheduling
CUDA Graph
KV Connector
cross-request shared pages
per-layer independent keep sets
per-head independent keep sets
```

这些属于后续 hardening / extension，不得从本设计自动外推。

---

# 2. V2 要解决的准确问题

P1 V1 已经完成：

```text
logical progress
        ≠
physical KV progress

positions
        ≠
cache_positions

Scheduler canonical ownership
        ≠
Worker BlockTable execution view
```

并完成：

```text
whole-block reclaim
→ Worker dense physical row
→ physical E
→ Scheduler reconcile
→ BlockPool safe release
→ subsequent real reuse
```

V2 不重做这些基础设施。

V2 新增的是：

> 给定一个 request 当前已经存在的 paged KV 和一个 externally supplied `keep_member_indices`，将被保留的 token-level K/V 按原有 member 顺序重新打包到当前 physical BlockTable 的连续前缀 `[0, K)`，使 `effective_kv_len = K`，并在 transaction 安全提交之后释放不再需要的 trailing pages。

形式化：

```text
Input:
    current paged KV
    current active physical BlockTable row
    E_source
    sorted unique keep_member_indices

Transformation:
    paged source
        → gather retained members
        → contiguous scratch
        → scatter/writeback to physical prefix

Output:
    valid KV = physical positions [0, K)
    E_new = K
    new_num_blocks = ceil(K / block_size)
    trailing canonical pages become reclaim candidates
```

其中：

```text
K = len(keep_member_indices)
```

---

# 3. V2 明确不解决什么

P1 V2 Core **不定义 retention policy**。

它不回答：

```text
哪个 token 更重要？
attention score 怎么计算？
是否使用 H2O / SnapKV / StreamingLLM？
是否训练 learned utility？
每层应该保留多少？
每个 head 应该保留多少？
压缩比如何自适应变化？
```

V2 Core 的边界是：

```text
Policy / Experiment / Explicit Plan
                 │
                 │ keep_member_indices
                 ▼
        Physical Compaction Core
```

因此：

> `keep_member_indices` 是 V2 Core 的输入，不是 V2 Core 内部推导出来的结果。

这是项目 scope 控制的核心设计之一。

---

# 4. 已冻结的基础语义

## 4.1 Logical position 不改变

假设原逻辑 token：

```text
logical positions:
0 1 2 3 4 5 6 7 8 9
```

选择：

```text
keep_member_indices:
0 2 5 8 9
```

compact 后这些 KV 被放到：

```text
physical positions:
0 1 2 3 4
```

但它们的模型语义仍来自原始 logical positions：

```text
0 2 5 8 9
```

因此：

```text
RoPE / model logical coordinate
        ≠
physical KV storage coordinate
```

V2 不能通过重新编号 logical position 来实现压缩。

## 4.2 `keep_member_indices` 的准确含义

`keep_member_indices` 索引的是：

> **compaction 执行前、当前 request 的 physical retained/member sequence。**

它不是：

```text
original prompt token index
logical position
raw global slot id
physical block id
(block_id, offset)
```

例如：

```text
E_source = 10

current member sequence:
m0 m1 m2 m3 m4 m5 m6 m7 m8 m9

keep_member_indices:
[0, 2, 5, 8, 9]
```

其语义是：

```text
keep:
m0 m2 m5 m8 m9
```

随后再根据当前 BlockTable 解析 source page/offset。

## 4.3 保序

V2 Core 要求：

```text
keep_member_indices strictly increasing
```

即：

```text
[0, 2, 5, 8, 9]     VALID
[0, 5, 2, 8, 9]     INVALID
[0, 2, 2, 8, 9]     INVALID
```

禁止 Core silently：

```text
sort()
deduplicate()
```

原因是：

> 调用方给出的 member order 本身就是 compaction semantics 的一部分；Core 只能 validate，不能修改 policy result。

---

# 5. Source-backed Runtime Facts

## 5.1 P1 / vLLM MRV2 normal execution chain

当前 P1 `GPUModelRunner.execute_model()` 的主要顺序：

```text
SchedulerOutput
        ↓
GPUModelRunner.execute_model()
        ↓
update_pp_decode_requests()
finish_requests()
free_states()
add_requests()
update_requests()
        ↓
effective_kv_len.apply_write()
block_tables.apply_staged_writes()
        ↓
prepare_inputs()
        ↓
prepare_attn()
        ↓
model.forward()
        ↓
model_output
        ↓
ExecuteModelState(...)
        ↓
execute_model() return
        ↓
sample_tokens()
        ↓
sample()
        ↓
ModelRunnerOutput
        ↓
postprocess_sampled()
```

当前 eager path 的关键源码 seam：

```python
model_output = self.model(**model_inputs)

# candidate V2 seam

if self.is_last_pp_rank:
    ...
```

随后才发布：

```python
self.execute_model_state = ExecuteModelState(...)
```

随后独立进入：

```python
sample_tokens(...)
```

因此源码存在一个干净的生命周期边界：

```text
model.forward completed
        ↓
all current-step layer KV writes have been issued
        ↓
sampling has not started
        ↓
logical postprocess has not happened
```

## 5.2 Tangram reference chain

Tangram audited path 的关键顺序：

```text
compression step begins
        ↓
model forward
        ↓
post-forward compression
        ↓
CompressionExecutor.run_request()
        ↓
KV gather / writeback
        ↓
effective lengths fold-in
        ↓
BlockTable trailing compaction
        ↓
freed IDs collected
        ↓
sampling / bookkeeping
        ↓
ModelRunnerOutput
        ↓
Scheduler
        ↓
KVCacheManager.free_blocks_by_ids()
```

Tangram 源码直接说明其 post-forward compression：

```text
this step's attention is already done
```

这一点是 P1 选择 post-forward placement 的 reference evidence。

但 P1 不继承 Tangram：

```text
importance scorer
compressor policy
cross-layer keep decision
ragged per-layer physical length
per-layer/group paging architecture
```

---

# 6. Decision 1 — Exact Runtime Hook

## 6.1 Approved Decision

```text
APPROVED_DESIGN

V2 compaction runtime hook
=
GPUModelRunner.execute_model()

AFTER:
    model forward returns

BEFORE:
    model output postprocessing
    ExecuteModelState publication
    sample_tokens()
```

当前 MVP 中更精确的概念位置：

```python
model_output = self.model(**model_inputs)

# -----------------------------------------
# P1 V2 post-forward compaction hook
# -----------------------------------------
self._run_v2_compaction_if_needed(...)

if self.is_last_pp_rank:
    ...
```

注意：

`_run_v2_compaction_if_needed` 只是本文的概念命名。

**函数名尚未冻结。**

冻结的是：

```text
lifecycle placement
```

而不是 Python symbol 名称。

---

# 7. Decision 1 为什么不是 pre-forward

假设：

```text
E_before = 32
q = 16
```

当前 forward 会生成 physical members：

```text
[32, 48)
```

如果 plan 针对 final current sequence：

```text
E_source = 48
```

那么 compaction 在 forward 前执行时：

1. 本轮新 K/V 尚不存在；
2. plan 无法选择 current-step newly written members；
3. 更严重的是，它会改变本轮 attention context。

即：

```text
compact old KV
        ↓
current attention sees compressed history
```

这已经不再只是 physical reclamation，而是在修改当前 forward 的 attention semantics。

P1 V2 选择：

```text
current forward first
        ↓
then compact
```

意味着：

```text
当前 final-prefill attention semantics
保持原样
```

compaction 只改变未来 physical KV state。

---

# 8. Decision 1 为什么不拖到 next decode 前

候选：

```text
final prefill
    ↓
sample
    ↓
Scheduler next iteration
    ↓
Worker before decode
    ↓
compact
```

虽然理论可行，但会制造一个不必要的 cross-step debt：

```text
Scheduler canonical state
        ?
Worker payload state
```

例如：

```text
forward completed E_source = 48
planned E_new = 21
```

如果 Worker 尚未 compact，下一轮 Scheduler 到底应该按：

```text
48
```

还是：

```text
21
```

进行 `allocate_slots()`？

P1 V1 已经建立：

```python
allocation_base = (
    request.effective_kv_len
    if request.effective_kv_len is not None
    else total_computed_tokens
)
```

因此正确方向是：

```text
current forward
        ↓
current Worker compaction
        ↓
current step result-time canonical commit
        ↓
next Scheduler iteration
```

而不是将 physical mutation 延迟到下一次 Worker execution。

---

# 9. Decision 1 为什么不放在 `sample_tokens()`

两个候选：

```text
A:
forward
→ compact
→ ExecuteModelState
→ sample_tokens

B:
forward
→ ExecuteModelState
→ sample_tokens
→ compact
→ sample
```

最终选择 A。

原因一：

```text
KV compaction
```

属于：

```text
Model Execution / KV Data Plane
```

而不是：

```text
Sampling / Token Bookkeeping
```

原因二：transaction publication 更清晰。

`ExecuteModelState` 可以继续表达：

> 本轮 model execution 和它要求的 physical KV post-operation 已经成功，可以进入 sampling。

因此顺序应该是：

```text
forward
    ↓
physical KV transaction
    ↓
ExecuteModelState publication
```

而不是：

```text
forward
    ↓
publish execution state
    ↓
physical KV transaction
```

原因三：失败语义更干净。

若 compaction 在 `ExecuteModelState` 发布前失败：

```text
execute_model_state remains unpublished
```

不会出现：

```text
published execution state
+
failed physical state mutation
```

---

# 10. Decision 1 与 CUDA Stream Ordering

`model.forward()` 返回不代表 CPU 与 GPU 全局同步。

实际更接近：

```text
CPU enqueue:
forward kernels
KV writes
V2 gather
V2 scatter
```

若 V2 primitive 与 model forward 使用同一 main CUDA stream：

```text
GPU:
forward KV write
      ↓
V2 gather
      ↓
V2 scatter
```

stream ordering 本身保证 Worker 内部数据依赖。

因此 MVP 不应为了 correctness 无条件加入：

```python
torch.cuda.synchronize()
```

但必须区分：

```text
same-stream ordering
```

只解决：

```text
Worker GPU data dependency
```

它不自动解决：

```text
Scheduler / BlockPool reuse lifetime
```

后者由 Decision 3 处理。

---

# 11. Decision 2 — `keep_member_indices` Ownership

## 11.1 Approved Decision

```text
APPROVED_DESIGN

P1 V2 Core does not own retention policy.

Exact retained members are supplied as:
    keep_member_indices

Plan producer:
    external / explicit producer outside V2 physical core

Control-plane binding:
    Scheduler side

Transport:
    SchedulerOutput

Consumer:
    GPUModelRunner post-forward V2 hook

Execution:
    Worker GPU compaction primitive
```

---

# 12. Decision / Plan / Execution 必须分层

必须区分三件事：

```text
Trigger:
    这一轮是否做 compaction？

Selection:
    保留哪些 members？

Execution:
    如何搬 paged KV？
```

P1 V2 Core 只负责第三层，并接收第二层结果。

最终架构：

```text
External Explicit Plan Producer
            │
            │ keep_member_indices
            ▼
Scheduler-side prepared plan
            │
            │ schedule-time materialization
            ▼
SchedulerOutput
            │
================ CONTROL / DATA ================
            ▼
GPUModelRunner
            │
            │ model.forward
            ▼
post-forward V2 hook
            │
            ▼
Paged KV Compaction Core
```

---

# 13. 为什么不让 Scheduler 内置 retention policy

Scheduler 应负责：

```text
request lifecycle
scheduling
canonical block ownership
allocation
preemption
transaction transport
```

它不应该变成：

```text
attention-score analyzer
token importance ranker
H2O/SnapKV controller
```

否则：

```text
control-plane allocator
```

与：

```text
model-dependent retention algorithm
```

被错误耦合。

Scheduler 可以决定：

```text
本次计划是否绑定到该 scheduled step
```

但不应该定义：

```text
token 5 比 token 8 更重要
```

---

# 14. 为什么不让 GPUModelRunner 自己发明 keep plan

禁止在 Core 中写：

```python
scores = ...
keep = topk(...)
```

否则 `GPUModelRunner` 同时承担：

```text
trigger policy
+
retention policy
+
physical execution
```

这使 future policy replacement 必须修改核心 runtime。

正确依赖方向：

```text
H2O / SnapKV / Tangram / offline oracle / random baseline
                           │
                           │ keep_member_indices
                           ▼
                 common compaction contract
                           │
                           ▼
                 same runtime + same kernel
```

---

# 15. 第一版 Plan Producer

第一版 approved producer 是：

```text
deterministic explicit plan
```

目的不是模拟真实 policy，而是验证：

```text
token-level compaction correctness
runtime transaction
real trailing-page reclaim
real continuation
real block reuse
```

例如：

```text
E_source = 10
keep_member_indices = [0, 2, 5, 8, 9]
```

测试只需要证明：

```text
paged source
→ correct retained payload
→ prefix
→ E=5
→ 2 destination blocks
→ trailing page safe release
→ generation continuation
```

这已经形成完整 AI Infra capability，不依赖 retention quality。

---

# 16. Plan 必须绑定 source state

不能只传：

```python
keep_member_indices: list[int]
```

而完全没有 source-version context。

原因：

`member index` 的含义依赖当前 physical member sequence。

如果 producer 认为：

```text
E_source = 48
```

但 Worker 执行时：

```text
actual source E = 49
```

继续执行是不安全的。

因此 transport contract 必须存在 stale-plan fence。

当前冻结的是语义：

```text
expected source extent must be validated
```

而不是具体字段名。

概念上：

```text
CompactionPlanData:
    keep_member_indices
    expected_source_effective_kv_len
    expected_source_num_blocks
    request/step identity if required
```

最终 exact dataclass 字段可在 implementation slice 中基于现有 Scheduler transport 结构做最小设计。

---

# 17. Plan 的时间语义

假设：

```text
forward-entry E_before = 32
current q = 16
```

post-forward source 是：

```text
E_source = 48
```

所以：

```text
keep_member_indices ∈ [0, 48)
```

不是：

```text
[0, 32)
```

即使 explicit plan 在 Scheduler schedule 时已经准备好，它描述的也是：

> **本轮 forward 完成后的 source physical member sequence。**

这点必须写入 plan contract。

---

# 18. V2 All-Layer Semantics

当前 MVP 采用：

```text
one request
one keep_member_indices
all attention layers apply the same member compaction
```

不支持：

```text
layer 0 keep set A
layer 1 keep set B
layer 2 keep set C
```

原因：

后者会引入：

```text
per-layer effective length
ragged per-layer paging
cross-layer aggregation
layer-specific ownership
```

这是 Tangram 更一般的架构问题，不属于 P1 V2 MVP。

---

# 19. Decision 3 — Transaction / Ownership Commit Protocol

## 19.1 Approved Decision

```text
APPROVED_DESIGN

V2 transaction is divided into:

Stage 0: fail-before-mutation validation
Stage 1: Worker KV data-plane compaction
Stage 2: Worker physical-view commit
Stage 3: Worker result publication
Stage 4: Scheduler canonical ownership commit
Stage 5: safe-free / BlockPool reuse commit
```

并明确：

> trailing block 在 payload compaction 完成并完成 canonical transaction 之前，绝不能重新进入可复用 BlockPool。

---

# 20. Decision 3 最重要的 Authority Rule

保持 V1 已验证原则：

```text
Scheduler canonical ownership
        ≠
Worker BlockTable execution view
```

具体：

```text
SingleTypeKVCacheManager.req_to_blocks
```

仍是 Scheduler 侧 canonical ownership。

Worker：

```text
BlockTables
```

只是 GPU execution addressing view。

V2 不允许因为 Worker 可以看到 block IDs 就把 allocator authority 迁到 Worker。

---

# 21. Stage 0 — Fail Before Mutation

任何 gather/scatter mutation 之前，Worker 至少验证：

```text
request still exists

plan exists for this step

actual E_source
    ==
expected_source_effective_kv_len

keep list non-empty
    [MVP scope; E_new=0 remains unsupported unless separately approved]

keep strictly increasing

keep unique

0 <= keep[i] < E_source

K = len(keep)

new_num_blocks = ceil(K / block_size)

new_num_blocks <= old_num_blocks

source physical row is valid for expected state
```

如果任何 validation 失败：

```text
NO PAYLOAD MUTATION
NO WORKER ROW MUTATION
NO E MUTATION
NO SCHEDULER RESULT
NO FREE
```

继承 V1 的核心：

```text
fail closed
```

---

# 22. Stage 1 — Data-Plane Commit

Correctness baseline 固定为：

```text
Paged source KV
        ↓
Gather retained members
        ↓
Contiguous scratch
        ↓
Scatter / writeback
        ↓
Current physical BlockTable prefix
```

禁止将 naive arbitrary direct in-place paged→paged copy 作为 baseline。

原因：

source 和 destination 可能 overlap。

例如：

```text
keep = [0, 1, 4, 5, 19, 20]

destination positions = [0, 1, 2, 3, 4, 5]
```

多个 GPU programs 并行执行时不存在全局顺序保证，因此 source 位置可能被 destination write 提前覆盖。

所以：

```text
scratch semantics
=
correctness baseline
```

后续如需优化，可另行证明 overlap-safe primitive，但不得改变 baseline correctness contract。

---

# 23. Concrete Compaction Example

假设：

```text
block_size = 4

source physical row:
[B7, B2, B11]

E_source = 10
```

当前 member：

```text
B7:
m0 m1 m2 m3

B2:
m4 m5 m6 m7

B11:
m8 m9 _ _
```

选择：

```text
keep_member_indices:
[0, 2, 5, 8, 9]
```

Gather：

```text
scratch:
[m0, m2, m5, m8, m9]
```

Scatter：

```text
B7:
m0 m2 m5 m8

B2:
m9 _ _ _

B11:
stale / no longer visible
```

于是：

```text
K = 5

E_new = 5

new_num_blocks
=
ceil(5 / 4)
=
2
```

新的 Worker execution row：

```text
[B7, B2]
```

candidate trailing block：

```text
[B11]
```

---

# 24. Stage 2 — Worker Physical-View Commit

Data-plane operation 已按正确 stream ordering enqueue 后，Worker 将自己的 physical execution view 更新为：

```text
Worker BlockTable:
old_row[:new_num_blocks]
```

以及：

```text
Worker effective_kv_len = K
```

V2 与 V1 在这里存在关键差异。

V1：

```text
old:
[B0 B1 B2 B3 B4 B5]

retained:
[B0 B1 B4 B5]
```

属于：

```text
arbitrary retained whole-block row
```

V2：

```text
old:
[B7 B2 B11]

new:
[B7 B2]
```

由于 retained token 已被重新打包到 destination prefix，block ownership 只需要：

```text
prefix trim
```

因此 V2 的 Scheduler canonical reconcile 可以比 V1 更简单。

---

# 25. V1 E 与 V2 E 的时间语义不同

这是禁止复用同一 transition contract 的核心理由之一。

## V1

V1 reclaim 在 forward 前。

例如：

```text
E_old = 94
whole-block reclaim target = 62
q = 1
```

Worker forward-entry：

```text
E_target = 62
```

本轮 forward 又写 1 token。

Scheduler completed state：

```text
E_completed = 63
```

所以 V1 result-time 规则：

```text
E = E_target + q
```

## V2

V2 在 forward 后。

例如：

```text
E_before = 32
q = 16

post-forward:
E_source = 48
```

compact：

```text
K = 21
```

那么：

```text
E_new = 21
```

已经是：

```text
post-forward completed physical state
```

Scheduler 必须：

```text
request.effective_kv_len = 21
```

不能：

```text
21 + 16
```

因此：

```text
V1 E_target
        ≠
V2 E_new
```

它们属于不同 lifecycle coordinate。

---

# 26. `ReclaimTransitionData` 保持 V1-only

当前 V1：

```text
ReclaimTransitionData
```

方向：

```text
Scheduler
    ↓
Worker
```

语义：

> 执行一个 pre-forward whole-block physical transition。

它本质属于：

```text
CONTROL COMMAND
```

其字段服务：

```text
retained_block_ids
new_effective_kv_len as forward-entry target
expected_old_num_blocks
```

V2 不应该把 token-level result 硬塞进这个 contract。

---

# 27. V2 使用独立的 Plan / Result Contract

V2 至少存在两个不同方向的语义对象。

## Scheduler → Worker

概念：

```text
CompactionPlanData
```

语义：

> 如果本轮 forward 成功完成，则基于指定 source state 对该 request 执行 token compaction。

概念字段：

```text
keep_member_indices
expected_source_effective_kv_len
expected_source_num_blocks
optional step/version fence
```

## Worker → Scheduler

概念：

```text
CompactionResultData
```

语义：

> Worker 已执行 post-forward physical compaction，现在 physical state 已缩短到该 shape，请 Scheduler reconcile canonical ownership。

概念字段至少表达：

```text
request identity

expected_source_effective_kv_len

expected_source_num_blocks

new_effective_kv_len

new_num_blocks

step/fence identity if existing runtime requires
```

再次强调：

这些名称是 architecture-level contract。

最终 exact Python dataclass 名称和字段布局由 implementation slice 最小化决定。

---

# 28. 为什么 Result 不应以 Worker-reported Freed IDs 为 Authority

Tangram 可以输出：

```text
compression_freed_block_ids
```

然后 Scheduler 走：

```text
free_blocks_by_ids
```

但 P1 V1 已经建立：

```text
Scheduler owns canonical req_to_blocks
```

因此 P1 V2 更合适的是：

Worker 报告：

```text
new_num_blocks = 2
```

Scheduler 当前 canonical row：

```text
[B7, B2, B11]
```

Scheduler 自己推导：

```text
retained:
canonical_row[:2]
=
[B7, B2]

removed:
canonical_row[2:]
=
[B11]
```

而不是：

```text
Worker says free B11
→ Scheduler blindly trusts
```

Worker-reported candidate IDs 可以用于 debug/evidence，但不应取代 canonical ownership authority。

---

# 29. Stage 4 — Scheduler Canonical Commit

Scheduler 收到：

```text
CompactionResultData
```

先读取 canonical row：

```text
canonical_row = req_to_blocks[request_id]
```

先完整 validation，例如：

```text
len(canonical_row)
    ==
expected_source_num_blocks

new_num_blocks
    ==
ceil(new_effective_kv_len / block_size)

0 < new_num_blocks <= old_num_blocks

new_effective_kv_len
    <=
expected_source_effective_kv_len
```

所有 validation 必须发生在 mutation 之前。

随后推：

```text
retained = canonical_row[:new_num_blocks]

removed = canonical_row[new_num_blocks:]
```

然后 canonical commit：

```text
req_to_blocks[request_id] = retained
```

只有 canonical reconcile 成功后，才：

```text
request.effective_kv_len = new_effective_kv_len
```

这个顺序继承 V1：

```text
reconcile ownership
        ↓
commit E
```

避免：

```text
E changed
+
ownership reconcile failed
```

形成 Scheduler internal split-brain。

---

# 30. Logical Release 与 Physical Reuse 必须分离

当 Scheduler 将：

```text
[B7 B2 B11]
```

变成：

```text
[B7 B2]
```

意味着：

```text
B11
```

已经不属于该 request 的 canonical active row。

但这不等于：

```text
B11 immediately reusable
```

必须区分：

```text
Request ownership detach
        ↓
Deferred retained block
        ↓
safe completion fence
        ↓
BlockPool free queue
        ↓
Reusable by another request
```

因此 lifetime state 至少概念上包含：

```text
REQUEST_OWNED
        ↓
DETACHED_BUT_FENCED
        ↓
FREE / REUSABLE
```

---

# 31. 为什么仅 CUDA same-stream ordering 不够

same stream 可以保证：

```text
forward KV write
        ↓
gather
        ↓
scatter
```

但如果 CPU Scheduler 太早：

```text
BlockPool.free(B11)
```

另外 request 可能：

```text
allocate(B11)
```

而 GPU 原 compaction step 尚未完成。

于是：

```text
Request A GPU
    still using B11

Request B
    already overwrites B11
```

这是：

```text
use-after-free / ownership race
```

所以：

```text
GPU dependency ordering
```

与：

```text
allocator lifetime ordering
```

必须分别解决。

---

# 32. V2 复用现有 Safe-Free / Step Fence

P1 V1 已经验证：

```text
processed_step_seq
deferred_frees
```

以及：

```text
Scheduler.update_from_output()
```

位于真实 execution completion/result boundary 之后。

accepted V1 evidence 已证明：

- removed blocks 不会在 GPU step 仍 in-flight 时提前进入 reusable pool；
- safe completion 后 deferred frees 才被 drain；
- 后续 request 能真实复用 reclaimed block；
- 没有 premature reuse / double free。

因此 V2 MVP：

```text
DO NOT invent a new lifetime system
```

优先：

```text
REUSE existing step fence / deferred-free semantics
```

如果 implementation audit 证明 post-forward token compaction 的结果 publication 与当前 V1 fence 语义不能直接对应，再提出最小 ADAPT。

未经 source contradiction，不允许先引入：

```text
per-compaction custom CUDA event allocator
new background free thread
independent lifetime manager
```

---

# 33. Complete V2 Transaction

最终 approved transaction：

```text
Scheduler / control plane

Explicit Plan Producer
        ↓
PreparedCompactionPlan
        ↓
schedule-time validation/materialization
        ↓
CompactionPlanData
        ↓
SchedulerOutput
        ↓

================ Worker =================

GPUModelRunner.execute_model()
        ↓
prepare inputs / attention
        ↓
model.forward()
        ↓
all current-step K/V writes issued
        ↓
POST-FORWARD V2 HOOK
        ↓
validate source fence
        ↓
gather paged KV → scratch
        ↓
scatter scratch → current page prefix
        ↓
Worker BlockTable = old prefix
        ↓
Worker effective_kv_len = K
        ↓
publish CompactionResultData
        ↓
ExecuteModelState publication
        ↓
sample_tokens()
        ↓
ModelRunnerOutput
        ↓

================ Scheduler =================

receive successful step result
        ↓
validate CompactionResultData
against canonical req_to_blocks
        ↓
derive:
retained = canonical prefix
removed  = canonical suffix
        ↓
canonical req_to_blocks commit
        ↓
request.effective_kv_len = K
        ↓
removed blocks become deferred-retained
        ↓
safe step fence satisfied
        ↓
BlockPool.free_blocks()
        ↓
future request can reuse blocks
```

---

# 34. End-to-End Example

参数：

```text
block_size = 4

logical progress L = 10

Worker source row:
[B7, B2, B11]

Scheduler canonical row:
[B7, B2, B11]

E_source = 10

keep_member_indices:
[0, 2, 5, 8, 9]
```

Worker：

```text
K = 5

new_num_blocks = ceil(5 / 4) = 2
```

Data movement：

```text
source:
B7   = m0 m1 m2 m3
B2   = m4 m5 m6 m7
B11  = m8 m9 _  _

gather:
[m0, m2, m5, m8, m9]

scatter:
B7   = m0 m2 m5 m8
B2   = m9 _  _  _
```

Worker physical state：

```text
BlockTable:
[B7, B2]

E:
5
```

Worker result：

```text
source_E          = 10
expected_blocks   = 3
new_E             = 5
new_num_blocks    = 2
```

Scheduler：

```text
canonical old:
[B7, B2, B11]

validate expected old blocks = 3

retained:
[B7, B2]

removed:
[B11]

canonical new:
[B7, B2]

request.E:
5
```

Lifetime：

```text
B11
    ↓
detached
    ↓
step-fenced
    ↓
safe completion
    ↓
BlockPool
    ↓
future request reuse
```

---

# 35. V1 → V2 Reuse Matrix

| Component / Invariant | V2 Decision | Explanation |
|---|---|---|
| `num_computed_tokens` logical truth | `REUSE_AS_IS` | 不因 KV compaction 改 logical progress |
| logical `positions` | `REUSE_AS_IS` | RoPE/model coordinate |
| physical `cache_positions` | `REUSE_AS_IS` | future append 继续从 physical E |
| Worker `effective_kv_len` | `REUSE_SEMANTICS / ADAPT_UPDATE_TIMING` | V2 在 post-forward 将 E 直接替换为 K |
| `effective_kv_seq_lens` | `REUSE_AS_IS` | 下一 forward FA2 读取 physical extent |
| `allocate_slots()` physical allocation base | `REUSE_AS_IS` | Scheduler committed E 驱动 future capacity |
| Scheduler canonical ownership | `REUSE_AS_IS` | 仍是 authority |
| BlockPool | `REUSE_AS_IS` | 真正 reusable block pool |
| fail-before-mutation principle | `REUSE_AS_IS` | V2 plan/data/state commit 均须 fail-closed |
| deferred-free / step fence | `REUSE_AS_IS / VERIFY MINIMAL ADAPT` | 优先继承 V1 lifetime closure |
| V1 `ReclaimTransitionData` | `V1_ONLY` | pre-forward command semantics |
| `_commit_reclaim_transition()` | `V1_ONLY` | whole-block physical metadata transition |
| block-aligned physical shrink checks | `V1_ONLY` | 与 arbitrary token compaction 冲突 |
| `L - E` block alignment invariant | `REMOVE FROM V2 PATH` | V2 可产生任意 token divergence |
| token payload gather | `V2_NEW` | data-plane capability |
| scratch buffer | `V2_NEW` | correctness baseline |
| token payload scatter/writeback | `V2_NEW` | data-plane capability |
| `CompactionPlanData` semantic contract | `V2_NEW` | Scheduler→Worker |
| `CompactionResultData` semantic contract | `V2_NEW` | Worker→Scheduler |
| Scheduler canonical prefix-trim reconcile | `V2_NEW` | V2 ownership commit |

---

# 36. V1 Whole-Block Guard 与 V2 的明确冲突

当前 V1 包含：

```text
physical shrink
    ==
removed_whole_blocks * block_size
```

并 enforce：

```text
physical_shrink % block_size == 0
```

以及：

```text
(logical_num_computed_tokens - new_effective_kv_len)
    % block_size
    == 0
```

这些是正确的 **V1 contract**。

它们不能被解释为全项目 invariant。

V2 例如：

```text
L = 94
E_new = 51

L - E_new = 43
```

完全可以合法：

```text
43 % 16 != 0
```

所以实现时必须：

```text
keep V1 validation intact on V1 path

do not weaken V1 globally

introduce V2-specific validation path
```

禁止为了 V2：

```text
直接删除 V1 whole-block guard
然后让 V1/V2 共用一个模糊 helper
```

---

# 37. V2 后续 Oracle 应验证什么

经过 Runtime Architecture Freeze 后，PyTorch Oracle 的责任终于明确。

它不是 runtime integration。

它验证的是纯 data-plane transformation：

```text
Input:
    paged KV tensor(s)
    active source block row
    E_source
    keep_member_indices
    block_size / layout metadata

Output:
    expected compacted paged prefix
    E_new = K
    new_num_blocks = ceil(K / block_size)
```

必须覆盖：

```text
1. keep-all no-op
2. arbitrary interior removals
3. cross-page gather
4. destination crosses page
5. partial destination tail
6. source/destination overlap case
7. K exactly block aligned
8. K not block aligned
9. invalid unsorted keep
10. duplicate keep
11. out-of-range keep
12. stale source length validation at integration layer
```

Oracle 不负责：

```text
Scheduler transport
BlockPool free
CUDA fence
policy quality
```

---

# 38. 后续 Triton Primitive Decomposition

设计冻结后，M4 可以自然拆成：

```text
P1-M4-T1
PyTorch token compaction oracle
        ↓
P1-M4-T2
Triton paged gather
        ↓
P1-M4-T3
scatter/writeback + primitive correctness gate
```

其中 correctness baseline：

```text
gather to scratch
        ↓
scatter to prefix
```

先证明：

```text
bitwise / tolerance equivalence to PyTorch oracle
```

再讨论：

```text
fusion
scratch reuse
vectorization
memory transaction optimization
```

---

# 39. Runtime Integration Slice 的准确职责

后续 M5 Runtime Integration 不应该重新设计 architecture。

它只需要将已批准 contract 接上：

```text
Scheduler plan transport
        ↓
GPUModelRunner exact hook
        ↓
all-layer primitive invocation
        ↓
Worker row/E commit
        ↓
CompactionResultData
        ↓
Scheduler canonical prefix reconcile
        ↓
deferred safe-free
```

并验证真实生命周期：

```text
Request A:
prefill
→ compaction
→ physical E shrinks
→ generation continues

Scheduler:
canonical row shrinks
→ free count grows only after safe fence

Request B:
reuses released physical block
```

---

# 40. Invariants

以下作为 V2 Core architecture invariants。

## I-V2-01 — Logical truth preserved

```text
num_computed_tokens
```

仍表示 model logical progress。

KV compaction 不减少 logical history。

## I-V2-02 — Physical frontier independent

```text
effective_kv_len
```

表示当前真实可见 physical KV member count。

V2 后：

```text
E_new = len(keep_member_indices)
```

## I-V2-03 — Logical positions never renumbered

Retained historical KV 保留原 logical positional semantics。

## I-V2-04 — Keep plan indexes current physical member sequence

不能混用 logical positions / slots / block IDs。

## I-V2-05 — Keep order preserved

Core validate but never silently reorder selection。

## I-V2-06 — Current forward semantics unaffected

Compaction 发生在 model forward 之后。

## I-V2-07 — Data-plane precedes ownership release

```text
gather/scatter
before
canonical trailing-page release
```

## I-V2-08 — Destination uses current prefix pages

V2 compact 后：

```text
new Worker / Scheduler row
=
old row prefix
```

不是 arbitrary destination block selection。

## I-V2-09 — Worker BlockTable is not allocator authority

Scheduler canonical row remains ownership truth。

## I-V2-10 — E commits only after successful reconcile

Scheduler 不允许先写 E 再发现 ownership mismatch。

## I-V2-11 — Detached does not mean reusable

Trailing pages 在 safe fence 满足前不得重新分配。

## I-V2-12 — V1 block-alignment rules remain V1-only

不得弱化 V1 contract 来“兼容”V2。

## I-V2-13 — Core is policy-neutral

Compaction primitive 不依赖具体 importance algorithm。

## I-V2-14 — ExecuteModelState is post-compaction publication boundary

在 current MVP，V2 physical operation 成功后才发布 `ExecuteModelState`。

---

# 41. Failure Semantics

## Plan stale

如果：

```text
actual E_source
!=
expected E_source
```

则：

```text
reject
no mutation
```

## Invalid selection

如果：

```text
unsorted
duplicate
out of range
```

则：

```text
reject
no mutation
```

## Primitive failure before writeback

保持旧 Worker physical state。

## Primitive failure during destructive scatter

第一版实现必须把该风险限制在 controlled step failure semantics 下。

不能伪装成 recoverable success。

若未来要求 runtime recovery，需要单独设计：

```text
double buffer
rollback source
transactional destination
```

不属于当前 MVP。

## Scheduler reconcile mismatch

如果 Worker result 与 canonical source ownership 不匹配：

```text
fail closed

no canonical E commit
no free
```

不得“猜测”正确 row。

---

# 42. Why Separate `CompactionResultData`

最终决断：

```text
ReclaimTransitionData
=
V1 Scheduler→Worker command

CompactionResultData
=
V2 Worker→Scheduler result
```

两者不可硬合并的原因：

1. **方向不同**
2. **发生时间不同**
3. **E 的时间坐标不同**
4. **V1 需要 arbitrary retained block IDs**
5. **V2 只需要 canonical prefix length/shape**
6. **V1 没有 payload movement**
7. **V2 result 必须表达 post-forward physical state**
8. **混合后容易产生 `E_target` / `E_completed` 语义错误**

因此：

```text
APPROVED:
keep ReclaimTransitionData V1-specific
introduce a V2-specific result contract
```

---

# 43. Tangram — Reuse vs Non-Reuse

## Reuse as semantic reference

```text
post-forward compression
all-layer cache access from runner
gather/writeback semantics
payload mutation before trailing-page release
effective-length fold-in
worker→scheduler result transport
```

## Do not inherit

```text
compression ratio semantics
gate scorer
cross-layer threshold
per-layer/group effective lengths
ragged paging architecture
sliding-window special cases
Tangram scheduler free-by-worker-reported-ID authority
```

P1 V2 的目标不是“mini Tangram”。

---

# 44. Architecture Summary

最终 V2 可以用一句话定义：

> **P1 V2 在 vLLM MRV2 的 `GPUModelRunner.execute_model()` 中，于当前 model forward 完成后、`ExecuteModelState` 发布前，消费由 control plane 显式传入且绑定当前 source state 的 `keep_member_indices`，通过 paged-KV → scratch → prefix writeback 完成 all-layer token-level physical compaction；Worker 将 BlockTable 和 `effective_kv_len` 提交到新的 physical shape，并通过独立 V2 result contract 通知 Scheduler。Scheduler 以自己的 canonical ownership 为 authority，仅按新 block count 截断 canonical prefix，并在 successful reconcile 后提交 `request.effective_kv_len`；trailing blocks 经过已有 step-fence/deferred-free lifetime 后才返回 BlockPool，从而完成真正安全的 physical reuse。**

---

# 45. Approved Design Record

## Decision 1 — Runtime Hook

```text
APPROVED

POST-FORWARD
PRE-MODEL-OUTPUT-POSTPROCESS
PRE-ExecuteModelState-PUBLICATION
PRE-SAMPLE
```

## Decision 2 — Keep Plan Ownership

```text
APPROVED

V2 Core is policy-neutral.

keep_member_indices:
external explicit input

binding:
Scheduler/control plane

transport:
SchedulerOutput

consumer:
GPUModelRunner post-forward hook
```

## Decision 3 — Commit Protocol

```text
APPROVED

validate
→ payload gather
→ scratch
→ payload scatter
→ Worker prefix row / E commit
→ V2 result
→ Scheduler canonical prefix reconcile
→ Scheduler E commit
→ deferred safe-free
→ BlockPool reuse
```

## Contract Split

```text
APPROVED

ReclaimTransitionData:
V1-only

V2:
separate plan/result semantics
```

---

# 46. What Is Frozen vs Still Adjustable

## Frozen

```text
post-forward insertion semantics
pre-ExecuteModelState publication
policy-neutral Core
keep_member_indices semantics
strict keep ordering
source-state fence requirement
scratch correctness baseline
destination physical prefix
Worker vs Scheduler authority split
V1/V2 E lifecycle distinction
separate V2 result contract
canonical prefix-trim reconciliation
E commit after reconcile
safe-free after fence
V1 block-alignment guards remain V1-only
```

## Still adjustable during implementation review

```text
exact Python class names
exact dataclass field names
exact helper function names
scratch allocation owner
scratch lifetime / reuse strategy
PyTorch Oracle file placement
Triton launch shape
vector width
whether Worker result contains debug candidate freed IDs
exact reuse of existing deferred-free helper
instrumentation / NVTX naming
```

任何 adjustable implementation 选择不得违反上面的 frozen semantics。

---

# 47. Next Allowed Work

Runtime architecture 已冻结。

下一个合法阶段：

```text
P1-M4-T1
PyTorch Token-Level Compaction Oracle
```

但进入实现前，应该先写一个非常小的 implementation contract：

```text
oracle input tensor/layout
block row representation
E_source
keep_member_indices
expected compacted output
```

然后才允许 Codex 实现。

当前禁止：

```text
直接写 retention policy
直接写 top-k scoring
直接集成 Tangram compressor
直接改 Scheduler ownership authority
直接删除 V1 block-alignment validation
直接进入 optimized in-place Triton
直接添加 CUDA synchronize 作为生命周期修复
```

---

# 48. Source / Evidence Index

Primary local evidence:

```text
04-experiments/project1_kv_reclaim/raw/
P1-V2-RUNTIME-SEAM-WALKTHROUGH.log
```

P1 V1 accepted evidence includes:

```text
P1-V1-CORE-GATE-REVIEW-01.md
P1-M1-T3 integration closure
V1 worker state / ownership / reuse tests
```

Pinned source identities:

```text
vLLM v0.26.0:
568afb3a13806beb53bb2e6bd518269357b237c0

P1 V1 accepted:
cd444b72ceecb2496bf015baf8c18bf883151fa1

P1 V2 observed worktree HEAD:
064f4efe082bc23087043381a4a401c170997f5c

Tangram audited:
8e1cbfa5cf82acc67fe1880df0c632f2a6485e35
```

---

# 49. Final Status

```text
P1 V2 Source Walkthrough
=
SUFFICIENT FOR CORE RUNTIME ARCHITECTURE

Decision 1
=
APPROVED

Decision 2
=
APPROVED

Decision 3
=
APPROVED

V2 Runtime Architecture
=
FROZEN FOR MVP

Next
=
P1-M4-T1
PyTorch Token-Level Compaction Oracle
```
