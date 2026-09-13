# P1 Learning — Source Reading, Runtime Trace & Implementation Deep Dives

> Project: **Physical KV Cache Reclamation for vLLM**  
> Scope: vLLM v0.26 / Model Runner V2 / Physical KV Reclamation / Token-Level KV Compaction  
> Purpose: 保存 P1 实现过程中形成的源码学习、运行时链路追踪、机制理解、测试分析与实现复盘。  
> This directory is **not** the canonical design contract and **not** the execution evidence namespace.

---

# 1. 这个目录是做什么的

`learning/` 保存的是：

```text
为了真正理解和实现 P1，
对 vLLM 源码、runtime state、KV addressing、
Scheduler / Worker transaction、
Triton compaction、测试与 E2E
做的深度分析与学习笔记。
```

它解决的是：

```text
“为什么系统是这样工作的？”
“这条源码链到底怎么走？”
“某个 state / index / block / position 到底表示什么？”
“P1 的改动具体插在哪里？”
“测试到底证明了什么？”
```

而不是：

```text
“项目正式 contract 是什么？”
“这一 Slice 实际改了哪些代码？”
“某次测试真实跑出了什么结果？”
```

---

# 2. 与其他目录的职责边界

整个 P1 workspace 应该按照下面的职责理解。

```text
01-projects/project1-physical-kv-reclamation-vllm/
│
├── docs/
│   = Canonical Project Documentation
│
├── learning/
│   = Source Reading / Runtime Deep Dive / Mechanism Understanding
│
└── ...
```

以及：

```text
04-experiments/project1_kv_reclaim/
=
Actual Execution Records / Scripts / Tests / Smoke / Evidence
```

更明确地说：

| 目录 | 负责什么 | 不负责什么 |
|---|---|---|
| `docs/` | 项目正式设计、Scope、Contract、Evaluation、Delivery、Reference | 不保存大量教学型源码分析 |
| `learning/` | 源码导读、运行时机制、完整链路、测试理解 | 不作为唯一 canonical truth |
| `04-experiments/project1_kv_reclaim/` | 实际代码变更记录、脚本、测试、E2E、Evidence、Handoff | 不承担长期知识整理 |
| `00-docs/source-reading/vllm/` | 与 P1 无关也成立的通用 vLLM 源码知识 | 不保存 P1-specific design |

因此：

```text
通用 vLLM 知识
→ 00-docs/source-reading/vllm/

P1 正式设计
→ P1/docs/

P1 深入源码理解
→ P1/learning/

P1 实际执行证据
→ 04-experiments/project1_kv_reclaim/
```

---

# 3. Truth / Authority Priority

如果不同文档之间出现冲突，按照以下优先级判断：

```text
Current Source
    >
PROJECT_STATE.md
    >
docs/ Canonical Design
    >
Accepted Experiment Evidence
    >
learning/ Notes
    >
Historical Records
```

即：

```text
learning/
允许保留探索过程、历史理解和教学解释，

但不能覆盖更新后的 source fact
或正式冻结的 project contract。
```

---

# 4. 当前 Learning 信息架构

```text
learning/
│
├── README.md
│
├── 00-foundations/
│
├── 10-v1-whole-block/
│
├── 20-v2-runtime/
│
├── 30-v2-compaction/
│
└── 40-testing-e2e/
```

推荐按：

```text
Foundations
    ↓
V1 Whole-Block
    ↓
V2 Runtime
    ↓
V2 Compaction
    ↓
Testing / E2E
```

阅读。

---

# 5. `00-foundations/` — P1 的状态与坐标基础

这里解决 P1 V1 / V2 共同依赖的基础问题。

核心概念：

```text
num_computed_tokens
effective_kv_len
effective_kv_seq_lens

logical position
physical KV position

positions
cache_positions

batch_idx
req_state_idx

RequestState
InputBatch
BlockTables
KV payload

slot mapping
attention visible KV length
```

---

## 5.1 推荐阅读 1

`P1-M1-T1-S1-MRV2-num-computed-tokens与Effective-KV-State生命周期详解-v2.md`

重点理解：

```text
num_computed_tokens = L
= logical progress

effective_kv_len = E
= persistent physical KV frontier

effective_kv_seq_lens
= current-forward visible physical KV extent
```

P1 最核心的状态解耦：

```text
Normal:

L ≈ E

After Physical Reclamation / Compaction:

L != E
```

例如：

```text
L = 33
E = 5
```

表示：

```text
模型逻辑已经推进到第 33 个 token，
但当前真正保留下来的 physical KV dense prefix 只有 5 个 member。
```

---

## 5.2 推荐阅读 2

`P1-M1-T1-S2-S3-vLLM-Logical-Physical-KV-解耦源码与实现详解-v3.md`

重点理解：

```text
logical token position
!=
physical KV write position
```

例如：

```text
L = 33
E = 5
```

下一步：

```text
RoPE / logical position = 33
physical cache position = 5
```

也就是：

```text
positions
继续使用 logical progress

cache_positions
从 compact / reclaim 后的 physical frontier 继续
```

这是 P1 所有 physical reclamation 能成立的前提。

---

# 6. 必须掌握的 MRV2 Index Domains

P1 中最容易混淆的是以下几个索引域。

---

## 6.1 `batch_idx`

属于：

```text
当前 InputBatch
```

例如：

```text
InputBatch:
["C", "A", "E"]

batch_idx:
0 → C
1 → A
2 → E
```

当前 step 的：

```text
effective_kv_seq_lens
override_valid
override_value
```

通常是 batch-indexed。

---

## 6.2 `req_state_idx`

属于：

```text
persistent RequestState / BlockTables
```

例如：

```text
RequestState:

slot 0 → B
slot 1 → E
slot 2 → empty
slot 3 → C
slot 4 → D
slot 5 → A
```

则：

```text
A req_state_idx = 5
```

---

## 6.3 `idx_mapping`

负责：

```text
batch_idx
→
req_state_idx
```

例如：

```text
InputBatch:
C, A, E

RequestState:
C→3
A→5
E→1

idx_mapping = [3,5,1]
```

---

## 6.4 BlockTables

BlockTables 使用：

```text
req_state_idx
```

进行 persistent addressing。

例如：

```text
row = [5,1,3]
```

表示：

```text
request-local page 0 → physical block 5
request-local page 1 → physical block 1
request-local page 2 → physical block 3
```

必须注意：

```text
block ID
!=
token/member index
```

---

# 7. `10-v1-whole-block/` — V1 Whole-Block Physical Reclamation

V1 的目标：

> 在 request 已经发生 logical / physical divergence 后，对整块 paged KV ownership 做安全的 physical reclaim。

推荐阅读顺序：

1. `P1-M1-T2-Whole-Block-Reclaim-vLLM-源码分析-问题定位-R1实现详解-v2.md`
2. `P1-M1-T3-Physical-Ownership-Reconciliation-Deep-Dive-v3-Chain-Design-Test.md`

---

# 8. V1 的主链

可以记成：

```text
Logical / Physical State Split
        ↓
Worker Physical View Transition
        ↓
Worker → Scheduler Result / ACK
        ↓
Scheduler Canonical Ownership Reconciliation
        ↓
Ownership Detach
        ↓
BlockPool Free
        ↓
Physical Block Reuse
```

V1 建立了 V2 继续使用的基础：

```text
effective_kv_len
physical-first allocation
Scheduler canonical ownership authority
Worker execution view
result-time reconciliation
safe release / reuse
```

因此：

```text
V2
不是重新做一套系统，

而是在 V1 state / ownership substrate 上
增加 token/member-level payload movement。
```

---

# 9. `20-v2-runtime/` — V2 Runtime Transaction

这里主要回答：

> 一个 V2 compaction request 如何从 Scheduler 进入 Worker，又如何从 Worker 回到 Scheduler？

推荐阅读：

1. `P1-V2-M5-T1-Runtime-Contract-Transport-Deep-Dive-Final.md`
2. `P1-M5-T2A-T2B-Implementation-Runtime-and-CPU-GPU-Deep-Dive.md`
3. `P1-M5-T2B-Design-Decision-and-Runtime-Lifecycle-Notes.md`
4. `P1-V2-EngineCore-to-Scheduler-Complete-Source-Walkthrough.md`

---

# 10. V2 Runtime 总事务

```text
Scheduler
│
├── CompactionPlanData
│
└── SchedulerOutput.compaction_plans
        ↓
Worker
│
├── _prepare_v2_compactions()
│
├── _PreparedCompaction
│
├── real model forward
│
├── _execute_v2_compactions()
│
├── payload compaction
│
├── BlockTable staged update
│
├── E override
│
└── CompactionResultData
        ↓
ModelRunnerOutput
        ↓
Scheduler.update_from_output()
        ↓
Canonical Reconciliation
```

---

# 11. V2 的四类 Carrier

## 11.1 `CompactionPlanData`

```text
Scheduler → Worker
```

表示：

```text
“这一 step 我要你 compact 什么”
```

---

## 11.2 `_PreparedCompaction`

```text
Worker internal
```

表示：

```text
“我已经基于当前 runtime state
验证并解析出了如何执行”
```

---

## 11.3 `CompactionResultData`

```text
Worker → Scheduler
```

表示：

```text
“Worker 实际完成了这个 physical transition”
```

---

## 11.4 `_PreparedCompactionReconciliation`

```text
Scheduler internal
```

表示：

```text
“Scheduler 已经核验 Plan / Result /
Request / canonical ownership，
允许正式 commit”
```

---

# 12. 为什么 Plan 与 Result 不能合并

因为：

```text
Plan
= command / expectation

Result
= execution receipt
```

Scheduler 不能直接假设：

```text
“Plan 发了”
=
“Worker 一定正确完成了”
```

因此必须有：

```text
Plan
    ↓
Worker execution
    ↓
Result
    ↓
Scheduler validation
    ↓
Commit
```

这也是 V2 transaction hardening 的核心。

---

# 13. `30-v2-compaction/` — V2 核心数据面与 Physical Reconciliation

这是整个 P1 V2 最核心的学习目录。

推荐阅读顺序：

1. `P1-V2-M4-Token-Level-Paged-KV-Compaction-Deep-Dive.md`
2. `P1-M5-T3-Scheduler-Canonical-Reconciliation-Frozen-Design.md`
3. `P1-M5-T4-Physical-Block-Release-Reuse-Final-Deep-Dive.md`
4. `P1-V2-Physical-KV-Compaction-Source-Trace-and-E2E-Deep-Dive.md`

---

# 14. V2 Payload Compaction

核心问题：

> 一个 paged KV 中稀疏保留的 member，如何重新压成 dense physical prefix？

例如：

```text
source_E = 33

keep =
[0,8,16,24,31]

K = 5
```

目标：

```text
old member 0  → new physical 0
old member 8  → new physical 1
old member 16 → new physical 2
old member 24 → new physical 3
old member 31 → new physical 4
```

最终：

```text
E = 5
```

---

# 15. KV Physical Address

假设：

```text
block_size = 16
BlockTable row = [5,1,3]
```

member：

```text
24
```

映射：

```text
local_page
=
24 // 16
=
1

offset
=
24 % 16
=
8

physical block
=
block_ids[1]
=
1
```

最终：

```text
kv_cache[1, head, 8, :]
```

因此：

```text
keep_member_index
```

表示：

```text
current request-local physical member index
```

不是：

```text
logical token ID
physical block ID
slot ID
```

---

# 16. 为什么是 Gather → Scratch → Writeback

不能直接原地：

```text
source → destination
```

因为：

```text
source / destination physical regions
可能重叠
```

前面的 writeback 可能覆盖：

```text
后面还没有 gather 的 source
```

因此：

```text
Paged Source
    ↓
Gather
    ↓
Independent Scratch
    ↓
Dense-Prefix Writeback
```

当前 M4 是：

```text
correctness-first
generic stable compaction
```

---

# 17. V2 Writeback 后的 Physical Geometry

例如：

```text
source_E = 33
block_size = 16

source blocks = 3
```

keep：

```text
K = 5
```

则：

```text
new_num_blocks
=
ceil(5 / 16)
=
1
```

如果旧 block row：

```text
[5,1,3]
```

则 compact 后 retained physical page 是：

```text
[5]
```

注意：

```text
retained physical blocks
=
block_ids[:new_num_blocks]
```

不是：

```text
“保留下来的 token 原来在哪些 blocks”
```

因为 payload 已经被搬回 prefix pages。

---

# 18. Worker State Commit

payload 成功以后：

```text
BlockTable
staged overwrite

E override
=
K

Result
=
CompactionResultData
```

例如：

```text
source_E = 33
K = 5

old block row:
[5,1,3]

Worker new execution row:
[5]
```

---

# 19. `post_update()`：Logical / Physical 的最终分叉

Normal path：

```text
L += delta
E += delta
```

V2：

```text
L += delta
E = K
```

例如：

```text
before:
L = 32
E = 32

q = 1
V2 compact K = 5

after:
L = 33
E = 5
```

这就是为什么 V2 不能只靠普通：

```text
E += computed_delta
```

而需要：

```text
absolute E override
```

---

# 20. Scheduler Canonical Reconciliation

Worker Result 回来以后：

```text
Scheduler.update_from_output()
```

需要核验：

```text
Result 是否有 matching Plan
request_id 是否一致
step_seq 是否一致
source E / source blocks 是否一致
request 是否仍 active
Scheduler 自己重算的 source E 是否一致
new geometry 是否满足 page invariant
canonical ownership row 是否仍处于 expected source shape
```

全部通过后才形成：

```text
_PreparedCompactionReconciliation
```

然后：

```text
commit
```

---

# 21. Scheduler 为什么不能直接相信 Worker

因为：

```text
Scheduler
=
canonical ownership authority
+
allocator authority

Worker
=
execution / payload authority
```

如果 Worker 返回：

```text
stale Result
wrong-step Result
duplicate Result
wrong source geometry
```

Scheduler 直接 free blocks 会造成：

```text
ownership corruption
block aliasing
unsafe reuse
```

因此必须 fail-closed。

---

# 22. Ownership Detach 与 Physical Free

假设：

```text
canonical row =
[5,1,3]

new_num_blocks =
1
```

Scheduler reconciliation：

```text
retained =
[5]

removed =
[1,3]

req_to_blocks[A]
=
[5]
```

这一步是：

```text
ownership detach
```

然后：

```text
BlockPool.free_blocks([1,3])
```

才是：

```text
physical allocator release
```

顺序必须是：

```text
detach
    ↓
free
```

不能反过来。

---

# 23. 当前同步 MVP 与 Deferred Free

当前 V2 functional baseline：

```text
async_scheduling = False
single in-flight execution regime
connector = off
```

因此 current result-time path 可以：

```text
ownership detach
→ direct BlockPool free
```

但未来 V3 async hardening 需要：

```text
detach
→ execution fence
→ deferred free
→ safe reuse
```

所以：

```text
current direct free
```

是明确受 support boundary 限制的。

---

# 24. `40-testing-e2e/` — 如何理解测试

这里保存：

```text
targeted tests
runtime contract tests
real-engine E2E interpretation
```

测试体系应按层理解：

```text
Kernel Tests
    ↓
Worker Tests
    ↓
Scheduler / Core Tests
    ↓
Real-Engine E2E
```

---

# 25. Kernel Test 证明什么

验证：

```text
Paged KV gather
scratch
writeback
geometry
overlap safety
```

不证明：

```text
Scheduler
ownership
allocator
真实 model forward
```

---

# 26. Worker Test 证明什么

验证：

```text
Plan
→ _PreparedCompaction

index domain
source fence
BlockTable
E override
CompactionResultData
failure-path behavior
```

---

# 27. Scheduler / Core Test 证明什么

验证：

```text
Plan transport
Result transport
Plan ↔ Result reconciliation
canonical row truncate
safe release
fail-closed
```

---

# 28. Real-Engine E2E 证明什么

Real-Engine smoke 不直接调用：

```text
_execute_v2_compactions()
```

而是只驱动：

```python
engine.step()
```

让真实 vLLM 自己完成：

```text
Scheduler.schedule()
        ↓
Model Runner
        ↓
Real Qwen Forward
        ↓
KV Write
        ↓
V2 Compaction
        ↓
Sampling / post_update
        ↓
ModelRunnerOutput
        ↓
Scheduler.update_from_output()
```

测试只在关键 seam：

```text
schedule()
update_from_output()
```

外面加 observer / trace。

---

# 29. E2E 中的 `trace` 是什么

`trace` 不是 profiler。

它只是测试代码维护的：

```python
dict
```

相当于：

```text
在关键 transaction seam 拍状态照片
```

例如：

```text
Schedule 后：

q
canonical row
E
L
free blocks
Plan
```

以及：

```text
Update 后：

Worker Result
new E
new blocks
canonical row
free blocks
```

---

# 30. Real-Engine E2E 的核心场景

固定：

```text
prompt_len = 32
block_size = 16
```

所以：

```text
32 tokens
=
2 pages
```

下一 decode：

```text
q = 1
```

变成：

```text
33 tokens
=
3 pages
```

因此：

```text
[1,2]
↓
append
↓
[1,2,3]
```

然后 V2：

```text
source_E = 33

keep =
[0,8,16,24,31]

K = 5
```

压成：

```text
3 pages
→
1 page
```

最终：

```text
[1,2,3]
→
[1]
```

---

# 31. E2E 当前证明的事实

当前 functional closure 验证：

```text
Scheduler Plan
成功进入当前 step

Worker
真实执行 model forward

post-forward source E
=
33

真实 paged KV compaction

33 members
→
5 members

3 pages
→
1 page

Worker Result
成功返回 Scheduler

Scheduler canonical ownership
成功 truncate

effective_kv_len
=
5

trailing blocks
成功回到 BlockPool

request
后续继续真实 generation 并 finish
```

---

# 32. E2E 没有证明什么

当前不能声称：

```text
keep=[0,8,16,24,31]
是优秀 retention policy

模型质量完全不下降

TPOT / TTFT 一定改善

async scheduling 已正确

prefix cache 已兼容

spec decode 已兼容

CUDA Graph 已兼容

next-step cache_position 已被直接观测为5
```

当前证明的是：

> **如果给定一个合法 physical keep set，V2 runtime mechanism 可以完成完整 physical transition，并继续真实 serving。**

---

# 33. 当前 V2 Functional Core 的完整主链

```text
Scheduler Policy / Test Producer
        ↓
CompactionPlanData
        ↓
Scheduler.schedule()
        ↓
SchedulerOutput.compaction_plans
        ↓
GPUModelRunner
        ↓
_prepare_v2_compactions()
        ↓
_PreparedCompaction
        ↓
Real Model Forward
        ↓
Current-Step KV Write
        ↓
Post-Forward V2
        ↓
Paged KV Gather
        ↓
Scratch
        ↓
Dense-Prefix Writeback
        ↓
Worker BlockTable Staged Update
        ↓
E Override
        ↓
CompactionResultData
        ↓
ModelRunnerOutput
        ↓
Scheduler.update_from_output()
        ↓
Plan / Result / Source Fence Validation
        ↓
_PreparedCompactionReconciliation
        ↓
Canonical Ownership Truncate
        ↓
Scheduler E = K
        ↓
Removed Blocks
        ↓
BlockPool.free_blocks()
        ↓
Next Scheduler Step
        ↓
Next Real Model Forward
```

---

# 34. 推荐完整阅读路线

如果从零开始理解 P1：

```text
Step 1
00-foundations/
理解 L / E / effective_kv_seq_lens

Step 2
00-foundations/
理解 positions / cache_positions / idx_mapping / BlockTables

Step 3
10-v1-whole-block/
理解 whole-block physical transition

Step 4
10-v1-whole-block/
理解 Scheduler ownership / Safe Free / Reuse

Step 5
20-v2-runtime/
理解 Plan / Prepared / Result / ExecuteModelState

Step 6
30-v2-compaction/
理解 Triton gather / scratch / writeback

Step 7
30-v2-compaction/
理解 Scheduler canonical reconciliation

Step 8
30-v2-compaction/
理解 physical release / reuse

Step 9
30-v2-compaction/
阅读完整 V2 Source Trace

Step 10
40-testing-e2e/
从测试反推整个 runtime contract
```

---

# 35. 当前最重要的几份 Learning 文档

## Foundations

```text
00-foundations/
├── P1-M1-T1-S1-MRV2-num-computed-tokens与Effective-KV-State生命周期详解-v2.md
└── P1-M1-T1-S2-S3-vLLM-Logical-Physical-KV-解耦源码与实现详解-v3.md
```

## V1

```text
10-v1-whole-block/
├── P1-M1-T2-Whole-Block-Reclaim-vLLM-源码分析-问题定位-R1实现详解-v2.md
└── P1-M1-T3-Physical-Ownership-Reconciliation-Deep-Dive-v3-Chain-Design-Test.md
```

## V2 Runtime

```text
20-v2-runtime/
├── P1-V2-M5-T1-Runtime-Contract-Transport-Deep-Dive-Final.md
├── P1-M5-T2A-T2B-Implementation-Runtime-and-CPU-GPU-Deep-Dive.md
├── P1-M5-T2B-Design-Decision-and-Runtime-Lifecycle-Notes.md
└── P1-V2-EngineCore-to-Scheduler-Complete-Source-Walkthrough.md
```

## V2 Compaction

```text
30-v2-compaction/
├── P1-V2-M4-Token-Level-Paged-KV-Compaction-Deep-Dive.md
├── P1-M5-T3-Scheduler-Canonical-Reconciliation-Frozen-Design.md
├── P1-M5-T4-Physical-Block-Release-Reuse-Final-Deep-Dive.md
└── P1-V2-Physical-KV-Compaction-Source-Trace-and-E2E-Deep-Dive.md
```

## Testing

```text
40-testing-e2e/
└── P1-S1-S2-S3-targeted-tests-源码逻辑与语法详解.md
```

---

# 36. Implementation Records 不放 Learning

以下类型不应该继续放在 `learning/`：

```text
IMPLEMENTATION-RECORD
CODE-CHANGE-RECORD
SOURCE-AUDIT
TEST-SUMMARY
FREEZE
GATE-REVIEW
E2E-CLOSURE
HANDOFF
```

这些应进入：

```text
04-experiments/project1_kv_reclaim/
```

推荐：

```text
records/
├── m1/
├── m4/
├── m5/
├── design-history/
└── closure/
```

---

# 37. 为什么 Learning 和 Records 必须分开

例如一份学习笔记可以说：

```text
“为什么 effective_kv_len 必须是 persistent physical frontier”
```

而 implementation record 应回答：

```text
“本 Slice 修改了 states.py / input_batch.py 哪几行，
跑了哪些测试，
结果是什么”
```

二者混在一起以后：

```text
GitHub reviewer
无法区分：
知识解释
vs
真实工程 evidence
```

所以：

```text
learning
=
Understand

records
=
Prove
```

---

# 38. P1 当前最值得对外展示的学习链

如果只选几份给面试官或 reviewer：

```text
1.
P1-M1-T1-S2-S3-vLLM-Logical-Physical-KV-解耦源码与实现详解-v3.md

2.
P1-M1-T3-Physical-Ownership-Reconciliation-Deep-Dive-v3-Chain-Design-Test.md

3.
P1-V2-M4-Token-Level-Paged-KV-Compaction-Deep-Dive.md

4.
P1-V2-Physical-KV-Compaction-Source-Trace-and-E2E-Deep-Dive.md
```

分别代表：

```text
State / Address Semantics

Ownership / Reclaim

Kernel / Data Movement

End-to-End Runtime Closure
```

---

# 39. 一句话总结 P1 Learning

```text
P1 learning/
不是“项目过程中的杂记”。

它是一套从：
vLLM runtime foundation
→ physical KV state
→ ownership
→ token-level compaction
→ allocator reclaim
→ real-engine E2E

逐层建立起来的源码学习与系统理解地图。
```

---

# 40. Current State

当前 P1 已经形成：

```text
V1 Functional Core
= CLOSED

V2 Functional Core
= CLOSED

Physical Release / Reuse
= CLOSED

Real-Engine E2E
= PASS

Next
= V3 Runtime Hardening
```

因此当前 `learning/` 的主要任务已经从：

```text
继续探索 V2
```

转为：

```text
冻结知识结构
补齐导航
整理 evidence boundary
为 V3 建立清晰入口
```

后续新文档必须优先判断属于：

```text
通用源码知识
P1 canonical design
P1 learning
P1 execution evidence
```

再决定落盘位置，避免重新回到平铺和混放状态。
