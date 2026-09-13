# P1 V2 Physical KV Compaction — 源码追踪、状态语义、事务链路与 Real-Engine E2E 全景笔记

> 项目：Physical KV Cache Reclamation for vLLM  
> 版本基线：vLLM v0.26.0  
> 主要实现分支：`p1/v2-token-compaction-v026`  
> 目标：完整记录 V2 token/member-level physical KV compaction 从 Scheduler 计划、Worker 执行、Triton payload 搬运、Worker 状态提交、Scheduler canonical ownership reconciliation、BlockPool 释放，到下一步真实推理继续执行的全链路。  
> 定位：源码学习 + 项目实现审计 + 简历项目技术说明底稿。  
> 本文只描述当前 synchronous / single-GPU / single-KV-group / eager MVP 已验证路径；不扩张到尚未验证的 async、PP>1、prefix cache、spec decode、KV connector、CUDA Graph 等模式。

---

# 0. Executive Summary

V2 要解决的核心问题不是“删掉几个 token”这么简单，而是：

> **在 vLLM Paged KV 架构中，允许一个 request 的 logical token progress 与 physical KV occupancy 分离；在某一步 forward 完成后，把当前稀疏保留的 KV member 稳定压缩成新的 dense physical prefix，并同步更新 Worker execution state、Scheduler canonical ownership 和 allocator capacity，使下一步真实推理继续从新的 physical frontier 追加。**

整个 V2 transaction 可以压缩成：

```text
Policy / Test Producer
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
真实 model forward 已完成
    ↓
_execute_v2_compactions()
    ↓
compact_paged_kv_triton_2d()
    ↓
gather → scratch → dense-prefix writeback
    ↓
Worker BlockTable staged overwrite
    ↓
effective_kv_len absolute override: E := K
    ↓
CompactionResultData
    ↓
ModelRunnerOutput
    ↓
Scheduler.update_from_output()
    ↓
_validate_compaction_results()
    ↓
_prepare_compaction_reconciliations()
    ↓
_PreparedCompactionReconciliation
    ↓
_commit_compaction_reconciliations()
    ↓
canonical req_to_blocks truncate
    ↓
request.effective_kv_len = K
    ↓
removed KVCacheBlock[]
    ↓
BlockPool.free_blocks()
    ↓
后续 schedule / model forward 继续执行
```

V2 的核心不是单个 kernel，而是 **physical state transition 的端到端闭环**。

---

# 1. 当前项目状态与支持边界

当前 functional closure：

```text
P1 V1 Functional Core = CLOSED
P1 V2 Functional Core = CLOSED
T4 Physical Release / Reuse = CLOSED
Real-Engine Functional E2E = PASS
DESIGN_CONFLICT = NONE
NEXT = V3 Runtime Hardening
```

Real-Engine E2E 的运行约束：

```text
GPU                  = NVIDIA A100 80GB PCIe
model                = Qwen3-0.6B
dtype                = BF16
model runner         = V2
block_size           = 16
max_model_len        = 128
max_num_seqs         = 1
max_num_batched_tokens = 32
enforce_eager        = True
async_scheduling     = False
PP / TP / DP / DCP   = 1
max concurrent batch = 1（当前实验约束）
KV connector         = off
prefix caching       = off
spec decode          = off
CUDA Graph           = off
single KV cache group
```

当前没有验证：

```text
async scheduling
PP > 1
KV connector / offload
prefix caching
speculative decoding
CUDA Graph
multi KV group
正式性能 benchmark
复杂 repeated-compaction lifecycle
```

因此当前结论是：

> V2 functional mechanism 在明确受限的同步执行 regime 下闭环，不等价于“已支持所有 vLLM runtime mode”。

---

# 2. V2 首先引入的关键抽象：Logical Progress 与 Physical Progress 解耦

上游普通执行通常隐含：

```text
logical progress ≈ physical KV progress
```

即：

```text
num_computed_tokens ≈ 当前 KV 中的有效 token 数
```

V2 必须允许：

```text
logical progress != physical KV occupancy
```

因此引入：

```text
L = num_computed_tokens
E = effective_kv_len
```

其中：

```text
L
= 模型逻辑上已经推进到哪里
= logical / RoPE position progress

E
= 当前真正持久存在的 physical KV dense-prefix 长度
= 下一次 KV append 的 physical frontier
```

典型 V2 状态：

```text
compaction 前：
L = 33
E = 33

keep 5 个 KV member 后：

L = 33
E = 5
```

下一步：

```text
logical position = 33
physical cache position = 5
```

即：

```text
RoPE / logical token numbering 不改变
KV physical address 被重新压紧
```

---

# 3. 三个必须分开的长度状态

## 3.1 `num_computed_tokens`

Persistent logical state：

```text
num_computed_tokens = L
```

语义：

> request 在模型语义上已经计算了多少 token。

V2 compaction 不应该把它改成 K。

---

## 3.2 `effective_kv_len`

Persistent physical state：

```text
effective_kv_len = E
```

语义：

> 当前 persistent physical KV dense prefix 的长度。

在普通执行中：

```text
E += computed_delta
```

在 V2 compaction step 中：

```text
E = K
```

这是 absolute replacement，不是 additive update。

---

## 3.3 `effective_kv_seq_lens`

这是当前 forward 的 batch-local 可见长度。

概念上：

```text
effective_kv_seq_lens[batch_idx]
=
persistent E
+
query_len_this_step
```

例如：

```text
step start:
E = 32

本轮 q = 1

forward-visible source:
effective_kv_seq_lens = 33
```

V2 是 **post-forward compaction**，因此它的 source E 应该来自：

```text
input_batch.effective_kv_seq_lens[batch_idx]
```

而不是 step-start persistent：

```text
req_states.effective_kv_len[req_state_idx]
```

---

# 4. `positions` 与 `cache_positions`：逻辑位置与物理写位置必须分离

V2 后：

```text
logical token position
!=
physical KV write position
```

因此当前输入准备语义：

```text
position
=
num_computed_tokens + local_offset

cache_position
=
effective_kv_len + local_offset
```

例如：

```text
L = 33
E = 5
query_len = 1
```

下一 token：

```text
position      = 33
cache_position = 5
```

这保证：

```text
RoPE 仍然看到原始逻辑序列位置
KV write 则写到 compact 后的新 physical frontier
```

这是整个 physical compaction 能成立的根本。

---

# 5. MRV2 中四类对象和三个索引域

理解 V2 最容易卡住的地方之一是：

```text
RequestState
InputBatch
BlockTables
KV cache payload
```

同时存在，并且索引域不同。

---

## 5.1 RequestState：persistent request table

文件：

```text
vllm/v1/worker/gpu/states.py
```

它不是“一个 request 对象”，而是固定容量 persistent table。

核心 mapping：

```python
req_id_to_index: dict[str, int]
index_to_req_id: dict[int, str]
```

其中 persistent row index：

```text
req_state_idx
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
A 的 req_state_idx = 5
```

`effective_kv_len`、`num_computed_tokens` 等 persistent tensors 都使用：

```text
req_state_idx
```

索引。

---

## 5.2 InputBatch：当前 forward 的临时 batch

文件：

```text
vllm/v1/worker/gpu/input_batch.py
```

例如：

```python
input_batch.req_ids = ["C", "A", "E"]
```

则：

```text
batch_idx 0 → C
batch_idx 1 → A
batch_idx 2 → E
```

当前 forward-local arrays：

```text
effective_kv_seq_lens
query_start_loc
override_valid
override_value
```

主要使用：

```text
batch_idx
```

---

## 5.3 `idx_mapping`

它桥接：

```text
batch_idx
→
req_state_idx
```

例如：

```text
InputBatch:
["C", "A", "E"]

persistent RequestState:
C → 3
A → 5
E → 1

所以：

idx_mapping = [3,5,1]
```

概念：

```python
idx_mapping[batch_idx] = req_state_idx
```

---

## 5.4 BlockTables

Worker `self.block_tables` 是 persistent execution addressing state。

它也按：

```text
req_state_idx
```

索引。

例如：

```text
A req_state_idx = 5
row = [5,1,3]
```

语义：

```text
request-local page 0 → physical block 5
request-local page 1 → physical block 1
request-local page 2 → physical block 3
```

这里的：

```text
[5,1,3]
```

不是 token index，而是 physical KV block IDs。

---

## 5.5 KV cache payload

逻辑 shape：

```text
[B, H, N, C]
```

其中：

```text
B = physical block pool size
H = KV heads
N = block_size
C = per-head content dimension
```

地址：

```text
kv_cache[
    physical_block_id,
    head,
    offset_inside_block,
    channel
]
```

---

## 5.6 一条完整地址链

例如：

```text
block_size = 16
member = 24
BlockTable row = [5,1,3]
```

则：

```text
member 24
↓
local_page = 24 // 16 = 1
offset     = 24 % 16  = 8
↓
physical block = BlockTable[1] = 1
↓
kv_cache[1, head, 8, :]
```

必须牢牢记：

```text
keep_member_index
!=
physical block ID
```

---

# 6. V2 Plan：CompactionPlanData

V2 当前没有内置 retention policy。

Plan 通过显式 producer / test hook 注入：

```python
_set_prepared_compaction_plan(
    request_id,
    keep_member_indices,
    expected_source_effective_kv_len,
    expected_source_num_blocks,
    step_seq,
)
```

内部形成：

```python
CompactionPlanData(
    request_id,
    keep_member_indices,
    expected_source_effective_kv_len,
    expected_source_num_blocks,
    step_seq,
)
```

语义：

```text
Scheduler → Worker：

“下一次这个 request 的 post-forward physical source
应该是 E=X / blocks=Y，
请只保留这些 current physical member indices。”
```

注意：

```text
keep_member_indices
```

是：

```text
当前 physical request-local member indices
```

不是：

```text
logical token IDs
physical block IDs
slot IDs
```

---

# 7. `keep_member_indices` 的稳定压缩语义

例如：

```text
source_E = 33

keep = [0,8,16,24,31]

K = 5
```

原来：

```text
physical member indices:
0 ... 32
```

保留：

```text
0
8
16
24
31
```

V2 的 stable compaction 后：

```text
old 0  → new 0
old 8  → new 1
old 16 → new 2
old 24 → new 3
old 31 → new 4
```

因此：

```text
new effective_kv_len = 5
```

如果：

```text
block_size = 16
```

则：

```text
new_num_blocks = ceil(5 / 16) = 1
```

最终 active physical layout：

```text
dense prefix [0,5)
```

---

# 8. Scheduler Plan 的一阶段 pending state 与 schedule-time binding

Plan 首先进入：

```text
Scheduler._prepared_compaction_plans[request_id]
```

但此时还不属于任何具体 Scheduler step。

只有 request 真正 allocation 成功并进入本轮 scheduled set 时：

```python
compaction_plan = self._prepared_compaction_plans.pop(
    request_id, None
)
```

然后：

```python
compaction_plans[request_id] = compaction_plan
```

最终进入：

```text
SchedulerOutput.compaction_plans
```

因此：

```text
pending plan
↓
request 成功 schedule
↓
current-step plan
```

这叫：

```text
schedule-time binding
```

如果 request 被 preempt 或根本没被 schedule，则 pending plan 不应错误发送给 Worker。

---

# 9. SchedulerOutput 为什么是重要边界

`SchedulerOutput` 表示：

> Scheduler 对当前 step 实际发出去的 execution contract。

其中现在额外携带：

```text
compaction_plans
```

所以 V2 的 Scheduler→Worker transport 是：

```text
Scheduler._prepared_compaction_plans
↓
schedule()
↓
SchedulerOutput.compaction_plans
↓
Worker
```

当前的 transport test 专门验证：

```text
request-a 的 Plan
不会串到 request-c
```

---

# 10. Worker 入口：`extract_compaction_plans()`

Worker 从：

```text
SchedulerOutput.compaction_plans
```

提取：

```text
dict[str, CompactionPlanData]
```

这里的职责非常轻：

```text
接收 Plan
确认 request mapping
交给后续 Worker prepare
```

真正 runtime validation 不在这里完成。

---

# 11. `_prepare_v2_compactions()`：先验证，不允许 mutation

文件：

```text
vllm/v1/worker/gpu/model_runner.py
```

职责：

```text
CompactionPlanData
↓
Worker 当前 runtime state 解析
↓
完整 validation
↓
_PreparedCompaction
```

它不是执行函数。

核心 contract：

> 所有可能在 mutation 前检查的条件必须尽量在这里完成。

---

## 11.1 `_PreparedCompaction`

典型字段：

```python
@dataclass(frozen=True)
class _PreparedCompaction:
    request_id: str
    req_state_idx: int
    batch_idx: int
    source_effective_kv_len: int
    source_num_blocks: int
    block_ids: tuple[int, ...]
    block_size: int
    keep_member_indices: tuple[int, ...]
    new_effective_kv_len: int
    new_num_blocks: int
    step_seq: int | None
```

它与 Plan 的区别：

```text
CompactionPlanData
= Scheduler 发来的意图

_PreparedCompaction
= Worker 结合当前真实 runtime state
  验证并解析后的可执行描述符
```

---

## 11.2 验证 index domain

Worker 需要同时得到：

```text
batch_idx
req_state_idx
```

并验证：

```text
idx_mapping[batch_idx] == req_state_idx
```

这是为了防止：

```text
当前 InputBatch 顺序
与 persistent RequestState row
发生 stale mismatch
```

---

## 11.3 验证 post-forward source E

V2 source 应来自：

```text
input_batch.effective_kv_seq_lens[batch_idx]
```

因为它代表：

```text
persistent E + 本轮 query_len
```

例如：

```text
step-start E = 32
q = 1

post-forward source E = 33
```

然后检查：

```text
source_E
==
plan.expected_source_effective_kv_len
```

---

## 11.4 验证 source blocks

从 Worker BlockTables：

```text
self.block_tables.num_blocks.np[0, req_state_idx]
```

取得 source page count。

例如：

```text
source_E = 33
block_size = 16

source_num_blocks = 3
```

并检查：

```text
source_num_blocks == expected_source_num_blocks
```

---

## 11.5 验证 keep set

必须：

```text
非空
都是 int
非负
严格递增
最后一个 < source_E
```

严格递增代表：

```text
stable compaction
```

不允许任意 reorder。

---

## 11.6 计算 destination geometry

```text
K = len(keep_member_indices)

new_effective_kv_len = K

new_num_blocks
=
ceil(K / block_size)
```

---

## 11.7 解析真实 physical block IDs

从：

```text
Worker BlockTable
```

读取 active block row：

```text
block_ids = (...)
```

例如：

```text
(5,1,3)
```

继续验证：

```text
block IDs 非负
不能重复
所有 layer 都有这些 block
KV cache shape/block size/device 一致
```

通过以后才构建：

```text
_PreparedCompaction
```

---

# 12. 为什么 prepare 阶段不能修改任何状态

测试明确验证 prepare 后：

```text
BlockTable unchanged
effective_kv_len unchanged
KV payload unchanged
```

因此：

```text
_prepare_v2_compactions()
=
validation / resolve phase
```

而不是 mutation phase。

这属于典型 transaction hardening：

```text
validate as much as possible
before touching destructive physical state
```

---

# 13. V2 的 payload kernel：`compact_paged_kv_triton_2d`

文件：

```text
vllm/v1/worker/gpu/kv_compaction.py
```

生产入口：

```python
compact_paged_kv_triton_2d(
    kv_cache,
    block_ids,
    source_effective_kv_len,
    keep_member_indices,
)
```

职责只有：

```text
KV payload sparse → dense
```

它不负责：

```text
Scheduler ownership
BlockPool free
persistent E
```

---

# 14. 为什么必须 Gather → Scratch → Writeback

不能直接原地：

```text
src → dst
```

因为 source/destination 会重叠。

例如：

```text
keep = [0,8,16,24,31]
```

最终要写到：

```text
0,1,2,3,4
```

如果一边读一边覆盖，很容易：

```text
前面的 write
覆盖后面还没读取的 source
```

所以：

```text
paged source
↓
完整 gather 到 independent scratch
↓
再写回 dense prefix
```

这是 correctness-first 设计。

---

# 15. Gather 详细地址计算

假设：

```text
block_size = 16
block_ids  = [5,1,3]
source_E   = 33
keep       = [0,8,16,24,31]
```

scratch：

```text
[K,H,C]
=
[5,H,C]
```

每个 retained member：

```text
source_member
↓
source_local_page = source_member // block_size
source_offset     = source_member % block_size
↓
physical_block = block_ids[source_local_page]
↓
kv_cache[physical_block, head, source_offset, :]
↓
scratch[retained_idx, head, :]
```

具体：

```text
old 0
→ block 5 offset 0

old 8
→ block 5 offset 8

old 16
→ block 1 offset 0

old 24
→ block 1 offset 8

old 31
→ block 1 offset 15
```

---

# 16. Writeback 详细语义

K=5：

```text
new_num_blocks = 1
dst_capacity = 16
```

writeback：

```text
scratch[0] → block 5 offset 0
scratch[1] → block 5 offset 1
scratch[2] → block 5 offset 2
scratch[3] → block 5 offset 3
scratch[4] → block 5 offset 4
```

然后：

```text
offset 5..15
→ zero
```

因此：

```text
active dense prefix = [0,5)
```

注意：

> 目标 page 取的是原 block row 的 prefix，即 `block_ids[:new_num_blocks]`。

不是：

> “保留 token 原来位于哪些 block，就保留那些 physical blocks”。

例如：

```text
old retained members 16/24/31
原来可能来自 block 1

但 writeback 后它们都已经被搬到 block 5
```

所以：

```text
retained physical blocks
=
block_ids[:new_num_blocks]
```

这里是：

```text
[5]
```

原来的 block 1、3 都可以成为 trailing ownership。

---

# 17. `_execute_v2_compactions()`：Worker mutation phase

该函数分两阶段。

---

## 17.1 Phase 1：所有 request × all layers 的 payload

核心：

```text
for each request:
    materialize keep tensor
    materialize block_ids tensor

    for each KV layer:
        compact_paged_kv_triton_2d(...)
```

这一步只动：

```text
KV payload
```

暂时不发布：

```text
BlockTable metadata
E override
CompactionResultData
```

---

## 17.2 为什么要两阶段

如果有：

```text
2 requests × N layers
```

假设第一个 request payload 成功，但第二个 request 某层失败。

如果前面已经发布了：

```text
Worker BlockTable
Result
```

Scheduler 可能认为 transaction 成功了一部分。

所以当前 hardening：

```text
Phase 1
先执行全部 payload

只有全部成功

Phase 2
才发布 metadata/result
```

这保证：

```text
payload failure
→ no success result publication
```

注意：

> 这不是完整 rollback。前面已经写过的 KV payload 不会自动恢复。

当前保证的是：

```text
no false success metadata publication
```

不是 ACID transaction。

---

# 18. Phase 2：Worker execution metadata commit

payload 全部成功后：

```python
retained_block_ids
=
prepared.block_ids[:prepared.new_num_blocks]
```

例如：

```text
[5,1,3]
→
[5]
```

然后：

```python
self.block_tables.append_block_ids(
    prepared.req_state_idx,
    (retained_block_ids,),
    overwrite=True,
)
```

注意：

```text
BlockTable
使用 req_state_idx
```

而不是 batch_idx。

接着：

```python
override_valid[prepared.batch_idx] = True
override_value[prepared.batch_idx] = K
```

这里又回到：

```text
batch_idx domain
```

因为 `post_update()` 是当前 batch 的 fold-back kernel。

最后构建：

```text
CompactionResultData
```

发送 Scheduler。

---

# 19. Worker BlockTable 为什么只是 staged write

`append_block_ids(overwrite=True)` 不代表立刻所有 GPU consumer 都看到新 row。

下一次 `execute_model()` 开头会：

```text
apply staged writes
```

因此时间线：

```text
Step N post-forward:
payload compacted
BlockTable overwrite staged
E override 同 step post_update

Step N+1 execute_model start:
BlockTable staged write apply
↓
prepare attention/input
```

这满足：

```text
current forward 已经结束
新 BlockTable 只需要在下一 forward 前可见
```

---

# 20. `override_valid / override_value` 为什么存在

普通路径：

```text
computed_delta = query_len - num_rejected

L += computed_delta
E += computed_delta
```

但 V2：

```text
L += computed_delta
E = K
```

不是：

```text
E += computed_delta
```

因此需要当前 batch 的 absolute override。

例如：

```text
before:
L=32
E=32

q=1
compact to K=5

after:
L=33
E=5
```

所以：

```text
override_valid=True
override_value=5
```

---

# 21. `ExecuteModelState`：为什么 override/result 先挂在这里

MRV2 拆成：

```text
execute_model()
↓
sample_tokens()
```

V2 compaction 在：

```text
model forward 已完成
```

后执行。

但 E 的最终 logical/physical post-update 需要等 sampling/rejection 信息。

因此：

```text
execute_model()
```

先把：

```text
effective_kv_len_override_valid
effective_kv_len_override
compaction_results
```

挂到：

```text
self.execute_model_state
```

中。

它是：

```text
single-step Worker-local carrier
```

不是持久 request state。

---

# 22. `sample_tokens()` 如何消费 ExecuteModelState

进入 `sample_tokens()` 后：

```text
从 execute_model_state 取：
override_valid
override_value
compaction_results
```

然后：

```text
self.execute_model_state = None
```

局部 Python 变量仍持有 tensor/list reference。

这里分两条线：

```text
override_valid/value
→ Worker local post_update

compaction_results
→ ModelRunnerOutput → Scheduler
```

---

# 23. 为什么 override 和 Result 都携带 K，但不能合并

二者可能都带：

```text
K = 5
```

但语义不同。

`override_value=5`：

> Worker 当前 batch 的 persistent physical state 要写成 5。

`result.new_effective_kv_len=5`：

> Worker 告诉 Scheduler：我已经把这个 request 的 physical shape compact 到 5。

因此：

```text
override
= Worker internal state-folding protocol

CompactionResultData
= Worker → Scheduler cross-component receipt
```

索引域也不同：

```text
override
按 batch_idx

Result
按 request_id
```

---

# 24. `post_update()`：V2 最重要的 physical-state commit

文件：

```text
vllm/v1/worker/gpu/input_batch.py
```

为什么放在 `input_batch.py`？

因为它本质是：

```text
current batch execution result
↓
通过 idx_mapping
↓
scatter/fold 回 persistent RequestState
```

而不是 RequestState 自己的简单 setter。

---

# 25. `_post_update_kernel` 的核心逻辑

先：

```text
batch_idx
↓
idx_mapping[batch_idx]
↓
req_state_idx
```

然后：

```text
computed_delta = query_len - num_rejected
```

logical：

```text
num_computed_tokens[req_state_idx]
+= computed_delta
```

physical：

```text
if override_valid[batch_idx]:

    effective_kv_len[req_state_idx]
    =
    override_value[batch_idx]

else if computed_delta != 0:

    effective_kv_len[req_state_idx]
    +=
    computed_delta
```

因此：

```text
normal:
L += delta
E += delta

V2:
L += delta
E = K
```

---

# 26. 为什么 override branch 不能放在 `computed_delta != 0` 里面

因为有场景：

```text
computed_delta = 0
```

但 V2 compaction 已经真实发生。

例如 sampling/rejection 导致逻辑没有新增 accepted token，但 physical KV 已经需要被压到 K。

此时仍然必须：

```text
E = K
```

所以：

```text
override
```

是 absolute state transition，与是否有普通 delta 独立。

---

# 27. ModelRunnerOutput：Worker → Scheduler receipt

V2 在 Worker 返回中增加：

```text
compaction_results: list[CompactionResultData]
```

`CompactionResultData`：

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

它的身份：

```text
Worker-reported receipt
```

不是 Scheduler-authoritative state。

Scheduler 不能直接相信。

---

# 28. Scheduler `update_from_output()`：Plan + Result + Canonical State 对账

签名：

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
)
```

为什么同时需要两者？

因为：

```text
scheduler_output
=
“这一 step 我当初发了什么”

model_runner_output
=
“Worker 这一 step 回来了什么”
```

`update_from_output()` 本质：

```text
command + result reconciliation
```

---

# 29. V2 在 `update_from_output()` 最前面先做 reconciliation

顺序：

```text
_validate_compaction_results()
↓
_prepare_compaction_reconciliations()
↓
_commit_compaction_reconciliations()
↓
再进入普通 sampled-token state update
```

原因：

> V2 physical E 已经是 absolute K，后续普通 Scheduler state update 不能再错误 `E += q`。

---

# 30. `_validate_compaction_results()`

第一层 carrier validation。

当前核心：

```text
不能有 duplicate request_id
```

这是最低限度的结果结构检查。

---

# 31. `_prepare_compaction_reconciliations()`：Scheduler 侧最关键的 transaction validation

这个函数不是做 compaction。

它的角色是：

```text
CompactionResultData
↓
Plan + Result + Scheduler Current State
三方/四方一致性核对
↓
_PreparedCompactionReconciliation
```

并且：

```text
prepare 阶段不允许修改 canonical ownership
```

---

# 32. `CompactionResultData` 与 `_PreparedCompactionReconciliation` 的区别

虽然字段很像，但 trust level 完全不同。

`CompactionResultData`：

```text
Worker 的“证词”
尚未被 Scheduler 信任
```

`_PreparedCompactionReconciliation`：

```text
Scheduler 已完成所有验证后
生成的最小 commit descriptor
```

因此：

```text
Result
↓ validate
Prepared Reconciliation
↓ commit
Canonical mutation
```

和 Worker 侧：

```text
Plan
↓ validate
_PreparedCompaction
↓ execute
```

是对称设计。

---

# 33. active Plan completeness：为什么要比较 `plan_ids - result_ids`

当前 step：

```text
plan_ids
=
Scheduler 真正发出去的 V2 Plan request IDs

result_ids
=
Worker 真正返回 V2 Result 的 request IDs
```

例如：

```text
plan_ids   = {A,B,C}
result_ids = {A,C}
```

则：

```text
plan_ids - result_ids
=
{B}
```

语义不是：

```text
找未结束 request
```

而是：

```text
找：
“Scheduler 发了 Plan，但 Worker 没返回 Result”
```

然后再检查 B 的 lifecycle。

如果：

```text
B 仍然 active
```

则：

```text
ERROR
```

因为 Scheduler 不知道：

```text
Worker 没执行？
执行失败？
执行成功但 Result 丢了？
payload 是否已经变？
```

不能继续。

如果：

```text
B 已 finished/removed
```

可以当 lifecycle stale。

---

# 34. Result → Plan reverse matching

每个 Result 也必须：

```text
找到同一步 matching Plan
```

否则：

```text
Worker 回了一个 Scheduler 从没发过的 transaction
```

属于非法状态。

因此 active transaction 强约束：

```text
Plan ↔ Result
```

需要一一对应。

---

# 35. `step_seq`：为什么 request_id 不够

request_id 只能标识：

```text
哪个 request
```

不能标识：

```text
这个 request 的哪一次 physical transition
```

未来 async / overlapping step 中可能：

```text
Step17 A Plan
Step18 A Plan
```

如果 Step17 Result 晚到：

```text
request_id = A
```

仍然会匹配 A。

所以：

```text
request_id + step_seq
```

一起标识 transaction instance。

在当前同步 single-batch MVP 中不容易自然触发，但这是 V3 async hardening 必需的 fence。

---

# 36. Source fence validation

Plan 中：

```text
expected_source_effective_kv_len
expected_source_num_blocks
```

Result 也必须带相同值。

例如 Scheduler 发：

```text
source_E = 33
blocks = 3
```

Worker 必须回：

```text
source_E = 33
blocks = 3
```

否则拒绝。

本质类似：

```text
Compare old/source state
before applying new state
```

---

# 37. Scheduler 为什么还要自己重算 source E

即使：

```text
Plan == Result
```

也还不够。

Scheduler 还要基于自己的 state：

```text
request.effective_kv_len
request.num_computed_tokens
scheduler_output.num_scheduled_tokens
```

重建：

```text
source_E
```

如果已有 explicit E：

```text
source_base = request.effective_kv_len
```

否则：

```text
source_base
=
request.num_computed_tokens - q
```

这里要减 q，是因为 Scheduler 在 schedule 后已经 optimistic advance 了 logical counter。

然后：

```text
source_E = source_base + q
```

最后要求：

```text
source_E
==
result.expected_source_effective_kv_len
```

这是 Scheduler 用自己的事实核验 Worker receipt。

---

# 38. Geometry validation

继续检查：

```text
0 < new_E <= source_E
```

以及：

```text
expected_source_num_blocks
==
ceil(source_E / block_size)

new_num_blocks
==
ceil(new_E / block_size)

0 < new_num_blocks <= source_num_blocks
```

当前 MVP 假设：

```text
active physical KV
=
dense paged prefix
```

所以 E 与 page count 必须严格匹配。

---

# 39. Canonical ownership validation

Scheduler 最后检查：

```text
KVCacheManager 当前 canonical row
```

仍然具有：

```text
expected_source_num_blocks
```

如果 Worker source 是 3 blocks，但 Scheduler canonical row 已经不是 3 blocks，则：

```text
ERROR
```

因为 ownership source 已经发生变化，不能盲目 truncate。

---

# 40. `_PreparedCompactionReconciliation`

所有验证通过后，生成：

```python
@dataclass(frozen=True)
class _PreparedCompactionReconciliation:
    request_id: str
    expected_source_num_blocks: int
    new_effective_kv_len: int
    new_num_blocks: int
    step_seq: int | None
```

它故意只保留 commit 阶段真正需要的数据。

例如：

```text
expected_source_effective_kv_len
```

已经用于验证完 source fence，commit 阶段不再需要。

---

# 41. `_commit_compaction_reconciliations()`

commit：

```text
for each prepared item
↓
kv_cache_manager.reconcile_compacted_blocks()
↓
request.effective_kv_len = K
↓
release removed blocks
```

这是 canonical mutation phase。

---

# 42. `reconcile_compacted_blocks()`：ownership detach，不负责 free

在 `SingleTypeKVCacheManager`：

```text
current_blocks
=
req_to_blocks[request_id]
```

例如：

```text
[5,1,3]
```

如果：

```text
new_num_blocks = 1
```

则：

```text
retained = [5]
removed  = [1,3]
```

然后：

```text
req_to_blocks[request_id] = [5]
```

最后返回：

```text
[block1, block3]
```

这里的职责是：

```text
T3 ownership detach
```

不是 allocator free。

---

# 43. 为什么先 detach 再 free

必须：

```text
canonical ownership:
[5,1,3]
→
[5]
```

然后：

```text
free [1,3]
```

不能反过来。

否则短暂窗口可能：

```text
BlockPool 已经把 block1 分给 Request B
但 canonical state 仍声称 Request A 拥有 block1
```

导致：

```text
double ownership / aliasing
```

所以 invariants：

```text
ownership detach
must happen before allocator reuse
```

---

# 44. `_release_reconciled_blocks()`

当前同步 MVP：

```python
self.kv_cache_manager.block_pool.free_blocks(
    reversed(removed)
)
```

直接回 BlockPool。

注意：

> 这个 helper 自己没有判断 `defer_block_free=False`。

它本身就是：

```text
synchronous-MVP-specific direct release path
```

当前之所以安全，是因为支持边界保证：

```text
async_scheduling=False
max in-flight batch=1
connector off
```

进入 V3 async hardening 后需要重新处理：

```text
detach
→ fence
→ deferred free
→ safe reuse
```

---

# 45. `num_cached_block` clamp

canonical row truncate 后：

```text
num_cached_block[request_id]
=
min(old_cached_count, len(retained))
```

这是为了防止 cached-block metadata 超出新的 canonical row 长度。

注意：

> V2 stable compaction 后，原 prefix hash/cache 语义是否仍完全成立是更广模式问题；当前 prefix caching 明确关闭，不应把这一点外推到 prefix-cache support。

---

# 46. V2 四层责任边界

可以把整个设计记成：

```text
1. Command Plane
Scheduler → Worker
CompactionPlanData

2. Data Plane
Worker
sparse KV → scratch → dense prefix

3. State Plane
Worker + Scheduler
E_source → K

4. Ownership Plane
Scheduler
canonical row truncate
→ BlockPool free/reuse
```

这四层不能混为一个函数。

---

# 47. 四个主要 carrier

## 47.1 `CompactionPlanData`

```text
Scheduler → Worker
```

表达：

```text
我要你 compact 什么
```

---

## 47.2 `_PreparedCompaction`

```text
Worker internal
```

表达：

```text
我已经验证并解析出怎么执行
```

---

## 47.3 `CompactionResultData`

```text
Worker → Scheduler
```

表达：

```text
我执行完了，physical shape 变成这样
```

---

## 47.4 `_PreparedCompactionReconciliation`

```text
Scheduler internal
```

表达：

```text
我已经核验 Worker receipt，可以 commit canonical state
```

---

# 48. V2 Real-Engine E2E：测试方法先理解清楚

测试脚本不是：

```text
自动 trace framework
```

也不是：

```text
直接手调 private compaction function
```

它采用：

```text
真实 Engine
+
runtime monkey patch observer
+
一次 explicit Plan injection
```

核心思想：

> 不改生产执行路径，只在 Scheduler 的两个关键 seam 外面包 observer。

---

# 49. 为什么要 `enable_multiprocessing=False`

这样：

```text
LLMEngine
和真正 EngineCore
在同一个 Python process
```

测试脚本才可以直接拿：

```python
scheduler = core.scheduler
runner = core.model_executor.driver_worker.model_runner
pool = scheduler.kv_cache_manager.block_pool
```

并 monkey-patch：

```text
scheduler.schedule
scheduler.update_from_output
```

---

# 50. `MethodType` 的作用

原来：

```text
scheduler.schedule
→ Scheduler.schedule
```

保存：

```python
original_schedule = scheduler.schedule
```

然后：

```python
scheduler.schedule = MethodType(
    schedule_trace,
    scheduler,
)
```

之后：

```text
engine.step()
↓
schedule_trace()
```

但 `schedule_trace()` 内部仍：

```python
output = original_schedule(...)
```

因此：

```text
真实 Scheduler.schedule
仍然执行
```

测试只是外面加：

```text
observe / record
```

同理：

```text
update_trace
```

也是包在真实：

```text
Scheduler.update_from_output()
```

外面。

---

# 51. `trace` 到底是什么

它不是 profiler。

只是普通 Python dict：

```python
trace = {
    "runtime": {...},
    "schedule_events": [],
    "compaction": None,
    "next_step": None,
}
```

本质：

```text
在关键 transaction seam 拍状态快照
```

可以理解成：

```text
测试人工维护的 evidence ledger
```

---

# 52. 为什么选 `schedule()` 和 `update_from_output()` 两个观察点

因为这两个点正好夹住 Worker。

```text
Scheduler.schedule()
        ↓
   Worker execution
        ↓
Scheduler.update_from_output()
```

因此：

`schedule_trace()` 可以观察：

```text
Scheduler 发了什么
```

`update_trace()` 可以观察：

```text
Worker 回来了什么
Scheduler commit 后变成什么
```

它们正好对应 V2 transaction 的两端。

---

# 53. `schedule_trace()` 记录什么

每次真实：

```text
Scheduler.schedule()
```

完成后，记录：

```text
num_scheduled_tokens
cached_req_ids
canonical_row
effective_kv_len
num_computed_tokens
free_blocks
compaction_plan（如果存在）
```

其中：

```text
canonical_row
```

回答：

> Scheduler 当前认为 request 拥有哪些 physical blocks？

`free_blocks` 回答：

> allocator capacity 如何变化？

`compaction_plan` 回答：

> pending plan 是否真的通过 schedule-time binding 进入 SchedulerOutput？

---

# 54. `update_trace()` 记录什么

它先调用：

```text
真实 Scheduler.update_from_output()
```

所以之后记录的：

```text
canonical row
E
free blocks
```

都是 commit 后状态。

如果 Worker 返回：

```text
CompactionResultData
```

测试记录：

```text
source_E
K
old_num_blocks
new_num_blocks
canonical_row_after_commit
free_blocks_after
effective_kv_len_after_commit
step_seq
```

因此能验证：

```text
Worker 说：
33 / 3 → 5 / 1

Scheduler 真正 state：
row → 1 page
E → 5
allocator capacity 恢复
```

---

# 55. 为什么 Plan 在 `update_trace()` 里注入

先让 request 正常跑出真实 KV。

测试 request：

```text
prompt_len = 32
block_size = 16
```

因此 prompt 执行后：

```text
32 physical members
≈ 2 pages
```

等：

```text
num_computed_tokens >= 32
```

以后才注入：

```text
下一 decode step 要执行 V2
```

这样 V2 source 是真实模型产生的 KV，不是 synthetic tensor。

---

# 56. 为什么 source E 是 33，不是 32

Plan 在当前 step 结束后注入。

它描述的是：

```text
下一 step post-forward source
```

当前：

```text
E = 32
```

下一 decode：

```text
q = 1
```

所以：

```text
source_E
=
32 + 1
=
33
```

由于：

```text
block_size=16
```

需要：

```text
ceil(33 / 16)
=
3 pages
```

这就是为什么 Plan 是：

```text
expected source:
E=33
blocks=3
```

尽管注入 Plan 当下 canonical row 还是：

```text
[1,2]
```

两者不矛盾。

---

# 57. 为什么这个 E2E 场景设计得好

它故意选择：

```text
prompt=32
block_size=16
下一 decode q=1
```

让 compaction step 产生：

```text
[1,2]
↓ append token 33
[1,2,3]
↓ compact 33 → 5
[1]
```

因此同一步验证：

```text
same-step new page allocation
+
post-forward compaction
+
trailing page release
```

不是只测：

```text
已有 3 pages 静态压到1 page
```

---

# 58. Real-Engine E2E 的真实观察

已观察：

```text
source_E = 33

keep_member_indices =
[0,8,16,24,31]

K = 5

old geometry = 3 blocks
new geometry = 1 block

plan injection 前:
canonical row = [1,2]

compaction schedule:
canonical row = [1,2,3]

commit 后:
canonical row = [1]

effective_kv_len = 5
```

allocator：

```text
计划注入前 free = 10865

为第 33 token 新申请 page：
10865 → 10864

compaction 释放 2 pages：
10864 → 10866
```

因此：

```text
本次 commit 释放 = 2 blocks
相对 plan injection 前净增 capacity = +1
```

---

# 59. 为什么 `max_tokens=3`

如果 compact 后 request 马上 finish，则无法验证：

```text
新 physical state 是否还能继续被真实模型使用
```

所以：

```text
max_tokens=3
ignore_eos=True
```

迫使 request 后续继续真实 decode。

目的：

```text
compaction
↓
后续 schedule
↓
后续 forward
↓
request finish
```

---

# 60. `engine.step()` 为什么是 E2E 的真正核心

测试没有手动写：

```text
scheduler.schedule()
runner.execute_model()
scheduler.update_from_output()
```

而只是不断：

```python
outputs = engine.step()
```

因此走的是 vLLM 真实 choreography：

```text
LLMEngine
↓
EngineCore
↓
Scheduler.schedule()
↓
ModelExecutor
↓
GPUModelRunner
↓
真实 Qwen forward
↓
V2 compaction
↓
sample/post_update
↓
ModelRunnerOutput
↓
Scheduler.update_from_output()
↓
EngineCoreOutput
```

这就是 E2E 和 unit test 的本质区别。

---

# 61. E2E 最终 assertions 证明什么

当前 assertions：

```text
Plan 确实注入过
Result/commit 确实发生

source_E = 33
K = 5

old blocks = 3
new blocks = 1

Scheduler commit 后 E = 5

compaction 后 persistent E 仍为 5

request 最终真实 finish
```

因此直接证明：

```text
Plan transport
Worker real execution
real KV compaction
Result transport
Scheduler reconciliation
canonical ownership truncate
physical E commit
allocator release
后续 generation continuity
```

在当前 baseline 下闭环。

---

# 62. E2E 没有直接证明什么

不能把 smoke 的结论扩张为：

```text
keep=[0,8,16,24,31] 是优秀策略
模型质量不下降
性能一定提升
TPOT/TTFT 改善
async 正确
prefix cache 正确
spec decode 正确
CUDA Graph 正确
```

另外，当前 smoke 没有直接 assert：

```text
Step N+1 cache_positions == 5
```

它证明：

```text
compaction 后真实 Engine 能继续执行并 finish
```

但：

```text
cache_position exactly 5
```

如果要作为强 evidence，应该单独 instrument。

---

# 63. 当前 E2E `next_step` 命名的小问题

代码在：

```text
compaction step 的 engine.step() 返回后
```

立即记录：

```text
trace["next_step"]
```

严格来说第一次写入它时更准确的名字应该是：

```text
post_compaction_state
```

因为它首先表示：

```text
Step N commit 后的 persistent state
```

不是直接记录：

```text
Step N+1 model forward 已经消费后的 state
```

后续 loop 的确继续真实执行并 finish，所以 continuity 仍然成立，但 instrumentation 命名可更精确。

---

# 64. 当前 E2E trace metadata 的小问题

trace 中：

```text
"max_concurrent_batches"
```

实际读取：

```text
scheduler_config.max_num_seqs
```

两者概念不应混淆。

这是：

```text
instrumentation metadata label issue
```

不是 runtime correctness bug。

建议后续直接改名：

```text
max_num_seqs
```

---

# 65. 已发现的一个值得复核的 multi-request implementation hole

在一个审计 snapshot 中：

```python
for request_id, prepared in prepared_compactions.items():
    ...
```

Phase 2：

```python
for prepared, _, _ in prepared_tensors:
    ...
    CompactionResultData(
        request_id=request_id,
        ...
    )
```

这里：

```text
request_id
```

可能是上一层 loop 遗留的最后一个 request ID。

如果多个 request 全部成功 compact：

```text
A
B
```

则 Phase 2 可能错误让两个 Result 都使用：

```text
B
```

更合理应为：

```python
request_id=prepared.request_id
```

当前 single-request E2E 无法发现。

现有 multi-request failure hardening test也主要验证：

```text
第二个 request failure 时
不能发布前一个 request 的 metadata/result
```

不等价于：

```text
两个 request 全部成功
result identity 都正确
```

因此建议加一个：

```text
two-request all-success test
```

验证：

```text
results[0].request_id == A
results[1].request_id == B
```

如果当前 HEAD 已经修复，则在后续 source re-audit 中将此项关闭。

---

# 66. Unit / Core / E2E 三层测试体系

## 66.1 Kernel Test

验证：

```text
gather/writeback bytes 正确
scratch overlap-safe
geometry 正确
```

不涉及：

```text
Scheduler
RequestState
BlockPool
```

---

## 66.2 Worker Tests

验证：

```text
Plan → _PreparedCompaction
index domain
source fence
BlockTable row
execute orchestration
override
Result
failure no-success-publication
```

通常不是真实 model forward。

---

## 66.3 Scheduler/Core Tests

验证：

```text
Plan transport
Result carrier
reconciliation validation
canonical ownership detach
release timing
fail-closed
```

多数可注入 fake ModelRunnerOutput。

---

## 66.4 Real-Engine E2E

验证：

```text
所有模块真的在生产 runtime chain 中组合起来
```

也就是：

```text
real Scheduler
real allocation
real model forward
real KV write
real Triton compaction
real Worker post_update
real ModelRunnerOutput
real Scheduler reconciliation
real BlockPool
real continuation
```

这才是 functional closure。

---

# 67. V2 关键 invariants

可以冻结成以下集合。

```text
INV-1
keep_member_indices 表示 current physical request-local member indices。

INV-2
logical/RoPE positions 不因 compaction 重新编号。

INV-3
V2 source E 是 post-forward physical extent。

INV-4
compaction 后 active KV 必须成为 dense prefix [0,K)。

INV-5
Worker execution BlockTable 与 Scheduler canonical ownership
最终必须收敛到相同 page geometry。

INV-6
ownership detach 必须早于 allocator reuse。

INV-7
new E = K；下一 append 从 physical K 开始。

INV-8
pre-mutation validation failure
不得改变 payload/ownership/E/allocator state。

INV-9
normal request 保持 upstream additive physical progress。

INV-10
V2 request 的 physical E 使用 absolute override，
不能在 K 上再错误 +q。

INV-11
active Plan 必须有 matching Result。

INV-12
Result 必须对应 current-step matching Plan。

INV-13
Plan / Result source fence 必须一致。

INV-14
Scheduler 必须独立重建并验证 source geometry。

INV-15
Result 不是 allocator authority；
Scheduler canonical ownership 才决定哪些 blocks 可 detach/free。
```

---

# 68. 从 V1 → V2 的本质升级

V1 更接近：

```text
whole-block physical ownership selection
```

V2 增加：

```text
token/member-level payload movement
```

因此 V2 多了一层最关键的数据面：

```text
source sparse member set
↓
physical gather
↓
scratch
↓
dense repack
```

但 V1 中已经建立的：

```text
logical / physical split
physical E
Scheduler ownership authority
BlockPool release/reuse
result-time commit
```

仍被 V2 复用。

因此 V2 不是一套完全独立系统，而是：

```text
V1 state/ownership substrate
+
V2 token-level payload compaction
```

---

# 69. V2 当前最重要的性能问题：movement amplification

当前 baseline：

```text
keep-list
+
full retained gather/writeback
```

如果：

```text
source = 4608
evict = 512
keep = 4096
```

control metadata：

```text
keep-list = 4096 indices
```

可能很大。

即便改成：

```text
delete-list = 512
```

只是 metadata 更小。

payload movement 仍可能：

```text
O(K)
```

因为 stable compaction 要把后面 surviving runs 向前移动。

因此未来性能分析应区分：

```text
policy representation cost
vs
physical movement cost
```

可以定义：

```text
movement amplification
=
actual KV members moved
/
members deleted
```

当前 M4 是：

```text
correctness-first generic stable compaction
```

不是最终 bandwidth-optimal implementation。

---

# 70. 后续可能的优化方向

在 benchmark 证明 payload movement 是瓶颈后，可以考虑：

```text
keep-list baseline
↓
evict-list / mask
↓
segment/run representation
↓
segment-aware stable compaction
↓
只移动 destination 发生变化的 surviving runs
```

但不能为了减少 movement 随意使用：

```text
tail-fill-holes
```

如果这种方案破坏：

```text
stable member order
```

则会改变当前 V2 semantic contract。

---

# 71. V3 Runtime Hardening 的自然入口

当前 functional core 后，真正下一阶段不是继续扩功能，而是处理：

```text
async safety
lifecycle safety
execution overlap
preemption
CUDA Graph
connector/offload interaction
multi in-flight batch
```

特别是：

```text
current:
ownership detach
→ immediate free

future async:
ownership detach
→ attach execution fence
→ deferred free
→ safe reuse
```

此外 `step_seq` 会从当前 hardening metadata 变成真正重要的：

```text
transaction version fence
```

---

# 72. 一句话讲给面试官

如果需要把 V2 在面试中压缩成 30 秒：

> 我在 vLLM 0.26 的 MRV2 上做了 token-level physical KV compaction。核心是把 logical progress 和 physical KV frontier 解耦：RoPE position 继续按 logical token 走，而 KV write 从 compact 后的 `effective_kv_len` 继续。每次 compaction 由 Scheduler 发送一个 physical-member keep plan，Worker 在 forward 后用 Triton 把 retained KV gather 到 scratch，再稳定写回 dense prefix；Worker 本地用 absolute override 提交新的 E，同时返回 result receipt。Scheduler 再对 Plan、Result 和 canonical ownership 做 reconciliation，截断 `req_to_blocks` 并把 trailing pages 还给 BlockPool。最终 real-engine smoke 验证了 Qwen3 上 33 个 KV members / 3 pages 压到 5 members / 1 page 后，请求仍能继续真实 generation 并完成。

---

# 73. 更详细的面试追问：为什么不能只改 BlockTable？

因为：

```text
BlockTable
只描述：
request-local page → physical block
```

如果不真的移动 payload：

```text
保留下来的 token 仍散落在旧 pages / offsets
```

而下一步：

```text
E = K
```

会假设：

```text
active KV 已经是 dense prefix [0,K)
```

所以：

```text
payload compaction
+
BlockTable geometry
+
E
```

三者必须一致。

---

# 74. 更详细的面试追问：为什么不用 logical token ID 做 keep index？

因为 V2 的 operation domain 是：

```text
current physical KV representation
```

compaction 后：

```text
logical token positions
仍保持原值
```

而：

```text
physical member indices
被重新 dense-renumber
```

如果把两者混在一起，就无法在多次 compaction 后正确解释下一次 source。

---

# 75. 更详细的面试追问：为什么 Scheduler 不能直接相信 Worker Result？

因为 Scheduler 是：

```text
canonical ownership + allocator authority
```

Worker 只是：

```text
execution + payload authority
```

如果直接相信 Worker：

```text
stale / duplicate / wrong-step Result
```

可能导致 Scheduler 错误 free 正在被其他状态引用的 block。

因此必须：

```text
Plan
+
Result
+
current Request state
+
canonical row
```

共同一致，才允许 commit。

---

# 76. 更详细的面试追问：为什么 E2E 还要保留 unit tests？

因为 E2E 失败时很难定位：

```text
是 Triton bytes 错？
index mapping 错？
BlockTable staged write 错？
post_update 错？
Scheduler reconciliation 错？
allocator free 错？
```

所以测试金字塔：

```text
kernel
↓
Worker
↓
Scheduler/Core
↓
real Engine E2E
```

每层证明不同 contract。

---

# 77. 当前 Source / Test / Evidence 索引

## Production source

```text
vllm/v1/core/sched/output.py
    CompactionPlanData
    SchedulerOutput.compaction_plans

vllm/v1/outputs.py
    ModelRunnerOutput.compaction_results

vllm/v1/core/sched/scheduler.py
    _prepared_compaction_plans
    _set_prepared_compaction_plan()
    schedule()
    _validate_compaction_results()
    _prepare_compaction_reconciliations()
    _commit_compaction_reconciliations()
    _release_reconciled_blocks()
    update_from_output()

vllm/v1/worker/gpu/model_runner.py
    extract_compaction_plans()
    _prepare_v2_compactions()
    _execute_v2_compactions()
    ExecuteModelState wiring
    execute_model()
    sample_tokens()

vllm/v1/worker/gpu/input_batch.py
    prepare_pos_seq_lens()
    post_update()
    _post_update_kernel()

vllm/v1/worker/gpu/states.py
    RequestState.effective_kv_len

vllm/v1/worker/gpu/block_table.py
    staged BlockTable overwrite

vllm/v1/worker/gpu/kv_compaction.py
    gather_paged_kv_triton_*
    writeback_paged_kv_triton_*
    compact_paged_kv_triton_2d()

vllm/v1/core/single_type_kv_cache_manager.py
    reconcile_compacted_blocks()

vllm/v1/core/kv_cache_coordinator.py
    reconcile_compacted_blocks()

vllm/v1/core/kv_cache_manager.py
    reconcile_compacted_blocks()

BlockPool
    free_blocks()
    get_new_blocks()
```

## Targeted tests

```text
tests/v1/core/test_reclaim_transport.py

tests/v1/core/test_compaction_reconciliation.py

tests/v1/worker/test_gpu_compaction_preparation.py

tests/v1/worker/test_gpu_kv_compaction.py

tests/v1/worker/test_gpu_effective_kv_state.py
```

## Real Engine smoke

```text
04-experiments/project1_kv_reclaim/scripts/p1_v2_e2e_smoke.py
```

核心 raw evidence：

```text
raw/p1-v2-e2e-closure/v2-gpu1-pass-attempt-20260913.log
```

---

# 78. 最终认知图

```text
                         Scheduler
┌────────────────────────────────────────────────────────────┐
│                                                            │
│ explicit producer / policy                                 │
│        │                                                   │
│        ▼                                                   │
│ CompactionPlanData                                         │
│        │                                                   │
│ _prepared_compaction_plans                                 │
│        │ schedule-time binding                             │
│        ▼                                                   │
│ SchedulerOutput.compaction_plans                           │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
                         Worker
┌────────────────────────────────────────────────────────────┐
│ extract plan                                               │
│      ↓                                                     │
│ _prepare_v2_compactions                                    │
│      ↓                                                     │
│ _PreparedCompaction                                        │
│                                                            │
│ real model forward                                         │
│      ↓                                                     │
│ current-step KV write                                      │
│      ↓                                                     │
│ source E = persistent E + q                                │
│      ↓                                                     │
│ compact_paged_kv_triton_2d                                 │
│      ↓                                                     │
│ gather → scratch → dense prefix                            │
│      ↓                                                     │
│ BlockTable staged overwrite                                │
│      ↓                                                     │
│ E override = K                                             │
│      ↓                                                     │
│ CompactionResultData                                       │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
                         Scheduler
┌────────────────────────────────────────────────────────────┐
│ update_from_output                                         │
│      ↓                                                     │
│ validate Result carrier                                    │
│      ↓                                                     │
│ Plan ↔ Result ↔ Request ↔ canonical row validation         │
│      ↓                                                     │
│ _PreparedCompactionReconciliation                          │
│      ↓                                                     │
│ canonical ownership                                        │
│ [B0,B1,B2] → [B0]                                         │
│      ↓                                                     │
│ Scheduler E = K                                            │
│      ↓                                                     │
│ removed [B1,B2]                                            │
│      ↓                                                     │
│ BlockPool.free_blocks                                      │
└───────────────┬────────────────────────────────────────────┘
                │
                ▼
                       Next Step
┌────────────────────────────────────────────────────────────┐
│ logical L continues                                        │
│ physical append starts from E=K                            │
│ Worker BlockTable uses compacted page row                  │
│ attention sees compacted physical KV                       │
│ request continues and finishes                             │
└────────────────────────────────────────────────────────────┘
```

---

# 79. Final Conclusion

V2 的真正价值不在于“写了一个 Triton gather/writeback kernel”，而在于完成了以下系统语义的闭环：

```text
logical / physical progress split
+
physical write coordinate
+
physical attention visibility
+
token/member-level payload compaction
+
Worker state commit
+
Worker→Scheduler receipt
+
canonical ownership reconciliation
+
allocator capacity release
+
next-step continuation
```

当前 Real-Engine closure 已经证明，在限定 baseline 下：

```text
source E = 33
3 physical pages
keep 5 members
↓
E = 5
1 physical page
↓
Scheduler canonical row 收缩
↓
trailing pages 释放
↓
真实模型继续 generation 并 finish
```

因此当前阶段可以把：

```text
V2 Functional Core
```

视为真正闭环。

下一阶段应集中在：

```text
V3 Runtime Hardening
```

而不是继续无边界扩展功能。

