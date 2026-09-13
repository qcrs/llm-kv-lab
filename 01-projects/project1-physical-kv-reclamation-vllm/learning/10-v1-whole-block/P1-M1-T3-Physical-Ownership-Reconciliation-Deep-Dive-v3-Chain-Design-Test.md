# P1-M1-T3 — Physical Ownership Reconciliation / Safe Free / Real Block Reuse

> Project: **P1 — Physical KV Cache Reclamation for vLLM**  
> Subtitle: **Decoupled Logical / Physical KV Positions with Real Block Reuse**  
> Baseline: **vLLM v0.26.0 / commit `568afb3a13806beb53bb2e6bd518269357b237c0`**  
> Scope: **MRV2 / A100 / BF16 / FA2 / eager / single KV group / block_size=16 / TP=PP=DP=DCP=PCP=1**  
> Out of scope: prefix caching / spec decode / async scheduling / CUDA Graph / multi-group / connector/offload / token-level compaction / performance claim

---

# 0. 先给结论：T3 到底在解决什么

T2 结束后，Worker 已经能用 compact 后的 row 做真实 forward：

```text
Scheduler canonical:
[B0 B1 B2 B3 B4 B5 B6]

Worker execution:
[B0 B1 B4 B5 B6]
```

但这时只完成了：

```text
execution-side compaction
```

还没有完成：

```text
allocator-visible physical reclamation
```

因为 Scheduler 仍 canonical-own `B2/B3`，BlockPool 也不能把它们交给其他 request。

所以 T3 真正闭合的是：

```text
Worker physical execution truth
        ↓
Scheduler canonical ownership truth
        ↓
BlockPool reusable truth
        ↓
future allocation physical accounting
```

对应三个问题：

```text
T3-A 什么时候 free 才安全？
T3-C canonical ownership 怎么改成 dense row？
T3-B reclaim 后 allocator 以后按 L 还是按 E 算容量？
```

---

# 1. 先从真实 vLLM 推理主链理解 T3

当前 baseline 下，一次正常 `EngineCore.step()` 可以简化成：

```text
EngineCore.step()
    │
    ▼
Scheduler.schedule()
    │
    ├─ 决定本轮哪些 request 执行
    ├─ 决定 num_scheduled_tokens
    └─ KVCacheManager.allocate_slots()
            │
            ▼
        KV allocator / BlockPool
    │
    ▼
SchedulerOutput
    │
    ▼
model_executor.execute_model(...)
    │
    ▼
Worker / GPUModelRunner
    │
    ├─ 更新 RequestState
    ├─ 更新 BlockTables
    ├─ build slot_mapping / attention metadata
    └─ real model forward
    │
    ▼
ModelRunnerOutput
    │
    ▼
Scheduler.update_from_output()
```

P1 没有另起一套 runtime，而是沿这条 upstream 链插入：

```text
Scheduler.schedule()
    ↓
Prepare reclaim transaction

Worker / GPUModelRunner
    ↓
T2 dense execution publication

real forward
    ↓

Scheduler.update_from_output()
    ↓
T3 result-time commit
```

所以理解 T3 的第一原则是：

> **它是一个 schedule → execute → commit 的资源事务。**

---

# 2. Test 流程为什么和真实推理入口不一样

测试里看到：

```python
scheduler = create_scheduler(...)
request = create_requests(...)
scheduler.schedule()
scheduler.update_from_output(...)
```

这不是 production runtime 入口。

`create_scheduler()` / `create_requests()` 来自：

```text
tests/v1/core/utils.py
```

属于 **test helper**。

它们会直接构造：

```text
ModelConfig
SchedulerConfig
CacheConfig
ParallelConfig
VllmConfig
KVCacheConfig
        ↓
Scheduler(...)
```

所以：

```text
真实 runtime:

EngineCore
  ↓
Scheduler.schedule()
  ↓
ModelExecutor / Worker / GPU
  ↓
ModelRunnerOutput
  ↓
Scheduler.update_from_output()


core test:

直接创建 Scheduler
  ↓
scheduler.schedule()
  ↓
[跳过真实 GPU]
  ↓
fake ModelRunnerOutput
  ↓
scheduler.update_from_output()
```

因此 core test 主要证明：

```text
Scheduler transaction
canonical ownership
BlockPool refcount/free
effective_kv_len accounting
preemption lifecycle
allocator reuse
```

而真实 Worker / FA2 / KV tensor 还需要 integration smoke 验证。

---

# 3. 四种 runtime truth

| Truth | Owner | 状态 | 回答的问题 |
|---|---|---|---|
| Logical progress truth | Scheduler `Request.num_computed_tokens` | `L` | 模型逻辑处理到哪里？ |
| Worker physical execution truth | Worker `RequestState.effective_kv_len` + `BlockTables` | Worker `E` | 当前 forward 如何写/读 physical KV？ |
| Scheduler canonical ownership truth | KV manager hierarchy / `req_to_blocks` | `[B...]` | request 真正拥有哪些 physical blocks？ |
| Allocator reuse truth | `BlockPool` | refcount/free queue | 哪些 physical block slots 可以给别人？ |

T3 本质上是让后两层追上 T2 已经建立的 Worker physical truth。

---

# 4. S1 / S2 / S3 / T2 / T3 的关系

## S1 — 独立 physical frontier

Worker 增加：

```text
RequestState.effective_kv_len
```

可以表达：

```text
L != E
```

## S2 — model position / KV write position 解耦

例如：

```text
L = 94
E = 62
q = 3
```

模型/RoPE：

```text
94,95,96
```

KV write：

```text
62,63,64
```

所以：

```text
positions       → logical/model coordinate
cache_positions → physical KV coordinate
```

## S3 — logical seq / physical KV visibility 解耦

Attention 使用 physical extent：

```text
effective_kv_seq_lens
```

而不是默认把 logical progress 当成真实 KV 长度。

## T2 — Worker 切换 dense execution row

```text
old:
[B0 B1 B2 B3 B4 B5]

retain:
[B0 B1 B4 B5]

same-step new:
[B6]

Worker execution:
[B0 B1 B4 B5 B6]
```

T2 解决的是：

> 当前 forward 用什么 physical block row。

## T3 — Scheduler / allocator 正式接受新的 physical reality

```text
Worker:
[B0 B1 B4 B5 B6]

Scheduler:
[B0 B1 B2 B3 B4 B5 B6]
```

T3 负责最终收敛。

---

# 5. Canonical example：整篇只用这一组数字

```text
block_size = 16

Before:
L = 94
E = 94

Scheduler canonical:
[B0 B1 B2 B3 B4 B5]
```

reclaim：

```text
retain:
[B0 B1 B4 B5]

drop:
[B2 B3]

removed capacity:
2 * 16 = 32
```

因此：

```text
E_target = 94 - 32 = 62
```

当前 step：

```text
q = 3
```

同一个 `Scheduler.schedule(step N)` 内 normal allocation 还可能追加：

```text
B6
```

所以 dispatch 前 Scheduler canonical：

```text
[B0 B1 B2 B3 B4 B5 | B6]
```

Worker T2：

```text
retained old:
[B0 B1 B4 B5]

same-step new:
[B6]

final execution:
[B0 B1 B4 B5 B6]
```

forward：

```text
logical positions:
94,95,96

physical writes:
62,63,64
```

完成：

```text
L = 97
E_completed = 65
```

---

# 6. 为什么 `reclaim_transition` 与 `new_block_ids` 仍然分开

这是最容易记错的一点。

`ReclaimTransitionData` 描述：

```text
旧 ownership 怎么缩
```

例如：

```text
expected_old_num_blocks = 6
retained_block_ids = [B0,B1,B4,B5]
new_effective_kv_len = 62
```

而：

```text
new_block_ids = [B6]
```

来自同一 scheduler step 的 normal allocation。

因此：

```text
transition
=
reclaim delta

new_block_ids
=
normal allocation delta
```

最终 compose：

```text
retained old + same-step new
=
[B0 B1 B4 B5 B6]
```

“allocation 后才 publish transition”不等于“transition 内部已经包含 B6”。

---

# 7. T3-A — Safe Free

问题不是：

```text
怎么 free
```

而是：

```text
什么时候 free 后允许别人 reuse
```

Worker row 切成：

```text
[B0 B1 B4 B5 B6]
```

不代表立刻可以：

```python
BlockPool.free_blocks(B2, B3)
```

因为：

```text
CPU-side state 已更新
≠
所有可能访问旧 physical resource 的 GPU work 已完成
```

错误时序可能导致：

```text
Request A 某个 in-flight work 仍访问 B2
同时 Request B 已拿到 B2
```

形成 use-after-free / physical aliasing。

所以 safe-free 是 **resource lifetime** 问题。

---

# 8. 为什么 `update_from_output()` 是 commit seam

真实顺序：

```text
Scheduler.schedule(step N)
    ↓
execute_model(...)
    ↓
future.result()
    ↓
ModelRunnerOutput
    ↓
Scheduler.update_from_output(step N)
```

Scheduler upstream 本身还维护：

```text
sched_step_seq
processed_step_seq
deferred_frees
```

所以 baseline 里 P1 不需要发明新的：

```text
CUDA Event
reclaim ACK
generation ID
per-block completion message
```

而是复用 matching：

```text
update_from_output(step N)
```

作为 result-time completion seam。

重要区分：

```text
prepare reclaim      ≠ safe free
Worker row changed   ≠ safe free
result-time completion + matching update_from_output = commit seam
```

---

# 9. T3-C — Canonical ownership hierarchy

Control plane：

```text
Scheduler
  ↓
KVCacheManager
  ↓
KVCacheCoordinator
  ↓
SingleTypeKVCacheManager
  ↓
req_to_blocks
```

Execution plane：

```text
SchedulerOutput
  ↓
GPUModelRunner
  ↓
BlockTables
  ↓
slot_mapping
  ↓
GPU KV
```

所以：

```text
req_to_blocks
=
canonical Request → blocks ownership

BlockTables
=
当前 forward 的 addressing view
```

T2 改 execution view。

T3-C 才改 canonical ownership。

---

# 10. Manager hierarchy 怎么记

```text
KVCacheManager
=
request-facing public API

KVCacheCoordinator
=
KV group coordination

SingleTypeKVCacheManager
=
one-group canonical ownership

BlockPool
=
physical block allocator / refcount / free queue
```

当前 baseline single-group，所以 Coordinator 较薄，但仍保留正确架构边界。

---

# 11. `reconcile_reclaimed_blocks()` 先只抓 5 步

核心算法：

```text
current = old + same_step_new

removed = old - retained

final = retained + same_step_new

publish final ownership

free removed
```

canonical example：

```text
current:
[B0 B1 B2 B3 B4 B5 | B6]

old:
[B0 B1 B2 B3 B4 B5]

new:
[B6]

retain:
[B0 B1 B4 B5]

removed:
[B2 B3]

final:
[B0 B1 B4 B5 B6]
```

其余大量 `if` 是 transaction validation，不是核心算法。

---

# 12. `block_id`、`position`、`KVCacheBlock object`

假设：

```text
old_blocks:
[B10, B37, B8, B21, B55]
```

`block_id`：

```text
10,37,8,21,55
```

是 physical block identity。

`old_positions`：

```text
10→0
37→1
8→2
21→3
55→4
```

这里的 position 只是：

> block 在当前 request `old_blocks` Python list 中的 index。

它不是：

```text
RoPE position
cache_position
token position
GPU address
```

作用只有：

```text
1. 验证 retained ID 属于 old ownership
2. 验证 retained 保持 old order
3. 根据 retained IDs 找回 KVCacheBlock objects
```

这个 position 不需要传出去。

---

# 13. retained order 为什么必须校验

old：

```text
[B10 B37 B8 B21 B55]
```

合法：

```text
retain:
[B10 B8 B55]

positions:
[0,2,4]
```

非法：

```text
retain:
[B8 B10 B55]

positions:
[2,0,4]
```

后者已经是 reorder，不是 reclaim。

所以 retained 必须是 old row 的：

```text
ordered subsequence
```

---

# 14. `final_ids` 和 `num_cached_block`

`final_ids`：

```python
[*retained_block_ids, *same_step_new_block_ids]
```

只是 local validation temporary，用来检查 final ownership 是否出现重复 physical ID。

`num_cached_block`：

```python
self.num_cached_block[request_id] = min(
    self.num_cached_block[request_id],
    len(final_blocks),
)
```

只是保证：

```text
num_cached_block <= final ownership length
```

当前 prefix caching OFF，所以不要把它理解成 reclaim 后精确重建 prefix-cache metadata。

---

# 15. 为什么先 publish ownership 再 free

禁止出现：

```text
req_to_blocks[A] 仍包含 B2
```

同时：

```text
B2 已进入 BlockPool free queue
```

否则：

```text
A canonical owns B2
B allocator also gets B2
```

所以 transaction 顺序：

```text
validate all
    ↓
build final_blocks
    ↓
req_to_blocks[A] = final_blocks
    ↓
BlockPool.free_blocks(removed)
```

---

# 16. T3-B — 为什么只 free 不改 allocator 一定失败

commit 后：

```text
L = 97
E = 65
owned blocks = 5
block_size = 16
```

下一轮：

```text
q = 1
```

如果继续按 logical：

```text
97 + 1 = 98
ceil(98/16) = 7
```

会错误再申请：

```text
7 - 5 = 2 blocks
```

刚 reclaim 两块，下一轮立刻补回来。

正确 physical：

```text
65 + 1 = 66
ceil(66/16) = 5
```

已有 5 blocks：

```text
new allocation = 0
```

---

# 17. `allocate_slots()` 的顺序与 P1 修改点

建议按 8 阶段读：

```text
1. 参数合法性
2. logical/cache computed bookkeeping
3. watermark / full-sequence admission
4. physical slot requirement       ← P1 seam
5. remove_skipped_blocks
6. get_num_blocks_to_allocate
7. capacity gate + allocate_new_blocks
8. cache metadata commit
```

数据流：

```text
request.num_computed_tokens
        ↓
num_local_computed_tokens
        ↓
total_computed_tokens
        │
        ▼
allocation_base
        ├─ E is None → total_computed_tokens
        └─ E exists  → effective_kv_len
        ↓
+ num_new_tokens
        ↓
num_tokens_main_model
        ↓
num_tokens_need_slot
        ↓
get_num_blocks_to_allocate()
        ↓
allocate_new_blocks()
```

P1 核心 semantic diff：

```python
allocation_base = (
    request.effective_kv_len
    if request.effective_kv_len is not None
    else total_computed_tokens
)

num_tokens_main_model = allocation_base + num_new_tokens
```

没有重写底层 allocator。

只是修改：

```text
physical capacity planning 的输入基准
```

---

# 18. 为什么修改恰好放这里

前面：

```text
logical/cache bookkeeping
```

后面：

```text
physical slot requirement
```

所以这是：

```text
logical/cache state
        ↓
选择 L or E
        ↓
physical capacity planning
```

最窄的 semantic seam。

底层 allocator 不需要知道 reclaim。

它只需要得到正确的：

```text
num_tokens_need_slot
```

---

# 19. `update_from_output()`：代码很多，但 T3 只占两块

整体：

```text
1. 解包 outputs
2. completion/deferred-free bookkeeping
3. step-level side data
4. per-request loop
5. request result commit
6. stop/finish cleanup
7. connector/events/stats
8. EngineCoreOutputs
```

T3 新增核心：

```text
A. reclaim_transitions map
B. per-request reconcile + E commit/advance
```

---

# 20. reclaim map 为什么这样构造

SchedulerOutput 里是 parallel arrays：

```text
req_ids
reclaim_transitions
new_block_ids
```

所以先构造：

```text
request_id
→
(transition, same-step new allocation)
```

后面 per-request loop：

```python
reclaim_state = reclaim_transitions.get(req_id)
```

直接查本 request 的 transaction。

---

# 21. `update_from_output()` 的三态 physical state machine

```text
Case 1: 本轮有 reclaim
    ↓
reconcile ownership
    ↓
free removed
    ↓
E = E_target + q

Case 2: 以前 reclaim 过，本轮 normal
    ↓
E += q

Case 3: 从未 reclaim
    ↓
E is None
    ↓
完全 upstream
```

为什么先 reconcile 再写 E：

```text
validation fail
→ no ownership mutation
→ no free
→ no Scheduler E commit
```

这就是 fail-closed。

---

# 22. E_target 与 E_completed

```text
E_target = 62
```

是本轮 forward-entry physical frontier。

q=3：

```text
physical writes:
62,63,64
```

完成：

```text
E_completed = 65
```

Scheduler 下一轮 allocator 应保存：

```text
65
```

不是 62。

---

# 23. Preempt 怎么理解

Preempt：

> request 没结束，但 Scheduler 为释放 KV/resource，把当前 execution KV state 全部丢掉，回 waiting，后续重新计算。

状态：

```text
RUNNING
   ↓
PREEMPTED / waiting
   ↓
later recompute
RUNNING
```

核心：

```text
free current KV ownership
num_computed_tokens = 0
effective_kv_len = None
requeue
```

为什么 `E=None`：

```text
preempt 后旧 physical KV 已不存在
```

如果：

```text
L=0
E=65
ownership=[]
```

就是 stale physical state。

`None` 的语义是：

```text
没有独立 physical mode
→ allocator 回 upstream logical path
```

---

# 24. T3 transaction 状态机

```text
IDLE
 │ prepare
 ▼
PREPARED
 │ schedule succeeds
 ▼
PUBLISHED
 │ Worker T2
 ▼
EXECUTING
 │ result
 ▼
EXECUTED
 │ update_from_output
 ▼
VALIDATING
 ├─ fail → keep old ownership / no free / no E commit
 └─ pass
      ↓
   COMMITTED
      ├─ dense canonical row
      ├─ removed refcnt→0/free queue
      └─ Scheduler E_completed
```



# 25. Test architecture：先学 pytest 在这里怎么“做实验”

pytest 在这份文件里只需要理解三个语法：

```python
def test_xxx():
```

表示一个 test。

```python
assert actual == expected
```

表示：

```text
actual 必须满足设计预期
```

否则 FAIL。

```python
with pytest.raises(ValueError):
```

表示：

```text
这段代码必须抛 ValueError
```

否则 FAIL。

所以这里真正难的不是 pytest，而是：

```text
如何构造 Scheduler state
如何 fake 掉不需要测试的 runtime
如何选择能证明 correctness 的 oracle
```

---

# 26. `_model_output()`：为什么可以假装 GPU 已经执行完

测试：

```python
def _model_output(
    request_id: str,
    sampled_token_ids: list[int]
) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=[request_id],
        req_id_to_index={request_id: 0},
        sampled_token_ids=[sampled_token_ids],
    )
```

例如：

```python
_model_output("request-a", [7])
```

不是 Qwen3 真的生成了 7。

它只是告诉 Scheduler：

```text
“假设 request-a 这一 execution step 已经完成，
现在这是最小合法 ModelRunnerOutput。”
```

因为当前 core test 的 SUT 是：

```text
Scheduler
KVCacheManager
SingleTypeKVCacheManager
BlockPool
```

而不是模型数学正确性。

---

# 27. `_create_reclaim_scheduler()`：为什么参数这样设计

```text
block_size = 16
num_blocks = 8
max_num_batched_tokens = 94
max_model_len = 128
max_num_seqs = 2
```

这里不是生产 workload。

它是人为构造的“小型 Scheduler 实验台”。

关键设计：

```text
Request A prompt = 97 tokens
max batch tokens = 94
```

所以：

```text
schedule #1:
94 tokens

schedule #2:
3 tokens
```

而：

```text
ceil(94 / 16) = 6 blocks
```

这样第一轮能稳定建立：

```text
6-block old ownership
```

第二轮再注入 reclaim。

---

# 28. `_schedule_reclaim_step()` 的真实目标

它不是“跑完 request”。

它的目标是：

> **把系统精确推进到 reclaim transaction 已经 schedule、same-step allocation 已经发生、但 result-time commit 还没有发生的时刻。**

即：

```text
prepare 已完成
schedule 已完成
transport 已完成
BUT
update_from_output 尚未执行
```

---

# 29. `_schedule_reclaim_step()` 逐步状态

创建：

```text
Request A
prompt = 97
```

第一次：

```python
first_output = scheduler.schedule()
```

因为 batch cap：

```text
schedule 94
```

于是 ownership 需要：

```text
ceil(94/16)=6 blocks
```

可以记成：

```text
[B0 B1 B2 B3 B4 B5]
```

然后测试直接：

```python
scheduler.update_from_output(
    first_output,
    _model_output(request_id, [])
)
```

表示：

```text
假装第一轮 94-token prefill 已经执行完成
```

读取：

```text
old_row = [B0 B1 B2 B3 B4 B5]
```

然后人工注入：

```text
retain positions:
0,1,4,5

E_target:
62
```

即：

```text
retain:
[B0 B1 B4 B5]

remove:
[B2 B3]
```

第二次：

```python
reclaim_output = scheduler.schedule()
```

剩：

```text
97 - 94 = 3 tokens
```

这轮：

```text
q=3
```

同一 step normal allocator 又追加：

```text
B6
```

所以 helper return 前的状态：

```text
Scheduler canonical:
[B0 B1 B2 B3 B4 B5 B6]

prepared reclaim:
retain B0 B1 B4 B5
remove B2 B3

same-step new:
B6

Scheduler effective_kv_len:
None

B2/B3:
still owned
```

这就是 Test1 的起点。

---

# 30. Test1 的问题定义

```python
test_reclaim_commit_releases_and_reuses_dense_blocks()
```

它不是只测一个函数。

它一次证明 4 个层级：

```text
1. schedule 阶段不能 premature free
2. update_from_output 后 ownership / BlockPool 真 commit
3. 下一轮 allocator 不会把 reclaim 容量补回来
4. 释放出来的 block 真能被别的 request 复用
```

所以它是一个 control-plane end-to-end core oracle。

---

# 31. Test1 逐行：Arrange

```python
scheduler = _create_reclaim_scheduler()
```

造 Scheduler under test。

---

```python
request, old_row, new_block_id, reclaim_output = (
    _schedule_reclaim_step(scheduler)
)
```

执行完后固定：

```text
old_row:
[B0 B1 B2 B3 B4 B5]

same-step new:
B6

current canonical:
[B0 B1 B2 B3 B4 B5 B6]

target final:
[B0 B1 B4 B5 B6]

removed:
[B2 B3]

E_target:
62
```

---

```python
pool = scheduler.kv_cache_manager.block_pool
```

只是取得 BlockPool 引用。

---

```python
free_before_commit = pool.get_num_free_blocks()
```

拍 commit 前 free capacity 快照：

```text
N
```

---

# 32. Test1：证明 schedule 不能提前 commit

```python
assert scheduler.kv_cache_manager.get_block_ids(
    request.request_id
)[0] == [
    *old_row,
    new_block_id,
]
```

`*old_row` 是 list unpack。

所以期望：

```text
[B0 B1 B2 B3 B4 B5 B6]
```

这证明：

```text
schedule 完成后 canonical 仍是 old + same-step new
```

还没有 dense reconcile。

---

```python
assert pool.get_num_free_blocks() == free_before_commit
```

证明：

```text
B2/B3 没有提前进入 free pool
```

---

```python
assert request.effective_kv_len is None
```

证明：

```text
Scheduler persistent E 也没提前 commit
```

三者共同证明：

```text
schedule / transport
≠
result-time commit
```

---

# 33. Test1：真正触发 commit

```python
scheduler.update_from_output(
    reclaim_output,
    _model_output(request.request_id, [7])
)
```

翻译：

```text
“假装本轮 Worker/GPU 已完成，现在让 Scheduler 正式处理结果。”
```

这里才进入：

```text
reconcile_reclaimed_blocks()
+
Scheduler E commit
```

---

# 34. Test1：验证 dense ownership

```python
removed_ids = {
    old_row[2],
    old_row[3],
}
```

即：

```text
{B2,B3}
```

---

```python
final_row = [
    old_row[0],
    old_row[1],
    old_row[4],
    old_row[5],
    new_block_id,
]
```

即：

```text
[B0 B1 B4 B5 B6]
```

---

```python
assert request.effective_kv_len == 65
```

因为：

```text
E_target=62
q=3
E_completed=65
```

---

```python
assert get_block_ids(...) == final_row
```

证明：

```text
canonical ownership
=
[B0 B1 B4 B5 B6]
```

---

# 35. Test1：验证不是 metadata-only reclaim

```python
assert pool.get_num_free_blocks() == free_before_commit + 2
```

证明：

```text
BlockPool 真收回 B2/B3 两个 physical slots
```

不是只改 Python list。

---

```python
assert all(
    pool.blocks[block_id].ref_cnt == 0
    for block_id in removed_ids
)
```

可以直接理解成：

```text
B2.ref_cnt == 0
AND
B3.ref_cnt == 0
```

证明 refcount release 真发生。

---

# 36. Test1：为什么还必须再 schedule 一次

```python
next_output = scheduler.schedule()
```

这是在验证 T3-B。

当前：

```text
L≈97
E=65
owned blocks=5
```

下一轮 decode：

```text
q=1
```

如果用 physical E：

```text
65+1=66
ceil(66/16)=5
```

已有 5 blocks：

```text
new allocation=0
```

如果错误继续用 logical L：

```text
97+1=98
ceil(98/16)=7
```

会 bogus allocate 2 blocks。

---

# 37. `cached_index` 到底是什么

```python
cached_index = (
    next_output.scheduled_cached_reqs.req_ids
    .index(request.request_id)
)
```

`scheduled_cached_reqs` 里的字段是 parallel arrays。

所以先找到 Request A 在：

```text
req_ids
```

里的位置，再用同一个 index 去取：

```text
new_block_ids[cached_index]
```

它只是 list index lookup，不是 KV position。

---

# 38. Test1：T3-B decisive oracle

```python
assert next_output.num_scheduled_tokens[
    request.request_id
] == 1
```

证明 normal decode q=1。

---

```python
assert (
    next_output.scheduled_cached_reqs
    .new_block_ids[cached_index]
    is None
)
```

这是非常重要的 assert：

> **下一轮没有申请新 block。**

它直接验证：

```text
allocate_slots()
已经按 E=65
而不是 L≈97
做 capacity planning
```

---

```python
assert get_block_ids(...) == final_row
```

进一步证明 ownership 没被 bogus append。

---

# 39. Test1：E 不能只 commit 一次

```python
scheduler.update_from_output(
    next_output,
    _model_output(request.request_id, [8])
)
```

fake normal decode completion。

---

```python
assert request.effective_kv_len == 66
```

验证：

```text
post-reclaim normal step:
E += q

65 → 66
```

如果没有：

```python
elif request.effective_kv_len is not None:
    request.effective_kv_len += num_tokens_scheduled
```

Scheduler E 会永远停在 65。

---

# 40. Test1：为什么最后还造 Request B

```python
(other_request,) = create_requests(
    num_requests=1,
    num_tokens=1,
    req_ids=["request-b"],
    block_size=16,
)
```

1 token：

```text
需要 1 KV block
```

加入 Scheduler：

```python
scheduler.add_request(other_request)
```

再 schedule：

```python
other_output = scheduler.schedule()
```

读取 B 的 ownership：

```python
allocated_to_other = get_block_ids("request-b")[0]
```

---

```python
assert len(allocated_to_other) == 1
```

1 token → 1 block。

---

```python
assert allocated_to_other[0] in removed_ids
```

这是 Test1 最终 decisive oracle。

前面：

```text
removed_ids={B2,B3}
```

现在 Request B 真拿到：

```text
B2 或 B3
```

所以完整闭环：

```text
A unlink B2/B3
    ↓
refcnt→0
    ↓
free queue
    ↓
B normal allocation
    ↓
得到相同 physical block ID
```

这比：

```text
free count +2
```

证据更强。

---

# 41. Test1 一张图复习

```text
create Scheduler under test
        ↓
Request A: 97 tokens
        ↓
schedule #1
        ↓
94 tokens
        ↓
6 blocks:
[B0 B1 B2 B3 B4 B5]
        ↓
fake completion
        ↓
prepare reclaim:
retain 0,1,4,5
E_target=62
        ↓
schedule #2
        ↓
q=3
same-step new B6
        ↓
commit 前 canonical:
[B0 B1 B2 B3 B4 B5 B6]
        ↓
assert:
no premature shrink
no premature free
E=None
        ↓
fake completion
        ↓
update_from_output
        ↓
T3 commit
        ↓
canonical:
[B0 B1 B4 B5 B6]
        ↓
B2/B3 refcnt→0
free capacity +2
E=65
        ↓
schedule #3
        ↓
q=1
no new block
        ↓
fake completion
        ↓
E=66
        ↓
create Request B
        ↓
B gets B2 or B3
        ↓
REAL BLOCK REUSE
```

---

# 42. Test2 — fail-closed

测试故意：

```python
transition.expected_old_num_blocks += 1
```

让 prepared snapshot 和 current canonical 不一致。

预期抛：

```text
ValueError("Canonical block count...")
```

但真正重要的是异常后：

```text
effective_kv_len 仍 None
canonical 仍 old + new
free capacity 不变
B2/B3 refcnt 仍 1
```

即：

```text
bad transaction
→ no partial commit
```

---

# 43. Test3 — reclaim 后再 preempt

成功 commit：

```text
canonical:
[B0 B1 B4 B5 B6]

B2/B3:
already free
```

后续 preempt 必须只 free：

```text
当前 canonical
```

不能再次碰 B2/B3。

同时：

```text
effective_kv_len → None
```

这个 test 验证：

```text
reclaim
+
upstream preemption lifecycle
```

可组合，不 double-free，不留下 stale E。

---

# 44. Test4 — normal path regression

普通 request：

```text
effective_kv_len=None
```

17 tokens：

```text
ceil(17/16)=2 blocks
```

验证 P1 修改没有让普通 request 进入 physical-decoupled allocation path。

---

# 45. 四个 core tests 的工程分类

| Test | 类型 | 问题 |
|---|---|---|
| Test1 reclaim/reuse | Happy path / core E2E | 成功路径是否真 release + reuse？ |
| Test2 validation failure | Negative / fail-closed | 坏 transaction 是否不污染状态？ |
| Test3 preemption | Lifecycle interaction | reclaim 后 upstream preempt 是否安全？ |
| Test4 normal request | Regression / fallback | 新 physical path 是否破坏普通路径？ |

以后看 vLLM test，先问：

```text
1. SUT 是谁？
2. Arrange 在制造什么状态？
3. Act 触发哪个 production path？
4. Assert 是哪个 invariant 的 oracle？
5. 哪些 runtime components 被 fake？
```



# 46. Core test 与真实 Engine smoke 的证据分工

Core test 证明：

```text
Scheduler
KVCacheManager
SingleType ownership
BlockPool refcount/free
effective_kv_len accounting
preemption
allocator reuse
```

Real Engine smoke 证明：

```text
SchedulerOutput
→ Worker T2
→ GPUModelRunner BlockTables
→ real Qwen forward
→ ModelRunnerOutput
→ Scheduler T3 commit
```

所以：

```text
core tests
=
deterministic control-plane oracle

integration smoke
=
真实 runtime closure
```

两者不能互相替代。

---

# 47. Real Engine integration closure

冻结环境：

```text
GPU: A100 80GB
model: Qwen3-0.6B
MRV2
BF16
FA2
eager
block_size=16
single KV group
TP/PP/DP/DCP/PCP=1
prefix/spec/async/CUDAGraph=OFF
```

真实观察：

```text
Scheduler old row:
[1,2,3,4,5,6]

Worker dense row:
[1,2,5,6]

E_target:
62

real forward q:
1

Worker E_completed:
63

Scheduler E after commit:
63

Scheduler final canonical:
[1,2,5,6]

released:
[3,4]

refcnt:
3→0
4→0

free capacity:
10860→10862

Request B:
allocated [4]
```

并且原 Request A 继续 generation。

这证明真实链：

```text
prepared reclaim
→ Worker dense execution
→ real model forward
→ result-time Scheduler commit
→ canonical shrink
→ BlockPool reuse
→ another request gets reclaimed physical ID
```

---

# 48. “physical free” 的精确定义

allocator 层级：

```text
Request ownership
        ↓
vLLM req_to_blocks
        ↓
vLLM BlockPool
        ↓
preallocated GPU KV tensor slots
        ↓
PyTorch CUDA allocator
        ↓
CUDA driver / HBM
```

T3 证明的是：

```text
vLLM BlockPool level
```

的 physical slot reclaim/reuse。

没有证明：

```text
shrink whole KV tensor
cudaFree
HBM returned to driver/OS
```

因此项目中的 “physical reclamation” 应精确表述为：

> **reclaim and reuse physical KV block slots inside vLLM's preallocated KV cache pool.**

---

# 49. Production patch semantic map

| 文件 / 函数 | upstream responsibility | P1/T3 semantic diff | 对应问题 |
|---|---|---|---|
| `vllm/v1/request.py` | Scheduler request state | 增加 `effective_kv_len` | T3-B |
| `KVCacheManager.allocate_slots()` | capacity planning | divergence 后以 completed physical E 为 allocation base | T3-B |
| `KVCacheManager.reconcile_reclaimed_blocks()` | request-level KV API | narrow dense reconcile public entry | T3-C |
| `KVCacheCoordinator.reconcile_reclaimed_blocks()` | group coordination | forwarding / current single-group guard | T3-C |
| `SingleTypeKVCacheManager.reconcile_reclaimed_blocks()` | canonical ownership | validate → dense row → free | T3-C |
| `BlockPool.free_blocks()` | allocator release | upstream refcount release/reuse | T3-C |
| `Scheduler.update_from_output()` | result-time lifecycle | completion seam + ownership/E commit | T3-A/B/C |
| `Scheduler._preempt_request()` | recompute preemption | `effective_kv_len=None` | lifecycle |
| Worker `RequestState.effective_kv_len` | execution state | physical write/visibility frontier | S1/S2/S3 |
| Worker `BlockTables` | forward addressing | dense execution row | T2 |

---

# 50. T3 核心 invariants

## I1 — Logical progress 仍是 model truth

```text
L = num_computed_tokens
```

reclaim 不把模型逻辑历史改写成 compact physical layout。

---

## I2 — Physical frontier 只通过 explicit reclaim 跳变

normal：

```text
E_next = E + committed_delta
```

reclaim：

```text
E_old
→ E_target
→ E_completed
```

---

## I3 — Worker physical row / cache positions / visibility 必须一致

```text
BlockTables
cache_positions
effective_kv_seq_lens
slot_mapping
```

必须共同解释同一个 compact physical address space。

---

## I4 — canonical ownership 由 KV manager hierarchy 修改

Scheduler 决定 transaction。

KV manager 执行 ownership mutation。

---

## I5 — reusable block 绝不能仍 canonical-owned

```text
block ∈ BlockPool free queue
⇒
block ∉ req_to_blocks[request]
```

---

## I6 — future allocation 必须从 completed E 增长

否则 logical history 会把 reclaimed capacity 补回来。

---

## I7 — recompute preemption invalidates E

```text
whole-request KV free
⇒
effective_kv_len=None
```

---

# 51. Failure semantics

当前原则：

```text
temporary retention
>
corruption risk
```

## Allocation / schedule failure

prepared reclaim 没真正进入 scheduled transaction：

```text
不 transport
不 Worker transition
不 commit
不 free
```

## Execution failure

没有成功走到 matching：

```text
update_from_output()
```

则：

```text
no ownership shrink
no free
```

## Validation failure

任何 transaction mismatch：

```text
no ownership mutation
no BlockPool free
no Scheduler E mutation
```

这就是 fail-closed。

---

# 52. 为什么不需要 Worker ACK / freed IDs

Scheduler 本来就是 reclaim decision producer。

它已经知道：

```text
old canonical
retained old blocks
expected old count
same-step new blocks
E_target
```

到 matching `update_from_output()` 时，可以确定性重建：

```text
final row
removed blocks
E_completed
```

所以当前 baseline 不需要额外：

```text
reclaim_freed_ids
Worker reclaim ACK
generation protocol
```

这是一个 deliberate minimal design，而不是遗漏。

---

# 53. 为什么不用 force-free

P1 保留 vLLM 原有：

```text
KVCacheBlock refcount
BlockPool.free_blocks()
```

语义。

removed block 正常：

```text
ref_cnt: 1 → 0
```

进入 free queue。

没有绕过 allocator ownership model 做 force-free。

---

# 54. 源码学习顺序

不要按文件从头看到尾。

## 第一轮 — execution completion

看：

```text
EngineCore.step()
Scheduler.schedule()
execute_model()
future.result()
Scheduler.update_from_output()
```

只回答：

```text
为什么 update_from_output 是 result-time seam？
```

## 第二轮 — ownership

看：

```text
KVCacheManager
KVCacheCoordinator
SingleTypeKVCacheManager.req_to_blocks
BlockPool
```

只回答：

```text
谁真正拥有 physical block？
```

## 第三轮 — allocation

看：

```text
KVCacheManager.allocate_slots()
num_tokens_main_model
num_tokens_need_slot
get_num_blocks_to_allocate()
allocate_new_blocks()
```

只回答：

```text
为什么 L 会造成 bogus allocation？
为什么 P1 只改 allocation_base？
```

## 第四轮 — T3 commit

看：

```text
SchedulerOutput.reclaim_transitions
SchedulerOutput.new_block_ids
Scheduler.update_from_output()
reconcile_reclaimed_blocks()
effective_kv_len
```

只回答：

```text
prepare → execute → commit 如何闭环？
```

## 第五轮 — tests

看：

```text
tests/v1/core/utils.py:create_scheduler
tests/v1/core/utils.py:create_requests
_model_output()
_schedule_reclaim_step()
Test1
Test2
Test3
Test4
```

只回答：

```text
哪些 runtime 被 fake？
每个 assert 对应哪个 invariant？
```

---

# 55. 面试 / 简历表达

可以这样讲：

> Worker-side block-table compaction 本身并不等于真实 memory reclamation。  
> 我进一步审计 vLLM 的 scheduler completion、canonical KV ownership 和 block allocator 路径，把问题拆成三个 runtime contract：
>
> 1. **Safe-free lifetime**：不能在 Worker block table 切换时立即复用 removed block，需要等对应 execution transaction completion；
> 2. **Canonical ownership reconciliation**：result-time 将 `req_to_blocks` 从旧 row 更新为 compact dense row，再通过 upstream refcounted `BlockPool.free_blocks()` 释放 removed blocks；
> 3. **Physical allocation accounting**：reclaim 后 logical progress 与 physical occupancy 分离，因此 future allocation 从 completed physical frontier `E` 增长，而不是继续只看 `num_computed_tokens`。
>
> 最终闭环：
>
> ```text
> Scheduler reclaim decision
> → Worker dense execution
> → real Qwen forward
> → result-time commit
> → ownership shrink
> → BlockPool real reuse
> → next-step physical allocation
> ```
>
> 并通过 core tests 验证 fail-closed、preemption/no-double-free、normal-path regression，再用真实 Engine smoke 验证 reclaimed physical block ID 被另一 request 实际重新分配。

---

# 56. 一页复习图

```text
                     BEFORE T3

Worker:
[B0 B1 B4 B5 B6]
E = compact physical frontier

Scheduler:
[B0 B1 B2 B3 B4 B5 B6]

BlockPool:
B2/B3 still owned

future allocator:
may still use logical L


                         ↓

                      T3-A
              execution completion seam

                         ↓

                      T3-C
            canonical dense reconciliation

[B0 B1 B2 B3 B4 B5 B6]
                ↓
[B0 B1 B4 B5 B6]

B2/B3:
refcnt → 0
free queue

                         ↓

                      T3-B
              completed physical frontier

E_target + q
=
E_completed

next allocate_slots()
uses E

                         ↓

                    FINAL STATE

Worker execution row
=
Scheduler canonical row

removed physical slots
=
BlockPool reusable

future request
=
can get same physical block ID

future allocation
=
does not reconstruct logical history
```

自测：

1. 为什么 T2 Worker row 更新后不能立刻 `free_blocks()`？
2. 为什么 `BlockTables` 不是 canonical ownership？
3. 为什么 `reclaim_transition` 和 `same_step_new_block_ids` 要保留不同语义？
4. `reconcile_reclaimed_blocks()` 的核心为什么只有 `removed=old-retained` 与 `final=retained+new`？
5. 为什么 `allocate_slots()` 只改 `allocation_base` 就能修复 bogus re-allocation？
6. 为什么 Test1 让 Request B 拿到 reclaimed physical block ID 是最强 oracle？

---

# 57. Evidence / Closure

当前状态：

```text
P1-M1-T3-IMPL-01A
PASS / WEB_REVIEW_ACCEPTED

P1-M1-T3-INTEGRATION-CLOSURE
PASS / WEB_REVIEW_ACCEPTED

P1-M1-T3
CLOSED
```

Evidence basis：

- `04-experiments/project1_kv_reclaim/notes/P1-M1-T3-IMPL-01A-Physical-Ownership-Reconciliation-实施记录.md`
- `04-experiments/project1_kv_reclaim/raw/m1-t3-impl-01a/m1-t3-impl-01a.log`
- `04-experiments/project1_kv_reclaim/notes/P1-M1-T3-INTEGRATION-CLOSURE-真实Engine生命周期验证报告.md`
- `04-experiments/project1_kv_reclaim/raw/m1-t3-integration-closure/m1-t3-integration-closure.log`
- `04-experiments/project1_kv_reclaim/notes/P1-CODE-DIFF-TRACKER.md`

Core evidence：

```text
Targeted: 4 passed
Regression: 39 passed
Combined: 43 passed
py_compile: PASS
git diff --check: PASS
```

当前 T3 已证明：

```text
safe free
dense canonical ownership reconciliation
refcount-safe BlockPool release
real physical-ID reuse
Scheduler completed physical E
no bogus next allocation
fail-closed validation
preemption reset/no double-free
normal path regression preserved
real Engine lifecycle closure
```

当前仍未证明：

```text
prefix caching
spec decode
async scheduling
PP>1
multi-group KV
DCP/PCP expansion
KV connector/offload
CUDA Graph
token-level compaction
Triton gather/scatter
general repeated reclaim policy
performance claim
HBM returned to CUDA allocator/OS
```

下一阶段：

```text
P1 V1/Core Gate Review
```

统一审查：

```text
S1
S2
S3
T2
T3
```

是否已经形成完整、可解释、可验证的 whole-block physical KV reclamation core。
