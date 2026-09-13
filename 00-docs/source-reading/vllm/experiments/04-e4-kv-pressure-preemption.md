# E4-vLLM-KV压力、准入等待与Preemption动态实验

> 实验阶段：vLLM 三周定向桥接学习 / KV 控制面动态验证  
> 实验编号：E4  
> 状态：PASS / Frozen  
> vLLM：v0.26.0  
> commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> 模型：`/data/models/Qwen3-0.6B`  
> 硬件：A100 80GB PCIe  
> 并行配置：TP=1 / PP=1 / DP=1  
> Engine：V1 Engine  
> Model Runner：V2 Model Runner  
> 调度：Async Scheduling + Batch Queue  
> 日期：2026-08-19

---

# 0. 实验结论摘要

E4 的目标不是单纯“制造一次 OOM”或“看到一次 preemption”，而是动态验证 vLLM V1 在 **KV Cache 容量不足** 时完整的控制面行为：

```text
KV capacity
    ↓
waiting request admission
    ↓
KVCacheManager.allocate_slots()
    ↓
full-sequence admission gate / current-step allocation gate
    ↓
allocation reject
    ↓
Scheduler victim selection
    ↓
preemption
    ↓
KV block ownership release
    ↓
request logical frontier reset
    ↓
PREEMPTED → WAITING
    ↓
resume / recompute
    ↓
eventual completion
```

实验最终完成三个阶段：

```text
Probe
→ 单请求真实 peak KV = 34 blocks

E4-A
→ scheduler_reserve_full_isl=True
→ 第二请求在 WAITING 阶段被 ALLOC_REJECT_FULL 拦截
→ 不发生 preemption
→ 第一请求完成后第二请求再 admission

E4-B
→ scheduler_reserve_full_isl=False
→ 第二请求被允许部分 Prefill 进入 RUNNING
→ 后续 KV 扩张时 ALLOC_REJECT_STEP
→ 第二请求被 preempt
→ KV block 立即归还
→ num_computed_tokens 清零
→ 回 WAITING
→ 重新 admission / recompute
→ 共发生 4 次 preemption
```

E4-A 与 E4-B 的核心区别可以概括为：

```text
E4-A：
宁可 WAITING，也不让一个完整输入明显装不下的请求过早进入 RUNNING。

E4-B：
当前 chunk 能放就先 admission；后续资源不够时再 preempt/recompute。
```

因此 E4 动态证明了：

> `scheduler_reserve_full_isl` 本质上是一个 KV admission policy。它把“是否有足够 KV 生存空间”的判断前移到 waiting admission 阶段，以降低 chunked prefill 在高 KV 压力下过度准入后产生的 preemption / recomputation / thrashing。

同时 E4-B 还进一步证明：

> Preemption 不是简单的“暂停请求”，而是 RUNNING request 的 KV ownership 释放 + logical computed frontier 重置 + PREEMPTED/WAITING requeue + 后续 recompute。

---

# 1. 为什么需要 E4

E1/E2/E3 已经分别回答：

```text
E1：单请求 Prefill → Decode 的 async scheduling 主链
E2：Chunked Prefill 如何受 token budget 切分，并持续增长 KV
E3：多个请求如何共享一个 global token budget，形成 continuous batching
```

但是这三个实验中的 KV Cache 都足够大，因此没有回答生产推理中非常重要的一组问题：

```text
如果多个请求总 KV 需求超过当前 GPU KV pool，会发生什么？

Scheduler 会让请求一直 WAITING，还是先让它运行？

allocate_slots() 在什么阶段返回 None？

WAITING 和 PREEMPTED 的区别是什么？

被 preempt 的 request 来自 waiting 还是 running？

preemption 后原先已经计算的 Prompt progress 是否保留？

KV block 什么时候归还 BlockPool？

Async scheduling 下，如果旧 step 尚未 reconcile，preemption 怎么维护状态一致性？
```

这些问题直接关系后续 QCache / KV compression / KV migration 项目，因为后续任何双池、压缩、迁移策略都必须建立在正确的生命周期认知上：

```text
admission
allocation
ownership
free
preemption
recompute
```

不能把它们混在一起。

---

# 2. 实验设计来源与 DoD

三周桥接手册对 E4 的定义是：

```text
先测单请求真实 block 数 B_single

num_gpu_blocks_override = ceil(1.25 * B_single)

并发两个相同长度请求，使：
总需求 > override

E4-A：scheduler_reserve_full_isl=true
观察第二请求因完整输入无法容纳而停留 waiting

E4-B：scheduler_reserve_full_isl=false
允许 over-admission
观察运行期 block pressure / preemption / recompute
```

要求记录：

```text
admission
allocation failure
waiting / preempted request ID
released blocks
num_preemptions
recompute / re-enter waiting
最终是否完成
```

实验停止条件不是“看起来像发生了抢占”，而是必须有真实 trace 把：

```text
allocation reject
→ preemption
→ block free
→ request state reset
→ waiting
→ recompute
```

串起来。

---

# 3. 固定环境

## 3.1 代码与运行环境

```text
Repo:
~/learning/llm-kv-lab/third_party/vllm

vLLM:
v0.26.0

Commit:
568afb3a13806beb53bb2e6bd518269357b237c0

Model:
/data/models/Qwen3-0.6B

GPU:
A100 80GB PCIe

TP=1
PP=1
DP=1
```

环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=1
export QCACHE_BRIDGE_TRACE=1
```

其中：

```text
QCACHE_BRIDGE_TRACE
```

是本学习分支自行增加的 instrumentation 开关，并不是 vLLM 官方环境变量。

---

## 3.2 Runtime 初始化确认

实际日志确认：

```text
Asynchronous scheduling is enabled
Using V2 Model Runner
step_fn=step_with_batch_queue
batch_queue_size=2
batch_queue_enabled=True
FlashAttention version 2
```

因此 E4 发生在我们已经验证过的 async batch-queue 主链上：

```text
Scheduler.schedule()
↓
SchedulerOutput
↓
Executor / Worker / MRV2 execute
↓
AsyncOutputFuture
↓
batch_queue
↓
ModelRunnerOutput
↓
Scheduler.update_from_output()
```

E4 不改变这条 execution pipeline，只是把 KV pool 压小，观察 Scheduler / KVCacheManager 在资源不足时的控制面决策。

---

# 4. Workload 与实验参数

正式 E4 使用两个相同长度 Prompt。

真实 tokenizer 长度：

```text
Prompt = 526 tokens
```

配置：

```text
block_size = 16
max_num_batched_tokens = 128
max_num_seqs = 4
max_model_len = 640
enable_prefix_caching = False
enforce_eager = True
max_tokens = 8
ignore_eos = True
num_gpu_blocks_override = 43
```

最终 E4-A/B 日志中的 `gpu_memory_utilization` 为：

```text
0.3
```

但由于使用了：

```text
num_gpu_blocks_override = 43
```

本实验真正决定 KV capacity 的是 override，而不是自动 profiling 算出的 block 数。

日志确认：

```text
Overriding num_gpu_blocks=13185 with num_gpu_blocks_override=43
GPU KV cache size: 688 tokens
Maximum concurrency for 640 tokens per request: 1.07x
```

因为：

```text
43 blocks × 16 token/block = 688 token slots
```

---

# 5. Probe：为什么 B_single = 34

正式 A/B 实验之前，先运行单请求 Probe。

Prompt：

```text
526 tokens
```

Prompt-only 理论 block 数：

```text
ceil(526 / 16)
= 33 blocks
```

真实 trace 的 Prefill KV 增长：

```text
Call 0：128 tokens → blocks [1..8]
Call 1：128 tokens → blocks [9..16]
Call 2：128 tokens → blocks [17..24]
Call 3：128 tokens → blocks [25..32]
Call 4：14  tokens → block  [33]
```

所以完整 Prompt 确实需要：

```text
33 blocks
```

但 Decode 继续推进时，33 个 blocks 只有：

```text
33 × 16 = 528 token slots
```

当 logical computed frontier 从：

```text
528 → 529
```

时必须分配第 34 个 block。

Probe 中真实观察到：

```text
new_block_ids=([34],)
```

因此不能用 Prompt-only 的 33 作为完整 request peak。

最终：

```text
B_single = 34 blocks
```

按照手册：

```text
ceil(1.25 × 34)
= ceil(42.5)
= 43
```

所以正式 E4 使用：

```text
num_gpu_blocks_override = 43
```

这个值满足：

```text
一个 request peak ≈ 34
34 < 43
→ 单请求可以完成

两个 request peak ≈ 68
68 > 43
→ 两请求不能完整同时驻留
```

这正是我们需要的 KV contention window。

---

# 6. 第一次正式运行为什么启动失败

最初正式运行保持：

```text
max_model_len = 2048
num_gpu_blocks_override = 43
```

Engine 在初始化阶段直接拒绝启动。

原因不是 Scheduler，也不是 trace 修改错误，而是 vLLM 在创建 KV Cache 时有 capacity validation：

```text
43 blocks × 16 = 688 token slots

max_model_len = 2048

688 < 2048
```

也就是说：

> 当前 KV pool 连一个配置上允许的 `max_model_len=2048` 请求都容纳不了。

初始化日志给出的估计最大长度正好是：

```text
688
```

因此 EngineCore 尚未进入真实 Scheduler runtime 就退出。

## 6.1 修复方式

没有提高 `num_gpu_blocks_override`，因为那会削弱我们故意制造的 KV pressure。

而是把：

```text
max_model_len = 2048
```

改成：

```text
max_model_len = 640
```

因为：

```text
ceil(640 / 16) = 40 blocks
40 <= 43
```

初始化可以通过。

同时真实 workload：

```text
526 prompt + 8 output = 534 tokens
```

仍然满足：

```text
534 < 640
```

所以不会改变真实实验 workload。

## 6.2 这个失败本身说明什么

`num_gpu_blocks_override` 不仅改变 Scheduler 运行时可分配的 KV blocks，也进入 Engine 初始化阶段的 capacity validation。

因此做小 KV pool 压力实验时必须同时满足：

```text
实验 pool 至少能承载一个合法 max_model_len request
```

否则还没进入 Scheduler，实验就已经被配置检查挡住。

---

# 7. 为什么 E4 需要新增 instrumentation

E1/E2/E3 已经有：

```text
CORE CALL_BEGIN / SCHEDULE / EXEC / QUEUE / RECONCILE
SCHED QUEUE_SNAPSHOT
SCHED BEFORE_COMMIT / AFTER_COMMIT
SCHED SETTLE / TOKEN_APPEND
MRV2 EXECUTE_BEGIN / SAMPLE_BEGIN
```

这些 trace 能回答：

```text
谁被 schedule
schedule 多少 token
logical frontier 怎么推进
block IDs 怎么增长
async queue 如何流动
```

但无法回答 E4 的关键问题：

```text
为什么 allocate_slots() 返回 None？

是 full-sequence admission reject，还是 current-step reject？

哪一个 RUNNING request 被 preempt？

preempt 前拥有哪些 blocks？

blocks 是否真的归还 free pool？

是 immediate free 还是 deferred free？

preempt 后 request status / computed / preemption count 怎么变化？
```

因此 E4 只增加最小必要观测点，没有修改调度逻辑。

---

# 8. E4 Trace 设计总览

新增事件：

```text
KVCacheManager.allocate_slots()
├─ [BRIDGE][KV][ALLOC_REJECT_FULL]
└─ [BRIDGE][KV][ALLOC_REJECT_STEP]

Scheduler._preempt_request()
├─ [BRIDGE][SCHED][PREEMPT_BEGIN]
└─ [BRIDGE][SCHED][PREEMPT_END]

Scheduler._free_request_blocks()
├─ [BRIDGE][KV][FREE_IMMEDIATE_BEGIN]
├─ [BRIDGE][KV][FREE_IMMEDIATE_END]
└─ [BRIDGE][KV][FREE_DEFERRED]
```

这三组 trace 分别回答：

```text
allocation gate 为什么失败？
↓
谁被抢占、状态怎么变化？
↓
KV blocks 怎么释放？
```

---

# 9. Trace 1：`ALLOC_REJECT_FULL`

文件：

```text
vllm/v1/core/kv_cache_manager.py
```

目标位置是 `KVCacheManager.allocate_slots()` 的：

```python
if full_sequence_must_fit:
    ...
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > self.block_pool.get_num_free_blocks():
        return None
```

增加：

```python
if required_blocks > self.block_pool.get_num_free_blocks():
    if os.getenv("QCACHE_BRIDGE_TRACE", "0") == "1":
        print(
            f"[BRIDGE][KV][ALLOC_REJECT_FULL] "
            f"req={request.request_id} "
            f"status={request.status.name} "
            f"num_tokens={request.num_tokens} "
            f"num_computed={request.num_computed_tokens} "
            f"num_in_flight={request.num_in_flight_tokens} "
            f"num_new_tokens={num_new_tokens} "
            f"full_num_tokens={full_num_tokens} "
            f"blocks_to_allocate={num_blocks_to_allocate} "
            f"watermark_blocks={watermark_blocks} "
            f"required_blocks={required_blocks} "
            f"free_blocks={self.block_pool.get_num_free_blocks()}",
            flush=True,
        )
    return None
```

它回答：

```text
这个 request 当前还是 WAITING / PREEMPTED 吗？
整个 input sequence 需要多少 block？
当前 free 有多少？
为什么 full-sequence admission 被拒绝？
```

注意 `full_num_tokens` 来自：

```python
full_num_tokens = min(request.num_tokens, self.max_model_len)
```

所以这里的 `full` 是 **当前完整 input sequence**，不是“Prompt + 未来所有未知 output token”。

---

# 10. Trace 2：`ALLOC_REJECT_STEP`

同样位于：

```text
KVCacheManager.allocate_slots()
```

当前 step 的真实 allocation gate：

```python
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks

if required_blocks > available_blocks:
    return None
```

为了 trace，把 free count 单独保存：

```python
free_blocks = self.block_pool.get_num_free_blocks()
available_blocks = free_blocks - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
```

然后打印：

```python
if required_blocks > available_blocks:
    if os.getenv("QCACHE_BRIDGE_TRACE", "0") == "1":
        print(
            f"[BRIDGE][KV][ALLOC_REJECT_STEP] "
            f"req={request.request_id} "
            f"status={request.status.name} "
            f"num_tokens={request.num_tokens} "
            f"num_computed={request.num_computed_tokens} "
            f"num_in_flight={request.num_in_flight_tokens} "
            f"num_new_tokens={num_new_tokens} "
            f"num_tokens_need_slot={num_tokens_need_slot} "
            f"blocks_to_allocate={num_blocks_to_allocate} "
            f"watermark_blocks={watermark_blocks} "
            f"reserved_blocks={reserved_blocks} "
            f"required_blocks={required_blocks} "
            f"free_blocks={free_blocks} "
            f"available_blocks={available_blocks}",
            flush=True,
        )
    return None
```

它回答的是：

```text
request 已经 RUNNING 后
当前这一轮继续推进需要新增多少 block？
当前真正 available block 有多少？
为什么运行期 expansion 失败？
```

因此这两个 allocation trace 的语义必须严格区分：

```text
ALLOC_REJECT_FULL
= admission pressure
= 完整 input sequence gate

ALLOC_REJECT_STEP
= runtime allocation pressure
= 当前 step 的真实 KV expansion gate
```

---

# 11. Trace 3：`PREEMPT_BEGIN / PREEMPT_END`

文件：

```text
vllm/v1/core/sched/scheduler.py
```

函数：

```text
Scheduler._preempt_request()
```

原始逻辑已经明确：

```python
assert request.status == RequestStatus.RUNNING
self._free_request_blocks(request)
self.encoder_cache_manager.free(request)
self._inflight_prefills.discard(request)
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
request.num_preemptions += 1
self.waiting.prepend_request(request)
```

因此 preemption victim 一定来自：

```text
RUNNING
```

而不是从 WAITING 中删除某个请求。

### PREEMPT_BEGIN

在 `_free_request_blocks()` 之前打印：

```text
request status
num_tokens
num_computed_tokens
num_in_flight_tokens
num_preemptions
当前 owned block IDs
free block count
```

### PREEMPT_END

在状态重置并 `waiting.prepend_request()` 之后打印：

```text
status
num_computed_tokens
num_in_flight_tokens
num_preemptions
waiting queue
free block count
```

这允许动态观察：

```text
RUNNING
↓
PREEMPTED
↓
computed = 0
↓
num_preemptions += 1
↓
prepend WAITING
```

---

# 12. Trace 4：为什么要观察 immediate/deferred free

`_preempt_request()` 调：

```python
self._free_request_blocks(request)
```

但 async scheduling 下不能简单假定：

```text
preempt
=
KV blocks 一定立即回 free pool
```

因为 `_free_request_blocks()` 真实实现中存在：

```python
if not self.defer_block_free or (
    request.last_sched_seq <= self.processed_step_seq
):
    self.kv_cache_manager.free(request)
    return

blocks = self.kv_cache_manager.pop_blocks_for_free(request)
if blocks:
    self.deferred_frees.append((self.sched_step_seq, blocks))
```

也就是说存在两种生命周期：

```text
A. Immediate Free
request ownership release
→ block 立即回 BlockPool

B. Deferred Free
request ownership 先移除
→ block 暂时不能安全复用
→ deferred_frees
→ 后续 fence 安全后再回 pool
```

所以增加：

```text
FREE_IMMEDIATE_BEGIN
FREE_IMMEDIATE_END
FREE_DEFERRED
```

E4-B 最终真实结果是：

```text
四次 preemption 全部走 FREE_IMMEDIATE
没有观察到 FREE_DEFERRED
```

因此文档只能得出：

> vLLM 当前源码支持 deferred free；但本次 E4 workload 没有触发该路径。

不能把“支持 deferred”写成“本实验发生了 deferred”。

---

# 13. E4-A：`scheduler_reserve_full_isl=True`

实验变量：

```text
mode = reserve
scheduler_reserve_full_isl = True
num_gpu_blocks_override = 43
```

Request：

```text
A：526 prompt + 8 output
B：526 prompt + 8 output
```

---

# 14. E4-A Call 0～3：A 先独占 token budget

A 的 Prefill：

```text
Call 0
scheduled=128
blocks [1..8]
computed 0 → 128

Call 1
scheduled=128
blocks [9..16]
computed 128 → 256

Call 2
scheduled=128
blocks [17..24]
computed 256 → 384

Call 3
scheduled=128
blocks [25..32]
computed 384 → 512
```

Call 1 起 B 已经进入 Scheduler：

```text
running=[A]
waiting=[B]
```

但是为什么 Call 1～3 没有 `ALLOC_REJECT_FULL`？

因为：

```text
max_num_batched_tokens = 128
A 每轮 scheduled=128
```

所以 running request A 已经把 global token budget 全部消耗：

```text
budget_start=128
A=128
budget_remaining=0
```

Scheduler 根本没有剩余 token budget 进入 waiting request admission。

这与 E3 已证明的：

```text
running first
→ leftover budget
→ waiting admission
```

完全一致。

---

# 15. E4-A Call 4：核心 admission gate 证据

A 只剩：

```text
526 - 512 = 14 tokens
```

所以：

```text
A scheduled=14
```

token budget 还剩：

```text
128 - 14 = 114
```

这时 Scheduler 才真正尝试 B。

真实 trace：

```text
[BRIDGE][KV][ALLOC_REJECT_FULL]
req=B
status=WAITING
num_tokens=526
num_computed=0
num_in_flight=0
num_new_tokens=114
full_num_tokens=526
blocks_to_allocate=33
required_blocks=33
free_blocks=9
```

这里最关键的是：

```text
当前 chunk：114 tokens
大约只需要 8 blocks

当前 free：9
```

也就是说：

```text
仅从“当前 chunk”看，B 是能进入的。
```

但 `scheduler_reserve_full_isl=True` 会提前检查完整 Input Sequence：

```text
B full input = 526
需要 33 blocks

free = 9

33 > 9
```

所以：

```text
allocate_slots() → None
B 保持 WAITING
```

这条 trace 是 E4-A 最核心证据。

---

# 16. E4-A Call 5～12：B 始终 WAITING

后续不断观察到：

```text
running=[A]
waiting=[B]
```

以及：

```text
ALLOC_REJECT_FULL
required_blocks=33
free_blocks=9
```

Decode 跨过 528-token block boundary 后，A 新增第 34 block：

```text
new_block_ids=([34],)
```

所以 free：

```text
9 → 8
```

随后 B admission 继续失败：

```text
33 > 8
```

整个过程中没有：

```text
ALLOC_REJECT_STEP
PREEMPT_BEGIN
```

所以 E4-A 的资源压力处理方式不是 preemption，而是：

```text
admission serialization
```

即：

```text
A RUNNING
B WAITING
↓
A finish
↓
B admission
```

---

# 17. E4-A Call 12 → 13：block reuse

Call 12 后半段 A 最终输出 reconcile：

```text
TOKEN_APPEND
num_output_tokens=8
num_tokens_after=534
num_computed=533
num_in_flight=0
```

A 生命周期结束。

Call 13：

```text
known_requests=[B]
running=[]
waiting=[B]
```

B 被 admission：

```text
kind=NEW
scheduled=128
new_block_ids=([34,33,32,31,30,29,28,27],)
```

这些 block IDs 正是 A 刚刚使用过的一部分。

因此即使 normal finish 没有专门打印 `FREE_IMMEDIATE`，也可以通过 block ID reuse 证明：

```text
A finish
↓
A block ownership release
↓
blocks 回到可分配资源
↓
B 立即复用
```

E4-A 最终两个请求都正常完成，输出 token 数均为 8。

---

# 18. E4-A 结论

E4-A = PASS。

动态证明：

```text
scheduler_reserve_full_isl=True
```

会把 KV capacity pressure 前移到：

```text
WAITING admission gate
```

完整链路：

```text
B WAITING
↓
当前 step 有 token budget
↓
full_sequence_must_fit=True
↓
完整 input 需要 33 blocks
↓
free 只有 9 / 8
↓
ALLOC_REJECT_FULL
↓
B stays WAITING
↓
NO PREEMPTION
↓
A finish
↓
B admission
↓
B 正常完成
```

E4-A 的代价是 B admission 更晚，但好处是没有浪费任何 B 的 Prefill progress。

---

# 19. E4-B：`scheduler_reserve_full_isl=False`

配置唯一关键变化：

```text
scheduler_reserve_full_isl = False
```

其余保持相同：

```text
Prompt=526
block_size=16
max_num_batched_tokens=128
num_gpu_blocks_override=43
max_model_len=640
max_tokens=8
```

这组实验用于回答：

> 如果不要求完整 Input Sequence 在 admission 时就能 fit，而是“当前 chunk 能放就先让它进”，运行期会发生什么？

---

# 20. E4-B Call 0～3：与 E4-A 相同

A：

```text
0 → 128 → 256 → 384 → 512
```

B：

```text
WAITING
```

A 每次占满 token budget，所以此时 A/B 两组没有差异。

真正分叉仍发生在 Call 4。

---

# 21. E4-B Call 4：Over-admission 真正发生

A：

```text
scheduled=14
new_block_ids=([33],)
```

剩余 token budget：

```text
114
```

B：

```text
kind=NEW
scheduled=114
new_block_ids=([34,35,36,37,38,39,40,41],)
```

因此同一 SchedulerOutput：

```text
A final prefill = 14
B first chunk   = 114
----------------------
total           = 128
```

这次没有：

```text
ALLOC_REJECT_FULL
```

因为 full-sequence admission gate 已关闭。

当前 114-token chunk 只需要 8 个 blocks，而当时这些 blocks 可以获得，所以 B：

```text
WAITING
↓
RUNNING
↓
num_computed=114
num_in_flight=114
```

这就是实验中所谓 **over-admission**：

> 当前 chunk 可以运行，所以 request 被 admission；但并没有保证整个剩余 Prompt 后续能持续拿到足够 KV。

---

# 22. E4-B Call 5：第一次 runtime allocation failure

Call 5 snapshot：

```text
running=[A,B]
waiting=[]
```

B 当前：

```text
num_computed=114
num_in_flight=114
```

下一轮希望：

```text
num_new_tokens=127
```

所以 KV coverage target：

```text
num_tokens_need_slot=241
```

需要新增：

```text
blocks_to_allocate=8
```

但是：

```text
free_blocks=1
reserved_blocks=0
watermark_blocks=0
available_blocks=1
required_blocks=8
```

因此：

```text
8 > 1
```

真实命中：

```text
[BRIDGE][KV][ALLOC_REJECT_STEP]
```

这与 E4-A 的核心区别是：

```text
E4-A：
status=WAITING
ALLOC_REJECT_FULL

E4-B：
status=RUNNING
ALLOC_REJECT_STEP
```

所以 E4-B 已经从“admission pressure”进入真正的“runtime allocation pressure”。

---

# 23. E4-B 第一次 preemption

紧接 `ALLOC_REJECT_STEP`：

```text
[BRIDGE][SCHED][PREEMPT_BEGIN]
req=B
status=RUNNING
num_tokens=526
num_computed=114
num_in_flight=114
num_preemptions=0
blocks=([34,35,36,37,38,39,40,41],)
free_blocks=1
```

这直接证明：

```text
preemption victim = B
```

并且 victim 来自：

```text
RUNNING
```

而不是 WAITING。

---

# 24. Preemption 时 KV block 如何释放

紧接：

```text
FREE_IMMEDIATE_BEGIN
blocks=[34..41]
free_before=1
```

B 拥有 8 blocks。

随后：

```text
FREE_IMMEDIATE_END
free_after=9
```

严格对得上：

```text
1 + 8 = 9
```

因此本次 preemption 中：

```text
B request ownership
↓
8 个 KV blocks 立即归还 BlockPool
```

本次没有：

```text
FREE_DEFERRED
```

---

# 25. Preemption 后 request 状态怎么变化

真实 `PREEMPT_END`：

```text
status=PREEMPTED
num_computed=0
num_in_flight=114
num_preemptions=1
waiting=[B]
free_blocks=9
```

因此源码状态转移被完整动态验证：

```text
B RUNNING
num_computed=114
KV=[34..41]
        │
        ▼
    PREEMPT
        │
        ├─ free KV
        ├─ status → PREEMPTED
        ├─ num_computed → 0
        ├─ num_preemptions 0 → 1
        └─ prepend WAITING
```

注意：

```text
num_in_flight
```

此刻仍是 114，并没有被 `_preempt_request()` 一起清零。

---

# 26. 为什么 preempt 后 `num_in_flight=114`

Call 4 中 B 的 114-token SchedulerOutput 已经：

```text
schedule
→ commit
→ submit execution
→ push batch_queue
```

Call 5 Scheduler 在前一份 transaction 尚未完成 reconciliation 前就继续下一次 scheduling。

因此 Call 5 可以先决定：

```text
B logical computed frontier 无效
→ num_computed=0
```

但旧 transaction 仍然需要从 bookkeeping 中 settle。

Call 5 后半段 consume Call 4 的结果时：

```text
SETTLE B
settled=114
in_flight_before=114
in_flight_after=0
num_computed=0
```

这说明：

```text
num_computed_tokens
= Scheduler 当前认可的 logical computed frontier

num_in_flight_tokens
= 已 submit 但尚未完成 Scheduler reconciliation 的 work accounting
```

Preemption 可以让当前 logical frontier 失效并归零，但不能跳过旧 SchedulerOutput transaction 的 reconciliation bookkeeping。

---

# 27. `num_in_flight > 0` 为什么还能 FREE_IMMEDIATE

这是 E4-B 一个非常重要的 async 语义发现。

第一次 preemption：

```text
num_in_flight=114
```

但是实际走：

```text
FREE_IMMEDIATE
```

而不是：

```text
FREE_DEFERRED
```

因此不能得到：

```text
num_in_flight > 0
⇒ GPU 此刻仍物理写这些 blocks
```

准确说：

```text
num_in_flight
```

是 Scheduler transaction / reconciliation 维度的状态，而 block 是否能物理安全复用还受：

```text
last_sched_seq
processed_step_seq
defer_block_free / fence
```

约束。

因此：

> Scheduler transaction lifecycle 与 GPU physical block lifetime 是相关但不等价的两个维度。

---

# 28. E4-B Call 6：第一次真正 recompute

Call 6：

```text
running=[A]
waiting=[B]
```

B 再次被 admission：

```text
kind=NEW
num_computed=0
scheduled=127
new_block_ids=([41,40,39,38,37,36,35,34],)
```

这里 `kind=NEW` 不代表创建了一个新的 API request。

request ID 没有变化。

它表示当前 V2 Model Runner/SchedulerOutput representation 中，preempted/resumed request 重新走 new-request data 路径。

真正 lifecycle 是：

```text
同一个 B
RUNNING
→ PREEMPTED
→ WAITING
→ resumed
```

而不是：

```text
old B destroy
→ new B create
```

---

# 29. 为什么这是 recompute，而不是 resume from 114

B 第一次已经计算过：

```text
114 tokens
```

但 preemption 后：

```text
num_computed_tokens=0
```

所以 Call 6：

```text
0 → 127
```

不是：

```text
114 → 241
```

也就是说 Prompt 前 114 token 的计算进度没有保留。

这就是当前 preemption mode 的：

```text
recompute
```

同时刚释放的 block IDs：

```text
34..41
```

马上又被 B 重新拿到。

因此真实资源链：

```text
allocate blocks
↓
compute chunk
↓
preempt
↓
free same blocks
↓
re-admit
↓
allocate same blocks
↓
recompute from prompt beginning
```

---

# 30. E4-B Call 7～11：KV Thrashing

B 随后陷入重复循环。

## 第 2 次 preemption

```text
Call 6:
B 0 → 127

Call 7:
B wants 127 → 254
needs +8 blocks
free=0

ALLOC_REJECT_STEP
↓
PREEMPT
↓
num_computed=0
num_preemptions=2
```

## 第 3 次 preemption

```text
Call 8:
B 0 → 127

Call 9:
needs +8
free=0

PREEMPT
num_preemptions=3
computed=0
```

## 第 4 次 preemption

```text
Call 10:
B 0 → 127

Call 11:
needs +8
free=0

PREEMPT
num_preemptions=4
computed=0
```

因此 B 真正陷入：

```text
admit
→ allocate 8 blocks
→ compute ~127
→ next expansion fails
→ preempt
→ free blocks
→ computed=0
→ waiting
→ re-admit
→ recompute
→ repeat
```

这就是本实验中的 **KV Cache Thrashing**。

---

# 31. E4-B 浪费的 Scheduler-level Prefill work

四次被 preempt 前，B 已经累计做过：

```text
第 1 次：114 tokens
第 2 次：127 tokens
第 3 次：127 tokens
第 4 次：127 tokens
```

这些 logical progress 都因为：

```text
num_computed_tokens → 0
```

而最终不能作为后续 request frontier 保留。

总计：

```text
114 + 127 + 127 + 127
= 495 tokens
```

B 的真实 Prompt：

```text
526 tokens
```

最后成功的一轮仍要重新完整完成约 526-token Prompt。

因此 B 的 Scheduler-level Prefill scheduled work 约为：

```text
495 wasted
+
526 useful
=
1021 token-work
```

相对于理想：

```text
526
```

约为：

```text
1021 / 526 ≈ 1.94x
```

因此当前极端 KV pressure workload 下：

> B 的 Prefill scheduled token-work 接近翻倍。

必须注意：

```text
1.94x
```

是 **Scheduler-level scheduled token-work** 的估计，不等于严格测量得到的 GPU FLOPs / latency / energy 放大倍数。

后者需要 Nsight / GPU profiling / benchmark 才能下结论。

---

# 32. 为什么 Call 12 以后 B 突然稳定推进

Call 11 时：

```text
B 第 4 次 preempt
A 已经接近 max_tokens=8
```

Call 12：

```text
B 再次从 0 开始
scheduled=128
```

同一个 EngineCore call 的后半段，A 最后一个 output transaction reconcile：

```text
TOKEN_APPEND
num_output_tokens=8
```

A 生命周期结束。

Call 13：

```text
known_requests=[B]
running=[B]
waiting=[]
```

从此 B 独占 KV pool。

于是：

```text
0 → 128
128 → 256
256 → 384
384 → 512
512 → 526
```

一路完成 Prompt，之后正常 Decode。

这说明前面的失败不是 B 自身不可运行，而是：

```text
A + B 同时驻留时 KV capacity 不足
```

A 离场释放 blocks 后，B 立刻恢复正常推进。

---

# 33. 为什么 E4-B B 比 E4-A 提前一个 step 开始最终成功 Prefill

这是 A/B 对照中一个细节。

E4-A Call 12 schedule 阶段：

```text
A 尚未完成最终 reconcile
```

所以 full gate 仍看到：

```text
B required=33
free=8
→ reject
```

只有 Call 12 后半段 A finish，Call 13 B 才 admission。

E4-B 没有 full gate，因此 Call 12：

```text
B 当前 128-token chunk 能分配
→ B 先 schedule=128
```

然后 Call 12 后半段 A finish。

所以 aggressive admission 并不是完全没有优点：

```text
True：更保守，减少 wasted work，但可能更晚 admission
False：更激进，可能更早 overlap，但 pressure 强时会 thrash
```

这才是 `scheduler_reserve_full_isl` 的真实 policy trade-off。

---

# 34. E4-B 最终输出正确性

两个请求最终都正常完成：

```text
prompt_tokens=526
num_output_tokens=8
```

两请求输出 token IDs 一致：

```text
[20286, 4128, 1614, 44378, 1598, 15136, 10091, 281]
```

因此：

```text
preemption
requeue
recompute
```

虽然引入大量额外 work，但没有破坏最终生成结果的一致性。

---

# 35. E4-A 与 E4-B 核心对照

| 维度 | E4-A | E4-B |
|---|---|---|
| `scheduler_reserve_full_isl` | `True` | `False` |
| KV pool | 43 blocks | 43 blocks |
| Prompt | 526 | 526 |
| Prompt blocks | 33 | 33 |
| 单请求 peak | 34 | 34 |
| 第二请求第一次获得剩余 budget | Call 4 | Call 4 |
| B 当时 current chunk | 114 | 114 |
| 完整 input gate | 开启 | 关闭 |
| 第一次关键 reject | `ALLOC_REJECT_FULL` | 后续 `ALLOC_REJECT_STEP` |
| reject 时 B 状态 | WAITING | RUNNING |
| B 是否部分进入 Prefill | 否 | 是 |
| Preemption | 0 | 4 |
| `num_computed` reset | 无 | 多次 →0 |
| KV block free | A finish 后 reuse | 每次 victim 8 blocks immediate free |
| Recompute | 无 | 有 |
| Thrashing | 无 | 明显 |
| 最终结果 | 正常 | 正常 |

---

# 36. 两种策略的状态机

## E4-A

```text
Pool = 43

A WAITING
↓
A RUNNING
↓
A blocks: 8→16→24→32→33→34
│
├─────────────────────────────┐
│                             │
│                         B WAITING
│                             │
│                    full input needs 33
│                             │
│                       free only 9/8
│                             │
│                    ALLOC_REJECT_FULL
│                             │
│                       stay WAITING
│                             │
└──────── A finish ────────────┘
              ↓
          blocks released
              ↓
       B WAITING → RUNNING
              ↓
         normal completion
```

## E4-B

```text
Pool = 43

A RUNNING
│
│ leftover token budget
▼
B current chunk fits
↓
B WAITING → RUNNING
↓
B gets 8 blocks
↓
B computes 114/127
↓
next chunk needs +8
↓
free = 0/1
↓
ALLOC_REJECT_STEP
↓
PREEMPT B
├─ free B blocks
├─ computed → 0
├─ preemptions++
└─ PREEMPTED → WAITING
          ↓
       re-admit
          ↓
       recompute
          ↓
     repeat until A finishes
```

---

# 37. Preemption victim selection：本实验能证明到什么程度

真实 E4-B：

```text
running=[A,B]
```

发生 pressure 后：

```text
PREEMPT_BEGIN req=B
```

因此日志直接证明：

```text
本次 victim = B
```

静态源码中普通 FCFS 路径存在：

```python
preempted_req = self.running.pop()
```

因此当前执行结果与“从 running 尾部选择后进入请求”一致。

Priority scheduling 路径则使用：

```python
max(
    self.running,
    key=lambda r: (r.priority, r.arrival_time),
)
```

所以不能把：

```text
preemption 永远抢最后一个 request
```

当成所有 policy 的普遍结论。

准确结论：

> Preemption victim 一定来自 RUNNING；当前 E4-B 的实际 victim 是后进入的 B，并与本次 FCFS `running.pop()` 路径一致。不同 scheduling policy 的 victim selection 需要看对应分支。

---

# 38. WAITING 与 PREEMPTED 的区别

E4 很好地把两者分开。

## WAITING

例如 E4-A 的 B：

```text
B 还没有进入 active running set
KV admission gate 不通过
↓
stay WAITING
```

它没有丢失已经计算的 Prefill progress，因为它根本没开始算。

## PREEMPTED

例如 E4-B 的 B：

```text
B 已经 RUNNING
已经拥有 KV blocks
已经有 computed progress
↓
运行期 allocation failure
↓
被 victim selection 选中
↓
free blocks
computed=0
status=PREEMPTED
↓
回 WAITING
```

所以：

```text
WAITING
≠
PREEMPTED
```

更不能把“第二请求一直 waiting”写成“发生了 preemption”。

---

# 39. `scheduler_reserve_full_isl` 到底应该怎么理解

不要理解成：

```text
True = 给整个 request 预留从 Prompt 到未来无限 Decode 的所有 KV
```

也不要简单写成：

```text
True 永远比 False 好
```

更准确：

```text
True
= waiting/preempted request admission 时
  先做完整 input-sequence fit 检查

False
= 不要求完整 input 在 admission 阶段 fit
  允许按当前 chunk 的真实需要逐步进入
```

因此：

```text
True：conservative admission
False：aggressive admission
```

在本次极端 KV pressure 中：

```text
True
→ 防止 over-admission
→ 无 preemption/recompute

False
→ B 可更早进入
→ 但发生 4 次 preemption
→ 明显 KV thrashing
```

所以它解决的是 production inference 中典型的：

```text
utilization / concurrency
vs
resource safety / recompute cost
```

权衡。

---

# 40. `allocate_slots()` 的两道防线

E4 最终让 `KVCacheManager.allocate_slots()` 的两个 gate 变得非常清楚。

## 第一层：Full-sequence admission gate

```text
适用：waiting / preempted admission
条件：full_sequence_must_fit=True

完整 input 所需 blocks
+
watermark
>
free blocks

→ return None
```

对应 E4-A：

```text
ALLOC_REJECT_FULL
```

## 第二层：Current-step allocation gate

```text
当前 step 为推进 num_new_tokens
需要追加 blocks
+
watermark
>
free - reserved

→ return None
```

对应 E4-B：

```text
ALLOC_REJECT_STEP
```

两者不能混成一个“KV 不够”。

控制面语义不同：

```text
FULL reject
= admission prevention

STEP reject
= already-running request cannot expand
```

---

# 41. E4 对 `num_tokens / num_computed / num_in_flight` 的进一步验证

E4-B 尤其强化了三者的区别。

## `num_tokens`

Request 当前已知 token 总数：

```text
Prompt
+
已经正式 append 的 output tokens
```

## `num_computed_tokens`

Scheduler 当前认可的 logical computed frontier。

schedule-time 会提前 commit。

Preemption 时可以：

```text
114 / 127 → 0
```

## `num_in_flight_tokens`

已经 schedule/submit、但尚未完成 Scheduler reconciliation 的 work accounting。

所以可能出现：

```text
PREEMPT_END:
computed=0
in_flight=114
```

随后旧 result 返回：

```text
SETTLE 114
in_flight 114→0
computed 仍=0
```

这个状态并不矛盾。

---

# 42. E4 对 block ownership 的验证

E4 同时观察了三类 block 生命周期。

## 42.1 Active allocation

```text
new_block_ids
```

证明 request 在 logical frontier 增长时获得新的 physical block IDs。

## 42.2 Preemption free

```text
FREE_IMMEDIATE_BEGIN
→ FREE_IMMEDIATE_END
```

证明 victim 的 active ownership 被释放，free count 精确增加。

## 42.3 Finish 后 reuse

E4-A 中 A finish 后，B 立即拿到 A 刚使用过的 IDs。

说明：

```text
block ID reuse
```

不表示旧 request 仍拥有 block。

真正需要看的是生命周期和 ownership boundary。

---

# 43. 一个未展开的小观察：43 blocks 与普通 block IDs

最终配置：

```text
num_gpu_blocks_override=43
```

但 request-owned block ID trace 主要观察到：

```text
1..42
```

并且：

```text
A=33 blocks 时 free=9
33+9=42
```

因此至少有 1 个 block 没以普通 request-owned block 的形式出现在当前 trace 中。

仅凭当前日志不能严格断言：

```text
block 0 一定是 null/reserved block
```

如果后续项目需要精确解释，再单独看 BlockPool 初始化 / null block 设计。

该细节不影响 E4 的 admission/preemption 结论。

---

# 44. 实验命令

## 44.1 环境

```bash
cd ~/learning/llm-kv-lab/third_party/vllm
source ./activate

export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=1
export QCACHE_BRIDGE_TRACE=1

cd ~/learning/llm-kv-lab
mkdir -p 04-experiments/vllm-bridge/raw/e4-kv-pressure
```

## 44.2 Probe

```bash
python 04-experiments/vllm-bridge/scripts/e4_kv_pressure.py \
    --mode probe \
    2>&1 \
    | tee 04-experiments/vllm-bridge/raw/e4-kv-pressure/probe.log
```

## 44.3 E4-A

```bash
python 04-experiments/vllm-bridge/scripts/e4_kv_pressure.py \
    --mode reserve \
    --num-gpu-blocks-override 43 \
    2>&1 \
    | tee 04-experiments/vllm-bridge/raw/e4-kv-pressure/e4-a-reserve.log
```

## 44.4 E4-B

```bash
python 04-experiments/vllm-bridge/scripts/e4_kv_pressure.py \
    --mode overadmit \
    --num-gpu-blocks-override 43 \
    2>&1 \
    | tee 04-experiments/vllm-bridge/raw/e4-kv-pressure/e4-b-overadmit.log
```

---

# 45. 建议日志过滤命令

## 45.1 核心 Scheduler / KV 状态

```bash
grep -E \
'\[BRIDGE\]\[(KV|SCHED)\]\[(ALLOC_REJECT_FULL|ALLOC_REJECT_STEP|PREEMPT_BEGIN|PREEMPT_END|FREE_IMMEDIATE_BEGIN|FREE_IMMEDIATE_END|FREE_DEFERRED|QUEUE_SNAPSHOT|BEFORE_COMMIT|AFTER_COMMIT|SETTLE|TOKEN_APPEND)\]' \
04-experiments/vllm-bridge/raw/e4-kv-pressure/e4-b-overadmit.log
```

## 45.2 只看 pressure/preemption

```bash
grep -E \
'ALLOC_REJECT_FULL|ALLOC_REJECT_STEP|PREEMPT_BEGIN|PREEMPT_END|FREE_IMMEDIATE|FREE_DEFERRED' \
04-experiments/vllm-bridge/raw/e4-kv-pressure/e4-b-overadmit.log
```

## 45.3 配置与最终结果

```bash
grep -E \
'mode:|constructed_prompt_tokens:|scheduler_reserve_full_isl:|num_gpu_blocks_override:|GPU KV cache size|prompt_tokens:|output_token_ids:|num_output_tokens:' \
04-experiments/vllm-bridge/raw/e4-kv-pressure/e4-b-overadmit.log
```

---

# 46. Trace 字段解释

## `ALLOC_REJECT_FULL`

```text
req
status
num_tokens
num_computed
num_in_flight
num_new_tokens
full_num_tokens
blocks_to_allocate
watermark_blocks
required_blocks
free_blocks
```

重点：

```text
full_num_tokens
required_blocks
free_blocks
```

回答完整 Input Sequence admission 是否可行。

## `ALLOC_REJECT_STEP`

```text
num_new_tokens
num_tokens_need_slot
blocks_to_allocate
reserved_blocks
watermark_blocks
required_blocks
free_blocks
available_blocks
```

重点：

```text
当前 step 结束后 KV coverage target
↓
还需要追加多少 blocks
↓
真正 available 多少
```

## `PREEMPT_BEGIN`

回答：

```text
谁被选为 victim？
被抢占前已经计算到哪里？
拥有哪些 KV blocks？
当前 free pool 状态？
```

## `PREEMPT_END`

回答：

```text
status 是否变 PREEMPTED？
computed 是否归零？
preemption count 是否增加？
是否重新回 WAITING？
```

## `FREE_IMMEDIATE`

回答 active blocks 是否立即重新进入可分配资源。

## `FREE_DEFERRED`

回答 async physical lifetime 是否需要延迟释放。

本实验未触发该事件。

---

# 47. 静态源码与动态证据对应

| 源码语义 | 动态证据 |
|---|---|
| `full_sequence_must_fit` 完整输入 gate | E4-A `ALLOC_REJECT_FULL required=33 free=9/8` |
| current-step allocation gate | E4-B `ALLOC_REJECT_STEP required=8 free=1/0` |
| victim 必须 RUNNING | `PREEMPT_BEGIN status=RUNNING` |
| `_free_request_blocks()` | `FREE_IMMEDIATE free 1→9 / 0→8` |
| `status=PREEMPTED` | `PREEMPT_END status=PREEMPTED` |
| `num_computed_tokens=0` | 114/127 → 0 |
| `num_preemptions += 1` | 0→1→2→3→4 |
| `waiting.prepend_request()` | `waiting=[B]` |
| resume/recompute | 同 request ID 再次 `num_computed=0 scheduled=127/128` |
| finished block reuse | A finish 后 B 获得 A 的旧 block IDs |

---

# 48. E4 最终证明了什么

## 48.1 KV admission 和 token scheduling 是两个不同约束

E3 已证明：

```text
global token budget
```

决定一个 step 能计算多少 token。

E4 进一步证明：

```text
有 token budget
```

并不等于：

```text
request 一定能被 schedule
```

还必须经过 KV capacity / allocation gate。

---

## 48.2 Chunked Prefill 会产生 over-admission 风险

Chunked Prefill 每次只需要部分 Prompt 的 KV。

因此：

```text
当前 chunk fit
```

不代表：

```text
整个 Prompt 能持续推进
```

在极端 memory pressure 下，如果 admission 只看当前 chunk，就可能：

```text
admit
→ compute
→ 后续 block 不够
→ preempt
→ recompute
```

---

## 48.3 `scheduler_reserve_full_isl=True` 是 prevention，不是 recovery

它不是在 preemption 发生后修复问题。

它是在 waiting admission 阶段提前阻止：

```text
明显没有完整 input 生存空间的 request
```

所以它的目标是减少：

```text
preemption / recomputation / thrashing
```

---

## 48.4 Preemption 是一个完整 request lifecycle transition

不是简单“停止 GPU”。

真实语义包括：

```text
RUNNING victim selection
↓
active KV ownership release
↓
status PREEMPTED
↓
logical computed frontier reset
↓
num_preemptions++
↓
requeue WAITING
↓
later resume/recompute
```

---

## 48.5 Waiting ≠ Preemption

E4-A B 一直 waiting，但从未 preempt。

所以后续日志分析必须严格区分：

```text
waiting due admission
```

与：

```text
preempted after running
```

---

## 48.6 `num_in_flight` 不是 physical GPU-running token count

E4-B 观察到：

```text
PREEMPT_END:
computed=0
in_flight=114/127
```

且同一时刻 block 可以 `FREE_IMMEDIATE`。

因此 `num_in_flight_tokens` 应理解为 async scheduler transaction accounting，而不是直接的 GPU kernel execution status。

---

## 48.7 KV block ID 是可复用资源 ID，不是永久归属

同一个 block ID 可以：

```text
先属于 A
A finish / B preempt 后 free
再属于 B
```

因此看到 block ID reuse 时必须结合 request lifecycle / ownership，而不能按 ID 静态绑定某个 request。

---

# 49. E4 与后续 QCache 项目的直接关系

未来 QCache 要引入：

```text
BF16 physical pool
INT4 physical pool
migration
compression
eviction
restore
```

E4 提供了关键控制面前提：

### 1. Admission policy 必须考虑未来资源可持续性

不能只因为：

```text
当前 chunk 有空间
```

就认为请求安全。

### 2. Preemption 会使 computed frontier 失效

如果 QCache 自己维护：

```text
compressed history metadata
migration state
logical→physical mapping
```

必须在 preemption/reset 时同步失效或重建。

### 3. Block ownership 与 physical lifetime 必须区分

因为 async runtime 可能存在 deferred free。

QCache 不能在 ownership release 后立刻无条件复用底层 memory，除非对应 fence 已安全。

### 4. Recompute 本身有真实代价

E4-B 仅一个 526-token request 就产生约 495 token 的 wasted logical Prefill progress。

未来设计新的 KV admission / compression policy 时应避免制造类似 thrashing。

---

# 50. 实验中明确没有证明的内容

为了防止过度解释，以下结论 E4 没有证明：

```text
1. FREE_DEFERRED 在什么 workload 下一定触发
2. GPU FLOPs 实际增加了 1.94x
3. TP/PP/DP 下 preemption 是否完全相同
4. Prefix Cache 打开后的 pressure 行为
5. KV Connector / LMCache 下 reserved_blocks 行为
6. Priority scheduler 所有 victim selection 规则
7. block 0 的精确内部用途
8. production workload 中 True 一定比 False 性能更好
```

这些都需要独立实验，不能从 E4 外推。

---

# 51. E4 DoD 检查

```text
[PASS] 单请求真实 B_single 已测得：34

[PASS] override 按 1.25× 设置：43

[PASS] 一个请求可以完成，两个请求总需求超过 pool

[PASS] E4-A 观察到 WAITING admission reject

[PASS] E4-A 观察到 ALLOC_REJECT_FULL

[PASS] E4-A 无 pressure-triggered preemption

[PASS] E4-B 允许第二请求部分 admission

[PASS] E4-B 观察到 ALLOC_REJECT_STEP

[PASS] 观察到真实 victim B

[PASS] 观察到 PREEMPT_BEGIN / END

[PASS] 观察到 released block IDs

[PASS] 观察到 free count 精确恢复

[PASS] 观察到 num_preemptions 0→4

[PASS] 观察到 num_computed_tokens reset→0

[PASS] 观察到 PREEMPTED → WAITING

[PASS] 观察到重新 admission / recompute

[PASS] 两请求最终完成且输出一致

[PASS] 区分 immediate / deferred free；本实验只触发 immediate
```

因此：

```text
E4 = PASS / Frozen
```

不需要继续追加 E4 runtime case。

---

# 52. 最终心智模型

E4 最终应该记住的不是参数名，而是下面这张生产推理控制面图：

```text
                           Scheduler
                               │
                               │ token budget
                               ▼
                    request 是否有调度机会
                               │
                               ▼
                       KVCacheManager
                               │
               ┌───────────────┴────────────────┐
               │                                │
      full-sequence admission          current-step allocation
               │                                │
       reserve_full_isl=True                    │
               │                                │
         full input 能否 fit             本 step 能否扩 KV
               │                                │
      ┌────────┴────────┐              ┌────────┴────────┐
      │                 │              │                 │
     Yes               No             Yes               No
      │                 │              │                 │
   admission      stay WAITING       schedule       allocation fail
                        │                                │
                ALLOC_REJECT_FULL                       ▼
                                                victim selection
                                                       │
                                                       ▼
                                                    RUNNING
                                                       │
                                                       ▼
                                                   PREEMPT
                                                       │
                           ┌───────────────────────────┼───────────────────────┐
                           │                           │                       │
                       free KV                computed → 0          status PREEMPTED
                           │                           │                       │
                           └───────────────────────────┼───────────────────────┘
                                                       ▼
                                                    WAITING
                                                       │
                                                       ▼
                                                later re-admit
                                                       │
                                                       ▼
                                                   recompute
```

---

# 53. 与 E1/E2/E3 的衔接

截至 E4，Scheduler/KV 动态实验已经形成一条完整递进：

```text
E1
单请求
完整 Prefill → Decode
↓
证明 async schedule / commit / queue / reconcile

E2
单请求 + 长 Prompt
↓
证明 chunked prefill / token budget / KV block growth

E3
多请求 + KV 足够
↓
证明 continuous batching / global token budget / running+waiting 混合

E4
多请求 + KV 极度受限
↓
证明 admission / allocation reject / preemption / free / recompute
```

所以现在已经建立了：

```text
token scheduling
+
request lifecycle
+
KV allocation lifecycle
+
async transaction lifecycle
```

四层统一认知。

下一阶段可以进入 E5 Prefix Cache，对照研究：

```text
computed frontier
```

除了通过真实模型计算推进之外，还如何通过跨请求 prefix hit 直接前移，以及 cached block / active block ownership 如何变化。

---

# 54. 最终结论

E4 最终不是简单证明：

```text
“vLLM KV 不够会 preempt。”
```

更重要的是证明了完整决策链：

```text
KV capacity 是 Scheduler admission 的一部分

Chunked Prefill 允许当前 chunk fit，但可能造成未来生存空间不足

scheduler_reserve_full_isl=True
通过完整 input admission gate 提前阻止这种 over-admission

scheduler_reserve_full_isl=False
允许更激进的并发，但高 pressure 下可能形成：

ALLOC_REJECT_STEP
→ PREEMPT
→ FREE
→ computed reset
→ WAITING
→ RECOMPUTE
→ THRASHING
```

本实验中 B 被 preempt 4 次，约 495 token 的 logical Prefill progress 被丢弃，最终仍能够在 A 结束、资源释放后重新完整计算并正确完成。

这说明 production inference runtime 的 KV Cache 管理不仅是：

```text
“有 block 就 allocate，没 block 就等。”
```

而是一个包含：

```text
admission policy
resource accounting
victim selection
lifecycle transition
async safety
recomputation cost
```

的完整控制系统。

**E4 到此正式收尾。**
