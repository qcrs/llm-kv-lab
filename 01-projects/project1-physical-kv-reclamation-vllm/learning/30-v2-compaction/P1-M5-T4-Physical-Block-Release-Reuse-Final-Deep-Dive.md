# P1-M5-T4 — Physical Block Release / Reuse Closure
## V1 + V2 最终源码链路、设计决策与实现复盘

> Project: Physical KV Cache Reclamation for vLLM  
> Baseline: vLLM v0.26.0  
> Branch: `p1/v2-token-compaction-v026`  
> Verified runtime slice: MRV2 / **explicit `async_scheduling=False`** / PP=1 / `max_concurrent_batches=1` / single KV group / KV connector OFF / prefix caching OFF / spec decode OFF / CUDA Graph OFF  
> T4 Core Status: **CLOSED / PASS within frozen synchronous MVP**  
> Async / PP>1 / KV connector runtime hardening: **DEFERRED / NOT CLAIMED**

---

# 0. 先回答一个核心问题：T4 的改动是不是其实很简单？

是。

**从代码量上看，T4 很简单。**

真正新增的核心逻辑几乎就是：

```python
def _release_reconciled_blocks(
    self,
    removed: list[KVCacheBlock],
) -> None:
    if removed:
        self.kv_cache_manager.block_pool.free_blocks(reversed(removed))
```

再把 V1 / V2 的 `removed blocks` 都交给这个 helper。

但 T4 难的地方不是“怎么写 free”，而是要先回答清楚下面四个问题：

```text
1. 一个 block 什么时候已经不再属于 request？
2. 一个 block 什么时候 GPU 不会再访问？
3. 谁应该决定 free 的时机？
4. free 以后什么时候才真正变成 allocator 可复用容量？
```

如果这四层没有分清楚，很容易写出看起来合理、实际上错误的逻辑，例如 V2 T3 曾经使用：

```python
self.deferred_frees.append(
    (self.sched_step_seq + 1, removed)
)
```

这段代码只有两行，但语义是错的。

所以 T4 的价值主要不是“增加多少代码”，而是把下面三个层次彻底拆开：

```text
Ownership
    ↓
Execution Lifetime
    ↓
Allocator Reuse
```

最终形成：

```text
SingleTypeKVCacheManager
    = WHICH blocks stop being owned

Scheduler
    = WHEN detached blocks can be released

BlockPool
    = HOW blocks become reusable allocator capacity
```

---

# 1. T4 在整个 P1 中处于什么位置？

P1 当前可以按以下层次理解：

```text
T1 / State
    logical progress / physical progress separation

        ↓

T2 / Worker Physical Transition
    V1: pre-forward whole-block reclaim
    V2: post-forward token/member compaction

        ↓

T3 / Ownership Reconciliation
    Worker result
    → Scheduler canonical req_to_blocks update
    → effective_kv_len commit

        ↓

T4 / Physical Release + Reuse
    detached KVCacheBlock
    → BlockPool.free_blocks()
    → free_block_queue
    → future allocation
```

因此：

```text
T2 = 搬 / 改物理 KV 数据

T3 = 改 canonical ownership

T4 = 真正把不再属于 request 的物理 block
     退回 allocator 并允许未来复用
```

这是理解 T4 最重要的定位。

---

# 2. 为什么只有 `req_to_blocks` 变短还不算 Physical Reclamation？

假设某个 request 原来持有：

```text
req_to_blocks["A"] = [B7, B2, B11]
```

V2 compaction 后只需要一个 block：

```text
req_to_blocks["A"] = [B7]
```

此时 `B2/B11` 已经不再属于 A，但这只能证明：

```text
ownership reclaimed
```

还不能证明：

```text
allocator capacity reclaimed
```

因为 vLLM 的实际 allocator 是：

```text
BlockPool
    ↓
free_block_queue
```

只有真正调用：

```python
BlockPool.free_blocks(...)
```

并让对应 block：

```text
ref_cnt → 0
```

进入：

```text
free_block_queue
```

以后，未来 `BlockPool.get_new_blocks(...)` 才可能重新取到这些 block。

因此完整物理回收必须是：

```text
canonical detach
        ↓
BlockPool.free_blocks()
        ↓
ref_cnt == 0
        ↓
free_block_queue
        ↓
future get_new_blocks()
        ↓
reuse
```

T4 正是在补这一层。

---

# 3. upstream vLLM 的 KV block 管理分层

核心层级可以简化为：

```text
Scheduler
    │ runtime lifecycle / scheduling
    ▼
KVCacheManager
    │ public KV ownership API
    ▼
KVCacheCoordinator
    │ KV-group coordination
    ▼
SingleTypeKVCacheManager
    │ canonical request → block ownership
    │ req_to_blocks
    ▼
BlockPool
    │ refcount / free queue
    ▼
FreeKVCacheBlockQueue
```

其中最重要的是：

```text
SingleTypeKVCacheManager.req_to_blocks
```

回答：

```text
“这个 request 当前 canonical-owned 哪些 block？”
```

而：

```text
BlockPool.free_block_queue
```

回答：

```text
“哪些 block 当前可以重新分配？”
```

这两个问题必须分开。

---

# 4. upstream 正常 request free 链路

正常 request 生命周期结束时，大致是：

```text
Scheduler
    ↓
_free_request()
    ↓
_free_blocks()
    ↓
_free_request_blocks()
    ↓
KVCacheManager.free(request)
    ↓
KVCacheCoordinator.free(request_id)
    ↓
SingleTypeKVCacheManager.free(request_id)
    ↓
pop_blocks_for_free()
    ↓
BlockPool.free_blocks(...)
```

`SingleTypeKVCacheManager.free()` 本质是：

```python
self.block_pool.free_blocks(
    reversed(self.pop_blocks_for_free(request_id))
)
```

这里其实包含两个动作：

```text
pop_blocks_for_free()
    = ownership detach

BlockPool.free_blocks()
    = allocator release
```

普通 request finish 可以整体释放，但 P1 的 V1/V2 是：

```text
request 仍然存活
只回收部分 physical KV
```

所以不能直接调用 whole-request free path，而必须有 partial reconciliation。

---

# 5. 先区分：Continuous Batching ≠ Multiple In-Flight Batches

这是理解 `defer_block_free` 之前最容易混淆的一点。

P1 当前一直在使用 continuous batching。一个 Scheduler step / execution batch 内可以同时包含多个 request：

```text
Step N:

Request A → decode 1 token
Request B → decode 1 token
Request C → prefill 64 tokens
Request D → decode 1 token
```

它们会被组织成同一个 `SchedulerOutput` / execution batch。

因此：

```text
一个 batch 内有多个 request / seq
≠
同时存在多个未完成 batch
```

可以把两个概念写成：

```text
Continuous Batching
= 一个 execution batch 内动态混合多个 request

Multiple In-Flight Batches
= 多个 SchedulerOutput / execution batch
  同时处于尚未完全 settle 的状态
```

例如同步 scheduling：

```text
Batch N
    ↓
GPU execute
    ↓
update_from_output(N)
    ↓
Batch N+1
```

此时：

```text
max_concurrent_batches = 1
```

即使 Batch N 里面有很多 request，也仍然只有一个 batch in flight。

而 async scheduling / PP 可以形成：

```text
Batch N      ────────────────>
Batch N+1         ────────────────>
Batch N+2              ────────────────>
```

此时才是：

```text
multiple in-flight batches
```

---

# 6. vLLM 0.26.0 的 `async_scheduling` 到底是什么？

## 6.1 “不配置”不是默认 False，而是 Auto

`SchedulerConfig` 中：

```python
async_scheduling: bool | None = None
```

所以用户不显式配置时，初始语义是：

```text
None
= auto
```

随后 `VllmConfig` 会根据 runner / executor / speculative decoding / platform 等兼容性条件，把它解析为最终的：

```text
True
or
False
```

对于普通 GPU generate、executor 支持 async 且没有不兼容项的场景，vLLM 0.26.0 会进入：

```python
self.scheduler_config.async_scheduling = True
```

因此必须区分：

```text
正式 vLLM 默认配置
= None → compatibility resolution → 常见情况下 True

P1 当前 core test fixture
= 显式传入 async_scheduling=False
```

P1 测试中看到：

```text
Asynchronous scheduling is disabled.
```

只能证明：

```text
当前测试 fixture 是 synchronous
```

不能推出：

```text
vLLM 正式 Engine 不配置时默认关闭 async
```

## 6.2 这里的 “async” 是哪个层面的异步？

这里不是 HTTP asyncio，也不是 KV Connector async load，更不是单纯说 CUDA kernel launch 本身异步。

这里的 `async_scheduling` 主要描述：

> GPU 正在执行 Step N 时，CPU Scheduler / Worker 可以继续推进 Step N+1 的 scheduling / preparation，从而让 Host 侧工作与 Device execution overlap。

同步模式近似：

```text
CPU schedule/prepare N
        ↓
GPU execute N
        ↓
CPU consume N
        ↓
CPU schedule/prepare N+1
        ↓
GPU execute N+1
```

Async Scheduling 目标是：

```text
GPU:
        [========= N =========]
                         [========= N+1 =========]

CPU:
schedule/prepare N

        schedule/prepare N+1

                         schedule/prepare N+2
```

也就是：

```text
GPU executes N
||
CPU prepares N+1
```

这是一个 Scheduler / Worker / GPU execution pipeline overlap。

## 6.3 CUDA async 与 vLLM Async Scheduling 的关系

更底层还有 CUDA 自身的异步执行：

```text
CPU enqueue CUDA work
→ 不等待 kernel 完成
→ CPU 可以继续执行
```

MRV2 的 async-first 设计尽量减少 CPU synchronization point，让 CPU 能继续准备后续 batch。

因此可以理解成两层：

```text
Layer 1:
CUDA asynchronous execution
CPU enqueue → GPU stream execution

        ↓ 被上层利用

Layer 2:
vLLM Async Scheduling
GPU executes N
||
CPU Scheduler / Worker prepares N+1
```

Layer 2 建立在 Layer 1 之上，但二者不是同一个配置概念。

---

# 7. `max_concurrent_batches` 为什么和 PP / MRV2 / Async 有关？

vLLM 0.26.0 的核心逻辑：

```python
@property
def max_concurrent_batches(self) -> int:
    pp_size = self.parallel_config.pipeline_parallel_size

    if self.scheduler_config.async_scheduling:
        if self.use_v2_model_runner:
            return pp_size + 1

        if pp_size <= 1:
            return 2

    return pp_size
```

这个量不是 batch size，也不是 max_num_seqs，而更接近：

```text
系统最多允许多少个 execution batch
同时处于未完全 settle / in-flight 状态
```

## 7.1 PP 本身为什么会带来多个 in-flight batch？

例如：

```text
PP = 2
```

为了填 pipeline：

```text
time →

Batch A:
Stage 0 ███
        Stage 1 ███

Batch B:
        Stage 0 ███
                Stage 1 ███
```

当 Batch A 在 Stage 1 时，Batch B 已经在 Stage 0。

所以即使：

```text
async_scheduling = False
```

仍然可能：

```text
max_concurrent_batches = pp_size = 2
```

因此：

```text
PP > 1
→ multiple_inflight_batches 可能成立
```

## 7.2 为什么 MRV2 + Async 是 `pp_size + 1`？

MRV2 本身不等于 async。

真正条件是：

```text
async_scheduling=True
AND
use_v2_model_runner=True
```

才走：

```python
return pp_size + 1
```

可以把它理解成：

```text
PP pipeline occupancy
+
1 个 async lookahead batch
```

例如：

```text
PP = 1
async = True
MRV2 = True
```

得到：

```text
max_concurrent_batches = 2
```

即：

```text
1 个 current GPU batch
+
1 个 async next/lookahead batch
```

如果：

```text
PP = 4
async = True
MRV2 = True
```

则：

```text
4 个 batch 用于填 PP pipeline
+
1 个额外 async lookahead batch
=
5
```

---

# 8. KV Connector Consumer 是什么？

vLLM KV transfer 中可以把角色理解成：

```text
Producer
= 本地已经算好 KV → 向外发送

Consumer
= 外部已经算好 KV → 加载进本地 GPU KV blocks

Both
= 同时具备 Producer + Consumer 能力
```

最典型场景是 Prefill / Decode 分离：

```text
Prefill Instance
    ↓
计算 Prompt KV
    ↓
KV Producer
    ↓
transfer
    ↓
KV Consumer
    ↓
Decode Instance
    ↓
直接基于外部 KV decode
```

Consumer 的关键不是“有一个 connector 对象”，而是它真的会参与本地 physical KV block 写入。

典型链路：

```text
Scheduler
    ↓
查询 local KV hit
    ↓
Connector 查询 external KV
    ↓
发现外部已有 KV
    ↓
Scheduler 为 request 分配本地 GPU KV blocks
    ↓
Connector 得到这些 block IDs / block objects
    ↓
external KV
    ↓
network / DMA / connector load
    ↓
local GPU KV blocks
```

也就是说 Consumer 是一个 external KV writer，它可能向刚从本地 `BlockPool` 分配出来的 physical blocks 写入 KV 数据。

---

# 9. 为什么 upstream deferred-free 是 `Consumer + Multiple In-Flight` 才启用？

Scheduler 初始化的核心逻辑：

```python
self.defer_block_free = False

multiple_inflight_batches = (
    self.vllm_config.max_concurrent_batches > 1
)

if (
    multiple_inflight_batches
    and kv_transfer_config.is_kv_consumer
):
    self.defer_block_free = True
```

这里是一个非常有针对性的 lifetime protection。

## 9.1 Multiple In-Flight 带来什么风险？

假设：

```text
Batch N
Request A
GPU 正在写 physical block B7
```

如果有 async scheduling / PP，Batch N 还未完全 settle 时，CPU Scheduler 可能已经继续推进后续生命周期。

于是可能出现：

```text
Request A 被 abort / finish
→ CPU ownership 移除 B7
→ B7 看起来已经可以 free
```

此时旧 GPU execution 却可能仍然在访问 B7。

## 9.2 Consumer 又带来什么？

如果 B7 立即进入：

```text
free_block_queue
```

后续 Consumer request B 可能：

```text
allocate B7
    ↓
Connector 将 remote KV load 到 B7
```

于是同时出现：

```text
                B7
               /  \
              /    \
Old GPU Batch N    KV Consumer Load
still writes       writes remote KV
```

两个写入者没有正确 ordering，这就是 physical block reuse race。

所以 upstream 的策略是：

```text
ownership 可以先 detach
但是 block 暂时不能 reusable
```

等待对应 execution fence 完成后：

```text
deferred_frees
    ↓
processed_step_seq >= fence_seq
    ↓
BlockPool.free_blocks()
```

## 9.3 为什么只有 Consumer 不够？

如果：

```text
max_concurrent_batches = 1
```

那么：

```text
Batch N
    ↓
GPU 完成
    ↓
update_from_output()
    ↓
后续 free / reuse
```

旧 batch 已经完成，因此 Consumer 后续 reuse B7 时不存在旧 GPU writer。

## 9.4 为什么只有 Multiple In-Flight 也不够？

例如：

```text
PP = 2
connector = None
```

则：

```text
multiple_inflight_batches = True
```

但没有 Consumer，也就没有 external KV load 对刚 reuse 的 B7 做 out-of-band write。

因此 upstream 这里保护的特定 hazard 是：

```text
Old GPU write
VS
New Connector Consumer write
```

需要特别强调：

```text
defer_block_free=False
```

并不等价于：

```text
max_concurrent_batches == 1
```

例如：

```text
PP = 2
connector OFF
```

可能：

```text
multiple_inflight_batches = True
defer_block_free = False
```

所以判断是否存在 overlap，要看 `max_concurrent_batches`，不能只看 `defer_block_free`。

---

# 10. `sched_step_seq / processed_step_seq` 的真实语义

Scheduler 有：

```python
self.sched_step_seq = 0
self.processed_step_seq = 0
```

以及：

```text
deferred_frees = deque[(fence_seq, blocks)]
```

关键是它们只有在 `defer_block_free=True` 时才推进：

```python
if self.defer_block_free and total_num_scheduled_tokens > 0:
    self.sched_step_seq += 1
```

以及：

```python
if (
    self.defer_block_free
    and scheduler_output.total_num_scheduled_tokens > 0
):
    self.processed_step_seq += 1
    self._drain_deferred_frees()
```

因此它们不是 generic engine-step counter，而是 conditional deferred-free protocol counter。

这正是 V2 T3 原实现出现问题的根源。

---

# 11. 当前 P1 中 KV Connector 实际走到了什么程度？

当前 P1 的验证配置：

```text
connector = None
defer_block_free = False
PP = 1
async_scheduling = False   # 当前 test fixture 显式关闭
max_concurrent_batches = 1
```

因此当前 P1 **没有启用 KV Connector**。

也就是说以下路径都没有参与当前 T4 correctness：

```text
remote KV lookup                 ❌
external KV hit                  ❌
KV Connector allocation binding ❌
WAITING_FOR_REMOTE_KVS           ❌
async KV load                    ❌
KV connector metadata            ❌
LMCache / NIXL transfer          ❌
```

当前链路是纯 local：

```text
Model Forward
    ↓
Local GPU KV
    ↓
SingleTypeKVCacheManager
    ↓
BlockPool
    ↓
Local GPU KV Pool
```

所以当前 T4 研究的是：

```text
partial local ownership detach
    ↓
什么时候把 removed block
重新交回 local BlockPool
```

而不是 remote/offloaded KV lifecycle。

---

# 12. 为什么 `update_from_output()` 是当前 T4 的关键 completion seam？

upstream 在 `Scheduler.update_from_output()` 中明确写：

```python
# Every GPU write enqueued by this and earlier steps has completed,
# so it is safe to return deferred-free blocks to the pool.
```

在当前 P1 **已验证的 test/runtime slice**：

```text
async_scheduling = False
PP = 1
max_concurrent_batches = 1
KV connector = None
```

因此：

```text
Worker execution
    ↓
ModelRunnerOutput 返回
    ↓
Scheduler.update_from_output()
```

可以作为当前 batch 的 result-time completion seam。

所以当前 T4 才可以安全采用：

```text
canonical detach
    ↓
result-time immediate release
```

注意这里的证明范围必须写清楚：

```text
当前 test fixture / frozen MVP
= 显式 synchronous

正式 vLLM 0.26.0 默认 Engine
= async_scheduling 初始 None / auto
= 兼容情况下可能自动解析成 True
```

因此当前 T4 的 safe-free correctness **不能自动外推**到默认 async Engine、PP > 1 或 KV Consumer Connector。

---

# 13. V1 在 T4 之前是什么样？

V1 是 whole-block physical reclaim，发生在 Worker **pre-forward**。

例如：

```text
old ownership:
[B0, B1, B2, B3]

retained old blocks:
[B0, B2]

same-step new:
[B4]
```

最终 canonical row：

```text
[B0, B2, B4]
```

removed：

```text
[B1, B3]
```

Worker 负责在 forward 前把 execution-facing physical row 改好；Scheduler 后续再提交 canonical ownership。

---

# 14. V1 T4 之前的 manager 行为

原来的：

```python
SingleTypeKVCacheManager.reconcile_reclaimed_blocks(...)
```

同时做：

```text
1. validation
2. canonical req_to_blocks update
3. num_cached_block clamp
4. BlockPool.free_blocks(removed)
5. return block IDs
```

也就是：

```text
manager
= ownership authority
+ release authority
```

这在同步 baseline 下功能上是成立的，因此必须强调：

```text
V1 T4 前不存在 V2 那种 allocator leak。
```

T4 对 V1 的主要作用是职责分离。

---

# 15. V1 T4 之后怎么变？

现在：

```python
SingleTypeKVCacheManager.reconcile_reclaimed_blocks(...)
    -> list[KVCacheBlock]
```

只做：

```text
validation
→ retained / removed
→ req_to_blocks = final_blocks
→ num_cached_block clamp
→ return removed_blocks
```

不再 `BlockPool.free_blocks()`。

因此职责变成：

```text
Manager:
“B1/B3 已经不属于 request A”

Scheduler:
“现在是否可以把 B1/B3 交回 allocator？”
```

---

# 16. V1 result-time 完整链路

现在 `Scheduler.update_from_output()` 中：

```python
removed = self.kv_cache_manager.reconcile_reclaimed_blocks(
    req_id,
    transition.retained_block_ids,
    transition.expected_old_num_blocks,
    same_step_new_block_ids,
)

request.effective_kv_len = (
    transition.new_effective_kv_len
    + num_tokens_scheduled
)

self._release_reconciled_blocks(removed)
```

完整顺序：

```text
Worker pre-forward physical transition
        ↓
forward
        ↓
ModelRunnerOutput
        ↓
Scheduler.update_from_output()
        ↓
canonical ownership commit
        ↓
physical E commit
        ↓
allocator release
```

---

# 17. V2 在 T4 之前是什么样？

V2 是 token/member-level physical compaction，发生在 Worker **post-forward**。

例如：

```text
E_before = 20
q = 1
```

forward 后：

```text
source_E = 21
```

如果 compaction 保留：

```text
K = 8
```

Worker 执行：

```text
source physical members [0,21)
        ↓
gather kept members
        ↓
scratch
        ↓
write back dense prefix [0,8)
        ↓
Worker effective E = 8
```

之后 Scheduler 才负责 ownership reconciliation。

---

# 18. V2 T3 已经完成了什么？

`reconcile_compacted_blocks(...)` 做：

```python
retained = current_blocks[:new_num_blocks]
removed = current_blocks[new_num_blocks:]

self.req_to_blocks[request_id] = retained
return removed
```

例如：

```text
before:
[B7, B2, B11]

new_num_blocks = 1

after:
[B7]

removed:
[B2, B11]
```

这部分本身是正确的，因为它只做 canonical ownership reconciliation。

---

# 19. V2 真正的问题：detach ≠ free

T3 原先在 Scheduler commit 后做：

```python
self.deferred_frees.append(
    (self.sched_step_seq + 1, removed)
)
```

意图是“晚一点再 reuse”。

但当前 MVP：

```text
defer_block_free = False
```

所以：

```text
sched_step_seq = 0
processed_step_seq = 0
```

并不会持续增长。

于是：

```text
removed = [B2, B11]
    ↓
fence = 1
    ↓
deferred_frees.append((1, removed))
```

drain 时：

```text
1 > 0
→ break
```

之后 processed 还是 0，于是永远不 free。

因此 T3 当时真实状态：

```text
V2 payload compaction         PASS
Worker effective E            PASS
canonical req_to_blocks       PASS
Scheduler E                   PASS
allocator free                FAIL
allocator reuse               FAIL
```

这就是 T4 真正修复的 bug。

---

# 20. 为什么不能简单把 `+1` 改成 `+0`？

因为问题不是 off-by-one，而是语义错位。

`sched_step_seq` 不是普通 Engine step sequence，而属于 conditional deferred-free protocol。

此外 upstream CoW 使用 fence 的生命周期点是在 schedule-time，它明确知道 copy 将在哪个 upcoming step 中发生；V2 removed blocks 是 result-time 才产生。

所以正确结论不是：

```text
把 fence 数字改对
```

而是：

```text
重新划清 ownership 与 execution lifetime 的职责
```

---

# 21. T4 新增统一 release helper

Scheduler 新增：

```python
def _release_reconciled_blocks(
    self,
    removed: list[KVCacheBlock],
) -> None:
    if removed:
        self.kv_cache_manager.block_pool.free_blocks(
            reversed(removed)
        )
```

它的语义：

```text
输入：
已经从 canonical ownership detach 的 KVCacheBlock

前提：
当前 result-time seam 下已经安全

动作：
交给 BlockPool

结果：
进入 allocator free/reuse 生命周期
```

---

# 22. 为什么 helper 应该在 Scheduler？

因为 Manager 能判断：

```text
“这个 block 是否还属于 request”
```

Scheduler 更接近判断：

```text
“对应 execution 是否已经完成”
```

所以：

```text
Ownership safety
→ manager

Execution lifetime safety
→ Scheduler/runtime

Allocator state transition
→ BlockPool
```

最终：

```text
SingleTypeKVCacheManager
    ↓ detach
Scheduler
    ↓ release timing
BlockPool
    ↓ allocator reuse
```

---

# 23. V2 T4 之后的完整 result-time 链路

V2 Worker 已完成：

```text
forward
    ↓
post-forward compaction
    ↓
CompactionResultData
```

Scheduler：

```text
update_from_output()
    ↓
_validate_compaction_results()
    ↓
_prepare_compaction_reconciliations()
```

这里验证：

```text
Plan / Result identity
step_seq
source E
source block count
new E
new block count
canonical ownership geometry
```

通过后生成：

```text
_PreparedCompactionReconciliation
```

随后：

```text
_commit_compaction_reconciliations()
    ↓
reconcile_compacted_blocks()
    ↓
removed KVCacheBlock[]
    ↓
request.effective_kv_len = K
    ↓
_release_reconciled_blocks(removed)
```

所以：

```text
V2 Result
→ validate
→ canonical detach
→ E = K
→ BlockPool free
```

闭环完成。

---

# 24. V1 和 V2 最终统一到哪里？

T4 后：

```text
                V1
      pre-forward reclaim
                ↓
             forward
                ↓
              result
                ↓
       canonical reconcile
                │
                │ removed KVCacheBlock[]
                ▼
      Scheduler release helper
                ▲
                │ removed KVCacheBlock[]
                │
       canonical reconcile
                ↑
              result
                ↑
     post-forward compaction
                V2
```

统一流向：

```text
_release_reconciled_blocks()
        ↓
BlockPool.free_blocks()
        ↓
free_block_queue
        ↓
future allocation
```

---

# 25. `BlockPool.free_blocks()` 真正代表什么？

概念上：

```text
block.ref_cnt -= 1
```

只有 refcount 到 0，block 才真正进入 allocator free state。

因此：

```text
req_to_blocks 删除
```

只是 ownership 变化；

```text
ref_cnt → 0 + free_block_queue
```

才是 allocator capacity reclamation。

---

# 26. 为什么 release 使用 `reversed(removed)`？

vLLM 原有 free path 通常按 reverse allocation order 释放，保持 tail-first / free queue ordering 习惯。

T4 没有重新设计 BlockPool policy，只沿用 upstream 行为。

---

# 27. T4 同时恢复了 upstream deferred-free protocol

T3 为了让 V2 deferred queue 尝试工作，曾扩大 `_drain_deferred_frees()` 的调用范围。

T4 恢复 upstream：

```python
if (
    self.defer_block_free
    and scheduler_output.total_num_scheduled_tokens > 0
):
    self.processed_step_seq += 1
    self._drain_deferred_frees()
```

这样：

```text
P1 synchronous partial transition release
```

和：

```text
upstream request/CoW/connector deferred-free
```

重新分离。

---

# 28. upstream deferred-free 现在仍服务于什么？

仍服务于 upstream 自己的：

```text
request free
CoW retained blocks
connector / overlap lifecycle
```

T4 没有修改：

```text
_free_request_blocks()
_free_cow_retained_blocks()
_drain_deferred_frees()
```

的核心 contract。

---

# 29. T4 具体修改文件

## `vllm/v1/core/single_type_kv_cache_manager.py`

`reconcile_reclaimed_blocks()`：

```text
Before:
detach + free + return IDs

After:
detach + return list[KVCacheBlock]
```

`reconcile_compacted_blocks()`：

```text
保持 detach-only
返回 removed KVCacheBlock[]
```

## `vllm/v1/core/kv_cache_coordinator.py`

V1/V2 reconcile API 改为：

```text
list[KVCacheBlock]
```

只做 topology validation / forwarding。

## `vllm/v1/core/kv_cache_manager.py`

同样透传 detached KVCacheBlock objects，不负责 partial-transition free。

## `vllm/v1/core/sched/scheduler.py`

核心改动：

```text
1. V1 capture removed blocks
2. V1 E commit 后 Scheduler release
3. V2 canonical commit 后 Scheduler release
4. 新增 _release_reconciled_blocks()
5. 删除 V2 fake (sched_step_seq + 1) fence
6. 恢复 upstream deferred drain guard
```

---

# 30. Before / After 对照

## V1 Before

```text
Scheduler
    ↓
KVCacheManager
    ↓
SingleTypeManager
    ↓
detach
    ↓
free
```

问题：ownership 和 release timing 耦合在 manager。

## V1 After

```text
Scheduler
    ↓
manager detach
    ↓
Scheduler E commit
    ↓
Scheduler release
```

结果：ownership authority / lifetime authority 分离。

## V2 Before

```text
Scheduler
    ↓
manager detach
    ↓
deferred_frees(sched_step_seq + 1)
```

同步 MVP 下 fence 不满足，allocator capacity 不恢复。

## V2 After

```text
Scheduler
    ↓
manager detach
    ↓
E = K
    ↓
Scheduler immediate release
```

allocator capacity 真正恢复。

---

# 31. V2 完整例子：21 token → 8 token

假设：

```text
block_size = 16
E_before = 20
q = 1
source_E = 21
```

ownership：

```text
[B7, B2]
```

V2 保留：

```text
K = 8
```

Worker：

```text
21 physical members
→ compact
→ dense [0,8)
```

只需 1 block。

Scheduler：

```text
req_to_blocks:
[B7, B2]
→ [B7]

removed:
[B2]

E:
21 → 8

B2
→ BlockPool.free_blocks()
→ free_block_queue
```

下一 request 的 allocation 未来就可以重新消费这部分 capacity。

---

# 32. V1 完整例子

假设：

```text
old canonical row:
[B0, B1, B2, B3]

retain:
[B0, B2]

same-step new:
[B4]
```

Worker pre-forward physical row：

```text
[B0, B2, B4]
```

result-time Scheduler：

```text
canonical:
[B0, B2, B4]

removed:
[B1, B3]
```

随后：

```text
E commit
→ release [B1,B3]
→ free queue
```

request 继续拥有 `[B0,B2,B4]`，这就是 partial reclaim 与 whole-request free 的区别。

---

# 33. T4 测试证据

Focused compaction / reconciliation tests：

```text
8 passed
```

覆盖：

```text
V2 detach-only
invalid geometry rejection
V1 manager detach-only
Plan/Result admission
Result without Plan
Scheduler release helper
no V2 fake deferred fence
```

随后重新运行：

```text
tests/v1/core/test_reclaim_ownership.py
```

最终：

```text
4 passed, 15 warnings
```

重新证明 T4 refactor 后：

```text
V1 release + reuse
validation fail-closed
preemption after commit
normal allocation path
```

仍然成立。

---

# 34. 为什么 `test_reclaim_ownership.py` 很重要？

因为 T4 最大风险不是语法，而是 lifecycle：

```text
detach 后 free 是否真的改变 allocator capacity？
以后是否真的可以 reuse？
finish/preemption 会不会 double free？
```

这组测试正是在验证这些语义。

---

# 35. 最终 ownership / lifetime / allocator 模型

```text
                    Logical State
                num_computed_tokens
                       │
                       ▼
              Physical Progress State
                effective_kv_len
                       │
                       ▼
              Worker Physical View
          BlockTables / effective source E
                       │
                V1 / V2 transition
                       │
                       ▼
          Canonical Ownership Authority
            SingleType.req_to_blocks
                       │
                  detach removed
                       │
                       ▼
              Scheduler Lifetime Gate
              update_from_output()
                       │
                       ▼
               BlockPool.free_blocks
                       │
                       ▼
               free_block_queue
                       │
                       ▼
              future get_new_blocks
```

这张图是 P1 到 T4 为止最重要的整体认知之一。

---

# 36. 为什么说 T4 是“小代码、大语义”？

代码上只是：

```text
V1 return removed objects
V2 return removed objects
Scheduler helper
两个调用点
恢复一个 upstream guard
```

但不做 source audit，很容易犯三种错误：

```text
错误 1：req_to_blocks 变短 = 已经释放容量
错误 2：sched_step_seq + 1 = 下一 Engine step
错误 3：Manager 知道 removed，所以 Manager 就最适合 free
```

T4 真正解决的是：

```text
free 应该在哪一层、在哪一个生命周期点发生
```

---

# 37. T4 最终冻结结论

在当前**已验证的 P1 runtime/test slice**：

```text
MRV2
async_scheduling = False   # P1 test fixture 显式关闭
PP = 1
max_concurrent_batches = 1
single KV group
KV connector OFF
prefix caching OFF
spec decode OFF
CUDA Graph OFF
```

注意：这不是 vLLM 0.26.0 正式 Engine 的天然默认。正式 `SchedulerConfig.async_scheduling` 初始是 `None / auto`，随后由 `VllmConfig` 做 compatibility resolution，兼容场景下可能自动变成 `True`。

因此这里冻结的是：

```text
P1 synchronous verified slice
```

而不是全部默认 vLLM runtime modes。

可以冻结：

```text
V1 ownership detach          PASS
V1 allocator release        PASS
V1 future reuse             PASS

V2 ownership detach          PASS
V2 allocator release        PASS
V2 fake deferred fence       REMOVED

upstream deferred-free       RESTORED / UNCHANGED
focused logic tests          PASS
Scheduler/BlockPool tests    PASS
```

因此：

```text
P1-M5-T4 Core
= CLOSED / PASS
```

---

# 38. 明确不宣称什么

T4 当前不宣称：

```text
async scheduling safe-free correctness
PP > 1 safe-free correctness
KV connector overlap correctness
multi-KV-group correctness
prefix-cache compatibility
spec decode compatibility
CUDA Graph lifecycle correctness
cross-component rollback
HBM / OS memory return
performance gain
```

这里尤其要区分：

```text
async scheduling
→ 可能让 Batch N+1 与 Batch N overlap

PP > 1
→ 即使 async=False，也可能有多个 pipeline batches in flight

KV consumer connector
→ 会把 external KV 写入新分配的 local GPU blocks
```

三者都会改变“ownership 已经 detach”到“physical block 可以安全 reuse”之间的 lifetime reasoning。

因此当前 T4 的 `update_from_output()` immediate release 结论只绑定到已验证的：

```text
async=False
PP=1
max_concurrent_batches=1
connector=None
```

这一 execution regime。

特别注意：

```text
BlockPool reclaim
```

意味着：

```text
vLLM allocator capacity recovered
```

不是：

```text
CUDA HBM returned to driver / OS
```

vLLM KV cache 本来就是预分配 pool。P1 做的是 pool 内 capacity reuse。

---

# 39. 关于未补 runtime fail-closed gate

最终 audit 确认：

```text
_set_prepared_reclaim_plan()
_set_prepared_compaction_plan()
```

当前没有额外针对 async / PP>1 / KV connector 的 P1 admission gate。

本轮决定不继续补该 hardening。

因此正确表述是：

```text
T4 Core 在冻结同步 MVP 范围内 CLOSED。
Unsupported runtime 主动 admission rejection
作为未来 hardening，不属于当前 core closure。
```

不能声称 P1 已支持 async / PP / connector。

---

# 40. 面试 / 汇报版本

可以这样讲：

> V1/V2 在 Worker 完成物理 KV transition 后，Scheduler 还需要同步 canonical block ownership，并把不再属于请求的物理页真正退回 vLLM 的 BlockPool。
>
> 我最开始在 V2 中复用了 vLLM 的 deferred-free step fence，但源码审计后发现 `sched_step_seq / processed_step_seq` 只属于特定 async/connector overlap 协议，并不是通用 engine-step counter；在同步路径下这个 fence 不会推进，会导致 detached blocks 永远无法重新进入 free queue。
>
> 最后我把 partial transition 的职责重新拆分：`SingleTypeKVCacheManager` 只负责 canonical ownership detach，Scheduler 在 `update_from_output()` 的 GPU-completion seam 统一 release，BlockPool 负责 refcount/free-queue/reuse。这样 V1/V2 都形成了 detach → free → reuse 的完整闭环。

这段已经能体现：

```text
源码理解
runtime lifetime
allocator ownership
bug diagnosis
architecture refactor
```

---

# 41. 最终记忆口诀

```text
detach ≠ free

free ≠ HBM return

Manager 管 ownership
Scheduler 管 lifetime
BlockPool 管 reuse
```

再结合 P1：

```text
T2 = 改物理数据
T3 = 改 canonical ownership
T4 = 回 allocator 并允许 reuse
```

到这里，P1 的 V1/V2 physical reclaim lifecycle 才真正闭环。
