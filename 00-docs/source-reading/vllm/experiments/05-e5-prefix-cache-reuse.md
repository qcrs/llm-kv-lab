# E5：vLLM Prefix Cache 命中、Block 复用与 Prefill 跳过动态实验

> 状态：**PASS / Frozen**  
> 对象：vLLM v0.26.0，V1 Engine + V2 Model Runner  
> 实验目标：用最小 A/B 对照动态证明 Prefix Cache 从“历史 KV block”到“新请求 computed frontier 前移”再到“ModelRunner 实际少算 Prompt token”的完整主链。

---

# 1. 实验背景

在 E1～E4 中，我们已经动态确认了 vLLM V1 Scheduler / KV Cache 的多条关键链路：

```text
Request
  ↓
Scheduler.schedule()
  ↓
根据 token budget 决定本轮 scheduled tokens
  ↓
KVCacheManager.allocate_slots()
  ↓
分配 / 扩展 physical KV blocks
  ↓
Scheduler 在 schedule-time 提交 logical progress
  ↓
SchedulerOutput
  ↓
Executor / Worker / V2 ModelRunner
  ↓
forward
  ↓
ModelRunnerOutput
  ↓
Scheduler.update_from_output()
  ↓
settle / append output / finish / free
```

前面的实验容易形成一个直觉：

```text
num_computed_tokens
≈
“这个 Request 自己已经在 GPU 上算过多少 token”
```

E5 专门验证这个理解是否完整。

Prefix Cache 引入以后，一个新 Request 即使自己还没有执行过任何 forward，也可能因为历史 Request 已经计算过相同 Prefix，而直接获得一段“已经有效计算”的 KV。

因此 E5 要回答的核心问题是：

```text
一个新请求 B：

num_computed_tokens = 0
        │
        ▼
Prefix Cache lookup
        │
        ├── 找到哪些历史 physical blocks？
        ├── 命中了多少 tokens？
        └── 历史 block 已经 ref_cnt=0 时还能不能复用？
        │
        ▼
这些 block 如何重新成为 B 的 active ownership？
        │
        ▼
Scheduler 的 computed frontier 是否直接前移？
        │
        ▼
num_new_tokens 是否减少？
        │
        ▼
V2 ModelRunner 是否真的只 forward miss tail？
```

这也是后续 KV offload / LMCache / remote KV restore 项目最关键的 runtime contract 之一：

```text
restore / reuse KV
        ↓
建立 valid KV ownership
        ↓
推进 Scheduler computed frontier
        ↓
skip duplicated Prefill compute
```

---

# 2. 本实验不研究什么

E5 的范围刻意控制在 runtime reuse 主链，不扩展到 Prefix Cache hash 算法内部实现。

本实验不研究：

```text
hash 算法选型
hash chaining 细节
salt / multimodal extra keys
Prefix Cache 性能优化
淘汰策略调优
KVConnector / remote KV
Mamba hybrid cache
```

本实验只回答：

```text
lookup
→ hit blocks
→ ownership/refcount
→ computed frontier
→ miss-tail scheduling
→ ModelRunner forward reduction
```

---

# 3. 固定实验环境

## 3.1 软件环境

```text
vLLM            : 0.26.0
PyTorch         : 2.11.0+cu129
Engine          : V1 Engine
Model Runner    : V2 Model Runner
Attention       : FlashAttention 2
Async scheduling: enabled
CUDA Graph      : disabled（enforce_eager=True）
```

运行环境：

```text
/home/qcrs/learning/llm-kv-lab/.venvs/
    vllm-v026-torch211-cu129-py310/bin/python
```

## 3.2 硬件

```text
GPU: NVIDIA A100 80GB PCIe
CUDA visible devices: 1
TP=1 / PP=1 / DP=1
```

## 3.3 模型

```text
/data/models/Qwen3-0.6B
```

## 3.4 E5 配置

```text
block_size              = 16
max_model_len           = 512
max_num_batched_tokens  = 256
max_num_seqs            = 2
gpu_memory_utilization  = 0.2
enforce_eager           = True
max_tokens              = 2
temperature             = 0
ignore_eos              = True
```

不再使用 E4 的：

```text
num_gpu_blocks_override=43
```

原因是 E5 研究 Prefix reuse，而不是 KV pressure / preemption。

实际初始化可用 KV Cache：

```text
GPU KV cache size: 136,736 tokens
Maximum concurrency for 512 tokens/request: 267.06x
```

因此本实验中的 memory pressure 基本可以忽略。

---

# 4. Workload 设计

## 4.1 为什么选择 97-token Prompt

脚本构造出的 Prompt：

```text
constructed_prompt_tokens = 97
block_size                = 16
```

所以：

```text
97 = 16 × 6 + 1
```

也就是：

```text
Block 1: token   0 ~ 15   FULL
Block 2: token  16 ~ 31   FULL
Block 3: token  32 ~ 47   FULL
Block 4: token  48 ~ 63   FULL
Block 5: token  64 ~ 79   FULL
Block 6: token  80 ~ 95   FULL
Block 7: token  96        PARTIAL
```

这个长度非常适合 Prefix Cache 实验，因为它天然制造：

```text
6 个完整可缓存 Prefix blocks
+
1 个 partial miss-tail block
```

如果 Prefix Cache 工作正常，第二个相同 Prompt 理论上应该命中前 96 tokens，然后只重新 forward 最后 1 token。

但实验前不把“96”当结论，最终仍以 trace 为准。

---

# 5. A/B 实验结构

E5 不是“只跑一次 Prefix ON”。

为了区分：

```text
普通 physical block reuse
```

和：

```text
真正 Prefix KV reuse
```

必须有 OFF / ON 对照。

## 5.1 E5-A：Prefix Cache OFF

```text
enable_prefix_caching=False
```

同一个 Engine 中顺序执行：

```text
Request 0: Prompt P
    ↓ finish
Request 1: 完全相同 Prompt P
```

预期：第二请求即使重新拿到相同 physical block IDs，也必须完整重新 Prefill。

## 5.2 E5-B：Prefix Cache ON

```text
enable_prefix_caching=True
```

仍然是同一个 Engine 中：

```text
Request 0: Prompt P（cache seed）
    ↓ finish
Request 1: 完全相同 Prompt P（replay）
```

预期：Request 0 的 full blocks 形成 cache identity；finish 后 active ownership 释放，但 cached KV 仍可命中；Request 1 重新 adopt 这些 block，并跳过对应 Prefill。

## 5.3 为什么两个请求必须在同一个 Engine 内

Prefix Cache 是 Engine / KV block pool 内部 runtime state。

如果这样做：

```text
Engine A
→ Request 0
→ destroy

Engine B
→ Request 1
```

那么历史 Prefix Cache state 已经不存在，无法验证跨 Request reuse。

所以必须：

```text
同一个 LLM / Engine
    ├── Request 0
    └── Request 1
```

---

# 6. 实验脚本

脚本：

```text
04-experiments/vllm-bridge/scripts/e5_prefix_reuse.py
```

核心结构：

```python
llm = LLM(... enable_prefix_caching=...)

prompt = build_prompt(tokenizer)  # 97 tokens

first = llm.generate([prompt], sampling_params)
second = llm.generate([prompt], sampling_params)
```

最关键的是：

```text
first 完整结束
↓
Engine 不退出
↓
second 使用同一个 Engine
```

A/B 唯一关键变量：

```text
enable_prefix_caching=False / True
```

---

# 7. 为什么现有 Trace 不够

E1～E4 已经有：

```text
[BRIDGE][SCHED][QUEUE_SNAPSHOT]
[BRIDGE][SCHED][BEFORE_COMMIT]
[BRIDGE][SCHED][AFTER_COMMIT]
[BRIDGE][MRV2][EXECUTE_BEGIN]
[BRIDGE][SCHED][SETTLE]
[BRIDGE][SCHED][TOKEN_APPEND]
```

这些能告诉我们：

```text
最终 num_computed_tokens 是多少
最终 scheduled 多少
ModelRunner 最终执行多少
```

但不能直接回答：

```text
为什么新请求 computed 会从 0 变成 96？
命中了哪些 cached blocks？
Request 0 finish 后这些 block 是否还保持 cache identity？
Request 1 命中后 ref_cnt 怎么变化？
```

因此 E5 增加三类最小 instrumentation：

```text
PREFIX_LOOKUP
PREFIX_TOUCH
PREFIX_FREE
```

---

# 8. Trace 1：PREFIX_LOOKUP

## 8.1 位置

文件：

```text
vllm/v1/core/sched/scheduler.py
```

静态主链：

```python
if request.num_computed_tokens == 0:
    ...
    (
        new_computed_blocks,
        num_new_local_computed_tokens,
        request.shared_prefix_boundary,
    ) = self.kv_cache_manager.get_computed_blocks(request)
```

`get_computed_blocks()` 是 waiting request 第一次进入 scheduling 时查询本地已缓存 Prefix 的关键入口。

## 8.2 最终 Trace

```python
# BRIDGE E5 TRACE BEGIN
if os.getenv("QCACHE_BRIDGE_TRACE", "0") == "1":
    print(
        f"[BRIDGE][PREFIX][LOOKUP] "
        f"req={request.request_id} "
        f"status={request.status.name} "
        f"num_tokens={request.num_tokens} "
        f"num_computed_before={request.num_computed_tokens} "
        f"local_hit_tokens={num_new_local_computed_tokens} "
        f"hit_block_ids="
        f"{new_computed_blocks.get_block_ids(allow_none=True)} "
        f"shared_prefix_boundary="
        f"{request.shared_prefix_boundary}",
        flush=True,
    )
# BRIDGE E5 TRACE END
```

## 8.3 为什么放在这里

这条 trace 放在 local lookup 分支完成之后、external KVConnector 查询之前。

因此语义非常明确：

```text
num_computed_before
= lookup 之前这个 Request 自己已有的 progress

local_hit_tokens
= 本地 Prefix Cache 本次直接命中的 token 数

hit_block_ids
= 本地命中的历史 physical KV blocks
```

对 E5-B 第二请求，我们最希望看到：

```text
num_computed_before=0
local_hit_tokens=96
hit_block_ids=([1,2,3,4,5,6],)
```

---

# 9. Trace 2：PREFIX_TOUCH

## 9.1 位置

文件：

```text
vllm/v1/core/block_pool.py
```

函数：

```python
def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
```

源码自身的注释已经明确说明：

```text
touch() 增加 block 的 reference count；
如果 ref_cnt==0，block 当前在 free list 中，touch 会把它从 free queue 移除；
这个函数正用于“另一个相同 prefix 的 request 命中该 block”时。
```

所以 `touch()` 是比在 KVCacheManager 外围猜 ownership 更直接的观察点。

## 9.2 Trace 语义

```text
TOUCH_BEGIN
    block
    ref_before
    was_free
    has_hash
    is_null

TOUCH_END
    block
    ref_after
    has_hash
```

最关键状态转移：

```text
cached block
ref_cnt=0
was_free=True
has_hash=True
        │
        ▼
touch()
        │
        ├── 从 free_block_queue remove
        └── ref_cnt += 1
        │
        ▼
active block
ref_cnt=1
```

因此这条 trace 验证：

```text
Prefix lookup 不只是“知道某个 hash 命中了”

而是历史 cached physical block
真正重新进入当前 Request 的 active ownership
```

---

# 10. Trace 3：PREFIX_FREE

## 10.1 位置

同样在：

```text
vllm/v1/core/block_pool.py
```

函数：

```python
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
```

核心源码逻辑：

```python
for block in ordered_blocks:
    block.ref_cnt -= 1

    if block.ref_cnt == 0 and not block.is_null:
        if block.block_hash is None:
            blocks_without_hash.append(block)
        else:
            blocks_with_hash.append(block)
```

这说明 free path 本身就区分：

```text
blocks_without_hash
blocks_with_hash
```

也就是：

```text
“没有任何 active owner”
和
“是否仍有 Prefix Cache identity”
是两个不同维度
```

## 10.2 Trace 字段

```text
block
ref_before
ref_after
has_hash
is_null
became_free
```

最重要的是观察：

```text
ref_cnt: 1 → 0
```

同时：

```text
has_hash=True / False
```

从而区分：

```text
active ownership release
```

和：

```text
cache identity destruction
```

---

# 11. Instrumentation 调试过程

第一次 E5 运行时：

```text
PREFIX_TOUCH
PREFIX_FREE
```

都正常出现，但：

```text
PREFIX_LOOKUP
```

没有打印。

由于 ON 第二请求已经出现：

```text
TOUCH block 1~6
num_computed=96
scheduled=1
MRV2 execute=1
```

所以 Prefix lookup 机制本身肯定已经发生。

问题被缩小为 instrumentation 问题。

随后检查：

```bash
rg -n -C 15 \
'\[BRIDGE\]\[PREFIX\]\[LOOKUP\]|get_computed_blocks\(request\)' \
vllm/v1/core/sched/scheduler.py
```

发现实际源码中确实没有 `[PREFIX][LOOKUP]`。

再验证 runtime import：

```text
runtime scheduler path:
/home/qcrs/learning/llm-kv-lab/third_party/vllm/
    vllm/v1/core/sched/scheduler.py

PREFIX_LOOKUP: False
QUEUE_SNAPSHOT: True
```

因此最终定位：

```text
不是 Prefix Cache 没命中
不是 env 没开
不是 grep 漏掉
不是 Python import 了另一份 vLLM

而是 PREFIX_LOOKUP trace 根本没有真正写入 scheduler.py
```

补上 trace 后重新运行，最终得到完整证据链。

这个过程也体现了实验方法本身：

```text
现象
→ 排除机制问题
→ 检查 trace source placement
→ 验证 runtime imported path
→ 修复 instrumentation
→ rerun
```

而不是看到缺日志就直接改变 runtime 结论。

---

# 12. E5-A：Prefix Cache OFF 结果

配置：

```text
enable_prefix_caching=False
```

---

## 12.1 Request 0：首次 Prompt

### Lookup

```text
[PREFIX][LOOKUP]
req=0-b0a2ecb1
status=WAITING
num_tokens=97
num_computed_before=0
local_hit_tokens=0
hit_block_ids=None
```

含义：

```text
这是第一次请求
没有任何历史 Prefix
所以 hit=0
```

### Scheduler

```text
BEFORE_COMMIT
num_computed=0
pending=97
scheduled=97
new_block_ids=([1,2,3,4,5,6,7],)
```

然后：

```text
AFTER_COMMIT
num_computed=97
num_in_flight=97
```

### ModelRunner

```text
MRV2 EXECUTE_BEGIN
total_scheduled=97
```

所以 Request 0 是完整 Prefill：

```text
0
→ schedule 97
→ execute 97
```

---

## 12.2 Request 0 finish

FREE trace：

```text
block 7: ref 1→0, has_hash=False
block 6: ref 1→0, has_hash=False
...
block 1: ref 1→0, has_hash=False
```

因为 Prefix Cache OFF，所以所有 block：

```text
has_hash=False
```

即使它们进入 free pool，也没有 Prefix Cache identity。

---

## 12.3 Request 1：相同 Prompt Replay

### Lookup

```text
[PREFIX][LOOKUP]
req=1-b0293880
num_computed_before=0
local_hit_tokens=0
hit_block_ids=None
```

第二请求没有命中任何 Prefix。

### Scheduler

```text
BEFORE_COMMIT
num_computed=0
pending=97
scheduled=97
new_block_ids=([7,6,5,4,3,2,1],)
```

### ModelRunner

```text
MRV2 EXECUTE_BEGIN
total_scheduled=97
```

所以第二请求仍然完整重新计算 97 tokens。

---

# 13. E5-A 最重要的结论：Physical block ID reuse ≠ Prefix Cache reuse

E5-A 中：

```text
Request 0 使用：
[1,2,3,4,5,6,7]

finish 后 free

Request 1 又使用：
[7,6,5,4,3,2,1]
```

如果只看 block ID，很容易错误地说：

```text
“B 复用了 A 的 KV”
```

但真实 trace 明确表明：

```text
local_hit_tokens=0
scheduled=97
MRV2 execute=97
has_hash=False
```

所以这只是：

```text
ordinary free block reuse
```

也就是：

```text
A 释放了一些 physical memory slots
↓
B 后来碰巧拿到了这些 slots
↓
B 仍然需要重新写入 KV
```

而不是：

```text
直接复用旧 KV 内容
```

因此以后读 vLLM trace 时必须区分：

```text
same physical block ID
≠
same valid cached KV
```

---

# 14. E5-B：Prefix Cache ON 结果

配置：

```text
enable_prefix_caching=True
```

---

# 15. E5-B Request 0：Cache Seed

## 15.1 Lookup

第一次请求：

```text
[PREFIX][LOOKUP]
req=0-9e4c32ad
num_computed_before=0
local_hit_tokens=0
hit_block_ids=None
```

符合预期：Prefix Cache 必须先有生产者，第一次请求不能凭空命中。

## 15.2 正常 Prefill

```text
BEFORE_COMMIT
num_computed=0
pending=97
scheduled=97
new_block_ids=([1,2,3,4,5,6,7],)
```

MRV2：

```text
total_scheduled=97
```

所以 Request 0 完整计算 97-token Prompt。

---

# 16. Request 0 finish：Prefix Cache residency 出现

Request 0 完成后：

```text
block=7 ref_before=1 ref_after=0 has_hash=False

block=6 ref_before=1 ref_after=0 has_hash=True
block=5 ref_before=1 ref_after=0 has_hash=True
block=4 ref_before=1 ref_after=0 has_hash=True
block=3 ref_before=1 ref_after=0 has_hash=True
block=2 ref_before=1 ref_after=0 has_hash=True
block=1 ref_before=1 ref_after=0 has_hash=True
```

这条证据非常重要。

它与 97-token / block_size=16 完全对齐：

```text
Block 1~6
= 6 个完整 block
= 96 tokens
= has_hash=True

Block 7
= 1-token partial tail
= has_hash=False
```

因此本实验动态证明：

```text
完整 Prefix blocks 获得 cache identity
partial tail 在本次路径中没有成为可命中的 full cached block
```

---

# 17. `ref_cnt=0` 不等于 KV 已经“消失”

这是 E5 最关键的架构认知之一。

Request 0 finish 后，block 1~6：

```text
ref_cnt = 0
has_hash = True
```

这不能解释成：

```text
ref_cnt=0
→ KV 数据已经无效 / 被清空
```

更准确的是：

```text
ref_cnt=0
→ 当前没有 active request ownership
```

它们进入：

```text
free_block_queue
```

成为：

```text
eviction candidates
```

但因为：

```text
has_hash=True
```

它们仍然保有 Prefix Cache identity，并且在真正被淘汰 / overwrite 之前仍然可以被未来请求命中。

因此 Prefix Cache 下的 free queue 不能简单理解为：

```text
“全都是空白可直接覆盖的块”
```

而应该理解成：

```text
free queue
= 没有 active owner 的 blocks

其中可能包含：

A. has_hash=False
   普通 free blocks

B. has_hash=True
   cached but inactive blocks
   仍具有复用价值，但也是 eviction candidates
```

---

# 18. E5-B Request 1：真正的 Prefix hit

第二个完全相同 Prompt 到来：

```text
[PREFIX][LOOKUP]
req=1-82e8158e
status=WAITING
num_tokens=97
num_computed_before=0
local_hit_tokens=96
hit_block_ids=([1,2,3,4,5,6],)
shared_prefix_boundary=0
```

这是 E5 最关键的 lookup 证据。

直接说明：

```text
Request B 自己还没有任何 computed progress
num_computed_before = 0

但是本地 Prefix Cache 一次命中：
96 tokens
6 physical blocks
```

并且：

```text
6 blocks × 16 tokens/block = 96 tokens
```

与 Prompt 结构完全一致。

---

# 19. Lookup 后为什么需要 TOUCH

Lookup 只解决：

```text
“哪些历史 block 是正确的 Prefix KV？”
```

但这些 block 当前状态是：

```text
ref_cnt=0
was_free=True
```

所以还需要真正把它们重新接入 B 的 active ownership。

日志：

```text
TOUCH_BEGIN block=1 ref_before=0 was_free=True has_hash=True
TOUCH_END   block=1 ref_after=1

...

TOUCH_BEGIN block=6 ref_before=0 was_free=True has_hash=True
TOUCH_END   block=6 ref_after=1
```

所以每个 cached block 都发生：

```text
ref_cnt: 0 → 1
```

同时由于 `was_free=True`：

```text
它原本在 free_block_queue 中
↓
touch()
↓
从 free queue remove
↓
重新成为 active block
```

这就是 Prefix Cache block 的 ownership 生命周期：

```text
Request A active
ref=1
        │
        ▼
A finish
ref=0 + has_hash=True
cached/inactive/free-queue
        │
        ▼
Request B lookup hit
        │
        ▼
touch()
ref=1
active again
```

---

# 20. Prefix Cache reuse 不是 KV copy

日志表现出的机制不是：

```text
找到历史 Prefix KV
↓
copy 到 6 个新的 blocks
```

而是：

```text
找到历史 physical blocks 1~6
↓
直接 touch / adopt 原 block
↓
新的 Request block ownership 指向这些 same physical blocks
```

因此高层可以理解为：

```text
Request A:
logical block 0 → physical block 1
logical block 1 → physical block 2
...

A finish

Request B:
logical block 0 → same physical block 1
logical block 1 → same physical block 2
...
```

这是 physical KV reuse，而不是把 KV 内容重新 materialize 一遍。

---

# 21. Scheduler computed frontier 如何被 Prefix hit 推进

Lookup / touch 之后：

```text
BEFORE_COMMIT
req=B
num_tokens=97
num_computed=96
num_in_flight=0
pending=1
scheduled=1
```

这是整个 E5 在 Scheduler 层最重要的一条证据。

Request B 本来：

```text
num_computed_before=0
```

它自己尚未 forward。

但是 Prefix Cache 命中：

```text
local_hit_tokens=96
```

于是 Scheduler 的有效 computed frontier 直接变成：

```text
0 → 96
```

因此：

```text
pending
= num_tokens - num_computed
= 97 - 96
= 1
```

最终本轮：

```text
scheduled=1
```

---

# 22. `num_computed_tokens` 的更准确语义

E5 修正了之前一个很重要的认知。

错误的简化理解：

```text
num_computed_tokens
= 当前 Request 自己在 GPU 上算过的 token 数
```

E5 证明，更准确应该是：

```text
num_computed_tokens
= 对当前 Request 而言，已经拥有有效计算结果、
  因而下一次 forward 不需要再重复计算的 logical frontier
```

这个 frontier 可以来自：

```text
1. 当前 Request 自己 forward 的 work

2. Local Prefix Cache hit

3. External KV / KVConnector

4. 未来 LMCache / offload restore
```

因此它本质上是：

```text
logical valid-computation frontier
```

而不是：

```text
GPU work counter
```

---

# 23. MRV2 最终只执行 1 token：真正的 Skip Prefill

如果只看到 Scheduler：

```text
scheduled=1
```

还可以质疑 ModelRunner 是否内部仍重新算了 97 tokens。

但 MRV2 trace 直接证明：

```text
[BRIDGE][MRV2][EXECUTE_BEGIN]
total_scheduled=1
per_req={'1-82e8158e': 1}
```

因此完整链是：

```text
Prompt total = 97
        │
        ▼
Prefix lookup hit = 96
        │
        ▼
computed frontier = 96
        │
        ▼
pending = 1
        │
        ▼
SchedulerOutput scheduled = 1
        │
        ▼
MRV2 actual execute = 1
```

这才是真正意义上的：

```text
Prefix Cache hit
→ Skip duplicated Prefill compute
```

---

# 24. 为什么最后 1 token 仍需要计算

本实验：

```text
97 = 96 + 1
```

前 96 tokens 对应 6 个完整 cached blocks。

第 97 token 位于 partial block 7。

日志已经明确：

```text
block 7: has_hash=False
```

所以它没有被本次 full-block Prefix Cache lookup 命中。

因此 B 至少还需要 forward：

```text
1 token
```

这也正好为生成第一个 output token 提供当前 Prompt 尾部所需的 logits。

---

# 25. `new_block_ids=[1..7]` 的语义修正

E5-B 第二请求日志：

```text
LOOKUP hit_block_ids=([1,2,3,4,5,6],)
```

并且 1~6 随后全部 TOUCH：

```text
ref 0→1
```

但是 `BEFORE_COMMIT` 又显示：

```text
new_block_ids=([1,2,3,4,5,6,7],)
```

这说明我们之前对 `new_block_ids` 的口头解释需要修正。

不能再简单理解为：

```text
“本轮所有 newly allocated physical blocks”
```

在 waiting + Prefix hit 路径中，它更接近：

```text
本次 admission 后新增附着到 Request / SchedulerOutput 的 block attachment
```

其中可能同时包含：

```text
1~6：new_computed_blocks
     Prefix Cache hit / touch / adopted blocks

7：真正为了 miss tail writable slot 而拿到的普通 block
```

所以：

```text
new_block_ids
≠
pure fresh allocation list
```

这一点以后分析 E5 / external KV / resumed request 时都要记住。

---

# 26. 为什么 block 7 最终仍然是 physical ID 7

Request A finish 时：

```text
block 7
ref=0
has_hash=False
```

它是普通 free block。

Request B 命中后：

```text
blocks 1~6
```

通过 Prefix Cache 被 touch。

然后 B 还需要最后 1 token，对应一个 writable partial tail。

BlockPool 此时又把 free block 7 分配给 B。

因此 B 最终 block layout 看起来仍然：

```text
[1,2,3,4,5,6,7]
```

但来源完全不同：

```text
1~6 = historical cached KV reuse
7    = ordinary free-block allocation for miss tail
```

---

# 27. OFF / ON 对照总结

| 项目 | Prefix OFF | Prefix ON |
|---|---:|---:|
| Request 0 lookup | 0 | 0 |
| Request 0 full Prefill | 97 | 97 |
| Request 0 finish full blocks has_hash | False | True（1~6） |
| Request 1 lookup hit | 0 | 96 |
| Request 1 hit blocks | None | 1~6 |
| cached block touch | 无 | 1~6，ref 0→1 |
| Request 1 computed before scheduling | 0 | 96 |
| Request 1 Prompt scheduled | 97 | 1 |
| MRV2 Prompt execute | 97 | 1 |
| 输出 token IDs | `[56483,6500]` | `[56483,6500]` |

第二请求 Prompt token-work：

```text
OFF: 97
ON : 1
```

被跳过：

```text
96 / 97 ≈ 98.97%
```

所以本 workload 下约 99% 的重复 Prompt forward token-work 被跳过。

注意：

```text
这证明 scheduled / model-forward token count 大幅减少
```

但不能直接把它等同于：

```text
TTFT 提升 97 倍
GPU latency 降低 99%
GPU FLOPs 精确降低 99%
```

因为仍有：

```text
hash lookup
Scheduler
metadata preparation
kernel launch
sampling
async orchestration
```

等固定或非线性开销。

---

# 28. 输出一致性

OFF / ON 两组：

```text
seed output_token_ids   = [56483, 6500]
replay output_token_ids = [56483, 6500]
```

所以在本次 deterministic workload 下：

```text
完整 Prefill
```

和：

```text
96-token Prefix reuse + 1-token miss-tail forward
```

产生相同输出。

这足以作为 E5 smoke correctness 证据。

但不把单次实验扩大成：

```text
所有模型 / 所有并发 / 所有 Prefix 场景 bitwise correctness 已证明
```

---

# 29. E5 动态主链与源码主链对齐

## 29.1 Scheduler lookup

```text
Scheduler.schedule()
        │
        ▼
waiting request
num_computed_tokens == 0
        │
        ▼
KVCacheManager.get_computed_blocks(request)
        │
        ▼
Prefix lookup / FullAttention cache manager
        │
        ├── new_computed_blocks
        └── num_new_local_computed_tokens
```

动态：

```text
LOOKUP
local_hit_tokens=96
hit_block_ids=[1..6]
```

## 29.2 Cached block adoption

Scheduler 随后调用：

```text
KVCacheManager.allocate_slots(...)
```

其中：

```text
coordinator.allocate_new_computed_blocks(...)
```

把 lookup 返回的历史 computed blocks 接到当前 Request。

最终落到：

```text
BlockPool.touch(blocks)
```

动态：

```text
block1~6
ref 0→1
```

## 29.3 Miss-tail allocation

随后：

```text
coordinator.allocate_new_blocks(...)
```

负责当前仍然缺少 slot 的部分。

本实验 miss tail 只有：

```text
1 token
```

对应 partial block 7。

## 29.4 Scheduler progress

Prefix hit 形成：

```text
num_computed=96
```

于是：

```text
num_new_tokens
= request.num_tokens - num_computed_tokens
= 97 - 96
= 1
```

## 29.5 Data plane

SchedulerOutput 最终：

```text
scheduled=1
```

交给 V2 ModelRunner：

```text
MRV2 EXECUTE_BEGIN total_scheduled=1
```

只对 miss tail forward。

## 29.6 Finish / free

B 完成：

```text
block 1~6
ref 1→0
has_hash=True

block 7
ref 1→0
has_hash=False
```

整个 Prefix Cache ownership 生命周期闭环。

---

# 30. 整体架构图

```text
                    ┌──────────────────────────┐
                    │      Request A / Seed    │
                    │        Prompt=97         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                      Scheduler.schedule()
                                 │
                    LOOKUP: local_hit=0
                                 │
                                 ▼
                    allocate normal KV blocks
                         physical 1..7
                                 │
                                 ▼
                       V2 ModelRunner
                       forward 97 tokens
                                 │
                                 ▼
                      Prefix block identity
                       full blocks 1..6
                         has_hash=True
                                 │
                                 ▼
                         Request A finish
                                 │
                                 ▼
                    BlockPool.free_blocks()
                                 │
            ┌────────────────────┴──────────────────┐
            │                                       │
     block 1..6                               block 7
     ref 1→0                                  ref 1→0
     hash=True                                hash=False
     cached/inactive                          ordinary free
            │
            │ same Engine
            ▼
                    ┌──────────────────────────┐
                    │     Request B / Replay   │
                    │        same Prompt       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                       Prefix Cache lookup
                                 │
                 local_hit_tokens = 96
                 hit_blocks = physical 1..6
                                 │
                                 ▼
                        BlockPool.touch()
                                 │
                        1..6: ref 0→1
                 remove from free_block_queue
                                 │
                                 ▼
                     Scheduler computed=96
                                 │
                      pending = 97-96 = 1
                                 │
                                 ▼
                 allocate miss-tail writable slot
                          physical block 7
                                 │
                                 ▼
                     SchedulerOutput: 1 token
                                 │
                                 ▼
                       V2 ModelRunner
                         forward 1 token
                                 │
                                 ▼
                         normal decode
                                 │
                                 ▼
                            finish
                                 │
                                 ▼
                    blocks 1..6 ref 1→0
                    hash identity retained
```

---

# 31. Prefix Cache 中几个容易混淆的概念

## 31.1 `ref_cnt=0` ≠ block 立即失效

```text
ref_cnt=0
```

表示：

```text
没有 active owner
```

不是：

```text
KV 必然已经被 overwrite
```

如果：

```text
has_hash=True
```

它仍可作为 Prefix Cache candidate 被命中。

---

## 31.2 free queue ≠ 全部是空白块

Prefix Cache ON 时 free queue 中可以有：

```text
ref_cnt=0 + has_hash=True
```

这种“cached but inactive” block。

它同时具有两种身份：

```text
1. 没有 active owner，因此可作为 eviction candidate
2. 仍保存有效 Prefix KV，因此在被淘汰前可直接复用
```

---

## 31.3 physical ID reuse ≠ cached KV reuse

OFF 实验已经证明：

```text
B 再拿到 A 的 block ID
```

不代表：

```text
B 跳过了计算
```

真正 Prefix reuse 必须看到：

```text
hash identity
+
Prefix lookup hit
+
touch/adopt
+
computed frontier advance
+
actual forward reduction
```

---

## 31.4 Lookup ≠ ownership

```text
LOOKUP
```

只是：

```text
发现哪些 cached physical blocks 与当前 Prefix 匹配
```

```text
TOUCH
```

才是：

```text
把这些 cached blocks 重新纳入当前 Request 的 active reference
```

所以：

```text
lookup = discovery

touch = ownership acquisition
```

---

## 31.5 computed frontier ≠ GPU work counter

Prefix hit 可以在没有新 forward 的情况下把：

```text
num_computed_tokens
0 → 96
```

所以它应该理解为：

```text
valid computation frontier
```

而不是：

```text
current-request GPU-compute counter
```

---

# 32. E5 对整个 vLLM 架构认知的补全

E1～E4 主要建立：

```text
Scheduler
= token / request / KV ownership 的控制面

ModelRunner
= 真正执行 forward 的数据面
```

E5 又补了一层：

```text
Scheduler 不只决定“新算多少”

Scheduler 还必须先知道：
“哪些 token 的有效计算结果已经存在？”
```

因此真正流程更完整地是：

```text
Request arrives
        │
        ▼
Scheduler establishes available computation frontier
        │
        ├── local Prefix Cache
        ├── external KV
        └── already-computed current-request work
        │
        ▼
决定剩余 num_new_tokens
        │
        ▼
确保相应 physical KV ownership / slots
        │
        ▼
只把 miss work 交给 ModelRunner
```

所以 production inference runtime 中：

```text
KV availability
```

本身就是调度输入，而不只是 ModelRunner 的一个“缓存实现细节”。

---

# 33. 对后续 KV Offload / LMCache 项目的意义

E5 与后续 Compression-Aware Async KV Offload 的逻辑高度同构。

Prefix Cache 当前做的是：

```text
local cached KV exists
        ↓
lookup
        ↓
restore active ownership
        ↓
computed frontier advance
        ↓
skip prefill
```

未来 external/offload 场景要做：

```text
CPU / SSD / remote packed KV exists
        ↓
lookup metadata
        ↓
H2D / restore / dequant
        ↓
写回 vLLM physical KV slots
        ↓
建立 ownership / valid state
        ↓
告诉 Scheduler external/local computed tokens
        ↓
computed frontier advance
        ↓
skip duplicated prefill
```

因此 E5 不是一个孤立 feature 实验，而是后续 KV restore 路径的最小 runtime 原型。

---

# 34. 实验结果最终判定

E5 已满足核心 DoD：

```text
✓ Prefix OFF：首次 Request lookup=0

✓ Prefix OFF：第二个相同 Prompt lookup=0

✓ Prefix OFF：第二请求完整 scheduled=97

✓ Prefix OFF：MRV2 完整 execute=97

✓ Prefix OFF：finish blocks has_hash=False

✓ Prefix ON：首次 Request lookup=0，并完整 Prefill

✓ Prefix ON：Request 0 finish 后 full blocks 1~6
  ref_cnt 1→0 且 has_hash=True

✓ Prefix ON：partial block 7 has_hash=False

✓ Prefix ON：第二 Request 直接 lookup hit 96 tokens

✓ Prefix ON：明确命中 physical blocks [1..6]

✓ Prefix ON：命中的 blocks 全部 touch
  ref_cnt 0→1

✓ Prefix ON：Scheduler computed frontier 直接到 96

✓ Prefix ON：miss tail = 1 token

✓ Prefix ON：Scheduler scheduled=1

✓ Prefix ON：V2 ModelRunner 实际 execute=1

✓ Prefix ON：最终 output 与 OFF/full recompute 一致

✓ finish 后 cached blocks 再次 ref_cnt 1→0
```

最终：

```text
E5 = PASS
Instrumentation = Complete
Status = Frozen
```

不需要继续增加 E5 必做实验。

---

# 35. 最终一句话模型

如果只保留一句话理解 E5：

> **vLLM Prefix Cache 的核心不是“第二个请求重新拿到相同 block ID”，而是历史请求完成后，full KV blocks 即使 ref_cnt 降到 0 仍以 hashed cached blocks 驻留；新请求通过 Prefix lookup 找回这些特定 physical blocks，`touch()` 将其从 free/evictable 状态恢复为 active ownership，Scheduler 据此把 `num_computed_tokens` 直接推进到命中边界，并只把 miss tail 交给 ModelRunner，从而真正跳过重复 Prefill。**

---

# 36. E1～E5 至此形成的动态控制面主线

经过 E1～E5，现在可以把 Scheduler / KV 控制面动态认知压缩为：

```text
E1
单请求 async scheduling
→ schedule-time logical commit
→ in-flight / reconcile

E2
Chunked Prefill
→ token budget 控制 chunk
→ KV block 随 computed frontier 增长

E3
Continuous Batching
→ running + waiting 混合
→ 多请求共享 global token budget
→ 请求动态加入/退出 batch

E4
KV Pressure
→ admission gate
→ step allocation failure
→ preemption
→ free
→ recompute / thrashing

E5
Prefix Cache
→ historical KV residency
→ lookup
→ touch / ownership restore
→ computed frontier jump
→ skip Prefill
```

因此当前已经不再只是“知道 vLLM 有 Scheduler / KVCacheManager / BlockPool”，而是已经用 runtime trace 把它们之间最重要的状态转换动态跑通。

