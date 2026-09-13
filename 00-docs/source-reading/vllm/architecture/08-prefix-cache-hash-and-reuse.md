# 08-vLLM Prefix Cache：Hash、注册、命中与 Prefill 跳过源码主链

> 环境基线：vLLM v0.26.0，commit `568afb3a13806beb53bb2e6bd518269357b237c0`  
> 当前主路径：V1 Engine + V2 Model Runner，普通 Full Attention，Prefix Cache 主链  
> 本文目标：不是研究 Prefix Cache 的所有优化，而是把 **Hash 如何产生、何时注册、如何命中、如何跳过 Prefill、命中的 KV 如何重新接回 ModelRunner 数据面** 这一整条链闭环。

---

# 0. 为什么需要单独梳理 Prefix Cache

前面已经把 vLLM 的主链梳理成：

```text
Request
↓
EngineCore
↓
Scheduler
↓
KVCacheManager
↓
BlockPool
↓
SchedulerOutput
↓
Worker / ModelRunner
↓
block_table / slot_mapping
↓
Attention
↓
KV write / paged KV read
```

但是 Prefix Cache 会在这条主链中插入一个新的问题：

> 一个新 Request 在真正进入 Prefill 之前，能不能发现“前面一段 Prefix 的 KV 已经被别的 Request 算过了”，从而直接复用？

它不会绕开 Scheduler，也不会在 ModelRunner 中突然执行一个 `skip_prefill()`。

真正的设计是：

```text
新 Request
↓
先检查 Prefix Cache
↓
得到：
1. 已经可用多少 token 的 KV
2. 这些 KV 对应哪些 physical blocks
↓
推进 computed frontier
↓
减少 num_new_tokens
↓
Scheduler 从一开始就不再调度命中的 Prefix
↓
ModelRunner 只计算 miss tail
同时通过 block_table 读取复用的历史 KV
```

所以 Prefix Cache 同时影响两个世界：

```text
控制面：
命中了多少 token？
→ num_computed_tokens
→ num_new_tokens 减少

数据面：
命中的 KV 在哪？
→ physical KV blocks
→ block_table
→ Attention 读取
```

---

# 1. 先建立最核心的抽象：Prefix Cache 到底缓存什么

假设：

```text
Request A:
[P0 P1 P2 ... P47] [A-specific tail...]

Request B:
[P0 P1 P2 ... P47] [B-specific tail...]
```

前 48 个 token 完全相同。

普通做法：

```text
Request A
↓
Prefill P0~P47
↓
得到 KV

Request B
↓
再次 Prefill P0~P47
↓
再次得到相同 Prefix 对应的 KV
```

Prefix Cache 想做的是：

```text
Request A
↓
算过 Prefix
↓
留下可复用 KV

Request B
↓
发现 Prefix 相同
↓
直接复用 A 留下的 KV
↓
不再计算 P0~P47
```

为了做到这一点，系统必须解决两个完全不同的问题：

```text
问题 1：
怎么知道 Request B 的 Prefix
和 Request A 的 Prefix 是同一个？

问题 2：
如果是同一个 Prefix，
它已经计算好的 KV 到底在 GPU 哪个 physical block？
```

vLLM 分别使用：

```text
BlockHash
= Prefix identity
= “这个 Prefix 是谁？”

KVCacheBlock / physical block_id
= KV physical identity/location
= “这个 Prefix 的 KV 在哪里？”
```

Prefix Cache 的核心索引最终就是：

```text
(BlockHash, KVCacheGroupId)
        ↓
physical KVCacheBlock
```

在当前普通单 KV group 的理解中，可以简化成：

```text
H0 → block17
H1 → block31
H2 → block46
```

---

# 2. 第一条必须牢牢记住的边界：Hash 不是 KV Cache

这是整个 Prefix Cache 最容易混淆的一点。

假设：

```text
hash_block_size = 16
Prompt = 32 tokens
```

Request 在刚创建时，就可能已经得到：

```text
H0 = prefix[0:16] 的 hash identity
H1 = prefix[0:32] 的 hash identity
```

但这个时间点：

```text
Scheduler 可能还没有 schedule
GPU 可能还没有 forward
KV 可能还没有写入 GPU cache
```

因此：

```text
Request.block_hashes 已经存在
```

并不意味着：

```text
Prefix Cache 中已经存在对应 KV
```

要把整个机制拆成三个世界：

```text
1. Token / Identity World
-------------------------
token_ids
↓
BlockHash
↓
“Prefix 是谁？”


2. Execution / KV World
-----------------------
Scheduler
↓
ModelRunner
↓
Attention
↓
K/V 计算并写入 physical KV blocks


3. Prefix Cache Index World
---------------------------
BlockHash
+
physical KVCacheBlock
↓
cached_block_hash_to_block
```

所以：

```text
Hash 生成
≠
KV 计算完成
≠
Prefix Cache 注册
```

这三个动作必须分开理解。

---

# 3. 第一阶段：EngineCore 创建 Prefix hasher

## 3.1 实际源码追踪

先通过：

```bash
rg -n \
"get_request_block_hasher|block_hasher" \
vllm/v1
```

定位到：

```text
vllm/v1/engine/core.py
vllm/v1/request.py
vllm/v1/core/kv_cache_utils.py
```

核心初始化代码：

```python
self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None

if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
    caching_hash_fn = get_hash_fn_by_name(
        vllm_config.cache_config.prefix_caching_hash_algo
    )
    init_none_hash(caching_hash_fn)

    self.request_block_hasher = get_request_block_hasher(
        hash_block_size, caching_hash_fn
    )
```

调用关系：

```text
EngineCore.__init__()
│
├── enable_prefix_caching ?
│       或
│   kv_connector != None ?
│
├── get_hash_fn_by_name(...)
│
├── init_none_hash(...)
│
└── get_request_block_hasher(...)
        ↓
self.request_block_hasher
```

## 3.2 为什么由 EngineCore 创建

因为下面这些东西属于 Engine runtime configuration：

```text
是否启用 Prefix Cache
prefix_caching_hash_algo
hash_block_size
是否存在 KV Connector
```

而不是 Request 自己的属性。

所以职责是：

```text
EngineCore
= 决定 Prefix hashing policy

Request
= 保存 token sequence，并使用 EngineCore 提供的 hasher
```

这也是一个典型的 dependency injection：

```text
EngineCore
↓
创建 callable
↓
注入 Request
```

Request 不需要知道当前配置到底使用什么 hash algorithm、hash block size 多大。

---

# 4. 第二阶段：hasher 如何进入 Request

Request 的预处理路径中：

```python
req = Request.from_engine_core_request(
    request,
    self.request_block_hasher
)
```

所以：

```text
EngineCore._preprocess_engine_core_request(...)
↓
Request.from_engine_core_request(...)
```

`Request.from_engine_core_request()`：

```python
@classmethod
def from_engine_core_request(
    cls,
    request: EngineCoreRequest,
    block_hasher: Callable[["Request"], list["BlockHash"]] | None,
) -> "Request":
    return cls(
        ...
        block_hasher=block_hasher,
        ...
    )
```

最终链路：

```text
EngineCore.request_block_hasher
↓
Request.from_engine_core_request(..., block_hasher)
↓
Request.__init__(..., block_hasher=...)
↓
self._block_hasher
```

最终 Request 内部形成：

```text
Request
│
├── all_token_ids
├── block_hashes
└── _block_hasher
```

---

# 5. 第三阶段：Request 一创建就会尝试产生 BlockHash

Request 初始化后：

```python
self.block_hashes: list[BlockHash] = []
self._block_hasher = block_hasher
self.update_block_hashes()
```

于是：

```text
Request.__init__()
↓
self.block_hashes = []
↓
self._block_hasher = block_hasher
↓
self.update_block_hashes()
```

`update_block_hashes()`：

```python
def update_block_hashes(self) -> None:
    """Compute block hashes for any new full blocks and append them."""
    if self._block_hasher is not None:
        self.block_hashes.extend(self._block_hasher(self))
```

关键点：

```text
Request 创建时
Prompt token_ids 已经知道
↓
即使 GPU 还没运行
也已经可以计算 Prefix identity
```

所以：

```text
Hash 依赖 token identity
不依赖 KV 是否已经写入 GPU
```

---

# 6. `request_block_hasher()` 如何增量计算 Hash

hasher 真正来自：

```python
get_request_block_hasher(...)
```

其内部返回：

```python
def request_block_hasher(request: Request) -> list[BlockHash]:
```

核心第一行：

```python
start_token_idx = len(request.block_hashes) * hash_block_size
```

这是整个增量机制的关键。

假设：

```text
hash_block_size = 4
request.block_hashes = [H0, H1]
```

说明已经处理：

```text
H0 → token[0:4]
H1 → token[0:8] 的 prefix identity
```

因此：

```text
start_token_idx
= 2 × 4
= 8
```

下一次不会重新计算 token0~7，而是直接从 token8 开始。

所以：

```text
已有 Hash
[H0, H1]

sequence 增长
↓
request_block_hasher()
只返回新增 Hash
↓
例如 [H2]

Request.update_block_hashes()
↓
extend

最终：
[H0, H1, H2]
```

---

# 7. 什么时候会生成一个新的 BlockHash

源码：

```python
if start_token_idx + hash_block_size > num_tokens:
    # Early stop when there no new full blocks created.
    return []
```

循环中：

```python
end_token_idx = start_token_idx + hash_block_size

if end_token_idx > num_tokens:
    # We only hash full blocks
    break
```

因此严格结论是：

> **只有凑满一个 `hash_block_size`，才生成新的 BlockHash。**

例如：

```text
hash_block_size = 16

num_tokens = 15
→ []

num_tokens = 16
→ [H0]

num_tokens = 17
→ [H0]

num_tokens = 31
→ [H0]

num_tokens = 32
→ [H0, H1]
```

这里说的是 `full hash block`，不是“GPU physical KV block 已写满”。

---

# 8. 为什么 Decode 过程中仍然不断调用 `update_block_hashes()`

在：

```python
append_output_token_ids(...)
```

尾部：

```python
self.update_block_hashes()
```

调用链：

```text
Sampler 产生新的 output token
↓
Scheduler / Request append output token
↓
Request.append_output_token_ids(...)
↓
_all_token_ids 增长
↓
update_block_hashes()
↓
如果又跨过一个 hash_block_size boundary
则产生新的 BlockHash
```

例如：

```text
hash_block_size = 4

初始已有 10 tokens
↓
H0 H1
剩余 token8 token9 不满一块

生成 O0
↓
11 tokens
仍然没有新 Hash

生成 O1
↓
12 tokens
↓
token8~11 满 4 个
↓
生成 H2
```

所以 `Request.block_hashes` 会随着 sequence 生命周期不断增长。

---

# 9. `hash_block_tokens()`：为什么 Hash 是链式的

实际代码：

```python
def hash_block_tokens(
    hash_function,
    parent_block_hash,
    curr_block_token_ids,
    extra_keys=None,
) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)

    return BlockHash(
        hash_function(
            (
                parent_block_hash,
                curr_block_token_ids_tuple,
                extra_keys,
            )
        )
    )
```

所以并不是：

```text
Hi = hash(current block tokens)
```

而是：

```text
H0 = hash(NONE_HASH, block0_tokens, extra_keys)
H1 = hash(H0, block1_tokens, extra_keys)
H2 = hash(H1, block2_tokens, extra_keys)
```

这意味着：

```text
H0
= prefix ending at block0

H1
= prefix ending at block1

H2
= prefix ending at block2
```

因此 `BlockHash` 更准确的理解是：

> **某个 hash boundary 位置上的完整 Prefix fingerprint。**

---

# 10. 为什么必须带 `parent_block_hash`

假设：

```text
Request A:
[A B C D]
[E F G H]

Request B:
[X Y Z W]
[E F G H]
```

第二个 token block 完全一样，但前面的 Prefix 不一样。

如果只计算：

```text
hash([E F G H])
```

A/B 第二块就会被错误地视为同一 cache identity。

实际：

```text
Request A:
H0A = hash(ROOT, [A B C D])
H1A = hash(H0A, [E F G H])

Request B:
H0B = hash(ROOT, [X Y Z W])
H1B = hash(H0B, [E F G H])
```

因为：

```text
H0A != H0B
```

所以：

```text
H1A != H1B
```

因此只有“从 token0 开始的整个 Prefix 一致”才可能命中。

---

# 11. Hash Producer 调用链总结

```text
EngineCore.__init__()
│
├── enable_prefix_caching / kv_connector
├── get_hash_fn_by_name(...)
└── get_request_block_hasher(...)
        ↓
self.request_block_hasher
        │
        ▼
EngineCore._preprocess_engine_core_request(...)
        │
        ▼
Request.from_engine_core_request(
    request,
    self.request_block_hasher
)
        │
        ▼
Request.__init__()
        │
        ├── self.block_hashes = []
        ├── self._block_hasher = block_hasher
        └── update_block_hashes()
                │
                ▼
        request_block_hasher(request)
                │
                ├── start_token_idx
                ├── full hash boundary check
                └── hash_block_tokens(...)
                        │
                        ▼
             chained BlockHash(es)
                        │
                        ▼
          Request.block_hashes.extend(...)
```

这条链只完成：

```text
token sequence
↓
Prefix identity
```

还没有完成：

```text
Prefix identity
↓
physical KV block
```

---

# 12. 第四阶段：Request A 进入正常 Scheduler / KV allocation

回到主链：

```text
Scheduler.schedule()
↓
KVCacheManager.allocate_slots()
↓
Coordinator
↓
SingleTypeKVCacheManager
↓
BlockPool
↓
physical KV blocks
```

假设：

```text
block_size = 16
Request A = 32 tokens
```

Scheduler 为它分配：

```text
logical block0 → physical block17
logical block1 → physical block31
```

此时同时有：

```text
Request.block_hashes = [H0, H1]
KV blocks = [17, 31]
```

现在才具备把 Prefix identity 和 physical block 绑定起来的条件。

---

# 13. Prefix Cache registration 从哪里触发

通过：

```bash
rg -n \
"num_tokens_to_cache|cache_blocks" \
vllm/v1/core/kv_cache_manager.py
```

定位到 `KVCacheManager.allocate_slots()` 内：

```python
if not self.enable_caching or delay_cache_blocks:
    return self.create_kv_cache_blocks(new_blocks)

num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,
    request.num_tokens,
)

self.coordinator.cache_blocks(
    request,
    num_tokens_to_cache,
)
```

因此注册链：

```text
Scheduler.schedule()
↓
KVCacheManager.allocate_slots(...)
↓
计算 num_tokens_to_cache
↓
Coordinator.cache_blocks(...)
↓
SingleTypeKVCacheManager.cache_blocks(...)
↓
BlockPool.cache_full_blocks(...)
↓
BlockPool._insert_block_hash(...)
↓
cached_block_hash_to_block.insert(...)
```

---

# 14. `num_tokens_to_cache` 在表达什么

```python
num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,
    request.num_tokens,
)
```

普通路径可理解为：

```text
total_computed_tokens
= 本轮之前已 considered computed 的 frontier

num_new_tokens
= 本轮 Scheduler 准备推进的 compute delta
```

所以：

```text
total_computed_tokens + num_new_tokens
≈ 本轮结束后可进入 cache bookkeeping 的 frontier
```

这里一个非常重要的时序点是：

> `cache_blocks()` 发生在 Scheduler / `allocate_slots()` 阶段，而不是 GPU forward 完成后的 callback。

因此时间线是：

```text
Scheduler.schedule()
↓
决定本轮 work
↓
allocate_slots()
↓
cache metadata bookkeeping
↓
SchedulerOutput
↓
Executor
↓
ModelRunner
↓
GPU forward / KV write
```

要区分：

```text
Prefix Cache control-plane registration
```

和：

```text
GPU kernel physical completion
```

它们不是同一时刻。

---

# 15. SingleTypeKVCacheManager：什么时候一个 physical block 可进入 full-block registration

关键逻辑：

```python
num_full_blocks = num_tokens // self.block_size

if num_cached_blocks >= num_full_blocks:
    return
```

然后：

```python
self.block_pool.cache_full_blocks(
    request=request,
    ...
    num_cached_blocks=num_cached_blocks,
    num_full_blocks=num_full_blocks,
    block_size=self.block_size,
    ...
)
```

假设：

```text
physical block_size = 16
```

则：

```text
15 tokens → 0 full block
16 tokens → 1 full block
31 tokens → 1 full block
32 tokens → 2 full blocks
```

所以标准 `cache_full_blocks()` 路径是按完整 physical block 注册。

---

# 16. `num_cached_blocks` 为什么存在

假设：

```text
之前 num_cached_blocks = 2
当前 num_full_blocks = 3
```

说明：

```text
block0 已注册
block1 已注册
block2 是本轮新完整的 block
```

因此只需要注册 block2。

这和 Hash 生产的增量机制非常类似：

```text
Hash side：
只计算新跨过的 hash boundary

Cache registration side：
只注册新跨过的 physical cache boundary
```

---

# 17. BlockPool 真正把 Hash 与 physical block 配起来

通过：

```bash
rg -n \
"def cache_full_blocks|_insert_block_hash|cached_block_hash_to_block" \
vllm/v1/core/block_pool.py
```

定位：

```text
cache_full_blocks()
↓
_insert_block_hash()
↓
cached_block_hash_to_block.insert(...)
```

`cache_full_blocks()` 会同时得到：

```text
new_full_blocks
= 新完整 physical blocks

new_block_hashes
= 对应 Prefix identities
```

中间还会经过：

```python
resolve_block_hashes(...)
```

因为 `hash_block_size` 与 physical `block_size` 不一定相同。

最后每个新 block 都形成：

```text
BlockHash
+
KVCacheBlock
↓
_insert_block_hash()
```

---

# 18. 真正注册 Prefix Cache 的最终位置

最终：

```python
self.cached_block_hash_to_block.insert(
    block_hash_with_group_id,
    block,
)
```

这一行可以视为 Prefix Cache registration 真正落地的位置。

也就是：

```text
Request.block_hashes
提供 Prefix identity

BlockPool / req_to_blocks
提供 physical KVCacheBlock

_insert_block_hash()
↓
Prefix Cache global lookup index
```

形成：

```text
(BlockHash, GroupId)
↓
KVCacheBlock
```

普通单 group 可简化成：

```text
H0 → block17
H1 → block31
```

---

# 19. Registration 调用链总结

```text
Scheduler.schedule()
        │
        ▼
KVCacheManager.allocate_slots(...)
        │
        ├── 分配/扩展 Request KV blocks
        ├── 计算 num_tokens_to_cache
        │
        ▼
Coordinator.cache_blocks(
    request,
    num_tokens_to_cache
)
        │
        ▼
SingleTypeKVCacheManager.cache_blocks(...)
        │
        ├── num_full_blocks
        ├── num_cached_blocks
        │
        ▼
BlockPool.cache_full_blocks(...)
        │
        ├── resolve_block_hashes(...)
        ├── new_block_hashes
        ├── new_full_blocks
        │
        ▼
BlockPool._insert_block_hash(...)
        │
        ▼
cached_block_hash_to_block.insert(...)
        │
        ▼
(BlockHash, group_id)
        →
physical KVCacheBlock
```

---

# 20. Request A 的 KV 实际写入 GPU 的链路

注册链只是控制面。

真正 KV data-plane 仍然是：

```text
SchedulerOutput
↓
Executor.execute_model()
↓
Worker.execute_model()
↓
GPUModelRunner.execute_model()
↓
prepare_inputs()
↓
prepare_attn()
↓
block_table / slot_mapping
↓
model forward
↓
QKV
↓
KV cache update
↓
physical GPU KV pages
```

因此 Request A 产生可复用 Prefix，可以拆成：

```text
1. Token identity
   ↓
   BlockHash

2. Control plane
   ↓
   physical block allocation
   ↓
   Hash → block metadata registration

3. Data plane
   ↓
   ModelRunner
   ↓
   slot_mapping
   ↓
   K/V 实际写入 physical KV pages
```

---

# 21. Request finish 后为什么 Prefix Cache 还能存在

Request A finish：

```text
Scheduler
↓
free Request ownership
↓
block.ref_cnt -= 1
```

可能出现：

```text
block17.ref_cnt = 0
```

但不能理解成：

```text
block17 KV 立刻清零
Hash 立即删除
```

Prefix Cache 必须区分：

```text
Active ownership
= 当前有多少活跃 Request 正在引用

Cached residency
= 这份 KV 是否仍保留，并具有可查找 Prefix identity
```

所以可以出现：

```text
block17
ref_cnt = 0
block_hash = H0

cached_block_hash_to_block
仍然：
H0 → block17
```

含义是：当前没人 active 使用，但未来相同 Prefix 仍可复用。

---

# 22. free 与 eviction 不是一回事

Request finish 后：

```text
free
```

主要表示 active ownership 释放。

未来 BlockPool 缺 block 时，才可能：

```text
选择 cached-but-unreferenced block
↓
evict Prefix Cache identity
↓
从 cached_block_hash_to_block 删除
↓
重新用于其他 Request
```

因此：

```text
free ≠ evict
```

更不是：

```text
free = GPU memory memset 0
```

---

# 23. Consumer side：Request B 到来

假设 Request A 已留下：

```text
H0 → block17
H1 → block31
H2 → block46
```

Request B 的前 48 token 与 A 完全一致。

Request B 创建时同样：

```text
Request.__init__()
↓
update_block_hashes()
↓
request_block_hasher()
```

得到：

```text
H0
H1
H2
H3
...
```

此时 B 自己还没 forward，所以：

```text
request.num_computed_tokens == 0
```

这会触发 Scheduler 的 Prefix Cache initial lookup。

---

# 24. Scheduler 在哪里开始查 Prefix Cache

waiting Request 路径：

```python
if request.num_computed_tokens == 0:
```

然后普通路径：

```python
(
    new_computed_blocks,
    num_new_local_computed_tokens,
    request.shared_prefix_boundary,
) = self.kv_cache_manager.get_computed_blocks(request)
```

调用链：

```text
Scheduler.schedule()
↓
waiting request handling
↓
if request.num_computed_tokens == 0
↓
KVCacheManager.get_computed_blocks(request)
```

这一步是在确定：

> 新 Request 的 initial computed frontier 是否可以通过已有 KV 直接向前移动。

---

# 25. `KVCacheManager.get_computed_blocks()` 做什么

关键代码：

```python
if not self.enable_caching or request.skip_reading_prefix_cache:
    return self.empty_kv_cache_blocks, 0, 0
```

没有 Prefix Cache：

```text
computed blocks = empty
computed tokens = 0
```

有 Prefix Cache：

```python
max_cache_hit_length = request.num_tokens - 1

computed_blocks, num_new_computed_tokens, num_uncached = (
    self.coordinator.find_longest_cache_hit(
        request.block_hashes,
        max_cache_hit_length
    )
)
```

因此：

```text
Scheduler
↓
KVCacheManager.get_computed_blocks()
↓
Coordinator.find_longest_cache_hit(...)
↓
具体 SingleType Manager
```

---

# 26. 为什么最多只 hit `num_tokens - 1`

源码：

```python
max_cache_hit_length = request.num_tokens - 1
```

原因：即使整个 Prompt KV 都存在，也仍然需要保留生成点附近的 forward 来获得 logits。

如果：

```text
Prompt = 32 tokens
32 tokens 全部 skip
```

则：

```text
没有 forward
↓
没有 logits
↓
无法 sample 下一个 token
```

当前 implementation 还会受 alignment / block boundary 约束，因此实际可能重算比一个 token 更多的尾部。

---

# 27. 为什么会出现多个 `find_longest_cache_hit()`

通过：

```bash
rg -n \
"class .*KVCacheManager|class FullAttention" \
vllm/v1/core/single_type_kv_cache_manager.py
```

得到：

```text
SingleTypeKVCacheManager
├── FullAttentionManager
├── SlidingWindowManager
├── ChunkedLocalAttentionManager
├── MambaManager
└── CrossAttentionManager
```

这些是不同 Attention/KV semantics 的实现，并不是一次请求会全部执行。

当前普通 Full Attention 只看：

```text
FullAttentionManager.find_longest_cache_hit()
```

---

# 28. FullAttentionManager：先适配 Hash 粒度

当前：

```python
class FullAttentionManager(SingleTypeKVCacheManager):
    supports_fine_grained_hash_lookup: ClassVar[bool] = True
```

然后：

```python
block_hashes = resolve_block_hashes(
    block_hashes,
    block_pool.hash_block_size,
    block_size,
    supports_fine_grained_hash_lookup=...,
    alignment_tokens=alignment_tokens,
)
```

说明：

```text
Request.block_hashes
= 通用 Prefix identity 序列

resolve_block_hashes(...)
= 按当前 physical block / alignment / attention type
  转换成 lookup 所需视图
```

这也是为什么不能简单把 `Request.block_hashes[i]` 永远理解成“physical block i 的 Hash”。

---

# 29. Phase 1：查最长连续 full-block Prefix

核心：

```python
for block_hash in itertools.islice(
    full_block_hashes,
    max_length // block_size
):
    cached_block = block_pool.get_cached_block(
        block_hash,
        kv_cache_group_ids
    )

    if not cached_block:
        break

    for computed, cached in zip(
        computed_blocks,
        cached_block
    ):
        computed.append(cached)
```

假设：

```text
H0 → block17
H1 → block31
H2 → block46
H3 → miss
```

执行：

```text
H0? → hit block17
H1? → hit block31
H2? → hit block46
H3? → miss → break
```

得到：

```text
computed_blocks = [17,31,46]
```

然后：

```python
hit_length = len(computed_blocks[0]) * block_size
```

若 `block_size=16`：

```text
hit_length = 3 × 16 = 48
```

---

# 30. 为什么一旦 miss 就可以 `break`

因为 Hash 是链式的：

```text
H0 = hash(ROOT, block0)
H1 = hash(H0, block1)
H2 = hash(H1, block2)
...
```

Prefix Cache 要找的是：

```text
从 token0 开始连续存在的最长 Prefix
```

而不是：

```text
任意 block 命中集合
```

因此 H1 miss 后，不存在“跳过 H1 再单独复用 H2”的语义。

---

# 31. `BlockPool.get_cached_block()` 才是真正查询 Prefix Cache Map 的地方

实际代码：

```python
def get_cached_block(
    self,
    block_hash: BlockHash,
    kv_cache_group_ids: list[int],
) -> list[KVCacheBlock] | None:

    cached_blocks = []

    for group_id in kv_cache_group_ids:
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, group_id
        )

        block = self.cached_block_hash_to_block.get_one_block(
            block_hash_with_group_id
        )

        if not block:
            return None

        cached_blocks.append(block)

    return cached_blocks
```

调用链：

```text
FullAttentionManager.find_longest_cache_hit()
↓
BlockPool.get_cached_block(block_hash, group_ids)
↓
make_block_hash_with_group_id(...)
↓
cached_block_hash_to_block.get_one_block(...)
↓
KVCacheBlock 或 None
```

所以真正 key/value：

```text
(BlockHash, KVCacheGroupId)
↓
KVCacheBlock
```

---

# 32. 为什么一个 group miss 就算整个 Prefix miss

生产配置可能存在多个 KV groups：

```text
same Prefix H0

group0 → block17
group1 → block41
```

若 group0 hit、group1 miss，则所需 KV state 不完整。

所以：

```python
if not block:
    return None
```

表达的是：所有要求的 groups 都存在才算有效 hit。

当前普通单 group Full Attention 可先忽略复杂 group 组合。

---

# 33. Fine-grained Phase 2：为什么可以命中 partial tail

当前：

```python
supports_fine_grained_hash_lookup = True
```

并且：

```python
fine_grained = (
    alignment_tokens < block_size
    and block_size % alignment_tokens == 0
)
```

例如：

```text
physical block_size = 16
alignment_tokens = 4
```

Phase 1 先查完整 block boundary：

```text
16
32
48
...
```

Phase 2 可以继续探测第一个非 full block 内部：

```text
32 + 12 = 44
32 + 8  = 40
32 + 4  = 36
```

源码从 high-to-low 探测最长可能命中，因此如果：

```text
44 miss
40 hit
```

最终：

```text
hit_length = 40
```

当前只需要知道这个能力存在，不继续展开所有 partial-tail registration 分支。

---

# 34. Prefix hit 返回的是两类信息

假设：

```text
H0 H1 H2 hit
block_size = 16
```

得到：

```text
new_computed_blocks = [block17, block31, block46]
num_new_local_computed_tokens = 48
```

两者用途不同：

```text
num_new_local_computed_tokens
= 控制面进度
= 有多少 token 不需要重新算

new_computed_blocks
= 物理地址
= 这些已有 KV 在哪些 physical blocks
```

所以一次 Prefix hit 同时返回：

```text
Progress + Address
```

---

# 35. Scheduler 如何把 Prefix hit 变成 computed frontier

普通 local Prefix 路径：

```python
(
    new_computed_blocks,
    num_new_local_computed_tokens,
    request.shared_prefix_boundary,
) = self.kv_cache_manager.get_computed_blocks(request)
```

如果有 external KV Connector：

```python
ext_tokens, load_kv_async = (
    self.connector.get_num_new_matched_tokens(
        request,
        num_new_local_computed_tokens
    )
)
```

然后：

```python
num_computed_tokens = (
    num_new_local_computed_tokens
    + num_external_computed_tokens
)
```

所以 computed frontier 可以来自：

```text
Local Prefix Cache
+
External KV Cache / KV Connector
```

---

# 36. `num_computed_tokens` 的正确语义

不要只理解为：

> 这个 Request 自己已经 forward 过多少 token。

更准确：

> **对于当前 Request 来说，有多少 token position 的计算结果/KV 已经可用，因此不需要本轮重新计算。**

来源可以是：

```text
1. Request 自己之前计算
2. Local Prefix Cache
3. External KV Connector
```

所以 computed frontier 描述的是：

```text
computation availability
```

而不是 computation ownership。

---

# 37. 真正 Skip Prefill 的最终位置

Scheduler：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
```

这就是控制面真正完成 Skip Prefix Prefill 的地方。

假设：

```text
Request B.num_tokens = 100
local Prefix hit = 48
external hit = 0
num_computed_tokens = 48
```

则：

```text
num_new_tokens = 100 - 48 = 52
```

于是：

```text
token[0:48)
根本不会进入本轮 scheduled workload

只需要：
token[48:100)
```

---

# 38. 为什么不存在一个 `skip_prefill()` 函数

错误理解：

```text
Scheduler
↓
把 100 token 全发给 ModelRunner
↓
ModelRunner 发现前48 cached
↓
skip 前48
```

真实逻辑：

```text
Scheduler
↓
先查 Prefix Cache
↓
发现 48 token 已 computed
↓
num_new_tokens = 100 - 48
↓
从控制面一开始只形成 52-token workload
↓
ModelRunner 根本不会收到“重新算前48”的任务
```

因此 Prefix Cache 的 Skip Compute 本质是：

> **Scheduler workload shrinking**

而不是 ModelRunner execution-time branch。

---

# 39. 这与 V1 Scheduler unified model 完全一致

之前的统一模型：

```text
known token frontier
-
computed frontier
=
pending work
```

没有 Prefix Cache：

```text
known = 100
computed = 0
pending = 100
```

有 Prefix hit：

```text
known = 100
computed = 48
pending = 52
```

如果：

```text
token_budget = 32
```

则：

```text
pending = 52
↓
本轮 schedule 32
↓
下一轮继续
```

所以 Prefix Cache 没有建立一套新的 Scheduler 模型。

它只是让一个新 Request 的 computed frontier 可以不从 0 开始。

---

# 40. 命中的 blocks 怎么继续进入数据面

Prefix Cache 不能只让 Scheduler 少算，它还必须把：

```text
block17
block31
block46
```

接入 Request B 当前 KV block view。

概念链：

```text
Prefix hit
↓
new_computed_blocks
↓
Request / KV manager 当前 block view
↓
SchedulerOutput
↓
GPUModelRunner persistent request state
↓
BlockTables
↓
prepare_inputs()
↓
prepare_attn()
↓
block_table
↓
Attention
```

假设：

```text
Request B

logical block0 → physical17   reused
logical block1 → physical31   reused
logical block2 → physical46   reused
logical block3 → physical62   newly allocated
logical block4 → physical71   newly allocated
```

那么：

```text
block_table = [17,31,46,62,71,...]
```

对于 ModelRunner 来说，不需要知道前三个 block 是 A 还是 B 计算的；它只需要知道当前 logical sequence 的历史 KV 在这些 physical blocks。

---

# 41. Prefix Cache 同时修改控制面和数据面

```text
                    Prefix Hit
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
     Progress Side               Address Side
          │                           │
48 cached tokens              blocks [17,31,46]
          │                           │
          ▼                           ▼
num_computed_tokens=48       Request block view
          │                           │
          ▼                           ▼
num_new_tokens reduced          block_table
          │                           │
          └─────────────┬─────────────┘
                        ▼
                  Model execution
                        │
               只算 miss tail
                        +
               读取 reused KV
```

---

# 42. 完整数值例子：Request A

设：

```text
hash_block_size = 16
physical block_size = 16
Request A prompt = 50 tokens
```

## A-1. Request 创建 / Hash

```text
token0~15  → H0
token16~31 → H1
token32~47 → H2
token48~49 不满 → 暂无 H3
```

所以：

```text
Request A.block_hashes = [H0,H1,H2]
```

此时只是 identity。

## A-2. Scheduler allocation

假设：

```text
logical0 → block17
logical1 → block31
logical2 → block46
logical3 → block62
```

## A-3. Prefix registration

标准 full-block registration：

```text
H0 → block17
H1 → block31
H2 → block46
```

## A-4. GPU execution

```text
SchedulerOutput
↓
ModelRunner
↓
slot_mapping
↓
KV write
↓
K/V 写入 physical blocks
```

## A-5. Finish

Request A finish 后：

```text
ref_cnt 可以降为 0
```

但 Prefix index 可继续保留：

```text
H0 →17
H1 →31
H2 →46
```

---

# 43. 完整数值例子：Request B 命中

Request B：

```text
100 tokens
```

且前 48 token 与 A 一样。

## B-1. Request 创建

产生：

```text
H0
H1
H2
H3
...
```

前三个与 A 相同。

## B-2. Scheduler admission

因为：

```text
request.num_computed_tokens == 0
```

调用：

```text
KVCacheManager.get_computed_blocks(B)
```

## B-3. longest Prefix lookup

```text
H0 → block17  hit
H1 → block31  hit
H2 → block46  hit
H3 → miss
↓
break
```

得到：

```text
new_computed_blocks = [17,31,46]
num_new_local_computed_tokens = 48
```

## B-4. Scheduler frontier

无 external cache：

```text
num_computed_tokens = 48
```

## B-5. Skip Prefix

```text
num_new_tokens
= request.num_tokens - num_computed_tokens
= 100 - 48
= 52
```

所以不调度：

```text
token[0:48)
```

只调度：

```text
token[48:100)
```

## B-6. KV blocks 接回 Request B

概念上：

```text
logical0 → block17   reused
logical1 → block31   reused
logical2 → block46   reused
logical3 → new block
...
```

## B-7. ModelRunner

```text
只 forward 后52 token
+
通过 block_table 读取 reused prefix KV
```

---

# 44. 完整源码调用链总览

## 44.1 Hash 生成链

```text
EngineCore.__init__()
↓
get_hash_fn_by_name(...)
↓
init_none_hash(...)
↓
get_request_block_hasher(...)
↓
self.request_block_hasher
↓
EngineCore._preprocess_engine_core_request(...)
↓
Request.from_engine_core_request(...)
↓
Request.__init__()
↓
Request.update_block_hashes()
↓
request_block_hasher(request)
↓
hash_block_tokens(...)
↓
Request.block_hashes
```

## 44.2 Prefix registration 链

```text
Scheduler.schedule()
↓
KVCacheManager.allocate_slots(...)
↓
num_tokens_to_cache
↓
Coordinator.cache_blocks(...)
↓
SingleTypeKVCacheManager.cache_blocks(...)
↓
num_full_blocks
↓
BlockPool.cache_full_blocks(...)
↓
resolve_block_hashes(...)
↓
BlockPool._insert_block_hash(...)
↓
cached_block_hash_to_block.insert(...)
```

结果：

```text
(BlockHash, GroupId)
→ physical KVCacheBlock
```

## 44.3 Prefix lookup 链

```text
Scheduler.schedule()
↓
waiting request
↓
if request.num_computed_tokens == 0
↓
KVCacheManager.get_computed_blocks(request)
↓
Coordinator.find_longest_cache_hit(...)
↓
FullAttentionManager.find_longest_cache_hit(...)
↓
resolve_block_hashes(...)
↓
Phase 1 full-block lookup
↓
BlockPool.get_cached_block(...)
↓
cached_block_hash_to_block.get_one_block(...)
↓
physical KVCacheBlock
↓
直到 first miss
↓
longest cached prefix
```

## 44.4 Skip Prefill 链

```text
get_computed_blocks()
↓
new_computed_blocks
+
num_new_local_computed_tokens
↓
可选 KV Connector
↓
num_external_computed_tokens
↓
num_computed_tokens
=
local + external
↓
num_new_tokens
=
request.num_tokens - num_computed_tokens
↓
Scheduler 只调度 miss tail
```

## 44.5 数据面 reuse 链

```text
new_computed_blocks
↓
Request / KV manager block view
↓
SchedulerOutput
↓
GPUModelRunner persistent request state
↓
BlockTables
↓
prepare_inputs()
↓
prepare_attn()
↓
block_table
↓
Attention
↓
读取 reused Prefix KV
+
计算当前 miss tail
```

---

# 45. 用“谁拥有状态”再梳理一次

## Request

维护：

```text
token sequence
Request.block_hashes
request.num_computed_tokens
request lifecycle
```

负责描述：

```text
“这个 Request 是什么”
“它有哪些 Prefix identities”
```

## Scheduler

维护：

```text
waiting / running
computed frontier
num_new_tokens
num_scheduled_tokens
admission
```

负责：

```text
“本轮真正还需要算多少”
```

## KVCacheManager / Coordinator / SingleType Manager

维护：

```text
Request → KV blocks
cache hit / allocation
block coverage
full block boundary
```

负责：

```text
“这个 Request 的 KV coverage 如何组织”
```

## BlockPool

维护：

```text
global physical block metadata
free queue
ref_cnt
cached_block_hash_to_block
cache eviction metadata
```

负责：

```text
“哪些 physical blocks 可用”
“哪个 Prefix identity 对应哪个 physical block”
```

## ModelRunner

维护/使用：

```text
persistent execution-side request state
BlockTables
InputBatch
slot_mapping
Attention metadata
```

负责：

```text
“GPU 这一轮具体如何读写 KV”
```

---

# 46. 为什么 Prefix Cache 的核心不是“缓存 token”

Token ids 本来 Request 就知道。

Prefix Cache 真正复用的是：

```text
这些 Prefix token 经模型计算后产生的 KV state
```

所以它减少的是：

```text
Transformer Prefill computation
```

而不是 tokenization。

---

# 47. 为什么 chained Hash 很适合 Prefix Cache

Prefix Cache 需要一个稳定、固定长度的 Prefix identity。

如果直接拿完整 token prefix tuple 当 key，Prefix 越长 key 越大。

链式 Hash：

```text
H0 = hash(ROOT, block0)
H1 = hash(H0, block1)
H2 = hash(H1, block2)
```

既保留：

```text
完整 Prefix dependency
```

又把每个 boundary identity 压缩成固定长度 fingerprint。

---

# 48. 为什么 Prefix Cache 与 Paged KV 天然契合

Paged KV 已经把 logical sequence 映射成离散 physical blocks。

所以 Prefix Cache 不需要复制一整段连续 KV，只需要让新 Request 的 block table 引用已有 physical blocks：

```text
Request A:
logical0 →17
logical1 →31
logical2 →46

Request B 命中 Prefix:
logical0 →17
logical1 →31
logical2 →46
```

这就是：

```text
Prefix reuse
+
Paged KV block indirection
```

结合的地方。

---

# 49. Prefix hit 与普通 KV allocation 的区别

Cache miss：

```text
Request B
↓
get_new_blocks()
↓
获得新的 physical blocks
↓
ModelRunner 重新计算并写 KV
```

Cache hit：

```text
Request B
↓
get_computed_blocks()
↓
find_longest_cache_hit()
↓
获得已有 physical blocks
↓
直接把这些 blocks 纳入 Request B 的 KV view
```

所以：

```text
get_new_blocks()
= 新 physical capacity

get_cached_block()
= 已有 computed KV reuse
```

---

# 50. 常见误区

## 误区 1

```text
“GPU block 写满以后才计算 Hash。”
```

错误。

正确：

```text
token sequence 每满 hash_block_size
就可以产生 BlockHash
```

Hash 可以早于 GPU execution。

## 误区 2

```text
“有 Request.block_hashes
就表示 Prefix Cache 已经存在。”
```

错误。

正确：

```text
block_hashes 只是 identity

还必须：
Hash → physical KVBlock
注册进 Prefix Cache index
```

## 误区 3

```text
“Prefix Cache 命中后
ModelRunner 自己跳过 Prefix。”
```

错误。

正确：

```text
Scheduler 通过：
num_new_tokens = num_tokens - num_computed_tokens

从一开始不再 schedule 命中的 Prefix
```

## 误区 4

```text
“Request finish 后 block free，Prefix Cache 就没了。”
```

错误。

正确：

```text
active ownership free
≠
cache residency eviction
```

## 误区 5

```text
“Prefix Cache 只需要知道 hit 多少 token。”
```

错误。

还必须知道：

```text
命中的 physical blocks 是哪些
```

否则 ModelRunner 无法读取历史 KV。

## 误区 6

```text
“BlockHash 就是当前 block token 的 hash。”
```

不准确。

正确：

```text
BlockHash 是 chained Prefix fingerprint
```

---

# 51. Prefix Cache 与当前 QCache 项目的关系

当前 QCache 主项目关注：

```text
KV physical representation
BF16 / INT4
recent / history
quantization / migration
low-bit read
```

Prefix Cache 关注：

```text
跨 Request 的 already-computed KV reuse
```

两者不是一个问题，但会在以下边界交叉：

```text
Block identity
physical pool
block_table
KV format
cache residency
reuse semantics
```

未来如果 physical KV block 有：

```text
BF16
INT4
scale metadata
```

Prefix hit 不只需要回答“Prefix 在哪个 block”，还可能需要保证该 block 当前 format 可被 execution/backend 正确消费。

这一点属于 QCache 后续设计，不在当前 Prefix 学习继续展开。

---

# 52. 当前学习范围内可以停止的地方

已经完成：

```text
Hash identity                      ✅
Hash incremental creation          ✅
chained Prefix semantics           ✅
physical block registration        ✅
Prefix cache global index          ✅
request finish / residency         ✅
longest prefix lookup              ✅
fine-grained lookup 基本认识        ✅
local/external computed frontier   ✅
Skip Prefill                       ✅
cached physical blocks reuse       ✅
与 block_table 数据面连接          ✅
```

当前不继续深入：

```text
Mamba Prefix retention
Sliding Window APC
Marconi-style shared boundary
EAGLE/MTP special drop
DCP/PCP Prefix hashing
Prefix Cache events
复杂 KV Connector
partial-tail registration 全部分支
cache eviction policy 优化
```

---

# 53. 最终总链

```text
====================== Request A ======================

EngineCore.__init__()
↓
get_request_block_hasher()
↓
Request.from_engine_core_request()
↓
Request.__init__()
↓
update_block_hashes()
↓
request_block_hasher()
↓
hash_block_tokens()
↓
Request.block_hashes
      │
      │ Prefix identity
      ▼
Scheduler.schedule()
↓
KVCacheManager.allocate_slots()
↓
physical KV blocks
↓
num_tokens_to_cache
↓
Coordinator.cache_blocks()
↓
SingleTypeKVCacheManager.cache_blocks()
↓
BlockPool.cache_full_blocks()
↓
_insert_block_hash()
↓
cached_block_hash_to_block

H0 → block17
H1 → block31
H2 → block46

↓
SchedulerOutput
↓
ModelRunner
↓
block_table / slot_mapping
↓
Attention
↓
K/V 真正写入 physical blocks

↓
Request A finish
↓
ref_cnt 可降 0
但 Prefix cache residency 可以继续保留


====================== Request B ======================

相同 Prefix
↓
Request 创建
↓
产生相同 chained hashes

H0 H1 H2 H3 ...

↓
Scheduler.schedule()
↓
request.num_computed_tokens == 0
↓
KVCacheManager.get_computed_blocks()
↓
Coordinator.find_longest_cache_hit()
↓
FullAttentionManager.find_longest_cache_hit()
↓
resolve_block_hashes()
↓
BlockPool.get_cached_block()

H0 → block17 hit
H1 → block31 hit
H2 → block46 hit
H3 → miss
↓
break

↓
new_computed_blocks = [17,31,46]
num_new_local_computed_tokens = 48

↓
可选 KV Connector
↓
num_computed_tokens = local + external

↓
num_new_tokens = request.num_tokens - num_computed_tokens

↓
Prefix tokens 不进入 scheduled workload

↓
SchedulerOutput
↓
ModelRunner
↓
block_table 包含 reused blocks

logical0 →17
logical1 →31
logical2 →46
logical3 →new block
...

↓
Attention
↓
读取 reused Prefix KV
+
只计算 miss tail
```

---

# 54. 一句话定义 Prefix Cache

> **vLLM Prefix Cache 通过 chained BlockHash 为 token Prefix 建立稳定 identity，再把该 identity 注册到承载 KV 的 physical KV blocks；新 Request 在首次 scheduling 时使用相同 BlockHash 查找最长连续 cached Prefix，同时得到“可跳过多少 token”和“这些历史 KV 位于哪些 physical blocks”，Scheduler 通过 `num_new_tokens = request.num_tokens - num_computed_tokens` 从控制面消除重复 Prefill，而 ModelRunner 则通过复用后的 block table 在数据面读取已有 Prefix KV。**

---

# 55. 当前源码追踪索引

以后回看 Prefix Cache，优先从这些入口重建主链。

```text
vllm/v1/engine/core.py
    EngineCore 初始化 request_block_hasher
    Request.from_engine_core_request(...)

vllm/v1/request.py
    Request.__init__()
    append_output_token_ids()
    update_block_hashes()

vllm/v1/core/kv_cache_utils.py
    get_request_block_hasher()
    request_block_hasher()
    hash_block_tokens()
    resolve_block_hashes()

vllm/v1/core/kv_cache_manager.py
    get_computed_blocks()
    allocate_slots()
    num_tokens_to_cache
    cache_blocks()

vllm/v1/core/single_type_kv_cache_manager.py
    SingleTypeKVCacheManager.cache_blocks()
    FullAttentionManager.find_longest_cache_hit()

vllm/v1/core/block_pool.py
    cache_full_blocks()
    _insert_block_hash()
    get_cached_block()
    cached_block_hash_to_block

vllm/v1/core/sched/scheduler.py
    waiting request Prefix lookup
    num_new_local_computed_tokens
    num_external_computed_tokens
    num_computed_tokens
    num_new_tokens = request.num_tokens - num_computed_tokens
```

这几处足以重建当前 Prefix Cache 主链。
