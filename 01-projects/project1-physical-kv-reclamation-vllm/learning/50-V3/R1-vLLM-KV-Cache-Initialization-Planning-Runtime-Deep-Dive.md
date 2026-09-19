# vLLM KV Cache 初始化、规划与 Runtime View 全链路深度梳理

**Project:** Project-1 — Physical KV Cache Reclamation for vLLM  
**Target:** vLLM v0.26 / qcrs fork  
**Scope:** KV Cache startup initialization path, memory planning path, storage-contract data model, worker-side raw allocation, backend-specific runtime view  
**Purpose:** 为后续 Ragged KV R1-A2 / A3 / BlockTable / Ownership 设计建立统一源码认知  
**Status:** Source-Trace / Architecture Learning Note

---

# 0. 这份文档解决什么问题

本轮不讨论 KV eviction / compaction / reclaim 算法本身。

目标是先把 vLLM 当前 Dense KV Cache 在启动阶段到底如何完成以下过程彻底贯通：

```text
用户配置
→ 模型 Attention 需求
→ per-layer KVCacheSpec
→ KV cache grouping
→ GPU memory planning
→ KVCacheConfig
→ KVCacheTensor allocation descriptor
→ raw GPU backing
→ backend-specific tensor view
→ Attention runtime KV cache
```

同时回答几个容易混淆的问题：

1. `CacheConfig` 和 `KVCacheConfig` 有什么区别？
2. `KVCacheSpec`、`KVCacheGroupSpec`、`KVCacheTensor` 分别描述什么？
3. `get_kv_cache_spec()` 到底是谁调用？
4. `get_kv_cache_groups()` 到底是谁调用？
5. `create_kv_cache_group_specs()` 为什么已经有 grouped names 还要 `merge()`？
6. 为什么每一种 Spec 可以有自己的 `merge()`？
7. `num_blocks` 在 Dense vLLM 中到底是什么？
8. `KVCacheTensor.block_stride=0` 到底是什么意思？
9. 为什么 worker 先申请 `torch.int8` raw tensor，再做 reshape？
10. `_reshape_kv_cache()`、`_reshape_attention_kv_cache()`、backend `get_kv_cache_shape()` / `get_kv_cache_stride_order()` 各自负责什么？
11. 为什么这条链对 Ragged KV A2/A3 极其关键？

---

# 1. 整体初始化调用链

KV Cache 的初始化并不是某一个函数一次性完成的。

真实调用横跨：

```text
EngineCore
→ Executor
→ Worker
→ GPUModelRunner
→ KV planner
→ Worker-side allocator
→ backend layout
```

完整主链可以压缩为：

```text
EngineCore.__init__
│
└─ EngineCore._initialize_kv_caches()
   │
   ├─ register_all_kvcache_specs(...)
   │
   ├─ model_executor.get_kv_cache_specs()
   │   │
   │   └─ Executor.collective_rpc("get_kv_cache_spec")
   │       │
   │       └─ Worker.get_kv_cache_spec()
   │           │
   │           └─ GPUModelRunner.get_kv_cache_spec()
   │               │
   │               └─ attn_utils.get_kv_cache_spec(vllm_config)
   │                   │
   │                   └─ 遍历每个 Attention layer
   │                       │
   │                       └─ attn_module.get_kv_cache_spec(vllm_config)
   │
   ├─ model_executor.determine_available_memory()
   │
   ├─ get_kv_cache_configs(...)
   │   │
   │   ├─ merge worker specs
   │   ├─ KVCacheSpecRegistry.check_kv_cache_spec_registry(...)
   │   ├─ get_kv_cache_groups(...)
   │   ├─ _project_kv_cache_groups_to_worker(...)
   │   ├─ memory admission check
   │   └─ get_kv_cache_config_from_groups(...)
   │
   ├─ generate_scheduler_kv_cache_config(...)
   │
   └─ model_executor.initialize_from_config(kv_cache_configs)
       │
       └─ Executor.collective_rpc("initialize_from_config")
           │
           └─ Worker.initialize_from_config()
               │
               └─ GPUModelRunner.initialize_kv_cache()
                   │
                   ├─ init_attn_backend()
                   ├─ BlockTables(...)
                   └─ init_kv_cache()
                       │
                       ├─ _allocate_kv_cache()
                       ├─ _reshape_kv_cache()
                       │   ├─ AttentionSpec
                       │   │   └─ _reshape_attention_kv_cache()
                       │   └─ MambaSpec
                       └─ bind_kv_cache()
```

这一整条属于 **Engine startup / KV cache initialization**。

它不是每个推理 step 都重复执行。

---

# 2. 两条主线：调用链与数据结构链

学习这段源码最容易混乱的原因，是函数调用和 dataclass 数据变化交织在一起。

建议同时记住两条主线。

## 2.1 调用链

```text
EngineCore
→ collect Specs
→ group Specs
→ plan memory
→ send KVCacheConfig to workers
→ allocate
→ reshape
→ bind
```

## 2.2 数据结构链

```text
VllmConfig
    │
    └─ CacheConfig
          ↓
dict[layer_name, KVCacheSpec]
          ↓
KVCacheGroupSpec[]
          ↓
KVCacheConfig
   ├─ num_blocks
   ├─ KVCacheGroupSpec[]
   └─ KVCacheTensor[]
          ↓
raw torch.Tensor
          ↓
runtime kv_caches[layer_name]
```

这两条链是一一对应的。

---

# 3. 第一层：VllmConfig

`VllmConfig` 是整个 vLLM Engine 的顶层配置对象。

概念上：

```python
VllmConfig(
    model_config=...,
    cache_config=...,
    parallel_config=...,
    scheduler_config=...,
    compilation_config=...,
    speculative_config=...,
    ...
)
```

KV Cache 相关配置只是其中一部分：

```python
vllm_config.cache_config
```

因此：

```text
VllmConfig
= Engine-level global configuration tree
```

---

# 4. 第二层：CacheConfig

位置：

```text
vllm/config/cache.py
```

`CacheConfig` 表达的是：

> 用户 / Engine 希望 KV Cache subsystem 怎么配置。

典型字段可以分为：

```text
storage granularity
- block_size

representation
- cache_dtype

memory budget
- gpu_memory_utilization
- kv_cache_memory_bytes

cache policy
- enable_prefix_caching
- prefix_caching_hash_algo

offload
- KV offloading related config

Mamba
- mamba_cache_mode
- mamba dtype

Project-1 Ragged
- page_group_size
```

例如：

```python
CacheConfig(
    block_size=16,
    cache_dtype="auto",
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True,
    page_group_size=2,
)
```

这里最关键的一点：

```text
CacheConfig
= 用户 / runtime policy

不是
= 某一个 layer 最终 storage geometry
```

它此时并不知道：

```text
当前 layer 的 Hkv
head_size
head_size_v
attention type
backend
```

---

# 5. CacheConfig 中为什么有 CacheDType / MambaDType / PrefixCachingHashAlgo

例如：

```python
CacheDType = Literal[
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    ...
]
```

这些 `Literal` 是配置 vocabulary。

它们用于表达：

```text
用户允许选择哪些 config 值
```

例如：

```text
cache_dtype="bfloat16"
cache_dtype="int8_per_token_head"
```

而 runtime 内部可能进一步转换成：

```text
KVQuantMode
backend-specific kernel mode
physical representation
```

同理：

```text
MambaDType
```

描述 Mamba state 的 dtype 配置。

```text
PrefixCachingHashAlgo
```

描述 prefix cache key 的 hash 算法。

```text
KVOffloadingBackend
```

描述 KV offload 后端选择。

所以把这些定义放在 `cache.py` 的原因是：

> 它们都是 KV/cache subsystem 的 Engine-level 配置语义，而不是某个 layer 的最终物理对象。

---

# 6. 为什么 Project-1 的 page_group_size 放在 CacheConfig

我们的新增字段：

```python
page_group_size: int | None
```

语义：

```text
Hp = 一个 physical Ragged page 内包含多少 KV heads
```

它应该属于 `CacheConfig`，而不是 `ModelConfig`。

原因：

```text
Hkv
= 模型 architecture 事实

Hp
= serving/storage policy
```

同一个模型：

```text
Hkv = 8
```

可以启动成：

```text
page_group_size=None → Dense
page_group_size=1    → per-head page
page_group_size=2    → 2 heads/page
page_group_size=4    → 4 heads/page
```

模型语义没变。

所以：

```text
Hkv → Model / Attention
Hp  → CacheConfig / storage policy
```

---

# 7. 从 CacheConfig 到 per-layer KVCacheSpec

模型加载后，EngineCore 会请求所有 worker 报告：

```text
“你当前 worker 上每个 layer 需要什么 KV storage？”
```

主链：

```text
EngineCore._initialize_kv_caches()
↓
model_executor.get_kv_cache_specs()
↓
Executor.collective_rpc("get_kv_cache_spec")
↓
Worker.get_kv_cache_spec()
↓
GPUModelRunner.get_kv_cache_spec()
↓
attn_utils.get_kv_cache_spec(vllm_config)
```

最终进入：

```text
vllm/v1/worker/gpu/attn_utils.py
```

的：

```python
def get_kv_cache_spec(
    vllm_config: VllmConfig,
) -> dict[str, KVCacheSpec]:
```

---

# 8. attn_utils.get_kv_cache_spec() 做什么

它首先找出 worker 上所有 Attention layers：

```python
attn_layers = get_layers_from_vllm_config(...)
```

然后：

```python
for layer_name, attn_module in attn_layers.items():
    if spec := attn_module.get_kv_cache_spec(vllm_config):
        ...
        kv_cache_spec[layer_name] = spec
```

也就是说：

```text
Attention runtime layer
        ↓
Attention.get_kv_cache_spec()
        ↓
per-layer storage contract
```

输出数据结构：

```python
dict[str, KVCacheSpec]
```

例如：

```python
{
    "L0": FullAttentionSpec(...),
    "L1": FullAttentionSpec(...),
    "L2": FullAttentionSpec(...),
}
```

这里仍然是：

```text
per-layer requirement discovery
```

还没有 allocator。

还没有 BlockPool。

还没有 page ID。

---

# 9. KVCacheSpec 数据模型

基础：

```python
@dataclass(frozen=True)
class KVCacheSpec:
    block_size: int
```

它是一个通用 cache resource contract。

它回答：

```text
一个 cache block/page 的通用语义是什么？
需要多少 bytes？
一个 request 最多需要多少？
这些 specs 是否可以 merge？
```

它不假设 cache 一定是传统 Attention K/V。

因此 vLLM 可以统一管理：

```text
Attention KV
Mamba state
Hybrid persistent state
```

---

# 10. AttentionSpec

```python
class AttentionSpec(KVCacheSpec):
```

在通用 Cache contract 上增加 Attention-specific geometry：

```text
num_kv_heads
head_size
dtype
kv_quant_mode
page_size_padded
indexes_kv_by_block_stride
```

因此：

```text
KVCacheSpec
= generic persistent cache contract

AttentionSpec
= Attention KV-specific cache contract
```

---

# 11. FullAttentionSpec

普通 Dense FullAttention 进一步形成：

```text
FullAttentionSpec
```

典型数据：

```text
block_size = 16
Hkv = 8
Dk = 128
Dv = 128
dtype = BF16
```

Dense page payload：

```text
P_dense
=
block_size
× Hkv
× (Dk + Dv)
× dtype_size
```

例：

```text
16 × 8 × 256 × 2
= 65536 B
= 64 KiB
```

一个 Dense physical page 可以理解为：

```text
one layer
× one token-block depth
× all local KV heads
```

---

# 12. RaggedAttentionSpec

Project-1 新增：

```python
RaggedAttentionSpec(AttentionSpec)
```

不是：

```text
RaggedAttentionSpec(FullAttentionSpec)
```

因为：

```text
Attention semantic type
≠
KV storage type
```

Ragged 仍然是 Attention KV，但它使用新的 physical storage geometry。

例如：

```text
Hkv = 8
Hp  = 2
G   = 4

block_size = 16
Dk = 128
Dv = 128
BF16
```

Ragged page：

```text
P_ragged
=
16 × 2 × 256 × 2
=
16 KiB
```

Identity 下：

```text
G × P_ragged
=
4 × 16 KiB
=
64 KiB
=
P_dense
```

所以 R1 identity 并不节省总显存。

它改变的是：

```text
physical allocation granularity
```

---

# 13. backend 信息为什么会重新写入 Spec

`attn_utils.get_kv_cache_spec()` 中，在拿到 layer Spec 后还会：

```python
backend = attn_module.get_attn_backend()
indexes = backend.indexes_kv_by_block_stride()

spec = replace(
    spec,
    indexes_kv_by_block_stride=indexes,
)
```

这里反映一个重要原则：

```text
最终 storage contract
=
model facts
+
CacheConfig
+
backend storage capability
```

因此 Spec 不是单纯模型 metadata。

它是 planner 最终可消费的 storage contract。

---

# 14. 从 per-layer Spec 到 KVCacheGroupSpec

下一阶段发生在：

```text
vllm/v1/core/kv_cache_utils.py
```

`get_kv_cache_configs()` 中：

```text
merge worker specs
↓
Registry check
↓
get_kv_cache_groups()
```

`get_kv_cache_groups()` 回答：

> 哪些 model layers 可以共享相同的 logical block-table / allocation lifecycle？

注意：

```text
sharing block-table lifecycle
≠
sharing actual KV payload
```

---

# 15. KVCacheGroupSpec

定义：

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    is_eagle_group: bool = False
```

例如：

```python
KVCacheGroupSpec(
    layer_names=["L0", "L1", "L2", "L3"],
    kv_cache_spec=FullAttentionSpec(...),
)
```

源码语义：

```text
这些 layers 在 KV cache manager 看来可以作为一个 manager layer
```

也就是：

```text
它们可以共享 block ID / block table 的 logical depth 语义
```

而不是：

```text
它们共享同一份 layer KV tensor
```

---

# 16. Dense FullAttention 为什么可以归成一个 Group

假设：

```text
Req A 已计算 32 tokens
block_size = 16
```

所有 FullAttention layers 都需要两层 token depth：

```text
L0 → 2 blocks
L1 → 2 blocks
L2 → 2 blocks
L3 → 2 blocks
```

depth 始终同步。

所以 scheduler 可以只记录：

```text
Req A block table = [5, 9]
```

然后每一层解释：

```text
L0 → L0[5], L0[9]
L1 → L1[5], L1[9]
L2 → L2[5], L2[9]
L3 → L3[5], L3[9]
```

注意：

```text
L0[5] ≠ L1[5]
```

它们不是同一块 GPU memory。

只是复用了同一个 logical ID `5`。

这就是 Dense vLLM 中极其重要的：

```text
synchronized ownership depth abstraction
```

---

# 17. grouped_layer_names 和 create_kv_cache_group_specs()

函数：

```python
def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
    grouped_layer_names: list[list[str]],
) -> list[KVCacheGroupSpec]:
```

这里：

```text
grouped_layer_names
```

已经是分组结果。

例如：

```python
[
    ["L0", "L1"],
    ["L2", "L3"],
]
```

所以该函数并不负责决定：

```text
“L0 应不应该和 L1 一组”
```

而是负责：

> 把已经决定好的 layer-name groups 正式 materialize 成 `KVCacheGroupSpec`。

---

# 18. create_kv_cache_group_specs() 逐步理解

核心：

```python
kv_cache_groups = []

for layer_names_one_group in grouped_layer_names:
    layer_specs = [
        kv_cache_spec[layer_name]
        for layer_name in layer_names_one_group
    ]

    merged_layer_spec = layer_specs[0].merge(layer_specs)

    kv_cache_groups.append(
        KVCacheGroupSpec(
            layer_names_one_group,
            merged_layer_spec,
        )
    )

return kv_cache_groups
```

假设：

```python
grouped_layer_names = [
    ["L0", "L1"],
    ["L2", "L3"],
]
```

第一轮：

```text
layer_names_one_group
=
["L0", "L1"]
```

然后：

```text
layer_specs
=
[spec_L0, spec_L1]
```

之后：

```text
merged_spec
=
spec_L0.merge([spec_L0, spec_L1])
```

最终：

```text
KVCacheGroupSpec(
    ["L0","L1"],
    merged_spec
)
```

所以：

```text
grouping
= 决定谁属于一组

merge
= 验证这一组在当前 Spec 类型下是否合法，
  并产生 group-level representative Spec
```

---

# 19. 为什么每一种 Spec 可以拥有自己的 merge()

默认 `KVCacheSpec.merge()` 可以使用：

```text
all specs equal
```

作为兼容条件。

但不同 cache type 对：

```text
“什么叫 compatible”
```

的定义不完全相同。

例如 FullAttention 可能需要处理：

```text
sliding_window
attention_chunk_size
non_causal
```

某些字段可能是：

```text
must equal
```

某些可能是：

```text
OR / ANY reduction
```

某些需要特殊 compatibility 规则。

因此：

```text
merge()
```

应该由具体 storage contract 类型自己定义。

我们的 `RaggedAttentionSpec.merge()` 也必须额外验证：

```text
Hp
head_size_v
```

特别是：

```text
L0 Hp=2
L1 Hp=4
```

不能被当成相同 physical page geometry。

所以：

```text
Spec.merge()
=
type-specific compatibility + group-level reduction
```

---

# 20. 从 Groups 到 KVCacheConfig

`get_kv_cache_configs()` 完成全局 grouping 后，还会考虑 PP worker projection：

```text
global groups
↓
_project_kv_cache_groups_to_worker(...)
↓
worker-local groups
```

然后每个 worker 调：

```python
get_kv_cache_config_from_groups(
    vllm_config,
    projected_groups,
    available_memory_one_worker,
)
```

这一步开始真正做：

```text
memory planning
```

---

# 21. CacheConfig 和 KVCacheConfig 的关键区别

这两个名字很像，但语义完全不同。

| 对象 | CacheConfig | KVCacheConfig |
|---|---|---|
| 阶段 | Engine config | memory planning result |
| 来源 | 用户 / config | planner 计算 |
| 作用 | 描述希望怎么配置 | 描述最终怎么分配 |
| 是否知道 GPU 可用显存 | 否 / 间接 | 是 |
| 是否包含 num_blocks | 配置/最终回填 | 是，核心规划结果 |
| 是否包含 layer groups | 否 | 是 |
| 是否包含 tensor allocation descriptors | 否 | 是 |

一句话：

```text
CacheConfig
= policy / intent

KVCacheConfig
= concrete memory plan
```

---

# 22. KVCacheConfig

定义：

```python
@dataclass
class KVCacheConfig:
    num_blocks: int
    kv_cache_tensors: list[KVCacheTensor]
    kv_cache_groups: list[KVCacheGroupSpec]
```

它同时包含：

```text
scheduler/manager view
+
worker memory allocation plan
```

其中：

```text
num_blocks
```

表示 scheduler / allocator 的 logical block namespace 大小。

```text
kv_cache_groups
```

告诉 scheduler / manager 有哪些 group。

```text
kv_cache_tensors
```

告诉 worker 真实 raw memory 应该怎样申请。

---

# 23. Dense num_blocks 的真正语义

普通 general case：

```python
num_blocks = (
    available_memory
    // page_size
    // num_layers
)
```

例如：

```text
available = 8 GiB
L = 4
P_dense = 64 KiB
```

得到：

```text
num_blocks = 32768
```

这里的一个 Dense block ID：

```text
block_id = 100
```

并不是一张唯一 physical page。

它隐含：

```text
L0 page[100]
L1 page[100]
L2 page[100]
L3 page[100]
```

所以一个 allocator block ID 的真实显存成本：

```text
L × P_dense
```

这就是为什么需要：

```text
available
/
page_size
/
num_layers
```

---

# 24. Dense block_id 的本质

Dense vLLM 中的 scheduler block ID 更接近：

> 跨 grouped layers 的 logical allocation depth ID。

而不是：

> global unique physical page ID。

例如：

```text
Req A block table = [3,8]
```

如果有四层：

```text
L0 → page3,page8
L1 → page3,page8
L2 → page3,page8
L3 → page3,page8
```

实际 physical pages：

```text
2 depth IDs × 4 layers = 8 pages
```

scheduler 只需要保存：

```text
[3,8]
```

这就是 Dense allocator abstraction 的核心。

---

# 25. KVCacheTensor

定义：

```python
@dataclass
class KVCacheTensor:
    size: int
    shared_by: list[str]
    offset: int = 0
    block_stride: int = 0
```

注意：

```text
KVCacheTensor
不是 torch.Tensor
```

它是：

> Worker-side physical allocation descriptor。

例如：

```python
KVCacheTensor(
    size=2 * GiB,
    shared_by=["L0"],
)
```

意思：

```text
worker 后面要申请一块 2GiB raw memory，
并让 L0 指向它
```

---

# 26. shared_by 的真正含义

`shared_by` 表示：

```text
哪些 layer_name alias 同一个 raw tensor object
```

不是：

```text
这些 layers 自动拥有同样的 KV payload
```

也不是：

```text
这些 layers 自动共享相同 block IDs
```

它只是：

```text
raw backing aliasing
```

之后具体如何寻址还取决于：

```text
offset
block_stride
block table
backend layout
```

---

# 27. 普通 Dense 的 KVCacheTensor 结果

例如：

```text
L=4
available=8GiB
num_blocks=32768
page_size=64KiB
```

每层：

```text
32768 × 64KiB
= 2GiB
```

于是：

```python
KVCacheConfig(
    num_blocks=32768,

    kv_cache_groups=[
        KVCacheGroupSpec(
            layer_names=["L0","L1","L2","L3"],
            kv_cache_spec=FullAttentionSpec(...),
        )
    ],

    kv_cache_tensors=[
        KVCacheTensor(
            size=2GiB,
            shared_by=["L0"],
        ),
        KVCacheTensor(
            size=2GiB,
            shared_by=["L1"],
        ),
        KVCacheTensor(
            size=2GiB,
            shared_by=["L2"],
        ),
        KVCacheTensor(
            size=2GiB,
            shared_by=["L3"],
        ),
    ]
)
```

这里可以看到：

```text
management:
1 KVCacheGroupSpec

physical allocation:
4 KVCacheTensor descriptors
```

因为：

```text
share block-table lifecycle
≠
share physical backing
```

---

# 28. KVCacheTensor.block_stride=0 的含义

非常重要：

```text
KVCacheTensor.block_stride
不是 torch.Tensor.stride()
```

它是 planner-level 的 packed layout metadata。

```text
block_stride = 0
```

表示：

```text
not packed
standalone contiguous raw allocation
```

因此：

```python
if kv_cache_tensor.block_stride > 0:
    ...
else:
    tensor = torch.zeros(...)
```

普通 Dense 通常：

```text
block_stride = 0
```

所以每个 descriptor 独立申请 raw tensor。

---

# 29. block_stride > 0 的 packed layout

假设：

```text
Layer A page = 64KiB
Layer B page = 128KiB
```

physical block packing：

```text
block0:
[A0 64KiB][B0 128KiB]

block1:
[A1 64KiB][B1 128KiB]
```

一个 packed block：

```text
192 KiB
```

于是：

```text
block_stride = 192KiB
```

A：

```text
offset = 0
```

B：

```text
offset = 64KiB
```

所以：

```text
offset
= 当前 layer 在 packed block 内的起始位置

block_stride
= block i → block i+1 的 byte distance
```

---

# 30. 从 KVCacheTensor 到真实 GPU Memory

EngineCore 最终调用：

```text
model_executor.initialize_from_config(kv_cache_configs)
```

Executor RPC：

```text
Worker.initialize_from_config()
```

Worker：

```text
GPUModelRunner.initialize_kv_cache()
```

随后：

```text
init_kv_cache()
```

其中：

```python
kv_cache_raw_tensors = _allocate_kv_cache(...)
```

普通非-packed：

```python
tensor = torch.zeros(
    kv_cache_tensor.size,
    dtype=torch.int8,
    device=device,
)
```

此时第一次真正申请 GPU KV memory。

---

# 31. 为什么 raw backing 用 torch.int8

这里的 `int8` 不是 KV quantization。

只是：

```text
1 element = 1 byte
```

因此：

```text
tensor.numel()
=
allocated bytes
```

它类似：

```c
void* ptr = malloc(num_bytes);
```

真正的 BF16 / FP16 / FP8 interpretation 留到后面。

因此：

```text
raw int8 tensor
=
byte-addressable backing storage
```

---

# 32. _allocate_kv_cache() 的输出

最终建立：

```python
dict[layer_name, raw_tensor]
```

普通 Dense：

```text
L0 → raw_tensor0
L1 → raw_tensor1
L2 → raw_tensor2
L3 → raw_tensor3
```

此时每个 raw tensor 仍然只是：

```text
1-D bytes
```

Attention backend 还不能直接消费。

---

# 33. 为什么还需要 _reshape_kv_cache()

因为 vLLM cache subsystem 不只管理 Attention KV。

`_reshape_kv_cache()` 是一个 generic runtime cache interpreter：

```text
raw memory
+
KVCacheSpec
+
backend
→ runtime cache object
```

它根据 Spec 类型 dispatch：

```python
if isinstance(kv_cache_spec, AttentionSpec):
    ...
elif isinstance(kv_cache_spec, MambaSpec):
    ...
```

所以：

```text
KV Cache subsystem
```

在系统层面实际表示：

```text
per-request persistent model state
```

不只传统 K/V。

---

# 34. Attention path 的职责

对于 `AttentionSpec`：

```text
raw bytes
↓
calculate num_blocks
↓
backend.get_kv_cache_shape(...)
↓
backend.get_kv_cache_stride_order()
↓
_reshape_attention_kv_cache(...)
```

这里必须区分：

```text
shape
```

和：

```text
physical stride order
```

---

# 35. backend.get_kv_cache_shape()

backend 决定逻辑 KV shape。

例如当前 FA 路径可抽象为：

```text
[B, H, N, 2D]
```

其中：

```text
B = num blocks
H = Hkv
N = block_size
2D = K + V packed content dimension
```

例如：

```text
[32768, 8, 16, 256]
```

这是 logical dimension structure。

---

# 36. backend.get_kv_cache_stride_order()

backend 还决定 physical memory order。

例如逻辑都表达：

```text
B,H,N,D
```

但物理可以是：

```text
HND:
B,H,N,D

NHD:
B,N,H,D
```

所以：

```text
shape
= logical dimensions

stride order
= physical dimension order
```

这两个不能混。

---

# 37. _reshape_kv_cache() 与 backend 的职责边界

正确分工：

```text
Attention Backend
→ 决定 logical shape
→ 决定 physical stride order

_reshape_kv_cache()
→ generic cache-type dispatcher
→ 获取 backend layout information

_reshape_attention_kv_cache()
→ 真正把 raw backing 构造成对应 Attention typed view
```

所以：

```text
NHD/HND 决策
```

不是 `_reshape_kv_cache()` 自己决定。

而是 backend 给出。

---

# 38. _reshape_attention_kv_cache()

它处理三种主要 physical layout 情况。

## 38.1 packed backing

如果：

```text
packing != None
```

通过：

```text
offset
block_stride
```

从大 backing 中抽取属于当前 layer 的 page region。

## 38.2 padded page

如果：

```text
page_size_padded != None
```

不能简单 contiguous reshape。

需要：

```python
torch.as_strided(...)
```

显式告诉 PyTorch：

```text
physical page i
→ page i+1
应该跨多少 bytes/elements
```

## 38.3 ordinary contiguous

最简单：

```python
kv_raw_tensor
    .view(dtype)
    .view(permuted_kv_cache_shape)
```

先把 raw bytes reinterpret 成：

```text
BF16 / FP16 / other dtype
```

然后按 backend physical order view。

最后：

```python
permute(*inv_order)
```

恢复统一 logical shape ordering。

---

# 39. 为什么最终 shape 相同但 stride 可以不同

例如 logical shape：

```text
[B,H,N,D]
```

backend 希望 NHD：

```text
physical = [B,N,H,D]
```

流程：

```text
raw bytes
↓
view [B,N,H,D] contiguous
↓
permute back
↓
logical [B,H,N,D]
```

最后：

```text
shape = [B,H,N,D]
```

但 tensor stride 仍然表达 NHD physical layout。

所以：

```text
logical API shape
```

和：

```text
physical memory layout
```

实现了解耦。

---

# 40. Mamba path

如果：

```python
isinstance(kv_cache_spec, MambaSpec)
```

则 raw backing 会被解释成：

```text
conv state
SSM state
...
```

可能最终返回：

```python
list[torch.Tensor]
```

因此：

```text
kv_caches[layer_name]
```

并不保证永远是一个普通 Attention tensor。

这也是 `KVCacheSpec` 必须高于 `AttentionSpec` 的原因。

---

# 41. bind_kv_cache()

经过 reshape 后：

```python
kv_caches = {
    "L0": attention_tensor0,
    "L1": attention_tensor1,
    ...
}
```

最后：

```python
bind_kv_cache(
    kv_caches,
    forward_context,
    runner_kv_caches,
    ...
)
```

把实际 runtime cache tensor references 接到模型执行路径。

因此初始化真正完成：

```text
Config
→ Spec
→ Group
→ Memory Plan
→ Raw Allocation
→ Backend View
→ Model Runtime Binding
```

---

# 42. 完整数据结构变化图

```text
┌──────────────────────────────────────┐
│ VllmConfig                           │
│                                      │
│ cache_config: CacheConfig            │
└──────────────────┬───────────────────┘
                   │
                   │ user/global policy
                   ▼
┌──────────────────────────────────────┐
│ CacheConfig                          │
│                                      │
│ block_size = 16                      │
│ cache_dtype = auto                   │
│ page_group_size = 2                  │
│ gpu_memory_utilization = ...         │
└──────────────────┬───────────────────┘
                   │
                   │ + model facts
                   │ + backend facts
                   ▼
┌──────────────────────────────────────┐
│ dict[layer_name, KVCacheSpec]        │
│                                      │
│ L0 → FullAttentionSpec(...)          │
│ L1 → FullAttentionSpec(...)          │
│ L2 → FullAttentionSpec(...)          │
└──────────────────┬───────────────────┘
                   │
                   │ get_kv_cache_groups()
                   ▼
┌──────────────────────────────────────┐
│ KVCacheGroupSpec                     │
│                                      │
│ layer_names=[L0,L1,L2]               │
│ kv_cache_spec=merged Spec            │
└──────────────────┬───────────────────┘
                   │
                   │ + available GPU memory
                   ▼
┌──────────────────────────────────────┐
│ KVCacheConfig                        │
│                                      │
│ num_blocks = N                       │
│ kv_cache_groups = [...]              │
│ kv_cache_tensors = [...]             │
└──────────────────┬───────────────────┘
                   │
                   │ physical allocation plan
                   ▼
┌──────────────────────────────────────┐
│ KVCacheTensor descriptors            │
│                                      │
│ size                                 │
│ shared_by                            │
│ offset                               │
│ block_stride                         │
└──────────────────┬───────────────────┘
                   │
                   │ _allocate_kv_cache()
                   ▼
┌──────────────────────────────────────┐
│ Raw torch.Tensor                     │
│                                      │
│ dtype=int8                           │
│ raw GPU bytes                        │
└──────────────────┬───────────────────┘
                   │
                   │ _reshape_kv_cache()
                   ▼
┌──────────────────────────────────────┐
│ Runtime Cache View                   │
│                                      │
│ Attention → [B,H,N,2D] logical view  │
│ Mamba     → state tensors            │
└──────────────────┬───────────────────┘
                   │
                   │ bind_kv_cache()
                   ▼
┌──────────────────────────────────────┐
│ Model Runtime                        │
│ Attention / backend reads & writes   │
└──────────────────────────────────────┘
```

---

# 43. 每个 dataclass 一句话钉死

| Object | Responsibility |
|---|---|
| `VllmConfig` | 整个 Engine 的总配置 |
| `CacheConfig` | 用户希望 KV/cache subsystem 怎么配置 |
| `KVCacheSpec` | 某个 layer 的通用 cache storage contract |
| `AttentionSpec` | Attention KV-specific storage contract |
| `FullAttentionSpec` | Dense/Full-Attention storage contract |
| `RaggedAttentionSpec` | Hp-granular Attention physical page contract |
| `KVCacheGroupSpec` | 哪些 layers 共用 allocator/block-table lifecycle |
| `KVCacheConfig` | 某 worker 最终 KV memory plan |
| `KVCacheTensor` | raw allocation 应如何创建 / alias / packed |
| raw `torch.Tensor` | GPU 上真正申请的 bytes |
| `kv_caches[layer]` | runtime 最终读写的 cache view |

---

# 44. Dense 示例完整贯通

假设：

```text
L = 4
Hkv = 8
block_size = 16
Dk = Dv = 128
BF16 = 2 bytes
available KV memory = 8 GiB
```

Dense page：

```text
P_dense
=
16 × 8 × (128+128) × 2
=
64 KiB
```

Planner：

```text
num_blocks
=
8 GiB
/
64 KiB
/
4
=
32768
```

生成：

```text
1 × KVCacheGroupSpec
layers=[L0,L1,L2,L3]
```

以及：

```text
4 × KVCacheTensor
each:
32768 × 64KiB
= 2GiB
```

Worker：

```text
L0 → raw 2GiB
L1 → raw 2GiB
L2 → raw 2GiB
L3 → raw 2GiB
```

reshape 后：

```text
L0 → BF16 [32768,8,16,256]
L1 → BF16 [32768,8,16,256]
L2 → BF16 [32768,8,16,256]
L3 → BF16 [32768,8,16,256]
```

如果 Scheduler 给：

```text
Req A block table = [5,9,12]
```

那么：

```text
Layer0 reads:
L0[5], L0[9], L0[12]

Layer1 reads:
L1[5], L1[9], L1[12]

...
```

因此：

```text
same block ID
+
different layer-local backing
```

正是 Dense KV architecture 的核心。

---

# 45. 对 Project-1 Ragged 的直接启示

当前 A1 只完成：

```text
CacheConfig.page_group_size
+
RaggedAttentionSpec
```

也就是：

```text
configuration vocabulary
+
storage contract vocabulary
```

尚未改变：

```text
KVCacheGroupSpec
KVCacheConfig
KVCacheTensor
raw backing
runtime view
BlockPool ownership
BlockTables
```

---

# 46. 为什么 Ragged 不能只改 page_size

假设：

```text
Hkv = 8
Hp = 2
G = 4
```

Dense page：

```text
64 KiB
```

Ragged page：

```text
16 KiB
```

如果只把：

```text
page_size
64KiB → 16KiB
```

但仍然：

```text
num_blocks
=
available
/
page_size
/
num_layers
```

则系统认为：

```text
每 layer 每 depth 只需要 1 个 16KiB page
```

但 Ragged 一个 layer 的一个 token depth 实际需要：

```text
G = 4 physical pages
```

所以 capacity accounting 已经错误。

---

# 47. 为什么 Ragged 也不能只改 global backing

即使 A2 改成：

```text
num_pages = available / 16KiB
```

并创建 global raw backing，

如果仍然走普通 FA：

```python
backend.get_kv_cache_shape(
    ...,
    num_kv_heads=Hkv,
    ...
)
```

backend 仍会尝试解释为：

```text
[Bphys,Hkv,N,2D]
```

但 Ragged physical page 实际应该是：

```text
[Bphys,Hp,N,2D]
```

因此：

```text
A2:
physical pool / namespace

A3:
physical page layout / virtual FA view
```

必须分开处理。

---

# 48. 为什么不能把 num_kv_heads 偷改成 Hp

因为：

```text
Hkv
= model semantic KV head count

Hp
= storage page width
```

如果：

```text
RaggedAttentionSpec.num_kv_heads = Hp
```

虽然短期可能让 backend reshape 出 `[B,Hp,N,2D]`，

但会破坏：

```text
GQA semantics
K/V input head count
head mapping
attention metadata
backend assumptions
```

所以正确设计必须保持：

```text
num_kv_heads = Hkv
page_group_size = Hp
```

后续通过：

```text
ragged-specific allocation/layout adapter
```

解决 physical geometry。

---

# 49. 对 R1-A2 / A3 的边界判断

从这条初始化链可以得到一个更准确的设计原则。

## R1-A2

负责：

```text
physical page capacity
global physical page namespace
raw backing planning
shared physical substrate
```

核心问题：

```text
page_id → raw memory
```

## R1-A3

负责：

```text
physical page layout
Hp-sized page interpretation
stride
zero-copy virtual member blocks
backend-facing view
```

核心问题：

```text
(page_id, column, token, channel)
→ exact address
```

但是：

```text
KVCacheConfig.num_blocks
```

现在同时连接：

```text
memory capacity
+
scheduler BlockPool ID namespace
```

因此 A2 设计不能只考虑 `torch.zeros()`。

必须同时审计：

```text
KVCacheGroupSpec
KVCacheManager
BlockPool
BlockTables
```

也就是说：

> 真正困难的不是创建一个 global tensor，而是重新定义 block/page ID 的 ownership semantics。

---

# 50. 当前阶段已经建立的核心认知

经过这轮源码梳理，可以冻结以下认知。

### 1. CacheConfig 和 KVCacheConfig 完全不是一层

```text
CacheConfig
= policy

KVCacheConfig
= memory plan
```

### 2. KVCacheSpec 不是 KV tensor

```text
Spec
= storage contract
```

### 3. KVCacheGroupSpec 不是 physical memory grouping

它主要表达：

```text
manager / block-table lifecycle grouping
```

### 4. KVCacheTensor 不是 torch.Tensor

```text
KVCacheTensor
= physical allocation descriptor
```

### 5. Dense block_id 不是 global physical page ID

它更接近：

```text
cross-layer logical allocation depth ID
```

### 6. raw int8 不表示 INT8 KV

它只是：

```text
byte-addressable backing storage
```

### 7. NHD/HND 是 backend 决策

```text
backend
→ shape / physical order

reshape helper
→ materialize typed view
```

### 8. vLLM KV subsystem 不只管理 Attention KV

还会管理：

```text
Mamba recurrent state
hybrid state
```

### 9. Project-1 Ragged 的核心冲突是 ownership abstraction

不是单纯：

```text
page shape 不一样
```

而是：

```text
Dense:
one logical block ID
implicitly expands across grouped layers

Ragged:
one page ID
should eventually identify one true physical page
```

---

# 51. 后续源码学习入口

本轮初始化链已经可以收尾。

下一阶段如果继续深入，应该进入 runtime ownership 链：

```text
KVCacheConfig
↓
Scheduler
↓
KVCacheManager
↓
SingleTypeKVCacheManager
↓
FullAttentionManager
↓
BlockPool
↓
KVCacheBlock
↓
req_to_blocks
↓
SchedulerOutput.block_ids
↓
Worker BlockTables
↓
Attention metadata
```

这一条要回答：

> `KVCacheConfig.num_blocks=N` 之后，Req A 到底如何获得 `[5,9,12]`，这些 block IDs 又如何最终索引各 layer-local KV tensor？

它会直接决定 Project-1 A2/B 该怎样重构 physical page ownership。

---

# 52. Source Map

本轮主要源码入口：

```text
vllm/v1/engine/core.py
- EngineCore.__init__
- EngineCore._initialize_kv_caches

vllm/v1/executor/abstract.py
- Executor.get_kv_cache_specs
- Executor.initialize_from_config

vllm/v1/worker/gpu_worker.py
- Worker.get_kv_cache_spec
- Worker.determine_available_memory
- Worker.initialize_from_config

vllm/v1/worker/gpu/model_runner.py
- GPUModelRunner.get_kv_cache_spec
- GPUModelRunner.initialize_kv_cache

vllm/v1/worker/gpu/attn_utils.py
- get_kv_cache_spec
- init_attn_backend
- init_kv_cache
- _allocate_kv_cache
- _reshape_kv_cache
- _reshape_attention_kv_cache

vllm/v1/core/kv_cache_utils.py
- get_kv_cache_configs
- get_kv_cache_groups
- create_kv_cache_group_specs
- get_kv_cache_config_from_groups
- get_num_blocks
- _project_kv_cache_groups_to_worker
- generate_scheduler_kv_cache_config

vllm/v1/kv_cache_interface.py
- KVCacheSpec
- AttentionSpec
- FullAttentionSpec
- RaggedAttentionSpec
- MambaSpec
- KVCacheGroupSpec
- KVCacheTensor
- KVCacheConfig
```

---

# 53. 最终心智模型

如果只保留一张图，保留下面这张：

```text
                    USER / ENGINE POLICY
                           │
                           ▼
                     CacheConfig
                           │
             + model facts│+ backend facts
                           ▼
                per-layer KVCacheSpec
                           │
                           ▼
               KVCacheGroupSpec
          logical lifecycle / grouping
                           │
                + available memory
                           ▼
                    KVCacheConfig
          ┌────────────────┴────────────────┐
          │                                 │
     num_blocks                       KVCacheTensor[]
 logical ID capacity                allocation descriptors
          │                                 │
          │                                 ▼
          │                         raw GPU backing
          │                                 │
          │                                 ▼
          │                      backend-specific reshape
          │                                 │
          │                                 ▼
          │                         runtime KV views
          │
          └──────────── future runtime ownership ────────────┐
                                                             │
                                                             ▼
                                                    Scheduler / BlockPool
                                                             │
                                                             ▼
                                                        Request block IDs
                                                             │
                                                             ▼
                                                       Worker BlockTables
                                                             │
                                                             ▼
                                                        Attention read/write
```

对 Project-1 来说，真正要打破的是中间这条 Dense invariant：

```text
one scheduler block ID
=
same logical depth across grouped layers
```

未来 Ragged 要逐步变成：

```text
one physical page ID
=
one actual reclaimable physical page
```

这正是从 R1-A1 进入 A2/B 的架构分界线。
