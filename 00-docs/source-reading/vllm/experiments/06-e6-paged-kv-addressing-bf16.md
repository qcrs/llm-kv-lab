# E6：Paged KV 地址映射 + Default Attention Backend + BF16 KV 数据面动态实验

> 实验状态：**PASS / Frozen**  
> vLLM：**v0.26.0**  
> commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> Model Runner：**V2**  
> GPU：**NVIDIA A100 80GB PCIe**  
> 模型：`/data/models/Qwen3-0.6B`  
> 实验日志：`04-experiments/vllm-bridge/raw/e6-paged-kv-attention/e6-default-bf16.log`

---

# 1. 实验背景

E1～E5 已经把 vLLM V1 Engine 的大部分 **控制面** 动态链路验证清楚：

```text
EngineCore
↓
Scheduler
↓
KVCacheManager
↓
SingleTypeKVCacheManager / Coordinator
↓
BlockPool
↓
physical block ownership
```

我们已经知道 Scheduler 会为 Request 分配类似：

```text
block_ids = [1, 2, 3]
```

也知道 Prefix Cache 命中时会把历史 cached block 重新 `touch/adopt` 到当前 Request。

但到 E5 为止仍然存在一个非常重要的数据面断层：

```text
Scheduler 说：
Request R 拥有 physical blocks [1,2,3]

              ↓

这些 block ID 怎么变成 GPU KV Cache 中具体 token 的写入地址？

              ↓

当前 token 的 K/V 到底是谁写进去？

              ↓

Attention 又通过什么 metadata 把历史 K/V 读回来？
```

因此本实验将原计划的 **Phase E：Paged KV 地址验证** 与 **E6：Default Attention Backend + BF16 KV** 合并。

最终目标是完整打通：

```text
Scheduler block ownership
↓
ModelRunner block_table
↓
position → slot_mapping
↓
Attention KV write
↓
BF16 KV Cache in HBM
↓
Attention paged historical KV read
```

这一步是从 vLLM **KV 控制面** 正式进入 **GPU KV 数据面** 的桥。

---

# 2. 本实验要回答的核心问题

E6 合并 Phase E 后，一共回答以下问题。

## 2.1 Scheduler 分配的 physical block 到底如何变成 token-level KV 地址？

我们希望验证：

```text
position
↓
logical block index
↓
block_table[logical block]
↓
physical block
↓
offset in block
↓
physical slot
```

并直接比较：

```text
expected_slot == actual_slot_mapping
```

而不是只凭源码推断。

---

## 2.2 `block_table` 和 `slot_mapping` 分别是什么？

这是此前很容易混淆的一点。

本实验希望建立：

```text
block_table
= Request / logical-block 粒度的物理页表

slot_mapping
= 当前 step 中每一个 scheduled token 的精确 KV write slot
```

即：

```text
block_table：粗粒度
slot_mapping：细粒度
```

---

## 2.3 默认 Attention Backend 实际选中了谁？

E6 不显式指定 backend。

需要让当前 A100 + vLLM v0.26 的 selector 自己完成 backend selection，然后用 runtime 日志确认最终 implementation。

不能写成：

```text
A100 → 我猜是 FlashAttention
```

而必须是：

```text
runtime selector → FLASH_ATTN
runtime impl     → FlashAttentionImpl
```

---

## 2.4 `kv_cache_dtype=auto` 最终真实 dtype 是什么？

不能只看 config。

必须看到实际 Tensor：

```text
kv_cache.dtype
```

---

## 2.5 KV Cache 的 shape 与真实 memory layout 是什么？

源码给出的 logical shape 与 HBM 的真实 memory order 不是同一个概念。

因此需要同时打印：

```text
kv_cache.shape
kv_cache.stride()
```

通过 stride 判断真实 layout。

---

## 2.6 当前 token 的 K/V 谁负责写入 Cache？

需要把：

```text
Attention.forward
↓
unified_kv_cache_update
↓
get_attention_context
↓
AttentionImpl.do_kv_cache_update
↓
reshape_and_cache_flash
```

和运行时 `slot_mapping` 对上。

---

## 2.7 Attention 如何读取历史 KV？

需要区分：

```text
WRITE
→ slot_mapping

READ
→ block_table + seq metadata
```

并验证 Decode 时：

```text
只写 1 个新 slot
但读取整个 Request 的 block table
```

---

# 3. 固定实验环境

```text
Repo:
~/learning/llm-kv-lab/third_party/vllm

vLLM:
0.26.0

commit:
568afb3a13806beb53bb2e6bd518269357b237c0

branch:
study-vllm-0.26

Python:
3.10

PyTorch:
2.11.0+cu129

GPU:
NVIDIA A100 80GB PCIe

Model:
/data/models/Qwen3-0.6B

TP=1
PP=1
DP=1
DCP=1

V1 Engine
V2 Model Runner
Async scheduling=True
```

关键配置：

```text
dtype=bfloat16
kv_cache_dtype=auto
block_size=16
max_model_len=512
max_num_batched_tokens=256
max_num_seqs=1
gpu_memory_utilization=0.2
enable_prefix_caching=False
enforce_eager=True
```

E6 特意：

```text
unset VLLM_ATTENTION_BACKEND
```

避免环境变量强制指定 backend。

Prefix Cache 关闭，是为了让本实验所有 KV 都来自当前 Request 的真实 forward，不混入 E5 的 historical-prefix reuse。

---

# 4. 为什么 workload 设计成 31-token Prompt

最终脚本构造：

```text
prompt_tokens = 31
block_size    = 16
max_tokens    = 4
```

这是刻意设计的 boundary workload。

31 个 Prompt token 会形成：

```text
position 0～15
→ logical block 0

position 16～30
→ logical block 1
```

第二个 block 还剩最后一个 slot：

```text
position 31
→ logical block 1
→ offset 15
```

再下一步：

```text
position 32
→ logical block 2
→ offset 0
```

因此实验会同时看到：

```text
31 → 32
```

这个跨 block boundary 的瞬间。

它能直接验证：

```text
旧 block 被填满
↓
Scheduler allocate 新 physical block
↓
block_table 增长
↓
ModelRunner position32 映射到新 block
↓
Attention 写入新 physical slot
```

这比随机长度 Prompt 更适合验证 Paged KV addressing。

---

# 5. 静态调用链：从 Scheduler 到 GPU KV Cache

先重新梳理这次实验对应的底层调用链。

---

## 5.1 Scheduler：只负责“Request 拥有哪些 physical blocks”

Scheduler / KVCacheManager 最终给 Request 建立类似：

```text
logical block 0 → physical block 1
logical block 1 → physical block 2
logical block 2 → physical block 3
```

Scheduler 关心的是：

```text
Request 当前需要多少 token
↓
需要多少 KV capacity
↓
需要几个 physical blocks
```

Scheduler 本身并不直接计算：

```text
position 32 应该写 HBM 哪个 token slot
```

这个 token-level 地址翻译在 ModelRunner。

---

## 5.2 ModelRunner：`prepare_inputs()` 先建立本 step token inputs

真实执行主链位于：

```text
vllm/v1/worker/gpu/model_runner.py
```

核心 runtime 结构：

```text
execute_model
↓
prepare_inputs(scheduler_output, batch_desc)
↓
InputBatch
```

`InputBatch` 中包含：

```text
req_ids
idx_mapping
query_start_loc
positions
seq_lens
num_scheduled_tokens
...
```

其中本实验 addressing 最关键的是：

```text
idx_mapping
query_start_loc
positions
```

---

## 5.3 `prepare_attn()`：同时构造 READ 页表与 WRITE 地址

源码：

```python
# Block tables: num_kv_cache_groups x [num_reqs_padded, max_num_blocks].
block_tables = self.block_tables.gather_block_tables(
    input_batch.idx_mapping,
    num_reqs_padded=input_batch.num_reqs_after_padding,
)

# Slot mappings: [num_kv_cache_groups, num_tokens_padded].
slot_mappings = self.block_tables.compute_slot_mappings(
    input_batch.idx_mapping,
    input_batch.query_start_loc,
    input_batch.positions,
    num_tokens_padded=input_batch.num_tokens_after_padding,
)
```

于是：

```text
Persistent BlockTables
        │
        ├── gather_block_tables()
        │       ↓
        │   block_tables
        │   → Attention READ
        │
        └── compute_slot_mappings()
                ↓
            slot_mappings
            → KV WRITE
```

这是本实验最重要的架构分叉之一。

---

# 6. `gather_block_tables()` 到底在做什么

ModelRunner 的 `BlockTables` 是 persistent state。

Scheduler 更新 Request 的 physical block ownership 后，ModelRunner 内部维护对应 row。

本 step batch 中 Request 可能不等于 persistent state 的 row 顺序，因此需要：

```text
batch_idx
↓
idx_mapping
↓
request-state idx
↓
persistent block table row
```

`gather_block_tables()` 将这些实际参与本 step 的 Request 行 gather 出来，形成：

```text
[num_reqs_padded, max_num_blocks]
```

例如正式 Prefill：

```text
block_table=[1,2]
```

表示：

```text
logical block0 → physical1
logical block1 → physical2
```

注意日志中：

```text
block_table_shape=(1,32)
block_table0=[1,2,0,0,0,...]
```

这里 `32` 是 metadata capacity / padded width。

真正有效的只有：

```text
[1,2]
```

后面的 `0` 不能理解成 Request 正在使用 physical block0。

---

# 7. `compute_slot_mappings()`：真正做 token-level 地址翻译

源码：

```text
vllm/v1/worker/gpu/block_table.py
```

Python wrapper：

```python
def compute_slot_mappings(
    self,
    idx_mapping,
    query_start_loc,
    positions,
    num_tokens_padded,
    out=None,
):
    ...
    _compute_slot_mappings_kernel[...] (
        ...
        idx_mapping,
        query_start_loc,
        positions,
        self.block_table_ptrs,
        self.block_table_strides,
        self.block_sizes_tensor,
        slot_mappings,
        ...
    )
```

真正公式在 Triton kernel：

```python
block_indices = positions // (block_size * CP_SIZE)
block_offsets = positions % (block_size * CP_SIZE)

block_numbers = tl.load(
    block_table_ptr
    + req_state_idx * block_table_stride
    + block_indices
)

if CP_SIZE == 1:
    slot_ids = block_numbers * block_size + block_offsets
```

当前实验：

```text
CP_SIZE = 1
```

所以直接退化为：

```text
logical_block = position // block_size

offset = position % block_size

physical_block = block_table[logical_block]

slot_id = physical_block * block_size + offset
```

这就是 Phase E 的核心公式。

---

# 8. 为什么 block ID 和 slot ID 不是一个概念

例如：

```text
physical block id = 3
block_size = 16
```

这个 physical block 代表的是一整页 capacity：

```text
slot 48
slot 49
...
slot 63
```

因为：

```text
3 * 16 = 48
```

所以：

```text
physical block
= KV page

physical slot
= page 内具体 token 的 KV location
```

例如：

```text
position32
logical block2
physical block3
offset0
→ slot48

position33
logical block2
physical block3
offset1
→ slot49
```

这是后面做 KV restore / compression / offload 必须掌握的区别。

---

# 9. `build_slot_mappings_by_layer()`：为什么 Addressing 结果能进入 Attention

文件：

```text
vllm/v1/worker/gpu/attn_utils.py
```

源码：

```python
def build_slot_mappings_by_layer(
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
) -> dict[str, torch.Tensor]:
    slot_mappings_by_layer = {}
    kv_cache_groups = kv_cache_config.kv_cache_groups

    for slot_mapping, kv_cache_group in zip(
        slot_mappings, kv_cache_groups
    ):
        for layer_name in kv_cache_group.layer_names:
            slot_mappings_by_layer[layer_name] = slot_mapping

    return slot_mappings_by_layer
```

所以：

```text
KV cache group slot_mapping
↓
按照 layer_name
↓
绑定到每一个 Attention layer
```

这一步解释了为什么 ModelRunner 算出来的：

```text
[16,17,...,46]
```

最终会出现在：

```text
model.layers.0.self_attn.attn
```

的 KV write path 中。

---

# 10. `model_state.prepare_attn()`：block_table 进入 Attention Metadata

ModelRunner 主链：

```python
block_tables, slot_mappings = self.prepare_attn(input_batch)

slot_mappings_by_layer = build_slot_mappings_by_layer(
    slot_mappings,
    self.kv_cache_config,
)

attn_metadata = self.model_state.prepare_attn(
    input_batch,
    batch_desc.cg_mode,
    block_tables,
    slot_mappings,
    self.attn_groups,
    self.kv_cache_config,
)
```

所以同一次 `prepare_attn()` 产生的两类 addressing metadata 最终走向不同：

```text
slot_mappings
↓
slot_mappings_by_layer
↓
forward_context.slot_mapping
↓
KV WRITE
```

以及：

```text
block_tables
↓
model_state.prepare_attn
↓
AttentionMetadata.block_table
↓
Attention READ
```

因此 WRITE 与 READ 使用的是同一份 Request ownership 的两种不同视图。

---

# 11. Attention 上层调用链重新梳理

文件：

```text
vllm/model_executor/layers/attention/attention.py
```

模型层调用 Attention 后，当前实现会走 unified custom-op wrapper。

核心结构可以理解成：

```text
Q/K/V projection 完成
↓
Attention.forward()
│
├── KV update path
│
└── Attention compute path
```

---

# 12. KV WRITE 调用链

上层：

```python
kv_cache_dummy_dep = unified_kv_cache_update(
    key,
    value,
    self.layer_name,
)
```

`unified_kv_cache_update()`：

```python
_, attn_layer, kv_cache, layer_slot_mapping = \
    get_attention_context(layer_name)

attn_layer.impl.do_kv_cache_update(
    attn_layer,
    key,
    value,
    kv_cache,
    layer_slot_mapping,
)
```

因此它完成的是：

```text
layer_name
↓
get_attention_context()
↓
拿到当前 layer 的：
- Attention object
- kv_cache Tensor
- layer_slot_mapping
↓
dispatch 给具体 Backend implementation
```

当前 E6 runtime：

```text
attn_layer.impl = FlashAttentionImpl
```

于是继续：

```text
FlashAttentionImpl.do_kv_cache_update()
```

---

# 13. `get_attention_context()` 为什么重要

其核心逻辑：

```python
attn_layer = forward_context.no_compile_layers[layer_name]
kv_cache = attn_layer.kv_cache
slot_mapping = forward_context.slot_mapping
layer_slot_mapping = slot_mapping.get(layer_name)
```

也就是把 ModelRunner 在 forward 前准备好的 metadata，通过 forward context 交给当前 layer。

因此：

```text
ModelRunner
build_slot_mappings_by_layer()
↓
forward_context.slot_mapping
↓
get_attention_context(layer_name)
↓
layer_slot_mapping
```

这就是 ModelRunner addressing plane 与 Attention data plane 的接口。

---

# 14. FlashAttention 的 KV WRITE

文件：

```text
vllm/v1/attention/backends/flash_attn.py
```

当前 `FlashAttentionImpl.do_kv_cache_update()`：

```python
key_cache, value_cache = (
    kv_cache.transpose(1, 2)
    .split(self.head_size, dim=-1)
)

reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    self.kv_cache_dtype,
    layer._k_scale,
    layer._v_scale,
)
```

所以真正的写入链：

```text
current key/value
+
slot_mapping
↓
reshape_and_cache_flash()
↓
scatter write
↓
GPU KV Cache
```

关键点：

```text
KV write address = slot_mapping
```

---

# 15. KV READ 调用链

上层调用：

```text
unified_attention_with_output()
```

核心：

```python
attn_metadata, self, kv_cache, _ = \
    get_attention_context(layer_name)

self.impl.forward(
    self,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output=output,
    ...
)
```

E6 runtime：

```text
self.impl = FlashAttentionImpl
```

于是：

```text
unified_attention_with_output
↓
FlashAttentionImpl.forward
```

后者拿到：

```text
query
kv_cache
attn_metadata.block_table
seq_lens
query_start_loc
max_query_len
max_seq_len
```

再进入 FlashAttention paged-attention implementation。

因此：

```text
Historical KV READ addressing
= block_table + sequence metadata
```

而不是 `slot_mapping`。

---

# 16. WRITE 与 READ 为什么故意设计成不同 metadata

这是 Paged KV 最重要的结构之一。

## WRITE

当前 step 只需要把新 token K/V 写进去。

因此最直接的数据结构是：

```text
scheduled token0 → physical slot X
scheduled token1 → physical slot Y
...
```

也就是：

```text
slot_mapping
```

---

## READ

Attention 计算需要访问整个有效历史：

```text
position 0
position 1
...
position seq_len-1
```

历史 KV 可以分散在多个 physical blocks 中。

因此需要：

```text
block_table
+
seq_len
```

让 backend 根据 logical sequence location 找对应 physical page。

所以可以压缩为：

```text
WRITE
= token-level exact address
= slot_mapping

READ
= sequence-level page table
= block_table + seq metadata
```

---

# 17. 本实验增加了哪些 Trace

为了证明上述链路，本实验只新增 addressing / attention data-path 相关 Trace，不继续修改 Scheduler/KVCacheManager。

最终新增：

```text
[BRIDGE][ADDR][BATCH]
[BRIDGE][ADDR][BLOCK_TABLE]
[BRIDGE][ADDR][MAPPING]

[BRIDGE][ATTN][KV_WRITE]
[BRIDGE][ATTN][KV_LAYOUT]
[BRIDGE][ATTN][KV_READ]
```

Attention Trace 只观察：

```text
model.layers.0.self_attn.attn
```

避免 Qwen3 多层 Attention 将日志淹没。

---

# 18. Trace 1：ADDR/BATCH、BLOCK_TABLE、MAPPING

修改文件：

```text
vllm/v1/worker/gpu/model_runner.py
```

位置：

```text
GPUModelRunner.prepare_attn()
```

放在：

```python
slot_mappings = self.block_tables.compute_slot_mappings(...)
```

之后、`return block_tables, slot_mappings` 之前。

---

## 18.1 为什么一定放这里

因为此时同时拥有：

```text
input_batch.positions
input_batch.query_start_loc
input_batch.req_ids
block_tables
actual slot_mappings
block_size
```

因此可以用源码公式自行重新计算一个 `expected_slot`，再与 Triton kernel 真正产生的 `actual_slot` 比较。

这不是只打印 kernel output，而是在做动态 invariant check。

---

## 18.2 核心 Trace 逻辑

概念代码：

```python
logical_block = position // block_size
block_offset = position % block_size

physical_block = block_table_cpu[
    batch_idx, logical_block
]

expected_slot = (
    physical_block * block_size
    + block_offset
)

actual_slot = slots_cpu[token_idx]

match = expected_slot == actual_slot
```

最终日志：

```text
[BRIDGE][ADDR][MAPPING]
req=...
position=32
logical_block=2
physical_block=3
offset=0
expected_slot=48
actual_slot=48
match=True
```

它直接证明：

```text
block_table + position
↓
slot_mapping
```

与 `_compute_slot_mappings_kernel` 的真实结果完全一致。

---

# 19. Trace 2：ATTN/KV_WRITE

修改文件：

```text
vllm/model_executor/layers/attention/attention.py
```

位置：

```text
unified_kv_cache_update()
```

放在：

```python
get_attention_context(layer_name)
```

之后、`attn_layer.impl.do_kv_cache_update()` 之前。

打印：

```text
layer
impl
key_shape
value_shape
kv_shape
kv_stride
kv_dtype
kv_device
slot_shape
slots
```

---

## 19.1 为什么这里比只改 FlashAttention 更好

`unified_kv_cache_update()` 是 backend-generic dispatch 层。

所以未来：

```text
E6 → FlashAttention
E7 → TritonAttention
E8 → Triton INT4
```

都可以复用同一个 Trace。

只需要观察：

```text
impl=...
```

发生变化。

同时它还能直接证明：

```text
ModelRunner slot_mapping
↓
get_attention_context
↓
具体 Backend 收到了同一组 slot IDs
```

---

# 20. Trace 3：ATTN/KV_LAYOUT

修改文件：

```text
vllm/v1/attention/backends/flash_attn.py
```

位置：

```text
FlashAttentionImpl.do_kv_cache_update()
```

放在：

```python
key_cache, value_cache = \
    kv_cache.transpose(1, 2).split(...)
```

之后。

打印：

```text
kv_cache.shape
kv_cache.stride
kv_cache.dtype

key_cache.shape
key_cache.stride

value_cache.shape
value_cache.stride
```

这个 Trace 用于回答：

```text
logical KV Tensor shape 是什么？
真实 physical memory layout 是什么？
FlashAttention 最终看到的 K/V view 又是什么？
```

---

# 21. Trace 4：ATTN/KV_READ

修改文件：

```text
vllm/model_executor/layers/attention/attention.py
```

位置：

```text
unified_attention_with_output()
```

放在：

```python
attn_metadata, self, kv_cache, _ = \
    get_attention_context(layer_name)
```

之后、`self.impl.forward()` 之前。

打印：

```text
layer
impl
query_shape
kv_shape
block_table_shape
block_table first row
seq_lens
query_start_loc
max_query_len
max_seq_len
```

它证明：

```text
FlashAttentionImpl.forward
```

拿到的是：

```text
整个 historical block table
+
sequence metadata
```

而不是只拿当前 token 的 slot。

---

# 22. 为什么只 Trace layer 0

Qwen3 有多层 Attention。

如果每层、每 step 都打印：

```text
KV_WRITE
KV_LAYOUT
KV_READ
```

日志会膨胀数十倍。

本实验不是验证不同 layer 的差异，而是验证统一的数据路径，所以选择：

```text
model.layers.0.self_attn.attn
```

作为代表层。

这属于 instrumentation 降噪，不改变实验语义。

---

# 23. 实验前出现的非 E6 故障

第一次启动时 vLLM import 失败：

```text
vllm/config/parallel.py
SyntaxError
```

检查 `git diff` 后发现该文件存在误编辑：

```python
if self.enable_eplb:
        raise ValueError(
    if not current_platform.is_cuda_alike():
```

而 HEAD 正确代码应为：

```python
if self.enable_eplb:
    if not current_platform.is_cuda_alike():
        raise ValueError(...)
```

该文件与 E6 Trace 无关，因此恢复：

```bash
git restore --source=HEAD -- vllm/config/parallel.py
```

并使用：

```bash
python -m py_compile ...
```

确认所有 Trace 文件语法通过后再跑实验。

这个问题不属于 E6 runtime 结论，只是实验前代码工作树排障。

---

# 24. 日志阶段划分：dummy、warmup、正式 Request

E6 日志不能从第一条 Attention Trace 就开始解读。

启动阶段出现：

```text
[MRV2][EXECUTE_BEGIN]
total_scheduled=256
per_req={'_dummy_req_0': 256}
```

随后：

```text
[ATTN][KV_WRITE]
kv_shape=(0,)
slot_shape=None
```

这不是异常。

这是 KV Cache 正式分配前的 profile/dummy forward。

随后 engine 才输出：

```text
Available KV cache memory: 14.61 GiB
GPU KV cache size: 136,736 tokens
```

之后 `_warmup_0_` 才看到真实：

```text
kv_shape=(8546, 8, 16, 256)
```

因此日志分析必须按：

```text
_dummy_req_
→ profile / ignore for real addressing

_warmup_
→ real KV allocated, mechanism smoke

req=0-a2c39288
→ 正式实验 workload
```

这也是后续 E7/E8 分析日志时应该沿用的规则。

---

# 25. Backend selection 实验结果

没有指定 Attention Backend。

runtime：

```text
Using FLASH_ATTN attention backend
out of potential backends:
['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']
```

随后：

```text
Using FlashAttention version 2
```

正式 Attention Trace：

```text
impl=FlashAttentionImpl
```

因此完整证据：

```text
Default selector
↓
FLASH_ATTN
↓
FlashAttentionImpl
↓
FlashAttention v2
```

结论：

```text
E6 Default Attention Backend = FLASH_ATTN / FA2
```

不是根据 GPU 型号推测，而是 runtime 证明。

---

# 26. KV Cache capacity 与 Tensor shape 对上

engine：

```text
GPU KV cache size: 136,736 tokens
```

runtime KV Tensor：

```text
kv_shape=(8546, 8, 16, 256)
```

其中：

```text
num_blocks = 8546
block_size = 16
```

计算：

```text
8546 × 16
= 136736 tokens
```

和 engine capacity 完全一致。

说明第一维：

```text
8546
```

确实就是这个 KV group 的 physical block capacity。

---

# 27. KV dtype：`auto` 最终解析为 BF16

配置：

```text
model dtype=torch.bfloat16
kv_cache_dtype=auto
```

runtime：

```text
kv_dtype=torch.bfloat16
```

因此可以正式得出：

```text
E6 实际 KV Cache dtype = BF16
```

而不是仅凭 config 猜测。

---

# 28. KV logical shape

FlashAttention backend 源码：

```text
[num_blocks,
 num_kv_heads,
 block_size,
 2 * head_size]
```

runtime：

```text
(8546, 8, 16, 256)
```

对应：

```text
num_blocks    = 8546
num_kv_heads  = 8
block_size    = 16
head_size     = 128
2*head_size   = 256
```

因此 logical packed representation：

```text
[B, Hkv, N, 2D]
```

其中最后 `2D` 是：

```text
K D
+
V D
```

---

# 29. 最关键的 layout 分析：shape 不等于 physical layout

runtime：

```text
kv_shape=(8546, 8, 16, 256)
kv_stride=(32768, 256, 2048, 1)
```

如果它在内存中真的按：

```text
[B,H,N,2D]
```

普通 contiguous 排列，那么 `H stride` 应接近：

```text
N × 2D
= 16 × 256
= 4096
```

但实际：

```text
H stride = 256
N stride = 2048
```

观察：

```text
H stride = 256
N stride = 8 × 256 = 2048
B stride = 16 × 8 × 256 = 32768
```

这正好对应 physical order：

```text
[B,N,H,2D]
```

即 NHD layout。

所以：

```text
logical shape
= [B,H,N,2D]

physical memory order
= [B,N,H,2D]
```

两者通过 stride 表达。

这是 E6 一个非常重要的结果：

> 看 KV Cache layout 时不能只看 `shape`，必须同时看 `stride`。

---

# 30. `transpose + split` 后的 K/V view

源码：

```python
key_cache, value_cache = (
    kv_cache.transpose(1, 2)
    .split(self.head_size, dim=-1)
)
```

runtime：

```text
key_cache_shape=(8546,16,8,128)
key_cache_stride=(32768,2048,256,1)

value_cache_shape=(8546,16,8,128)
value_cache_stride=(32768,2048,256,1)
```

逻辑过程：

```text
packed KV
[B,H,N,2D]

↓ transpose(1,2)

[B,N,H,2D]

↓ split(D,D)

K = [B,N,H,D]
V = [B,N,H,D]
```

而 stride：

```text
(32768,2048,256,1)
```

正是 `B,N,H,D` 的自然 memory order。

因此 FlashAttention 的 `reshape_and_cache_flash()` 最终拿到的是：

```text
key_cache  [block, token-in-block, kv-head, head-dim]
value_cache[block, token-in-block, kv-head, head-dim]
```

这对理解 physical KV layout 很关键。

---

# 31. Q/K/V shape 顺便验证 GQA

正式 Prefill：

```text
query_shape=(31,16,128)
key_shape=(31,8,128)
value_shape=(31,8,128)
```

所以：

```text
Q heads  = 16
KV heads = 8
D        = 128
```

即：

```text
16 / 8 = 2
```

每 2 个 Query heads 共享 1 个 KV head。

这是 Qwen3-0.6B 当前 runtime 的 GQA 结构。

---

# 32. 正式 Call 0：31-token Prefill

Scheduler：

```text
num_tokens=31
num_computed=0
pending=31
scheduled=31
new_block_ids=([1,2],)
```

意味着 Scheduler 为 Request 建立：

```text
logical block0 → physical1
logical block1 → physical2
```

ModelRunner：

```text
[ADDR][BLOCK_TABLE]
block_table=[1,2]
```

控制面与 addressing plane 第一次直接对上。

---

# 33. Prefill position 0～15：physical block1

公式：

```text
logical_block = position // 16
offset        = position % 16
physical      = block_table[logical_block]
slot          = physical*16 + offset
```

position0：

```text
logical=0
physical=1
offset=0
slot=16
```

runtime：

```text
expected_slot=16
actual_slot=16
match=True
```

position15：

```text
logical=0
physical=1
offset=15
slot=31
```

runtime：

```text
expected_slot=31
actual_slot=31
match=True
```

所以 physical block1 对应本次 Request 的：

```text
positions 0～15
slots 16～31
```

---

# 34. Prefill position16：第一次 page boundary

position16：

```text
logical_block=1
physical_block=block_table[1]=2
offset=0
slot=2*16=32
```

runtime：

```text
position=16
logical_block=1
physical_block=2
offset=0
expected_slot=32
actual_slot=32
match=True
```

这直接证明：

```text
position15
→ block1 / slot31

position16
→ block2 / slot32
```

Paged KV 的 page boundary 翻译正确。

---

# 35. Prefill 的所有 31 个 token 全部匹配

日志从：

```text
position0 → slot16
```

一直到：

```text
position30 → slot46
```

全部：

```text
match=True
```

因此 Phase E 不只是抽样成功，而是正式 Prefill 的全部 31 个 token 都通过了 invariant check。

---

# 36. Prefill Address Trace 与 KV WRITE 完全对上

ModelRunner 产生：

```text
actual slots:
[16,17,18,...,46]
```

紧接着 Attention：

```text
[ATTN][KV_WRITE]
slot_shape=(31,)
slots=[16,17,18,...,46]
```

这条日志非常关键，因为它直接把两层连接起来：

```text
ModelRunner.compute_slot_mappings()
↓
[16..46]

build_slot_mappings_by_layer()
↓
forward_context.slot_mapping
↓
get_attention_context()
↓
FlashAttentionImpl.do_kv_cache_update()
↓
slots=[16..46]
```

也就是说：

> ModelRunner 根据 `block_table + positions` 算出的 token-level physical slots，确实原样进入了 FlashAttention KV write path。

---

# 37. Prefill KV WRITE

runtime：

```text
key_shape=(31,8,128)
value_shape=(31,8,128)
slot_shape=(31,)
```

说明本轮 Prefill：

```text
31 个 token
→ 31 组 K
→ 31 组 V
→ 31 个 write slots
```

写入的目标 Tensor：

```text
kv_shape=(8546,8,16,256)
kv_dtype=torch.bfloat16
```

因此当前 Request 的 Prefill K/V 被写入全局 BF16 Paged KV Cache。

---

# 38. Prefill KV READ

同轮 Attention：

```text
query_shape=(31,16,128)
block_table0=[1,2,0,0,...]
seq_lens=[31]
query_start_loc=[0,31]
max_query_len=31
max_seq_len=31
```

有效 historical page table：

```text
[1,2]
```

因此 Attention forward 已经拿到完整 Request 的 paged addressing metadata。

注意：

```text
WRITE slots = [16..46]
```

而 READ metadata 是：

```text
block_table=[1,2]
seq_len=31
```

两者粒度不同，但描述的是同一个 Request 的 KV ownership。

---

# 39. Call 1：position31，仍然使用 physical block2

Async Scheduler 下一轮：

```text
num_tokens=31
num_computed=31
pending=1
scheduled=1
new_block_ids=None
```

为什么不需要新 block？

因为：

```text
position31
logical block1
offset15
```

仍然是第二个 16-token block 的最后一个位置。

ModelRunner：

```text
block_table=[1,2]
```

Address：

```text
position=31
logical_block=1
physical_block=2
offset=15
expected_slot=47
actual_slot=47
match=True
```

Attention write：

```text
key_shape=(1,8,128)
value_shape=(1,8,128)
slots=[47]
```

Attention read：

```text
query_shape=(1,16,128)
block_table=[1,2]
seq_lens=[32]
```

完整链：

```text
position31
↓
logical block1
↓
physical block2
↓
slot47
↓
write new K/V to slot47
↓
read history through blocks [1,2], seq_len32
```

---

# 40. Call 2：position32，真正跨 Decode block boundary

这是整个 E6 最关键的一轮。

上一轮结束后：

```text
positions 0～31
```

已经刚好占满两个 16-token blocks。

Scheduler 下一步：

```text
num_tokens=32
scheduled=1
new_block_ids=([3],)
```

说明控制面发现：

```text
下一 token position32
已经无法写入 block2
```

因此新分配：

```text
physical block3
```

于是 ownership 变成：

```text
logical0 → physical1
logical1 → physical2
logical2 → physical3
```

ModelRunner：

```text
block_table=[1,2,3]
```

Address：

```text
position=32
logical_block=2
physical_block=3
offset=0
expected_slot=48
actual_slot=48
match=True
```

Attention write：

```text
slots=[48]
```

Attention read：

```text
block_table=[1,2,3]
seq_lens=[33]
```

完整动态证据链：

```text
Scheduler:
new_block_ids=[3]

↓

ModelRunner:
block_table=[1,2,3]

↓

position32
logical block2

↓

physical block3

↓

slot48

↓

FlashAttention KV_WRITE:
slots=[48]

↓

FlashAttention KV_READ:
block_table=[1,2,3]
seq_len=33
```

这就是 Scheduler 控制面到 GPU KV 数据面的完整动态桥。

---

# 41. Call 3：position33，验证 block3 内部继续写

下一轮：

```text
new_block_ids=None
```

因为 physical block3 仍有空间。

ModelRunner：

```text
position=33
logical_block=2
physical_block=3
offset=1
expected_slot=49
actual_slot=49
match=True
```

Attention：

```text
KV_WRITE slots=[49]

KV_READ block_table=[1,2,3]
seq_lens=[34]
```

这进一步证明：

```text
position32 → block3 offset0 → slot48
position33 → block3 offset1 → slot49
```

不是只在 boundary 单点碰巧正确。

---

# 42. 为什么 Decode 每轮只写 1 个 slot，却要读整个 block table

Decode 时：

```text
query_shape=(1,16,128)
key_shape=(1,8,128)
value_shape=(1,8,128)
```

所以本轮只产生当前 1 个 token 的 Q/K/V。

WRITE：

```text
slot_mapping=[47]
```

或者：

```text
[48]
```

只需把当前 K/V 写入一个位置。

但是 Attention 计算当前 query 时，需要历史：

```text
K0,V0
K1,V1
...
Kcurrent,Vcurrent
```

所以 READ：

```text
block_table=[1,2]
seq_len=32
```

或：

```text
block_table=[1,2,3]
seq_len=33
```

因此 Decode 最典型的结构是：

```text
WRITE very small
READ history grows
```

这也是 Decode 阶段 KV bandwidth 逐渐重要的根本原因之一。

---

# 43. Async Scheduling 与 Addressing 如何同时成立

E6 仍然开启：

```text
Asynchronous scheduling
batch_queue_size=2
```

Call0 提交 31-token Prefill 后，没有马上 reconcile：

```text
QUEUE_PUSH len=1
YIELD_WITHOUT_CONSUME
```

Call1 又继续 schedule position31：

```text
num_computed=31
num_in_flight=31
scheduled=1
```

此时 Prefill output 还没被 Scheduler settle，但 Scheduler 的 logical frontier 已经 commit。

随后 queue 满：

```text
len=2
```

才开始消费 Call0 result。

这说明我们以前 E1 得到的认知继续成立：

```text
num_computed_tokens
= Scheduler logical committed frontier
```

而不是：

```text
已经被 Scheduler reconcile 的 GPU token 数
```

更重要的是：

> 即使存在 outstanding batch，block ownership 与 token addressing 仍然按逻辑提交状态正确推进。

因此 Call1 的 position31 和 Call2 的 position32 都获得了正确 physical address。

---

# 44. 为什么最终输出4个 token，却只有3次 Decode forward

最终：

```text
output_token_ids=[38297,323,279,1372]
num_output_tokens=4
```

实际 forward：

```text
Call0
Prefill 31 tokens
→ sample output token #1

Call1
Decode position31
→ sample output token #2

Call2
Decode position32
→ sample output token #3

Call3
Decode position33
→ sample output token #4
```

因此：

```text
4 output tokens
=
1 Prefill sample
+
3 Decode samples
```

不是：

```text
4 output tokens
=
4 additional Decode forwards
```

Call4：

```text
total_scheduled=0
```

只是 async pipeline 继续 drain / reconcile 最后一批 output。

这和 E1 的结论完全一致：

> 最后一个 sampled token 不需要再为“打印出来”执行额外 forward。

---

# 45. Call4/5/6 为什么还有空 batch

日志后面：

```text
Call4 total_scheduled=0
Call5 total_scheduled=0
Call6 drain queue
```

这不是模型还在生成 token。

原因是 async batch queue 需要把之前已经提交的 transaction 完整消费掉。

可以理解为：

```text
Request 已 finish
≠
EngineCore batch queue 已 drain
```

因此：

```text
request lifecycle
```

和：

```text
async batch transaction lifecycle
```

不是完全同一件事。

---

# 46. Prefix Trace 为什么仍然出现，但没有影响 E6

正式日志有：

```text
[PREFIX][LOOKUP]
local_hit_tokens=0
```

同时配置：

```text
enable_prefix_caching=False
```

我们保留了 E5 instrumentation，所以相关 trace code 仍可能打印 lookup bookkeeping 状态。

但实际结果：

```text
local_hit_tokens=0
hit_block_ids=None
```

且所有最终 free：

```text
has_hash=False
```

所以本实验不存在 historical Prefix Cache reuse。

E6 的全部 KV 都是当前 Request 自己写出的。

---

# 47. E6 建立的三层架构认知

现在可以把 vLLM Paged KV 完整分为三层。

## 第一层：KV Control Plane

```text
Scheduler
KVCacheManager
BlockPool
```

负责：

```text
Request 需要多少 KV capacity
↓
Request 拥有哪些 physical blocks
```

输出语义：

```text
logical block → physical block
```

例如：

```text
[1,2,3]
```

---

## 第二层：Addressing Plane

```text
GPUModelRunner
BlockTables
```

负责：

```text
block_table
+
current token positions
↓
exact physical slot
```

输出：

```text
slot_mapping
```

例如：

```text
position32
→ logical2
→ physical3
→ slot48
```

---

## 第三层：Attention Data Plane

```text
Attention layer
Attention Backend
GPU op/kernel
```

WRITE：

```text
current K/V
+
slot_mapping
↓
reshape_and_cache_flash
↓
KV Cache
```

READ：

```text
query
+
block_table
+
seq metadata
↓
FlashAttention paged read
↓
Attention output
```

三层职责明确：

```text
Scheduler
= ownership

ModelRunner
= address translation

Attention Backend
= layout + write/read semantics

GPU op
= real memory operation
```

---

# 48. E5 与 E6 终于接起来了

E5 证明：

```text
历史 KV block 已经存在
↓
Prefix lookup hit
↓
touch/adopt
↓
num_computed_tokens frontier advance
↓
skip duplicated Prefill
```

但 E5 没回答：

```text
那些有效 KV 在 GPU 上到底放在哪里？
```

E6 证明：

```text
physical block ownership
↓
block_table
↓
slot_mapping
↓
KV Cache physical location
↓
Attention read
```

所以合起来：

```text
E5
回答“什么时候可以不重算”

E6
回答“已经有效的 KV 在 GPU 中怎么定位和使用”
```

这两步是未来做 External KV / LMCache / compressed restore 的直接基础。

---

# 49. 对未来 Compression-Aware KV Offload 的接口启示

未来如果做：

```text
CPU / LMCache / compressed KV
↓
restore to GPU
```

不能只说：

```text
把 KV 放回 GPU
```

真正需要满足 vLLM runtime 契约：

```text
1. Scheduler / KV manager
   给 Request 建立正确 physical block ownership

2. restore path
   知道 target physical block / target slot

3. 按当前 Attention Backend 的 KV layout
   写入正确 K/V representation

4. logical computed frontier
   被正确 advance

5. 后续 Attention
   不需要知道 KV 来自 GPU compute 还是 external restore
   只通过 block_table 正常读取
```

E6 已经把第 2～5 条最底层的 addressing/layout contract 解释清楚。

---

# 50. 这次实验没有证明什么

需要保持证据边界。

E6 已经证明：

```text
FlashAttentionImpl.forward 被执行

它收到：
block_table
seq_lens
query metadata

KV write 收到：
slot_mapping
```

结合源码，可以确认 paged KV metadata path。

但 E6 **没有**做：

```text
Nsight kernel-level load/store trace
```

所以不能写成：

```text
我们逐条观察了 FlashAttention CUDA kernel
从 block1/block2/block3 发出的具体 HBM load 指令
```

如果未来要研究：

```text
kernel name
HBM bandwidth
memory transaction
warp behavior
```

那属于 Nsight / backend kernel profiling，不属于本 E6。

---

# 51. E6 PASS Gate

| Gate | 结果 |
|---|---|
| Default backend 未显式指定 | PASS |
| Runtime backend = FLASH_ATTN | PASS |
| FlashAttention version = 2 | PASS |
| Runtime KV dtype = BF16 | PASS |
| KV logical shape 可见 | PASS |
| KV stride / physical layout 可解释 | PASS |
| K/V view shape + stride 可见 | PASS |
| block_table 可见 | PASS |
| positions 可见 | PASS |
| slot_mapping 可见 | PASS |
| expected_slot == actual_slot | PASS |
| Prefill boundary 15→16 | PASS |
| Decode boundary 31→32 | PASS |
| Scheduler new block 与 ModelRunner table 对应 | PASS |
| KV write implementation = FlashAttentionImpl | PASS |
| ModelRunner slots 与 KV_WRITE slots 完全一致 | PASS |
| historical KV read metadata 可见 | PASS |
| Prefill query path 可见 | PASS |
| Decode query_len=1 path 可见 | PASS |
| 正常生成 4 tokens | PASS |

结论：

```text
Phase E + E6 = PASS
Instrumentation = Complete
Status = Frozen
```

不需要再为 E6 补第二个 workload。

---

# 52. 最重要的一条动态证据链

最终把整个实验压缩成 position32 这一轮：

```text
Request 已有 32 个有效 token
block_size=16

↓

前两个 logical blocks 全满

↓

Scheduler
allocate physical block3

↓

block ownership:
logical0 → physical1
logical1 → physical2
logical2 → physical3

↓

ModelRunner.prepare_attn()

↓

gather_block_tables()
block_table=[1,2,3]

↓

compute_slot_mappings()

position32
logical_block=32//16=2
offset=32%16=0
physical_block=block_table[2]=3

↓

expected_slot=3*16+0=48
actual_slot=48
match=True

↓

build_slot_mappings_by_layer()

↓

get_attention_context()

↓

FlashAttentionImpl.do_kv_cache_update()
slot_mapping=[48]

↓

reshape_and_cache_flash()

↓

BF16 KV Cache
physical block3 / offset0

↓

FlashAttentionImpl.forward()
block_table=[1,2,3]
seq_len=33

↓

Attention 可以访问完整 positions 0～32 的历史 KV
```

这就是 vLLM Paged KV Cache 从：

```text
Scheduler 分配 physical block
```

一路到：

```text
GPU Attention 实际写入/读取 KV
```

的完整动态桥。

---

# 53. 最终应形成的长期认知

以后看到下面几个对象时，应直接建立如下语义：

```text
Request.block_ids / BlockPool blocks
= ownership / capacity

block_table
= logical block → physical block page table

position
= sequence logical token position

slot_mapping
= 当前 scheduled token 的 exact physical KV write address

kv_cache
= backend-defined physical storage

AttentionMetadata.block_table
= historical KV read page table
```

因此完整关系是：

```text
physical block ownership
        ↓
    block_table
        ↓
position + block_table
        ↓
   slot_mapping
        ↓
 current KV write

同时：

block_table + seq_lens
        ↓
historical KV read
```

这套模型就是后面继续理解 Triton Attention、INT4 KV、external KV restore、LMCache 对接时最重要的地址与数据面基础。
