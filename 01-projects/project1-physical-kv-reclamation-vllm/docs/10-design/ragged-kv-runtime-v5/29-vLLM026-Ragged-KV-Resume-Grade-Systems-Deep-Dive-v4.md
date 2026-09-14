> **v5 canonical addendum:** R2+ allocation/reclamation 必须遵循 `38-Scheduler-Side-Ragged-Physical-State-and-Allocation-Contract.md`；metadata shape/lifetime 遵循 `39-Ragged-Step-Metadata-Shape-Lifetime-and-Attention-Contract.md`。

# vLLM 0.26 Ragged KV Runtime — Resume-Grade Systems Deep Dive v4

**Project:** Non-Uniform / Ragged KV Cache Runtime for vLLM 0.26 Model Runner V2  
**Date:** 2026-09-14  
**Status:** Canonical Architecture / Implementation Deep-Dive  
**Primary implementation base:** `qcrs/vllm:p1/v2-token-compaction-v026`  
**Reference system:** `aiha-lab/tangram:main`

---

# 0. Executive Verdict

本项目最合理的定位不是：

> “复现 Tangram”  
> “实现一个新的 KV eviction scorer”  
> “给 vLLM 加一个 token deletion 策略”

而是：

> **在 vLLM 0.26 Model Runner V2 上重新构建支持 per-layer / per-head-group 非均匀 KV retention 的 Ragged KV Runtime，并打通 Scheduler / Allocator / BlockTable / KV Write / FlashAttention Read / Physical Reclamation / TP / CUDA Graph。**

真正值得写进简历的不是某一个 scorer，而是以下完整系统闭环：

```text
Non-uniform retention decision
        ↓
per-(request, layer, head-group) physical KV state
        ↓
head-group page allocation
        ↓
member-aware KV write
        ↓
member-major FlashAttention read
        ↓
paged KV compaction
        ↓
scheduler-authoritative ownership reconciliation
        ↓
BlockPool free
        ↓
released pages reused by another request
        ↓
original request continues decoding
```

如果最终能完成：

```text
R1 Identity Ragged Paging
R2 Per-group Physical State
R3 Manual Non-uniform Physical Reclamation
R4 One Real Compression Policy
+ Preemption
+ TP=2 Uniform
+ Piecewise CUDA Graph
+ AOT Clustering（推荐）
```

这个项目已经能够稳定体现一条完整的 **LLM Serving / GPU Runtime / Distributed Inference / Compiler Runtime / Triton Kernel** 技术栈。

本轮审计后，最重要的新增结论有四个：

1. **Ragged KV 的第一硬点不是 scorer，而是 physical layout / stride contract。**
2. **vLLM 0.26 已经把 KV write 与 Attention read 拆开，因此必须分别设计 Ragged KV Write 与 Ragged Attention Read。**
3. **TP 的核心不是“多卡能跑”，而是 rank-local head geometry 与 scheduler-visible physical free boundary 必须一致。**
4. **CUDA Graph 不应另起炉灶；应复用 v0.26 已有 `unified_kv_cache_update` / `unified_attention_with_output` splitting seam，把动态 Ragged metadata 隔离在 eager island。**

---

# 1. Source Pins

本项目所有设计判断应固定到具体 source pin，不使用模糊的 “当前 vLLM”。

## 1.1 qcrs/vLLM

```text
Repository:
https://github.com/qcrs/vllm

Branch:
p1/v2-token-compaction-v026

HEAD:
bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00

Baseline:
568afb3a13806beb53bb2e6bd518269357b237c0
```

当前 P1/V2 branch 相对 baseline 为 5 commits ahead。

核心已经存在的 P1 能力：

- logical `num_computed_tokens` 与 physical `effective_kv_len` 分离；
- `cache_positions`；
- Worker-side token-level compaction；
- PyTorch reference oracle；
- Triton gather/writeback；
- Worker → Scheduler compaction result；
- Scheduler canonical ownership reconciliation；
- physical block free / reuse closure。

## 1.2 Tangram

```text
Repository:
https://github.com/aiha-lab/tangram

Branch:
main

HEAD:
6fa551fc8f6edcc118a2a39b3554ee1520e91edd
```

Tangram 在本项目中的角色是：

```text
Reference Architecture
+ Semantics Oracle
+ Test/Invariant Source
+ Optimization Reference
```

而不是直接 cherry-pick 的 implementation base。

原因：Tangram fork 的 vLLM 结构与当前 qcrs/v0.26 MRv2 已经存在明显 API / runtime drift。

---

# 2. 项目到底体现什么技术栈

这个项目的价值在于，它不是只停留在某个 kernel 或某个算法，而是横跨 serving system 的多个层级。

| 技术层 | 本项目实际涉及内容 | 最终可展示能力 |
|---|---|---|
| LLM Serving Runtime | vLLM Scheduler / continuous batching / preemption | 理解 request lifecycle 与 serving state machine |
| KV Memory Management | PagedAttention / BlockPool / BlockTable / page ownership | 能从逻辑 token retention 推导到真实 GPU page reclamation |
| Attention Runtime | FlashAttention 2 / varlen / GQA | 能修改 attention metadata 与 physical KV visibility |
| GPU Data Layout | NHD/HND / page stride / virtual block | 理解 layout、stride、alias、zero-copy view |
| CUDA / Triton | paged gather / writeback / slot mapping / in-place compaction | 能写 reference→optimized kernel 闭环 |
| PyTorch Runtime | custom op / forward context / metadata builder | 理解 eager / compile seam |
| torch.compile | splitting ops / piecewise graph | 能处理动态 runtime state 与 compiler boundary |
| CUDA Graph | persistent buffers / fixed address / replay safety | 能做 decode hot-path capture compatibility |
| Distributed Inference | TP=2 / NCCL / `torch.distributed.all_reduce` | 能处理 rank-local state 与 global physical contract |
| CPU↔GPU Metadata | UVA / pinned memory / staged writes | 理解 serving metadata hot path |
| Systems Correctness | scheduler-authoritative ownership / stale fence | 能设计 transaction 与 failure-safe state transition |
| Performance Engineering | TTFT / TPOT / throughput / allocator overhead / profiler | 能做 evidence-driven optimization |
| Offline Runtime Optimization | AOT head clustering | 能把 offline profile 转换成 serving placement optimization |

因此最终简历叙事应当是：

> **KV Cache Runtime / Serving System 项目**

而不是：

> **KV Cache Algorithm 项目**

---

# 3. 必须追踪的 vLLM 完整主链

后续每一轮实现都应该能把修改点放进下面这条真实链路，而不是只看局部文件。

---

## 3.1 Startup / KV Geometry Chain

```text
ModelConfig / CacheConfig
        ↓
Attention.__init__
        ↓
Attention.get_kv_cache_spec()
        ↓
get_kv_cache_spec()
        ↓
KVCacheSpec grouping / merge
        ↓
kv_cache_utils.get_kv_cache_config_from_groups()
        ↓
KVCacheConfig
        ↓
GPUModelRunner.initialize_kv_cache()
        ↓
init_attn_backend()
        ↓
BlockTables(...)
        ↓
init_kv_cache()
        ↓
_allocate_kv_cache()
        ↓
_reshape_kv_cache()
        ↓
bind_kv_cache()
```

qcrs source：

- `vllm/model_executor/layers/attention/attention.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/gpu/model_runner.py`

Tangram 对应参考：

- `vllm/v1/kv_cache_interface.py::RaggedAttentionSpec`
- `vllm/v1/core/kv_cache_utils.py`

### Ragged 必须插入的点

```text
Attention.get_kv_cache_spec
        ↓
RaggedAttentionSpec
        ↓
ragged page bytes
        ↓
global page pool capacity
        ↓
single shared raw backing
        ↓
Ragged physical cache view
```

这一链决定的是：

> “GPU 上究竟分配了什么”。

如果这一层不正确，后面的 BlockTable、Attention、reclaim 全部没有意义。

---

## 3.2 Scheduler / Allocator Chain

```text
Scheduler.schedule()
        ↓
KVCacheManager.allocate_slots()
        ↓
KVCacheCoordinator
        ↓
SingleTypeKVCacheManager.get_num_blocks_to_allocate()
        ↓
SingleTypeKVCacheManager.allocate_new_blocks()
        ↓
BlockPool
        ↓
KVCacheBlocks
        ↓
SchedulerOutput.new_block_ids
```

qcrs source：

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`

Tangram Ragged reference：

- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`

### Dense 当前语义

通常：

```text
one logical token block
→ one scheduler block id
→ block id addresses per-layer physical storage
```

### Ragged 目标语义

```text
one token-block-depth
× G_total head-groups
→ G_total independent physical page ids
```

其中：

```math
G_{local} = H_{kv,local} / H_p
```

```math
G_{total} = L \times G_{local}
```

这直接打破了 Dense “request block row 单一深度”的假设。

---

# 4. Tangram → qcrs/v0.26 的精确映射

下表应作为后续实现的 canonical mapping。

| Tangram Surface | Tangram 作用 | qcrs/v0.26 Target | Action |
|---|---|---|---|
| `config/cache.py::page_group_size` | Ragged enable / geometry | `vllm/config/cache.py` | ADAPT |
| `RaggedAttentionSpec` | page bytes / groups | `v1/kv_cache_interface.py` | ADAPT |
| Ragged spec validation | reject unsupported | config + Attention spec | ADAPT |
| `get_num_blocks()` ragged branch | global page pool | `v1/core/kv_cache_utils.py` | PORT + ADAPT |
| single shared `KVCacheTensor` | all layers share pool | `get_kv_cache_config_from_groups` | PORT |
| Ragged BlockPool ring | bulk IDs | `core/block_pool.py` | DELAY |
| Ragged manager flat IDs | scheduler ownership | new Ragged manager | REDESIGN |
| `RaggedBlockTable` | `[req,group,block]` | new MRv2 `RaggedBlockTables` | REDESIGN |
| `ragged_layout.py` | virtual blocks/member map | new MRv2 module | PORT/ADAPT |
| `virtual_block_id` | reuse FA paging | same semantics | PORT |
| `ragged_forward.py` decode | member-major decode | existing attention seam | ADAPT |
| `ragged_forward.py` prefill | token→member copy | existing attention seam | ADAPT |
| Tangram KV write | inside ragged attention path | `unified_kv_cache_update` | REDESIGN |
| Tangram metadata builder | RaggedStepViews | FA MetadataBuilder | ADAPT |
| per-group effective len | non-uniform physical state | new `RaggedKVState` | REDESIGN |
| compression executor | plan/writeback | R4 | SELECTIVE PORT |
| Torch writeback | reference | P1 + Tangram | REUSE-P1 |
| Triton in-place writeback | optimized compaction | R8 | DELAY |
| BudgetScope uniform | TP-compatible | R4/R6 | PORT |
| layer/global budget | TP1 nonuniform | optional | DELAY |
| KeyDiff | scorer | R4 | SELECTIVE PORT |
| FastKVZip | checkpoint gate | none | SKIP |
| AOT clustering | placement | R7 | PORT/ADAPT |
| TP kept-length MAX | free boundary sync | R6 | PORT |
| Tangram custom ragged op | eager CG seam | existing v0.26 ops | DO NOT PORT FIRST |
| Piecewise CG | dynamic metadata | native v0.26 compilation | REUSE |
| snapshot/restore row | old persistent batch semantics | MRv2 persistent row | MOSTLY SKIP |
| preemption reset | recompute lifecycle | MRv2 preempt hook | ADAPT |
| prefix cache disable | incompatible mutable layout | config validation | PORT |
| DCP/PCP | unsupported | reject | FREEZE OUT |
| MLA | incompatible geometry | reject | FREEZE OUT |
| quantized KV | additional layout | reject initially | FREEZE OUT |

这张表里真正属于项目“自主工程设计”的地方主要是：

```text
Global Pool integration
MRv2-native RaggedBlockTables
Ragged KV Write seam
Ragged per-group state
Scheduler ownership/reconciliation
TP free-boundary protocol
Piecewise CG integration
```

---

# 5. R1 最容易被低估的难点：Physical Layout / Stride Contract

这是本轮最重要的新发现。

---

## 5.1 qcrs 当前 FlashAttention KV shape

qcrs 当前 FA2 logical cache shape：

```text
[B, H, N, 2D]
```

其中：

```text
B = physical blocks
H = KV heads
N = block_size
D = head_size
```

qcrs backend 支持：

```text
NHD
HND
```

Dense 默认在没有 connector 强制 layout 时使用：

```text
NHD
```

也就是实际物理排列倾向：

```text
[B, N, H, 2D]
```

---

## 5.2 Tangram zero-copy virtual block 的隐含条件

Tangram 明确采用：

```text
[2, B, Hp, N, D]
```

`Hp` 放在 `block_size` 外部。

因此：

```text
(physical_block, column)
```

两个维度连续，可以直接：

```text
virtual_block_id
= physical_block * Hp + column
```

再 zero-copy reshape 为：

```text
[2, B*Hp, N, 1, D]
```

这正是 Tangram 可以继续使用标准 single-head paged FlashAttention 的关键。

---

## 5.3 为什么 qcrs 默认 NHD 不能直接套这个公式

如果 Ragged physical page 被表示成：

```text
logical:
[B, Hp, N, 2D]
```

但实际 stride 是 NHD：

```text
physical:
[B, N, Hp, 2D]
```

那么：

```text
B 与 Hp 不是相邻连续维度
```

不能假定：

```text
[B, Hp, ...]
→
[B*Hp, ...]
```

是 zero-copy contiguous virtual-block view。

错误后果有两类：

1. reshape 触发 materialization/copy，Ragged hot path 成本被隐藏；
2. 更严重的是 virtual block arithmetic 与真实 stride 不一致。

---

# 6. 推荐的 Ragged Physical Layout 决策

## 6.1 推荐方案：Dedicated Ragged HND-like layout

Ragged cache 单独冻结：

```text
[Bphys, Hp, N, 2D]
```

并要求：

```text
stride(Hp)
= N * 2D
```

从而：

```text
[Bphys, Hp, N, 2D]
→ view
[Bphys * Hp, 1, N, 2D]
```

其中：

```text
virtual_block
= physical_page * Hp + column
```

必须是 zero-copy。

### 为什么不要全局强制 HND

Dense vLLM 的 layout 选择还可能受到：

- KV connector；
- backend；
- transfer layout；
- future features

影响。

所以更合理的是：

> **Ragged cache 使用 dedicated storage contract，不改变 Dense path 的全局 layout 行为。**

---

## 6.2 R1 必须新增 Layout Gate

在任何 real-engine attention 之前先测：

```text
physical.shape
physical.stride()
virtual.shape
virtual.stride()
physical.data_ptr() == virtual.data_ptr()
virtual block address oracle
virtual slot address oracle
```

接受条件：

```text
virtual view is zero-copy
physical page/column→virtual block bijection
slot address exact
```

这个 Gate 非常重要，因为它把一个很隐蔽的 performance/correctness 风险提前变成可测试 contract。

---

# 7. Ragged KV Spec：不能直接复制 Tangram

qcrs 当前 `AttentionSpec` 比 Tangram baseline 新很多，包括：

```text
KVQuantMode
page_size_padded
indexes_kv_by_block_stride
storage_block_size
max_num_blocks_per_req
head_size_v
```

因此推荐：

```python
@dataclass(frozen=True, kw_only=True)
class RaggedAttentionSpec(AttentionSpec):
    page_group_size: int
    head_size_v: int
```

Core scope：

```text
KVQuantMode.NONE
head_size_v == head_size
full attention
TP1 initially
DCP=1
PCP=1
PP=1
no MLA
no prefix cache
no spec decode
no KV connector
```

page bytes：

```math
P_{ragged}
=
2 \cdot B \cdot H_p \cdot D \cdot sizeof(dtype)
```

groups per layer：

```math
G = H_{kv}/H_p
```

identity total memory：

```math
G \cdot P_{ragged}
=
2 \cdot B \cdot H_{kv} \cdot D \cdot sizeof(dtype)
=
P_{dense}
```

因此必须牢记：

> **Ragged layout 本身不节省理论 KV bytes。**

它提供的是：

> **更细粒度、可独立释放的 physical allocation unit。**

真正的 memory reduction 来自：

```text
non-uniform retained lengths
+
independent group page depth
```

---

# 8. Global Page Pool：这是架构亮点，不只是 allocator 修改

Tangram 的 Ragged path 不是：

```text
each layer owns num_blocks pages
```

而是：

```text
one global page namespace
```

Dense planner：

```math
N_{dense}
\approx
available\_memory / page\_size / num\_layers
```

Ragged：

```math
N_{ragged}
\approx
available\_memory / ragged\_page\_size
```

因为每个 physical page 本身已经只代表：

```text
one layer/head-group allocation
```

而不再代表所有 layer 的同深度 block。

Tangram 明确把所有 attention layers 指向同一 raw KV tensor：

```text
KVCacheTensor(
    size = page_size * num_blocks,
    shared_by = all_attention_layers
)
```

各 layer / group 的隔离由：

```text
BlockTable block IDs
```

完成。

---

# 9. qcrs v0.26 上 Global Pool 怎么实现

qcrs 当前 `_allocate_kv_cache()` 已经支持：

```text
one KVCacheTensor
shared_by=[layer0, layer1, ...]
```

从而多个 layer name 可以 alias 同一个 raw tensor。

这给我们的实现提供了非常好的 MRv2-native seam。

但 `_reshape_kv_cache()` 当前会传：

```text
kv_cache_spec.num_kv_heads
```

给 backend 的 `get_kv_cache_shape()`。

Dense 是正确的。

Ragged 不正确，因为 physical page 只含：

```text
Hp = page_group_size
```

所以需要 dedicated Ragged reshape branch：

```text
semantic heads = Hkv
storage heads = Hp
```

这是另一个重要设计原则：

> **不要把 semantic attention geometry 和 physical storage geometry 混在一个字段里。**

建议显式区分：

```text
RaggedGeometry:
  num_kv_heads_semantic
  page_group_size_storage
  groups_per_layer
  num_layers
  total_groups
```

---

# 10. MRv2 RaggedBlockTables：不要复制 Tangram 老实现

qcrs 当前 `BlockTables` 已经有非常好的 serving runtime abstraction：

```text
StagedWriteTensor
UvaBackedTensor
FusedStagedWriter
persistent input_block_tables
persistent slot_mappings
Triton gather
Triton slot mapping
```

尤其：

```text
dummy block tables
dummy slot mappings
```

明确要求 fixed memory address，用于 CUDA Graph capture。

因此 Ragged 应沿着这个体系重建。

---

## 10.1 推荐 internal storage

```text
persistent block table:
[R * G_total, Bmax]
```

逻辑 view：

```text
[R, G_total, Bmax]
```

其中：

```math
flat\_row
=
req\_idx \times G_{total}
+
group\_flat
```

`num_blocks`：

```text
[R, G_total]
```

而不是 Dense：

```text
[group, R]
```

---

## 10.2 为什么 flat rows 适配 MRv2

这样可以直接复用：

```text
StagedWriteTensor.stage_write(row,...)
```

并且 GPU-side gather 仍是：

```text
row gather
```

只不过 row 数从：

```text
R
```

变成：

```text
R * G_total
```

这比复制 Tangram 的 Python/CPU 3D BlockTable 更符合当前 MRv2。

---

# 11. Scheduler canonical ownership：建议比 Tangram 再明确一步

Tangram ragged manager 采用：

```text
req_to_block_ids[req]
=
flat int32 array
```

分配顺序决定如何重新拆回 group rows。

这个实现是高性能且可行的，但在项目第一版里不建议把“flat list + implicit distribution”直接当唯一 canonical semantics。

更清晰的 semantic model 应当是：

```text
request
  ↓
group_flat
  ↓
ordered page row
```

建议第一版 scheduler-side state：

```text
req_to_group_blocks:
    req_id
      → [group0 pages]
      → [group1 pages]
      → ...
```

或者：

```text
flat_ids
+
explicit per_group_counts
```

建议 SchedulerOutput 也显式携带：

```text
RaggedBlockDelta:
    request_id
    target_group_counts
    appended_group_counts
    flat_new_ids
```

这样 worker 不需要通过总长度反推每个 group 拿了多少 page。

等 Core semantics 稳定后，再优化为 Tangram 那种：

```text
np.int32 flat array
+
bulk ring allocator
```

---

# 12. vLLM MRv2 Worker Step 的完整链

当前 qcrs worker 主链可以抽象为：

```text
GPUModelRunner.execute_model
        ↓
extract_compaction_plans
        ↓
update_pp_decode_requests
        ↓
finish_requests
        ↓
free_states
        ↓
add_requests
        ↓
update_requests
        ↓
RequestState staged writes
        ↓
BlockTables staged writes
        ↓
prepare input positions/lengths
        ↓
prepare_attn
```

Attention preparation：

```text
Ragged/Dense persistent block state
        ↓
gather_block_tables
        ↓
compute_slot_mappings
        ↓
build_slot_mappings_by_layer
        ↓
DefaultModelState.prepare_attn
        ↓
build_attn_metadata
        ↓
AttentionMetadataBuilder
        ↓
set_forward_context
        ↓
model.forward
```

这条链是后面所有 debug 的主线。

---

# 13. Logical Position 与 Physical Position

P1 已经完成了非常关键的抽象：

```text
logical:
num_computed_tokens

physical:
effective_kv_len
```

并产生：

```text
positions
cache_positions
```

Dense/P1：

```text
position(t)
=
logical_num_computed + t

cache_position(t)
=
effective_kv_len + t
```

Ragged 后：

```text
position(t)
=
logical_num_computed + t
```

仍然只有一份。

但：

```text
cache_position[g,t]
=
effective_kv_len[g] + t
```

每个 group 都不同。

因此不要扩展 P1：

```text
effective_kv_len: [R]
```

去变成有时 1D、有时 2D 的混合 tensor。

新建：

```text
RaggedKVState
```

例如：

```text
effective_kv_lens_cpu: [R, G_total]
effective_kv_lens_gpu: [R, G_total]
```

---

# 14. Ragged KV Write：这是本项目与 Tangram “直接移植”最大的差异之一

qcrs 当前 FlashAttention：

```text
forward_includes_kv_cache_update = False
```

Attention layer 把：

```text
KV cache update
```

和：

```text
attention compute
```

拆开。

主 seam：

```text
unified_kv_cache_update
        ↓
unified_attention_with_output
```

因此 Ragged 不应该复制旧 Tangram：

```text
ragged_forward
同时 write + read
```

而应该：

```text
Ragged KV Write
        ↓
Ragged Attention Read
```

分别闭环。

---

# 15. R1-D1：Ragged KV Write

输入：

```text
K/V:
[T, Hkv, D]
```

Dense slot mapping：

```text
[T]
```

Ragged 目标：

```text
member-major K/V:
[T * Hkv, 1, D]
```

对于 KV head `h`：

```text
group = h // Hp
column = h % Hp
```

当前 token 在 group 中的 physical position：

```text
p = effective_kv_len[group] + local_query_offset
```

group block row：

```text
page_id = block_table[req, group, p // block_size]
```

virtual block：

```text
vbid = page_id * Hp + column
```

virtual slot：

```text
slot = vbid * block_size + p % block_size
```

最终仍然调用：

```text
reshape_and_cache_flash
```

但它看到的是：

```text
single-head virtual cache
+
member-major K/V
+
member virtual slots
```

这意味着：

> **可以复用 FlashAttention KV write kernel，而不写新的 K/V scatter math kernel。**

这是很强的工程亮点。

---

# 16. R1-D2：Ragged Decode Read

Decode：

```text
q_len = 1
```

对于每个 request：

```text
one request token
× Hkv members
```

Q：

```text
[T, Hq, D]
```

利用 GQA：

```text
q_per_kv = Hq / Hkv
```

reshape：

```text
[T * Hkv, q_per_kv, D]
```

K/V cache：

```text
virtual single-head pages
```

member sequence：

```text
one (request, KV-head)
=
one FA varlen sequence
```

`seqused_k`：

```text
[R * Hkv]
```

block table：

```text
[R * Hkv, Bmax]
```

Tangram 已证明 decode path 可主要依靠 reshape。

我们应当 **先完成 decode synthetic equality**，再进入 prefill。

---

# 17. R1-D3：Prefill / Mixed Batch 是 R1 最大 attention 难点

Prefill 中：

```text
request A: q_len = 128
request B: q_len = 37
request C: decode q_len = 1
```

原始 token-major tensor 的布局：

```text
[token, head, dim]
```

无法通过单一 view 自动变成：

```text
member0所有token
member1所有token
...
```

因为不同 request 的 token span 不同。

所以 Tangram 明确 materialize：

```text
token-major
→ permute
→ contiguous
→ member-major
```

完成 FA 后再 inverse copy：

```text
member-major output
→ token-major output
```

因此 prefill path 的首要目标不是“零 copy”。

首要目标是：

```text
correctness
```

之后 profile：

```text
member-major materialization cost
```

再决定是否优化。

---

# 18. R2：Per-Group Physical State

R2 不做 scorer。

只人为设置：

```text
group0 E=128
group1 E=192
group2 E=96
...
```

然后 decode one token。

逻辑 model position 对所有 group 相同：

```text
position = L
```

physical write：

```text
g0 → 128
g1 → 192
g2 → 96
```

forward 后：

```text
E_new[g] = E_old[g] + q_len
```

核心 acceptance：

```text
logical positions unchanged
group-specific KV writes correct
group-specific reads correct
no ownership/free yet
```

---

# 19. R3：真正决定项目是否成立的系统闭环

R3 不需要 scorer。

输入：

```text
manual keep plan
```

例如：

```text
L0/G0: keep 96
L0/G1: keep 160
L0/G2: keep 64
...
```

执行：

```text
old group pages
        ↓
compact payload
        ↓
new effective length
        ↓
new required page count
        ↓
worker ragged row shrink
        ↓
CompactionResult
        ↓
Scheduler validates source fence
        ↓
canonical ownership trim
        ↓
BlockPool free
        ↓
request B allocate
        ↓
reuse released page
```

这是整个项目最重要的 evidence。

---

# 20. P1 Transaction 为什么可以直接升级为 Ragged

P1 已经建立：

```text
Scheduler produces plan
Worker executes data movement
Worker reports physical result
Scheduler validates result
Scheduler mutates canonical ownership
Scheduler frees pages
```

Ragged 只需要把 scalar transition：

```text
new_effective_kv_len
new_num_blocks
```

升级为：

```text
new_effective_kv_lens[group]
new_num_blocks[group]
```

但 ownership 原则不变：

> **Worker 不成为 allocator authority。**

建议：

```text
RaggedCompactionPlan:
    request_id
    expected_source_group_counts[]
    expected_source_effective_lens[]
    per_group_keep_indices[]
    step_seq

RaggedCompactionResult:
    request_id
    new_group_counts[]
    new_effective_lens[]
    step_seq
```

Scheduler validation：

```text
source version still matches?
source group counts still match?
new counts <= old counts?
ceil(E[g]/B) == new_count[g]?
no duplicate physical page?
```

全部 PASS 后才 commit/free。

---

# 21. TP=2：真正要解决的不是“多卡运行”

TP=2 是一个很值得写简历的 breadth 项。

但项目里需要明确：

> **TP correctness = rank-local attention correctness + rank-consistent physical state transition。**

---

## 21.1 KV head geometry 必须使用 vLLM authoritative API

不能简单假设：

```text
local_Hkv = total_Hkv / TP
```

应该使用：

```python
model_config.get_num_kv_heads(parallel_config)
```

这是 vLLM 对当前 rank 的 authoritative geometry。

然后：

```math
G_{local}
=
H_{kv,local}/H_p
```

启动时必须验证：

```text
Hkv_local % Hp == 0
```

---

## 21.2 什么是 rank-local

每个 TP rank：

```text
拥有自己的 KV head shard
拥有自己的 Ragged physical cache
拥有自己的 score
拥有自己的 keep position
```

因此不同 rank 可以保留不同 token positions。

这是允许的。

---

## 21.3 什么必须 rank-consistent

Scheduler 只有一个 request lifecycle 和 canonical physical allocation contract。

所以在 physical free boundary 上：

```text
每个 rank 最终 group depth/page count
```

必须 compatible。

否则：

```text
rank0 认为第 8 页已经 free
rank1 仍然会 read 第 8 页
```

会产生严重 correctness bug。

---

# 22. TP=2 推荐 Core Contract

第一阶段只支持：

```text
TP = 2
compression_budget_scope = uniform
identity head grouping
```

不要第一版支持：

```text
layer budget
global budget
cross-rank clustering
```

Tangram 也是把 layer/global scope 约束在 TP1，而 TP>1 使用 uniform。

---

## 22.1 `all_reduce(MAX)` 的意义

假设某个对应 group：

```text
rank0 kept_len = 95
rank1 kept_len = 111
```

physical boundary 要保守一致，可以：

```text
all_reduce(MAX)
→ 111
```

然后两 rank 都按照：

```text
ceil(111 / block_size)
```

决定 page depth。

这可能牺牲少量 rank-local memory efficiency，但换来：

```text
same physical detach/free boundary
```

Tangram 当前实现已经使用：

```python
torch.distributed.all_reduce(
    kept_lengths_gpu,
    op=torch.distributed.ReduceOp.MAX,
    group=tp_group.device_group,
)
```

这是我们 TP 设计最直接的 reference。

---

# 23. TP 的 debug / acceptance 设计

Debug build 可以在 free 前：

```text
all_gather:
  new_group_counts
  detached page count
  row checksum
```

强制：

```text
same scheduler-visible shape transition
```

最终 acceptance：

```text
TP2 Dense vs Ragged Identity output parity
TP2 compressed correctness
rank-local FA correctness
rank-consistent group depth
scheduler free count correct
request B reuses released block IDs
request A continues decode
```

---

# 24. 为什么 TP=2 值得写简历

因为它证明的不是“会用 torchrun”。

它证明：

```text
Tensor Parallel head sharding
+
rank-local KV geometry
+
distributed retention
+
collective synchronization
+
allocator ownership invariant
```

可以写成：

> **扩展 Ragged KV Runtime 至 TP=2，基于 rank-local KV head geometry 构建 head-group page layout，并通过 TP collective 对齐压缩后的 physical free boundary，保证跨 rank page ownership 与 scheduler reclamation 一致。**

只有真实 TP2 evidence 完成后才使用这条。

---

# 25. CUDA Graph：本项目最合适的是 Piecewise，不是 Full

vLLM 0.26 当前已经存在：

```text
CUDAGraphMode.NONE
PIECEWISE
FULL
FULL_DECODE_ONLY
FULL_AND_PIECEWISE
```

并且 compilation config 默认把：

```text
vllm::unified_attention_with_output
```

放入 splitting ops。

更关键的是当前 v0.26 还会把：

```text
vllm::unified_kv_cache_update
```

排除出普通 compiled graph / CUDA graph region。

这恰好与 Ragged 所需动态行为匹配。

---

# 26. 不建议第一版复制 Tangram `unified_attention_ragged`

Tangram fork 创建独立：

```text
vllm::unified_attention_ragged
```

作为 piecewise splitting point。

但 qcrs/v0.26 已经有：

```text
unified_kv_cache_update
unified_attention_with_output
```

两个天然 boundary。

所以建议第一版：

```text
Attention.forward
        ↓
existing unified_kv_cache_update
        ↓
dispatch dense/ragged KV write

existing unified_attention_with_output
        ↓
dispatch dense/ragged attention read
```

优点：

```text
不增加 compile surface
不新增 custom-op registration complexity
不需要修改 splitting_ops model
保持 Dense path 不动
```

---

# 27. Ragged + Piecewise CUDA Graph 的正确结构

推荐：

```text
Captured Piece A
  QKV projection
  RoPE / model compute
        ↓
Eager Island
  Ragged KV update
        ↓
Eager Island
  Ragged metadata-dependent attention
        ↓
Captured Piece B
  projection / FFN / later model compute
```

真正需要 capture-stable 的不是 Ragged metadata 内容。

而是：

```text
graph-facing tensor address
shape class
buffer pointer
```

---

# 28. CUDA Graph 最大难点：dynamic metadata allocation

危险代码包括：

```text
torch.empty() every step
new Tensor from Python list
new per-layer block table
new query_start_loc
new seq_lens overlay
new dataclass carrying fresh tensors
GPU→CPU .item()
```

这些会导致：

```text
capture-time address != replay-time address
```

或者 CPU overhead 抵消 CG 收益。

Tangram 因此直接拒绝 Ragged FULL graph，而保留 Piecewise。

---

# 29. R9 推荐的 Persistent RaggedStepBuffers

建议定义：

```text
RaggedStepBuffers
```

持有：

```text
persistent_member_block_tables
persistent_virtual_slot_mapping
persistent_member_seq_lens
persistent_member_query_start_loc
persistent_group_effective_lens
persistent_decode_overlays
```

CPU → GPU metadata：

```text
pinned memory
or
UVA-backed buffer
```

每步只更新内容，不换地址。

---

# 30. CUDA Graph acceptance 不只是“没有 crash”

要测：

```text
A. Dense + Piecewise CG
B. Ragged eager
C. Ragged + Piecewise CG
```

关注：

```text
TPOT
decode token/s
CPU step time
CUDA launch gaps
attention eager-island time
metadata build time
```

如果：

```text
Ragged eager → Ragged Piecewise
```

显著降低 CPU launch gap / TPOT，

才形成简历亮点。

可写：

> **将动态 Ragged KV 路径接入 vLLM piecewise torch.compile/CUDA Graph，通过复用现有 attention/KV-update splitting seam，并将动态 metadata 隔离在 eager island、固定 replay-facing buffer 地址，降低 decode launch overhead。**

不能只因为“能开 cudagraph”就写性能优化。

---

# 31. Preemption：v3 需要一个 MRv2-specific 修正

Tangram 老路径存在：

```text
snapshot_row
restore_row
```

因为 compressed ragged row：

```text
group depth non-uniform
```

无法通过普通 flat block list 自动重建。

但 qcrs MRv2 当前：

```text
RequestState
BlockTables
```

本身按：

```text
req_idx
```

做 persistent state。

如果 request 只是：

```text
本 step 没被调度
```

但没有被 preempt/free，

其 persistent RaggedBlockTables 根本没有消失。

所以：

> **普通 skipped step 不需要 snapshot/restore。**

---

# 32. 真正要处理的是 True Scheduler Preemption

当前 worker 会把：

```text
preempted_req_ids
```

合并进 remove path。

真正的 preemption 语义：

```text
scheduler frees KV
request goes back waiting
worker removes request runtime state
later recompute
```

因此 preemption hook 需要 reset：

```text
RaggedBlockTables row
RaggedKVState E[r,g]
compression req state
slot-score state
pending compaction plan
```

resume 时：

```text
scheduled_new_req
→ new req_idx / state
→ overwrite canonical row
→ recompute
```

而不是恢复旧 KV row。

这个实现比直接 port Tangram snapshot 更符合当前 MRv2。

---

# 33. Preemption 为什么值得写简历

Preemption 是系统项目非常好的 correctness evidence，因为它证明：

```text
KV ownership
≠
worker cached metadata
≠
logical request state
```

你必须真正理解三层 lifecycle：

```text
logical request
physical pages
compression state
```

简历可以描述：

> **完善 Ragged KV request lifecycle，区分 transient unscheduled 与 scheduler preemption/recompute，避免恢复 stale per-group physical state，并在 KV-pressure workload 下验证释放、重算和继续解码。**

---

# 34. AOT Head Clustering：最推荐的 Optimization

如果 Core 完成后只选一个“额外优化”，优先推荐 AOT Clustering。

原因：

```text
不是另一个 scorer
不是无意义 kernel micro-opt
直接解决 Ragged grouping fragmentation
与项目主线强相关
CPU/offline部分实现难度可控
效果可以清楚量化
```

---

# 35. AOT Clustering 解决什么

一个 page group 内包含 Hp 个 members。

实际 page count 由：

```math
pages(group)
=
ceil(max(member kept length)/B)
```

决定。

如果：

```text
head A keep 90%
head B keep 30%
```

被分在同一 group，

B 仍然跟着 A 分配接近 90% depth。

这是：

```text
max-pool over-allocation
```

Tangram 的解决办法：

```text
page_group_size=1 profile
        ↓
per-(layer,head) retention profile
        ↓
rank/stability
        ↓
cluster similar heads
        ↓
new (cluster,column) map
```

---

# 36. 我们推荐先做 per-layer clustering

Tangram 支持 cross-layer/global clustering。

但对简历项目第一版不推荐。

因为 cross-layer clustering 会重新打开：

```text
layer ownership
physical placement
metadata lookup
TP shard mapping
```

这些问题。

推荐：

```text
R7-A per-layer clustering
= normal target

R7-B cross-layer clustering
= stretch
```

---

# 37. AOT 要测什么

必须同时报告：

```text
Per-head Ideal
Adjacent Grouping
AOT Clustered Grouping
```

指标：

```text
allocated pages
residual fragmentation
KV bytes
max concurrency
throughput
```

Tangram README 中 42.7% → 2.9% 是 Tangram 自己的 reference result。

项目不能直接当成自己的数字。

---

# 38. Triton In-place Writeback：要不要做，取决于 profiler

P1 当前已经有非常好的：

```text
PyTorch reference
→ Triton Gather
→ scratch
→ Triton Writeback
```

这个路径极适合 R3/R4 correctness。

不要为了“用了 Triton”过早重写。

---

# 39. Tangram In-place Writeback 的核心安全性

Tangram 的关键条件：

对于排序后的 retained source：

```math
s_j \ge j
```

所以 destination 永远不会位于 source 之后。

source 分两类：

```text
source >= kept
→ 不可能被 destination overwrite
→ parallel

source < kept
→ endangered
→ sequential read-before-write prefix
```

最后：

```text
zero-pad last page
```

必须发生在所有可能 source read 之后。

---

# 40. Triton Optimization 的正确 evidence

比较：

```text
Reference scratch path
vs
In-place path
```

指标：

```text
temporary scratch bytes
DRAM bytes estimate
compaction latency
GB/s
end-to-end compression boundary overhead
TPOT impact
```

如果 compaction 只占总时间：

```text
< 1%
```

那么不应该宣称：

> “显著提升整体吞吐”

应该把它作为：

> **Triton kernel / memory movement engineering demo**

---

# 41. Bulk Allocator / Ring Buffer：另一个可选系统亮点

Ragged 把一个 Dense page 拆成很多 smaller pages。

page ID 数量会明显增加。

Tangram 因此把普通：

```text
KVCacheBlock Python objects
+
linked free queue
```

优化成：

```text
np.int32 ring buffer
+
bulk allocate/free
```

并尽量避免：

```text
KVCacheBlock materialization
```

这是非常合理的系统优化。

---

# 42. 但 Bulk Allocator 不应成为 R1 blocker

推荐顺序：

```text
first:
semantic correctness using existing allocator abstraction

then profile:
CPU alloc/free overhead
Python object/GC overhead

if material:
port vectorized int32 path
```

如果最终测出收益，可写：

> **针对 head-group paging 导致的高 page-count allocator pressure，引入 int32 bulk allocation/free path，降低 Python object 与 free-queue 管理开销。**

---

# 43. Continuous Batching 是必须隐式通过的系统测试

Ragged 不是单请求离线 algorithm。

因此 Identity 和 Reclaim 必须至少验证：

```text
request A prefill
request B arrives
A decode + B prefill mixed
A compressed
B continues
C arrives
freed pages reused
```

这证明：

```text
ragged metadata
block-table gather
member-major attention
physical ownership
```

没有偷偷依赖 static batch。

---

# 44. Chunked Prefill 是 Prefill correctness 的关键 workload

因为 chunked prefill 会制造：

```text
same request
multiple forward boundaries
```

同时与 decode request 混合。

这会 stress：

```text
query_start_loc
member-major packing
effective length
same-step append
compression boundary
```

R1-E 至少应包含一个 chunked-prefill case。

---

# 45. 不值得现在打开的方向

以下内容技术上有意义，但会明显拉大项目而不提高简历 ROI：

```text
PP > 1
DCP / PCP
Spec Decode
Prefix Cache
KV Connector
LMCache
PD Disaggregation
MLA
Quantized KV
Cross-layer AOT
TP layer/global budget
Full CUDA Graph
```

这些应保持：

```text
explicit unsupported
```

而不是“以后再说”的隐式状态。

---

# 46. 难点排序

| 难点 | 难度 | 为什么难 | 首选策略 | Fallback |
|---|---:|---|---|---|
| Physical layout / stride | 9/10 | zero-copy virtual block 必须与真实 storage 一致 | dedicated Ragged HND contract | explicit copy oracle |
| Global pool ownership | 9/10 | page ID 不再隐含 layer | single namespace + explicit group rows | semantic manager first |
| Prefill member-major | 9/10 | variable q_len 无法纯 view | Tangram materialization | decode-only debug |
| Scheduler per-group ownership | 9/10 | Dense canonical row 假设被打破 | explicit group counts | flat ids + counts |
| TP free-boundary | 8/10 | ranks keep positions不同但 free必须一致 | all_reduce MAX | TP1 |
| CUDA Graph metadata | 8/10 | dynamic address / CPU overhead | Piecewise + persistent buffers | eager |
| KV write seam | 8/10 | MRv2 与 Tangram旧路径不同 | existing unified_kv_cache_update | dedicated helper |
| Compaction correctness | 7/10 | in-place overwrite / state commit | P1 scratch oracle | Torch only |
| Preemption | 7/10 | stale physical/compression state | reset on true preempt | Core without pressure |
| AOT clustering | 5/10 | profile/map validation | per-layer first | adjacent grouping |
| Bulk allocator | 5/10 | CPU bookkeeping | profiler-driven | existing BlockPool |

---

# 47. 最值得写进简历的亮点优先级

## S 级：必须做

### 1. MRv2-native Ragged KV Representation

```text
Head-group page
global page pool
virtual single-head view
```

证明：

```text
懂 PagedAttention physical representation
```

### 2. Non-uniform Physical Reclamation

```text
per-group compaction
→ real BlockPool free
→ cross-request reuse
```

证明：

```text
不是逻辑 mask
而是真正 memory capacity
```

### 3. Scheduler / Worker Transaction

```text
worker transforms payload
scheduler owns canonical pages
```

证明：

```text
serving correctness / state management
```

---

# 48. A 级：强烈推荐

### TP=2

证明：

```text
distributed inference / rank consistency
```

### Piecewise CUDA Graph

证明：

```text
compiler/runtime integration
```

### Preemption

证明：

```text
serving lifecycle correctness
```

---

# 49. B 级：选一到两个，不要全堆

推荐顺序：

```text
AOT Clustering
> Triton In-place if profiling supports
> Bulk Allocator if profiling supports
```

不要为了数量把所有 Tangram optimization 都复制一次。

---

# 50. 最佳 Resume Completion Boundary

我建议最终目标冻结成：

```text
MUST
R1 Identity
R2 Per-group State
R3 Physical Reclaim
R4 One Policy

STRONGLY RECOMMENDED
Preemption
TP=2 Uniform
Piecewise CUDA Graph

ONE OPTIMIZATION
AOT Clustering

PROFILE-DRIVEN
Triton In-place
Bulk Allocator
```

这是风险与简历价值最平衡的版本。

---

# 51. Benchmark Matrix

最终 benchmark 必须是分层记账。

| ID | Runtime | Compression | TP | CG | 目的 |
|---|---|---|---:|---|---|
| B0 | Dense v0.26 | off | 1 | Piecewise/Eager | canonical baseline |
| B1 | Ragged Identity | off | 1 | eager | representation overhead |
| B2 | Ragged Identity | off | 1 | piecewise | CG integration |
| B3 | Ragged | manual nonuniform | 1 | eager | physical capacity proof |
| B4 | Ragged | one scorer | 1 | eager | automatic policy |
| B5 | Ragged | one scorer | 1 | piecewise | serving optimized |
| B6 | Ragged | uniform | 2 | eager | TP correctness |
| B7 | Ragged | uniform | 2 | piecewise | distributed serving |
| B8 | Ragged | one scorer+AOT | 1 | piecewise | fragmentation optimization |

---

# 52. 必须记录的 Metrics

Memory：

```text
total physical pages
free pages
allocated pages/request
KV bytes
residual group fragmentation
max concurrency
```

Performance：

```text
TTFT
TPOT
request throughput
token throughput
CPU step time
metadata build time
compaction time
```

Correctness：

```text
dense/ragged identity output parity
reference/optimized compaction equality
free/reuse exact block IDs
continuation after reclaim
TP rank consistency
preemption recompute correctness
```

Quality：

```text
小规模 RULER / long-context task
```

Core 不需要复现整篇论文 benchmark。

---

# 53. Resume Bullet：完成 R3 后

可以写：

> 基于 vLLM 0.26 Model Runner V2 实现 Head-Group Ragged KV Cache Runtime，将 KV page 从全 head block 细化为 per-head-group allocation，并通过 virtual block addressing 复用 FlashAttention paged kernel。

> 建立 per-group physical KV state 与 scheduler-authoritative reclamation transaction，实现 token-level compaction 后真实 GPU page 回收和跨请求 block reuse，并验证原请求持续 decode 正确性。

---

# 54. Resume Bullet：完成 R4 后

再增加：

> 将 importance policy 与物理 KV runtime 解耦，引入单一 deterministic scorer 驱动非均匀 retention，完成 decision → compaction → allocator capacity 的完整闭环。

不要写：

> “提出新的 KV importance algorithm”

因为项目价值不在这里。

---

# 55. Resume Bullet：完成 TP/CG/AOT 后

只有对应 evidence 完成后才写：

> 扩展至 TP=2，通过 rank-local head-group geometry 与 collective 同步压缩后 physical free boundary，保证跨 rank page ownership 一致。

> 将动态 Ragged KV 路径接入 vLLM piecewise torch.compile/CUDA Graph，复用 KV-update / attention splitting seam，并通过 persistent metadata buffers 降低 decode CPU launch overhead。

> 基于 offline retention profile 实现 AOT head clustering，降低 head-group max-pool fragmentation，并量化其对 physical pages、最大并发与 serving throughput 的影响。

---

# 56. 90 秒面试叙事

可以按以下顺序讲：

> vLLM 的 PagedAttention 虽然解决了 sequence-level KV fragmentation，但一个 block 仍然绑定一个 layer 内全部 KV heads，因此如果压缩策略想让不同 head 保留不同长度，逻辑上可以删，但物理 page 无法独立释放。

> 我参考 Tangram 的 Head-Group Page，把一个 dense page 拆成只保存 `page_group_size` 个 KV heads 的更小 page，然后给每个 request、layer、head-group 独立 block row。为了不改 FlashAttention kernel，我把 `(physical_page, column)` 映射成 virtual single-head block。

> 但我的实现基于 vLLM 0.26 Model Runner V2，和 Tangram fork 已经不同。v0.26 把 KV cache update 和 attention read 拆成两个 custom-op seam，所以我分别重建了 Ragged KV write 和 member-major attention read，并用 MRv2 的 persistent staged buffers 实现 block table。

> 压缩后 Worker 只负责 KV payload transformation，Scheduler 仍然是 canonical allocator owner。Worker 返回新的 per-group physical shape，Scheduler 校验 source fence 后再 trim/free block，最后验证 block 被另一个 request 重用且原 request 能继续 decode。

> 后续我还把它扩展到 TP=2 和 piecewise CUDA Graph，解决 rank-local retention 与 physical free boundary 一致性，以及动态 Ragged metadata 与 CUDA Graph fixed-address requirement 的冲突。

---

# 57. 面试高频问题

### Q1：Ragged paging 本身为什么不省内存？

因为：

```math
G \times P_{ragged} = P_{dense}
```

它只改变 allocation granularity。

### Q2：真正省内存发生在哪里？

不同 group 可以持有不同 page depth，压缩后独立 free tail pages。

### Q3：为什么不改 FlashAttention kernel？

通过 virtual single-head block 把 `(page,column)` 映射成标准 paged single-head KV。

### Q4：为什么逻辑 position 不能跟着 compact？

RoPE/model token position 是模型语义；只改变 physical cache coordinate。

### Q5：为什么 Worker 不能直接 free block？

Scheduler 是 canonical ownership authority；否则 scheduler/worker 可能出现 reuse-after-free / stale ownership。

### Q6：为什么 TP 要 MAX reduce kept length？

各 rank 头不同，可以保留不同 positions，但 scheduler-visible physical depth/free boundary 必须兼容。

### Q7：为什么只做 Piecewise CUDA Graph？

Ragged metadata 每 step 动态变化；Full graph 的固定地址要求成本高、ROI 低。Piecewise 能保留大部分模型 compute capture，同时把动态 attention metadata 放 eager island。

### Q8：Prefill 为什么比 decode 难？

decode q_len=1，member-major 可以主要 reshape；prefill 不同 request query length 不同，需要 materialize varlen member-major layout。

### Q9：AOT clustering 优化什么？

不是 score quality，而是 head grouping 造成的 max-member page over-allocation。

### Q10：为什么项目不用很多 scorer？

因为项目目标是 physical runtime；scorer 只是产生 keep plan 的 control plane。

---

# 58. Canonical Execution Order v4

```text
R0
freeze pins / baseline

R1-A1
RaggedAttentionSpec

R1-A2
Global Page Pool

R1-A3   [NEW EXPLICIT GATE]
Physical Layout / Stride Contract

R1-B
MRv2 RaggedBlockTables

R1-C
Virtual Block / Slot Oracle

R1-D1
Ragged KV Write

R1-D2
Decode Attention Read

R1-D3
Prefill / Mixed Read

R1-E
Real Engine Identity

R1-F
Piecewise CUDA Graph Compatibility Smoke

R2
Per-group E[r,g]

R3
Manual Non-uniform Reclaim + Free + Reuse

R4
One scorer + automatic policy

R5
True Preemption / Recompute

R6
TP=2 Uniform

R7
AOT Per-layer Clustering

R8
Triton In-place [only if profiler]

R9
Piecewise CG Hardening / Persistent Metadata

R10
Final benchmark / resume closure
```

这里相对 v3 最重要的变化：

```text
Layout / Stride Contract
```

被提升成独立 gate；

并且：

```text
Preemption
```

按 MRv2 语义重新定义，不再把 ordinary skipped step snapshot 当核心要求。

---

# 59. Go / No-Go Gates

## Gate 1 — Spec

必须：

```text
page bytes正确
identity memory等价
unsupported明确reject
```

## Gate 2 — Global Pool

必须：

```text
one physical namespace
capacity accounting正确
all layers alias intended backing
```

## Gate 3 — Layout

必须：

```text
zero-copy virtual view
stride oracle通过
```

## Gate 4 — RaggedBlockTables

必须：

```text
group rows独立
staged write正确
persistent address
```

## Gate 5 — KV Write

必须：

```text
member virtual slot写入正确
```

## Gate 6 — Attention

必须：

```text
decode synthetic equality
prefill synthetic equality
```

## Gate 7 — Identity Engine

必须：

```text
dense/ragged real model parity
continuous batch
chunked prefill
```

## Gate 8 — Physical Reclaim

必须：

```text
payload correct
row shrink
BlockPool free↑
reuse
continue decode
```

只有 Gate 8 通过，项目的核心 systems claim 才成立。

---

# 60. 推荐源码阅读顺序

## Stage 1 — Geometry / Allocation

qcrs：

```text
vllm/v1/kv_cache_interface.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/worker/gpu/attn_utils.py
```

Tangram：

```text
vllm/v1/kv_cache_interface.py
vllm/v1/core/kv_cache_utils.py
```

## Stage 2 — Ownership

qcrs：

```text
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/block_pool.py
vllm/v1/core/sched/scheduler.py
```

Tangram：

```text
vllm/v1/core/block_pool.py
vllm/v1/core/single_type_kv_cache_manager.py
```

## Stage 3 — Worker Physical Mapping

qcrs：

```text
vllm/v1/worker/gpu/block_table.py
vllm/v1/worker/gpu/input_batch.py
vllm/v1/worker/gpu/states.py
vllm/v1/worker/gpu/model_runner.py
```

Tangram：

```text
vllm/v1/worker/ragged_block_table.py
vllm/v1/attention/backends/ragged_layout.py
```

## Stage 4 — Attention

qcrs：

```text
vllm/model_executor/layers/attention/attention.py
vllm/v1/attention/backends/flash_attn.py
vllm/v1/worker/gpu/model_states/default.py
vllm/v1/worker/gpu/attn_utils.py
```

Tangram：

```text
vllm/v1/attention/backends/ragged_forward.py
vllm/v1/attention/backends/flash_attn.py
```

## Stage 5 — Compression

qcrs：

```text
vllm/v1/worker/gpu/kv_compaction.py
scheduler P1 reconciliation
```

Tangram：

```text
vllm/v1/attention/compression/executor.py
vllm/v1/attention/compression/eviction_writeback.py
```

## Stage 6 — TP / CG

qcrs：

```text
vllm/config/model.py
vllm/config/compilation.py
vllm/v1/worker/gpu/cudagraph_utils.py
vllm/v1/attention/backends/flash_attn.py
```

Tangram：

```text
vllm/v1/worker/compression_model_runner_mixin.py
vllm/config/cache.py
```

---

# 61. Pinned Source Index

## qcrs/vLLM

Branch:

https://github.com/qcrs/vllm/tree/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00

KV spec:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/kv_cache_interface.py

KV cache planner:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/core/kv_cache_utils.py

Scheduler:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/core/sched/scheduler.py

KV manager:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/core/kv_cache_manager.py

Single-type manager:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/core/single_type_kv_cache_manager.py

MRv2 BlockTables:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/worker/gpu/block_table.py

RequestState:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/worker/gpu/states.py

GPUModelRunner:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/worker/gpu/model_runner.py

Attention util:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/worker/gpu/attn_utils.py

Default model state:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/worker/gpu/model_states/default.py

Attention layer:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/model_executor/layers/attention/attention.py

FA2:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/attention/backends/flash_attn.py

Compilation:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/config/compilation.py

CUDA Graph:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/worker/gpu/cudagraph_utils.py

P1 compaction oracle:

https://github.com/qcrs/vllm/blob/bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00/vllm/v1/worker/gpu/kv_compaction.py

## Tangram

Pinned tree:

https://github.com/aiha-lab/tangram/tree/6fa551fc8f6edcc118a2a39b3554ee1520e91edd

Ragged spec:

https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/kv_cache_interface.py

Global pool planner:

https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/core/kv_cache_utils.py

Ragged manager:

https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/core/single_type_kv_cache_manager.py

Ragged BlockTable:

https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/worker/ragged_block_table.py

Ragged layout:

https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/attention/backends/ragged_layout.py

Ragged forward:

https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/attention/backends/ragged_forward.py

Compression runner:

https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/worker/compression_model_runner_mixin.py

Writeback:

https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/vllm/v1/attention/compression/eviction_writeback.py

AOT clustering:

https://github.com/aiha-lab/tangram/blob/6fa551fc8f6edcc118a2a39b3554ee1520e91edd/tools/head_group_clustering/README.md

---

# 62. Final Recommendation

这个项目后续不要再继续扩展横向 topic。

真正最有价值的是把下面四条 evidence 做硬：

```text
1.
Ragged Identity
真的能在 v0.26 MRv2 跑

2.
Non-uniform group lengths
真的能改变 physical page ownership

3.
released pages
真的进入 BlockPool 并被其他 request reuse

4.
系统 breadth:
TP2 / Preemption / Piecewise CG
至少各有一个真实 closure
```

最终最强的项目故事不是：

> “我实现了 Tangram 的很多功能。”

而是：

> **“我基于 Tangram 的 Ragged Paging 思想，在新版 vLLM Model Runner V2 上重新设计了非均匀 KV physical runtime，并把它和 allocator ownership、FlashAttention、TP、CUDA Graph、Triton compaction 真实打通。”**

这才是项目与简单 paper reproduction 的本质区别。
