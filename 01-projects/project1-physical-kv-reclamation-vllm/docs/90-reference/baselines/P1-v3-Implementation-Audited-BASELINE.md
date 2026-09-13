# Project 1 — Physical KV Cache Reclamation for vLLM
## Decoupled Logical/Physical KV Positions + Real Block Reuse + Triton KV Compaction

**定位：主项目 / LLM Inference Runtime / KV Cache Memory Management / Triton / Profiling**

**建议基线**
- vLLM: `v0.26.0`
- pinned commit: `568afb3a13806beb53bb2e6bd518269357b237c0`
- GPU: NVIDIA A100 80GB
- 初始约束：单机单卡，TP=1 / PP=1 / DP=1，dense full attention，BF16 KV
- MVP 关闭：prefix caching、speculative decoding、async scheduling、full CUDA Graph
- MVP 额外约束：`num_kv_cache_groups == 1`、dense full attention、reclaim 后先只支持 single-token decode (`q_len=1`)
- 主模型：先使用小模型完成 correctness/runtime bring-up，再切长上下文模型做容量与延迟实验
- **Reference portability 注意**：Tangram audited commit 使用 `torch==2.9.0`，而本项目 pinned vLLM v0.26 使用 `torch==2.11.0`；Tangram 只作为 source-level design/reference，不直接 cherry-pick 整体 patch。

---

# 0. 一句话定义

这个项目不是“做一个 KV eviction policy”。

它要解决的是：

> **当一部分历史 KV 已经决定不再参与后续 attention 时，如何让 vLLM 不只是“逻辑上忽略它”，而是真正修改 KV page ownership、压缩有效 KV 地址空间、把物理 block 归还给 BlockPool，并让请求在 logical token position 不变的情况下继续 decode。**

完整目标：

```text
Long-context request
        │
        ▼
Paged KV grows
        │
        ▼
Retention / Reclamation Decision
        │
        ├── V1: block-aligned zero-copy reclamation
        │
        └── V2: token-granular Triton compaction
        │
        ▼
physical KV occupancy ↓
        │
        ▼
freed KV blocks
        │
        ▼
BlockPool
        │
        ▼
other requests can really reuse them
```

项目的核心成功条件不是“模型质量更好”或者“TPOT 一定提升”。

核心成功条件是：

```text
retained KV < original KV
        ↓
request-owned physical blocks ↓
        ↓
BlockPool free blocks ↑
        ↓
freed blocks are safely reused
        ↓
original request continues decode correctly
```

这是一个**确定性的 runtime capability**。

---

# 0.5 Pre-Implementation Certainty Audit：先验确定性审计

这部分专门回答一个问题：

> **开始编码以后，会不会才发现某个关键环节其实没有参考，需要临时发明一套新的、不确定的系统设计？**

审计结论不是“所有代码都有现成 copy”。

更准确的结论是：

> **P1 的必要 runtime contracts 已经有 source-level reference；存在两个没有 exact drop-in implementation 的点，但都不是项目成立所依赖的未知机制，并且都有明确 fallback。**

---

## 0.5.1 Reference 等级

本文后续统一使用四级标记：

| 等级 | 含义 | 是否允许进入 Core |
|---|---|---|
| **A — Exact Reference** | 公开仓库存在相同或几乎相同的 runtime contract / implementation | 可以 |
| **B — Reference-backed Adaptation** | 机制已被参考项目证明，我们裁剪/迁移到 vLLM v0.26 | 可以 |
| **C — Mechanically Specified Own Implementation** | 没有 exact drop-in，但输入/输出/layout/oracle 都已有 reference，不需要发明算法 | 可以，但必须有 fallback |
| **D — Experimental / Unknown Benefit** | 需要 profiler 或 benchmark 才知道是否值得，或需要新的较大设计 | **不能作为项目成功条件** |

---

## 0.5.2 P1 必要链路逐项审计

| 必要能力 | 等级 | 具体 reference | 是否需要我们发明新方案 |
|---|---|---|---|
| Request → physical block ownership | A | vLLM `SingleTypeKVCacheManager.req_to_blocks` | 否 |
| block allocation/free | A | vLLM `BlockPool`, `free_blocks()` | 否 |
| full-attention 历史 block 的主动释放 | A/B | vLLM `remove_skipped_blocks()` + Tangram compression free path | 否 |
| 压缩后 allocation 按 effective occupancy 计算 | **A** | Tangram `KVCacheManager.allocate_slots(... effective_num_cached_tokens=...)` | 否 |
| logical position 与 physical slot position 分离 | **A/B** | Tangram `GPUModelRunner`: logical `positions_np` 保持不变，而 `slot_positions = effective_seq_lens_cpu + arange` | 否 |
| attention 使用压缩后的有效 KV 长度 | A/B | Tangram `effective_seq_lens_cpu` → attention metadata | 否 |
| worker 完成变更后再通知 scheduler free | **A** | Tangram `ModelRunnerOutput.compression_*` → Scheduler → `free_blocks_by_ids()` | 否 |
| request block-table 行变更 | A/B | Tangram ragged BlockTable compaction/snapshot；upstream `add_row/append_row` | 否，uniform 2D 版本需裁剪 |
| zero-copy whole-block middle reorder | **B** | pinned vLLM `BlockTable.add_row()` 已提供 arbitrary row reconstruction；Tangram 提供 reclaim protocol | 需要新增 reclaim-specific transition，但不需要新 BlockTable architecture |
| token-level PyTorch compaction | A/B | Tangram `CompressionExecutor` | 否 |
| Triton paged-KV source gather | **C** | Tangram 给 compaction semantic；vLLM 给 paged addressing；未找到 exact gather kernel | 需要我们写 kernel |
| Triton destination paged write | A/B | vLLM `triton_reshape_and_cache_flash.py` | 否/轻度适配 |
| invalid-input atomicity 测试 | A/B | Sparse-vLLM `test_static_eviction_compaction.py` | 否 |
| per-head ragged paging | A | Tangram | 有 reference，但不进入 Core |
| async/spec/CUDA Graph compatibility | D | Tangram/Sparse-vLLM 有更复杂实现 | 不进入 Core |

这一项在 Round-3 再审计后需要进一步修正：

- pinned vLLM 本身已经提供 `BlockTable.add_row()` 做 arbitrary full-row reconstruction；
- `GPUInputBatch.add_request()` 正在使用它；
- ModelRunner 也已有 preemption-resume 的 full block-list replacement semantic。

因此 **whole-block zero-copy reorder 不再算一个缺失 primitive**，而是一个基于现成 row-rebuild primitive 的 reclaim-specific state transition。

当前 Core 中真正没有 exact drop-in kernel 的主要实现点只剩：

1. 我们自己的 **arbitrary keep-index Triton paged-KV gather kernel**。

它也不是“研究一个新算法是否成立”.

它们都有完全确定的：

```text
input
output
memory layout
correctness oracle
fallback
```

---

## 0.5.3 Zero-copy V1：Round-3 后已从“reference gap”降级为 reference-backed adaptation

我们偏好的 V1：

```text
[B0 B1 B2 B3 B4 B5]
        ↓ keep [0,1,4,5]
[B0 B1 B4 B5]
```

surviving physical block 本身不移动，只改 block-table active row。

Round-3 对 pinned vLLM 重新核实后，这一部分比此前判断更确定：

- `BlockTable.add_row()` 已经可以用任意 physical block-id list 重建 active row；
- `GPUInputBatch.add_request()` 正在正常 runtime 中使用这个 primitive；
- preemption resume 已经存在 full block-list replacement。

因此我们不需要新发明 row-replacement mechanism。

更准确的等级是：

> **B：reference-backed reclaim transition。**

我们需要实现的是“reclaim 何时触发、effective length 如何一起 commit、worker ACK 后何时 free”，而不是重新设计 BlockTable。

但它不会把项目变成未知设计，因为：

```text
keep block indices 必须排序
→ surviving tokens 的 attention 顺序明确

block table 是 indirection
→ physical block ID 不要求连续

prefix caching 关闭
→ 没有 shared/ref-counted prefix page ownership

reclaim 只发生在 full prefill 完成之后
→ 不存在后续 prompt chunk 继续写旧 logical slot

q_len=1 decode
→ attention 只读取 retained physical sequence
```

### Hard fallback

如果在真实 FlashAttention / BlockTable integration 中发现 zero-copy reorder 存在隐藏语义：

直接退回 **V1-R reference-exact fallback**：

```text
whole-block keep set
↓
Tangram-style gather/writeback
↓
surviving KV 写成 physical prefix
↓
trim trailing pages
↓
free trailing block IDs
```

它多一次 copy，但 system contract 与 Tangram 已验证路径一致。

因此：

> **zero-copy 是 preferred engineering optimization，不是项目能否做成的单点风险。**

---

## 0.5.4 Triton V2 的 reference gap 与 fallback

我们没有找到一颗可以直接拿来的：

```text
triton_gather_paged_kv(keep_indices)
```

这点必须提前说清楚。

已有 reference 分成两半：

### Tangram 提供 semantic oracle

```text
source paged KV
+ keep indices
→ retained KV
→ compact writeback
```

### vLLM 提供 address oracle

`triton_reshape_and_cache_flash.py` 已经明确：

```text
slot
→ block_idx = slot // block_size
→ block_offset = slot % block_size
→ paged KV address
```

所以我们自己写的 gather 只是在完成：

```text
keep_idx
→ source block/offset
→ tl.load(K/V head vector)
→ contiguous scratch
```

这不是未定义问题。

### Fallback

如果第一版 Triton 性能没有赢：

```text
correct Triton kernel
+
PyTorch oracle
+
NCU/NSYS 分析
```

仍然完成 V2 的 GPU-infra 目标。

项目**不要求 Triton kernel 一定带来 E2E speedup**。

---

## 0.5.5 最危险的 hidden bug：free timing

旧方案如果这样做：

```text
Scheduler 先把 block free
↓
另一个 request 立即复用
↓
Worker 旧 BlockTable 仍指向该 block
```

会产生真实 alias/corruption 风险。

因此 Core 设计必须改为 Tangram 已验证的顺序：

```text
last-prefill forward 完成
        ↓
Worker 先更新 physical view
  - zero-copy row replacement
  或
  - compaction/writeback
        ↓
Worker 产出:
  new effective KV len
  freed block IDs
        ↓
ModelRunnerOutput
        ↓
Scheduler 收到 ACK
        ↓
KVCacheManager 修改 canonical ownership
        ↓
BlockPool.free_blocks(...)
        ↓
下一轮 scheduler 才允许复用
```

也就是：

> **先让 data-plane 停止引用，再让 control-plane 释放 ownership。**

这个顺序不是我们新发明的；Tangram 就是 worker result → scheduler → `free_blocks_by_ids()`。

---

## 0.5.6 第二个 hidden bug：不能把全局 `seq_lens` 简单改成 physical length

Tangram 的 source audit 给了非常关键的答案。

它保留：

```text
positions_np
=
logical num_computed_tokens + query offset
```

全局：

```text
seq_lens
=
logical num_computed_tokens + scheduled tokens
```

仍然是逻辑长度，因为这些字段还参与：

- sampling/discard 判断；
- request progression；
- scheduler/model state；
- 其他非-attention semantics。

只有 KV write slot 使用：

```text
slot_positions
=
effective_seq_lens_cpu + write offset
```

同时把：

```text
effective_seq_lens_cpu
```

单独传给 attention metadata。

因此我们的 uniform MVP 也应该遵循：

```text
logical seq len             # 不变
physical/effective KV len   # 新字段
logical model position      # 不变
physical cache position     # 新字段
```

而不是把 `self.seq_lens` 全局替换成 physical length。

这是一个**已经被 Tangram 源码明确解决的设计问题**，不再需要我们自己猜。

---

## 0.5.7 第三个 hidden bug：allocation 必须按 physical occupancy 继续增长

压缩后如果 Scheduler 仍然：

```text
num_required_blocks
=
ceil(logical_num_computed / block_size)
```

它会马上把释放的 page 又“补回来”。

Tangram 已经有 exact reference：

```text
KVCacheManager.allocate_slots(
    ...,
    effective_num_cached_tokens=...
)
```

有该值时：

```text
num_tokens_need_slot
=
effective_num_cached_tokens
+ num_new_tokens
+ lookahead
```

而不是使用 logical `num_computed_tokens`。

这就是 P1 V1 scheduler-side allocation 的主要 template。

Reference：

- `aiha-lab/tangram/vllm/v1/core/kv_cache_manager.py`
- `aiha-lab/tangram/vllm/v1/core/sched/scheduler.py`

---

## 0.5.8 P1 的 deterministic gain 到底是什么

这里必须修正一个容易夸大的说法。

vLLM 启动时已经为 KV Cache 建立固定 GPU block pool。

因此：

```text
BlockPool free pages ↑
```

通常**不代表**：

```text
nvidia-smi allocated HBM ↓
```

因为 CUDA KV tensor 本身仍然预分配在那里。

真正确定的收益是：

```text
request-owned physical KV pages ↓
        ↓
vLLM internal BlockPool free pages ↑
        ↓
same fixed KV pool 能服务更多 future allocations
```

所以 Core 指标应该叫：

- effective KV occupancy；
- blocks/request；
- free blocks；
- allocator utilization；
- admission capacity。

而不是：

- “CUDA HBM allocation 被释放”；
- “nvidia-smi 显存一定下降”。

在 memory-pressure workload 中：

```text
concurrency ↑
preemption ↓
admission reject ↓
```

是高概率但仍需 workload 验证的 serving-level 收益。

---

## 0.5.9 Core Hard Gates：任何一个没满足就先不继续扩 scope

在写 V1 功能前，先做 8 个 gate：

1. `num_kv_cache_groups == 1`；
2. prefix caching = off；
3. spec decode = off；
4. async scheduling = off；
5. full CUDA Graph = off / eager bring-up；
6. dense full attention；
7. reclaim 只在 last-prefill forward 完成之后；
8. reclaim 后第一版只允许 `q_len == 1` decode。

如果运行时不满足：

```text
fail fast
```

不要静默进入未经验证路径。

Preemption/resume 单独作为 compatibility gate：

- 单请求 correctness bring-up 阶段不依赖它；
- 要开始“preemption reduction” benchmark 前，必须先补 preempt/resume correctness test；
- 若暂时未支持，benchmark 只能报告 capacity/admission，不允许声称 preemption 改善。

---

## 0.5.10 最终先验结论

### V1

**核心方案没有“必须临时发明一个未知 subsystem”的阻塞点。**

唯一 preferred-only 的 zero-copy reorder 有 fallback。

因此：

> **V1 finishability：高。**

### V2

Triton gather 没有 exact drop-in kernel，但 memory contract 完整，且 PyTorch/Tangram/vLLM 三重 oracle 足够。

因此：

> **V2 correctness finishability：高；performance speedup：不保证。**

### V3

全部视为：

> **D — optional experimental optimization**

绝不拿来定义项目是否成功。

---


# 0.6 Round-3 GitHub Implementation Audit：从“方案可行”继续核实到“按什么 contract 实现”

这一轮不再问：

```text
有没有类似论文 / 有没有类似项目？
```

而是逐个回答：

```text
V1 / V2 每一个必要状态、调用边界、数据结构和 commit boundary，
到底有没有公开代码可以参考？

如果没有 exact implementation：
我们缺的是一个局部 glue，还是缺一套新的 runtime architecture？

如果某个 preferred path 失败：
是否存在 reference-backed fallback，让项目仍然可以完成？
```

本轮额外审计了：

- pinned vLLM v0.26 本身；
- 2026-08-24 时点更靠后的 vLLM upstream；
- Tangram；
- Sparse-vLLM；
- NVIDIA KVPress。

## 0.6.1 本轮结论先说清楚

P1 的结论比 v2 **更确定**：

> **V1 现在几乎不存在“为了实现 whole-block reclaim 必须自己发明一个新的 BlockTable primitive”的问题。**

原因是 pinned vLLM v0.26 已经提供：

```text
BlockTable.add_row(...)
```

它的语义就是：

```text
当前 request row
→ reset active row length
→ 按给定 physical block-id list 重建 row
```

而且这不是一个“没人用过的 helper”。

`GPUInputBatch.add_request()` 就在正常 runtime 中使用：

```text
request.block_ids
→ BlockTable.add_row(...)
```

同时 `GPUModelRunner._update_states()` 已经存在两种 block-state transition：

```text
normal running
→ append new block IDs

preemption resume
→ replace full block-id list
```

因此 P1 V1 实际上只需要增加第三种明确 transition：

```text
reclaim commit
→ replace request's current physical block-id list
```

我们不是在给 vLLM 发明“可变 BlockTable”。

我们是在利用它本来就存在的 **full-row reconstruction primitive**，新增一个 reclaim-specific runtime protocol。

### Reference

Pinned vLLM:

```text
vllm/v1/worker/block_table.py
vllm/v1/worker/gpu_input_batch.py
vllm/v1/worker/gpu_model_runner.py
```

commit:

```text
568afb3a13806beb53bb2e6bd518269357b237c0
```

---

## 0.6.2 因此 V1 的 reference 等级需要更新

上一版将：

```text
uniform 2D whole-block zero-copy reorder
```

整体记为 C。

这一轮应拆成两部分：

| 子问题 | 新等级 | 原因 |
|---|---:|---|
| arbitrary physical block-id row reconstruction | **A** | pinned vLLM 已有 `BlockTable.add_row()` |
| CachedRequestState full block-list replacement | **A/B** | preemption resume 已有 replace semantic |
| reclaim-specific state transition | **B** | 基于上述 primitive + Tangram reclaim protocol 做适配 |
| reclaim 后 effective KV length | **A/B** | Tangram 已有 exact general-case semantic |
| worker ACK 后 scheduler free | **A** | Tangram 已有完整闭环 |
| whole-block keep policy | **A/B** | sink/recent 等成熟 policy；不是项目风险 |
| zero-copy correctness | **B** | block table indirection + no tensor movement；需要我们的 E2E 验证 |

所以 P1 V1 真正需要我们设计的，不再是：

```text
“怎样让一个 vLLM BlockTable 支持 replacement？”
```

而是：

```text
什么时候 replacement
replacement 与 effective_kv_len 何时一起 commit
freed IDs 什么时候才可进入 BlockPool
下一轮 allocation / slot mapping 如何消费新 state
```

这些正好都有 Tangram 与 vLLM runtime 的 reference。

---

# 0.7 V1 Implementation Contract：不写代码，先把完整状态机定死

建议把 V1 看成一个 **transactional state transition**，而不是“调用一个 free_blocks”。

## 0.7.1 Reclaim 前的 canonical state

对 request A：

```text
Scheduler:
logical_num_computed = L
canonical_blocks      = [B0, B1, ..., Bn-1]

Worker CachedRequestState:
block_ids             = [B0, B1, ..., Bn-1]
logical_num_computed  = L

Persistent InputBatch:
BlockTable row        = [B0, B1, ..., Bn-1]
effective_kv_len      = L
```

V1 first release 要求：

```text
L 是完整 prefill 已完成后的 committed prompt length
```

并且：

```text
prefix cache off
spec off
async off
q_len after reclaim = 1
```

因此没有：

- shared page；
- rejected speculative token；
- in-flight future step；
- 后续 prefill chunk 还会写被回收 page。

---

## 0.7.2 Reclaim decision 只产生逻辑 keep-set，不直接 free

例如：

```text
current physical row:
[B0 B1 B2 B3 B4 B5 B6 B7]

keep logical block columns:
[0, 1, 6, 7]

retained physical IDs:
[B0, B1, B6, B7]

candidate freed IDs:
[B2, B3, B4, B5]
```

### Hard invariant

keep column 必须保持：

```text
ascending logical order
```

因为 attention 看到的 compacted physical sequence：

```text
column 0
column 1
column 2
...
```

仍然代表 retained token 的原始顺序。

这里不要求：

```text
physical block id 连续
```

PagedAttention 本来就依赖：

```text
block-table indirection
```

因此：

```text
[B0, B1, B6, B7]
```

是合法 physical page list。

---

## 0.7.3 Worker commit 是 V1 的真正 commit point

不能：

```text
Scheduler 先 free
```

正确顺序固定为：

```text
last-prefill model forward finishes
        ↓
worker decides/receives keep-set
        ↓
worker constructs retained physical block-id list
        ↓
CachedRequestState.block_ids
        =
retained block IDs
        ↓
InputBatch BlockTable
        =
rebuild current row with retained block IDs
        ↓
effective_kv_len
        =
retained physical token count
        ↓
block-table/effective metadata committed to device
        ↓
Worker reports:
new_effective_kv_len
candidate_freed_block_ids
        ↓
only now control plane may free
```

这意味着一个 reclaim event 应该被理解成：

> **physical-view transaction**

而不是 allocator API 调用。

---

## 0.7.4 Scheduler commit 是第二个 commit point

Scheduler 收到 Worker output 后：

```text
validate:
freed IDs ⊂ canonical request ownership

validate:
retained + freed == previous owned blocks
（第一版 whole-block path）

validate:
no duplicates
no null block
no shared prefix page
        ↓
canonical ownership
→ remove freed IDs
        ↓
BlockPool.free_blocks(freed)
        ↓
request.effective_num_cached_tokens
→ new effective length
```

之后才允许：

```text
Request B
```

在下一轮 allocation 获得这些 page。

### 为什么必须做 ownership validation

如果 Worker 与 Scheduler state drift：

```text
worker thinks B3 was owned
scheduler thinks B3 belongs elsewhere
```

直接 force-free 是最危险的 bug 类型。

所以：

```text
worker result
```

不是 allocator truth。

Scheduler-side `req_to_blocks` 才是 canonical ownership truth。

---

# 0.8 V1 下一轮 decode 的 exact semantic

reclaim 后假设：

```text
logical_num_computed = 8192
effective_kv_len     = 4096
```

下一 token 必须同时存在两套 position：

```text
model / RoPE position:
8192

physical KV append position:
4096
```

因此 V1 不是简单：

```text
num_computed_tokens = 4096
```

那会破坏 model semantics。

## 0.8.1 Tangram 已经证明的 general pattern

Tangram 保持：

```text
positions_np
=
logical num_computed_tokens + query offset
```

但是写 KV slot 时使用：

```text
slot_positions
=
effective_seq_lens + query offset
```

因此我们直接抽取这个 pattern：

```text
logical_position
≠
cache_position
```

### 必须保持 logical 的 state

包括：

- request token progression；
- RoPE/model position；
- sampling eligibility；
- prompt/decode phase 判断；
- max model length accounting。

### 可以改成 physical/effective 的 state

只包括：

- KV slot mapping position；
- attention retained-KV member length；
- KV allocator occupancy。

---

# 0.9 V1 allocation：为什么 Tangram reference 已经足够具体

最容易在 coding 中踩的坑是：

```text
我们刚 free 4 个 block
↓
下一步 allocate_slots 看 logical_num_computed 仍然是 8192
↓
认为 request 应该持有 8192-token KV
↓
把刚刚 free 的 block 又补回来
```

Tangram 已经解决 exactly 这个问题：

```text
effective_num_cached_tokens
```

进入 allocation calculation。

因此我们不需要自己推一套新的 allocator policy。

P1 v1 的原则就是：

```text
logical progress
→ scheduler request semantics

effective physical occupancy
→ KV allocation sizing
```

### V1 要验证的 allocation invariant

每一步 decode 后：

```text
effective_kv_len_next
=
effective_kv_len_before_step
+
num_committed_new_tokens
```

直到下一个 reclaim event。

例如：

```text
after reclaim:
4096

decode 1:
4097

decode 2:
4098
```

而 logical 侧：

```text
8192
8193
8194
```

两者独立增长。

---

# 0.10 V1 safe-free：新的 upstream vLLM 又给了一层独立佐证

本轮额外检查了更靠后的 vLLM upstream：

```text
audit snapshot:
0ecc284790e5403f74b899524ef82ecb69f83cb3
```

注意：

> 它不是 P1 implementation baseline，只作为“后来 upstream 怎么解决相同 lifetime 问题”的 reference。

它现在的 `KVCacheManager.allocate_slots()` 在释放 sliding-window 等 skipped blocks 时，明确不是使用 optimistic total computed boundary，而是：

```text
processed / committed boundary
=
total_computed_tokens - num_in_flight_tokens
```

源码注释直接指出：

```text
in-flight steps may still read old blocks
rejected speculative tokens can roll back progress
```

这和我们 P1 的 worker-ACK 原则完全一致。

## 0.10.1 对 P1 Core 的意义

同步 MVP：

```text
async scheduling off
```

所以：

```text
num_in_flight = 0
```

worker output 本身就是天然 safety fence。

这进一步说明：

> 当前 V1 的 hard gate 不是“偷懒”，而是把 lifetime protocol 先限制到最容易证明正确的 case。

---

## 0.10.2 对 V3 async compatibility 的意义

未来打开 async 后，不应该凭感觉扩展。

直接参考 modern upstream：

```text
processed-safe boundary
+
deferred physical free
```

它还有专门：

```text
tests/v1/core/test_deferred_block_free.py
```

测试：

- finished request 还有 future GPU step 在 flight；
- abort 发生时仍有 inflight writes；
- preemption 时 canonical bookkeeping 可以先清，但 physical page 必须延迟 free；
- 多个 in-flight step 时必须等 newest fence；
- 只有与该 request 相关的 future step 才应阻止 free。

因此 P1 V3 async 已经可以从：

```text
D: 大范围未知
```

降低成：

```text
B: newer-upstream-backed porting task
```

虽然仍然不建议放进 Core。

---

# 0.11 V1 最小 patch surface：进一步收紧

这一轮后，V1 不建议一开始新增过多 abstraction。

更稳的 patch surface 可以控制在：

```text
Scheduler/Request state
    │
    ├── effective physical occupancy
    └── reclaim metadata/result

KVCacheManager / SingleTypeKVCacheManager
    │
    ├── allocation against effective occupancy
    └── commit freed ownership after worker ACK

SchedulerOutput / ModelRunnerOutput
    │
    └── reclaim transaction metadata

CachedRequestState / GPU InputBatch
    │
    ├── effective_kv_len
    └── block row full replacement

GPUModelRunner
    │
    ├── last-prefill reclaim commit
    └── logical position vs cache position split

Attention metadata builder
    │
    └── effective retained KV length
```

### 一个重要修正

v2 里建议新增：

```text
BlockTable.replace_row()
```

现在不应该把它写成必要 patch。

Pinned vLLM 已经有：

```text
BlockTable.add_row()
```

做 full-row reconstruction。

第一版优先复用现成 primitive。

只有在 profiling/correctness 发现：

```text
add_row 的命名或 tail clearing 语义不够清晰
```

才增加一个非常薄的 reclaim-specific wrapper。

不要为了项目看起来“改得多”而创造多余 API。

---

# 0.12 V1 如何证明“真的不是 toy”

推荐把 V1 correctness 进一步组织成四层 oracle。

## Layer A — pure state oracle

不给 GPU。

只验证：

```text
old blocks
keep columns
→ retained IDs
→ freed IDs
```

以及所有 invalid input：

```text
duplicate keep
unsorted keep
out of bounds
keep empty
```

第一版应该：

```text
reject before mutation
```

---

## Layer B — BlockTable/slot oracle

构造：

```text
retained row = [B0, B1, B6, B7]
effective_kv_len = 64
```

验证 next token：

```text
logical position = 128
physical slot position = 64
```

通过 BlockTable 得到：

```text
physical slot ∈ current retained/append blocks
```

---

## Layer C — allocator reuse oracle

最重要：

```text
A releases IDs
↓
free count exactly increases
↓
B allocation really receives one or more of those IDs
↓
A continues decode
```

只测：

```text
free block count ↑
```

还不够。

---

## Layer D — attention semantic oracle

构造 retained K/V：

```text
full KV
↓
select retained tokens
```

Dense PyTorch reference：

```text
Q attends selected K/V in retained order
```

vLLM path：

```text
Q
+
reconstructed BlockTable
+
effective KV length
```

二者 compare。

这能把：

```text
allocator correctness
```

和：

```text
attention correctness
```

分开定位。

---

# 0.13 V2：Triton compaction 的 implementation contract 再细一层

V2 的风险不是“有没有办法实现”。

风险是：

```text
一开始为了少一次 global-memory copy
直接写复杂 in-place paged→paged
```

导致 alias/race/debug 成本失控。

所以 V2 Required Path 固定：

```text
Phase 0
PyTorch semantic oracle

Phase 1
Triton paged gather
paged source → contiguous scratch

Phase 2
known destination slot mapping
contiguous scratch → compacted paged destination

Phase 3
worker commits new physical row/effective length

Phase 4
scheduler frees trailing pages
```

## 0.13.1 Paged gather 的 source addressing 已有 exact primitive

Pinned vLLM 的：

```text
triton_reshape_and_cache_flash.py
```

已经明确实现：

```text
slot
→ block_idx = slot // block_size
→ block_offset = slot % block_size
→ page/head/dim addressing
```

并且同文件已经有：

```text
grid = (num_tokens, num_heads)

load one token/head
→ fp32 absmax
→ scale
→ quantize
→ paged store
```

因此 V2 自己写 gather 时：

```text
Triton launch structure
head vector tiling
block/slot decomposition
stride handling
```

都不是未知。

唯一新逻辑是：

```text
source logical token index
=
keep_indices[dst_compact_idx]
```

然后把 source mapping 反向应用到 load。

---

## 0.13.2 为什么第一版 destination scatter 应尽量复用 upstream semantic

不要同时自己发明：

```text
source gather addressing
+
destination paged layout
+
in-place hazard protocol
```

第一版的工程目标是隔离变量：

```text
our Triton gather
→ known contiguous scratch
→ upstream-style paged write
```

这样如果出错：

```text
gather result wrong
```

和：

```text
paged scatter wrong
```

可以分开测试。

---

## 0.13.3 Scratch 不是失败，而是 intentional staging design

V2 first release 有：

```text
paged read
→ scratch write
→ scratch read
→ paged write
```

确实多 global-memory traffic。

但这是为了：

```text
deterministic correctness
```

而不是追求最终最优 kernel。

只有 NCU/NSYS 证明：

```text
scratch traffic 成为 dominant cost
```

V3 才做 fused paged→paged。

---

# 0.14 V2 成功条件进一步收紧

V2 Required：

```text
arbitrary sorted keep_indices
        ↓
correct paged gather
        ↓
correct compact writeback
        ↓
correct effective length
        ↓
correct freed trailing IDs
        ↓
real allocator reuse
```

性能方面只要求：

```text
measure
explain
profile
```

不要求：

```text
Triton > PyTorch by X%
```

也不要求：

```text
E2E TPOT 必然改善
```

### 为什么仍然有简历价值

因为它展示的是：

```text
KV physical layout
Triton memory addressing
runtime lifetime
allocator integration
profiling
```

而不是“我调了 num_warps”。

---

# 0.15 V1/V2 Implementation Readiness Matrix

| 工作项 | 是否已有 reference | 需要自己的设计量 | fallback | 开工风险 |
|---|---|---:|---|---:|
| whole-row block replacement | pinned vLLM exact primitive | 很低 | `add_row()` existing path | 低 |
| request full block-list replacement | pinned runner resume path | 低 | remove/re-add persistent row | 低 |
| effective physical occupancy | Tangram | 低 | same semantic, uniform scalar | 低 |
| logical/cache position split | Tangram | 中低 API adaptation | copy-based compact path still same split | 低 |
| worker→scheduler freed-ID ACK | Tangram | 低 | exact pattern | 低 |
| canonical ownership removal | Tangram + vLLM manager | 中低 porting | conservative no-free on mismatch | 低 |
| allocator grows from physical frontier | Tangram | 低 | exact accounting template | 低 |
| V1 zero-copy KV movement | Paged indirection | 低 | Tangram-style copy compaction | 低 |
| PyTorch token compaction | Tangram semantic | 低 | — | 低 |
| Triton paged gather | vLLM address primitive | 中 | PyTorch oracle + staged Triton | 中低 |
| Triton scatter | pinned vLLM store primitive | 中低 | existing reshape/cache semantic | 低 |
| fused in-place compaction | 无必要 exact Core ref | 高 | staged V2 | 高，V3 only |
| async reclaim lifetime | newer vLLM exact analogous mechanism | 中 | keep MVP sync | 中，V3 |
| policy quality | many mature policies | 低 | sink+recent | 不影响 Core |

结论：

> **V1 已经可以进入 implementation planning；V2 也不存在 architecture research blocker。**

真正需要研发试错的部分只有：

```text
Triton performance tuning
V3 compatibility/optimization
```

它们都不控制项目是否能完成。


# 1. 为什么做这个问题

## 1.1 长上下文推理的 KV Cache 是真实容量瓶颈

对于 decoder-only Transformer，decode 阶段每生成一个 token，都需要保留历史 token 的 K/V。

对一个请求，KV memory 大体随 sequence length 线性增长：

```text
KV_bytes
≈
num_layers
× 2(K,V)
× num_kv_heads
× head_dim
× sequence_length
× dtype_bytes
```

因此长上下文服务中会出现：

```text
sequence length ↑
    ↓
KV blocks/request ↑
    ↓
BlockPool free blocks ↓
    ↓
admission capacity ↓
preemption probability ↑
concurrency ↓
```

vLLM 的 PagedAttention 解决的是：

> 如何以 block/page 的形式高效管理 KV，而不是给每个 request 分配一块固定连续 tensor。

但标准 full-attention 请求仍然有一个基本语义：

```text
已经生成的历史 KV
通常一直占有物理 block
直到 request 完成
```

如果我们在算法上说：

```text
这些历史 token 不再需要
```

但 allocator 里：

```text
这些 block 仍属于 request
```

那么**GPU memory 并没有真正回来**。

所以本项目解决的不是“attention mask”。

解决的是：

> **logical retention → physical memory reclamation**

---

# 2. 为什么这个项目适合作为 vLLM 主项目

它同时覆盖四层 AI Infra 能力：

```text
Control Plane
Scheduler / KVCacheManager / BlockPool
                │
                ▼
Runtime State
Request state / persistent batch / block ownership
                │
                ▼
Data Plane
BlockTable / slot_mapping / attention metadata
                │
                ▼
GPU Primitive
Triton paged-KV gather / compact / writeback
```

并且可以继续用：

```text
NVTX
Nsight Systems
Nsight Compute
```

做真实 profiling。

因此它不是：

```text
写一个独立 eviction Python demo
```

也不是：

```text
写一个孤立 Triton microbenchmark
```

而是一个跨：

- Scheduler
- allocator
- worker state
- paged addressing
- attention metadata
- Triton kernel
- correctness
- benchmark
- profiling

的 production-runtime integration 项目。

---

# 3. Source Fact：标准 vLLM 为什么不能直接支持“压缩后的物理 KV”

这里必须先看 upstream 的真实约束。

---

## 3.1 vLLM `BlockTable`：logical position 同时承担 physical addressing

参考：

**vLLM**
- repo: `vllm-project/vllm`
- commit: `568afb3a13806beb53bb2e6bd518269357b237c0`
- file: `vllm/v1/worker/block_table.py`

源码中标准 `BlockTable` 是二维结构：

```text
[max_num_reqs, max_num_blocks_per_req]
```

`append_row()` 将 scheduler 分配的 physical block id 顺序追加进 request row。

而 `compute_slot_mapping()` 的核心语义是：

```text
position
    ↓
position // block_size
    ↓
block_table[row][block_index]
    ↓
physical block id
    ↓
physical slot
```

也就是说 upstream 默认假设：

```text
model logical token position
≈
KV cache position
```

Reference:
- https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/block_table.py

---

## 3.2 `GPUModelRunner._prepare_inputs()` 把 `num_computed_tokens` 同时用于 position / seq_len / slot mapping

参考：

- `vllm/v1/worker/gpu_model_runner.py`

源码实际构造：

```python
positions = num_computed_tokens + query_pos
seq_lens = num_computed_tokens + num_scheduled_tokens
block_table.compute_slot_mapping(..., positions)
```

因此标准路径中：

```text
num_computed_tokens
      │
      ├── model/RoPE position
      ├── attention seq_len
      └── KV slot mapping position
```

三种语义实际上被绑定在一个计数器上。

Reference:
- https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/gpu_model_runner.py

这就是本项目最核心的 runtime problem。

---

## 3.3 `SingleTypeKVCacheManager` 才是 canonical physical ownership owner

参考：

- `vllm/v1/core/single_type_kv_cache_manager.py`

源码有：

```python
self.req_to_blocks:
    request_id -> list[KVCacheBlock]
```

同时：

```python
num_req_blocks = len(self.req_to_blocks[request_id])
```

参与后续 allocation 计算。

标准 FullAttention 默认：

```python
get_num_skipped_tokens(...) -> 0
```

也就是说 full attention 不会主动释放中间历史 block。

但同一个 manager 已经存在成熟的：

```text
remove_skipped_blocks
_remove_blocks_in_range
block_pool.free_blocks(...)
```

这说明 allocator 层已经有“释放 request-owned blocks”的成熟 primitive，只是 full-attention 标准路径不会这样做。

Reference:
- https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/core/single_type_kv_cache_manager.py

---

# 4. 最重要的 Reference：Tangram 已经证明“物理 KV reclamation”可以进入 vLLM runtime

参考仓库：

**Tangram**
- repo: `aiha-lab/tangram`
- audited commit: `8e1cbfa5cf82acc67fe1880df0c632f2a6485e35`

我们不是要复现 Tangram。

我们抽取的是它已经验证过的几个 engineering primitives。

---

## 4.1 Tangram `CompressionExecutor` 已经实现 KV keep → gather → writeback

文件：

```text
vllm/v1/attention/compression/executor.py
```

源码明确描述：

```text
Runs after model.forward

keep_idx =
sink
∪ locked
∪ topk
∪ window

gather cached KV
↓
scatter kept K/V back
↓
block-aligned writeback
```

它说明：

> token-level KV compaction 的 tensor semantics 已经有公开 production-oriented reference。

Reference:
- https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/attention/compression/executor.py

---

## 4.2 Tangram `BlockTable.compact_after_compress_all_layers()` 真正收集 freed physical block ids

文件：

```text
vllm/v1/worker/block_table.py
```

它做：

```text
old blocks:
[b0 b1 b2 b3 b4 b5]

new occupancy:
4 blocks

↓ trim

[b0 b1 b2 b3]

freed:
[b4 b5]
```

函数会：

1. 校验 new block count 不能大于 old；
2. 找到 `[new_count, old_count)`；
3. 收集 physical block IDs；
4. 清空对应 BlockTable entry；
5. 更新 per-row block count；
6. 返回 `freed_ids`。

Reference:
- https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/worker/block_table.py

---

## 4.3 Tangram 把 worker-side compression result 回传 scheduler

文件：

```text
vllm/v1/outputs.py
```

`ModelRunnerOutput` 新增了：

```text
compression_new_eff_seq_lens
compression_freed_block_ids
```

也就是说它没有让：

```text
Worker 自己偷偷释放 block
```

而是：

```text
Worker
  │
  │ compression result
  ▼
ModelRunnerOutput
  │
  ▼
Scheduler
  │
  ▼
KVCacheManager
  │
  ▼
canonical allocator state
```

Reference:
- https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/outputs.py

---

## 4.4 Tangram Scheduler 会同步 effective occupancy，并把 freed ids 归还 KVCacheManager

文件：

```text
vllm/v1/core/sched/scheduler.py
```

源码做两件非常关键的事：

### A. 保存压缩后的 effective sequence length

```text
request.compress_max_eff_seq_len
```

后续 allocation 不再只按 logical `num_computed_tokens` 估算。

### B. 真正释放 physical block

```python
kv_cache_manager.free_blocks_by_ids(...)
```

因此 Tangram 公开代码已经证明：

```text
worker compaction
        ↓
effective physical occupancy
        ↓
scheduler accounting
        ↓
canonical block ownership update
        ↓
physical block return
```

是一个完整闭环。

Reference:
- https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/core/sched/scheduler.py
- https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/core/kv_cache_manager.py

---

# 5. 第二个 Reference：Sparse-vLLM 给我们 architecture ownership 和 correctness 方法

参考：

**Sparse-vLLM**
- repo: `CURRENTF/Sparse-vLLM`
- audited commit: `c14643387104c9db923a4fe690b699fe7fbd1790`

---

## 5.1 Architecture reference

它的 architecture map 明确把：

```text
persistent method state
physical mutation
metadata
view materialization
```

放到：

```text
engine/cache_manager/
```

ModelRunner 主要负责：

```text
construct + connect runtime components
```

Attention data plane 消费 generic compute view，而不应该理解每个 method 的名字。

这个 ownership 原则非常适合我们的 P1：

```text
reclamation policy/state
        ↓
KV cache owner

ModelRunner
        ↓
只消费 physical cache state
```

Reference:
- https://github.com/CURRENTF/Sparse-vLLM/blob/c14643387104c9db923a4fe690b699fe7fbd1790/.agents/skills/add-sparse-method/references/file-map.md

---

## 5.2 Correctness reference

文件：

```text
tests/test_static_eviction_compaction.py
```

Sparse-vLLM 没有只做一个 E2E smoke test。

它用：

```text
batched compaction
vs
scalar oracle
```

比较：

- page table
- free slot stack
- sequence lengths

并测试：

```text
duplicate keep index
out-of-bound keep index
```

失败时要求：

```text
state must not mutate
```

这套测试思想应该直接迁移到我们的 P1。

Reference:
- https://github.com/CURRENTF/Sparse-vLLM/blob/c14643387104c9db923a4fe690b699fe7fbd1790/tests/test_static_eviction_compaction.py

---

# 6. 第三个 Reference：vLLM 自己已经有成熟 Triton paged-KV addressing primitive

参考：

```text
vllm/v1/attention/ops/triton_reshape_and_cache_flash.py
```

在我们固定的 vLLM commit 中已经存在：

```python
@triton.jit
reshape_and_cache_kernel_flash(...)
```

它已经处理：

```text
slot_mapping
↓
physical slot
↓
block_idx = slot // block_size
block_offset = slot % block_size
↓
paged KV address
↓
tl.load / tl.store
```

同一个文件甚至包含 per-token/per-head quantization kernel：

```text
load one KV head
↓
absmax
↓
scale
↓
quantize
↓
store into paged KV
```

所以我们写 Triton compaction 并不是在未知 layout 上盲写。

Reference:
- https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py

---

# 7. Project Decision：我们不复现 Tangram，而是抽取更可控的 subsystem

Tangram 做得更大：

```text
per-layer
per-head-group
ragged paging
importance score
cross-layer decisions
compression-aware allocator
...
```

我们的主项目不应该一开始进入这个 scope。

核心原则：

> **先解决 logical/physical decoupling + real block reclamation，再增加 token-level GPU compaction。**

因此分成：

```text
V1 = Runtime Core
V2 = Triton Data-Plane Core
V3 = Optional Optimization
```

V1/V2 都是 reference-backed engineering。

V3 才允许出现我们自己的 optimization。

---

# 8. 核心状态抽象

必须显式拆开：

```text
logical_num_computed_tokens
physical_kv_len
logical_position
physical_cache_position
```

压缩前：

```text
logical_num_computed_tokens
=
physical_kv_len
```

例如：

```text
8192 logical tokens
8192 physical KV tokens
```

压缩后：

```text
logical_num_computed_tokens = 8192
physical_kv_len            = 4096
```

下一 token：

```text
model / RoPE position
=
8192

physical KV append position
=
4096
```

这是整个项目最重要的不变量。

---

# 9. 为什么 logical position 不能改

已经写入 KV Cache 的 K 向量包含它生成时的 positional encoding / RoPE 语义。

如果保留：

```text
logical token 7000
```

即使它被移动到：

```text
physical cache position 2500
```

它仍然应该保持：

```text
original logical position = 7000
```

不能重新赋予 position 2500。

所以：

```text
physical compaction
≠
model token position renumbering
```

我们只压缩：

```text
memory layout
```

不重写：

```text
model semantics
```

---

# 10. V1 — Block-Level Zero-Copy Physical Reclamation

## 10.1 目标

V1 不做 token copy。

只允许 block-aligned retention。

例如 block_size=16：

```text
original block table:

B0 B1 B2 B3 B4 B5 B6 B7
```

保留：

```text
sink:
B0 B1

recent:
B6 B7
```

新的 physical block list：

```text
B0 B1 B6 B7
```

释放：

```text
B2 B3 B4 B5
```

因为 Paged KV 本身已经通过 block table 做 indirection，所以这些 surviving blocks 不需要移动 KV tensor。

这就是 V1 最漂亮的地方：

> **real physical memory reclamation，但 GPU data copy = 0。**

---

# 11. V1 Runtime Chain

V1 的关键修正是：**不能由 Scheduler 在 Worker 更新 block view 之前先 free page。**

正确链路采用 Tangram 已验证的 worker-ack pattern：

```text
Scheduler
detects request is closing last prefill
        │
        ▼
SchedulerOutput
contains reclaim metadata / keep decision trigger
        │
        ▼
GPUModelRunner executes last-prefill forward
        │
        ▼
Worker-side reclaim commit
        │
        ├── preferred: block-row zero-copy replacement
        │
        └── fallback: Tangram-style copy/compact
        │
        ▼
Worker updates:
  BlockTable row
  effective_kv_len
        │
        ▼
ModelRunnerOutput
  reclaimed_req_id
  new_effective_kv_len
  freed_block_ids
        │
        ▼
Scheduler.update_from_output()
        │
        ▼
KVCacheManager
  remove freed IDs from canonical request ownership
        │
        ▼
BlockPool.free_blocks(...)
        │
        ▼
future schedule iteration
can safely allocate released IDs
        │
        ▼
decode continues
```

这个 ordering 的安全不变量：

```text
free(block_id)
```

只能发生在：

```text
worker no longer has a live attention/write view that references block_id
```

之后。

Tangram 的 exact pattern：

```text
CompressionExecutor / BlockTable compaction
↓
ModelRunnerOutput.compression_freed_block_ids
↓
Scheduler
↓
KVCacheManager.free_blocks_by_ids
```

就是我们的 reference。

### 为什么不做 Scheduler-first free

Scheduler-first：

```text
canonical owner drop
→ BlockPool free
→ Request B reuses
→ Worker A stale row still points to same ID
```

可能直接造成 KV alias/corruption。

所以它被明确禁止。

# 12. V1 建议修改位置

下面不是“猜几个文件”，而是按 ownership contract 拆。

## 12.1 新模块：policy / state contract

```text
vllm/v1/core/kv_reclaim.py
```

建议：

```text
ReclaimConfig
ReclaimDecision
ReclaimRequestState
```

第一版 policy 只做：

```text
sink blocks + recent blocks
```

并要求：

```text
keep_block_indices sorted ascending
whole-block only
```

### Reference

- retention policy 本身可以来自 StreamingLLM 类 sink+recent 思路；
- physical lifecycle 不由 policy 负责；
- Sparse-vLLM architecture map 支持“policy/controller 与 cache-manager physical mutation 分离”。

---

## 12.2 `single_type_kv_cache_manager.py`

这里仍是 canonical ownership owner。

不是在第一步直接 free；而是增加 scheduler 收到 Worker ACK 后调用的类似：

```text
commit_reclaimed_blocks(
    request_id,
    freed_block_ids,
    new_effective_kv_len
)
```

职责：

1. 验证 freed IDs 确实属于 request；
2. 从 request canonical ownership 中移除；
3. 防止 duplicate/free/null block；
4. 调 `BlockPool.free_blocks(...)`；
5. 保持 allocation accounting 与 worker committed state 一致。

### Exact reference

Tangram 的 exact chain：

```text
KVCacheManager.free_blocks_by_ids()
↓
KVCacheCoordinator.free_blocks_by_ids()
    ├── manager.free_blocks_by_ids(...)
    │     # 从 per-request ownership 删除
    └── block_pool.free_by_block_ids(...)
          # 真正归还 free pool
```

因此“ownership mutation”和“free queue reclaim”两个动作都有明确 reference，不需要我们自己猜。

---

## 12.3 `kv_cache_manager.py::allocate_slots`

这是 V1 最关键 patch point 之一。

标准 vLLM：

```text
num_tokens_need_slot
≈ logical num_computed + new
```

压缩后必须：

```text
num_tokens_need_slot
≈ effective physical KV occupancy + new
```

### Exact reference

Tangram 已增加：

```python
allocate_slots(
    ...,
    effective_num_cached_tokens=...
)
```

并在该值存在时使用：

```text
effective_num_cached_tokens + num_new_tokens + lookahead
```

这是直接 template，不需要我们重新设计 allocator accounting。

---

## 12.4 Scheduler / Request state

Scheduler 继续拥有：

```text
logical num_computed_tokens
```

另外维护：

```text
effective_num_cached_tokens
```

或者等价命名。

注意：

```text
logical progress
```

不能被 reclaim 回退。

Tangram 的：

```text
request.compress_max_eff_seq_len
```

就是 exact semantic reference。

---

## 12.5 `ModelRunnerOutput`

新增/裁剪 Tangram-style channel：

```text
reclaim_new_effective_kv_lens
reclaim_freed_block_ids
```

不要让 worker 直接修改 Scheduler-side BlockPool。

### Exact reference

Tangram：

```text
compression_new_eff_seq_lens
compression_freed_block_ids
```

---

## 12.6 `GPUModelRunner._update_states()`

upstream 普通 running request：

```text
append new blocks
```

preemption resume：

```text
replace block list
```

reclamation 需要显式第三种 transition：

```text
worker reclaim commit
→ replace compacted/current block view
```

preferred zero-copy V1 使用：

```text
replace row with kept physical block IDs
```

fallback 使用：

```text
compacted prefix block IDs
```

不要把 reclamation 伪装成 preemption。

---

## 12.7 `BlockTable`

uniform MVP **优先复用 pinned vLLM 已存在的 `BlockTable.add_row()` 语义**，而不是先新增一个 `replace_row()`。

它已经执行：

```text
reset active row length
→ append provided physical block IDs
```

并且 `GPUInputBatch.add_request()` 正在 production path 中使用这个 primitive 重建 request row。

因此 reclaim 需要新增的是：

```text
reclaim-specific full-row replacement transition
```

而不是一套新的 BlockTable API。

如果为了可读性后续增加 reclaim-specific wrapper，也应该只是对 `add_row()` 的薄封装，不引入第二套 row mutation semantic。

---

## 12.8 Persistent InputBatch

增加：

```text
effective_kv_len_cpu
```

不要重载：

```text
num_computed_tokens_cpu
```

### Exact semantic reference

Tangram：

```text
effective_seq_lens_cpu
```

我们的 uniform 版本只需要：

```text
[max_num_reqs]
```

而不是 Tangram 的：

```text
[max_num_reqs, num_head_groups]
```

---

## 12.9 Attention metadata

这是原文档容易漏掉的 patch point。

不能只改：

```text
slot_mapping
```

还必须让 attention backend 看到：

```text
effective KV length
```

同时保持全局 logical `seq_lens` 不变。

### Reference

Tangram：

```text
CommonAttentionMetadata(
    ...,
    effective_seq_lens_cpu=...
)
```

随后 ragged FlashAttention builder 用它构造真正的 physical/member KV lengths。

我们的 uniform path需要一个更小的 equivalent：

```text
effective_kv_lens
```

并在 FlashAttention metadata builder 中只覆盖 attention 的 KV length semantic。

---

## 12.10 这一部分哪些是我们自己真正要设计的

需要自己写但边界确定：

1. uniform `ReclaimDecision` 数据结构；
2. reclaim-specific full-row replacement transition（底层复用现有 `BlockTable.add_row()`）；
3. uniform `effective_kv_len` 如何最小侵入送进当前 v0.26 FA metadata；
4. reclaim-specific ModelRunnerOutput 命名/字段。

这些属于：

```text
API adaptation / glue design
```

不是：

```text
未知算法 / 未知 runtime mechanism
```

因为 Tangram 已经提供对应的 general-case contract。

# 13. V1 `_prepare_inputs()` 的核心变化

这部分必须严格区分**四种语义**，不能简单把 upstream 的 `seq_lens` 改成 physical length。

## 13.1 Upstream

标准路径大致是：

```text
logical positions =
num_computed_tokens + query_pos

logical seq_lens =
num_computed_tokens + num_scheduled_tokens

slot mapping input =
logical positions
```

所以 logical/physical 被绑定。

---

## 13.2 Tangram 给出的 exact general pattern

Tangram 在 compression/ragged path 中：

### Model position 仍是 logical

```text
positions_np =
logical num_computed_tokens + arange
```

### slot position 改用 effective physical occupancy

```text
slot_positions =
effective_seq_lens_cpu + arange
```

再：

```text
block_table.compute_slot_mapping(
    ...,
    slot_positions
)
```

### 全局 `seq_lens` 仍保持 logical

Tangram 没有把它整体改成 physical length。

而是另外把：

```text
effective_seq_lens_cpu
```

传入 `CommonAttentionMetadata`，由 attention builder 构造实际 retained-KV view。

这是本项目要遵循的核心 pattern。

---

## 13.3 我们的 uniform MVP

因此建议：

```text
logical_positions =
logical_num_computed + query_pos
```

用于：

- model forward；
- RoPE；
- token identity；
- sampling/request progression。

新增：

```text
cache_positions =
effective_kv_len + query_pos
```

只用于：

```text
BlockTable.compute_slot_mapping()
```

并新增：

```text
attention_effective_kv_len =
effective_kv_len + scheduled
```

送入 attention metadata。

但：

```text
self.seq_lens
```

继续保持 upstream logical semantic，除非某个 consumer 经源码核实只代表 attention KV length。

---

## 13.4 为什么这很重要

如果直接：

```text
self.seq_lens = physical length
```

可能破坏：

- partial prefill discard；
- logits/sample eligibility；
- request progression；
- spec/async future compatibility；
- scheduler/worker consistency。

所以：

> **“logical/physical split”不是把一个变量改值，而是把原本复用一个变量的多个 semantic consumer 拆开。**

这是 P1 最核心的 runtime engineering point，同时已有 Tangram reference，不需要我们从零发明。

# 14. V1 Correctness

V1 必须有五类测试。

## C1. Ownership oracle

输入：

```text
blocks = [10,11,12,13,14,15]
keep   = [0,1,4,5]
```

期望：

```text
request blocks
=
[10,11,14,15]

freed
=
[12,13]
```

---

## C2. Allocator oracle

前：

```text
BlockPool.free = N
```

释放 2 blocks：

```text
BlockPool.free = N + 2
```

---

## C3. Multi-request reuse

最重要的 E2E test：

```text
Request A
owns block 100,101,102,103
↓
A releases 101,102
↓
Request B admitted
↓
B actually receives 101/102
↓
A still decodes correctly
```

这个 test 能证明：

> memory 不是“统计数字变小”，而是真正被复用。

---

## C4. Logical/physical append oracle

例如：

```text
logical = 128
physical = 64
```

next decode：

```text
RoPE position = 128
KV slot        = 64
```

生成之后：

```text
logical = 129
physical = 65
```

---

## C5. Retention=1 identity

```text
retention_ratio = 1.0
```

应该：

```text
0 blocks freed
same block table
same model output
```

这是非常重要的 regression baseline。

---

# 15. V1 预期收益

这里把收益分成“allocator-level deterministic”和“serving-level conditional”。

## 15.1 确定收益：vLLM 内部物理 page occupancy

只要 reclaim correctness 成立：

```text
request-owned physical KV blocks ↓
BlockPool free blocks ↑
```

这是 deterministic。

例如：

```text
before:
A owns 512 blocks
pool free = 300

after reclaim 256:
A owns 256 blocks
pool free = 556
```

并可进一步证明：

```text
B receives an ID that A just released
```

这就是“real reclamation”。

---

## 15.2 必须纠正：不保证 `nvidia-smi` HBM 下降

vLLM 的 KV cache 本身是预先建立的固定 block pool。

因此：

```text
BlockPool free page
```

代表：

```text
vLLM allocator 可以重新使用
```

通常不代表：

```text
CUDA allocation 被释放给系统
```

所以不要把核心收益写成：

```text
peak HBM allocation ↓
nvidia-smi memory ↓
```

更准确的 metric：

```text
effective KV occupancy
allocated blocks/request
free blocks
allocator utilization
```

---

## 15.3 高概率但需 workload 验证

在 KV pool pressure 下：

```text
admission capacity ↑
max concurrent long-context requests ↑
preemption ↓
admission reject ↓
```

这些不是数学必然，因为 scheduler policy / workload arrival 会影响结果。

必须 benchmark。

---

## 15.4 条件性性能收益

retained KV 变短后：

```text
attention reads fewer K/V tokens
```

理论 memory traffic 会下降。

可能：

```text
TPOT ↓
throughput ↑
```

但 V1 自身也引入：

- metadata preparation；
- block-table replacement；
- extra CPU bookkeeping。

V2 还引入 compaction latency。

因此：

> **TPOT/throughput speedup 不是项目成功条件。**

如果最终结论是：

```text
capacity improves materially,
but reclaim overhead only amortizes beyond context length X
```

仍然是完整 systems result。

# 16. V1 实现确定性

评级：

> **整体高；但不是“所有代码都有 exact copy”。**

## 16.1 已经 exact reference 的难点

| 难点 | Reference | 等级 |
|---|---|---|
| canonical block ownership | vLLM `req_to_blocks` | A |
| block free/reuse | vLLM `BlockPool` | A |
| compression 后按 effective occupancy allocation | Tangram `effective_num_cached_tokens` | A |
| logical position 与 physical slot 分离 | Tangram `slot_positions = effective_seq_lens + arange` | A/B |
| worker→scheduler free ACK | Tangram `compression_freed_block_ids` | A |
| attention effective lengths | Tangram `effective_seq_lens_cpu` | A/B |
| invalid compaction state tests | Sparse-vLLM | A/B |

---

## 16.2 不是 exact copy 的难点

### Whole-block zero-copy reorder

等级：

> C

原因：

- 没找到 audited repo 中与我们 uniform 2D path 完全相同的 drop-in code；
- 但 block-table indirection、排序要求和 attention view 都是确定的。

Fallback：

```text
Tangram-style copy compaction
```

---

### vLLM v0.26 API adaptation

Tangram audited commit 与我们的 pinned vLLM/torch 版本不同。

所以我们要：

```text
port semantic
≠
copy patch
```

这会产生调 API 的工作量，但不是 architecture unknown。

---

## 16.3 V1 最大实际风险

不是“physical reclamation 能不能做”。

而是：

```text
Scheduler canonical state
Worker BlockTable state
effective KV length
logical token progression
```

四份 state 是否在同一个 commit boundary 上一致。

因此开发顺序必须是：

```text
state invariant tests
→ single request
→ multi request reuse
→ preemption compatibility
→ capacity benchmark
```

而不是先跑性能。

# 17. V2 — Token-Level Triton KV Compaction

V1 有一个限制：

```text
只能整 block 丢弃
```

例如 block_size=16，一个 block 中：

```text
token 160~175
```

如果只想保留其中：

```text
160, 165, 171
```

不能直接释放整个 block。

这时需要真正移动 KV。

---

# 18. V2 目标

输入：

```text
paged KV
+
keep_indices
```

输出：

```text
dense physical retained KV prefix
+
shorter physical occupancy
+
freed trailing blocks
```

例如：

```text
logical tokens:
0 ... 8191

keep:
0...127
+
selected history
+
7680...8191
```

最终：

```text
physical KV:
0 ... retained_len-1
```

但：

```text
logical positions
```

仍然保留原始 position semantics。

---

# 19. V2 不直接写复杂 in-place kernel

为了确定性，V2 分两层。

## Phase A — PyTorch reference

先实现：

```text
paged KV
↓
gather by keep_indices
↓
contiguous scratch K/V
↓
write to compacted slots
```

它是 correctness oracle。

---

## Phase B — Triton

第一版 Triton：

```text
Paged KV
   │
   │ Triton gather
   ▼
contiguous scratch
   │
   │ Triton/vLLM-style scatter
   ▼
Compacted Paged KV
```

而不是：

```text
source paged KV
→ complicated in-place move
```

这样能避免 overlapping read/write corruption。

---

# 20. V2 Triton Reference 来源

V2 必须把“已有 reference”和“我们自己写的 kernel”分开。

## Reference A — Tangram：compaction semantic

Tangram `CompressionExecutor` 已经给出：

```text
keep decision
↓
source KV gather
↓
retained KV
↓
writeback
↓
new effective occupancy
↓
freed trailing blocks
```

所以：

```text
what should the kernel compute?
```

不是未知问题。

---

## Reference B — vLLM：paged KV address semantic

固定 vLLM commit：

```text
vllm/v1/attention/ops/triton_reshape_and_cache_flash.py
```

已经给出：

```text
slot
↓
block_idx = slot // block_size
block_offset = slot % block_size
↓
KV cache pointer arithmetic
↓
tl.load/tl.store
```

所以：

```text
how do we address paged KV?
```

也不是未知问题。

---

## Reference C — Sparse-vLLM：oracle/testing semantic

提供：

```text
batched compaction
vs
scalar oracle
```

以及：

```text
invalid keep indices
→ reject
→ no mutation
```

---

## 明确的 reference gap

当前 audit **没有找到 exact drop-in Triton kernel**：

```text
paged KV + arbitrary keep_indices
→ contiguous retained K/V
```

因此：

```text
triton_gather_paged_kv
```

是我们自己实现。

等级：

> **C — mechanically specified own implementation**

但它不是创新算法：

```text
source layout
destination layout
index mapping
expected tensor result
```

都已定义。

如果性能不理想，correctness + NCU 分析仍然完成 GPU-data-plane 项目目标。

# 21. V2 Triton Kernel 建议

## 21.1 先建立 PyTorch oracle

```text
reference_gather_paged_kv()
reference_scatter_compacted_kv()
```

先做到：

```text
arbitrary sorted keep_indices
→ exact retained K/V tensor
```

---

## 21.2 Triton gather

```text
triton_gather_paged_kv(
    src_k_cache,
    src_v_cache,
    source_block_table,
    keep_indices,
    dst_k,
    dst_v
)
```

建议 grid：

```text
(retained_tokens, kv_heads)
```

每个 program：

```text
1 retained token
×
1 KV head
```

核心：

```text
source_logical_idx = keep_indices[token]
source_block_col   = source_logical_idx // block_size
source_offset      = source_logical_idx % block_size

physical_block =
    source_block_table[source_block_col]

source address
    ↓
tl.load K/V head vector
    ↓
tl.store scratch
```

head/vector tile 直接参考 vLLM 的 Triton paged KV store/quantization kernel。

---

## 21.3 Destination scatter

第一版优先复用或高度贴近：

```text
reshape_and_cache_flash
```

输入：

```text
scratch K/V
+
new compacted slot_mapping
```

输出：

```text
paged KV physical prefix
```

不要一开始做 in-place fused source→destination。

---

## 21.4 Race / alias hard rule

在 compaction 完成之前：

```text
freed blocks
```

仍然属于 request。

必须在 GPU compaction 完成且 worker block view committed 后：

```text
ModelRunnerOutput
→ Scheduler free
```

---

## 21.5 V2 的“成功”定义

不是：

```text
must beat PyTorch / must improve TPOT
```

而是：

1. Triton 与 PyTorch oracle 对齐；
2. 能进入真实 paged-KV path；
3. block reuse correctness 通过；
4. NCU/NSYS 能解释 kernel memory behavior；
5. 性能正负都有可解释结果。

# 22. V2 Worker → Scheduler 闭环

这一阶段不建议 Scheduler 预先释放 block。

必须：

```text
Worker completes compaction
        │
        ▼
new physical length
freed physical IDs
        │
        ▼
ModelRunnerOutput
        │
        ▼
Scheduler
        │
        ▼
KVCacheManager
        │
        ▼
canonical ownership update
        │
        ▼
BlockPool
```

这直接借鉴 Tangram 的：

```text
compression_new_eff_seq_lens
compression_freed_block_ids
```

设计思想。

---

# 23. V2 Correctness

## Kernel oracle

```text
PyTorch result
vs
Triton result
```

检查：

```text
K
V
retained length
destination slot
```

BF16 应 bitwise/near-exact，具体取决于是否发生转换。

---

## Attention oracle

构造小 tensor：

```text
Q
full K/V
keep_indices
```

Reference：

```text
dense retained K/V
→ PyTorch attention
```

Project：

```text
compacted paged K/V
→ attention backend
```

比较 output。

---

## Invalid input atomicity

借 Sparse-vLLM：

```text
duplicate keep indices
out of bounds
unsorted if contract requires sorted
```

必须：

```text
raise
+
no cache state mutation
```

---

# 24. V2 预期收益

确定：

```text
token-level retention granularity
>
block-level retention granularity
```

以及：

```text
retained physical KV bytes ↓
```

条件性：

```text
shorter decode attention
→ TPOT improvement
```

成本：

```text
gather
+
scratch
+
writeback
```

因此 V2 必须 benchmark：

```text
reclaim saved bytes
vs
compaction latency
```

---

# 25. V2 实现确定性

评级：

> **Correctness finishability：高；性能收益确定性：中/未知。**

## 已知

- compaction semantic：Tangram；
- paged addressing：vLLM Triton；
- output correctness：PyTorch oracle；
- allocator free lifecycle：V1/Tangram；
- invalid-state testing：Sparse-vLLM。

## 我们自己实现

- paged gather Triton kernel；
- launcher；
- tile/num_warps；
- scratch layout；
- v0.26 integration glue。

## 不需要自己发明

- retention 算法；
- allocator；
- page ownership protocol；
- paged address model；
- worker/scheduler feedback protocol。

## 不保证

```text
Triton compaction cost
<
saved decode attention time
```

所以项目不把 speedup 当 gate。

### Fallback ladder

```text
Triton fused attempt fails/perf poor
        ↓
two-stage Triton gather + upstream-style scatter
        ↓
still difficult
        ↓
PyTorch reference compaction remains runtime-correct baseline
```

V2 最终至少必须交付正确 Triton implementation；不要求它成为 runtime 默认路径。

# 26. V3 — 后续优化方向：只做 reference-backed、profiler-driven 的扩展

V3 的原则：

> **不因为“看起来高级”就加 scope。只有 V1/V2 完整、且已有公开机制或 profiler 指向明确瓶颈时才进入。**

本轮 GitHub 再审计后，推荐优先级需要调整。

---

## 26.1 V3-A — Async-safe / Deferred-Free Compatibility【最高优先级】

这是我最推荐的 P1 V3。

原因不是它最炫，而是它最像 production runtime hardening。

较新的 vLLM upstream 已经出现完整 reference：

```text
processed safe boundary
=
total computed
-
num_in_flight_tokens
```

并配套：

```text
deferred block free
```

测试直接覆盖：

- finish while future step is in flight；
- abort while prefill/decode is in flight；
- preemption while old GPU step may still touch page；
- multi-step async pipeline fence；
- request-local safety，而不是全局粗暴 barrier。

### P1 如何迁移这个思想

同步 V1：

```text
Worker ACK
→ immediately safe to free
```

Async V3：

```text
reclaim decision generated
↓
identify newest in-flight step that may still read/write candidate pages
↓
worker stops new scheduling from depending on dropped pages
↓
candidate freed pages enter deferred-free set
↓
process outputs until request-specific fence is satisfied
↓
BlockPool.free
```

### 成功条件

不是 throughput speedup。

而是：

```text
async scheduling enabled
+
stress test
+
no stale read/write alias
+
reclaimed page eventually reusable
```

### Reference

Current/newer vLLM audit snapshot:

```text
vllm/v1/core/kv_cache_manager.py
tests/v1/core/test_deferred_block_free.py
```

这让这一项从“自己研究 async reclamation protocol”变成：

> **port a proven lifetime/fence pattern to reclaim-specific pages。**

---

## 26.2 V3-B — Periodic Decode Reclamation【第二优先级】

V1/V2 只在：

```text
prefill → decode
```

边界压缩一次。

长 decode 后 physical KV 又会继续增长。

因此自然扩展：

```text
decode
↓
physical_kv_len crosses trigger
↓
periodic reclaim
↓
physical_kv_len returns toward target
```

### 不要先做复杂 adaptive policy

第一版只需要三个 knob：

```text
compression_interval
target_physical_tokens
min_reclaimable_blocks
```

推荐使用 hysteresis：

```text
if current_physical_len
<
target + min_reclaim_span:
    skip
```

避免每几步就做一次小 reclaim。

### Policy reference

NVIDIA KVPress 的 `DecodingPress` 已经采用：

```text
compression_interval
target_size
periodic decode compression
```

并显式维护 compression state/buffer。

KVPress 是 Transformers runtime，不是 vLLM allocator reference。

我们只借它：

```text
trigger semantics
```

不借它：

```text
physical memory implementation
```

### 推荐 first implementation

先只支持：

```text
V1 whole-block periodic reclaim
```

不要一开始 periodic + V2 token compaction 同时打开。

原因：

```text
whole-block = zero copy
```

可以先单独验证：

```text
repeated state transition
allocator reuse
logical/physical counters
```

全部稳定后，再允许 V2 compaction 被 periodic trigger 调用。

---

## 26.3 V3-C — Preemption / Resume Compatibility【在声称 preemption benefit 前必须完成】

如果最终 benchmark 要写：

```text
preemption count ↓
```

那项目必须先证明：

```text
reclaimed request itself can later be preempted/resumed correctly
```

### 核心问题

preemption 会丢掉原 physical cache，然后 resume 重新 prefill / reload。

因此必须定义：

```text
logical progress
effective_kv_len
reclaim state
```

在 preempt 时如何 reset。

推荐：

```text
preempt
→ discard old physical reclaim state
→ reset effective_kv_len to scheduler-resolved resumed physical state
→ worker uses existing full block-list replacement semantics
```

不要试图让：

```text
old reclaim mapping
```

跨一次 full physical cache loss 继续存活。

### Reference

Pinned ModelRunner 已经对 resume：

```text
replace full block IDs
```

而不是 append。

这正是 reset boundary。

---

## 26.4 V3-D — CUDA Graph Compatibility

V1 first release 使用 eager 是正确的。

开启 CUDA Graph 后需要核实：

```text
BlockTable GPU buffer address
effective length buffer address
attention metadata shape
decode batch shape
```

是否仍保持 graph-compatible。

### 推荐原则

不要动态分配新 tensor。

尽量：

```text
preallocated fixed-size buffers
+
in-place content update
```

因为：

```text
physical row content changes
```

本身不等于：

```text
graph shape changes
```

### 成功条件

```text
reclaim path under supported decode graph mode
→ same correctness
→ no unexpected recapture explosion
```

这更像 runtime compatibility work，不要求速度一定更高。

---

## 26.5 V3-E — Fused Paged-to-Paged Triton Compaction

只有 V2 NCU 明确显示：

```text
scratch write + read
```

是 dominant overhead，才做。

目标：

```text
paged source
→ Triton
→ paged destination
```

### 最大风险：alias

当：

```text
destination physical prefix
```

与：

```text
source blocks
```

发生重叠时，一个 program 过早写 destination 可能覆盖另一个 program 尚未读的 source。

所以不能简单：

```text
parallel gather+store
```

### 可控实现策略

优先级：

1. 只处理已证明 non-overlap 的 source/destination ranges；
2. tile-local temporary registers/scratch；
3. 两阶段 move plan；
4. 最后才考虑 general in-place scheduling。

如果需要复杂 dependency graph：

> 不值得在简历项目时间预算内继续。

保留 staged V2 就够。

---

## 26.6 V3-F — Better Retention Policy

runtime core完成后，可以把：

```text
keep-set generation
```

做成 plugin。

可复用：

- StreamingLLM sink + recent；
- KNorm；
- SnapKV；
- H2O；
- Chunk-level keep；
- KeyDiff。

NVIDIA KVPress 提供了大量成熟 selection implementation，可用于：

```text
policy correctness / quality reference
```

但不要把本项目改造成：

```text
“研究哪个 eviction algorithm 最好”
```

### 推荐最多加入一个非 trivial policy

例如：

```text
sink + recent baseline
vs
K-norm / simple attention-score
```

足以展示：

```text
runtime mechanism is policy-agnostic
```

没有必要跑十几个压缩算法。

---

## 26.7 V3-G — Ragged Per-Layer / Per-Head Paging【最后才考虑】

Tangram 已经证明：

```text
per-layer/per-head effective length
ragged block table
```

可以做。

但它会把状态从：

```text
one scalar effective_kv_len/request
```

升级成：

```text
effective length per layer/head-group
```

并改变：

- BlockTable shape；
- allocator accounting；
- attention metadata；
- worker result format；
- block ownership granularity。

这不是“再加一点功能”。

这是第二套更大的 runtime architecture。

因此建议：

> **除非 V1/V2 提前大量完成，否则不做。**

简历价值不会因为没做到 Tangram full generality 而降低。

---

## 26.8 V3 推荐顺序

```text
V1
whole-block physical reclamation
        ↓
V2
token-level staged Triton compaction
        ↓
V3-A
async/deferred-free compatibility
        ↓
V3-B
periodic decode reclaim
        ↓
V3-C
preemption/resume hardening
        ↓
V3-D
CUDA Graph compatibility
        ↓
V3-E
fused compaction（only if profiler says yes）
        ↓
V3-F
one better retention policy
        ↓
V3-G
ragged per-head/layer（only if scope allows）
```

这里最重要的变化：

> **V3 优先做 runtime hardening，而不是继续堆算法创新。**

这更符合 AI Infra 简历项目定位。


# 27. Profiling 设计

建议 NVTX：

```text
P1_RECLAIM_DECISION
P1_CACHE_MANAGER_MUTATE
P1_BLOCKTABLE_REPLACE
P1_PREPARE_LOGICAL_POS
P1_PREPARE_CACHE_POS
P1_TRITON_GATHER
P1_TRITON_SCATTER
P1_ATTENTION
```

---

## Nsight Systems

回答：

```text
reclaim 在 CPU critical path 占多少？
BlockTable H2D copy 是否增加？
compaction 和 model forward 是否串行？
attention duration 是否下降？
```

---

## Nsight Compute

只针对 V2 kernel：

```text
DRAM throughput
load/store efficiency
occupancy
register pressure
kernel launch count
```

V1 没有必要硬上 NCU。

---

# 28. Benchmark Matrix

至少三组：

```text
Baseline
No reclamation

V1
Block-level reclamation

V2
Token-level Triton compaction
```

变量：

```text
prompt length
batch size
retention ratio
block size
decode length
```

---

## Memory / capacity

记录：

```text
blocks/request
BlockPool free blocks
effective KV occupancy
fixed KV-pool utilization
max concurrent requests
admission reject
preemption count
```

---

## Latency

记录：

```text
TTFT
TPOT
E2E latency
reclaim latency
compaction latency
```

---

## Throughput

```text
requests/s
tokens/s
goodput(optional)
```

---

## Quality

选择小而清晰的长上下文集合：

- Needle-in-a-Haystack
- RULER subset
- LongBench subset

目的不是论文 SOTA。

而是画：

```text
memory saving
vs
quality degradation
```

trade-off。

---

# 29. Project Success Criteria

## Core Done

必须：

- [ ] prefill 后能执行 block-level reclaim；
- [ ] canonical request block ownership 正确；
- [ ] freed block 真正回到 BlockPool；
- [ ] another request 能复用 freed block；
- [ ] original request 能继续 decode；
- [ ] logical position 与 physical KV position 分离；
- [ ] retention=1 identity；
- [ ] multi-request correctness；
- [ ] benchmark 能展示 capacity change。

---

## Full Main Project Done

再加：

- [ ] PyTorch token compaction oracle；
- [ ] Triton paged-KV gather；
- [ ] Triton compact writeback；
- [ ] worker→scheduler freed-block synchronization；
- [ ] Triton correctness test；
- [ ] NSYS timeline；
- [ ] NCU kernel report。

---

# 30. 代码量合理预估

不是 KPI，只用于 scope。

```text
V1 runtime core       550–900 LOC
V2 Triton core        300–550 LOC
tests                  450–750 LOC
bench/profile          300–550 LOC
──────────────────────────────
total incremental     ~1700–2750 LOC
```

如果 V3：

```text
+ 150–400 LOC
```

这是比较合理的主项目体量。

---

# 31. 我们自己的 Ownership 到底是什么

## Upstream / reference 提供

### vLLM
- allocator；
- BlockPool；
- BlockTable；
- slot mapping；
- Paged KV；
- Scheduler/ModelRunner 基础 runtime；
- Triton paged KV addressing。

### Tangram
- real physical reclamation 的完整系统先例；
- effective seq length；
- worker→scheduler freed IDs；
- KV gather/writeback semantic；
- ragged paging 的高级 reference。

### Sparse-vLLM
- cache manager ownership；
- physical mutation architecture；
- compaction correctness oracle pattern。

---

## 我们实现

- scoped vLLM v0.26 subsystem；
- uniform block reclamation；
- logical/physical state split；
- scheduler/worker state contract；
- block table replacement；
- safe block reuse；
- PyTorch compaction reference；
-自己的 Triton paged-KV compaction primitive；
- benchmark harness；
- NVTX / NSYS / NCU evidence。

---

# 32. 项目不是“创新论文项目”

V1/V2 都不需要声称新算法。

正确描述：

> I surveyed Tangram, Sparse-vLLM and vLLM’s KV runtime, extracted the physical-reclamation primitives, and implemented a scoped reclamation subsystem in vLLM. The project decouples logical model positions from compacted KV storage positions, returns unused pages to the allocator, and adds a Triton token-level KV compaction path.

V3 如果最后 profiler 找到可以优化的具体问题，再说：

```text
profiler-driven optimization
```

而不是一开始要求创新。

---

# 33. 实现确定性总结

| 子问题 | Reference 等级 | 实现确定性 | 收益确定性 | 结论 |
|---|---:|---:|---:|---|
| vLLM block ownership/free | A | 很高 | page 可复用：确定 | Core |
| effective-occupancy allocation | A（Tangram） | 很高 | 避免被 logical length 重新撑回：确定 | Core |
| logical/physical position split | A/B（Tangram） | 高 | correctness capability：确定 | Core |
| worker ACK 后 free | A（Tangram） | 高 | 防 alias：确定 | Core |
| 2D zero-copy whole-block reorder | B（pinned `BlockTable.add_row()` + Tangram protocol） | 高 | copy cost=0：correctness 可确定 | preferred，仍保留 copy fallback |
| copy-based compact fallback | A/B | 高 | reclaim：确定 | safety net |
| PyTorch token compaction | A/B | 很高 | finer granularity：确定 | V2 oracle |
| Triton paged gather | C | 高（correctness） | speedup：不确定 | V2 |
| Triton scatter | A/B | 高 | speedup：不确定 | V2 |
| capacity/admission improvement | — | — | 高概率，需 workload | benchmark |
| TPOT/throughput improvement | — | — | 不确定 | benchmark only |
| `nvidia-smi` HBM decrease | — | — | **不成立为 Core claim** | 不写 |
| fused paged→paged | D | 可做但有 race complexity | 不确定 | V3 |
| periodic reclaim | D | 中 | 不确定 | V3 |
| ragged per-head paging | A reference but scope 大 | 可行 | 条件性 | V3 |

## Final risk statement

P1 现在不存在：

> “实现到一半才发现核心机制没有 reference，只能自己发明一个新 runtime architecture”

这种已知风险。

仍存在正常 systems engineering 风险：

- API port；
- state synchronization；
- FlashAttention metadata adaptation；
- kernel correctness/performance。

而且每一个 Core 风险都有：

```text
reference contract
+
oracle
+
fallback
```

所以它和此前依赖未验证算法收益的项目风险性质不同。

# 34. Reference Inventory

## vLLM — pinned implementation baseline

1. `vllm/v1/worker/block_table.py`  
   https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/block_table.py

2. `vllm/v1/worker/gpu_model_runner.py`  
   https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/gpu_model_runner.py

3. `vllm/v1/core/single_type_kv_cache_manager.py`  
   https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/core/single_type_kv_cache_manager.py

4. `vllm/v1/core/block_pool.py`  
   https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/core/block_pool.py

5. `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`  
   https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py

6. vLLM build baseline (`torch==2.11.0`)  
   https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/pyproject.toml

---

## Tangram — general physical-reclamation reference

Audited commit:

```text
8e1cbfa5cf82acc67fe1880df0c632f2a6485e35
```

7. `vllm/v1/attention/compression/executor.py`  
   https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/attention/compression/executor.py

8. `vllm/v1/worker/block_table.py`  
   https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/worker/block_table.py

9. `vllm/v1/worker/gpu_input_batch.py` — `effective_seq_lens_cpu` / snapshots  
   https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/worker/gpu_input_batch.py

10. `vllm/v1/worker/gpu_model_runner.py` — logical positions vs effective slot positions  
    https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/worker/gpu_model_runner.py

11. `vllm/v1/outputs.py` — worker→scheduler freed IDs  
    https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/outputs.py

12. `vllm/v1/core/sched/scheduler.py` — `compress_max_eff_seq_len` lifecycle  
    https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/core/sched/scheduler.py

13. `vllm/v1/core/kv_cache_manager.py` — `effective_num_cached_tokens` allocation  
    https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/core/kv_cache_manager.py

14. `vllm/v1/core/single_type_kv_cache_manager.py` — per-request `free_blocks_by_ids()` ownership filtering  
    https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/core/single_type_kv_cache_manager.py

15. `vllm/v1/core/kv_cache_coordinator.py` — ownership filtering + `block_pool.free_by_block_ids()`  
    https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/core/kv_cache_coordinator.py

16. `vllm/v1/attention/backends/flash_attn.py` / `ragged_layout.py` — effective member KV lengths  
    https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/attention/backends/flash_attn.py  
    https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/vllm/v1/attention/backends/ragged_layout.py

17. Tangram build baseline (`torch==2.9.0`) — portability warning  
    https://github.com/aiha-lab/tangram/blob/8e1cbfa5cf82acc67fe1880df0c632f2a6485e35/pyproject.toml

---

## Sparse-vLLM — ownership/test reference

18. Architecture map  
    https://github.com/CURRENTF/Sparse-vLLM/blob/c14643387104c9db923a4fe690b699fe7fbd1790/.agents/skills/add-sparse-method/references/file-map.md

19. Static eviction/compaction tests  
    https://github.com/CURRENTF/Sparse-vLLM/blob/c14643387104c9db923a4fe690b699fe7fbd1790/tests/test_static_eviction_compaction.py

---

## Reference gaps explicitly recorded

当前没有把下面两项伪装成 exact reference：

1. **uniform 2D whole-block zero-copy reorder**；
2. **arbitrary keep-index Triton paged-KV gather**。

两者均已在 0.5 节给出 contract、oracle 和 fallback。


## 34.1 Round-3 新增 Reference Inventory

### Pinned vLLM：V1 row reconstruction / state replacement

20. `BlockTable.add_row()` — full-row reconstruction primitive  
    https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/block_table.py

21. `GPUInputBatch.add_request()` — normal runtime directly rebuilds BlockTable row  
    https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/gpu_input_batch.py

22. `GPUModelRunner._update_states()` — append vs preemption-resume full replacement  
    https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/worker/gpu_model_runner.py

### Newer vLLM upstream：只作为 post-baseline lifetime reference

Audit snapshot:

```text
0ecc284790e5403f74b899524ef82ecb69f83cb3
```

23. `KVCacheManager.allocate_slots()` / processed-token safe free boundary  
    https://github.com/vllm-project/vllm/blob/0ecc284790e5403f74b899524ef82ecb69f83cb3/vllm/v1/core/kv_cache_manager.py

24. Deferred block free tests under in-flight execution  
    https://github.com/vllm-project/vllm/blob/0ecc284790e5403f74b899524ef82ecb69f83cb3/tests/v1/core/test_deferred_block_free.py

这些文件**不直接 cherry-pick 到 v0.26**；它们用于 V3 async/lifetime design 的 source reference。

### NVIDIA KVPress：只作为 policy/trigger reference

25. KVPress project  
    https://github.com/NVIDIA/kvpress

26. Periodic decode compression trigger (`DecodingPress`)  
    https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/decoding_press.py

KVPress 不承担：

```text
vLLM allocator
block ownership
physical reuse
```

只用于：

```text
compression_interval
target_size
decode-time trigger
```

的 policy reference。


# 35. 最终判断

这个项目值得做的原因不是：

```text
KV compression 很热门
```

而是：

1. **问题真实**：long-context KV memory pressure；
2. **系统深度够**：Scheduler → allocator → ModelRunner → Paged KV → Triton；
3. **reference 足够完整**：Tangram + Sparse-vLLM + upstream vLLM；
4. **Core 确定**：physical page reclamation / allocator reuse 可以独立成立，并有 worker-ACK 与 effective-allocation reference；
5. **收益可验证**：request-owned blocks / BlockPool free pages / effective occupancy 是 deterministic metric；capacity/admission 是 workload-level benchmark；
6. **性能结果允许正负**：不会因为 TPOT 没提升导致项目失败；
7. **Triton 是正式组成**：不是为了简历硬塞一个 kernel；
8. **后续扩展自然**：fused compaction、periodic reclaim、ragged paging、CUDA Graph。

因此它非常适合作为主简历项目。
