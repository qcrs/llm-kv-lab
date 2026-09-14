> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Learning and Source Reading Guide

这不是泛读清单，而是按研发阶段给出的“你现在必须看懂什么”。

---

## 1. R1 前必须具备

### PagedAttention / BlockTable

必须能解释：

```text
request token position
 → logical block index
 → block_table[row, block]
 → physical block id
 → slot id
```

重点读：

```text
qcrs/vllm:
  vllm/v1/worker/gpu/block_table.py
  vllm/v1/worker/gpu/attn_utils.py
  vllm/v1/attention/ops/triton_reshape_and_cache_flash.py
```

掌握到：能手算给定 `block_ids=[7,2,11]`、block_size=16、position=35 的 physical slot。

---

## 2. GQA / Head geometry

必须清楚：

```text
num_query_heads
num_kv_heads
num_queries_per_kv
TP rank-local heads
```

否则无法理解“一个 KV member 携带一组 query heads”。

重点：

```text
Attention.__init__
FlashAttentionImpl
model QKV projection partition
```

练习：给 Qwen 配置，写出 TP1/TP2 下 local Q/KV head 数。

---

## 3. FlashAttention varlen / paged interface

重点不是 FlashAttention 算法推导，而是 API contract：

```text
q
k_cache/v_cache
cu_seqlens_q
seqused_k
block_table
max_seqlen_q/k
```

必须理解：Tangram 为什么能通过 virtual block table 把 `(request, KV head)` 变成标准 varlen sequence，而不改 softmax kernel。

重点读：

```text
qcrs/vllm/vllm/v1/attention/backends/flash_attn.py
Tangram/vllm/v1/attention/backends/ragged_forward.py
Tangram/vllm/v1/attention/backends/ragged_layout.py
```

---

## 4. MRv2 Persistent Buffers

这是 port 成败关键。

重点看：

```text
StagedWriteTensor
UvaBackedTensor
FusedStagedWriter
BlockTables.input_block_tables
slot_mappings
```

必须理解为什么 CUDA Graph 需要地址稳定，而不是每 step `torch.tensor(list)`。

重点源码：

```text
vllm/v1/worker/gpu/buffer_utils.py
vllm/v1/worker/gpu/block_table.py
vllm/v1/worker/gpu/states.py
vllm/v1/worker/gpu/input_batch.py
```

---

## 5. R2 前：Logical vs Physical State

复习 P1：

```text
num_computed_tokens
positions

effective_kv_len
cache_positions
effective_kv_seq_lens
```

重点读：

```text
qcrs P1 branch:
  gpu/states.py
  gpu/input_batch.py
  attention/backend.py
  flash_attn.py
```

你需要能明确回答：

> 为什么 compact KV 以后 RoPE position 不能跟着 compact？

---

## 6. R3 前：Scheduler / Allocator Ownership

重点：

```text
KVCacheManager
SingleTypeKVCacheManager
KVCacheCoordinator
BlockPool
Scheduler.update_from_output / reconciliation
```

必须区分：

```text
worker physical view
scheduler canonical ownership
allocator free list
```

重点读 P1 的：

```text
scheduler.py
single_type_kv_cache_manager.py
kv_cache_coordinator.py
test_compaction_reconciliation.py
test_reclaim_ownership.py
```

目标：可以画出一个 block ID 从 allocate → request ownership → compaction detach → free → B reuse 的生命周期。

---

## 7. R4 前：Compression semantics

只需要掌握：

```text
score ≠ keep count
keep count ≠ retained positions
retained positions ≠ physical free
```

重点读 Tangram：

```text
compression/compressor.py
compression/budget_scope.py
compression/executor.py
compression/eviction_regime.py
```

不要一开始读所有 scorer。

---

## 8. R6 前：Tensor Parallel

需要理解：

- ColumnParallel / RowParallel 基本概念
- Q/K/V head sharding
- rank-local KV cache
- TP process group
- `all_reduce(MAX)` 的语义

源码重点：

```text
vllm/distributed/parallel_state.py
attention/model projection code
Tangram compression_model_runner_mixin.py
Tangram budget_scope.py
```

练习：解释为什么 TP2 uniform 可以只同步 kept length，而 layer/global scope 需要更复杂的 cross-rank information。

---

## 9. R5 前：Continuous batching / Preemption

掌握：

```text
running / waiting
request skipped this step
persistent batch state
scheduler preemption
recompute
```

重点看：

```text
vllm/v1/core/sched/scheduler.py
request/input batch add-remove paths
Tangram RaggedBlockTable snapshot/restore
Tangram preemption benchmark
```

目标：可以准确区分“request 从 GPU input batch 消失”和“request KV 被 scheduler free”。

---

## 10. R7 前：AOT Clustering

数学基础很轻：

- 排序
- grouping
- Spearman rank correlation
- max-pool fragmentation

重点看：

```text
Tangram/tools/head_group_clustering/README.md
build_profile.py
build_cluster_map.py
clustering.py
validate.py
```

你需要会自己解释：为什么 clustering 是 placement optimization，而不是 importance algorithm。

---

## 11. R8 前：Triton

必须掌握：

```text
tl.program_id
tl.arange
tl.load/store + mask
stride address
constexpr
grid decomposition
program dependency
```

然后再读：

```text
P1 kv_compaction.py
Tangram eviction_writeback.py
```

核心练习不是抄 kernel，而是写出 in-place safety proof。

---

## 12. R9 前：CUDA Graph / torch.compile

掌握：

```text
capture/replay
stable address
full vs piecewise CUDA Graph
custom op
opaque attention op
graph break/eager island
```

重点：

```text
qcrs/vllm/config/compilation.py
qcrs/vllm/v1/worker/gpu/cudagraph_utils.py
qcrs/vllm/model_executor/layers/attention/attention.py
Tangram/vllm/config/compilation.py
Tangram/vllm/config/vllm.py
```

目标：能解释“为什么 dynamic ragged attention 不必让整个 model enforce_eager”。

---

## 13. 每轮阅读方法

不要从文件第一行读到最后一行。

推荐：

```text
1. 先写本轮 invariant
2. 找 source-of-truth state
3. 找 producer
4. 找 transport
5. 找 consumer
6. 找 commit point
7. 找 cleanup/failure path
```

例如 R3：

```text
freed block IDs
producer      = worker compaction
transport     = ModelRunnerOutput
consumer      = scheduler
commit        = ownership mutate + BlockPool.free
cleanup       = failed/stale result handling
```

这种方式比“看懂所有函数”更适合 AI Infra 源码学习。
