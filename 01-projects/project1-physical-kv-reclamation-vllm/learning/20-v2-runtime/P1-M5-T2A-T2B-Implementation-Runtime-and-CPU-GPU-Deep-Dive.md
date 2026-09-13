# P1 M5-T2A / M5-T2B 实现与 Runtime Lifecycle 深度梳理

> Project: Physical KV Cache Reclamation for vLLM  
> Upstream: vLLM v0.26.0  
> Branch: `p1/v2-token-compaction-v026`  
> Current implementation baseline HEAD: `1f04cdca301fa72d20046a791672cf265bf9df09`（working tree 有未提交 T2A/T2B 修改）  
> Scope: P1 V2 — Token-Level Physical KV Compaction  
> Position: M5-T1 CLOSED → M5-T2A CLOSED/FROZEN → M5-T2B PASS_CANDIDATE / HARDEN-01 PASS → M5-T3 NOT STARTED  
> MVP: MRV2 / A100 / BF16 / FA2 / eager / TP=PP=DP=DCP=PCP=1 / single KV group / block_size=16 / prefix caching OFF / spec decode OFF / async scheduling OFF / CUDA Graph OFF / KV connector OFF

---

## 0. 这轮到底解决了什么

P1 V1 解决的是 **whole-block physical reclaim**：Scheduler 预先决定删掉完整 block，Worker 在 forward 前把 `effective_kv_len` 切到新的 block-aligned 边界，然后本轮 query 在这个新边界后继续写。

P1 V2 的问题完全不同。

V2 允许在一个已经执行完 forward 的 KV 序列内部选择任意 member：

```text
source members:
m0 m1 m2 m3 m4 m5 m6 m7 m8 m9 ...

keep:
m0 m2 m5 m8 m9 ...
```

并要求最终物理布局变成：

```text
physical prefix:
m0 m2 m5 m8 m9
```

因此 V2 不是“删掉几个 block”，而是一个真正的：

```text
post-forward payload transformation
+
Worker execution metadata transition
+
persistent physical-length transition
+
future Scheduler ownership reconciliation
```

M5-T2A 和 M5-T2B 共同完成的是这条链中的 Worker 侧核心部分：

```text
Scheduler CompactionPlanData
        ↓
M5-T2A
resolve + validate + snapshot
        ↓
_PreparedCompaction
        ↓
M5-T2B
all requests × all layers payload compaction
        ↓
Worker BlockTable next-state stage
        ↓
batch-indexed E absolute override
        ↓
CompactionResultData
        ↓
ExecuteModelState
        ↓
sample_tokens()
        ↓
post_update()
        ↓
Worker persistent E = K
```

当前尚未做：

```text
Scheduler canonical req_to_blocks reconcile
removed suffix ownership transition
safe deferred free
reuse
continued-generation E2E closure
```

这些属于 M5-T3 / M5-T4。

---

# 1. 先建立四层状态模型

理解 T2A/T2B 最重要的不是某一行代码，而是先区分四类状态。

## 1.1 Scheduler Plan：意图

`CompactionPlanData` 表示 Scheduler/外部 producer 对 Worker 发出的 compaction 意图。

典型字段：

```text
request_id
keep_member_indices
expected_source_effective_kv_len
expected_source_num_blocks
step_seq optional
```

它表达的是：

> “我预期 Worker 当前这个 request 的 physical source 是 E_source / N blocks，并希望保留这些 member。”

它不是 Worker runtime truth。

所以 Worker 不能拿到 plan 后直接 destructive write。

---

## 1.2 Worker Prepared：经过 runtime 验证的执行描述

`_PreparedCompaction` 是 T2A 的输出。

当前 hardened 版本包含：

```text
request_id
req_state_idx
batch_idx
source_effective_kv_len
source_num_blocks
block_ids
block_size
keep_member_indices
new_effective_kv_len
new_num_blocks
step_seq
```

它不是新状态本身，而是：

```text
Scheduler intention
+
current batch facts
+
persistent Worker facts
+
BlockTable facts
+
KV tensor geometry
↓
validated immutable execution descriptor
```

生命周期严格限定为：

```text
_prepare_v2_compactions()
↓
_execute_v2_compactions()
↓
结束
```

HARDEN-01 已删除 `ExecuteModelState.prepared_compactions`，避免 Prepared descriptor 在已经执行成功后还跨越 sample boundary。

---

## 1.3 Worker Physical State：真正影响下一轮 execution 的状态

Worker 侧至少有三类 physical state：

```text
KV payload
BlockTable execution-addressing view
effective_kv_len = E
```

其中：

- KV payload：真正的 K/V 数据；
- BlockTable：逻辑 page → physical block ID 的执行寻址表；
- `effective_kv_len`：Worker 下一次计算 cache positions / attention visible extent 时使用的 physical logical-length abstraction。

V2 必须让三者最终一致。

---

## 1.4 Scheduler Canonical State：ownership authority

Scheduler 侧的：

```text
SingleTypeKVCacheManager.req_to_blocks
```

才是 canonical allocator / ownership authority。

Worker 的 BlockTable 不是 ownership authority。

因此 T2B 即使已经把 Worker active row：

```text
[B7, B2, B11]
```

缩成：

```text
[B7, B2]
```

也不能在 Worker 自己 `free(B11)`。

正确边界是：

```text
T2B:
Worker execution view updated

T3:
Scheduler canonical ownership reconciled

T4:
safe free / reuse
```

---

# 2. 六种容易混淆的“索引 / 位置”

V2 的复杂度很大程度来自多个 index domain 同时存在。

## 2.1 logical token position

模型语义位置，例如：

```text
position = 100, 101, 102 ...
```

它决定 RoPE / causal semantics。

V2 compaction **不能重新编号 logical positions**。

一个原来 position=1000 的历史 KV，即使被压缩到 physical member index=20，它的语义仍然是 position=1000 对应的历史 K/V。

---

## 2.2 physical member index

`keep_member_indices` 所属的 index domain。

例如：

```text
current physical sequence:
m0 m1 m2 m3 m4 m5 m6 ...

keep_member_indices:
[0, 2, 5]
```

这里的 0/2/5 不是 token ID，也不是 logical position，更不是 block ID。

它只表示：

> 当前 compaction source 中第几个 physical retained member。

---

## 2.3 block-table logical page index

例如：

```text
request row = [B7, B2, B11]
```

其中：

```text
page index 0 -> B7
page index 1 -> B2
page index 2 -> B11
```

page index 是 row 内偏移。

---

## 2.4 physical block ID

`7 / 2 / 11` 才是 physical block IDs。

它们用于定位实际 KV cache storage 的 block/page。

---

## 2.5 batch_idx

当前 continuous batch 中 request 的 row。

例如：

```text
current batch:
batch 0 -> req-X
batch 1 -> req-A
batch 2 -> req-Z
```

那么 `req-A.batch_idx = 1`。

---

## 2.6 req_state_idx

Worker persistent RequestState table 中 request 的 row。

例如：

```text
persistent req_states:
row 7 -> req-A
```

则：

```text
req-A:
batch_idx = 1
req_state_idx = 7
```

二者绝对不能假定相等。

这就是 T2A 中：

```python
if int(input_batch.idx_mapping_np[batch_idx]) != req_state_idx:
    raise ValueError(...)
```

存在的原因。

---

# 3. T2A：从 Plan 到 Prepared 的完整逻辑

核心函数：

```python
def _prepare_v2_compactions(
    self,
    compaction_plans: dict[str, CompactionPlanData],
    input_batch: InputBatch,
) -> dict[str, _PreparedCompaction]:
```

一句话定义：

> T2A 不修改 KV，不修改 BlockTable，不修改 persistent E；它只从多个 runtime source 读取事实、核对 Plan、派生 execution 参数，并生成 immutable Prepared descriptor。

---

## 3.1 T2A 的输入从哪里来

表面输入只有：

```text
compaction_plans
input_batch
```

实际上函数内部还读取：

```text
self.req_states
self.block_tables
self.kv_caches
```

完整来源关系：

```text
CompactionPlanData
├─ request_id
├─ expected_source_E
├─ expected_source_num_blocks
├─ keep_member_indices
└─ step_seq
          │
          │
InputBatch│
├─ req_ids
├─ idx_mapping_np
└─ effective_kv_seq_lens
          │
          │
req_states│
└─ req_id_to_index
          │
          │
BlockTables
├─ num_blocks.np
└─ block_tables[0].gpu
          │
          │
self.kv_caches
└─ all layer KV tensor geometry
          │
          ▼
_PreparedCompaction
```

---

# 4. T2A 每个字段具体怎么得到

## 4.1 request identity

首先：

```python
if request_id != plan.request_id:
    raise ValueError(...)
```

避免 dict key 和 Plan 自身 identity 不一致。

这属于 transport/runtime fence，而不是 retention policy。

---

## 4.2 `req_state_idx`

```python
req_state_idx = self.req_states.req_id_to_index.get(request_id)
```

得到 persistent RequestState row。

如果找不到：

```text
Plan 指向了 Worker 当前不存在的 request
```

必须 fail before mutation。

---

## 4.3 `batch_idx`

函数先构造：

```python
batch_indices = {
    request_id: batch_idx
    for batch_idx, request_id in enumerate(input_batch.req_ids)
}
```

然后：

```python
batch_idx = batch_indices.get(request_id)
```

得到当前 step 的 batch row。

随后用：

```python
input_batch.idx_mapping_np[batch_idx]
```

验证：

```text
current batch row
确实映射到
这个 persistent req_state_idx
```

因此：

```text
batch_idx
= transient execution topology

req_state_idx
= persistent state topology
```

---

# 5. `source_effective_kv_len` 为什么从 InputBatch 读

代码：

```python
source_effective_kv_len = int(
    input_batch.effective_kv_seq_lens[batch_idx].item()
)
```

这是整个 T2B timing 设计的关键。

假设 forward 前：

```text
persistent E_before = 32
q = 16
```

本轮 input preparation 会让：

```text
cache_positions = 32 ... 47
forward effective extent = 32 + 16 = 48
```

但这时 persistent：

```text
req_states.effective_kv_len.gpu
```

还可能仍然是 32，因为正常 `post_update()` 尚未执行。

而 V2 要 compact 的 source 包含本轮刚刚写入的 q，因此 source 必须是：

```text
E_source = 48
```

所以 T2A 读取的是：

```text
input_batch.effective_kv_seq_lens[batch_idx]
```

而不是 persistent `req_states.effective_kv_len`。

这正是 V2 必须 post-forward 的根本原因之一。

---

# 6. `.item()`：这里发生了什么 CPU/GPU 数据运动

如果：

```python
input_batch.effective_kv_seq_lens[batch_idx]
```

是 CUDA 上的 scalar tensor：

```text
tensor(48, device='cuda:0')
```

`.item()` 会把这个单元素值取回 Python：

```text
CUDA scalar
↓ device-to-host value extraction
Python scalar 48
```

因此 T2A 当前允许一个小的 control-plane GPU → CPU synchronization point。

这不是 data-plane 优化路径，而是 correctness-first 的 MVP design。

当前阶段不应声称：

```text
T2A is fully asynchronous
```

也不应声称：

```text
no CPU/GPU synchronization
```

正确表述是：

> T2A 为了形成 host-side immutable Prepared descriptor，会读取少量 GPU control state；这可能引入同步开销，但目前没有 profiling，因此不做 performance claim。

---

# 7. `source_num_blocks = num_blocks.np[0, req_state_idx]`

代码：

```python
source_num_blocks = int(
    self.block_tables.num_blocks.np[0, req_state_idx]
)
```

这里 `.np` 是 `num_blocks` 的 host / NumPy-side view。

二维含义：

```text
num_blocks[
    kv_cache_group_id,
    req_state_idx
]
```

当前 MVP：

```text
single KV group
```

所以第一维固定：

```text
0
```

因此：

```python
num_blocks.np[0, req_state_idx]
```

表示：

> KV group 0 下，该 persistent request 当前 active BlockTable 的 block 数。

例如：

```text
req_state_idx = 7
num_blocks.np[0, 7] = 3
```

对应 active row 可能是：

```text
[B7, B2, B11]
```

注意 `.np` 与后面的 `.gpu` 不同：

```text
num_blocks.np
= host-visible metadata

block_tables[0].gpu
= GPU execution-addressing table
```

---

# 8. exact page-count guard：为什么 T2A 必须补

现在 hardened contract：

```python
expected_source_num_blocks = cdiv(
    source_effective_kv_len,
    block_size,
)

if source_num_blocks != expected_source_num_blocks:
    raise ValueError(...)
```

即：

\[
N_{source}
=
\left\lceil \frac{E_{source}}{B} \right\rceil
\]

例如：

```text
E_source = 48
B = 16
```

必须：

```text
source_num_blocks = 3
```

不是“至少能 cover 48”：

```text
4 blocks capacity also enough
```

而是：

```text
exactly 3 active source pages
```

原因不是 vLLM universally 保证 active row 永远 exact，而是：

```text
P1 current MVP
+
M4 exact source-page input contract
```

共同要求这样做。

所以需要明确：

```text
exact page-count equality
= P1 MVP invariant

NOT
= universal vLLM invariant
```

HARDEN-01 已删除后面的 capacity-only 冗余 guard，避免同一位置出现两个看似不同的 contract。

---

# 9. `keep_member_indices` 的验证

Plan 中：

```text
keep_member_indices
```

必须满足：

```text
non-empty
all int
bool rejected
non-negative
strictly increasing
unique implicitly guaranteed by strictly increasing
last < E_source
```

例如：

```text
VALID:
[0, 2, 5, 8, 9]

INVALID:
[0, 5, 2]
[0, 2, 2]
[-1, 2, 5]
[0, 2, 48]  when E_source=48
```

为什么不能自动 sort / dedup？

因为 Plan 的 order 本身就是 semantic contract。

自动修复 Plan 会掩盖 producer bug。

所以原则是：

```text
invalid control decision
→ fail
而不是
→ silently normalize
```

---

# 10. `block_ids_tensor`：真正从 Worker GPU BlockTable 读取 source row

代码逻辑：

```python
block_ids_tensor = self.block_tables.block_tables[0].gpu[
    req_state_idx,
    :source_num_blocks
]
```

假设：

```text
req_state_idx = 7
source_num_blocks = 3
```

可能得到：

```text
CUDA Tensor:
[7, 2, 11]
```

含义：

```text
logical page 0 -> physical block 7
logical page 1 -> physical block 2
logical page 2 -> physical block 11
```

随后：

```python
block_ids = tuple(
    int(block_id)
    for block_id in block_ids_tensor.tolist()
)
```

形成：

```text
(7, 2, 11)
```

---

# 11. `.tolist()`：为什么是 GPU → CPU tuple

这一段的数据流是：

```text
Worker GPU BlockTable row
        ↓ slice
CUDA tensor [7, 2, 11]
        ↓ .tolist()
Python list [7, 2, 11]
        ↓ tuple()
immutable tuple (7, 2, 11)
        ↓
_PreparedCompaction.block_ids
```

`.tolist()` 对 CUDA tensor 来说需要把值 materialize 到 host，因此也是一个 control-plane D2H / synchronization point。

这里的目的不是性能，而是形成：

```text
stable snapshot
+
immutable Prepared descriptor
```

使 T2A 和 T2B 之间的 contract 非常清楚：

```text
T2A:
read live mutable runtime state
→ validate
→ snapshot

T2B:
consume snapshot
```

当前 V2 correctness MVP 接受这部分额外控制开销。

---

# 12. T2A 对 KV cache geometry 的 preflight

对于每层：

```python
for layer_idx, kv_cache in enumerate(self.kv_caches):
```

至少验证：

```text
kv_cache.ndim == 4
kv_cache.shape[2] == block_size
max_block_id < kv_cache.shape[0]
kv_cache.device == block_ids_tensor.device
```

当前 FA2/P1 MVP 中 runner-facing logical view 可以理解为：

```text
[num_blocks, num_kv_heads, block_size, 2 * head_size]
```

但要注意：

```text
logical shape
≠ backing storage physical contiguous order
```

NHD/HND 可以通过 stride/view 表达不同 backing arrangement。

M4 使用 stride-aware 2D Triton path，因此真正重要的是：

```text
logical block dimension
logical head dimension
logical token-within-block dimension
channel dimension
+
correct strides
```

而不是假定所有 layer 都是同一种 contiguous layout。

---

# 13. `self.kv_caches` 从哪里来

`GPUModelRunner.initialize_kv_cache()` 中：

```python
self.kv_caches: list[torch.Tensor] = []

kv_caches_dict = init_kv_cache(
    self.kv_caches,
    ...
)
```

`init_kv_cache()` 内部大致完成：

```text
_allocate_kv_cache()
↓
真正分配 backing GPU storage

_reshape_kv_cache()
↓
根据 attention backend 构造 layer logical views

bind_kv_cache()
↓
把这些 layer KV views 绑定进 runner_kv_caches
```

因此最终：

```text
self.kv_caches
=
[
    layer0 KV view,
    layer1 KV view,
    ...
]
```

T2B：

```python
for kv_cache in self.kv_caches:
```

就是对所有 transformer attention layer 应用同一个 request-level keep decision。

---

# 14. T2A 的最终产物

完成全部 validation 后构造：

```python
_PreparedCompaction(
    request_id=...,
    req_state_idx=...,
    batch_idx=...,
    source_effective_kv_len=...,
    source_num_blocks=...,
    block_ids=...,
    block_size=...,
    keep_member_indices=...,
    new_effective_kv_len=...,
    new_num_blocks=...,
    step_seq=...,
)
```

其中：

```text
K = new_effective_kv_len
  = len(keep_member_indices)

new_num_blocks
  = ceil(K / block_size)
```

到这里必须仍然成立：

```text
KV payload unchanged
Worker BlockTable unchanged
persistent E unchanged
Scheduler state unchanged
no block freed
```

因此 T2A 的一句话定义：

> **T2A 是 destructive boundary 前最后一道完整 preflight fence。**

---

# 15. T2B：真正的 destructive Worker physical transition

核心函数：

```python
@torch.inference_mode()
def _execute_v2_compactions(...)
```

它从 `_PreparedCompaction` 开始，完成：

```text
Prepared
↓
GPU execution tensors materialization
↓
all request × all layer M4 payload compaction
↓
metadata phase
↓
BlockTable prefix stage
↓
E override generation
↓
CompactionResultData generation
```

---

# 16. `@torch.inference_mode()` 是什么

它告诉 PyTorch：

```text
这个函数是 inference-only
不需要 autograd graph
不需要训练相关 tracking
```

它不代表：

```text
自动异步
自动 kernel fusion
自动 CUDA Graph
自动让 Triton 更快
```

这里使用它是合理的，因为 KV compaction 是纯 serving/runtime mutation，不需要梯度。

---

# 17. 为什么 Prepared 中的 tuple 又变回 CUDA tensor

T2A 保存：

```text
block_ids = (7, 2, 11)
keep = (0, 2, 5, 8, 9)
```

T2B 要调用 Triton M4：

```python
block_ids = torch.tensor(
    prepared.block_ids,
    dtype=torch.int32,
    device=self.device,
)

keep = torch.tensor(
    prepared.keep_member_indices,
    dtype=torch.int64,
    device=self.device,
)
```

所以当前控制数据路径确实是：

```text
GPU live BlockTable
↓
CPU immutable Prepared snapshot
↓
new CUDA control tensor
↓
Triton M4
```

这是一条 correctness-first 路径，不是一条优化后的 zero-copy control path。

`torch.tensor(..., device="cuda")` 这里不应被描述成我们专门设计的 async H2D pipeline。

它只是重新 materialize 小型 GPU control tensors。

未来只有 profiling 证明这里是显著 overhead 时，才值得考虑：

```text
device-resident Prepared metadata
reusable control buffers
pinned host buffers
non_blocking explicit H2D
```

当前没有必要提前复杂化。

---

# 18. M4 data plane 做什么

T2B 唯一 production primitive：

```python
compact_paged_kv_triton_2d(
    kv_cache,
    block_ids,
    source_effective_kv_len,
    keep,
)
```

M4 已经冻结的语义：

```text
paged source
↓
gather selected members
↓
independent contiguous scratch
↓
writeback to current paged prefix
↓
zero unused tail in final active page
```

为什么一定要 scratch？

因为 direct arbitrary paged → paged in-place copy 可能 source/destination overlap。

例如某个 destination 提前覆盖了后面还没读取的 source member，就会 corrupt。

因此：

```text
gather
→ independent scratch
→ writeback
```

是 correctness requirement。

---

# 19. canonical example

设：

```text
block_size = 4
E_source = 10
old row = [B7, B2, B11]

physical members:
B7  = [m0, m1, m2, m3]
B2  = [m4, m5, m6, m7]
B11 = [m8, m9, xx, xx]

keep_member_indices = [0, 2, 5, 8, 9]
```

M4 gather：

```text
[m0, m2, m5, m8, m9]
```

writeback：

```text
B7  = [m0, m2, m5, m8]
B2  = [m9, 0, 0, 0]
B11 = unchanged by M4
```

得到：

```text
K = 5
new_num_blocks = ceil(5 / 4) = 2
```

所以新的 Worker active row 必须是：

```text
[B7, B2]
```

而不是重新申请 `[B20, B21]`。

---

# 20. retained block IDs 为什么就是 old row prefix

代码：

```python
retained_block_ids = list(
    prepared.block_ids[:prepared.new_num_blocks]
)
```

例如：

```text
prepared.block_ids = (7, 2, 11)
new_num_blocks = 2
```

得到：

```text
retained_block_ids = [7, 2]
```

不要和：

```text
keep_member_indices = [0, 2, 5, 8, 9]
```

混淆。

二者层级完全不同：

```text
keep_member_indices
= token/member granularity

retained_block_ids
= physical page granularity
```

M4 已经把 retained member 写进原 physical page row 的 prefix，因此 Worker metadata 只需缩短 active row。

---

# 21. `(retained_block_ids,)` 为什么多一层 tuple

调用：

```python
self.block_tables.append_block_ids(
    prepared.req_state_idx,
    (retained_block_ids,),
    overwrite=True,
)
```

如果：

```text
retained_block_ids = [7, 2]
```

则：

```python
(retained_block_ids,)
```

是：

```text
([7, 2],)
```

表示：

```text
KV group 0 -> [7, 2]
```

接口按 KV cache group 接收 block IDs。

若理论上有多个 group，结构可能类似：

```text
(
    [7, 2],       # group 0
    [18, 21, 9],  # group 1
)
```

P1 MVP 只有 single KV group，所以只有一个元素。

这和：

```python
num_blocks.np[0, req_state_idx]
```

中的第一维 `0` 是同一个 group 概念。

---

# 22. `append_block_ids(... overwrite=True)` 不是普通 append

名字容易误导。

在这里：

```python
overwrite=True
```

语义是：

```text
从这个 request row 的 offset 0
stage 新的 active block prefix
```

而不是：

```text
在旧 row 末尾追加 block
```

因此：

```text
old active row:
[B7, B2, B11]

new staged active row:
[B7, B2]
```

Worker 不需要重新 allocator blocks，因为 payload 已经 compact 到 B7/B2 中。

---

# 23. HARDEN-01：为什么必须是“两阶段 transaction ordering”

初版 T2B 是：

```text
for request A:
    compact all layers
    stage A metadata
    append A result

for request B:
    compact all layers
    ...
```

问题：

```text
A all layers success
↓
A BlockTable already staged
↓
B layer 3 fails
```

此时：

```text
A metadata = next state
B payload = partial failure
whole T2B call = exception
```

虽然没有返回 result，但 Worker metadata 已经部分进入 next state。

因此 hardening 后冻结为：

```text
PHASE 1 — payload only

request A × layer0...N
request B × layer0...N
request C × layer0...N
...
↓
ALL payload calls succeeded

PHASE 2 — metadata/result

stage A BlockTable
stage B BlockTable
...
populate E overrides
construct Results
```

这不是 atomic transaction。

因为如果 B layer3 失败：

```text
A payload
以及 B 前几层 payload
可能已经被 destructive modified
```

当前 MVP 不 rollback。

我们能保证的是：

```text
no metadata commit before all payload success
no successful result after payload failure
```

不能保证：

```text
payload unchanged after failure
```

---

# 24. 当前 `_execute_v2_compactions()` 的 final transaction semantics

HARDEN-01 后可以抽象为：

```python
# control outputs
override_valid = zeros[num_reqs]
override_value = zeros[num_reqs]
results = []

# Phase 1
prepared_tensors = []

for prepared in prepared_compactions.values():
    keep = materialize_cuda(...)
    block_ids = materialize_cuda(...)

    for kv_cache in self.kv_caches:
        new_E, new_blocks = compact_paged_kv_triton_2d(...)

        if new_E != prepared.new_effective_kv_len:
            raise RuntimeError(...)

        if new_blocks != prepared.new_num_blocks:
            raise RuntimeError(...)

    prepared_tensors.append(...)

# Phase 2 only if all Phase 1 succeeded
for prepared, ... in prepared_tensors:
    retained = old_row[:prepared.new_num_blocks]

    append_block_ids(... overwrite=True)

    override_valid[prepared.batch_idx] = True
    override_value[prepared.batch_idx] = prepared.new_effective_kv_len

    results.append(
        CompactionResultData(
            ...
            step_seq=prepared.step_seq
        )
    )
```

这里 `prepared_tensors` 的核心意义不是成为长期 state，而只是：

> payload phase 成功后，metadata phase 需要的已准备控制对象。

---

# 25. 为什么 BlockTable metadata 必须 payload 后改

假设 32 layer。

错误：

```text
layer0 compact
↓
BlockTable shrink
↓
layer1...
```

如果 layer5 fail：

```text
BlockTable 已告诉 attention：
active row 是新 prefix

但 layer5..31：
仍然旧 payload
```

payload 和 addressing metadata 立即分裂。

正确：

```text
all layers payload
↓
success
↓
BlockTable stage
```

HARDEN-01 更进一步把范围从：

```text
single request all layers
```

提升为：

```text
all requests × all layers
```

完成后才允许任何 request metadata phase。

---

# 26. Worker BlockTable 为什么可以 staged，而不必 same-step 立即 GPU rewrite

这是当前实现里很重要的 sync/visibility 设计。

`BlockTables` 使用 staged-write machinery。

概念上：

```text
append_block_ids(... overwrite=True)
↓
记录新的 row write descriptor
↓
host-visible active num_blocks 更新
↓
GPU block-table row 的真正 materialization
在 apply_staged_writes() 生命周期执行
```

T2B 发生位置：

```text
model.forward()
↓
T2A
↓
T2B
↓
sampling
↓
post_update
↓
return
```

同一个 step 的 forward 已经结束。

在 T2B 后、下一次 execute_model 前，没有新的 attention consumer 需要使用 compact 后 BlockTable 做本轮第二次 KV addressing。

因此 correctness requirement 是：

```text
next forward before attention
GPU BlockTable new view visible
```

不是：

```text
T2B function return 前
GPU BlockTable bytes 必须立即被覆盖
```

所以复用已有 staged-write path 比新增 immediate mutation API 更合理。

---

# 27. `.np` 已变，不等于 `.gpu` 已变

这点以后调试很容易踩坑。

例如：

```text
append_block_ids(... overwrite=True)
```

可能让：

```text
num_blocks.np
```

立即反映新的 active block count。

但 GPU row 写可能仍是 staged。

因此：

```text
host metadata says 2 blocks
```

不等于你可以立刻在任意 CUDA consumer 中假定：

```text
GPU BlockTable row 已经 materialized 为 [B7, B2]
```

必须遵守 runtime lifecycle：

```text
stage write
↓
apply_staged_writes()
↓
GPU consumer
```

这也是为什么我们明确分析 same-step 是否存在需要读取新 BlockTable 的 consumer。

当前 MVP 判断：没有。

---

# 28. StagedWriteTensor / async copy 应该怎么理解

vLLM 这里的 staged state 设计核心不是“所有东西都有 CPU 和 GPU 两份 authoritative state”，而是：

```text
host side:
记录待执行的 write descriptors / values

GPU side:
真正 serving kernel 使用的 persistent tensor
```

`stage_write` 类 API：

```text
只记录 pending write
不立即修改 GPU runtime tensor
```

后续 `apply_write()` / `apply_staged_writes()`：

```text
把 descriptors / contents 放到 device 可访问位置
↓
在 CUDA stream 上执行更新 kernel / copy
↓
clear staged descriptors
```

现有路径中涉及的 non-blocking copy / same-stream ordering，应理解为：

```text
CPU 发出异步 copy/kernel
不代表后续 GPU consumer 可以无序读取

same CUDA stream
保证：
earlier copy/write
happens-before
later consumer
```

所以“async”不是：

```text
没有顺序
```

而是：

```text
host 不必等待 GPU 完成
+
GPU stream 仍维持明确 execution order
```

---

# 29. T2A 中 GPU→CPU 为什么和 staged async path 是两回事

当前 T2A 的：

```python
.item()
.tolist()
```

是为了立即在 Python control flow 中得到值：

```text
if actual != expected:
    raise
```

这类操作要求 host 看到具体值。

因此通常会形成同步点。

而 staged-write path 的 async copy 目标不同：

```text
host 不需要立即读取 copy 结果
只需要保证 future GPU consumer 前写入完成
```

因此二者不能混为：

```text
“只要用了 CUDA 就都是 async”
```

正确分类：

```text
T2A .item/.tolist
= host consumes device value now
= likely synchronization point

StagedWrite apply / async_copy_to_gpu
= host enqueues future GPU-visible update
= can be non-blocking, ordered by CUDA stream
```

---

# 30. T2B 的 override 为什么保持在 GPU，不再回 CPU

T2B 创建：

```python
override_valid = torch.zeros(
    input_batch.num_reqs,
    dtype=torch.bool,
    device=self.device,
)

override_value = torch.zeros(
    input_batch.num_reqs,
    dtype=torch.int32,
    device=self.device,
)
```

对 V2 request：

```python
override_valid[prepared.batch_idx] = True
override_value[prepared.batch_idx] = K
```

这些 tensor 后续直接：

```text
ExecuteModelState
↓
sample_tokens()
↓
postprocess_sampled()
↓
post_update()
↓
Triton _post_update_kernel
```

所以 E override control path 是：

```text
GPU tensor
→ Python object carries tensor reference
→ GPU kernel pointer
```

没有必要再把 K 取回 CPU 后又写回 persistent E。

这也是选择 batch-indexed GPU override tensor 的好处之一。

---

# 31. 为什么 E 不能在 `_execute_v2_compactions()` 直接写

这是 T2B 最核心设计决策。

普通流程：

```text
E_before
↓
forward writes q tokens
↓
post_update:
E += computed_delta
```

V2 假设：

```text
E_before = 32
q = 16
source after forward = 48

compact source
↓
K = 20
```

正确最终 physical state：

```text
E_next = 20
```

如果 T2B post-forward 直接：

```text
E = 20
```

然后原本 `post_update()` 再：

```text
E += 16
```

就得到：

```text
36
```

这是错误的 double fold-in。

payload 实际只有：

```text
[0, 20)
```

但 metadata 声称：

```text
[0, 36)
```

下一轮 cache position / attention extent 都会错。

---

# 32. 为什么选择 post_update absolute override

我们最终比较过三种方案：

```text
A.
T2B 立即 E=K
+
post_update suppression mask

B.
T2B 不直接 persistent write
+
post_update absolute override E=K

C.
post_update 先 E+=delta
+
之后 repair E=K
```

最终冻结 B。

原因：

```text
post_update 继续作为 persistent E final fold-in owner
normal/V1 path 保留 additive semantics
V2 path 清晰变成 absolute replacement
无需额外“已经写过所以这次别加”的 suppression state
避免 temporary wrong E
```

最终语义：

```python
computed_delta = query_len - num_rejected

if computed_delta != 0:
    num_computed_tokens += computed_delta

if override_valid:
    effective_kv_len = override_value
elif computed_delta != 0:
    effective_kv_len += computed_delta
```

关键：

```text
override_valid branch
不能嵌套在
computed_delta != 0
里面
```

即使：

```text
computed_delta == 0
```

只要 payload compaction 成功：

```text
E = K
```

仍必须 commit。

---

# 33. logical / physical state 正式解耦

原生普通路径隐含：

```text
logical progress
≈
physical progress
```

V2 后必须明确拆开：

\[
L_{next} = L_t + \Delta
\]

physical：

\[
E_{next} =
\begin{cases}
K, & \text{V2 compaction committed}\\
E_t + \Delta, & \text{normal / V1 incremental path}
\end{cases}
\]

所以：

```text
num_computed_tokens
```

继续记录 logical execution progress；

而：

```text
effective_kv_len
```

记录当前真正 physical retained KV extent。

这是整个 P1 从 V1 走向 token-level compaction 后最重要的状态模型变化。

---

# 34. 为什么 override 按 `batch_idx` 建，而 persistent state 按 `req_state_idx` 写

当前 post-update execution topology：

```text
Triton program_id
≈ current batch row
```

概念上：

```python
batch_idx = tl.program_id(0)
req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
```

override：

```text
override_valid[batch_idx]
override_value[batch_idx]
```

persistent：

```text
effective_kv_len[req_state_idx]
num_computed_tokens[req_state_idx]
```

例如：

```text
batch_idx = 1
idx_mapping[1] = 7

override_valid[1] = True
override_value[1] = 20
```

kernel 最终：

```text
effective_kv_len[7] = 20
```

这也是为什么 `req_state_idx != batch_idx` 必须有测试。

---

# 35. `_post_update_kernel` 中的 `if` 会不会造成 warp divergence

当前核心逻辑：

```python
if computed_delta != 0:
    ...

if override_valid:
    E = override
elif computed_delta != 0:
    E += delta
```

这里条件是 per-request / per-program scalar control state。

不是典型：

```text
同一个 program 内
不同 token lane 各自 True/False
```

的 data-dependent lane divergence。

即使编译后存在少量 control-flow/predication overhead，这也是一个小 bookkeeping kernel，不是 GEMM/attention hot kernel。

当前阶段 correctness 优先，没有任何证据表明这里需要为了 branch 做优化。

---

# 36. ExecuteModelState 的最终职责

`ExecuteModelState` 是：

```text
execute_model()
→ subsequent sample_tokens()
```

之间的一次性 ephemeral mailbox。

T2B 初版曾把：

```text
prepared_compactions
```

也放进去。

HARDEN-01 已删除。

原因：

```text
Prepared
= before execution

Result / Override
= after successful execution
```

T2B 已在 execute_model post-forward seam 消费 Prepared，因此后面只需要携带：

```text
effective_kv_len_override_valid
effective_kv_len_override
compaction_results
```

所以最终 lifecycle：

```text
Plan
↓
Prepared
↓
_execute_v2_compactions
↓
Prepared dies

                   ┌──────────────────────┐
                   │                      │
                   ▼                      ▼
              E override             Result
                   │                      │
                   └──── ExecuteModelState┘
                           │
                     sample_tokens
                      /          \
                     /            \
            postprocess         ModelRunnerOutput
                ↓                    ↓
            post_update            Scheduler
```

---

# 37. 为什么 Override 和 Result 必须区分

二者来自同一次 physical compaction，但属于不同 state owner。

## Worker override

目的：

```text
让 Worker 自己下一轮 execution 使用正确 E
```

路径：

```text
_execute_v2_compactions
↓
E override
↓
ExecuteModelState
↓
sample_tokens
↓
postprocess_sampled
↓
post_update
↓
Worker persistent E
```

## CompactionResultData

目的：

```text
通知 Scheduler：
Worker 已经成功从 source state
transition 到 new physical state
```

路径：

```text
_execute_v2_compactions
↓
CompactionResultData
↓
ExecuteModelState
↓
sample_tokens
↓
ModelRunnerOutput
↓
Scheduler
```

一句话：

```text
override
= Worker 给自己改 physical execution state

Result
= Worker 给 Scheduler 的 execution receipt
```

---

# 38. `step_seq` 为什么要透传

HARDEN-01 已让：

```text
CompactionPlanData.step_seq
↓
_PreparedCompaction.step_seq
↓
CompactionResultData.step_seq
```

保持一致。

它不是 block ownership authority，也不是唯一 correctness fence。

但它提供了：

```text
plan/result correlation
step identity
debug/evidence trace
```

几乎零成本，保留比丢掉更合理。

---

# 39. failure semantics：精确边界

必须分两个 region。

## Region A：第一次 destructive M4 write 前

T2A 完整 prevalidation。

如果失败：

```text
KV payload unchanged
BlockTable unchanged
E unchanged
Result none
Scheduler unchanged
```

这是强保证。

## Region B：第一次 destructive M4 write 后

如果后续 layer/request M4 runtime failure：

```text
earlier payload MAY already be modified
```

当前无 rollback。

HARDEN-01 保证：

```text
no metadata phase
no success result
```

即：

```text
BlockTable next-state 不会提前 stage
num_blocks metadata 不进入 next state
override/result 不构造为 completed success
```

但是不能说：

```text
transaction is atomic
```

---

# 40. 为什么 Worker 不 free trailing blocks

canonical example：

```text
old Scheduler ownership:
[B7, B2, B11]

T2B Worker active view:
[B7, B2]
```

此时 B11 虽然已经不是 Worker next attention view 的一部分，但 Scheduler canonical `req_to_blocks` 还可能认为它属于 request。

所以 Worker 若直接 free：

```text
execution layer
越权修改 allocator ownership
```

会破坏 Scheduler authority。

因此 T2B 只生成：

```text
new_num_blocks = 2
CompactionResultData
```

后续 T3 根据 canonical old ownership 推导：

```text
retained = [B7, B2]
removed = [B11]
```

然后 T4 才决定何时 safe free。

---

# 41. T2A/T2B 的 CPU/GPU 数据流全景

可以把当前实现的数据移动分成四类。

```text
A. Host-only metadata read
num_blocks.np[0, req_state_idx]

B. GPU → CPU control snapshot
effective_kv_seq_lens[batch_idx].item()
block_table_gpu_row.tolist()

C. CPU → GPU control materialization
torch.tensor(prepared.block_ids, device=cuda)
torch.tensor(prepared.keep_member_indices, device=cuda)

D. GPU-resident state transition
M4 Triton payload
override tensors
post_update Triton kernel
```

这意味着当前路径不是：

```text
fully device resident
```

而是：

```text
hybrid control plane
+
GPU data plane
```

这对于当前 correctness MVP 完全合理。

---

# 42. 同步 / 异步的准确说法

## 会让 host 立即需要结果的路径

```text
.item()
.tolist()
```

Python 后续要用这些值进行：

```text
if / equality / tuple snapshot
```

因此 host 必须拿到真实 device value。

应视为可能的 synchronization point。

## 不需要 host 立即读取结果的路径

例如 staged BlockTable update：

```text
stage
↓
future GPU apply
↓
future GPU consumer
```

可以利用：

```text
non_blocking copy
same-stream ordering
```

host 不必同步等待每个 copy 完成。

## GPU kernel 之间

只要在同 CUDA stream：

```text
M4 write
↓
later GPU op
```

会按照 stream order 执行。

async 是 host/device overlap 语义，不是取消 dependency。

---

# 43. 为什么现在不优化这些 control-plane round trips

因为当前目标是：

```text
prove correctness
prove lifecycle
prove state authority boundary
```

而不是：

```text
claim low-overhead token compaction
```

任何下列优化都必须以 profiling evidence 为前提：

```text
Prepared device-resident
GPU-side validation
control buffer reuse
pinned host metadata
batched keep arrays
fused multi-layer orchestration
CUDA Graph compatibility
async compaction stream
```

在 T2B acceptance 前提前引入这些，会扩大 bug surface。

---

# 44. T2A/T2B 与 V1 的关系

Normal：

```text
persistent E_t
↓
forward q
↓
post_update
E_next = E_t + delta
```

V1：

```text
pre-forward reclaim
E := R
↓
forward q
↓
post_update
E_next = R + delta
```

V2：

```text
E_t
↓
forward q
↓
source = E_t + q
↓
post-forward compact entire source
↓
K
↓
post_update absolute override
E_next = K
```

所以不能把 V2 直接照搬 V1：

```text
new_E + num_tokens_scheduled
```

因为 V2 的 K 已经包含本轮 forward source 经 retention 后的最终 physical state。

---

# 45. 当前实现 touch points

生产代码：

```text
vllm/v1/worker/gpu/model_runner.py

_PreparedCompaction
_prepare_v2_compactions
_execute_v2_compactions
execute_model post-forward seam
sample_tokens handoff
ExecuteModelState
```

以及：

```text
vllm/v1/worker/gpu/input_batch.py

_post_update_kernel
post_update
```

M4 data plane：

```text
vllm/v1/worker/gpu/kv_compaction.py
```

本轮没有修改其 frozen semantic contract。

---

# 46. HARDEN-01 最终收敛的四个关键点

当前已经完成：

```text
1.
all requests × all layers payload first
then metadata/result

2.
ExecuteModelState.prepared_compactions removed

3.
redundant capacity-only guard removed

4.
step_seq preserved Plan → Prepared → Result
```

因此目前 Prepared 生命周期已经干净：

```text
Plan
→ Prepared
→ Execute
→ Prepared ends
```

---

# 47. 测试与当前 acceptance 状态

最新 evidence：

```text
T2A/T2B focused + M5-T1 transport
26 passed

M4 regression
52 passed
0 skipped

ruff
PASS

py_compile
PASS

git diff --check
PASS
```

HARDEN-01 本身可以判：

```text
PASS
```

但最近一次：

```text
test_gpu_effective_kv_state.py
+
test_gpu_reclaim_commit.py
```

因 NVIDIA driver/NVML 不可用：

```text
29 skipped
```

因此这次 skip 不能作为 current tree 的 CUDA correctness pass evidence。

所以最严谨的项目状态是：

```text
M5-T2A
= PASS / CLOSED / FROZEN

M5-T2B-HARDEN-01
= PASS

M5-T2B implementation
= PASS_CANDIDATE

M5-T2B final acceptance
= pending targeted GPU regression rerun

M5-T3
= NOT STARTED
```

---

# 48. 下一步：不是继续改 T2B，而是补 acceptance gate

GPU/NVML 恢复后，优先跑：

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="$PWD" \
/home/qcrs/learning/llm-kv-lab/.venv/bin/python -m pytest -q \
  tests/v1/worker/test_gpu_effective_kv_state.py \
  tests/v1/worker/test_gpu_reclaim_commit.py
```

如果当前 working tree 得到 CUDA tests pass / 0 unexpected skips，则可以正式：

```text
M5-T2B
= PASS / ACCEPTED / CLOSED / FROZEN
```

然后进入 M5-T3。

---

# 49. M5-T3 将解决什么

T2B 当前已经让 Worker physical execution state 变成：

```text
payload compacted
Worker active BlockTable prefix staged
Worker E = K
```

但 Scheduler 还需要：

```text
CompactionResultData
↓
validate canonical source state
↓
derive retained prefix / removed suffix
↓
req_to_blocks = retained prefix
↓
Scheduler request.effective_kv_len = K
↓
enqueue removed suffix for later safe free
```

关键设计仍然是：

```text
Scheduler V2 E
= absolute K
```

不能复制 V1：

```text
K + scheduled_tokens
```

---

# 50. 最终架构图

```text
                    Scheduler
                       │
                       │ CompactionPlanData
                       ▼
┌──────────────────────────────────────────────────────┐
│ Worker                                               │
│                                                      │
│  T2A                                                 │
│  _prepare_v2_compactions()                           │
│      │                                               │
│      ├─ resolve request / batch indexes               │
│      ├─ read post-forward E_source                    │
│      ├─ read active page count                        │
│      ├─ snapshot GPU BlockTable row                   │
│      ├─ validate keep                                 │
│      ├─ validate exact pages                          │
│      └─ validate all layer geometry                   │
│                 │                                    │
│                 ▼                                    │
│         _PreparedCompaction                          │
│                 │                                    │
│                 ▼                                    │
│  T2B _execute_v2_compactions()                       │
│                                                      │
│  Phase 1                                             │
│  all requests × all layers                           │
│      │                                               │
│      └─ compact_paged_kv_triton_2d                   │
│                 │                                    │
│          ALL PAYLOAD SUCCESS                         │
│                 │                                    │
│  Phase 2                                             │
│      ├─ stage BlockTable retained prefix              │
│      ├─ override_valid[batch_idx] = True              │
│      ├─ override_value[batch_idx] = K                 │
│      └─ CompactionResultData                          │
│                 │                                    │
│                 ▼                                    │
│          ExecuteModelState                           │
└─────────────────┬────────────────────────────────────┘
                  │
                  ▼
            sample_tokens()
              /          \
             /            \
            ▼              ▼
   postprocess_sampled   ModelRunnerOutput
            │              │
            ▼              ▼
       post_update        Scheduler
            │              │
            ▼              │
 Worker persistent E=K     │
                           │
                           ▼
                      future M5-T3
                 canonical reconciliation
                           │
                           ▼
                      future M5-T4
                      safe free/reuse
```

---

# 51. SOURCE_FACT 与 DESIGN_DECISION

| 项目 | 类型 | 说明 |
|---|---|---|
| `batch_idx != req_state_idx` 可能成立 | SOURCE_FACT | current batch topology 与 persistent RequestState topology 不同 |
| `InputBatch.effective_kv_seq_lens` 表示本轮 attention 使用的 current physical extent | SOURCE_FACT | T2A source E 的 runtime basis |
| `num_blocks.np[group, req_state_idx]` 是 host-side active page count view | SOURCE_FACT | Worker BlockTables state |
| Worker GPU BlockTable row 提供 active physical block IDs | SOURCE_FACT | M4 source addressing |
| `self.kv_caches` 是 layer KV execution views | SOURCE_FACT | ModelRunner KV initialization/binding |
| `append_block_ids(... overwrite=True)` 使用 staged BlockTable update | SOURCE_FACT | next-step execution metadata lifecycle |
| T2A exact page equality | DESIGN_DECISION + M4 CONTRACT | P1 MVP fence，不是 universal vLLM invariant |
| T2B post-forward destructive timing | DESIGN_DECISION | source 必须包含本轮 forward 新写入 KV |
| M4 gather→scratch→writeback | FROZEN DATA-PLANE CONTRACT | 避免 overlap corruption |
| all requests × all layers payload before metadata | DESIGN_DECISION | hardening transaction boundary |
| Worker BlockTable 只保留 old-row prefix | DESIGN_DECISION + M4 SEMANTICS | destination 复用 current physical prefix |
| persistent E 由 post_update absolute override | DESIGN_DECISION | 避免 K+delta double fold-in |
| override 按 batch_idx | DESIGN_DECISION | 与 post_update kernel execution topology 对齐 |
| Worker 不 free trailing blocks | AUTHORITY DECISION | Scheduler 是 canonical ownership authority |
| no rollback MVP | EXPLICIT LIMITATION | failure after destructive boundary may leave partial payload |
| no performance claim | EXPLICIT LIMITATION | control-plane round trips 尚未 profiling |

---

# 52. 最值得记住的十句话

1. `CompactionPlanData` 是意图，不是 runtime truth。  
2. T2A 是 destructive boundary 前的 preflight fence。  
3. `_PreparedCompaction` 是 validated execution snapshot，不是长期 state。  
4. `keep_member_indices` 索引 physical member，不索引 block/token ID。  
5. V2 source 必须是 post-forward extent，因此不能像 V1 一样 pre-forward compact。  
6. M4 只处理 payload，不处理 ownership/free。  
7. payload compact 到 old BlockTable physical prefix，所以 Worker row 只需 shrink prefix。  
8. 所有 request × 所有 layer payload 成功后，才能进入 metadata/result phase。  
9. V2 的 K 已是最终 physical length，因此 `post_update` 必须 `E=K`，不能 `K+delta`。  
10. Worker execution truth 在 T2B 收敛，Scheduler ownership truth 要到 T3/T4 才闭环。

---

# 53. 一句话总结

> **M5-T2A 将 Scheduler 的 compaction 意图与 Worker 当前 batch、persistent RequestState、BlockTable 和 layer KV geometry 对齐，形成一个经过完整验证的 immutable execution descriptor；M5-T2B 在本轮 forward 后先对所有 request × 所有 layer 使用 M4 完成真实 KV payload compaction，只有全部 payload 成功后才 stage Worker BlockTable prefix、生成 batch-indexed absolute E override 与 execution receipt。`effective_kv_len` 的 persistent commit 延迟到 `post_update()`，从而把 logical token progression 与 physical retained-KV progression 正式解耦，同时保持 Scheduler ownership/free 权限在后续 T3/T4。**

---

# 54. Evidence / Source Anchors

Current HARDEN-01 evidence:

```text
04-experiments/project1_kv_reclaim/
P1-M5-T2B-HARDEN-01-CODE-CHANGE-RECORD.md

04-experiments/project1_kv_reclaim/raw/
P1-M5-T2B-HARDEN-01.log
```

Hardened source anchors:

```text
vllm/v1/worker/gpu/model_runner.py
  _PreparedCompaction            L129-L140
  _prepare_v2_compactions        L1812-L1978
  _execute_v2_compactions        L1981-L2042
  execute_model T2B seam         L2265-L2289
  ExecuteModelState              L2550-L2563

vllm/v1/worker/gpu/input_batch.py
  _post_update_kernel            L495-L564
  post_update                    L566-L626

tests/v1/worker/test_gpu_compaction_preparation.py
  single request / step_seq      L182-L229
  layer failure                  L232-L264
  multi-request failure          L267-L315
```

Current acceptance caveat:

```text
T2A/T2B + M5-T1: 26 passed
M4:               52 passed / 0 skipped

latest V1/effective-state rerun:
29 skipped because NVIDIA driver/NVML unavailable

=> HARDEN-01 PASS
=> T2B PASS_CANDIDATE
=> final freeze waits for targeted GPU regression evidence
```
