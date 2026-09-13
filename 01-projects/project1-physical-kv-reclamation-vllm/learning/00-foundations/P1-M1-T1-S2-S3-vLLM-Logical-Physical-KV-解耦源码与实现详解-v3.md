# P1-M1-T1 S2→S3：从 vLLM 原生 Position / Slot / Attention 链路到 Logical / Physical KV 解耦

> Project 1 — **Physical KV Cache Reclamation for vLLM**  
> 主题：`P1-M1-T1-S2 Logical Position / Physical Cache Position Split` + `P1-M1-T1-S3 Logical / Effective KV Attention Length Split`  
> 目标：不是单纯记录“改了哪些代码”，而是把我们从 **vLLM 原生链路理解 → 问题暴露 → Tangram 参考核验 → v0.26 Port Mapping → 实现 → Review Fix → 测试闭环** 的全过程重新梳理成一份可复习的源码/项目笔记。
>
> 本文默认已经理解 S1 的基本结论：MRV2 中新增了独立的 Worker GPU persistent state `effective_kv_len`，它与 logical `num_computed_tokens` 在正常执行时按同一 actual delta 推进，但未来 reclaim commit 可以独立缩短。
> **本版新增重点**：§17.1–§17.8 解释字段为什么放在 `RequestState / InputBuffers / InputBatch / CommonAttentionMetadata / FlashAttentionMetadata`；§23.1–§23.7 逐函数追踪 S3 从 `InputBatch` 到 `seqused_k` 的完整 vLLM Attention metadata 链。

---

# 0. 这份笔记到底解决什么问题

S1 结束时，我们只是让系统**有能力保存**下面这种状态：

```text
logical_num_computed_tokens = 128
effective_kv_len            = 64
```

但此时原生 vLLM 的 execution consumer 仍然全部假设：

```text
logical progress == physical KV progress
```

所以 S1 只是建立了第二条 physical state，并没有真正让 runtime 使用它。

S2/S3 真正解决的是两个 execution-side correctness 问题：

```text
S2:
下一批 token 的 K/V 到底写到 KV cache 的哪里？

S3:
FlashAttention 本轮到底应该读取多少个有效 KV？
```

这两个问题都来自同一个根因：

> **原生 vLLM 可以把 logical sequence coordinate 和 physical KV sequence coordinate 共用，是因为 baseline 下二者始终相等；P1 reclaim 以后，这个隐含等式不再成立。**

最终我们需要得到：

```text
                         ┌───────────────────────────────┐
                         │       LOGICAL DOMAIN          │
                         └───────────────────────────────┘

RequestState.num_computed_tokens
               │
               ├── positions
               │      ↓
               │   model / RoPE
               │
               └── seq_lens
                      ↓
                 logical runtime


                         ┌───────────────────────────────┐
                         │      PHYSICAL KV DOMAIN       │
                         └───────────────────────────────┘

RequestState.effective_kv_len
               │
               ├── cache_positions
               │      ↓
               │ compute_slot_mappings
               │      ↓
               │   KV WRITE
               │
               └── effective_kv_seq_lens
                      ↓
               Attention metadata
                      ↓
              FA2 `seqused_k`
                      ↓
                   KV READ
```

这就是 S2 + S3 的最终形态。

---

# 1. 固定基线与 Source Identity

本阶段所有 source fact 都来自正式 implementation worktree：

```text
worktree:
/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim

branch:
p1/physical-kv-reclaim-v026

vLLM baseline:
v0.26.0

commit:
568afb3a13806beb53bb2e6bd518269357b237c0
```

固定 MVP：

```text
A100 80GB
V2 Model Runner / MRV2
TP=1
PP=1
DP=1
DCP=1
PCP=1
block_size=16
FlashAttention2
BF16
prefix caching OFF
spec decode OFF
async scheduling OFF
CUDA Graph OFF
dense causal attention
single KV cache group
```

Tangram reference：

```text
repo:
third_party/tangram

commit:
8e1cbfa5cf82acc67fe1880df0c632f2a6485e35
```

Tangram 在这里是 **source-level reference**，不是整仓 cherry-pick 对象。它的 runtime 更 general，包含 ragged paging、per-head-group effective length 等机制；P1 V1 只裁剪其中和 uniform whole-block reclamation 有关的 contract。

---

# 2. S1 为什么自然把我们带到 S2

先回忆 S1 最关键的结果。

原生 MRV2 的 Worker GPU logical execution state：

```text
RequestState.num_computed_tokens.gpu
```

S1 新增：

```text
RequestState.effective_kv_len.gpu
```

两者语义不同：

| state | semantic |
|---|---|
| `num_computed_tokens` | 模型逻辑上已经计算了多少 token |
| `effective_kv_len` | Worker 物理 KV view 当前保留多少 KV token |

baseline：

```text
logical = 128
physical = 128
```

正常推进一个 token：

```text
129 / 129
```

未来 reclaim 后可以出现：

```text
128 / 64
```

然后正常执行一个 accepted token：

```text
129 / 65
```

关键是：

```text
physical 不是 logical 的 mirror
```

不能做：

```python
effective_kv_len = num_computed_tokens
```

而是两条独立 persistent state，在 normal execution 时共享 actual `computed_delta`。

但 S1 完成后，`effective_kv_len` 还没有 execution consumer。

原生链路仍然是：

```text
num_computed_tokens.gpu
        ↓
prepare_pos_seq_lens()
        ↓
positions
        ├── model / RoPE              # 这里仍然正确
        └── compute_slot_mappings()   # reclaim 后将变错
```

所以 S2 的问题不是“再加一个 state”。

而是：

> **让 physical state 真正接管 KV addressing，同时 logical state 继续负责模型位置。**

---

# 3. 先建立 4 个坐标系：这是理解 S2 最重要的前提

我们前面最容易混淆的是“logical token 位置”“cache position”“physical slot”。

实际上至少要区分四层：

## 3.1 Logical token position

例如：

```text
128
```

表示模型逻辑序列里的第 128 个位置，影响 RoPE / position embedding / model token identity。它不是 HBM 地址。

## 3.2 Cache sequence position

例如：

```text
64
```

表示在当前 active / retained KV sequence 中，这个新 KV 应 append 到第 64 个位置。它仍然不是 raw physical block ID。

## 3.3 Block-table column

若 `block_size=16`：

```text
cache_position = 64
block_table_column = 64 // 16 = 4
block_offset       = 64 % 16  = 0
```

## 3.4 Raw physical block / slot

再通过 request 的 block table：

```text
block_table[req, 4] = physical_block_id
raw_slot = physical_block_id * block_size + block_offset
```

完整映射：

```text
cache_position
      ↓ // block_size
block-table column
      ↓ block_table[row, col]
physical block id
      ↓ * block_size + offset
raw KV slot
```

因此我们后来选择名字 `cache_positions`，而不是 `physical_positions`：它表示的是 **dense KV cache-sequence coordinate**，不是 raw HBM address。

---

# 4. vLLM 原生链路：`execute_model()` 怎么走到 `positions`

S2 source audit 从 `GPUModelRunner.execute_model()` 往下追。

正常真实 batch：

```text
execute_model()
    ↓
prepare_inputs()
    ↓
prepare_attn()
    ↓
model_state.prepare_attn()
    ↓
model forward
```

源码审计确认：

```python
input_batch = self.prepare_inputs(scheduler_output, batch_desc)
block_tables, slot_mappings = self.prepare_attn(input_batch)
```

model input 中：

```python
model_inputs = {
    "input_ids": input_ids,
    "positions": input_batch.positions,
    ...
}
```

所以 `InputBatch.positions` 明确是模型 forward 使用的 position tensor。

这已经提示：

> `positions` 不能为了 KV reclaim 被重新定义成 physical cache position。

---

# 5. `prepare_inputs()`：MRV2 每一步如何组织 batch

`GPUModelRunner.prepare_inputs()` 根据 `SchedulerOutput` 构造：

```text
req_ids
num_scheduled_tokens
idx_mapping
query_start_loc
```

其中：

```text
idx_mapping:
batch row → persistent RequestState row

query_start_loc:
每个 request 在 packed token tensor 中的起始位置
```

例如：

```text
Req A q_len=1
Req B q_len=2

query_start_loc = [0, 1, 3]
```

于是：

```text
query_len =
query_start_loc[i+1] - query_start_loc[i]
```

这些信息随后进入 `prepare_pos_seq_lens()`。

---

# 6. 原生 `_prepare_pos_seq_lens_kernel()` 到底算什么

原生逻辑：

```python
req_state_idx = tl.load(idx_mapping_ptr + req_id)
num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

start = tl.load(query_start_loc_ptr + req_id)
end = tl.load(query_start_loc_ptr + req_id + 1)
query_len = end - start

seq_len = num_computed_tokens + query_len
tl.store(seq_lens_ptr + req_id, seq_len)

for ...:
    pos = num_computed_tokens + block
    tl.store(pos_ptr + start + block, pos)
```

所以：

```text
positions =
num_computed_tokens + local_query_offset

seq_lens =
num_computed_tokens + query_len
```

## 6.1 一个容易混淆但非常重要的点：kernel 是 producer，不是字段 owner

这里最容易产生的误解是：

> “既然 `effective_kv_seq_lens` 是 `InputBatch` 的字段，为什么又说 `_prepare_pos_seq_lens_kernel()` 在处理它？”

答案是：**字段属于哪个对象，和字段由谁计算出来，是两件事。**

可以把这三层先拆开：

```text
_prepare_pos_seq_lens_kernel()
    = producer / 计算器

InputBuffers
    = GPU storage owner / 结果写入的 backing tensor

InputBatch
    = current-step carrier / 当前 forward 对这些 tensor 的语义 view
```

原生 MRV2 也是同样模式。`positions` 和 `seq_lens` 虽然是 `InputBatch` 里后续要使用的字段，但它们并不是在 `InputBatch` dataclass 中“自己算出来”的，而是先由 `_prepare_pos_seq_lens_kernel()` 写进 `InputBuffers`，再由 `prepare_inputs()` 把当前 slice 组织进 `InputBatch`。

原生链路实际上是：

```text
RequestState.num_computed_tokens
        ↓
_prepare_pos_seq_lens_kernel()
        ↓ 写 GPU buffer
InputBuffers.positions / seq_lens
        ↓ slice / view
InputBatch.positions / seq_lens
        ↓
下游 consumer
```

P1 只是把同一个 producer 扩成两套 base：

```text
logical base  = num_computed_tokens
physical base = effective_kv_len
```

于是同一次 kernel launch 再写出：

```text
InputBuffers.cache_positions
InputBuffers.effective_kv_seq_lens
```

随后仍然由 `InputBatch` 携带给下游。

所以一定要记住：

```text
InputBatch.effective_kv_seq_lens
    表示“这个字段属于当前 forward 的 batch semantic”

_prepare_pos_seq_lens_kernel()
    表示“这个字段在 GPU hot path 中由谁生产”
```

两者并不冲突。

## 6.2 为什么这个 producer 最适合计算 `effective_kv_seq_lens`

因为它原本就已经知道：

```text
req_state_idx
query_start_loc
query_len
num_computed_tokens
local token offset
```

S1 再把 `effective_kv_len` 作为第二个 persistent base 传进来以后，它一次就拥有：

```text
logical base
physical base
query_len
local offset
```

所以可以非常机械地计算：

```text
per-token:
positions       = logical_base  + local_offset
cache_positions = physical_base + local_offset

per-request:
seq_lens              = logical_base  + query_len
effective_kv_seq_lens = physical_base + query_len
```

如果反过来等到 `InputBatch` 构造后再用 Python/PyTorch 额外计算 `effective_kv_seq_lens`，会引入额外 op / kernel launch 或 CPU/GPU plumbing，而且会把本来高度对称的 logical/physical metadata producer 拆散。

因此这里的设计原则不是“因为 kernel 名字里有 seq_lens”，而是：

> **这个 kernel 已经是 MRV2 原生 per-forward position/length metadata producer，所以 physical counterpart 也应该在同一 producer 中派生。**

若：

```text
num_computed_tokens = 128
query_len = 4
```

则：

```text
positions = [128,129,130,131]
seq_lens = 132
```

`positions` 是当前 scheduled query token 的 **logical model positions**；`seq_lens` 是包含当前 query 后的 **logical sequence end**。

---

# 7. `InputBatch` 是什么：不是 persistent owner，而是 per-forward view

源码结构可分三层：

```text
RequestState.*
    = request-lifetime persistent state

InputBuffers.*
    = runner 预分配 reusable GPU storage

InputBatch.*
    = 当前 forward 的 per-step semantic view
```

原生关键字段包括：

```text
query_start_loc
seq_lens
seq_lens_cpu_upper_bound
num_computed_tokens_np
positions
...
```

所以 S2/S3 的新字段：

```text
cache_positions
effective_kv_seq_lens
```

天然应该落在 `InputBuffers + InputBatch`，而不是再造 request persistent state。

---

# 8. Source search 结果：为什么 `positions` 不能整体 physical 化

`rg` 搜索 MRV2 中 `positions` 的 consumer 后发现，除了 `model_runner.prepare_attn()`，还存在：

```text
model_inputs["positions"]
sampler
rejection sampler
speculator
cudagraph input
attention metadata positions
PCP path
...
```

说明 `positions` 是广泛的 logical/model semantic tensor。

因此最先被拒绝的方案：

```python
positions = effective_kv_len + offset
```

会让模型 / RoPE position 从 128 错误回退到 64。

---

# 9. 为什么 KV relocation 不能重新编号 RoPE position

假设 logical token 7000 的 K/V 被保留，但在 compact physical layout 中被放到 cache sequence position 2500：

```text
logical token identity = 7000
physical cache position = 2500
```

已缓存的 K 本身已经包含生成时的 positional/RoPE 语义，因此：

```text
physical KV compaction
≠
logical token renumbering
```

P1 改的是 storage layout，不是 model sequence semantics。

---

# 10. `prepare_attn()`：原生 `positions` 为什么又影响 KV WRITE

原生：

```python
block_tables = self.block_tables.gather_block_tables(
    input_batch.idx_mapping,
    num_reqs_padded=input_batch.num_reqs_after_padding,
)

slot_mappings = self.block_tables.compute_slot_mappings(
    input_batch.idx_mapping,
    input_batch.query_start_loc,
    input_batch.positions,
    num_tokens_padded=input_batch.num_tokens_after_padding,
)
```

这就是关键 coupling：

```text
InputBatch.positions
        │
        ├── model/RoPE
        └── compute_slot_mappings
```

baseline 可行，因为 logical position == cache sequence position；reclaim 后不再成立。

---

# 11. `compute_slot_mappings()` 真正做什么

Pinned MRV2 的 slot-mapping kernel 核心公式：

```python
positions = ...
block_indices = positions // block_size
block_offsets = positions % block_size

block_numbers = block_table[
    req_state_idx,
    block_indices
]

slot_ids =
    block_numbers * block_size + block_offsets
```

即：

```text
sequence coordinate
        ↓
column / offset
        ↓
block table
        ↓
physical block
        ↓
raw slot
```

因此真正错的不是 `compute_slot_mappings()` 算法，而是 reclaim 后继续把 logical `positions` 当它的输入。

S2 没有重写 slot-mapping kernel，只改变它接收哪套 coordinate。

---

# 12. `gather_block_tables()` 与 stale-tail hazard

Source audit 还确认：

```text
gather_block_tables()
```

根据 active `num_blocks` 只 gather 当前 active columns。

但 `compute_slot_mappings()` 的 position 输入本身不携带 active `num_blocks` bound。

未来 reclaim 后若：

```text
active columns = 0..3
```

却仍拿 logical position 128：

```text
column = 128 // 16 = 8
```

则可能寻址 backing row 中非 active / stale column。

目前 source 能证明的是：

```text
logical position 可能寻址到 active view 之外的 backing column
```

不能提前夸大为“必然 crash / corruption”；真实 failure mode 要等 real reclaim E2E。

---

# 13. 数值例子：为什么 S2 必须存在

固定：

```text
block_size = 16
logical_num_computed = 128
effective_kv_len = 64
query_len = 1
```

logical model position：

```text
positions = [128]
logical block index = 128 // 16 = 8
```

physical cache append position：

```text
cache_positions = [64]
cache block index = 64 // 16 = 4
offset = 0
```

真正 slot mapping 应使用：

```text
block_table[req, 4]
```

而不是 `block_table[req, 8]`。

因此：

```text
position 128
!=
cache position 64
```

必须成为两套 coordinate。

---

# 14. Tangram 对 S2 给了什么参考

Pinned Tangram `gpu_model_runner.py` 中：

logical positions 仍按：

```python
positions_np =
    num_computed_tokens_cpu[req_indices]
    + arange
```

compression 后不会把 model position 改小。

KV WRITE 则改用：

```python
slot_positions = (
    effective_seq_lens_cpu[req_indices].T
    + arange[None, :]
)

compute_slot_mapping(
    req_indices,
    slot_positions,
)
```

Tangram 源码注释明确说明 compression 后 KV writes target `effective_seq_lens + intra-sequence offset`，它会与 model/prompt position diverge。

所以 Tangram 已给出 exact semantic contract：

```text
logical positions
≠
effective slot positions
```

P1 V1 只是把：

```text
Tangram:
effective_seq_lens[req, head_group]

裁成：

P1:
effective_kv_len[req]
```

因此：

```text
cache_positions =
effective_kv_len + local_offset
```

属于 reference-backed mechanical adaptation。

---

# 15. S2 最终 Design Freeze

```text
num_computed_tokens
        ↓
positions
        ↓
model / RoPE


effective_kv_len
        ↓
cache_positions
        ↓
compute_slot_mappings
        ↓
KV WRITE
```

公式：

```text
positions[token]
=
num_computed_tokens + local_query_offset

cache_positions[token]
=
effective_kv_len + local_query_offset
```

---

# 16. S2 被拒绝的替代方案

1. **直接把 `positions` 改成 physical**：拒绝，RoPE/model logical identity 会错。
2. **修改 `compute_slot_mappings()` 算法**：拒绝，算法正确，错的是输入 coordinate semantic。
3. **Scheduler 生成 physical positions**：拒绝，S2 是 Worker execution-side addressing；Scheduler 后续负责 canonical allocator ownership。
4. **再加第二个 kernel**：未采用。现有 `_prepare_pos_seq_lens_kernel()` 已掌握 `idx_mapping/query_start_loc`，加入 `effective_kv_len` 后可以同一 launch 生成 logical/physical 两套结果。

---

# 17. S2 实现：一个 producer 同时产出四类信息

最终 producer 使用：

```text
logical base:
num_computed_tokens

physical base:
effective_kv_len
```

同一次 Triton launch 生成：

```python
seq_len =
    num_computed_tokens + query_len

effective_kv_seq_len =
    effective_kv_len + query_len

position =
    num_computed_tokens + local_offset

cache_position =
    effective_kv_len + local_offset
```

所以最终不是 S2 kernel + S3 kernel，而是一个 producer 同时输出：

```text
positions
cache_positions
seq_lens
effective_kv_seq_lens
```


---


## 17.1 最重要的问题：这些字段为什么加在这些地方

到这里如果只记住：

```text
加了 cache_positions
加了 effective_kv_seq_lens
```

其实还没有真正理解实现。

真正需要理解的是：

> **一个字段应该放在哪个对象里，不取决于“哪个文件方便改”，而取决于它的 owner、生命周期和 consumer。**

MRV2 这里可以先分成五层：

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. RequestState                                             │
│    request-lifetime persistent truth                        │
│    “这个请求跨 step 持久保存什么状态？”                    │
├─────────────────────────────────────────────────────────────┤
│ 2. InputBuffers                                             │
│    reusable GPU storage                                     │
│    “这一类 per-step tensor 的显存由谁预分配和持有？”       │
├─────────────────────────────────────────────────────────────┤
│ 3. InputBatch                                               │
│    per-forward semantic view                                │
│    “当前这一轮 forward 实际有哪些 active/padded tensor？”  │
├─────────────────────────────────────────────────────────────┤
│ 4. CommonAttentionMetadata                                  │
│    generic attention transport                              │
│    “Runner 如何把本轮 attention 信息交给各 backend？”      │
├─────────────────────────────────────────────────────────────┤
│ 5. FlashAttentionMetadata                                   │
│    backend-specific execution metadata                      │
│    “FA kernel 最终看到的参数是什么？”                       │
└─────────────────────────────────────────────────────────────┘
```

S1/S2/S3 的字段正好分布在这五层。

### 17.1.1 `effective_kv_len` 为什么属于 `RequestState`

`effective_kv_len` 表示进入当前 forward 前这个 request 当前 retained physical KV occupancy。它会跨 step 持续存在，例如：

```text
step N:   logical=128, effective_kv_len=64
step N+1: logical=129, effective_kv_len=65
```

所以它是 request-lifetime persistent state，owner 必须是 `RequestState`。如果把它放在 `InputBatch`，下一轮 batch 重建时 physical truth 就丢失。

### 17.1.2 `cache_positions` 为什么不属于 `RequestState`

`cache_positions` 是：

```text
effective_kv_len + 本轮每个 token 的 local query offset
```

例如 `effective_kv_len=64, q_len=4`，这一轮是 `[64,65,66,67]`；下一轮 physical base 变成 68 后可能只剩 `[68]`。它每轮重新派生，因此是 transient per-forward data，不能成为 `RequestState.cache_positions`。

### 17.1.3 为什么同时有 `InputBuffers.cache_positions` 和 `InputBatch.cache_positions`

这不是重复保存两份数据。MRV2 本来就是：

```text
InputBuffers = GPU tensor storage owner
InputBatch   = 当前 step 的 active slice / semantic carrier
```

原生 `positions` 已经按这种模式存在：

```text
InputBuffers.positions
        ↓ slice
InputBatch.positions
```

S2 只是平行增加：

```text
InputBuffers.cache_positions
        ↓ slice
InputBatch.cache_positions
```

`InputBuffers` 负责预分配和复用 GPU storage，避免每 step 动态分配；`InputBatch` 负责把当前 active/padded view 交给 downstream。

### 17.1.4 `effective_kv_seq_lens` 为什么不是 persistent state

它与 `effective_kv_len` 的时间语义不同：

```text
effective_kv_len
= 进入本轮 forward 前已存在的 physical KV

effective_kv_seq_lens
= 当前 query 的新 KV 也纳入本轮 attention view 后的有效 KV 总长度
```

例如 `effective_kv_len=64, q_len=4`，本轮 `effective_kv_seq_lens=68`。所以前者是 persistent base，后者是 per-forward derived metadata。

### 17.1.5 为什么 `cache_positions` 是 per-token，而 `effective_kv_seq_lens` 是 per-request

shape 本身就反映 semantic：

```text
cache_positions: [num_tokens_after_padding]
effective_kv_seq_lens: [num_reqs_after_padding]
```

每个 scheduled token 都有自己的 append coordinate；而每个 request 只需要给 Attention 一个本轮有效 KV 总长度。因此它们分别与原生字段构成对称：

```text
positions  ↔ cache_positions
seq_lens   ↔ effective_kv_seq_lens
```

## 17.2 S1/S2/S3 字段所有权总表

| 字段 | Owner | 生命周期 | 粒度 | 为什么放这里 |
|---|---|---|---|---|
| `num_computed_tokens` | `RequestState` | persistent | per-request | logical progress 跨 step 持续存在 |
| `effective_kv_len` | `RequestState` | persistent | per-request | retained physical occupancy 跨 step 持续存在 |
| `positions` backing storage | `InputBuffers` | reusable | per-token | GPU 热路径预分配 |
| `cache_positions` backing storage | `InputBuffers` | reusable | per-token | physical coordinate 同样需要 GPU 预分配 |
| `seq_lens` backing storage | `InputBuffers` | reusable | per-request | 本轮 logical length |
| `effective_kv_seq_lens` backing storage | `InputBuffers` | reusable | per-request | 本轮 physical read extent |
| `positions` | `InputBatch` | per-forward | per-token | 当前 model/RoPE view |
| `cache_positions` | `InputBatch` | per-forward | per-token | 当前 KV write view |
| `seq_lens` | `InputBatch` | per-forward | per-request | 当前 logical request view |
| `effective_kv_seq_lens` | `InputBatch` | per-forward | per-request | 当前 physical KV visibility |
| `effective_kv_seq_lens` | `CommonAttentionMetadata` | per-attention-build | per-request | generic runner→backend transport |
| selected physical length | `FlashAttentionMetadata.seq_lens` | backend execution | per-request | FA 已用此字段作为 `seqused_k` |

## 17.3 S2 字段真正从哪里出生，到哪里结束

```text
RequestState.effective_kv_len
        │ persistent physical base
        ▼
GPUModelRunner.prepare_inputs()
        │ 把 physical base 交给 producer
        ▼
_prepare_pos_seq_lens_kernel()
        │ effective_kv_len + local token offset
        ▼
InputBuffers.cache_positions
        │ current-step slice
        ▼
InputBatch.cache_positions
        ▼
GPUModelRunner.prepare_attn()
        ▼
BlockTables.compute_slot_mappings()
        ▼
slot_mapping
        ▼
reshape_and_cache_flash()
        ▼
KV WRITE
```

`cache_positions` 在生成 `slot_mapping` 后就完成使命，因此不需要继续进入 `CommonAttentionMetadata` 或 `FlashAttentionMetadata`。

## 17.4 S3 字段为什么比 S2 多穿几层

S3 真正 consumer 不在 `prepare_attn()`，而在：

```text
FlashAttentionImpl.forward()
→ seqused_k
```

所以它必须经过：

```text
InputBatch
→ DefaultModelState.prepare_attn
→ build_attn_metadata
→ CommonAttentionMetadata
→ FlashAttentionMetadataBuilder
→ FlashAttentionMetadata
```

这不是 P1 额外绕路，而是 vLLM 原生 attention metadata architecture 的标准 transport path。

## 17.5 为什么 S2/S3 最适合一起在 `_prepare_pos_seq_lens_kernel()` 生成

这个 kernel 原本已经拥有 `idx_mapping`、`query_start_loc`、`num_computed_tokens`，即它已经知道 request、query_len、logical base 和 local offset。S1 再提供 `effective_kv_len` 后，它同时拥有 logical/physical 两个 base。

所以一次 launch 就可产生：

```text
positions
cache_positions
seq_lens
effective_kv_seq_lens
```

如果拆成两个 kernel，会重复读取 `idx_mapping/query_start_loc` 并额外增加一次 launch。这里合并的本质是：两套 metadata 使用相同 local offset/query_len，只是 base 不同。

## 17.6 128 / 64 / q=4：按对象跟着真实数据跑一遍

### Persistent request state

```text
RequestState
num_computed_tokens = 128
effective_kv_len    = 64
```

### `prepare_inputs()` 构造当前 query layout

```text
query_len = 4
query_start_loc = [0,4]
```

### Producer 写入 `InputBuffers`

```text
positions = [128,129,130,131]
cache_positions = [64,65,66,67]
seq_lens = [132]
effective_kv_seq_lens = [68]
```

### `InputBatch` 只是拿当前 slice/view

```text
positions             → model/RoPE
cache_positions       → prepare_attn → slot mapping
seq_lens              → logical runtime
effective_kv_seq_lens → attention metadata → FA read extent
```

同一轮里 128..131、64..67、132、68 同时存在且各司其职。

## 17.7 为什么这些字段不能“为了简单”重新合并

reclaim 后 logical/physical 会长期 divergence。若为了少一个字段重新用 `positions` 或 `seq_lens` 同时表达两种 semantic，就会重新引入 P1 正在消除的 baseline coupling。因此字段拆分不是临时 hack，而是 logical/physical decoupling 的必要数据模型。

## 17.8 Producer → Storage → Carrier → Consumer：把 `effective_kv_seq_lens` 从出生到使用完整追一遍

这一节专门回答一个容易混淆的问题：

> `effective_kv_seq_lens` 明明是 `InputBatch` 使用/携带的字段，那 `_prepare_pos_seq_lens_kernel()` 到底是干嘛的？

正确理解是五个角色：

```text
1. Persistent source/base
   RequestState.effective_kv_len

2. Producer
   _prepare_pos_seq_lens_kernel()

3. Storage owner
   InputBuffers.effective_kv_seq_lens

4. Per-forward carrier
   InputBatch.effective_kv_seq_lens

5. Consumer
   Attention metadata → FA seqused_k
```

### 17.8.1 Step A：`RequestState.effective_kv_len` 只是 base，不是最终 Attention length

例如：

```text
forward entry:
effective_kv_len = 64

current query_len = 4
```

`64` 只说明：

```text
本轮执行前已经存在 64 个 retained KV
```

但本轮 Attention 执行时，当前 4 个 query token 的新 K/V 也已经属于有效 KV view，所以真正 read extent 是：

```text
64 + 4 = 68
```

这就是为什么不能直接把 `effective_kv_len` 传到 FA 当 `seqused_k`。

### 17.8.2 Step B：kernel 负责把 persistent base 转成 per-forward derived metadata

`_prepare_pos_seq_lens_kernel()` 运行时读取：

```text
num_computed_tokens
effective_kv_len
query_start_loc
idx_mapping
```

对于 request：

```text
logical_base  = 128
physical_base = 64
query_len     = 4
```

它在 GPU 上直接写：

```text
positions = [128,129,130,131]
cache_positions = [64,65,66,67]
seq_lens = [132]
effective_kv_seq_lens = [68]
```

这里 kernel 只是**生产者**。kernel launch 结束后，它本身不会“持有”任何字段。

### 17.8.3 Step C：结果实际写在 `InputBuffers`

GPU memory owner 是：

```text
self.input_buffers.positions
self.input_buffers.cache_positions
self.input_buffers.seq_lens
self.input_buffers.effective_kv_seq_lens
```

可以把 `InputBuffers` 想成 runner 事先准备好的工作台。kernel 每轮把新的结果写进这些固定 buffer，而不是每轮重新申请一套 tensor。

因此：

```text
kernel = 填数据的人
InputBuffers = 真正放数据的 GPU backing storage
```

### 17.8.4 Step D：`InputBatch` 只是把当前有效 slice 包装成当前 forward 的语义对象

`prepare_inputs()` 随后构造：

```python
InputBatch(
    ...
    positions=self.input_buffers.positions[:num_tokens_after_padding],
    cache_positions=self.input_buffers.cache_positions[:num_tokens_after_padding],
    seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
    effective_kv_seq_lens=(
        self.input_buffers.effective_kv_seq_lens[:num_reqs_padded]
    ),
    ...
)
```

所以 `InputBatch` 并没有重新计算 `68`。

它做的是：

```text
“这一轮有效的 physical KV length tensor 就是 backing buffer 的这一段。”
```

因此：

```text
InputBatch = carrier / semantic view
不是 producer
```

### 17.8.5 Step E：S2 和 S3 从 `InputBatch` 开始分叉

S2：

```text
InputBatch.cache_positions
        ↓
GPUModelRunner.prepare_attn()
        ↓
BlockTables.compute_slot_mappings()
        ↓
slot_mapping
        ↓
reshape_and_cache_flash()
        ↓
KV WRITE
```

S3：

```text
InputBatch.effective_kv_seq_lens
        ↓
DefaultModelState.prepare_attn()
        ↓
build_attn_metadata()
        ↓
CommonAttentionMetadata.effective_kv_seq_lens
        ↓
FlashAttentionMetadataBuilder
        ↓
FlashAttentionMetadata.seq_lens
        ↓
seqused_k
        ↓
KV READ
```

这解释了为什么同一个 producer 生成 S2/S3 两类 physical metadata，但它们在 `InputBatch` 之后走向不同 consumer。

### 17.8.6 用一句工程化的话概括各层职责

```text
RequestState
回答：请求跨 step 的真实状态是什么？

_prepare_pos_seq_lens_kernel
回答：基于这些 persistent state，本轮 execution metadata 应该是多少？

InputBuffers
回答：这些 GPU tensor 的 backing storage 放在哪里？

InputBatch
回答：当前 forward 实际使用 backing storage 的哪一段，以及这些 tensor 的语义是什么？

Attention / BlockTables
回答：这些 metadata 最终如何驱动 KV WRITE / READ？
```

### 17.8.7 为什么 kernel 名字现在看起来不够完整

原生 kernel 只生成：

```text
positions
seq_lens
```

所以叫：

```text
_prepare_pos_seq_lens_kernel
```

P1 扩展后它还生成：

```text
cache_positions
effective_kv_seq_lens
```

从功能上现在更接近：

```text
_prepare_logical_physical_pos_seq_lens_kernel
```

但我们没有为命名做大范围 rename，因为：

```text
minimal patch > naming cleanup
```

因此以后读源码时，不要因为函数仍叫 `prepare_pos_seq_lens` 就误以为它只处理 logical metadata。


---

# 18. S2 实际修改的代码面

## 18.1 `vllm/v1/worker/gpu/input_batch.py`

新增 GPU buffers：

```text
InputBuffers.cache_positions
InputBuffers.effective_kv_seq_lens
```

新增 `InputBatch` fields：

```text
cache_positions
effective_kv_seq_lens
```

扩展：

```text
_prepare_pos_seq_lens_kernel()
prepare_pos_seq_lens()
```

并保留 backward-compatible fallback：

```text
cache_pos is None
→ cache_pos = pos

effective_kv_seq_lens is None
→ effective_kv_seq_lens = seq_lens

effective_kv_len is None
→ effective_kv_len = num_computed_tokens
```

即旧 caller 自动退化成：

```text
logical == physical
```

## 18.2 `vllm/v1/worker/gpu/model_runner.py`

`prepare_inputs()` 将：

```text
RequestState.effective_kv_len.gpu
```

送入 producer。

构造 `InputBatch` 时发布：

```text
cache_positions
effective_kv_seq_lens
```

S2 最关键的 consumer substitution：

```text
Before:
compute_slot_mappings(... input_batch.positions ...)

After:
compute_slot_mappings(... input_batch.cache_positions ...)
```

这就是 S2 真正改变 runtime behavior 的地方。

---

# 19. 为什么 S2 做完还不够：S3 是怎么暴露出来的

假设：

```text
logical = 128
physical = 64
query_len = 1
```

S2 做完后：

```text
model position = 128
KV append position = 64
```

新 K/V 已经写到正确 physical cache frontier。

但写完后真实 cache view 是：

```text
64 old KV + 1 new KV = 65 KV
```

如果 Attention 仍然收到：

```text
seq_lens = 129
```

它还是会按 129 的 K length 解释 paged cache。

因此：

```text
WRITE address 正确
≠
READ extent 正确
```

S3 专门解决 read extent。

---

# 20. S3 第一轮 Source Audit：`seq_lens` 到底是什么

原生 producer 已证明：

```text
seq_lens =
num_computed_tokens + query_len
```

因此 reclaim 后：

```text
logical=128
physical=64
q=1

seq_lens 仍然应该是 129
```

关键在于：

> `129` 是 logical request length，不再是 physical KV occupancy。

所以必须继续追 consumer，不能因为名字叫 `seq_lens` 就整体改成 physical。

---

# 21. `seq_lens` 有非 Attention consumer：这是 S3 最关键的 source fact 之一

Source audit 找到：

```text
combine_sampled_and_draft_tokens
get_num_sampled_and_rejected
```

等逻辑会使用 `seq_lens` 做类似：

```python
seq_len = ...
prefill_len = ...

if seq_len <= prefill_len:
    ...

is_chunked_prefilling =
    seq_len < prefill_len
```

这些明显在判断：

```text
request logical progression
prefill lifecycle
```

而不是：

```text
物理 KV cache 还剩多少 token
```

因此方案：

```text
InputBatch.seq_lens = effective physical length
```

被正式否决。

S3 的核心原则：

> **global / generic `seq_lens` 不能被整体重定义。**

---

# 22. `seq_lens_cpu_upper_bound` 为什么也保留 logical

`prepare_inputs()` 还会算：

```python
seq_lens_cpu_upper_bound =
    num_computed_tokens_np + num_scheduled_tokens
```

`DefaultModelState.prepare_attn()` 用它得到：

```text
max_seq_len
```

而 `CommonAttentionMetadata` 的注释明确允许：

```text
max_seq_len may be an upper bound
```

只要：

```text
0 <= effective_kv_len <= num_computed_tokens
```

就有：

```text
effective_kv_seq_lens <= logical seq_lens
```

所以 logical `max_seq_len` 仍然是 physical KV length 的安全上界。

因此 S3 没有引入：

```text
effective_kv_len CPU mirror
GPU→CPU synchronization
physical max_seq_len
```

---

# 23. Attention metadata 链：从 `InputBatch.seq_lens` 到 FA

Pinned v0.26 normal path：

```text
InputBatch.seq_lens
        ↓
DefaultModelState.prepare_attn()
        ↓
build_attn_metadata()
        ↓
CommonAttentionMetadata.seq_lens
        ↓
FlashAttentionMetadataBuilder.build()
        ↓
FlashAttentionMetadata.seq_lens
        ↓
FlashAttentionImpl.forward()
        ↓
seqused_k
        ↓
flash_attn_varlen_func(...)
```

这是 S3 必须读通的完整 transport 链。


## 23.1 把 Attention 链真正展开：每一跳到底做什么

上面的链如果只背函数名仍然不够。下面按“caller / 输入 / 这一层职责 / 输出 / 为什么 S3 必须经过这里”展开。

### 23.1.1 `GPUModelRunner.prepare_inputs()`

这一层属于 general model execution input preparation。它已经产生：

```text
InputBatch.seq_lens
InputBatch.effective_kv_seq_lens
```

但它不决定“FA 最终使用哪一个”，因为 GPUModelRunner 是通用 runner，不应把 FA-specific lowering 写死在这里。

### 23.1.2 `GPUModelRunner.prepare_attn()`

这里先处理 paged-KV addressing：

```text
gather_block_tables()
compute_slot_mappings()
```

S2 在这里完成 `cache_positions → slot_mapping`。但 S3 的 effective length 不在这里直接消费，因为“读多少 KV”属于 attention backend metadata，而不是 BlockTables addressing。

### 23.1.3 `DefaultModelState.prepare_attn()`

它拿到 `InputBatch + block_tables + slot_mappings`，整理 generic attention build 参数：

```text
query_start_loc
max_query_len
seq_lens
seq_lens_cpu_upper_bound
positions
...
```

S3 只新增一件事：把 `input_batch.effective_kv_seq_lens` 传给 `build_attn_metadata()`。这里不重新计算，只做 ModelState-level adapter / transport。

### 23.1.4 `build_attn_metadata()`

该函数按 KV-cache group 构造 `CommonAttentionMetadata`，然后调用各 `AttentionMetadataBuilder`。原生 `seq_lens` 会先 slice 到 `num_reqs`；S3 的 `effective_kv_seq_lens` 也遵循同一 slicing contract，再进入 Common metadata。

为什么这里要 slice？因为 `InputBuffers` 可能按 padded/max capacity 存储，而本轮 attention build 只处理有效 `num_reqs` 范围。

### 23.1.5 `CommonAttentionMetadata`

它是 Runner/ModelState 与具体 Attention backend 之间的 generic ABI。这里必须同时保留：

```text
seq_lens = logical
effective_kv_seq_lens = optional physical
```

不能直接把 `seq_lens` 改成 physical，因为 source search 已证明它还有 `compute_num_computed_tokens()` 以及 Triton/Mamba/FlexAttention/ROCm/FlashInfer/MLA 等 generic consumers。

### 23.1.6 `FlashAttentionMetadataBuilder.build()`

这是 S3 真正发生 semantic substitution 的地方，因为这里已经进入 FlashAttention-specific lowering。

当前 scope guard：

```text
effective field exists
AND dcp_world_size == 1
AND not use_cascade
```

满足时：

```text
FlashAttention backend seq_lens = effective_kv_seq_lens
```

否则回退：

```text
FlashAttention backend seq_lens = logical seq_lens
```

所以 semantic change 被限制在 backend boundary，而不是污染 generic Common metadata。

### 23.1.7 `FlashAttentionMetadata`

到了这里，它已经是 kernel-facing backend execution metadata。于是：

```text
CommonAttentionMetadata.seq_lens
= logical

FlashAttentionMetadata.seq_lens
= P1 normal path 下的 physical KV length
```

两个同名字段位于不同 abstraction layer，语义不同是有意设计，不是冲突。

### 23.1.8 `FlashAttentionImpl.forward()`

forward 原生：

```python
seqused_k = attn_metadata.seq_lens
max_seqlen_k = attn_metadata.max_seq_len
block_table = attn_metadata.block_table
```

因此 S3 不需要改 forward ABI 或 `flash_attn_varlen_func()`。builder 已经把正确 physical tensor 放进 `FlashAttentionMetadata.seq_lens`，kernel 完全不需要知道 P1 的存在。

## 23.2 `seqused_k` 与 block table 如何共同描述 paged KV READ

Paged KV READ 实际回答两个问题：

```text
1. 第 N 个 cache token 在哪个 physical page？
   → block_table

2. 这个 request 有多少个 cache token 有效？
   → seqused_k
```

所以：

```text
block_table = address translation
seqused_k   = valid extent
```

S2 解决 write-side address generation；S3 解决 read-side valid extent。只改 block table/slot mapping 而不改 `seqused_k`，Attention 仍可能认为更长 tail 有效。

## 23.3 为什么 `max_seq_len` 不需要复制成 physical exact 值

`seqused_k` 是 exact per-row valid length；`max_seq_len/max_seqlen_k` 是 safe upper bound / planning information。只要：

```text
effective_kv_len <= num_computed_tokens
```

就有：

```text
effective_kv_seq_lens <= logical seq_lens
```

所以 logical max 仍是 physical length 的安全上界。当前 V1 因此不增加 physical CPU mirror、GPU→CPU sync 或新的 effective max field。

## 23.4 为什么 S3 不是“给 FA 再加一个字段”

如果直接新增 `FlashAttentionMetadata.effective_kv_seq_lens`，还要继续修改 `FlashAttentionImpl.forward()` 和可能的 kernel-facing helper。现在直接复用 FA 已有的：

```text
FlashAttentionMetadata.seq_lens → seqused_k
```

只在 builder 决定这个字段本轮装 logical 还是 physical tensor，因此 kernel ABI 完全不变。

## 23.5 S3 transport 的完整对象图

```text
RequestState.effective_kv_len
        │ persistent base
        ▼
_prepare_pos_seq_lens_kernel
        ▼
InputBuffers.effective_kv_seq_lens
        │ slice
        ▼
InputBatch.effective_kv_seq_lens
        ▼
DefaultModelState.prepare_attn
        ▼
build_attn_metadata
        ▼
CommonAttentionMetadata.effective_kv_seq_lens
        │ backend-local selection
        ▼
FlashAttentionMetadata.seq_lens
        ▼
FlashAttentionImpl.forward
        ▼
seqused_k
        ▼
flash_attn_varlen_func
```

## 23.6 为什么 `CommonAttentionMetadata.unpadded()` 是 source audit 必查点

新增 dataclass field 不能只搜 constructor，还要搜 copy/replace/slice/rebuild/split/dummy/capture。此次 source search 找到 `replace()`、`unpadded()` 和多个 `CommonAttentionMetadata(...)` construction sites。

`replace()` 走 dataclass replace，会自然保留未覆盖字段；但 `unpadded()` 显式重新调用 constructor，所以必须手动传播：

```text
effective_kv_seq_lens = maybe_slice_reqs(...)
```

否则 physical channel 会 silently disappear。

## 23.7 这条链对理解 vLLM Attention 架构有什么帮助

通过 S3 可以把 attention architecture 分成：

```text
GPUModelRunner
    负责 execution batching

ModelState
    负责 model/generic attention adaptation

CommonAttentionMetadata
    负责 backend-neutral transport

AttentionMetadataBuilder
    负责 backend-specific lowering

FlashAttentionMetadata / forward
    负责 kernel-facing execution
```

以后遇到 attention 需求时，应先判断它属于 request state、per-step batch metadata、generic attention semantic，还是 backend-specific lowering，而不是直接冲到 `flash_attn.py` 改 kernel。


---

# 24. `CommonAttentionMetadata.seq_lens` 为什么必须继续 logical

`CommonAttentionMetadata` 不是只给 FA 用的 dumb struct。

它还有：

```python
def compute_num_computed_tokens(self):
    query_lens = ...
    return self.seq_lens - query_lens
```

Source search 还发现很多 backend 直接消费：

```text
common_attn_metadata.seq_lens
```

包括：

```text
FlashAttention
Triton attention
ROCm
FlexAttention
Mamba
FlashInfer
MLA
CPU backend
...
```

所以如果：

```text
CommonAttentionMetadata.seq_lens = physical
```

影响面会从 P1 normal FA2 瞬间扩大到多个未审计 backend，还会让 generic `compute_num_computed_tokens()` 从 logical 语义变成 physical。

最终设计因此是：

```text
CommonAttentionMetadata.seq_lens
    = logical

CommonAttentionMetadata.effective_kv_seq_lens
    = optional physical channel
```

---

# 25. FA2 最终真正读哪个长度

Pinned normal FA path：

```python
seqused_k = attn_metadata.seq_lens
max_seqlen_k = attn_metadata.max_seq_len
block_table = attn_metadata.block_table
```

然后：

```python
flash_attn_varlen_func(
    ...
    seqused_k=seqused_k,
    max_seqlen_k=max_seqlen_k,
    block_table=block_table,
    ...
)
```

所以 paged KV READ 可以抽象成：

```text
block_table + seqused_k
```

其中：

```text
block_table
决定 physical pages 如何映射

seqused_k
决定每个 request 有多少有效 K/V
```

这就是 S3 的 exact consumer。

---

# 26. KV WRITE 与 KV READ 是两条不同路径

## WRITE

```text
cache_positions
    ↓
compute_slot_mappings
    ↓
slot_mapping
    ↓
reshape_and_cache_flash
```

也就是：

```text
KV WRITE
由 slot_mapping 决定
```

## READ

```text
block_table
+
effective sequence length
    ↓
flash_attn_varlen_func
```

也就是：

```text
KV READ
由 page translation + valid length 决定
```

因此：

```text
S2 = 写地址
S3 = 读范围
```

是两个独立 correctness surface。

---

# 27. Tangram 对 S3 给了什么 reference

Tangram 在 ragged/compression 模式维护：

```text
effective_seq_lens_cpu
```

shape：

```text
[num_reqs, num_head_groups]
```

因为它支持不同 head-group 拥有不同 retained length。

P1 V1 只做 uniform whole-block view，所以只需要：

```text
[num_reqs]
```

---

# 28. Tangram 的 effective length 是 pre-step 还是 post-step

Port audit 专门核实：

```text
Tangram effective_seq_lens_cpu
=
当前 step 开始前，
compression 后已经保留的 cache occupancy
```

builder 再做：

```text
effective + current scheduled tokens
```

源码语义：

```python
seq_lens_cluster =
    effective_seq_lens
    + num_scheduled
```

这与 P1 S1 的：

```text
effective_kv_len
=
forward-entry physical occupancy
```

直接对应。

所以：

```text
effective_kv_seq_lens
=
effective_kv_len + query_len
```

不是现场猜出来的，而是 Tangram formula 的 uniform specialization。

---

# 29. Tangram 为什么不能直接整套搬过来

Tangram 后续更复杂：

```text
effective_seq_lens_cpu
        ↓
per-cluster length
        ↓
seq_lens_grouped
        ↓
per-member / per-layer view
        ↓
layer-specific FA metadata
```

因为它做：

```text
per-head-group ragged KV
```

例如：

```text
group0 = 64
group1 = 47
group2 = 80
```

P1 V1 是：

```text
所有 KV heads 共用一个 request-level effective length
```

所以不需要：

```text
cluster map
ragged block table
seq_lens_grouped
member virtual block table
per-layer overlay
```

准确关系是：

```text
Tangram general contract
      ↓
P1 uniform specialization
```

---

# 30. S3 Port Mapping Audit：哪些是 Exact，哪些是 Adaptation

| Tangram | P1 v0.26 | 分类 |
|---|---|---|
| logical sequence progress | `num_computed_tokens` | A / exact semantic |
| effective physical occupancy | `effective_kv_len` | B / MRV2 adaptation |
| logical `seq_lens` 继续存在 | `InputBatch.seq_lens` | A |
| `effective + scheduled` | `effective_kv_len + query_len` | A/B |
| separate effective metadata channel | `effective_kv_seq_lens` | B/C glue |
| `seq_lens_grouped` | 删除，不需要 | generality removal |
| per-head-group lengths | uniform per-request length | B |
| per-layer physical FA seq lens | normal FA2 selected length | B/C |
| FA kernel ABI | 不变 | A |

最重要的审计结论：

```text
没有出现 D — novel runtime mechanism
```

S3 是 reference-backed port，不是临时发明。

---

# 31. S3 最终设计

新增：

```text
effective_kv_seq_lens
```

公式：

```text
effective_kv_seq_lens
=
effective_kv_len + query_len
```

transport：

```text
effective_kv_len
      ↓
_prepare_pos_seq_lens_kernel
      ↓
effective_kv_seq_lens
      ↓
InputBatch
      ↓
DefaultModelState.prepare_attn
      ↓
build_attn_metadata
      ↓
CommonAttentionMetadata.effective_kv_seq_lens
      ↓
FA2 builder
      ↓
FlashAttentionMetadata.seq_lens
      ↓
seqused_k
```

而：

```text
InputBatch.seq_lens
CommonAttentionMetadata.seq_lens
```

继续 logical。

---

# 32. 为什么 physical channel 要等到 FA builder 才接管

在 generic metadata 层直接重写 `seq_lens` 会污染所有 backend。

所以采用：

```text
Common metadata:
    logical seq_lens
    optional effective_kv_seq_lens

              ↓

FA2 builder:
    根据批准 scope 选择 backend 最终 length
```

blast radius 被限制在 normal pinned FA2。

---

# 33. `CommonAttentionMetadata.unpadded()` 为什么也要改

Source audit 发现：

```text
replace()
```

走 dataclass replace，一般会保留新增 field。

但：

```text
CommonAttentionMetadata.unpadded()
```

会显式重新构造 `CommonAttentionMetadata(...)`。

若不处理，新增：

```text
effective_kv_seq_lens
```

会 silent drop。

因此增加：

```text
effective_kv_seq_lens =
    maybe_slice_reqs(self.effective_kv_seq_lens)
```

这是典型 version glue，也提示后续改 runtime dataclass 时必须搜：

```text
field definition
+
all constructors
+
explicit reconstruction
+
slice/copy/replace paths
```

---

# 34. Dummy path 为什么 `logical == physical`

`InputBatch.make_dummy()` 模拟 fresh shape：

```text
base = 0
seq_len = query_len
```

因此 dummy：

```text
cache_positions = positions
effective_kv_seq_lens = seq_lens
```

不需要 invent 新 semantic。

---

# 35. S3 Review Fix：为什么第一版还不能直接 PASS

第一版实现中 `kv_seq_lens` 不仅影响 normal FA，还顺手进入：

```text
DCP branch
cascade branch
```

一度出现：

```text
DCP:
context_kv_lens = kv_seq_lens - query_lens

Cascade:
suffix_kv_lens = kv_seq_lens - common_prefix_len
```

这从直觉上不一定错，但我们没有审计：

```text
DCP + reclaim
cascade attention + reclaim
```

所以它属于 **scope leakage**。

最终 review fix 将 physical substitution 限制为：

```text
effective field exists
AND
dcp_world_size == 1
AND
not use_cascade
```

于是：

```text
Normal pinned FA2:
    physical effective length

DCP:
    upstream logical behavior

Cascade:
    upstream logical behavior
```

工程原则：

> **“看起来合理”不等于“当前 Slice 已支持”。没有 source/reference audit 的路径不要自动声明支持。**

---

# 36. S2/S3 最终 producer contract

例：

```text
logical_num_computed = 128
effective_kv_len = 64
query_len = 4
```

同一个 kernel 生成：

```text
LOGICAL
positions = [128,129,130,131]
seq_lens = 132

PHYSICAL KV
cache_positions = [64,65,66,67]
effective_kv_seq_lens = 68
```

本质：

```text
same local offset
different base
```

---

# 37. 完整 consumer contract

```text
positions
    ↓
model / RoPE

seq_lens
    ↓
logical runtime
sampling / prefill semantics
generic CommonAttentionMetadata


cache_positions
    ↓
BlockTables.compute_slot_mappings
    ↓
slot_mapping
    ↓
reshape_and_cache_flash


effective_kv_seq_lens
    ↓
CommonAttentionMetadata optional field
    ↓
normal FA2 builder
    ↓
FlashAttentionMetadata.seq_lens
    ↓
seqused_k
```

最终矩阵：

| | Logical domain | Physical KV domain |
|---|---|---|
| persistent | `num_computed_tokens` | `effective_kv_len` |
| per-token | `positions` | `cache_positions` |
| per-request | `seq_lens` | `effective_kv_seq_lens` |
| consumer | model / RoPE / logical lifecycle | slot mapping / FA KV read |

---

# 38. 代码实际修改了哪些文件

Production：

```text
vllm/v1/worker/gpu/states.py
vllm/v1/worker/gpu/input_batch.py
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/model_states/default.py
vllm/v1/worker/gpu/attn_utils.py
vllm/v1/attention/backend.py
vllm/v1/attention/backends/flash_attn.py
```

Tests：

```text
tests/v1/worker/test_gpu_effective_kv_state.py
```

### 旧记录与最终 review-fix 的区别

pre-fix implementation 快照曾记录：

```text
7 production files
87 insertions
5 deletions
10 tests
```

这不是最终状态。

review-fix 后最终汇报：

```text
13 targeted tests
13 passed, 15 warnings in 8.87s
```

并加入 DCP/cascade scope guard。

最终 accumulated production diff 汇报：

```text
7 files changed
95 insertions
4 deletions
```

后续若做精确 accounting，应以：

```text
m1-t1-s2-s3-review-fix/03-post-fix.patch
m1-t1-s2-s3-review-fix/03-post-fix-diff.log
```

为准，而不是继续引用 pre-fix `87/5`。

---

# 39. 各文件职责：从 vLLM 架构角度理解，而不是背 diff

## `states.py`

request-lifetime GPU execution state：

```text
num_computed_tokens
effective_kv_len
```

`effective_kv_len` 属于 S1，是 S2/S3 的 physical base。

## `input_batch.py`

S2/S3 producer 核心：

```text
InputBuffers
InputBatch
_prepare_pos_seq_lens_kernel
prepare_pos_seq_lens
post_update
```

## `model_runner.py`

把 persistent request state 组织成当前 forward。

S2 的核心 consumer split：

```text
model → positions
KV slot mapping → cache_positions
```

## `model_states/default.py`

`InputBatch → attention metadata build` 的 transport seam。

## `attn_utils.py`

构造 generic `CommonAttentionMetadata`，只做机械 transport，不改变 generic logical semantics。

## `attention/backend.py`

定义 common metadata、`unpadded()`、`compute_num_computed_tokens()`。这里让我们明确：generic metadata 是跨 backend 接口层，semantic 改动要极谨慎。

## `flash_attn.py`

`CommonAttentionMetadata → FlashAttentionMetadata` 的 backend specialization seam。P1 只在 approved normal FA2 path 把 backend length 切到 physical，kernel ABI 不改。


---

# 40. 测试如何一步步补齐 evidence

最终 targeted tests 共 13 个。

## Case A — baseline equality

```text
logical=128
physical=128
q=1
block_size=16
```

得到：

```text
positions=[128]
cache_positions=[128]
seq_lens=129
effective_kv_seq_lens=129
logical/cache block index 都为 8
```

意义：

```text
logical == physical
```

时新实现与 upstream 等价。

---

## Case B — single-token divergence

```text
logical=128
physical=64
q=1
```

得到：

```text
positions=[128]
cache_positions=[64]
logical length=129
physical length=65
logical/cache block index=8/4
```

---

## Case C — multi-token

```text
logical=128
physical=64
q=4
```

得到：

```text
positions=[128,129,130,131]
cache_positions=[64,65,66,67]
seq_lens=132
effective_kv_seq_lens=68
```

---

## Case D — mixed batch

```text
A:
logical=128
physical=64
q=1

B:
logical=96
physical=80
q=2
```

得到：

```text
logical=[129,98]
physical=[65,82]
```

并验证 `unpadded()` 同步 slice physical channel。

---

## Case E — fallback

```text
effective field exists
→ physical

effective field None
→ logical upstream
```

保证 backward compatibility。

---

# 41. 为什么后来还必须补两个更强的 test

第一轮 oracle 主要证明：

```text
producer 算对了
helper 选对了
```

但 Web review 指出：

```text
producer 正确
不等于
wiring 一定正确
```

所以补了更强 evidence。

## 41.1 `test_prepare_attn_wires_cache_positions`

真正调用：

```text
GPUModelRunner.prepare_attn()
```

捕获：

```text
compute_slot_mappings(...)
```

第三个参数并断言：

```text
is input_batch.cache_positions
is not input_batch.positions
```

这样 S2：

```text
producer
→ InputBatch
→ prepare_attn
→ slot mapping
```

整条 wiring 都被验证。

## 41.2 `test_flash_builder_outputs_effective_lengths_and_fallback`

不再只测试 `_select_kv_seq_lens()`。

而是真正调用：

```text
FlashAttentionMetadataBuilder.build()
```

验证：

```text
logical [129,98]
physical [65,82]

→ FlashAttentionMetadata.seq_lens = [65,82]
```

physical field 为 `None`：

```text
→ [129,98]
```

所以 S3：

```text
Common metadata
→ FA metadata
```

真正被覆盖。

## 41.3 `test_effective_length_scope_guard`

验证 selector 在 physical substitution 显式关闭时回到 logical，作为 DCP/cascade 当前不启用 P1 physical semantics 的窄 oracle。

---

# 42. 测试与 Evidence 结果

最终 targeted：

```bash
CUDA_VISIBLE_DEVICES=2 \
/home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310/bin/python \
-m pytest -q tests/v1/worker/test_gpu_effective_kv_state.py
```

结果：

```text
13 passed, 15 warnings in 8.87s
```

同时：

```text
py_compile: PASS
git diff --check: PASS
```

没有运行 real vLLM engine smoke，因为当时目标 GPU 被外部任务占用。

当前 evidence 边界：

```text
execution semantic unit/oracle PASS
```

而不是：

```text
real reclaim E2E PASS
```

这两者必须严格区分。

---

# 43. 这一阶段保存了哪些源码审计日志

重要 source/evidence：

```text
s2-position-slot-chain.log
s2-focused-source-chain.log
attention-chain-focused.log

P1-M1-T1-S2 Final Implementation Boundary Audit
P1-M1-T1-S3 Effective Attention Length Audit
P1-M1-T1-S3 Final Port Seam Audit

04-experiments/project1_kv_reclaim/raw/
    m1-t1-s2-s3-implementation/

04-experiments/project1_kv_reclaim/raw/
    m1-t1-s2-s3-review-fix/
```

Source audit 主要采用：

```text
rg -n -C
nl -ba
sed -n
```

并用 `tee` 保存 raw evidence。

这形成了一套很值得复用的 vLLM 阅读方法：

```text
先找 producer
→ 再找所有 consumer
→ 判断每个 field semantic
→ 找 explicit reconstruction
→ 找 backend boundary
→ 最后才决定哪里能改
```

---

# 44. 这一阶段最值得保留的源码检索方法

## 44.1 S2 position chain

```bash
rg -n -C 8 \
'prepare_pos_seq_lens|positions|prepare_attn|compute_slot_mappings|gather_block_tables' \
vllm/v1/worker/gpu
```

目标：

```text
producer → InputBatch → consumer
```

## 44.2 查 `positions` 的全部 consumer

```bash
rg -n \
'\bpositions\b' \
vllm/v1/worker/gpu \
--glob '*.py'
```

目标：证明 `positions` 是广泛 logical/model semantic，而非只服务于 slot mapping。

## 44.3 查 Common metadata 所有 constructor

```bash
rg -n \
'CommonAttentionMetadata\(' \
vllm/v1 \
--glob '*.py'
```

目标：新增 field 后检查 transport / reconstruction blast radius。

## 44.4 查 generic `seq_lens` consumer

```bash
rg -n \
'common_attn_metadata\.seq_lens\b' \
vllm/v1/attention/backends \
--glob '*.py'
```

目标：证明不能把 Common `seq_lens` 全局 physical 化。

## 44.5 查 logical helper

```bash
rg -n \
'compute_num_computed_tokens\(' \
vllm/v1
```

目标：确认 generic metadata 仍依赖 logical sequence semantics。

---

# 45. 通过 S2/S3，我们实际学到了哪些 vLLM 架构

S2/S3 不是只完成一个 P1 patch，而是把 MRV2 一个非常核心的 runtime 主链读通。

## 45.1 Request persistent state 和 per-forward metadata 是两层

```text
RequestState
    = request-lifetime truth

InputBatch
    = current execution view
```

不能因为两个值 baseline 下暂时相等，就把 ownership / lifecycle 混为一谈。

---

## 45.2 `positions` 和 KV slot 不是同一个概念

baseline 下仅仅是：

```text
logical sequence coordinate
==
cache sequence coordinate
```

它不意味着：

```text
logical token position == raw physical slot
```

raw slot 还要经过 block table translation。

---

## 45.3 Paged KV 的核心就是 indirection

```text
sequence position
→ block table
→ physical block
→ slot
```

这也是 V1 zero-copy whole-block reclaim 可行的基础：

```text
surviving physical block ID 不需要连续
只要 active block-table order 正确
```

---

## 45.4 Attention backend metadata 是 generic runtime 和 kernel ABI 的分界

```text
CommonAttentionMetadata
```

是 generic interface。

```text
FlashAttentionMetadata
```

是 backend-specific representation。

P1 S3 最适合的 adaptation seam 正是：

```text
Common → FA builder
```

而不是直接改 FA native kernel ABI。

---

## 45.5 `seqused_k` 与 `max_seqlen_k` 语义不同

```text
seqused_k
=
exact per-request valid KV length

max_seqlen_k
=
kernel planning / safe upper-bound information
```

因此：

```text
physical exact → seqused_k

logical conservative upper bound → max_seqlen_k
```

可以同时成立。

---

## 45.6 KV WRITE 与 KV READ 的控制元数据不同

```text
WRITE:
slot_mapping

READ:
block_table + sequence length
```

以后遇到 paged attention correctness 问题时，先区分是：

```text
write address 错
还是
read visibility 错
```

而不是笼统说“KV cache 出错”。

---

# 46. 这一阶段解决了 P1 的什么核心风险

P1 的最终目标：

```text
logical sequence 不回退
physical KV occupancy 可以下降
```

如果不先完成 S1/S2/S3：

即使未来真的 free 了 block，也可能出现：

```text
新 KV 又按 logical position 写回旧高地址
```

或者：

```text
FA 仍按 logical sequence length 读取
```

那么 physical reclaim 无法持续。

所以 S1/S2/S3 是 real physical reclamation 的 execution-side prerequisite。

现在已经具备：

```text
logical progress 可以继续增长
physical KV progress 可以独立缩短
model position 不受 physical compaction 影响
KV write 可以沿 physical frontier 继续 append
attention read 可以只看到 physical effective extent
```

这才让下一步真正 block reclaim 有意义。

---

# 47. 现在还没有解决什么

当前没有实现：

```text
retention decision
real reclaim transaction
BlockTable active-row replacement
stale-tail isolation E2E
worker → scheduler ACK
scheduler canonical ownership update
BlockPool free
cross-request block reuse
```

所以目前：

```text
logical=128
physical=64
```

主要仍由 test fixture 构造。

系统还不会真实、安全地从：

```text
128 / 128
```

transition 到：

```text
128 / 64
```

这就是下一阶段的问题。

---

# 48. 为什么下一步从 execution semantics 进入 ownership transaction

S1/S2/S3 已回答：

```text
如果 physical KV 已经变短，
Worker 如何继续正确执行？
```

下一步要回答：

```text
谁让 physical KV 变短？
什么时候 commit？
谁更新 BlockTable？
谁有权 free block？
Scheduler 什么时候可以把它重新分配给别的 request？
```

也就是：

```text
Worker physical execution truth
        vs
Scheduler allocator ownership truth
```

最危险的 race：

```text
Scheduler 提前 free block
        ↓
Request B 复用它
        ↓
Worker A 旧 BlockTable 仍引用
        ↓
alias / corruption
```

所以后续要沿 Tangram 的：

```text
Worker 完成 physical view 更新
        ↓
产出 freed block IDs / new effective length
        ↓
Scheduler 收到 ACK
        ↓
canonical ownership mutation
        ↓
BlockPool free/reuse
```

继续审计。

---

# 49. Tangram 对整个 P1 的定位：借了什么，没有借什么

## Tangram 已提供强 reference

```text
logical position 与 slot position 分离
effective sequence length
compression 后 attention effective length
worker-side physical result
freed block IDs
scheduler-side safe-free pattern
ragged physical KV architecture
```

## P1 V1 当前采用

```text
uniform per-request effective length
whole-block reclaim
normal 2D block table
FA2
single GPU
```

## 当前没有采用

```text
per-head-group ragged paging
cluster map
virtual member block table
Tangram compression policy
Tangram full runtime complexity
```

因此准确描述不是“照抄 Tangram”，也不是“自己发明 logical/physical split”。

更准确：

> **从 Tangram 的 general physical-reclamation runtime 中抽取已验证 logical/physical contract，再针对 pinned vLLM v0.26 MRV2 做 uniform、GPU-state-oriented 的机械适配。**

---

# 50. Reference / Adaptation 等级总结

| 能力 | 等级 | 说明 |
|---|---|---|
| logical model position 保持原值 | A | Tangram + vLLM semantic |
| effective physical state | A/B | Tangram contract → MRV2 GPU state |
| logical / cache position split | A/B | Tangram slot_positions pattern |
| cache_position → slot mapping | A | vLLM 原生 BlockTables |
| global logical seq_lens 保留 | A | Tangram + vLLM consumer audit |
| separate effective attention length | A/B | Tangram effective length pattern |
| P1 GPU `effective_kv_seq_lens` transport | B/C | v0.26 version glue |
| FA2 builder substitution | B/C | backend-local adaptation |
| FA kernel ABI | 不改 | upstream |
| DCP/cascade reclaim | 未审计 | 不在当前支持范围 |

S2/S3 没有把任何 D 级未知机制作为成功条件。

---

# 51. 最终状态

```text
P1-M1-T1-S1
Persistent Logical / Physical State
PASS / ACCEPTED

P1-M1-T1-S2
Logical Position / Physical Cache Position
PASS / ACCEPTED

P1-M1-T1-S3
Logical Length / Effective KV Attention Length
PASS / ACCEPTED
```

因此：

```text
execution-side logical / physical contract
=
CLOSED
```

但：

```text
real physical reclaim
=
NOT IMPLEMENTED
```

---

# 52. 一页复习版

## 原生 vLLM

```text
num_computed_tokens
    ↓
prepare_pos_seq_lens
    ├── positions ───────────────┐
    │                            ├→ model / RoPE
    │                            └→ slot mapping
    │
    └── seq_lens
             ↓
      CommonAttentionMetadata
             ↓
      FlashAttentionMetadata
             ↓
          seqused_k
```

隐含 invariant：

```text
logical progress == physical KV progress
```

## P1 reclaim 后

```text
logical = 128
physical = 64

positions = 128
cache_position = 64

logical seq_len = 129
physical KV seq_len = 65
```

## 修改后

```text
num_computed_tokens
    ├── positions ─────────────> model/RoPE
    └── seq_lens ──────────────> logical runtime

effective_kv_len
    ├── cache_positions
    │      ↓
    │  slot_mapping
    │      ↓
    │   KV WRITE
    │
    └── effective_kv_seq_lens
           ↓
       FA2 metadata
           ↓
        seqused_k
           ↓
         KV READ
```

---

# 53. 关键源码地图

```text
vllm/v1/worker/gpu/states.py
    RequestState
    ├── num_computed_tokens
    └── effective_kv_len

vllm/v1/worker/gpu/input_batch.py
    InputBuffers
    InputBatch
    _prepare_pos_seq_lens_kernel
    prepare_pos_seq_lens
    post_update

vllm/v1/worker/gpu/model_runner.py
    prepare_inputs
    prepare_attn
    execute_model / postprocess_sampled

vllm/v1/worker/gpu/block_table.py
    BlockTables
    gather_block_tables
    compute_slot_mappings

vllm/v1/worker/gpu/model_states/default.py
    DefaultModelState.prepare_attn

vllm/v1/worker/gpu/attn_utils.py
    build_attn_metadata

vllm/v1/attention/backend.py
    CommonAttentionMetadata
    unpadded
    compute_num_computed_tokens

vllm/v1/attention/backends/flash_attn.py
    FlashAttentionMetadataBuilder.build
    FlashAttentionImpl.forward
    reshape_and_cache_flash
    flash_attn_varlen_func
```

---

# 54. Evidence / Reference Index

## P1 Source Audit

```text
s2-position-slot-chain.log
s2-focused-source-chain.log
attention-chain-focused.log
P1-M1-T1-S2 Final Implementation Boundary Audit
P1-M1-T1-S3 Effective Attention Length Audit
P1-M1-T1-S3 Final Port Seam Audit
```

## Implementation

```text
04-experiments/project1_kv_reclaim/raw/
m1-t1-s2-s3-implementation/
```

## Review Fix

```text
04-experiments/project1_kv_reclaim/raw/
m1-t1-s2-s3-review-fix/
```

关键：

```text
01-pre-fix.patch
02-targeted-tests.log
03-post-fix.patch
03-post-fix-diff.log
pre-post.patch.delta
```

## Tangram Reference

```text
third_party/tangram
commit 8e1cbfa5cf82acc67fe1880df0c632f2a6485e35

vllm/v1/worker/gpu_model_runner.py
    logical positions
    effective slot positions
    effective metadata transport
    effective state advancement

vllm/v1/worker/gpu_input_batch.py
    effective_seq_lens_cpu

vllm/v1/attention/backends/ragged_layout.py
    effective + current scheduled length

vllm/v1/attention/backends/flash_attn.py
    effective per-layer FA metadata
```

---

# 55. 最后一句话

S2/S3 的本质不是“给 vLLM 再加几个 tensor”。

真正完成的是：

> **把原生 vLLM 中依赖 `logical progress == physical KV progress` 的隐含耦合显式拆开：模型继续沿原 logical token / RoPE 时间轴前进，而 KV storage 可以沿独立的 compact physical frontier 写入，并被 Attention 按真实 retained length 读取。**

这一步完成以后，P1 才真正具备实现：

```text
whole-block reclaim
→ active BlockTable rewrite
→ worker physical commit
→ scheduler ownership ACK
→ BlockPool free/reuse
```

的 execution foundation。
