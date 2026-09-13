# P1 V2 M4 — Token-Level Paged KV Compaction 数据面学习笔记

**项目：** P1 — Physical KV Cache Reclamation for vLLM  
**阶段：** P1 V2 — Token-Level Physical KV Compaction  
**当前状态：** M4 Data Plane = PASS / CLOSED  
**用途：** 后续复习 vLLM / KV Cache / Triton / Paged KV Compaction 时直接对照源码使用

---

## 1. M4 到底做了什么

P1 V1 已经解决了 logical / physical state、Worker / Scheduler ownership、effective_kv_len、BlockTable、safe free 等控制面与生命周期问题，但 V1 只能做 whole-block reclaim。

V2 M4 解决的是：

> **把一个 request 当前 paged KV 中需要保留的 token-level KV members 搬到当前 physical block row 的前缀，使得后面的 blocks 成为真正可回收的 trailing blocks。**

核心数据流：

```text
old paged KV
    ↓
根据 keep_member_indices 找 retained members
    ↓
Triton Gather
    ↓
independent scratch[K,H,C]
    ↓
Triton Writeback
    ↓
current physical block-row prefix
    ↓
new effective physical KV prefix [0,K)
```

典型例子：

```text
block_size = 4

old block row:
[B7, B2, B11]

source:
B7  = m0 m1 m2 m3
B2  = m4 m5 m6 m7
B11 = m8 m9 _  _

E_source = 10

keep = [0,2,5,8,9]
```

compact 后：

```text
B7:
m0 m2 m5 m8

B2:
m9 0 0 0

B11:
unchanged
```

此时：

```text
E_new = 5
new_num_blocks = 2
```

M4 只完成 **KV payload transformation**。它还没有 trim Worker BlockTable、修改 Scheduler canonical ownership、free B11 或让其他 request reuse B11。这些属于 M5 runtime transaction。

---

## 2. 当前 KV 的逻辑 shape 与 layout

当前 FA2 MVP 使用：

```text
kv_cache logical shape = [B,H,N,C]
```

其中：

```text
B = num physical blocks
H = num_kv_heads
N = block_size
C = 2 * head_size
```

逻辑 shape 固定，不代表底层 physical layout 一定相同。当前主要考虑：

```text
HND physical: [B,H,N,C]
NHD physical: [B,N,H,C]
```

即使 logical view 都表现为 `[B,H,N,C]`，它们的 stride 不一样。因此 kernel 中真正决定地址的是：

```text
stride_b
stride_h
stride_n
stride_c
```

而不是只看 shape。

---

## 3. M4 的三层实现

整个实现可以分成三层：

```text
Layer 1 — PyTorch Reference
定义“正确答案”

Layer 2 — Triton Gather / Writeback
实现 GPU payload movement

Layer 3 — Full Triton Compaction Primitive
Gather + Scratch + Writeback
```

核心源码：

```text
vllm/v1/worker/gpu/kv_compaction.py
```

测试：

```text
tests/v1/worker/test_gpu_kv_compaction.py
```

---

## 4. PyTorch Reference 的角色

Reference 不是生产实现，而是：

```text
correctness oracle
```

也就是：给定完全相同的输入，生成“正确的 compacted KV 应该是什么样”。

主要函数：

```python
gather_paged_kv_reference(...)
compact_paged_kv_reference(...)
```

Reference 追求的是：

```text
简单
确定
容易理解
无副作用
方便比较
```

而不是性能。

---

## 5. `_validate_inputs()`

作用：统一校验 Gather / Reference Compaction 的输入 contract。

输入：

```text
kv_cache
block_ids
source_effective_kv_len
keep_member_indices
```

输出：

```text
block_size
```

主要约束：

```text
kv_cache 必须是 4D [B,H,N,C]

block_ids:
- 1D
- int32 / int64
- physical block id 合法
- 不能重复

source_effective_kv_len:
- Python int
- > 0

keep_member_indices:
- 1D
- int32 / int64
- 非空
- 属于 [0,E_source)
- 严格递增
```

为什么 keep 必须严格递增？因为 V2 core 的 contract 是：

```text
外部 policy 决定保留谁
physical core 只验证，不修改 decision
```

所以这里不能偷偷 `sort()` / `unique()`，否则 core 就改变了 policy semantics。

---

## 6. `gather_paged_kv_reference()`

作用：把 paged KV 转成 request-local member sequence，再根据 keep 选出 retained KV。

输入：

```text
kv_cache: [B,H,N,C]
block_ids: [P]
E_source
keep: [K]
```

输出：

```text
scratch: [K,H,C]
```

核心过程：

```text
global KV pool
    ↓ index_select(block_ids)
active_pages [P,H,N,C]
    ↓ permute
[P,N,H,C]
    ↓ reshape
[P*N,H,C]
    ↓ truncate E_source
[E_source,H,C]
    ↓ index_select(keep)
[K,H,C]
    ↓ clone
independent scratch
```

`active_pages = kv_cache.index_select(0, block_ids)` 完成的是：

```text
global physical block order
→ request physical page order
```

`permute(0,2,1,3).reshape(-1,H,C)` 把 `page × token_offset` 合并成 request-local physical member index。

最终：

```python
members.index_select(0, keep_member_indices).clone()
```

得到 retained scratch。

---

## 7. 为什么必须有 independent scratch

这是 V2 correctness 的核心。

如果直接做：

```text
paged KV → paged KV
```

arbitrary in-place compaction，可能出现：

```text
source member 还没读
destination write 已经覆盖了它
```

因此 V2 baseline 冻结为：

```text
Paged KV
    ↓
Gather
    ↓
Independent Scratch
    ↓
Writeback
```

这样 source storage 与 destination storage 独立，就不存在 read-after-write overlap hazard。

---

## 8. `compact_paged_kv_reference()` 为什么 clone KV Cache

这个函数不是“执行生产 compaction”，而是在生成：

```text
expected final global KV state
```

所以：

```python
expected = kv_cache.clone()
```

目的有两个。

第一，不修改输入 kv_cache。测试时可以同时保留：

```text
before
expected
actual Triton result
```

第二，可以验证整个 global KV pool，而不只是 active pages：

```text
B7 / B2 写对了
B11 没被误写
其他 request blocks 没被误写
整个 global pool 没有额外 side effect
```

最终可以直接：

```python
torch.equal(actual, expected)
```

这比只比较 destination pages 更强。

---

## 9. Reference Compaction 完整流程

```text
_validate_inputs
    ↓
gather_paged_kv_reference
    ↓
scratch[K,H,C]
    ↓
E_new = K
    ↓
new_num_blocks = ceil(K / block_size)
    ↓
clone global KV -> expected
    ↓
allocate zero dense buffer
    ↓
dense[:K] = scratch
    ↓
reshape back into pages
    ↓
write pages into block_ids[:new_num_blocks]
```

为什么 dense 长度不是 K？因为最终 physical storage 必须按完整 page 组织。

例如：

```text
K = 5
block_size = 4
```

需要：

```text
2 pages = 8 slots
```

因此 dense 是：

```text
[m0,m2,m5,m8,m9,0,0,0]
```

这也定义了：

```text
final active page tail = deterministic zero
```

---

## 10. Triton Gather 两种实现

当前保留两种 work decomposition。

### Variant A — NHD-only 1D

Kernel：

```python
_gather_paged_kv_nhd_1d_kernel
```

Grid：

```text
(K,)
```

一个 Triton program：

```text
负责一个 retained token
搬完整 [H,C]
```

即：

```text
program j → scratch[j,:,:]
```

为什么只能 NHD？因为 NHD physical layout 为：

```text
[B,N,H,C]
```

固定 `physical_block` 和 `token_offset` 后，整个 `[H,C]` 连续，因此可以 flatten 成 `H*C` 一次搬运。

---

## 11. 1D Gather 的地址计算

核心：

```python
j = tl.program_id(0)
member = keep_member_indices[j]
page_idx = member // block_size
token_offset = member % block_size
physical_block = block_ids[page_idx]
```

这条链完成：

```text
retained output j
→ source member
→ request page / token offset
→ global physical block
```

NHD 下 source：

```python
src = (
    physical_block * stride_b
    + token_offset * stride_n
    + offs
)
```

destination：

```python
dst = j * (H*C) + offs
```

对应：

```text
scratch[j,:,:]
```

---

## 12. `tl.arange()` 不等于 CUDA thread 数

例如：

```python
offs = tl.arange(0, BLOCK_HC)
```

如果 `BLOCK_HC = 2048`，不要理解成 2048 个 CUDA threads。

它表示当前 Triton program 内部的一组 block tensor offsets。Triton compiler 会进一步映射到 warp / lane / vectorized instructions。

所以：

```text
Triton program != CUDA thread
```

---

## 13. 为什么需要 mask

通常：

```python
BLOCK_HC = triton.next_power_of_2(H*C)
```

例如：

```text
H*C = 1500
BLOCK_HC = 2048
```

那么：

```python
mask = offs < hc
```

保证只有真实的 1500 elements 访问 memory，多出来的 lanes 被屏蔽。

---

## 14. Variant B — stride-safe 2D Gather

Kernel：

```python
_gather_paged_kv_2d_kernel
```

Grid：

```text
(K,H)
```

一个 program：

```text
一个 retained token
× 一个 KV head
× [C]
```

也就是：

```text
program(j,h) → scratch[j,h,:]
```

和 1D 最大区别：

```text
1D: 一次搬 [H,C]
2D: 一次只搬 [C]
```

---

## 15. 为什么 2D 可以支持 HND / NHD

因为 2D kernel 不假设 `H*C` 连续，而是显式使用真实 stride：

```python
src = (
    physical_block * stride_b
    + h * stride_h
    + token_offset * stride_n
    + offs * stride_c
)
```

它对应：

```text
kv_cache[physical_block,h,token_offset,c]
```

所以 HND / NHD 只要 stride 正确，同一个公式都成立。

---

## 16. 1D 和 2D 的真正差别

不要只记“1D grid vs 2D grid”，真正差别是 **program ownership**。

### 1D

```text
program j
负责 token j 的完整 [H,C]
```

特点：

```text
page/block lookup 一次
程序粒度更粗
H*C 一次搬运
NHD locality 自然
但 program 更大
```

### 2D

```text
program(j,h)
负责 token j 的一个 head [C]
```

特点：

```text
程序更小
program 数量更多
HND / NHD 通用
block/page lookup 会对每个 h 重复
```

当前没有性能优劣结论，只证明 correctness 都成立。

---

## 17. `_validate_triton_inputs()`

在 `_validate_inputs()` 基础上增加 CUDA backend 要求：

```text
kv_cache.is_cuda
block_ids.is_cuda
keep_member_indices.is_cuda
```

并保证三者位于同一个 CUDA device。

---

## 18. `_validate_nhd_layout()`

作用：确认 logical `[B,H,N,C]` tensor 确实由 NHD physical storage 支撑。

NHD physical：

```text
[B,N,H,C]
```

对应 logical strides：

```text
stride_c = 1
stride_h = C
stride_n = H*C
stride_b = N*H*C
```

只有满足该 contract，1D kernel 才允许执行，否则 fail closed。

---

## 19. Gather Wrapper 的职责

以：

```python
gather_paged_kv_triton_2d(...)
```

为例，wrapper 负责：

```text
1. validation
2. 创建 scratch[K,H,C]
3. 计算 BLOCK_C
4. 配置 grid
5. launch Triton kernel
6. 返回 scratch
```

kernel 只负责 GPU address + load/store；wrapper 负责 Python-side contract / allocation / launch。

---

## 20. Triton Writeback 的角色

Gather 解决：

```text
旧 KV 中保留谁
```

Writeback 解决：

```text
这些 retained KV compact 后放哪里
```

Gather 后 scratch 已经是正确 retained 顺序：

```text
scratch[0]
scratch[1]
...
scratch[K-1]
```

因此 Writeback 不需要 `keep_member_indices`。

destination member 永远就是：

```text
dst_member = j
```

---

## 21. Writeback destination mapping

对于 compact 后的 `dst_member`：

```python
dst_page = dst_member // block_size
dst_offset = dst_member % block_size
physical_block = block_ids[dst_page]
```

因此 destination 是：

```text
kv_cache[physical_block,...,dst_offset,...]
```

所以：

```text
Gather source 是 irregular
Writeback destination 是 regular prefix
```

这是两者的本质区别。

---

## 22. 1D Writeback Kernel

Kernel：

```python
_writeback_paged_kv_nhd_1d_kernel
```

Grid：

```text
(dst_capacity,)
```

其中：

```text
dst_capacity = new_num_blocks * block_size
```

一个 program 写完整 destination token `[H,C]`。

如果：

```text
dst_member < K
```

读取 `scratch[dst_member]`；否则写 zero。

---

## 23. 为什么 Writeback grid 不是 K

例如：

```text
K = 5
block_size = 4
new_num_blocks = 2
```

如果只 launch `(5,)`，只能写 5 个 retained tokens，但最后 active page 剩下的旧 slots 可能仍然有旧 KV。

因此需要：

```text
dst_capacity = 2*4 = 8
```

launch program 0...7：

```text
0..4 -> retained KV
5..7 -> zero
```

这样一个 kernel 同时完成：

```text
retained writeback + tail zero
```

---

## 24. Masked Load 为什么不会越界读 scratch

Writeback：

```python
values = tl.load(
    scratch + dst_member * hc + offs,
    mask=is_retained & valid_hc,
    other=0.0,
)
```

当 `dst_member >= K` 时：

```text
is_retained = False
```

对应 memory load 被 mask，不会真正访问 `scratch[K]` 等越界位置，同时 `values` 由 `other=0.0` 得到 zero。

---

## 25. 2D Writeback Kernel

Kernel：

```python
_writeback_paged_kv_2d_kernel
```

Grid：

```text
(dst_capacity,H)
```

一个 program：

```text
一个 destination token
× 一个 head
× [C]
```

source：

```text
scratch[dst_member,h,:]
```

destination：

```text
kv_cache[physical_block,h,dst_offset,:]
```

同样通过完整真实 stride 支持 HND / NHD。

---

## 26. `_validate_writeback_inputs()`

作用：验证 scratch 是否可以安全写回 paged KV。

输入：

```text
kv_cache
block_ids
scratch
```

返回：

```text
block_size
retained_members = K
new_num_blocks
```

主要检查：

```text
scratch shape = [K,H,C]
K > 0
scratch H/C 与 kv_cache 一致
dtype 一致
scratch contiguous
CUDA device 一致
block_ids 合法且唯一
len(block_ids) >= new_num_blocks
```

注意这里是 `>=`，因为 compaction 前 block row 可能仍有 trailing blocks。

---

## 27. `writeback_paged_kv_triton_2d()`

Wrapper 做：

```text
validate
    ↓
计算 K / new_num_blocks
    ↓
dst_capacity = new_num_blocks * block_size
    ↓
grid = (dst_capacity,H)
    ↓
launch 2D writeback
    ↓
return K, new_num_blocks
```

它不会修改 Worker BlockTable，也不会 free trailing blocks，只修改 KV payload。

---

## 28. `compact_paged_kv_triton_2d()`

这是当前完整 M4 production candidate。

核心逻辑：

```python
scratch = gather_paged_kv_triton_2d(...)
return writeback_paged_kv_triton_2d(
    kv_cache,
    block_ids,
    scratch,
)
```

数据流：

```text
paged KV
    ↓
2D Gather
    ↓
scratch[K,H,C]
    ↓
2D Writeback
    ↓
compacted paged prefix
```

输出：

```text
new_effective_kv_len = K
new_num_blocks
```

但“返回 K”不代表 runtime 已经 commit `effective_kv_len = K`，真正 runtime state commit 在 M5。

---

## 29. 为什么 Full Primitive 默认使用 2D

不是因为已经证明 2D 更快，目前没有 performance benchmark。

主要原因是：

```text
2D stride-safe
```

天然支持：

```text
HND
NHD
```

而 1D variant 只支持 NHD。

所以当前定位：

```text
2D = general correctness baseline / production candidate
1D = specialized implementation candidate
```

---

## 30. M4 已建立的真实 GPU Evidence

环境：

```text
GPU: NVIDIA A100 80GB PCIe
Compute Capability: 8.0
PyTorch: 2.11.0+cu129
Triton: 3.6.0
CUDA runtime: 12.9
```

M4-T2：

```text
36 passed
0 skipped
```

Triton-only：

```text
22 passed
```

1D realistic shape：

```text
H = 8
C = 256
H*C = 2048
compile = PASS
launch = PASS
correctness = PASS
```

M4-T3 最终：

```text
52 passed
0 skipped
```

覆盖：

```text
NHD 1D Gather
NHD 1D Writeback
2D Gather
2D Writeback
HND
NHD
partial tail
exact boundary
cross-page
heavy shrink
overlap keep
detached trailing block
full compaction
```

最重要的 oracle comparison：

```python
torch.equal(actual, expected)
```

通过，即：

```text
Triton full data plane == PyTorch global KV oracle
```

---

## 31. M4 最重要的 invariants

### Invariant 1

`keep_member_indices` 只表示当前 physical member sequence 中保留谁，不是 logical position / block id / global slot / prompt token index。

### Invariant 2

retained order 不变，keep 必须严格递增。

### Invariant 3

```text
destination = current physical block-row prefix
```

### Invariant 4

```text
E_new = K
```

### Invariant 5

```text
new_num_blocks = ceil(K / block_size)
```

### Invariant 6

最后 active page：

```text
[0,K) = valid KV
[K,page_end) = zero
```

### Invariant 7

trailing blocks 在 M4 中不修改、不 free。

---

## 32. M4 没有解决什么

M4 CLOSED 不等于 V2 CLOSED。

M4 后：

```text
KV payload = 已经 compact
```

但 runtime 可能仍然是：

```text
Worker BlockTable: [B7,B2,B11]
Worker effective_kv_len: old E
Scheduler req_to_blocks: [B7,B2,B11]
BlockPool: B11 still owned
```

因此 M4 是：

```text
Data Plane Closure
```

下一阶段 M5 才解决：

```text
Runtime Transaction Closure
```

---

## 33. M5 接下来要解决什么

完整 runtime transaction：

```text
Stage 0
Scheduler / control plane 产生 CompactionPlan
    ↓
Stage 1
Worker model forward 完成
    ↓
Stage 2
POST_FORWARD_PRE_EXECUTE_STATE
    ↓
Stage 3
compact_paged_kv_triton_2d()
    ↓
Stage 4
Worker physical view commit
    ↓
BlockTable prefix
effective_kv_len = K
    ↓
Stage 5
CompactionResultData
    ↓
Stage 6
Scheduler canonical ownership reconcile
    ↓
Stage 7
trailing blocks detached + fenced
    ↓
Stage 8
safe deferred free
    ↓
Stage 9
BlockPool reuse
```

这才是真正完整的 Token-Level Physical KV Reclamation。

---

## 34. 学习时最值得记住的一句话

> **Gather 解决“旧 KV 中保留谁”，Writeback 解决“这些 retained KV compact 后放哪里”。**

展开：

```text
Gather:
keep[j]
→ source member
→ source page
→ physical block
→ scratch[j]

Writeback:
scratch[j]
→ destination member j
→ destination page
→ physical block prefix
```

因此：

```text
Gather source 是 irregular
Writeback destination 是 regular
```

这是理解整个实现最关键的一条。

---

## 35. 最终架构图

```text
                   Scheduler / Policy
                         │
                         │ keep_member_indices
                         ▼
                ┌──────────────────┐
                │  Paged KV Pool   │
                │    [B,H,N,C]     │
                └────────┬─────────┘
                         │
                         │ block_ids + E_source + keep
                         ▼
              ┌─────────────────────┐
              │ Triton Gather 2D    │
              │ grid = (K,H)        │
              └─────────┬───────────┘
                        │
                        ▼
                 scratch[K,H,C]
                        │
                        │ independent
                        ▼
              ┌─────────────────────┐
              │ Triton Writeback 2D │
              │ grid=(capacity,H)   │
              └─────────┬───────────┘
                        │
                        ▼
             compacted physical prefix
                        │
          ┌─────────────┴─────────────┐
          │                           │
      E_new = K              new_num_blocks
                                    =
                              ceil(K / N)
                        │
                        ▼
                  M5 Runtime
                        │
             Worker / Scheduler
                        │
             ownership / safe free
```

---

## 36. 当前阶段结论

```text
P1 V2 M4
Token-Level Paged KV Compaction Data Plane

PyTorch Oracle
= PASS

Triton Gather
= PASS

Triton Writeback
= PASS

Full Gather -> Scratch -> Writeback
= PASS

HND / NHD Correctness
= PASS

Tail Zero
= PASS

Detached Block Preservation
= PASS

Real A100 Evidence
= PASS

M4
= CLOSED
```

下一阶段：

```text
M5
Runtime Integration
+
Physical State Commit
+
Canonical Ownership Reconciliation
+
Safe Free / Reuse
```
