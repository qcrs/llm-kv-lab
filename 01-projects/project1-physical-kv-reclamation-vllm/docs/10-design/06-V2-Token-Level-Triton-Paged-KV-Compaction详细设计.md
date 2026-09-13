# 06 — V2 Token-Level Triton Paged-KV Compaction 详细设计

## 1. Why V2 Exists

V1 只能整 block retain/drop。V2 允许 arbitrary sorted `keep_member_indices`，把 retained tokens compact 成短的 physical prefix，然后释放 trailing pages。

目标：

```text
paged source KV + keep_member_indices
→ exact retained K/V in original retained order
→ compact paged destination
→ shorter effective_kv_len
→ freed trailing physical pages
```

Logical model positions不重编号。

## 2. Reference Boundary

### 已有 reference

- Tangram `CompressionExecutor`：keep→gather→writeback semantic；
- pinned vLLM Triton reshape/cache：slot→block_idx/block_offset→paged address；
- Sparse-vLLM：scalar/batched compaction oracle、invalid input atomicity；
- V1：worker commit→scheduler free lifecycle。

### 我们自己实现

```text
arbitrary member-index Triton paged gather
launcher / tile / num_warps
scratch layout
v0.26 integration glue
```

它是 **C — mechanically specified own implementation**，不是 algorithm research。

## 3. Required Staged Path

第一版禁止直接 general in-place paged→paged：

```text
Phase 0  PyTorch semantic oracle
Phase 1  Triton paged source → contiguous scratch
Phase 2  known/upstream-style scratch → compact paged destination
Phase 3  worker commits row/effective length
Phase 4  scheduler frees trailing pages
```

Scratch 多一次 global-memory roundtrip是 deliberate correctness staging，不是失败。

## 4. PyTorch Oracle First

Oracle 需要覆盖：

- K/V；
- multiple KV heads；
- block_size boundary；
- non-contiguous physical block IDs；
- partial source/destination tail；
- empty/duplicate/unsorted/out-of-bounds `keep_member_indices`；
- arbitrary sorted current-member keep pattern；
- retention=1 identity。

invalid input必须在任何 destination/cache mutation前 reject。

## 5. Triton Gather Address Contract

推荐 launch concept：

```text
grid = (retained_tokens, kv_heads)
```

每个 program：

```text
dst_compact_idx
→ source_member_idx = keep_member_indices[dst_compact_idx]
→ source_block_col = source_member_idx // block_size
→ source_offset    = source_member_idx % block_size
→ physical_block_id = source_block_table[source_block_col]
→ compute K/V head pointer
→ tl.load vector
→ tl.store contiguous scratch
```

head/vector tiling、stride/address pattern参考 pinned vLLM Triton paged KV store/quant code。

## 6. Destination Scatter

第一版优先复用/贴近 pinned `reshape_and_cache_flash` semantic：

```text
scratch K/V
+ compact destination slot_mapping [0..retained_len-1]
→ paged destination prefix
```

不要同时自己发明 source gather、destination layout、alias protocol。

## 7. Aliasing / Lifetime Rule

在 compaction GPU work 完成且 worker physical view committed 前：

```text
candidate freed blocks still belong to request
```

不能提前进入 BlockPool。

为什么不直接 general in-place：destination prefix与 source page可能 overlap；一个 program 的 store 可能覆盖另一个 program 尚未 load 的 source。除非 V3 有明确 dependency/non-overlap proof，否则 staged scratch 是 Core path。

## 8. Logical Position after Token Compaction

V2 把 current retained/member sequence 中的 arbitrary members compact 到 physical prefix；retained K 仍携带它们生成时的 original logical positional semantics。Core 限制 `q_len=1`，避免 multi-token query 的 causal boundary需要恢复更复杂 original key/query position metadata。

因此：

```text
keep_member_indices order = current retained/member sequence order
physical compact index != new logical position
```

如果未来 q_len>1，必须重新审计 causal metadata，不可仅放宽 assert。

## 9. Partial Destination Tail

`retained_len` 不一定 block aligned：

```text
num_dst_pages = ceil(retained_len / block_size)
```

scatter 只写 `[0, retained_len)` valid slots；attention effective length = retained_len；下一 decode append slot = retained_len。trailing unused slots不算 KV member。

## 10. Correctness Gates

V2 required：

1. PyTorch oracle PASS；
2. Triton gather vs oracle PASS；
3. scatter/compacted paged KV vs oracle PASS；
4. attention output vs dense retained-KV reference PASS；
5. freed trailing IDs正确；
6. B real reuse + A continues decode；
7. NSYS/NCU能解释 kernel behavior。

性能不要求：

```text
Triton > PyTorch
TPOT must improve
E2E speedup > X
```

如果 kernel正确但 staging cost高，依然完成 V2，并由 profiler决定是否进入 fused V3。

## 11. Fallback Ladder

```text
fused/direct attempt problematic
→ staged Triton gather + upstream-style scatter
→ runtime integration still problematic
→ PyTorch reference stays correctness baseline; fix integration before performance
```

V2 强交付要求最终至少有一颗正确 Triton gather/codec primitive，但它不必须成为 production default path。

## 12. Index Domain Contract（v1.2）

Core 中 `keep_member_indices` 的定义是：

> index into the **current retained physical/member sequence before this compaction**。

第一次 prefill-boundary V2 时，current member order 与原 prompt logical order一致；但未来 periodic reclaim 后二者不再必然数值相等。因此 kernel 不把 `keep_member_indices` 当成 model position。

如果 V3 policy 需要根据 original logical token position 反复压缩，则另行维护：

```text
member_to_logical_position
```

或 policy-side metadata。该 mapping 不进入 V2 Core。
