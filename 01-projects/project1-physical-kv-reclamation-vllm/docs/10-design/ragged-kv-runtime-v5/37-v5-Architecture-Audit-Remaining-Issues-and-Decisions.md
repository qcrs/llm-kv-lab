# v5 Architecture Audit — Remaining Issues, Corrections & Frozen Decisions

**Status:** CANONICAL / supersedes conflicting legacy notes  
**Purpose:** 审核 v4 是否还有会阻塞真实实现的问题，并将剩余设计选择冻结到可执行状态。

---

# 1. Verdict

v4 的总体方向正确，但在真正进入 R1/R3 前仍有五个需要冻结的问题：

1. **Scheduler 侧缺少 per-group physical frontier。** Worker 有 `E[r,g]` 还不够；allocator 下一步必须依据 compact 后每个 group 的真实长度申请 page。
2. **canonical ownership representation 需要明确。** “flat IDs”只是存储形式；语义必须始终是 `request → group → ordered page row`。
3. **allocation 与 compaction commit 必须区分 reservation / ownership commit。** 首版同步路径可以简化，但协议必须能够解释失败与 stale result。
4. **member metadata 的 shape/lifetime 还需冻结。** 尤其 prefill、decode、CG replay 不能每层临时构造不同格式。
5. **旧文档部分 R5/R6、snapshot/restore 语义存在历史残留。** v5 已直接修正文档，不再只依赖 precedence note。

这些不是新增 feature，而是让 Core 从“能跑一次”变成“压缩后可以继续长期 decode”的必要条件。

---

# 2. Frozen Authority Model

```text
Scheduler-side Ragged KV manager
= canonical physical ownership + canonical per-group physical frontier

Worker RaggedBlockTables
= current execution mirror of page rows

Worker RaggedKVState
= current execution mirror of effective lengths

Compression policy
= decision only

BlockPool
= physical page allocator/free list
```

绝不能出现：

```text
Worker free page first
Scheduler later learns it
```

也不能出现：

```text
Scheduler only knows max(E[g])
Worker knows actual E[g]
```

因为第二种会导致下一步 allocation 重新把所有 group 拉齐。

---

# 3. Canonical Group Identity

首版只做 per-layer grouping：

```text
local_layer_idx in [0, L_local)
group_in_layer   in [0, G_layer)

group_flat = local_layer_idx * G_layer + group_in_layer
GT = L_local * G_layer
```

所有模块共享同一 `RaggedGeometry`，禁止自己重新计算 flatten 顺序。

TP 下 `local_layer_idx` 仍是本 rank PP-stage 的 local layer order；Core 首版 PP=1，所以 source of truth 很简单。未来 PP 才需要 global-layer-id↔local-layer-id map。

---

# 4. Canonical Scheduler Representation

首版推荐语义：

```text
req_id
  ├─ effective_lens[GT]
  ├─ counts[GT]
  └─ group_rows[GT][variable pages]
```

实现层可以选择：

```text
A. list[np.ndarray] per group
B. one flat np.int32 array + offsets/counts
```

但 API 层必须暴露 group 语义，不能依赖“allocation order 恰好能反推”。

推荐首版采用：

```text
flat_ids + counts[GT]
```

因为：

- 比 nested `KVCacheBlock` object 更接近 Tangram 后续 bulk allocator；
- counts 明确支持 non-uniform depth；
- 可以 deterministic reshape/scatter；
- scheduler reconcile 容易验证。

---

# 5. Allocation After Non-Uniform Compression

这是 v5 的关键修正。

压缩后：

```text
E = [96, 160, 64, ...]
count = [6, 10, 4, ...]     # B=16
```

下一步 decode `q=1`：

```text
required_E[g] = E[g] + 1
required_pages[g] = ceil(required_E[g] / 16)
```

只有跨 page boundary 的 group 才需要新 page。

不能使用：

```text
request.num_computed_tokens + q
```

也不能长期只使用：

```text
max(E[g]) + q
```

后者虽然保守正确，但会把较短 group 重新扩成最大 group 的深度，逐步吃掉 non-uniform reclamation 的收益。

Tangram 当前 scheduler/manager 已显式引入 `effective_num_cached_tokens` / `compress_max_eff_seq_len`，证明“压缩后 allocator 不能继续只看 logical num_computed_tokens”这一问题是真实存在的；v5 在此基础上进一步选择 per-group frontier，匹配我们的长期 physical-ownership 目标。

---

# 6. Append / Compaction Ordering

首版 Core 强制：

```text
max_concurrent_batches = 1
PP = 1
async scheduling = OFF
spec decode = OFF
```

因此同一 request 不存在两个 GPU step 同时修改同一 Ragged row。

一个 step 的物理状态：

```text
E_before
  ↓ reserve pages for E_before + q
worker writes q tokens
  ↓
E_post_append = E_before + q
  ↓ optional compaction
E_final <= E_post_append
  ↓ worker result
scheduler reconcile/free
```

R3 第一版允许 compression boundary 与 page-growth step 分离；之后再补 same-step append+new-page+compression stress case。

---

# 7. R1 Identity Does Not Need Scheduler Per-Group E Yet

为了降低 R1 复杂度：

```text
R1 identity:
all E[g] equal
all counts[g] equal
```

可以使用 derived identity state。

但 R2 开始必须建立真正 scheduler-side vector state，且 R3 禁止继续依赖 scalar `Request.effective_kv_len`。

---

# 8. Prefix Cache Decision

Core 明确禁止 APC/prefix cache，不做兼容层。

原因不仅是“复杂”：

```text
prefix hash identity
↔ immutable logical token block
```

而 Ragged compaction 会改变 physical row / payload membership；不同 group 又可能保留不同 token subset。

在没有定义：

```text
per-group cache identity
hash validity after compaction
shared-prefix ownership semantics
```

之前不能声称兼容。

这应写入 config validation，而不是 README limitation only。

---

# 9. CUDA Graph Decision

Core acceptance：

```text
R1-F = compatibility smoke
```

Resume-grade optimization：

```text
R9 = persistent metadata + piecewise CG hardening
```

Full CUDA Graph 明确不做。

如果 R9 benchmark 没有可测 TPOT/CPU-gap 改善，则简历只写“兼容 piecewise CUDA Graph”，不写“优化 decode latency”。

---

# 10. TP Decision

R6 首版：

```text
TP=2
uniform budget
rank-local identity grouping
same Hp on all ranks
```

不做：

```text
cross-rank cluster
layer/global budget
TP-aware AOT
```

TP evidence 必须是：

```text
rank-local FA correctness
+ rank-consistent physical group counts/free boundary
+ scheduler reuse closure
```

不是只启动成功。

---

# 11. Optimization Decision

默认优化优先级：

```text
AOT per-layer clustering
> Piecewise CG hardening
> Triton in-place (only if compaction is material)
> bulk allocator (only if CPU allocator pressure is material)
```

AOT 与项目 physical fragmentation 主线最一致，因此是最推荐的“额外亮点”。

---

# 12. Final Project Boundary

项目完成优先级：

```text
Tier 1 — systems core
R1 + R2 + R3

Tier 2 — flagship
+ R4

Tier 3 — serving/distributed breadth
+ R5 + R6 + R9 compatibility/hardening

Tier 4 — one measured optimization
+ R7 preferred
```

R8 / ring allocator 不应阻塞简历交付。
