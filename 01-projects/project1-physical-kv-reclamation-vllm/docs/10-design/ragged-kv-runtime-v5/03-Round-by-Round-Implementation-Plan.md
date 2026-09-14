# Round-by-Round Implementation Plan — v4

原则：**按风险隔离推进，不按 Tangram 文件目录推进。** 每个 Slice 必须同时具备：source fact、design decision、implementation surface、acceptance、evidence、fallback。上一 Gate 未关闭，不进入下一 Gate。

---

# R0 — Baseline / Reference Freeze

## Goal

固定三套源码与运行环境，不改代码。

```text
clean v0.26 baseline
current P1/V2
Tangram reference
```

## Actions

- 记录 exact SHA / branch / worktree。
- Qwen3-0.6B dense smoke。
- P1 existing compaction/reconciliation focused tests。
- Tangram compression-off smoke；能跑则留一条 minimal compression smoke。
- 记录 A100 / CUDA / torch / Python / FA backend。

## Exit

```text
source pins fixed
environment fixed
baseline output preserved
existing P1 evidence reproducible
```

---

# R1 — Identity Ragged Paging

R1 **不允许删除任何 KV**。只证明新的 physical representation 能正确运行。

## R1-A1 — RaggedAttentionSpec / Config Contract

### Implement

- `CacheConfig.page_group_size` 最小开关。
- 新 `RaggedAttentionSpec`，继承当前 v0.26 spec 语义。
- `num_kv_heads` 保持 semantic Hkv；新增/记录 storage `page_group_size=Hp`。
- startup hard gates。

### Required validation

```text
Hp > 0
Hkv_local % Hp == 0
KVQuantMode.NONE
head_size_v == head_size
FullAttention only
prefix/spec/connector/DCP/PCP/MLA/hybrid off
```

### Acceptance

```text
G = Hkv/Hp
G * ragged_page_bytes == dense_page_bytes
invalid Hp rejects
dense path byte-for-byte behavior unchanged
```

## R1-A2 — Global Physical Page Pool / Shared Backing

### Implement

Ragged planner：

```text
page_bytes = 2 * block_size * Hp * D * dtype_size
num_physical_pages = floor(available_memory / page_bytes)
```

不再除一次 `num_layers`。

所有 supported local attention layers 共享一个 raw backing；不同 layer/group 的隔离由 global physical page IDs 完成。

### Acceptance

```text
allocation bytes exact
one global page-ID namespace
all intended layers alias same backing
dense planner regression unchanged
```

## R1-A3 — Physical Layout / Stride Contract

### Goal

证明 virtual block 是 **zero-copy address transform**，不是逻辑想象。

### Frozen first design

```text
physical: [Bphys, Hp, N, 2D]
virtual:  [Bphys*Hp, 1, N, 2D]
```

### Acceptance

```text
physical.data_ptr == virtual.data_ptr
shape/stride oracle exact
(page,column) ↔ vbid bijection
(vbid,token_offset) ↔ physical address exact
no hidden contiguous/materialization copy
```

这一步未过，禁止 R1-D。

## R1-B — MRv2 RaggedBlockTables

### Representation

```text
R = max_num_reqs
GT = local_layers * groups_per_layer
M = max blocks per group

persistent rows physical shape: [R*GT, M]
logical view:                  [R, GT, M]
counts:                        [R, GT]
flat_row = req_idx*GT + group_flat
```

### Reuse

优先复用：

```text
StagedWriteTensor
UvaBackedTensor
persistent GPU buffers
Triton row gather
```

### Acceptance

- append one group / multiple groups；
- block-boundary growth；
- two requests interleaved；
- remove/reset；
- active row gather；
- counts independent；
- pointer stability；
- no duplicate page ownership。

## R1-C — Virtual Block / Member / Slot Oracle

### Identity map

```text
group = head // Hp
column = head % Hp
flat_group = layer*G + group
vbid = physical_page*Hp + column
```

### Slot

```text
p = physical cache position
page_depth = p // block_size
offset = p % block_size
page_id = row[page_depth]
slot = (page_id*Hp + column)*block_size + offset
```

### Acceptance

纯 reference oracle 与 GPU slot mapping exact equality。

## R1-D1 — Ragged KV Write

### Input

```text
K/V [T, Hkv, D]
```

### Transform

```text
member K/V [T*Hkv, 1, D]
member virtual slots [T*Hkv]
```

### Seam

```text
unified_kv_cache_update
→ ragged dispatch
→ reshape_and_cache_flash
```

### Acceptance

人工 page IDs + member slots 下逐 cell 比较 cache payload。

## R1-D2 — Decode FA2 Read

### Decode geometry

```text
q_len=1
Q [T,Hq,D]
→ [T*Hkv, Hq/Hkv, D]
member block table [R*Hkv,M]
member seq lens [R*Hkv]
```

### Acceptance

synthetic dense vs ragged decode output equality。

## R1-D3 — Prefill / Mixed FA2 Read

Prefill 不能假装成纯 reshape：不同 request q_len 不同，需要 token-major → member-major materialization，再 inverse output copy。

### Test case

```text
Req A q=128
Req B q=37
Req C q=1 decode
```

### Acceptance

- pure prefill；
- mixed prefill/decode；
- uneven q_len；
- chunked-prefill-like split；
- output equality。

## R1-E — Real Engine Identity

### Scope

```text
A100 / FA2 / TP1 / PP1
Qwen3-0.6B first
BF16/FP16
compression off
page_group_size=2 or 4
```

### Must run

```text
single request prefill+decode
multi-request continuous batching
mixed prefill/decode
chunked prefill
```

### Exit

```text
token output parity
optional logits tolerance
identity physical memory accounting
no page leak
```

## R1-F — Piecewise CG Compatibility Smoke

不是性能优化，只回答：

```text
Ragged Identity + v0.26 native Piecewise CG
能否正确运行？
```

如果失败，记录 unsupported reason；不阻塞 R2/R3 Core。

---

# R2 — Per-Group Physical State

R2 从 v5 开始必须同时建立两份语义一致但 authority 不同的 state：

```text
Scheduler-side Ragged physical state
  E[req,GT] / counts[req,GT] / canonical rows
  = allocation authority

Worker RaggedKVState.effective_kv_lens [R,GT]
  = execution mirror
```

下一步 allocation 必须逐 group 使用：

```text
required_pages[g] = ceil((E[g] + q) / block_size)
```

禁止继续只按 logical `num_computed_tokens` 或 scalar max(E) 长期扩容。

手工制造：

```text
G0=128
G1=192
G2=96
```

同一个逻辑 decode token：

```text
logical position identical
physical write position differs by group
```

不做 free，不做 scorer。

Exit：group-specific append/read 正确；scheduler allocation-after-nonuniform 正确；短 group 在后续 decode 不会被重新拉齐。

---

# R3 — Manual Non-Uniform Physical Reclamation

这是 Core 最重要的 systems gate。

```text
manual keep lengths / indices
→ reference compaction
→ ragged row tail trim
→ worker result
→ scheduler source-fence validation
→ canonical ownership trim
→ BlockPool free
→ request B reuse
→ request A continue decode
```

### Acceptance is atomic

必须同时满足：

```text
target lengths differ
payload equals reference
worker row shrinks
scheduler row matches
free pages increase
freed IDs reusable
A continues correctly
```

任何一项缺失，不算 R3 closed。

---

# R4 — One Real Compression Policy

## R4-A Control Plane

只做：

```text
UniformScope
+ one deterministic scorer
+ sink/top-k/tail plan
+ explicit compression boundary
```

不做 scorer zoo。

## R4-B Automatic Non-Uniform

TP1 下允许 layer/per-group non-uniform allocation，让 automatic policy 真正产生不同 group depth。

Exit：policy-driven physical divergence + R3 same allocator closure。

---

# R5 — True Preemption / Recompute

目标不是 snapshot ordinary skipped rows。

压测制造 KV pressure，让 scheduler 真正 preempt：

```text
free KV
remove worker Ragged state
request returns waiting
later recompute
```

必须 reset Ragged state，不恢复 stale row。

Exit：preempt→recompute→continue correct，无 stale page ownership。

---

# R6 — TP=2 Uniform

### Supported scope

```text
TP2
uniform budget
identity/per-layer grouping
no layer/global cross-rank allocation
```

### Key contract

```text
local Hkv = model_config.get_num_kv_heads(parallel_config)
Glocal = Hkv_local / Hp
```

各 rank 可选不同 positions，但 scheduler-visible physical retained depth 必须 compatible；必要时 `all_reduce(MAX)` kept lengths。

Exit：TP2 identity + compressed closure + free/reuse + continuation。

---

# R7 — AOT Per-Layer Clustering

先 `page_group_size=1` 收集 retention profile：

```text
[layer,head]
```

再 per-layer cluster similar-retention heads。

比较：

```text
per-head ideal
adjacent grouping
AOT grouping
```

Metrics：allocated pages / residual waste / concurrency / throughput。

Cross-layer clustering 只是 stretch。

---

# R8 — Triton In-place Writeback

只有 profiler 证明 compaction data movement materially matters 时才做。

必须保留 reference scratch backend。

Compare：

```text
scratch bytes
compaction latency
DRAM traffic estimate
boundary overhead
TPOT impact
```

如果 e2e <1%，把它定位成 kernel engineering，不夸大 throughput。

---

# R9 — Piecewise CUDA Graph Hardening

目标：

```text
persistent ragged metadata buffers
fixed replay-facing addresses
pinned/UVA staging
reduce Python allocation
reduce decode CPU launch gap
```

Compare：

```text
Dense+Piecewise
Ragged eager
Ragged+Piecewise
```

Exit：不只是“不 crash”，而是有 CPU step / launch gap / TPOT evidence。

---

# R10 — Final Closure

最终输出：

```text
ARCHITECTURE.md
SOURCE_MAP.md
DESIGN_DECISIONS.md
IMPLEMENTATION_LOG.md
BENCHMARK.md
LIMITATIONS.md
EVIDENCE/
RESUME.md
INTERVIEW_QA.md
```

最终 benchmark 不复现整篇 Tangram；只回答项目自己的 systems claims。
