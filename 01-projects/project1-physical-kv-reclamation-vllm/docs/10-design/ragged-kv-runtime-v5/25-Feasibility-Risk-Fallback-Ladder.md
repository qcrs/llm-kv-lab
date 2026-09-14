> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Feasibility, Risk & Fallback Ladder — v3

目标不是保证所有 stretch feature都成功，而是保证**项目核心一定有一个强而完整的落点**。

## 1. 风险排序

### P0 — 必须解决，否则项目不成立

1. Global page pool/backing memory accounting。
2. Ragged block rows能正确表达 `(request,layer,group)`。
3. KV write member mapping正确。
4. FA decode/prefill member layout正确。
5. scheduler/worker ownership一致。

### P1 — Core需要，但已有强 reference

6. per-group E。
7. manual compaction/free/reuse。
8. one scorer data-plane wiring。

### P2 — 可以失败但项目仍成立

9. TP2。
10. preemption breadth。
11. AOT clustering收益。
12. Triton in-place正 speedup。
13. CUDA Graph性能收益。

## 2. 每个硬点的 fallback

### Global backing reshape失败

Fallback顺序：

```text
A. dedicated contiguous ragged shape in attn_utils
B. custom as_strided view with strict alias tests
C. temporarily allocate one explicit ragged backing object outside generic helper
```

禁止 fallback 到“每 layer一个pool”，因为那会破坏跨 layer/group page reuse的核心 claim。

### MRv2 flattened row太难

Fallback：先实现独立 `RaggedBlockTables` CPU mirror + persistent GPU tensor copy，证明正确性；等 R1-E 后再替换成 staged-write优化。

这会慢，但不损害语义。

### Prefill member-major太复杂

Fallback：先完成 decode-only synthetic不是项目完成条件；但可以把 real-engine identity拆成：

```text
phase 1: single request prefill
phase 2: multi-request prefill
phase 3: mixed batch
```

不能永久跳过 prefill，因为真实 request必须建立 cache。

### KV write custom-op integration卡住

Fallback：先在 attention adapter里显式执行 write，`enforce_eager` 下验证；然后再恢复 `unified_kv_cache_update` compile seam。

### Standard BlockPool对象数量太大

semantic tests：降低 `kv_cache_memory_bytes` / override。

capacity benchmark前：port Tangram ring allocator。

### In-place Triton失败

保留：

```text
P1 PyTorch/Torch scratch oracle
或 existing Triton gather→scratch→writeback
```

项目主 claim不是“零 scratch”。

### AOT clustering没有收益

仍可以给出 negative result：

```text
adjacent grouping waste
vs clustered grouping waste
```

如果目标模型 retention profile本来接近，说明 grouping不是主要瓶颈；不影响 runtime claim。

### TP2失败

TP=1 flagship仍成立。TP2标 stretch，不允许拖住招聘交付。

## 3. 最小可交付台阶

### L1 — Strong Systems Demo

```text
R1 Identity
+ R2 per-group state
+ R3 manual non-uniform free/reuse
```

已经能展示：

> 我重构了 vLLM 0.26 的 KV physical representation，使 head-groups具有独立 physical depth，并完成真实 page reclamation/reuse。

这本身就比“实现一个 eviction policy”强。

### L2 — Flagship

```text
+ R4 one scorer + automatic non-uniform
```

完整故事：policy→runtime→allocator。

### L3 — Production Breadth

```text
+ TP2
+ preemption
```

### L4 — Optimization

```text
+AOT/Triton/CG 任意2项有可信 evidence
```

## 4. 为什么这个路线比旧 P1 更稳

旧 P1 最大问题是项目价值容易落在“我把保留token copy了一遍然后free blocks”。新路线的系统对象更明确：

```text
physical unit changed
allocation namespace changed
block-table topology changed
attention sequence abstraction changed
runtime state changed from scalar→matrix
```

即使 policy效果一般，系统工程本身仍有独立价值。

而且 Tangram 已经证明这套 representation可工作，所以科学风险显著低于从零设计。
