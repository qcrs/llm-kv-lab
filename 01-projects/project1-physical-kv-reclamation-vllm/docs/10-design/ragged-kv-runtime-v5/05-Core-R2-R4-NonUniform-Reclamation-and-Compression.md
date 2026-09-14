> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Core R2–R4 — Non-Uniform State, Reclamation, Compression

## 1. 为什么 R2/R3 比 scorer 更重要

如果先接 scorer，会同时出现：

```text
score 是否正确？
keep 是否正确？
per-group length 是否正确？
slot mapping 是否正确？
payload 是否正确？
block ownership 是否正确？
```

任何错误都难定位。

所以 Core 必须按：

```text
state → manual action → automatic policy
```

推进。

---

## 2. R2 — `E[r,l,g]` 成为 source of truth

### 状态定义

建议：

```text
RaggedKVState:
    effective_kv_lens[max_requests, G_total]
```

每项表示“下一次写入之前，此 group 已 compact 后真正有效的 physical token 数”。

本 step 调度 `q` 个 token：

```text
physical write positions(group)
= E[group] + arange(q)

physical read length(group)
= E[group] + q

logical model positions
= num_computed_tokens + arange(q)
```

### transaction 纪律

不要在 forward 前无条件 commit 新 E。

推荐：

```text
prepare
 → execute/write
 → compression/reclaim succeeds
 → postprocess commit E
```

失败路径必须保留上一步 canonical E。

---

## 3. R3 — Manual target length

### 为什么手动目标长度最好

它把系统 correctness 与算法质量完全分开。

例如：

```text
Layer0/G0: 1024
Layer0/G1: 2048
Layer1/G0: 1536
Layer1/G1: 768
```

只要能把这个 layout 真实实现，就证明 runtime 已支持 non-uniform state。

### Physical compaction

对于每 `(layer,group)`：

```text
source length
keep positions
 → compact payload to prefix
 → new length
 → n_blocks_new = ceil(new length / block_size)
```

然后只 free tail blocks。

### P1 reference 的角色

P1 `kv_compaction.py` 当前是：

```text
paged all-head KV
 → contiguous scratch
 → paged prefix
```

Ragged 后一个 page 只包含 `page_group_size` heads，因此可以自然把 reference 缩成 one-group compaction。

第一版可保留 scratch；它是 correctness oracle，不是最终性能路径。

---

## 4. Scheduler reconciliation

R3 最容易做错的是：Worker free 了 payload，但 Scheduler ownership 仍认为 request 拥有那些 IDs。

正确 transaction：

```text
worker:
  compact + detach logical row tail
  ↓
  report freed IDs / new per-group block counts

scheduler:
  validate request/current ownership
  ↓
  mutate canonical ownership
  ↓
  BlockPool free
```

需要保留 P1 的原则：

> payload transformation 不拥有 allocator free 权限。

### 新 result 结构建议

不要把 ragged result 塞进原 `CompactionResultData` scalar fields。

建议新建：

```text
RaggedCompactionResultData:
    request_id
    group_new_lengths
    freed_block_ids
    optional expected ownership generation/step
```

具体字段要在源码审计时再冻结，但 shape/ownership 语义必须独立。

---

## 5. R3 强验收

构造两请求：A、B。

### Before

```text
A owns groups with several pages
BlockPool free = F0
```

### Action

A 手动 non-uniform compact。

### After worker

```text
A ragged rows shortened
payload prefix matches oracle
```

### After scheduler

```text
freed IDs no longer in A canonical ownership
BlockPool free = F1 > F0
```

### Reuse

调度 B：

```text
B allocation intersects freed_ids(A)
```

### Continuation

A 再 decode N tokens，结果正确。

这比单纯报告 memory allocated bytes 有说服力得多。

---

## 6. R4 — 最小 Compression Control

### Module contract

```text
Scorer:
  Q/K/... → score[layer, head, position]

BudgetScope:
  score → count[layer, group]

KeepPlanner:
  count + protected region → sorted retained positions

Executor:
  keep plan → physical compaction
```

不要让 scorer 知道 BlockPool，也不要让 allocator 知道 score。

### 第一版只选 UniformScope

优点：

- 每 group count 一致，容易验证；
- 为 R6 TP=2 铺路；
- 与 non-uniform positions 仍兼容；
- physical system 已经是 ragged，即使 count uniform，也能验证 per-head/member mapping。

真正 layer/global budget 可以留为 optional extension，不纳入项目完成条件。

### Scorer 选择

推荐：

1. KeyDiff：实现相对直接、无需额外 checkpoint；或
2. 一个 Tangram 已验证且容易 port 的 deterministic scorer。

避免第一版 FastKVzip，因为 gate checkpoint / hidden-state capture 会引入额外 compilation/runtime side effects。

---

## 7. Compression boundary

建议 R4 第一版只在明确的 prefill compression boundary 执行，而不是每 decode step 动态驱逐。

优点：

- state transition 少；
- 容易与 chunked prefill 对齐；
- performance profiling 更清晰；
- 不会立刻引入频繁 reclaim 与 allocator churn。

后续需要时再扩 budget regime 的 repeated boundary。

---

## 8. Core Done 的 resume narrative

完成 R4 后可以准确描述：

> 基于 vLLM 0.26 Model Runner V2 重构 Head-Group Ragged KV physical layout，将 per-head/group retention 映射为独立 page depth；实现 worker-side KV compaction、scheduler-side ownership reconciliation 与 BlockPool physical reuse，使 logical KV compression 转化为真实 GPU capacity。

这已经是完整 AI Infra 项目核心，不需要依赖 TP/CG 才成立。

---

# v2 Addendum — R4 拆成 A/B

为了同时保证可完成性与最终 non-uniform claim：

- **R4-A**：one scorer + UniformScope。只证明 scoring/control-plane能驱动 R3 physical data plane。
- **R4-B**：same scorer + LayerScope。证明真实 policy自动产生 unequal `E[r,l,g]` 与 page depths。

Core Functional可以在 R4-A关闭；Flagship Non-Uniform在 R4-B关闭。这样即使 policy调优不理想，R3的 manual non-uniform runtime证据仍然成立，不会让整个项目失败。

