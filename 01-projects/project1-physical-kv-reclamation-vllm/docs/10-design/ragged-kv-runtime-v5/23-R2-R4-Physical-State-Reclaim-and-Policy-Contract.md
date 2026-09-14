> **v5 canonical addendum:** R2+ allocation/reclamation 必须遵循 `38-Scheduler-Side-Ragged-Physical-State-and-Allocation-Contract.md`；metadata shape/lifetime 遵循 `39-Ragged-Step-Metadata-Shape-Lifetime-and-Attention-Contract.md`。

> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v5 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# R2–R4 Exact Contract — From Ragged Identity to Real Non-Uniform Reclamation

R1 只证明 representation。R2–R4 才把项目真正变成 non-uniform physical runtime。

## R2 — Per-group physical state

### v5 R2 prerequisite — Scheduler-side physical frontier

R2 不再只新增 Worker `E[r,g]`。必须同时建立 scheduler-side canonical `E[req,g] / counts[req,g] / rows[req,g]`，下一步 allocation 用 `ceil((E[g]+q)/B)` 逐 group 计算。详细合同见 `38-...Allocation-Contract.md`。

关键验收新增：**R3 reclaim 后继续 decode 时，短 group 不得因为 allocator 只看 logical length 而重新长回 max depth。**

# 1. 不改 P1 scalar state

P1 现有：

```text
effective_kv_len[r]
```

Ragged 新建独立：

```text
RaggedKVState.effective_kv_lens[r, GT]
GT = local_layers * groups_per_layer
```

不要把旧字段改成 1D/2D union。

### 2. Logical vs physical

对每 request：

```text
logical model position:
Plogical = num_computed_tokens[r] + query_offset

physical cache position for group g:
Pphysical[g] = E[r,g] + query_offset
```

RoPE / model position必须使用 logical；slot mapping使用 physical。

### 3. Current-step update

若本 step query_len=q：

```text
source_E_after_forward[g] = E_before[g] + q
```

只有 compaction成功后：

```text
E_after[g] = kept_len[g]
```

### 4. Acceptance

人工构造：

```text
E groups = [48,32,16,...]
```

下一 decode token写到：

```text
[48,32,16,...]
```

且每 group read length分别为 `[49,33,17,...]`。

---

## R3 — Manual Non-Uniform Reclaim

R3 不接 scorer。输入就是人工指定 per-group keep/target。

### 1. 为什么先 manual

这样把错误空间限制为：

```text
layout
payload compaction
row trimming
scheduler ownership
```

而不是同时怀疑 scorer/policy。

### 2. Transaction source

复用 P1 思路，plan必须描述**post-forward source**。

建议 Ragged plan：

```text
request_id
step_seq/source_version
expected_source_effective_lens [GT]
expected_source_num_blocks     [GT]
keep decision / target lengths
```

Worker result：

```text
request_id
step_seq
new_effective_lens [GT]
new_num_blocks     [GT]
```

### 3. Payload backend order

第一版：

```text
Torch/P1 scratch gather → compact retained positions → writeback prefix
```

不是立刻做 in-place Triton。

原因：R3 的目标是验证 physical semantics，不是 kernel performance。

### 4. Trim semantics

对每 group：

```text
new_blocks[g] = ceil(new_E[g]/B)
old_blocks[g] = current_count[g]

freed tail IDs = row[g][new_blocks[g]:old_blocks[g]]
```

worker execution row可先 trim；allocator canonical free必须由 Scheduler/KV manager reconcile。

### 5. P1 ownership reuse

P1 已有最重要的不变量：

```text
Worker transforms payload/execution view
Scheduler owns canonical physical allocation/free
```

Ragged只把 scalar fence/count扩成 vector，不改变 authority。

### 6. R3 canonical scenario

建议固定一条非常容易读懂的 evidence case：

```text
block_size = 16
Hp = 2 or 4
at least 2 groups

before:
G0 E=48 blocks=[1,2,3]
G1 E=48 blocks=[4,5,6]

manual compact:
G0 keep 40 → 3 blocks
G1 keep 16 → 1 block

result:
G0 [1,2,3]
G1 [4]
freed = {5,6}

second request allocation
must receive one of {5,6}

first request continues decode
G0 writes at pos 40
G1 writes at pos 16
```

这条 evidence 足以证明项目最核心 claim。

### 7. Same-step allocation

先分两阶段：

R3.1：禁止 compaction boundary 同 step 发生新 page allocation。

R3.2：补 support：source fence必须包含 current-step newly allocated pages。

这样降低第一次 closure难度。

---

## R4-A — One real scorer, safe policy

### Scorer选择建议

优先：**KeyDiff**。

理由：

- Tangram 有独立 source `compression/keydiff.py`；
- gate-free；
- 不依赖 checkpoint；
- query-independent；
- 可以对 cached keys rescore；
- 非常适合简历项目的“runtime focus，policy不是主角”。

如果只追实现最简单，也可先 StreamingLLM，但亮点弱于 KeyDiff。

### Budget第一阶段

先做 uniform retained count：

```text
每 group 保留相同 K
位置由 scorer决定
```

它验证 scorer→executor data plane，但尚未产生 count-level non-uniform。

---

## R4-B — Automatic non-uniform assignment

这是 flagship policy closure。

### 推荐先做 per-layer scope

同一 layer 内 groups 共享总 budget：

```text
sum_g K[l,g] = layer_budget
K[l,g] can differ
```

不要第一步做 global cross-layer scope。

理由：

- 避免 cross-layer score scale问题；
- 不打开 cross-layer cluster ownership；
- Tangram 本身也明确区分 layer/global/uniform scope；
- TP1限定清晰。

### Required output

至少一次真实 prompt中看到：

```text
K[l,g0] != K[l,g1]
```

随后：

```text
physical block counts differ
free pages increase
request continues decode
```

### Quality check

项目不是 accuracy paper，不需要大规模 SOTA；但至少做：

```text
Dense
vs Ragged identity
vs Compression at 2-3 retention levels
```

用一个长上下文任务/困惑度或简单 benchmark确认不是完全破坏质量。

---

## R4 after closure

一旦 R4-B 完成，项目最核心系统 claim 已闭环：

```text
non-uniform decision
→ non-uniform physical state
→ real page reclamation
→ reuse
```

后续 TP/preemption/AOT/Triton/CG 都是 breadth/optimization，不再改变项目是否成立。

---

# v4 Clarification

R3 的 closure 必须视为 atomic transaction：

```text
payload correct
+ worker row shrink
+ scheduler canonical ownership match
+ BlockPool free increase
+ exact released ID reuse
+ original request continuation
```

缺一项都不能把 R3 标为 CLOSED。

