> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Codex Slice Execution Contracts

本项目建议继续使用：

```text
ChatGPT Web = Architect / Mentor / Reviewer
Codex CLI    = Slice Execution Agent
```

每次只交一个可验证 Slice。

---

## 1. 通用 Slice Contract

每个 Slice prompt 必须包含：

```text
BASELINE
GOAL
IN-SCOPE
OUT-OF-SCOPE
SOURCE PATHS TO READ
FROZEN INVARIANTS
EXPECTED FILES
ACCEPTANCE
EVIDENCE REQUIRED
STOP CONDITIONS
```

禁止只写：

> 帮我实现 Ragged KV。

---

## 2. R1-A Prompt Skeleton

```text
Task: R1-A RaggedAttentionSpec + Physical Geometry Contract

Baseline:
qcrs/vllm branch p1/v2-token-compaction-v026
HEAD bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00

Reference:
aiha-lab/tangram commit 6fa551fc8f6edcc118a2a39b3554ee1520e91edd

Goal:
Introduce the minimum vLLM 0.26-native KV spec/config representation needed
for head-group pages. No runtime ragged attention yet.

Read first:
- vllm/config/cache.py
- vllm/v1/kv_cache_interface.py
- vllm/model_executor/layers/attention/attention.py
- vllm/v1/core/kv_cache_utils.py
- Tangram corresponding RaggedAttentionSpec implementation

Frozen invariant:
At compression=off, total max KV bytes for one request/layer are equivalent
to FullAttentionSpec: smaller page_size × more head groups.

Supported slice scope:
- Full attention
- KVQuantMode.NONE
- head_size_v == head_size
- TP=PP=DCP=PCP=1 for runtime assumptions

Out of scope:
- BlockTables
- slot mapping
- attention forward
- compression
- allocator free
- TP
- CUDA Graph changes

Acceptance:
- config disabled path byte/behavior unchanged
- validation of page_group_size
- page byte math unit tests
- max-memory accounting tests
- unsupported combos fail explicitly

Do not enter the next slice.
```

---

## 3. R1-B Prompt Skeleton

重点强调：

```text
Do not port Tangram RaggedBlockTable verbatim.
Design against MRv2 BlockTables / StagedWriteTensor first.
```

要求 Codex 先输出：

```text
CURRENT DATA SHAPE
PROPOSED DATA SHAPE
ROW FLATTEN FORMULA
OPERATIONS AFFECTED
CUDA-GRAPH POINTER-STABILITY IMPACT
```

Review 通过后再实现。

---

## 4. R1-D Prompt Skeleton

```text
Goal:
Implement the minimum ragged FA2 identity path.

Order:
1. decode-only synthetic
2. prefill/member-major synthetic
3. real metadata integration

Never change FlashAttention mathematical kernel.
If implementation appears to require changing the FA kernel itself, STOP and
report DESIGN_CONFLICT.
```

这是一个重要 stop condition，防止 scope 失控。

---

## 5. R3 Prompt Skeleton

```text
Goal:
Given an explicit per-(layer,group) target/keep plan, compact KV and produce
freed physical block IDs, then reconcile scheduler ownership and prove reuse.

No scorer / automatic policy is allowed in this slice.

Required evidence:
- before/after group block rows
- payload reference equality
- freed block IDs
- BlockPool free count delta
- second request reuse
- first request continuation
```

---

## 6. R6 TP Prompt Skeleton (legacy template, v5 numbering)

```text
Scope:
TP=2 + UniformScope only.

Required design facts before coding:
- global num_kv_heads
- rank-local num_kv_heads
- groups per rank
- head/global-local mapping
- which state is rank-local
- which state must be synchronized
- collective type and exact commit point

If LayerScope/GlobalScope is needed to pass the slice, STOP: scope is wrong.
```

---

## 7. R8 Triton Prompt Skeleton

要求先 benchmark reference，再进入实现：

```text
Gate:
If reference compaction is not a material cost in the selected workload,
implement only the kernel microbenchmark/optimization slice; do not claim
end-to-end speedup.
```

必须保留 Torch/reference path，不允许 optimization 替换掉唯一 oracle。

---

## 8. R9 CUDA Graph Prompt Skeleton

```text
Do not attempt Full CUDA Graph.

Target:
Reuse v0.26 piecewise CG and existing attention eager-break seam.

First deliverable:
A compatibility audit that traces:
Attention.forward
→ unified_attention_with_output
→ compilation splitting op
→ CudaGraphManager capture/replay

Only after the audit propose code changes.
```

---

## 9. Agent Review Checklist

每次 Codex 返回后，Web Review 固定问：

1. 改动是否越过 Slice scope？
2. 是否改变 dense/default path？
3. source-of-truth state 是谁？
4. commit point 在哪里？
5. failure path 是否留 stale state？
6. test 是否证明 invariant，而不只是函数能运行？
7. 是否把 Tangram old-vLLM assumption 带进 MRv2？
8. 是否引入不必要 Python hot-path overhead？
9. 是否为后续 TP/CG 留下不可维护 shape union？
10. Evidence 能否单独支持 acceptance？

通过后再进入下一 Slice。

---

# v2 Codex Contract Rule

R1 不再一次给 `Spec + BlockTable + FA`。依次发：`A1 Spec` → `A2 Global Pool` → `B BlockTable` → `C Layout` → `D1 Write` → `D2 Decode` → `D3 Prefill` → `E Engine`。每个 Slice prompt 必须写：allowed files / forbidden surfaces / source facts / design decision / acceptance / evidence / STOP_ON_DESIGN_CONFLICT。特别是 D1 必须在任何 attention read实现前单独关闭。

