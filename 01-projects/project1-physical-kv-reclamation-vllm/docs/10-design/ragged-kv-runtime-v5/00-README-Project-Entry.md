# Tangram → vLLM 0.26 Ragged KV Runtime Porting Program — Definitive Engineering Plan v5

**Date:** 2026-09-14  
**Primary implementation base:** `qcrs/vllm:p1/v2-token-compaction-v026@bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00`  
**Clean baseline:** `qcrs/vllm:study-vllm-0.26@568afb3a13806beb53bb2e6bd518269357b237c0`  
**Reference:** `aiha-lab/tangram@6fa551fc8f6edcc118a2a39b3554ee1520e91edd`  
**Project class:** LLM Serving Runtime / KV Memory Management / GPU Runtime / Distributed Inference

---

## 0. v5 Executive Freeze

> **v5 critical addition:** 从 R2 开始，per-group physical frontier 必须同时存在于 scheduler-side Ragged KV manager 与 Worker execution mirror。Scheduler/Allocator 不能只依赖 logical `num_computed_tokens` 或 scalar max(E)，否则压缩后的短 group 会在后续 decode 被重新扩容，破坏 non-uniform physical ownership。

新的 canonical docs：

```text
37 Architecture Audit / remaining decisions
38 Scheduler-side Ragged Physical State / allocation
39 Step Metadata shape / lifetime / attention
40 Resume project success / deliverables / stop rules
41 Slice / file ownership / dependency DAG
```

发生冲突时，优先级：

```text
00 > 37–41 > 22/23/28/29–36 > older background docs
```

## 0. v4 Historical Executive Freeze

本项目不是 Tangram paper reproduction，也不是 scorer 项目。

项目唯一主线冻结为：

> **在 vLLM 0.26 Model Runner V2 上重新实现支持 per-layer / per-head-group 非均匀 KV retention 的 Ragged KV Runtime，把逻辑 retention 变成真实 GPU physical page reclamation，并打通 Scheduler、Allocator、BlockTable、KV Write、FlashAttention Read、TP、Preemption 与 Piecewise CUDA Graph。**

最终系统闭环必须能够画成：

```text
Retention / Budget Decision
        ↓
RaggedKVState E[request, layer, group]
        ↓
Head-Group Physical Pages
        ↓
MRv2 RaggedBlockTables
        ↓
Member-aware KV Write
        ↓
Member-major FA2 Read
        ↓
Paged KV Compaction
        ↓
Worker Compaction Receipt
        ↓
Scheduler Source-Fence Validation
        ↓
Canonical Ownership Trim
        ↓
BlockPool Free
        ↓
Cross-request Physical Page Reuse
        ↓
Original Request Continues Decode
```

只有最后四步真实成立，项目才不是“逻辑删除”。

---

## 1. v4 相对 v3 的关键变化

### 1.1 新增 R1-A3：Physical Layout / Stride Contract

这是 v4 最重要的架构修正。

qcrs/v0.26 FA cache 支持 NHD/HND；无 connector 时默认倾向 NHD physical order。Tangram 的 virtual block zero-copy 则依赖 `(physical_page, column)` 连续，因此不能只看 logical shape 就直接 flatten。

v4 冻结：

```text
Ragged physical storage:
[Bphys, Hp, N, 2D]

virtual single-head view:
[Bphys * Hp, 1, N, 2D]
```

且必须满足：

```text
physical.data_ptr == virtual.data_ptr
stride contract exact
physical (page,column) ↔ virtual block bijection
```

在这条 Gate 未通过前，禁止进入 real-engine attention。

### 1.2 MRv2 Write / Read 必须分开设计

当前 v0.26 已存在：

```text
vllm::unified_kv_cache_update
        ↓
vllm::unified_attention_with_output
```

因此 Ragged 不能照搬旧 Tangram 一体化 forward；必须分别完成：

```text
R1-D1 Ragged KV Write
R1-D2 Decode Attention Read
R1-D3 Prefill/Mixed Attention Read
```

### 1.3 TP 的目标重新定义

TP=2 不是“torchrun 能启动”，而是：

```text
rank-local KV head geometry
+ rank-local retention
+ rank-consistent scheduler-visible physical depth/free boundary
```

首版只做：

```text
TP=2
uniform budget scope
identity/per-layer local grouping
```

使用 vLLM authoritative per-rank KV-head geometry；必要时通过 TP collective 对齐 physical retained length/page depth。

### 1.4 CUDA Graph 不另造体系

首版不新增 `unified_attention_ragged`。

直接复用当前 v0.26 splitting seam：

```text
unified_kv_cache_update
unified_attention_with_output
```

Ragged 动态 metadata 留在 eager island；模型主体仍走 piecewise compilation/CUDA Graph。后期优化 persistent buffers / pinned-UVA staging / decode CPU overhead。

### 1.5 Preemption 按 MRv2 lifecycle 重写

普通“本 step 未调度”不等于 preemption；persistent request row 不应 snapshot/restore。

真正 scheduler preemption：

```text
free KV
remove worker runtime state
request returns waiting
later recompute
```

必须 reset：

```text
RaggedBlockTables row
RaggedKVState E[r,g]
compression state
slot-score state
pending plan
```

不能恢复 stale compressed row。

---

## 2. Completion Boundary

### MUST — Core

```text
R1 Identity Ragged Paging
R2 Per-Group Physical State
R3 Manual Non-Uniform Physical Reclamation
R4 One Real Policy
```

Core 必须证明：

```text
representation supports non-uniform state
attention consumes it correctly
logical retention changes real physical ownership
BlockPool free count increases
freed page is reused
original request continues decode
```

### STRONGLY RECOMMENDED — Systems Breadth

```text
R5 True Preemption / Recompute
R6 TP=2 Uniform
R9 Piecewise CUDA Graph Hardening
```

### ONE RECOMMENDED OPTIMIZATION

```text
R7 AOT Per-Layer Head Clustering
```

### PROFILE-DRIVEN ONLY

```text
R8 Triton In-place Writeback
Bulk int32 Ring Allocator
```

不要为了 feature 数量阻塞 Core。

---

## 3. Canonical v4 Execution Order

```text
R0    Freeze Source / Env / Reference

R1-A1 RaggedAttentionSpec / Config Contract
R1-A2 Global Physical Page Pool / Shared Backing
R1-A3 Physical Layout / Stride Contract
R1-B  MRv2 RaggedBlockTables
R1-C  Virtual Block / Slot / Member Oracle
R1-D1 Ragged KV Write
R1-D2 Decode FA2 Read
R1-D3 Prefill / Mixed FA2 Read
R1-E  Real Engine Identity
R1-F  Piecewise CUDA Graph Compatibility Smoke

R2    E[request, group] Physical Divergence
R3    Manual Reclaim → Scheduler Reconcile → Free → Reuse
R4    One Scorer / Automatic Policy
R5    True Preemption / Recompute
R6    TP=2 Uniform
R7    AOT Per-Layer Clustering
R8    Triton In-place — only after profiling
R9    Piecewise CG Hardening / Persistent Metadata
R10   Final Benchmark / Resume Closure
```

---

## 4. Canonical Document Reading Order

v4 建议按下面顺序读，不需要从 01 顺序读到最后：

1. `00-README-Project-Entry.md` — 项目入口与冻结范围。
2. `29-vLLM026-Ragged-KV-Resume-Grade-Systems-Deep-Dive-v4.md` — 第四轮源码审计完整结论。
3. `30-vLLM026-End-to-End-Source-Chain-and-Modification-Map.md` — 从 startup 到 free/reuse 的 vLLM 主链与修改点。
4. `34-Physical-Layout-Stride-and-Virtual-Block-Exact-Contract.md` — R1-A3 的 zero-copy / stride contract。
5. `22-R1-Exact-Engineering-Contract.md` — R1 A1→F 完整实现契约。
6. `23-R2-R4-Physical-State-Reclaim-and-Policy-Contract.md` — Core 非均匀状态与 reclaim。
7. `31-TP2-Distributed-Ragged-KV-Exact-Engineering-Contract.md` — TP=2。
8. `33-Preemption-Lifecycle-and-State-Reset-Exact-Contract.md` — lifecycle。
9. `32-Piecewise-CUDA-Graph-Exact-Engineering-Contract.md` — compile / CG。
10. `07-Optimization-AOT-Triton-CUDAGraph.md` — AOT / Triton / CG optimization rationale。
11. `35-Resume-Technical-Stack-and-Evidence-Matrix.md` — 简历 claim 与 evidence。
12. `36-Execution-State-Evidence-and-Handoff-Template.md` — 后续每个 Slice 的状态/evidence/handoff 模板。

`01`–`28` 中未被 v4 重写的文件保留为 detailed background / provenance；若与 v4 canonical docs 冲突，以 `00 / 22 / 28 / 29–36` 为准。

---

## 5. Core Supported Slice

第一阶段明确只支持：

```text
GPU: A100
Attention: FlashAttention 2
Model family: Qwen-style full attention
KV dtype: BF16 / FP16
KVQuantMode.NONE
head_size_v == head_size
TP=1 initially
PP=1
DCP=1
PCP=1
prefix cache off
spec decode off
KV connector off
offload off
MLA off
SWA/hybrid off
```

R6 只增加 TP=2 uniform。

---

## 6. 项目技术栈

这个项目最终应当明确体现：

| Layer | Technology / Concept | Project Evidence |
|---|---|---|
| Serving | Scheduler / continuous batching / preemption | R3/R5 real-engine lifecycle |
| KV Memory | PagedAttention / BlockPool / ownership | R1/R3 physical page evidence |
| Attention | FA2 varlen / GQA | R1-D2/D3 |
| GPU Layout | NHD/HND / stride / alias / virtual block | R1-A3/R1-C |
| GPU Kernel | Triton slot mapping / compaction | P1 + R8 optional |
| Runtime | PyTorch custom op / forward context | R1-D1/D2 |
| Compiler | torch.compile / splitting ops | R1-F/R9 |
| CUDA Graph | persistent buffer / replay address | R9 |
| Distributed | TP / NCCL collective | R6 |
| Metadata | UVA / pinned memory / staged writes | R1-B/R9 |
| Correctness | source fence / scheduler authority | R3 |
| Performance | TTFT / TPOT / page accounting / profiler | R10 |
| Offline Optimization | AOT head clustering | R7 |

这也是为什么项目最终应描述为：

> **KV Cache Runtime / Serving System**

而不是：

> **KV Cache Algorithm**

---

## 7. Hard Stop Rules

以下规则后续不再反复讨论：

```text
R1-E 未过 → 禁止进入 scorer
R3 未过 → 禁止把项目称为 physical reclamation runtime
reference path 未稳定 → 禁止先写 in-place Triton
Ragged eager 未稳定 → 禁止调 CUDA Graph
TP1 Core 未稳定 → 禁止开 TP2
profiler 未证明 allocator/compaction 是瓶颈 → 不做 bulk allocator / in-place optimization
```

---

## 8. Immediate Next Slice

当前立即执行：

```text
R1-A1 — RaggedAttentionSpec + Config Contract
```

只允许触碰：

```text
vllm/config/cache.py
vllm/v1/kv_cache_interface.py
vllm/v1/kv_cache_spec_registry.py / equivalent registration
vllm/model_executor/layers/attention/attention.py  # spec return only
focused tests
```

不允许触碰：

```text
RaggedBlockTables
FA forward
compression
scheduler free
TP collectives
CUDA Graph modifications
```

R1-A1 通过后立即进入 R1-A2；**R1-A2 通过后必须先做 R1-A3 layout/stride gate，再做 BlockTable/Attention。**
