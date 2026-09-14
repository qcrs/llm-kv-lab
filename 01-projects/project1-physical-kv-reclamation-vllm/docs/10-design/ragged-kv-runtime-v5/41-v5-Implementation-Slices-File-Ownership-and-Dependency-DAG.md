# v5 Implementation Slices — File Ownership & Dependency DAG

**Status:** CANONICAL execution map

---

# 1. Dependency DAG

```text
R1-A1 Spec
  ↓
R1-A2 Global Pool
  ↓
R1-A3 Layout/Stride
  ↓
R1-B RaggedBlockTables
  ↓
R1-C Address Oracle
  ↓
R1-D1 KV Write
  ↓
R1-D2 Decode Read
  ↓
R1-D3 Prefill/Mixed Read
  ↓
R1-E Real Engine Identity
  ↓
R1-F Piecewise CG Smoke
  ↓
R2 Scheduler+Worker Per-group Physical State
  ↓
R3 Manual Reclaim Transaction
  ↓
R4 One Policy
  ↓
 ┌───────────┬─────────────┬──────────────┐
 R5 Preempt   R6 TP2        R7 AOT
                               ↓
                         optional R8 Triton

R9 CG hardening can start after R1-F,
but final optimization measurement should use R4/Core runtime.
```

---

# 2. R1-A1 — Spec Only

Primary files：

```text
vllm/config/cache.py
vllm/v1/kv_cache_interface.py
vllm/v1/kv_cache_spec_registry.py
vllm/model_executor/layers/attention/attention.py
```

Forbidden：

```text
scheduler
block pool implementation
attention forward
compression
```

Acceptance：page math / guards / dense regression。

---

# 3. R1-A2 — Global Pool / Shared Backing

Primary：

```text
vllm/v1/core/kv_cache_utils.py
vllm/v1/worker/gpu/attn_utils.py
```

Acceptance：

```text
ragged num_blocks = available_memory/page_bytes
one backing shared_by all supported layers
capacity accounting exact
```

不要 port ring allocator。

---

# 4. R1-A3 — Layout

建议新模块：

```text
vllm/v1/attention/backends/ragged_layout.py
```

以及最小 reshape hook：

```text
vllm/v1/worker/gpu/attn_utils.py
```

Acceptance：zero-copy address oracle。

---

# 5. R1-B — RaggedBlockTables

建议新文件：

```text
vllm/v1/worker/gpu/ragged_block_table.py
```

复用：

```text
buffer_utils.StagedWriteTensor
UvaBackedTensor
```

首版 API：

```text
add_request_identity
append_group_pages
remove_request
move_request
gather_step_rows
get_counts
```

Core 不实现 snapshot/restore。

---

# 6. R1-C — Pure Addressing

主要在：

```text
ragged_layout.py
```

禁止改 Scheduler/FA kernel。

---

# 7. R1-D1 — KV Write

Primary：

```text
attention.py dispatch hook
flash_attn.py do_kv_cache_update adapter
ragged_layout.py member slots
```

原则：尽量继续调用现有 `reshape_and_cache_flash`。

---

# 8. R1-D2/D3 — Attention Read

Primary：

```text
vllm/v1/attention/backends/flash_attn.py
new ragged_forward.py or ragged helper module
vllm/v1/worker/gpu/attn_utils.py metadata integration
```

D2 只 decode。
D3 才 materialize prefill member-major。

---

# 9. R1-E — Engine Glue

Primary：

```text
GPUModelRunner
DefaultModelState.prepare_attn
InputBatch only if necessary
```

要求 hook 数量尽量小；发现大量 `if ragged` 散到 generic runner 时停止 review。

---

# 10. R2 — Scheduler + Worker Vector State

新增/修改：

```text
scheduler-side ragged manager state
worker RaggedKVState
SchedulerOutput block delta
GPUModelRunner consume delta
```

必须先实现 allocation-after-compression test，再进入 R3。

---

# 11. R3 — Reclamation

复用：

```text
P1 CompactionPlan/Result transaction semantics
P1 reference compaction
Tangram per-group writeback reference
```

修改：

```text
sched/output.py vector carrier
scheduler.py reconciliation
ragged manager canonical trim/free
worker compaction per group
```

Acceptance：free/reuse/continue decode。

---

# 12. R4 — One Policy

新增 compression control-plane package时保持隔离：

```text
scorer
budget
plan builder
```

不允许 policy 直接碰 BlockPool。

---

# 13. R5 — Preemption

Primary：

```text
scheduler preempt output
GPUModelRunner.finish/remove path
RaggedBlockTables.reset
RaggedKVState.reset
compression state reset
```

不增加 ordinary skipped snapshot。

---

# 14. R6 — TP2

Primary：

```text
RaggedGeometry local Hkv
compression budget reconcile
TP collective
result validation
```

不修改 global head semantic assumptions到处散落；统一从 ModelConfig authoritative API 获取 local Hkv。

---

# 15. R7 — AOT

主要独立于 serving core：

```text
tools/head_group_clustering/
cluster map loader
ragged_layout member map
```

首版 per-layer only。

---

# 16. R8 — Triton

只有 profiler gate PASS 后：

```text
reference writeback retained
tuned Triton backend added
bitwise/close differential required
```

---

# 17. R9 — CUDA Graph

Primary：

```text
persistent RaggedStepBuffers
metadata staging
existing compilation splitting seam
```

禁止默认新建第三个 attention custom op。
