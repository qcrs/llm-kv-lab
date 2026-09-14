# Implementation File Plan & Dependency DAG — v4

目标：把 feature complexity 从 `GPUModelRunner` / generic managers 中隔离出来，让 Dense/P1 path 尽可能不受影响。

---

# 1. Recommended New Modules

```text
vllm/v1/worker/gpu/
  ragged_block_table.py
    # [R*GT,M] staged/persistent rows
    # per-group counts
    # active-row gather
    # member/group slot mapping

  ragged_kv_state.py
    # E[request,group]
    # CPU/GPU staged mirrors
    # reset/preemption semantics

  ragged_runtime.py
    # runner-facing façade
    # step preparation
    # block/member views
    # post-forward state transition

vllm/v1/attention/backends/
  ragged_layout.py
    # physical stride contract
    # virtual block view
    # member map
    # CPU/PyTorch addressing oracle

  ragged_forward.py
    # member-major decode adapter
    # prefill/mixed materialization
    # inverse output mapping

vllm/v1/core/
  ragged_kv_cache_manager.py   # preferred if generic manager branching grows

vllm/v1/attention/compression/
  ragged_plan.py
  ragged_executor.py
  ragged_writeback.py          # R4/R8, not R1
```

---

# 2. Existing Files / Maximum Allowed Responsibility

## `vllm/config/cache.py`

```text
page_group_size
minimal feature flags
validation dispatch
```

不要放 runtime logic。

## `vllm/v1/kv_cache_interface.py`

```text
RaggedAttentionSpec
page bytes
max blocks/geometry properties
```

保持：

```text
semantic Hkv != storage Hp
```

## `vllm/v1/kv_cache_spec_registry.py`

只做 spec registration / manager selection。

## `vllm/v1/core/kv_cache_utils.py`

Ragged dedicated branch：

```text
global page count
one shared raw backing
capacity accounting
```

不要偷偷改变 normal `get_num_blocks()` 所有 caller 的语义。

## `vllm/v1/worker/gpu/attn_utils.py`

```text
Ragged raw backing allocation/reshape dispatch
metadata builder plumbing
```

## `vllm/v1/worker/gpu/model_runner.py`

只保留 orchestration：

```text
self.ragged_runtime = maybe_create(...)
ragged_runtime.apply_scheduler_delta(...)
ragged_runtime.prepare_step(...)
ragged_runtime.post_forward(...)
```

不把 scorer / clustering / compaction kernels 全塞入 runner。

## `vllm/model_executor/layers/attention/attention.py`

只做：

```text
unified_kv_cache_update dense/ragged dispatch
unified_attention_with_output dense/ragged dispatch
```

## `vllm/v1/core/sched/output.py`

R3 才新增：

```text
RaggedCompactionPlanData
RaggedCompactionResultData
RaggedBlockDelta if needed
```

## `vllm/v1/core/sched/scheduler.py`

R1 不进入 compression reconciliation。
R3 开始接 source-fence validation / canonical ownership trim。

---

# 3. Exact Per-Round Modification Surface

| Round | Existing Files | New Modules | Forbidden |
|---|---|---|---|
| R1-A1 | cache.py, kv_cache_interface.py, registry, attention.py(spec only) | optional validation helper | scheduler, FA forward, compression |
| R1-A2 | kv_cache_utils.py, attn_utils.py | planner helper | block table, scorer |
| R1-A3 | attn_utils.py/backend shape hook | ragged_layout.py | scheduler/free |
| R1-B | model_runner small hook | ragged_block_table.py | attention math |
| R1-C | none/minimal | ragged_layout.py | compression |
| R1-D1 | attention.py, update op path | ragged runtime/layout helpers | scheduler free |
| R1-D2/D3 | flash_attn.py/attention seam | ragged_forward.py | policy |
| R1-E/F | runner/model_state minimal | none | compression |
| R2 | scheduler manager/state + sched/output + worker state/runner hooks | ragged_kv_state.py + scheduler ragged physical-state helper | free/policy |
| R3 | sched/output.py, scheduler.py, manager | compaction plan/executor | scorer zoo |
| R4 | minimal runner hook | compression control plane | TP layer/global first |
| R5 | preempt/remove hooks | state reset helper | row snapshot for ordinary skip |
| R6 | config/parallel hooks | TP sync helper | DCP/PCP |
| R7 | none/minimal runtime map load | offline clustering tools | cross-layer first |
| R8 | none | triton writeback | remove reference backend |
| R9 | compilation/metadata small hooks | RaggedStepBuffers | Full CG first |

---

# 4. Dependency DAG

```text
R0
 ↓
A1 Spec
 ↓
A2 Global Pool
 ↓
A3 Layout/Stride Gate
 ↓
B MRv2 RaggedBlockTables
 ↓
C Virtual Addressing
 ↓
D1 KV Write
 ↓
D2 Decode Read
 ↓
D3 Prefill/Mixed Read
 ↓
E Real Identity
 ├────→ F Piecewise Smoke
 ↓
R2 scheduler+worker E[r,g] / counts[g]
 ↓
R3 Reclaim Transaction
 ↓
R4 Policy
 ├────→ R5 Preemption
 ├────→ R6 TP2
 ├────→ R7 AOT
 ├────→ R8 Triton (profile gate)
 └────→ R9 Piecewise hardening
               ↓
              R10
```

---

# 5. Immediate Slice

当前下一刀：`R1-A1`。

### Files

```text
vllm/config/cache.py
vllm/v1/kv_cache_interface.py
vllm/v1/kv_cache_spec_registry.py
vllm/model_executor/layers/attention/attention.py
focused tests
```

### Acceptance

```text
RaggedAttentionSpec constructs
page bytes correct
Hkv/Hp validation
unsupported combinations reject
dense config unchanged
```

### Next

A1 PASS 后：

```text
A2 Global Pool
→ A3 Layout/Stride
```

不要直接跳 BlockTable/FA。
