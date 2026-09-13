# P1 V1 Core Accepted Source Manifest

## 1. Accepted Identity

当前 accepted implementation 已固化为 immutable checkpoint：

```text
vLLM v0.26.0 upstream commit 568afb3a13806beb53bb2e6bd518269357b237c0
`cd444b72ceecb2496bf015baf8c18bf883151fa1`

Accepted tag：`p1-v1-core-accepted`

历史上该 commit 由 upstream + 已审计 dirty diff 形成。
```

Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
Checkpoint worktree：`CLEAN`

## 2. Production Manifest

| 文件 | 修改 symbol | Slice | 语义目的 |
|---|---|---|---|
| `vllm/v1/request.py` | `Request.effective_kv_len` | T3 | Scheduler completed physical frontier；normal request保持 `None`。 |
| `vllm/v1/core/sched/output.py` | `ReclaimTransitionData`, `CachedRequestData.reclaim_transitions` | T2 | prepared reclaim transition 与 request arrays 对齐传输。 |
| `vllm/v1/core/sched/scheduler.py` | `_PreparedReclaimPlan`, `_materialize_reclaim_transition`, `schedule`, `update_from_output`, `_preempt_request` | T2/T3 | pre-allocation materialization、successful completion commit、E persistence、preemption reset。 |
| `vllm/v1/core/kv_cache_manager.py` | `allocate_slots`, `reconcile_reclaimed_blocks` | T3 | physical E allocation override 与 public dense reconcile seam。 |
| `vllm/v1/core/kv_cache_coordinator.py` | `reconcile_reclaimed_blocks` | T3 | ownership hierarchy forwarding 与 single-group guard。 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `reconcile_reclaimed_blocks` | T3 | validate → dense ownership publication → `BlockPool.free_blocks`。 |
| `vllm/v1/worker/gpu/states.py` | `RequestState.effective_kv_len` | S1 | Worker persistent physical state。 |
| `vllm/v1/worker/gpu/input_batch.py` | `cache_positions`, `effective_kv_seq_lens`, builders/kernels | S2/S3 | logical positions 与 physical KV write/visibility 分离。 |
| `vllm/v1/worker/gpu/model_runner.py` | `update_requests`, `_commit_reclaim_transition`, `execute_model` | S1/S2/T2 | Worker dense row、physical publication、forward 前 staged state flush。 |
| `vllm/v1/worker/gpu/attn_utils.py` | effective-length wiring | S3 | attention metadata physical visibility。 |
| `vllm/v1/attention/backend.py` | effective metadata field | S3 | common attention metadata transport。 |
| `vllm/v1/attention/backends/flash_attn.py` | FA2 effective-length selection | S3 | baseline FA2 physical read extent。 |
| `vllm/v1/worker/gpu/model_states/default.py` | effective state wiring | S1/S3 | model state lifecycle compatibility。 |

## 3. Tests / Harness Manifest

| 文件 | 类型 | Slice | 用途 |
|---|---|---|---|
| `tests/v1/core/test_reclaim_transport.py` | Core test | T2 | Scheduler prepared plan / aligned transport / failure lifecycle。 |
| `tests/v1/worker/test_gpu_reclaim_commit.py` | Worker test | R1/T2/S1 | one-write dense row、descriptor、effective state、normal branch。 |
| `tests/v1/worker/test_gpu_effective_kv_state.py` | Worker test | S1/S2/S3 | logical/physical state、positions、attention visibility。 |
| `tests/v1/core/test_reclaim_ownership.py` | Core test | T3 | ownership reconcile、safe free、refcount、reuse、preemption。 |
| `04-experiments/project1_kv_reclaim/scripts/p1_m1_t3_integration_closure.py` | Smoke harness | T3 closure | real `LLMEngine`/Qwen3 forward、A/B reuse、trace。 |

## 4. Source Ownership Boundary

上述 production files 构成 accepted V1 diff；未列出的 source 不属于本次 freeze 的新增语义。`block_table.py`、`buffer_utils.py` 未被扩容或重设计，仅作为既有 staged-write primitive 依赖。
