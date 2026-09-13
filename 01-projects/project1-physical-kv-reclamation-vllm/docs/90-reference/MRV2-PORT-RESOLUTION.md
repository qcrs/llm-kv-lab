# MRV2 Port Resolution (2026-08-25)

## Decision

`MRV2_FEASIBLE` for the pinned vLLM `0.26.0` source at commit
`568afb3a13806beb53bb2e6bd518269357b237c0`.

This is a source-map port, not a new P1 design. P1 V1/V2 feature names and
the approved logical/physical ownership contract are unchanged. No runtime
source was modified by this audit.

## Runtime Identity

```yaml
vllm_engine: V1 Engine (vllm/v1/**)
model_runner: V2 (MRV2)
model_runner_env: VLLM_USE_V2_MODEL_RUNNER=1
model_runner_source: vllm/v1/worker/gpu/model_runner.py
legacy_runner_v1: vllm/v1/worker/gpu_model_runner.py
legacy_status: historical_reference_only; not the P1 target
p1_feature_v1: whole-block zero-copy reclamation
p1_feature_v2: token-level Triton KV compaction
```

## F1-F7 Feasibility Evidence

| Criterion | Result | Exact source evidence | P1 implication |
|---|---|---|---|
| F1 persistent request state | FEASIBLE | `vllm/v1/worker/gpu/states.py::RequestState`, persistent arrays and `add_request()`/`remove_request()` | A future physical-occupancy field has a request-lifetime owner; field addition is not approved here. |
| F2 physical view replacement | FEASIBLE | `gpu/block_table.py::BlockTables.append_block_ids(..., overwrite=True)` resets the active row length and stages replacement; `apply_staged_writes()` publishes it | Whole-row replacement has a narrow MRV2 primitive. |
| F3 active-range correctness | FEASIBLE WITH TO_VERIFY | `gather_block_tables()` copies only `num_blocks`; `compute_slot_mappings()` indexes the persistent row from `positions` without an in-kernel active-range check | Stale tail is hidden from gathered attention tables, but future physical-position bounds/defensive clearing must be proven in the implementation Slice. |
| F4 logical/physical separation | FEASIBLE WITH CANDIDATE_SEAM | `input_batch.py::prepare_pos_seq_lens()` derives positions and seq lens from `num_computed_tokens`; `BlockTables.compute_slot_mappings()` consumes those positions | A narrow adapter seam exists, but the current coupling remains TO_VERIFY; no counter is changed by this port. |
| F5 scheduler ownership | FEASIBLE | `SingleTypeKVCacheManager.req_to_blocks`, `allocate_slots()`, `pop_blocks_for_free()`, `BlockPool.free_blocks()` | Scheduler remains canonical owner and publisher of reusable blocks. |
| F6 worker ACK boundary | FEASIBLE | `vllm/v1/outputs.py::ModelRunnerOutput` is serialized to scheduler; `EngineCore` forwards it to `Scheduler.update_from_output()` | Existing transport can carry future small commit metadata; no field is added now. |
| F7 fixed supported config | FEASIBLE SUBJECT TO EXISTING GATES | MRV2 path is selected before execution and has no source blocker for the fixed TP=PP=DP=1, eager, dense, one-group configuration | Exact backend, block-size, DCP/PCP/cascade and Qwen3 positional gates remain TO_VERIFY. |

## MRV1 -> MRV2 Source Map

| Historical seam | Current MRV2 seam | Status |
|---|---|---|
| `vllm/v1/worker/gpu_model_runner.py` | `vllm/v1/worker/gpu/model_runner.py` | semantic port |
| `CachedRequestState` | `gpu/states.py::RequestState` plus transient `InputBatch` | partial equivalent |
| `GPUInputBatch` persistent state | `RequestState` + per-step `InputBatch` | semantic port |
| `vllm/v1/worker/block_table.py::BlockTable` | `gpu/block_table.py::BlockTables` | semantic port |
| `BlockTable.add_row()` | `append_block_ids(..., overwrite=True)` | candidate equivalent |
| `_update_states()` | `add_requests()` + `update_requests()` | semantic port |
| `_prepare_inputs()` | `prepare_inputs()` | semantic port |
| old slot mapping path | `compute_slot_mappings()` | semantic port; bounds TO_VERIFY |
| resume full-row replacement | MRV2 overwrite/full request re-add | candidate equivalent |
| scheduler `req_to_blocks` | unchanged scheduler-side ownership | equivalent |
| `ModelRunnerOutput` | current serialized worker->scheduler boundary | equivalent transport seam |

## Historical Snapshot Rule

`docs/90-reference/baselines/P1-v3-Implementation-Audited-BASELINE.md` is not
rewritten. It records an earlier audit against MRV1 seams. This resolution is
the 2026 MRV2 revalidation and must not be projected backward onto that
historical snapshot.

## Remaining TO_VERIFY

- prove physical positions cannot address a stale tail after a shortened row;
- freeze the narrow logical-position versus physical-slot adapter for Qwen3;
- verify exact attention backend, block-size equality, DCP/PCP/cascade flags and
  driver/model environment;
- validate result ordering and canonical-ownership checks before any ACK field
  is approved.

## Scope Boundary

No `effective_kv_len` implementation, reclaim patch, scheduler free patch,
Triton kernel, benchmark, NSYS, NCU, or P2 runtime work was performed.
