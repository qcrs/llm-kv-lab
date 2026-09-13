# P1-V1-CORE-GATE-REVIEW-01

## 1. Executive Summary

**Recommended Gate Status：PASS（建议，等待 ChatGPT Web / Work 最终裁定）**

当前 pinned baseline 下，S1/S2/S3/T2/T3 已形成可解释、可执行、可验证的 whole-block physical KV reclamation core。源码链路、Worker dense execution view、Scheduler completed physical frontier、canonical ownership reconcile、`BlockPool` refcount-safe release、后续 physical allocation 和真实 Engine reuse smoke 相互闭合。

该建议严格限定于已验证配置：MRV2、A100、BF16、FA2、eager、TP/PP/DP/DCP/PCP=1、single KV group、`block_size=16`、prefix/spec/async/CUDA Graph/connector 全关闭。不应外推到未验证配置。

## 2. Source Identity

- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/physical-kv-reclaim-v026`
- HEAD：`568afb3a13806beb53bb2e6bd518269357b237c0`
- vLLM：v0.26.0
- Python：3.10.20
- Torch：2.11.0+cu129
- GPU：NVIDIA A100 80GB；本轮 source/evidence review 使用 GPU 0 targeted tests，既有真实 Engine closure 使用 GPU 2。
- Worktree：DIRTY；既有 S1/S2/S3/R1/T2/T3 修改和 tests 全部保留。

Raw review evidence：`/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/p1-v1-core-gate-review-01/p1-v1-core-gate-review-01.log`

## 3. End-to-End Runtime Chain

```text
EngineCore.step_with_batch_queue()
  → Scheduler.schedule()
  → KVCacheManager.allocate_slots()
  → SchedulerOutput
  → GPUModelRunner.execute_model()
  → add_requests()/update_requests()
  → RequestState / BlockTables publication
  → positions + cache_positions
  → slot_mapping + attention metadata
  → real model forward
  → ModelRunnerOutput
  → Scheduler.update_from_output()
  → dense canonical ownership reconcile
  → BlockPool.free_blocks()
  → next allocation uses Scheduler E
```

Slice placement：

- S1：`RequestState.effective_kv_len`、`states.py` staging/publication、model runner post-forward effective frontier。
- S2：`InputBatch.cache_positions` 与 slot-mapping 路径；logical `positions` 保持模型/RoPE 坐标。
- S3：`effective_kv_seq_lens`、`CommonAttentionMetadata` 和 FA2 backend length selection。
- T2：`ReclaimTransitionData`、Scheduler prepared-plan materialization、Worker `_commit_reclaim_transition()`，reclaim/normal append 互斥。
- T3：`Request.effective_kv_len`、`allocate_slots()` physical override、三层 `reconcile_reclaimed_blocks()`、`update_from_output()` completion commit、`BlockPool.free_blocks()`。

## 4. S1/S2/S3/T2/T3 Cross-Module Map

| Slice | 问题 | 源码位置 | 消费者 | Invariant | Evidence |
|---|---|---|---|---|---|
| S1 | Worker 需要独立 physical frontier | `vllm/v1/worker/gpu/states.py`, `model_runner.py` | slot/input/forward lifecycle | normal 时 E 随 q 推进；reclaim 后 E 可与 L 分离 | `test_gpu_effective_kv_state.py`、真实 closure |
| S2 | logical position 不能直接作为 physical KV 写位置 | `worker/gpu/input_batch.py`, `model_runner.py` | slot mapping/cache write | `positions` logical，`cache_positions` physical | S1/S2/S3 worker tests |
| S3 | attention visibility 不能读取 logical 全范围 | `attention/backend.py`, `backends/flash_attn.py` | FA2 metadata/backend | effective physical extent 独立选择 | effective-state tests、真实 FA2 closure |
| T2 | retained 与 current new 的同窗双 descriptor 冲突 | `core/sched/output.py`, `core/sched/scheduler.py`, `worker/gpu/model_runner.py` | BlockTables | retained + new → one final row overwrite | R1/T2 tests、transport tests |
| T3 | Worker dense row 与 Scheduler ownership/allocator 脱节 | `request.py`, `core/kv_cache_manager.py`, coordinator/manager、scheduler | allocator/free queue/next step | reconcile → free → E commit；后续按 E 分配 | `test_reclaim_ownership.py`、GPU2 real Engine closure |

## 5. Gate Findings

### Gate 1 — Logical / Physical State

`Request.num_computed_tokens` 仍是 Scheduler logical progress；`Request.effective_kv_len` 只有 reclaim 成功后才变为 completed physical frontier。Worker 侧 `RequestState.num_computed_tokens` 与 `effective_kv_len` 也由不同 staged state 维护。normal request 维持 upstream logical path；reclaim 后允许 `L != E`。当前审计未发现把 L 重新当作 physical KV length 的 P1 baseline consumer。

### Gate 2 — Position

`positions` 继续服务 model/RoPE semantics；`cache_positions` 进入 slot mapping 与 physical block/offset。对 `L=94, E_target=62, q=3`，logical positions 是 `94,95,96`，physical cache positions 是 `62,63,64`。worker input builder 和 model runner 将两者分别传给对应消费者。

### Gate 3 — Attention Visibility

`seq_lens` 保留 logical sequence semantics，`effective_kv_seq_lens` 被 FA2 backend 在有效 scope 内选择，用于 physical KV visibility。对 `L=97, E=65`，attention 不应把 logical 97 当作存在 97 个连续 physical KV positions；当前 effective metadata 将可读 physical extent 限制到 E。DCP/cascade 等 scope guard 保留 fallback，未外推支持。

### Gate 4 — Worker Dense BlockTables

Scheduler 从 prepared plan 的 retained indices 基于 `KVCacheManager.get_block_ids()` materialize retained IDs；`CachedRequestData.new_block_ids` 仍表示 current-step allocation delta。Worker reclaim branch调用 R1 helper，以一次 `append_block_ids(..., overwrite=True)` 生成 dense row；normal branch仍使用 `overwrite=False`。R1 测试证明 one descriptor；T2 transport 与真实 closure证明 dense row publication。当前 baseline 未观察到 NULL hole 或 duplicate final row。

### Gate 5 — Canonical Ownership

`SingleTypeKVCacheManager.req_to_blocks` 是 Scheduler canonical ownership；Worker `BlockTables` 是 execution addressing view，二者不混同。T3 primitive 从 current row 拆出 expected old 与 same-step tail，验证 retained subset/order/uniqueness，然后写 dense `req_to_blocks`，最后调用 `BlockPool.free_blocks(reversed(removed_blocks))`。真实 closure 观察到 `[1,2,3,4,5,6] → [1,2,5,6]`。

### Gate 6 — Fail-Closed

验证覆盖 `expected_old_num_blocks`、same-step tail、retained membership/order/duplicate、old/final uniqueness、removed live/non-null/refcount。测试 `test_reclaim_validation_failure_is_fail_closed` 观察到 mismatch 时 ownership、free count、E 均不变。当前实现的 physical E assignment 位于 reconcile 返回之后，因此 validation failure 不会提交 E。

### Gate 7 — Safe-Free / Lifetime

`Scheduler.update_from_output()` 位于真实 `future.result()` 之后，并已有 `processed_step_seq` / `deferred_frees` 语义：GPU writes 完成后才 drain deferred frees。T3 removed blocks不是提前进入 free queue；它们只在对应 step output 被处理且 reconcile成功后释放。真实 closure 的 free count 增长与后续 reuse证明没有 premature reuse。

### Gate 8 — Physical Allocation Accounting

`allocate_slots()` 保留 `num_local_computed_tokens`/`total_computed_tokens` 的 logical/prefix bookkeeping，仅在计算 `num_tokens_main_model` 的 allocation seam 使用：

```python
allocation_base = (
    request.effective_kv_len
    if request.effective_kv_len is not None
    else total_computed_tokens
)
```

不直接修改 `total_computed_tokens`，因为它还服务 prefix/external/cache semantics；不修改底层 allocator，因为 manager 仍应接收“目标所需 blocks”并维护 ownership。对 `L=97,E=65,q=1,block_size=16,owned=5`，physical target 为 66，仍需 5 blocks，因此新 allocation 为 0；测试与真实 closure均通过。

### Gate 9 — Result-Time E

- Case A：本轮 reclaim，`E = E_target + q`。
- Case B：已有 physical state，本轮 normal，`E += q`。
- Case C：从未 reclaim，`E is None`，upstream path。

`E_target` 是 forward-entry value；`E_completed` 才是 Scheduler persistence。真实 closure 观察 `E_target=62`、Worker/Scheduler completed E=`63`。

### Gate 10 — Preemption / Lifecycle

`_preempt_request()` 是 recompute preemption：释放当前 canonical ownership，`num_computed_tokens=0`，`effective_kv_len=None`，request requeued。E 必须 reset，否则 resumed request会以旧 physical row解释新分配。T3 test证明 commit后 preemption只释放 dense final row，removed IDs不会 double-free。

## 6. Invariant Matrix

| ID | Invariant | Status | Basis |
|---|---|---|---|
| I1 | Logical state remains model truth | PASS | Request/Worker separated state；S1/S2/S3 tests |
| I2 | Physical E can diverge from L | PASS | Worker effective-state tests；closure E=63 vs L=95 |
| I3 | positions remain logical | PASS | input/model runner source audit；S2 tests |
| I4 | cache_positions remain physical | PASS | input builder/slot mapping source；S2 tests |
| I5 | attention visibility uses physical extent | PASS（baseline scope） | FA2 effective length tests；closure |
| I6 | Worker BlockTables can become dense | PASS | R1 one-row tests；closure |
| I7 | BlockTables != canonical ownership | PASS | T2 pre/post ownership observations |
| I8 | req_to_blocks is canonical ownership | PASS | manager source + T3 reconcile |
| I9 | no reusable block remains canonical-owned | PASS | ownership-before-free ordering + closure |
| I10 | ownership mutation precedes BlockPool reuse | PASS | manager primitive ordering；reuse oracle |
| I11 | validation failure is fail-closed | PASS | negative T3 test |
| I12 | Scheduler E commits only after successful reconcile | PASS | `update_from_output()` order/source |
| I13 | future allocation uses E after reclaim | PASS | allocation override + no-bogus-allocation test |
| I14 | normal request retains upstream logical path | PASS | normal allocation test、Worker normal append test |
| I15 | preemption invalidates E | PASS | `_preempt_request()` + T3 preemption test |
| I16 | reclaimed blocks are not double-freed | PASS | post-commit preemption test |
| I17 | reclaimed physical block can be reallocated | PASS | T3 core reuse + real request B reuse |
| I18 | real Engine closes Worker→Scheduler transaction | PASS（baseline scope） | GPU2 Qwen3 closure smoke |

## 7. Test / Evidence Matrix

| Contract | Source | Unit/Core Test | Integration Evidence | Status |
|---|---|---|---|---|
| T2 transport alignment | `scheduler.py`, `output.py` | `test_reclaim_transport.py` | closure schedule trace | PASS |
| Worker one-write composition | `model_runner.py`, `block_table.py` | `test_gpu_reclaim_commit.py` | closure dense row | PASS |
| logical/physical state | `states.py`, `input_batch.py` | `test_gpu_effective_kv_state.py` | closure E trace | PASS |
| dense ownership reconcile | manager hierarchy | `test_reclaim_ownership.py` | closure canonical row | PASS |
| safe free/refcount | `block_pool.py` | T3 exact release/preemption tests | closure refcount 0 | PASS |
| real reuse | `block_pool.py`, `allocate_slots()` | T3 request B test | closure request B gets block 4 | PASS |
| normal fallback | allocation and Worker branches | normal-path tests | Engine no-plan steps | PASS |
| static validity | all modified source | `py_compile`, `git diff --check` | raw logs | PASS |

Core test与真实 Engine的边界：`create_scheduler()` / `create_requests()` 是 test helper；core tests使用真实 Scheduler 与 manager，但通常注入 fake `ModelRunnerOutput`，不能单独证明 kernel/model forward。GPU2 closure 才证明真实 EngineCore、Qwen3 model forward、Worker、Scheduler result-time commit和generation continuation。

## 8. Failure Semantics

- allocation failure：prepared plan 在 allocation成功前不进入 `CachedRequestData`；self-preemption会丢弃 plan。
- execution failure：没有对应 successful `update_from_output()` 时不执行 T3 reconcile/free；fail-safe 偏向 retention 而不是 aliasing。
- validation failure：reconcile 在所有 validation后才 mutation；E assignment位于其后；ownership/free/E保持旧值。
- preemption：释放当前 dense canonical row，reset logical/physical state，重新排队 recompute；无 double-free removed IDs。
- finish：在 successful step commit 后，finish/free 只处理当前 canonical row。
- normal fallback：`effective_kv_len is None` 时 allocation 和 Worker append保持 upstream行为。

## 9. Scope / Limitation

### 已验证

MRV2、A100、BF16、FA2、eager、TP/PP/DP/DCP/PCP=1、single KV group、`block_size=16`、prefix/spec/async/CUDA Graph/connector关闭；GPU2真实 Qwen3-0.6B generation、T2/R1/S1/T3 targeted tests。

### NOT VALIDATED

prefix caching、spec decode、async scheduling、CUDA Graph、PP>1、DCP/PCP>1、多 KV group、KV connector/offload、repeated reclaim在更广生命周期下的组合、错误执行后的重新调度恢复。

### 明确 out of scope

retention policy、token-level compaction、Triton compaction kernel、性能 benchmark、HBM/OS memory return、V2 implementation。

## 10. Gaps / Risks

- **MAJOR（scope risk）**：`Request.effective_kv_len` 和 reconcile API 只在 single-group baseline中验证；multi-group guard会拒绝，而不是宣称支持。
- **MINOR（test strength）**：真实 Engine closure 的 reclaim step使用 q=1，same-step new-block composition 的跨 block boundary主要由 core/Worker oracle覆盖；closure本身未再现带 `B6` 的真实 dense row。
- **MINOR（environment）**：日志显示 vLLM `_version` 未生成 warning，以及 NCCL process group exit warning；均未影响本轮结果，但不是 release packaging结论。
- **OBSERVATION**：R1 reclaim-event-only路径仍有一次 GPU→CPU `.item()`；本 Gate未把它升级为 correctness blocker。

没有发现 BLOCKER，也没有发现当前 approved design 与 pinned baseline 的 `DESIGN_CONFLICT`。

## 11. DESIGN_CONFLICT

`NONE`

## 12. Recommended Gate Status

建议：`PASS`。

理由：I1–I18 在声明的 baseline scope 内均为 PASS；core tests、GPU2真实 Engine closure、ownership/refcount/reuse evidence和静态检查相互一致。正式将 P1 V1/Core Gate 标记为 accepted/closed 仍需 ChatGPT Web / Work 的 architecture adjudication。

## 13. Evidence Index

- Review raw evidence：`/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/p1-v1-core-gate-review-01/p1-v1-core-gate-review-01.log`
- T3 implementation：`/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t3-impl-01a/m1-t3-impl-01a.log`
- Real Engine closure：`/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t3-integration-closure/m1-t3-integration-closure.log`
- Closure report：`/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/notes/P1-M1-T3-INTEGRATION-CLOSURE-真实Engine生命周期验证报告.md`
- T3 tests：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim/tests/v1/core/test_reclaim_ownership.py`
- T2 transport tests：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim/tests/v1/core/test_reclaim_transport.py`
- Worker tests：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim/tests/v1/worker/test_gpu_reclaim_commit.py`
- State/attention tests：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim/tests/v1/worker/test_gpu_effective_kv_state.py`

## 14. Review Conclusion

当前 whole-block physical reclaim core 已完成从 execution、ownership、lifetime 到 allocation accounting 的闭环审计。建议保持当前 scope，不在本 Gate下扩展未验证配置或进入 V2；后续工作必须由新的 approved Slice单独授权。
