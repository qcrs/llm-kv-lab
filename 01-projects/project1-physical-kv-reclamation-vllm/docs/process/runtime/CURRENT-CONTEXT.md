# P1 Current Context

> 这是由 `PROJECT_STATE.md` 派生的恢复入口，不是技术事实的最高来源。

## 当前状态

```text
P1 V1/Core = PASS / ACCEPTED / FROZEN
Closed Slice = P1-V2-R1-C-EXECUTION-FOUNDATION-01
Closed Slice Status = PASS / CLOSED
Current Slice = P1-V2-R1-D-RAGGED-GPU-EXECUTION-01
Slice Status = PASS_PENDING_WEB_REVIEW
Canonical V1 restore point = `cd444b72ceecb2496bf015baf8c18bf883151fa1`
Current branch = `p1/v2-token-compaction-v026`
C final implementation commit = `db3cc1179f53e2dc8df2096ed17b5ec86ef033d3`
D start HEAD = `dc4287411704abc512f6cca3ca669c7d9ec254c5`
Next Allowed Action = WEB_REVIEW_CURRENT_SLICE
```

## 已关闭的 C Slice

Web architecture review 已将 C 判定为 `PASS / CLOSED`。C 只关闭 C0–C4
execution-addressing foundation，不启用 production Ragged runtime。已
完成 geometry hardening、CPU scalar physical address oracle、Hp-wide zero-copy layout、
shared backing materialization 和 member metadata transforms。focused C0–C4/Dense 为
`32 passed`，Ragged state regression 为 `37 passed`，Dense `attn_utils` regression 为
`5 passed`。

实现 trace：`04-experiments/project1_kv_reclaim/notes/P1-V2-R1-C-EXECUTION-FOUNDATION-01-Code-Trace.md`。
raw evidence：`04-experiments/project1_kv_reclaim/raw/p1-v2-r1-c-execution-foundation-01-final/`。

## 已冻结的 V1

V1 已闭合 whole-block physical KV reclamation：`L`（logical progress）与 `E`
（physical KV frontier）分离；`positions` 与 `cache_positions` 分离；FA2 physical
visibility、Worker dense `BlockTables`、Scheduler canonical ownership reconcile、
`BlockPool.free_blocks()` 的 refcount-safe release、真实 block reuse、基于完成态 `E`
的后续 allocation，以及 fail-closed / preemption reset 均有 source、core test 和
真实 Engine evidence 支撑。

## Canonical runtime chain

```text
EngineCore.step
→ Scheduler.schedule / KVCacheManager.allocate_slots
→ SchedulerOutput / reclaim transition
→ GPUModelRunner RequestState + BlockTables
→ positions / cache_positions / attention metadata
→ real model forward
→ ModelRunnerOutput / Scheduler.update_from_output
→ dense canonical ownership reconcile
→ BlockPool free/reuse
→ future allocation based on completed E
```

主要入口：`vllm/v1/request.py`、`vllm/v1/core/sched/output.py`、
`vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/kv_cache_manager.py`、
`vllm/v1/core/single_type_kv_cache_manager.py`、`vllm/v1/worker/gpu/model_runner.py`。

## Evidence 入口

- Gate review：`04-experiments/project1_kv_reclaim/notes/P1-V1-CORE-GATE-REVIEW-01.md`
- T3 implementation：`04-experiments/project1_kv_reclaim/notes/P1-M1-T3-IMPL-01A-Physical-Ownership-Reconciliation-实施记录.md`
- T3 Engine closure：`04-experiments/project1_kv_reclaim/notes/P1-M1-T3-INTEGRATION-CLOSURE-真实Engine生命周期验证报告.md`
- Freeze report：`04-experiments/project1_kv_reclaim/notes/P1-V1-CORE-FREEZE-01.md`
- Freeze raw：`04-experiments/project1_kv_reclaim/raw/p1-v1-core-freeze-01/p1-v1-core-freeze-01.log`

## 范围与限制

验证基线为 A100 80GB、MRV2、BF16、FA2、eager、TP/PP/DP/DCP/PCP=1、single KV
group、`block_size=16`，并关闭 prefix caching、speculative decoding、async、CUDA
Graph、KV connector/offload。其余配置为 `NOT VALIDATED`；retention policy、token-level
compaction、Triton/CUDA compaction、性能结论、HBM/OS return 与未授权的下一 Slice 为
`OUT OF SCOPE`。

## 当前 D Slice

当前执行已批准的完整 Slice `P1-V2-R1-D-RAGGED-GPU-EXECUTION-01`，覆盖
`D-WRITE`、`D-DECODE`、`D-PREFILL-MIXED`，但不做 production activation。
D 已完成 focused real-CUDA correctness closure，当前状态为 `PASS_PENDING_WEB_REVIEW`；
raw evidence 位于 `04-experiments/project1_kv_reclaim/raw/p1-v2-r1-d-ragged-gpu-execution-01/`。

## 冻结规则

```text
DO NOT MODIFY V1 CORE WITHOUT REOPEN
```

任何 S1/S2/S3/T2/T3 修改必须先创建 `P1-V1-CORE-REOPEN-XX`，说明 reason、affected
invariant、affected evidence 和 regression plan。D 完成后必须停在
`WEB_REVIEW_CURRENT_SLICE`；不得自动进入 E。
