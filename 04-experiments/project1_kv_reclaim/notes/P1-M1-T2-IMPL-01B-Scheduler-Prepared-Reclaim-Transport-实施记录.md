# P1-M1-T2-IMPL-01B 实施记录

## 1. Slice 身份与目标

- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/physical-kv-reclaim-v026`
- Pinned HEAD：`568afb3a13806beb53bb2e6bd518269357b237c0`
- 本轮目标：将已通过 R1 的 Worker final-row reclaim primitive 接入 continuing RUNNING request 的 Scheduler → `CachedRequestData` → `update_requests()` 路径。
- 范围：single KV cache group、whole-block、prepared decision；不实现 policy、T3、free/reuse、ownership reconciliation。

## 2. 问题与设计边界

R1 只证明 Worker 可以把 retained old blocks 与同一步新 block 合为一个 row。若不接入 runtime，Scheduler 的 canonical row、Worker 的 execution row 与 physical `effective_kv_len` 仍不会在同一 forward 中对齐。本 Slice 只增加 prepared-plan ingress 和 transport；plan 不是 retention policy，removed blocks 继续留在 Scheduler canonical ownership 中，处于 pending-free/quarantine 语义。

## 3. Source Facts

| 标签 | Source anchor | 观察 | 意义 |
|---|---|---|---|
| SOURCE_FACT | `Scheduler.schedule()` | RUNNING request 先计算 `num_new_tokens`，调用 `allocate_slots()`，成功后写入 `req_to_new_blocks`，最后构造 `CachedRequestData`。 | materialization 可在 allocation 前读取旧 row，但 transition 只有 allocation 成功后才进入 output。 |
| SOURCE_FACT | `KVCacheManager.get_block_ids()` | 返回 request 的 canonical `tuple[list[int], ...]` block IDs。 | 使用 public API materialize retained IDs，不穿透 manager internals。 |
| SOURCE_FACT | `CachedRequestData.new_block_ids` | 表示当前 step incremental allocation delta，不是完整 row。 | R1 直接复用该 delta，避免重复追加。 |
| SOURCE_FACT | `GPUModelRunner.update_requests()` | cached request 更新 logical CPU mirror，并对无特殊状态的 new blocks 使用 `overwrite=False`。 | reclaim 与 normal append 必须互斥。 |
| SOURCE_FACT | `execute_model()` | `update_requests()` 后调用 block-table publication，再进入 prepare/forward。 | 新增 `effective_kv_len.apply_write()` 使 physical E 在 forward 前可见。 |
| SOURCE_FACT | `new_block_ids_to_zero` / `kv_cache_block_copies` | zeroing 与 CoW 是独立 side channels。 | reclaim branch 不提前 return，不改变其 lifecycle。 |

## 4. Reference Facts

- `Tangram` pinned commit `8e1cbfa5cf82acc67fe1880df0c632f2a6485e35` 的 worker/source 展示了 effective sequence length、压缩后物理 view 与 deferred/free lifecycle 的分离。适用点是 logical/effective channel 分离；其 ragged paging、cluster map、per-head-group machinery 不适用于本 V1 uniform slice。
- `Sparse-vLLM` pinned commit `c14643387104c9db923a4fe690b699fe7fbd1790` 提供 invalid-input atomicity 与 deferred device lifecycle 的测试方法；本轮只借鉴验证思想，不搬运实现。
- 本地没有将 CacheGen/LMCache source 引入实现；其 serialization/offload/network 机制不属于 01B。

以上是 `REFERENCE_FACT`，不是当前 vLLM API。当前实现以 pinned v0.26 source 为准。

## 5. Design Choices

1. `CachedRequestData.reclaim_transitions` 使用 parallel list，与 `req_ids`、`new_block_ids`、`num_computed_tokens` 严格按 index 对齐；normal entry 为 `None`。
2. `_PreparedReclaimPlan` 只保存 retained block indices 与 target E。Scheduler 在 allocation 前通过 `get_block_ids()` materialize physical IDs，避免 policy 长期持有 allocator IDs。
3. allocation 成功后才把 `ReclaimTransitionData` 写入本 step map；allocation failure/self-preemption 清理 prepared plan，不进入 transport。
4. Worker reclaim branch 与 normal `overwrite=False` branch 互斥；reclaim branch 调用 R1 一次 final-row overwrite。
5. `effective_kv_len.apply_write()` 单独 flush R1 staged physical state，不调用完整 `RequestState.apply_staged_writes()`，避免无关 UVA fields 产生额外 publication。

## 6. Before / After Runtime Chain

Before：

```text
Scheduler → allocate_slots → new_block_ids → CachedRequestData
→ Worker incremental append
```

After：

```text
prepared plan
→ pre-allocation canonical snapshot
→ allocate_slots
→ allocation-success transition + current new_block_ids
→ CachedRequestData.reclaim_transitions
→ Worker R1 final-row composition
→ effective_kv_len.apply_write()
→ BlockTables.apply_staged_writes()
→ prepare_inputs / prepare_attn / forward
```

## 7. Canonical Example

```text
L=94, E=94, S=16
Scheduler old row: [B0 B1 B2 B3 B4 B5]
prepared keep indices: [0,1,4,5]
retained: [B0 B1 B4 B5]
new_E: 62
same-step allocation: B6
Scheduler canonical row: [B0 B1 B2 B3 B4 B5 B6]
Worker final row: [B0 B1 B4 B5 B6]
B2/B3: pending-free / quarantine
```

`B6` 是当前 step allocation capacity，不能用于把 E 设置成 `5*16`；成功 forward 后才由 S1 normal post-update 按 computed delta 增长 E。

## 8. 文件级修改说明

### `vllm/v1/core/sched/output.py`

新增 `ReclaimTransitionData`，并给 `CachedRequestData` 新增 aligned `reclaim_transitions` field；`make_empty()` 返回空 list，`anon_repr()` 只输出 retained count summary。删除：无。

### `vllm/v1/core/sched/scheduler.py`

新增 private `_PreparedReclaimPlan`、`_set_prepared_reclaim_plan()` 与 `_materialize_reclaim_transition()`。`schedule()` 在 allocation 前读取 canonical row、在成功后注册 transition，并在 self-preemption 路径清理 stale plan。`_make_cached_request_data()` 增加 aligned transport。删除：无。未修改 ownership/free。

### `vllm/v1/worker/gpu/model_runner.py`

`update_requests()` 增加 reclaim/normal mutually-exclusive routing，继续执行 zeroing 与 CoW；`execute_model()` 在 update 后、block table publication 前 flush `effective_kv_len`。R1 helper 本身未重写。删除：无。

### `tests/v1/core/test_reclaim_transport.py`

新增本地 `/data/models/Qwen3-0.6B` Scheduler transport oracle：canonical materialization、new delta 不进入 retained、canonical row 不缩减、empty/normal `None` alignment，以及未成功调度的 prepared plan 不作为 transport state。删除：无。

### `tests/v1/worker/test_gpu_reclaim_commit.py`

新增 `update_requests()` reclaim publication 与 normal incremental append tests，验证 one descriptor、GPU-visible E=62、L=94 和 normal frontier 不改变。删除：无。

## 9. Added / Modified / Replaced / Deleted

### ADDED

- `ReclaimTransitionData`
- `CachedRequestData.reclaim_transitions`
- `_PreparedReclaimPlan` 与两个 private Scheduler seam
- Scheduler/Worker targeted tests
- 本 implementation record 与主 raw evidence

### MODIFIED

- Scheduler schedule/materialization and `_make_cached_request_data()`。
- Worker `update_requests()` and `execute_model()` publication boundary。

### REPLACED

仅对 reclaim request：原 incremental append 分支替换为 R1 final-row overwrite。Normal request 仍保留 upstream `append_block_ids(..., overwrite=False)`。

### DELETED

DELETED: NONE。

## 10. Tests 与结果

命令：

```text
CUDA_VISIBLE_DEVICES=0 /home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310/bin/python -m pytest -q tests/v1/core/test_reclaim_transport.py tests/v1/worker/test_gpu_reclaim_commit.py tests/v1/worker/test_gpu_effective_kv_state.py
```

结果：`30 passed, 15 warnings`。使用本地 `/data/models/Qwen3-0.6B`，未联网；GPU 0 上其他进程未终止。

静态检查：相关 `py_compile` PASS；`git diff --check` PASS。

## 11. T2 / T3 边界与限制

本轮没有 `SchedulerOutput` reclaim field、ownership shrink、`BlockPool.free`、reuse、ACK、allocator physical accounting 或 retention policy。Scheduler canonical row 仍包含 removed old blocks；Worker 只更新 execution view。candidate blocks 必须等 T3 worker commit/ACK 与 scheduler ownership reconciliation 后才可能 free。当前仅支持 continuing RUNNING、single KV group、eager baseline；不覆盖 resumed reclaim、preemption restore、prefix caching、DCP/PCP、cascade、spec decode、async/CUDA Graph。R1 的 reclaim-event-only GPU scalar `.item()` 仍存在。

## 12. Evidence

- `/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01b-runtime-integration/m1-t2-01b-runtime-integration.log`

## 13. Next Allowed Action

Review P1-M1-T2-IMPL-01B implementation, tests, evidence, and implementation record.

## 14. Web Review Closure

原 01B 主日志混合记录了 fixture 初次失败、修正后的 core 失败、最终通过结果以及静态检查，出现 `2 failed, 28 passed → 30 passed` 等多个阶段，不能直接把最后一行解释为每个独立 suite 的最终结果。本轮在当前 worktree final state 重新执行并记录了独立 exit code：core Scheduler suite 为 `3 passed, 15 warnings`、exit code 0；combined suite 为 `30 passed, 15 warnings`、exit code 0。历史失败输出没有删除。

原 `test_empty_and_normal_cached_transport_have_no_transition` 只是手工构造 `CachedRequestData`，未证明 continuing RUNNING lifecycle；现已修正为真实流程：`WAITING → first schedule → update_from_output() → RUNNING → continuing cached schedule`，并先断言 `scheduled_cached_reqs.req_ids == ["request-a"]`，再断言 `reclaim_transitions == [None]`。

原 allocation-failure case 曾在首次 admission 就因资源不足产生 `KeyError`，不符合测试目标。现使用真实首次 admission + `update_from_output()` 进入 RUNNING，再以 deterministic monkeypatch 让本 step `allocate_slots()` 返回 `None`；Scheduler 经过 materialization 和 self-preemption failure path 后，`scheduled_cached_reqs` 不包含 transition，private prepared plan 被清理。该测试不执行任何 free/reuse。

victim contract：transition 仅存于单次 `schedule()` 的 local `req_to_reclaim_transition`；priority 分支从 `scheduled_running_reqs` 移除 victim 并同步 pop transition，self-preemption 清理 private prepared plan。`_make_cached_request_data()` 只遍历最终 scheduled list，因此被移除 victim 不会 transport；没有跨 step persistent transition。默认非-priority loop 的 pinned 行为是 pop 当前失败 request，本轮未添加无意义改动。

本轮未修改 production code；只修改了 `tests/v1/core/test_reclaim_transport.py` 的 fixture/oracle，并更新本记录与 `PROJECT_STATE.md`。production 的 normal `overwrite=False`、reclaim R1 mutual exclusion、zeroing/CoW、`effective_kv_len.apply_write()` publication 均保持不变。

最终 evidence：

`/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01b-web-review-fix/m1-t2-01b-web-review-fix.log`
