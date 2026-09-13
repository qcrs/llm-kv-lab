# P1 Code Diff Tracker

## P1-M1-T2-IMPL-01A-R1

### Source Identity

- HEAD：`568afb3a13806beb53bb2e6bd518269357b237c0`
- Branch：`p1/physical-kv-reclaim-v026`
- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`

### Modified Files

| 文件 | 类型 | 函数/类 | 修改摘要 |
|---|---|---|---|
| `vllm/v1/worker/gpu/model_runner.py` | 修改/替换 | `GPUModelRunner._commit_reclaim_transition` | retained 与 same-step new 合并为 final row，一次 overwrite。 |
| `tests/v1/worker/test_gpu_reclaim_commit.py` | 修改/新增 | reclaim commit oracle tests | 验证 one descriptor、state transition、invalid atomicity。 |
| `block_table.py` / `buffer_utils.py` | 未修改但依赖 | `BlockTables`, `StagedWriteTensor` | 复用 pinned primitive。 |

### Added

新增 `same_step_new_block_ids` 参数、final-row composition、descriptor-count 与 boundary tests；新增 validation 覆盖 duplicate、count、alignment、no-op。

### Modified

`effective_kv_len` 仅在 shrink 时 staged；logical state 不变；测试将 canonical same-step composition 作为 single-write oracle。

### Replaced

原逻辑 `retained overwrite + separate append` 替换为 `retained + new → final row → one overwrite`。

### Deleted

删除：无。没有删除历史 DESIGN_CONFLICT evidence。

### Behavioral Diff

Before：same request → two staged descriptors → capacity conflict。After：same request → one composed final-row descriptor → publication succeeds；`[10,11,12,13,14,15]`、E=94 变为 `[10,11,14,15,16]`、E=62，L=94。

### Test Diff

新增/修改 `test_single_write_final_row_avoids_descriptor_conflict`、same-step append、no-op、invalid transition 与 boundary oracle；历史 failing conflict 的意义保留在 implementation record 与 evidence 中。

### Evidence

证据目录：`/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a-r1/`。历史 two-descriptor conflict evidence 未删除。

### Git Diff Stat

累计 diff：7 个 production files，174 insertions，4 deletions；R1 semantic diff 仅涉及上述 `model_runner.py` 与 `test_gpu_reclaim_commit.py`，不能将累计 S1/S2/S3 additions 归为 R1。

## P1-M1-T3-IMPL-01A

### Source Identity

- HEAD：`568afb3a13806beb53bb2e6bd518269357b237c0`
- Branch：`p1/physical-kv-reclaim-v026`
- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`

### Modified Files

| 文件 | 类型 | 函数/类 | 修改摘要 |
|---|---|---|---|
| `vllm/v1/request.py` | 新增 | `Request.effective_kv_len` | Scheduler completed physical frontier。 |
| `vllm/v1/core/kv_cache_manager.py` | 修改 | `allocate_slots()` | reclaimed request 按 physical frontier 分配。 |
| `vllm/v1/core/kv_cache_manager.py` | 新增 | `reconcile_reclaimed_blocks()` | public single-group reconcile path。 |
| `vllm/v1/core/kv_cache_coordinator.py` | 新增 | `reconcile_reclaimed_blocks()` | ownership hierarchy forwarding。 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 新增 | `reconcile_reclaimed_blocks()` | dense ownership + refcount-safe release。 |
| `vllm/v1/core/sched/scheduler.py` | 修改 | `update_from_output()`, `_preempt_request()` | completion commit、持续推进 E、preemption reset。 |
| `tests/v1/core/test_reclaim_ownership.py` | 新增 | T3 T1-T10 combined oracle | real free/reuse 与 fail-closed。 |

### Added

新增 Scheduler physical state、三层 dense reconcile API、T3 ownership/reuse tests、中文实施记录和 raw evidence。

### Modified

allocation 的 normal path保持 logical；仅 `effective_kv_len is not None` 时使用 physical base。`update_from_output()` 成为 successful reclaim transaction commit point。

### Replaced

对 successful reclaim request，T2 quarantine-only ownership替换为 dense canonical ownership与真实 BlockPool release。Normal request 不替换。

### Deleted

删除：无。

### Behavioral Diff

Before：Worker dense row，但 Scheduler仍持有 removed blocks，free capacity不增长。After：completion 时 E=65、canonical row dense、removed refcount归零并可被另一 request复用；下一 q=1不分配新 block。

### Test Diff

新增 `test_reclaim_ownership.py` 四个 test，组合覆盖 T1-T10。最终 targeted `4 passed`，existing regression `39 passed`，final combined `43 passed`。

### Evidence

`/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t3-impl-01a/m1-t3-impl-01a.log`

### Git Diff Stat

完成时 cumulative worktree diff 为 13 个 tracked production files、923 insertions、10 deletions，另有 untracked tests。该统计包含 S1/S2/S3/R1/01B；T3 semantic diff 仅为上表 5 个 production files和 `test_reclaim_ownership.py`，不能把 cumulative additions归为本 Slice。
