# P1-M1-T3-IMPL-01A Physical Ownership Reconciliation 实施记录

## 1. 背景与验证动机

T2 已把 Worker execution row 从 `[B0 B1 B2 B3 B4 B5]` 压缩为 `[B0 B1 B4 B5]`，并能与 same-step `B6` 合成 `[B0 B1 B4 B5 B6]`。但 Scheduler canonical ownership 仍保留 `[B0 B1 B2 B3 B4 B5 B6]`，allocator 仍以 logical `L` 计算容量，因此 reclaimed blocks 既不可复用，也会在下一步被逻辑历史重新分配回来。

本 Slice 验证并实现以下 invariant：成功 step 到达 `Scheduler.update_from_output()` 后，Scheduler completed physical frontier、dense ownership 与 BlockPool refcount 同步提交；在此之前不释放任何 candidate block。

## 2. 环境与版本

- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/physical-kv-reclaim-v026`
- HEAD：`568afb3a13806beb53bb2e6bd518269357b237c0`
- Python：`/home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310/bin/python`
- GPU：GPU 0，NVIDIA A100 80GB PCIe
- Model：`/data/models/Qwen3-0.6B`，仅用于本地 config/source oracle，不联网
- Baseline：single KV group、block size 16、prefix cache/spec decode/async/CUDA Graph 关闭

## 3. SOURCE_FACT

1. `Request.num_computed_tokens` 是 logical progress；reclaim 不得降低它。
2. `KVCacheManager.allocate_slots()` 的 running fast path 最终通过 required block count 与当前 `req_to_blocks` 长度求增量。
3. `SingleTypeKVCacheManager.req_to_blocks` 是 request canonical ownership；whole-request `free()` 不适合 selective reclaim。
4. `BlockPool.free_blocks()` 逐 block 降低 `ref_cnt`，归零后进入 free queue；不需要 force-free。
5. `Scheduler.update_from_output()` 是 processed GPU step 的 completion seam；KV load failure、已结束或 stale request 在逐 request commit 前被过滤。
6. T2 transition 与 current-step `new_block_ids` 已保存在同一 `SchedulerOutput.scheduled_cached_reqs`，Scheduler 可确定性重建 commit transaction，无需 Worker ACK/result ABI。

## 4. DESIGN_CHOICE

1. `Request.effective_kv_len: int | None` 保存 Scheduler completed physical frontier。`None` 完全保留 upstream logical allocation；`int` 才启用 physical base。
2. reclaim step completion 设置 `E = E_target + q`；后续 successful non-reclaim step持续执行 `E += q`。
3. `update_from_output()` 从当前 `SchedulerOutput` 的 aligned transition/new-block data 读取事务，不新增 persistent ACK map。
4. 新增 `KVCacheManager → KVCacheCoordinator → SingleTypeKVCacheManager` narrow dense reconcile path；Scheduler 不穿透 manager internals。
5. reconcile 先完整验证，再发布 dense ownership，最后通过 `BlockPool.free_blocks()` 释放 removed blocks。allocator 不会看到“仍被 request ownership 引用但已经 reusable”的中间状态。
6. recompute preemption 在释放剩余 canonical blocks后将 `effective_kv_len` 清为 `None`，resumed request 回到 upstream logical path。

## 5. Runtime 事务

```text
schedule step N
→ prepared reclaim transport
→ Worker dense row publication / forward
→ Scheduler.update_from_output(step N)
→ validate old ownership + retained order + same-step tail
→ reconcile dense canonical ownership
→ refcount-safe free removed blocks
→ commit Scheduler completed E
→ future allocate_slots uses physical E
```

实现中 validation 与 ownership reconcile 成功后才完成 physical E commit；若 validation 失败，ownership、free queue 与 `request.effective_kv_len` 均不变。reconcile 内部在所有可能失败的 validation 完成后才 mutation。

## 6. Canonical Worked Example

```text
block_size = 16
L = 94
E_old = 94
old = [B0 B1 B2 B3 B4 B5]
retained = [B0 B1 B4 B5]
E_target = 62
q = 3
same-step new = [B6]

update_from_output 前 Scheduler:
[B0 B1 B2 B3 B4 B5 B6]

Worker:
[B0 B1 B4 B5 B6]

update_from_output 后:
L = 97
Scheduler E = 65
canonical = [B0 B1 B4 B5 B6]
released = [B2 B3]

next q = 1:
physical target = 66
owned blocks = 5
new allocation = 0
completion 后 Scheduler E = 66
```

## 7. 修改清单

### ADDED

| file | symbol | purpose |
|---|---|---|
| `vllm/v1/request.py` | `Request.effective_kv_len` | Scheduler completed physical frontier；`None` 保留 normal path。 |
| `vllm/v1/core/kv_cache_manager.py` | `reconcile_reclaimed_blocks()` | single-group public ownership reconcile API。 |
| `vllm/v1/core/kv_cache_coordinator.py` | `reconcile_reclaimed_blocks()` | manager hierarchy forwarding 与 single-group guard。 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `reconcile_reclaimed_blocks()` | fail-before-mutation validation、dense row publication、refcount-safe release。 |
| `tests/v1/core/test_reclaim_ownership.py` | 4 tests / combined T1-T10 oracles | commit、dense ownership、exact free、reuse、next allocation、fail-closed、preemption、normal regression。 |
| 本实施记录与 raw log | documentation/evidence | 保存 source identity、命令、结果、限制和 diff。 |

### MODIFIED

| file | function/class | before | after | why |
|---|---|---|---|---|
| `vllm/v1/core/kv_cache_manager.py` | `allocate_slots()` | 以 logical computed tokens 计算 running capacity | `effective_kv_len` 非空时以 physical completed frontier 计算 | 防止 reclaim 后 bogus re-allocation。 |
| `vllm/v1/core/sched/scheduler.py` | `update_from_output()` | 只处理 logical/output lifecycle | successful reclaim commit dense ownership 与 E；后续 step推进 E | completion point 才能安全释放。 |
| `vllm/v1/core/sched/scheduler.py` | `_preempt_request()` | reset logical state | 同时 reset physical decoupling state | recompute resumed path不能沿用旧 physical frontier。 |

### REPLACED

仅对已经成功执行的 reclaim transaction：T2 的“Worker compact、Scheduler ownership 保持 quarantine”状态，在 `update_from_output()` 被替换为“Scheduler dense ownership + refcount-safe real release”。Normal request lifecycle 没有被替换。

### DELETED

DELETED: NONE。

## 8. Oracle 与实际结果

| Oracle | Setup / expected | OBSERVED |
|---|---|---|
| O1 no premature free | schedule 后、completion 前 ownership/full refcount不变 | PASS |
| O2 physical commit | `E_target=62, q=3 → E=65` | PASS |
| O3 dense ownership | 7-block canonical → 5-block dense row，无 NULL hole | PASS |
| O4 exact release | 仅 `{B2,B3}` | PASS |
| O5 refcount/free queue | removed `ref_cnt 1→0`，free count `+2` | PASS |
| O6 real reuse | request B 获得 `{B2,B3}` 中的 physical ID | PASS |
| O7 no bogus allocation | `E=65, q=1, owned=5 → new_block_ids=None` | PASS；completion 后 E=66 |
| O8 fail closed | expected old count mismatch | PASS；no E/ownership/free mutation |
| O9 post-commit preemption | 只释放当前 dense row，无 removed block double-free | PASS |
| O10 normal path | `effective_kv_len=None` | PASS；17 tokens正常分配2 blocks |

第一次 sandbox pytest 在 fixture 初始化前因 NVML/device detection 不可见得到 `4 failed`，不是 runtime oracle 结果；主机 GPU 0 复跑进入逻辑后曾有一个 test 常量错误：测试未扣除 pinned `null_block`，错误期待 free count 8。修正为 `pool.num_gpu_blocks - 1` 后 production 未因该失败修改。另一次 code review 发现 post-reclaim non-reclaim completion 必须持续推进 E，已补充 production 与第二次 completion oracle。

## 9. 最终测试

Targeted：

```text
CUDA_VISIBLE_DEVICES=0 .../bin/python -m pytest -q tests/v1/core/test_reclaim_ownership.py
4 passed, 15 warnings, rc=0
```

Regression：

```text
CUDA_VISIBLE_DEVICES=0 .../bin/python -m pytest -q \
  tests/v1/core/test_reclaim_transport.py \
  tests/v1/worker/test_gpu_reclaim_commit.py \
  tests/v1/worker/test_gpu_effective_kv_state.py \
  tests/v1/core/test_single_type_kv_cache_manager.py
39 passed, 15 warnings, rc=0
```

同一 final state 的 combined run 另得 `43 passed, 15 warnings`。`py_compile` 与 `git diff --check` 均为 `rc=0`。

## 10. 该结果证明了什么

在冻结 baseline 中，whole-block reclaim 已从 Worker metadata-only compact 闭合到 Scheduler completed physical accounting、canonical dense ownership、BlockPool free queue 增长和另一 request 的真实 physical ID reuse。原 request 下一步不会按 logical history补回被 reclaim 的 blocks。

## 11. 该结果没有证明什么

本结果不是 HBM 返还 OS/CUDA allocator 的证明，也不是性能/吞吐结论。未覆盖 prefix caching、spec decode、async、PP>1、DCP/PCP、多 KV group、connector/offload、CUDA Graph、token-level compaction、policy、跨失败恢复或重复 reclaim 在更广配置下的语义。

## 12. Evidence Index

- `/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t3-impl-01a/m1-t3-impl-01a.log`
- `/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim/tests/v1/core/test_reclaim_ownership.py`

## 13. 下一步

`Next Allowed Action: WEB_REVIEW_CURRENT_SLICE`

`Proposed Next Action: Review P1-M1-T3-IMPL-01A implementation, tests, evidence, and ownership transaction invariants.`
