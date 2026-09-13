# P1 V1 Core End-to-End Architecture

## 1. Canonical Runtime Chain

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
→ future allocate_slots() based on completed E
```

## 2. Slice Placement

- S1：Worker `RequestState.effective_kv_len`，normal delta 与 reclaim target 的 persistent physical state。
- S2：`positions` 保持 logical model/RoPE coordinates；`cache_positions` 进入 physical slot mapping。
- S3：`effective_kv_seq_lens` 进入 baseline FA2 attention metadata/backend，限制 physical KV visibility。
- T2：Scheduler prepared whole-block plan → `CachedRequestData.reclaim_transitions` → Worker `_commit_reclaim_transition()`；retained + same-step new 只形成一个 final-row overwrite descriptor。
- T3：successful `update_from_output()` → manager hierarchy dense reconcile → `BlockPool.free_blocks()` → Scheduler completed E；下一步 allocator 使用 E。

## 3. State Planes

```text
Scheduler control plane:
Request.num_computed_tokens       = logical L
Request.effective_kv_len          = completed physical E or None
req_to_blocks                     = canonical ownership

Worker execution plane:
RequestState.num_computed_tokens  = logical scheduling state
RequestState.effective_kv_len     = physical KV frontier
BlockTables                       = dense physical addressing view
```

`BlockTables` 不等于 canonical ownership。Scheduler 只有在 successful completion reconcile 后才移除 removed blocks 并让它们进入 BlockPool reusable state。

## 4. Canonical Transaction

```text
old canonical:  [B0 B1 B2 B3 B4 B5]
retained:       [B0 B1 B4 B5]
same-step new:  [B6]
Worker row:      [B0 B1 B4 B5 B6]
```

以 `L=94, E_target=62, q=3` 为例：logical positions 为 `94,95,96`，physical cache positions 为 `62,63,64`。真实 forward 完成后 `E_completed=65`；Scheduler completion 执行 `validate → reconcile dense ownership → free removed → commit E`。

## 5. Freeze Boundary

当前 V1 只接受 whole-block、single-group、MRV2 baseline。任何 token-level compaction、policy、multi-group、prefix/spec/async/CUDA Graph、V2 gather/scatter 变化都必须通过新的设计审查，不得在 V1 frozen core 上顺手修改。

