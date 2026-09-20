# P1-V2 R1-B12 Ragged Physical State Control Plane Code Trace

## 1. Slice 信息

- Slice：`R1-B12 — Ragged Physical State Control Plane`。
- 执行模式：`SLICE_EXECUTE`，只实现 Scheduler authority、Scheduler→Worker wire contract、Worker persistent mirror；不进入 execution/data path。
- 日期：2026-09-20，时区 `Asia/Shanghai`。
- implementation worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`。
- branch：`p1/v2-token-compaction-v026`。
- start HEAD：`bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00`。
- end HEAD：仍为 `bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00`；本轮未 commit，结果位于 WORKTREE。
- Python：3.10.20；Torch：2.11.0+cu129；Torch CUDA runtime：12.9；`torch.cuda.is_available() == True`。
- verdict：`PASS_PENDING_WEB_REVIEW`。

开始前已有 dirty state，归属 A1/A2，不是本 B12 创建：

```text
M  vllm/config/cache.py
M  vllm/v1/core/kv_cache_utils.py
M  vllm/v1/kv_cache_interface.py
M  vllm/v1/worker/gpu/attn_utils.py
?? tests/v1/core/test_ragged_kv_cache_planner.py
?? tests/v1/test_ragged_attention_spec.py
```

B12 没有修改上述 A1/A2 文件。本轮与既有 dirty state 的唯一共享 tracked 文件是
`vllm/v1/core/sched/output.py`：其中既有 V1 `Compaction*` 与中文注释保留，本轮只新增
Ragged transport dataclass、`SchedulerOutput.ragged_kv_updates`，以及为通过整文件 ruff
对两处既有长行做等价换行/`noqa` 标记。

## 2. Source Audit

### 2.1 vLLM Dense source facts

- `vllm/v1/core/single_type_kv_cache_manager.py:36` 的
  `SingleTypeKVCacheManager` 使用 `req_to_blocks: request_id -> list[KVCacheBlock]`
  作为单一 scalar row ownership。Dense allocation 由 token scalar 经 `cdiv` 推导 block
  count，free 从该 canonical row 取真实 `KVCacheBlock` 后调用 `BlockPool.free_blocks()`。
- `vllm/v1/core/block_pool.py:143` 的 `BlockPool` 保存全部 `KVCacheBlock` identity；
  `get_new_blocks()` 在 free capacity 不足时先失败，成功时增加 `ref_cnt`；
  `free_blocks()` 根据真实 block identity/refcount 归还 free queue。page 0 是 `null_block`。
- `vllm/v1/worker/gpu/block_table.py:17` 的 `BlockTables` axis 0 是
  `num_kv_cache_groups`，其 persistent shape 是每个 KVCacheGroup 一张
  `[max_requests,max_blocks]` 表，不是 physical cluster axis。
- `NewRequestData.block_ids` 与 `CachedRequestData.new_block_ids` 的 outer tuple/list
  均表达 KVCacheGroup。B12 不能把 `C` 个 physical clusters 伪装成 `C` 个 KVCacheGroups。

### 2.2 Tangram pinned reference facts

审计 source：`third_party/tangram@6fa551fc8f6edcc118a2a39b3554ee1520e91edd`。

| Reference | 机制 | 是否采用 | 原因 |
|---|---|---:|---|
| `worker/ragged_block_table.py::RaggedBlockTable` | persistent `[request,group,depth]` + per-group counts | 采用问题建模，重新实现 | 证明 cluster-major、non-uniform row 可行；本项目用独立 CPU mirror `[Rmax,C,M]`，不复制旧 runner 容器 |
| `RaggedBlockTable._append_row_grouped()` | 由 uniform target depth、总数可整除性从 flat IDs 重建 row | 拒绝 | 永久 contract 必须显式携带 `appended_page_counts[C]`，允许 `[1,0,2,0]`，禁止 `len(ids)/C` 推断 |
| `snapshot_row()/restore_row()` | Worker 保存 exact compressed row，re-add 时恢复 | 采用其暴露的问题，拒绝 authority 位置 | flat Dense block list 无法重建 non-uniform row 是事实；恢复 authority 改为 Scheduler Full Snapshot，Worker mirror 可丢弃 |
| `gpu_input_batch.py::effective_seq_lens_cpu` | request×group vector frontier | 采用 vector frontier | B12 使用 `E[C]` / `[Rmax,C]`，不再使用 scalar `compress_max_eff_seq_len` 作为 canonical frontier |
| `outputs.py::compression_freed_block_ids` | Worker 报 freed IDs | 拒绝 | Worker 只报告 source/final physical shape；Scheduler 从 canonical `KVCacheBlock` rows 推导 detached pages 并执行 free |

### 2.3 Adopt / Reject 结论

采用 cluster-major non-uniform state、vectorized effective frontier、exact row-boundary metadata。
拒绝 uniform reconstruction、Worker canonical snapshot、`layer*G+group` 永久 identity、Worker
allocator-authoritative freed IDs。Tangram 只作为 reference，不是代码移植源。

## 3. Canonical State

`RaggedRequestPhysicalState` 位于
`vllm/v1/core/ragged_kv_cache_manager.py:19`：

```text
request_id
  -> effective_lens[C]
  -> page_rows[C][variable depth] of KVCacheBlock
  -> page_counts[C] = len(page_rows[c])
```

`C` 是唯一 canonical physical ownership axis。manager 不接收 layer/head/G，不包含
`layer * G + group` 公式。R1 identity builder 未来可以在 manager 外部构造 `C`；
MemberPlacementMap 属于后续 A3/C。

验证 invariant：

```text
len(E) == C
len(rows) == C
E[c] >= 0
E[c] <= len(rows[c]) * block_size
ordinary page_id != 0
同一 request 内 live page_id 唯一
```

`page_counts` 是 derived property，没有第二份 mutable source of truth。Scheduler rows 保存
`KVCacheBlock` 而不是 int，保留 BlockPool identity/refcount。

### E != capacity

`E[c]` 是已成功写入、已 committed、execution 可读的 frontier；capacity 是
`len(page_rows[c]) * B`。reserve 允许产生 `E=32, pages=3, capacity=48`，但不能在
forward success 前把 E 推进到 33。

## 4. Scheduler Authority

`RaggedAttentionManager` 位于
`vllm/v1/core/ragged_kv_cache_manager.py:41`，继承
`SingleTypeKVCacheManager` 只为未来 Registry interface compatibility，不使用 inherited
`req_to_blocks` 作为 canonical Ragged ownership。

- canonical map 是 `req_to_ragged_state`。
- inherited scalar `get_num_blocks_to_allocate()` / `allocate_new_blocks()` fail-fast：
  `Ragged manager requires vector capacity API`。
- `enable_caching=True` fail-fast；R1 prefix caching OFF。
- `get_num_common_prefix_blocks()` 返回 0，`find_longest_cache_hit()` 返回 no-hit。
- `free()` 明确委托 `free_request()`，不会静默走 Dense 空 row。

Scheduler 是唯一 allocator authority，因为只有它持有真实 `KVCacheBlock` rows，并能在
ownership transition 后安全调用 `BlockPool.free_blocks()`。Worker 只保存 int page IDs，
不持有 refcount、free queue 或 allocator mutation API。

## 5. Plan → Reserve → Commit

### `RaggedCapacityPlan`

`RaggedCapacityPlan` 是 frozen dataclass，包含：

```text
request_id
source_effective_lens[C]
source_page_counts[C]
target_effective_lens[C]
required_page_counts[C]
appended_page_counts[C]
total_new_pages
```

`plan_capacity()`（line 108）是 pure：不存在 request 时读取零向量但不创建 state，不调用
BlockPool。`required[c]=ceil(target_E[c]/B)`；
`need[c]=max(required[c]-current_counts[c],0)`。

### 具体非均匀例子

```text
B = 16
E = [32,56,20,48]
counts = [2,4,2,3]
target = [33,57,21,49]
required = [3,4,2,4]
need = [1,0,0,1]
```

`apply_capacity_plan()`（line 129）先验证 current E/counts 与 plan source 一致，并重新推导
required/appended/total，任何 stale/malformed plan 在 pool mutation 前失败。之后只调用一次
`BlockPool.get_new_blocks(2)`，假设返回 `[101,102]`，按显式 counts scatter：

```text
c0 += [101]
c1 += []
c2 += []
c3 += [102]
```

成功只发布 new rows/capacity，E 仍是 `[32,56,20,48]`。随后
`commit_effective_lens()`（line 180）以 expected source E 做 stale fence，验证 grow-only normal
transition 与 capacity bound，最后 atomic vector commit 为 `[33,57,21,49]`。

grow-only 只属于 normal commit primitive；`RaggedRequestPhysicalState` 本身不声明 E 永久单调，
compaction 使用独立 shrink transition。

## 6. Transport

`vllm/v1/core/sched/output.py` 新增：

- `RaggedRequestStateSnapshotData`：`request_id`、`effective_lens[C]`、
  `page_counts[C]`、cluster-major `flat_page_ids`。
- `RaggedPageAllocationDeltaData`：source E/counts fence、
  `appended_page_counts[C]`、cluster-major `flat_new_page_ids`。
- `RaggedCompactionResultData`：未来 Worker→Scheduler 只报告 source/final E/counts，
  可选 `state_version/step_seq`；没有 freed IDs。
- `RaggedKVUpdateData`：`snapshots` 与 `allocations` 两个 request-indexed aggregate。
- `SchedulerOutput.ragged_kv_updates: RaggedKVUpdateData | None = None`。

Full Snapshot 用于 first materialization、resume/re-add、Worker rebuild、future migration/resync，
它不依赖 Worker 旧状态。Allocation Delta 只用于 Worker 已持有 exact source state 的连续运行。

不复用 `NewRequestData.block_ids` / `CachedRequestData.new_block_ids`，因为它们 outer dimension
是 KVCacheGroup，复用会把 physical cluster axis 错误编码为 fake KVCacheGroups，并污染现有
Dense BlockTables contract。

## 7. Worker Mirror

`RaggedWorkerPhysicalState` 位于
`vllm/v1/worker/gpu/ragged_kv_state.py:23`，persistent CPU representation：

```text
rows           [Rmax,C,M] int32
counts         [Rmax,C]   int32
effective_lens [Rmax,C]   int32
```

API 接收已有 request-lifetime `req_index`，不创建第二套 req_id allocator。

- `apply_snapshot()`（line 68）：validate-all，构造 candidate row，最后一次替换 rows/counts/E。
- `apply_allocation_delta()`（line 98）：同时 fence current counts 与 current E；显式 scatter，
  只更新 rows/counts，不推进 E。
- `commit_effective_lens()`（line 146）：normal grow-only frontier commit，不能超过 capacity。
- `remove_request()`（line 170）：只清 mirror row，不 free BlockPool。
- `gather()`（line 176）：返回 backend-neutral `RaggedClusterStepView`，shape 为
  `[R,C,M]` / `[R,C]` / `[R,C]`，返回 copy，避免 step consumer 改写 persistent state。

本 Slice 没有 UVA/StagedWriteTensor/GPU buffer；接口保留 cluster-major neutral boundary，未来
可替换存储实现而不改变 wire/ownership semantics。

## 8. Re-add / Resume

核心 oracle `test_scheduler_wire_worker_non_uniform_end_to_end_and_readd()`：

```text
Scheduler reserve+commit initial E=[32,56,20,48]
-> Full Snapshot counts=[2,4,2,3]
-> Worker apply snapshot
-> Scheduler reserve target=[33,57,21,49], need=[1,0,0,1]
-> Worker apply delta，双方 capacity 更新但 E 不变
-> 模拟 forward success，双方 commit E
-> Worker remove_request(req_index)
-> Scheduler export fresh Full Snapshot
-> Worker 仅依靠该 Snapshot exact restore rows/counts/E
```

该 oracle 证明 Worker snapshot 不是恢复 authority；Worker mirror 可以丢弃并由 Scheduler
canonical snapshot 完全重建。

## 9. Compaction / Free Authority

`RaggedAttentionManager.reconcile_compaction()`（line 210）输入 source E/counts 与 new
E/counts。它先 validate 全部 cluster，构造 retained candidate rows，验证 new E 不超过 new
capacity，然后：

```text
commit Scheduler ownership
-> derive detached = old_row[c][new_count[c]:]
-> BlockPool.free_blocks(detached)
```

测试从 `[4,4,4,4]` shrink 到 `[2,3,1,4]`，free count 精确增加 6；Request B 随后通过正常
`get_new_blocks()` 至少复用一个 released page ID。没有 force-free、没有直接篡改 refcount。

未来 Worker compaction result 只允许报告 source/final shape；Scheduler 从 canonical rows 推导
detached IDs。Worker 永远不调用 `BlockPool.free_blocks()`。

`free_request()` 同样先从 `req_to_ragged_state` 删除 canonical ownership，再 free 全部真实
`KVCacheBlock`，支持不同 cluster 不同 depth。

## 10. 逐文件修改表

| 文件 | symbol | 操作 | 修改前 | 修改后 | invariant |
|---|---|---|---|---|---|
| `vllm/v1/core/ragged_kv_cache_manager.py` | `RaggedRequestPhysicalState` | 新增 | 无 Ragged Scheduler state | E + cluster rows，counts derived | Scheduler 保存真实 `KVCacheBlock`；C-only |
| 同上 | `RaggedCapacityPlan` | 新增 | 无 vector plan | immutable source/target/count plan | planning pure；stale fence |
| 同上 | `RaggedAttentionManager` | 新增 | Dense scalar manager only | vector ownership manager | 不使用 `req_to_blocks`；prefix off |
| 同上 | `plan_capacity` | 新增 | scalar token plan | `target_E[C]` pure plan | non-uniform first-class |
| 同上 | `apply_capacity_plan` | 新增 | 无 explicit cluster scatter | one pool reserve + explicit counts scatter | reserve 不推进 E |
| 同上 | `commit_effective_lens` | 新增 | 无 vector commit | expected-source atomic grow commit | E<=capacity |
| 同上 | `export_snapshot` | 新增 | 无 self-contained Ragged snapshot | cluster-major exact snapshot | Worker 可无旧状态重建 |
| 同上 | `reconcile_compaction` | 新增 | Dense suffix reconcile only | vector shrink、Scheduler derive/free | ownership commit before free |
| 同上 | `free_request`/`free` | 新增/替换 inherited behavior | inherited Dense row free | Ragged canonical rows free | Worker 无 allocator authority |
| `vllm/v1/core/sched/output.py` | 4 个 Ragged dataclass | 新增 | Dense/V1 scalar carriers | Full Snapshot、Delta、shape result、aggregate | 不重载 KVCacheGroup fields |
| 同上 | `SchedulerOutput.ragged_kv_updates` | 新增 | 无 Ragged channel | optional independent channel | Dense default `None` |
| `vllm/v1/worker/gpu/ragged_kv_state.py` | `RaggedWorkerPhysicalState` | 新增 | Dense BlockTables only | CPU `[Rmax,C,M]` mirror | mirror 可丢弃；int page IDs only |
| 同上 | `RaggedClusterStepView` | 新增 | 无 neutral gather | `[R,C,M]`/`[R,C]` view | 不含 vbid/member table |
| `tests/v1/core/test_ragged_kv_cache_manager.py` | manager tests | 新增 | 无 B1 oracle | pure plan、atomicity、scatter、compaction、reuse、free | non-uniform 为主要 oracle |
| `tests/v1/worker/test_ragged_kv_state.py` | control-plane/Worker tests | 新增 | 无 B2 oracle | snapshot/delta/re-add/gather/invalid input | validate-all→atomic commit |
| `block_pool.py`、`single_type_kv_cache_manager.py`、`kv_cache_manager.py`、`kv_cache_coordinator.py` | Dense ownership/allocator | 仅审计未修改 | Dense behavior | 不变 | B12 不激活 runtime |
| `block_table.py`、`model_runner.py`、`states.py` | Dense Worker lifecycle | 仅审计未修改 | KVCacheGroup tables / existing req index | 不变 | 不形成半激活 runtime |
| Tangram reference files | Ragged reference | 仅审计未修改 | reference implementation | 不复制 | Adopt problem/shape，Reject authority/assumptions |

本轮没有删除 production symbol；没有替换 Dense ownership、BlockTables 或 execution mapping。

## 11. Tests

B12 targeted tests 共 23 个：

- pure plan：验证 pool free count 与 request map 在 plan 前后不变。
- initial non-uniform state：`E=[32,56,20,48]`、counts `[2,4,2,3]`。
- incremental explicit scatter：target `[33,57,21,49]`、need `[1,0,0,1]`。
- reserve/E separation：Scheduler 与 Worker capacity 更新后 E 均保持 source value。
- frontier commit：双方成功后 exact E/rows/counts equality。
- stale/malformed plan、insufficient pool：zero pool mutation、zero canonical mutation。
- invalid snapshot：wrong flat length、E>capacity、NULL 0、duplicate ID 均 atomic reject。
- invalid/stale delta：source counts/E mismatch、wrong flat length、delta duplicate、existing ID、NULL 0 均 atomic reject。
- compaction/reuse：Scheduler derive detached、free count +6、Request B 正常复用。
- remove/re-add：Worker row 清空后仅靠 Scheduler snapshot exact restore。
- gather：request-index order 与 cluster-major shape 正确，返回 copy。
- future compaction result：只含 shape，不含 freed page IDs。
- Dense default：`SchedulerOutput.make_empty().ragged_kv_updates is None`。

最终合并 pytest：

```text
57 passed, 15 warnings in 18.87s
```

组成：B12 23；SingleType Dense 9；既有 reclaim ownership/transport/compaction 19；
Dense BlockTables 2；BlockPool/prefix primitives 4。

warnings 是已知 `vllm._version` 缺失提示与 Torch JIT deprecation，不是 test failure。

## 12. Future Compatibility

- non-uniform compaction：state/plan/snapshot/delta/reconcile 从第一天使用 `C` vector；R2/R3
  不需要改变 ownership axis。
- arbitrary MemberPlacementMap：manager/wire/mirror 不包含 layer/head/G；后续 map 只需把 semantic
  member 映射为 `(cluster,column)`。
- AOT clustering：只改变 placement map 与 cluster topology builder，不改变 B12 API。
- Worker rebuild/migration/resync：Full Snapshot self-contained，可覆盖空 Worker row。
- async：source E/counts 已提供 optimistic concurrency fence；
  `RaggedCompactionResultData` 预留 `state_version/step_seq`。本轮未实现 async ordering/epoch。
- future fused kernel：`RaggedClusterStepView` 提供 cluster-major neutral input；persistent NumPy arrays
  可替换为 UVA/GPU buffer。B12 未实现 fused gather。
- future sharing：R1 只检查同一 request 内 page ID 唯一，没有声明 physical page 跨 request 永久独占；
  将来可由 BlockPool refcount extension 支持 sharing。

尚未实现：MemberPlacementMap、physical address generation、vbid adapter、KV write、FA2 read、
compaction payload move、runtime scheduling、state-version protocol、GPU/UVA mirror。

## 13. Scope Audit

全仓引用审计确认 `RaggedAttentionManager`、`RaggedWorkerPhysicalState`、
`ragged_kv_updates` 没有出现在 production activation surface。

本轮未修改、未激活：

```text
Attention.get_kv_cache_spec
KVCacheSpecRegistry
KVCacheCoordinator construction
KVCacheManager.allocate_slots
Scheduler.schedule main logic
GPUModelRunner lifecycle
BlockTables
attn_utils execution path
FlashAttention / FA2
R1-A3 / R1-C / R1-D1 / R1-D2 / R1-D3 / R1-E
R2 / R3
```

因此本轮只建立 isolated control-plane contract，没有让 Engine 支持 Ragged，也没有同时维护
Dense BlockTables + Ragged execution rows 的半激活状态。

## 14. Diff 总结

新增：

- `vllm/v1/core/ragged_kv_cache_manager.py`：281 行。
- `vllm/v1/worker/gpu/ragged_kv_state.py`：186 行。
- `tests/v1/core/test_ragged_kv_cache_manager.py`：228 行。
- `tests/v1/worker/test_ragged_kv_state.py`：232 行。

删除：无 production file/symbol 删除。

替换/修改：

- `vllm/v1/core/sched/output.py`：B12 相对当前 worktree新增 transport contract 与 optional
  `SchedulerOutput` field；tracked diff 为 `48 insertions, 2 deletions`，两处 deletion 是既有长行的
  等价格式调整。

仅审计：Dense allocator/manager/coordinator/BlockTables/ModelRunner/RequestState 与 Tangram pinned
reference files，均未修改。

注意：普通 `git diff --stat` 不显示 untracked 新文件，因此当前 raw output 只显示 tracked dirty
files；上述 B12 新文件行数由 `wc -l` 单独记录。

## 15. Evidence Index

- `04-experiments/project1_kv_reclaim/raw/r1-b12-impl-01/pytest-final.log`
- `04-experiments/project1_kv_reclaim/raw/r1-b12-impl-01/py-compile.log`
- `04-experiments/project1_kv_reclaim/raw/r1-b12-impl-01/ruff.log`
- `04-experiments/project1_kv_reclaim/raw/r1-b12-impl-01/git-diff-check-focused.log`
- `04-experiments/project1_kv_reclaim/raw/r1-b12-impl-01/git-diff-check-full.log`
- `04-experiments/project1_kv_reclaim/raw/r1-b12-impl-01/git-status.log`
- `04-experiments/project1_kv_reclaim/raw/r1-b12-impl-01/git-branch.log`
- `04-experiments/project1_kv_reclaim/raw/r1-b12-impl-01/git-head.log`
- `04-experiments/project1_kv_reclaim/raw/r1-b12-impl-01/git-diff-stat.log`

`py_compile`、ruff、B12 focused `git diff --check` 均 rc=0。全 worktree
`git diff --check` rc=2，唯一输出是 A2 既有 dirty 文件
`vllm/v1/core/kv_cache_utils.py:973` trailing whitespace；B12 未修改该行，也未擅自清理其他
Slice 的 dirty state。

## 16. 结果边界与下一步

本结果证明 isolated Scheduler authority→wire contract→Worker mirror control-plane 对非均匀
cluster rows 成立，且 invalid/stale input 的关键路径保持 atomic。结果不证明 production Engine
已激活、不证明 KV payload layout/地址/读写正确、不证明 compaction kernel 或 serving 性能。

`Next Allowed Action: WEB_REVIEW_CURRENT_SLICE`

`Proposed Next Action: Web review R1-B12 diff、contracts 与 evidence；不自动进入 A3/C/D/E/R2/R3。`
