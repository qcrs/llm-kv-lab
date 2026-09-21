# P1-V2 R1-A3-IMPL-01 Contract Alignment Code Trace

## 1. Slice 信息与结论

- Slice：`P1-V2 R1-A3-IMPL-01 — Pre-C1 Contract Alignment`。
- 执行模式：`SLICE_EXECUTE`。
- 日期：2026-09-20。
- implementation worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`。
- branch：`p1/v2-token-compaction-v026`。
- start HEAD：`47e2d319b2c37934d21613c06856ffcb4e28a15b`。
- end HEAD：仍为 `47e2d319b2c37934d21613c06856ffcb4e28a15b`；本轮未 commit。
- start worktree：clean。
- verdict：`PASS_PENDING_WEB_REVIEW`。
- Next Allowed Action：`WEB_REVIEW_CURRENT_SLICE`。

本 Slice 只完成 A3 freeze 中 `MUST CHANGE BEFORE C1` 的 CPU/control-plane contract
hardening。没有实现 `resolve_kv_address()`，没有修改 GPU backing、KV write、FlashAttention、
production Scheduler/ModelRunner wiring 或 compaction payload movement。

## 2. 背景与验证动机

B12 已建立：

```text
Scheduler canonical ownership
→ Snapshot / Allocation Delta
→ Worker discardable mirror
```

A3 review 确认该方向正确，但指出五个进入 C1 前必须关闭的 contract gap：

1. `C` 仍是独立 `num_clusters` 参数，没有 canonical placement source of truth；
2. E/counts shape fence 无法识别 physical ownership ABA；
3. Snapshot/Delta/Worker 没有统一 generation contract；
4. 同一 request 的 Snapshot 与 Delta 没有 XOR 约束；
5. compaction prefix-retention 已存在，但未被明确锁定为测试 invariant。

本轮验证的核心 invariant 是：

```text
placement 唯一决定 C
physical ownership mutation 由单一 state_version fence
normal frontier commit 不制造第二套 clock
Snapshot XOR Delta
compaction retained row = old row prefix
```

## 3. Canonical Input 与 Source Audit

实现严格依据：

- `P1-V2-R1-A3-Placement-and-Physical-Address-Architecture-Freeze.md`；
- `P1-V2-R1-B12-IMPL-01-Control-Plane-Code-Trace.zh-CN.md`；
- 当前 HEAD 的 B12 manager、transport、Worker mirror；
- 当前 A1/A2 `RaggedAttentionSpec` 与 planner。

确认未出现 `DESIGN_CONFLICT`：A3 文档审计的 HEAD 与本轮 start HEAD 相同，当前源码中的
pre-C1 gaps 与 freeze 描述一致。

## 4. MemberPlacementMap

新增中立模块：

```text
vllm/v1/ragged_kv_layout.py
```

`MemberPlacementMap` 是 model/rank static metadata，不属于 request state。它保存：

```text
num_layers = L
num_kv_heads = Hkv
page_group_size = Hp
member_to_cluster[L * Hkv]
member_to_column[L * Hkv]
```

Canonical flat member identity：

```text
m = layer_idx * Hkv + local_kv_head_idx
```

Derived topology：

```text
M = L * Hkv
C = M / Hp
```

`num_clusters` 不再由 manager 或 Worker caller 独立传入。两者都从
`placement.num_clusters` 取得 C。

R1 constructor validation 强制：

- `L/Hkv/Hp > 0`；
- `M % Hp == 0`；
- mapping 长度等于 M；
- cluster 位于 `[0,C)`；
- column 位于 `[0,Hp)`；
- `(cluster,column)` 无重复；
- mapping 恰好覆盖全部 `C × Hp` slots。

`MemberPlacementMap.identity()` 实现 freeze 中的 deterministic adjacent-head mapping：

```text
cluster = layer * (Hkv // Hp) + head // Hp
column = head % Hp
```

自定义有效 bijection 不会被重写成 identity，因此后续 AOT/custom placement 不需要修改 B12
Scheduler ownership contract。

## 5. Scheduler state_version 语义

`RaggedRequestPhysicalState` 新增唯一 generation：

```text
state_version
```

没有新增 `frontier_version`、`allocation_epoch` 或 `compaction_epoch`。

本轮实现严格遵循 freeze：

| Transition | state_version |
|---|---|
| reserve 实际 append physical pages | `+1`，一次 transition 只加一次 |
| normal `commit_effective_lens()` | 验证 expected version，不递增 |
| compaction/repacking reconciliation | 成功后 `+1` |
| Snapshot export / Worker apply | 不改变 Scheduler generation |
| Worker remove/re-add same Snapshot | 不改变 Scheduler generation |
| request free 后同 ID 新 physical incarnation | 通过 last-generation ledger 使用更大的 generation |

`RaggedCapacityPlan` 增加 `source_state_version`。`apply_capacity_plan()` 在任何
BlockPool mutation 前同时验证 version、E 与 counts。reserve 成功后返回：

```text
expected_source_state_version
new_state_version
```

`commit_effective_lens()` 继续保留 expected E value fence，并新增
`expected_state_version`。正常 write commit 只推进 E，不改变 physical generation。

为满足 request incarnation rule，manager 保存轻量 `_last_state_versions[request_id]`。
`free_request()` 删除 canonical ownership 后保留最后 generation；同 request ID 再次规划时的
空 source generation 大于旧 incarnation，后续 reserve 不会复用旧 generation。

## 6. Transport Generation Fence

`vllm/v1/core/sched/output.py` 调整如下：

```text
RaggedRequestStateSnapshotData
    state_version

RaggedPageAllocationDeltaData
    expected_source_state_version
    new_state_version

RaggedCompactionResultData
    expected_source_state_version
```

删除 compaction result 中语义模糊的 optional `state_version`。Worker→Scheduler compaction
carrier 仍只报告 source/final shape，不携带 allocator-authoritative freed/retained page IDs。

Snapshot 的语义是无条件用 Scheduler canonical generation 完整覆盖 Worker mirror；它不要求
Worker 已持有某个 source generation。Delta 的语义是增量 ownership transition，必须同时
匹配 source version、E 与 counts。

## 7. Snapshot XOR Delta

`RaggedKVUpdateData.__post_init__()` 强制：

```text
set(snapshots) ∩ set(allocations) == ∅
```

同一个 request 同时进入两个 channel 时立即失败。不同 request 可以分别使用 Snapshot 和
Delta。

如果未来某 request 同一步既要 materialize 又要 reserve，contract 要求 Scheduler 先完成
canonical reserve，再只发送最终 Full Snapshot。本 Slice 没有引入 application ordering 或
ACK/stream protocol。

## 8. Worker Persistent Mirror

`RaggedWorkerPhysicalState` constructor 改为接收同一 `MemberPlacementMap`，并由它派生 C。
Persistent CPU state 现在是：

```text
rows            [Rmax,C,M] int32
counts          [Rmax,C]   int32
effective_lens  [Rmax,C]   int32
state_versions  [Rmax]     int64
```

未 materialize request 的 Worker generation sentinel 是 `-1`；合法 Snapshot generation
必须非负。

`apply_snapshot()` 先完整验证 version、shape、capacity、page IDs，再 atomic publish
rows/counts/E/version。

`apply_allocation_delta()` 先验证：

```text
current version == expected source version
current E/counts == expected source E/counts
flat length == sum(appended counts)
page IDs valid and non-duplicate
new version == source version + 1  (有 page append)
```

全部通过后才 atomic publish rows/counts/version；E 不变。

Worker `commit_effective_lens()` 验证 expected version 与 expected E，成功只推进 E；
`remove_request()` 清 rows/counts/E 并把 version 重置为 `-1`。`gather()` 同时返回所选 request
的 generation copy，仍未生成任何 virtual address。

## 9. ABA Fence Oracle

核心 regression 明确构造：

```text
Worker current:
E/counts = X
version = 9

old Delta:
expected E/counts = X
expected version = 3
```

尽管 shape 完全相同，`apply_allocation_delta()` 因 generation 不匹配拒绝。测试同时断言
rows、counts、E、version 均未发生部分 mutation。

Scheduler compaction 也有同类 oracle：source E/counts 完全匹配、只有 version stale 时，
canonical state 与 BlockPool free count 均不改变。

## 10. Compaction Prefix-Retention

`reconcile_compaction()` 继续只从 Scheduler canonical rows 推导：

```text
retained[c] = old_row[c][:new_count[c]]
detached[c] = old_row[c][new_count[c]:]
```

测试对所有 cluster 逐 row 比较 exact prefix，并验证 detached IDs 恰好来自 suffix。
例如：

```text
[B0,B1,B2,B3], new_count=2
→ retained=[B0,B1]
→ detached=[B2,B3]
```

成功 reconciliation 的顺序仍为：

```text
validate source version/E/counts
→ build prefix candidate
→ publish Scheduler canonical ownership with version+1
→ BlockPool.free_blocks(detached)
```

Worker carrier 不存在 `retained_page_ids` 或 `freed_page_ids`，不能任意选择保留页或获得
allocator authority。

## 11. 逐文件修改表

| 文件 | symbol | 操作 | Contract 结果 |
|---|---|---|---|
| `vllm/v1/ragged_kv_layout.py` | `MemberPlacementMap` | 新增 | canonical member placement、bijection validation、derived C、identity builder |
| `vllm/v1/core/ragged_kv_cache_manager.py` | `RaggedRequestPhysicalState` | 修改 | 新增 canonical `state_version` |
| 同上 | `RaggedCapacityPlan` | 修改 | 增加 source generation fence |
| 同上 | `RaggedAttentionManager.__init__` | 替换 | 删除 arbitrary `num_clusters`，改为 placement-derived C |
| 同上 | `apply_capacity_plan` | 修改 | version/E/counts stale validation；page append 推进 generation |
| 同上 | `commit_effective_lens` | 修改 | 验证 generation；E-only commit 不递增 |
| 同上 | `reconcile_compaction` | 修改 | 验证 source generation；prefix retained；成功 generation+1 |
| 同上 | `free_request` | 修改 | 保留 last-generation ledger，防同 ID incarnation ABA |
| `vllm/v1/core/sched/output.py` | Ragged DTOs | 修改 | Snapshot/Delta/compaction generation contract |
| 同上 | `RaggedKVUpdateData.__post_init__` | 新增 | Snapshot XOR Delta |
| `vllm/v1/worker/gpu/ragged_kv_state.py` | constructor/state arrays | 修改 | placement-derived C + persistent mirror generation |
| 同上 | snapshot/delta/frontier/remove/gather | 修改 | validate→candidate→atomic publish；version fence |
| `tests/v1/test_ragged_kv_layout.py` | placement tests | 新增 | identity/custom bijection/invalid topology oracle |
| `tests/v1/core/test_ragged_kv_cache_manager.py` | manager tests | 修改 | version semantics、incarnation、stale compaction、exact prefix |
| `tests/v1/worker/test_ragged_kv_state.py` | transport/mirror tests | 修改 | end-to-end version、ABA、atomicity、XOR、re-add |

## 12. Tests 与观察

最终合并命令覆盖：

```text
tests/v1/test_ragged_kv_layout.py
tests/v1/core/test_ragged_kv_cache_manager.py
tests/v1/worker/test_ragged_kv_state.py
tests/v1/core/test_ragged_kv_cache_planner.py
tests/v1/test_ragged_attention_spec.py
tests/v1/core/test_single_type_kv_cache_manager.py
tests/v1/core/test_reclaim_ownership.py
tests/v1/core/test_reclaim_transport.py
tests/v1/core/test_compaction_reconciliation.py
tests/v1/worker/test_gpu_block_table.py
```

`OBSERVED`：

```text
78 passed, 15 warnings in 23.18s
```

warnings 为已知 `vllm._version` 缺失提示与 Torch JIT deprecation，不是 failure。

质量检查：

```text
python -m py_compile ...  PASS
uvx ruff check ...       PASS
git diff --check         PASS
```

这些结果证明 placement/version/XOR/prefix contract 与既有 B12、A1/A2、Dense/V1 ownership /
transport regression 兼容。它们不证明 GPU Ragged execution correctness 或性能。

## 13. Scope Audit

本轮 production source 修改仅限：

```text
vllm/v1/ragged_kv_layout.py
vllm/v1/core/ragged_kv_cache_manager.py
vllm/v1/core/sched/output.py
vllm/v1/worker/gpu/ragged_kv_state.py
```

未修改：

```text
KVCacheSpecRegistry
KVCacheCoordinator
KVCacheManager.allocate_slots
Scheduler.schedule
GPUModelRunner
BlockTables production integration
attn_utils.py
FlashAttention / FA2 / FA3
KV write / slot mapping
Triton / CUDA compaction
payload movement
```

源码与 tests 中没有新增 `resolve_kv_address`、`virtual_block_id` 或 `virtual_slot` 实现。
Ragged 仍未 production activate，Dense path 默认行为不变。

## 14. Remaining GPU Blocker 与 C1 状态

已知 blocker 保持不变：planner 以 `Hp`-wide physical page 计算 backing，但当前 GPU
`_reshape_kv_cache` / backend path 仍可能按 Dense `Hkv` geometry materialize。

本 Slice 按 A3 freeze 明确未修复该问题。它属于后续 GPU backing execution Slice，不属于 C1。

A3 pre-C1 contract hardening 已完成，因此：

```text
C1 CPU Reference Addressing
= contract-level unblocked after Web review

GPU production execution
= still blocked
```

Codex 不自动开始 C1。当前唯一允许的下一动作是：

```text
WEB_REVIEW_CURRENT_SLICE
```

## 15. Evidence Index

Raw evidence 目录：

```text
04-experiments/project1_kv_reclaim/raw/r1-a3-impl-01/
```

包含：

- `source-identity.log`：branch、HEAD、最近五个 commits；
- `git-status.log`：最终 implementation worktree 状态；
- `git-diff-stat.log`：tracked diff stat；
- `git-diff-check.log`：whitespace check 与 exit code；
- `pytest.log`：最终 78-test 合并结果；
- `ruff.log`：本 Slice Python 文件 lint；
- `py-compile.log`：本 Slice Python 文件 compile 结果。

`git diff --stat` 不显示未 tracked 的两个新增文件；它们是：

```text
vllm/v1/ragged_kv_layout.py               87 lines
tests/v1/test_ragged_kv_layout.py         108 lines
```
