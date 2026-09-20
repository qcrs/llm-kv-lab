# P1-V2-R1-A2 实现代码追踪

## 1. Slice 信息

- Slice：`R1-A2 — Physical Namespace / Planner / Global Backing`
- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/v2-token-compaction-v026`
- Start HEAD：`bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00`
- End HEAD：`WORKTREE`（未创建 commit）
- 日期：2026-09-19
- 状态：`PASS`

## 2. A2 设计目标

Dense planner 的 `num_blocks` 表示跨 layer 共享的 logical token-block depth，因此 Dense general path 会按 layer/group 数量除分配内存。Ragged planner 的 `num_blocks` 改为 worker-local physical page slots：每个 slot 的大小是 `RaggedAttentionSpec.page_size_bytes`，所有支持的 local layer 通过一个 `KVCacheTensor.shared_by` alias 到同一 raw backing。R1-A2 只冻结 namespace、capacity 和 descriptor，不引入 Scheduler ownership、placement map 或执行 layout。

## 3. 修改文件总表

| 文件 | symbol / 区域 | 操作 | 修改前 | 修改后 | 原因 / invariant |
|---|---|---|---|---|---|
| `vllm/v1/core/kv_cache_utils.py` | imports | 新增 | 未导入 Ragged spec | 导入 `RaggedAttentionSpec` | planner 必须按 resolved Spec dispatch |
| `vllm/v1/core/kv_cache_utils.py` | `get_kv_cache_config_from_groups` | 新增 | 仅 Dense/packed/uniform 分支 | exactly-one Ragged 分支，`Nphys=floor(M/P)`，单一 global tensor，override/mixed fail-fast | physical namespace 与 Dense logical namespace 分离 |
| `vllm/v1/core/kv_cache_utils.py` | `get_max_concurrency_for_kv_cache_config` | 新增 | 使用 Dense `memory_per_block` | Ragged 使用 `L * max_memory / P` 的 physical-page accounting | 防止遗漏 layer 数导致并发高估 |
| `vllm/v1/core/kv_cache_utils.py` | `_max_memory_usage_bytes_from_groups` | 未修改但审计 | general group-size 计算 | 保持不变 | Ragged spec 已将每层 memory 表达为 `D * G * P`，最终为 `L*D*G*P` |
| `vllm/v1/core/kv_cache_utils.py` | `_pool_bytes_per_block` | 未修改但审计 | Dense/override capacity helper | 保持不变 | R1 Ragged 禁止 `num_gpu_blocks_override`，无需重构 unsupported path |
| `vllm/v1/worker/gpu/attn_utils.py` | `_allocate_kv_cache` | 未修改但审计 | `shared_by` layers alias 同一 tensor | 保持不变 | 已能表达 global raw backing，A3 才处理 reshape |
| `tests/v1/core/test_ragged_kv_cache_planner.py` | T1-T6 | 新增 | 无 A2 planner tests | 4 个高信息量 tests 覆盖 T1-T6 语义 | 保持测试克制 |
| `04-experiments/.../P1-V2-R1-A2-IMPL-01-Code-Trace.zh-CN.md` | 本记录 | 新增 | 无本 Slice trace | 记录 source chain、scope、命令和限制 | 满足中文 evidence 要求 |

没有删除代码；没有修改 `CacheConfig`、A1 spec、worker allocation 或治理状态文件。

## 4. 逐文件代码追踪

### `vllm/v1/core/kv_cache_utils.py`

#### `get_kv_cache_config_from_groups`

原逻辑在 non-uniform Dense 情况按 `group_size` 计算 `get_num_blocks()`，并生成每个 layer slot 的 tensor。新增逻辑先收集 `RaggedAttentionSpec` group，并要求输入恰好一个 Ragged group；Ragged 与 Dense 混合、多个 Ragged group 都抛出 `ValueError`。随后直接计算 `page_size = spec.page_size_bytes`、`num_blocks = max(available_memory // page_size, 0)`，返回一个 `KVCacheTensor(size=num_blocks*page_size, shared_by=group.layer_names)`。Ragged 输入遇到 `num_gpu_blocks_override` 立即拒绝。没有删除 Dense 分支，也没有调用 Dense `get_num_blocks()`。对应 invariant 是 `Nphys=floor(M/P)`、descriptor 数为 1、raw backing 包含全部 slots（包括未来由 BlockPool 保留的 page 0）。

#### `get_max_concurrency_for_kv_cache_config`

原逻辑使用 Dense `memory_per_block = page_size * num_layer_per_group`。新增 Ragged-only 分支要求 exactly one Ragged group，计算 `per_request_bytes = L * spec.max_memory_usage_bytes(vllm_config)`，再用 `ceil(per_request_bytes / P)` 得到 physical pages/request，最终返回 `num_blocks / physical_pages_per_request`。因此在 identity 下为 `L * D * G`，不会错误退化为 `G * D`。没有改变 Dense path。

#### `_max_memory_usage_bytes_from_groups`

`AUDITED / NO CHANGE`。Ragged `max_memory_usage_bytes()` 已返回 `D * G * P`；现有 single/general group-size 逻辑乘上 `L` 后得到 `L * D * G * P`，与 identity memory oracle 一致。

#### `_pool_bytes_per_block`

`AUDITED / NO CHANGE`。该 helper 主要用于 `num_gpu_blocks_override` 的 capacity adjustment；A2 明确禁止 Ragged override，因此不为 unsupported path 扩展 helper。

### `vllm/v1/worker/gpu/attn_utils.py`

`_allocate_kv_cache()` 已按 `KVCacheTensor.size` 创建一个 raw tensor，并把同一对象写入 `shared_by` 中的每个 layer name。该语义足以承载一个 Ragged group 的 global backing，故 `AUDITED / NO CHANGE`。没有进入 `_reshape_kv_cache()` 或 FA2 view。

## 5. 测试代码追踪

- T1/T2：合并在 `test_ragged_planner_global_backing_and_identity_memory`，验证 `Nphys=available_memory//P`、只有一个 descriptor、size、`shared_by`、`Hkv=8/Hp=2/G=4` 与 `CacheConfig.page_group_size is None`。
- T3：同一 case 验证 `L*D*G*P_ragged == L*D*P_dense`。
- T4：`test_ragged_concurrency_counts_all_layer_physical_pages` 使用 `L=3,G=4,D=4`，验证 full request 需要 `48` physical pages，而非 `16`。
- T5：`test_dense_planner_keeps_general_namespace` 验证 Dense general path 仍按 group size 除分配并生成两个 per-layer descriptors。
- T6：`test_ragged_planner_rejects_mixed_groups_and_override` 验证 mixed group 和 `num_gpu_blocks_override` 明确失败。

没有测试 Scheduler ownership、BlockPool page allocation、worker Ragged reshape、FA2 correctness 或 engine startup，因为这些属于后续 Slice。

## 6. Scope Audit

`git diff` 仅新增 `kv_cache_utils.py` 的 planner/concurrency 逻辑与 A2 测试；既有 dirty 的 A1 文件保留但未被本 Slice 重写。grep/diff 审计确认本 Slice 没有修改或注册：`Attention.get_kv_cache_spec`、`KVCacheSpecRegistry` production mapping、`FullAttentionManager`、`RaggedAttentionManager`、`KVCacheCoordinator`、`BlockPool`、Scheduler ownership、`SchedulerOutput`、`BlockTables`、ModelRunner Ragged state、`_reshape_kv_cache()`、FlashAttention、slot mapping、compression、CUDA Graph、Tangram allocator。

## 7. 命令与证据

- `git status --short && git branch --show-current && git rev-parse HEAD`：branch 为 `p1/v2-token-compaction-v026`，start HEAD 为 `bd5a0e9...`，保留既有 dirty A1 文件。
- `/home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310/bin/python -m py_compile vllm/v1/core/kv_cache_utils.py tests/v1/core/test_ragged_kv_cache_planner.py`：通过。
- `git diff --check`：通过。
- `ruff check vllm/v1/core/kv_cache_utils.py tests/v1/core/test_ragged_kv_cache_planner.py`：通过。
- `/home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310/bin/python -m pytest -q tests/v1/core/test_ragged_kv_cache_planner.py`：`4 passed, 9 warnings`。
- 目标 venv 已包含测试所需依赖；警告仅为 `vllm._version` 缺失和 CUDA NVML 不可用，不影响本地 CPU/planner 测试。
- `git diff --stat`：见最终 handoff；A1 既有 dirty diff 与本 Slice diff 同时存在，未清理。

## 8. 最终 Diff 解释

新增：Ragged planner branch、Ragged concurrency branch、A2 targeted tests、本中文 trace。

删除：无。

替换：无 Dense 语义替换；仅在 planner/concurrency 函数前置增加 Ragged resolved-Spec 分支。

仅审计未修改：`_max_memory_usage_bytes_from_groups`、`_pool_bytes_per_block`、worker `_allocate_kv_cache` 及全部 runtime activation surface。

## 9. 下一步边界

A2 只证明 `RaggedAttentionSpec → physical page namespace → planner/global backing descriptor`。它没有证明 Scheduler ownership、worker physical layout、FA2 correctness 或真实 engine activation。下一步必须由 Web Review 决定，不能在本 Slice 自动进入 A3、B1 或 B2。
