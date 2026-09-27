# P1-V2-R1-E MRV2 Production Ragged Identity 接入计划

## 1. 本轮目标与边界

目标是在 pinned vLLM worktree 中将 Production Ragged Identity 接入 MRV2，恢复 `vllm/v1/worker/gpu/model_runner.py` 为 Ragged production runner，并撤销 E commit 为 MRV1 临时增加的 runner fallback、lifecycle 与重复 reshape。本轮不改 Scheduler canonical ownership、transport contract、C/D physical layout 和 Attention backend，不进入 non-uniform compaction 或性能优化。

## 2. 启动审计

- OBSERVED：implementation worktree `/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim` 位于 `p1/v2-token-compaction-v026`，HEAD 为 `7ed376375479c3de2fc392cb8356d35aa695a7dc`，初始工作树 clean，无 staged/unstaged diff。
- OBSERVED：该 commit 的 `gpu_worker.py` 改动以 `page_group_size is None` 限制 V2 runner；`gpu_model_runner.py` 改动新增 Ragged Worker mirror、condense/reindex、StepViews、ForwardContext staging、Worker commit 和 Ragged reshape。
- OBSERVED：MRV2 `RequestState` 维护 `req_id_to_index`、`index_to_req_id` 和 `free_indices`；`prepare_inputs()` 生成 `InputBatch.idx_mapping_np`，顺序为当前 batch 到 persistent request slot 的映射。
- OBSERVED：MRV2 当前 `execute_model()` 依次执行 `finish_requests()`、`add_requests()`、`update_requests()`，然后 prepare/forward；`sample_tokens()` 在 `postprocess_sampled()` 后继续返回 output。
- OBSERVED：C 阶段 `gpu/attn_utils.py` 已调用共享 `ragged_physical_cache_shape()` 建立 Ragged cache view；MRV1 E 的 reshape 重复定义几何。
- OBSERVED：`RaggedWorkerPhysicalState.reindex()` 只有 MRV1 runner 调用；现有 worker-state tests 未调用该 API。
- OBSERVED：环境为 `/home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310/bin/python`、Torch `2.11.0+cu129`、CUDA runtime `12.9`，`vllm` import 指向 implementation worktree。
- OBSERVED：主机可见 NVIDIA A100 80GB；本轮审计时 GPU 0 已被其他进程占用且报告 100% utilization。不得终止该进程；真实 GPU 验证需等待 GPU 0 可用。

## 3. 修改分类

| 分类 | 范围 | 处理 |
|---|---|---|
| KEEP | `KVCacheManager` / `Scheduler` Ragged plan-reserve-transport 与 success commit | 不修改；只读核对 source E、version、scheduled token 数的一致性 |
| KEEP | `ForwardContext` Ragged fields、`attention.py` unified Ragged dispatch、`ragged_layout.py` / `ragged_forward.py`、`gpu/attn_utils.py` reshape | 不修改；MRV2 作为 producer 接入既有 seam |
| REWRITE-IN-MRV2 | `gpu/model_runner.py` initialization、transport apply、request lifecycle、step gather、metadata bypass、ForwardContext 与 Worker commit | 按 persistent `req_idx` 重新接入；active request 必须由 `idx_mapping_np` gather |
| REVERT | `gpu_worker.py` E runner fallback；`gpu_model_runner.py` 中 E 专属 Ragged lifecycle / reshape | 仅撤销 `7ed3763` 引入的内容，不触碰其余 MRV1 历史实现 |
| REVERT | `RaggedWorkerPhysicalState.reindex()` | 仅在 MRV2 tests 不依赖且移除 MRV1 caller 后删除，保持 mirror slot identity 稳定 |
| NO_CHANGE_REVIEWED | Scheduler authority、physical page allocation、Snapshot/Delta schema、C/D geometry 与 unified attention | 本轮不新增 owner、allocator、placement 或 backend |

## 4. 实施顺序

1. 恢复 `Worker.use_v2_model_runner = vllm_config.use_v2_model_runner`；Ragged + MRV1 在 central config validation fail-fast。
2. 在 MRV2 `initialize_kv_cache()` 基于唯一 Ragged group 和 `group.layer_names` 初始化 Worker mirror、identity placement tensors 与 layer index；cache reshape继续走 `attn_utils`。
3. 在 MRV2 lifecycle 完成 request add/update 后应用 Snapshot/Delta；remove 时使用删除前 `req_idx` 清镜像。Ragged active 却缺 transport、req slot 或状态 source 时 fail closed。
4. 当前 step 用 `idx_mapping_np[:num_reqs]` gather mirror，验证 version、capacity、page IDs，再构造 `RaggedStepViews`；Ragged 分支跳过 Dense attention metadata producer，并向 `set_forward_context()` 传 StepViews/layer indices。
5. 在 `ExecuteModelState` 保存 forward-entry source view；last-rank sampling/postprocess 成功后，按 batch→persistent slot 映射提交 Worker vector E。Scheduler success-tail commit维持现有 ownership并与相同 snapshot/delta source 对齐。
6. 删除 MRV1-only runner lifecycle 和重复 reshape；补 MRV2 gather、snapshot/delta、slot reuse、reserve-before-success、Dense regression focused tests。
7. 静态检查与可运行 focused tests；GPU 0 空闲后执行 MRV2 startup/profile、single prefill/decode、boundary、Dense/Ragged parity、mixed batch、并发、finish/reuse。禁止杀进程或切换到其他 GPU。
8. 更新 MRV2 reconciliation Code Trace 和 raw evidence；不创建 commit、不关闭 Parent Task、不进入下一 Slice。

## 5. 核心不变量

- Scheduler 是 canonical page ownership authority；Worker mirror 可丢弃；StepViews 仅为当前 step 派生数据。
- Worker mirror row key 固定为 MRV2 persistent `RequestState req_idx`；不得 per-step compact/reindex。
- 当前 batch 对 mirror 的唯一索引来源是 `InputBatch.idx_mapping_np`，不是 `range(num_reqs)`。
- Scheduler reserve 与 Worker Snapshot/Delta apply 仅扩 capacity；forward 和 sampling/postprocess 成功前不推进 E。
- 写 slot 使用 source E；attention seq lens 使用 source E + 本轮 query length；Worker 与 Scheduler commit 使用同一 forward-entry source/version/token count。
- Ragged 不进入 Dense slot mapping / attention metadata fallback；physical cache 仍为 `[P, Hp, B, 2D]`，复用唯一 zero-copy reshape。

## 6. 验证与停止条件

- Static：修改 Python 文件 `py_compile`、focused `ruff`、`git diff --check`。
- Focused tests：Ragged worker state/layout/forward、MRV2 runner lifecycle、Scheduler transport、Dense MRV2 regression。
- A100：使用 `CUDA_VISIBLE_DEVICES=0`、`/data/models/Qwen3-0.6B`、BF16、`gpu_memory_utilization≈0.2`、`max_model_len=128`、`max_num_seqs=2`、`page_group_size=2`，关闭 prefix/async/spec/CUDA Graph；以实际 CLI help 确认参数。
- GPU 0 被其他 workload 占用时，只运行不触碰 CUDA 的工作；不杀进程、不临时改用 GPU 1/2。若 GPU 0 未能在本次执行窗口释放，则 A100 项记 `BLOCKED_BY_SHARED_GPU0`，不宣称 acceptance PASS。
- 如实际 MRV2 lifecycle 与冻结 authority 不可调和，停止并记录 `DESIGN_CONFLICT`；普通代码/环境/test failure 应修复或如实记录，不升级为架构冲突。

## 7. 状态与限制

当前本文件是执行计划，不是 Slice PASS 证据。最终是否 PASS 取决于源修改、测试与 A100 matrix 的实际结果。所有新增工程文档使用中文；raw command output 保持原样。
