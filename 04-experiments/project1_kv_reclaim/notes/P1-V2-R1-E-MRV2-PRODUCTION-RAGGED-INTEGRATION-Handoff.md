# P1-V2-R1-E MRV2 Ragged Identity 执行交接

## 当前判断

状态：`PASS_PENDING_WEB_REVIEW`，不是最终 PASS。三个 closure evidence gap 已完成，仍需 Web review 接受 evidence，不能自动关闭 Slice。GPU 0 在 2026-09-27 仍有其他进程且 GPU 利用率为 100%，但剩余约 70 GiB 显存足够；未终止或干扰其他进程。

## Resume / Snapshot Lifecycle 审计

- `SOURCE_FACT`：MRV2 Scheduler 在 `scheduler.py:1145-1154` 将 `scheduled_resumed_reqs` 并入 `scheduled_new_reqs`，并据此构造 `NewRequestData`。
- `SOURCE_FACT`：同一 `Scheduler.schedule()` 在 `scheduler.py:1174-1196` 为 scheduled new/resumed requests 导出 Full Snapshot；running requests 才走可选 Allocation Delta。
- `SOURCE_FACT`：MRV2 `execute_model()` 在 `model_runner.py:2261-2269` 按 finish、add、update、apply Ragged transport 顺序执行。
- `SOURCE_FACT`：`add_requests()` 为 new/resumed `NewRequestData` 创建 MRV2 `RequestState` slot；`_apply_ragged_kv_updates()` 随后通过 `req_id_to_index` 将 Snapshot 应用到该持久 slot，见 `model_runner.py:856-897`。
- `SOURCE_FACT`：E commit 中 MRV1 原路径也在其 add/condense lifecycle 后按 Snapshot 替换 Worker mirror row；resume 走 `scheduled_new_reqs` 的 MRV1 add/re-add 分支，见 `git show 7ed3763:vllm/v1/worker/gpu_model_runner.py` 的 `_update_states()` 与 Ragged transport 段。
- 审计结论：以上固定源码没有显示 MRV2 resume/Snapshot 与 frozen authority contract 冲突；目前没有依据把它分类为 `DESIGN_CONFLICT` 或直接新增 runtime fallback/兼容逻辑。
- `TO_VERIFY`：当前新增 MRV2 runner tests 覆盖非连续 persistent slot `[7,2]`、Snapshot/Delta mirror materialization、capacity reserve 不推进 E、frontier commit 与缺失 transport fail-closed；尚未覆盖真实 Scheduler preempt→resume→MRV2 `add_requests()`→Snapshot→slot materialization 的闭环。因此 preemption/resume 的 runner-level lifecycle 仍未被测试证明。
- `OBSERVED`：新增测试 `test_scheduler_preempt_resume_materializes_full_snapshot_at_new_req_idx` 已真实驱动 Ragged priority Scheduler：A 被 preempt，B 完成后 A 以 `scheduled_new_reqs` resume；resume output 只含 A 的 Full Snapshot，不含 Allocation Delta。旧 mirror slot `7` 清空，当前 slot `2` 完整 materialize `rows/counts/effective_lens/state_version`。
- `OBSERVED`：Worker success commit 已使用 forward-entry capture 的 persistent `req_idx`、source `state_version`、source E 和 target E；没有在 sample/postprocess 阶段重新从 `SchedulerOutput` 推导 transaction source。

## 已完成改动（未提交）

- `gpu_worker.py` 恢复 runner generation 只由 `use_v2_model_runner` 决定；`page_group_size` 不再选择 MRV1。
- `VllmConfig.validate_ragged_core()` 对 Ragged + MRV1 显式 fail-fast。
- MRV2 `gpu/model_runner.py` 增加 Ragged worker mirror 初始化、Snapshot/Delta transport apply、persistent `req_idx` gather、RaggedStepViews、Dense metadata bypass、ForwardContext staging 与 sample/postprocess 成功后的 Worker E commit。
- `gpu_model_runner.py` 撤销 E commit 新增的 MRV1 Ragged lifecycle、重复 reshape 和 ForwardContext wiring；共用 Scheduler/Attention/C/D substrate 保留。
- 删除无其他调用者的 `RaggedWorkerPhysicalState.reindex()`；增加 slot 清理后 Snapshot 全量替换测试。
- 新增 `tests/v1/worker/test_gpu_model_runner_v2_ragged.py`。
- `vllm/v1/worker/gpu/warmup.py`：发现 generic V2 warmup 使用无 Ragged transport 的 synthetic `SchedulerOutput`，导致真实 A100 startup 在 `_apply_ragged_kv_updates()` fail closed；Ragged active 时跳过该 generic warmup，避免伪造 allocator/mirror state。
- `tests/v1/core/test_scheduler.py`：priority helper 增加可选 `kv_cache_spec`，供 Ragged preempt/resume 测试复用。

这些是当前 worktree 的实现事实，不代表完整 E gate 已关闭。未创建 commit。

## 已运行验证

- `tests/v1/worker/test_gpu_model_runner_v2_ragged.py` + `test_ragged_kv_state.py`：`25 passed`。
- Ragged spec/layout/state、manager/planner、Scheduler transport/ownership、Ragged execution 与 `attn_utils`：`86 passed, 7 skipped`；skip 为 CUDA-only tests 在 `CUDA_VISIBLE_DEVICES=` 下跳过。
- `py_compile`、`ruff --select I,F,E9`、`git diff --check`：通过。
- 完整 Ruff 对 MRV1 文件仍报告 6 个既有 `UP038`；未清理无关 lint debt。
- `vllm serve --help=all` 已确认本地 CLI 参数名，包括 `--page-group-size`、`--max-model-len`、`--max-num-seqs`、`--gpu-memory-utilization`、`--dtype`、`--enforce-eager`、`--async-scheduling`、`--enable-prefix-caching`。
- 新增 resume test 和 Ragged focused suite：`51 passed`；新增 resume 单测通过。
- 一组 Dense/MRV2 与旧 V1 worker tests 失败按 `TEST_STALENESS` 分类：默认引用本地不存在的 `facebook/opt-125m`，且网络代理缺少 `socksio`；此前隐藏 CUDA 的失败按 `CUDA_REQUIRED` 分类。没有观察到本 Slice 相关 focused `REGRESSION`。
- A100 GPU 0 低资源验证成功：Qwen3-0.6B、BF16、`gpu_memory_utilization=0.2`、`max_model_len=64/128`、`max_num_seqs=1/2`、`page_group_size=2`、prefix/async/spec/CUDA Graph 关闭。日志确认 `Using V2 Model Runner`、Ragged generic warmup skip；instrumented run 观察到 `Ragged KV write entered` 和 `Ragged attention forward entered`。
- A100 输出：single request、15-token boundary prompt、two concurrent requests、mixed-style concurrent requests、finish 后新 request 均成功；Dense/Ragged 同 prompt、`temperature=0` 输出文本一致。

## 本轮最终证据闭环（2026-09-27）

- Exact token parity：本地 `LLM.generate()` 读取 generated token IDs；Dense/Ragged 两侧均为 `[12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279]`，`exact_equal=true`。证据：`raw/p1-v2-r1-e-final-evidence-closure/05-final-a100-token-parity-smoke.log`。
- True same-step mixed：真实单步 A decode `1` token、source `E>0` 与 B new prefill `13` tokens、source `E=0`；MRV2 记录 `idx_mapping_np=[1,0]`、`query_start_loc=[0,1,14]`、post-write A=`9`/B=`13`，同一 forward `requests=2` 进入 Ragged write/read。证据：`raw/p1-v2-r1-e-final-evidence-closure/03-mixed-step-trace.log`。
- Physical page free/reuse：focused test 记录 A ownership `(1,2,3,4)`、A finish 后 free `7→11` 且 A canonical state 消失；B ownership `(4,3)`、free `11→9`，自然观察 overlap `[3,4]`，但测试不依赖固定 ID。证据：`raw/p1-v2-r1-e-final-evidence-closure/04-focused-tests-final.log`。
- `Ragged generic V2 warmup skip` scope：Validated for current `enforce-eager` / CUDA-Graph-disabled E scope; future CUDA Graph/compile warmup requires separate design.

## 未完成 Gate

本 Slice 仍不覆盖 prefix caching、async scheduling、spec decode、TP/PP/DCP/PCP、CUDA Graph/compile warmup、quantized KV、MLA/sliding window 或 non-uniform compaction。旧 Dense fixture suite 的失败继续分类为 `TEST_STALENESS`/`CUDA_REQUIRED`，没有观察到当前 focused source regression。

开始环境和审计记录位于：
`04-experiments/project1_kv_reclaim/raw/p1-v2-r1-e-mrv2-production-ragged-integration/`。

## 当前工作树

implementation worktree：`worktrees/p1-vllm-reclaim`；branch：`p1/v2-token-compaction-v026`；开始 HEAD：`7ed376375479c3de2fc392cb8356d35aa695a7dc`。当前 source/tests 仍为 dirty；没有覆盖用户的 untracked design 文档，没有创建 commit。

Gate 建议：Architecture `PASS`，CPU `PASS`，Resume `PASS`，A100 Startup `PASS`，A100 Ragged Execution `PASS`，Exact Token Parity `PASS`，True Mixed-Step `PASS`，Physical Page Reuse `PASS`；Overall 为 `PASS_PENDING_WEB_REVIEW`，`Next Allowed Action: WEB_REVIEW_CURRENT_SLICE`。不得进入下一 Slice。
