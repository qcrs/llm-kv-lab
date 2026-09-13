# P1-M5-T1 源码地图

本地图基于 implementation worktree `worktrees/p1-vllm-reclaim` 在本 Slice 开始前的实际源码与 `nl -ba` 输出。Source Identity：branch `p1/v2-token-compaction-v026`，HEAD `9ea5d804ca6cc3a7a7212d83d626acb1ddd023ea`，工作树干净。

| ID | File | Symbol | Pre-change Lines | Role | Why Relevant |
| --- | --- | --- | --- | --- | --- |
| SRC-01 | `vllm/v1/core/sched/output.py` | `SchedulerOutput` | L208-L291 | Scheduler→Worker transport | 需要增加可选 V2 plan carrier；`make_empty()` 是空输出构造点。 |
| SRC-02 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` / `SchedulerOutput(...)` | L438-L458；L1172-L1191 | 生成调度输出 | 这是 plan 绑定当前 scheduled step 并进入 output 的生产 seam。 |
| SRC-03 | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.execute_model` | L1773-L1780；L1946-L2014 | Worker 消费 SchedulerOutput | forward 调用在 L1984；`ExecuteModelState` 在 L2002-L2009 发布；本 Slice 只读取/保存 plan，不执行 compaction。 |
| SRC-04 | `vllm/v1/outputs.py` | `ModelRunnerOutput` | L231-L294 | Worker→Scheduler 序列化结果 carrier | 可增加独立 `CompactionResultData` 列表，不复用 V1 command。 |
| SRC-05 | `vllm/v1/engine/core.py` | `EngineCore.step` / `step_with_batch_queue` | L576-L606；L617-L705 | 结果返回与 Scheduler 接收 | 两条真实结果路径都将 `ModelRunnerOutput` 传给 `Scheduler.update_from_output()`。 |
| SRC-06 | `vllm/v1/core/sched/output.py` | `ReclaimTransitionData` | L112-L118 | V1-only Scheduler→Worker command | V1 whole-block pre-forward transition；本 Slice 保持定义和字段不变。 |
| SRC-07 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.update_from_output` | L1661-L1706 | Scheduler 结果消费 seam | 本 Slice 只提取/验证 V2 result carrier，不修改 canonical ownership、E 或 free lifecycle。 |
| SRC-08 | `vllm/v1/worker/gpu/model_runner.py` | `ExecuteModelState` | L2266-L2273 | post-forward worker per-step state | 可保存当前 step 的 plan，供后续 M5-T2 seam 使用；不加入 payload/E/row mutation。 |
| SRC-09 | `vllm/v1/core/sched/scheduler.py` | `processed_step_seq` / `deferred_frees` | L331-L337；L1675-L1679；L2345-L2384 | 现有 lifetime fence | 仅阅读；本 Slice 不修改 lifetime、deferred-free 或 safe-free 逻辑。 |

## 传输选择的 Source Fact

`SchedulerOutput` 已大量使用 request-indexed `dict[str, ...]`，而 `ModelRunnerOutput` 注释明确说明会序列化到 Scheduler 且 torch tensor 传输昂贵。因此本 Slice 的实现选择是：plan 使用 `dict[str, CompactionPlanData]` 保留 request identity；result 使用 `list[CompactionResultData]` 保持独立 per-request records，并由 Scheduler 端按 `request_id` 提取。两者都使用 Python 标量/list，不提前物化 CUDA tensor。

## 结果链路

`Scheduler.schedule()` → `SchedulerOutput` → `model_executor.execute_model()` → `GPUModelRunner.execute_model()` / `sample_tokens()` → `ModelRunnerOutput` → `EngineCore.step()` → `Scheduler.update_from_output()`。本轮只建立字段和结构校验 seam；M5-T2 才负责 post-forward physical transition 与 result publication。
