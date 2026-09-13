# P1-M1-T3-INTEGRATION-CLOSURE 真实 Engine 生命周期验证报告

## 1. 背景与验证动机

`P1-M1-T3-IMPL-01A` 的 unit/integration oracle 已证明 dense ownership reconcile、safe free 和 reuse primitive。本轮进一步验证真实 `LLMEngine`、真实 Qwen3 model forward、Worker T2 transition 与 Scheduler T3 completion commit 能否在同一个生命周期中闭合，并确认原 request reclaim 后仍能继续 generation。

## 2. 环境与版本

- 日期：2026-09-01，Asia/Shanghai
- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/physical-kv-reclaim-v026`
- HEAD：`568afb3a13806beb53bb2e6bd518269357b237c0`
- Model：`/data/models/Qwen3-0.6B`
- GPU：GPU 2，NVIDIA A100 80GB PCIe
- dtype：BF16
- TP/PP/DP/DCP/PCP：1/1/1/1/1
- MRV2、eager、normal FA2
- prefix caching/spec decode/async scheduling/CUDA Graph/connector 均关闭
- `VLLM_ENABLE_V1_MULTIPROCESSING=0`：仅为进程内可观测性，仍使用真实 EngineCore、Scheduler、GPU Worker 和 model forward。

## 3. Smoke 构造

1. 创建 request A：94 个 prompt token、`max_tokens=8`。
2. 第一次真实 forward 完成后，Scheduler request 的 logical progress 为 94，canonical row 为 6 blocks。
3. smoke-only wrapper 调用既有 `_set_prepared_reclaim_plan()`：保留 indices `(0,1,4,5)`，`E_target=62`。
4. 下一真实 Engine step 走 `Scheduler.schedule()`、T2 transport、Worker dense row publication、Qwen3 forward 和 `update_from_output()`。
5. T3 commit 后加入 request B，验证它能获得被 A 释放的 block ID。
6. 继续 `engine.step()`，直至 A 和 B 都以 length reason 完成。

instrumentation 仅位于实验脚本，通过 wrapper 读取已有 Scheduler/Worker state；production source 未修改。

## 4. OBSERVED Runtime Chain

```text
Scheduler prepared plan
→ cached reclaim transition
→ Worker row [1,2,3,4,5,6] → [1,2,5,6]
→ forward-entry E_target=62
→ real Qwen3 q=1 forward
→ Worker completed E=63
→ Scheduler.update_from_output()
→ canonical [1,2,3,4,5,6] → [1,2,5,6]
→ blocks 3/4 ref_cnt=0
→ free count 10860 → 10862
→ request B receives released block 4
→ request A continues to 8 generated tokens and finishes
```

## 5. Ownership / E / Reuse Evidence

| 项目 | OBSERVED |
|---|---|
| prepared old row | `[1,2,3,4,5,6]` |
| retained / Worker post-forward row | `[1,2,5,6]` |
| transition forward-entry E | `62` |
| Worker post-forward completed E | `63` |
| Scheduler post-commit E | `63` |
| Scheduler post-commit canonical row | `[1,2,5,6]` |
| released IDs | `[3,4]` |
| released refcount | `3:0, 4:0` |
| free blocks | `10860 → 10862` |
| request B row | `[4]` |
| real reused ID | `4` |
| request A output | 8 tokens，`finish_reason=length` |
| request B output | 2 tokens，`finish_reason=length` |

## 6. 调试过程

历史失败均保留在同一 raw log：

1. wrapper 未透传 `schedule(throttle)` 参数，首个 schedule 前失败；同时发现 EngineArgs 默认 async，随后显式关闭。
2. Engine internal request ID 带 UUID suffix，最初使用 external ID 导致 plan 未注入；A 的 normal generation仍完成。
3. 首次 closure assertion 将 forward-entry E=62误当 post-forward E；真实 S1 forward完成后 E=63，修正 smoke oracle后通过。

这些是 harness/evidence 问题，均未修改 production architecture。

## 7. 测试结果

最终 Engine smoke：

```text
P1_M1_T3_INTEGRATION_CLOSURE_PASS
engine_smoke_final_rc=0
```

既有 T3/T2/R1/S1 regression 在本 Slice 前 final state 已为 `43 passed`；本轮收尾另在 GPU 2 复跑 targeted suite并记录于 raw log。

## 8. 该结果证明了什么

在冻结 baseline 中，真实 Scheduler decision、Worker dense transition、真实 Qwen3 forward、Scheduler completion commit、physical E、canonical ownership、BlockPool release/reuse和原 request继续 generation已形成闭环。

## 9. 该结果没有证明什么

不构成 benchmark或性能结论；不证明 HBM返还系统；不覆盖 policy、prefix/spec/async/CUDA Graph、PP/DCP/PCP、多 KV group、connector/offload或重复 reclaim广义支持。

## 10. Evidence

- `/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/m1-t3-integration-closure/m1-t3-integration-closure.log`
- `/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/scripts/p1_m1_t3_integration_closure.py`

## 11. Closure 建议

当前 Slice evidence 支持将 `P1-M1-T3` 标记为 `CLOSED_PENDING_WEB_REVIEW`。根据 workspace authority，Codex 不自行关闭 Parent Task；最终 CLOSED 状态由用户 + Web Review确认。

`Next Allowed Action: WEB_REVIEW_CURRENT_SLICE`
