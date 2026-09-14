> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Risk Register, Fallbacks & Decision Gates

项目目标是可完成的 resume system project，所以每个高风险点都必须提前有 fallback，而不是遇到问题后重新设计全局架构。

| Risk | 触发信号 | 优先方案 | Fallback | 是否阻塞 Core |
|---|---|---|---|---|
| Global backing reshape不适配当前 allocator | alias/shape test失败 | dedicated ragged planner + one backing | 独立 ragged allocation helper | 是，R1 |
| NHD无法 zero-copy virtualize | stride不满足 | ragged-specific HND-like storage | dedicated virtual adapter/copy（仅debug） | 是，需最终zero-copy |
| MRv2 flat rows复杂 | staged kernel难泛化 | `[R*G,B]` + helper | 3D storage但内部flatten write | 是 |
| normal FA impl有static-head假设 | descale/sink shape错 | dedicated RaggedFA adapter | 直接调用 FA varlen primitive | 是 |
| Prefill member copy成本高 | identity tax明显 | correctness first | 后期 fused transpose kernel | 否 |
| Python BlockPool metadata爆炸 | init时间/RAM异常 |先小 pool验证 | 提前移植 bulk int32 ring | capacity benchmark前阻塞 |
| P1 scheduler protocol不够表达 per-group | result ambiguity | vectorized plan/result | first R3 one-request synchronous path | 是 |
| same-step allocation+compaction复杂 | fence mismatch | 先禁止重叠 | 后续专门 Slice | 否（早期） |
| TP ranks depths diverge | checksum mismatch | MAX-reduce lengths | abort compaction, keep full state | R6 only |
| Preemption stale physical/compression state | stale page read/double free | explicit lifecycle invalidation | preemption时完全 reset/recompute | R5 only |
| Piecewise CG不兼容 | capture/replay error | attention eager split | enforce eager entire model | 否 |
| AOT map收益小 | waste差异不显著 | report negative result | adjacent map保持 | 否 |
| Triton writeback无端到端收益 | boundary占比低 |不默认启用 |保留 Torch/scratch | 否 |

## Gate A — R1-A2 Global Pool

只有满足以下条件才允许写 RaggedBlockTables：

- `num_global_pages` formula verified；
- one raw backing verified；
- identity total memory verified；
- host metadata规模评估完成。

## Gate B — R1-D1 KV Write

attention read之前，write-only differential必须 PASS。若失败，不允许用 end-to-end生成结果“猜”地址问题。

## Gate C — R1-E Identity Tax

Ragged Identity可以比 dense慢；R1只要求 correctness。但必须记录 prefill/decode初始 overhead，作为 R9 optimization baseline。

## Gate D — R3 Physical Closure

“worker row变短”不是完成。必须同时看到 scheduler canonical row、free count和 cross-request reuse。

## Gate E — Optimization Admission

AOT 可以直接做（低风险、capacity-oriented）。Triton/CG 必须先 profile：

```text
compaction boundary cost material? → Triton
CPU/launch gap material?           → CG metadata optimization
```

不允许为了简历关键词预设正收益。

## Stop-loss Rule

任何 Stretch feature如果连续需要改动两个以上本来不在该 feature ownership 的核心 subsystem，暂停并降级；优先保证 Core/Breadth evidence。典型例子：PP若同时要求重构 global cluster map、allocator ownership、compression coordinator，则停止，不影响项目完成。
