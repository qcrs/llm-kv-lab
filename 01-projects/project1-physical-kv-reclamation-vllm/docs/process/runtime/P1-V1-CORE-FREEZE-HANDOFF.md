# P1 V1 Core Freeze Handoff

## 当前 accepted architecture

P1 V1 已完成 whole-block physical KV reclamation core：`L` 与 `E` 解耦，logical positions 与 physical cache positions 解耦，FA2 physical visibility、Worker dense BlockTables、Scheduler canonical ownership reconcile、safe free、real block reuse 和 physical-E-based allocation 已闭合。

## Gate basis

`P1-V1-CORE-GATE-REVIEW-01` 推荐并记录 `PASS`；Web/Architecture Gate 已最终裁定 `PASS / ACCEPTED`。真实 GPU2 Qwen3-0.6B Engine closure 证明 request A reclaim 后继续生成，request B 复用 released block。

## Source identity

Accepted V1 checkpoint：`cd444b72ceecb2496bf015baf8c18bf883151fa1`，tag
`p1-v1-core-accepted`，branch `p1/physical-kv-reclaim-v026`，worktree `CLEAN`。
Upstream parent：`568afb3a13806beb53bb2e6bd518269357b237c0`。Freeze 当时的 dirty diff
仅作历史 provenance。

## Baseline scope

A100、MRV2、BF16、FA2、eager、TP/PP/DP/DCP/PCP=1、single group、`block_size=16`、prefix/spec/async/CUDA Graph/connector off。其余配置保持 `NOT VALIDATED`。

## Frozen rule

`DO NOT MODIFY V1 CORE WITHOUT REOPEN`。任何 V1 source change 必须先走 `P1-V1-CORE-REOPEN-XX`。

## Next phase

下一窗口默认角色是 ChatGPT Web = Architect / Mentor / Reviewer。唯一允许动作：`P1-V2-DESIGN-REVIEW-01`；V2 尚未开始，不能直接实现 Triton/CUDA 或 production patch。
