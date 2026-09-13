# P1 V1 Core Freeze Handoff

## 交接状态

```text
P1 V1/Core = PASS / ACCEPTED / FROZEN
S1 = CLOSED; S2 = CLOSED; S3 = CLOSED; T2 = CLOSED; T3 = CLOSED
```

## Accepted architecture

当前 accepted core 是 whole-block physical KV reclamation：logical/physical state、
model/physical positions、attention visibility、Worker dense `BlockTables`、Scheduler
canonical ownership、safe-free、real reuse 与 physical-E allocation 已闭合。

## Source identity

Accepted V1 commit：`cd444b72ceecb2496bf015baf8c18bf883151fa1`
；tag：`p1-v1-core-accepted`；branch：`p1/physical-kv-reclaim-v026`；checkpoint
worktree：`CLEAN`。Upstream parent 为 `568afb3a13806beb53bb2e6bd518269357b237c0`。
历史 dirty diff 仅作为 provenance 保留。

## Evidence basis

Gate review、T3 `4 + 39 = 43` historical suite、Gate rerun `4 + 30` selected subset、
以及 GPU2 `/data/models/Qwen3-0.6B` real Engine closure 均已索引。不同 selected suite
数字属于不同 review 时点，不互相覆盖。

## Baseline and limits

A100 80GB、MRV2、BF16、FA2、eager、TP/PP/DP/DCP/PCP=1、single KV group、
`block_size=16`；prefix/spec/async/CUDA Graph/connector/offload 关闭。未验证配置保持
`NOT VALIDATED`，retention policy、token-level compaction、Triton/CUDA compaction、
性能/HBM 结论与 V2 保持 `OUT OF SCOPE`。

## Freeze rule and next phase

```text
P1 V1/Core is frozen.
DO NOT MODIFY V1 CORE WITHOUT REOPEN.
```

任何 V1 修改先创建 `P1-V1-CORE-REOPEN-XX`。下一窗口角色为 ChatGPT Web =
Architect / Mentor / Reviewer；唯一允许动作是 `P1-V2-DESIGN-REVIEW-01`，不是 V2
implementation。
