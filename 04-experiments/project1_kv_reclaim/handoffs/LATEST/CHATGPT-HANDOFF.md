# P1 Latest Handoff

当前状态：`P1 V1/Core = PASS / ACCEPTED / FROZEN`。

Accepted source identity：commit `cd444b72ceecb2496bf015baf8c18bf883151fa1`，tag
`p1-v1-core-accepted`，branch `p1/physical-kv-reclaim-v026`，worktree `CLEAN`。
Upstream parent：`568afb3a13806beb53bb2e6bd518269357b237c0`。

S1、S2、S3、T2、T3 均 `CLOSED`。V1 baseline 为 A100、MRV2、BF16、FA2、eager、
TP/PP/DP/DCP/PCP=1、single KV group、`block_size=16`；prefix/spec/async/CUDA Graph/
connector/offload 关闭。其他配置 `NOT VALIDATED`；V2 与 token-level compaction 等
`OUT OF SCOPE`。

V1 frozen rule：`DO NOT MODIFY V1 CORE WITHOUT REOPEN`。未来修改先创建
`P1-V1-CORE-REOPEN-XX`，列明 reason、affected invariant、affected evidence 与
regression plan。

下一允许动作：`P1-V2-DESIGN-REVIEW-01`（仅 design review，不是 implementation）。
此文件为 derived handoff；权威状态见 project `PROJECT_STATE.md`。
