# P1 Risk / Fallback Decision Tree

## Source or Environment

identity/ABI mismatch → BLOCKED_ENV / SOURCE_CONFLICT；不 install/cherry-pick；先修 M0。

## Row Reconstruction

add_row/replacement semantic可用 → preferred zero-copy。
hidden attention/order issue → stop and review V1-R copy fallback。
需要 ragged architecture → reject scope expansion。

## Ownership

ACK validation passes → commit/free。
ownership/request-state mismatch → no-free + debug。
worker live reference不清 → no-free / deferred-free V3，不加 global sync掩盖。

## Logical / Physical

separate consumers可实现 → proceed。
必须全局 physical seq_lens → design conflict。
attention adapter暂不可行 → reference backend/oracle fallback。

## Allocation

effective frontier works → reuse test。
logical length回补 → fix accounting before E2E。
only free counter changes, no ID reuse → Gate fail。

## V2

PyTorch oracle fail → no Triton。
gather correct/scatter fail → upstream-style scatter。
staged correct/perf poor → retain correct negative result。
fused alias complexity → do not pursue。

## Performance / Quality

capacity improves, latency not → valid result。
quality unacceptable at target retention → lower compression/limit scope。
no profiler signal for V3 → skip with reason。

所有 fallback 都需在 Task Card允许或 Web + 用户批准；Codex不得自动选择。

