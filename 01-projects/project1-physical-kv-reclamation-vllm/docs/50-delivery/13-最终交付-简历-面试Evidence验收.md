# P1 Final Delivery / Resume / Interview Evidence

## Required Delivery

- architecture and state machine report；
- exact source map/pins/environment；
- diff/commit history；
- unit/integration/E2E suites；
- allocator reuse proof；
- capacity/serving/quality tables；
- Nsys/NCU reports for applicable stages；
- reproduction commands/manifests；
- limitations/negative results；
- artifact index。

## Resume Claim Gate

每条 bullet 必须映射到 project-owned diff + tests + raw metric/profiler。上游/reference能力用“integrated/leveraged/surveyed”，不用“built from scratch”。

V1 完成可表达：在 pinned vLLM 中实现 logical/physical KV decoupling 和 transaction-safe whole-block reclamation，并验证 allocator released-page reuse。

V2 完成可表达：实现 PyTorch/Triton staged paged-KV compaction，并用 NCU/NSYS分析 compaction cost 与 retained-KV收益。

数字只在最终 evidence 后填写。

## Interview Must Explain

- 为什么 logical retention 不等于 physical reclaim；
- four state semantics；
- two commit points and alias risk；
- effective allocation；
- why non-contiguous IDs work；
- free page vs HBM；
- V1 zero-copy fallback；
- V2 scratch/race/crossover；
- upstream/reference/our ownership；
- negative result如何改变决策。

## Close Gate

clean-shell reproduce、all mandatory Gates、unsupported fail-fast、raw chain、negative results、no inflated claim。

## v1.2 Explicit Non-Claims

V1/V2 Core 未做 prefill-time reclamation，因此 README / 简历 / 面试不得声称：

- 单请求最大首次 prompt/context 因 P1 自动提高；
- final-prefill 前的 KV peak 一定下降；
- BlockPool free pages 等价于 CUDA/HBM allocation returned。

允许的 strongest capacity claim 是：在 request 完成 full prefill 并 reclaim 后，request-owned pages 减少、free pool 恢复，后续/并发 request 可真实复用 released page IDs；serving concurrency/preemption 仍以 workload evidence 为准。
