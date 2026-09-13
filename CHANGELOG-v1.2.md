# CHANGELOG v1.2 — Final Contract Hardening

## Governance

- Codex PROJECT_STATE write权限改为 factual whitelist；执行后 `Next Allowed Action = WEB_REVIEW_CURRENT_SLICE`。
- `Proposed Next Action` 与 Allowed Next 分离。
- P1/P2 Core M0–M5 默认禁用 AUTO_TASK。
- historical baseline precedence 明确低于 BASELINE-RESOLUTION/current approved state。
- governance files（AGENTS/CURRENT/ADR/Supported Matrix/Master Plan/BASELINE-RESOLUTION）在普通 implementation Slice 中强制 read-only，防止 Agent 自我扩权。
- P1/P2 新增 `REFERENCE-PORTING-LEDGER.md`。

## P1

- 明确 Core 只在 final-prefill 后 reclaim，不降低 first-prefill peak KV，也不提高 single-request max prompt。
- Supported Matrix 新增 DCP/PCP=1、manager/kernel block size 1:1、blocks_per_kv_block=1、CP interleave=1、cascade off。
- `BlockTable.add_row()` 从“primitive exists”与“live transition proven”分离；stale-tail 成为 M1/M2 mandatory seam。
- feature-off identity 与 feature-on retention=1 no-op identity 分离。
- V2 `keep_indices` canonical 改为 `keep_member_indices`，定义 current retained sequence index domain。
- capacity benchmark加入 rolling/staggered arrival、prefill peak、post-reclaim recovery time 与 simultaneous-prefill negative regime。

## P2

- `Block-INT8` 的 block正式定义为 LMCache object-domain codec quant group：K/V × layer × KV head × fixed token group × head_dim；不要求对齐 vLLM BlockPool page。
- Source Manifest加入 LMCache real FS Serde E2E test 与 TurboQuant layout/size tests。
- FSL2 capacity claim收紧：serialized bytes是 runtime fact；固定 byte-budget capacity是外部 quota 下 derived result，不是 backend 自动 admission。
- filesystem crossover新增 buffered-warm、large-working-set/new-key、verified O_DIRECT regimes。
- exact `.data` byte invariant明确使用 logical file length (`stat.st_size`)，不使用 `du`/allocated blocks。
- 新增 `FILESYSTEM-BENCHMARK-REGIMES.md`。

## Scope

- 未修改 vLLM / LMCache source。
- 未运行正式 benchmark/profiler。
- 当前状态保持 REVIEW_ONLY；Approved Slice = NONE；P2 仍 BLOCKED_BY_PROJECT1_CORE。
