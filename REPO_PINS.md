# Repository Pins

未知字段保持 TO_FREEZE。REFERENCE snapshot 不得自动升级成 implementation baseline。

## Project 1

| Item | Value | Status |
|---|---|---|
| Upstream repository | vllm-project/vllm | PINNED UPSTREAM |
| Upstream version | v0.26.0 | PINNED |
| Upstream base commit | 568afb3a13806beb53bb2e6bd518269357b237c0 | PINNED |
| P1 implementation branch | p1/v2-token-compaction-v026 | FROZEN |
| P1 implementation commit | bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00 | FROZEN FUNCTIONAL CORE |
| GPU | NVIDIA A100 80GB | FIXED HARDWARE TARGET |
| Topology | single GPU; TP=1; PP=1; DP=1 | APPROVED BASELINE |
| Attention | dense full attention | APPROVED BASELINE |
| KV dtype | BF16 | APPROVED BASELINE |
| Runner mode | eager MVP | APPROVED BASELINE |
| Prefix caching | off | APPROVED BASELINE |
| Spec decode | off | APPROVED BASELINE |
| Async scheduling | off | APPROVED BASELINE |
| CUDA Graph | off for MVP | APPROVED BASELINE |
| num_kv_cache_groups | 1 for MVP | APPROVED BASELINE |
| first post-reclaim query | q_len=1 decode | APPROVED BASELINE |
| Python / torch / Triton / CUDA toolkit | TO_FREEZE | M0 GATE |
| Local worktree | worktrees/p1-vllm-reclaim | FROZEN LOCAL TOPOLOGY |
| Model / tokenizer | TO_FREEZE；first supported target must be Qwen3-family standard RoPE decoder | M0 HARD GATE |
| Positional semantic | standard RoPE; no dual-chunk/relative-bias physical-index reconstruction | M0 HARD GATE / TO_VERIFY LOCAL |

Reference-only:

- Tangram audited commit 8e1cbfa5cf82acc67fe1880df0c632f2a6485e35; source-level reference, not cherry-pick baseline.
- Sparse-vLLM audited commit c14643387104c9db923a4fe690b699fe7fbd1790; ownership/test reference.
- newer vLLM audit snapshot 0ecc284790e5403f74b899524ef82ecb69f83cb3; V3 lifetime/offload reference only.
- NVIDIA KVPress; policy/periodic-trigger reference only.

## Project 2

| Item | Value | Status |
|---|---|---|
| vLLM repository | vllm-project/vllm | IMPLEMENTATION BASELINE |
| vLLM version / commit | v0.26.0 / 568afb3a13806beb53bb2e6bd518269357b237c0 | PINNED |
| LMCache repository | LMCache/LMCache | IMPLEMENTATION BASELINE |
| LMCache commit | a7afadebb9248b62b5c533ce2c12297e9d94fc4a | PINNED COMPATIBILITY BASELINE |
| GPU | NVIDIA A100 80GB | FIXED HARDWARE TARGET |
| Topology | single machine / single GPU first | APPROVED BASELINE |
| L2 backend | local filesystem L2 | APPROVED V1 BASELINE |
| torch | 2.11.x ABI family | REQUIRED ABI GATE |
| Python / exact torch build / Triton / CUDA | TO_FREEZE | P2 M0 GATE |
| Worktrees / branches / import paths | TO_FREEZE | P2 M0 GATE |
| Model / tokenizer / LMCache config | TO_FREEZE | P2 M0 GATE |
| Block-INT8 quant group | TO_FREEZE from MemoryObj/chunk layout; physical-block mapping not assumed | P2 M1 DESIGN GATE |
| Serialized byte invariant | fixed-size format + FSL2 only; per-object/per-store | P2 V1 ACCEPTANCE SCOPE |

LMCache later audit snapshot f9addd2e4e074f21f26cbda84c3abd932d32ef33 is REFERENCE ONLY because its build pin moved to torch 2.13. It cannot replace the a7afade compatibility baseline without a new ABI review.

CacheGen is motivation/reference only. Newer vLLM offload snapshot 0ecc284790e5403f74b899524ef82ecb69f83cb3 is V3 architecture/lifetime reference only.

## ABI Gate

P2 must prove all of the following before functionality work:

- torch runtime exactly identified as 2.11.x compatible build;
- vLLM native extensions import from the intended worktree;
- LMCache native extensions import without undefined symbol;
- CUDA/Triton backend is actually enabled, not silent fallback;
- package metadata, source HEAD and import paths agree.

Failure means BLOCKED_ENV. Do not work around by silently changing pinned source.

