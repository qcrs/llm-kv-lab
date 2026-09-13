# P1 Environment Snapshot

> Unknown values remain TO_FREEZE. Do not infer from old llm-kv-lab.

| Field | Value | Status |
|---|---|---|
| GPU target | NVIDIA A100 80GB | BASELINE |
| Host/GPU count/PCIe/NUMA | TO_VERIFY (nvidia-smi unavailable in this session) | M0 |
| Driver | TO_VERIFY (nvidia-smi unavailable in this session) | M0 |
| Python | 3.10.20 | OBSERVED |
| torch | 2.11.0+cu129 | OBSERVED |
| torch CUDA runtime | 12.9 | OBSERVED |
| CUDA toolkit | 12.1.66 (`/usr/local/cuda-12.1`) | OBSERVED |
| Triton | 3.6.0 | OBSERVED |
| Nsight Systems/NCU | TO_FREEZE | M0/profile |
| vLLM | v0.26.0 / 568afb3... | PINNED |
| model/tokenizer | TO_FREEZE | M0 |
| backend | TO_FREEZE | M0 |
| TP/PP/DP | 1/1/1 | BASELINE |
| KV | BF16 | BASELINE |
| eager/APC/spec/async/graph | true/off/off/off/off | BASELINE |
| block size/max len/batch | TO_FREEZE | M0 |
| Python source | P1 worktree editable finder | OBSERVED |
| Native mode | PRECOMPILED_REUSE; no rebuild performed | OBSERVED |
| Native artifact seed | `third_party/vllm/vllm` | OBSERVED |

M0 需保存 exact commands/output/import path/native extensions。环境激活不等于 source identity。
Import smoke loaded stable_libtorch, MoE, qutlass, FA2 and FA3 from the P1
worktree and observed registered `_C_cache_ops::reshape_and_cache_flash`.
