# P1 Native Artifact Provenance

## Status

```yaml
mode: PRECOMPILED_REUSE
source_seed: /home/qcrs/learning/llm-kv-lab/third_party/vllm
target_worktree: /home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim
source_commit: 568afb3a13806beb53bb2e6bd518269357b237c0
git_tracking: NOT_COMMITTED
build_performed: false
```

The target P1 worktree is a clean Git checkout of the pinned vLLM commit. Existing
validated native artifacts were copied from the study/reference tree into the same
relative package paths. They are local execution artifacts, ignored by Git, and are
not part of the P1 source diff.

## Artifact Inventory

| Relative path | Seed | Target | Bytes | Purpose |
|---|---|---|---:|---|
| `vllm/_C_stable_libtorch.abi3.so` | `third_party/vllm` | `worktrees/p1-vllm-reclaim` | 495514296 | stable custom-op registration |
| `vllm/_moe_C_stable_libtorch.abi3.so` | `third_party/vllm` | same relative path | 355799416 | MoE native operators |
| `vllm/_qutlass_C.abi3.so` | `third_party/vllm` | same relative path | 2723696 | qutlass native extension |
| `vllm/_flashmla_C.abi3.so` | `third_party/vllm` | same relative path | 6615616 | FlashMLA native extension |
| `vllm/_flashmla_extension_C.abi3.so` | `third_party/vllm` | same relative path | 785168 | FlashMLA extension |
| `vllm/_rust_tool_parser.abi3.so` | `third_party/vllm` | same relative path | 994328 | Rust parser extension |
| `vllm/cumem_allocator.abi3.so` | `third_party/vllm` | same relative path | 27264 | CUDA memory allocator |
| `vllm/fs_io_C.abi3.so` | `third_party/vllm` | same relative path | 17176 | filesystem I/O extension |
| `vllm/spinloop.abi3.so` | `third_party/vllm` | same relative path | 264808 | spinloop extension |
| `vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so` | `third_party/vllm` | same relative path | 418372472 | A100/SM80 FlashAttention 2 |
| `vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so` | `third_party/vllm` | same relative path | 148001008 | optional FlashAttention 3 availability |

The package also contains five Python-version-specific DeepGEMM `.so` files under
`vllm/third_party/deep_gemm/`; they were copied as part of the complete package-tree
inventory. The active Python 3.10 runtime selects the `cp310` artifact when needed.

## Verified Identity

With `.venvs/vllm-v026-torch211-cu129-py310/bin/python`:

- `vllm.__file__` resolves under `worktrees/p1-vllm-reclaim/vllm/`.
- stable_libtorch, MoE, qutlass, FA2 and FA3 modules resolve under the same P1
  worktree namespace.
- importing stable_libtorch registers `_C_cache_ops::reshape_and_cache_flash` and
  related cache operators. This is an import/registration smoke, not a model or
  benchmark run.

## Rebuild Policy

- Python source changes: restart the process; no native rebuild by default.
- Triton `@triton.jit` changes: allow Triton JIT/cache regeneration; no vLLM native
  rebuild by default.
- C++/CUDA, `csrc/`, stable_libtorch custom-op or FA2 source changes:
  `NATIVE_REBUILD_REQUIRED`. Stop for Web/User review; Codex must not start a broad
  native build automatically.

The study tree remains read-only for P1 implementation purposes. Do not symlink P1
artifacts back to it, force-add `.so` files, or treat binary reuse as proof that a
future native source change is included.
