# P1 Repository / Native / Environment Topology Report

Date: 2026-08-24 Asia/Shanghai

## 1. Workspace Git (Repo A)

- Path: `/home/qcrs/learning/llm-kv-lab`
- Git initialized: yes
- Branch: `main`
- Owner: workspace docs, project control plane, experiments, reports and handoffs
- `.gitignore`: excludes `.venvs/`, `.cache/`, `third_party/`, `worktrees/`, native
  binaries, model weights and profiler binaries while keeping active docs and
  experiment namespaces trackable.
- No commit was created.

## 2. vLLM Git (Repo B)

- Path: `/home/qcrs/learning/llm-kv-lab/third_party/vllm`
- HEAD: `568afb3a13806beb53bb2e6bd518269357b237c0`
- Branch: `study-vllm-0.26`
- Role: study/reference tree and validated native artifact seed
- Status: preserves pre-existing user dirty changes; no reset, clean, checkout,
  rebase or source modification was performed.

## 3. Preserved Study Tree

The study tree remains at `third_party/vllm`. Its dirty files are unchanged. The
old registered worktree path under the archived QCache project was stale and was
removed from Git's worktree registry only; archived files remain under
`archive/legacy-projects/project1-qcache-vllm/worktrees/`.

## 4. Removed Obsolete Worktrees

Removed Git registration:

```text
/home/qcrs/learning/llm-kv-lab/01-projects/project1-qcache-vllm/worktrees/vllm-v026-qcache
branch: project1/m0-qcache-v026
HEAD: 568afb3a13806beb53bb2e6bd518269357b237c0
reason: superseded/archived QCache implementation worktree
```

The directory content was already in `archive/legacy-projects`; only Repo B's
stale registration was removed.

## 5. P1 Implementation Worktree

- Path: `/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Owner: Repo B worktree, not Repo A
- Branch: `p1/physical-kv-reclaim-v026`
- HEAD: `568afb3a13806beb53bb2e6bd518269357b237c0`
- Initial tracked-source status: clean
- `.so`, DeepGEMM artifacts and `__pycache__` are ignored local artifacts.

## 6. Existing Python Environment

- Path: `/home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310`
- Python: `3.10.20`
- torch: `2.11.0+cu129`
- PyTorch CUDA runtime: `12.9`
- Triton: `3.6.0`
- CUDA toolkit: `12.1.66`, explicit `CUDA_HOME=/usr/local/cuda-12.1`
- `third_party/lmcache`: absent; no P2 environment/worktree was created.
- `nvidia-smi` was unavailable in this session, so GPU/driver fields remain
  `TO_VERIFY` in the P1 environment snapshot.

## 7. Precompiled Native Mode

```text
Python source:       editable P1 worktree
Native C++/CUDA:      copied precompiled .so, no rebuild
Triton:               runtime JIT/cache
```

The editable finder and `direct_url.json` were minimally updated from the archived
QCache worktree to `/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`.
No `pip install -e`, dependency resolution or build command was run.

## 8. Native Artifact Inventory

Seed: `third_party/vllm/vllm/`  
Target: `worktrees/p1-vllm-reclaim/vllm/`  
Mode: copy, same relative paths, not committed.

| Artifact | Bytes | Purpose |
|---|---:|---|
| `_C_stable_libtorch.abi3.so` | 495514296 | stable custom-op registration |
| `_moe_C_stable_libtorch.abi3.so` | 355799416 | MoE operators |
| `_qutlass_C.abi3.so` | 2723696 | qutlass extension |
| `_flashmla_C.abi3.so` | 6615616 | FlashMLA extension |
| `_flashmla_extension_C.abi3.so` | 785168 | FlashMLA extension |
| `_rust_tool_parser.abi3.so` | 994328 | Rust parser |
| `cumem_allocator.abi3.so` | 27264 | CUDA allocator |
| `fs_io_C.abi3.so` | 17176 | filesystem I/O |
| `spinloop.abi3.so` | 264808 | spinloop |
| `vllm_flash_attn/_vllm_fa2_C.abi3.so` | 418372472 | A100/SM80 FA2 |
| `vllm_flash_attn/_vllm_fa3_C.abi3.so` | 148001008 | optional FA3 availability |

Five Python-version-specific DeepGEMM binaries were copied as part of the complete
`.so` inventory; the Python 3.10 runtime selects the `cp310` artifact as needed.
Full provenance is in `01-projects/project1-physical-kv-reclamation-vllm/docs/process/runtime/NATIVE-ARTIFACTS.md`.

## 9. Editable Binding Before / After

Before:

```text
01-projects/project1-qcache-vllm/worktrees/vllm-v026-qcache/vllm
```

After:

```text
worktrees/p1-vllm-reclaim/vllm
```

The update was made directly in the existing editable finder and `direct_url.json`
after saving before-state copies. It is reversible metadata-only path switching.

## 10. Python Import Identity

```text
vllm.__file__ = worktrees/p1-vllm-reclaim/vllm/__init__.py
```

The activation smoke also prints this identity and the native module paths.
Import emits a non-fatal warning that `_version` is unavailable in this source
checkout; the package import itself succeeds.

## 11. Native Import Identity

All of the following imported from the P1 worktree:

```text
vllm._C_stable_libtorch
vllm._moe_C_stable_libtorch
vllm._qutlass_C
vllm.vllm_flash_attn._vllm_fa2_C
vllm.vllm_flash_attn._vllm_fa3_C
```

## 12. Dispatcher Smoke

After importing `_C_stable_libtorch`, 13 `_C_cache_ops::*` dispatcher names were
visible in this process, including:

```text
_C_cache_ops::reshape_and_cache_flash
```

This is only import/operator-registration introspection. No model request or formal
benchmark was executed.

## 13. FA2 Availability

`vllm.vllm_flash_attn._vllm_fa2_C` imported successfully from the P1 worktree. This
proves binary availability only; it does not claim runtime dispatch selected FA2.

## 14. `.so` Git Tracking

Precompiled `.so` files are ignored and **NOT COMMITTED** to Repo A or Repo B. They
are local runtime artifacts. The P1 worktree tracked-source status is clean.

## 15. Native Rebuild Policy

- Python source: restart process.
- Triton JIT source: regenerate JIT/cache as needed.
- C++/CUDA/custom-op/FA2 source: `NATIVE_REBUILD_REQUIRED`; stop for Web/User review.

## 16. Governance

P1 remains `REVIEW_ONLY`, current Parent Task remains `P1-M0-T1`, and Approved Slice
remains `NONE`. P2 remains `BLOCKED_BY_PROJECT1_CORE`. The topology task did not
change architecture, support matrix, Gate status or project scope.

## 17. Runtime Source / Formal Benchmark

```text
Runtime source changes: NONE
Reclaim implementation: NONE
Offload implementation: NONE
Formal benchmark: NONE
NSYS/NCU capture: NONE
Native rebuild: NONE
```

## 18. Next Allowed Action

```text
WEB_REVIEW_P1_M0_T1
```
