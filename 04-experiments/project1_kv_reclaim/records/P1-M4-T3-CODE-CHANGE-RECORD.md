# P1-M4-T3 代码变更记录

## Slice Identity

| 项目 | 内容 |
|---|---|
| Task | P1-M4-T3 |
| Scope | Triton Paged-KV Writeback + Full Compaction Primitive |
| Branch | `p1/v2-token-compaction-v026` |
| HEAD before/after | `064f4efe082bc23087043381a4a401c170997f5c` / unchanged |
| Date | 2026-09-08 |
| Commit | 未创建；working-tree changes only |

## Changed Files Overview

| File | Change Type | Purpose |
|---|---|---|
| `vllm/v1/worker/gpu/kv_compaction.py` | MODIFIED | 新增 NHD 1D、stride-safe 2D writeback 和完整 2D compaction primitive。 |
| `tests/v1/worker/test_gpu_kv_compaction.py` | MODIFIED | 新增 writeback、tail、detached-page 和 full-oracle CUDA tests。 |
| `04-experiments/project1_kv_reclaim/records/P1-M4-T3-CODE-CHANGE-RECORD.md` | ADDED | 本 Slice 的修改追踪和验证证据。 |

## Source-Level Modification Ledger

### CHG-01

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `_writeback_paged_kv_nhd_1d_kernel`

Change Type: ADDED

Before: 只有 gather kernel；没有 scratch 到 paged KV 的 writeback。

After: 新增 `@triton.jit`、`grid=(dst_capacity,)` kernel；每个 program 由 `dst_member=tl.program_id(0)` 负责一个完整 `[H,C]` token，`dst_member >= K` 时 masked load zero。

Reason: 实现 NHD-only one-token/one-program destination writeback 和 deterministic tail zero。

Frozen Design Mapping: Variant A Writeback、`M4-T3-GRID-1D`、`M4-T3-TAIL-ZERO`。

Behavioral Effect: 只访问 `block_ids[:new_num_blocks]`，不读取 scratch 越界，不触碰 detached trailing blocks。

Out of Scope: 不做 free、BlockTable trim 或 runtime metadata 更新。

Verification: `test_triton_writeback_nhd_1d_shape_coverage`、`test_triton_writeback_nhd_1d_rejects_hnd`。

### CHG-02

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `writeback_paged_kv_triton_nhd_1d`

Change Type: ADDED

Before: Symbol did not exist。

After: 新增 wrapper，复用 writeback validation，严格复用 M4-T2 NHD stride contract，并按 `ceil(K / block_size)` launch。

Reason: 提供 Variant A public API，拒绝 HND silent fallback。

Frozen Design Mapping: `M4-T3-NHD-VALIDATION`、`M4-T3-VARIANT-A`。

Behavioral Effect: 原地修改传入 `kv_cache`，返回 `(new_effective_kv_len, new_num_blocks)`。

Out of Scope: `num_warps=4` 只是 baseline implementation parameter，不是性能结论。

Verification: A100 `H=2,C=16`、`H=4,C=32`、`H=8,C=256` 全部通过。

### CHG-03

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `_writeback_paged_kv_2d_kernel`

Change Type: ADDED

Before: 只有 stride-safe gather 2D kernel。

After: 新增 `@triton.jit`、`grid=(dst_capacity,H)` kernel；`(dst_member,h)` program 搬运 `[C]`，destination 使用 `stride_b/stride_h/stride_n/stride_c`。

Reason: 支持 NHD/HND logical `[B,H,N,C]` view 的通用 writeback。

Frozen Design Mapping: Variant B Writeback、`M4-T3-GRID-2D`、`M4-T3-STRIDE-SAFE`。

Behavioral Effect: retained members 写入 regular destination member `j`；tail program 通过 masked load 写 zero。

Out of Scope: 不使用 `keep_member_indices` 做 destination addressing，不做直接 in-place arbitrary gather。

Verification: `test_triton_writeback_2d_matches_reference`、full oracle tests，HND/NHD 均通过。

### CHG-04

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `writeback_paged_kv_triton_2d`

Change Type: ADDED

Before: Symbol did not exist。

After: 新增 stride-safe public wrapper，输出 metadata `(K, ceil(K/block_size))`，只要求 `len(block_ids) >= new_num_blocks`。

Reason: 提供 Variant B writeback API，并保留 old row trailing pages。

Frozen Design Mapping: `M4-T3-WRITEBACK-API`、`M4-T3-DETACHED-PAGE-SAFETY`。

Behavioral Effect: 只修改 destination physical prefix，detached pages untouched。

Out of Scope: 不修改 canonical ownership 或释放 trailing blocks。

Verification: full A100 suite `52 passed`。

### CHG-05

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `_validate_writeback_inputs`

Change Type: ADDED

Before: 没有 writeback-specific validation。

After: 新增 rank、shape、dtype、contiguous scratch、CUDA/device、一致 block ID、range、destination capacity validation。

Reason: writeback source 已是 contract-defined contiguous `[K,H,C]` scratch；非法输入必须 fail closed。

Frozen Design Mapping: `M4-T3-VALIDATION`。

Behavioral Effect: validation 在 kernel launch 前完成，不隐藏输入错误。

Out of Scope: 不修改 frozen `_validate_inputs()` semantic gather contract。

Verification: `ruff check`、`py_compile`、现有 invalid-input tests。

### CHG-06

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `compact_paged_kv_triton_2d`

Change Type: ADDED

Before: 只有独立 gather/reference compaction；没有 Triton full data-plane primitive。

After: 新增 `gather_paged_kv_triton_2d()` → independent scratch → `writeback_paged_kv_triton_2d()`，返回 `(new_effective_kv_len, new_num_blocks)`。

Reason: 建立 M4-T1 PyTorch compact reference 与 Triton gather+writeback 的 end-to-end correctness oracle。

Frozen Design Mapping: `M4-T3-FULL-PRIMITIVE`、`M4-T3-SCRATCH-BOUNDARY`。

Behavioral Effect: physical payload 被 compact 到旧 row prefix；无 Worker/Scheduler/BlockPool 副作用。

Out of Scope: 未新增 Triton compact Variant A、未做 runtime integration、free 或 ownership reconcile。

Verification: canonical、overlap-style、HND/NHD full-oracle tests 全部通过。

### CHG-07

File: `tests/v1/worker/test_gpu_kv_compaction.py`

Symbols: `test_triton_writeback_*`、`test_compact_paged_kv_triton_2d_full_oracle`、tail/detached tests

Change Type: ADDED

Before: 只有 M4-T1/M4-T2 gather tests。

After: 新增 16 个 writeback/full compaction tests，覆盖 K=5 tail zero、K=4 exact boundary、K=1 heavy shrink、`[0,1,4,5,19,20]` overlap-style keep、non-monotonic `[7,2,11]`、HND/NHD、A100 realistic `H=8,C=256` 和 detached sentinel。

Reason: 证明 writeback 和 full primitive 与 `compact_paged_kv_reference` bitwise 相等。

Frozen Design Mapping: `M4-T3-TEST`、`M4-T3-TAIL-ZERO`、`M4-T3-DETACHED-SAFETY`、`M4-T3-ORACLE`。

Behavioral Effect: 明确验证 `torch.equal(actual, expected)`，而非 tolerance comparison。

Out of Scope: 不做 benchmark/performance claim。

Verification: A100 全套 `52 passed, 0 skipped`。

## Test Evidence Mapping

| Test | Validates | What It Proves |
|---|---|---|
| `test_triton_writeback_2d_matches_reference` | CHG-03, CHG-04 | Variant B HND/NHD writeback 与 PyTorch compact reference 完全一致。 |
| `test_triton_writeback_nhd_1d_shape_coverage` | CHG-01, CHG-02 | Variant A 在 `HC=32/128/2048` 真实 A100 上通过。 |
| `test_triton_writeback_nhd_1d_rejects_hnd` | CHG-02, CHG-05 | NHD-only wrapper 对 HND fail closed。 |
| `test_compact_paged_kv_triton_2d_full_oracle` | CHG-06, CHG-07 | Gather + writeback entire cache 与 reference bitwise equal。 |
| `test_writeback_tail_zero_and_detached_block_unchanged` | CHG-01, CHG-03, CHG-07 | K=5 tail zero，B11 detached sentinel unchanged。 |
| `test_writeback_exact_boundary_does_not_touch_next_page` | CHG-03, CHG-07 | K=4 不启动/修改下一 physical page。 |

## Verification

Environment:

```text
GPU: NVIDIA A100 80GB PCIe, Compute Capability 8.0
PyTorch: 2.11.0+cu129
Triton: 3.6.0
CUDA_VISIBLE_DEVICES=1
```

Commands and results:

```text
CUDA_VISIBLE_DEVICES=1 .../.venv/bin/python -m pytest -vv -s tests/v1/worker/test_gpu_kv_compaction.py
52 passed, 15 warnings

CUDA_VISIBLE_DEVICES=1 .../.venv/bin/python -m pytest -vv -s tests/v1/worker/test_gpu_kv_compaction.py -k 'writeback or compact'
52 passed, 15 warnings

ruff check vllm/v1/worker/gpu/kv_compaction.py tests/v1/worker/test_gpu_kv_compaction.py
PASS

python -m py_compile ...
PASS

git diff --check
PASS
```

Warnings are existing vLLM version metadata and TorchScript deprecation warnings; no test failure occurred.

## Diff Audit

`git status --short` lists only the two existing working-tree source/test files; this record is tracked in the workspace repository.

Because production/test files remain untracked in the implementation worktree, `git diff --stat` and `git diff --numstat` emit no rows. File-level audit reports the M4-T3 additions above; no unrelated production file was modified.

No Triton autotune, benchmark, runtime integration, Scheduler/BlockPool change, ownership mutation, free, or scatter beyond the approved writeback primitive was introduced.

## Gate Interpretation

M4-T3 correctness closure is established on real A100 for the implemented writeback/full 2D primitive and both layout families. Variant A NHD writeback realistic `H=8,C=256` passed. The Slice remains data-plane only and stops before M5 runtime integration.
