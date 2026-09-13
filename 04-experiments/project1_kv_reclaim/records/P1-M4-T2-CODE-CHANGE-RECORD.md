# P1-M4-T2 代码变更记录

## Slice Identity

| 项目 | 内容 |
|---|---|
| Task | P1-M4-T2 |
| Scope | Triton Gather Variants |
| Branch | `p1/v2-token-compaction-v026` |
| HEAD before | `064f4efe082bc23087043381a4a401c170997f5c` |
| HEAD after | unchanged（未 commit，working-tree changes only） |
| Date | 2026-09-08 |

## Changed Files Overview

| File | Change Type | Purpose |
|---|---|---|
| `vllm/v1/worker/gpu/kv_compaction.py` | MODIFIED | 新增两个独立 Triton gather kernels 及 CUDA/stride validation wrappers。 |
| `tests/v1/worker/test_gpu_kv_compaction.py` | MODIFIED | 新增 CUDA Triton correctness、layout、shape 和 fail-closed 测试。 |
| `04-experiments/project1_kv_reclaim/records/P1-M4-T2-CODE-CHANGE-RECORD.md` | ADDED | 记录本 Slice 的代码追踪、设计映射和实际验证限制。 |

## Source-Level Modification Ledger

### CHG-01

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `_gather_paged_kv_nhd_1d_kernel`

Change Type: ADDED

Before: M4-T1 只有 PyTorch reference，没有 Triton kernel。

After: 新增 `@triton.jit` kernel，`grid=(K,)`，`j=tl.program_id(0)`；每个 program 计算 `member -> page_idx/token_offset -> physical_block`，以 contiguous `[H*C]` offsets 搬运一个完整 token payload。

Reason: 验证 NHD physical layout 下 one-token/one-program decomposition。

Design Mapping: `M4-T2-VARIANT-A`、`M4-T2-GRID-1D`。

Behavioral Effect: 只写入独立 contiguous scratch `[K,H,C]`，不修改 paged source。

Out of Scope: 不支持 HND，不做 scatter/writeback，不做 autotune 或性能选择。

Verification: `test_triton_gather_nhd_semantics`、`test_triton_nhd_1d_shape_coverage`（当前环境 CUDA unavailable，skip）。

### CHG-02

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `gather_paged_kv_triton_nhd_1d`

Change Type: ADDED

Before: Symbol did not exist。

After: 新增 Variant A wrapper；复用 `_validate_inputs()`，要求 CUDA 同设备，并严格检查 NHD-backed strides：`stride_c=1`、`stride_h=C`、`stride_n=H*C`、`stride_b=N*H*C`；不满足即 `raise`。

Reason: 禁止依据配置字符串或 silent fallback 假设 NHD。

Design Mapping: `M4-T2-NHD-VALIDATION`、`M4-T2-VARIANT-A`。

Behavioral Effect: Variant A 使用 `BLOCK_HC=next_power_of_2(H*C)`，输出 BF16 scratch；HND 输入 fail closed。

Out of Scope: 未宣称 `BLOCK_HC` 最优，未改变 M4-T1 reference。

Verification: `test_triton_nhd_1d_rejects_hnd`、`py_compile`、`ruff check`。

### CHG-03

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `_gather_paged_kv_2d_kernel`

Change Type: ADDED

Before: M4-T1 只有 PyTorch reference，没有 Triton kernel。

After: 新增 `@triton.jit` kernel，`grid=(K,H)`，`j=tl.program_id(0)`、`h=tl.program_id(1)`；每个 program 搬运一个 token/head 的 `[C]`，地址使用 `stride_b/stride_h/stride_n/stride_c`。

Reason: 提供 NHD/HND 通用 stride-safe decomposition。

Design Mapping: `M4-T2-VARIANT-B`、`M4-T2-GRID-2D`、`M4-T2-STRIDE-SAFE`。

Behavioral Effect: 只写入 contiguous scratch `[K,H,C]`，不假设 source logical view contiguous。

Out of Scope: 不写回 paged destination，不融合多 request/layer。

Verification: `test_triton_gather_nhd_semantics`、`test_triton_2d_supports_hnd_and_nhd`（当前环境 skip）。

### CHG-04

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `_validate_triton_inputs`

Change Type: ADDED

Before: 没有 Triton-specific device validation。

After: 复用 frozen semantic validation，并增加 all-input CUDA、same-device 检查。

Reason: 保持 M4-T1 validation 单一来源，同时让 Triton backend fail closed。

Design Mapping: `M4-T2-VALIDATION`。

Behavioral Effect: CPU tensor 或跨 device 输入在 launch 前抛出 `ValueError`。

Out of Scope: 不改变 `_validate_inputs()` 的 semantic contract。

Verification: CPU M4-T1 suite、`py_compile`、`ruff check`。

### CHG-05

File: `tests/v1/worker/test_gpu_kv_compaction.py`

Symbols: `_cuda_cache()`、`test_triton_*`

Change Type: ADDED

Before: 只有 M4-T1 PyTorch tests。

After: 新增 deterministic BF16 CUDA fixture；覆盖 Keep All、Interior Drop、Cross Source Page、Heavy Shrink、Exact Boundary、non-monotonic physical row、overlap keep `[0,1,4,5,19,20]`，并覆盖 HND/NHD、`H=2/4/8`、`C=16/32/128/256`、Variant A HND rejection。

Reason: 直接比较两个 Triton decomposition 与 M4-T1 oracle 的 `torch.equal`。

Design Mapping: `M4-T2-TEST`、`M4-T2-NHD-VALIDATION`、`M4-T2-SHAPE-COVERAGE`。

Behavioral Effect: CUDA 可用时真实 launch kernels；CUDA 不可用时测试显式 skip，不伪造 PASS。

Out of Scope: 不做 benchmark、scatter、runtime integration。

Verification: `14 passed, 22 skipped`；22 个 skip 均为 CUDA-required tests。

## Test Evidence Mapping

| Test | Validates Change IDs | What It Proves |
|---|---|---|
| `test_triton_gather_nhd_semantics` | CHG-01, CHG-02, CHG-03 | 两个 variant 在多种 member mapping 下应与 PyTorch oracle bitwise 相等。 |
| `test_triton_2d_supports_hnd_and_nhd` | CHG-03, CHG-05 | Variant B 覆盖真实 HND/NHD-backed logical view。 |
| `test_triton_nhd_1d_shape_coverage` | CHG-01, CHG-02, CHG-05 | Variant A 的 `HC` shape coverage，包括 `HC=2048`。 |
| `test_triton_nhd_1d_rejects_hnd` | CHG-02, CHG-04, CHG-05 | Variant A 对 HND 输入 fail closed，不 silent fallback。 |
| M4-T1 CPU tests | CHG-04 | frozen reference semantics 未回归。 |

## Diff Classification

Files added: 1 record artifact（相对 M4-T1 working tree，production/test 文件为 MODIFIED）

Files modified: 2

Files removed: 0

Production lines added in this Slice: kernel/wrapper block at `kv_compaction.py` lines 914-1065；由于该文件在当前 worktree 尚未被 Git tracked，`git diff --numstat` 不展示 untracked baseline diff。

Test lines added in this Slice: CUDA fixture/tests appended to `test_gpu_kv_compaction.py`；同样因 untracked baseline，`git diff --numstat` 不展示。

No unrelated production code was modified. No scatter, runtime integration, scheduler, BlockPool, or performance tuning was introduced.

## Verification and Environment

- Branch: `p1/v2-token-compaction-v026`
- HEAD: `064f4efe082bc23087043381a4a401c170997f5c`，未创建 commit。
- Python: `/home/qcrs/learning/llm-kv-lab/.venv/bin/python`，Torch `2.11.0+cu129`，Triton `3.6.0`。
- `ruff check vllm/v1/worker/gpu/kv_compaction.py tests/v1/worker/test_gpu_kv_compaction.py`: PASS。
- `py_compile`: PASS。
- `git diff --check`: PASS。
- Targeted pytest: `14 passed, 22 skipped`。
- `torch.cuda.is_available() == False`，`nvidia-smi` 无法与 NVIDIA driver 通信；因此 Triton kernels 本轮没有真实 compile/launch evidence。

## Gate Interpretation

静态实现、API、stride contract、测试矩阵和 fail-closed 逻辑已完成；由于 GPU/driver 不可用，不能将 G9/G10 的 Triton runtime correctness 或 Variant A realistic `HC` feasibility 标记为已证明。Slice 状态为 `BLOCKED`，等待可用 CUDA 环境复跑同一 targeted suite 后再进行 Web review。
