# P1-M4-T1 代码变更记录

## Slice Identity

| 项目 | 内容 |
|---|---|
| Task | P1-M4-T1 |
| Branch | `p1/v2-token-compaction-v026` |
| HEAD before | `064f4efe082bc23087043381a4a401c170997f5c` |
| HEAD after | unchanged（未 commit，working-tree changes only） |
| Date | 2026-09-08 |
| Scope | PyTorch FA2 Paged-KV Token Compaction Reference |

## Changed Files Overview

| File | Change Type | Purpose |
|---|---|---|
| `vllm/v1/worker/gpu/kv_compaction.py` | ADDED | 增加单 request、单 layer 的 PyTorch gather/compaction correctness oracle。 |
| `tests/v1/worker/test_gpu_kv_compaction.py` | ADDED | 覆盖 T1-T10 reference semantics、stride、overlap 与 invalid input。 |
| `04-experiments/project1_kv_reclaim/records/P1-M4-T1-CODE-CHANGE-RECORD.md` | ADDED | 记录本 Slice 的 source-level modification、设计映射和测试证据。 |

## Source-Level Modification Ledger

### CHG-01

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `_validate_inputs()`

Change Type: ADDED

Before: Symbol did not exist；reference compaction 没有实现。

After: 新增对 rank、integral index dtype、正的 `source_effective_kv_len`、精确 source page 数、block ID 范围/唯一性、非空且严格递增 `keep_member_indices` 的 fail-closed 校验。

Reason: 保护 member domain 和 physical page contract，禁止 sort/dedup/clip/repair。

Design Mapping: `M4-T1-VALIDATION`、V2 invariant `keep_member_indices` 是当前 physical member sequence 的显式输入。

Behavioral Effect: 非法输入立即 `raise`，不产生部分输出。

Out of Scope: 不推导 retention policy，不修改 runtime state 或 ownership。

Verification: `test_invalid_inputs`。

### CHG-02

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `gather_paged_kv_reference()`

Change Type: ADDED

Before: Symbol did not exist。

After: 新增 `index_select` active pages、`permute(0, 2, 1, 3).reshape(-1, H, C)` member view、effective-length 截断和 `index_select(...).clone()` scratch gather。

Reason: 提供后续 Triton gather/scatter 的唯一 PyTorch correctness oracle，并避免把 raw storage 当作 contiguous token payload。

Design Mapping: `M4-T1-GATHER`、`M4-T1-SCRATCH`、`M4-T1-NHD-HND`。

Behavioral Effect: 输出 `[K, H, C]` 独立 scratch，保留输入 member 顺序；普通 PyTorch indexing 尊重 HND/NHD-backed stride。

Out of Scope: 无 Triton/CUDA kernel、无 runtime integration。

Verification: `test_keep_all`、`test_interior_drop_and_cross_source_page`、`test_hnd_and_nhd_strides`、`test_overlap_hazard_uses_independent_scratch`。

### CHG-03

File: `vllm/v1/worker/gpu/kv_compaction.py`

Symbol: `compact_paged_kv_reference()`

Change Type: ADDED

Before: Symbol did not exist。

After: 新增 full reference：先 gather 到 scratch，clone `kv_cache`，构造 zero-padded dense destination members，按 `block_ids[:new_num_blocks]` 写回，并返回 `expected_kv_cache`、`new_effective_kv_len`、`new_num_blocks`。

Reason: 固化 prefix destination、partial-tail zero 和输入不可变的 reference semantics。

Design Mapping: `M4-T1-PREFIX-WRITEBACK`、`M4-T1-TAIL-ZERO`、`M4-T1-FUNCTIONAL-REFERENCE`。

Behavioral Effect: `E_new == len(keep_member_indices)`；`new_num_blocks == ceil(K / block_size)`；只写旧 active row 的 destination prefix，detached trailing pages 保持原值。

Out of Scope: 不更新 BlockTable、Scheduler、BlockPool 或 runtime `effective_kv_len`。

Verification: `test_cross_destination_page_and_partial_tail`、`test_exact_boundary_and_heavy_shrink`、`test_overlap_hazard_uses_independent_scratch`。

### CHG-04

File: `tests/v1/worker/test_gpu_kv_compaction.py`

Symbols: `_cache()`、`_members()`、`test_*`

Change Type: ADDED

Before: Symbol did not exist；目标测试文件不存在。

After: 新增 deterministic BF16 payload fixture 与 14 个 targeted tests，覆盖 T1 Keep All、T2 Interior Drop、T3 Cross Source Page、T4 Cross Destination Page、T5 Partial Destination Tail、T6 Exact Block Boundary、T7 Heavy Shrink、T8 overlap hazard、T9 HND/NHD、T10 invalid inputs，并检查 input immutability。

Reason: 将 approved reference contract 转化为可执行 regression oracle。

Design Mapping: `M4-T1-TEST`，其中 overlap regression 显式映射 `M4-T1-SCRATCH`。

Behavioral Effect: 测试只验证 one-layer/one-request PyTorch reference，不引入 runtime scope。

Out of Scope: 不执行 benchmark、profiler、Triton、CUDA 或 engine integration。

Verification: `/home/qcrs/learning/llm-kv-lab/.venv/bin/python -m pytest -q tests/v1/worker/test_gpu_kv_compaction.py` → `14 passed`。

## Test Evidence Mapping

| Test | Validates Change IDs | What It Proves |
|---|---|---|
| `test_keep_all` | CHG-02, CHG-03 | 保留全部 source members 时 payload 与长度保持一致。 |
| `test_interior_drop_and_cross_source_page` | CHG-02 | source page 边界跨越时按 member index 正确 gather。 |
| `test_cross_destination_page_and_partial_tail` | CHG-03 | destination 跨页写回、`K=5` 时最终 page tail zero。 |
| `test_exact_boundary_and_heavy_shrink` | CHG-01, CHG-03 | exact block boundary 与 heavy shrink 的 page count 正确。 |
| `test_overlap_hazard_uses_independent_scratch` | CHG-02, CHG-03 | `[0,1,4,5,19,20]` 在 `E_source > 20` 下经 independent scratch，且 input 不变。 |
| `test_hnd_and_nhd_strides` | CHG-02 | HND-like 与 NHD-like stride 均遵守 logical `[B,H,N,C]` 语义。 |
| `test_invalid_inputs` | CHG-01 | 非法 rank、dtype、长度、重复/乱序/越界输入 fail closed。 |

## Diff Classification

Files added: 3

Files modified: 0

Files removed: 0

Production lines added: 113 (`kv_compaction.py`，untracked file，因此 `git diff --numstat` 不展示）

Production lines removed: 0

Test lines added: 100 (`test_gpu_kv_compaction.py`，untracked file，因此 `git diff --numstat` 不展示）

Test lines removed: 0

Record lines added: this record file

No unrelated production code was modified. No Triton or runtime integration was introduced.

## Verification Summary

- `python -m pytest -q tests/v1/worker/test_gpu_kv_compaction.py`: `14 passed`，19 warnings（环境中 vLLM version metadata 与 NVML warning，不影响测试结果）。
- `git diff --check`: PASS。
- `git status --short`: 仅列出本记录及两个批准的新增 source/test 文件。
- 未创建 commit；HEAD after 保持 `064f4efe082bc23087043381a4a401c170997f5c`。
