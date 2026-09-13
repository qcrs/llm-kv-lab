# PROJECT_STATE — Project 2

> Authoritative detailed state。当前 runtime blocked；derived context不得解锁。

## Repository / Worktree

~~~yaml
project: Compressed KV Offload for vLLM + LMCache
project_root: TO_FREEZE
vllm_worktree: TO_FREEZE
lmcache_worktree: TO_FREEZE
vllm_branch_head: TO_FREEZE / 568afb3a13806beb53bb2e6bd518269357b237c0
lmcache_branch_head: TO_FREEZE / a7afadebb9248b62b5c533ce2c12297e9d94fc4a
last_good_commits: TO_FREEZE
working_tree_status: TO_FREEZE
import_paths: TO_FREEZE
~~~

## Environment

~~~yaml
date_timezone: 2026-08-24 Asia/Taipei
gpu_target: NVIDIA A100 80GB
python: TO_FREEZE
torch_required: 2.11.x ABI family
torch_exact: TO_FREEZE
cuda_runtime_toolkit_driver: TO_FREEZE
triton: TO_FREEZE
vllm: v0.26.0 / 568afb3a13806beb53bb2e6bd518269357b237c0
lmcache: a7afadebb9248b62b5c533ce2c12297e9d94fc4a
native_extension_status: TO_FREEZE
model_tokenizer: TO_FREEZE
filesystem_l2_root: TO_FREEZE
connector_config: TO_FREEZE
serde_config: disabled for V0
~~~

## Current Position

~~~yaml
current_milestone: P2-M0 Environment / ABI / Connector Freeze
current_parent_task: P2-M0-T1
execution_mode: REVIEW_ONLY
current_slice: NONE
parent_task_status: BLOCKED_BY_PROJECT1_CORE
slice_status: NOT_STARTED
approved_implementation_slice: NONE
runtime_unlock_gate: P1-G3 V1_RUNTIME_CORE_DONE accepted
last_completed_gate: DOCUMENTATION_AUDIT_ONLY
next_parent_task: NONE
~~~

## Approved Designs

- ADR-002 high-level P2 identity and V0/V1/V2/V3 boundary。
- ADR-003 P1-first sequence。
- vLLM/LMCache pinned baselines and ABI gate。
- filesystem L2 V1 baseline。

No implementation API/format/field/kernel/placement/Slice approved。

## Design Drafts

- fixed metadata + FP16 scales + INT8 payload。
- quant_group symmetric INT8；preferred **LMCache object-domain** K/V-separated × layer × KV-head × fixed token group × head_dim。`Block` 是 codec quantization block，不以对齐 vLLM physical page 为目标。
- exact MemoryLayoutDesc-derived offsets。
- PyTorch V1 default；Triton V2 line A/line B separation。
- layerwise V3 first，resource pool/transfer scheduling later。

## Known Source Facts

| Status | Fact | Anchor |
|---|---|---|
| SOURCE_FACT | vLLM baseline has LMCacheConnectorV1 hooks | lmcache_connector.py |
| SOURCE_FACT | LMCache RequestTracker maps allocated blocks to slots | vllm_v1_adapter.py |
| SOURCE_FACT | Serde ABC + AsyncSerdeProcessor exist | distributed/serde |
| SOURCE_FACT | SerdeL2AdapterWrapper composes temp serialize→inner L2 | serde_wrapper.py |
| SOURCE_FACT | FSL2Adapter writes MemoryObj byte_array and reports bytes | fs_l2_adapter.py |
| SOURCE_FACT | FP8 serde is simple factory/template | fp8.py |
| SOURCE_FACT | TurboQuant demonstrates Triton codec and CPU/CUDA staging | turboquant/* |
| SOURCE_FACT | a7afade baseline aligns torch2.11; f9addd snapshot pins torch2.13 | pyproject/commit audit |
| SOURCE_FACT | generic Serde may occur after raw GPU→CPU materialization | architecture audit |
| SOURCE_FACT | newer vLLM offload provides page views/pinned pools/fences reference | 0ecc284 snapshot |

均来自 v3 audited baseline；P2 M0 local source/runtime revalidation blocked until unlock。

## To Verify After Unlock

worktrees/import/native ABI/backend actual enablement；connector example；RequestTracker shapes；MemoryObj layout/device；raw FS-L2 object/file/bytes；Serde wrapper actual code vs docs；StorePolicy/L1 retention；Triton staging；layer hooks；exact allowed files；fixed-size FSL2 byte invariant scope/pre-existing-key behavior；MemoryLayoutDesc exact object shape/head_dim；quant_group token-group size；filesystem page-cache/O_DIRECT benchmark regime。

## Evidence

NONE runtime。Only documentation audit。正式根：04-experiments/project2_kv_offload。

## Modified Files / Experiments

Only package docs；no source modification；no experiment。

## Blockers

P1 Core not done；P2 M0 not reviewed；Approved Slice NONE。

## Verification

Documentation Audit only。ABI/source/unit/E2E/benchmark/profile NOT_STARTED。

## Git Checkpoint

TO_FREEZE；no commit/source change。

## Exact Next Action

等待 P1-G3 V1_RUNTIME_CORE_DONE accepted。解锁后由 ChatGPT Web + 用户 review P2-M0-T1 的 worktree/ABI/connector/environment gate；不能提前执行。

## Next Allowed Action

NONE for P2 runtime。允许只读文档 review；不得创建 Slice。

## Proposed Next Action

NONE while blocked。解锁后的 Codex 只能提出 proposal，不能自行批准 P2 next Slice。

## State Write Authority

Codex 仅事实性更新；任何未来 P2 Slice 完成后：

```text
Next Allowed Action: WEB_REVIEW_CURRENT_SLICE
```

format/placement/Gate PASS 与下一 Slice仍由用户 + Web 批准。
