# PROJECT_STATE — Project 1

> Authoritative detailed project state。CURRENT-CONTEXT、CHATGPT-HANDOFF 与 LATEST pack 均由此文件和 raw evidence 派生。

## Repository / Worktree

~~~yaml
project: Physical KV Cache Reclamation for vLLM
project_root: /home/qcrs/learning/llm-kv-lab/01-projects/project1-physical-kv-reclamation-vllm
implementation_repository: vllm-project/vllm
implementation_worktree: /home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim
reference_or_study_tree: /home/qcrs/learning/llm-kv-lab/third_party/vllm
branch: p1/v2-token-compaction-v026
head: db3cc1179f53e2dc8df2096ed17b5ec86ef033d3
last_good_commit: cd444b72ceecb2496bf015baf8c18bf883151fa1
accepted_tag: p1-v1-core-accepted
working_tree_status: DIRTY_SLICE_IN_PROGRESS
editable_import_path: /home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim/vllm
implementation_runner_generation: MRV2
implementation_runner_env: VLLM_USE_V2_MODEL_RUNNER=1
implementation_runner_source: vllm/v1/worker/gpu/model_runner.py
legacy_runner_v1_status: NOT_P1_TARGET / historical_reference_only
~~~

## Environment

~~~yaml
date_timezone: 2026-09-02 Asia/Shanghai
gpu_target: NVIDIA A100 80GB
python: 3.10.20
torch: 2.11.0+cu129
torch_cuda_runtime: 12.9
cuda_toolkit: 12.1.66 (/usr/local/cuda-12.1)
driver: 565.57.01
triton: 3.6.0
vllm_version: v0.26.0
vllm_commit: 568afb3a13806beb53bb2e6bd518269357b237c0
model: /data/models/Qwen3-0.6B
tokenizer: /data/models/Qwen3-0.6B
attention_backend: FLASH_ATTN v2
tp_pp_dp: 1/1/1
kv_dtype: BF16
block_size: 16
max_model_len: 128
eager: true
prefix_caching: false
spec_decode: false
async_scheduling: false
cuda_graph: false
num_kv_cache_groups: 1
dcp: 1
pcp: 1
kernel_block_size_equals_kv_manager_block_size: NOT_VALIDATED beyond accepted baseline
blocks_per_kv_block: 1
cp_kv_cache_interleave_size: 1
cascade_attention: OFF in accepted baseline; broader variants NOT_VALIDATED
post_reclaim_q_len: 1
~~~

## Current Position

~~~yaml
project_version: V2 / R1
current_milestone: P1 V2 / R1 execution-addressing foundation
current_parent_task: P1-V2-R1-D-RAGGED-GPU-EXECUTION-01
execution_mode: SLICE_EXECUTE
current_slice: P1-V2-R1-D-RAGGED-GPU-EXECUTION-01
parent_task_status: REVIEW_REQUIRED
slice_status: PASS_PENDING_WEB_REVIEW
approved_implementation_slice: P1-V2-R1-D-RAGGED-GPU-EXECUTION-01
last_completed_gate: P1-V1-CORE-GATE-REVIEW-01 PASS / ACCEPTED
next_parent_task: P1-V2-DESIGN-REVIEW-01
proposed_next_parent_task: P1-V2-R1-E-PRODUCTION-RAGGED-IDENTITY-01 (only after Web review)
~~~

## Approved Designs

- ADR-001 high-level project identity and V1/V2/V3 boundary。
- ADR-003 P1-first execution order。
- MVP constraints listed in REPO_PINS。

P1 V1/Core Gate 已接受并冻结；canonical implementation 为 commit `cd444b72ceecb2496bf015baf8c18bf883151fa1`、tag `p1-v1-core-accepted`。任何 V1 修改必须先走 `P1-V1-CORE-REOPEN-XX`。

## Design Drafts

- 历史 DESIGN_DRAFT：uniform effective_kv_len/request；已由 accepted V1 state contract 落地并冻结。
- 历史 DESIGN_DRAFT：reclaim-specific state transition / `BlockTables.append_block_ids(..., overwrite=True)`；已由 T2/R1 落地并冻结。
- 历史 DESIGN_DRAFT：attention metadata separate effective length；已由 S3 落地并冻结。
- DESIGN_DRAFT：Tangram-style ModelRunnerOutput channel；V1 不采用。
- DESIGN_DRAFT：V2 staged scratch gather/scatter。
- DESIGN_DRAFT：sink + recent whole-block policy。
- DESIGN_DRAFT / CONDITIONAL：explicit reclaim_generation / transaction ID，只有 fixed-source audit 证明 stale-result identity 需要时才批准。

V2 staged gather/scatter、retention policy 与 token-level compaction 仍未批准。

## Known Source Facts

| Status | Fact | Audited Anchor |
|---|---|---|
| SOURCE_FACT | pinned baseline 是 vLLM v0.26.0 / 568afb3... | repository pin |
| SOURCE_FACT | MRV2 BlockTables 可 staged full-row replacement，按 num_blocks 发布 active range | vllm/v1/worker/gpu/block_table.py |
| SOURCE_FACT | MRV2 runner 通过 add_requests/update_requests 处理新 request 与 cached request | vllm/v1/worker/gpu/model_runner.py |
| SOURCE_FACT | SingleTypeKVCacheManager.req_to_blocks 是 canonical request ownership | single_type_kv_cache_manager.py |
| SOURCE_FACT | slot mapping 将 position 分解为 block index/offset | block_table.py |
| SOURCE_FACT | Tangram 展示 effective allocation、worker output、scheduler free 闭环 | audited commit 8e1cbfa... |
| SOURCE_FACT | vLLM Triton reshape/cache 提供 paged address primitive | triton_reshape_and_cache_flash.py |
| SOURCE_FACT | Sparse-vLLM 提供 invalid-input atomicity/oracle methodology | audited commit c146433... |
| SOURCE_FACT | MRV2 RequestState owns request-lifetime indexed state | vllm/v1/worker/gpu/states.py::RequestState |
| SOURCE_FACT | MRV2 BlockTables supports staged full-row replacement with overwrite=True and active num_blocks | vllm/v1/worker/gpu/block_table.py |
| SOURCE_FACT | ModelRunnerOutput is the serialized worker-to-scheduler result boundary | vllm/v1/outputs.py; vllm/v1/engine/core.py |
| OBSERVED | MRV2 feasibility audit result is MRV2_FEASIBLE with F3/F4 narrow TO_VERIFY seams | 04-experiments/project1_kv_reclaim/raw/source-audit/20260825_mrv2-feasibility-audit.log |

这些来自 v3 audited baseline；P1-M0 必须在实际 local fixed worktree 复核 path/signature/import identity。

## Historical To Verify

以下条目是 freeze 之前的审计记录；不表示 accepted V1 仍待实现。

- worktree/branch/HEAD/status/import/native extension identity；
- Python/torch/CUDA/Triton/model/backend；
- exact v0.26 symbol signatures and callers；
- current attention metadata builder 的 minimal effective-length seam；
- ModelRunnerOutput → Scheduler update exact path；
- canonical ownership validation/free APIs；
- scheduler allocation input；
- q_len=1/eager/one-group fail-fast locations；
- Qwen3/RoPE positional-semantic gate：no dual-chunk/relative-bias physical-index reconstruction；
- partial-tail effective length / next append slot semantics；
- live MRV2 `append_block_ids(overwrite=True)` stale-tail observability：证明所有 consumer 仅访问 active/effective range，或批准 clear-old-range + rebuild wrapper；
- kernel block size / manager block size / blocks_per_kv_block / DCP/PCP/cascade exact runtime gate；
- reclaim-after-prefill limitation and rolling-arrival benchmark semantics；
- whether current result ordering + canonical ownership validation already makes explicit reclaim_generation unnecessary；
- copied fallback feasibility；
- all allowed/read-only files。

## Source Anchors

见 docs/10-design/02-vLLM-v0.26固定源码与Paged-KV-Reclaim-Source-Map.md；当前均为 AUDITED_BASELINE，local line numbers TO_FREEZE。

## Evidence

V1 Core Gate、T3 targeted/regression、真实 Engine closure 与 freeze evidence 均已记录；正式 evidence 根：04-experiments/project1_kv_reclaim/。

## Modified Files

- `worktrees/p1-vllm-reclaim/vllm/v1/worker/gpu/states.py`
- `worktrees/p1-vllm-reclaim/vllm/v1/worker/gpu/input_batch.py`
- `worktrees/p1-vllm-reclaim/vllm/v1/worker/gpu/model_runner.py`
- `worktrees/p1-vllm-reclaim/vllm/v1/core/sched/output.py`
- `worktrees/p1-vllm-reclaim/vllm/v1/core/sched/scheduler.py`
- `worktrees/p1-vllm-reclaim/vllm/v1/worker/gpu/model_states/default.py`
- `worktrees/p1-vllm-reclaim/vllm/v1/worker/gpu/attn_utils.py`
- `worktrees/p1-vllm-reclaim/vllm/v1/attention/backend.py`
- `worktrees/p1-vllm-reclaim/vllm/v1/attention/backends/flash_attn.py`
- `worktrees/p1-vllm-reclaim/tests/v1/worker/test_gpu_effective_kv_state.py`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s1/`
- `04-experiments/project1_kv_reclaim/records/m1-t1-s1/`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-implementation/`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/`
- `04-experiments/project1_kv_reclaim/records/m1/P1-M1-T1-S2-S3-IMPLEMENTATION-RECORD.zh-CN.md`
- `worktrees/p1-vllm-reclaim/tests/v1/worker/test_gpu_reclaim_commit.py`（01A conflict evidence；未接受）
- `worktrees/p1-vllm-reclaim/tests/v1/worker/test_gpu_reclaim_commit.py`（01A-R1 final-row oracle）
- `worktrees/p1-vllm-reclaim/tests/v1/core/test_reclaim_transport.py`（01B Scheduler transport oracle）
- `worktrees/p1-vllm-reclaim/vllm/v1/request.py`（T3 Scheduler physical frontier）
- `worktrees/p1-vllm-reclaim/vllm/v1/core/kv_cache_manager.py`（T3 physical allocation override / reconcile API）
- `worktrees/p1-vllm-reclaim/vllm/v1/core/kv_cache_coordinator.py`（T3 reconcile forwarding）
- `worktrees/p1-vllm-reclaim/vllm/v1/core/single_type_kv_cache_manager.py`（T3 dense ownership / safe free）
- `worktrees/p1-vllm-reclaim/tests/v1/core/test_reclaim_ownership.py`（T3 ownership/reuse oracle）
- `04-experiments/project1_kv_reclaim/raw/m1-t3-impl-01a/`
- `04-experiments/project1_kv_reclaim/notes/P1-M1-T3-IMPL-01A-Physical-Ownership-Reconciliation-实施记录.md`
- `04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01b-web-review-fix/`
- `04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a/`
- `04-experiments/project1_kv_reclaim/raw/m1-t2-impl-01a-r1/`
- `04-experiments/project1_kv_reclaim/notes/P1-M1-T2-IMPL-01A-Worker-Reclaim-Commit-Primitive-实施记录.md`
- `04-experiments/project1_kv_reclaim/notes/P1-M1-T2-IMPL-01A-R1-Final-Row-Reclaim-Composition-实施记录.md`
- `04-experiments/project1_kv_reclaim/notes/P1-CODE-DIFF-TRACKER.md`

## Closed Slice — P1-V2-R1-C-EXECUTION-FOUNDATION-01

- Web architecture review 结论：`PASS / CLOSED`。
- C final implementation commit：`db3cc1179f53e2dc8df2096ed17b5ec86ef033d3`。
- 本 Slice 只关闭 C0–C4 execution-addressing foundation，不启用 production Ragged dispatch。
- 本 Slice 修改的 source：`vllm/v1/kv_cache_interface.py`、`vllm/v1/ragged_kv_layout.py`、`vllm/v1/attention/backends/ragged_layout.py`（新增）、`vllm/v1/worker/gpu/attn_utils.py`。
- 本 Slice 修改的 tests：`tests/v1/test_ragged_attention_spec.py`、`tests/v1/test_ragged_kv_layout.py`、`tests/v1/worker/test_attn_utils.py`。
- 开始前已存在且只做 `NO_CHANGE_REVIEWED` 的 dirty 内容：`vllm/v1/core/ragged_kv_cache_manager.py`、`vllm/v1/core/sched/output.py`，以及 `vllm/v1/ragged_kv_layout.py` 中既存解释性注释；未回滚或覆盖。
- 代码追踪：`04-experiments/project1_kv_reclaim/notes/P1-V2-R1-C-EXECUTION-FOUNDATION-01-Code-Trace.md`。
- raw evidence：`04-experiments/project1_kv_reclaim/raw/p1-v2-r1-c-execution-foundation-01-final/`。
- 未证明：actual Ragged KV write、FlashAttention read、real Engine production Ragged identity、GPU engine smoke、Scheduler/ModelRunner wiring。
- closure action：提升权限后的 tiny CUDA alias smoke `PASS`；placement tensor lifetime contract `FROZEN`。证据：`raw/.../23-cuda-alias-smoke.log`（sandbox `BLOCKED_BY_ENV`）与 `raw/.../24-cuda-alias-smoke-escalated.log`（CUDA `PASS`）。

## Experiments

P1-M1-T1-S1 targeted GPU tests：T1 初始化、T2 normal delta、T3 logical/effective divergence、T4 request slot reuse；4 passed。完整命令与 warning 见 `04-experiments/project1_kv_reclaim/raw/m1-t1-s1/`。

P1-M1-T1-S2/S3 GPU2 targeted tests：S1 lifecycle、Case A-E logical/physical split 与 request-length padding 共 10 tests；`10 passed, 15 warnings in 8.09s`。`py_compile` 与 `git diff --check` 均通过。GPU2 最终约占用 21.4 GiB 且利用率 100%，因此未额外启动 engine smoke，也未终止外部进程。

P1-M1-T1-S2/S3 review-fix：normal FA2 physical-length scope guard、`prepare_attn()` cache-position wiring、builder-level physical/fallback oracle 与 selector scope guard 已加入；`13 passed, 15 warnings in 8.87s`，`py_compile` / `git diff --check` 通过。

P1-M1-T2-IMPL-01A：targeted tests `20 passed, 3 failed`。canonical transition 单独成立，但同一 publication 前对同 row stage reclaim overwrite + normal append 产生两条 descriptors，超过 `StagedWriteTensor` 按 `num_rows` 配置的 descriptor capacity；触发 `DESIGN_CONFLICT`，未进入 01B。

P1-M1-T2-IMPL-01A-R1：PASS。resolved by retained + same-step new delta → one final physical row → one overwrite descriptor；历史 CUDA oracle `25 passed`，本次复跑因 NVIDIA driver unavailable 为 `25 skipped`。

P1-M1-T2-IMPL-01B：PASS。prepared whole-block plan 在 allocation 成功后由 Scheduler 基于 `KVCacheManager.get_block_ids()` materialize，按 aligned `CachedRequestData.reclaim_transitions` 传输；Worker reclaim/normal 路径互斥，`effective_kv_len.apply_write()` 在 forward 前发布。GPU 0 targeted/regression：`30 passed, 15 warnings`。

P1-M1-T2-IMPL-01B-WEB-REVIEW-FIX：PASS / WEB_REVIEW_ACCEPTED。当前 worktree final state 独立 core suite `3 passed`，combined suite `30 passed`，exit code 均为 0；修正 normal cached lifecycle 与 continuing RUNNING allocation-failure oracle，production code 未修改。

本轮 Web Review Closure 未修改 production；仅修改 `tests/v1/core/test_reclaim_transport.py`，将 normal cached 与 allocation-failure tests 改为真实 Scheduler lifecycle，并保留历史失败 evidence。

P1-M1-T3-IMPL-01A：PASS_PENDING_WEB_REVIEW。Scheduler completed physical frontier、physical allocation override、dense canonical ownership reconcile 与 refcount-safe release已实现。GPU 0 targeted `4 passed`；T2/R1/S1/single-type manager regression `39 passed`；final combined `43 passed`。Oracle证明 removed blocks使 free capacity `+2`、另一 request真实获得 released physical ID，且原 request下一 q=1不发生 bogus allocation。无 ACK、force-free、BlockPool redesign 或 benchmark。

P1-M1-T3-INTEGRATION-CLOSURE：PASS_PENDING_WEB_REVIEW。GPU 2、真实 `/data/models/Qwen3-0.6B`、MRV2/eager/FA2 运行通过：Worker row `[1,2,3,4,5,6]→[1,2,5,6]`，forward-entry E=62、post-forward/commit E=63；Scheduler canonical row完成相同 dense reconcile，free blocks `10860→10862`，released IDs 3/4 refcount归零，request B真实复用 block 4；request A继续生成并以8 output tokens完成，request B以2 output tokens完成。Production未修改，仅新增 smoke script、closure report并修正文档 commit ordering。

P1-V2-R1-C-EXECUTION-FOUNDATION-01：`32 passed` focused C0–C4/Dense suite、`37 passed` Ragged planner/manager/worker regression、`5 passed` Dense `attn_utils` regression；focused `ruff`、`py_compile` 和 `git diff --check` 均 PASS。closure action 的 escalated CUDA alias smoke `PASS`；placement tensor lifetime contract `FROZEN`。完整 changed-file `ruff` 仅报告三个 HEAD 既有问题：`kv_cache_interface.py:289 E501`、`attn_utils.py:94 E501`、`attn_utils.py:671 UP038`，本轮未修改。Address/layout raw evidence、命令与限制见 trace 和 `raw/p1-v2-r1-c-execution-foundation-01-final/`。

P1-V2-R1-C-EXECUTION-FOUNDATION-01：Web architecture review 已完成，最终状态为 `PASS / CLOSED`；final implementation commit 为 `db3cc1179f53e2dc8df2096ed17b5ec86ef033d3`。

## Current Slice — P1-V2-R1-D-RAGGED-GPU-EXECUTION-01

- D 状态：`PASS_PENDING_WEB_REVIEW`；D-WRITE、D-DECODE、D-PREFILL-MIXED 均在 focused real-CUDA closure 中通过。
- D actual start HEAD：`dc4287411704abc512f6cca3ca669c7d9ec254c5`；worktree 保留用户既有 HEAD delta，未创建新 commit。
- 主要 source：`vllm/v1/attention/backends/ragged_layout.py`、`vllm/v1/attention/backends/ragged_forward.py`。
- 主要 tests：`tests/v1/attention/ragged_reference.py`、`tests/v1/attention/test_ragged_execution.py`、`tests/v1/test_ragged_kv_layout.py`。
- Code Trace：`04-experiments/project1_kv_reclaim/notes/P1-V2-R1-D-RAGGED-GPU-EXECUTION-01-Code-Trace.md`。
- raw evidence：`04-experiments/project1_kv_reclaim/raw/p1-v2-r1-d-ragged-gpu-execution-01/`。
- real CUDA execution/layout：`29 passed`；CPU/C/Ragged/Dense focused：`71 passed, 6 skipped`；Dense `attn_utils`：`5 passed`；static checks PASS。
- 一次 `test_attention_backends.py` 全量尝试因缺少远端 `meta-llama/Meta-Llama-3-8B` 本地配置在第 6 个测试失败；前 5 个通过，未归因于 D source。
- 未启用 production Ragged dispatch；未修改 Scheduler、ModelRunner、ForwardContext、Dense `flash_attn.py` 或新增 custom op。

## Decisions

- high-level ADR-001/003 only。
- P1-M1-T1-S1：新增独立 GPU `RequestState.effective_kv_len`，只与 normal computed delta 同步推进；不新增 CPU mirror，不实现 reclaim consumer。
- P1-M1-T1-S2/S3：`positions` / `seq_lens` 保持 logical；新增 `cache_positions` 驱动 slot mapping，新增 optional `effective_kv_seq_lens` 驱动 FA2 physical visibility；`None` 保留 upstream fallback，未修改 FA kernel ABI。
- S2/S3 review-fix：仅在 `dcp_world_size == 1` 且 `common_prefix_len == 0` 的 normal FA2 path 使用 effective lengths；DCP/cascade 恢复 upstream logical length；新增 wiring 与 builder-level tests。

## Blockers

- Accepted V1/Core 已冻结；不得在 V2 开发中顺手修改 V1。
- V1 accepted source identity 已由 immutable checkpoint 固化：commit `cd444b72ceecb2496bf015baf8c18bf883151fa1`、tag `p1-v1-core-accepted`；该 checkpoint 当时 worktree CLEAN。当前 implementation worktree 因本 Slice 为 `DIRTY_SLICE_IN_PROGRESS`，历史 freeze 阶段的 upstream + dirty diff 仅作 provenance 保留。
- prefix/spec/async/CUDA Graph/PP/DCP/PCP>1/multi-group/connector 仍 NOT VALIDATED。
- retention policy、token-level compaction、Triton/CUDA compaction、benchmark、HBM/OS return、V2 implementation 均 OUT OF SCOPE。

## Verification

- Documentation Audit：PASS；P1 V1/Core Gate Review：PASS / ACCEPTED。
- Source audit：MRV2 baseline scope evidence complete；未验证配置不外推。
- M0 static/oracle closure：READY_FOR_WEB_REVIEW；row replacement stale-tail、gather active range、current slot mapping 和 partial-tail oracle evidence 已记录，未批准 M1。
- 01A historical：FAIL/BLOCKED（20 passed, 3 failed）；R1 historical CUDA targeted/regression：25 passed；R1 本次复跑：25 skipped（driver unavailable）；`py_compile` / diff-check：PASS；real reclaim integration/E2E/benchmark/profiler：NOT_STARTED。
- 01B targeted/regression：`30 passed, 15 warnings`（GPU 0，本地 `/data/models/Qwen3-0.6B`）；`py_compile` / `git diff --check`：PASS；无 benchmark/E2E/profiler。
- 01B Web Review Fix：core `3 passed`、combined `30 passed`，均 exit code 0；focused diff、no-T3 audit、py_compile、diff-check 见 `raw/m1-t2-impl-01b-web-review-fix/m1-t2-01b-web-review-fix.log`。
- T3-IMPL-01A：targeted `4 passed`、existing regression `39 passed`、final combined `43 passed`；`py_compile` / `git diff --check` 均为 rc=0。Evidence：`raw/m1-t3-impl-01a/m1-t3-impl-01a.log`。
- T3-INTEGRATION-CLOSURE：真实 Engine smoke `P1_M1_T3_INTEGRATION_CLOSURE_PASS`、exit code 0。Evidence：`raw/m1-t3-integration-closure/m1-t3-integration-closure.log`。
- V1-CORE-GATE-REVIEW-01：I1–I18 baseline scope PASS；report 与 raw evidence 已冻结。
- V1-CORE-FREEZE-01：accepted source manifest、source identity、architecture note、freeze report、handoff 与 artifact index 已生成。
- P1-V2-R1-C-EXECUTION-FOUNDATION-01：`32 passed` focused、`37 passed` Ragged state regression、`5 passed` Dense regression；`py_compile`、focused `ruff`、`git diff --check` PASS。Evidence：`raw/p1-v2-r1-c-execution-foundation-01-final/` 与对应 Code Trace。

## Git Checkpoint

~~~yaml
start_commit: cd444b72ceecb2496bf015baf8c18bf883151fa1
last_good_commit: cd444b72ceecb2496bf015baf8c18bf883151fa1
accepted_tag: p1-v1-core-accepted
commit_created: cd444b72ceecb2496bf015baf8c18bf883151fa1
rollback_point: p1-v1-core-accepted
~~~

## Exact Next Action

D Slice 已完成，等待 `WEB_REVIEW_CURRENT_SLICE`；不得启用 production Ragged runtime 或进入 E。

## Next Allowed Action

WEB_REVIEW_CURRENT_SLICE

## Proposed Next Action

完成 D 后固定回到 `WEB_REVIEW_CURRENT_SLICE`；建议 review 通过后再考虑
`P1-V2-R1-E-PRODUCTION-RAGGED-IDENTITY-01`。V1 Core remains frozen；任何 V1 change
仍需 `P1-V1-CORE-REOPEN-XX`。

## State Write Authority

Codex 仅可更新事实字段。V1/Core 的 accepted/frozen 状态来自 Web/Architecture adjudication；
本 Slice 只记录实际实现、测试和 evidence，不批准下一 Slice、不修改 Approved Design、不解锁
P2/V2 runtime。后续任何 V1 修改必须先创建 `P1-V1-CORE-REOPEN-XX`。
