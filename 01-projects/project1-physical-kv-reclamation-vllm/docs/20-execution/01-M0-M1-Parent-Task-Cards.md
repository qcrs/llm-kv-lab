# P1 M0–M1 Parent Task Cards

所有 Parent Task 均遵循：没有 one-shot Approved Slice Prompt 时 REVIEW_ONLY；候选 S1/S2 只是 PLANNED_DECOMPOSITION。Allowed Files 需在 M0 映射为实际路径；超出即停止。

## P1-M0-T1 — Source / Environment / Worktree Freeze

- Goal：冻结独立 implementation worktree、branch/HEAD/status、import/native extension、环境与 MVP flags。
- Learning Goal：用户能解释“环境 identity”和“source identity”为何是两道 Gate。
- Read First：REPO_PINS、SOURCE-IDENTITY template、vLLM pyproject/build metadata。
- Source Anchors：git worktree/HEAD；vllm.__file__；native .so __file__；torch/CUDA/Triton versions。
- Approved Design Needed：worktree/branch naming 与是否复用已有环境；尚未批准。
- Allowed Files：process/runtime identity/env、M0 evidence、project state；source read-only。
- Read-only Files：所有 vLLM source、study/reference trees、已有 raw evidence。
- Out Of Scope：install/rebuild、source patch、benchmark、M0-T2。
- Expected Patch Shape：只读 probes + manifests/docs；0 source LOC。
- Tests：git identity/status；import identity；native extension import；GPU inventory；minimal feature-off smoke only if separately approved。
- Evidence：repo_state、env manifest、import paths、commands/stdout、dirty-file manifest。
- Pass/Fail Gate：全部 identity 一致且无用户改动风险；未知为 TO_FREEZE 则 BLOCKED。
- Failure Signals：import 指错树、HEAD mismatch、undefined symbol、silent backend fallback、dirty overlap。
- Rollback：不改变环境/source；撤销仅 project-owned docs。
- Handoff：列出 exact source/env、what not changed、review questions。
- Next Task：P1-M0-T2，仅 Gate PASS 后。
- Candidate decomposition：S1 repo/worktree audit；S2 env/import/ABI audit；S3 evidence/state close。均未创建。

## P1-M0-T2 — Fixed Source Map Freeze

- Goal：在 local pinned source 核对 scheduler→manager→runner→BlockTable→attention→output→free 链。
- Learning Goal：用户能指出 canonical owner、worker view、publication points。
- Read First：10-design/02、P1 v3 source inventory。
- Source Anchors：req_to_blocks、BlockPool、allocate_slots、SchedulerOutput/update、ModelRunnerOutput、MRV2 `add_requests/update_requests`、`prepare_inputs`、`append_block_ids(overwrite=True)`、`compute_slot_mappings`、attention builder。
- Approved Design Needed：NONE for read-only audit；任何 patch seam 只可 DESIGN_DRAFT。
- Allowed Files：source-audit record/source map/evidence/state。
- Read-only：pinned source。
- Out Of Scope：实现字段/API、trace instrumentation。
- Expected Patch Shape：0 source LOC；definition/caller/callee/owner/state/shape/test table。
- Tests：rg/static trace；existing relevant test inventory；no runtime mutation。
- Evidence：exact paths/signatures/commit、reference diff table、TO_VERIFY list。
- Gate：all Core anchors mapped or explicit blocker/fallback。
- Failure：v3 assumption absent、caller chain unknown、current source differs。
- Rollback：N/A；preserve audit negative result。
- Handoff：facts vs design implications separated。
- Next：P1-M0-T3。
- Candidate decomposition：ownership chain；worker/data path；attention/output/free path。

## P1-M0-T3 — Native Baseline and Oracle Freeze

- Goal：冻结 no-reclaim baseline、MVP config、tiny allocator/attention oracle plan。
- Learning Goal：理解 fixed BlockPool capacity 与 CUDA HBM allocation 区别。
- Read First：evaluation correctness/capacity docs。
- Anchors：native allocation/free tests、runner smoke、attention backend。
- Approved Design Needed：baseline model/workload/config。
- Allowed：test harness/config/evidence only after Slice approval。
- Read-only：production source。
- Out：reclaim patch、formal performance。
- Patch Shape：project-owned smoke/oracle harness and manifest。
- Tests：feature-off identity；allocator tiny sequence；Qwen3/RoPE positional-semantic feasibility；partial-tail slot feasibility。
- Evidence：raw config/results/manifest；no invented numbers。
- Gate：reproducible baseline and exact expected oracle。
- Failure：flaky model output、uncontrolled GPU、capacity metric ambiguous。
- Fallback：smaller model/tensor oracle；do not alter pin。
- Handoff：baseline limitations。
- Next：P1-M1-T1。
- Candidate decomposition：baseline config；allocator oracle；attention oracle。

## P1-M1-T1 — Logical / Physical State Contract

- Goal：批准 logical_num_computed、effective_kv_len、logical_position、physical_cache_position 的 owner/lifecycle。
- Learning Goal：能手推 logical=128/physical=64 的下一步。
- Read First：10-design/04、MRV2 source map、`prepare_inputs` consumers。
- Anchors：request state、persistent input batch、positions、seq_lens、slot mapping、metadata。
- Approved Design Needed：field ownership、initialization/reset/update、batch row/request-state identity；是否需要 explicit generation 仅作为 TO_VERIFY。
- Allowed：decision record/tests/spec；source remains read-only until approved。
- Out：runtime implementation。
- Patch Shape：state table、transition oracle、API draft。
- Tests：pure state transitions；retention=1；row reuse/invalid input。
- Evidence：Approved Design record + oracle cases。
- Gate：所有 consumer 被分类 logical/physical，no overloaded ambiguous field。
- Failure：必须全局改 seq_lens、req_idx reusable identity 被忽略。
- Fallback：separate adapter/view；conservative no reclaim。
- Handoff：explicit approved/draft/to-verify。
- Next：P1-M1-T2。
- Candidate decomposition：consumer audit；state schema/oracle；design approval。

## P1-M1-T2 — Whole-Block Reclaim Transition Contract

- Goal：冻结 keep validation、row reconstruction、effective length 与 copy fallback。
- Learning Goal：解释 physical IDs 不连续为何合法、顺序为何必须稳定。
- Read First：10-design/03、MRV2 `append_block_ids(overwrite=True)` path、Sparse oracle。
- Anchors：MRV2 BlockTables row lifecycle、RequestState、`add_requests/update_requests` replacement semantics。
- Approved Design Needed：trigger、keep policy baseline、partial-tail rule、fallback condition；explicit generation only if source audit proves needed。
- Allowed：pure policy/state oracle and tests after approval。
- Read-only：runtime source。
- Out：scheduler free、attention integration。
- Patch Shape：small project-owned transition helper/oracle。
- Tests：sorted/unique/in-bounds、no mutation on invalid、old=retained∪freed。
- Evidence：state snapshots and exact expected IDs。
- Gate：deterministic transition independent of GPU。
- Failure：foreign/shared IDs、partial mutation、row capacity mismatch。
- Rollback：feature off; retain old row。
- Next：P1-M1-T3。
- Candidate decomposition：validation oracle；row semantic adapter；fallback review。

## P1-M1-T3 — Worker ACK / Scheduler Commit Contract

- Goal：批准 two-commit transaction、output schema、ownership validation、idempotency。
- Learning Goal：解释 scheduler-first free alias。
- Read First：10-design/05、Tangram output/free reference。
- Anchors：ModelRunnerOutput、scheduler output handling、KV manager ownership/free。
- Approved Design Needed：ACK payload、canonical ownership/idempotency、duplicate/stale/error behavior；transaction ID/generation 为 conditional draft。
- Allowed：contract/oracle/test doubles；runtime source read-only before approval。
- Out：actual runtime patch。
- Patch Shape：transaction dataclasses/spec + mock state machine。
- Tests：ACK loss、duplicate、stale、ownership mismatch、no-free safety；验证无需 generation 是否已足够，若不足才提出 generation。
- Evidence：sequence diagrams/state tables/test output。
- Gate：no path publishes a page before worker view commit。
- Failure：worker directly mutates pool、scheduler cannot validate request owner。
- Fallback：conservative no-free / synchronous copy path。
- Next：P1-M2-T1。
- Candidate decomposition：output channel audit；transaction oracle；approval close。



### M0/M1 Additional Hard Gates (v1.2 final contract)

- 首个模型冻结为 Qwen3-family standard RoPE decoder path，禁止把 positional argument 泛化到 ALiBi/relative-bias/dual-chunk。
- 必须加入 partial-tail (`effective_kv_len % block_size != 0`) oracle。
- `reclaim_generation` 不得因为文档旧稿存在就自动批准；必须给出 fixed-source stale-result necessity evidence。

### v1.2 Final-Contract Gates

M0/M1 除上述 gate 外还必须关闭：

- runtime确认 `DCP=PCP=1`、`blocks_per_kv_block=1`、manager/kernel block size相同、CP interleave=1；cascade attention保持 off/unsupported；
- MRV2 `append_block_ids(overwrite=True)` live shorter-row replacement 的 stale-tail observability：source proof 或 defensive clear-old-range design；
- 明确 Core reclaim 发生在 final-prefill 后，不能把 post-reclaim capacity 写成 first-prefill peak/max-context improvement；
- V2 future contract中 selection index命名/语义使用 `keep_member_indices`（current retained member domain）；
- feature-off identity 与 feature-on/retention=1 no-op identity 分开。
