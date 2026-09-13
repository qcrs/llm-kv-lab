# P2 M0–M2 Parent Task Cards

当前 P2 runtime 全部 BLOCKED_BY_PROJECT1_CORE。以下是未来 Parent Task contract；候选 Slice 仅 PLANNED_DECOMPOSITION，不是授权。

## P2-M0-T1 — Runtime Unlock, Worktrees and ABI Gate

- Goal：先验证P1 unlock；再冻结vLLM/LMCache worktrees、branches/HEAD/status/import/native ABI/environment。
- Learning Goal：解释为何a7afade是implementation baseline、f9addd2仅reference。
- Read First：REPO_PINS、design 02、SOURCE-IDENTITY。
- Anchors：both pyproject、native modules、import paths、backend capability。
- Approved Design Needed：worktree/env strategy；P1 unlock accepted。
- Allowed：identity/env docs/evidence；source read-only。
- Out：install/rebuild/source patch/connector workload。
- Patch Shape：0 source LOC；read-only manifests。
- Tests：version/import/.so symbols/backend actual enablement/GPU。
- Evidence：commands/stdout/git/import/ABI manifest。
- Gate：all identities consistent；no silent fallback。
- Failure：torch mismatch/undefined symbol/wrong tree/dirty overlap。
- Rollback：no environment mutation；BLOCKED_ENV。
- Handoff：exact blocker。
- Next：M0-T2。
- Candidate：unlock check；worktree identity；ABI/backend。

## P2-M0-T2 — Connector / MemoryObj / Serde / L2 Source Map

- Goal：fixed commits内追vLLM connector→LMCache tracker→MemoryObj→Serde wrapper→FS。
- Learning：谁持有raw/temp/object、completion/cleanup、slots。
- Read First：design 02/03/04。
- Anchors：connector hooks、RequestTracker/ReqMeta、base/async/wrapper/fs/fp8/turboquant。
- Approved Design Needed：none read-only；design implications remain draft。
- Allowed：source audit/evidence/state。
- Read-only：both sources。
- Out：runtime/config changes。
- Patch：0 source LOC；object flow table。
- Tests：existing tests inventory/static caller trace；必须纳入 pinned `test_serde_fs_e2e.py` 与 `test_turboquant.py` 作为后续 test-pattern reference。
- Evidence：exact signatures/code-vs-doc drift/device/layout TO_VERIFY。
- Gate：all V0/V1 owners/lifetimes mapped。
- Failure：source baseline lacks required wrapper/backend、docs misleading。
- Fallback：raw local backend scope report；do not change pin。
- Next：M0-T3。
- Candidate：connector/tracker；serde/wrapper；FS/staging。

## P2-M0-T3 — Connector Reuse Baseline

- Goal：复现upstream shared-prefix connector reuse，Serde disabled。
- Learning：cache hit与generation resume evidence。
- Read First：local reuse example、source map。
- Anchors：connector scheduler/worker hooks。
- Approved Design Needed：model/prefix/config/workload。
- Allowed：configs/harness/evidence; no core source。
- Out：distributed FS/custom codec/performance。
- Patch：project-owned smoke。
- Tests：A store B reuse；feature off/cold compare；repeat。
- Evidence：cached token/hit signal、outputs、raw logs。
- Gate：real reuse, no source drift。
- Failure：no hit/request mapping mismatch。
- Fallback：small supported model/config；pin unchanged。
- Next：M1-T1。
- Candidate：config smoke；hit proof；Gate close。

## P2-M1-T1 — Raw Filesystem-L2 Store/Load

- Goal：distributed FS-L2 with Serde disabled，actual file/write/read。
- Learning：ObjectKey/chunk/MemoryObj→file。
- Read First：design 03、fs adapter。
- Anchors：StoreController/FSL2Adapter/L2StoreResult。
- Approved Design Needed：filesystem root/config/cleanup policy。
- Allowed：config/harness/evidence; adapter source read-only。
- Out：codec/Serde。
- Patch：baseline config/probe only。
- Tests：store/load exact raw object、file exists、repeat/partial failure；byte exactness使用 `stat.st_size`，不使用 `du`。
- Evidence：raw bytes/temp/file (`stat.st_size`)/reported bytes、commands；latency若记录必须标 buffered/page-cache/O_DIRECT regime，不在 G1 做 universal disk claim。
- Gate：deterministic raw path。
- Failure：permission/key/size/read mismatch。
- Fallback：new empty project-owned FS dir；no external backend。
- Next：M1-T2。
- Candidate：store; load; byte manifest。

## P2-M1-T2 — Raw FS-L2 + vLLM E2E Reuse

- Goal：raw FS object load resumes cache-hit request。
- Learning：connector local vs distributed L2 path区别。
- Read First：M0/M1 evidence。
- Anchors：wrapper disabled path/controller completion。
- Approved Design Needed：exact request sequence/tolerance。
- Allowed：harness/evidence/minimal approved config。
- Out：custom Serde。
- Tests：cold/store/restart-or-evict/load/resume；bad/missing file。
- Evidence：file, hit, generation output, bytes/latency baseline。
- Gate：G1。
- Failure：L1 hit masks L2、file not actual source、generation corrupt。
- Fallback：force/verify L2 access via approved config。
- Next：M2-T1。
- Candidate：L2 provenance; E2E; Gate close。

## P2-M2-T1 — Block-INT8 Layout / Size Contract Approval

- Goal：基于 frozen LMCache `MemoryLayoutDesc` 批准 **object-domain** quant_group、scale dtype、regions/offsets、fixed-size exact estimator；不要求对齐 vLLM BlockPool page。
- Learning：payload/scale/metadata/padding如何计真实bytes。
- Read First：design 04、FP8 actual code、wrapper allocation、pinned `test_turboquant.py` object-layout/size tests。
- Anchors：MemoryLayoutDesc、estimate API、byte_array。
- Approved Design Needed：actual K/V/layer/token/hidden axes、`num_kv_heads/head_dim` interpretation、fixed token group G、scale/layout choices。
- Allowed：spec/PyTorch size calculator/tests；runtime source read-only。
- Out：Triton/Serde registration。
- Patch：format descriptor + pure size tests。
- Tests：multiple shapes/dtypes、object-domain axis/head_dim、overflow/alignment/unsupported；不得把 codec block当 vLLM physical block。
- Evidence：formula/examples/exact expected bytes。
- Gate：same layout⇒same bytes；no data-dependent length。
- Failure：estimator必须worst-case raw或backend writes padding unexplained。
- Fallback：simpler headerless fixed layout/FP32 scale config。
- Next：M2-T2。
- Candidate：layout audit；size oracle；design approval。

## P2-M2-T2 — PyTorch Serializer / Deserializer Oracle

- Goal：implement pure Block-INT8 quant/dequant and byte pack/unpack。
- Learning：quant math vs serialized layout separation。
- Read First：approved format、FP8 template。
- Anchors：Serde sync interface only。
- Approved Design Needed：M2-T1。
- Allowed：project codec module/tests；no wrapper integration。
- Out：async/Triton/E2E。
- Patch：small serializer/deserializer reference。
- Tests：round-trip、PyTorch oracle、zero/edge/nonfinite policy、shape/dtype、exact bytes。
- Evidence：MAE/RMSE/max/cosine、payload compare。
- Gate：deterministic correct oracle。
- Failure：offset overlap/size mismatch/unsupported silent cast。
- Rollback：format spec only。
- Next：M2-T3。
- Candidate：quant oracle；pack/unpack；Serde adapter unit。

## P2-M2-T3 — Serde Factory / Async Wrapper / FS Integration

- Goal：register custom Serde and traverse existing AsyncSerdeProcessor + wrapper + FS。
- Learning：sync codec如何嵌入现成async lifecycle。
- Read First：actual wrapper/async/factory。
- Anchors：registration、submit/query/event/temp lifecycle。
- Approved Design Needed：factory name/config/failure semantics。
- Allowed：minimal serde integration/config/tests；no async framework rewrite。
- Out：vLLM E2E/Triton。
- Patch：factory glue + approved module。
- Tests：completion exactly once、failure cleanup、temp release、FSL2 scoped per-object/per-store byte equality (`stat.st_size`)；control already-existing keys；优先适配 pinned real FS Serde E2E test structure。
- Evidence：object lifecycle trace/file/stat/report。
- Gate：G2 fixed-size + FSL2 scoped exact invariant。
- Failure：temp leak/full allocated padding/file mismatch。
- Rollback：serde disabled raw path。
- Next：M3-T1。
- Candidate：factory/config；async lifecycle；FS byte Gate。

## v1.2 Filesystem Performance Boundary

M0–M2 的 raw/Serde bring-up 先证明 object/file/byte/lifecycle correctness。正式 filesystem crossover 进入 M5 时才按 `FILESYSTEM-BENCHMARK-REGIMES.md` 区分 buffered-warm、large-working-set/new-key 与 verified O_DIRECT。不得在 baseline smoke 中把 page-cache latency包装成 NVMe/FS bandwidth。
