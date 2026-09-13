# P1 M2–M3 Parent Task Cards

所有 Parent Task 均遵循：没有 one-shot Approved Slice Prompt 时 REVIEW_ONLY；候选 S1/S2 只是 PLANNED_DECOMPOSITION。Allowed Files 需在 M0 映射为实际路径；超出即停止。

## P1-M2-T1 — Worker-Side Row / Effective State Integration

- Goal：feature-gated worker commit of retained row and effective length after last-prefill forward。
- Learning Goal：定位 data-plane publication 与 persistent batch update。
- Read First：design 03/04；approved M1 state/transaction ADR。
- Anchors：MRV2 `GPUModelRunner.execute_model/add_requests/update_requests`、`RequestState`、`BlockTables` staged writes。
- Approved Design Needed：M1 all；trigger timing and buffer publication。
- Allowed Files：exact worker/state files named by approved Slice；tests/project evidence。
- Read-only：scheduler/KV manager/attention until later slices。
- Out：free、serving benchmark、V2。
- Patch Shape：small feature flag + one reclaim transition; no unrelated refactor。
- Tests：worker unit/state snapshots、feature-off identity、feature-on/retention=1 no-op、invalid pre-mutation、shorter-row stale-tail case。
- Evidence：diff/source record/test raw。
- Gate：worker no longer exposes candidate IDs after commit。
- Failure：stale device row/tail can become observable、partial update、current prefill still writes dropped page。
- Rollback：feature off / Slice commit。
- Next：M2-T2。
- Candidate decomposition：observer; CPU row transition; GPU publication; worker output。

## P1-M2-T2 — Scheduler Ownership / Allocation Integration

- Goal：consume worker ACK, validate canonical owner, free pages, allocate from effective occupancy。
- Learning Goal：解释 owner list、pool publication、allocation frontier。
- Read First：design 05、approved transaction。
- Anchors：scheduler update_from_output、KVCacheManager.allocate_slots、SingleType manager、BlockPool。
- Approved Design Needed：ACK API、effective cached field owner。
- Allowed：approved scheduler/manager files + focused tests。
- Read-only：attention/kernel。
- Out：V2/policy/performance。
- Patch Shape：minimal output consumption + commit_reclaimed_blocks equivalent + effective accounting。
- Tests：ownership mismatch、duplicate free、released-ID reuse in manager oracle、feature off。
- Evidence：before/after owner/pool snapshots。
- Gate：free exactly once, allocation does not logical-backfill。
- Failure：B gets ID before A commit、foreign free、finish double free。
- Rollback：no-free conservative path / feature off。
- Next：M2-T3。
- Candidate decomposition：ACK parse; ownership commit; effective allocation。

## P1-M2-T3 — Slot Mapping and Attention Effective Length

- Goal：logical positions remain upstream; cache positions/effective attention length consume physical state。
- Learning Goal：手推 slot and dense retained attention。
- Read First：design 04、source map attention builder。
- Anchors：MRV2 `prepare_inputs`、`compute_slot_mappings`、CommonAttentionMetadata/backend builder。
- Approved Design Needed：minimal metadata seam and supported backend。
- Allowed：approved runner/metadata/backend files + tests。
- Read-only：unrelated sampling/spec/graph。
- Out：global seq_lens rewrite、new backend、V2 kernel。
- Patch Shape：separate cache position/effective length buffer；fixed shape。
- Tests：logical=128/physical=64、dense attention oracle、retention=1、q_len fail-fast。
- Evidence：positions/slots/metadata dumps and output compare。
- Gate：next decode correct and non-attention semantics unchanged。
- Failure：sampling progression changes、slot hits freed page、backend length mismatch。
- Fallback：supported reference attention path / copy compact prefix。
- Next：M3-T1。
- Candidate decomposition：consumer instrumentation; cache slot split; attention view。

## P1-M3-T1 — Pure / Unit Correctness Matrix

- Goal：close policy/state/transaction/ownership/slot unit layers。
- Learning Goal：能从 failure signal定位层。
- Read First：Correctness Oracle doc。
- Anchors：all project-owned helpers and focused upstream tests。
- Approved Design Needed：none beyond M1/M2。
- Allowed：tests/oracles/bug fixes within approved files。
- Out：formal E2E/performance。
- Patch Shape：test expansion only/minimal fixes。
- Tests：invalid atomicity、feature-off identity、feature-on/retention=1 no-op、duplicate/stale result without double-free、stale-tail unreachability/clearing、row reuse、effective allocation、unsupported granularity fail-fast。
- Evidence：test list/raw output/coverage map。
- Gate：deterministic all pass。
- Failure：flaky/hidden global sync/cross-request state。
- Rollback：last good M2 Slice。
- Next：M3-T2。
- Candidate decomposition：state tests; ownership tests; slot/attention tests。

## P1-M3-T2 — Single-Request E2E Continuation

- Goal：prefill→reclaim→multi-token decode correctness under MVP。
- Learning Goal：将 source chain 与 runtime trace 对齐。
- Read First：evaluation and Nsys instrumentation plan（不正式 profile）。
- Anchors：scheduler/worker/attention/output complete chain。
- Approved Design Needed：workload/model/tolerance。
- Allowed：approved test/trace instrumentation。
- Out：multi-request/capacity/benchmark。
- Tests：feature-off upstream equality；feature-on retention=1 no-op equality；reclaim retained-attention oracle；finish/abort。
- Evidence：command/config/raw outputs/state snapshots。
- Gate：original request continues without corruption and logical progress correct。
- Failure：first/second decode divergence beyond oracle、leak/double free。
- Rollback：feature off / fallback path review。
- Next：M3-T3。
- Candidate decomposition：one-step; repeated decode; cleanup.

## P1-M3-T3 — Multi-Request Reuse and Capacity

- Goal：A releases pages, B receives released ID, A continues；measure internal capacity。
- Learning Goal：区分 counter change、real reuse、serving outcome。
- Read First：Capacity benchmark doc。
- Anchors：BlockPool allocation/free and scheduler admission。
- Approved Design Needed：deterministic request sequence、**rolling/staggered arrival** workload、simultaneous-prefill negative regime。
- Allowed：test/benchmark harness/evidence; source instrumentation only if approved。
- Out：claiming HBM return、large benchmark matrix。
- Tests：exact released set/intersection、cross-request isolation、finish order、rolling A-reclaim→B-arrival reuse、prefill-peak vs post-reclaim owned pages、simultaneous-prefill pressure negative regime。
- Evidence：IDs/state/raw metrics/run manifests；必须包含 `prefill_peak_owned_blocks`、`post_reclaim_owned_blocks`、`capacity_recovery_latency`。
- Gate：real reuse + no corruption + fixed pool accounting conservation。
- Failure：only counter changes、ID alias、uncontrolled scheduler ordering。
- Rollback：deterministic manager integration test before serving。
- Next：M3-T4。
- Candidate decomposition：deterministic allocator reuse; E2E two requests; capacity sweep.

## P1-M3-T4 — V1 Core Release Review

- Goal：assemble G3 evidence, limitations, fallback and claim audit。
- Learning Goal：用户能完整解释 transactional reclaim。
- Read First：all M3 records、delivery claim rules。
- Anchors：artifact index, git checkpoint。
- Approved Design Needed：release decision and P2 unlock decision。
- Allowed：docs/state/delivery; no new feature patch。
- Out：V2/P2 automatic start。
- Patch Shape：release report/manifest/state only。
- Tests：re-run minimal clean-shell reproduction。
- Evidence：traceable checklist and negative results。
- Gate：all Core Done items; no unverified claims。
- Failure：missing raw chain/dirty state/unsupported config not fail-fast。
- Rollback：BLOCKED and exact missing action。
- Next：M4 or P2-M0 review, only user decision。
- Candidate decomposition：evidence audit; reproduction; release decision。
