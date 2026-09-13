# P1 Execution Control and Gates

所有 Parent Task 均遵循：没有 one-shot Approved Slice Prompt 时 REVIEW_ONLY；候选 S1/S2 只是 PLANNED_DECOMPOSITION。Allowed Files 需在 M0 映射为实际路径；超出即停止。

## Milestones

| M | Theme | Parent Tasks | Gate |
|---|---|---|---|
| M0 | Foundation / Source Freeze | T1 identity/env; T2 source map; T3 native baseline/oracle plan | G0 |
| M1 | V1 State Contract | T1 state semantics; T2 reclaim transition; T3 ACK transaction | G1 |
| M2 | V1 Runtime Integration | T1 worker; T2 scheduler/allocator; T3 slot/attention | G2 |
| M3 | Correctness / Capacity / Release | T1 unit; T2 single E2E; T3 multi reuse/capacity; T4 release | G3 V1_RUNTIME_CORE_DONE |
| M4 | V2 Primitive | T1 PyTorch; T2 Triton gather; T3 scatter/atomicity | G4 |
| M5 | V2 Runtime/Profile | T1 integration; T2 micro/NCU; T3 E2E/NSYS | G5 P1_MAIN_PROJECT_STRONG_DONE |
| M6 | Optional V3 | explicit separate GO | G6 |
| M7 | Delivery | report/reproduction; claim audit | G7 |

## Gate Evidence

G0：exact worktree/branch/HEAD/status/import/env/native smoke；source map definition/caller/owner/tests；baseline config。

G1：Approved state schema、owner、idempotency、keep validation、two commit points、failure/fallback；显式 generation 仅在 source audit 证明需要时批准；pure transition oracle。

G2：feature-gated minimal runtime chain；logical/cache positions and attention effective length；safe canonical free；feature-off regression。

G3：retention=1、single request、multi-request released-ID reuse、A continues decode、capacity accounting、negative results、Core Done review。

G4：PyTorch oracle、Triton gather/scatter correctness、invalid atomicity、no premature free。

G5 P1_MAIN_PROJECT_STRONG_DONE：runtime staged V2、kernel microbenchmark、NCU、NSYS、saved bytes vs compaction cost、serving/quality limits。

G6：item-specific compatibility/profiler evidence。

G7：clean reproduction、artifact chain、final report、resume claims only from evidence。

## Status Rule

Gate has PASS/BLOCKED only. Slice PASS never implies Parent Task/Gate PASS. PROJECT_STATE alone records current state and exactly one Next Allowed Action.

