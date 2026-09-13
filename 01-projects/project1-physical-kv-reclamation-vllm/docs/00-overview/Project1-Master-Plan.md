# Project 1 Master Plan

## Goal

交付 production-runtime scoped、reference-backed、可复现的 physical KV reclamation 项目，而不是孤立 policy 或 kernel demo。

## Milestones and Gates

| Milestone | Purpose | Completion Gate |
|---|---|---|
| M0 Foundation / Source Freeze | 冻结 source/env/worktree/anchors/baseline | P1-G0 |
| M1 V1 State Contract | 冻结 logical/physical/ownership/output transaction | P1-G1 |
| M2 V1 Runtime Integration | worker/scheduler/allocation/slot/attention 闭环 | P1-G2 |
| M3 V1 Correctness / Capacity / Release | single/multi request、reuse、capacity、release audit | P1-G3 V1_RUNTIME_CORE_DONE |
| M4 V2 PyTorch + Triton Primitive | oracle、gather、scatter、atomicity | P1-G4 |
| M5 V2 Runtime / Profiling | paged runtime、Nsys/NCU、cost model | P1-G5 P1_MAIN_PROJECT_STRONG_DONE |
| M6 Optional V3 | hardening only by reference/profiler | separate GO |
| M7 Delivery | report、reproduction、resume evidence | P1-G7 |

## Gate Principles

- Gate 无 partial pass。
- G0 前无 source patch。
- G1 前不实现 runtime glue。
- G2 前不做 serving benchmark。
- G3 是 P2 runtime unlock gate。
- G4/G5 的 Triton speedup 不是 PASS 条件。
- M6 必须用户明确 GO。
- 每个 Parent Task 可拆 Slice，但 package 初始化不创建 Approved Slice。

## Core Invariants

1. logical_num_computed = model/request progress。
2. physical/effective_kv_len = retained cache occupancy。
3. logical_position = original model/RoPE position。
4. physical_cache_position = effective frontier / slot position。
5. ownership only mutates after worker commit ACK。
6. request owner list、BlockTable row、effective length、allocator accounting 必须在同一 reclaim state transition 上一致；显式 generation 仅在 source audit 证明需要时引入。
7. invalid keep/free input 在 mutation 前拒绝。
8. feature-off identity 与 feature-on/retention=1 no-op identity 分开验证。

## Deterministic vs Conditional Outcomes

Deterministic：request blocks↓、free pool↑、released-ID reuse、physical occupancy↓。

Conditional：concurrency/admission/preemption/TPOT/throughput/quality。

Not a claim：nvidia-smi HBM 必然下降、Triton 必然更快、single-request maximum first-prefill context 自动提高、first-prefill peak KV 自动下降。

## Current Control

Current Parent Task P1-M0-T1；REVIEW_ONLY；Slice NONE。详细任务见 20-execution。



## Technical Baseline Preservation

完整 v3 reasoning snapshot 位于 `../90-reference/baselines/`；历史冲突以 `../90-reference/BASELINE-RESOLUTION.md` 为 canonical。`SUPPORTED-CONFIG-MATRIX.md` 是 M0/V1/V2 的 support gate。

## v1.2 Core Scope Clarifications

- V1/V2 reclaim 发生在 final-prefill commit 之后；容量收益是 post-reclaim allocator recovery / later reuse。
- Core 只支持 `DCP=PCP=1`、manager/kernel block size一对一、`blocks_per_kv_block=1`、CP interleave=1、cascade attention off。
- MRV2 `BlockTables.append_block_ids(..., overwrite=True)` 是当前 row replacement candidate，但 live reclaim 仍必须关闭 stale-tail composite seam。
- V2 selection index 是 current retained/member sequence 的 `keep_member_indices`，不是永久等同 original logical token index。
