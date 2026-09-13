# P1-M1-T1 Implementation Records 索引

> Parent Task：`P1-M1-T1 — Logical / Physical State Contract`

## 1. Parent Task 主链

```text
S1 persistent physical state
RequestState.effective_kv_len
        |
        v
S2 write-coordinate split
positions != cache_positions
        |
        v
S3 read-extent split
seq_lens != effective_kv_seq_lens
        |
        v
Web review-fix
physical length only when DCP=1 AND no cascade
```

## 2. Canonical Implementation Records

| Slice | 解决的问题 | Canonical record | 当前证据结论 |
|---|---|---|---|
| `P1-M1-T1-S1` | request-lifetime GPU physical progress state | `P1-M1-T1-S1-IMPLEMENTATION-RECORD.zh-CN.md` | 独立 storage；`128/64 -> 129/65`；S1 当时 4 tests PASS |
| `P1-M1-T1-S2` | logical model position 与 KV write coordinate 分离 | `P1-M1-T1-S2-S3-IMPLEMENTATION-RECORD.zh-CN.md` | `prepare_attn()` wiring 使用 `cache_positions` |
| `P1-M1-T1-S3` | logical request length 与 FA2 physical read extent 分离 | 同上 | normal FA2 builder 使用 guarded physical length；累计 13 tests PASS |
| S2/S3 review-fix | 收紧未审计 feature scope | 同上 `Web Review Fix` 章节 | 仅 `effective field present AND DCP=1 AND no cascade` 使用 physical；否则 logical fallback |

## 3. Learning 与 Evidence

- S1 状态生命周期长篇讲解：`../../../../01-projects/project1-physical-kv-reclamation-vllm/learning/00-foundations/P1-M1-T1-S1-MRV2-num-computed-tokens与Effective-KV-State生命周期详解-v2.md`
- S1 raw evidence：`04-experiments/project1_kv_reclaim/raw/m1-t1-s1/`
- S1 原始 record：`04-experiments/project1_kv_reclaim/records/m1-t1-s1/`
- S2/S3 implementation evidence：`04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-implementation/`
- S2/S3 review-fix evidence：`04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/`

## 4. 阅读顺序

1. 本索引：先理解 T1 的 Slice 分工。
2. S1 implementation record：理解 persistent state owner 和 normal delta。
3. S1 learning note：需要深入理解 Scheduler/Worker/GPU 三状态域时阅读。
4. S2/S3 implementation record：理解 write/read coordinate split 和 review-fix scope。
5. `PROJECT_STATE.md`：确认当前权威状态与下一允许动作。

## 5. 边界

`P1-M1-T1` 只建立 execution-side logical/physical state contract。以下内容不因这些 records 存在而视为完成：

- real reclaim transaction；
- BlockTable row replacement integration；
- stale-tail runtime correctness；
- Scheduler canonical ownership update；
- physical free / cross-request block reuse；
- worker ACK。

当前 `Next Allowed Action` 仍由 `PROJECT_STATE.md` 决定，本文不创建或批准下一 Slice。
