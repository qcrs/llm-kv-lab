> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Ragged Runtime State Machine, Ownership & Transaction Protocol

这份文档定义谁拥有 physical page、什么时候 worker state可信、什么时候必须丢弃 snapshot。R3、R5、R6 实现前必须冻结。

## 1. Two authorities

```text
Scheduler / KVCacheManager
= canonical physical ownership authority

Worker / ModelRunner
= execution view + payload transformation authority
```

Worker 可以报告“我把 state compact 成什么形状”，但不能单方面决定 allocator 中哪些 page 已经 free。

## 2. Request lifecycle states

### S0 — NEW

Scheduler 已决定 admission/allocate，worker尚未建立 row。

### S1 — ACTIVE_IN_BATCH

worker `RaggedBlockTables` row + `RaggedKVState` 是 execution source；scheduler拥有同一组 physical IDs 的 canonical bookkeeping。

### S2 — CACHED_OUT_OF_BATCH

request 暂时不在 persistent active batch，但 scheduler **仍拥有 KV**。必须保存：

```text
exact per-group block IDs
per-group counts
effective lengths
```

uniform flat list不足以重建 compressed row。

### S3 — PREEMPTED_RECOMPUTE

scheduler已经 free request KV；worker所有 physical snapshot都失效：

```text
block row snapshot = DROP
effective lengths   = RESET
scorer state         = RESET
pending compaction   = DROP
```

以后靠 logical token history recompute。

### S4 — FINISHED

scheduler free ownership，worker彻底清 state。

## 3. Compression transaction

### Phase A — Decision / fence

Scheduler或policy产生：

```text
request_id
step_seq / source_version
expected source effective lengths [G]
expected source block counts [G]
keep/target decision
```

source 是**本轮 forward 完成后**的 state，包括本 step 新写 query。

### Phase B — Worker payload mutation

Worker验证 fence：

```text
actual source lengths/counts == expected
```

然后 compaction payload；成功前不发送 result。

### Phase C — Worker result

建议结构：

```text
request_id
source_version
new_effective_lens [G]
new_num_blocks [G]
```

不把 worker看到的 freed IDs 当 allocator authority。

### Phase D — Scheduler canonical reconcile

Scheduler在自己拥有的 group rows上：

```text
old canonical row
→ retain prefix new_num_blocks[g]
→ derive detached tail IDs
→ verify no duplicate/cross-owner violation
→ free pool IDs
→ commit new canonical counts
```

### Phase E — next-step worker visibility

下一 step worker state必须已与 scheduler canonical row一致；若采用 worker post-forward直接trim，scheduler只做验证/ownership commit；若采用scheduler next-output回传 canonical shape，则需要显式 pending状态。两种实现选一个，不能混。

## 4. Same-step allocation ordering

危险 case：

```text
old E=31, block_size=16
current step writes 2 tokens
→ source E=33 and scheduler may allocate page #3
then compaction target E'=20
```

source fence必须描述 `33 / 3 pages`，而不是 forward-entry 的 `31 / 2 pages`。

R3 首 Slice 为降低变量，允许先硬 gate：compression boundary不能与 new page allocation同一步发生；等 basic closure后再补 mixed case。

## 5. TP=2 distributed commit

每 rank可以选择不同 keep positions，因为 KV heads本来 sharded；但 physical depth/free boundary若 scheduler page IDs在 ranks间同构，则必须一致。

协议：

```text
rank-local kept_lengths
  ↓
all_reduce(MAX)
  ↓
reconciled kept_lengths
  ↓
each rank extends/executes local writeback to same physical depth
  ↓
optional debug all_gather checksum(lengths/counts)
  ↓
worker result
  ↓
scheduler free once
```

若任一 rank fence不一致：hard abort该 compaction transaction，不允许 best-effort free。

## 6. Preemption invalidation table

| State | scheduler still owns pages? | restore exact row? | restore E? | restore scorer? |
|---|---:|---:|---:|---:|
| active batch reorder | yes | yes/move | yes | yes |
| temporarily out of batch | yes | yes | yes | yes |
| true preemption/recompute | no | **no** | **no** | **no** |
| finished/abort | no | no | no | no |

## 7. Debug invariants

开发阶段每个 boundary可选开启：

- all active physical IDs unique per request unless null-page semantic明确允许；
- scheduler-owned ID set == union of request rows；
- `ceil(E[g]/B) <= num_blocks[g]`；
- after compact，尾部 detached IDs不再出现在任何 request row；
- free ID refcount / ring-state consistent；
- TP ranks `new_num_blocks[g]` checksum一致。

这些 assertion不必进入 production hot path，但必须成为 early development evidence。
