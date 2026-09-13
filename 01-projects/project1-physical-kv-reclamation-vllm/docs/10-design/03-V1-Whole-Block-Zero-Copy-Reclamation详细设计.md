# 03 — V1 Whole-Block Zero-Copy Reclamation 详细设计

## 1. Design Intent

V1 解决的是 **physical ownership reclamation**，不是 attention mask。它只允许 block-aligned keep/drop，因此 surviving KV page 本体不搬动：

```text
old active row:       [B0 B1 B2 B3 B4 B5 B6 B7]
keep logical columns: [ 0  1              6  7]
retained IDs:         [B0 B1 B6 B7]
candidate free IDs:   [B2 B3 B4 B5]
```

Paged KV 的 block-table indirection 允许 physical IDs 不连续；但是 retained **logical order 必须保持**。V1 的关键价值是：

```text
GPU KV tensor copy = 0 for whole-block survivors
request-owned pages ↓
BlockPool reusable pages ↑
```

这不等于 CUDA allocator 将 HBM 归还系统。

## 2. Source-Backed Primitives

### Pinned vLLM v0.26 (current MRV2 source map)

- `gpu/block_table.py::BlockTables.append_block_ids(..., overwrite=True)`：把 staged row 从起点替换并更新 `num_blocks`；这是当前 MRV2 full-row replacement candidate。
- `gpu/model_runner.py::add_requests()`：新 request 将 scheduler block IDs 写入 RequestState/BlockTables；`update_requests()` 处理 cached requests。
- stale tail 对 gathered table 由 `num_blocks` 隔离，但 slot mapping kernel 的 persistent-row边界仍是 `TO_VERIFY`；不得把它写成已关闭的安全证明。

### Tangram

提供 general-case：

- effective physical sequence length；
- logical model position 与 physical slot position 分离；
- worker 完成 physical mutation 后把 freed IDs / new effective length送回 scheduler；
- scheduler/KV manager 再修改 canonical ownership 与 free pool；
- allocation sizing 从 effective occupancy 继续增长。

因此 V1 的 own work 是 scoped API adaptation / glue，不是未知 architecture。

## 3. Hard Preconditions

V1 runtime feature enable 前必须全部满足：

```text
num_kv_cache_groups == 1
prefix caching == off
spec decode == off
async scheduling == off
CUDA Graph == off/eager bring-up
dense causal full attention
first supported model passes RoPE positional-semantic gate
reclaim trigger only after final prefill forward has committed
first post-reclaim q_len == 1
DCP == PCP == 1
kernel_block_size == kv_manager_block_size
blocks_per_kv_block == 1
cp_kv_cache_interleave_size == 1
cascade attention == off / unsupported until audited
```

失败时 fail-fast/feature-off，不进入“可能也能跑”的路径。

## 4. State Before Reclaim

对 request A：

```text
Scheduler canonical state
  logical_num_computed = L
  req_to_blocks[A]     = [B0 ... Bn-1]

Worker RequestState + BlockTables (MRV2)
  block_ids            = [B0 ... Bn-1]
  num_computed_tokens  = L

Persistent InputBatch
  BlockTable row       = [B0 ... Bn-1]
  effective_kv_len     = L
```

第一版只在 full prefill 完成后触发，因此没有后续 prompt chunk 再写被 drop 的 logical cache position。

## 5. Reclaim Decision Contract

Policy 只产生 decision，不允许直接 free：

```text
ReclaimDecision (conceptual)
  request_id
  keep_block_columns       # sorted, unique
  retained_valid_tokens    # physical/effective valid token count
```

第一版 policy：`sink blocks + recent blocks`。Policy 不拥有 BlockPool，也不修改 Worker BlockTable。

### Keep validation — mutation 前完成

必须检查：

- sorted ascending；
- unique；
- in bounds；
- whole-block semantics；
- retained order stable；
- keep set 与 partial-tail rules 一致；
- 不涉及 shared/refcounted prefix page（MVP 已关闭 prefix cache）。

invalid input 必须 **no mutation**。

## 6. Partial-Tail Block：必须显式处理

若：

```text
block_size = 16
prompt / current effective len = 130
```

最后一页只有 2 个有效 KV token。即使 active row 有 9 个 block：

```text
effective_kv_len = 130
physical_block_count = 9
```

不是 144。

Reclaim 后必须继续保持：

```text
physical_block_count = ceil(effective_kv_len / block_size)
```

下一 decode 的 physical cache position 应等于 `effective_kv_len`，因此写到 retained tail page 的下一个有效 slot；不能覆盖 tail，不能把 unused slot 暴露给 attention。

V1 correctness 必须含 `0 < effective_kv_len % block_size < block_size` case。


## 6.5 `append_block_ids(overwrite=True)` 的组合语义：必须处理 stale tail

Pinned `add_row()` 的核心是把 `num_blocks_per_row[row]` 归零后 append 新 IDs；它**不自动清空旧 row 中超出新 active length 的 tail entries**。因此：

```text
old active row: [B0 B1 B2 B3 B4 B5]
new active row: [B0 B1 B4 B5]
```

内存中的 row tail 可能仍残留旧值，但 active count 已缩短。

这本身不证明错误；如果所有后续 consumer 都严格由 `effective_kv_len / active block count` 生成 slot/address，则 stale tail 不可观察。但 live reclaim 是新的组合场景，所以 M1/M2 必须明确证明二选一：

### Option A — active-range proof

source audit + unit test 证明：

- slot mapping 只会生成 `< ceil(effective_kv_len / block_size)` 的 column；
- attention metadata 不会把 stale tail 计入 member length；
- scheduler append/allocation 不会从 stale tail 读取 ownership；
- request remove/move/swap 不会把 stale tail 重新提升为 active。

### Option B — defensive thin replacement semantic

在 reclaim transition 中显式：

```text
old_active_count snapshot
→ clear old active range / stale tail
→ rebuild retained row with existing primitive
→ publish new active count/effective len
```

这只是薄的 live-reclaim safety wrapper，不是新 BlockTable architecture。

**Gate：** 在这个问题没有被 source+test 关闭前，不能把 V1 zero-copy row replacement 标成 implementation-ready PASS。

## 6.6 Core 不减少首次 prefill peak

V1 reclaim 发生在 final-prefill forward 已完成之后。因此该 request 在 reclaim 前仍需完整持有 prompt KV：

```text
full prompt prefill
→ peak request-owned pages reached
→ reclaim
→ post-reclaim pages decrease
```

Core 的确定收益是 **post-reclaim allocator capacity recovery / later reuse**，不是 `max single-request prefill context ↑`。如果未来要在 prefill 过程中回收，需要新的 chunk/prefill-time correctness 设计，不属于 V1/V2。


## 7. Worker Physical-View Commit — 第一 commit point

正确链：

```text
final-prefill forward completes
        ↓
validate decision
        ↓
map keep columns → retained physical block IDs
        ↓
RequestState/BlockTables active block view = retained IDs
        ↓
BlockTables.append_block_ids(req_index, retained IDs, overwrite=True)
        ↓
effective_kv_len = retained valid token count
        ↓
commit BlockTable/effective metadata to device
        ↓
produce worker result
```

Worker result 的最小候选 payload：

```text
request_id
new_effective_kv_len
candidate_freed_block_ids
(optional diagnostic: retained_block_ids / old count)
error/status
```

**不默认要求 `reclaim_generation`。** 是否需要显式 transaction ID 要在 M0/M1 根据 fixed output consumption/lifetime 证明；当前只允许作为 conditional design draft。

Worker commit 的定义：新一步 attention/write view 已不再引用 candidate freed pages。

## 8. Scheduler / KV Manager Commit — 第二 commit point

Scheduler 收到 worker result 后，canonical owner 必须重新验证：

```text
request still exists
candidate IDs unique and non-null
candidate IDs currently belong to request
candidate IDs not in retained current ownership
new_effective_kv_len fits retained page capacity
no shared/refcounted page
worker status indicates full commit
```

然后：

```text
canonical req_to_blocks remove freed IDs
        ↓
update request effective physical occupancy
        ↓
BlockPool.free(...)
        ↓
only future scheduler iteration may reuse
```

Worker result 不是 allocator truth；scheduler-side canonical ownership 才是。

### Duplicate/stale result safety

同步 MVP 首选 **ownership-idempotency**：

- 如果 freed ID 已不属于该 request，禁止再次 free；
- 保存 evidence 并 no-free / reject；
- 只有 fixed-source audit 证明现有 identity 不足以排除 stale result 时，才批准额外 generation/transaction ID。

## 9. Next Decode Semantics

假设：

```text
logical_num_computed = 8192
effective_kv_len     = 4096
```

下一步：

```text
model/RoPE logical position = 8192
physical KV append position = 4096
```

生成一个 committed token 后：

```text
logical_num_computed = 8193
effective_kv_len     = 4097
```

两者独立增长，直到下一次 reclaim（V1 Core 只有一次 prefill-boundary reclaim）。

## 10. Allocation Contract

如果 allocation 继续使用 logical 8192，会把释放 page 重新补回。因此 sizing 必须使用：

```text
effective_num_cached_tokens
+ num_new_tokens
+ allowed lookahead
```

而 logical progress 仍用于 scheduler/model semantics。Tangram 的 `effective_num_cached_tokens` 是直接 reference template。

## 11. Failure / Fallback

| Failure | Required behavior |
|---|---|
| invalid keep | reject before mutation |
| ownership mismatch | no-free + evidence + stop |
| worker result lost | pages remain canonically owned；capacity leak > corruption |
| duplicate result | no duplicate free；ownership validation rejects |
| partial worker commit | fail-stop / rollback to last good commit |
| FA metadata cannot consume effective length | feature stays off；review seam |
| zero-copy path reveals hidden backend semantic | use V1-R copy-compaction fallback |

### V1-R fallback

```text
whole-block keep
→ Tangram-style gather/writeback to compact physical prefix
→ worker commits compacted row/effective length
→ scheduler frees only trailing pages
```

它多一次 copy，但不改变 logical/physical/safe-free contract。

## 12. V1 Required Oracles

1. pure state: old/keep → retained/freed；invalid atomicity；
2. partial-tail: next append slot exact；
3. slot mapping: logical position ≠ cache position；
4. ownership: free count exact；
5. real reuse: B 获得 A released ID；
6. A 在 B reuse 后继续 decode；
7. feature disabled identity：完全走 upstream path；
8. feature enabled + retention=1 no-op identity：reclaim infrastructure runs but frees nothing；
9. attention semantic oracle against dense retained K/V reference。

做到这里，V1 才是 real runtime subsystem，而不是 free-counter demo。
