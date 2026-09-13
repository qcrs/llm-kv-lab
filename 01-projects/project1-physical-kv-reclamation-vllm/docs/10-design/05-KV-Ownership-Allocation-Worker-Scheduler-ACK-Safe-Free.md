# 05 — KV Ownership / Allocation / Worker→Scheduler ACK / Safe-Free

## 1. Ownership Model

三层必须严格分开：

```text
Scheduler-side KV manager
= canonical request ownership truth

Worker BlockTable / persistent batch
= current data-plane view

BlockPool free queue
= reusable capacity publication
```

Worker 不能直接把 ID 交给 free pool；BlockPool free count 也不能反过来当 request ownership truth。

Pinned vLLM `SingleTypeKVCacheManager.req_to_blocks`（MRV2 不改变该 scheduler-side owner）是 V1 canonical owner reference。Tangram 的 scheduler free chain用于 port semantic，不整体 cherry-pick。当前 worker result transport seam 是 serialized `ModelRunnerOutput`；不预先添加 ACK 字段。

## 2. Why Scheduler-First Free Is Unsafe

禁止：

```text
Scheduler removes B3 and publishes free
→ Request B allocates B3
→ Worker A stale BlockTable still references B3
→ A/B alias same physical KV page
```

这种 bug 可能表现为 silent generation corruption，而不是 deterministic crash。

核心顺序必须是：

```text
Worker stops referencing
→ Worker reports
→ Scheduler validates canonical ownership
→ Scheduler removes ownership
→ BlockPool publishes reusable page
```

## 3. Minimal Result Contract — 不过度设计

当前建议最小 worker result：

```text
request_id
new_effective_kv_len
candidate_freed_block_ids
status/error
optional diagnostic snapshot/count
```

### `reclaim_generation` policy

当前 **不是 mandatory field**。

M0/M1 要先审：

- ModelRunnerOutput 是否按 step 顺序消费；
- 同步 MVP 是否可能对同一 request 有多个 reclaim result in flight；
- request ID reuse / preemption/finish 是否可能让旧 result 作用到新 physical state；
- canonical ownership validation 是否已经足够 idempotent。

只有答案表明仍存在 stale-result ambiguity，才批准 generation/transaction ID。

## 4. Scheduler Commit Validation

在任何 free 前：

1. request 仍存在；
2. worker status = committed；
3. freed IDs unique；
4. freed IDs 非 null/invalid；
5. every freed ID 当前确实属于 request canonical ownership；
6. retained/current ownership与 freed 不重叠；
7. new effective length 不超过 retained page capacity，且 partial-tail valid token count正确；
8. prefix/shared ownership在 MVP 不存在；
9. feature/runtime gate仍满足。

失败策略：**conservative no-free**。宁可临时损失容量，也不允许 double-free/foreign-free。

## 5. Duplicate / Lost / Stale Handling

### Lost result

如果 worker 已完成 row mutation但 scheduler没收到成功 result：

- scheduler canonical owner仍保留 candidate pages；
- 造成 capacity leak / accounting divergence，需要 fail-stop/recovery；
- 但不能“猜测”并 free。

### Duplicate result

第二次处理时 freed IDs通常已不属于 current canonical ownership：

```text
validation fails
→ no second free
→ record duplicate evidence
```

### Stale result

同步 Core 首先通过 current request/ownership/state validation拒绝。只有 source audit 证明 stale result 可能在 ownership set 偶然再次匹配时，才引入 explicit generation。

## 6. Allocation after Reclaim

allocation 目标不再由 logical progress决定。必须把：

```text
logical_num_computed
```

留给 model/request progression，而用：

```text
effective_num_cached_tokens + num_new_tokens + allowed_lookahead
```

计算 physical slots/blocks requirement。

否则：

```text
free pages
→ next schedule sees logical 8192
→ immediately allocates back toward 8192
```

等于 reclamation 失效。

## 7. Publication / Real Reuse Oracle

仅看到：

```text
free_count N → N+k
```

不足以证明 real reclaim。

必须做：

```text
A owns released IDs R
→ worker commits A view without R
→ scheduler removes R
→ pool free count +|R|
→ B is admitted
→ B allocation contains at least one ID from R
→ A concurrently/afterward continues decode correctly
```

同时检查 finish/abort 不再次 free R。

## 8. Async Boundary

V1/V2 Core async scheduling关闭，因此 worker result可作为简单 safety boundary。V3 async 时参考 newer vLLM：

```text
processed-safe boundary
+ request-specific deferred free/fence
```

而不是全局 `cudaDeviceSynchronize()` 粗暴串行化。

## 9. Failure Priority

优先级固定：

```text
correct ownership/lifetime
> capacity recovery
> latency
```

任何“为了避免 leak 而 force-free”的方案都不允许进入 Core。
