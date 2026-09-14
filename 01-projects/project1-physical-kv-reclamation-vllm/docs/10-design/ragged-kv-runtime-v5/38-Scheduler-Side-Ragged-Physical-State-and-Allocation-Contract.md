# Scheduler-Side Ragged Physical State & Allocation Contract

**Status:** CANONICAL — required before R2/R3  
**Problem solved:** non-uniform compaction 后如何继续正确、持续地申请 page，而不会重新长回 uniform depth。

---

# 1. Why Worker-Only `E[r,g]` Is Insufficient

Worker 需要 `E[r,g]` 来：

```text
compute physical write position
build member seq_lens
build attention metadata
```

但 Scheduler/Allocator 同样需要 physical frontier 来决定：

```text
下一 step 哪些 group 已经有容量？
哪些 group 需要新 page？
需要申请多少 page？
canonical row 应该 append 到哪里？
```

如果 Scheduler 只知道 logical：

```text
request.num_computed_tokens
```

non-uniform compression 之后 allocation 必然错误或过度保守。

---

# 2. Canonical State

推荐新增 scheduler-side helper：

```text
RaggedSchedulerPhysicalState
```

由 Ragged-specific KV manager 持有，而不是把 vector 塞入 generic `Request`。

状态：

```text
req_to_effective_lens: dict[str, np.ndarray[int32]]  # [GT]
req_to_group_counts:  dict[str, np.ndarray[int32]]  # [GT]
req_to_flat_block_ids: dict[str, np.ndarray[int32]]
```

约束：

```text
sum(counts) == len(flat_block_ids)
counts[g] >= ceil(E[g] / B)
Core normal state: counts[g] == ceil(E[g] / B), except temporary reservation
```

flat layout：

```text
[group0 row][group1 row]...[groupGT-1 row]
```

通过 prefix-sum counts 获得 offset。

---

# 3. Why Manager-Owned Instead of `Request`-Owned

`Request` 是 model/logical lifecycle object。

Ragged physical vector 属于：

```text
KV allocation implementation
```

放进 manager 的优势：

- generic request 不需要理解 GT；
- preemption free 时 manager 可以原子清理 ownership + E + counts；
- scheduler reconciliation 不需要跨对象维护一致性；
- 后续 bulk allocator 可替换而不污染 Request API。

P1 scalar `Request.effective_kv_len` 可以继续服务 Dense/P1 path；Ragged 不扩成 union type。

---

# 4. Identity Allocation

R1：

```text
E[g] = logical E
counts[g] = ceil(E/B)
```

需要 append `q`：

```text
target_E[g] = E[g] + q
target_count[g] = ceil(target_E[g]/B)
need[g] = target_count[g] - counts[g]
```

Identity 下所有 `need[g]` 相同。

allocate：

```text
new_ids = pool.allocate(sum(need))
scatter(new_ids, need[g])
append to group rows
```

SchedulerOutput 给 Worker 的 delta 必须保留 group count 信息。

---

# 5. Non-Uniform Allocation

压缩之后：

```text
E = [31, 55, 19]
counts = [2, 4, 2]
B=16
```

下一 decode `q=1`：

```text
target_E = [32,56,20]
target_count = [2,4,2]
need = [0,0,0]
```

再下一 token：

```text
target_E = [33,57,21]
target_count = [3,4,2]
need = [1,0,0]
```

只给 group0 新增 page。

这是 Ragged physical capacity 能长期保持的必要条件。

---

# 6. SchedulerOutput Delta Contract

建议新增 transport：

```text
RaggedBlockDeltaData:
    request_id: str
    source_group_counts: tuple[int,...] | compact array
    appended_group_counts: tuple[int,...]
    flat_new_block_ids: tuple[int,...]
```

Worker deterministic scatter：

```text
offset = 0
for g:
    k = appended_group_counts[g]
    ids = flat_new_block_ids[offset:offset+k]
    block_table.append(g, ids)
    offset += k
```

不要让 Worker 根据：

```text
len(flat_new_ids) / GT
```

反推；R2 后根本不一定整除。

---

# 7. Append State Commit

首版同步 scope 下推荐：

```text
Scheduler schedules q
→ manager reserves required pages
→ SchedulerOutput contains block delta
→ Worker executes forward
→ output processed
→ canonical E += actual committed q
```

如果 compaction 同 step：

```text
final E = compaction_result.new_E
```

即 compaction result 覆盖 post-append frontier。

为了避免 R3 首 slice 同时处理 reservation rollback，可以先 gate：

```text
compression step 不触发新的 page allocation
```

随后补 same-step page-growth case。

---

# 8. Reconciliation Contract

Worker result：

```text
expected_source_E[GT]
expected_source_counts[GT]
new_E[GT]
new_counts[GT]
step_seq
```

Scheduler-side manager validation：

```text
canonical source E/counts == expected source
0 < new_E[g] <= source_E[g]
new_counts[g] == ceil(new_E[g]/B)
new_counts[g] <= source_counts[g]
```

然后：

```text
old row[g]
→ keep row[g][:new_counts[g]]
→ detached = row[g][new_counts[g]:]
```

全部 group 先 validate，任何一个失败：

```text
NO FREE
NO PARTIAL COMMIT
```

全部通过后一次性：

```text
commit rows/counts/E
free detached IDs
```

---

# 9. Allocation / Compaction Invariants

必须始终成立：

```text
I1. E[g] <= counts[g] * B
I2. counts[g] == len(group_row[g])
I3. group rows for same request contain no duplicate non-null page ID
I4. detached page IDs no longer appear in canonical request rows before free
I5. Worker row after acknowledgement == Scheduler canonical row
I6. logical num_computed_tokens never由 compaction 减少
```

---

# 10. Tangram Reference vs v5 Design

Tangram 已经识别到 compression 后 allocation 不能继续只用 logical `num_computed_tokens`，因此在 `KVCacheManager.allocate_slots` 增加 `effective_num_cached_tokens`，Scheduler 传 `request.compress_max_eff_seq_len`。

v5 的差异：

```text
Tangram conservative scalar frontier
→ max effective cache length

v5 canonical frontier
→ per-group E[GT]
```

这样做复杂度更高，但与本项目要证明的“长期 non-uniform physical ownership”完全一致。

---

# 11. Minimal Tests

只需要先做 5 个：

```text
T1 identity need[g] all equal
T2 unequal E only one group crosses boundary
T3 delta scatter reconstructs exact rows
T4 stale source vector => zero mutation/free
T5 after reclaim, next decode does not regrow shorter groups unnecessarily
```

T5 是 v5 新增最关键 test。
