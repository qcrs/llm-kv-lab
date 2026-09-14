# TP=2 Distributed Ragged KV — Exact Engineering Contract

**Prerequisite:** R4 Core closed under TP1.  
**Supported first slice:** `TP=2 + uniform budget + FA2 + full attention + no DCP/PCP/PP`。

---

# 1. Problem Definition

Tensor Parallel 会把 attention heads 分片到不同 rank。Ragged compression 又允许每个 local head/group 保留不同 positions。

因此 TP correctness 不是：

```text
all ranks return same token
```

而是：

```text
rank-local attention state is correct
AND
scheduler-visible physical ownership transition is compatible across ranks
```

---

# 2. Local Geometry

只使用 vLLM API：

```python
Hkv_local = model_config.get_num_kv_heads(parallel_config)
Hq_local  = model_config.get_num_attention_heads(parallel_config)
```

定义：

```text
Hp = page_group_size
Glocal = Hkv_local / Hp
q_per_kv = Hq_local / Hkv_local
```

startup：

```text
Hkv_local > 0
Hkv_local % Hp == 0
Hq_local % Hkv_local == 0
```

---

# 3. Rank-local State

每 rank 独立：

```text
physical KV backing
RaggedBlockTables
RaggedKVState
member map
scores
keep positions
compaction payload
```

不同 rank **不要求** keep positions 一样。

---

# 4. Shared Physical Contract

Scheduler 不能接受：

```text
rank0 new_depth=6
rank1 new_depth=7
```

却只提交一个 canonical shape transition。

第一版最简单 contract：

```text
per corresponding group retained length
→ all_reduce(MAX)
→ common physical retained length
→ common page count
```

各 rank 的 payload 可以在 common retained capacity 内不同。

---

# 5. Synchronization Point

推荐 sequence：

```text
local score/keep candidate
→ local kept_len
→ GPU tensor kept_lengths
→ TP all_reduce(MAX)
→ common kept_len
→ each rank builds final physical plan
→ compaction
→ result
```

不要：

```text
先 rank0 free
再等 collective
```

free boundary 必须在 collective 之后冻结。

---

# 6. Scheduler Receipt

建议 scheduler-visible result 不依赖 rank-local keep indices，只携带：

```text
request_id
common new_group_counts
common effective physical depths
step_seq/source fence
```

Worker/rank-local correctness由 executor内部保证。

如果 ModelRunnerOutput 只由 rank0 上报，则 debug build 必须先验证 rank consensus。

---

# 7. Debug Consensus

仅 debug：

```text
all_gather(group_counts)
all_gather(detached_count)
all_gather(hash(group-count-vector))
```

assert all same。

不要在 production hot path做 Python object all_gather。

---

# 8. Tests

## T1 Local Geometry

```text
Qwen config
TP=2
check Hkv_local/Glocal/q_per_kv
```

## T2 Identity Member Map

每 rank local head → local `(group,column)` bijection。

## T3 Identity Engine

```text
Dense TP2 vs Ragged TP2
```

output parity。

## T4 Local Keep Difference

人为让 rank0/rank1 keep candidates 不同；验证 MAX 后 common physical length一致。

## T5 Manual Reclaim

```text
TP2 manual compression
→ group counts identical across ranks
→ scheduler free count exact
```

## T6 Reuse

第二请求复用 scheduler released block；两 rank 继续正确执行。

## T7 Policy

uniform budget + one scorer。

---

# 9. Failure Modes

### F1: assume global_Hkv/TP

某些 model/parallel semantics 可能并不等价；始终使用 ModelConfig API。

### F2: all-reduce keep positions

没有必要；只同步 physical boundary。

### F3: rank0 scheduler result hides rank1 divergence

debug consensus必须在 submit 前检查。

### F4: AOT cluster crosses TP shard

R6 首版禁止 cross-rank clustering。

### F5: MAX sync happens after compaction/free

顺序错误，会造成不同 rank payload/page depth 不兼容。

---

# 10. Performance Metrics

```text
collective time per compression boundary
compression frequency
TPOT delta
throughput delta
extra retained pages due MAX synchronization
```

需要单独记录：

```text
rank-local ideal pages
common-MAX pages
```

这能量化 distributed consistency 的 capacity tax。

---

# 11. Resume Gate

只有 T3–T6 全部 PASS 才能写：

> 扩展至 TP=2，通过 rank-local KV head geometry 与 collective 同步 physical retained boundary，保证跨 rank page ownership 与 scheduler reclamation 一致。
