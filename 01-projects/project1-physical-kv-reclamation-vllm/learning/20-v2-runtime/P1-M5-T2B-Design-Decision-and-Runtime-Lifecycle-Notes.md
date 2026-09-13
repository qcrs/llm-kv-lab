# P1 M5-T2B 最终设计抉择与 Worker Runtime Lifecycle 全链路笔记

**Project:** P1 — Physical KV Cache Reclamation for vLLM  
**Phase:** V2 — Token-Level Physical KV Compaction  
**Milestone:** M5-T2B — Real Post-Forward Compaction + Worker Physical-State Commit  
**Status:** `DESIGN FROZEN / READY FOR IMPLEMENTATION`  
**Target runtime:** vLLM v0.26.0 MRV2 / FA2 / BF16 / eager / A100 / single GPU / single KV group  
**Reference:** vLLM v0.26.0 + Tangram（仅架构参考，不是 patch/cherry-pick baseline）  
**Date:** 2026-09-10

---

## 0. 结论先行

M5-T2B 的核心目标不是“再做一个 KV 删除算法”，而是把 M4 已经完成的 **KV payload compaction primitive** 正确接入 vLLM Worker runtime，并建立一条不会破坏逻辑序列语义、物理 KV 语义、BlockTable 寻址以及 Scheduler ownership 的 runtime transaction。

最终冻结的主链路是：

```text
Scheduler
  │
  │ CompactionPlanData
  ▼
GPUModelRunner.execute_model()
  │
  ├─ update Worker runtime state
  ├─ prepare attention / slot mappings
  ├─ model.forward()
  │      └─ 当前 step 的 q 个新 KV 已写入 paged KV
  │
  ├─ _prepare_v2_compactions()          [T2A / validate only]
  │      ├─ request identity
  │      ├─ req_state_idx / batch_idx
  │      ├─ post-forward source E
  │      ├─ exact source page count
  │      ├─ active BlockTable row
  │      ├─ keep_member_indices
  │      └─ all-layer structural checks
  │
  ├─ _execute_v2_compactions()          [T2B / destructive]
  │      ├─ M4 compact layer 0
  │      ├─ M4 compact layer 1
  │      ├─ ...
  │      ├─ all payload compactions succeed
  │      ├─ stage retained BlockTable prefix
  │      ├─ build CompactionResultData
  │      └─ produce per-batch E override = K
  │
  └─ ExecuteModelState
         ├─ completed compaction_results
         └─ effective_kv_len override state
             │
             ▼
        sample_tokens()
             │
             ▼
        postprocess_sampled()
             │
             ▼
        post_update()
             ├─ logical progress: L += computed_delta
             └─ physical progress:
                    normal / V1 → E += computed_delta
                    V2 compact  → E = K   (absolute replacement)
             │
             ▼
        ModelRunnerOutput.compaction_results
             │
             ▼
Scheduler M5-T3
  ├─ validate result against canonical ownership
  ├─ reconcile req_to_blocks
  ├─ Scheduler request.effective_kv_len = K
  └─ removed suffix enters safe deferred-free lifecycle
```

这里最重要的设计抉择有三个：

1. **KV payload compaction 必须发生在 `model.forward()` 之后，而不是 pre-forward，也不是塞进 `post_update()`。**
2. **persistent `effective_kv_len` 的最终提交仍由 `post_update()` 统一负责；V2 使用 absolute override，而不是先写 `E=K` 再想办法 suppress 正常增量。**
3. **Worker 只负责物理 payload 与 execution-addressing view；Scheduler 仍然是 canonical block ownership / free authority。**

这三个决定构成 T2B 的架构骨架。

---

# 1. 为什么 V2 不是 V1 的简单延伸

P1 V1 和 V2 都叫“physical KV reclamation”，但二者实际处理的对象不同。

V1 是 **whole-block physical reclamation**。它的目标是：

```text
旧 physical row
[B0, B1, B2, B3, B4, B5]

retain
[B0, B1, B4, B5]

下一 step 再 append 新 block
```

V1 的关键特点是：reclaim transition 在 forward 前生效。新的 `effective_kv_len` 是 forward 的起点，当前 step 的 `q` 个 token 尚未写入，因此后续正常的：

```text
E += q
```

仍然正确。

V2 不一样。V2 是 **token-level physical compaction**：

```text
forward 前:
physical E = E_t

forward:
写入当前 q 个 token

forward 后:
source extent = E_t + q

从整个 source extent 中选择 keep members
并压缩成 [0, K)
```

也就是说，当前 step 的 q 个新 KV 本身已经属于 V2 的 source。

因此 V2 的正确 next-state 是：

\[
E_{next}=K
\]

而不是：

\[
E_{next}=K+q
\]

这是 T2B 设计里最容易犯错、也最根本的问题。

---

# 2. 必须先区分的六种“位置 / 索引”

整个 V2 如果不先区分 index domain，很容易出现“代码能跑但语义错”的问题。

| 概念 | 含义 | 例子 | 是否会因 V2 改写 |
|---|---|---:|---|
| `logical_position` | 模型语义位置 / RoPE position | token 原始位置 128 | **不改** |
| `member_index` | 当前 physical member sequence 内的顺序 | `0..E-1` | keep 按它索引 |
| `cache_position` | 当前 physical KV prefix 的 token offset | `0..K-1` | **会压缩** |
| `block_id` | physical KV pool page ID | B7、B2、B11 | block 本身不等于 token |
| `req_state_idx` | Worker persistent request-state row | 7 | persistent state 域 |
| `batch_idx` | 当前 step batch 中 request row | 1 | current-batch 域 |

其中最重要的语义是：

```text
keep_member_indices
≠ logical positions
≠ token IDs
≠ block IDs
≠ slot IDs

keep_member_indices
= compaction 前 current physical member sequence 的下标
```

例如：

```text
physical member sequence:
[m0, m1, m2, m3, m4, m5, m6, m7, m8, m9]

keep_member_indices:
[0, 2, 5, 8, 9]
```

则 retained KV 是：

```text
[m0, m2, m5, m8, m9]
```

这些 KV 的物理存储位置会变成：

```text
cache_position = [0,1,2,3,4]
```

但它们原来的模型 logical positions **不会被改成 0,1,2,3,4**。

KV 在产生时已经使用原本的 RoPE / model position 编码；V2 只改变存储排列与后续 physical addressing，不改历史语义坐标。

这也是为什么 P1 必须明确拆开：

```text
logical positions
vs
physical cache positions
```

---

# 3. vLLM v0.26.0 原生 runtime 给我们的约束

## 3.1 Scheduler allocation 是 ownership 层，不是 Worker execution view

在当前 P1 branch 中，Scheduler 的 canonical ownership 最终仍由：

```text
SingleTypeKVCacheManager.req_to_blocks
```

维护。

Worker 的 BlockTable 是：

```text
Scheduler ownership
        ↓ transport
Worker BlockTable
        ↓
attention / slot-mapping execution-addressing view
```

所以 Worker 不能因为自己 compact 完了，就直接把某些 block IDs 当作“已经可释放”。

正确 authority 顺序必须是：

```text
Worker physical transition success
        ↓ result
Scheduler validates canonical source state
        ↓
Scheduler changes req_to_blocks
        ↓
only then block becomes reclaimable/reusable
```

这也是为什么 `CompactionResultData` 不携带“authoritative freed IDs”。

---

## 3.2 allocation 的 physical base 已经被 P1 改成 E，而不是纯 logical computed tokens

P1 之前最重要的改造之一是：

```python
allocation_base = (
    request.effective_kv_len
    if request.effective_kv_len is not None
    else total_computed_tokens
)
num_tokens_main_model = allocation_base + num_new_tokens
```

这里意味着：当 request 已进入 P1 physical-state tracking 后，Scheduler 下一 step 申请 block 的依据是 physical E，而不是逻辑 `num_computed_tokens`。

因此在当前 MVP：

```text
E_before = 当前已存在的 physical prefix
q        = 当前 scheduled tokens

allocation target
= E_before + q
```

它与本轮 forward 后的 V2 source extent 一致：

\[
E_{source}=E_{before}+q
\]

这一步是 V2 能在真正物理收缩后继续生成的基础。

---

## 3.3 当前 MVP 下 active page count 必须精确覆盖 E_source

`SingleTypeKVCacheManager.allocate_new_blocks()` 的核心逻辑是：

```python
num_required_blocks = cdiv(num_tokens, self.block_size)
num_new_blocks = num_required_blocks - len(req_blocks)
```

在当前被冻结的 P1 MVP 中：

```text
prefix caching = OFF
spec decode = OFF
lookahead = 0
KV connector = OFF
single full-attention group
single GPU
```

因此：

\[
source\_num\_blocks
=
\left\lceil
\frac{E_{source}}{B}
\right\rceil
\]

这是一个 **P1 MVP invariant**，不是通用 vLLM invariant。

为什么不能泛化？

因为 vLLM 源码自己已经明确存在例如 speculative decoding 的情况：

```text
draft token block 已经分配
→ draft 后来 rejected
→ num_required_blocks 可能小于当前 num_req_blocks
```

所以文档与代码里必须写：

```text
P1 MVP invariant
```

而不能写成：

```text
vLLM universal invariant
```

这次 source-page-count audit 的价值就在这里：它把 T2A 原先较宽的：

```text
source_num_blocks * block_size >= source_E
```

收紧为与 M4 一致的：

```text
source_num_blocks == cdiv(source_E, block_size)
```

这样所有 deterministic source mismatch 都能在第一次 destructive KV write 之前失败。

---

# 4. M4 已经解决了什么，T2B 不应该重复解决什么

M4 的职责是 **data plane**。

最终 production primitive：

```text
compact_paged_kv_triton_2d(...)
```

语义是：

```text
paged KV source
+ exact active block IDs
+ E_source
+ keep_member_indices
        ↓
gather
        ↓
independent contiguous scratch
        ↓
writeback
        ↓
retained KV compacted into current physical prefix [0,K)
```

并返回：

```text
(new_effective_kv_len, new_num_blocks)
```

其中：

\[
K = |keep|
\]

\[
new\_num\_blocks=\left\lceil \frac{K}{B}\right\rceil
\]

M4 已经明确处理：

- arbitrary keep pattern；
- preserve original member order；
- overlap-safe scratch；
- stride-safe NHD/HND；
- final active block tail zeroing；
- physical block addressability；
- exact source page count；
- gather / writeback。

M4 **不应该**负责：

- request identity；
- batch/request row mapping；
- Scheduler plan staleness；
- Worker persistent E；
- Worker BlockTable lifecycle；
- Scheduler ownership；
- BlockPool free；
- Result transport。

所以 T2B 的本质不是“再实现一次 compaction”，而是：

> **把已经正确的数据面 primitive 放进正确的 runtime transaction。**

---

# 5. 为什么必须是 post-forward compaction

这是我们与 Tangram 对照后最确定的设计之一。

## 5.1 pre-forward 不成立

假设：

```text
E_t = 32
q   = 16
```

如果 V2 在 forward 前 compact：

```text
source 只能看到 32 个旧 KV
```

但当前 step 的 16 个新 KV 尚未产生。

这会导致两种错误：

1. 当前 q 不参与 retention / compaction；
2. V2 的 source contract 不再是“当前真实 physical sequence”。

而我们定义的 V2 是：

```text
current paged KV after forward
→ compact complete source sequence
```

所以它必须等待 forward 写完 KV。

---

## 5.2 太晚也不合适

另一个极端是等 sampling / output 以后再随便找地方 compact。

这样会把：

```text
forward success
sampling success
physical transition success
```

混在一起，failure boundary变得模糊。

更合理的 transaction seam 是：

```text
model.forward()
↓
KV source complete
↓
validate
↓
compact
↓
only then publish state to sampling/output path
```

如果 destructive compaction 本身失败，则当前 execute step直接失败，而不是先让上层认为 inference step已经完整成功。

---

# 6. Tangram 给出的最有价值架构先例

本轮参考的 Tangram GitHub snapshot 是：

```text
aiha-lab/tangram
snapshot analyzed:
6fa551fc8f6edcc118a2a39b3554ee1520e91edd
```

它只是 architecture reference。

不能直接认为：

```text
Tangram implementation
=
P1 implementation
```

但它提供了三个很重要的系统级先例。

---

## 6.1 Tangram 也把真正 KV reorder 放在 model.forward 之后

Tangram `CompressionExecutor` 的模块说明直接定义：

```text
Runs after model.forward
```

它在 compression boundary 上：

```text
forward
↓
_run_compression_layer_loop()
↓
CompressionExecutor.run_request()
↓
KV writeback
```

也就是说它与我们的判断一致：

> retention decision可能在 forward 内收集证据，但 destructive KV reorder 应在 forward 完成后执行。

这不是偶然的实现习惯，而是由 KV source completeness决定。

---

## 6.2 Tangram 明确分离 KV payload writeback 与 BlockTable compaction

Tangram `CompressionExecutor.run_request()` 的职责是：

```text
把 retained KV 写入新的 prefix
```

它明确说明：

```text
block_table is NOT mutated
```

随后 caller 再调用：

```text
compact_after_compress_all_layers(...)
```

去修改 metadata / block-table view。

也就是说 Tangram也是：

```text
PAYLOAD
first

METADATA
second
```

这对我们的 T2B 非常关键。

P1 因此冻结：

```text
for every relevant KV layer:
    M4 payload compaction

ALL success
    ↓
stage Worker BlockTable retained prefix
```

绝对不能在 layer 0刚 compact 完时就先提交新的 BlockTable，再继续 layer 1。

---

## 6.3 Tangram 对 effective length 采用“压缩行 absolute fold-in，普通行 additive increment”

Tangram 在 post-forward compression boundary 上大致做：

```text
compression boundary request:
    compute post-eviction kept length
    pending_eff_seq_lens[req] = absolute new physical length

non-boundary request:
    effective length += raw scheduled tokens
```

它的 runner 甚至显式：

```text
boundary rows excluded from normal increment
because their post-evict length has already been folded in
```

这与我们遇到的 double-fold-in问题高度一致。

但 P1 没有照搬 Tangram的实现位置。

Tangram直接维护自己的 `effective_seq_lens_cpu`，且其状态是：

```text
per layer
× per head group
× ragged BlockTable
```

P1 当前只有一个 request-level E，并且已有：

```text
post_update()
```

作为 persistent RequestState 的现成 fold-in点。

因此我们借鉴的是：

```text
“compressed request需要 absolute physical next-state，
不能继续走普通 additive physical progression”
```

而不是照抄 Tangram的状态容器。

---

# 7. Tangram 与 P1：哪些借鉴，哪些明确不借鉴

| 维度 | Tangram | P1 V2 | 决策 |
|---|---|---|---|
| compaction timing | post-forward | post-forward | **借鉴 / 一致** |
| KV writeback | executor | M4 Triton primitive | **同一分层思想** |
| metadata shrink | ragged BlockTable after writeback | dense BlockTable prefix after all-layer success | **简化** |
| physical length | per-layer/per-head-group | request-level single E | **不照搬** |
| retention/scoring | runtime compressor / scorer | T2B不负责 policy | **不照搬** |
| TP consensus | MAX reduce | TP=1 MVP | **暂不引入** |
| sliding layers | special lifecycle | out of scope | **不引入** |
| Worker freed IDs | Worker可产生 freed ids | Worker Result不作为 authoritative free source | **明确拒绝** |
| block ownership | Tangram扩展的 ragged flow | Scheduler `req_to_blocks` canonical authority | **保留 vLLM/P1 authority** |
| normal length fold-in | additive | additive | 一致 |
| compressed length fold-in | absolute post-compression state | `post_update` absolute E override | **借鉴语义，不照搬实现** |

因此 Tangram真正提供给 P1 的不是“代码”，而是一个很重要的 architecture validation：

```text
post-forward reorder
→ payload / metadata separation
→ compressed physical state uses absolute next-state
```

---

# 8. T2B 最核心冲突：double fold-in

这是最终设计抉择的起点。

假设：

```text
persistent E_before = 32
current q           = 16
```

forward 阶段：

```text
cache positions = 32 ... 47
```

所以：

\[
E_{source}=48
\]

现在 retention policy 给出：

```text
K = 20
```

M4 compact 完以后真实物理布局已经变成：

```text
valid KV = physical prefix [0,20)
```

因此正确 next state：

```text
E_next = 20
next append cache_position = 20
```

### 错误方案

如果 post-forward先：

```text
E = 20
```

然后原来的 `post_update()`继续：

```text
E += computed_delta
```

在 q=16、无 spec 的 MVP：

```text
computed_delta = 16
```

最终：

```text
E = 36
```

但真实 KV只有：

```text
[0,20)
```

这会直接制造：

```text
20 ... 35
```

的 physical hole。

下一 step：

```text
cache_position starts at 36
```

将与实际 compacted payload不一致。

attention read extent、slot mapping和后续 append都会被破坏。

这不是小的 off-by-one，而是 physical state model错误。

---

# 9. 三种候选方案的比较

## 方案 A：post-forward立即写 `E=K`，post_update用 suppression mask

流程：

```text
post-forward:
    compact KV
    E := K

post_update:
    logical += delta
    if not compacted:
        E += delta
```

### 优点

- post-forward时 GPU persistent E立刻反映 K；
- 语义直接。

### 问题

- 同一个 persistent E有两个 writer：
  - post-forward compaction path；
  - post_update normal path。
- 需要新增“skip physical fold-in”控制；
- runtime state ownership更分散；
- 更容易出现未来某个 caller忘记 suppression。
- 对当前实际 lifecycle并没有必要，因为 forward后到 post_update前不存在需要用 K 做下一轮 attention的 consumer。

所以：

```text
CORRECT
但不是首选
```

---

## 方案 B：payload post-forward compact，persistent E 在 post_update absolute override

流程：

```text
post-forward:
    compact KV payload
    stage BlockTable
    produce override K

post_update:
    logical += delta

    if override_valid:
        E = K
    else:
        E += delta
```

### 优点

- `post_update()`继续是 persistent E唯一 final fold-in owner；
- normal path不变；
- V1 path不变；
- V2只增加一种 physical transition mode：
  - incremental
  - absolute replacement。
- 与 Tangram“boundary row不再做普通 additive physical increment”语义一致；
- 不需要先写 E再 suppression；
- 更清楚地表达：
  ```text
  logical progression != physical progression
  ```

### 结论

```text
CHOSEN
```

---

## 方案 C：先正常 `E += delta`，之后再 repair 成 K

流程：

```text
post_update:
E += delta

later:
E = K
```

这相当于先让 runtime进入一个已知错误 state，再依赖后处理修复。

它的问题是：

- intermediate state无效；
- failure boundary差；
- 多 writer；
- 顺序要求隐蔽；
- 很难证明所有未来 consumer都在 repair之后。

因此：

```text
REJECTED
```

---

# 10. 最终冻结的 post_update 语义

最终状态转移必须是：

```python
computed_delta = query_len - num_rejected

# logical state
if computed_delta != 0:
    num_computed_tokens += computed_delta

# physical state
if effective_kv_len_override_valid:
    effective_kv_len = effective_kv_len_override
elif computed_delta != 0:
    effective_kv_len += computed_delta
```

这里最重要的一条：

> **absolute override 不能嵌套在 `computed_delta != 0` 里面。**

错误：

```python
if computed_delta != 0:
    num_computed += computed_delta
    if override_valid:
        E = K
```

因为：

```text
physical replacement
```

和：

```text
logical computed delta
```

是两个独立的状态事件。

即便未来出现：

```text
computed_delta == 0
```

只要 physical payload transition已经成功：

```text
E = K
```

仍然必须提交。

数学上：

\[
L_{next}=L_t+\Delta
\]

而：

\[
E_{next}=
\begin{cases}
K,& \text{if V2 compaction committed}\\
E_t+\Delta,& \text{otherwise}
\end{cases}
\]

这就是 P1 从“普通 vLLM状态”走向“logical/physical decoupling”的核心。

---

# 11. 为什么 override state 应按 batch_idx 表示

`post_update` 的执行天然围绕当前 batch：

```text
batch_idx
↓
idx_mapping
↓
req_state_idx
```

所以最自然的控制数据是：

```text
effective_kv_len_override_values[batch_idx]
effective_kv_len_override_valid[batch_idx]
```

例如 mixed batch：

```text
batch row 0: request A normal
batch row 1: request B V2 K=20
batch row 2: request C normal
```

控制状态：

```text
valid = [0, 1, 0]
value = [0,20, 0]
```

post_update：

```text
A:
L += delta
E += delta

B:
L += delta
E = 20

C:
L += delta
E += delta
```

这比按 `req_state_idx`额外建立一个 persistent-size control buffer更贴合当前 kernel execution topology。

---

# 12. `_PreparedCompaction` 的最终定位

T2A 当前定义的 `_PreparedCompaction` 是：

```text
Scheduler Plan
        ↓
Worker resolve / validate
        ↓
immutable execution descriptor
```

其语义角色是正确的，应冻结。

典型字段：

```text
request_id
req_state_idx
batch_idx
source_effective_kv_len
source_num_blocks
block_ids
block_size
keep_member_indices
new_effective_kv_len = K
new_num_blocks
```

但是：

```text
PreparedCompaction
```

不应该在最终 T2B 中继续充当：

```text
execute_model → sample_tokens
```

的 trigger。

原因是：

```text
Prepared = “已经证明可以执行”
Result   = “已经执行成功”
```

真正 destructive compaction应在 `execute_model()` post-forward seam立即消费 Prepared。

因此最终：

```text
PreparedCompaction
= execute_model-local transient

ExecuteModelState
= completed compaction result + E override carrier
```

这也符合 vLLM MRV2 本身对 `ExecuteModelState` 的定位：

```text
execute_model()
和
subsequent sample_tokens()
之间的 ephemeral state
```

它不是一个延迟执行 destructive KV transform 的任务队列。

---

# 13. `_execute_v2_compactions()` 应该做什么

建议其职责严格限定为：

```text
validated PreparedCompaction
→ actual Worker physical transition
```

概念结构：

```python
def _execute_v2_compactions(prepared_compactions):
    results = []
    overrides = ...

    for prepared in prepared_compactions.values():

        block_ids = to_device(prepared.block_ids)
        keep = to_device(prepared.keep_member_indices)

        # destructive boundary starts here
        for kv_cache in self.kv_caches:
            new_E, new_num_blocks = compact_paged_kv_triton_2d(
                kv_cache,
                block_ids,
                prepared.source_effective_kv_len,
                keep,
            )

            assert new_E == prepared.new_effective_kv_len
            assert new_num_blocks == prepared.new_num_blocks

        retained_block_prefix = prepared.block_ids[:prepared.new_num_blocks]

        self.block_tables.append_block_ids(
            prepared.req_state_idx,
            (list(retained_block_prefix),),
            overwrite=True,
        )

        overrides.valid[prepared.batch_idx] = True
        overrides.value[prepared.batch_idx] = prepared.new_effective_kv_len

        results.append(
            CompactionResultData(...)
        )

    return results, overrides
```

这只是结构草图，不是要求 Codex逐字照抄。

真正必须冻结的是责任边界与顺序。

---

# 14. 为什么 all-layer payload success 必须早于 BlockTable stage

假设 32 层模型。

错误顺序：

```text
layer 0 compact
↓
BlockTable shrink
↓
layer 1 compact
↓
...
```

如果 layer 5失败：

```text
BlockTable已经声称只剩新的 prefix
但 layer 5..31仍是旧 payload
```

metadata与payload马上分裂。

正确顺序：

```text
layer 0 compact
layer 1 compact
...
layer 31 compact
↓
ALL SUCCESS
↓
stage BlockTable prefix
↓
produce Result / override
```

这样即使目前没有 rollback，至少 metadata不会在 payload transaction尚未完成时提前声称成功。

---

# 15. BlockTable 为什么只保留 prefix，而不是重排 block IDs

canonical example：

```text
B = 4
source E = 10
old row = [B7, B2, B11]

keep members = [0,2,5,8,9]
K = 5
new_num_blocks = 2
```

M4写回的是：

```text
B7:
[m0,m2,m5,m8]

B2:
[m9,0,0,0]

B11:
旧数据可以仍然存在
但已不属于 active destination prefix
```

所以 Worker下一状态：

```text
active BlockTable row
= [B7,B2]

num_blocks
= 2
```

这里不需要寻找新的物理 blocks。

我们复用旧 row 的前：

\[
\left\lceil K/B \right\rceil
\]

个 page作为 destination pages。

因此：

```text
new_row = old_row[:new_num_blocks]
```

这与 M4的“compact into current physical prefix”完全一致。

---

# 16. BlockTable staging 为什么可以延迟到下一 execute_model GPU materialization

当前 Worker BlockTable API 的重要语义是：

```text
overwrite=True
→ 从 row offset 0 stage 新 prefix
→ CPU/host-visible active num_blocks更新
→ GPU table真正 materialize 在 apply_staged_writes()
```

T2B发生在：

```text
forward 已结束
```

之后。

同一 step后续是：

```text
sampling
bookkeeping
post_update
output
```

没有第二次 attention forward需要读取新的 BlockTable。

因此：

```text
same-step immediate GPU BlockTable rewrite
```

不是正确性要求。

我们需要的是：

```text
before next model forward
new BlockTable GPU view must be visible
```

而下一次 `execute_model()`入口本来就会 `apply_staged_writes()`。

所以使用现有 staged path比新增一个“立即GPU重写 BlockTable” API更符合 runtime。

---

# 17. source page count 为什么必须在 T2A exact fence

M4 的 data-plane contract明确要求：

```text
len(block_ids)
=
ceil(E_source / block_size)
```

原因是 M4 的 source member：

```text
member i
→ page index i // B
→ block_ids[page index]
```

它需要的是精确覆盖 source的 page row，不是 allocator capacity upper bound。

如果 T2A只检查：

```text
source_num_blocks * B >= E_source
```

那么可能发生：

```text
Request A:
valid
→ layer compaction已执行

Request B:
source row比 M4 expected pages多
→ M4 validation失败
```

于是本来可以在 destructive boundary之前发现的问题，被推迟到 partial mutation之后。

因此：

```text
T2A exact source-page guard
```

属于 transaction safety，不只是重复 M4 validation。

最终：

```python
expected_source_num_blocks = cdiv(
    source_effective_kv_len,
    block_size,
)

if source_num_blocks != expected_source_num_blocks:
    fail before first M4 mutation
```

---

# 18. failure semantics：我们现在能保证什么，不能保证什么

T2B MVP **没有 rollback**。

必须明确分成两个 failure region。

## Region A：第一次 M4 destructive write 前

包括：

```text
request identity mismatch
batch mapping stale
source E mismatch
source page count mismatch
keep invalid
block IDs invalid
layer shape/device incompatible
```

要求：

```text
FAIL
→ no V2 payload mutation
```

这就是 T2A存在的意义。

---

## Region B：第一次 M4 destructive write 后

例如：

```text
layer 0 success
layer 1 success
layer 2 kernel/runtime failure
```

此时：

```text
部分 KV payload可能已经 compact
```

MVP不提供 inverse transform / rollback。

所以只能：

```text
abort step
surface error
do NOT publish CompactionResultData success
do NOT claim metadata transaction complete
do NOT claim payload unchanged
```

同样，多 request情况下：

```text
request A complete
request B runtime failure
```

也可能存在 A payload已经被修改。

因此 T2B的 transaction性质是：

```text
prevalidated destructive transaction
but not atomic rollback transaction
```

这必须在项目文档和简历表述里说准确。

---

# 19. 为什么 Worker 不直接 free trailing blocks

V2 Worker已经知道：

```text
old row [B7,B2,B11]
new active prefix [B7,B2]
```

看起来它似乎可以直接：

```text
free B11
```

但这是错误 authority。

Worker拥有：

```text
execution-addressing view
```

Scheduler拥有：

```text
canonical ownership
```

在 Scheduler还没有做：

```text
req_to_blocks:
[B7,B2,B11]
→
[B7,B2]
```

之前，B11在 canonical state里仍然属于该 request。

如果 Worker先 free：

```text
BlockPool可能把 B11复用给别的 request
但 Scheduler旧 ownership仍然指向 B11
```

这会造成 use-after-reuse级别的问题。

所以 T2B 只输出：

```text
old source descriptor
new E
new num_blocks
```

T3由 Scheduler根据自己的 canonical row推导 trailing removed blocks。

这个设计比让 Worker回传 authoritative freed IDs更稳。

---

# 20. `CompactionResultData` 为什么已经足够

T1定义的 Result：

```text
request_id
expected_source_effective_kv_len
expected_source_num_blocks
new_effective_kv_len
new_num_blocks
step_seq(optional)
```

已经足够 T3完成 ownership reconcile。

Scheduler知道：

```text
canonical current row
```

Result告诉 Scheduler：

```text
Worker是基于哪个 source状态完成transition
以及新的 active prefix长度是多少
```

Scheduler于是可以：

```text
validate old E / old block count
↓
old_row[:new_num_blocks] 作为 retained prefix
↓
old_row[new_num_blocks:] 作为 removed suffix
↓
publish canonical row
↓
safe-free removed suffix
```

所以 T2B不需要扩大 Result contract。

---

# 21. normal / V1 / V2 三条 physical progression 如何统一

最终 Worker physical E 有三种情况。

### Normal

```text
E_before
↓ forward q
↓ post_update
E_next = E_before + delta
```

### V1

```text
E_before
↓ pre-forward reclaim
E = R
↓ forward q
↓ post_update
E_next = R + delta
```

### V2

```text
E_before
↓ forward q
source = E_before + q
↓ post-forward compaction
payload prefix = [0,K)
↓ post_update
E_next = K
```

统一表达：

```text
normal/V1:
physical transition mode = incremental

V2:
physical transition mode = absolute
```

这比在 runtime中继续假设：

```text
logical_delta == physical_delta
```

更准确。

---

# 22. 一个完整的 canonical T2B example

设：

```text
block_size B = 4

before step:
logical L = 10
physical E = 10

BlockTable:
[B7, B2, B11]

current q = 2
```

Scheduler先确保：

```text
source extent after forward
= E + q
= 12

source_num_blocks
= ceil(12/4)
= 3
```

forward写入：

```text
cache_position 10
cache_position 11
```

现在 source physical members：

```text
[m0,m1,m2,m3,
 m4,m5,m6,m7,
 m8,m9,m10,m11]
```

假设 keep：

```text
[0,2,5,8,9]
```

则：

```text
K = 5
new_num_blocks = 2
```

M4：

```text
gather scratch:
[m0,m2,m5,m8,m9]

writeback:

B7 = [m0,m2,m5,m8]
B2 = [m9,0,0,0]

B11 physical bytes不要求立即清除
```

Worker stage：

```text
BlockTable active prefix:
[B7,B2]

num_blocks=2
```

然后：

```text
E override:
batch row X → K=5
```

sampling之后：

```text
logical L_next:
10 + 2 = 12

physical E_next:
5
```

注意：

```text
L = 12
E = 5
```

完全合法。

这不是状态损坏，而是 P1真正要建立的：

```text
logical history
与
physical retained KV state
```

之间的分离。

下一 step append：

```text
cache_position = 5
```

因此落在：

```text
B2 offset 1
```

正好接在 `m9`之后。

---

# 23. T2B 预计最小 code touch points

这不是最终 patch specification，但当前最合理的修改面应保持很小。

### `vllm/v1/worker/gpu/model_runner.py`

职责：

- T2A exact source-page guard；
- `_execute_v2_compactions()` orchestration；
- post-forward hook；
- Prepared变为 execute-local；
- build Result；
- build/carry E overrides；
- `ExecuteModelState` transport completed state。

这里可以放 orchestration，但不应该把 Triton data-plane重新实现进去。

vLLM v0.26.0 自己对这个核心 runner文件也强调“保持稳定、尽量最小”，因此 feature-specific重逻辑继续留在 `kv_compaction.py` 是更符合 upstream code-style意图的。

### `vllm/v1/worker/gpu/input_batch.py`

职责：

- 扩展 `post_update()` / `_post_update_kernel`；
- 支持 batch-indexed：
  ```text
  override_valid
  override_value
  ```
- normal/V1默认行为完全保持。

### `vllm/v1/worker/gpu/kv_compaction.py`

预期：

```text
NO CHANGE
```

除非 integration真正发现已冻结 M4 primitive存在 bug。

T2B不能借 integration之名重写 M4。

### `vllm/v1/core/...`

T2B预期：

```text
NO Scheduler ownership/free mutation
```

Scheduler reconcile属于 T3。

---

# 24. T2B 必须覆盖的测试

最低测试矩阵应覆盖：

| Test | 要证明什么 |
|---|---|
| canonical single-request | M4→BlockTable→Result→E override完整闭环 |
| multi-layer | 所有 layer 都 compact |
| NHD/HND M4 regression | integration不破坏 stride-safe contract |
| `req_state_idx != batch_idx` | 两套 index domain不混 |
| mixed batch normal + V2 | 只有V2 row走 absolute E |
| keep-all | 合法 no-op / state一致 |
| source E stale | T2A pre-mutation fail |
| source block count stale | T2A pre-mutation fail |
| excess source pages | exact-page fence fail |
| invalid keep | pre-mutation fail |
| tail zero | final active page deterministic |
| BlockTable prefix | `new_row=old_row[:new_num_blocks]` |
| post_update override | `E=K`而不是`K+q` |
| `computed_delta==0 + override` | override仍提交 |
| V1 regression | V1仍然 `R+delta` |
| normal regression | 普通请求仍 `E+delta` |
| M5-T1 transport regression | Plan/Result contract不破坏 |
| M4 regression | 52类 data-plane correctness仍成立 |
| failure injection | runtime failure后不得发布成功 Result |

最重要的测试不是“多跑”，而是：

```text
把每一个 frozen semantic contract钉到测试上
```

---

# 25. 最终冻结的 Design Decisions

下面可以直接作为 T2B implementation contract 的 decision ledger。

### D-T2B-01 — Timing

```text
KV payload compaction
MUST execute immediately post-forward.
```

不是 pre-forward，不延迟到 sampling trigger。

### D-T2B-02 — Source

```text
source_effective_kv_len
= current post-forward physical extent
```

不是 stale persistent E。

### D-T2B-03 — Exact pages

当前 P1 MVP：

```text
source_num_blocks
= cdiv(source_effective_kv_len, block_size)
```

必须在 destructive mutation前验证。

### D-T2B-04 — Prepared descriptor

`_PreparedCompaction` 是 immutable validated execution descriptor。

### D-T2B-05 — Prepared lifetime

T2B完成后：

```text
Prepared
= execute_model-local
```

不作为 `sample_tokens()` 的 destructive-operation trigger。

### D-T2B-06 — Data plane

唯一 production payload primitive：

```text
compact_paged_kv_triton_2d()
```

T2B不重新实现 gather/writeback。

### D-T2B-07 — All-layer ordering

```text
all KV layer payload compactions
BEFORE
BlockTable metadata stage
```

### D-T2B-08 — Destination row

```text
new Worker active row
= old active row[:new_num_blocks]
```

### D-T2B-09 — BlockTable API

优先复用现有：

```text
overwrite=True
```

staged row replacement机制。

### D-T2B-10 — Persistent E ownership

```text
post_update()
remains the single final persistent E fold-in point.
```

### D-T2B-11 — Physical transition modes

```text
normal/V1 → incremental
V2        → absolute replacement
```

### D-T2B-12 — Override precedence

```text
if override_valid:
    E = K
else:
    E += computed_delta
```

### D-T2B-13 — Logical independence

```text
num_computed_tokens += computed_delta
```

不因 V2 physical compaction而 suppress。

### D-T2B-14 — Override independent from delta

即使：

```text
computed_delta == 0
```

V2 absolute E override仍必须生效。

### D-T2B-15 — Result publication

只有：

```text
payload success
+
Worker metadata stage success
```

之后才创建/发布 success `CompactionResultData`。

### D-T2B-16 — Scheduler boundary

T2B：

```text
NO canonical req_to_blocks mutation
NO BlockPool free
NO reuse claim
```

这些属于 T3/T4。

### D-T2B-17 — Failure semantics

第一次 M4 write以后：

```text
NO rollback guarantee
```

失败必须 abort，禁止伪装 atomic。

### D-T2B-18 — Performance claim

T2A/T2B当前允许少量 control-plane GPU↔CPU / CPU→GPU tensor materialization。

没有 profiling之前：

```text
NO performance claim
```

---

# 26. 下一阶段 T3 / T4 的边界

T2B结束后，不代表 physical reclamation真正闭环。

### M5-T3 — Scheduler Canonical Reconciliation

目标：

```text
CompactionResultData
↓
validate Scheduler canonical source state
↓
derive retained prefix
↓
req_to_blocks = final dense prefix
↓
Scheduler effective_kv_len = K
```

注意 Scheduler这里也必须是：

```text
E = K
```

不能沿用 V1：

```text
K + num_tokens_scheduled
```

因为 V2的 K已经包含当前 step q经过 retention后的最终 post-forward physical state。

---

### M5-T4 — Safe Free / Reuse / Continued Generation

目标：

- removed suffix何时真正 free；
- deferred-free fence；
- refcount；
- block reuse；
- continued generation；
- next append = K；
- repeated compaction；
- preemption / request finish边界。

真正的“物理显存收益”要到 T3/T4完成才成立。

T2B只证明：

```text
payload + Worker physical execution state
```

能正确 transition。

---

# 27. 整个 P1 V2 到目前为止的架构图

```text
                         P1 V2
                          │
             ┌────────────┴────────────┐
             │                         │
       Control Plane               Data Plane
             │                         │
             │                         │
     M5-T1 Plan/Result             M4 Triton
        transport               gather/scratch/writeback
             │                         │
             └────────────┬────────────┘
                          │
                       M5-T2A
                Source Fence / Prepare
                          │
                          ▼
                 _PreparedCompaction
                          │
                          ▼
                       M5-T2B
             Post-forward Worker Transaction
                          │
           ┌──────────────┼──────────────┐
           │              │              │
       KV payload     BlockTable      E override
       compaction       stage         + Result
           │              │              │
           └──────────────┴──────────────┘
                          │
                          ▼
                      post_update
                          │
                          ▼
                      M5-T3
              Scheduler canonical ownership
                          │
                          ▼
                      M5-T4
                  safe free / reuse
```

这张图也说明了为什么项目不是“只写一个 kernel”。

真正困难的是：

```text
data-plane transform
×
runtime state transition
×
ownership authority
×
lifecycle safety
```

四者必须同时正确。

---

# 28. SOURCE_FACT 与 DESIGN_DECISION 的最终区分

| 内容 | 分类 |
|---|---|
| Scheduler在P1中以 `effective_kv_len` 作为 allocation base | `SOURCE_FACT / P1 EXISTING` |
| current q 在 forward期间写入 KV | `SOURCE_FACT` |
| post-forward source extent包含当前 q | `DERIVED SOURCE FACT` |
| M4要求 exact source page count | `SOURCE_FACT / M4 CONTRACT` |
| M4使用 scratch避免overlap | `FROZEN DATA-PLANE DESIGN` |
| vLLM Worker BlockTable服务slot mapping/addressing | `SOURCE_FACT` |
| `ExecuteModelState`连接 execute与sample | `SOURCE_FACT` |
| Tangram post-forward执行compression writeback | `REFERENCE SOURCE FACT` |
| Tangram payload writeback后再compact block table | `REFERENCE SOURCE FACT` |
| Tangram boundary row absolute physical length、普通row additive | `REFERENCE SOURCE FACT` |
| T2B选择post-forward hook | `DESIGN_DECISION supported by source/reference` |
| T2B选择post_update absolute E override | `DESIGN_DECISION` |
| Worker不authoritatively free blocks | `DESIGN_DECISION / AUTHORITY CONTRACT` |
| Prepared只在execute_model内消费 | `DESIGN_DECISION` |
| BlockTable使用old prefix作为destination row | `FROZEN V2 SEMANTIC DECISION` |
| no rollback MVP | `EXPLICIT LIMITATION` |

这张表非常重要。

以后做项目汇报时，不要把：

```text
“我们觉得这样好”
```

和：

```text
“vLLM源码本来就这样”
```

混在一起。

---

# 29. 最终项目状态

截至当前设计冻结：

```text
P1 V1/Core
= PASS / ACCEPTED / FROZEN

M4 Data Plane
= PASS / CLOSED / FROZEN

M5-T1
Plan / Result Contract + Transport
= PASS / CLOSED / FROZEN

M5-T2A
Source Fence + Execution Preparation
= PASS / CLOSED / FROZEN
with P1 MVP qualification

M5-T2B
Real Post-Forward Compaction
+ Worker Physical-State Commit
= DESIGN FROZEN
= READY FOR CODEX IMPLEMENTATION

NEXT:
1. T2A exact-page guard alignment patch
2. _execute_v2_compactions()
3. post_update absolute E override
4. Result transport
5. focused correctness/regression tests
6. Web acceptance review
```

---

# 30. 一句话总结 T2B

如果最后要向别人解释这一阶段，可以这样讲：

> **M5-T2B 不是在 attention 里做“删除 token”，而是在本轮 forward 已经产生完整 KV source 后，用 M4 将保留 KV 物理压缩到 paged-cache 前缀，再更新 Worker 的下一步寻址视图；逻辑 token 进度仍正常推进，而 physical `effective_kv_len` 在 `post_update` 中用压缩后的 K 做 absolute commit。Scheduler 的 canonical block ownership 与真正 free 则延后到 T3/T4。**

这个表述基本覆盖了这一阶段真正的系统价值。

---

# 31. 证据与源码索引

## P1 本地审计 / 当前分支

- `P1-M5-T2B-FINAL-DESIGN-WALKTHROUGH.log`
- `P1-M5-T2B-SOURCE-PAGE-COUNT-AUDIT.log`
- `P1-M5-T2B-SOURCE-PAGE-COUNT-AUDIT(1).log`
- `P1-M5-T2A-KV-LAYOUT-AUDIT.md`
- `P1-M5-T2A-CODE-CHANGE-RECORD.md`

重点源码：

```text
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/input_batch.py
vllm/v1/worker/gpu/block_table.py
vllm/v1/worker/gpu/kv_compaction.py

vllm/v1/core/kv_cache_manager.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/sched/scheduler.py
vllm/v1/core/sched/output.py
vllm/v1/outputs.py
```

## vLLM upstream

Repository:

```text
https://github.com/vllm-project/vllm
tag: v0.26.0
```

重点看：

```text
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/input_batch.py
vllm/v1/worker/gpu/block_table.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/single_type_kv_cache_manager.py
```

注意：P1 当前分支在 v0.26.0 MRV2 上增加了 `effective_kv_len` 等 logical/physical separation，因此不能把 P1新增行为误写成 pristine upstream behavior。

## Tangram reference

Repository:

```text
https://github.com/aiha-lab/tangram
snapshot analyzed in this review:
6fa551fc8f6edcc118a2a39b3554ee1520e91edd
```

重点源码：

```text
vllm/v1/attention/compression/executor.py
vllm/v1/attention/compression/eviction_writeback.py
vllm/v1/worker/compression_model_runner_mixin.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/ragged_block_table.py
```

Tangram仅用于验证和启发：

```text
post-forward compression timing
payload → metadata ordering
compressed physical length absolute fold-in
```

P1 不照搬其：

```text
ragged per-layer/per-group state
compression policy/scorer
sliding-layer lifecycle
TP consensus
Worker-authoritative freed-id flow
```

---

## Final Freeze

```text
M5-T2B DESIGN FREEZE:

SOURCE:
post-forward complete physical KV extent

PRECONDITION:
all T2A deterministic validation passes
including exact page count

MUTATION:
M4 compact all layer KV payloads

METADATA:
stage retained Worker BlockTable prefix only after payload success

PHYSICAL LENGTH:
post_update absolute E override = K

LOGICAL LENGTH:
normal computed_delta progression remains

RESULT:
publish CompactionResultData only after Worker transition success

OWNERSHIP/FREE:
not T2B; Scheduler T3/T4 only

ROLLBACK:
not provided in MVP
```
