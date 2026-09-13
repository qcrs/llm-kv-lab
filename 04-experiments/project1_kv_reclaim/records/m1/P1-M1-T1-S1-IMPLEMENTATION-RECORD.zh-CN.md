# P1-M1-T1-S1 实现记录

> Slice：`P1-M1-T1-S1 — MRV2 Effective-KV Persistent State Plumbing`  
> Parent Task：`P1-M1-T1 — Logical / Physical State Contract`  
> 状态：Slice execution 已完成，等待 Parent Task 级 Web review；不等于 M1 Gate PASS。

## 1. 文档定位

这是 S1 的 canonical implementation record。此前 S1 记录没有丢失，而是只保存在 experiment record 目录中，因此没有和 S2/S3 一起出现在项目 `docs/30-evaluation/` 导航下。

相关资料按用途区分：

| 类型 | 路径 | 用途 |
|---|---|---|
| 本文 | `04-experiments/project1_kv_reclaim/records/m1/P1-M1-T1-S1-IMPLEMENTATION-RECORD.zh-CN.md` | S1 正式实现摘要与复盘入口 |
| 原始 code change record | `04-experiments/project1_kv_reclaim/records/m1-t1-s1/20260825-170000_m1-t1-s1-code-change-record.md` | S1 执行当时的文件、行号、测试记录 |
| 深入学习笔记 | `../../../../01-projects/project1-physical-kv-reclamation-vllm/learning/00-foundations/P1-M1-T1-S1-MRV2-num-computed-tokens与Effective-KV-State生命周期详解-v2.md` | Scheduler/Worker/GPU 状态生命周期教学 |
| raw evidence | `04-experiments/project1_kv_reclaim/raw/m1-t1-s1/` | 未翻译的命令、source audit 和测试输出 |

## 2. Source Identity

```text
worktree: /home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim
branch: p1/physical-kv-reclaim-v026
base HEAD: 568afb3a13806beb53bb2e6bd518269357b237c0
runner: vLLM V1 Engine + MRV2
Slice ID: P1-M1-T1-S1
```

S1 完成后没有 commit；后续 S2/S3 和 review-fix 修改继续叠加在同一 working tree 上。

## 3. S1 解决的问题

Upstream normal execution 中通常隐含：

```text
logical progress == physical KV progress
```

P1 reclaim 后必须允许：

```text
logical progress != physical KV progress
```

S1 只建立 request-lifetime、GPU-resident 的 physical progress state：

```text
RequestState.effective_kv_len
```

它与 logical `num_computed_tokens` 使用独立 storage。S1 不让它驱动 slot mapping 或 attention，也不执行 reclaim。

## 4. Approved State Contract

| State | Owner | Storage | 生命周期 | S1 语义 |
|---|---|---|---|---|
| `num_computed_tokens.gpu` | MRV2 `RequestState` | `StagedWriteTensor`, GPU, `int32 [max_num_reqs]` | request-lifetime | logical model progress |
| `num_computed_tokens_np` | MRV2 `RequestState` | CPU NumPy mirror | request-lifetime | optimistic logical CPU mirror |
| `effective_kv_len.gpu` | MRV2 `RequestState` | 独立 `StagedWriteTensor`, GPU, `int32 [max_num_reqs]` | request-lifetime | forward-entry physical KV progress |

S1 明确没有新增 `effective_kv_len_np`。两个 GPU tensor 在 normal execution 中共享同一个 `computed_delta`，但不共享 storage，也不执行绝对值同步。

## 5. 文件级修改

### `vllm/v1/worker/gpu/states.py`

修改：

- `RequestState.__init__` 新增独立 `effective_kv_len: StagedWriteTensor`；
- `add_request()` 使用 initial `num_computed_tokens` 初始化 physical state；
- `apply_staged_writes()` publication 时提交该 staged write；
- `remove_request()` 不增加 zeroing，下一次 `add_request()` 显式覆盖复用 row。

原因：request 创建时尚未发生 reclaim，因此 initial logical 与 physical 相等；之后必须允许两者独立分叉。

### `vllm/v1/worker/gpu/input_batch.py`

修改：

- `_post_update_kernel()` / `post_update()` 增加 `effective_kv_len` 参数；
- 使用同一个：

  ```text
  computed_delta = query_len - num_rejected
  ```

  分别执行：

  ```text
  num_computed_tokens += computed_delta
  effective_kv_len    += computed_delta
  ```

原因：normal execution 同时增加 logical progress 和 physical occupancy，但未来 reclaim 可以只改变 physical state。

### `vllm/v1/worker/gpu/model_runner.py`

修改：normal `postprocess_sampled() -> post_update()` 调用传入：

```text
self.req_states.effective_kv_len.gpu
```

原因：把 persistent physical state 接入当前 fixed `PP=1` generation completion path。

### `tests/v1/worker/test_gpu_effective_kv_state.py`

S1 创建该 targeted test 文件，最初加入 T1-T4。后续 S2/S3 在同一个文件继续增加测试，因此当前文件行数和 test 数量不能反推 S1 当时的独立 diff。

## 6. 函数与 Kernel 签名变化

修改前：

```text
post_update(idx_mapping, num_computed_tokens, last_sampled_tokens, ...)
```

修改后：

```text
post_update(
    idx_mapping,
    num_computed_tokens,
    effective_kv_len,
    last_sampled_tokens,
    ...,
)
```

`effective_kv_len` 是独立 GPU tensor。新增参数表示“应用相同 normal delta”，不表示 `effective_kv_len = num_computed_tokens`。

## 7. 行为变化

```text
Before
computed_delta
    └─> logical num_computed_tokens += delta

After S1
computed_delta
    ├─> logical  num_computed_tokens += delta
    └─> physical effective_kv_len    += delta
```

分叉 oracle：

```text
before: logical=128, physical=64
delta:  1
after:  logical=129, physical=65
```

这个 oracle 证明 physical state 不是 logical counter 的 alias 或 absolute mirror。

## 8. S1 Targeted Tests

S1 closure 当时共有 4 个 tests：

| Test | Invariant | S1 结果 |
|---|---|---|
| `test_initialization` | add + staged commit 后为 `37/37` | PASS |
| `test_baseline_advancement` | `128/128 -> 129/129` | PASS |
| `test_independent_divergence_advancement` | `128/64 -> 129/65` | PASS |
| `test_request_slot_reuse_overwrites_state` | req_idx 复用后由新 request 覆盖为 `37/37` | PASS |

S1 当时结果：

```text
4 passed, 15 warnings in 3.01s
```

后续 S2/S3 review-fix 最终在同一文件达到 13 tests；这是累计结果，不应回写成“S1 有 13 tests”。

## 9. S1 明确没有完成什么

- 没有让 `effective_kv_len` 成为 KV write-address consumer；
- 没有增加 `cache_positions`；
- 没有增加 `effective_kv_seq_lens`；
- 没有修改 `BlockTables` 或 slot mapping；
- 没有修改 attention metadata / FlashAttention；
- 没有执行 block reclaim、ownership commit、safe-free 或 reuse；
- 没有增加 worker ACK；
- 没有修改 Scheduler、`KVCacheManager`、`BlockPool`；
- 没有新增 CPU physical mirror。

这些边界分别由后续 S2/S3 或尚未批准的 future Slice 处理。

## 10. Evidence Index

```text
04-experiments/project1_kv_reclaim/raw/m1-t1-s1/20260825-142836_m1-t1-s1-state-lifecycle-audit.log
04-experiments/project1_kv_reclaim/raw/m1-t1-s1/20260825-164028_m1-t1-s1-implementation.log
04-experiments/project1_kv_reclaim/records/m1-t1-s1/20260825-170000_m1-t1-s1-code-change-record.md
```

## 11. 与 S2/S3 的关系

```text
S1
effective_kv_len persistent physical base
    |
    v
S2
cache_positions -> KV slot mapping / write address
    |
    v
S3
effective_kv_seq_lens -> normal FA2 physical read extent
```

完整 Parent Task 导航见 `P1-M1-T1-IMPLEMENTATION-INDEX.zh-CN.md`。

## 12. Next Allowed Action

S1 已作为已执行 Slice 被后续 S2/S3 继承。当前整个 `P1-M1-T1` 的 workflow authority 仍以 `PROJECT_STATE.md` 为准：

```text
Next Allowed Action: WEB_REVIEW_CURRENT_SLICE
```

本文只是补齐 canonical 文档归档，不批准新的 implementation Slice。
