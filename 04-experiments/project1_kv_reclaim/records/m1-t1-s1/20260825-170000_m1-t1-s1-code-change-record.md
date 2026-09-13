# P1-M1-T1-S1 代码变更记录

## 1. 源码身份

- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/physical-kv-reclaim-v026`
- Base HEAD：`568afb3a13806beb53bb2e6bd518269357b237c0`
- Final state：仅包含本 Slice 批准的 source/test 修改，worktree 为 dirty
- Slice ID：`P1-M1-T1-S1`

## 2. 变更文件

### 文件
`vllm/v1/worker/gpu/states.py`

### 变更区域

- `RequestState.__init__`，L60-L62
- `RequestState.add_request`，L112-L115
- `RequestState.apply_staged_writes`，L119-L125

### 具体变更

- 新增独立的 GPU resident `effective_kv_len: StagedWriteTensor`。
- shape、`torch.int32` dtype 和 device 与 `num_computed_tokens` 对齐。
- `add_request()` 使用 request 初始 `num_computed_tokens` 初始化它。
- `apply_staged_writes()` 提交 effective state 的 staged write。

### 变更原因

这是 worker execution-side、request-lifetime 的 physical progress state。发生 reclaim
之前，initial physical progress 与 initial logical progress 相等。本 Slice 没有新增
CPU mirror，因为当前没有 CPU physical-state consumer。

### 文件
`vllm/v1/worker/gpu/input_batch.py`

### 变更区域

- `_post_update_kernel`，L459-L519
- `post_update`，L522-L562

### 具体变更

- 为现有 Triton post-update kernel 增加 `effective_kv_len_ptr`。
- 使用同一个 `computed_delta = query_len - num_rejected`，分别推进
  `num_computed_tokens` 和 `effective_kv_len`。
- 为 `post_update()` 增加 GPU tensor 参数。

### 变更原因

普通 execution 中两个 progress state 必须同步推进，但必须保持独立 storage。代码没有
把 effective length 赋值为 logical counter，因此未来的 `128/64` 状态可以正确推进为
`129/65`。

### 文件
`vllm/v1/worker/gpu/model_runner.py`

### 变更区域

- `GPUModelRunner.postprocess_sampled()` 调用 `post_update`，L1133-L1137

### 具体变更

- 将 `self.req_states.effective_kv_len.gpu` 传入现有 normal post-update path。

### 变更原因

当前支持的 PP=1 last-rank generation path 使用
`postprocess_sampled() → post_update()`。没有修改 excluded PP/non-last-rank helper
path，也没有新增 effective state consumer。

### 文件
`tests/v1/worker/test_gpu_effective_kv_state.py`

### 变更区域

- `TestEffectiveKVState`，L53-L91

### 具体变更

- 新增四个 CUDA targeted tests，覆盖初始化、普通推进、logical/effective 分叉和
  request index reuse。

### 变更原因

这些测试直接覆盖 Slice acceptance criteria，不引入 broad E2E framework。

## 3. 函数 / Kernel 签名变化

修改前：

```text
post_update(idx_mapping, num_computed_tokens, last_sampled_tokens, ...)
```

修改后：

```text
post_update(idx_mapping, num_computed_tokens, effective_kv_len,
            last_sampled_tokens, ...)
```

新增参数是独立 GPU tensor，接收相同的 normal execution delta；它不是 alias，也不是
logical state 的绝对值镜像。

## 4. 行为变化

修改前：

```text
computed_delta → num_computed_tokens += delta
```

修改后：

```text
computed_delta → logical num_computed_tokens += delta
              → physical effective_kv_len += delta
```

两个 tensor 只共享 normal execution delta。未来 reclaim transition 可以建立
`logical=128, effective=64`，本 Slice 不会把它们重新合并。

## 5. 新增测试与结果

测试文件：`tests/v1/worker/test_gpu_effective_kv_state.py`

- `test_initialization`：add + staged commit 后为 `37/37`，PASS。
- `test_baseline_advancement`：`128/128 → 129/129`，PASS。
- `test_independent_divergence_advancement`：`128/64 → 129/65`，PASS。
- `test_request_slot_reuse_overwrites_state`：复用 index 后初始化为 `37/37`，PASS。

执行命令：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 \
/home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310/bin/python \
-m pytest -q tests/v1/worker/test_gpu_effective_kv_state.py
```

结果：`4 passed, 15 warnings in 3.01s`。

## 6. 明确未修改区域

本 Slice 没有修改 scheduler、`KVCacheManager`、`BlockPool`、`BlockTables`、slot
mapping、`prepare_pos_seq_lens` semantics、attention metadata、reclaim policy、ACK、
Triton compaction、async/preemption/CUDA Graph、native C++/CUDA 或 `third_party/vllm`。

`post_update_num_computed_tokens()` 已审计但未修改；它属于 excluded non-last-PP/pooling
paths，后续 scope 仍为 `TO_VERIFY`。

## 7. Diff 摘要

```text
3 个 runtime 文件修改，新增 12 行
git diff --check：PASS
```

测试文件是 vLLM worktree 中新增的 untracked file，属于本 Slice；完整 diff 仍由 Git
保留。

## 8. 后续边界

本 Slice 已完成：independent request-lifetime GPU physical progress state，以及
normal computed-delta plumbing，并获得 T1-T4 evidence。

尚未实现：reclaim transition、physical slot consumer、attention effective length、
BlockTables row replacement integration、scheduler ownership/free、ACK 和 compaction。

下一允许动作：`WEB_REVIEW_CURRENT_SLICE`。不会自动开始 S2。
