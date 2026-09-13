# P1-M5-T2A — KV Cache Layout Audit

## 1. Verdict

`PASS_WITH_MVP_QUALIFICATION`

审计未发现当前 T2A layout guard 会拒绝 M4 stride-safe 2D primitive 正式支持的
NHD/HND layout。结论限定在当前 P1 MVP：MRV2、FA2、BF16、single KV group、single
GPU、eager attention path。

本轮未修改 production code 或 tests。

## 2. Actual KV Cache Layout

### SOURCE_FACT

`FlashAttentionBackend.get_kv_cache_shape()` 在
`vllm/v1/attention/backends/flash_attn.py:L142-L152` 返回：

```text
[num_blocks, num_kv_heads, block_size, 2 * head_size]
```

`GPUModelRunner.initialize_kv_cache()` 在
`vllm/v1/worker/gpu/model_runner.py:L508-L520` 通过 `init_kv_cache()` 接收并保存
reshaped layer tensors 到 `self.kv_caches`。

因此当前 runner-facing logical cache 为：

```text
shape:
    [B, H, N, C]

B:
    num physical blocks/pages

H:
    num_kv_heads

N:
    physical block_size / token-within-block

C:
    2 * head_size，K/V packed in the content dimension
```

stride 取决于 `get_kv_cache_stride_order()`：

```text
NHD backing:
    physical contiguous storage [B, N, H, C]
    runner logical view        [B, H, N, C]

HND backing:
    physical/logical contiguous [B, H, N, C]
```

注意：NHD/HND 在这里主要是 block 内 `head` 与 `token` 的 backing stride
排列差异，不是不同的 outer logical shape。

### RUNTIME_PROBE

`NOT_RUN`。本次没有在未确认配置的 runner fixture 上启动完整 MRV2+FA2 runtime；
结论来自 pinned source 和现有 M4 tests。

## 3. M4 Layout Contract

### M4 reference path

`vllm/v1/worker/gpu/kv_compaction.py:L98-L180` 的 reference validator 将输入
定义为 logical `[B,H,N,C]`，从 `kv_cache.shape[2]` 取得 `block_size`，并以
`kv_cache.shape[0]` 检查 physical block ID。

reference gather 在 `L229-L236` 通过：

```text
[B,H,N,C] → permute(B,N,H,C) → flatten request-local members
```

### M4 stride-safe 2D path

`_gather_paged_kv_2d_kernel()` 位于
`vllm/v1/worker/gpu/kv_compaction.py:L410-L488`。它显式使用：

```text
stride_b, stride_h, stride_n, stride_c
```

计算 `kv_cache[physical_block, head, token_offset, :]`，因此同时支持：

```text
NHD:
    backing [B,N,H,C]
    logical view [B,H,N,C]

HND:
    backing/logical [B,H,N,C]
```

writeback 的同一 contract 位于 `L798-L875`，并在
`L1069-L1125` 通过真实 strides 写回。

现有 `tests/v1/worker/test_gpu_kv_compaction.py`：

- `_cache()` / `_cuda_cache()` 构造物理 `[B,N,H,C]` storage；
- `nhd=False` 返回 HND contiguous view；
- `nhd=True` 返回 NHD-backed logical view；
- stride-safe 2D gather/writeback 与 full compaction tests 对 `[False, True]` 参数均执行。

NHD-only 1D variant 另有显式 `_validate_nhd_layout()`，并且测试确认它拒绝 HND；
这不影响 T2A 对未来 `compact_paged_kv_triton_2d()` 的 layout 判断。

当前 production runner 的 exact runtime cache layout 选择由
`get_kv_cache_layout()` / `VLLM_KV_CACHE_LAYOUT` 与 backend stride order 决定，
但 runner-facing outer logical shape 仍为 `[B,H,N,C]`。

## 4. T2A Guard Review

### `kv_cache.ndim == 4`

`PASS_MVP_ONLY`

证据：FA2 shape factory 返回四维 `[B,H,N,C]`；M4 `_validate_inputs()` 在
`kv_compaction.py:L131-L132` 也明确要求 4D。之所以标记 MVP-only，是因为
`self.kv_caches` 作为广义 runner cache 容器并不等于所有未来 KV spec（例如
Mamba/hybrid state）都必须是该 4D attention layout；当前 T2A 又明确要求 single
KV group MVP。

### `kv_cache.shape[2] == block_size`

`PASS_MVP_ONLY`

证据：FA2 logical shape 的 dim 2 是 `block_size`；M4 在
`kv_compaction.py:L146` 将 `kv_cache.shape[2]` 作为 block size。NHD/HND 只改
stride order，不改 logical dim 2。限定 MVP 是因为该判断不应泛化到非-FA2 或其他
KV spec 的 packed/custom shape。

### `max_block_id < kv_cache.shape[0]`

`PASS`

证据：dim 0 是 `num_blocks`；M4 reference 在 `L155-L160`、writeback 在
`L970-L975` 都直接对 `kv_cache.shape[0]` 做 physical block addressability
校验。该 guard 对 NHD/HND 均成立。

### `kv_cache.device == block_ids_tensor.device`

`PASS_MVP_ONLY`

证据：M4 Triton gather 在 `kv_compaction.py:L514-L528`、writeback 在
`L953-L968` 要求 KV、block IDs、scratch 位于同一 CUDA device。T2A 当前比较
`kv_cache.device` 与 Worker BlockTable block-id tensor device，是同一 addressability
contract 的提前检查；single-GPU 是当前 MVP 限定，而不是 layout 类型判断。

## 5. Recommendation

`KEEP_CURRENT_GUARDS`

理由：

1. 当前 guard 检查的是 cheap structural/addressability facts，不检查 contiguous
   stride，因此不会错误排除 M4 支持的 NHD/HND。
2. M4 helper 的 canonical detailed validation 仍应保留在未来实际 gather/writeback
   execution path；T2A guard 不是 M4 validator 的替代品。
3. T2A 必须在 destructive T2B 前发现 deterministic incompatibility，因此保留这些
   minimal preflight checks 是合理的。
4. 没有证据表明需要改为 `MINIMIZE_GUARDS` 或 `REUSE_M4_VALIDATION`。

## 6. Code Mutation Audit

- production code modified：`NO`
- tests modified：`NO`
- audit artifacts added：`YES`
- T2B started：`NO`

## 7. Evidence Paths

- [P1-M5-T2A-KV-LAYOUT-AUDIT.md](/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/P1-M5-T2A-KV-LAYOUT-AUDIT.md)
- [P1-M5-T2A-KV-LAYOUT-AUDIT.log](/home/qcrs/learning/llm-kv-lab/04-experiments/project1_kv_reclaim/raw/P1-M5-T2A-KV-LAYOUT-AUDIT.log)

## 8. Stop Boundary

本轮只完成 layout contract review。没有修改 `_prepare_v2_compactions()`，没有调用
`compact_paged_kv_triton_2d()`，也没有进入 M5-T2B。
