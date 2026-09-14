> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Source Pins and Key Files

## qcrs/vllm

### Baseline

```text
branch: study-vllm-0.26
commit: 568afb3a13806beb53bb2e6bd518269357b237c0
```

### Current P1 V2

```text
branch: p1/v2-token-compaction-v026
commit: bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00
message: feat(p1): close V2 KV compaction runtime reconciliation
```

P1 V2 相对 baseline 当前修改面包含：

```text
attention/backend.py
attention/backends/flash_attn.py
attention/ops/triton_reshape_and_cache_flash.py
core/kv_cache_coordinator.py
core/kv_cache_manager.py
core/single_type_kv_cache_manager.py
core/sched/output.py
core/sched/scheduler.py
worker/gpu/attn_utils.py
worker/gpu/block_table.py
worker/gpu/input_batch.py
worker/gpu/kv_compaction.py
worker/gpu/model_runner.py
worker/gpu/states.py
```

这说明 P1 已经覆盖了新项目最关键的一部分 worker→scheduler physical-state transaction surface。

---

## aiha-lab/tangram

### Reference

```text
branch: main
commit: 6fa551fc8f6edcc118a2a39b3554ee1520e91edd
```

Tangram `.vllm_base_commit`：

```text
6fb0215eee44cf5e4b28f57e6739ef4a51945127
```

因此它的 implementation surface 与 v0.26 MRv2 不能逐文件一一复制。

---

## Tangram Core Files — 推荐阅读优先级

### S：Representation / Attention

```text
vllm/v1/kv_cache_interface.py
vllm/v1/worker/ragged_block_table.py
vllm/v1/attention/backends/ragged_layout.py
vllm/v1/attention/backends/ragged_forward.py
vllm/v1/attention/backends/flash_attn.py
```

### S：Ownership / Physical Reclamation

```text
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/block_pool.py
vllm/v1/attention/compression/executor.py
vllm/v1/attention/compression/eviction_writeback.py
vllm/v1/worker/compression_model_runner_mixin.py
```

### A：Policy

```text
vllm/v1/attention/compression/compressor.py
vllm/v1/attention/compression/budget_scope.py
vllm/v1/attention/compression/eviction_regime.py
```

### A：Optimization

```text
tools/head_group_clustering/README.md
tools/head_group_clustering/build_profile.py
tools/head_group_clustering/build_cluster_map.py
tools/head_group_clustering/clustering.py
vllm/config/compilation.py
vllm/config/vllm.py
```

---

## v0.26 MRv2 Core Files — 推荐阅读优先级

### S：KV initialization / physical metadata

```text
vllm/v1/kv_cache_interface.py
vllm/v1/core/kv_cache_utils.py
vllm/v1/worker/gpu/attn_utils.py
vllm/v1/worker/gpu/block_table.py
```

### S：Model Runner state / execution

```text
vllm/v1/worker/gpu/states.py
vllm/v1/worker/gpu/input_batch.py
vllm/v1/worker/gpu/model_runner.py
```

### S：Attention seam

```text
vllm/model_executor/layers/attention/attention.py
vllm/v1/attention/backend.py
vllm/v1/attention/backends/flash_attn.py
```

### S：Ownership

```text
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/block_pool.py
vllm/v1/core/sched/scheduler.py
```

### A：Compilation

```text
vllm/config/compilation.py
vllm/v1/worker/gpu/cudagraph_utils.py
```

---

## Source Fact vs Project Decision

本包中需要始终区分：

### Source Fact

例如：

- Tangram `RaggedAttentionSpec` 将 page head count 改为 `page_group_size`。
- Tangram `RaggedBlockTable` 带 group axis。
- v0.26 `BlockTables` 使用 staged/persistent buffers。
- Tangram threshold-based layer/global scope 当前 TP>1 需要额外跨 rank 信息。

### Project Decision

例如：

- 在 MRv2 中优先尝试 flat virtual rows，而不是复制旧 `RaggedBlockTable`。
- 新建 `RaggedKVState`，不把 scalar `effective_kv_len` 改成 union shape。
- R6 只做 TP=2 uniform。
- Full CUDA Graph 不作为目标。

后续源码审计如果推翻某个 Project Decision，可以修改；不要把它误写成 Tangram/source 的事实。

---

# v2 新增必读源码

```text
qcrs/vllm:
  vllm/v1/kv_cache_spec_registry.py
  vllm/v1/worker/gpu/buffer_utils.py
  vllm/model_executor/layers/attention/attention.py
  vllm/v1/attention/backends/flash_attn.py::do_kv_cache_update
  vllm/v1/core/kv_cache_utils.py::{get_num_blocks,get_kv_cache_config_from_groups}

Tangram:
  vllm/config/compression.py
  vllm/v1/core/kv_cache_utils.py  # global ragged pool branch
  tests/v1/attention/ragged_reference.py
  tests/v1/attention/test_ragged_layout.py
```

其中 `kv_cache_spec_registry.py` 与 `buffer_utils.py` 是上一版 source map 未充分强调、但对 MRv2-native实现非常关键的两个 surface。

