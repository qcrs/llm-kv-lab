> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Tangram Source Provenance Ledger — v3

目的：每一个我们准备实现的重要机制，都必须回答三个问题：

1. Tangram 哪里已经实现/验证？
2. qcrs/vLLM 0.26 对应接入点在哪里？
3. 我们是直接 port、API adapt、还是重新设计？

Source pin：

```text
Tangram main:
6fa551fc8f6edcc118a2a39b3554ee1520e91edd

qcrs/vllm branch:
p1/v2-token-compaction-v026
bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00
```

## 1. Provenance Matrix

| Capability | Tangram source | qcrs/v0.26 target | Action | 说明 |
|---|---|---|---|---|
| config knob | `vllm/config/cache.py::page_group_size` | `vllm/config/cache.py` | ADAPT | 只加最小 ragged fields，不搬全 compression config |
| startup validation | `vllm/config/compression.py::validate_extended_fields` | 新 `config/ragged.py` 或 cache validator | PORT-SELECTIVE | 只保留 Hp/divisibility/backend/scope gates |
| Ragged spec | `vllm/v1/kv_cache_interface.py::RaggedAttentionSpec` | 同文件 | ADAPT | v0.26 `AttentionSpec` 字段更多，不能 copy dataclass |
| spec generation | Tangram `vllm/attention/layer.py::get_kv_cache_spec` | `model_executor/layers/attention/attention.py::get_kv_cache_spec` | REDESIGN-HOOK | 当前 attention module 路径不同 |
| global pool sizing | Tangram `core/kv_cache_utils.py::get_num_blocks` | `v1/core/kv_cache_utils.py` | REDESIGN | v0.26 planner支持 packed/hybrid，需独立 ragged branch |
| one shared backing | Tangram `get_kv_cache_config_from_groups` ragged branch | `kv_cache_utils.py` + `gpu/attn_utils.py::_allocate_kv_cache` | ADAPT | v0.26 已支持 `shared_by`, 可利用 |
| bulk page allocator | Tangram `core/block_pool.py` ragged ring | `core/block_pool.py` | LATER-PORT | semantic tests先可小 pool；capacity benchmark前必做 |
| ragged manager | Tangram `single_type_kv_cache_manager.py` ragged branch | 推荐新 `ragged_kv_cache_manager.py` | REDESIGN | v0.26 generic manager复杂度更高，避免大量 `if ragged` |
| 3D block table semantics | Tangram `worker/ragged_block_table.py` | 新 `worker/gpu/ragged_block_table.py` | REDESIGN | MRv2 staged/UVA data plane不同 |
| snapshot/restore | Tangram `RaggedBlockTable.snapshot_row/restore_row` | MRv2 persistent RaggedBlockTables | REFERENCE-ONLY | 普通 skipped step 不需要；true preemption 必须 reset/recompute |
| virtual block mapping | Tangram `ragged_layout.py` | 新 `v1/attention/backends/ragged_layout.py` | NEAR-PORT | 算法不变，physical shape adapter变 |
| virtual slot mapping | 同上 | 同上 | NEAR-PORT | `p*Hp+c` 核心不变 |
| reference oracle | `tests/v1/attention/ragged_reference.py` | 新 focused tests | PORT | 优先拿来做 CPU oracle |
| decode member path | `ragged_forward.py::_ragged_decode_forward` | 新 ragged adapter | PORT-ADAPT | current custom-op seam不同 |
| prefill member path | `_ragged_member_major_forward` | 新 ragged adapter | PORT-ADAPT | copy逻辑基本可借鉴 |
| ragged metadata | Tangram `flash_attn.py::FlashAttentionMetadataBuilder` | v0.26 FA metadata builder / common metadata | REDESIGN | MRv2 metadata结构不同 |
| KV write | Tangram mostly inside ragged attention `impl.forward` | `Attention.unified_kv_cache_update` + FA `do_kv_cache_update` | **V026-NATIVE** | 这是本项目最关键的接口漂移之一 |
| per-group effective length | Tangram `gpu_input_batch.py::effective_seq_lens_cpu` | 推荐新 `gpu/ragged_kv_state.py` | REDESIGN | 不污染 P1 scalar `effective_kv_len` |
| payload executor | Tangram `attention/compression/executor.py` | 先复用 P1 scratch oracle | REUSE-P1-FIRST | correctness先于优化 |
| tail trimming | Tangram `RaggedBlockTable.compact_after_compress_all_layers` | RaggedBlockTables + scheduler reconcile | PORT-SEMANTICS | allocator authority保持scheduler |
| scheduler transaction | Tangram compression flow | qcrs P1 `CompactionPlanData/ResultData` | **REUSE-P1** | 不重新发明 ownership |
| uniform budget | Tangram `compression/budget_scope.py` | R4 | PORT-SELECTIVE | TP2安全路径 |
| scorer | Tangram `keydiff.py` / `streamingllm.py` | R4 | PORT-ONE | 只选一个 |
| TP uniform | Tangram budget/config + distributed kept length handling | R6 | PORT-SEMANTICS | 只做 uniform count |
| exact row restore | Tangram旧 persistent-batch snapshot fields | MRv2 persistent request state | MOSTLY-SKIP | 仅未来 worker 主动丢弃仍被 scheduler 持有的 row 时需要 |
| AOT clustering | `tools/head_group_clustering/*` | tools/ | PORT-SELECTIVE | 首版限定 per-layer map |
| Triton writeback | Tangram executor/writeback | R8 | PORT-LATER | 必须先和 scratch oracle differential |
| CUDA Graph | Tangram dedicated ragged custom op + piecewise | v0.26 existing `unified_attention_with_output` eager split | REDESIGN-USE-NATIVE | 不必复制 dedicated op 才能支持 piecewise |

## 2. Tangram 关键文件阅读顺序

如果只读与本项目直接相关的 Tangram，按下面顺序即可：

```text
1. vllm/v1/kv_cache_interface.py
   └─ RaggedAttentionSpec

2. vllm/v1/core/kv_cache_utils.py
   └─ global pool sizing / one backing

3. vllm/v1/core/block_pool.py
   └─ ragged bulk allocator

4. vllm/v1/worker/ragged_block_table.py
   └─ per-group row / slot / trim / snapshot

5. vllm/v1/attention/backends/ragged_layout.py
   └─ physical→virtual mapping

6. tests/v1/attention/ragged_reference.py
   └─ independent address oracle

7. vllm/v1/attention/backends/ragged_forward.py
   └─ decode/prefill member-major

8. vllm/v1/attention/backends/flash_attn.py
   └─ metadata construction

9. vllm/v1/attention/compression/executor.py
   └─ compact/writeback

10. vllm/v1/attention/compression/budget_scope.py
11. vllm/v1/attention/compression/keydiff.py
12. tools/head_group_clustering/*
```

不要一开始读 Tangram 的所有 scorer/multi-turn/benchmark glue。

## 3. qcrs/v0.26 必须对齐的源码面

```text
vllm/model_executor/layers/attention/attention.py
  - Attention.get_kv_cache_spec
  - unified_kv_cache_update
  - unified_attention_with_output
  - eager_break_during_capture

vllm/v1/worker/gpu/attn_utils.py
  - get_kv_cache_spec
  - init_attn_backend
  - _allocate_kv_cache
  - _reshape_kv_cache
  - build_slot_mappings_by_layer

vllm/v1/core/kv_cache_utils.py
  - group creation
  - get_num_blocks
  - get_kv_cache_config_from_groups
  - max concurrency accounting

vllm/v1/worker/gpu/block_table.py
vllm/v1/worker/gpu/buffer_utils.py
  - StagedWriteTensor / UVA path

vllm/v1/worker/gpu/input_batch.py
vllm/v1/worker/gpu/states.py
  - P1 logical/physical split

vllm/v1/attention/backends/flash_attn.py
  - current KV cache shape
  - do_kv_cache_update
  - forward

vllm/v1/core/sched/output.py
vllm/v1/core/sched/scheduler.py
  - P1 plan/result/reconcile

vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/core/block_pool.py
  - canonical allocator authority
```

## 4. “有 Tangram 出处”不等于“可以复制”

以下四类一定要区分：

### Exact-portable semantics

```text
page_group_size / page byte formula
virtual_block = physical*Hp+column
member sequence = (request,kv_head)
ragged row non-uniform count
snapshot/restore semantics
```

### Portable algorithms, API needs adapt

```text
member_virtual_slots
member_virtual_block_table
decode reshape
prefill member-major permutation
compaction writeback safety
```

### Tangram architecture is reference, v0.26 must redesign

```text
global pool planner
RaggedBlockTables data plane
attention metadata builder integration
CUDA Graph seam
manager registration
```

### qcrs-specific reusable work

```text
logical vs physical length split
post-forward source fence
compaction plan/result protocol
scheduler canonical ownership
free/reuse closure
P1 scratch compaction oracle
```

这四类如果混在一起，后续最容易出现“看起来有 Tangram 代码，但在 v0.26 接不上”的问题。
