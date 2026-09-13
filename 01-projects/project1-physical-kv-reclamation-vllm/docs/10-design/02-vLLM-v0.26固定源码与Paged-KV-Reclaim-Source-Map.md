# 02 — vLLM v0.26 Fixed Source Map

## Pin

vLLM v0.26.0 / 568afb3a13806beb53bb2e6bd518269357b237c0。实际 local path/line/signature 在 M0 冻结，不复用 current-main line number。

## Current Runtime Identity

```yaml
vllm_engine: V1 Engine
model_runner: V2 / MRV2
env: VLLM_USE_V2_MODEL_RUNNER=1
source: vllm/v1/worker/gpu/model_runner.py
legacy_runner: vllm/v1/worker/gpu_model_runner.py (historical_reference_only)
```

历史 v3 audit 使用 MRV1 seams；当前 active mapping 见
`docs/90-reference/MRV2-PORT-RESOLUTION.md`。历史 baseline snapshot 刻意不改写。

## Control and Data Chain

~~~text
Scheduler
→ KVCacheManager / SingleTypeKVCacheManager
→ SchedulerOutput
→ Executor / Worker
→ GPUModelRunner V2: add_requests / update_requests / prepare_inputs
→ RequestState / BlockTables
→ slot mapping / attention metadata
→ attention backend / KV cache
→ ModelRunnerOutput
→ Scheduler.update_from_output
→ canonical owner mutation / BlockPool
~~~

## Fixed Anchors to Reconfirm

| Concern | Pinned Path / Symbol | Owner / Role | Status |
|---|---|---|---|
| canonical request pages | vllm/v1/core/single_type_kv_cache_manager.py::req_to_blocks | control-plane owner | AUDITED / local verify |
| allocation/free | kv_cache_manager.py; block_pool.py | capacity/reuse | AUDITED |
| scheduler output/update | vllm/v1/core/sched/*; vllm/v1/outputs.py | transaction channel | TO_VERIFY exact |
| row storage/rebuild | vllm/v1/worker/gpu/block_table.py::append_block_ids(overwrite=True) | worker mapping | AUDITED MRV2 |
| persistent request state | vllm/v1/worker/gpu/states.py::RequestState | request lifetime | AUDITED MRV2 |
| append/resume transition | gpu/model_runner.py::add_requests/update_requests | worker state | AUDITED MRV2 |
| logical positions/seq | gpu/model_runner.py::prepare_inputs + input_batch.py::prepare_pos_seq_lens | model semantics | AUDITED MRV2 |
| slot mapping | gpu/block_table.py::compute_slot_mappings | cache position→page | AUDITED MRV2; bounds TO_VERIFY |
| attention metadata | backend builder / CommonAttentionMetadata | physical KV view | TO_VERIFY exact seam |
| paged Triton address | triton_reshape_and_cache_flash.py | block/offset pointer math | AUDITED |

## Important Source Facts

Upstream commonly reuses num_computed_tokens for model position、seq length、slot mapping. P1 cannot globally replace seq_lens with physical length because other consumers may use logical semantics.

BlockTable is indirection：retained IDs need not be contiguous, but logical block columns in the active row must preserve retained token order.

SingleTypeKVCacheManager canonical ownership, worker BlockTable row and effective length must converge at transaction commit。

## Reference Mapping

Tangram：

- CompressionExecutor for keep/gather/writeback；
- effective_seq_lens_cpu for attention/cache semantics；
- ModelRunnerOutput compression_*；
- scheduler → free_blocks_by_ids；
- allocate_slots effective_num_cached_tokens。

Sparse-vLLM：

- cache-manager ownership；
- scalar/batched oracle；
- invalid keep input no mutation。

## M0 Required Output

每个 anchor 记录 exact path/signature、caller/callee、persistent state、owner、shape/device、mutation、tests、feature-off。无法确认则 TO_VERIFY；与 Approved Design 冲突则 DESIGN_CONFLICT。

## v1.2 Composite Seam Anchors

M0/M1 额外把下面几项作为 exact source anchors，而不是隐藏在 “BlockTable works” 结论里：

| Concern | Exact source question | Core decision |
|---|---|---|
| live row shrink | `add_row()` 是否清 old tail？哪些 consumer 读取 raw row vs active count？ | source proof or clear-old-range wrapper |
| block granularity | `kernel_block_size`, manager block size, `blocks_per_kv_block` | Core 只接受 1:1 |
| context parallel mapping | DCP/PCP/CP interleave 是否改 slot semantics | Core 全部 1 |
| cascade attention | backend 是否建立额外 shared-prefix/cascade metadata | Core OFF |
| reclaim timing | final-prefill 前是否仍持有 full prompt KV | Core 明确不降低 first-prefill peak |

这些是“primitive→composed transition”的 integration seams；未关闭前不能把 V1 标成 Runtime Gate PASS。
