# vLLM v0.26 V1 Engine / MRV2 架构与 KV 主链

> `SOURCE_FACT` 以 pinned source `568afb3a13806beb53bb2e6bd518269357b237c0` 为准；
> `P1_RELEVANCE` 只说明与 P1 的关系，不是批准的实现接口。

## 1. 三个版本名

本仓库使用 vLLM **V1 Engine**（`vllm/v1/**`）。在该 Engine 内，环境变量
`VLLM_USE_V2_MODEL_RUNNER=1` 选择 **Model Runner V2 (MRV2)**，入口为
`vllm/v1/worker/gpu/model_runner.py`（选择逻辑见 `vllm/config/vllm.py:550-553`
和 `vllm/v1/worker/gpu_worker.py:402-416`）。旧
`vllm/v1/worker/gpu_model_runner.py` 仅作历史参考。P1 V1/V2 则是项目功能
版本，分别表示 whole-block reclaim 与 token-level Triton compaction。

## 2. Runtime 分层

```text
Frontend / LLM
  -> Engine -> EngineCore -> Scheduler
  -> KVCacheManager / BlockPool -> SchedulerOutput
  -> Executor -> GPUWorker -> GPUModelRunner (MRV2)
  -> RequestState / BlockTables / InputBatch
  -> attention metadata -> attention backend -> paged KV tensors
  -> ModelRunnerOutput -> EngineCore/Scheduler
```

Scheduler/KV manager 是 control plane；worker runner、attention metadata 与
KV tensor 访问是 data plane。`req_to_blocks` 是 scheduler-side canonical
request ownership（`vllm/v1/core/single_type_kv_cache_manager.py`）；worker
只维护执行所需的 block view。

## 3. Scheduler 与 KV 控制面

Scheduler 按 request 的 `num_computed_tokens`、token budget 调用
`allocate_slots()`，把 block IDs 放入 `req_to_blocks`，并通过
`SchedulerOutput` 发送到 worker。`pop_blocks_for_free()` 与
`BlockPool.free_blocks()` 负责 ownership removal 和 reusable free queue。
因此 physical free 不能由 worker 直接发布（`P1_RELEVANCE`）。

## 4. MRV2 RequestState

`vllm/v1/worker/gpu/states.py::RequestState` 是 request-lifetime owner，维护
`req_id_to_index`、`prompt_len`、`prefill_len`、`total_len`、
`num_computed_prefill_tokens`、`num_computed_tokens` 及 NumPy mirror；
`add_request()` 建立索引，`remove_request()` 回收索引，
`apply_staged_writes()` 发布 staged state。每步 `InputBatch` 是 transient
的输入组装，而不是旧 MRV1 `CachedRequestState` 的当前描述。

## 5. MRV2 BlockTables

`BlockTables` 持久化每个 request 的 row 与 `num_blocks`。新 request 或完整
替换可调用 `append_block_ids(..., overwrite=True)`，再由
`apply_staged_writes()` 发布。`gather_block_tables()` 按 `num_blocks` 复制
active range 到当前 step 的表；`compute_slot_mappings()` 使用 positions、
block size 和持久 row 计算 physical slot。旧 tail 因 active count 不会进入
gathered table；slot kernel 对 persistent row 的边界保护仍是 P1 `TO_VERIFY`。

## 6. prepare_inputs 与 prepare_attn

MRV2 `prepare_inputs()`（`model_runner.py:874-1053`）从 SchedulerOutput 得到
scheduled request IDs，映射为 worker indices，构造 query start locations，
再调用 `prepare_pos_seq_lens()`。当前 kernel（`input_batch.py:246-300`）从
`num_computed_tokens` 计算 `seq_len` 与 logical `positions`。

`prepare_attn()`（`model_runner.py:1055-1080`）随后调用
`gather_block_tables()` 与 `compute_slot_mappings()`，形成 attention metadata
和 KV write slot mapping。当前 source 仍把 logical progress 与 physical slot
输入耦合；P1 只记录一个 candidate adapter seam，不在本文件批准拆分字段。

## 7. KV 写入与读取

新 K/V 经 `slot_mapping` 写入 paged KV cache；attention backend 依据 gathered
block table 读取历史 K/V。该文档只覆盖固定 dense causal Qwen3-family 路径；
FA2/FA3 的实际选择仍需环境 gate，不能由 `.so` 存在推断 dispatch。

## 8. ModelRunnerOutput 返回边界

MRV2 `sample_tokens()` 构造 `ModelRunnerOutput`（`model_runner.py:1395+`）。
`vllm/v1/outputs.py` 明确该对象会 serialized 并发送到 scheduler；EngineCore
等待执行结果后调用 `Scheduler.update_from_output()`（`engine/core.py:588-596,
700-704`）。这是未来 worker physical commit metadata 的 transport seam
（`P1_RELEVANCE`），本轮不新增字段。

## 9. 一次 request 的主链

以 32-token prompt、一个固定 block size `N` 为例：scheduler 为 request 分配
`ceil(32/N)` 个 block，并把 IDs 放进 `req_to_blocks` 和 SchedulerOutput；
MRV2 `add_requests()` 建立 RequestState 与 BlockTables row。prefill 时
`num_computed_tokens` 生成 positions/seq lens，slot mapping 写入新 K/V；
首个 decode 继续由 scheduler 更新 token budget 与 block ownership，runner
更新 RequestState，重新 gather row 并写入 decode K/V。每步结果通过
ModelRunnerOutput 返回 scheduler。logical progress、scheduler ownership、
worker physical view 是三个相关但不同的状态。

## 10. P1 Seam Map

| P1 需求 | 当前 MRV2 观察 | 状态 |
|---|---|---|
| persistent physical occupancy | `RequestState` arrays | `CANDIDATE_SEAM` |
| whole-row replacement | `append_block_ids(overwrite=True)` | `CANDIDATE_SEAM` |
| physical slot calculation | `compute_slot_mappings()` | `CANDIDATE_SEAM`, bounds `TO_VERIFY` |
| logical model position | `prepare_pos_seq_lens()` | `SOURCE_FACT` |
| canonical ownership | `req_to_blocks` / `BlockPool` | `SOURCE_FACT` |
| worker ACK transport | serialized `ModelRunnerOutput` | `SOURCE_FACT` |

本文件不批准任何 field、patch 或 Slice。后续 source-first 顺序仍是：source
→ correctness → runtime → benchmark → profiler。
