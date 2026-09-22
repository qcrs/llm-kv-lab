# P1 Diff Summary

## 当前 Slice

`P1-V2-R1-C-EXECUTION-FOUNDATION-01` 已 `PASS / CLOSED`；C final implementation commit 为
`db3cc1179f53e2dc8df2096ed17b5ec86ef033d3`。

| 文件 | 变更 | 目的 |
|---|---|---|
| `vllm/v1/kv_cache_interface.py` | MODIFY | 收紧 Ragged Core geometry：equal K/V head size、unquantized、unpadded。 |
| `vllm/v1/ragged_kv_layout.py` | MODIFY | 新增 immutable `ResolvedKVAddress` 和 CPU `physical_position` oracle。 |
| `vllm/v1/attention/backends/ragged_layout.py` | ADD | 新增 Hp-wide physical shape、zero-copy virtual view、member metadata transforms。 |
| `vllm/v1/worker/gpu/attn_utils.py` | MODIFY | 为 Ragged raw allocation 建立 `[P,Hp,B,2D]` dedicated materialization；Dense path 保持。 |
| `tests/v1/test_ragged_attention_spec.py` | MODIFY | 更新合法 geometry expectation，增加 Core rejection coverage。 |
| `tests/v1/test_ragged_kv_layout.py` | MODIFY | 增加 scalar/vector、custom placement、PAD、zero-copy alias oracle。 |
| `tests/v1/worker/test_attn_utils.py` | MODIFY | 增加 Ragged shared backing materialization，并保留 Dense regression。 |

开始前已存在、仅 `NO_CHANGE_REVIEWED` 的 dirty 内容：
`vllm/v1/core/ragged_kv_cache_manager.py`、`vllm/v1/core/sched/output.py`，以及
`vllm/v1/ragged_kv_layout.py` 中既存解释性注释；这些内容没有被回滚。

## 行为边界

- 没有删除 production behavior。
- 没有替换 Dense `_reshape_kv_cache` path；只增加 Ragged dedicated branch。
- 没有修改 public API signature。
- 没有加入 `reshape`/`contiguous`/`clone`/copy fallback。
- 没有启用 production Ragged dispatch、KV write 或 FlashAttention read。

## 证据

Code Trace：`04-experiments/project1_kv_reclaim/notes/P1-V2-R1-C-EXECUTION-FOUNDATION-01-Code-Trace.md`。
Raw evidence：`04-experiments/project1_kv_reclaim/raw/p1-v2-r1-c-execution-foundation-01-final/`。

当前已完成 `P1-V2-R1-D-RAGGED-GPU-EXECUTION-01`；D 状态为 `PASS_PENDING_WEB_REVIEW`。

## D Slice

| 文件 | 变更 | 目的 |
|---|---|---|
| `vllm/v1/attention/backends/ragged_layout.py` | MODIFY | cached placement tensor API、group physical slots、`RaggedStepViews`、source/post-write metadata。 |
| `vllm/v1/attention/backends/ragged_forward.py` | ADD | real `reshape_and_cache_flash` write、FA2 decode/prefill/mixed read。 |
| `tests/v1/attention/ragged_reference.py` | ADD | 独立 scalar address / PyTorch attention reference。 |
| `tests/v1/attention/test_ragged_execution.py` | ADD | D-WRITE、D-DECODE、D-PREFILL-MIXED real CUDA coverage。 |
| `tests/v1/test_ragged_kv_layout.py` | MODIFY | 适配 cached-tensor helper signature。 |

D production activation 保持 OFF。
