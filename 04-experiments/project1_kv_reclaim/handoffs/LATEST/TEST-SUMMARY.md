# P1 Test Summary

## 已通过

- `VLLM_TARGET_DEVICE=cpu python -m pytest -q tests/v1/test_ragged_attention_spec.py tests/v1/test_ragged_kv_layout.py tests/v1/worker/test_attn_utils.py`：`32 passed`。
- `VLLM_TARGET_DEVICE=cpu python -m pytest -q tests/v1/core/test_ragged_kv_cache_planner.py tests/v1/core/test_ragged_kv_cache_manager.py tests/v1/worker/test_ragged_kv_state.py`：`37 passed`。
- `VLLM_TARGET_DEVICE=cpu python -m pytest -q tests/v1/worker/test_attn_utils.py`：`5 passed`。
- changed Python `py_compile`：PASS。
- focused `ruff check`：PASS。
- `git diff --check`：PASS。
- tiny CUDA backing/alias smoke（用户授权提升权限）：`CUDA_SMOKE: PASS`。
- `PLACEMENT_LIFETIME_CONTRACT`：`FROZEN`；D1/D2 hot path 必须复用 initialization-time placement tensors，不得每 step/layer 重新 `torch.tensor(...)`。

## 证据与限制

完整 raw 输出在 `04-experiments/project1_kv_reclaim/raw/p1-v2-r1-c-execution-foundation-01-final/`。
测试覆盖 geometry、scalar address、custom placement、zero-copy alias、Hp-wide shared
backing、member table/slot/seq-len transforms 与 Dense regression；不覆盖 GPU engine smoke、
Ragged KV write、FlashAttention read、production activation 或 benchmark。

CUDA raw evidence：`23-cuda-alias-smoke.log` 首次 sandbox 运行因 device 不可见而为
`BLOCKED_BY_ENV`；用户授权提升权限后，`24-cuda-alias-smoke-escalated.log` 实际 CUDA
运行通过。

完整 changed-file `ruff` 仍会报告三个 HEAD 既有问题：
`vllm/v1/kv_cache_interface.py:289 E501`、`vllm/v1/worker/gpu/attn_utils.py:94 E501`、
`vllm/v1/worker/gpu/attn_utils.py:671 UP038`；本 Slice 未修改这些无关 debt。
