# P1-V2-R1-D-RAGGED-GPU-EXECUTION-01 Code Trace

## 1. Slice 与身份

- Slice：`P1-V2-R1-D-RAGGED-GPU-EXECUTION-01`
- Mode：`SLICE_EXECUTE`
- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/v2-token-compaction-v026`
- Reviewed D architecture HEAD：`db3cc1179f53e2dc8df2096ed17b5ec86ef033d3`
- Actual D start HEAD：`dc4287411704abc512f6cca3ca669c7d9ec254c5`
- HEAD delta：只包含用户已有的中文解释性注释；source audit 未发现 C behavior、physical layout、virtual block equation、ownership 或 production activation 变化。
- C final status：`PASS / CLOSED`
- C final implementation commit：`db3cc1179f53e2dc8df2096ed17b5ec86ef033d3`

## 2. Frozen Contract

```text
physical cache: [P,Hp,B,2D]
virtual cache:  [P*Hp,1,B,2D]
virtual_block = physical_page * Hp + column
virtual_slot  = virtual_block * B + offset
```

本 Slice 只实现 focused/synthetic real-CUDA write/read closure；不启用 production Ragged Engine，不修改 Scheduler、ModelRunner lifecycle、Attention production dispatch 或 Dense execution behavior。

## 3. Change Trace

### Entry D-CLOSURE-SYNC

File: `PROJECT_STATE.md`、`CURRENT-CONTEXT.md`、`CHATGPT-HANDOFF.md`、`handoffs/LATEST/*`、C Code Trace

Symbol: C Slice status/provenance

Change Type: MODIFY

Before: C 状态仍含 `PASS_PENDING_WEB_REVIEW`、`018e...` 或未提交 provenance。

After: C 统一为 `PASS / CLOSED`，final implementation commit 统一为 `db3cc1179f53e2dc8df2096ed17b5ec86ef033d3`；D 记录 actual start HEAD `dc4287411704abc512f6cca3ca669c7d9ec254c5`。

Why: 落实已经完成的 ChatGPT Web architecture review，并在 D source 修改前关闭 stale control-plane state。

Contract: C behavior 不变；只同步事实状态和 provenance。

Test: 文档定向 `rg` 与最终 state audit。

Status: PASS

### Entry D-LAYOUT

File: `vllm/v1/attention/backends/ragged_layout.py`

Symbol: `member_virtual_block_table`、`member_virtual_slots`、`member_seq_lens`、`group_physical_slots`、`RaggedStepViews`、`build_ragged_step_views`

Change Type: MODIFY / ADD

Before: C convenience helpers直接消费 `MemberPlacementMap`，内部每次调用 `placement_to_tensors()`；不存在 group physical slot producer 或统一 step derived metadata。

After: low-level helper显式消费 cached `member_to_cluster` / `member_to_column` tensors；新增 vectorized group slot构造、source-E write地址、post-write seq-len和 immutable `RaggedStepViews`。

Why: 关闭 D execution metadata producer，并从 API 上阻止 hot path重复 placement tensor construction。

Contract: physical/virtual equation、canonical `[R,M,...]` axis、source E vs post-write E 均保持 architecture freeze。

Test: T1、cached tensor reuse、custom placement、CUDA write/read tests。

Status: PASS

### Entry D-TESTS

File: `tests/v1/attention/ragged_reference.py`、`tests/v1/attention/test_ragged_execution.py`、`tests/v1/test_ragged_kv_layout.py`

Symbol: scalar history writer、independent PyTorch attention reference、T1-T8 focused coverage、C helper call sites

Change Type: ADD / MODIFY

Before: C tests只验证 layout/address transforms；expected metadata仍调用 placement-based helper signature。

After: test-only reference只使用 frozen scalar `resolve_kv_address` 和直接 PyTorch causal GQA数学；新增 real CUDA exact write、decode/prefill/uneven/mixed/non-uniform E/custom placement coverage；C tests显式构造并复用 placement tensors。

Why: 避免 production helper自证，并覆盖 D acceptance mechanism而不扩展到 Engine/E2E。

Contract: production code不 import test reference；CUDA tests真实调用现有 cache kernel和 FA2。

Test: `tests/v1/attention/test_ragged_execution.py` 及 C/Dense focused regressions。

Status: PASS

### Entry D-FORWARD

File: `vllm/v1/attention/backends/ragged_forward.py`

Symbol: `_virtual_kv_cache`、`ragged_kv_cache_update`、`ragged_attention_forward`

Change Type: ADD

Before: 不存在 Ragged GPU write/read execution helper。

After: 通过 zero-copy virtual cache复用 `reshape_and_cache_flash`；decode采用 request-major/head-minor；prefill/mixed采用 head-major/request-minor并显式 inverse layout；read直接调用 FA2 `flash_attn_varlen_func`。

Why: 在不修改 Dense path和 production dispatch的前提下完成 focused real-CUDA execution closure。

Contract: NO new custom op；KV cache view zero-copy；current-step Q/K/V允许 reshape/materialization；caller-provided output。

Test: T2-T8 real CUDA exact-cell与独立 PyTorch equality。

Status: PASS

### Entry D-ABI-FIX

File: `vllm/v1/attention/backends/ragged_forward.py`

Symbol: `ragged_kv_cache_update`

Change Type: MODIFY

Before: 将 canonical `member_slot_mapping` 的 `int32` 直接传给当前 v0.26 native `reshape_and_cache_flash`，真实 CUDA ABI 报 `expected scalar type Long but found Int`。

After: 仅在 native cache-kernel 调用边界将 flattened slots 转为 `torch.long`；step metadata canonical dtype保持不变。

Why: 适配 fixed local ABI，不改变物理地址值或 cache geometry。

Contract: physical slot equation不变；无新增 kernel/op；cache仍由 zero-copy virtual view提供。

Test: `04-cuda-focused-escalated-slot-long.log`、`07-cuda-regression-layout.log`。

Status: PASS

## 5. Acceptance Evidence

- D-WRITE：真实 CUDA `reshape_and_cache_flash` exact-cell oracle通过；custom placement通过；跨 `15→16` block boundary地址通过。
- D-DECODE：`query_lens=[1]`、`[1,1]`，GQA `q_per_kv=2`，FA2与独立 PyTorch reference一致，最大误差 `0.000977`。
- D-PREFILL-MIXED：`[3]`、`[3,2]`、`[1,3]` 均通过 head-major/request-minor packing和 inverse layout，最大误差 `0.000977`。
- Non-uniform E：`[[15,31],[17,5]]` 通过不同 cluster physical position与 post-write member seq lens。
- zero-copy：复用 C 已关闭的 `[P,Hp,B,2D] → [P*Hp,1,B,2D]` alias contract。
- production activation：未修改，仍 OFF。

## 6. Verification

- CPU/C/Ragged/Dense focused：`71 passed, 6 skipped`。
- Real CUDA execution/layout：`29 passed`。
- Dense `attn_utils` regression：`5 passed`。
- `py_compile`、focused `ruff`、`git diff --check`：PASS。
- `test_attention_backends.py` 全量尝试在第 6 个测试因缺少远端 `meta-llama/Meta-Llama-3-8B` 本地配置失败；前 5 个通过，未归因于 D source。

## 7. Overengineering Audit

- duplicate validation：无新增 placement bijection、quant、ownership 或 state-version validation。
- unused abstraction：仅新增 frozen design要求的 `RaggedStepViews`；无 `RaggedGeometry`/manager/cache singleton。
- per-step placement reconstruction：无；helper显式消费初始化后 cached tensors。
- hidden cache copy：无；physical→virtual使用 `view`，current-step Q/K/V reshape仅限小 tensor。
- duplicate authority：无；step view是 derived metadata，不持有 Scheduler/BlockPool/lifetime state。
- new custom op/kernel：无；直接复用 `reshape_and_cache_flash` 与 FA2 `flash_attn_varlen_func`。
- Dense pollution：无；`flash_attn.py`、production dispatch 和 Dense slot path未修改。

## 8. Boundary / Verdict

仍未证明：production `page_group_size` activation、Scheduler→Worker real wiring、ForwardContext real Ragged dispatch、real Qwen Engine Ragged identity、continuous batching、chunked-prefill E2E、physical compaction/free/reuse、性能。

Verdict：`PASS_PENDING_WEB_REVIEW`

Next Allowed Action：`WEB_REVIEW_CURRENT_SLICE`

Proposed Next Action：Web review current D Slice；不得自动执行 E。
