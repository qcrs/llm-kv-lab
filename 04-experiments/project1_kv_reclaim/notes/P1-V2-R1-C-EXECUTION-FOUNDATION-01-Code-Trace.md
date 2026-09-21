# P1-V2-R1-C-EXECUTION-FOUNDATION-01 Code Trace

## 1. Slice 信息与执行身份

- Slice：`P1-V2-R1-C-EXECUTION-FOUNDATION-01`。
- 执行模式：`SLICE_EXECUTE`。
- 日期：2026-09-21（Asia/Shanghai）。
- implementation worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`。
- branch：`p1/v2-token-compaction-v026`。
- reviewed HEAD：`018e68f47f3bcdfb0b935f5ffe0e579b033c6268`。
- actual start HEAD：`018e68f47f3bcdfb0b935f5ffe0e579b033c6268`。
- HEAD delta：无；未触发 `DESIGN_CONFLICT`。
- start worktree：3 个既存 dirty files：
  - `vllm/v1/core/ragged_kv_cache_manager.py`
  - `vllm/v1/core/sched/output.py`
  - `vllm/v1/ragged_kv_layout.py`
- 既存 dirty 变更判断：manager/output 只有中文解释性注释；`ragged_kv_layout.py` 的既存 dirty 部分也是解释性注释，没有改变 A3 placement contract。本 Slice 保留这些变更，不回滚、不覆盖。

本轮只关闭 C0–C4 execution-addressing foundation：geometry hardening、CPU address oracle、v0.26-native zero-copy layout、Ragged backing view 和 member metadata transforms。没有启用 production Ragged dispatch，也没有实现 KV write、FlashAttention read 或 Scheduler/ModelRunner wiring。

## 2. Source Fact 与验证动机

### SOURCE FACT

- `RaggedAttentionSpec.num_kv_heads` 是 semantic `Hkv`，`page_group_size` 是 physical `Hp`。
- Planner 已按 `Hp` 计算 page bytes，并为 Ragged layers 生成一个 `KVCacheTensor(shared_by=all Ragged layers)`。
- v0.26 Dense `_reshape_kv_cache` 会调用 backend `get_kv_cache_shape(..., Hkv, ...)`，不能直接解释 Ragged `[P,Hp,B,2D]` raw storage。
- `MemberPlacementMap` 是 semantic member 到 `(cluster,column)` 的唯一 static placement authority。

### DESIGN CONTRACT

```text
physical cache: [P, Hp, B, 2D]
virtual view:   [P*Hp, 1, B, 2D]
virtual_block_id = physical_page_id * Hp + column
virtual_slot = virtual_block_id * B + block_offset
```

因此本轮必须同时证明：scalar address equation 成立、planner 的 Hp-wide bytes 能被真实 materialize、virtual view 没有 hidden copy、vector metadata 与 scalar oracle exact 一致。

## 3. Change Ledger

### [00] Start dirty files — NO_CHANGE_REVIEWED

**File:** `vllm/v1/core/ragged_kv_cache_manager.py`  
**Symbol:** `RaggedRequestPhysicalState` / `_empty_state`  
**Change Type:** `NO_CHANGE_REVIEWED`  
**Reason:** start 前已经存在的中文解释性注释；本 Slice 不涉及 Ragged Scheduler ownership state。  
**Status:** 保留，未由本 Slice 新增。

**File:** `vllm/v1/core/sched/output.py`  
**Symbol:** `RaggedKVUpdateData.__post_init__`  
**Change Type:** `NO_CHANGE_REVIEWED`  
**Reason:** start 前已经存在的 overlap 说明注释；本 Slice 不修改 transport。  
**Status:** 保留，未由本 Slice 新增。

### [01] Core geometry hardening

**File:** `vllm/v1/kv_cache_interface.py`  
**Symbol:** `RaggedAttentionSpec.__post_init__`  
**Change Type:** `MODIFY`  
**Before:** 只拒绝 `Hp <= 0` 和 `Hkv % Hp != 0`，`head_size_v` 可与 `head_size` 不同，quantized/padded geometry 可以构造。  
**After:** 在同一 construction boundary 继续保留 `Hp/Hkv` geometry checks，并拒绝：`head_size_v != head_size`、`kv_quant_mode != KVQuantMode.NONE`、`page_size_padded is not None`。`head_size_v=None` 仍先归一化为 `head_size`。  
**Contract:** Core 只支持 equal K/V head size、unquantized、unpadded Ragged geometry；不在后续 layout helper 重复检查。  
**Tests:** `tests/v1/test_ragged_attention_spec.py` 的 valid replace/type test、non-divisible test、unsupported geometry parameterization。  
**Status:** IMPLEMENTED / TESTED。

### [02] CPU scalar address result

**File:** `vllm/v1/ragged_kv_layout.py`  
**Symbol:** `ResolvedKVAddress`  
**Change Type:** `ADD`  
**Before:** 只有 immutable `MemberPlacementMap`，没有可复用的 resolved address result。  
**After:** 新增 immutable result，字段为 `member_index`、`cluster_index`、`column_index`、`page_depth`、`block_offset`、`physical_page_id`、`virtual_block_id`、`virtual_slot`。  
**Contract:** result 只表达 CPU/reference address，不携带 GPU pointer、Scheduler state 或 backend dependency。  
**Tests:** scalar example、boundary、custom placement tests。  
**Status:** IMPLEMENTED / TESTED。

### [03] CPU scalar address resolver

**File:** `vllm/v1/ragged_kv_layout.py`  
**Symbol:** `resolve_kv_address`  
**Change Type:** `ADD`  
**Before:** 没有 `physical_position` 到 physical/virtual address 的 reference oracle。  
**After:** 按 `MemberPlacementMap` 解析 member 和 `(cluster,column)`；再按 `physical_position // block_size`、`% block_size` 读取 `active_row[cluster][page_depth]`，计算 physical page、virtual block 和 virtual slot。  
**Reason:** 为后续 D1/D2 提供单一 scalar correctness oracle，避免 runtime helper 重写 identity placement formula。  
**Contract:** 允许 reserve-before-write 的 position，因此不检查 `physical_position < effective_len`；只拒绝负 position、越过 supplied row depth 和 null page `0`。  
**Dependency:** `MemberPlacementMap` 是唯一 placement authority；模块保持无 `torch`、Scheduler、Worker、FlashAttention 依赖。  
**Tests:** `layer=1, head=2, physical_position=21` exact expected；`0/15/16/31` boundary；outside-row；non-identity placement。  
**Status:** IMPLEMENTED / TESTED。

### [04] v0.26-native Ragged Torch layout

**File:** `vllm/v1/attention/backends/ragged_layout.py`  
**Symbol:** `ragged_physical_cache_shape`  
**Change Type:** `ADD`  
**Before:** 没有 dedicated Ragged physical shape helper。  
**After:** 返回 `[P, Hp, B, 2D]`，不调用 Dense backend shape/stride API。  
**Contract:** v0.26 packed K/V content 继续使用最后一维 `2D`，不复制 Tangram `[2,P,Hp,B,D]` rank。  
**Tests:** `_reshape_kv_cache` raw materialization shape assertion。  
**Status:** IMPLEMENTED / TESTED。

**File:** `vllm/v1/attention/backends/ragged_layout.py`  
**Symbol:** `as_virtual_block_view`  
**Change Type:** `ADD`  
**Before:** 没有 physical-to-virtual zero-copy view。  
**After:** 对 contiguous `[P,Hp,B,2D]` 输入直接调用 `.view(P*Hp,1,B,2D)`；拒绝错误 rank、错误 `Hp` 或 non-contiguous input。没有 `reshape/contiguous/clone/copy` fallback。  
**Contract:** `physical[p,c,t,x]` 与 `virtual[p*Hp+c,0,t,x]` 必须 alias 同一 storage。  
**Tests:** `Hp=1/2/4` 的 shape/stride/data pointer/mutation alias，以及 non-contiguous rejection。  
**Status:** IMPLEMENTED / TESTED。

### [05] Member metadata transforms

**File:** `vllm/v1/attention/backends/ragged_layout.py`  
**Symbol:** `placement_to_tensors`、`member_virtual_block_table`、`member_virtual_slots`、`member_seq_lens`  
**Change Type:** `ADD`  
**Before:** 没有 Torch execution representation；只有 tuple-based `MemberPlacementMap`。  
**After:** 从 `MemberPlacementMap` 派生 `member_to_cluster/member_to_column` tensors，并实现：

```text
[R,C,MaxPages] -> [R,M,MaxPages]
[Q,C]          -> [Q,M]
[R,C]          -> [R,M]
```

`member_virtual_block_table` 对 physical page `0` 保留 null/padding `0`；`member_virtual_slots` 只对 `PAD_SLOT_ID=-1` 做 sentinel preservation。  
**Contract:** vector transforms 不产生第二套 placement authority；custom valid placement 与 identity placement 都必须工作。  
**Tests:** identity/custom table、normal/block-boundary/PAD slot、non-uniform sequence lengths；测试内独立 scalar loop 与 vector result exact equal。
**Status:** IMPLEMENTED / TESTED。

### [06] Ragged backing materialization

**File:** `vllm/v1/worker/gpu/attn_utils.py`  
**Symbol:** `_reshape_kv_cache`  
**Change Type:** `MODIFY`  
**Before:** 所有 `AttentionSpec` 都进入 backend `get_kv_cache_shape(..., Hkv, ...)` 和 Dense stride-order path。  
**After:** `RaggedAttentionSpec` 使用 dedicated `[P,Hp,B,2D]` view；不调用 backend Dense shape。只保留必要 runtime fence：`kernel_block_size == spec.block_size`、raw backing page alignment、packed Ragged path 直接 fail。Dense path unchanged。  
**Reason:** Planner 已按 Hp-wide page bytes 分配；继续用 Dense Hkv shape 会把同一 raw allocation 解释成错误 geometry。  
**Contract:** raw allocation 的 `P = raw_bytes / spec.page_size_bytes`；Ragged view 与 raw allocation alias，同一 backing 可由多个 layers 共享。  
**Dependency:** C0 geometry contract、`ragged_physical_cache_shape`、现有 `_allocate_kv_cache` 的 `shared_by` aliasing。  
**Tests:** fake backend 禁止调用 Dense shape；多个 layer 使用同一 raw backing；shape/data pointer/virtual alias；既有 Dense padded/HND/quantized tests。  
**Status:** IMPLEMENTED / TESTED。

### [07] Dense-only mixed-layout guard

**File:** `vllm/v1/worker/gpu/attn_utils.py`  
**Symbol:** `_align_mixed_attention_kv_cache_views`  
**Change Type:** `MODIFY`  
**Before:** 对所有 `AttentionSpec` 尝试查询 backend block dimension；Ragged dedicated view 没有必要进入 Dense mixed-layout restride。  
**After:** `RaggedAttentionSpec` 在该 Dense-only alignment helper 中直接跳过。  
**Reason:** 避免 Ragged backing 被 Dense K/V-first/blocks-first alignment 误处理；当前 planner 仍禁止 Ragged mixed groups。  
**Tests:** Ragged `_reshape_kv_cache` shared-backing test；Dense regression suite。  
**Status:** IMPLEMENTED / TESTED。

### [08] Independent scalar vector-oracle tests

**File:** `tests/v1/test_ragged_kv_layout.py`  
**Symbol:** `_scalar_member_virtual_block_table`、`_scalar_member_virtual_slots`  
**Change Type:** `ADD`  
**Before:** C4 table/slot assertions主要依赖手写 expected tensor，不能直接表达 vector result 与独立 scalar reference 的逐项一致性。  
**After:** 测试内增加简单 Python scalar loops，分别计算 physical page/column 到 virtual block、physical slot 到 virtual slot，并对 identity/custom placement、block boundary、`PAD_SLOT_ID=-1` 和 non-uniform `E` 做 exact comparison。  
**Reason:** 直接满足 T7/T8 的 vector-vs-scalar oracle contract；不向 production code 添加测试专用 abstraction。  
**Tests:** 最终 focused C0–C4/Dense suite `32 passed`。  
**Status:** IMPLEMENTED / TESTED。

## 4. Address Evidence

使用 `L=2, Hkv=4, Hp=2, B=16` identity placement；raw 输出见
`04-experiments/project1_kv_reclaim/raw/p1-v2-r1-c-execution-foundation-01-final/04-address-layout-evidence.log`：

```text
layer1/head2/physical_position21
→ member6
→ cluster3,column0
→ page_depth1,offset5
→ physical page41
→ virtual block82
→ virtual slot1317
```

另有 custom placement 测试证明 scalar resolver 不会把 `cluster = layer * G + head // Hp` 当成 runtime authority。

## 5. Layout Evidence

针对 `P=3, Hp∈{1,2,4}, B=4, 2D=6` 的 contiguous input，observed stride contract 为：

```text
physical shape: [3, Hp, 4, 6]
physical stride: [Hp*4*6, 4*6, 6, 1]
virtual shape: [3*Hp, 1, 4, 6]
virtual stride: [4*6, 4*6, 6, 1]
```

本轮 raw 复核使用 `P=3, Hp=2, B=4, 2D=6`，观察到：

```text
physical_shape=(3, 2, 4, 6)
physical_stride=(48, 24, 6, 1)
virtual_shape=(6, 1, 4, 6)
virtual_stride=(24, 24, 6, 1)
physical_data_ptr == virtual_data_ptr
physical_storage_ptr == virtual_storage_ptr
alias_mutation=-123.0
```

`Hp=1/2/4` parameterized test 同时覆盖 shape/stride/data pointer/mutation alias；未调用
`contiguous()`、`clone()` 或 copy fallback。

## 6. Test Evidence

以下命令使用 `/home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310/bin/python`，`VLLM_TARGET_DEVICE=cpu` 仅用于 focused CPU/control-plane validation。

1. `python -m py_compile` changed Python files：PASS；raw：
   `raw/p1-v2-r1-c-execution-foundation-01-final/03-pycompile.log`。
2. `VLLM_TARGET_DEVICE=cpu python -m pytest -q tests/v1/test_ragged_attention_spec.py tests/v1/test_ragged_kv_layout.py tests/v1/worker/test_attn_utils.py`：最终 `32 passed`。
   证明 C0、C1、C2、C3、C4 以及 Dense `attn_utils` focused behavior；raw：
   `raw/p1-v2-r1-c-execution-foundation-01-final/13-final-focused-ragged-dense.log`。
3. `VLLM_TARGET_DEVICE=cpu python -m pytest -q tests/v1/core/test_ragged_kv_cache_planner.py tests/v1/core/test_ragged_kv_cache_manager.py tests/v1/worker/test_ragged_kv_state.py`：`37 passed`。
   证明 C0 不破坏现有 planner/manager/worker control-plane tests；raw：
   `raw/p1-v2-r1-c-execution-foundation-01-final/02-ragged-state-regression.log`。
4. `VLLM_TARGET_DEVICE=cpu python -m pytest -q tests/v1/worker/test_attn_utils.py`：`5 passed`。
   证明 Dense padded/HND/diff-KV/per-token-scale reshape regression 与新 Ragged case；raw：
   `raw/p1-v2-r1-c-execution-foundation-01-final/06-dense-regression.log`。
5. `/opt/miniconda/bin/ruff check` focused on newly added layout、C1 layout module 和修改后的
   tests：最终 PASS；raw：`raw/p1-v2-r1-c-execution-foundation-01-final/14-final-ruff-focused.log`。
   对全部修改文件运行时仅报告三个 HEAD 既有问题：`kv_cache_interface.py:289 E501`、
   `attn_utils.py:94 E501`、`attn_utils.py:671 UP038`；raw：
   `raw/p1-v2-r1-c-execution-foundation-01-final/20-final-ruff-all-implementation-files.log`，本轮未修改这些无关 debt。
6. `git diff --check`：最终 PASS；raw：
   `raw/p1-v2-r1-c-execution-foundation-01-final/16-final-diff-check.log`。

测试环境观察到 `Can't initialize NVML`、`vllm._version` warning 和 Torch JIT deprecation warning；没有改变测试结果。本轮没有 GPU driver/e2e/benchmark 证据。

## 7. Removed / Replaced Behavior

- No production behavior deleted。
- No existing Dense path replaced；只在 `_reshape_kv_cache` 增加 `RaggedAttentionSpec` dedicated branch。
- No public API signature changed。
- One existing invalid test expectation（`head_size_v=64` Ragged construction）被替换为合法 equal-geometry expectation；同时新增 explicit rejection tests。
- No fallback copy path added。

## 8. Overengineering Audit

- 重复 validation：未在 C2/C3/C4 重复 `head_size_v`、quant、padding；这些由 `RaggedAttentionSpec` construction boundary 负责。C3 只保留 block-size、packing 和 raw page alignment 这三个本地 silent-wrong-risk fence。
- unused abstraction：没有新增 `RaggedGeometry` 或 runtime façade；只新增当前 Slice 直接消费的 pure helpers。
- hidden copy：`as_virtual_block_view` 使用 `.view`，C3 raw materialization 也使用 dtype view；没有 `reshape/contiguous/clone/copy`。
- duplicate authority：placement 仍由 `MemberPlacementMap` 唯一持有；Torch tensors 只是 derived execution representation。
- defensive try/except：没有新增 broad exception/fallback。
- pre-existing dirty comments：`ragged_kv_cache_manager.py`、`sched/output.py` 的 start dirty 内容未被本 Slice 扩展。

## 9. NOT YET PROVEN

- actual Ragged KV write：未调用 `reshape_and_cache_flash`，留给 `P1-V2-R1-D1-RAGGED-KV-WRITE-01`。
- actual FlashAttention read：未实现 member-major adapter，留给 D2/D3。
- real Engine identity：production `Attention.get_kv_cache_spec()` 仍未根据 `page_group_size` 激活 Ragged，符合本 Slice out-of-scope。
- GPU backing on real CUDA allocation：本轮 focused backing test 使用 CPU tensor；layout contract 已通过 Torch view alias proof，但未做 engine/GPU smoke。
- scheduler/model-runner production wiring、prefill/mixed、quantized/unequal K/V、TP2、prefix cache、Triton/CUDA Graph：均未覆盖。

## 10. Slice 状态与下一步

本 Slice 的实现和 focused verification 已完成，状态为：`PASS_PENDING_WEB_REVIEW`。

```text
Next Allowed Action: WEB_REVIEW_CURRENT_SLICE
Proposed Next Action: Web review current C0-C4 diff/evidence; only after review consider P1-V2-R1-D1-RAGGED-KV-WRITE-01.
```

本记录不批准下一 Slice，不启用 production Ragged runtime，也不改变 P1/P2 Gate。
