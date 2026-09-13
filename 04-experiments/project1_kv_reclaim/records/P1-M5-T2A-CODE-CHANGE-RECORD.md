# P1-M5-T2A — Source Fence + Execution Preparation Code Change Record

## Slice Summary

- Purpose：在真实 vLLM v0.26.0 Worker post-forward seam 建立只读 V2 source fence 和 future execution preparation。
- Source HEAD before：`9ea5d804ca6cc3a7a7212d83d626acb1ddd023ea`
- Source HEAD after：`9ea5d804ca6cc3a7a7212d83d626acb1ddd023ea`（未创建 commit）
- Files Added：`tests/v1/worker/test_gpu_compaction_preparation.py`；本 workspace 的本记录与 raw log。
- Files Modified：`vllm/v1/worker/gpu/model_runner.py`；另有 M5-T1 carried-forward files。
- Files Deleted：`NONE`
- Production LOC added/deleted：T2A production semantic change `171/1`（model_runner diff；另有 1 行 M5-T1 comment cleanup）。
- Test LOC added/deleted：`178/0`（untracked focused module）；M5-T1 carried-forward test diff 不计入本轮。
- Architecture Deviations：`NONE`
- Runtime Mutation Introduced?：`NO`
- KV Payload Mutation Introduced?：`NO`
- Ownership Mutation Introduced?：`NO`

## 1. Slice 结论

`P1-M5-T2A = PASS`。

本轮在 `GPUModelRunner` 的 `model.forward()` 返回后、`ExecuteModelState`
发布前建立只读 source fence。实现会解析 request 的 persistent
`req_state_idx` 与 current-batch `batch_idx`，验证 post-forward physical
source E、active block prefix、keep set、目标 block count 以及所有 layer KV
cache 的确定性结构条件，并生成 private `_PreparedCompaction`。

本轮未调用 M4 compaction primitive，未修改 KV payload、Worker
`effective_kv_len`、Worker BlockTable、Scheduler canonical ownership，未释放 block。

## 2. Source Identity

- Workspace：`/home/qcrs/learning/llm-kv-lab`
- P1 worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Branch：`p1/v2-token-compaction-v026`
- Start HEAD：`9ea5d804ca6cc3a7a7212d83d626acb1ddd023ea`
- End HEAD：`9ea5d804ca6cc3a7a7212d83d626acb1ddd023ea`
- Commit created：`NO`
- Start status：M5-T1 的 5 个 tracked modified files；本轮未 reset、discard 或覆盖。

## 3. Source Facts 与实现判断

### SOURCE_FACT

- `InputBatch.req_ids[batch_idx]` 表达 current forward batch identity。
- `InputBatch.idx_mapping_np[batch_idx]` 将 `batch_idx` 映射到 persistent
  `req_state_idx`。
- `InputBatch.effective_kv_seq_lens[batch_idx]` 是包含本轮 query 的
  post-forward physical extent；它不能由 stale persistent
  `req_states.effective_kv_len[req_state_idx]` 替代。
- `BlockTables.num_blocks.np[0, req_state_idx]` 给出 single-group active row
  长度；`BlockTables.block_tables[0].gpu[req_state_idx, :source_num_blocks]`
  是 exact active prefix。
- physical allocation block size 来自 `BlockTables.block_sizes[0]`。
- `model.forward()` 在 `GPUModelRunner.execute_model()` 内完成后才创建
  `ExecuteModelState`，因此该 publication seam 可以读取 post-forward source。

### DESIGN_DECISION

- 使用 private frozen dataclass `_PreparedCompaction`，避免扩大 public transport
  contract；字段完整保存 future T2B 所需的 identity、index domains、source row、
  keep set 与 derived destination shape。
- `_prepare_v2_compactions()` 对空 plan mapping 立即返回 `{}`，不对不参与 V2 的
  normal execution 强加 single-group 或 KV layout guard。
- `keep_member_indices` 只验证，不排序、不去重、不修复；prepared form 转为 tuple，
  保持 producer member order 并防止后续 accidental mutation。
- layer prevalidation 检查所有 `self.kv_caches`：4D shape、physical block size、
  source block addressability 和 device compatibility。任何确定性不兼容都在 future
  destructive T2B 开始前失败。
- 未采用 `source_num_blocks == ceil(source_E / block_size)` 作为 invariant；只验证
  capacity `source_num_blocks * block_size >= source_E`。

## 4. Change Entries

### CHG-T2A-01

File:
`vllm/v1/worker/gpu/model_runner.py`

Change type:
`MODIFIED`

Final symbol:
`_PreparedCompaction`

Final line range:
`L126-L137`

Previous responsibility:
不存在 T2A prepared execution representation；M5-T1 仅将
`CompactionPlanData` 保存到 `ExecuteModelState`。

New responsibility:
以 private immutable representation 保存 `request_id`、`req_state_idx`、
`batch_idx`、source E/block row、`block_size`、keep set、K 与
`new_num_blocks`。

Exact change:
新增 `dataclasses.dataclass` import 与 frozen `_PreparedCompaction` dataclass。

Why required:
future T2B 需要消费经过 source fence 的确定输入，而不是重新解释 stale plan 或
混淆 batch/persistent index domain。

Requirement mapping:
T2A §4、§8、§10。

Classification:
`DESIGN_DECISION`

Runtime mutation introduced:
`NO`

Compatibility impact:
private symbol；无 public API 或 serialization 变化。

Tests covering it:
`test_prepare_v2_compaction_resolves_distinct_index_domains_without_mutation`、
`test_prepare_v2_compaction_keep_all_is_valid`。

### CHG-T2A-02

File:
`vllm/v1/worker/gpu/model_runner.py`

Change type:
`MODIFIED`

Final symbol:
`GPUModelRunner._prepare_v2_compactions()`

Final line range:
`L1806-L1934`

Previous responsibility:
不存在 post-forward V2 source-resolution/source-fence helper。

New responsibility:
只读解析 request identity、`req_state_idx`、`batch_idx`、post-forward source E、
active BlockTable prefix；验证 plan fences、keep set、capacity、single KV group 和
all-layer structure；计算 K 与 `new_num_blocks`。

Exact change:

- 空 plan fast path；
- single-KV-group、positive physical `block_size` 与 initialized layer guard；
- request existence/current-batch/index-mapping fence；
- 从 `effective_kv_seq_lens[batch_idx]` 读取并校验 post-forward source E；
- 从 `num_blocks.np[0, req_state_idx]` 和 persistent block table exact prefix 解析
  source blocks；
- keep non-empty/integer/non-negative/strictly-increasing/in-range validation；
- `K=len(keep)` 和 `new_num_blocks=cdiv(K, block_size)` derivation；
- all-layer 4D/block-size/block-address/device prevalidation；
- 返回 request-indexed `_PreparedCompaction`，不写入 runtime state。

Why required:
future T2B 必须在任何 destructive mutation 前证明 plan 与实际 post-forward
physical source 一致。

Requirement mapping:
T2A §6、§7、§8、§9、§10、§11。

Classification:
`SOURCE_FACT_DRIVEN`

Runtime mutation introduced:
`NO`

Compatibility impact:
仅当当前 execution 实际携带 V2 plan 时执行；正常空-plan path 返回 `{}`。

Tests covering it:
`tests/v1/worker/test_gpu_compaction_preparation.py` 全部 16 tests。

### CHG-T2A-03

File:
`vllm/v1/worker/gpu/model_runner.py`

Change type:
`MODIFIED`

Final symbol:
`GPUModelRunner.execute_model()` / `ExecuteModelState`

Final line range:
`L2166-L2177`、`L2435-L2442`

Previous responsibility:
M5-T1 在 forward 后直接把 raw `compaction_plans` 放入
`ExecuteModelState`，未解析 source fence。

New responsibility:
在 `model.forward()` 成功返回后调用 `_prepare_v2_compactions()`，然后只发布
`prepared_compactions`。

Exact change:
用 post-forward prepared result 替换 `ExecuteModelState.compaction_plans` raw plan
field；未新增执行、commit 或 result publication。

Why required:
保证 future T2B seam 位于 frozen `POST_FORWARD_PRE_EXECUTE_STATE` boundary。

Requirement mapping:
T2A §5、§10、§11。

Classification:
`SOURCE_FACT_DRIVEN`

Runtime mutation introduced:
`NO`

Compatibility impact:
`ExecuteModelState` 为内部 carrier；T2A 仅替换 M5-T1 的尚未消费 raw plan field。

Tests covering it:
focused helper tests；M5-T1 transport regression；M4/V1 regression。

### CHG-T2A-04

File:
`tests/v1/worker/test_gpu_compaction_preparation.py`

Change type:
`ADDED`

Final symbol:
T2A focused test module

Final line range:
`L1-L178`

Previous responsibility:
不存在 Worker T2A source-fence focused suite。

New responsibility:
覆盖 valid plan、distinct index domains、missing/out-of-batch request、stale E/block
count/index mapping、invalid keep variants、keep-all、derived block count、capacity、
single-group guard、all-layer prevalidation、empty-plan normal path和无 mutation。

Exact change:
新增 16 个 CPU-only focused tests；使用 minimal fake runner state，不调用 model 或
M4 kernel。

Why required:
以最低成本验证 source-resolution contract 和 fail-before-mutation behavior。

Requirement mapping:
T2A §12。

Classification:
`TEST_ONLY`

Runtime mutation introduced:
`NO`

Compatibility impact:
无 production compatibility impact。

Tests covering it:
本文件自身，最终与 M5-T1 suite 合并结果 `23 passed`。

### CHG-T2A-05

File:
`vllm/v1/core/sched/output.py`

Change type:
`MODIFIED`

Final symbol:
`SchedulerOutput.compaction_plans` adjacent comment

Final line range:
`L320-L324`

Previous responsibility:
M5-T1 已新增 plan field；其相邻注释有一个 trailing whitespace，导致
`git diff --check` 失败。

New responsibility:
语义不变。

Exact change:
仅删除注释末尾 whitespace。

Why required:
满足 mandatory diff hygiene check；没有格式化其他代码。

Requirement mapping:
T2A §17、§18 diff evidence。

Classification:
`EVIDENCE_ONLY`

Runtime mutation introduced:
`NO`

Compatibility impact:
`NONE`

Tests covering it:
`git diff --check`。

## 5. Carried-forward M5-T1 Worktree Changes

以下 tracked modifications 在 T2A 开始前已经存在，属于 frozen M5-T1，不是本轮
新增语义：

- `tests/v1/core/test_reclaim_transport.py`
- `vllm/v1/core/sched/output.py`（除 CHG-T2A-05 whitespace cleanup）
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/outputs.py`
- `vllm/v1/worker/gpu/model_runner.py` 的 T1 transport 部分

本轮未 reset、discard、重写或 generalize 这些 T1 contracts。

## 6. Runtime Mutation Audit

- KV payload mutation：`NO`
- Worker `effective_kv_len` V2 result mutation：`NO`
- Worker BlockTable V2 prefix mutation：`NO`
- Scheduler canonical `req_to_blocks` mutation：`NO`
- BlockPool free / deferred-free change：`NO`
- `compact_paged_kv_triton_2d()` invocation：`NO`
- CompactionResult publication：`NO`（属于 T2B）

focused test 会 snapshot KV caches、persistent BlockTable 与 persistent
`effective_kv_len`，调用 preparation 后逐项比较 unchanged。

## 7. Test Evidence

### T2A + M5-T1

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" \
/home/qcrs/learning/llm-kv-lab/.venv/bin/python -m pytest -q \
tests/v1/worker/test_gpu_compaction_preparation.py \
tests/v1/core/test_reclaim_transport.py
```

OBSERVED：exit 0，`23 passed, 15 warnings in 6.55s`。

### M4 regression

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" \
/home/qcrs/learning/llm-kv-lab/.venv/bin/python -m pytest -q \
tests/v1/worker/test_gpu_kv_compaction.py
```

OBSERVED：exit 0，`52 passed, 15 warnings in 18.05s`，0 skip。

### V1 reclaim regression

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" \
/home/qcrs/learning/llm-kv-lab/.venv/bin/python -m pytest -q \
tests/v1/worker/test_gpu_reclaim_commit.py
```

OBSERVED：exit 0，`14 passed, 15 warnings in 6.87s`。

### effective-state regression

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" \
/home/qcrs/learning/llm-kv-lab/.venv/bin/python -m pytest -q \
tests/v1/worker/test_gpu_effective_kv_state.py
```

OBSERVED：exit 0，`13 passed, 15 warnings in 5.68s`。

## 8. Static / Diff Checks

- changed Python files `py_compile`：exit 0。
- project `.venv` 的 exact `python -m ruff check ...`：exit 1，环境缺少
  `ruff`，错误为 `No module named ruff`。
- available `/opt/miniconda/bin/ruff check` 对 T2A production/test files：exit 0，
  `All checks passed!`。
- `git diff --check`：exit 0。

## 9. Files Inventory

### ADDED

- `tests/v1/worker/test_gpu_compaction_preparation.py` — T2A focused tests。
- `04-experiments/project1_kv_reclaim/P1-M5-T2A-CODE-CHANGE-RECORD.md` — 本记录。
- `04-experiments/project1_kv_reclaim/raw/P1-M5-T2A-IMPLEMENTATION.log` — raw evidence。

### MODIFIED

- `vllm/v1/worker/gpu/model_runner.py` — T2A source fence、prepared representation、
  post-forward publication。
- `vllm/v1/core/sched/output.py` — 仅清理 M5-T1 comment trailing whitespace；
  transport semantics 不变。
- 另有 4 个 start-status 中已存在的 M5-T1 carried-forward modified files；见 §5。

### DELETED

`NONE`

## 10. 该结果证明了什么

OBSERVED evidence 证明：plan 可在 post-forward seam 被解析为正确 request 的两个
index domains；source E 来自 current batch physical extent；block row 只读取 active
prefix；keep 与 derived shape fence 按 contract 失败或成功；all-layer deterministic
structure 在 future mutation 前完成检查；相关 T1/M4/V1 tests 未回归。

## 11. 该结果没有证明什么

本轮没有证明 KV compaction payload correctness、transactional all-layer mutation、
Worker row/E commit、result publication、Scheduler reconcile/free 或性能收益。这些均不
属于 T2A。

## 12. Next Allowed Action

`WEB_REVIEW_CURRENT_SLICE`

Proposed Next Action：review T2A diff/evidence 后再决定是否批准 M5-T2B。Codex 在本轮
停止，不实现 T2B。
