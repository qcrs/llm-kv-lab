# P1-M1-T1-S2/S3 实现记录

> Parent Task 导航：`P1-M1-T1-IMPLEMENTATION-INDEX.zh-CN.md`  
> S1 persistent-state record：`P1-M1-T1-S1-IMPLEMENTATION-RECORD.zh-CN.md`

## 1. Slice 信息

| 项目 | 值 |
|---|---|
| Parent Task | `P1-M1-T1 — Logical / Physical State Contract` |
| Slice | `P1-M1-T1-S2` + `P1-M1-T1-S3` |
| 执行日期 | 2026-08-26（Asia/Shanghai） |
| implementation worktree | `/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim` |
| branch | `p1/physical-kv-reclaim-v026` |
| baseline HEAD | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| 起始 dirty state | 已批准且未提交的 S1 修改：`states.py`、`input_batch.py`、`model_runner.py` 和新 test |
| Tangram reference | `8e1cbfa5cf82acc67fe1880df0c632f2a6485e35` |
| GPU test device | 物理 GPU2，通过 `CUDA_VISIBLE_DEVICES=2` 映射为进程内 CUDA device 0 |

本记录描述相对 pinned HEAD 的当前 S1+S2+S3 working-tree state，并在文件级说明哪些内容来自继承的 S1、哪些内容属于本轮 S2/S3。没有创建 commit。

## 2. 本轮目标

S2 修复 reclaim 后的 KV write address：模型仍使用 logical `positions`，但 `BlockTables.compute_slot_mappings()` 改为消费由 `effective_kv_len` 派生的 `cache_positions`。

S3 修复 reclaim 后的 KV read extent：`InputBatch.seq_lens` 和 `CommonAttentionMetadata.seq_lens` 继续表达 logical request length；独立的 `effective_kv_seq_lens` 被传到 FlashAttention2 builder，并最终成为 `FlashAttentionMetadata.seq_lens` / `seqused_k` 所见的 physical KV length。

本轮不执行 reclaim，不改变 block ownership，也不释放或复用任何 block。

## 3. 修改前问题

Upstream MRV2 在一个 coordinate channel 中同时派生模型位置、KV 写入位置和 attention 长度：

```text
num_computed_tokens
├─> positions ─> model/RoPE
├─> positions ─> compute_slot_mappings ─> KV write slot
└─> seq_lens ─> FlashAttention seqused_k
```

当 logical progress 与保留的 physical KV occupancy 相等时，这个共享 channel 正确。未来 whole-block reclaim 后若状态为 `logical=128, physical=64`，继续使用 upstream coupling 会把下一 token 写入 logical block 8，而不是 compact view 的 block 4，并让 attention 按 129 而不是 65 的长度读取。S2/S3 的任务是拆开这两个 execution-side channel，而不是实现产生 `128/64` 状态的 reclaim transaction。

## 4. Approved Contract

| 字段 | Owner | dtype / shape | 生命周期 | Producer | Consumer | 语义 |
|---|---|---|---|---|---|---|
| `num_computed_tokens` | `RequestState` | GPU `int32 [max_num_reqs]` | request-lifetime persistent | `add_request()` / `post_update()` | position、logical length 等 | logical progress |
| `effective_kv_len` | `RequestState` | GPU `int32 [max_num_reqs]` | request-lifetime persistent | S1 init / `post_update()` | `prepare_pos_seq_lens()` | forward-entry physical occupancy，不含当前 query |
| `positions` | `InputBatch` | GPU `int64 [num_tokens_after_padding]` | per-forward | `_prepare_pos_seq_lens_kernel()` | model / RoPE | logical token coordinate |
| `cache_positions` | `InputBatch` | GPU `int64 [num_tokens_after_padding]` | per-forward | `_prepare_pos_seq_lens_kernel()` | `compute_slot_mappings()` | effective KV sequence coordinate，不是 raw GPU slot |
| `seq_lens` | `InputBatch` / common metadata | GPU `int32 [num_reqs_after_padding]` | per-forward | `_prepare_pos_seq_lens_kernel()` | logical runtime consumers | logical base + current query length |
| `effective_kv_seq_lens` | `InputBatch` / common metadata | GPU `int32 [num_reqs_after_padding]` | per-forward | `_prepare_pos_seq_lens_kernel()` | FA2 builder | physical base + current query length |

`max_seq_len` 没有新增 physical mirror，继续作为 logical conservative upper bound；本轮没有引入 GPU→CPU synchronization。

## 5. Tangram Reference Mapping

Tangram source identity 和实际片段保存在 `01-tangram-reference.log`。

| Tangram anchor | Tangram 行为 | P1 uniform adaptation | 分类 |
|---|---|---|---|
| `vllm/v1/worker/gpu_model_runner.py:L1396-L1419` | compression 开启后，用 `effective_seq_lens_cpu + intra_seq_offset` 计算 slot positions，logical `positions_np` 保持独立 | 以 request-uniform GPU `effective_kv_len` 生成 `cache_positions`，logical `positions` 不变 | B — Mechanical Adaptation |
| 同文件 `L1664-L1695` | 把 post-compression effective length 传入 common attention metadata | 新增 optional GPU `effective_kv_seq_lens` channel | B — Mechanical Adaptation |
| 同文件 `L2997-L3009` | normal step 用 scheduled increment 推进 effective length | 继承 S1 的 `computed_delta` 同步推进两个独立 persistent tensor | B — Mechanical Adaptation |

P1 没有移植 Tangram 的 ragged paging、per-head-group length、cluster map、virtual member block table 或 compression algorithm。P1 V1 只采用每 request 单一 effective length 的 whole-block specialization。

## 6. 文件级修改清单

### `vllm/v1/worker/gpu/states.py`

- 修改位置：`RequestState.__init__`（约 L55-L64）、`add_request()`（L112-L115）、`apply_staged_writes()`（L119-L125）。
- 新增：S1 的独立 `StagedWriteTensor effective_kv_len`，初始化和 staged publication。
- 删除：无。
- 行数：`+5/-0`。
- 原因：为 S2/S3 producer 提供 request-lifetime physical base；没有 CPU mirror。

### `vllm/v1/worker/gpu/input_batch.py`

- 修改位置：`InputBuffers.__init__`（约 L23-L35）、`InputBatch`（L89-L98）、`make_dummy()`（L132-L193）、`_prepare_pos_seq_lens_kernel()` / `prepare_pos_seq_lens()`（L262-L337）、`_post_update_kernel()` / `post_update()`（约 L495-L568）。
- 新增：`cache_positions`、`effective_kv_seq_lens` buffers 与 dataclass fields。
- 修改：同一次 Triton launch 读取 logical/physical bases 并派生四个输出；padding branch 同时清零两套 request lengths。
- 继承 S1：`post_update()` 接收 `effective_kv_len`，对两个独立 tensor 应用相同 `computed_delta`。
- 删除：无语义删除。
- 行数：`+43/-0`。
- 原因：在 producer 处一次性建立 logical/physical per-forward channels，避免第二次 kernel launch。

### `vllm/v1/worker/gpu/model_runner.py`

- 修改位置：`prepare_inputs()`（约 L963-L1062）、`prepare_attn()`（L1065-L1086）、normal `post_update()` 调用（约 L1135）。
- 新增：把 persistent `effective_kv_len.gpu` 传给 producer；构造 `InputBatch` 时发布两个 derived fields。
- 修改：slot mapping consumer 从 `input_batch.positions` 替换为 `input_batch.cache_positions`。
- 继承 S1：normal post-update 调用传入 `effective_kv_len.gpu`。
- 删除：旧 slot-mapping 参数 `input_batch.positions` 一处。
- 行数：`+14/-1`。
- 原因：KV write address 必须使用 physical append coordinate，同时不改变 model/RoPE positions。

### `vllm/v1/worker/gpu/model_states/default.py`

- 修改位置：`DefaultModelState.prepare_attn()`（约 L176-L188）。
- 新增：向 `build_attn_metadata()` 传入 `input_batch.effective_kv_seq_lens`。
- 删除：无。
- 行数：`+1/-0`。
- 原因：把 per-forward physical length 从 runner transport 到 common metadata builder。

### `vllm/v1/worker/gpu/attn_utils.py`

- 修改位置：`build_attn_metadata()`（约 L568-L625）。
- 新增：optional 参数、与 `seq_lens` 相同的 request slicing，以及 common metadata field publication。
- 删除：无。
- 行数：`+4/-0`。
- 原因：机械传递 physical channel，保留 `None` fallback 和其他 caller 的兼容性。

### `vllm/v1/attention/backend.py`

- 修改位置：`CommonAttentionMetadata`（约 L430-L445）、`unpadded()`（约 L562-L577）。
- 新增：optional `effective_kv_seq_lens`；显式 dataclass reconstruction 时使用 `maybe_slice_reqs()`，避免 silent drop。
- 删除：无。
- 行数：`+4/-0`。
- 原因：`CommonAttentionMetadata.seq_lens` 必须继续保持 logical semantics，physical read extent 使用独立 channel。

### `vllm/v1/attention/backends/flash_attn.py`

- 修改位置：`_select_kv_seq_lens()`（L71-L77）、`FlashAttentionMetadataBuilder.build()`（约 L456-L679）。
- 新增：带 scope guard 的 selector。只有 `effective_kv_seq_lens` 存在、`dcp_world_size == 1` 且 `common_prefix_len == 0`（no cascade）时，normal FA2 才使用 physical length；否则回退到 logical `seq_lens`。
- 修改：normal FA2 的 scheduler `seqlens` 和 `FlashAttentionMetadata.seq_lens` 使用 guarded `kv_seq_lens`；DCP 的 `context_kv_lens` 与 cascade 的 `suffix_kv_lens` 明确继续使用 logical `seq_lens`。
- 删除：3 行旧的直接 logical/physical 混用，被 scope-guarded channel 替换。
- 行数：相对 pinned HEAD `+24/-3`（累计 diff；其中包含 S2/S3 原有修改和本轮 review-fix）。
- 原因：最终 `seqused_k` 只在已批准 normal FA2 baseline 看到 actual physical KV length；不提前声明 DCP/cascade reclaim 支持，FA kernel ABI 不变。

### `tests/v1/worker/test_gpu_effective_kv_state.py`

- 修改位置：新文件 L1-L334。
- 新增：S1 T1-T4、S2/S3 Case A-E、padded request-length oracle、`prepare_attn()` cache-position wiring、builder-level physical/fallback oracle、scope guard oracle，共 13 个 CUDA tests。
- 删除：无。
- 行数：新文件 334 行；因 untracked，未包含在普通 `git diff --numstat`。
- 原因：验证 state independence、producer formulas、metadata preservation、normal FA2 guarded selection、builder 最终输出、wiring 和 fallback。

## 7. 精确 Change Ledger

| 文件 | 函数/类 | 原始行为 | 新行为 | 新增 | 删除/替换 | 修改原因 | Reference |
|---|---|---|---|---|---|---|---|
| `states.py` | `RequestState` | 只有 logical persistent counter | 独立 physical persistent counter | `effective_kv_len` | 无 | S1 physical base | Tangram L2997-L3009 |
| `input_batch.py` | `InputBuffers` / `InputBatch` | 只有 `positions`、`seq_lens` | 增加 physical per-forward channels | 两个 tensor fields | 无 | 拆分 write/read coordinates | Tangram L1396-L1419、L1664-L1695 |
| `input_batch.py` | `_prepare_pos_seq_lens_kernel()` | logical base 生成两类输出 | logical/physical bases 在同一 launch 生成四类输出 | 3 pointer inputs/outputs | 无 | 单 producer、无额外 launch | B adaptation |
| `model_runner.py` | `prepare_attn()` | slot mapping 消费 logical `positions` | 消费 `cache_positions` | 英文语义注释 | 替换 1 个参数 | 修复 KV write address | Tangram L1407-L1417 |
| `attn_utils.py` | `build_attn_metadata()` | 只运输 logical length | optional transport physical length | 参数和 field | 无 | metadata transport | Tangram L1664-L1695 |
| `backend.py` | `CommonAttentionMetadata.unpadded()` | 不存在 physical field | slice/preserve optional physical field | field + slice | 无 | 防止 explicit reconstruction 丢字段 | source-derived |
| `flash_attn.py` | builder `build()` | scheduler / metadata 使用 logical length | 仅在 `effective field present AND DCP=1 AND no cascade` 时使用 physical；否则 logical fallback；DCP/cascade 分支保持 logical | `_select_kv_seq_lens(..., use_effective=...)` 与 scope guard | 3 行旧混用路径 | 只在已审计 normal FA2 path 修复 read extent | Tangram effective metadata + Web review |
| test | 13 tests | 无 S1-S3 semantic oracle | 覆盖 A-E、lifecycle/padding、wiring、builder final metadata、fallback、scope guard | 334 行 | 无 | correctness evidence | approved Slice + Web review fix |

## 8. 新增字段表

新增字段的 owner、dtype、shape、persistent 属性、producer 和 consumer 已完整列在第 4 节。关键区别是：`effective_kv_len` 是 persistent；`cache_positions` 与 `effective_kv_seq_lens` 都是 per-forward derived state。`cache_positions` 表示 KV sequence coordinate，最终还需经 block table 映射成 raw physical slot。

## 9. Dataflow Before / After

修改前：

```text
num_computed_tokens
       |
       +--> positions --> model/RoPE
       |        |
       |        +------> slot mapping --> KV write
       |
       +--> seq_lens --> common metadata --> FA seqused_k
```

修改后：

```text
Logical path
num_computed_tokens
       +--> positions ----------------------> model/RoPE
       +--> seq_lens -----------------------> logical runtime consumers

Physical path
effective_kv_len
       +--> cache_positions ----------------> slot mapping --> KV write
       +--> effective_kv_seq_lens
                 --> CommonAttentionMetadata optional field
                 --> FA builder kv_seq_lens --> seqused_k

Normal completion
computed_delta
       +--> num_computed_tokens += delta
       +--> effective_kv_len     += delta
```

两套 persistent state 只共享 normal execution delta，不执行绝对值镜像。

## 10. Oracle Tests

| Case | 输入 | OBSERVED / 断言 | 结果 |
|---|---|---|---|
| A baseline equality | logical=128, physical=128, q=1, block=16 | `positions=[128]`，`cache_positions=[128]`，lengths 都为 129，block indices 都为 8 | PASS |
| B divergence | logical=128, physical=64, q=1 | `positions=[128]`，`cache_positions=[64]`，lengths 为 129/65，block indices 为 8/4 | PASS |
| C multi-token | logical=128, physical=64, q=4 | logical `[128..131]`，physical `[64..67]`，lengths 132/68 | PASS |
| D mixed batch | A=128/64/q1，B=96/80/q2 | common logical `[129,98]`、physical `[65,82]`；`unpadded()` 同步 slice | PASS |
| E fallback | physical field 分别 present / `None` | normal selector 在 scope 允许时返回 physical；字段为 `None` 时返回 logical | PASS |
| Padding | 1 request，capacity 4 | logical `[129,0,0,0]`、physical `[65,0,0,0]` | PASS |
| Wiring | fake `BlockTables` + `GPUModelRunner.prepare_attn()` | `compute_slot_mappings()` 第三个参数 identity 等于 `cache_positions`，不是 `positions` | PASS |
| Builder final metadata | logical `[129,98]`，physical `[65,82]` | normal builder 输出 `metadata.seq_lens=[65,82]`；`None` 时输出 `[129,98]` | PASS |
| Scope guard | physical field 存在但 selector 关闭 | `use_effective=False` 返回 logical tensor，作为 DCP/cascade 禁用 physical substitution 的 oracle | PASS |

同一 test 文件中的 S1 T1-T4 也通过：初始化 37/37、baseline 128/128→129/129、divergence 128/64→129/65、remove/reuse 后显式覆盖为 37/37。最终共 13 个 CUDA tests。

## 11. 执行命令

关键命令如下，完整 stdout/stderr 见 Evidence Index：

```bash
CUDA_VISIBLE_DEVICES=2 \
  /home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310/bin/python \
  -m pytest -q tests/v1/worker/test_gpu_effective_kv_state.py

/home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310/bin/python \
  -m py_compile <本 Slice 涉及的 8 个 Python 文件>

git diff --check
git diff --stat
git diff --numstat
git status --short
git diff
```

## 12. Test Evidence

- GPU2 final targeted test：`13 passed, 15 warnings in 8.87s`。
- warning 仅包括 editable source 缺少 generated `vllm._version` 和 Torch `script_method` deprecation；没有 test failure。
- `py_compile`：PASS，无输出。
- `git diff --check`：PASS，无输出。
- GPU2 在最终快照中约占用 21.4 GiB 且利用率 100%，因此没有再启动 `gpu_memory_utilization=0.2` 的 vLLM engine smoke；没有终止外部进程。此前 targeted tests 本身不创建 engine，也不使用该 engine 参数。

## 13. Git Diff Summary

相对 pinned HEAD、仅 implementation worktree tracked production files：

```text
7 files changed, 95 insertions(+), 4 deletions(-)
```

```text
4   0  vllm/v1/attention/backend.py
24  3  vllm/v1/attention/backends/flash_attn.py
4   0  vllm/v1/worker/gpu/attn_utils.py
43  0  vllm/v1/worker/gpu/input_batch.py
14  1  vllm/v1/worker/gpu/model_runner.py
1   0  vllm/v1/worker/gpu/model_states/default.py
5   0  vllm/v1/worker/gpu/states.py
```

另有 untracked 新 test 334 行。上述 tracked diff 包含先前批准的 S1/S2/S3 以及本轮 review-fix；没有 commit，HEAD 仍为 baseline。

## 14. 未修改范围

- Scheduler：未修改。
- `KVCacheManager`：未修改。
- `BlockPool`：未修改。
- `BlockTables` representation/algorithm：未修改。
- slot-mapping kernel：未修改，只替换其 position input channel。
- FA forward kernel ABI / native `.cpp` / `.cu`：未修改。
- `third_party/tangram` 与 `third_party/vllm`：未修改。
- reclaim policy、block ownership/free/reuse、worker ACK：未实现。

## 15. Remaining Risks

- real reclaim 尚未实现，因此未验证真实 `128/64` 状态如何由 transaction 安全产生。
- block-table row replacement、stale-tail isolation、allocator ownership commit、cross-request reuse 尚未集成。
- 本 Slice 只对 A100 + FA2 + eager + single GPU baseline 执行 targeted semantic oracle；DCP/PCP、cascade、spec decode、CUDA Graph、FA3/FA4 等不在支持承诺内。
- `max_seq_len` 仍为 logical conservative upper bound，这是批准行为，不是 exact physical max。
- GPU2 被其他进程持续占用，本轮没有额外 engine-level smoke；已有 unit oracle 覆盖 producer 和 selected FA length，但不等价于 real reclaim E2E。

没有发现 `DESIGN_CONFLICT`。

## 16. Evidence Index 与下一步

- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-implementation/20260826-110036_00-source-identity.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-implementation/01-tangram-reference.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-implementation/02-targeted-tests-gpu2-final.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-implementation/02-gpu2-occupancy.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-implementation/03-pycompile.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-implementation/03-diff-check.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-implementation/05-final-validation.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/00-source-identity.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/01-pre-fix.patch`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/01-pre-fix-diff.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/02-pycompile.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/02-targeted-tests.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/03-post-fix.patch`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/03-post-fix-diff.log`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/pre-post.patch.delta`
- `04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/04-final-validation.log`

本 Slice 已完成 execution-side logical/physical write coordinate 与 attention read length 的最小拆分。最终 FA2 行为是：`physical-if-present AND DCP=1 AND no cascade`；否则使用 logical fallback，且 DCP/cascade 不声明支持。未完成真实 reclaim、block ownership transaction 或 allocator reuse。

`Next Allowed Action: WEB_REVIEW_CURRENT_SLICE`

review-fix 文档 finalization 已完成；停止，不自动进入下一 Slice，等待 ChatGPT Web review。

## Web Review Fix

### 1. Review Finding

Web review 给出 `CONDITIONAL PASS`，要求收紧 S3 physical-length substitution 的适用范围，并补齐两条 wiring/oracle 证据。原实现将 `kv_seq_lens` 同时用于 normal、DCP 和 cascade 分支；虽然当前 fixed baseline 是 `DCP=1`、cascade disabled，但 DCP/cascade reclaim contract 尚未完成 reference-backed audit，因此不能让它们自动继承 S3 semantics。

### 2. 修改前行为

`vllm/v1/attention/backends/flash_attn.py::FlashAttentionMetadataBuilder.build()` 在约 `L471-L474` 无条件通过 `_select_kv_seq_lens()` 选择 physical channel；因此：

```text
DCP:     context_kv_lens = kv_seq_lens - query_lens
Cascade: suffix_kv_lens = kv_seq_lens[:num_reqs] - common_prefix_len
Normal:  scheduler seqlens = kv_seq_lens
```

本轮 review fix 前，三条路径都可能使用 `effective_kv_seq_lens`。

### 3. 修改后行为

`build()` 现在先计算：

```text
use_effective_kv_seq_lens = (
    effective field exists
    and dcp_world_size == 1
    and not use_cascade
)
```

随后：

```text
Normal FA2: kv_seq_lens = effective_kv_seq_lens
DCP:        context_kv_lens = seq_lens - query_lens
Cascade:    suffix_kv_lens = seq_lens - common_prefix_len
```

最终 `FlashAttentionMetadata.seq_lens` 在 pinned normal path 使用 physical length；DCP/cascade 保留 upstream logical length。`_select_kv_seq_lens(..., use_effective=False)` 提供了可测试的 scope guard，但没有引入新的 metadata 或 execution channel。

### 4. Scope Rationale

这不是断言 DCP/cascade 一定错误，而是当前 P1 fixed scope 没有完成它们与 reclaim 的 positional/partition/cascade contract audit。保守做法是只在已批准的 `DCP=1`、无 cascade normal FA2 path 开启 physical visibility，避免未验证路径产生隐含支持承诺。

### 5. Added Tests

- `test_prepare_attn_wires_cache_positions`：构造轻量 fake `BlockTables`，直接调用 `GPUModelRunner.prepare_attn()`，捕获第三个参数并断言它 identity 等于 `input_batch.cache_positions`，且不等于 logical `positions`。
- `test_flash_builder_outputs_effective_lengths_and_fallback`：实际调用 `FlashAttentionMetadataBuilder.build()`；physical field 存在时断言 metadata `[65,82]`，为 `None` 时断言 fallback `[129,98]`。
- `test_effective_length_scope_guard`：验证 selector 在 `use_effective=False` 时返回 logical tensor，作为 DCP/cascade 禁用 physical substitution 的窄 oracle。

### 6. Test Result

命令：

```bash
CUDA_VISIBLE_DEVICES=2 \
/home/qcrs/learning/llm-kv-lab/.venvs/vllm-v026-torch211-cu129-py310/bin/python \
-m pytest -q tests/v1/worker/test_gpu_effective_kv_state.py
```

结果：`13 passed, 15 warnings in 8.87s`。`py_compile` 与 `git diff --check` 均通过。

Evidence：

```text
04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/00-source-identity.log
04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/01-pre-fix.patch
04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/01-pre-fix-diff.log
04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/02-pycompile.log
04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/02-targeted-tests.log
04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/03-post-fix.patch
04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/03-post-fix-diff.log
04-experiments/project1_kv_reclaim/raw/m1-t1-s2-s3-review-fix/pre-post.patch.delta
```

### 7. Change Ledger

本轮 review-fix 只修改两个文件：

| 文件 | 修改 | 删除/替换 | 原因 |
|---|---|---|---|
| `vllm/v1/attention/backends/flash_attn.py` | `_select_kv_seq_lens()` 增加 `use_effective`；`build()` 提前计算 `use_cascade` 与 scope guard；normal 使用 physical | DCP `context_kv_lens` 和 cascade `suffix_kv_lens` 恢复使用 logical `seq_lens` | 消除未经 audit 的 scope leakage |
| `tests/v1/worker/test_gpu_effective_kv_state.py` | 增加 wiring、builder-level、fallback/scope tests | 无 | 证明 producer→InputBatch→prepare_attn 以及最终 builder metadata |

Review-fix delta 相对 pre-fix patch 可由 `pre-post.patch.delta` 查看；S1/S2/S3 accumulated diff 仍保留在 `03-post-fix.patch`。

### 8. Remaining Risks

DCP、cascade、PCP、spec decode、CUDA Graph、FA3/FA4 仍未验证；本轮只保证 normal pinned FA2 path。真实 reclaim transaction、BlockTable stale-tail isolation、scheduler ownership commit、physical free/reuse 仍未实现。
