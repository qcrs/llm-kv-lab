# P1 M0 Static / Oracle Closure Report

日期：2026-08-25（Asia/Shanghai）  
结论类型：`READY_FOR_WEB_REVIEW`，不是 `P1-G0 PASS`

## 1. 背景与边界

P1 M1 implementation 之前必须关闭 pinned source、固定配置和最小 data-plane oracle
的不确定性。本轮只做 source audit、GPU0 bounded probe、项目-owned oracle、文档和
raw evidence；不修改 `worktrees/p1-vllm-reclaim/vllm/**` 或 `third_party/vllm/**`，不
添加 `effective_kv_len`，不实现 reclaim、scheduler free、ACK、Triton、benchmark、
NSYS 或 NCU。M0 closure 只能提交 Web review，不能自行批准 M1 Slice。

## 2. Fixed identity

```yaml
workspace: /home/qcrs/learning/llm-kv-lab
worktree: /home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim
branch: p1/physical-kv-reclaim-v026
head: 568afb3a13806beb53bb2e6bd518269357b237c0
engine: V1 Engine
runner: MRV2 / vllm/v1/worker/gpu/model_runner.py
environment: .venvs/vllm-v026-torch211-cu129-py310
gpu: NVIDIA A100 80GB PCIe, GPU0
model: /data/models/Qwen3-0.6B
```

## 3. Static config closure

| 项目 | 目标 | 证据 | 状态 |
|---|---|---|---|
| MRV2 | `VLLM_USE_V2_MODEL_RUNNER=1` | `config/vllm.py:550-553`; smoke `Using V2 Model Runner` | PASS |
| TP/PP/DP | `1/1/1` | engine log；`config/parallel.py` fields/defaults | PASS |
| DCP/PCP | `1/1` | runtime world/rank log；DCP default 1；P1 fixed matrix | PASS for exercised baseline |
| one KV group | `num_kv_cache_groups=1` | MRV2 `BlockTables` single-group path；single dense Qwen3 smoke | PASS for exercised baseline |
| cascade | disabled | `ModelConfig.disable_cascade_attn` default `True` (`config/model.py:248-253`) | PASS by fixed default |
| block size | `16` | smoke constructor；cache/block-size resolution | PASS for requested value |
| kernel block size | equal to manager | `prepare_kernel_block_sizes()` (`v1/worker/utils.py:351-392`); oracle `[16]/[16]` | PASS for oracle path; full inventory TO_VERIFY |
| blocks per KV block | `1` | `BlockTables.blocks_per_kv_block = bs // kbs`; oracle yielded 1 | PASS for oracle path |
| CP interleave | `1` | `ParallelConfig.cp_kv_cache_interleave_size` default 1 | PASS for fixed baseline |

完整 static search 输出见 `20260825_m0-static-config-audit.log`。

## 4. Qwen3 positional source chain

exact source chain：

```text
Qwen3ForCausalLM → Qwen3Model → Qwen3Attention
→ get_rope() → self.rotary_emb(positions, q, k)
→ self.attn(q, k, v)
```

anchors：`vllm/model_executor/models/qwen3.py:65-167, 264-271`。这证明当前 text
Qwen3 attention 在 attention 前使用传入 `positions` 对 Q/K 应用 RoPE。`qwen3_vl.py`
的 mRoPE 是另一条 multimodal 路径，不是 `/data/models/Qwen3-0.6B` target。该 source
prerequisite 不等于 reclaim 后 positional correctness 已证明。

## 5. BlockTables structure

`BlockTables` 维护 persistent `block_tables`、active `num_blocks`、per-step
`input_block_tables` 和 `slot_mappings`；`blocks_per_kv_block` 为
`block_size // kernel_block_size`。source：`vllm/v1/worker/gpu/block_table.py:17-205`。

## 6. Row replacement oracle

项目-owned oracle：`04-experiments/project1_kv_reclaim/oracles/mrv2_blocktables_oracle.py`。
初始 row `[10,11,12,13,14,15,16,17]` 替换为 `[10,11,16,17]`：

```text
short_num_blocks [[4]]
short_row [10, 11, 16, 17, 14, 15, 16, 17]
```

结论：`overwrite=True` 更新 active count 并覆盖前缀，但不会清零旧 tail；stale tail
是可观察 backing storage，不能把 primitive 存在写成 live reclaim safety 已证明。

## 7. Gather 与 slot mapping oracle

同一 shortened row 的 `gather_block_tables()` observed：

```text
gathered_row [10, 11, 16, 17, 0, 0, 0, 0]
```

因此 gather 按 `num_blocks` 复制 active prefix 并清零剩余输出。但这不自动证明
`compute_slot_mappings()` 的 persistent-row访问安全。

positions `[0,1,16,17,32,33,48,49]` 的 observed slots：

```text
[160, 161, 176, 177, 256, 257, 272, 273]
```

这证明 current kernel 按 logical positions 计算 block index，再从 persistent row 取
ID；它不会自动把 logical position 转成 compact physical position，也不会用
`num_blocks` 重写 positions。

## 8. Partial-tail oracle

```text
effective_kv_len=70
block_size=16
physical_block_count=5
last_block_offset=6
```

未来 `effective_kv_len` 必须表示 retained valid token count，而不是
`num_blocks * block_size`；下一 decode 应写到第 5 个 physical block 的 offset 6。

## 9. F4 status

```yaml
architecture_feasibility: PASS
current_source_coupling: OBSERVED/PASS
slot_mapping_oracle: PASS (current behavior characterized)
stale_tail_behavior: OBSERVED (tail remains in backing row; gather masks it)
logical_physical_split_implementation: NOT_IMPLEMENTED
reclaim_correctness: OPEN
```

## 10. 证明与未证明

已证明：MRV2 fixed baseline 可运行；Qwen3 text path 为 positions → RoPE → attention；
row replacement、active gather、slot mapping 和 partial-tail 行为已由 exact source +
GPU oracle 记录。

未证明：reclaim transition、effective physical state、attention effective range、
logical position preservation、scheduler ownership commit、safe free、cross-request
reuse 或性能收益。

## 11. Closure recommendation

```text
M0_STATIC_ORACLE_CLOSURE: READY_FOR_WEB_REVIEW
```

这是 review recommendation，不是 Codex 对 Gate 的批准。由于 stale-tail 已被观察，
M1 review 必须明确 active-range proof 或 defensive clear-old-range 方案后，才能批准
runtime implementation。

## 12. Evidence Index / 下一步

```text
04-experiments/project1_kv_reclaim/raw/m0-closure/*_agents-rule-audit.log
04-experiments/project1_kv_reclaim/raw/m0-closure/20260825_m0-static-config-audit.log
04-experiments/project1_kv_reclaim/raw/m0-closure/20260825_qwen3-rope-source-audit.log
04-experiments/project1_kv_reclaim/raw/m0-closure/20260825_mrv2_blocktables_oracle.log
04-experiments/project1_kv_reclaim/raw/source-audit/20260825_f4-f7-gpu0-validation.log
```

下一步：`WEB_REVIEW_M0_STATIC_ORACLE_CLOSURE`。review 前不创建 M1 Slice，不修改
vLLM runtime source，不进入 reclaim/Triton/benchmark。
