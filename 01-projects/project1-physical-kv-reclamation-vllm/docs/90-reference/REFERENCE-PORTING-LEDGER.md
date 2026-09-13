# P1 Reference / Porting Ledger

目的：明确每个实现点到底“借了什么”，防止 reference primitive、semantic adaptation 与 project-owned code 混在一起。实现后必须把 `Our Target` 从 `TO_FREEZE` 更新成真实文件/symbol。

| Our capability | Reference repo | Commit | Reference symbol/path | Use Type | Our Target | Notes |
|---|---|---|---|---|---|---|
| BlockTables row rebuild | vLLM | `568afb3...` | `vllm/v1/worker/gpu/block_table.py::BlockTables.append_block_ids(overwrite=True)` | SOURCE_API | TO_FREEZE | MRV2 primitive; live stale-tail seam still audited separately |
| running append vs resume replacement | vLLM | `568afb3...` | `gpu/model_runner.py::add_requests/update_requests` | SEMANTIC_REFERENCE | TO_FREEZE | reclaim is a third explicit transition |
| canonical request ownership | vLLM | `568afb3...` | `single_type_kv_cache_manager.py::req_to_blocks` | SOURCE_API | TO_FREEZE | scheduler-side allocator truth |
| effective occupancy allocation | Tangram | `8e1cbfa...` | `kv_cache_manager.py` effective cached tokens | SEMANTIC_REFERENCE | TO_FREEZE | port semantic, do not cherry-pick wholesale |
| logical position vs cache slot | Tangram | `8e1cbfa...` | effective slot positions (interpreted against MRV2 `prepare_pos_seq_lens` / `compute_slot_mappings`) | SEMANTIC_REFERENCE | TO_FREEZE | uniform scalar adaptation; current seam remains TO_VERIFY |
| worker→scheduler freed IDs | Tangram | `8e1cbfa...` | `outputs.py` + scheduler free path | SEMANTIC_REFERENCE | TO_FREEZE | publication/lifetime protocol |
| compaction tensor semantic | Tangram | `8e1cbfa...` | `attention/compression/executor.py` | SEMANTIC_REFERENCE | TO_FREEZE | V2 oracle/flow |
| invalid compaction atomicity | Sparse-vLLM | `c146433...` | `tests/test_static_eviction_compaction.py` | TEST_PATTERN | TO_FREEZE | copy test methodology, not implementation |
| paged Triton addressing | vLLM | `568afb3...` | `triton_reshape_and_cache_flash.py` | SOURCE_API / KERNEL_PATTERN | TO_FREEZE | own arbitrary gather kernel |
| async deferred free | newer vLLM | `0ecc284...` | deferred block free / processed-safe boundary | SEMANTIC_REFERENCE | V3 only | post-baseline reference |
| periodic trigger | NVIDIA KVPress | current audited snapshot | `DecodingPress` | MOTIVATION_ONLY / POLICY_PATTERN | V3 only | no allocator code borrowed |

## Use Type Vocabulary

- `SOURCE_API`：直接使用 upstream/pinned API/primitive；
- `SEMANTIC_REFERENCE`：重新实现相同 contract，不复制整个 patch；
- `TEST_PATTERN`：借测试结构/oracle；
- `KERNEL_PATTERN`：借 addressing/tiling pattern，kernel 逻辑由本项目实现；
- `ADAPTED_CODE`：若未来直接改写具体代码，必须记录原文件/license/header；
- `VERBATIM_CODE`：原则上避免；如发生必须保留原版权与 Apache-2.0 要求；
- `MOTIVATION_ONLY`：不进入代码 dependency。

vLLM / Tangram / Sparse-vLLM 当前 audited reference 均为 Apache-2.0；任何直接复制/改写仍需保留 attribution/header/NOTICE 要求。
