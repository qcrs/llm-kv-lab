# P2 Reference / Porting Ledger

目的：在实现/面试中明确 upstream framework、reference pattern 与 project-owned codec 的边界。实现后将 `Our Target` 更新为真实路径/symbol。

| Our capability | Reference repo | Commit | Reference symbol/path | Use Type | Our Target | Notes |
|---|---|---|---|---|---|---|
| vLLM↔LMCache bridge | vLLM | `568afb3...` | `LMCacheConnectorV1` | SOURCE_API | upstream use | 不声称自建 connector |
| request/block→slot tracking | LMCache | `a7afade...` | `vllm_v1_adapter.py` | SOURCE_API | upstream use | allocator mapping由 adapter承担 |
| Serde ABC/factory | LMCache | `a7afade...` | distributed serde base/factory | SOURCE_API | TO_FREEZE | custom Block-INT8 plugin |
| simple codec plugin pattern | LMCache | `a7afade...` | `serde/fp8.py` | SEMANTIC_REFERENCE / API_TEMPLATE | TO_FREEZE | 不复制“首次 compression” claim |
| temp/lifecycle async chaining | LMCache | `a7afade...` | `SerdeL2AdapterWrapper` | SOURCE_API | upstream use | 不重写 eventfd/L1 lock/thread loop |
| real filesystem Serde E2E | LMCache | `a7afade...` | `tests/v1/distributed/serde/test_serde_fs_e2e.py` | TEST_PATTERN | TO_FREEZE | replace FP8 with Block-INT8 + byte assertions |
| MemoryLayout/codec config tests | LMCache | `a7afade...` | `tests/v1/distributed/serde/test_turboquant.py` | TEST_PATTERN | TO_FREEZE | object-domain layout oracle |
| Triton codec architecture | LMCache | `a7afade...` | `serde/turboquant/*` | KERNEL_PATTERN | TO_FREEZE | own simpler INT8 encode/decode |
| FS bytes accounting | LMCache | `a7afade...` | `fs_l2_adapter.py` | SOURCE_API | evidence | `.data` / bytes_transferred oracle |
| compressed KV motivation | CacheGen | audited reference | project/readme | MOTIVATION_ONLY | none | no direct code dependency |
| pre-D2H page/lifetime primitive | newer vLLM | `0ecc284...` | generic KV offload worker | SEMANTIC_REFERENCE | V3 only | not baseline dependency |

## Use Type Vocabulary

- `SOURCE_API`：直接依赖 pinned upstream contract；
- `API_TEMPLATE` / `SEMANTIC_REFERENCE`：按同类 plugin contract 自己实现；
- `TEST_PATTERN`：复用测试结构/oracle；
- `KERNEL_PATTERN`：借 Triton reduction/staging/layout思路，kernel由本项目实现；
- `ADAPTED_CODE` / `VERBATIM_CODE`：若未来发生必须记录原文件并保留 Apache-2.0 attribution/header；
- `MOTIVATION_ONLY`：不进入代码依赖。

vLLM 与 LMCache audited baseline 均为 Apache-2.0。项目 README/面试必须区分“integrated/leveraged/reference”与“implemented”。
