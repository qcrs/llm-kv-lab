# P2 Source Manifest

## Technical Baseline

P2 v3 Implementation-Audited design is primary。旧 KV Retention/QOffload/QCache 不继承。

## Implementation Pins

- vLLM 568afb3...。
- LMCache a7afadebb9248b62b5c533ce2c12297e9d94fc4a。
- torch 2.11 ABI family。
- local filesystem L2。

## Roles

| Role | Source | Provides |
|---|---|---|
| UPSTREAM | vLLM pinned | KV Connector、paged runtime、layer hooks |
| UPSTREAM | LMCache pinned | MemoryObj、Serde ABC、AsyncSerdeProcessor、L2 wrapper、filesystem、FP8/TurboQuant、RequestTracker |
| REFERENCE | CacheGen | compressed KV loading motivation |
| REFERENCE | newer vLLM 0ecc284 | canonical pages、pinned host、stream/event/buffer pools、transfer fences |
| REFERENCE ONLY | LMCache f9addd2 | later evolution; torch2.13 incompatible baseline |
| TEST REFERENCE | LMCache a7afade | `tests/v1/distributed/serde/test_serde_fs_e2e.py` real FS Serde lifecycle |
| TEST REFERENCE | LMCache a7afade | `tests/v1/distributed/serde/test_turboquant.py` MemoryLayout/size/codec test pattern |

## Our Ownership

Block-INT8 spec、Serde/factory/estimator、PyTorch/Triton codec、E2E、byte invariant、crossover/profiling。不能声称创建 connector/async/L2 framework或首次给 LMCache压缩。

## Source Truth

same-commit implementation + runtime observation > same-commit design docs > later snapshots/reference > hypothesis。LMCache docs/code漂移时以 code + E2E为准。

## Explicit Gaps

- pinned path first raw D2H前的 codec hook未确认；
- actual MemoryObj device/layout需冻结；
- Triton INT8 kernel/layout是own mechanical implementation；
- performance/crossover需测。

这些不阻塞 V1，pre-D2H留V3。



## v1.2 Final-Contract Baseline

完整 v3 reasoning snapshot：`docs/90-reference/baselines/`（从项目根路径理解；若当前文件已在 docs 内则相对路径按目录导航）。Canonical corrections：`docs/90-reference/BASELINE-RESOLUTION.md`。Snapshot 是 read-only reference，不自动批准 implementation API。

## v1.2 Canonical Codec Domain

`Block-INT8` 的 block 是 LMCache object-domain quantization block，默认按 `K/V × layer × KV head × fixed token group × head_dim` 定义；不要求与 vLLM BlockPool page 对齐。Connector/RequestTracker 已承担 physical slot 到 object 的桥接。

具体 reference 使用方式与 project ownership 见 `../90-reference/REFERENCE-PORTING-LEDGER.md`。
