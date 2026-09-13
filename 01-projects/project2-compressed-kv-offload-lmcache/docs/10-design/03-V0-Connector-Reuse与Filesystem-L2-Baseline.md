# 03 — V0 Connector Reuse / Filesystem-L2 Baseline

P2 不一次 debug connector + FS + Serde + codec。V0 拆两个 baseline。

## V0-A — vLLM↔LMCache Reuse Smoke

目标：复现 upstream local reuse，证明：

```text
request A computes/stores shared prefix
request B reuses cached prefix
load/resume generation succeeds
```

记录 connector config、request tracker、hit provenance、exact source identities。此阶段不启 custom Serde。

## V0-B — Raw Distributed Filesystem L2

目标链：

```text
raw L1 MemoryObj
→ FSL2Adapter
→ real .data file
→ load raw object
→ restore/resume
```

记录：

- ObjectKey/chunk boundaries；
- raw object `byte_array` length；
- `.data` file size；
- `L2StoreResult.bytes_transferred`；
- raw store/load latency；
- pre-existing key behavior；
- actual MemoryObj device/layout。

## Why Separate

- V0-A fail → connector/request tracking问题；
- V0-B fail → L2 config/ObjectKey/FS问题；
- 两者都 PASS 后才允许 M1 format / M2 Serde。

## Key Namespace Rule

FS adapter 对已存在 file 的 key 会 skip store。任何 byte equality / store latency experiment 必须：

- 使用 isolated namespace/cache_salt，或
- 明确清理目标 files，或
- 把 already-present keys 从 expected written bytes中排除。

否则 `bytes_transferred=0` 可能只是 cache already existed，不是 codec bug。
