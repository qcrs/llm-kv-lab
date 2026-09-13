# P1 Reference and Ownership Index

## Upstream vLLM Provides

BlockPool、BlockTable、KV managers、Paged KV、slot mapping、attention runtime、runner/scheduler framework、paged Triton address primitive。

## Tangram Inspires

effective occupancy、logical/cache position split、worker→scheduler freed IDs、copy compaction/ragged advanced reference。我们不复现或声称其 architecture。

## Sparse-vLLM Inspires

cache-manager ownership、scalar/batched compaction tests、invalid-input atomicity。

## Newer vLLM / KVPress

deferred-free/offload lifetime 与 periodic trigger，仅 V3 reference。

## We Implement

scoped v0.26 uniform reclaim transaction、whole-block zero-copy transition、safe canonical reuse、effective allocation/slot/attention、V2 staged Triton primitive、oracles/benchmark/profiling。

## Prohibited Claims

“实现 vLLM allocator”“发明 physical KV compression”“返回 CUDA HBM”“复现 Tangram全部能力”“Triton必然加速”。

## v1.2 Provenance Ledger

实现期间持续维护 `docs/90-reference/REFERENCE-PORTING-LEDGER.md`：记录 reference repo/commit/symbol、使用类型（API / semantic / test / kernel pattern / adapted code）和 project-owned target。最终面试/README 的 ownership 声明必须能回溯到该 ledger。
