# P1 Nsight Systems / NCU Profiling

## Tool Boundary

NSYS：CPU/GPU timeline、NVTX、kernel/memcpy、同步、critical path。

NCU：V2 gather/scatter 单 kernel 的 memory/occupancy/register。V1 zero-copy 没必要为“显得底层”强跑 NCU。

## NVTX Ranges

- P1_RECLAIM_DECISION
- P1_WORKER_COMMIT
- P1_BLOCKTABLE_REPLACE
- P1_MODEL_LOGICAL_POSITION
- P1_CACHE_PHYSICAL_POSITION
- P1_SCHEDULER_OWNERSHIP_COMMIT
- P1_BLOCKPOOL_PUBLICATION
- P1_TRITON_GATHER / SCATTER
- P1_ATTENTION

Range 必须低开销、feature-gated、名称稳定。

## NSYS Questions

- reclaim 位于 last-prefill 之后吗？
- worker commit 与 scheduler free 的时间顺序？
- block table/metadata H2D cost？
- compaction 是否在 critical path？
- V2 scratch traffic/kernel launch？
- shorter attention duration是否在相同 workload出现？
- hidden sync / cudaDeviceSynchronize？
- multi-process capture是否覆盖正确 process？

## NCU Metrics

DRAM throughput、bytes、load/store efficiency、occupancy、register pressure、waves/SM、launch count。只在 correctness和稳定shape后采。

## Protocol

先一次未 profile baseline；再 NSYS coarse capture；用 NVTX/SQLite 定位；只对目标 kernel NCU。避免同时跑 NSYS+NCU。保存 exact command、report、export/summary 与版本。

## Conclusion Rule

Profiler observation 先写 OBSERVED，再写 mechanical interpretation，最后由 Web review形成结论。micro kernel speedup不能替代 E2E。

