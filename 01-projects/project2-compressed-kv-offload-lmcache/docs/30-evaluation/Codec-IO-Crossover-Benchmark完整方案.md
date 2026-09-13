# Codec / I/O Crossover Benchmark Plan

## Comparisons

Raw BF16、LMCache FP8、Our Block-INT8 PyTorch、Our Block-INT8 Triton、TurboQuant（环境允许且公平）。

## Three Separate Suites

1. Kernel micro：CUDA tensor→codec，encode/decode GB/s。
2. L2 object I/O：serialized bytes/file store/load，不含serving。
3. E2E serving：cold/cache-hit TTFT、load/resume、throughput。

不得用一个suite替代另一个。

## Variables

object/context length、layers/chunks/shape、batch/concurrency、backend/cache state、codec。先1K/2K/4K/...候选，实际范围由模型/Gate冻结。

## Metrics

codec ms/GBs/error；estimated/temp/file/reported bytes；FS store/load ms；D2H/H2D bytes/duration（profiler only）；cold/hit TTFT、E2E、throughput。

## Theory

~~~text
T_comp = T_encode + bytes_comp/BW + T_decode
T_raw = bytes_raw/BW
~~~

理论预测与实测分开。actual path若含 raw D2H或CPU→GPU staging，模型需加入这些项。

## Protocol

Git/env/model/config/warmup/repeats/command/raw/processed；clear/verify cache provenance；避免L1 hit掩盖FS；invalid run保留原因。

## Conclusion

报告 crossover point/interval、codec dominant vs IO dominant、negative regimes、quality。first-D2H claim只来自NSYS memcpy evidence。

## Filesystem I/O Regimes（v1.2 Mandatory）

Filesystem L2 的 latency/crossover 至少区分：

1. **Storage footprint only**：只比较 serialized/file bytes，不把 latency解释成 disk bandwidth；
2. **Buffered warm**：允许 Linux page cache warm，明确报告这是 memory/page-cache dominated path；
3. **Buffered large-working-set / new-key**：通过大于可用 page cache 的 working set 或新 namespace降低重复 cache hit，仍报告 buffered I/O；
4. **O_DIRECT**：仅当 object size / buffer alignment满足 FSL2 与 filesystem block-size要求时启用；若源码/runtime回退 buffered I/O，run 必须标记 invalid-for-ODIRECT claim。

每个 run 额外记录：

```text
filesystem type / mount
storage device
use_odirect
alignment / object bytes
working-set size
cache state (warm / new-key / controlled)
L1 cleared?
```

不允许用第二次读取同一文件的 page-cache hit，直接得出“NVMe/FS load bandwidth”结论。

## Capacity Claim Scope

FSL2 baseline 本身没有 fixed max-capacity admission/eviction signal。因此：

```text
serialized bytes ↓                 # measured runtime fact
objects under external N-byte budget ↑  # derived math / controlled experiment
```

第二条必须明确“external byte budget/quota model”，不能写成 LMCache FSL2 自动 admission capacity提高。
