# P2 Filesystem Benchmark Regimes

本文件只解决一个问题：**让 codec/I/O crossover 的 filesystem 数字可解释、可复现。**

## 1. Why

FSL2 默认是 normal buffered filesystem I/O，Linux page cache 可能让重复 load 变成内存拷贝；同时 `use_odirect=True` 只有在 size/alignment满足要求时才真正走 O_DIRECT，否则实现会回退 buffered I/O。因此“load ms”必须绑定 I/O regime。

## 2. Required Regimes

### A. Footprint

- 只测 `.data` size / serialized bytes；
- 不做 disk bandwidth claim。

### B. Buffered Warm

- 明确预热相同 key；
- 目标是测 serde + warm page-cache path；
- 报告为 `buffered-warm`，不是 disk/NVMe throughput。

### C. Buffered Large-Working-Set / New-Key

- 使用大 working set 或每轮新 namespace/key；
- 目标是减少相同 page 的重复命中；
- 仍然是 OS-managed buffered I/O，不称为严格 cold disk。

### D. O_DIRECT

只有：

```text
use_odirect = true
object bytes aligned to filesystem/device block requirement
runtime confirms no fallback
```

时才纳入 O_DIRECT suite。否则记录 fallback 并从 O_DIRECT figure 排除。

## 3. Required Metadata

- filesystem / mount path；
- backing device / type；
- FSL2 config；
- object bytes / alignment；
- warm/new-key/working-set state；
- L1 state；
- OS/cache-control method（若使用）；
- repetitions/warmup；
- raw `.data` files and `bytes_transferred` evidence。

## 4. Interpretation

P2 的核心 deterministic evidence 是 byte footprint。I/O latency/crossover 是 backend/regime dependent。报告必须将：

```text
codec cost
filesystem/page-cache cost
GPU↔CPU movement
```

拆开，不用一个 crossover 数字代表全部 storage layers。

## 5. File Size Metric

Payload exactness 使用 `stat.st_size`。`du` / allocated blocks 只允许作为 filesystem physical-allocation secondary metric，并且不得与 Serde estimator 直接做 equality assertion。
