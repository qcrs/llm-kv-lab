# 05 — V2 Triton Codec Placement / Crossover 详细设计

## 1. V2 Has Two Independent Questions

不能把“kernel 很快”和“E2E data movement更好”混成一个结论。

### Line A — Codec Kernel

控制输入为 CUDA BF16 KV representation：

```text
CUDA BF16
→ Triton INT8 encode
→ CUDA serialized payload/scales
→ Triton decode
→ CUDA BF16
```

只回答：correctness、encode/decode GB/s、launch/DRAM/occupancy。

### Line B — Actual LMCache Placement

真实 Serde path 记录：

```text
src MemoryObj device?
temp serialized object device?
raw H2D/D2H?
compressed H2D/D2H?
CPU threadpool delay?
filesystem I/O?
```

只有 Line B + NSYS能支持 PCIe claim。

## 2. Triton Reference Boundary

LMCache TurboQuant 已证明：reduction、quantization、low-bit packing、scale/zero、encode/decode、CPU/CUDA staging都能在 Serde体系中工作。我们的 INT8 是更简单 transform，但 serialized offsets、tiling、launcher仍属于自己实现。

推荐：

```text
grid = quant_groups × K/V × heads
program:
  load group
  fp32 absmax
  compute scale
  quantize int8
  write scale/payload
```

Decode反向：load scale + int8 → fp32 multiply → BF16 destination。

## 3. V2 Progression

```text
T0 PyTorch quant/dequant oracle
T1 Triton encode CUDA→CUDA
T2 Triton decode CUDA→CUDA
T3 shape/scale/payload byte oracle
T4 plug into same Serde contract
T5 filesystem L2 E2E
T6 NSYS actual movement map
T7 NCU kernel report
```

任何一步失败不允许跳到下一步用 E2E现象猜原因。

## 4. Crossover Model Must Match Real Boundary

只有当某个真实 transfer/storage boundary搬运 compressed bytes时，才可用：

```text
T_saved = (bytes_raw - bytes_comp) / BW_boundary
T_comp  = T_encode + T_decode
```

如果 generic Serde发生在 raw D2H之后，那么 V2 Core crossover主要是：

```text
codec cost vs L2 filesystem I/O saved bytes
```

不是 first PCIe D2H crossover。

## 5. Deterministic vs Conditional Claims

Deterministic：

- fixed serialized L2 bytes lower than BF16；
- format-size contract；
- Triton/PyTorch numeric agreement within defined quantization tolerance。

Conditional：

- L2 latency；
- cache-hit TTFT；
- Triton vs CPU/PyTorch speed；
- end-to-end throughput。

Forbidden without profiler：

```text
first D2H reduced
PCIe bytes reduced
Triton makes serving faster
```

## 6. Fallback

如果实际 source MemoryObj在 CPU，TurboQuant-style GPU codec可能产生：

```text
CPU raw → H2D raw → GPU encode → D2H compressed
```

这可能比 CPU V1更差。此时：

- V1 CPU/fixed Serde保持系统默认；
- Triton作为独立正确 GPU codec + profiling evidence；
- 是否前移 codec 到 pre-D2H由 V3-D gates决定。
