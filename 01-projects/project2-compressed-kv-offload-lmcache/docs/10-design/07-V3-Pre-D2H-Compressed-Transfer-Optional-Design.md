# 07 — V3 Pre-D2H Compressed Transfer（Optional）

目标：

```text
GPU paged KV
→ GPU codec
→ compressed D2H
→ host/L2
```

reverse：

```text
compressed host
→ H2D compressed
→ GPU decode
→ vLLM destination slots
```

这是 P2 最强但 integration cost最高的 V3，不进入 V1/V2 completion gate。

## Reference Status

newer vLLM offload subsystem 已提供 analogous primitives：

- canonical GPU page byte views；
- block transfer specs；
- pinned host registration；
- stream/event pools；
- compute→transfer fence；
- completion/lifetime accounting。

因此机制不是完全未知；但把这些 semantic port到 pinned v0.26 + LMCache仍是 substantial B/C extension。

## Five Mandatory Gates

1. **Raw transfer owner located**：source + NSYS确认 first GPU→host copy函数/bytes；
2. **Source page lifetime fenced**：model last write → codec read → D2H read → safe reuse；
3. **Compressed buffer owner explicit**：GPU temp/host temp/L1/L2谁分配谁释放；
4. **Reverse load path explicit**：compressed bytes能在对应 attention layer前恢复到 allocated vLLM physical slots；
5. **Raw fallback works**：unsupported layout/codec failure回 existing connector path。

任何 gate 不满足：不做 V3-D。

## Implementation Guidance

- Triton负责 codec，不默认替代 bulk DMA；
- copy engine负责 bulk GPU↔CPU transfer，除非 profiler给出反证；
- small reusable stream/event pool优于 per-layer无限创建；
- 只有 NSYS显示 actual D2H/H2D memcpy bytes变小后，README/简历才允许写 PCIe reduction。
