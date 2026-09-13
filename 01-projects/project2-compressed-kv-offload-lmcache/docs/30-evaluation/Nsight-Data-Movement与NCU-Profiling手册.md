# Nsight Data Movement and NCU Profiling

## NVTX

P2_KV_EXTRACT、P2_RAW_D2H、P2_SERIALIZE、P2_TRITON_ENCODE、P2_L2_STORE、P2_L2_LOAD、P2_TRITON_DECODE、P2_DESERIALIZE、P2_H2D、P2_RESUME。

## NSYS Questions

- raw GPU→CPU owner/function/size？
- serializer src and temp device？
- CPU source是否raw H2D staging？
- compressed bytes何时成为source of truth？
- L1→L2/file bytes？
- load reverse copies？
- codec/IO/compute overlap？
- CPU threadpool/eventfd delay？
- hit→first decode critical path？

保存timeline与memcpy sizes；仅dtype/file size不能推断PCIe。

## NCU

只看encode/decode：DRAM BW、load/store、occupancy、register、reduction、launch。正确性/稳定shape先行。

## Capture Order

baseline normal run → NSYS coarse + NVTX → SQLite/summary → targeted NCU。NSYS/NCU不同时重负载。multi-process选对进程。

## Claim

OBSERVED memcpy bytes/duration才支持PCIe；kernel速度不支持E2E；negative staging evidence必须保留。

