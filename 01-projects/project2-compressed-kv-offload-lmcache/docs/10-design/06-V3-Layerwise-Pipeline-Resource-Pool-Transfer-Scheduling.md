# 06 — V3 Layerwise Pipeline / Resource Reuse / Transfer Scheduling

V3 只在 V2 strong done + profiler evidence + user explicit GO 后进入。

## V3-A — Layerwise Load/Save Pipeline（最高优先级）

Connector已经有 layer hooks：

```text
start_load_kv
wait_for_layer_load(layer)
save_kv_layer(layer)
wait_for_save
```

因此优先利用 layer boundary，而不是自己造 arbitrary chunk scheduler。

Load目标：

```text
load layer i+1
while compute layer i
```

Save目标：

```text
codec/store layer i
while later layers compute
```

成功条件首先是 NSYS 显示 overlap / exposed critical-path time变化；不要求固定 speedup。

## V3-B — Temp/Pinned/Stream/Event Resource Reuse

进入条件：profile证明 allocation/reserve/release/object churn可见。newer vLLM offload worker提供 pinned host region、stream/event/buffer pool的 ownership/lifecycle reference。

不能绕过 LMCache L1Manager accounting建立隐形 raw-pointer pool。

## V3-C — Transfer Scheduling out of Token Critical Path

若 store/serialize阻塞 next-token path，尝试在安全 boundary defer submission。必须保留 source read-lock/lifetime fence：codec/store仍读 source时 backing storage不能 reuse/delete。

## Priority Rule

```text
profile evidence
→ one scoped V3
→ correctness
→ profiler again
```

不并行堆多个 optimization。
