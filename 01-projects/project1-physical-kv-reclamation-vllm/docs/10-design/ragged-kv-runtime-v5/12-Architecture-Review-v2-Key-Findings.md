> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Architecture Review v2 — Key Findings

本文件记录在上一版计划完成后，对 Tangram 当前 main 与 qcrs/vLLM P1-V2 branch 再做一轮 source-level cross-audit 得到的增量结论。它只列会改变实现顺序或数据结构的发现。

## 1. Global pool 不是“多一点 block”，而是 allocator semantic change

**[FACT:v0.26]** `core/kv_cache_utils.py::get_num_blocks` 普通路径用 `available_memory // page_size // num_layers`；`get_kv_cache_config_from_groups` 再根据 KV groups/layer sharing 创建 tensors。  
**[FACT:Tangram]** Ragged 路径改为 `available_memory // page_size`，并将所有 attention layers 放入一个 `KVCacheTensor.shared_by`，由 globally unique page IDs 区分 `(layer,group)`。  
**[DECISION]** 把 Global Pool 从 R1-B allocator 附属问题提前为 R1-A2 独立 Slice。

错误实现的典型症状：identity Ragged 看似工作，但 capacity 会多/少一个 layer factor；或者两个 layers 得到同 page ID namespace 导致 silent overwrite。

## 2. v0.26 两阶段 Attention seam 改变了 port 方法

**[FACT:v0.26]** 当前 FlashAttention backend 的 KV update 不在 forward 内；`unified_kv_cache_update` 与 `unified_attention_with_output` 分离，并用 dummy dependency 维护副作用顺序。  
**[FACT:Tangram]** 其旧基线上 ragged special forward 内部负责 member reshape、cache update、FA execution。  
**[DECISION]** 在 MRv2 保留两阶段 contract，新增 R1-D1 专门解决 ragged write，read 继续在 R1-D2/D3。

这比新增一套 `unified_attention_ragged` 更符合 v0.26 当前 architecture，也更利于 piecewise CUDA Graph。

## 3. Physical layout 需要 v0.26-native adaptation

**[FACT:v0.26]** FA cache semantic shape 是 `[B,H,N,2D]`，通过 stride order 支持 NHD/HND；forward 再把 `2D` split 为 K/V。  
**[FACT:Tangram]** 其 virtual-block trick依赖 page 内 column 在 block_size 之前，使 `(physical_page,column)` 连续并可 flatten。  
**[DECISION]** Ragged 首版专属 storage 采用 `[B,Hp,N,2D]` contiguous，virtual view `[B*Hp,1,N,2D]`。不要求 dense path 改全局 layout。

## 4. Flattened virtual rows 与 MRv2 staged buffers 天然匹配

**[FACT:v0.26]** `StagedWriteTensor` 以 first dimension 为 row，stage-write kernel按 row/start 写 persistent GPU buffer。  
**[DECISION]** persistent ragged block rows 使用 `[R*G_total,Bmax]`，API 层 expose `[R,G_total,Bmax]`。这样可复用 MRv2 pointer stability、UVA descriptors、Triton gather，而不是退回 Tangram 旧 CPU-table commit 方式。

## 5. Spec registry 是干净隔离 dense/ragged 的入口

**[FACT:v0.26]** `KVCacheSpecRegistry` 已支持 spec→manager 绑定及 uniform-type compatibility。  
**[DECISION]** Ragged 用专用 spec/manager 注册，而不是在 `SingleTypeKVCacheManager` 大量 `if ragged`。这与 `GPUModelRunner` 文件头要求“feature-specific complexity stay outside central runner”一致。

## 6. R1 必须硬 gate static-head side paths

普通 `FlashAttentionImpl.forward` 有 quant descales、sink、ALiBI、DCP、cascade 等路径，它们部分按原 model `self.num_kv_heads` 构造 metadata。member-major attention 把每个 KV head变成一条 sequence，不能未经审计就复用全部分支。

**首版 hard gate：** no quant KV、no sink、no alibi/cascade、no DCP/PCP、full causal attention、`head_size_v==head_size`。

## 7. AOT clustering 分两级更安全

Tangram map 可以 cross-layer；这要求 arbitrary member→cluster mapping，cluster physical ownership甚至不再 layer-local。  
**[DECISION]** 标准 R7 先做 **per-layer AOT**；cross-layer/global clustering 作为 R7-B stretch。这样仍能展示“offline profile→placement optimization”，但不把 Core runtime复杂度重新打开。

## 8. Core completion 分成 functional 与 flagship

上一版 R4 只写 one policy，容易出现“automatic policy 仍 uniform，但简历写 non-uniform”的表述风险。

**[DECISION]**

```text
R4-A: one scorer + UniformScope = policy wiring closure
R4-B: same scorer + LayerScope = automatic non-uniform closure
```

R3 的 manual unequal depths 已足够证明 runtime non-uniform capability；R4-B 则让最终 flagship claim更完整。

## 9. Preemption 必须是状态机，不是一个 restore function

定义三态：

```text
ACTIVE_IN_BATCH
CACHED_OUT_OF_BATCH   # still owns KV
PREEMPTED_RECOMPUTE   # KV ownership released
```

第二态需要 exact row/effective snapshot；第三态绝不能 restore physical snapshot。

## 10. Benchmark 必须显式记 Ragged tax

Tangram 自己的 speedup script 注释指出某些 speedup baseline 是 uncompressed Tangram/ragged，而非 vanilla vLLM。我们的最终报告必须同时放：

```text
Dense v0.26
Ragged Identity
Ragged + Compression
Ragged + each optimization
```

否则 compression gain 可能掩盖 representation overhead，无法回答面试中的系统 trade-off 问题。
