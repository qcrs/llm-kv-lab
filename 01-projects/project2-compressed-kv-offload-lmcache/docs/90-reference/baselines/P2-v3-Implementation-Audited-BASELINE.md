# Project 2 — Compressed KV Offload / Reuse for vLLM + LMCache
## Custom Block-INT8 L2 Serde + Triton Codec + Data-Movement Crossover Analysis

**定位：第二项目 / KV Cache Reuse / LMCache / vLLM KV Connector / GPU-CPU Memory Hierarchy / Triton / PCIe Profiling**

**建议基线**
- vLLM: `v0.26.0`
- vLLM pinned commit: `568afb3a13806beb53bb2e6bd518269357b237c0`
- **LMCache implementation baseline: `a7afadebb9248b62b5c533ce2c12297e9d94fc4a`**
- LMCache later audit snapshot: `f9addd2e4e074f21f26cbda84c3abd932d32ef33`（只用于后续 source comparison，不直接作为 torch2.11 环境安装基线）
- GPU: NVIDIA A100 80GB
- 单机单卡优先
- V1 固定使用 local filesystem L2 + Serde wrapper，避免 Redis/Mooncake/NIXL/multi-node 依赖
- 第一版不扩展到 RDMA / Mooncake / multi-node / P-D disaggregation
- **ABI gate**：vLLM v0.26 固定 torch 2.11；LMCache implementation baseline 也必须保持 torch 2.11 native-extension compatibility

---

# 0. 一句话定义

这个项目解决：

> **当长上下文 KV 被外部缓存复用时，raw BF16 KV 的外部存储与数据搬运成本可能很大；我们先在 vLLM + LMCache 的真实 L2 Serde 路径中加入透明、可验证的 Block-INT8 codec，再用 profiler 明确各层 data movement，并分析“codec cost vs saved L2 bytes”的系统 crossover。**

不是重新做 LMCache。

也不是声称发明 KV quantization。

核心是：

```text
vLLM paged KV
        │
        ▼
LMCache KV Connector
        │
        ▼
KV object
        │
        ▼
Custom Block-INT8 Codec
        │
        ▼
compressed representation
        │
        ▼
serialized L2 KV storage
        │
        ▼
load
        │
        ▼
decode
        │
        ▼
restore vLLM KV
        │
        ▼
resume generation
```

---

# 0.5 Pre-Implementation Certainty Audit：先验确定性审计

这部分专门检查：

> **P2 会不会在开始做以后才发现 custom codec 接不到真实 path、版本不兼容、或者为了“减少 PCIe”不得不自己发明一个没有参考的新 connector architecture？**

审计结果比上一版更严格：

> **P2 V1（真实 vLLM+LMCache reuse + L2 Serde compression）确定性很高；V2 Triton codec 也有成熟 reference。**
>
> **但“generic Serde 一定减少 GPU↔CPU PCIe bytes”不成立。真正 pre-D2H compression 目前没有在我们 pinned vLLM0.26 + LMCache path 中确认到 exact drop-in implementation，因此它被移出 Core，变成 optional V3/experimental path。**

这项修正很重要。

---

## 0.5.1 Reference 等级

| 等级 | 含义 | Core 是否可依赖 |
|---|---|---|
| **A — Exact Reference** | 同类接口/路径已有公开 implementation | 可以 |
| **B — Reference-backed Adaptation** | general mechanism 已存在，我们实现更简单 variant | 可以 |
| **C — Mechanically Specified Own Implementation** | format/kernel 自己写，但 contract/oracle 已知 | 可以 |
| **D — Experimental / Benefit Unknown** | 需要新 integration 或 profiler 才能证明 | 不能作为成功条件 |

---

## 0.5.2 先发现并修掉一个真实环境风险：LMCache / torch ABI

上一版使用：

```text
LMCache commit:
f9addd2e...
```

但它的 `pyproject.toml` 在该 snapshot 中已经 pin：

```text
torch==2.13.0
```

而我们的 vLLM v0.26 pinned commit 明确是：

```text
torch==2.11.0
```

这不是小问题。

LMCache 自己在 commit：

```text
a7afadebb9248b62b5c533ce2c12297e9d94fc4a
```

`[CI] fix abi mismatch and smoke checks (#4423)`

明确记录了：

- stable vLLM 0.26.0 使用 torch 2.11；
- LMCache 曾错误地跟 vLLM main 升到 torch 2.13；
- 已发布构建出现 native extension `undefined symbol`；
- 更危险的是，某些 fallback 会让 serving 看似能跑，但 native CUDA kernels 实际被静默禁用；
- 该修复重新把 stable build pin 回 torch 2.11。

因此 P2 implementation baseline 现在正式改为：

```text
LMCache:
a7afadebb9248b62b5c533ce2c12297e9d94fc4a
```

而不是 `f9addd...`。

更重要的是，在这个兼容 commit 上我们需要的能力已经存在：

- FP8 Serde；
- `SerdeL2AdapterWrapper`；
- TurboQuant；
- Triton store/decode kernels；
- filesystem L2；
- vLLM reuse example。

所以不用为了 ABI 兼容牺牲项目 reference。

### Environment Gate

进入任何功能开发前先验证：

```text
torch == 2.11.x
vLLM == pinned v0.26 commit
LMCache == a7af...
native extension imports successfully
CUDA/Triton backend actually enabled
```

不能只看：

```text
import lmcache
```

成功就算环境没问题。

这一步专门防止“开始项目后才发现环境基座不兼容”。

---

## 0.5.3 P2 必要链路逐项审计

| 必要能力 | 等级 | Reference | 是否需要新设计 |
|---|---|---|---|
| vLLM → LMCache Connector | A | vLLM `LMCacheConnectorV1` | 否 |
| shared-prefix KV reuse | A | LMCache local backend example | 否 |
| request/block/slot mapping | A | LMCache `vllm_v1_adapter.py` | 否 |
| custom serializer/deserializer interface | A | LMCache Serde ABC | 否 |
| async task/event lifecycle | A | `AsyncSerdeProcessor` | 否 |
| Serde 接 L2 store/load | A | `SerdeL2AdapterWrapper` | 否 |
| deterministic local L2 backend | A | `fs_l2_adapter.py` | 否 |
| simple codec template | A | FP8 serde | 否 |
| GPU Triton low-bit codec | A/B | TurboQuant | 否 |
| 我们的 Block-INT8 format | C | 标准 symmetric quant + LMCache format contract | 格式由我们定义，但不是研究问题 |
| 我们的 Triton INT8 kernel | C | TurboQuant + PyTorch oracle | 自己实现 kernel |
| L2 serialized bytes reduction | **确定** | byte layout / FS file size | 否 |
| L1 CPU raw cache footprint reduction | **不由 Serde 自动提供** | StoreController/L1 design | 若要做需额外 policy |
| GPU↔CPU raw D2H reduction | **D** | generic Serde 不保证 | 需要更靠前的 codec hook |
| TTFT/throughput improvement | D | benchmark | 不保证 |

---

## 0.5.4 第二个重要修正：Serde 是 L1→L2 层，不等于 L1 本身被压缩

LMCache `SerdeL2AdapterWrapper` 的真实 store path：

```text
raw L1 MemoryObj
        │
        ▼
temporary serialized buffer
        │
        ▼
Serde
        │
        ▼
inner L2 adapter
```

Load：

```text
L2 serialized bytes
        │
        ▼
temporary serialized buffer
        │
        ▼
deserialize
        │
        ▼
raw L1 MemoryObj
```

因此 custom Serde 最确定的收益是：

```text
L2 serialized footprint ↓
```

不是自动：

```text
L1 raw CPU footprint ↓
```

LMCache 的 StoreController 文档还明确说明，默认 StorePolicy：

```text
successful L2 store
→ does not delete L1 by default
```

所以“同样的 CPU/L1 cache 能装更多对象”不能作为 V1 默认 claim。

如果以后配置/实现：

```text
L2 store success
→ delete raw L1 copy
```

那是另一个 storage policy 问题。

---

## 0.5.5 第三个重要修正：generic Serde 不保证减少第一段 D2H

可能的真实链：

```text
vLLM GPU paged KV
↓
raw BF16 D2H
↓
LMCache raw L1 object
↓
Serde compression
↓
L2
```

这里：

```text
L2 bytes ↓
```

但第一段：

```text
GPU → L1
```

仍然搬 raw BF16。

TurboQuant 的 serializer 甚至明确支持：

```text
CPU source
↓
copy raw source to CUDA
↓
Triton codec
↓
copy compressed result back CPU
```

所以如果 source 已经是 CPU：

> GPU codec 反而可能额外产生 PCIe traffic。

因此 V2 不再预设：

```text
Triton Serde
=
reduced PCIe
```

---

## 0.5.6 P2 Core 的 deterministic physical backend：filesystem L2

为了让 V1 完全不依赖外部服务，固定使用：

```text
LMCache filesystem L2 adapter
```

Reference：

```text
lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py
```

它把每个 ObjectKey 存成独立：

```text
.data
```

文件。

因此 V1 的最硬 proof 可以直接是：

```text
same KV object

raw L2 file bytes
vs
Block-INT8 L2 file bytes
```

这比“看日志说压缩了”更可靠。

V1 不需要：

- Redis；
- Mooncake；
- NIXL；
- RDMA；
- second machine。

---

## 0.5.7 P2 V0 baseline 也有 exact example

LMCache 自己有：

```text
examples/kv_cache_reuse/local_backends/offload.py
```

它做：

```text
LLM(
  kv_transfer_config=LMCacheConnectorV1
)
```

并用两个共享长 prefix 的请求验证：

```text
request A store
request B reuse
```

所以：

```text
vLLM → LMCache → shared-prefix reuse
```

不是我们自己需要设计的。

注意：

> 这个 local-backend example 用来验证 connector/reuse；custom Serde 的确定性实验仍然固定走 distributed L2 + `SerdeL2AdapterWrapper`，不要把两条 backend path 混为一谈。

---

## 0.5.8 我们真正自己定义的 Block-INT8 是否会变成“未知方案”

不会。

它是 C 级 own implementation：

```text
input:
known BF16 MemoryObj layout

quantization:
symmetric INT8

output:
fixed byte layout + scale

size:
deterministic

oracle:
PyTorch quant/dequant
```

需要我们决定的只是：

- scale granularity；
- scale dtype；
- byte layout；
- padding/alignment。

这些是工程 format 选择，不是“有没有算法能工作”的研究问题。

而且有两套 reference：

```text
FP8 serde
→ 最小 Serde plugin structure

TurboQuant
→ complex quantized byte layout + Triton codec structure
```

---

## 0.5.9 Triton V2 的风险边界

我们的 INT8 Triton kernel没有 exact drop-in copy。

但 TurboQuant 已经完成更复杂的：

- reduction；
- quantization；
- scale/zero；
- 3/4-bit packing；
- Triton store/decode；
- CUDA staging。

所以 INT8 是其更简单子问题。

等级：

> C

项目要求：

```text
correct Triton encode/decode
+
PyTorch oracle
+
kernel benchmark/profile
```

不要求：

```text
must beat TurboQuant
must reduce TTFT
must reduce PCIe
```

---

## 0.5.10 Pre-D2H compression：明确移到 optional V3

如果后续 source audit / NSYS 证明：

```text
save_kv_layer
↓
存在干净的 GPU-side codec insertion point
↓
可以在 raw D2H 之前得到 compressed buffer
```

再做：

```text
GPU paged KV
→ GPU encode
→ compressed D2H
```

这是很好的 V3。

但截至本轮审计：

> **我们有 connector hook、CacheGen system idea、TurboQuant CUDA codec reference，但没有确认到 pinned vLLM0.26 + LMCache 中“compressed-before-first-D2H”的 exact drop-in path。**

所以不能把它写成 V1/V2 必做。

这正是本次审计要提前暴露、而不是 coding 到一半才发现的问题。

---

## 0.5.11 P2 deterministic gains

### Core 可保证

```text
serialized L2 bytes ↓
filesystem L2 file size ↓
L2 capacity / fixed byte budget ↑
L1→L2 serialized transfer bytes ↓
```

前提是 inner L2 adapter 接收的确实是 Serde output，这由 wrapper contract保证。

### 需要实验

```text
L2 store/load latency
cache-hit TTFT
throughput
Triton codec speed
```

### Core 不保证

```text
L1 raw CPU capacity ↑
GPU→CPU PCIe bytes ↓
overall HBM ↓
end-to-end speedup
beat FP8/TurboQuant
```

---

## 0.5.12 最终先验结论

### V0/V1

所有必要 system path 都有 exact reference。

> **Finishability：很高。**

### V2 Triton codec

format 是我们定义，kernel 自己写，但有 TurboQuant 与 PyTorch oracle。

> **Correctness finishability：高；speedup 不保证。**

### V3 pre-D2H / overlap

存在真实 integration gap，需要 profiler/source audit 后决定。

> **不进入项目成功条件。**

---


# 0.6 Round-3 GitHub Implementation Audit：把 V1/V2 从“有接口”细化到“真实对象怎么流动”

P2 这一轮重点不是再确认：

```text
LMCache 有 Serde
```

而是继续追：

```text
谁持有 raw KV？
谁构造 slot_mapping？
Serde 的 input/output MemoryObj 是什么？
temp buffer 谁分配？
真正写进 filesystem 的 bytes 到底是什么？
异步 completion 谁负责？
V2 Triton 放在哪一层才不会误判 PCIe benefit？
V3 有没有成熟 offload runtime 可以借？
```

本轮额外检查：

- LMCache compatibility baseline `a7afade...` 的实际 Python implementation；
- LMCache Serde 文档 **以及实际 FP8 code**；
- `SerdeL2AdapterWrapper` implementation；
- filesystem L2 actual write/read；
- 更靠后的 vLLM generic CPU offload runtime；
- CacheGen；
- vLLM/LMCache layerwise connector hooks。

## 0.6.1 Round-3 总结

P2 的确定性也进一步提高：

### V1

现在可以明确说：

> **Custom fixed-size Block-INT8 L2 Serde 不需要设计新的 async/storage architecture。**

我们只负责：

```text
raw MemoryObj
→ deterministic INT8 byte representation

serialized bytes
→ raw destination MemoryObj
```

其他：

- temp allocation；
- async thread pool；
- eventfd；
- serialize→L2 chaining；
- L2→deserialize chaining；
- failure lifecycle；
- temp-buffer cleanup；
- filesystem store/load；

都有现成 implementation。

### V2

Triton 也不是“给 LMCache 硬加一个 GPU kernel”。

已有 TurboQuant 证明：

```text
LMCache Serde
+
GPU Triton codec
+
CPU/CUDA staging
```

是合法结构。

我们做的 INT8 是更简单的 codec。

### V3

新的最大收获是：

> **较新的 vLLM upstream 已经有成熟 generic CPU offload worker，可作为 pinned buffer / stream / event / transfer fence / canonical GPU page mapping 的工程 reference。**

因此 V3 不再需要凭想象设计 async data movement runtime。

---

# 0.7 V1 End-to-End Object Flow：把真实路径拆成两个 baseline 再闭环

P2 V1 最稳的推进方式仍然是：

```text
V0-A
先证明 vLLM ↔ LMCache reuse

V0-B
再证明 distributed filesystem L2 raw path

V1
最后在 L2 path 中插入 Block-INT8 Serde
```

不要一次把：

```text
connector + L2 + serde + codec
```

四层一起 debug。

---

## 0.7.1 V0-A：vLLM Connector / Request Tracking / Slot Mapping

在兼容 baseline 上：

```text
vLLM
↓
LMCacheConnectorV1
↓
LMCacheConnectorV1Impl
```

worker-side 已有：

```text
register_kv_caches
start_load_kv
wait_for_layer_load
save_kv_layer
wait_for_save
```

scheduler-side 已有：

```text
get_num_new_matched_tokens
update_state_after_alloc
build_connector_meta
request_finished
```

因此 vLLM↔LMCache 生命周期不用我们设计。

### RequestTracker 已经解决 block→slot mapping

LMCache `vllm_v1_adapter.py` 保存：

```text
prompt_len
token_ids
allocated_block_ids
num_saved_tokens
decode phase
lmcache cached tokens
```

`ReqMeta.from_request_tracker()` 根据：

```text
allocated_block_ids
+
block_size
```

生成：

```text
slot_mapping
```

核心逻辑是：

```text
physical slot
=
physical block id × block_size
+
offset in block
```

因此 P2 不需要自己解释：

```text
一个 LMCache chunk 到底如何对应 vLLM physical KV slot
```

这个 bridge 已经存在。

---

## 0.7.2 V0-B：Raw distributed filesystem L2

第二个 baseline 要单独证明：

```text
raw L1 MemoryObj
↓
StoreController
↓
FSL2Adapter
↓
real .data file
```

然后：

```text
file
↓
FSL2Adapter load
↓
raw MemoryObj
↓
vLLM restore/resume
```

这一阶段：

```text
serde = disabled
```

目的不是性能。

是确定：

- distributed L2 config 正确；
- keys/chunks 对得上；
- FS write/load 正常；
- L2 hit 可以恢复；
- byte accounting 可观测。

---

# 0.8 V1 Serde 的真实 runtime contract

当开启：

```text
fs adapter
+
serde = block_int8
```

真实 store path 应理解成：

```text
StoreController
holds read-locked raw L1 object
        ↓
SerdeL2AdapterWrapper.submit_store_task
        ↓
wrapper asks L1Manager:
reserve temporary serialized MemoryObj
        ↓
AsyncSerdeProcessor.submit_serialize
        ↓
BlockInt8Serializer
raw object → serialized temp
        ↓
serialize completion event
        ↓
wrapper changes temp lifecycle:
write-complete → read source for inner L2
        ↓
FSL2Adapter.submit_store_task
        ↓
filesystem write
        ↓
inner completion
        ↓
temp released
        ↓
L2StoreResult
```

Load：

```text
caller supplies raw destination MemoryObj
        ↓
wrapper reserves serialized temp object
        ↓
FSL2Adapter loads .data bytes into temp
        ↓
load completion bitmap
        ↓
AsyncSerdeProcessor.submit_deserialize
        ↓
BlockInt8Deserializer
serialized temp → raw destination
        ↓
wrapper releases temp
        ↓
load completion
```

这说明我们的 codec 根本不需要控制：

```text
thread
eventfd
poll
L1 lock
filesystem coroutine
temp cleanup
```

这正是项目实现确定性高的原因。

---

# 0.9 一个新的重要 audit finding：LMCache design doc 与同 commit code 有细节漂移

本轮同时检查：

```text
docs/design/v1/distributed/serde/README.md
```

和：

```text
lmcache/v1/distributed/serde/fp8.py
```

在同一 compatibility commit 上发现：

文档还描述 FP8 serialized-size estimator 有额外 headroom；

但实际 `fp8.py` 已经改成：

```text
exact 1 byte / element
```

而代码注释明确说明：

> 如果 estimator 膨胀，wrapper 分配的 serialized MemoryObj 也会膨胀，inner L2 会把整个对象持久化，直接侵蚀压缩收益。

### 对我们项目的规则

以后 P2 不允许：

```text
只读 design doc
→ 推断 runtime behavior
```

必须：

```text
design doc
+
same-commit implementation
+
E2E measurement
```

三者对齐。

### V1 format 因此采用 exact fixed-size layout

Block-INT8 V1 不做：

- entropy coding；
- data-dependent variable-length payload；
- pickle；
- variable header；
- complex alignment policy。

目标：

```text
estimate_serialized_size
==
physical serialized MemoryObj size
==
filesystem file size
```

尽量做到三者完全一致。

---

# 0.10 为什么 filesystem L2 是极好的 deterministic oracle

实际 `FSL2Adapter._execute_store()` 做：

```text
buf = MemoryObj.byte_array
size = len(buf)
write exactly buf
bytes_written += size
```

然后：

```text
L2StoreResult(success, bytes_written)
```

Load 同样：

```text
expected = len(destination byte_array)
read exactly expected bytes
```

因此 V1 可以同时得到三种互相验证的 physical evidence：

```text
1. serializer returned actual bytes
2. L2StoreResult.bytes_transferred
3. os file size
```

### 推荐必须建立 invariant

对于我们的 fixed Block-INT8：

```text
estimated_size
==
temp MemoryObj bytes
==
FS file bytes
==
reported stored bytes
```

如果不相等：

```text
先查 size contract
```

不要进入 E2E performance benchmark。

这会让 P2 比普通“量化 tensor 然后测误差”的项目扎实很多。

---

# 0.11 Block-INT8 V1 Format：进一步把设计收紧成 engineering format

不追求 codec novelty。

推荐只实现一个 format。

## 0.11.1 Quantization granularity

最优先选择：

```text
per physical KV block
×
per K/V
×
per KV head
```

具体如何从 LMCache `MemoryObj` shape 找到 K/V/head 维度，要在 Environment/Shape Gate 中先打印并固定。

如果 current MemoryObj 把 K/V 合并成一个 tensor：

```text
先固定 layout contract
```

再决定：

```text
K 和 V 各自 scale
```

还是：

```text
combined scale
```

推荐 K/V 分开 scale，因为：

- 仍然简单；
- error behavior 更可解释；
- scale metadata 成本仍很小。

### 不在 V1 做

```text
per-channel K + per-token V
mixed bit width
learned scale
outlier channel
```

这些会把第二项目又拖回 quantization research。

---

## 0.11.2 Serialized region

建议抽象为：

```text
small fixed metadata
+
scale array
+
INT8 payload
```

所有 offset 都由：

```text
MemoryLayoutDesc
+
known codec config
```

推导。

不依赖 KV 数值。

因此：

```text
same shape/dtype
→ same serialized bytes
```

这非常契合 LMCache wrapper “按 batch first layout 预分配 temp”的 contract。

---

## 0.11.3 Scale dtype

V1 推荐：

```text
FP16 scale
```

而不是 BF16/FP32。

理由不是性能玄学，而是：

- metadata 更小；
- decode 简单；
- A100/Torch/Triton 支持稳定；
- PyTorch oracle 容易一致。

如果数值测试发现 FP16 scale error 不可接受：

```text
切 FP32
```

是 configuration change，不是 architecture change。

---

# 0.12 V1 implementation ownership：真正属于我们的代码只有哪些

建议项目 ownership 明确写成：

```text
1. BlockInt8 format specification
2. Serializer
3. Deserializer
4. exact size estimator
5. serde factory/registration
6. tensor-level oracle/tests
7. LMCache distributed L2 config
8. E2E reuse verification
9. benchmark/profiling instrumentation
```

不把这些 upstream 能力包装成自己的：

```text
async event loop
L1 locking
filesystem adapter
vLLM connector
slot mapping framework
```

反而更可信。

---

# 0.13 V1 Failure Tree：开始 coding 前就定义出现问题往哪里查

## Case A — Connector baseline fail

症状：

```text
shared-prefix second request 没 hit
```

不碰 codec。

查：

```text
request tracker
chunk boundary
connector config
LMCache key lookup
```

---

## Case B — Raw FS L2 fail

症状：

```text
connector works
local path works
distributed FS path load/store fail
```

仍然不碰 codec。

查：

```text
StoreController
FSL2Adapter
ObjectKey
L1/L2 config
```

---

## Case C — Serde tensor unit test fail

查：

```text
layout interpretation
scale granularity
quant math
payload offsets
size estimator
```

---

## Case D — Serde unit correct，FS file size wrong

优先查：

```text
estimated serialized layout
temp MemoryObj allocation
padding/alignment
inner adapter writes full object
```

---

## Case E — FS compressed load succeeds，但 generation corrupt

查：

```text
deserialize output layout
slot mapping
dtype/shape
lossy error magnitude
```

而不是立即怀疑：

```text
LMCache async runtime
```

这就是分阶段 baseline 的价值。

---

# 0.14 V2 Triton：分成“codec kernel”与“真实 placement”两件事

V2 不能只写：

```text
把 PyTorch quant 换成 Triton
```

应该明确两条实验线。

## Line A — Kernel correctness/performance

控制输入：

```text
CUDA BF16 KV tensor
```

输出：

```text
CUDA serialized INT8/scales
```

只测：

```text
quant/dequant correctness
encode GB/s
decode GB/s
NCU
```

这里可以公平看 Triton kernel 本身。

---

## Line B — Actual LMCache Serde path

真实：

```text
src MemoryObj device = ?
dst temp device = ?
```

由 runtime 决定。

测：

- serializer source device；
- temp device；
- 是否发生 raw H2D；
- 是否发生 compressed D2H；
- codec stream；
- CPU threadpool delay；
- FS I/O。

### 为什么必须分开

如果真实 src 是 CPU：

```text
CPU raw
→ H2D raw
→ Triton encode
→ D2H compressed
```

那么：

```text
Triton kernel 很快
```

也不代表：

```text
E2E 更快
```

更不代表：

```text
PCIe ↓
```

两条实验线分开后，negative result 很容易解释。

---

# 0.15 V2 Triton implementation reference 到底覆盖到哪

## Exact/strong reference

LMCache TurboQuant 已覆盖：

- GPU reduction；
- quantization；
- low-bit packing；
- scale/zero；
- encode/decode；
- CPU/CUDA staging。

Pinned vLLM 也有 per-token/per-head INT8 paged quantization primitive：

```text
grid=(token, head)
absmax
scale
quantize
store
```

因此我们的：

```text
per block/head INT8
```

属于更简单的 GPU transform。

## 我们自己的部分

仍然包括：

```text
serialized layout offsets
kernel tiling
scale region write
payload region write
launcher
```

这是 C 级 own implementation。

但：

```text
input/output
numerics
size
```

都可由 PyTorch oracle 完全定义。

没有 research blocker。

---

# 0.16 V2 推荐 implementation progression

不要直接上完整 E2E。

```text
T0
PyTorch quant/dequant oracle

T1
Triton encode
CUDA tensor → CUDA serialized tensor

T2
Triton decode
CUDA serialized → CUDA BF16

T3
bit/scale/shape oracle

T4
Serde wrapper integration
without changing L2 backend

T5
filesystem L2 E2E

T6
NSYS actual device movement

T7
NCU kernel analysis
```

### Required fallback

如果 Triton codec 真实 Serde path 因 CPU staging 不划算：

```text
PyTorch/CPU V1 stays default
Triton remains GPU codec benchmark/reference path
```

项目仍完成。

不要为了“必须用 Triton”破坏系统设计。

---

# 0.17 更靠后的 vLLM upstream 给 P2 V3 的新 reference

本轮额外检查：

```text
vllm/v1/kv_offload/
vllm/distributed/kv_transfer/kv_connector/v1/offloading/
```

audit snapshot：

```text
0ecc284790e5403f74b899524ef82ecb69f83cb3
```

它不是 pinned v0.26 implementation baseline。

但它非常有价值，因为 upstream 已经把 generic KV CPU offload 做成：

```text
canonical GPU KV page views
        ↓
GPU/CPU load-store specs
        ↓
async transfer worker
        ↓
pinned host storage
        ↓
stream/event lifetime
        ↓
transfer completion/fencing
```

这直接回答了 P2 V3 很多原本可能需要自己设计的问题。

---

## 0.17.1 Canonical GPU page mapping reference

`OffloadingConnectorWorker.register_kv_caches()` 会把真实 GPU KV cache 转成：

```text
[num_blocks, page_size_bytes]
```

的 byte-level canonical view，并处理：

- layer mapping；
- page stride；
- packed layouts；
- aliased layers；
- group refs。

因此如果未来做：

```text
pre-D2H GPU codec
```

我们不需要从零思考：

```text
如何从 vLLM layer tensor 找到一个 physical KV page 的 byte region？
```

newer upstream 已经有 mature generalization。

这会降低 V3 pre-D2H 的 architecture uncertainty。

但 porting 仍然很大，所以仍不进入 Core。

---

## 0.17.2 Pinned host memory reference

newer CPU offload path：

```text
cudaHostRegister
```

整个 mmap host region。

这说明 P2 如果以后做：

```text
reusable host staging
```

可以直接参考成熟 pinned-memory lifecycle。

不建议在 V1 自己手写：

```text
random torch.empty(pin_memory=True)
```

散落各处。

---

## 0.17.3 Stream/Event/Descriptor Pool reference

newer offload worker 已经维护：

```text
_stream_pool
_event_pool
_buffer_pool
```

transfer 完成后复用。

因此 V3 pinned-buffer/pool optimization 可以升级成：

> **reference-backed resource recycling**

而不是“我感觉 allocation 很慢，所以做个 pool”。

仍然应该由 NSYS 证明：

```text
allocation / object churn
```

值得优化后再做。

---

## 0.17.4 Transfer ordering reference

newer worker 在 GPU→CPU store 前：

```text
transfer stream waits current compute stream
```

因为 live KV 仍可能被 model 写。

并让多个 transfer：

```text
event ordered
```

这正是 pre-D2H codec 将来必须遵守的 lifetime rule：

```text
encode must not read before model finishes writing source page

page must not be reused
until encode + transfer no longer read it
```

这和 P1 的 block lifetime thinking 非常一致。

---

## 0.17.5 不要用 Triton 替代所有 memcpy

newer vLLM source 甚至明确区分：

```text
GPU→CPU:
bandwidth-bound
dedicated copy engine preferred

CPU→GPU small aligned payload:
Triton may win
```

所以 P2 的 V3 设计应该是：

```text
Triton = codec
DMA/copy engine = bulk transfer
```

除非 profiler 给出反证。

这比“所有数据搬运都写 Triton kernel”更 production-like。

---

# 0.18 CacheGen 再核实后的角色：保持 reference，不升级为实现依赖

CacheGen README 直接说：

```text
latest update and integration
→ LMCache
```

它旧版 implementation 仍然能看到：

```text
serialize KV
save encoded bytes
deserialize
feed past_key_values
```

但那条路不是我们要做的 production vLLM connector path。

因此 CacheGen 继续只承担：

```text
compressed KV loading motivation
codec-vs-transfer system framing
```

而不进入：

```text
P2 code dependency
```

这是正确取舍。

---

# 0.19 V1/V2 Implementation Readiness Matrix

| 工作项 | reference | 自己设计量 | fallback | 风险 |
|---|---|---:|---|---:|
| vLLM↔LMCache connector | exact | 很低 | upstream example | 低 |
| request/block/slot tracking | exact | 很低 | existing adapter | 低 |
| distributed FS L2 | exact | 很低 | local raw smoke | 低 |
| temp serialized MemoryObj | exact wrapper | 无 | upstream | 低 |
| async serde lifecycle | exact | 无 | upstream | 低 |
| fixed Block-INT8 format | mechanical | 中低 | FP8 baseline | 低 |
| exact serialized size | exact contract | 低 | conservative format | 低 |
| file-byte proof | exact FS impl | 无 | OS stat + L2 metrics | 低 |
| PyTorch codec | standard | 低 | — | 低 |
| Triton INT8 encode/decode | TurboQuant + vLLM quant refs | 中 | PyTorch oracle | 中低 |
| E2E speedup | benchmark only | — | negative result acceptable | 不控制完成 |
| first-D2H reduction | newer offload gives mapping/lifetime refs | 高 porting | stay L2 serde | V3 |
| pinned/resource pools | newer vLLM exact analogous ref | 中 | current wrapper | V3 |
| layerwise pipeline | LMCache/vLLM connector hooks | 中 | synchronous layer wait | V3 |
| adaptive per-object codec | no need for Core | 中高 format complexity | config-level selection | 低优先 |

结论：

> **P2 V1/V2 可以直接进入 implementation planning；未知集中在 V3 placement optimization，而不是 codec/reuse Core。**


# 1. 为什么做这个问题

## 1.1 Prefix caching 只覆盖单一 runtime 内部的本地 KV 生命周期

长上下文应用经常有：

```text
same/similar context
↓
multiple requests
```

如果每次都：

```text
prefill from scratch
```

会重复执行大量 attention / MLP compute。

因此 external KV cache 的价值是：

```text
computed KV
↓
store outside active GPU KV pool
↓
future request hit
↓
load KV
↓
skip recompute
```

---

# 2. 但 raw KV reuse 又产生新的系统成本

如果 raw format 是 BF16：

```text
2 bytes / element
```

对于很长 context：

```text
KV object size
```

非常大。

因此 external reuse 会受到：

```text
host/cache capacity
transfer volume
memory bandwidth
PCIe bandwidth
serialization cost
```

共同影响。

这就出现一个非常系统的问题：

> **压缩减少 bytes，但 encode/decode 自身需要时间；什么时候压缩值得？**

---

# 3. 核心 cost model

Raw：

```text
T_raw
≈
bytes_raw / BW_transfer
```

Compressed：

```text
T_comp
≈
T_encode
+
bytes_comp / BW_transfer
+
T_decode
```

因此理论 crossover：

```text
T_encode + T_decode
<
(bytes_raw - bytes_comp) / BW_transfer
```

这比单独说：

```text
INT8 比 BF16 小 2×
```

更有 AI Infra 价值。

因为真正问题是：

```text
size saving
vs
codec overhead
vs
transfer
vs
cache-hit TTFT
```

---

# 4. 为什么这个项目不是 toy

它接的不是自己写的：

```text
save_tensor.pt
load_tensor.pt
```

而是 vLLM 已经存在的：

```text
KVConnector
```

以及 LMCache 已经存在的：

```text
Serde
AsyncSerdeProcessor
SerdeL2AdapterWrapper
L1Manager
L2 adapter
```

因此项目链条是：

```text
vLLM scheduler/runtime
        │
        ▼
LMCacheConnectorV1
        │
        ▼
LMCache integration
        │
        ▼
Serde / L2
        │
        ▼
compressed KV
```

---

# 5. Source Fact：vLLM v0.26 已经有 LMCacheConnectorV1

参考：

**vLLM**
- repo: `vllm-project/vllm`
- commit: `568afb3a13806beb53bb2e6bd518269357b237c0`
- file:
  `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`

worker-side 已定义：

```text
register_kv_caches
start_load_kv
wait_for_layer_load
save_kv_layer
wait_for_save
```

其中 `save_kv_layer()` 的参数直接包括：

```text
kv_layer: paged KV buffer
attn_metadata
```

而 `start_load_kv()` 是在 model forward 前发起 load。

这说明：

> 我们不需要自己造一个 vLLM↔external cache bridge。

真正 production integration point 已经存在。

Reference:
- https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py

---

# 6. Source Fact：LMCache 的 vLLM integration 已经维护 request/block/slot mapping

参考：

```text
LMCache/lmcache/integration/vllm/vllm_v1_adapter.py
```

`RequestTracker` 已经维护：

```text
request id
prompt length
token ids
allocated block ids
num_saved_tokens
decode phase
LMCache cached token count
```

并根据：

```text
allocated block ids
+
block_size
```

生成：

```text
slot_mapping
```

这说明 LMCache integration 不是一个 tensor dump 工具，而是理解 vLLM paged KV block mapping 的 runtime adapter。

Reference:
- https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/integration/vllm/vllm_v1_adapter.py

---

# 7. Source Fact：LMCache 已经把 custom codec 做成正式 Serde extension point

参考：

```text
docs/design/v1/distributed/serde/README.md
```

它明确定义：

```text
Serializer
Deserializer
SerdeProcessor
AsyncSerdeProcessor
factory registration
```

用户真正需要实现的是 sync layer：

```text
serialize(src, dst, key)
estimate_serialized_size(layout)
deserialize(src, dst, key)
```

Async 部分：

```text
submit
event notifier/eventfd
query
thread pool
lifecycle
```

由 `AsyncSerdeProcessor` 提供。

因此项目不需要自己重写 async framework。

Reference:
- https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/docs/design/v1/distributed/serde/README.md

---

# 8. Source Fact：SerdeL2AdapterWrapper 已经把 codec 接入 store/load pipeline

参考：

```text
docs/design/v1/distributed/l2_adapters/serde_wrapper.md
```

Store：

```text
caller
↓
wrapper
↓
temporary buffer
↓
serialize
↓
inner L2 store
```

Load：

```text
inner L2 load
↓
temporary serialized bytes
↓
deserialize
↓
destination KV object
```

同时 wrapper 负责：

```text
temp buffer lifecycle
failure handling
eventfd
locking
completion
```

因此我们的 custom codec 不需要触碰绝大多数 controller/lifecycle code。

Reference:
- https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/docs/design/v1/distributed/l2_adapters/serde_wrapper.md

---

# 9. Source Fact：LMCache 自带 FP8 serde，是最好的最小模板

参考：

```text
lmcache/v1/distributed/serde/fp8.py
```

它非常短。

Store：

```text
src tensor
↓
cast FP8
↓
view uint8
↓
copy into serialized buffer
```

Load：

```text
uint8 bytes
↓
view FP8
↓
reshape
↓
cast original dtype
```

最后：

```python
register_serde_factory("fp8", ...)
```

所以我们的 Block-INT8 custom serde 有非常明确的最小实现模板。

Reference:
- https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/serde/fp8.py

---

# 10. Source Fact：LMCache 已经有更复杂的 Triton TurboQuant backend

这是确定性的关键。

参考：

```text
lmcache/v1/distributed/serde/turboquant/
```

包括：

```text
turboquant.py
store_kernel.py
decode_kernel.py
```

Store kernel 明确是：

```text
Fused Triton kernels for TurboQuant KV store
```

包含：

```text
tl.load
min/max
scale
quantize
3-bit/4-bit packing
slot addressing
tl.store
```

这说明：

> 在 LMCache serde 体系中使用 Triton codec 已经有成熟 production reference。

Reference:
- https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/serde/turboquant/store_kernel.py

---

# 11. Source Fact：TurboQuant Serializer 已经处理 CPU/CUDA MemoryObj staging

这点非常重要。

`TurboQuantSerializer.serialize()` 会检查：

```text
src_tensor.is_cuda
dst_tensor.is_cuda
```

如果 MemoryObj 是 CPU/pinned：

```text
CPU src
↓
temporary CUDA src
↓
Triton codec
↓
temporary CUDA compressed buffer
↓
copy compressed bytes back CPU dst
```

如果本身是 CUDA：

```text
直接使用 CUDA tensor
```

因此它公开证明两件事：

1. Triton serde 在 LMCache 中是可行的；
2. **codec 放置位置会直接决定 PCIe 上到底搬 raw bytes 还是 compressed bytes。**

Reference:
- https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/serde/turboquant/turboquant.py

---

# 12. Critical Design Point：不要把“compressed L2 storage”自动说成“PCIe bytes reduced”

这部分现在进一步收紧成三个完全不同的数据层级。

## Layer 1 — vLLM GPU paged KV

```text
GPU resident KV
```

## Layer 2 — LMCache L1 raw MemoryObj

通常是：

```text
raw KV object
```

## Layer 3 — L2 serialized object

Serde wrapper 作用于：

```text
L1 raw object
→ serialized temp
→ L2
```

因此三类 claim 必须分开：

| Claim | generic Serde 是否自动成立 |
|---|---|
| L2 serialized bytes ↓ | **是** |
| L1 raw CPU bytes ↓ | **否** |
| first GPU→CPU D2H bytes ↓ | **否** |

---

### Path A — generic Serde after raw L1 materialization

```text
GPU paged KV
↓ raw D2H
L1 raw object
↓ Serde
compressed L2 bytes
```

结果：

```text
L2 footprint ↓
L1→L2 bytes ↓
```

但：

```text
first D2H unchanged
```

---

### Path B — pre-D2H GPU codec

```text
GPU paged KV
↓ GPU codec
compressed GPU bytes
↓ D2H
L1/L2
```

只有这种 path 才能直接减少：

```text
GPU-host PCIe bytes
```

---

### Project rule

V1/V2 Core：

```text
只承诺 L2 serialized compression
```

V3 optional：

```text
pre-D2H GPU codec
```

只有当 NSYS 明确看到：

```text
compressed D2H memcpy size < raw D2H memcpy size
```

之后，README 才允许写：

```text
reduced PCIe transfer volume
```

# 13. CacheGen 的角色

参考：

**CacheGen**
- repo: `UChi-JCL/CacheGen`

它本身就是：

```text
KV cache encoding/decoding
for compressed context loading/streaming
```

README 现在直接写：

> latest update and integration should use LMCache.

因此 CacheGen 对我们的价值是：

```text
system motivation / historical design reference
```

而不是：

```text
我们 fork CacheGen 然后改一点
```

Reference:
- https://github.com/UChi-JCL/CacheGen

---

# 14. Project Decision：自研 codec 用简单 Block-INT8，而不是复制 TurboQuant

LMCache 已经有：

```text
FP8
TurboQuant 8/4/3 bit variants
```

所以我们不能把：

```text
“我给 LMCache 加了压缩”
```

当作项目贡献。

正确 project ownership 是：

> **实现一个足够透明的 Block-INT8 codec，用它建立完整 correctness / transfer / crossover benchmark，并在真实 vLLM+LMCache path 中验证 codec placement。**

codec 不追求算法 novelty。

---

# 15. Block-INT8 Format

建议使用：

```text
symmetric per-block/per-head INT8
```

对于每个：

```text
block × KV head
```

计算：

```text
amax = max(abs(x))
scale = max(amax / 127, eps)

q = round(x / scale)
q = clamp(q, -127, 127)
```

存储：

```text
int8 payload
+
scale
+
small layout metadata if needed
```

decode：

```text
x_hat = q * scale
```

---

# 16. 为什么选择 INT8

不是因为它最先进。

是因为它：

1. A100 友好；
2. 量化数学简单；
3. PyTorch oracle 非常容易写；
4. size deterministic；
5. serde size estimator 简单；
6. Triton reduction + quantization 很标准；
7. 能清晰展示 codec vs transfer trade-off；
8. 不会让项目变成 quantization research。

这非常符合第二项目定位：

```text
engineering completeness
>
novel codec
```

---

# 17. V1 — Custom Block-INT8 Serde + Real LMCache E2E

V1 的 scope 现在固定得更窄、更确定：

```text
V0-A
vLLM + LMCache shared-prefix reuse smoke
        ↓
V0-B
distributed filesystem L2 raw baseline
        ↓
V1
filesystem L2 + custom Block-INT8 Serde
        ↓
store / load / deserialize
        ↓
vLLM resume
```

V1 不要求 Triton。

V1 的核心 proof 是：

```text
real connector reuse
+
real SerdeL2AdapterWrapper
+
real filesystem L2
+
compressed .data bytes
+
successful load/resume
```

而不是：

```text
独立 tensor codec demo
```

---

## 为什么固定 filesystem L2

因为它：

- upstream 已实现；
- 本机即可；
- 不需要服务部署；
- object 最终是实际 `.data` file；
- byte footprint 可直接检查；
- 与 SerdeL2AdapterWrapper contract 配套。

这使 V1 的 system dependency 极小。

# 18. V1 Step 1 — Reproduce Upstream Baseline

先分成两个 baseline，避免把 LMCache 的 legacy/local path 与 distributed L2 Serde path混在一起。

## V0-A — Connector / reuse smoke

直接参考：

```text
examples/kv_cache_reuse/local_backends/offload.py
```

它已经展示：

```text
vLLM LLM
+
KVTransferConfig(
    kv_connector="LMCacheConnectorV1"
)
```

两次共享长 prefix 请求：

```text
A → store KV
B → reuse cached KV
```

我们先复现它，验证：

- vLLM connector 可用；
- LMCache request tracking 可用；
- same-prefix hit 可用；
- load/resume generation 可用。

---

## V0-B — Distributed filesystem L2 baseline

然后切到：

```text
filesystem L2 adapter
+
serde disabled
```

验证：

```text
raw L1 object
→ FS L2
→ .data
→ load
→ L1
→ vLLM resume
```

记录：

```text
raw L2 bytes
raw .data file size
store latency
load latency
cache-hit TTFT
```

---

## Environment/ABI Gate

在 V0-A 前先固定：

```text
vLLM = 568afb...
torch = 2.11.x
LMCache = a7afade...
```

并验证 native extension 真正 load，而不是 silent fallback。

如果环境 gate 不通过：

```text
stop
```

先修环境，不进入 codec 开发。

这样不会把环境问题误判成 Serde/codec bug。

# 19. V1 Step 2 — PyTorch Block-INT8 Reference

建议文件：

```text
lmcache/v1/distributed/serde/block_int8.py
```

或者如果不直接污染 upstream tree：

```text
project_ext/block_int8.py
```

实现：

```text
BlockInt8Serializer
BlockInt8Deserializer
_create_block_int8_serde
register_serde_factory(...)
```

严格遵循 LMCache serde ABC。

---

# 20. V1 Serialized Layout

要明确设计，不要随意 Python pickle。

例如：

```text
Header / fixed metadata
─────────────────────────────
num_blocks
num_heads
head_dim
block_size
scale dtype
payload offsets

Scales
─────────────────────────────
[num_blocks, num_heads, ...]

Payload
─────────────────────────────
int8 flattened KV bytes
```

更简单的第一版甚至可以：

```text
fixed known layout
```

不写复杂 header。

利用 LMCache 的 `MemoryLayoutDesc` 恢复 shape。

---

# 21. V1 `estimate_serialized_size`

这是正式 contract。

必须做到：

```text
estimate >= actual bytes
```

最好对于固定 INT8 format：

```text
estimate == exact bytes
```

例如 raw BF16：

```text
2 × N bytes
```

INT8：

```text
1 × N
+
scale bytes
```

所以压缩率大致接近：

```text
~2×
```

但会被 scale metadata 稍微稀释。

---

# 22. V1 Correctness

## Tensor round-trip

随机：

```text
BF16 KV
↓
serialize
↓
deserialize
```

检查：

```text
shape
dtype
serialized bytes
MAE
RMSE
max error
cosine similarity
```

---

## PyTorch oracle

自己写：

```text
reference_quantize
reference_dequantize
```

Serde 结果必须和 reference 对齐。

---

## Size contract

检查：

```text
actual_bytes <= estimate
```

以及：

```text
raw_bytes
compressed_bytes
compression ratio
```

---

## Async contract

LMCache 已提供 `AsyncSerdeProcessor`。

测试：

```text
submit
↓
pending
↓
event
↓
query exactly once
```

不需要自己实现 eventfd。

---

## E2E

```text
save compressed KV
↓
load
↓
restore
↓
vLLM generation
```

验证：

```text
request completes
cache hit recognized
no corrupted KV
```

因为 quantization 是 lossy，不要求 token-by-token 与 BF16 完全一致。

---

# 23. V1 可以带来的确定收益

V1 的确定收益现在只写 source contract 能保证的部分。

## 23.1 L2 serialized footprint

Raw BF16：

```text
~2 bytes / element
```

Block-INT8：

```text
~1 byte / element
+
scale metadata
```

因此只要 format 实现正确：

```text
serialized L2 bytes ↓
```

是 deterministic。

filesystem L2 还能直接用：

```text
.data file size
```

做物理证据。

---

## 23.2 Fixed L2 byte budget capacity

在固定 L2 storage budget 下：

```text
serialized object size ↓
→ number of stored KV chunks ↑
```

这是直接的容量关系。

---

## 23.3 L1→L2 bytes

SerdeL2AdapterWrapper 把 serialized temp 交给 inner L2 adapter。

因此：

```text
inner L2 store bytes
```

也应该按 serialized size 下降。

可以使用：

```text
L2StoreResult.bytes_transferred()
```

作为软件侧证据。

---

## 23.4 不再宣称的“确定收益”

V1 默认不能说：

```text
L1 CPU capacity ↑
GPU↔CPU PCIe bytes ↓
cache-hit TTFT ↓
throughput ↑
```

这些属于其他 layer 或需要 benchmark。

# 24. V1 条件性收益

需要 benchmark 的：

```text
L2 store latency
L2 load latency
cache-hit TTFT
E2E latency
```

原因：

```text
saved I/O bytes
vs
serialize/de-serialize cost
```

存在 crossover。

例如可能：

```text
small object:
codec overhead > saved I/O

large object:
saved I/O > codec overhead
```

这就是项目应该研究的真实 systems trade-off。

另外：

```text
L1 CPU raw footprint
```

取决于 StorePolicy 是否在 L2 store 成功后删除 L1 raw object。

默认 policy 不删除，因此不要把 L1 capacity improvement 混入 Serde result。

# 25. V1 实现确定性

评级：

> **很高。**

不是因为“INT8 很简单”，而是整条 V1 path 都有 exact reference：

| 环节 | Exact reference |
|---|---|
| vLLM connector | vLLM `LMCacheConnectorV1` |
| shared-prefix reuse | LMCache `examples/kv_cache_reuse/local_backends/offload.py` |
| request/block mapping | `vllm_v1_adapter.py` |
| Serde interface | `serde/base.py` / design README |
| async wrapper | `AsyncSerdeProcessor` |
| L2 Serde composition | `SerdeL2AdapterWrapper` |
| deterministic local backend | `FSL2Adapter` |
| simple codec structure | FP8 Serde |
| advanced quantized codec | TurboQuant |

我们真正自己定义：

```text
Block-INT8 byte format
scale granularity
scale dtype
```

这些全部有：

```text
exact size formula
PyTorch oracle
round-trip test
```

所以不是未知研究方案。

### 最大实际风险

1. version/ABI；
2. MemoryObj layout；
3. serialized-size accounting；
4. config/registration；
5. lossy codec quality。

其中 ABI 风险已经通过切换到 LMCache `a7af...` implementation baseline 提前消掉。

# 26. V2 — Triton Block-INT8 Codec + Placement Verification

V2 Required Core 现在定义为：

```text
PyTorch Block-INT8
        ↓
Triton encode/decode
        ↓
same Serde contract
        ↓
filesystem L2 E2E
        ↓
kernel benchmark + NCU/NSYS
```

也就是说：

> **Triton 是 P2 的 GPU implementation strengthening，但 V2 不再以“减少第一段 PCIe D2H”为完成条件。**

V2 必须回答：

1. Triton 与 PyTorch quant/dequant oracle 是否一致；
2. codec throughput 是多少；
3. L2 serialized bytes 是否不变；
4. codec 在 end-to-end store/load critical path 占多少；
5. context/object size 增长后，codec overhead 与 L2 I/O savings 在哪里 crossover。

然后把：

```text
pre-D2H compressed transfer
```

移到 V3 optional。

# 27. V2 Triton Encode

建议 grid：

```text
(num_blocks × num_kv_heads × K/V)
```

每个 program：

```text
load one block/head vector
↓
fp32 absmax reduction
↓
scale
↓
quantize INT8
↓
store payload
↓
store scale
```

核心 Triton primitive：

```text
tl.arange
tl.load
tl.abs
tl.max
tl.where
tl.clamp
tl.store
```

这比 TurboQuant 明显简单。

---

# 28. V2 Triton Decode

每个 program：

```text
load scale
↓
load int8 payload
↓
cast fp32
↓
multiply scale
↓
cast BF16
↓
store reconstructed KV
```

可以和 PyTorch oracle 精确比较量化结果。

---

# 29. V2 为什么 reference 很充分

## LMCache TurboQuant

已经实现：

```text
Triton store
Triton decode
low-bit packing
per-vector scale/zero
CUDA staging
```

我们的 INT8 是它的简化子问题。

---

## vLLM Triton KV Store

upstream 又提供：

```text
paged slot mapping
block addressing
per-token-head quantization
```

所以无论 codec 输入是：

```text
LMCache contiguous KV object
```

还是后续要更靠近：

```text
vLLM paged KV
```

都有 reference。

---

# 30. V2 最重要的实验：确定 codec placement

V2 仍然必须做 codec placement profiling，但目的改为：

> **弄清楚数据到底在哪一层被压缩，不提前假设 PCIe benefit。**

用 NVTX + NSYS 画：

```text
vLLM paged KV
        │
        ▼
GPU connector/extract
        │
        ▼
raw L1 MemoryObj?
        │
        ▼
Serde Triton encode
        │
        ▼
serialized temp
        │
        ▼
filesystem L2
```

要确认：

- raw GPU→L1 copy 在哪里；
- serializer 输入是 CPU 还是 CUDA；
- serializer 是否产生 CUDA staging；
- compressed bytes 从哪里开始成为 source of truth；
- L1→L2 实际 bytes；
- load 方向是否对称。

输出不是“必须证明 PCIe 下降”，而是一张：

```text
actual data-movement map
```

这张图本身就是非常好的 infra evidence。

# 31. 如果 generic Serde 已经拿到 CUDA source

如果实测 generic Serde 的 `src MemoryObj` 本身就在 CUDA：

```text
CUDA raw object
↓
Triton encode
↓
serialized CUDA temp
↓
后续 transfer
```

那么可以进一步检查：

```text
后续 D2H 是否直接搬 compressed temp
```

若是：

```text
PCIe reduction
```

可能自然成立。

但它仍必须由：

```text
NSYS memcpy size
```

证明。

不能只根据 tensor dtype/shape 推断。

# 32. 如果 generic Serde 拿到的是 CPU source

这是当前更需要防范的情况。

如果：

```text
GPU KV
↓ raw D2H
CPU/L1 source
↓ Triton serializer
```

TurboQuant-style CUDA codec 可能变成：

```text
raw CPU
↓ H2D raw
GPU encode
↓ D2H compressed
CPU temp
```

这对 PCIe 不但未必节省，反而增加搬运。

因此：

## V2 Core

仍然只做：

```text
Triton codec correctness
L2 serialized compression
L2 I/O/crossover analysis
```

## V3 optional

如果要真正做：

```text
compressed-before-D2H
```

必须把 codec hook 往前移动到：

```text
vLLM/LMCache GPU connector extraction path
```

而不是强行通过 generic L2 Serde 完成。

截至本次 audit，这个 exact pre-D2H hook 没有被确认成现成 drop-in reference，所以不进入 V2 必做。

# 33. V2 确定性

分成三个问题。

## A. Triton codec correctness

> **高**

Reference：

- LMCache TurboQuant Triton store/decode；
- FP8 Serde；
- PyTorch oracle。

---

## B. Triton codec performance

> **不保证**

INT8 很可能是 bandwidth-oriented kernel，但：

```text
launch overhead
reduction cost
staging
object size
```

会决定收益。

正确但没有 speedup 仍是有效结果。

---

## C. 第一段 GPU-host PCIe reduction

> **不作为 V2 Core；当前是 D 级 optional。**

原因：

generic Serde placement 不保证 encode 发生在 raw D2H 之前。

如果未来实现 V3 connector-side pre-D2H codec，再单独验证。

# 34. V2 Benchmark

比较：

```text
Raw BF16
LMCache FP8
Our Block-INT8 PyTorch
Our Block-INT8 Triton
LMCache TurboQuant（环境允许时）
```

注意：

> 我们不是必须击败 TurboQuant。

TurboQuant 是 production advanced baseline。

我们的价值是：

```text
simple codec
+
transparent implementation
+
full-system analysis
```

---

# 35. Metrics

## Codec

```text
encode ms
decode ms
GB/s effective codec throughput
compression ratio
MAE/RMSE
```

---

## Memory/storage

```text
raw bytes
serialized L2 bytes
filesystem .data bytes
objects cached under fixed L2 byte budget
```

---

## Transfer

```text
D2H bytes
H2D bytes
D2H duration
H2D duration
effective PCIe bandwidth
```

只有 profiler 能证明。

---

## Serving

```text
cold TTFT
cache-hit TTFT
E2E latency
throughput
```

---

# 36. Crossover Sweep

变量：

```text
context length:
1K
2K
4K
8K
16K
...

batch size

codec:
raw
FP8
INT8
TurboQuant
```

画：

```text
context size
vs
load latency
```

找到：

```text
crossover point
```

---

# 37. 理论 vs 实测

理论：

```text
T_saved_transfer
=
(bytes_raw - bytes_comp) / BW_pcie
```

codec：

```text
T_codec
=
T_encode + T_decode
```

如果：

```text
T_codec < T_saved_transfer
```

理论上 compression 对 transfer path 有利。

然后拿 NSYS/benchmark 的：

```text
measured crossover
```

和理论比较。

这会成为 P2 最好的系统分析部分。

---

# 38. Profiling

NVTX：

```text
P2_KV_EXTRACT
P2_SERIALIZE
P2_TRITON_ENCODE
P2_D2H
P2_L2_STORE
P2_L2_LOAD
P2_H2D
P2_TRITON_DECODE
P2_DESERIALIZE
P2_RESUME
```

---

## NSYS

看：

```text
codec kernel
memcpy H2D/D2H
stream overlap
CPU thread pool
eventfd wake-up
load→first decode critical path
```

---

## NCU

只看 Triton codec：

```text
DRAM BW
occupancy
register pressure
load/store efficiency
```

---

# 39. V3 — 后续优化扩展：优先做已有 runtime primitive 可借的优化

V3 不再按“想得到什么就列什么”。

本轮 GitHub audit 后，建议按 reference 完整度和简历价值重新排序。

---

## 39.1 V3-A — Layerwise Load/Save Pipeline【最高推荐】

这是 P2 最值得优先做的 V3。

原因：

vLLM/LMCache connector 已经有明确 hook：

```text
start_load_kv(...)
wait_for_layer_load(layer_name)

save_kv_layer(layer_name, kv_layer, attn_metadata)
wait_for_save()
```

connector 文档甚至明确说：

```text
wait_for_layer_load
is useful for layer-by-layer pipelining
```

因此可以设计：

### Load

```text
preload layer 0
        ↓
compute layer 0
        │
        ├── meanwhile load layer 1
        ↓
compute layer 1
        │
        ├── meanwhile load layer 2
...
```

### Save

```text
layer i finishes writing KV
        ↓
launch save/codec for layer i
        │
        ├── model computes later layers
...
```

### 为什么比“任意 chunk pipeline”更推荐

因为：

```text
layer boundary
```

已经是 connector API 的显式 synchronization boundary。

我们不是再造：

```text
chunk scheduler
```

### V3-A success condition

不是一定 TTFT 更低。

先证明：

```text
NSYS:
load/save activity overlaps model execution
```

然后测：

```text
critical-path exposed transfer/codec time ↓
```

如果没有改善，也能解释：

- layer granularity 太粗；
- codec dominant；
- IO backend dominant；
- transfer stream serialization。

---

## 39.2 V3-B — Temp / Pinned Buffer / Stream / Event Reuse【第二推荐】

LMCache wrapper 每次 serde task 都需要 temp MemoryObj。

如果 profiler 发现：

```text
reserve/release churn
thread scheduling
temporary allocation
```

明显，可以做 pool。

### 这次有更强 reference

newer vLLM generic CPU offload worker 已经维护：

```text
stream pool
event pool
pinned descriptor buffer pool
pinned mmap host region
```

并在 transfer completion 后 recycle。

因此我们可以参考：

```text
resource ownership
pool lifecycle
shutdown cleanup
```

而不是自己发明。

### 注意

不要绕过 LMCache 的 L1Manager lock/accounting。

更合理的方向是：

```text
在遵守 L1Manager contract 的前提下
复用 temporary storage / backing allocation
```

而不是：

```text
另建一套不可见的 raw pointer pool
```

### 进入条件

NSYS/CPU profile 至少证明：

```text
allocation/lifecycle overhead
```

是可见项。

否则不做。

---

## 39.3 V3-C — Transfer Scheduling：把 store 移出 token critical path

newer vLLM offload worker 有一个很值得借的策略：

```text
prepare store
→ defer actual store to next engine step
```

理由是：

```text
avoid delaying token-generation related transfers
```

这类 scheduling 比“加更多 stream”更有系统价值。

### 我们可以做的 scoped variant

如果 V2 profiling 发现：

```text
serialize/store
```

阻塞了：

```text
sample / next decode critical path
```

可以尝试：

```text
mark store-ready
↓
defer background serialize/L2 submission
↓
next safe runtime boundary launches it
```

### 必须保留 lifetime fence

在 store/codec 仍读 source object 时：

```text
source backing storage
```

不能被 reuse/delete。

这需要借 LMCache existing L1 read lock 或 connector completion。

不要自己裸管理 pointer lifetime。

---

## 39.4 V3-D — Pre-D2H GPU Compression【最强但风险较高】

这是原文档最吸引人的 V3：

```text
GPU paged KV
→ GPU INT8 encode
→ compressed D2H
→ host/external storage
```

Load：

```text
compressed host
→ H2D compressed
→ GPU decode
→ paged KV
```

### 本轮之后，它不再是“完全没有参考”

newer vLLM upstream 已经给出：

```text
CanonicalKVCaches
CanonicalKVCacheRef
GPULoadStoreSpec
GPU page byte views
async CPU offload worker
transfer fences
```

所以我们现在知道：

```text
如何描述 raw GPU physical pages
如何建立 block transfer plan
如何 fence compute/transfer
如何统计 transfer bytes
```

这些都有成熟 reference。

### 但为什么仍然不提升成 V2/Core

因为我们的 implementation baseline 是：

```text
vLLM 0.26
+
LMCache a7af...
```

而 newer vLLM generic offload architecture 并不在 pinned baseline 中。

把它移植进 LMCache Serde path 仍然会涉及：

- connector integration；
- compressed staging ownership；
- ObjectKey/chunk format；
- deserialize destination；
- async completion；
- page lifetime；
- error fallback。

这是 substantial port，不是简单 patch。

所以等级可以从：

```text
D: 几乎未知
```

提升为：

```text
B/C reference-backed extension,
but high integration cost
```

仍然是 optional V3。

---

### 39.4.1 如果真的做，必须先通过 5 个 gate

#### Gate 1 — raw transfer owner 已定位

NSYS + source trace 能明确：

```text
哪一个函数
把 vLLM paged KV 的 bytes
从 GPU 搬到 host
```

#### Gate 2 — source page lifetime 可 fence

必须知道：

```text
model 最后一次写
→ encode read
→ D2H read
→ page safe for reuse
```

#### Gate 3 — compressed representation 有明确 owner

不能出现：

```text
GPU temp
CPU temp
LMCache L1 temp
L2 temp
```

四份拷贝都没人清楚谁释放。

#### Gate 4 — load reverse path 清楚

```text
compressed bytes
```

最终必须被写回：

```text
vLLM allocated physical KV slots
```

并在对应 attention layer 使用前完成。

#### Gate 5 — fallback

任何异常：

```text
codec unavailable
unsupported layout
```

都能回退：

```text
raw existing connector path
```

如果这五项不满足：

> 不做 V3-D。

---

## 39.5 V3-E — Codec / Transfer Overlap

只有当 V3-D 或 actual Serde device path 已经允许 GPU-side codec 时才考虑。

Store pipeline：

```text
encode chunk/layer N+1
while
D2H compressed N
```

Load：

```text
H2D compressed N+1
while
decode N
```

### 参考

- LMCache layerwise load/save hooks；
- newer vLLM stream/event transfer worker。

### 不推荐的做法

一次创建大量 stream：

```text
one stream per layer forever
```

容易：

- event explosion；
- memory pressure；
- copy-engine contention。

更推荐：

```text
small reusable stream pool
+
explicit events
```

---

## 39.6 V3-F — Config-level Adaptive Raw vs Compressed

原方案写：

```text
if saved_transfer_time > codec_time:
    compressed
else:
    raw
```

思想没错，但 first version 不建议做：

```text
per-object dynamic variable format
```

原因是 LMCache serialized temp allocation依赖：

```text
layout → estimated size
```

而 filesystem adapter 会写整个 allocated MemoryObj。

如果同一个 serde type 对不同 object 输出：

```text
raw or compressed
```

但 estimator 必须给 worst-case：

```text
raw size
```

可能直接把 L2 physical saving 抹掉。

### 推荐更可实现的版本

先做：

```text
config/workload-level policy
```

例如：

```text
object size < threshold
→ use raw adapter

object size >= threshold
→ use block_int8 adapter
```

通过不同 adapter/config path 选择，而不是让一个 object format data-dependent。

只有 LMCache 后续支持：

```text
per-object exact variable serialized length
```

且 backend 只 persist actual bytes，才升级成真正 dynamic policy。

---

## 39.7 V3-G — More Aggressive Codec【低优先级】

在 Block-INT8 全部完成后，可以尝试：

```text
INT4
K8V4
K/V mixed precision
```

但 LMCache 已经有 TurboQuant。

所以这项对简历的边际价值反而不高。

如果做，理由必须来自：

```text
current INT8 crossover 太晚
→ codec overhead acceptable
→ byte saving 不够
```

然后再试更低 bit。

不要为了“更低 bit 看起来更前沿”而做。

---

## 39.8 V3 推荐路线

推荐优先级：

```text
V1
fixed Block-INT8 L2 Serde
        ↓
V2
Triton codec + placement map
        ↓
V3-A
layerwise load/save overlap
        ↓
V3-B
temp/pinned/stream/event resource reuse
        ↓
V3-C
transfer scheduling out of critical path
        ↓
V3-D
pre-D2H compressed transfer
        ↓
V3-E
codec/transfer overlap
        ↓
V3-F
config-level adaptive raw/compressed
        ↓
V3-G
lower-bit codec
```

### 为什么这样排

前 3 项：

```text
都能在现有 runtime contracts 上做
```

而 pre-D2H 虽然很强，但 porting 面明显更大。

因此第二简历项目最合理的策略是：

> **先拿到一个完整 V1/V2，再选择一个 profiling 最有证据的 V3。**

不是要求 V3 全做。


# 40. Project Success Criteria

## V0 Environment / Baseline Gate

- [ ] vLLM pinned commit 可运行；
- [ ] torch 2.11 confirmed；
- [ ] LMCache `a7af...` checked out；
- [ ] native extension really loads；
- [ ] shared-prefix reuse example reproduced；
- [ ] filesystem L2 raw store/load reproduced。

---

## V1 Core Done

- [ ] custom Block-INT8 serde registered；
- [ ] deterministic byte layout；
- [ ] `estimate_serialized_size` contract；
- [ ] tensor round-trip / PyTorch oracle；
- [ ] async serde lifecycle；
- [ ] SerdeL2AdapterWrapper real path；
- [ ] filesystem L2 `.data` size lower than raw BF16；
- [ ] `L2StoreResult.bytes_transferred` measured；
- [ ] compressed L2 object can load；
- [ ] vLLM cache-hit request can resume；
- [ ] quality/error metrics documented。

做到这里：

> P2 已经是完整可交付项目。

---

## V2 Strong Done

- [ ] Triton encode；
- [ ] Triton decode；
- [ ] Triton vs PyTorch oracle；
- [ ] kernel throughput benchmark；
- [ ] NCU report；
- [ ] NSYS actual data-movement map；
- [ ] raw / FP8 / INT8 / TurboQuant（可用时）对比；
- [ ] L2 I/O + codec crossover；
- [ ] 明确记录 serializer source device；
- [ ] **不未经证据声称 PCIe reduction**。

---

## V3 Optional

只有 source audit 与 profiler 支持时：

- [ ] pre-D2H GPU codec hook；
- [ ] compressed D2H/H2D；
- [ ] overlap；
- [ ] pinned/temp buffer reuse；
- [ ] adaptive compression policy。

V3 失败不影响 P2 完成。

# 41. 代码量合理预估

```text
V1 serde/runtime core    300–500 LOC
V2 Triton core           180–350 LOC
tests                    250–450 LOC
bench/profile            250–400 LOC
─────────────────────────────────
total                   ~980–1700 LOC
```

第二项目明显比 P1 小一档。

这是合理的。

---

# 42. 我们自己的 Ownership

## Upstream / reference

### vLLM
- KV Connector contract；
- paged KV runtime；
- layer save/load hooks。

### LMCache
- cache engine；
- vLLM adapter；
- MemoryObj；
- Serde ABC；
- AsyncSerdeProcessor；
- SerdeL2AdapterWrapper；
- L1/L2 lifecycle；
- FP8 baseline；
- TurboQuant reference。

### CacheGen
- compressed context/KV loading motivation；
- historical codec-based KV streaming architecture。

---

## 我们实现

- transparent Block-INT8 format；
- PyTorch codec oracle；
- LMCache custom serde；
- configuration/registration；
- Triton encode/decode；
- E2E integration validation；
- storage/transfer measurements；
- crossover cost model；
- NVTX/NSYS/NCU evidence；
- optional pinned buffer/overlap optimization。

---

# 43. 不能怎么写简历

不要写：

```text
Invented a new KV compression algorithm
```

因为没有。

不要写：

```text
Added KV compression to LMCache
```

因为 LMCache 已经有 FP8/TurboQuant。

不要在 V1 就写：

```text
Reduced PCIe transfer by 2×
```

除非 NSYS 证明 encode 在 D2H 前发生。

---

# 44. 正确的项目描述

V1 完成后：

> Implemented a custom Block-INT8 KV serde for LMCache and integrated it with vLLM’s real KV Connector and filesystem-L2 path, reducing serialized L2 footprint while preserving cache-hit load/resume semantics.

V2 完成后：

> Added Triton Block-INT8 encode/decode kernels and profiled the end-to-end codec/L2 path on A100, separating codec latency from serialized I/O cost and identifying the object-size crossover where compression becomes beneficial.

只有 V3 真正验证 pre-D2H compressed movement 后，才增加：

> Moved compression ahead of the raw GPU-host transfer boundary and verified reduced D2H/H2D bytes with Nsight Systems.

这个表述非常可信。

---

# 45. 实现确定性总结

| 子问题 | Reference 等级 | 实现确定性 | 收益确定性 | 是否 Core |
|---|---:|---:|---:|---|
| vLLM↔LMCache connector | A | 很高 | reuse capability 确定 | V0 |
| shared-prefix reuse | A | 很高 | recompute skip/hit 可验证 | V0 |
| filesystem L2 | A | 很高 | real bytes/file 可验证 | V0/V1 |
| Serde plugin | A | 很高 | transparent L2 transform 确定 | V1 |
| async wrapper | A | 很高 | lifecycle 确定 | V1 |
| Block-INT8 format | C | 很高 | L2 bytes ↓ 确定 | V1 |
| L2 serialized capacity | — | — | 确定 | V1 |
| L1 CPU capacity | — | — | 默认不成立 | 非 Core |
| Triton INT8 encode/decode | C + A/B reference | 高 | speedup 不保证 | V2 |
| L2 I/O latency | — | — | 条件性 | benchmark |
| cache-hit TTFT | — | — | 条件性 | benchmark |
| generic Serde 减少 first D2H | D | — | **不保证** | 非 Core |
| connector-side pre-D2H codec | B/C（newer vLLM 提供 canonical page/transfer/lifetime reference） | 中，主要风险变成 pinned-baseline porting | 需实测 | V3 |
| pinned buffer reuse | A/B follow-up hint | 高 | 条件性 | V3 |
| stream overlap | D | 中 | 条件性 | V3 |
| beat TurboQuant | — | — | 不要求 | 否 |

## Final risk statement

P2 当前 Core 已经避免两个最危险的“开工后才发现”问题：

1. **ABI/version mismatch**：implementation baseline 改到与 vLLM0.26 / torch2.11 对齐的 LMCache commit；
2. **PCIe placement assumption**：pre-D2H compression 移出 Core。

因此 V1/V2 必做部分都能落到：

```text
exact upstream interface
+
reference implementation
+
local deterministic backend
+
oracle
```

上。

剩余未知只存在于 optional V3。

# 46. Reference Inventory

## vLLM — pinned baseline

1. LMCache connector  
   https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py

2. Triton paged KV store/quant reference  
   https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py

3. vLLM build pin (`torch==2.11.0`)  
   https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/pyproject.toml

---

## LMCache — implementation baseline

**Pinned compatibility commit**

```text
a7afadebb9248b62b5c533ce2c12297e9d94fc4a
```

Commit title：

```text
[CI] fix abi mismatch and smoke checks (#4423)
```

GitHub：
https://github.com/LMCache/LMCache/commit/a7afadebb9248b62b5c533ce2c12297e9d94fc4a

4. vLLM local reuse example  
   https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/examples/kv_cache_reuse/local_backends/offload.py

5. vLLM adapter  
   https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/integration/vllm/vllm_v1_adapter.py

6. Serde design  
   https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/docs/design/v1/distributed/serde/README.md

7. Serde L2 wrapper  
   https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/docs/design/v1/distributed/l2_adapters/serde_wrapper.md

8. L2 Store/Prefetch architecture / StorePolicy  
   https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/docs/design/v1/distributed/l2_adapters/overall.md

9. AsyncSerdeProcessor  
   https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/serde/async_processor.py

10. FP8 serde  
    https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/serde/fp8.py

11. TurboQuant serde  
    https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/serde/turboquant/turboquant.py

12. TurboQuant Triton store kernel  
    https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/serde/turboquant/store_kernel.py

13. filesystem L2 adapter  
    https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py

14. LMCache build pin at compatibility commit (`torch==2.11.0`)  
    https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/pyproject.toml

---

## Later audit snapshot

```text
f9addd2e4e074f21f26cbda84c3abd932d32ef33
```

它用于确认后续 Serde/TurboQuant evolution。

但该 snapshot 的 build pin 已是 `torch==2.13.0`，不能在我们的 vLLM0.26/torch2.11 环境中未经 ABI rebuild/verification 直接作为 implementation baseline。

---

## CacheGen

15. CacheGen  
    https://github.com/UChi-JCL/CacheGen

用途：

```text
compressed context loading / KV streaming system motivation
```

不是我们项目的直接代码基座。

---

## Explicit reference gap

当前没有把下面一项伪装成已解决：

```text
pinned vLLM0.26 + LMCache
在 first raw D2H 之前
直接 encode compressed KV
```

它属于 V3 source-audit target，不属于 V1/V2 completion gate。


## 46.1 Round-3 新增 Reference Inventory

### LMCache implementation-level references

16. Actual `SerdeL2AdapterWrapper` implementation  
    https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/l2_adapters/serde_wrapper.py

17. Actual FP8 serde implementation / exact estimator behavior  
    https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/serde/fp8.py

18. Filesystem L2 actual write/read / `L2StoreResult.bytes_transferred`  
    https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py

19. LMCache vLLM connector hook surface  
    https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/integration/vllm/lmcache_connector_v1.py

20. LMCache RequestTracker / ReqMeta / slot mapping  
    https://github.com/LMCache/LMCache/blob/a7afadebb9248b62b5c533ce2c12297e9d94fc4a/lmcache/integration/vllm/vllm_v1_adapter.py

### Newer vLLM upstream：V3 offload runtime reference

Audit snapshot:

```text
0ecc284790e5403f74b899524ef82ecb69f83cb3
```

21. Generic offloading connector worker / canonical GPU KV page mapping  
    https://github.com/vllm-project/vllm/blob/0ecc284790e5403f74b899524ef82ecb69f83cb3/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py

22. Generic offloading scheduler / transfer lifecycle  
    https://github.com/vllm-project/vllm/blob/0ecc284790e5403f74b899524ef82ecb69f83cb3/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py

23. CPU offload GPU worker / pinned host registration / stream-event-buffer pools  
    https://github.com/vllm-project/vllm/blob/0ecc284790e5403f74b899524ef82ecb69f83cb3/vllm/v1/kv_offload/cpu/gpu_worker.py

这些是：

```text
V3 architecture/lifetime reference
```

不是：

```text
P2 pinned baseline dependency
```

因此只迁移 engineering primitive，不整体移植 newer vLLM offload subsystem。


# 47. 最终判断

P2 值得做，是因为它和 P1 形成互补。

P1 回答：

```text
GPU 内部
怎样让 KV page 真正释放和复用？
```

P2 回答：

```text
GPU 外部
怎样让可复用 KV 以更小的 L2 serialized representation 存储，
并量化 codec / I/O / data movement 的 crossover？
```

两者共同形成：

```text
          KV Cache Lifecycle
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
   GPU resident        external reuse
        │                 │
        ▼                 ▼
Physical Reclaim    Compressed Offload
        │                 │
        ▼                 ▼
vLLM allocator      LMCache / PCIe
        │                 │
        └────────┬────────┘
                 ▼
       Long-context serving
```

它的最大优点是：

- Core 有成熟 reference；
- 不依赖算法创新；
- V1 很确定；
- V2 可以展示 Triton；
- performance claim 由 profiler 决定，不提前吹；
- 即使 codec 没有在某些 workload 上变快，仍能产出完整 crossover analysis。

因此非常适合作为第二个简历项目。
