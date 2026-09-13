# E8：TRITON_ATTN + INT4 Per-Token-Head KV Cache 三层桥接实验报告

> 目标：完成从 **源码语义 → Runtime 数据路径 → GPU Kernel / Nsight 证据** 的三层桥接，并回答一个核心问题：  
> 在 vLLM 0.26、A100、TRITON_ATTN、Eager 条件下，`int4_per_token_head` KV Cache 到底改变了什么？它带来了什么收益，又带来了什么代价？
>
> 本文定位是 **三层桥接学习收尾文档**，不是 INT4 KV 微架构优化论文。  
> 对 `_attn_packed` 的 Roofline、NCU、fusion、Hadamard 微架构优化等问题，只保留为后续专项入口。

---

## 0. 实验环境与固定条件

### 0.1 软件 / 硬件

- vLLM：`v0.26.0`
- commit：`568afb3a13806beb53bb2e6bd518269357b237c0`
- branch：`study-vllm-0.26`
- GPU：NVIDIA A100 80GB PCIe，SM80
- PyTorch：2.11
- Python：3.10
- ModelRunner：V2
- Engine：V1
- TP / PP / DP：1 / 1 / 1
- 主模型：`/data/models/Qwen3-0.6B`

### 0.2 E8 固定配置

- Attention Backend：`TRITON_ATTN`
- Model dtype：BF16
- `enforce_eager=True`
- CUDA Graph：关闭
- `block_size=16`
- `max_model_len=2048`
- `max_num_batched_tokens=256`
- Prefix Cache：关闭

### 0.3 E7 → E8 唯一核心变量

E7：

```text
TRITON_ATTN
+
BF16 KV Cache
+
Eager
```

E8：

```text
TRITON_ATTN
+
int4_per_token_head KV Cache
+
Eager
```

因此 E8 的目标不是重新研究 Scheduler，而是沿着已经建立好的 Paged KV 主链继续下钻：

```text
SchedulerOutput
    ↓
GPUModelRunner
    ↓
prepare_inputs / prepare_attn
    ↓
slot_mapping / block_table
    ↓
TritonAttentionImpl
    ↓
KV write
+
Paged KV read / Attention
    ↓
GPU kernels
```

---

# 1. E8 的研究问题

E8 实际拆成了五类问题。

## 1.1 表示层

`int4_per_token_head` 到底如何存 K/V？

包括：

- raw cache dtype
- physical layout
- scale 放在哪里
- 每个 token / head 占多少字节
- block / slot 语义是否改变

## 1.2 数据路径层

BF16 和 INT4 在：

```text
KV write
+
KV read / Attention
```

上分别走什么函数和 kernel？

## 1.3 数值层

INT4 KV 是否明显改变输出分布？

要区分：

```text
Free-running
```

和：

```text
Fixed-history
```

否则 autoregressive divergence 会混淆“单步量化误差”。

## 1.4 系统性能层

INT4 带来约 3.76× KV capacity 后：

- Prefill 是否更快？
- Decode 是否更快？
- Context 越长是否会出现 crossover？

## 1.5 三层桥接层

最终要能回答：

```text
源码里的 INT4 路径
为什么在 Runtime 中多出这些操作
又为什么在 Nsight 中表现成这些 GPU kernels
```

---

# 2. E8-A：源码路径与 INT4 表示

## 2.1 dtype 不是 torch.int4

vLLM 中：

```text
int4_per_token_head
```

实际底层 storage dtype 是：

```text
torch.uint8
```

因为一个 byte 中打包两个 INT4：

```text
high nibble = 4 bit
low nibble  = 4 bit
```

所以：

```text
2 × INT4
→
1 byte
```

---

## 2.2 per-token-head 的含义

量化粒度不是：

```text
整个 layer 一个 scale
```

也不是：

```text
整个 head 永远一个 scale
```

而是：

```text
每一个 token
×
每一个 KV head
```

都有自己的量化参数。

这是一种较细粒度的动态 KV quantization。

---

## 2.3 RHT / Rotation

INT4 路径会在量化前对数据做 Randomized Hadamard Transform（RHT）一类的预旋转。

目的不是压缩，而是：

```text
减弱 outlier
↓
让数值分布更均匀
↓
提高 4-bit quantization 的可用性
```

这在后面的 GPU profiler 中会直接表现为：

```text
hadamard_transform_kernel
```

---

## 2.4 scale 与 zero-point

当前模式使用：

- K / V 分别的 per-token-head scale
- 4-bit zero-point 被编码进 scale carrier 的低位信息
- scale 通过 cache storage 的 view / `as_strided` 方式嵌入管理，而不是额外分配一大块独立 scale cache

因此不要把 INT4 KV 理解成：

```text
一个 packed K/V tensor
+
一个完全独立的大 scale tensor
```

实际表示更紧凑。

---

# 3. E8-B：Runtime layout 与容量验证

Qwen3-0.6B 的关键结构：

```text
Q heads  = 16
KV heads = 8
head_dim = 128
GQA      = 2 Q heads / KV head
```

---

## 3.1 BF16

每 token / KV head：

```text
K = 128 × BF16 = 256 B
V = 128 × BF16 = 256 B

K + V = 512 B
```

8 个 KV head：

```text
512 × 8 = 4096 B / token
```

---

## 3.2 INT4

每个 K 或 V：

```text
128 × 4 bit
=
64 B packed payload
```

再加：

```text
4 B scale carrier
```

所以：

```text
K = 68 B
V = 68 B

K + V = 136 B / token / KV head
```

8 个 KV head：

```text
136 × 8
=
1088 B / token
```

---

## 3.3 理论压缩比

```text
4096 / 1088
≈ 3.7647
```

Runtime 实际 capacity：

```text
BF16  = 136,736 tokens
INT4  = 514,768 tokens
```

比例：

```text
514768 / 136736
≈ 3.7647
```

理论与 Runtime 完全吻合。

这是 E8 最干净、最确定的一个结果：

> `int4_per_token_head` 在当前 Qwen3-0.6B / vLLM 配置下，KV capacity 确实提高约 3.765×。

---

# 4. INT4 并没有改变 Paged KV 的逻辑地址语义

这一点非常重要。

INT4 改变的是：

```text
physical bytes / slot
```

但没有改变：

```text
logical block
slot
block_table
slot_mapping
```

的基本含义。

因此三层桥接可以继续沿用：

```text
position
↓
logical block / offset
↓
block_table / slot_mapping
↓
physical block / slot
↓
Attention Backend 解释该 slot 的具体 byte layout
```

也就是说：

```text
slot_mapping = WRITE address
block_table  = READ address map
```

仍然成立。

低比特 KV 是：

> **改变物理表示，不改变 Paged KV 控制面的逻辑寻址模型。**

---

# 5. E8 数据路径：BF16 vs INT4

## 5.1 BF16 write

```text
GPUModelRunner
↓
slot_mapping
↓
TritonAttentionImpl.do_kv_cache_update
↓
triton_reshape_and_cache_flash
↓
reshape_and_cache kernel
↓
Paged KV Cache
```

Runtime 典型 kernel：

```text
reshape_and_cache_kernel_flash
```

---

## 5.2 BF16 read

```text
query
+
block_table
+
BF16 paged KV
↓
TritonAttentionImpl.forward
↓
unified_attention
↓
kernel_unified_attention
+
reduce_segments
```

---

## 5.3 INT4 write

概念链：

```text
当前 K/V
↓
RHT / rotation
↓
dynamic per-token-head quantization
↓
INT4 pack
↓
scale / zero-point handling
↓
写入 packed Paged KV
```

Runtime 主 kernel：

```text
_reshape_cache_int4_kernel
```

但完整写路径周围还存在若干：

```text
Hadamard
cast
scale
elementwise
```

GPU kernel。

---

## 5.4 INT4 read

```text
Q
↓
rotation / transform
↓
packed INT4 KV
+
scale
+
block_table
↓
unified_attention_int4
↓
_attn_packed
↓
reduce_segments
↓
output transform
```

关键边界：

> 当前实现不是“先把整个历史 KV 解压成 BF16，然后再调用普通 BF16 Attention”。

而是专门的 packed INT4 Attention 路径。

---

# 6. E8-C：数值影响

E8 分成两个实验。

---

## 6.1 C1：Free-running

配置：

- 4 个 prompt
- 长度约：12 / 42 / 175 / 529
- greedy
- 每个生成 16 token
- 比较 BF16 vs INT4 自回归轨迹

结果：

```text
64 个生成位置
trajectory agreement ≈ 0.3125
mean top-k Jaccard ≈ 0.1568
mean common logprob delta ≈ 1.713
max delta ≈ 9.30
```

其中：

- 某些 case step0 就在 near-tie 上分叉
- 某些 case 前几步相同后再分叉
- 长 prompt 529-token case 甚至 16 个 token 全部一致

### C1 的正确解释

不能简单说：

```text
INT4 accuracy = 31.25%
```

这是错误的。

因为一旦：

```text
step0 token 不同
```

后续输入 history 就不同：

```text
BF16:
A B C X ...

INT4:
A B C Y ...
```

从 step1 开始已经不是相同输入条件。

因此 C1 测的是：

> **用户最终观察到的 free-running 行为差异。**

不是纯量化数值误差。

---

# 7. E8-C2：Fixed-history

为了隔离纯量化误差，固定完全相同的 input token history，只比较“下一 token 分布”。

测试长度：

```text
15
16
17
64
128
255
256
257
512
```

结果：

```text
top1 agreement = 7 / 9
mean top5 Jaccard ≈ 0.459
mean top20 Jaccard ≈ 0.422
mean common-top20 logprob delta ≈ 1.474
max delta ≈ 10.817
nonfinite = 0
```

### 说明

这证明：

> 即使输入 history 完全一致，INT4 KV 在部分 context 下也会造成明显的 next-token distribution distortion。

因此 C1 中的差异不能全部归因于“首次分叉后的 autoregressive amplification”。

---

## 7.1 但没有观察到明显 block-boundary bug

特别测试：

```text
15 / 16 / 17
255 / 256 / 257
```

是为了观察：

```text
block boundary
```

附近是否出现系统性异常。

当前没有看到：

```text
16
256
```

边界突然崩坏的模式。

所以目前没有证据支持：

```text
INT4 的错误来自 block_table / slot_mapping / block boundary 寻址 bug
```

---

# 8. 数值实验的正确分层

以后低比特 KV 实验建议一直保留五层：

```text
L1 Representation Error
K/V quantization error

L2 Distribution Error
logits / probability / ranking

L3 Decoding Behavior
free-running token trajectory

L4 Model / Task Quality
PPL / LongBench / task accuracy

L5 System
memory / latency / throughput
```

E8 当前主要覆盖：

```text
L2
L3
L5
```

还没有完成完整 task-quality 评估。

---

# 9. E8-E：Clean GPU 性能

条件：

```text
A100 idle
Qwen3-0.6B
concurrency = 4
TRITON_ATTN
eager
```

Workload：

### Prefill

```text
4 × 512 input
output = 1
```

### Decode

```text
4 × 32 input
output = 128
```

采用：

```text
warmup + 3 reps
取 median
```

---

## 9.1 结果

### BF16

```text
Prefill elapsed
= 0.202518 s

Prefill input throughput
= 10112.705 tok/s

Decode elapsed
= 3.276275 s

Decode output throughput
= 156.275 tok/s
```

### INT4

```text
Prefill elapsed
= 0.318361 s

Prefill input throughput
= 6432.951 tok/s

Decode elapsed
= 5.237719 s

Decode output throughput
= 97.752 tok/s
```

---

## 9.2 比例

```text
Prefill:
INT4 / BF16
≈ 0.636×

Decode:
INT4 / BF16
≈ 0.626×
```

即：

```text
Prefill throughput ≈ -36.4%
Decode throughput  ≈ -37.4%
```

同时：

```text
KV capacity
≈ +3.765×
```

---

## 9.3 E8-E 结论

在当前受控条件下：

> INT4 获得了约 3.765× KV capacity，但 Prefill 和 Decode throughput 均明显低于 BF16。

必须保留范围边界：

```text
A100
Qwen3-0.6B
TP1
concurrency4
TRITON_ATTN
Eager
vLLM0.26
```

不能推广成：

```text
INT4 KV 天生更慢
```

---

# 10. E8-F：Decode Context Scaling

E8-E 之后出现一个合理问题：

> INT4 虽然在短 context 慢，但随着 context 增长，KV read 越来越重，INT4 减少 HBM traffic 的收益会不会最终超过 quant/dequant overhead？

于是 E8-F 做 context sweep。

---

## 10.1 模型

- Qwen3-0.6B
- Qwen3-8B
- Qwen3-32B
- Llama3-8B

Context：

```text
128
256
512
1024
1536
1920
```

输出：

```text
64 tokens
```

Concurrency：

```text
4
```

---

# 11. E8-F 主要结果

所有模型在 128–1920 内：

> **没有观察到 INT4 / BF16 decode crossover。**

INT4 TPOT 大致保持为 BF16 的：

```text
~1.45× – 1.60×
```

Qwen3-32B 有轻微 relative narrowing，但没有变成 INT4 更快。

---

## 11.1 Qwen3-0.6B 示例

```text
ctx128:
BF16 TPOT 25.88 ms
INT4 TPOT 39.29 ms

ctx1920:
BF16 TPOT 25.31 ms
INT4 TPOT 39.03 ms
```

非常平。

---

## 11.2 Qwen3-32B 示例

```text
ctx128:
BF16 TPOT ≈ 58.70 ms
INT4 TPOT ≈ 89.74 ms

ctx1920:
BF16 TPOT ≈ 65.45 ms
INT4 TPOT ≈ 96.81 ms
```

INT4/BF16 ratio 有轻微缩小，但绝对 TPOT gap 仍约 31ms。

所以：

> 目前没有证据说明在 ≤2K context 上，INT4 的 HBM 节省已经转换成实际 latency 优势。

---

# 12. E8-F 的方法学修正

原脚本中的：

```text
batch_decode_tps
```

不能解释成纯 steady-state Decode throughput。

原因：

```text
concurrency = 4
max_num_batched_tokens = 256
long prompt
```

会造成：

```text
request A 先完成某个 prefill chunk
↓
A 开始 decode

与此同时
B/C/D 仍可能 prefill
```

因此：

```text
first_token_ts → last_token_ts
```

中混有：

```text
decode
+
other request prefill
+
request-start staggering
```

所以 E8-F 中：

- `median_tpot_ms` 更有价值
- `batch_decode_tps` 只能视为 mixed effective throughput
- 不能用它做“纯 KV bandwidth scaling”结论

这也是后面 E8-G 改成：

```text
concurrency = 1
+
Decode-only NVTX
```

的原因。

---

# 13. E8-G：Nsight 三层桥接

## 13.1 问题

E8-E/F 已经知道：

```text
INT4 更慢
```

E8-G 不再问：

```text
“慢不慢？”
```

而是：

```text
“慢在哪里？”
```

---

# 14. 为什么不能继续 profile 整个 `generate()`

Whole-generate 包含：

```text
Prefill
+
Decode
+
Sampling
+
Scheduler
```

所以需要把：

```text
纯 Decode Model Forward
```

单独标出来。

---

# 15. E8-G Instrumentation

在：

```text
GPUModelRunner.execute_model()
```

中，当前 eager path 是：

```text
self.model(**model_inputs)
```

我们增加实验性 NVTX：

```text
E8G_DECODE_FORWARD
```

只在受控条件满足时触发：

```text
concurrency = 1
speculative decode = off
num_reqs = 1
scheduler 本轮该请求 scheduled token = 1
```

因此：

```text
Prefill:
scheduled tokens > 1

Decode:
scheduled tokens = 1
```

在这个受控 microbenchmark 中可作为 Decode classifier。

---

# 16. 为什么 NVTX 只包 model forward

我们要研究的是：

```text
Attention / KV GPU 数据面
```

所以只包：

```text
self.model(...)
```

而不是整个：

```text
execute_model()
```

从而排除：

- prepare_inputs
- prepare_attn
- sampling
- scheduler

---

# 17. 为什么不加 `torch.cuda.synchronize()`

因为 CUDA 本来就是异步执行。

如果为了让 GPU kernel 落在 NVTX wall range 内而同步：

```text
launch
↓
synchronize
↓
launch
```

会改变真实 runtime。

因此 E8-G 使用：

```text
NVTX host range
↓
CUDA Runtime launch
↓
correlationId
↓
GPU kernel
```

做关联。

不是简单：

```text
kernel.start BETWEEN nvtx.start AND nvtx.end
```

---

# 18. E8-G Runtime 归因链

SQLite：

```text
NVTX_EVENTS
↓
CUPTI_ACTIVITY_KIND_RUNTIME
    start / end
    globalTid
    correlationId
↓
CUPTI_ACTIVITY_KIND_KERNEL
    correlationId
↓
kernel
```

这条链使我们能够回答：

> 某一个 Decode `self.model()` 到底提交了哪些 CUDA kernels。

---

# 19. BF16 Decode：结构结果

以 Qwen3-0.6B 为例：

```text
Decode steps = 15
```

每 step：

```text
kernel_count      = 367
attention calls   = 28
reduce calls      = 28
KV write calls    = 28
```

Qwen3-0.6B 恰好：

```text
28 Transformer layers
```

因此形成非常漂亮的闭环：

```text
Layer 0
→ 1 × Attention
→ 1 × KV write
→ 1 × reduce

...

Layer 27
→ 1 × Attention
→ 1 × KV write
→ 1 × reduce
```

说明 correlation-based attribution 是稳定的。

---

# 20. BF16 Context Scaling

当前 profiler 观察到：

```text
ctx ↑
```

时：

```text
kernel_count
不变

KV write
基本不变

reduce
基本不变

Attention duration
明显增加
```

这是 Decode 数据面的基本规律：

```text
KV write
=
只写当前新 token
≈ O(1)

Historical KV read / Attention
=
读取已有 context
≈ 随 context 增长
```

因此长 context 的 Decode 增量主要落在：

```text
Attention historical KV read
```

而不是：

```text
KV Cache write
```

---

# 21. INT4 Decode：最关键的结构发现

INT4 ctx128 / ctx1920 都观察到：

```text
kernel_count = 1179 / decode step
```

而 BF16：

```text
kernel_count = 367 / decode step
```

差：

```text
1179 - 367
=
812 extra kernels / token
```

---

# 22. 这些额外 kernel 是什么

INT4 中最突出的额外 kernel：

```text
unrolled_elementwise_kernel
3375 / 15
≈ 225 / step

vectorized_elementwise_kernel
5460 / 15
= 364 / step

elementwise_kernel
1680 / 15
= 112 / step

hadamard_transform_kernel
1680 / 15
= 112 / step
```

合计：

```text
225 + 364 + 112 + 112
=
813 / step
```

和：

```text
INT4 - BF16
≈ 812 extra kernels
```

几乎完全对齐。

---

# 23. 这四类 kernel 应该如何理解

## 23.1 `hadamard_transform_kernel`

它具有明确算法语义：

```text
RHT / Hadamard rotation
```

用于 INT4 量化前后的旋转/反旋转。

Qwen3-0.6B：

```text
112 / 28
=
4 / layer
```

当前路径可以概念对应为：

```text
RHT(K)
RHT(V)
RHT(Q)
inverse RHT(output)
```

所以：

```text
4 × 28
=
112
```

与 profiler 完全对齐。

---

## 23.2 `elementwise / vectorized / unrolled_elementwise`

这些不是某个具体“INT4 算法名称”。

它们更接近 PyTorch / CUDA 通用逐元素执行模板，可能来自：

- dtype cast
- scale
- sign
- multiply
- copy / contiguous
- conversion
- quantization preparation

当前可以确定的是：

> 它们高度绑定 INT4 per-layer 路径，是 INT4 相比 BF16 新增的大量 runtime work。

但三层桥接阶段不再继续把每一个 generic kernel 精确归因到每一行 Python/Torch 源码。

---

# 24. INT4 Context Scaling

INT4：

```text
ctx128
→
ctx1920
```

时：

```text
kernel_count
1179 → 1179

Hadamard / elementwise
基本固定

KV write
基本固定

_attn_packed
明显增长
```

因此可以把 INT4 Decode 成本拆为：

```text
固定 per-token / per-layer overhead
│
├─ rotation
├─ elementwise
├─ quant / scale / pack
└─ cache write

+

context-dependent overhead
│
└─ packed historical KV Attention
```

---

# 25. E8-G 当前 timing 的证据边界

E8-G 后半段 profile 时 GPU2 仍存在其它进程占用。

因此当前：

```text
kernel call count
kernel identity
每层重复结构
ctx 增长时调用次数是否变化
```

这些结构事实可信度很高。

但：

```text
20.2 us
35.8 us
4.99 ms
45.5 ms
1.77×
```

这类绝对 timing 应标为：

```text
provisional / contaminated
```

不作为最终 benchmark。

---

# 26. 但结构结论已经足够完成三层桥接

E8-G 已经完成：

```text
源码：
INT4 RHT / quant / pack / packed attention

↓ Runtime

每层额外 transform
独立 INT4 cache write
独立 packed Attention

↓ GPU profiler

hadamard_transform_kernel
大量 elementwise kernels
_reshape_cache_int4_kernel
_attn_packed
```

这已经达到本阶段学习目标。

---

# 27. E8 总结：三个最重要结论

## 27.1 低比特 KV 首先是“物理表示变化”

Paged KV 的：

```text
block / slot
block_table
slot_mapping
```

逻辑寻址不变。

改变的是：

```text
slot 内部 byte layout
+
Attention Backend 如何解释它
```

---

## 27.2 INT4 的收益非常明确

```text
KV capacity
≈ 3.765×
```

这是实打实的系统收益。

---

## 27.3 但低比特不是免费收益

当前 `TRITON_ATTN + int4_per_token_head` 会引入：

```text
rotation
quantization
packing
scale handling
generic elementwise transforms
packed Attention
```

因此：

```text
memory footprint ↓
```

并不自动推出：

```text
latency ↓
```

在 E8 的 clean 性能实验中，INT4 反而比 BF16 慢约 36–37%。

---

# 28. E8 对“三层桥接学习”的最终价值

现在已经可以从配置：

```python
kv_cache_dtype="int4_per_token_head"
```

一路解释到：

```text
CacheConfig
↓
KV physical layout
↓
GPUModelRunner
↓
slot_mapping / block_table
↓
TritonAttentionImpl
↓
INT4 write/read
↓
GPU kernels
↓
Nsight timeline
```

这就是本阶段真正需要建立的能力：

> 不再把 `kv_cache_dtype` 当成一个黑盒参数，而是知道它如何改变数据表示、执行路径以及最终硬件行为。

---

# 29. 暂不继续深挖的问题

以下问题保留给后续 Quantized KV 专项，不作为 E8 三层桥接的阻塞项：

- `_attn_packed` 是 memory-bound 还是 compute-bound？
- INT4 DRAM bytes / L2 / SM throughput 到底是多少？
- Hadamard 是否应该 fuse？
- 812 个 extra kernel 有多少可以消除？
- CUDA Graph 能否显著减少 INT4 eager host launch overhead？
- 是否应该避免 online rotation？
- 是否需要重新设计 mixed-precision paged KV format？
- Recent BF16 + History INT4 的最优切换策略是什么？

---

# 30. E8 最终结论

> 在 vLLM 0.26、A100、V2 ModelRunner、TRITON_ATTN、Eager 路径下，`int4_per_token_head` 将 KV Cache 的物理存储压缩到 BF16 的约 26.6%，使可用 KV token capacity 提升约 3.765×，同时保持 Paged KV 的 block/slot/block_table/slot_mapping 逻辑不变。  
> 但该表示会改变 Attention Backend 的真实数据路径：BF16 的普通 cache-write + unified attention 被替换为包含 RHT、动态 per-token-head quantization、INT4 packing、专用 cache write 和 packed Attention 的路径。数值实验表明该 INT4 路径在部分 context 下会造成明显 next-token distribution distortion；系统实验则表明，在当前实现与 workload 中，INT4 的容量收益没有转化成 latency/throughput 收益。Nsight 三层桥接进一步验证了：INT4 Decode 每 step 会引入大量额外 transform / elementwise / Hadamard kernels，并通过 `_attn_packed` 读取历史 packed KV。Context 增长时，主要增长项仍然是 historical KV Attention，而不是 KV write。  
> 到此，E8 已完成“源码 → Runtime → GPU kernel”的三层桥接目标；更深的 INT4 kernel / Roofline / NCU 优化问题转入后续 Quantized KV 专项。
