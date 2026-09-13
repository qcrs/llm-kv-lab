# E7 实验报告：显式 `TRITON_ATTN + BF16` 数据面验证

> 项目：vLLM 三周定向桥接学习  
> 阶段：E7  
> 版本：vLLM v0.26.0  
> Commit：`568afb3a13806beb53bb2e6bd518269357b237c0`  
> Branch：`study-vllm-0.26`  
> GPU：NVIDIA A100 80GB PCIe，SM80  
> Model：`/data/models/Qwen3-0.6B`  
> Model Runner：V2  
> Engine：V1  
> Execution Mode：`enforce_eager=True`  
> Prefix Cache：Disabled  
> TP/PP/DP：1/1/1  
> Block Size：16  
> `max_model_len=2048`  
> `max_num_batched_tokens=256`  
> `max_num_seqs=4`

---

# 1. 实验背景

E6 已经完成默认 Attention Backend 的数据面定位，当前环境实际选择：

```text
Default selector
    ↓
FLASH_ATTN
    ↓
FlashAttentionImpl
    ↓
FlashAttention v2
    ↓
native CUDA extension
```

E6 同时已经证明：

```text
slot_mapping
=
当前新 token 的 KV WRITE 地址

block_table
=
历史 KV 的 Paged READ 地址表
```

并动态验证过一个具体地址：

```text
position=32
logical_block=2
physical_block=3
offset=0

expected_slot=48
actual_slot=48
match=True
```

E6.5 / E6.6 又完成了 Nsight Profiling Bridge 和 eager / CUDA Graph A/B，因此 E7 不再继续混入 CUDA Graph、Profiler 或调度变量，而是只替换 Attention Backend。

E7 的目标不是“看看 Triton 能不能启动”，而是完成一个受控的 backend substitution：

```text
E6:
FLASH_ATTN + BF16 KV + eager

          ↓ 唯一主要变量

E7:
TRITON_ATTN + BF16 KV + eager
```

需要验证：

1. 当前 commit 是否真的支持显式 `TRITON_ATTN`；
2. A100 / SM80 + BF16 + block_size=16 + head_size=128 是否满足 backend 约束；
3. runtime 是否真正使用 `TritonAttentionImpl`，而非 fallback 到 FlashAttention；
4. KV Cache 是否仍然为 BF16；
5. `slot_mapping` 的 WRITE 地址语义是否保持不变；
6. `block_table` 的 READ 地址语义是否保持不变；
7. Triton 后端实际走哪些 KV write / Attention read kernel；
8. FA2 与 Triton 在相同 BF16 条件下能否得到一致的生成结果。

---

# 2. 实验假设

在实验前提出以下预测。

## 2.1 控制面应保持不变

更换 Attention Backend 不应该改变：

```text
Scheduler
    ↓
KVCacheManager
    ↓
physical block allocation
    ↓
SchedulerOutput
    ↓
ModelRunner
    ↓
block_table / slot_mapping
```

因此：

```text
slot_mapping[token]
=
physical_block * block_size + offset
```

以及：

```text
block_table[logical_block]
=
physical_block
```

应当继续成立。

## 2.2 数据面应发生替换

E6 的 Attention 数据面：

```text
FlashAttentionImpl
    ↓
FA2 native CUDA
```

E7 预期变为：

```text
TritonAttentionImpl
    ↓
vLLM Triton kernels
```

其中 WRITE 与 READ 应分别落到：

```text
WRITE:
TritonAttentionImpl.do_kv_cache_update()
    ↓
triton_reshape_and_cache_flash()
    ↓
reshape_and_cache_kernel_flash
```

```text
READ:
TritonAttentionImpl.forward()
    ↓
unified_attention()
    ↓
kernel_unified_attention
```

---

# 3. 实验结构

E7 分成三个阶段：

```text
E7-A
Source / Config Preflight
    ↓

E7-B
Runtime Data Path
    ↓

E7-C
FA2 vs Triton Correctness A/B
```

最终 Gate：

```text
E7-A PASS
E7-B PASS
E7-C PASS
    ↓
E7 FINAL PASS
```

---

# 4. E7-A：Source / Config Preflight

## 4.1 目的

E7-A 不加载模型、不执行 inference，只回答：

> 当前 active commit 的源码和配置系统是否允许我们在 A100 上显式使用 `TRITON_ATTN + BF16`？

这一步避免直接运行后再猜：

- backend 名字是否正确；
- Python API 是否支持；
- A100 是否满足限制；
- BF16 KV 是否支持；
- block_size/head_size 是否兼容；
- TRITON_ATTN 对应的是哪个 backend class；
- WRITE / READ 代码实际落在哪里。

---

## 4.2 环境确认

实际结果：

```text
repo=/home/qcrs/learning/llm-kv-lab/third_party/vllm

python=
/home/qcrs/learning/llm-kv-lab/
.venvs/vllm-v026-torch211-cu129-py310/bin/python

python_version=Python 3.10.20

branch=study-vllm-0.26

commit=
568afb3a13806beb53bb2e6bd518269357b237c0

commit_match=True
```

GPU capability：

```text
platform: cuda
device capability:
DeviceCapability(major=8, minor=0)
```

即：

```text
A100
=
SM80
```

因此实验环境与固定基线一致。

---

# 5. Attention Backend 配置链

源码与 Python introspection 共同确认：

```text
AttentionConfig
    │
    │ backend="TRITON_ATTN"
    ▼
validate_backend_before()
    ↓
AttentionBackendEnum.TRITON_ATTN
    ↓
selector.get_attn_backend()
    ↓
_cached_get_attn_backend()
    ↓
current_platform.get_attn_backend_cls()
    ↓
TritonAttentionBackend
    ↓
TritonAttentionImpl
```

实际 Python 输出：

```text
AttentionConfig(string).backend:
AttentionBackendEnum.TRITON_ATTN

AttentionConfig(enum).backend:
AttentionBackendEnum.TRITON_ATTN
```

Registry：

```text
TRITON_ATTN
=
vllm.v1.attention.backends.triton_attn.TritonAttentionBackend
```

Backend identity：

```text
backend class:
TritonAttentionBackend

backend name:
TRITON_ATTN

impl class:
TritonAttentionImpl
```

因此 `TRITON_ATTN` 不是一个模糊字符串，而是一条明确的配置映射链。

---

# 6. Backend 静态合法性检查

Triton backend 声明：

```text
supported model dtypes:
torch.float16
torch.bfloat16
torch.float32
```

KV Cache dtype：

```text
auto
float16
bfloat16
fp8
fp8_e4m3
fp8_e5m2
int4_per_token_head
int8_per_token_head
fp8_per_token_head
```

当前配置：

```text
model dtype     = bfloat16
KV dtype        = auto / bfloat16
block_size      = 16
head_size       = 128
GPU             = SM80
```

实际检查：

```text
supports block_size=16:
True

supports head_size=128:
True
```

Generic validation：

```text
validate kv_cache_dtype=auto:
[]

validate kv_cache_dtype=bfloat16:
[]
```

其中：

```text
[]
=
没有 invalid reason
```

因此 E7-A 可以得出：

> 在当前 commit 的 backend contract 层面，`TRITON_ATTN + BF16 + block_size=16 + head_size=128 + A100/SM80` 是合法配置。

需要注意：

这只是“配置兼容”，不是“runtime 已经成功执行”。

---

# 7. TRITON_ATTN 的源码数据面

## 7.1 KV WRITE

Triton backend：

```text
forward_includes_kv_cache_update = False
```

说明 KV 更新仍是独立阶段。

源码链：

```text
unified_kv_cache_update()
    ↓
TritonAttentionImpl.do_kv_cache_update()
    ↓
triton_reshape_and_cache_flash()
    ↓
@triton.jit
reshape_and_cache_kernel_flash
```

Kernel 内：

```text
slot_idx = slot_mapping[token_idx]

block_idx =
slot_idx // block_size

block_offset =
slot_idx % block_size
```

因此源码直接证明：

```text
slot_mapping
    ↓
global physical slot
    ↓
physical block + offset
    ↓
KV WRITE address
```

这里函数名虽然包含：

```text
flash
```

但该实现是：

```text
@triton.jit
```

因此仍然属于 Triton implementation，并非 fallback 到 `FLASH_ATTN` backend。

---

## 7.2 Paged KV READ

源码链：

```text
TritonAttentionImpl.forward()
    ↓
unified_attention()
    ↓
kernel_unified_attention
```

Triton kernel 内部：

```text
physical_block_idx =
block_table[
    seq_offset // BLOCK_SIZE
]
```

随后构造 K/V 地址：

```text
physical_block
+
offset_in_block
+
kv_head
+
head_dim
```

因此源码层面确认：

```text
block_table
=
logical KV block
→ physical KV block
```

---

# 8. E7-A 中的 stride-order 报错

E7-A 直接调用：

```text
get_kv_cache_stride_order()
```

出现：

```text
AssertionError:
Current vLLM config is not set
```

这不是 Triton backend 不支持 layout。

原因是：

```text
get_kv_cache_stride_order()
    ↓
get_kv_cache_layout()
    ↓
依赖 current VllmConfig
```

而 E7-A 没有初始化完整 LLM runtime，也没有进入：

```text
set_current_vllm_config(...)
```

上下文。

因此不为此修改测试代码，而是在 E7-B 中直接观察真实：

```text
kv_shape
kv_stride
```

这是更强的 runtime 证据。

---

# 9. E7-A 结论

```text
Config API                          PASS
Enum → Backend                     PASS
TritonAttentionBackend             PASS
TritonAttentionImpl                PASS
A100 / SM80 validation             PASS
BF16 validation                    PASS
block_size=16                      PASS
head_size=128                      PASS
WRITE source path                  PASS
READ source path                   PASS
slot_mapping WRITE consumer        PASS
block_table READ consumer          PASS
```

结论：

```text
E7-A PASS
```

---

# 10. E7-B：Runtime Data Path

## 10.1 目的

E7-B 不再修改源码，直接复用 E6 已经加入的 backend-generic instrumentation：

```text
[BRIDGE][ADDR][BLOCK_TABLE]
[BRIDGE][ADDR][MAPPING]

[BRIDGE][ATTN][KV_WRITE]
[BRIDGE][ATTN][KV_READ]
```

目标：

> 证明 formal request 真正运行在 `TritonAttentionImpl` 上，并动态验证 KV dtype/layout、slot_mapping WRITE 和 block_table READ。

---

# 11. Runtime backend selection

运行参数：

```text
dtype = bfloat16
kv_cache_dtype = auto
backend = TRITON_ATTN
block_size = 16
enable_prefix_caching = False
enforce_eager = True
```

EngineCore 实际打印：

```text
Using AttentionBackendEnum.TRITON_ATTN backend.
```

同时：

```text
[BRIDGE][ATTN][KV_WRITE]
impl=TritonAttentionImpl
```

以及：

```text
[BRIDGE][ATTN][KV_READ]
impl=TritonAttentionImpl
```

因此：

```text
AttentionConfig(TRITON_ATTN)
    ↓
TritonAttentionBackend
    ↓
TritonAttentionImpl
```

已在 runtime 闭环。

没有出现：

```text
FlashAttentionImpl
```

作为 formal request 的 Attention impl，因此不存在 silent fallback。

---

# 12. Attention backend 并不是整个 vLLM backend

同一次运行同时出现：

```text
Using TRITON_ATTN backend
```

以及：

```text
Using FlashInfer for top-p & top-k sampling.
```

Engine config 中还存在：

```text
linear_backend='auto'
moe_backend='auto'
```

Norm：

```text
rms_norm=['vllm_c', 'native']
```

因此 runtime 直接证明：

```text
Model execution
│
├─ Norm
│   └─ vllm_c / native
│
├─ Linear
│   └─ 自己选择 backend
│
├─ Attention
│   └─ TRITON_ATTN
│
├─ MLP
│   └─ 自己的 kernel/backend
│
└─ Sampling
    └─ FlashInfer
```

`AttentionConfig.backend` 只是 Attention 子系统的 backend 选择，不是整个 vLLM 的全局 GPU backend 开关。

---

# 13. Dummy / Warmup / Formal request 区分

运行时分成三类请求。

## 13.1 Dummy

```text
_dummy_req_
```

出现：

```text
kv_shape=(0,)
```

属于 profile / init 阶段。

不作为正式实验数据。

---

## 13.2 Warmup

```text
_warmup_0_
...
_warmup_3_
```

已经使用真实 KV Cache：

```text
kv_shape=(8546,8,16,256)
```

并使用：

```text
TritonAttentionImpl
```

---

## 13.3 Formal

通过：

```text
[E7-B][FORMAL_BEGIN]
```

显式划定正式请求。

Formal prompt：

```text
prompt_tokens = 47
```

后续所有关键数据面分析都以这段请求为证据。

---

# 14. BF16 KV runtime 验证

正式请求：

```text
kv_shape=
(8546, 8, 16, 256)

kv_dtype=
torch.bfloat16
```

因此：

```text
kv_cache_dtype=auto
    ↓
runtime
    ↓
torch.bfloat16
```

E6 → E7 没有引入 KV dtype 变化。

---

# 15. KV Cache shape 含义

真实 shape：

```text
(8546, 8, 16, 256)
```

解释：

```text
8546
=
physical KV blocks

8
=
num_kv_heads

16
=
block_size / token slots per block

256
=
2 * head_size
=
K 128 + V 128
```

逻辑可写成：

```text
[B, H, N, 2D]
```

其中：

```text
B = block
H = KV head
N = slot/token inside block
D = head_dim
```

---

# 16. 为什么以前学习 block/slot 时没有讲 head

Paged KV 的 block allocator 管的是：

```text
token capacity
```

不是：

```text
head capacity
```

一个 slot 不是一个 scalar，而是：

```text
一个 token 的完整 KV payload
```

本模型一个 slot 内逻辑包含：

```text
K:
8 heads × 128

V:
8 heads × 128
```

因此以前的：

```text
physical block
    ↓
slot
```

完整展开是：

```text
physical block
    ↓
slot / token
    ↓
KV head
    ↓
head dimension
```

Scheduler / KVCacheManager 只需要管理前两层。

Attention Backend 才需要处理：

```text
head
head_dim
physical tensor layout
```

---

# 17. NHD / HND

定义：

```text
N = token / slot
H = KV head
D = head dimension
```

加上 physical block：

```text
NHD:
[B, N, H, D]

HND:
[B, H, N, D]
```

区别：

```text
NHD
=
Block
→ Token
→ Head
→ Dim
```

适合从“一个 token 的所有 heads”视角理解。

而：

```text
HND
=
Block
→ Head
→ Token
→ Dim
```

适合从“一个 head 的历史 tokens”视角理解。

---

# 18. Runtime physical layout

真实：

```text
kv_shape=
[B,H,N,C]

kv_stride=
[32768,256,2048,1]
```

如果 `[B,H,N,C]` 是普通 contiguous，应该是：

```text
[32768,4096,256,1]
```

但真实不是。

现在：

```text
H stride = 256
N stride = 2048 = 8 * 256
```

这更符合 physical：

```text
[B,N,H,C]
```

即 NHD。

可以理解为：

```text
logical view:
[B,H,N,C]

physical order:
[B,N,H,C]
```

---

# 19. `transpose(1,2)` 的意义

Triton backend 中：

```text
kv_cache.transpose(1,2)
```

只是交换第 1、2 维：

```text
[B,H,N,C]
    ↓
[B,N,H,C]
```

通常不会复制底层显存，而只是改变 Tensor 的：

```text
shape
stride
```

解释方式。

执行后：

```text
shape:
[B,N,H,C]

stride:
[32768,2048,256,1]
```

此时 Tensor view 与实际 NHD physical order 对齐。

因此：

> `transpose()` 不是重新排布 KV Cache，而是在不搬数据的情况下，把 Tensor 的逻辑视图切换为 backend/kernel 更自然的维度顺序。

---

# 20. Formal Prefill 的 block allocation

Formal prompt：

```text
47 tokens
```

block_size：

```text
16
```

因此：

```text
ceil(47 / 16)
=
3 blocks
```

Scheduler 实际：

```text
block_table=
[1,2,3]
```

对应：

```text
position 0~15
→ logical block 0
→ physical block 1

position 16~31
→ logical block 1
→ physical block 2

position 32~46
→ logical block 2
→ physical block 3
```

---

# 21. slot_mapping runtime 验证

所有正式 Prefill token 都满足：

```text
expected_slot
=
actual_slot

match=True
```

例如：

```text
position=32
logical_block=2
physical_block=3
offset=0

expected_slot=
3 * 16 + 0
=
48

actual_slot=
48
```

因此：

```text
slot_mapping
=
当前 token 的 physical WRITE slot
```

在 Triton backend 下仍保持与 E6 完全相同的语义。

---

# 22. 跨 physical block boundary 的动态验证

47-token Prefill 后：

```text
position 32~46
```

已经占用：

```text
physical block 3
offset 0~14
```

下一 Decode：

```text
position=47
logical_block=2
physical_block=3
offset=15

expected_slot=63
actual_slot=63
```

正好填满 block 3。

再下一位置：

```text
position=48
```

Scheduler：

```text
new_block_ids=([4],)
```

block table：

```text
[1,2,3,4]
```

地址：

```text
position=48
logical_block=3
physical_block=4
offset=0

expected_slot=64
actual_slot=64
```

这条证据把：

```text
Scheduler block allocation
    ↓
block_table extension
    ↓
slot_mapping
    ↓
ModelRunner
    ↓
Attention KV WRITE
```

完整串起来。

---

# 23. Triton KV WRITE runtime

Formal Prefill：

```text
[BRIDGE][ATTN][KV_WRITE]

impl=
TritonAttentionImpl

key_shape=
(47,8,128)

value_shape=
(47,8,128)

kv_shape=
(8546,8,16,256)

kv_stride=
(32768,256,2048,1)

kv_dtype=
torch.bfloat16

slots=
[16 ... 62]
```

结合 E7-A source：

```text
TritonAttentionImpl.do_kv_cache_update()
    ↓
triton_reshape_and_cache_flash()
    ↓
reshape_and_cache_kernel_flash
```

因此 WRITE path 已闭环。

---

# 24. Triton Paged KV READ runtime

Formal Prefill：

```text
[BRIDGE][ATTN][KV_READ]

impl=
TritonAttentionImpl

query_shape=
(47,16,128)

block_table0=
[1,2,3,...]

seq_lens=
[47]

query_start_loc=
[0,47]

max_query_len=
47

max_seq_len=
47
```

随后 runtime 直接出现：

```text
Triton kernel JIT compilation during inference:
kernel_unified_attention
```

因此不仅知道：

```text
impl=TritonAttentionImpl
```

还得到真实 kernel-level 证据：

```text
TritonAttentionImpl.forward()
    ↓
unified_attention()
    ↓
kernel_unified_attention
    ↓
Triton JIT
    ↓
GPU
```

---

# 25. Decode 中的 `reduce_segments`

Decode：

```text
query_shape=(1,16,128)
seq_lens=[48]
```

runtime 又出现：

```text
Triton kernel JIT compilation during inference:
reduce_segments
```

这说明 Triton unified attention 在该 shape/config 下进入了 segmented / 3D execution path。

概念上：

```text
Attention history
    ↓
分成多个 segment
    ↓
每个 segment 计算：
partial output
max
exp_sum
    ↓
reduce_segments
    ↓
合并最终 Attention output
```

它不是 backend fallback，而是 `TRITON_ATTN` 内部的 kernel strategy。

---

# 26. Triton JIT Warning 的正确解释

日志：

```text
Triton kernel JIT compilation during inference:
kernel_unified_attention
```

以及：

```text
reduce_segments
```

说明 warmup 没有覆盖 formal workload 的全部 shape/config。

因此首次运行时：

```text
compile latency
```

混入 inference latency。

结论：

```text
E7-B 的 throughput
不能与 E6 FA2 throughput 比较
```

本轮只验证：

```text
runtime dispatch
correctness
data path
```

不做性能结论。

---

# 27. E7-B 结论

```text
显式 TRITON_ATTN config             PASS
runtime backend selection           PASS
TritonAttentionImpl                 PASS
无 silent fallback                  PASS
BF16 KV                             PASS
真实 KV allocation                  PASS
KV shape                            PASS
KV stride                           PASS
NHD physical layout                 PASS
slot_mapping semantics              PASS
block boundary allocation           PASS
block_table semantics               PASS
kernel_unified_attention runtime     PASS
reduce_segments runtime             PASS
formal inference completion          PASS
```

因此：

```text
E7-B PASS
```

---

# 28. E7-C：FA2 vs Triton Correctness A/B

## 28.1 目的

E7-A/B 已经证明：

```text
TRITON_ATTN
可以正确初始化
并且真实走 Triton kernel
```

但“能跑”还不等于：

```text
与 baseline correctness 一致
```

因此 E7-C 保持其他条件不变，仅替换：

```text
FLASH_ATTN
vs
TRITON_ATTN
```

使用 greedy decoding 比较生成 token IDs。

---

# 29. E7-C 控制变量

共同配置：

```text
Model:
Qwen3-0.6B

Model dtype:
BF16

KV cache dtype:
auto → BF16

block_size:
16

enforce_eager:
True

prefix caching:
False

Sampling:
temperature=0

max_tokens:
16
```

唯一主要变量：

```text
FLASH_ATTN
vs
TRITON_ATTN
```

---

# 30. Backend runtime 再确认

FA2：

```text
Using AttentionBackendEnum.FLASH_ATTN backend.

Using FlashAttention version 2
```

Triton：

```text
Using AttentionBackendEnum.TRITON_ATTN backend.
```

因此 A/B backend 条件明确。

---

# 31. Correctness workloads

三组 prompt：

```text
case 0:
12 prompt tokens

case 1:
42 prompt tokens

case 2:
175 prompt tokens
```

覆盖：

```text
短 prompt
多 block prompt
更长 prefill workload
```

---

# 32. Correctness 结果

Case 0：

```text
prompt_tokens_fa=12
prompt_tokens_triton=12

token_ids_equal=True
```

Case 1：

```text
prompt_tokens_fa=42
prompt_tokens_triton=42

token_ids_equal=True
```

Case 2：

```text
prompt_tokens_fa=175
prompt_tokens_triton=175

token_ids_equal=True
```

最终：

```text
[E7-C][FINAL]
token_ids_all_equal = True
```

因此三个 smoke workload 下：

> FA2 与 vLLM Triton Attention 的 greedy decoding 生成 token IDs 逐 token 完全一致。

---

# 33. Correctness 结论的边界

当前已经证明：

```text
三个代表性 workload
+
greedy decoding
+
16 output tokens
+
BF16 KV
```

得到：

```text
token-level exact match
```

但这并不等价于：

```text
所有 sequence length
所有 batch size
所有 sampling strategy
所有模型
所有数值条件
均数学意义完全一致
```

因此实验结论应该写：

> E7 的 smoke correctness PASS。

而不是扩大成：

> 两个 backend 对所有输入严格等价。

---

# 34. E7-C 结论

```text
FA2 runtime backend        PASS
Triton runtime backend     PASS

case 0 token IDs           identical
case 1 token IDs           identical
case 2 token IDs           identical

token_ids_all_equal        True
```

因此：

```text
E7-C PASS
```

---

# 35. E7 最终 Gate

```text
E7-A
Source / Config Preflight
PASS

E7-B
Runtime Data Path
PASS

E7-C
FA2 vs Triton Correctness
PASS

================================

E7
Explicit TRITON_ATTN + BF16
FINAL PASS

================================
```

---

# 36. E6 vs E7 对照

| 项目 | E6 | E7 |
|---|---|---|
| Model | Qwen3-0.6B | Qwen3-0.6B |
| Model dtype | BF16 | BF16 |
| KV dtype | BF16 | BF16 |
| block_size | 16 | 16 |
| Engine | V1 | V1 |
| Model Runner | V2 | V2 |
| eager | Yes | Yes |
| Prefix Cache | Off | Off |
| Scheduler | 相同 | 相同 |
| KVCacheManager | 相同 | 相同 |
| slot_mapping 语义 | WRITE physical slot | 相同 |
| block_table 语义 | READ physical block map | 相同 |
| Attention Backend | FLASH_ATTN | TRITON_ATTN |
| Attention Impl | FlashAttentionImpl | TritonAttentionImpl |
| Core Attention | FA2 native CUDA | Triton unified attention |
| KV WRITE | FA backend KV update | Triton reshape/cache kernel |
| READ kernel | FA2 | `kernel_unified_attention` |
| Correctness | baseline | 与 FA2 token IDs 一致 |

---

# 37. E7 最重要的架构结论

E7 最核心的发现不是：

```text
Triton 也能跑
```

而是：

> Paged KV 的控制面地址语义与 Attention kernel implementation 是可以解耦的。

完整结构：

```text
                    Scheduler
                        ↓
                 KVCacheManager
                        ↓
              physical block allocation
                        ↓
              block_table / slot_mapping
                        ↓
                   ModelRunner
                        ↓
               Attention Backend
                 /             \
                /               \
        FLASH_ATTN           TRITON_ATTN
            ↓                    ↓
      FA2 native CUDA       Triton kernels
```

Backend 替换后：

保持不变：

```text
Scheduler semantics
KV block allocation
slot_mapping
block_table
request lifecycle
```

发生变化：

```text
Attention implementation
KV physical layout contract
KV write kernel
Paged KV read kernel
Attention kernel
```

这正是 production inference runtime 常见的分层设计：

```text
control plane
≠
data plane implementation
```

---

# 38. 关于 FlashAttention 与 TritonAttention 的正确认知

不要简单理解成：

```text
FlashAttention = 一种算法
TritonAttention = 另一种算法
```

更准确：

```text
高效 tiled / online-softmax Attention
                │
        ┌───────┴────────┐
        │                │
   FA2 native CUDA    vLLM Triton
        │                │
FLASH_ATTN backend   TRITON_ATTN backend
```

当前 E6：

```text
FLASH_ATTN
    ↓
FlashAttentionImpl
    ↓
FA2 native CUDA extension
```

当前 E7：

```text
TRITON_ATTN
    ↓
TritonAttentionImpl
    ↓
vLLM 自己维护的 Triton unified attention
```

vLLM 的 `TRITON_ATTN` 不是简单调用 Dao-AILab 的实验 Triton FlashAttention 文件，而是一套为 vLLM inference / Paged KV 场景适配的 Triton backend。

---

# 39. 本实验没有证明什么

E7 没有做：

```text
FA2 vs Triton steady-state 性能比较

不同 batch size 性能

不同 sequence length 性能

CUDA Graph 下比较

Nsight kernel 时间比较

NCU kernel efficiency

INT4 correctness

INT4 memory saving

INT4 throughput

Prefix Cache + Triton

高并发 Triton
```

尤其：

```text
E7-B / E7-C 中
Triton 首次遇到部分 shape 时发生 JIT
```

所以不能依据该轮 wall time / tok/s 得出：

```text
Triton 比 FA2 快
```

或者：

```text
Triton 比 FA2 慢
```

性能问题必须单独 warmup 后再测。

---

# 40. 下一阶段：E8

E7 建立了一个非常重要的 BF16 Triton baseline：

```text
TRITON_ATTN
+
BF16 KV
+
Paged KV
+
slot_mapping
+
block_table
+
correctness PASS
```

因此 E8 才可以只改变：

```text
KV dtype
```

进入：

```text
TRITON_ATTN
+
INT4 per-token-head KV
```

E8 将引入新的问题：

```text
BF16 K/V
    ↓
per-token-head quantization
    ↓
INT4 packing
    ↓
scale cache
    ↓
quantized KV physical layout
    ↓
INT4 KV WRITE
    ↓
Paged INT4 KV READ
    ↓
dequant / fused attention
```

这时我们才开始真正进入：

```text
low-bit Paged KV Cache
```

而不是同时调试：

```text
backend
+
quantization
+
layout
+
kernel
```

多个变量。

---

# 41. 最终结论

E7 完成了从 FA2 BF16 baseline 到 Triton BF16 backend 的受控替换。

最终证据链：

```text
Config
    ↓
AttentionBackendEnum.TRITON_ATTN
    ↓
TritonAttentionBackend
    ↓
TritonAttentionImpl
    ↓
BF16 Paged KV
    ↓
slot_mapping WRITE
    ↓
Triton KV write kernel
    ↓
block_table READ
    ↓
kernel_unified_attention
    ↓
reduce_segments
    ↓
generated tokens
```

同时：

```text
FA2
vs
Triton
```

在三组 greedy smoke workload 上：

```text
token_ids_all_equal=True
```

因此：

```text
========================================

E7 — Explicit TRITON_ATTN + BF16

Source / Config       PASS
Runtime Data Path     PASS
Correctness A/B       PASS

FINAL                 PASS

========================================
```

E7 可以正式收尾，并作为 E8 `TRITON_ATTN + INT4 per-token-head KV` 的 BF16 对照基线。
