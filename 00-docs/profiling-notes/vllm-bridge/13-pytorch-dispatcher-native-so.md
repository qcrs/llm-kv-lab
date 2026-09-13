# 13-vLLM：执行包装层、PyTorch Dispatcher 与 Native `.so` 来源梳理

> 目标：解释 E6/E6.5 过程中经常混淆的两个问题：  
> 1. **EngineCore → Executor → WorkerWrapper → Worker → ModelRunner 为什么有这么多包装层？**  
> 2. **Python 最终如何从 `torch.ops` 走到真正的 C++/CUDA `.so` 与 GPU kernel？**

这两件事直接决定：

```text
应该在哪一层加 NVTX？
改哪个 .py 会生效？
改 .cu 为什么可能完全没效果？
Nsight 里一个 kernel 应该反向对应到哪个业务函数？
```

---

# 1. 先看完整软件栈

不要把 vLLM 理解成：

```text
Python
↓
GPU
```

更准确：

```text
Frontend / Engine Runtime
│
├── EngineCore
├── Scheduler
└── Executor
      ↓
WorkerWrapper
      ↓
GPU Worker
      ↓
GPUModelRunner
      ↓
Model Python code
      ↓
Python operator wrapper
      ↓
PyTorch Dispatcher
      ↓
Native Extension .so
      ↓
CUDA Runtime / Driver
      ↓
GPU Kernel
```

每一层解决的是不同变化维度。

---

# 2. EngineCore 为什么不直接调用 GPUModelRunner

假设最简单设计：

```text
EngineCore
↓
GPUModelRunner
```

那 EngineCore 必须同时理解：

```text
CUDA / CPU / XPU
TP / PP / DP
single process / multiprocessing / Ray
rank / local_rank
device init
model loading
RPC / worker lifecycle
GPUModelRunner lifecycle
```

耦合会非常严重。

因此 vLLM 把变化维度拆开：

```text
EngineCore
= orchestration / request progress

Scheduler
= scheduling policy / KV control plane

Executor
= execution topology / deployment strategy

WorkerWrapper
= remote/local worker lifecycle boundary

Worker
= device/runtime ownership

ModelRunner
= actual model input preparation + execution data plane
```

---

# 3. 当前 TP1 UniProc 路径的对象链

当前实验：

```text
TP=1 / PP=1 / DP=1
V2 Model Runner
UniProcExecutor
```

大致：

```text
EngineCore
│
├── Scheduler
│
└── model_executor: UniProcExecutor
      │
      └── WorkerWrapperBase
            │
            └── gpu_worker.Worker
                  │
                  └── GPUModelRunner(V2)
```

这里既有继承，也有 has-a ownership。

最重要的不是背类图，而是知道职责边界。

---

# 4. Executor：为什么它是“部署/拓扑层”

`Executor.get_class()` 根据：

```text
distributed_executor_backend
```

选择：

```text
uni
mp
ray
external launcher
```

所以 Executor 主要回答：

> **这次 model execution 在哪里、通过什么 topology 执行？**

而不是回答：

```text
Attention 怎么算？
KV slot 是多少？
```

这些属于 ModelRunner / operator backend。

---

# 5. WorkerWrapper：为什么有一层 wrapper

WorkerWrapperBase 把：

```text
worker class construction
环境变量
rank / rpc rank
init_worker
init_device
load_model
```

变成统一 lifecycle interface。

好处：Executor 不需要直接依赖每一种设备 Worker 的实现细节。

可以理解：

```text
Executor
知道“我要一个可执行 worker”

WorkerWrapper
负责把一个具体 worker 生命周期包装成统一接口

Worker
真正持有设备与 ModelRunner
```

---

# 6. Worker 与 ModelRunner 的边界

Worker 更接近：

```text
CUDA device lifecycle
process/rank
model load
memory / KV cache init
execute_model entry
```

ModelRunner 更接近：

```text
本轮 scheduler output
↓
request state / input batch
↓
input_ids / positions
↓
block_table / slot_mapping
↓
attention metadata
↓
model forward
↓
output state
```

所以我们 profiling 时：

```text
Worker 层打 MRV2_EXECUTE
```

适合看“整次 model runner execution wall”。

而：

```text
ModelRunner 内部打 PREPARE_INPUTS / MODEL_FORWARD
```

适合继续拆执行数据面。

---

# 7. `non_block=True` 为什么属于 Executor/Future 抽象，不是 GPU 硬件保证

当前 Core 调用：

```text
model_executor.execute_model(non_block=True)
```

这个 flag 的意义属于：

```text
Executor API
Future / queued result contract
```

不能直接翻译成：

```text
GPUWorker.execute_model 立即返回
```

实际 TP1 UniProc trace：

```text
EXEC_SUBMIT wall
≈
MRV2_EXECUTE wall
≈
20+ms eager
```

说明 host execution path 仍然同步走了大量 work。

因此：

```text
抽象层“non-blocking”
≠
内部每层函数都不阻塞
≠
硬件 timeline 一定 overlap
```

---

# 8. Python 源码 provenance：先确认 `.py` 到底从哪里加载

第一步通常：

```python
import vllm
print(vllm.__file__)
```

当前确认：

```text
~/learning/llm-kv-lab/third_party/vllm/vllm/__init__.py
```

说明：

```text
EngineCore
Scheduler
ModelRunner
FlashAttentionImpl Python 部分
```

来自当前 study repo。

所以改：

```text
third_party/vllm/vllm/.../*.py
```

重启 Python/EngineCore 后通常立即生效。

---

# 9. 但 Python provenance 不等于 CUDA implementation provenance

例如 Python：

```python
vllm._custom_ops.reshape_and_cache_flash(...)
```

可能只是 wrapper。

继续进去看到：

```python
torch.ops._C_cache_ops.reshape_and_cache_flash(...)
```

这里已经跨过一个重要边界：

```text
普通 Python 调用
↓
PyTorch custom operator dispatcher
```

真正实现不再是 `_custom_ops.py`。

---

# 10. `torch.ops.xxx.yyy` 到底是什么

例如：

```python
torch.ops._C_cache_ops.reshape_and_cache_flash
```

对应逻辑 operator 名：

```text
_C_cache_ops::reshape_and_cache_flash
```

拆成：

```text
namespace = _C_cache_ops
operator  = reshape_and_cache_flash
```

`torch.ops` 不是把 `.so` 当普通 Python module 直接调用，而是访问 PyTorch 已注册的 native operator。

---

# 11. Dispatcher 是什么

可以把 PyTorch Dispatcher 理解成：

> **operator registry + runtime router。**

一个逻辑 operator 可能有：

```text
CPU implementation
CUDA implementation
Meta implementation
Autograd implementation
...
```

运行时根据：

```text
operator name
+
dispatch key / tensor device
```

选择真正实现。

例如输入是 CUDA Tensor：

```text
_C_cache_ops::reshape_and_cache_flash
        ↓
Dispatcher sees CUDA
        ↓
CUDA registered implementation
```

Dispatcher 属于 PyTorch runtime，不是 CUDA driver。

完整层次：

```text
Python
↓
PyTorch Dispatcher
↓
C++/CUDA registered implementation
↓
CUDA Runtime
↓
GPU
```

---

# 12. 谁把 operator 注册到 Dispatcher

通常是 native extension `.so` 被 import/dlopen 时执行注册代码。

概念上类似：

```cpp
TORCH_LIBRARY(...)
TORCH_LIBRARY_IMPL(..., CUDA, ...)
```

具体 vLLM 实现可能通过不同 helper/macro，但原理相同：

```text
import native .so
↓
shared library initialization
↓
register operator schema / CUDA impl
↓
torch.ops.xxx.yyy 可用
```

---

# 13. 本轮非常关键的 BEFORE / AFTER import 证据

KV write operator：

```text
_C_cache_ops::reshape_and_cache_flash
```

BEFORE：

```text
import vllm._C_stable_libtorch 之前
Dispatcher 没有对应 CUDA registration
```

执行：

```text
import vllm._C_stable_libtorch
```

实际加载：

```text
.../vllm/_C_stable_libtorch.abi3.so
```

AFTER：

```text
CUDA implementation registered
build metadata points to:
/workspace/csrc/libtorch_stable/torch_bindings.cpp:908
```

因此可以建立强证据：

```text
_C_stable_libtorch.abi3.so
负责把该 cache op 的 CUDA implementation 注册进 Dispatcher
```

不是只根据文件名猜。

---

# 14. 为什么 registration path 是 `/workspace/...`，而不是本地 repo

Dispatcher metadata 里：

```text
/workspace/csrc/libtorch_stable/torch_bindings.cpp:908
```

而本地源码：

```text
~/learning/llm-kv-lab/third_party/vllm
```

不矛盾。

`/workspace/...` 是：

```text
当初编译 precompiled .so 时
build machine 内的源码路径
```

这个路径被编译/注册 metadata 留在 binary 中。

所以：

```text
.so 现在所在路径
≠
它当初编译时的源码绝对路径
```

---

# 15. 当前环境为什么是“editable Python + precompiled native”混合体

当前 pip version：

```text
0.26.0+precompiled
```

同时 Python import 指向 study repo。

因此环境近似：

```text
Python .py
→ 当前 repo editable source

Native C++/CUDA
→ 已编译好的 .so
```

后果：

```text
改 Python .py
→ 重启即可生效

改 native .cpp/.cu
→ 不会自动改变已经安装/加载的 .so
→ 必须 rebuild / replace native extension
```

这是做 KV CUDA/Triton 改造前必须确认的事情。

---

# 16. Triton JIT 是第三类，不要和 `.so` 混在一起

代码大致分三类：

## A. Python source

```text
EngineCore
Scheduler
ModelRunner
Python wrappers
```

改 `.py`：

```text
重启 process
→ 生效
```

## B. Triton JIT

```python
@triton.jit
```

执行时：

```text
Python
↓
Triton JIT / cache
↓
GPU kernel
```

改 Triton Python source 通常不需要 rebuild vLLM native `.so`，但会触发新的 JIT/cache path。

## C. Native C++/CUDA extension

```text
_C_stable_libtorch.abi3.so
_vllm_fa2_C.abi3.so
```

改 `.cu/.cpp` 必须 rebuild binary。

---

# 17. KV WRITE：当前真实执行链

从 Attention 语义层：

```text
slot_mapping
↓
FlashAttentionImpl.do_kv_cache_update()
↓
vllm._custom_ops.reshape_and_cache_flash()
↓
torch.ops._C_cache_ops.reshape_and_cache_flash(...)
↓
PyTorch Dispatcher
↓
CUDA implementation registered by native extension
↓
_C_stable_libtorch.abi3.so
↓
reshape_and_cache_flash CUDA kernel
↓
scatter current K/V → physical KV slots
```

因此：

```text
slot_mapping
= WRITE physical address
```

---

# 18. KV READ / Attention：A100 当前走 FA2

平台选择：

```text
A100 = SM80
```

当前 `get_flash_attn_version()`：

```text
2
```

因此：

```text
FlashAttentionImpl.forward()
↓
flash_attn_varlen_func(..., fa_version=2)
↓
flash_attn_interface.py
↓
torch.ops._vllm_fa2_C.varlen_fwd(...)
↓
_vllm_fa2_C.abi3.so
↓
FA2 native CUDA implementation
```

Paged KV read 时：

```text
query
+
key_cache / value_cache
+
block_table
↓
FA2 varlen_fwd
↓
根据 block_table 读取 physical KV pages
```

因此：

```text
block_table
= READ logical→physical mapping
```

---

# 19. 为什么 FA2/FA3 `.so` 都可能加载，但 runtime 仍明确走 FA2

环境里可能同时看到：

```text
_vllm_fa2_C.abi3.so
_vllm_fa3_C.abi3.so
```

不能因此说：

```text
当前同时执行 FA2 和 FA3
```

真正 backend 由平台/runtime dispatch 决定。

A100/SM80 当前实测：

```text
fa_version=2
```

所以真实 read path 是：

```text
_vllm_fa2_C
```

加载存在 ≠ 当前路径使用。

---

# 20. 为什么 source call chain 和 Nsight kernel chain 必须结合

源码只能告诉你：

```text
FlashAttentionImpl.forward
→ torch.ops._vllm_fa2_C.varlen_fwd
```

Nsight 只能告诉你：

```text
某时刻 GPU 出现 flash_fwd_splitkv_kernel
```

两边结合：

```text
Runtime semantic NVTX
↓
CUDA API correlation
↓
GPU kernel name
↓
source/native provenance
```

才能回答：

> **这个 kernel 是哪一条业务链发出的、对应哪个 source operator、如果要修改应该改 Python/Triton/native 的哪一层。**

---

# 21. 为什么 instrumentation 要分层放，而不是只在最底层 kernel wrapper 加

如果只在：

```text
FlashAttention wrapper
```

加 NVTX，你能看到 Attention，但看不到：

```text
Scheduler
Executor submit
ModelRunner preparation
Sampling
```

如果只在 EngineCore 加：

```text
EXEC_SUBMIT
```

又无法知道 22ms 是：

```text
prepare_inputs
还是 model forward
```

所以本轮 instrumentation 逐层：

```text
Core NVTX
= runtime orchestration

Worker NVTX
= whole model runner execution

ModelRunner phase NVTX
= data-plane coarse decomposition

CUDA correlation
= operator submission → GPU execution
```

这是最合理的 observability hierarchy。

---

# 22. 包装层本身是不是性能 overhead

“层多”不等于“性能就差”。

需要 profiler 证据。

本轮：

```text
EXEC_SUBMIT ≈ MRV2_EXECUTE
Scheduler ≈0.1ms
```

说明：

```text
Executor/WorkerWrapper 外层函数包装
不是 20ms 主瓶颈
```

真正 95% 落在：

```text
MODEL_FORWARD
```

所以不要凭类数量猜 overhead。

---

# 23. 以后改代码前，先判断自己在哪一层

## 场景 1：改 Scheduler policy

```text
vllm/v1/core/sched/...
```

Python source，重启即可。

## 场景 2：改 block_table / slot_mapping preparation

```text
ModelRunner / input batch Python or Triton path
```

看具体函数是否 `@triton.jit`。

## 场景 3：改 KV write CUDA native implementation

如果当前走：

```text
_C_stable_libtorch.abi3.so
```

只改 repo `.cu` 不够，必须 rebuild native extension。

## 场景 4：改 FA2 implementation

真实路径：

```text
_vllm_fa2_C.abi3.so
```

同样需要对应 native build。

## 场景 5：写自己的 Triton dequant/gather

如果新增 `@triton.jit` Python/Triton path：

```text
不一定需要 rebuild vLLM native .so
```

---

# 24. 一套 provenance 排查 checklist

```text
1. print(vllm.__file__)
   → Python 从哪加载？

2. inspect Python wrapper source
   → 这里是不是最终实现？

3. 是否出现 torch.ops.xxx.yyy？
   → 如果是，进入 Dispatcher/native 边界

4. torch._C._dispatch_dump_table(...)
   → 有哪些 backend registration？

5. BEFORE / AFTER import native module
   → 哪个 import 导致 op registration？

6. module.__file__ / /proc/self/maps
   → 具体加载哪个 .so？

7. 平台/backend dispatch
   → FA2 还是 FA3？

8. Nsight correlation
   → runtime 真正 launch 的 GPU kernel 是谁？

9. 修改前判断：
   Python / Triton / Native C++/CUDA 哪一类？

10. Native 修改后是否 rebuild 对应 binary？
```

---

# 25. 一页恢复版

```text
Runtime：
EngineCore
↓
Scheduler
↓
Executor
↓
WorkerWrapper
↓
Worker
↓
GPUModelRunner

数据面：
prepare_inputs
↓
block_table / slot_mapping
↓
model forward
↓
Attention

KV WRITE：
slot_mapping
↓
FlashAttentionImpl.do_kv_cache_update
↓
vllm._custom_ops
↓
torch.ops._C_cache_ops.reshape_and_cache_flash
↓
Dispatcher
↓
_C_stable_libtorch.abi3.so
↓
CUDA kernel

KV READ：
block_table
↓
FlashAttentionImpl.forward
↓
flash_attn_varlen_func(fa_version=2)
↓
torch.ops._vllm_fa2_C.varlen_fwd
↓
_vllm_fa2_C.abi3.so
↓
FA2 CUDA kernel

修改规则：
.py      → restart
Triton   → JIT/cache
.cpp/.cu → rebuild .so
```

这份层次图是后续做 Quantized Paged KV Cache / KV Offload / Triton fused gather-dequant 时最应该长期保留的“代码生效与性能定位地图”。
