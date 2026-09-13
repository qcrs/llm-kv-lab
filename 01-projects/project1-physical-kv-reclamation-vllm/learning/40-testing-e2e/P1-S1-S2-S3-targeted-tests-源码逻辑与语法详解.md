# P1 S1/S2/S3 Targeted Test 详解：从测试反推 vLLM Runtime Contract

> 基于 `tests/v1/worker/test_gpu_effective_kv_state.py`。  
> 阅读目标：不要逐行翻译，而是固定按 **“这个函数/测试要证明什么 → 为什么这样构造 → 调哪个 production function → assert 什么 → 能抓什么 bug”** 来看。

---

# 0. 先看整份测试到底在测什么

这份测试可以分成四层：

```text
第 1 层：Persistent State
RequestState 中 logical / physical state 的初始化、推进、复用

第 2 层：Producer
prepare_pos_seq_lens() 是否正确生成本轮 metadata

第 3 层：Wiring
prepare_attn() 是否真的把 cache_positions 接到 slot mapping

第 4 层：Backend
FlashAttentionMetadataBuilder 是否真的使用 effective KV length
```

完整链路：

```text
RequestState
   │
   ▼
prepare_pos_seq_lens()
   │
   ├── positions --------------------------> model / RoPE
   ├── cache_positions --------------------> prepare_attn → slot mapping → KV WRITE
   ├── seq_lens ---------------------------> logical runtime
   └── effective_kv_seq_lens
                 │
                 ▼
        CommonAttentionMetadata
                 │
                 ▼
        FlashAttentionMetadataBuilder
                 │
                 ▼
        FlashAttentionMetadata.seq_lens
                 │
                 ▼
              seqused_k
                 │
                 ▼
               KV READ
```

所以这份 test 本质上是 **executable specification**：它把 S1/S2/S3 的 frozen contract 写成可执行断言。

---

# 1. `@unittest.skipUnless(current_platform.is_cuda(), "requires CUDA")` 是什么意思

代码：

```python
@unittest.skipUnless(current_platform.is_cuda(), "requires CUDA")
class TestEffectiveKVState(unittest.TestCase):
    ...
```

这是 Python decorator。

语义：

```text
current_platform.is_cuda() == True
→ 运行整个测试类

False
→ 整个测试类 SKIP
→ reason = "requires CUDA"
```

可以粗略理解成：

```python
if current_platform.is_cuda():
    run_tests()
else:
    skip_tests("requires CUDA")
```

为什么不是 FAIL？

因为这里很多测试会真正创建：

```python
torch.device("cuda")
```

并运行 GPU/Triton 相关 production code。

如果机器没有 CUDA，属于：

```text
测试环境不满足
```

而不是：

```text
代码逻辑错误
```

所以应该 `SKIPPED`，不应该 `FAILED`。

常见相关语法：

```python
@unittest.skipUnless(condition, "reason")
# condition=False → skip

@unittest.skipIf(condition, "reason")
# condition=True → skip
```

这里 decorator 放在 class 上，表示 class 内所有 test 都要求 CUDA。

---

# 2. 两个 TestCase 先分开理解

## 2.1 `TestEffectiveKVState`

它主要测 S1：

```text
RequestState 中：
num_computed_tokens
vs
effective_kv_len
```

重点是：

```text
初始化是否正确
正常推进是否正确
divergence 后是否独立推进
request row reuse 时旧 state 是否被清掉/覆盖
```

也就是：

> persistent logical / physical state 本身是否可靠。

---

## 2.2 `TestLogicalPhysicalInputContract`

它主要测 S2/S3：

```text
persistent state
   ↓
prepare_pos_seq_lens
   ↓
per-forward logical / physical metadata
```

再往后验证：

```text
cache_positions
是否真正进入 compute_slot_mappings

effective_kv_seq_lens
是否真正进入 FlashAttention metadata
```

也就是：

> state 已经存在后，execution side 有没有真正使用正确 channel。

---

# 3. Helper：`make_request_state()` 做什么

```python
def make_request_state() -> RequestState:
    return RequestState(
        max_num_reqs=1,
        max_model_len=256,
        max_num_batched_tokens=8,
        num_speculative_steps=0,
        vocab_size=128,
        device=torch.device("cuda"),
    )
```

目标：

> 创建一个最小可测试的 `RequestState`。

它没有启动：

```text
LLM
Scheduler
EngineCore
真实模型 forward
```

因为这一组测试只关心 state lifecycle。

`max_num_reqs=1` 表示当前 test 只需要一个 request slot。

`RequestState` 可以粗略理解为固定容量 table：

```text
row 0 → request A
row 1 → request B
row 2 → request C
...
```

每个 row 上有：

```text
num_computed_tokens[row]
effective_kv_len[row]
total_len[row]
last_sampled_tokens[row]
...
```

然后：

```text
req_id_to_index
```

负责：

```text
request id → RequestState row
```

这正是后面 slot reuse test 的基础。

语法：

```python
def make_request_state() -> RequestState:
```

`-> RequestState` 是 return type hint，不是运行时强制检查。

---

# 4. Helper：`add_request()` 做什么

```python
def add_request(state, req_id, initial_tokens) -> int:
```

它做三件事：

```text
1. 调 RequestState.add_request()
2. apply staged writes
3. 返回这个 request 使用的 state row
```

第一步：

```python
state.add_request(...)
```

会为新 request 分配一个 `req_idx`，并初始化对应 row。

例如：

```text
request-a → row 0

row 0:
num_computed_tokens = 128
effective_kv_len    = 128
```

第二步：

```python
state.apply_staged_writes()
```

体现 `StagedWriteTensor` 的 lifecycle：

```text
add_request()
→ stage write

apply_staged_writes()
→ publish 到 GPU state
```

第三步：

```python
torch.accelerator.synchronize()
```

GPU 操作通常异步。测试马上会 `.item()` / `.cpu()` 读结果，所以显式等待 GPU 前序工作完成。

最后：

```python
return state.req_id_to_index[req_id]
```

把 request id 映射到对应 state row。

---

# 5. Helper：`advance_one_token()` 做什么

这个 helper 在最小化模拟：

> request 正常完成一次 token execution 后，production `post_update()` 如何推进 state。

核心：

```python
post_update(...)
```

关键参数：

```python
num_computed_tokens=state.num_computed_tokens.gpu,
effective_kv_len=state.effective_kv_len.gpu,
```

两条 persistent state 都传入真正 production code。

当前 test 设定：

```text
query_len = 1
num_rejected = 0
```

所以：

```text
computed_delta = 1
```

正确 contract：

```text
num_computed_tokens += 1
effective_kv_len    += 1
```

为什么不直接在 test 写 `+=1`？

因为这样只能证明测试自己的公式对，不能证明 production `post_update()` 对。

---

# 6. `test_initialization`：测试什么、怎么测

目标：

> 新 request 刚加入时，logical / physical 应相等。

setup：

```python
state = make_request_state()
req_idx = add_request(state, "request-a", 37)
```

预期：

```text
logical = 37
physical = 37
```

assert：

```python
self.assertEqual(state.num_computed_tokens.gpu[req_idx].item(), 37)
self.assertEqual(state.effective_kv_len.gpu[req_idx].item(), 37)
```

为什么重要：

```text
P1 不是一开始就 divergence。
正常新 request：logical == physical。
只有未来 reclaim commit 后才可能 divergence。
```

`.item()`：把 0-d Tensor 转成 Python scalar。

---

# 7. `test_baseline_advancement`：测试什么

setup：

```text
128 / 128
```

调用：

```python
advance_one_token(...)
```

预期：

```text
129 / 129
```

这个测试不是在证明新 feature，而是在证明：

> 没有 reclaim 时，新设计退化为原始 vLLM behavior。

这是典型 regression / baseline equivalence test。

---

# 8. `test_independent_divergence_advancement`：S1 核心测试

先创建：

```text
logical = 128
physical = 128
```

然后人为：

```python
state.effective_kv_len.gpu[req_idx] = 64
```

得到：

```text
logical = 128
physical = 64
```

注意：这不是 production reclaim。

它是 test fixture，用来模拟：

```text
“假设未来 reclaim 已经发生，后续 normal execution 是否还正确？”
```

再执行：

```python
advance_one_token(...)
```

正确：

```text
129 / 65
```

错误实现可能是：

```python
effective_kv_len = num_computed_tokens
```

那会变成：

```text
129 / 129
```

直接把 divergence 抹掉。

所以这个 test 固定了：

```text
logical += delta
physical += delta
```

而不是：

```text
physical = logical
```

---

# 9. `test_request_slot_reuse_overwrites_state`：详细解释

这是你刚才最没理解的测试。

它测试的不是 request id，而是：

> **RequestState 的同一个固定 tensor row 被新 request 复用时，旧 request 的 state 是否被彻底覆盖。**

---

# 10. 为什么 RequestState 会复用 row

假设：

```text
max_num_reqs = 4
```

RequestState 预先有：

```text
row 0
row 1
row 2
row 3
```

例如：

```text
row 0 → request A
row 1 → request B
row 2 → free
row 3 → free
```

这些 tensor row 的 GPU memory 是预分配的。

删除 request A 时，不会把整张 tensor 重新 free/realloc，只是：

```text
row 0 从 active 变成 free
```

以后新 request 可以拿 row 0。

这叫：

```text
slot / row reuse
```

这样避免每个 request 到来都重新分配 GPU state memory。

---

# 11. Slot reuse test 一步步怎么测

## Step 1：加入 request A

```python
req_idx_a = add_request(state, "request-a", 128)
```

假设：

```text
req_idx_a = 0
```

此时：

```text
row 0:
logical  = 128
physical = 128
```

---

## Step 2：故意把 A 的 physical 改成特殊值 64

```python
state.effective_kv_len.gpu[req_idx_a] = 64
```

得到：

```text
row 0:
logical  = 128
physical = 64
```

为什么要写 64？

因为后面很容易判断：

```text
旧值 64 有没有残留到新 request
```

这是 sentinel-like test setup。

---

## Step 3：删除 request A

```python
state.remove_request("request-a")
```

现在逻辑上：

```text
row 0 → free
```

但要注意：

```text
row 0 这块 GPU tensor memory 仍然存在
```

不是整块消失。

---

## Step 4：加入 request B

```python
req_idx_b = add_request(state, "request-b", 37)
```

如果 row 0 被复用：

```text
request B → row 0
```

正确初始化应该：

```text
logical  = 37
physical = 37
```

---

## Step 5：为什么先 assert `req_idx_b == req_idx_a`

```python
self.assertEqual(req_idx_b, req_idx_a)
```

这句是整个 test 的前提保证。

如果：

```text
A 用 row 0
B 用 row 1
```

那么你根本没有测试：

```text
旧 row 0 的 stale state 是否会泄漏
```

只有：

```text
A row 0
B row 0
```

才真正命中 slot reuse 场景。

---

# 12. 什么是 stale state leakage

假设 `add_request()` 只记得初始化：

```text
num_computed_tokens
```

却忘了初始化新增的：

```text
effective_kv_len
```

那么：

```text
request A:
row 0
logical  = 128
physical = 64

remove A

request B reuse row 0
logical 被写成 37
physical 没被覆盖
```

结果：

```text
request B:
logical  = 37
physical = 64   ← 错
```

这就是：

```text
stale state leakage
```

旧 request 的状态泄漏到了新 request。

---

# 13. 为什么这个 bug 对 P1 特别危险

S2 以后：

```text
cache_positions = effective_kv_len + offset
```

如果 request B 本来应该：

```text
physical = 37
```

却残留：

```text
physical = 64
```

那么后面 producer 会算：

```text
cache_positions 从 64 开始
```

而不是从 37 开始。

再经过：

```text
compute_slot_mappings()
```

就可能写到错误 block-table coordinate。

所以这不是“一个统计字段忘了清”，而可能变成真正 KV addressing correctness bug。

---

# 14. Slot reuse test 一张图

```text
Request A 加入
──────────────
row 0:
logical  = 128
physical = 128

人为 divergence：
row 0:
logical  = 128
physical = 64


Request A remove
────────────────
row 0:
FREE

注意：GPU tensor row 仍存在


Request B 加入
──────────────
复用 row 0

正确：
logical  = 37
physical = 37

错误：
logical  = 37
physical = 64  ← request A stale state
```

因此测试必须同时证明：

```text
1. B 确实复用了 A 的 row
2. B 的 logical state 被覆盖
3. B 的 physical state 也被覆盖
```

---

# 15. 第二组核心 helper：`assert_position_contract()`

这个 helper 在测试 S2/S3 producer contract。

输入：

```text
logical_base
effective_base
query_len
```

期望输出：

```text
positions
cache_positions
seq_lens
effective_kv_seq_lens
```

也就是四个公式：

```text
positions = logical_base + local_offset
cache_positions = effective_base + local_offset

seq_lens = logical_base + query_len
effective_kv_seq_lens = effective_base + query_len
```

所以这个 helper 可以理解成：

> “调用真实 production producer，检查 frozen logical/physical contract。”

---

# 16. `idx_mapping` 在测试中是什么意思

```python
idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
```

含义：

```text
当前 batch row 0
对应 RequestState row 0
```

真实 runtime 中 batch row 和 persistent state row 不一定相同，所以需要 mapping。

---

# 17. `query_start_loc` 是什么

单 request、q=4：

```python
query_start_loc = [0,4]
```

表示：

```text
request 0 在 packed token tensor 中占 [0,4)
```

所以：

```text
query_len = 4
```

两个 request：

```text
Req A q=1
Req B q=2
```

则：

```text
query_start_loc=[0,1,3]
```

---

# 18. 为什么测试先创建 output buffers

```python
positions = torch.zeros(...)
cache_positions = torch.zeros_like(positions)
seq_lens = torch.zeros(...)
effective_seq_lens = torch.zeros_like(seq_lens)
```

因为 production `prepare_pos_seq_lens()` 是：

```text
写入预分配 tensor
```

而不是：

```text
return 新 tensor
```

这符合 vLLM GPU hot path 的 buffer reuse 设计。

`torch.zeros_like(x)`：创建 shape/dtype/device 与 x 相同的全 0 tensor。

---

# 19. 为什么这里是真正 producer test

测试调用：

```python
prepare_pos_seq_lens(...)
```

它会进入真正 production：

```text
_prepare_pos_seq_lens_kernel
```

所以不是 test 自己手算公式，而是在验证实际实现。

---

# 20. `test_baseline_equality`

输入：

```text
logical=128
physical=128
q=1
```

预期：

```text
positions=[128]
cache_positions=[128]
seq_lens=129
effective=129
```

意义：

```text
没有 divergence 时
新实现和 upstream 等价
```

---

# 21. `test_single_token_divergence`

输入：

```text
logical=128
physical=64
q=1
block_size=16
```

预期：

```text
positions=[128]
cache_positions=[64]
logical_len=129
physical_len=65
```

再：

```text
128 // 16 = 8
64  // 16 = 4
```

说明 logical coordinate 和 cache coordinate 已经指向不同 block-table column。

Python `//` 是 floor/integer division。

---

# 22. `test_divergent_positions_and_lengths`

输入：

```text
logical=128
physical=64
q=4
```

预期：

```text
positions=[128,129,130,131]
cache_positions=[64,65,66,67]
seq_lens=132
effective=68
```

为什么不能只测 q=1？

因为 q=4 才能验证：

```text
same local offset
不同 base
```

对 offset 0/1/2/3 都成立。

---

# 23. Padding test 为什么先填 `-1`

```python
logical_seq_lens = torch.full((4,), -1, ...)
```

如果一开始就是：

```text
[0,0,0,0]
```

即使 kernel 没清 padding，test 也可能误通过。

先放 sentinel：

```text
[-1,-1,-1,-1]
```

运行后要求：

```text
logical=[129,0,0,0]
physical=[65,0,0,0]
```

就能证明 padding row 真被 kernel 主动清零。

`torch.full((4,), -1)`：shape=(4,)，所有元素初值 -1。

---

# 24. `test_common_metadata_preserves_logical_seq_lens`

输入：

```text
logical=[129,98]
physical=[65,82]
```

构造：

```python
CommonAttentionMetadata(
    seq_lens=logical,
    effective_kv_seq_lens=effective,
)
```

这个 test 在固定：

```text
Common metadata 中两条 channel 必须同时存在
```

而不是：

```text
seq_lens 被 physical 覆盖
```

再测试：

```python
unpadded = metadata.unpadded(2, 1)
```

要求：

```text
logical [129,98] → [129]
physical [65,82] → [65]
```

防止 explicit reconstruction 时 physical field silent drop。

---

# 25. `_select_kv_seq_lens()` 测试什么

两种输入：

```text
physical field exists
```

和：

```text
physical field = None
```

要求：

```text
exists → physical
None   → logical
```

这是 backward compatibility。

---

# 26. `assertIs` 和 `assertEqual` 区别

```python
self.assertEqual(a, b)
```

比较：

```text
值是否相等
```

```python
self.assertIs(a, b)
```

比较：

```text
是不是同一个 Python object
```

等价于：

```python
a is b
```

selector test 用 `assertIs`，是要确认：

```text
直接选原 logical/physical tensor
```

而不是重新创建 copy。

---

# 27. `kwargs` 和 `**kwargs`

```python
kwargs = dict(
    query_start_loc=...,
    seq_lens=logical,
    ...
)
```

就是一个 dict。

然后：

```python
CommonAttentionMetadata(
    effective_kv_seq_lens=effective,
    **kwargs,
)
```

`**kwargs` 会把 dict 展开成 keyword arguments。

用于避免 physical / fallback 两套 metadata 重复写大量共同参数。

---

# 28. `test_prepare_attn_wires_cache_positions`

这是 S2 的关键 wiring test。

前面的 producer test 只能证明：

```text
cache_positions 算对了
```

还不能证明：

```text
prepare_attn 真用了它
```

所以这个测试专门抓：

```text
GPUModelRunner.prepare_attn()
→ compute_slot_mappings(..., 第三个参数, ...)
```

到底传了谁。

---

# 29. 为什么用 `FakeBlockTables`

它只实现：

```text
gather_block_tables()
compute_slot_mappings()
```

因为 `prepare_attn()` 当前只需要这些。

这叫 test double / fake object。

它不真的算 slot，而是在：

```python
captured["positions"] = positions
```

把第三个参数抓出来。

最后：

```python
assert captured[...] is cache_positions
assert captured[...] is not logical_positions
```

直接证明：

```text
prepare_attn
→ compute_slot_mappings
→ cache_positions
```

---

# 30. `SimpleNamespace` 是什么

```python
runner = SimpleNamespace(
    pcp_manager=None,
    block_tables=FakeBlockTables(),
)
```

快速构造一个有属性的简单对象：

```text
runner.pcp_manager
runner.block_tables
```

不用为测试专门定义完整 `FakeRunner`。

同理 `input_batch = SimpleNamespace(...)` 只提供 `prepare_attn()` 会访问的字段。

---

# 31. 为什么可以这样调用 production method

```python
GPUModelRunner.prepare_attn(runner, input_batch)
```

Python instance method：

```python
def prepare_attn(self, input_batch):
```

本质：

```text
obj.prepare_attn(x)
≈
GPUModelRunner.prepare_attn(obj, x)
```

所以可以让 lightweight `runner` 当 `self`，只要它具备函数内部需要访问的属性。

---

# 32. `_make_minimal_flash_builder()` 为什么用 `__new__`

```python
builder = FlashAttentionMetadataBuilder.__new__(FlashAttentionMetadataBuilder)
```

正常：

```text
Class(...)
→ __new__
→ __init__
```

直接 `__new__`：

```text
只创建对象
不跑复杂 __init__
```

为什么？

因为完整 builder 初始化可能需要大量：

```text
model config
parallel config
cache config
runner state
```

而当前 test 只想测：

```text
build()
```

所以手工补 `build()` 真正需要的属性。

这是一种 lightweight unit test 技巧。

---

# 33. `_make_builder_metadata()` 的 `torch.Tensor | None`

```python
effective_kv_seq_lens: torch.Tensor | None
```

Python 3.10+ union type hint：

```text
Tensor 或 None
```

旧写法：

```python
Optional[torch.Tensor]
```

这里正好用来构造：

```text
physical present
physical absent
```

两条测试路径。

---

# 34. `test_flash_builder_outputs_effective_lengths_and_fallback`

这个比 helper selector test 更强。

它真正执行：

```text
CommonAttentionMetadata
        ↓
FlashAttentionMetadataBuilder.build()
        ↓
FlashAttentionMetadata
```

要求：

```text
physical exists:
metadata.seq_lens=[65,82]

physical=None:
metadata.seq_lens=[129,98]
```

所以它证明 S3 真正进入 backend output，而不是只有 helper 函数写对了。

---

# 35. `test_effective_length_scope_guard`

测试：

```text
use_effective=True
→ physical

use_effective=False
→ logical
```

它固定一个工程边界：

```text
physical field 存在
不等于
所有 path 都必须 physical 化
```

当前 DCP/cascade 未 audit，所以必须有 scope guard。

---

# 36. 整份测试总结表

| Test | 要证明什么 | Setup | 调哪个 production function | Assert | 防什么 bug |
|---|---|---|---|---|---|
| initialization | 初始 logical/physical 相等 | new req=37 | `add_request` | 37/37 | physical 未初始化 |
| baseline advancement | baseline 不退化 | 128/128 | `post_update` | 129/129 | normal path 被破坏 |
| divergence advancement | 两状态独立推进 | 人工 128/64 | `post_update` | 129/65 | physical 被 mirror logical |
| slot reuse | request row reuse 安全 | A=128/64 删除，B=37 | remove/add | same row + 37/37 | stale physical state 泄漏 |
| baseline equality | producer baseline | 128/128/q1 | `prepare_pos_seq_lens` | logical==physical | fallback 行为错 |
| single divergence | S2/S3 split | 128/64/q1 | producer | 128/64 + 129/65 | 未解耦 |
| multi-token | local offsets | 128/64/q4 | producer | 两套连续坐标 | offset 错 |
| padding | tail 清零 | sentinel=-1 | producer | tail=0 | stale padded metadata |
| common metadata | logical 保留 | logical+physical | `unpadded` | 两套同步 slice | field 丢失 |
| selector fallback | backward compatibility | physical / None | selector | physical/logical | 旧 caller 失效 |
| prepare_attn wiring | S2 真 consumer | fake block tables | `prepare_attn` | third arg=cache | 仍传 logical positions |
| builder output | S3 真 consumer | minimal builder | `build` | physical/fallback | builder 没接 physical |
| scope guard | 未批准 path fallback | bool | selector | physical/logical | DCP/cascade scope 泄漏 |

---

# 37. 常见语法速查

| 语法 | 含义 |
|---|---|
| `class X(unittest.TestCase)` | unittest 测试类 |
| `@unittest.skipUnless(cond, reason)` | cond=False 时 skip |
| `@unittest.skipIf(cond, reason)` | cond=True 时 skip |
| `self.assertEqual(a,b)` | 比较值 |
| `self.assertIs(a,b)` | 比较是不是同一个对象 |
| `self.assertIsNot(a,b)` | 必须不是同一对象 |
| `-> RequestState` | 返回值 type hint |
| `list[int]` | int list type hint |
| `torch.Tensor | None` | Tensor 或 None |
| `dict[str, torch.Tensor]` | str→Tensor dict type |
| `**kwargs` | dict 展开为 keyword args |
| `SimpleNamespace(...)` | 快速创建带属性对象 |
| `Foo.__new__(Foo)` | 创建对象但跳过 `__init__` |
| `torch.zeros_like(x)` | 与 x shape/dtype/device 相同的 0 tensor |
| `torch.full((4,), -1)` | shape 4、值全 -1 |
| `.item()` | 0-d Tensor → Python scalar |
| `.cpu().tolist()` | GPU Tensor → Python list |
| `//` | floor/integer division |
| `is` | 是否同一个 Python object |
| `torch.accelerator.synchronize()` | 等设备前序工作完成 |

---

# 38. 以后读测试代码固定用这个模板

看到一个 test，不要先逐行啃语法。

固定问五个问题：

```text
1. 这个 test 要证明什么？
2. 它构造了什么 runtime state？
3. 它调用了哪个 production function？
4. 最后的 assert 在固定什么 contract？
5. 如果 assert 失败，真实 runtime 会出现什么 bug？
```

例如 slot reuse：

```text
目标：新 request 复用旧 state row 时不能继承旧 physical state

Setup：
A = 128/64
remove A
B = 37

Action：
remove_request + add_request

Assert：
B row == A row
B logical == 37
B physical == 37

防止：
stale effective_kv_len 泄漏 → cache_positions 错 → KV addressing 错
```

这就比逐行翻译更容易理解。

---

# 39. 最后一张图：整份测试在保护什么

```text
                 ┌──────────────────────┐
                 │  Backend behavior    │
                 │ FA builder output    │
                 └──────────▲───────────┘
                            │
                 ┌──────────┴───────────┐
                 │      Wiring          │
                 │ prepare_attn → cache │
                 └──────────▲───────────┘
                            │
                 ┌──────────┴───────────┐
                 │ Producer formulas    │
                 │ logical / physical   │
                 └──────────▲───────────┘
                            │
                 ┌──────────┴───────────┐
                 │ Persistent state     │
                 │ lifecycle / reuse    │
                 └──────────────────────┘
```

所以这份 targeted test 很值得结合 production source 一起读：

> production code 告诉你“代码怎么实现”；targeted test 告诉你“哪些行为是设计上绝对不能被破坏的”。
