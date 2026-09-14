# Physical Layout / Stride / Virtual Block — Exact Contract

这是 R1-A3 的独立规范。它解决一个容易被 logical shape 掩盖的问题：**virtual block 公式只有在真实 physical stride 满足条件时才是 zero-copy。**

---

# 1. Dense v0.26 Context

FlashAttention backend logical KV shape可表达为：

```text
[B,H,N,2D]
```

实际 physical layout 可能由 stride order 变成：

```text
NHD: [B,N,H,2D]
HND: [B,H,N,2D]
```

因此 logical dimension order ≠ physical contiguous order。

---

# 2. Ragged Requirement

一个 physical page保存：

```text
Hp KV heads × N tokens × K/V payload
```

我们希望 `(page,column)` 组成 virtual block：

```text
vbid = page*Hp + column
```

FA看到：

```text
one KV head per virtual block
```

---

# 3. Dedicated First Layout

冻结：

```text
[Bphys, Hp, N, 2D]
```

其中最后三维连续。

标准 contiguous stride：

```text
stride_C  = 1
stride_N  = 2D
stride_Hp = N*2D
stride_B  = Hp*N*2D
```

因此：

```text
physical offset(page,col,n,c)
= ((page*Hp + col)*N + n)*(2D) + c
```

正好等于：

```text
virtual offset(vbid,n,c)
= (vbid*N + n)*(2D) + c
```

---

# 4. Zero-Copy View

目标 API 语义：

```python
virtual = physical.view(Bphys*Hp, 1, N, 2*D)
```

不允许内部发生：

```text
permute(...).contiguous()
clone()
materialization
```

hot-path virtual view 必须 alias same storage。

---

# 5. K/V Packed Content

qcrs/v0.26 FA 当前将 K/V packed 到 content dim：

```text
2D
```

因此首版 Ragged storage继续遵守现有 packed representation，不照抄 Tangram `[2,B,Hp,N,D]` 外观。

两者 semantic equivalent：

```text
Tangram separate K/V plane
vs
qcrs packed 2D content
```

实现选择应以 qcrs backend native shape为准。

---

# 6. Backend Contract

Dense backend 的 `get_kv_cache_shape/stride_order` 不要被全局改写。

建议 Ragged 专用路径：

```text
if RaggedAttentionSpec:
  use ragged storage shape/stride contract
else:
  existing dense backend path
```

不要通过全局环境强制 HND 影响 Dense/connector path。

---

# 7. Addressing Oracle

测试参数：

```text
Bphys=7
Hp=2/4
N=16
D=64/128
```

填充 physical tensor：

```text
value = encode(page,col,token,channel)
```

验证：

```text
virtual[vbid,0,token,channel]
== physical[page,col,token,channel]
```

随机至少覆盖：

```text
first/last page
first/last column
block boundary
last channel
```

---

# 8. Slot Oracle

给定 group row：

```text
[page7,page2,page11]
```

head column `c`，physical position `p`：

```text
depth = p//N
offset = p%N
page = row[depth]
vbid = page*Hp+c
slot = vbid*N+offset
```

将 slot 解码回 physical tuple，应得到完全相同地址。

---

# 9. Performance Gate

R1-A3 必须检查：

```text
view creation does not allocate
```

可用：

```text
same storage/data_ptr
memory_allocated delta ~0
profile without memcpy kernel
```

不需要正式 benchmark，只需证明没有隐藏 copy。

---

# 10. Fallback

如果 current backend reshape infrastructure无法直接产生该 view：

1. 新建 Ragged-specific reshape helper；
2. raw backing仍由 KV cache allocator提供；
3. 单独构造 storage view；
4. 不改 Dense global layout。

不能接受的 fallback：

```text
每个 step copy/transpose entire KV cache
```

那会破坏项目 runtime价值。
