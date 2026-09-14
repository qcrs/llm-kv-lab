> **v5 canonical addendum:** R2+ allocation/reclamation 必须遵循 `38-Scheduler-Side-Ragged-Physical-State-and-Allocation-Contract.md`；metadata shape/lifetime 遵循 `39-Ragged-Step-Metadata-Shape-Lifetime-and-Attention-Contract.md`。

# R1 Exact Engineering Contract — Identity Ragged Paging v4

R1 唯一目标：**零压缩条件下改变 physical KV representation，同时真实模型输出与 Dense v0.26 等价。**

R1 按 `A1 → A2 → A3 → B → C → D1 → D2 → D3 → E → F` 严格推进。

---

# R1-A1 — Spec / Config Contract

## Source Basis

Tangram：

```text
RaggedAttentionSpec
CacheConfig.page_group_size
extended validation
```

qcrs：

```text
vllm/config/cache.py
vllm/v1/kv_cache_interface.py
vllm/v1/kv_cache_spec_registry.py
vllm/model_executor/layers/attention/attention.py::get_kv_cache_spec
```

## Frozen Semantics

```text
Hkv = semantic KV heads on current rank
Hp  = KV heads stored per physical page
G   = Hkv / Hp
```

`RaggedAttentionSpec.num_kv_heads` 保持 Hkv，不把它偷偷改成 Hp。

page bytes：

```math
P = 2 * block_size * Hp * head_size * dtype_size
```

## Startup Gates

```text
FlashAttention backend
FullAttention only
KVQuantMode.NONE
head_size_v == head_size
Hkv % Hp == 0
TP1 for R1
PP1
DCP1
PCP1
prefix cache off
spec decode off
KV connector/offload off
MLA/SWA/hybrid off
```

## Unit Acceptance

```text
G * P_ragged == P_dense
Hp=1 / Hp=Hkv valid
non-divisible Hp rejected
Hp<=0 rejected
quantized KV rejected
dense spec unchanged
```

---

# R1-A2 — Global Page Pool / Shared Raw Backing

## Problem

Dense scheduler block ID 常隐含“同一 token block depth across layer resources”。Ragged physical page 已经代表一个 `(layer,head-group)` resource，因此 page ID 必须成为 global physical namespace。

## Planner

```text
num_pages = floor(available_memory / ragged_page_bytes)
```

不能再除 `num_layers`。

## Raw Tensor

生成：

```text
one KVCacheTensor
size = page_bytes * num_pages
shared_by = all supported local attention layer names
```

## Acceptance

```text
planner page count exact
raw bytes exact
all layer aliases share backing base
physical page IDs globally unique
Dense planner regression unchanged
```

---

# R1-A3 — Physical Layout / Stride Contract

这是 v4 新增强制 Gate。

## Why

qcrs FA2 logical shape 与 physical stride 可以不同。不能看到 `[B,H,N,2D]` 就假定 `(B,H)` 可 flatten。

Tangram virtual-block 技巧要求每个 `(physical_page,column)` 是完整连续 single-head block。

## Frozen First Layout

```text
physical logical view:
[Bphys, Hp, N, 2D]
```

contiguous strides 理论：

```text
stride_B  = Hp*N*2D
stride_Hp = N*2D
stride_N  = 2D
stride_C  = 1
```

virtual：

```text
[Bphys*Hp, 1, N, 2D]
```

## Oracle

对于：

```text
page=b
column=c
position=n
channel=x
```

physical linear offset：

```text
b*stride_B + c*stride_Hp + n*stride_N + x
```

virtual block：

```text
vbid = b*Hp + c
```

virtual offset：

```text
vbid*(N*2D) + n*(2D) + x
```

二者必须严格相等。

## Acceptance

```text
physical.data_ptr == virtual.data_ptr
storage_offset equal
no .contiguous() required
stride oracle passes
random page/column/token address checks pass
```

如果 zero-copy 做不到：

```text
STOP
```

先决定 dedicated layout/reshape hook；不要带着 hidden copy 继续做 FA。

---

# R1-B — MRv2-native RaggedBlockTables

## Data Model

```text
R  = max_num_reqs
L  = local supported layers
G  = groups_per_layer
GT = L*G
M  = max blocks per group

persistent table physical: [R*GT,M]
logical:                  [R,GT,M]
counts CPU/UVA:           [R,GT]
```

### Canonical indices

```text
flat_group = layer_idx*G + group_idx
flat_row = req_idx*GT + flat_group
```

## Operations

必须定义：

```text
append_group_pages(req_idx, group_flat, ids)
overwrite_group_row(...)
trim_group_tail(...)
reset_request(req_idx)
apply_staged_writes()
gather_active_rows(req_mapping,...)
get_group_counts(...)
```

identity 阶段虽然每组深度相同，但 API 不允许假设 uniform。

## Allocation Placement Order

首版固定：

```text
layer-major
→ group-major
→ depth
```

用于 scheduler delta → worker rows 的 deterministic placement。

## Acceptance

```text
single request add
cross-block append
2 requests interleave
group-specific append
remove/reset
gather exact
counts exact
IDs unique
persistent pointer unchanged across steps
```

---

# R1-C — Member Map / Virtual Block / Slot Mapping

## Identity Member Map

```text
head h:
group = h // Hp
column = h % Hp
```

## Group-level Position

identity R1：

```text
physical position == logical cache position
```

future R2：

```text
physical position = E[group] + query_offset
```

## Virtual Slot

```text
page_depth = p // block_size
offset = p % block_size
page_id = row[req, group, page_depth]
vbid = page_id*Hp + column
slot = vbid*block_size + offset
```

## Acceptance

PyTorch CPU oracle 与 GPU kernel exact equality：

```text
multiple requests
multiple layers
groups
block boundary
padding/PAD slot
```

---

# R1-D1 — Ragged KV Write

## Current MRv2 Contract

KV cache write 与 attention read 分开：

```text
unified_kv_cache_update
→ unified_attention_with_output
```

Ragged 必须遵守这一 seam。

## Member-major K/V

input：

```text
K/V [T,Hkv,D]
```

转换：

```text
[T,Hkv,D]
→ [T*Hkv,1,D]
```

对应 member virtual slots：

```text
[T*Hkv]
```

调用现有：

```text
reshape_and_cache_flash
```

使其看到：

```text
single-head virtual cache
```

## Acceptance

先 synthetic：

```text
known page IDs
known K/V values
known virtual slots
```

逐 cell 验证 `(request,token,head)` 写到正确 physical `(page,column,offset)`。

---

# R1-D2 — Decode Attention Read

## Geometry

```text
q_len=1
Hq/Hkv = q_per_kv
Q [T,Hq,D]
→ Qm [T*Hkv,q_per_kv,D]
```

每个 `(request,KV head)` 是一个 FA varlen sequence。

需要构造：

```text
member query_start_loc
member block table
member seqused_k
virtual cache view
```

## Acceptance

```text
Dense reference attention
vs
Ragged member-major FA2
```

多种：

```text
B=1/2 requests
short/long context
Hp=1/2/4
GQA ratio >1
```

---

# R1-D3 — Prefill / Mixed Read

## Hard Part

不同 request q_len 不同，token-major 无法用一个简单 view 变 member-major。

首版允许：

```text
token-major
→ explicit permute/index
→ contiguous member-major buffer
→ FA
→ inverse scatter/copy
```

先正确再 profile。

## Acceptance Cases

```text
pure prefill one request
pure prefill multiple uneven q
mixed prefill + decode
chunked prefill continuation
```

必须与 Dense reference 在 tolerance 内一致。

---

# R1-E — Real Engine Identity

## Test Config

```text
Qwen3-0.6B
A100
FA2
TP1
PP1
BF16/FP16
block_size=16
page_group_size=2/4
compression off
```

## Workloads

1. single short prompt + decode；
2. single long prompt crossing multiple pages；
3. 2 requests continuous batching；
4. A decode + B prefill mixed；
5. chunked prefill split。

## Evidence

```text
output token IDs
optional logits max diff
physical page/accounting dump
Ragged row geometry
no-leak free count after finish
```

---

# R1-F — Piecewise CG Compatibility Smoke

只验证 existing v0.26 piecewise seam 是否容纳 identity Ragged。

不优化 metadata，不新增 custom op。

如果失败，记录：

```text
capture/replay failure surface
address instability
unsupported builder path
```

然后继续 Core eager；R9 再解决。

---

# R1 Closure Gate

R1 只有以下全部 PASS 才 CLOSED：

```text
Spec
Global Pool
Layout/Stride
BlockTables
Virtual Slot
KV Write
Decode Read
Prefill/Mixed Read
Real Engine Identity
```

R1-F Piecewise smoke 可标记 `DEFERRED`，不能掩盖 R1-E correctness。
