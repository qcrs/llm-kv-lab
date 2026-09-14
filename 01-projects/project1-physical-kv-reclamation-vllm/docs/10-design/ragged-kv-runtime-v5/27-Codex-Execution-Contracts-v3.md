> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Codex Execution Contracts — v3

原则：一次只给 Codex 一个可以独立验收的 Slice。禁止让它“参考 Tangram把整个 ragged paging移植过来”。

## Contract R1-A1

```text
TASK: R1-A1 RaggedAttentionSpec / Config only.

Baseline:
qcrs/vllm branch p1/v2-token-compaction-v026
HEAD bd5a0e950e4138ab3cb6fa2e12eecdcce6607a00
Tangram reference 6fa551fc8f6edcc118a2a39b3554ee1520e91edd.

Read first:
- qcrs vllm/config/cache.py
- qcrs vllm/v1/kv_cache_interface.py
- qcrs model_executor/layers/attention/attention.py::get_kv_cache_spec
- qcrs vllm/v1/kv_cache_spec_registry.py
- Tangram vllm/v1/kv_cache_interface.py::RaggedAttentionSpec
- Tangram vllm/config/compression.py validation

Implement only:
1. minimal page_group_size config;
2. v0.26-native RaggedAttentionSpec;
3. spec generation hook;
4. hard guards for initial supported envelope;
5. focused page-math/validation tests.

Invariant:
G=Hkv/Hp and G*ragged_page_bytes=dense_page_bytes.

Do not touch:
block table, allocator implementation, attention forward, compression,
scheduler, TP, preemption.

Return:
- changed files
- source facts vs design choices
- test commands/results
- any design conflict; do not silently broaden scope.
```

## Contract R1-A2

```text
TASK: Ragged global physical page pool + one shared backing only.

Read:
- qcrs core/kv_cache_utils.py
- qcrs gpu/attn_utils.py::_allocate_kv_cache/_reshape_kv_cache
- Tangram core/kv_cache_utils.py ragged path

Implement:
- explicit ragged planner branch;
- page count available_memory/page_bytes, no layer divisor;
- one raw KVCacheTensor shared by supported attention layers;
- minimal ragged backing reshape helper if needed.

Use a small override in tests; do not yet port ring allocator.

Acceptance:
exact bytes/page count, shared data_ptr, dense regression.
```


## Contract R1-A3 — v5 REQUIRED

```text
TASK: Freeze and prove Ragged physical layout / stride / virtual-block contract.

Implement no attention math.

Required proof:
- dedicated Ragged storage has page/column-contiguous layout;
- virtual view shares data_ptr/storage;
- physical(page,column,offset) maps exactly to virtual(block,offset);
- no hidden contiguous()/copy() in hot-path adapter.

If zero-copy cannot be proven, stop with DESIGN_CONFLICT before R1-B/D1.
```

## Contract R1-B

```text
TASK: MRv2-native RaggedBlockTables identity path.

Reference semantics:
Tangram worker/ragged_block_table.py.
Do not copy its buffer architecture.

Required logical state:
[R, GT, M] block table + [R,GT] counts.
Prefer flattened persistent rows [R*GT,M] over v0.26 StagedWriteTensor/UVA.

Implement identity append/move/remove/gather/slot only.
Do NOT implement snapshot/restore in Core; MRv2 persistent rows survive ordinary unscheduled steps.
No compaction yet.
```

## Contract R1-C/D1/D2/D3

每个必须分开：

```text
C: pure layout/address tests only.
D1: write-only cache differential; no attention.
D2: decode attention equality only.
D3: prefill/mixed equality.
```

## Contract R2/R3

```text
R2:
add independent RaggedKVState E[r,g]; prove unequal-depth append/read.
No scorer.

R3:
manual target lengths/keep sets; use reference compaction backend;
worker reports shape; scheduler derives/free tails from canonical rows;
prove second request reuses released page and first continues decode.
```

## Contract R4

```text
First port exactly one scorer, preferably KeyDiff.
First close uniform-count policy wiring.
Then add per-layer pooled budget to produce automatic unequal counts.
Do not add more scorers before physical non-uniform closure passes.
```

## Review rule

每个 Slice review只回答：

```text
1. semantic invariant passed?
2. source contract respected?
3. unsupported scope still hard-gated?
4. evidence minimal but sufficient?
5. next Slice dependency satisfied?
```

不要因为“顺手”加入额外 feature。
