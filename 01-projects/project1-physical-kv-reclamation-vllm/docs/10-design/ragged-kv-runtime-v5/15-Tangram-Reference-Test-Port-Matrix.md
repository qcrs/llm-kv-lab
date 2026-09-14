> **v5 precedence note:** 本文件保留为背景/细节资料；若与 v4 canonical docs 冲突，以 `00`, `22`, `28`, `29–41` 为准。

# Tangram Reference → vLLM 0.26 Test Port Matrix

原则：**移植行为规范，不机械复制测试代码。** Tangram tests基于旧 vLLM data structures；我们的测试应保留 independent oracle，但 target 是 MRv2 新实现。

| Round | Tangram reference | 需要证明的 invariant | v0.26 target test |
|---|---|---|---|
| R1-A1 | `kv_cache_interface.py::RaggedAttentionSpec` | page bytes / divisibility | `test_ragged_spec.py` |
| R1-A2 | `core/kv_cache_utils.py` ragged branch | global pool不除 layer；one backing | `test_ragged_kv_config.py` |
| R1-B | `tests/v1/worker/test_block_table.py` | group rows/counts/snapshot topology | `test_gpu_ragged_block_tables.py` |
| R1-C | `tests/v1/attention/ragged_reference.py` | virtual block/slot equations | `test_ragged_layout_reference.py` |
| R1-C | `test_ragged_layout.py` | virtual view aliases physical cells | `test_ragged_cache_alias.py` |
| R1-D1 | Tangram per-member write tests | `reshape_and_cache` writes correct member | `test_ragged_kv_write.py` |
| R1-D2 | `test_ragged_custom_op.py` decode cases | decode member sequences equal dense | `test_ragged_fa_decode.py` |
| R1-D3 | `test_ragged_custom_op.py` prefill/mixed | uneven varlen reorder/inverse correct | `test_ragged_fa_prefill.py` |
| R2 | `ragged_layout.build_ragged_step_views` | group-specific seq lens/slots | `test_ragged_effective_state.py` |
| R3 | `test_eviction_writeback.py` + P1 oracle | payload compact + tail detach | `test_ragged_manual_reclaim.py` |
| R4 | `test_budget_scope.py` | exact budget/count semantics | `test_ragged_budget_scope.py` |
| R6 | Tangram TP uniform path | rank physical depth agreement | distributed smoke + checksum |
| R5 | Tangram preemption benchmark + MRv2 remove/reset path | true preempt invalidates physical/compression state | lifecycle integration test |
| R7 | clustering tests | map bijection / grouping / waste | offline CPU tests |
| R8 | `test_eviction_kernels.py` | in-place==Torch reference | Triton differential |
| R9 | ragged CG tests/README behavior | piecewise correctness | capture/replay smoke |

## 1. R1-C independent oracle must stay out of runtime

保留一个纯 test module，自己写 identity arithmetic：

```text
cluster(h)=h//Hp
column(h)=h%Hp
virtual_block=p*Hp+column
virtual_slot=virtual_block*N+offset
```

runtime helper若改错，不能连 oracle一起改成同一个 bug。

## 2. v0.26-specific new test: packed-content alias

Tangram reference使用 `[2,B,Hp,N,D]`，我们使用 proposed `[B,Hp,N,2D]`。因此必须新增它没有的 test：

```text
physical[p,c,t,0:D]   == K virtual[v,t]
physical[p,c,t,D:2D]  == V virtual[v,t]
```

并确认 `data_ptr/storage_offset/stride` 表明 view无 copy。

## 3. v0.26-specific new test: split write/read seam

只调用 `do_kv_cache_update`/ragged update，不调用 attention，然后检查 cache。再单独构造已经填好的 cache测 read。这样一旦 mismatch，可以确定是 WRITE 还是 READ，而不是一个 end-to-end FA error。

## 4. Minimum evidence philosophy

每个 Slice 只保留能够卡住 semantic regression 的少量测试：

- pure CPU/index math；
- one CUDA kernel differential；
- one real-engine integration。

不要把整个 upstream vLLM suite 当每个 Slice 的 acceptance；Core close 后再跑相关 regression subset。
