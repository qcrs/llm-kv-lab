# P1 Baseline Resolution / Canonical Errata

本文件解决 P1 v3 完整设计文档中因多轮 audit 迭代留下的历史结论冲突。它不创造新的 runtime fact；它只指定当前 package 对 baseline 的 canonical interpretation。

## R1 — Whole-block zero-copy 不再是 C 级 reference gap

旧段落曾将 uniform 2D whole-block zero-copy reorder 记为 **C**。Round-3 重新核实 pinned vLLM 后，该结论已 superseded。

Canonical classification：

| 子问题 | Canonical level | 依据 |
|---|---:|---|
| MRV2 BlockTables full-row reconstruction | A | pinned `gpu/block_table.py::append_block_ids(..., overwrite=True)` staged replacement plus `num_blocks` |
| MRV2 normal runtime 使用 full-row rebuild | A | `gpu/model_runner.py::add_requests()` writes scheduler IDs into RequestState/BlockTables |
| request block-list full replacement semantic | A/B | MRV2 `add_requests()`/`update_requests()` provide current transition seam |
| reclaim-specific worker transition | B | 复用上述 primitive，并借 Tangram effective-length + worker-result + scheduler-free protocol |
| arbitrary keep-index Triton paged gather | C | input/output/address/oracle 明确，但没有 exact drop-in gather kernel |

因此：**P1 Core 的主要 exact-kernel gap 只剩 V2 arbitrary keep-index Triton paged gather。**

Pinned source：
- vLLM `568afb3a13806beb53bb2e6bd518269357b237c0`
- `vllm/v1/worker/gpu/block_table.py`
- `vllm/v1/worker/gpu/states.py`
- `vllm/v1/worker/gpu/model_runner.py`

## R2 — `reclaim_generation` 不是 Core 必需协议

旧 package 草稿把 `transaction generation / reclaim_generation` 写成必须 invariant，证据不足。

Canonical rule：

- 同步 MVP：async/spec 都关闭，优先依赖 `request_id + current canonical ownership + worker result + scheduler-side validation`。
- duplicate result：若 freed IDs 已不属于当前 canonical request ownership，必须 **no-free / reject / record evidence**，不能二次 free。
- stale result 是否可能跨 state transition 到达，必须在 M0/M1 对 fixed EngineCore/ModelRunnerOutput 消费顺序做 source audit。
- 只有 source audit 证明 request/result 生命周期存在无法通过 current ownership/state identity 排除的 stale-result risk 时，才允许批准 `reclaim_generation` 或 transaction ID。

所以 `reclaim_generation` 状态为：**DESIGN_DRAFT / CONDITIONAL，非 APPROVED Core field。**

## R3 — Position semantic 只对首个支持模型做明确保证

P1 v3 的“retained K 已带原 positional/RoPE semantic”不能泛化成所有 dense attention model。

首个支持目标必须通过 M0 gate：

- decoder-only；
- Qwen3-family first validation target；
- standard RoPE path；
- causal decoder attention；
- 不启用 dual-chunk attention / bidirectional embedding mode；
- attention backend 不以 compact physical index 重新构造 key logical position bias。

Pinned Qwen3 source 中，`Qwen3Attention.forward()` 先调用：

```text
q, k = self.rotary_emb(positions, q, k)
```

再调用：

```text
attn_output = self.attn(q, k, v)
```

因此对该首个验证路径，KV 物理重排不应重编号 model/RoPE logical position。ALiBi、relative-position bias、mRoPE/特殊 positional path、dual-chunk 等全部 **OUT_OF_SCOPE / TO_VERIFY**。

## R4 — Partial-tail block 是 V1 必测 invariant

`effective_kv_len` 是 retained **valid token count**，不是 `len(retained_blocks) * block_size`。

必须满足：

```text
physical_block_count = ceil(effective_kv_len / block_size)
```

但当存在 partial tail 时：

```text
effective_kv_len != physical_block_count * block_size
```

如果 prompt/reclaim 后最后 retained page 只有 `r` 个有效 token（`0 < r < block_size`），下一 decode 必须写到该 page 的 offset `r`，不能覆盖 retained KV，也不能把未初始化 tail 计入 attention effective length。

## R5 — Gate 命名

- `P1-G3` canonical 名称：**P1-G3 V1_RUNTIME_CORE_DONE**。
- 它不是整个主项目完成；它只证明 V1 runtime core + correctness/capacity/reuse。
- `P1-G5` canonical 名称：**P1-G5 P1_MAIN_PROJECT_STRONG_DONE**，包含 V2 Triton + runtime + profiling。
- P2 的 runtime review unlock 仍以 accepted P1-G3 为高层流程 gate，但是否立即切到 P2 仍由用户 + Web 明确决定；P1 V2 不是 optional feature。

## R6 — Async reference classification

newer vLLM 的 processed-safe boundary / deferred block free 是 V3 的强 analogous reference，因此 async lifetime design 可以记为 **B reference-backed porting**；但它仍然不是 pinned v0.26 Core，也不能提前进入 V1/V2 completion gate。

## R7 — MRV2 row replacement primitive ≠ live reclaim composite proof

`append_block_ids(..., overwrite=True)` 已是 A 级 row-rebuild primitive，但它只更新 active count并写入新 IDs；旧 row 超出新 active count 的 tail entry 不一定自动清空。因此 live reclaim 必须在 M1/M2 关闭 stale-tail seam：证明所有 consumer 严格由 effective/active range 限界，或采用 clear-old-active-range + rebuild 的薄 replacement wrapper。该 seam 不把 V1 降回 architecture-unknown，但属于必须验证的 integration contract。

## R8 — Core reclaim 不降低首次 full-prefill peak

V1/V2 Core trigger 在 final-prefill commit 之后。因此 request 在 reclaim 前仍需完整持有 prompt KV。Canonical claim 是 post-reclaim allocator capacity recovery / real page reuse；不得声称 single-request maximum prefill context 或 first-prefill peak KV 自动改善。容量评测应加入 rolling/staggered arrival。
