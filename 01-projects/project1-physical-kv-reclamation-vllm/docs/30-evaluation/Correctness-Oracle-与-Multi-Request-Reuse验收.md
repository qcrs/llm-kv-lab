# P1 Correctness Oracle and Multi-Request Reuse

## Layered Oracle

### A. Pure Keep/Ownership

old=[10,11,12,13,14,15]，keep columns=[0,1,4,5]，expected retained=[10,11,14,15]、freed=[12,13]。duplicate/unsorted/OOB/empty 在 mutation 前拒绝。

### B. BlockTable / Slot

retained row 非连续 IDs；effective=64；logical=128。next logical position=128，cache position=64，slot 必须落在 retained/append active row，不指向 candidate freed page。

### C. Transaction

worker commit 前 pool unchanged；ACK 后 canonical owner exact removal；duplicate/stale/mismatch ACK no mutation；finish/abort no double free。

### D. Attention

PyTorch dense reference 对 retained K/V（原顺序）做 attention，与 vLLM reconstructed row + effective length output 比较。分别记录 dtype/tolerance；logical position不重写。

### E. Runtime Regression

必须分开两条 identity：

1. **feature disabled identity**：reclamation 完全关闭，走 upstream path；
2. **feature enabled + retention=1 no-op identity**：decision/metadata plumbing 可执行，但 `freed=[]`、effective length/ownership/outputs不变。

另外覆盖 prefill→reclaim→decode N、finish/abort、unsupported configs fail-fast。

### F. Multi-Request Real Reuse

强制可控 pool：

~~~text
A owns released_set R
→ A worker commit + scheduler commit
→ free pool includes R
→ B allocation intersects R
→ A active row excludes R
→ A and B continue correctly
~~~

仅 free count↑不够。必须保存 before/after owner lists、row、pool、B allocated IDs 与 outputs。

## Conservation

~~~text
owned active blocks
+ free blocks
+ reserved/null/in-flight explicitly accounted
= fixed pool inventory
~~~

每个 transition 前后验证，避免只看一个 counter。

## Quality

V1 policy lossy。先测试 retention=1 exact，再测 sink+recent 的生成/long-context quality；quality 下降不等于 runtime incorrect，但必须作为 trade-off。

## Failure Triage

- ID/owner mismatch → control plane。
- slot points wrong page → logical/physical state。
- owner correct但 attention mismatch → metadata/order。
- single correct multi corrupt → lifetime/request-state identity。
- free count不增 → canonical commit/allocation accounting。
- nvidia-smi不降 → expected fixed pool behavior，不判失败。



## v1.2 Mandatory Edge Cases

- partial-tail retained page：`effective_kv_len % block_size != 0`，next append offset exact；
- Qwen3/RoPE logical position vs physical slot oracle；
- unsupported positional mode/q_len>1 must fail-fast；
- duplicate worker result must not double-free even without explicit generation field。

## v1.2 Additional Mandatory Cases

### Live-row stale-tail

构造 old row 比 retained row 长的 case。reclaim 后必须证明：

- 后续 slot mapping 永不引用旧 tail IDs；
- 或 reclaim replacement 已显式清理旧 active range；
- B 复用 freed IDs 后，A 的任何 metadata/row consumer都不能重新触达这些 stale IDs。

### Granularity gate

启动时若 `blocks_per_kv_block != 1`、DCP/PCP>1、CP interleave>1 或 cascade path未获批准，feature 必须 fail-fast/保持 off。

### Prefill peak limitation

记录 reclaim 前 request-owned blocks peak 与 reclaim 后 owned blocks。测试不得把 post-reclaim capacity recovery误判成“首次 prompt 本身能超出 upstream KV pool”。
