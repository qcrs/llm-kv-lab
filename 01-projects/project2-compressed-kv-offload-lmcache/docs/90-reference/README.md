# P2 Reference Traceability

只登记P2 v3、pinned vLLM/LMCache与reference snapshots。旧KV Retention/QOffload不作为active design。later snapshot明确REFERENCE。



## v1.2 Final-Contract Baseline

完整 v3 reasoning snapshot：`docs/90-reference/baselines/`（从项目根路径理解；若当前文件已在 docs 内则相对路径按目录导航）。Canonical corrections：`docs/90-reference/BASELINE-RESOLUTION.md`。Snapshot 是 read-only reference，不自动批准 implementation API。

## Provenance

- Canonical corrections: `BASELINE-RESOLUTION.md`
- Implementation/reference usage ledger: `REFERENCE-PORTING-LEDGER.md`
- Historical full reasoning: `baselines/`

P2 codec/test 实现必须优先使用 pinned LMCache object-domain source/tests，而不是从 historical “physical block” wording 反推 allocator coupling。
