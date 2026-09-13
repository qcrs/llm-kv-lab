# P1 Reference Traceability

这里只登记新 P1 v3 baseline、pinned source 与 reference inventory。旧 QCache 技术文档不作为 active reference。SOURCE-MANIFEST 是入口；actual source excerpts/evidence 在 M0 后登记，不复制大段源码。



## v1.2 Final-Contract Baseline

完整 v3 reasoning snapshot：`docs/90-reference/baselines/`（从项目根路径理解；若当前文件已在 docs 内则相对路径按目录导航）。Canonical corrections：`docs/90-reference/BASELINE-RESOLUTION.md`。Snapshot 是 read-only reference，不自动批准 implementation API。

## Provenance

- Canonical corrections: `BASELINE-RESOLUTION.md`
- Implementation/reference usage ledger: `REFERENCE-PORTING-LEDGER.md`
- Historical full reasoning: `baselines/`

Implementation 时优先读 resolution + current design + ledger；historical snapshot 不直接授权 patch。
