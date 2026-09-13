# P1-V1-CORE-FREEZE-01

## 1. Freeze Status

```text
P1 V1/Core = ACCEPTED / FROZEN
```

该状态来自已完成的 Web/Architecture Gate adjudication；本轮只固化 repository state 与 evidence identity。

## 2. Accepted Scope

GPU A100 80GB、MRV2、BF16、FA2、eager、TP/PP/DP/DCP/PCP=1、single KV group、`block_size=16`、prefix/spec/async/CUDA Graph/connector/offload 关闭。

## 3. Accepted Runtime Architecture

S1→S2→S3→T2→T3 的 canonical chain 见 `P1-V1-CORE-END-TO-END-ARCHITECTURE.md`：logical/physical state 分离、physical positions/visibility、Worker dense BlockTables、Scheduler canonical reconcile、safe free、real reuse、future allocation based on E。

## 4. Accepted Source Identity

```text
Freeze initially captured：upstream `568afb3a13806beb53bb2e6bd518269357b237c0` + audited working-tree diff。

Subsequent immutable checkpoint：`cd444b72ceecb2496bf015baf8c18bf883151fa1`，tag
`p1-v1-core-accepted`，worktree `CLEAN`。checkpoint 未改变已接受实现内容，仅提交了
此前 Gate-approved 的 17-file state。
```

没有创建 commit/tag。完整文件和 symbol manifest 见 `P1-V1-CORE-ACCEPTED-SOURCE-MANIFEST.md`。

## 5. Evidence Snapshot

- T3 implementation/closure historical suite：targeted `4` + regression `39` = `43 passed`。
- V1 Core Gate review rerun subset：targeted `4` + selected regression `30`，两组均 exit code 0。
- Real Engine closure：GPU 2、Qwen3-0.6B、`P1_M1_T3_INTEGRATION_CLOSURE_PASS`；A/B generation 完成，released block 4 被 request B 复用。
- Static：`py_compile` PASS；`git diff --check` PASS。

以上数字属于不同 review 时点和 selected suite，不互相覆盖，也不构成 evidence conflict。

## 6. Freeze Rules

1. P1 V1/Core is frozen。
2. 不得在 V2 开发期间顺手修改 S1/S2/S3/T2/T3 production semantics。
3. 任何 V1 修改必须先创建 `P1-V1-CORE-REOPEN-XX`，说明 reason、affected invariant、affected evidence 和 regression plan。
4. 本 freeze 不批准任何 V2 implementation、Triton/CUDA kernel、性能 benchmark 或未验证配置支持。

## 7. Limitations

`prefix caching`、`speculative decoding`、`async scheduling`、`CUDA Graph`、PP/DCP/PCP>1、multi-group KV、KV connector/offload、broader repeated reclaim：`NOT VALIDATED`。

retention policy、token-level compaction、Triton compaction、HBM/OS return、performance claim、V2：`OUT OF SCOPE`。

## 8. Next Allowed Action

```text
P1-V2-DESIGN-REVIEW-01
```

这只允许 architecture/design review，不是 V2 implementation authorization。

## 9. Freeze Deliverables

- accepted source manifest：`01-projects/project1-physical-kv-reclamation-vllm/docs/process/runtime/P1-V1-CORE-ACCEPTED-SOURCE-MANIFEST.md`
- source identity：`01-projects/project1-physical-kv-reclamation-vllm/docs/process/runtime/P1-V1-CORE-SOURCE-IDENTITY.md`
- canonical architecture：`01-projects/project1-physical-kv-reclamation-vllm/docs/process/runtime/P1-V1-CORE-END-TO-END-ARCHITECTURE.md`
- Web handoff/context/index：同目录下的 `P1-V1-CORE-FREEZE-HANDOFF.md`、`CURRENT-CONTEXT.md`、`CHATGPT-HANDOFF.md`、`ARTIFACT-INDEX.md`
- freeze raw evidence：`04-experiments/project1_kv_reclaim/raw/p1-v1-core-freeze-01/p1-v1-core-freeze-01.log`

checkpoint 文档同步日期：2026-09-03。当前 canonical restore point 为
`p1-v1-core-accepted`。
