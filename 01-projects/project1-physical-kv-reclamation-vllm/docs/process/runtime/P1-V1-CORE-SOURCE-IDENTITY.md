# P1 V1 Core Source Identity

## Upstream Identity

- Upstream repository：`vllm-project/vllm`
- Upstream version：`v0.26.0`
- Upstream commit：`568afb3a13806beb53bb2e6bd518269357b237c0`

## Accepted P1 V1 Identity

- Accepted commit：`cd444b72ceecb2496bf015baf8c18bf883151fa1`
- Accepted tag：`p1-v1-core-accepted`
- Accepted branch：`p1/physical-kv-reclaim-v026`
- Checkpoint worktree：`CLEAN`
- Checkpoint date：2026-09-03 Asia/Shanghai
- Worktree：`/home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim`
- Historical freeze date：2026-09-02 Asia/Shanghai

## Relationship and Restore Point

```text
cd444b72ceecb2496bf015baf8c18bf883151fa1
= 568afb3a13806beb53bb2e6bd518269357b237c0
  + accepted S1/S2/S3/R1/T2/T3 production/test changes
```

历史 freeze identity 曾是 upstream + audited dirty diff；checkpoint 创建后该 identity 已被 immutable commit/tag supersede。恢复或检查使用：`git show p1-v1-core-accepted`。

## Accepted Runtime Baseline

GPU A100 80GB；MRV2；BF16；FA2；eager；TP/PP/DP/DCP/PCP=1；single KV group；`block_size=16`；prefix caching、speculative decoding、async scheduling、CUDA Graph、KV connector/offload 关闭。

## Evidence Identity

- Gate Review：`04-experiments/project1_kv_reclaim/notes/P1-V1-CORE-GATE-REVIEW-01.md`
- T3 implementation：`04-experiments/project1_kv_reclaim/notes/P1-M1-T3-IMPL-01A-Physical-Ownership-Reconciliation-实施记录.md`
- T3 closure：`04-experiments/project1_kv_reclaim/notes/P1-M1-T3-INTEGRATION-CLOSURE-真实Engine生命周期验证报告.md`
- Freeze raw：`04-experiments/project1_kv_reclaim/raw/p1-v1-core-freeze-01/p1-v1-core-freeze-01.log`
- Gate raw：`04-experiments/project1_kv_reclaim/raw/p1-v1-core-gate-review-01/p1-v1-core-gate-review-01.log`
- T3 closure raw：`04-experiments/project1_kv_reclaim/raw/m1-t3-integration-closure/m1-t3-integration-closure.log`

## Recovery Checkpoint

`p1-v1-core-accepted` 是 canonical V1 restore point。未来 regression 应优先从该 tag/commit 检出并核对工作树，不应把 upstream parent 当作完整 P1 implementation。
