# P1 Learning — Source Reading and Runtime Deep Dives

这里保存 Project 1 实现过程中形成的源码学习、运行时链路追踪和机制理解笔记。

## 目录职责

```text
docs/
= 项目正式设计、Scope、Contract、Evaluation、Delivery

learning/
= P1 源码导读、机制分析、完整运行链路笔记

04-experiments/project1_kv_reclaim/
= 实际执行记录、代码变更记录、测试脚本、Smoke、Evidence
阅读顺序
1. 00-foundations/

基础状态与坐标语义：

num_computed_tokens
effective_kv_len
effective_kv_seq_lens
logical position / physical KV position
batch_idx / req_state_idx
BlockTable / slot mapping

建议：

P1-M1-T1-S1-MRV2-num-computed-tokens与Effective-KV-State生命周期详解-v2.md
P1-M1-T1-S2-S3-vLLM-Logical-Physical-KV-解耦源码与实现详解-v3.md
2. 10-v1-whole-block/

V1 主线：

Logical / Physical KV Split
→ Whole-Block Reclaim
→ Ownership Reconciliation
→ Safe Free
→ Physical Block Reuse
3. 20-v2-runtime/

V2 runtime transaction：

Scheduler Plan
→ SchedulerOutput
→ Worker Prepared
→ Worker Execution
→ ExecuteModelState
→ post_update
→ CompactionResultData
→ Scheduler
4. 30-v2-compaction/

V2 核心：

Paged KV Compaction
→ Worker State Transition
→ Scheduler Reconciliation
→ Physical Release / Reuse
→ Real-Engine E2E
5. 40-testing-e2e/

测试与 E2E 链路：

targeted test
→ runtime contract test
→ real-engine smoke
Truth Priority

如果文档之间出现冲突：

current source
>
PROJECT_STATE.md
>
docs/ canonical design
>
accepted experiment evidence
>
learning notes
>
historical records
