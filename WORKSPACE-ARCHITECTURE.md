# LLM KV Lab Workspace Architecture

> 生成时间：2026-08-24 Asia/Shanghai
>
> 用途：提供给 ChatGPT Web / Work 做 workspace 架构复核。本文描述职责、权威链、
> 项目边界、目录语义和当前状态；逐文件路径以同目录的
> `WORKSPACE-FILE-MANIFEST.txt` 为准。

## 1. 一句话结论

这是一个“文档控制面 + 学习资料 + evidence + pinned source/reference”的 AI
Infra 学习 workspace。当前只允许审查 P1 的 M0 source/environment/worktree
freeze；P2 只有文档，runtime 被 P1 core gate 阻塞。当前没有 Approved
Implementation Slice，也没有因本次架构说明而产生 source 或实验变更。

## 2. 当前控制状态

来源：`CURRENT.md`、`REPO_PINS.md`、两个项目 `PROJECT_STATE.md`。

```yaml
phase: Documentation / Source Freeze
canonical_package_phase: NEW_PROJECT_DOCUMENTATION_FINAL_CONTRACT_V1_2
primary_project: P1 Physical KV Cache Reclamation for vLLM
current_parent_task: P1-M0-T1
execution_mode: REVIEW_ONLY
current_slice: NONE
approved_implementation_slice: NONE
p1_status: DOCUMENTATION_READY / REVIEW_ONLY
p2_status: BLOCKED_BY_PROJECT1_CORE
next_allowed_action: WEB_REVIEW_P1_M0_T1
```

P1 的实现 baseline 是 vLLM `v0.26.0`，commit
`568afb3a13806beb53bb2e6bd518269357b237c0`。P2 使用同一 vLLM pin，并将
LMCache `a7afadebb9248b62b5c533ce2c12297e9d94fc4a` 作为兼容性 baseline；
两边的 worktree、branch、environment、import path 仍然 `TO_FREEZE`。

## 3. 权威链与角色

### 技术事实链

```text
fixed source / raw evidence
  -> accepted ADR / APPROVED_DESIGN
  -> project PROJECT_STATE.md
  -> Master Plan
  -> Parent Task / Task Card
  -> derived CURRENT-CONTEXT / handoff / prompt
```

### Agent 行为链

```text
root AGENTS.md
  -> 00-docs/roadmap/02-ChatGPT-Web-Work-Codex协作与文档角色矩阵.md
  -> project AGENTS.md
  -> current Task Card
  -> one-shot Slice Prompt
```

ChatGPT Web / Work + 用户负责 architecture、设计批准、Gate、Slice 规划和结果
解释。Codex 是 Repository Execution Agent，只能执行明确批准的最小 Slice，且
只能事实性更新 `PROJECT_STATE.md`。

状态标签必须保持 `SOURCE_FACT`、`OBSERVED`、`DESIGN_DRAFT`、
`APPROVED_DESIGN`、`HYPOTHESIS`、`TO_VERIFY` 的区别。若 Approved Design 与
fixed source 冲突，必须输出 `# DESIGN_CONFLICT` 并停止 patch。

## 4. 根目录分层

```text
llm-kv-lab/
├── README.md / AGENTS.md / CURRENT.md / REPO_PINS.md
├── 00-docs/                         跨项目 active 文档控制面
├── 01-projects/                     P1/P2 active 项目文档与状态
├── 04-experiments/                  raw/processed evidence 与 handoff
├── 05-models/                       模型身份/许可索引，不含模型本体
├── third_party/vllm/                pinned study/reference tree，当前 dirty
├── archive/                         本次迁移归档，历史-only
├── 99-archive/                      更早轮次的历史 evidence/reference
├── .venvs/                          本地 Python 环境
├── .cache/                          pip/torch/Triton/HuggingFace 缓存
├── .agents/ / .codex/               Agent/tool 本地元数据
└── activate                         环境入口脚本
```

`archive/` 和 `99-archive/` 都不是当前技术 source of truth。Codex 不得从其中
恢复旧 QCache、KV Retention、QOffload 或 nano-vLLM 的实现决策。

## 5. Active 文档控制面：`00-docs/`

| 子目录 | 文件数 | 责任 |
|---|---:|---|
| `roadmap/` | 7 | 总控顺序、协作矩阵、Task/Slice/Gate 规则 |
| `design-notes/` | 4 | P1/P2 高层 accepted ADR |
| `source-reading/` | 23 | vLLM/public-source revalidation 与源码学习 |
| `profiling-notes/` | 28 | profiling 方法、NVTX/Nsight/SQLite 与优化记录 |
| `context-packs/` | 3 | Web 快速接入的 derived context |
| `interview/` | 1 | 已验证交付 evidence 入口 |
| `weekly/` | 1 | 周计划/复盘入口 |
| 根文件 `INDEX.md` | 1 | 文档导航 |

这里的学习资料是 active reference，不等于项目 implementation approval。尤其
`source-reading/vllm/`、`profiling-notes/vllm-bridge/` 和 `04-experiments/vllm-bridge/`
可以被 P1/P2 复用，但 bridge 结果不能直接升级为项目能力结论。

## 6. P1：Physical KV Cache Reclamation

路径：`01-projects/project1-physical-kv-reclamation-vllm/`，共 47 个文件。

### 文件分工

- `README.md`：项目入口、范围和阅读顺序。
- `AGENTS.md`：P1 特有范围、logical/physical ownership、不变量和 stop 条件。
- `PROJECT_STATE.md`：唯一详细状态；当前 M0-T1 review required、Slice none。
- `docs/00-overview/`：项目导航、Master Plan、SOURCE-MANIFEST、support matrix。
- `docs/10-design/`：问题定义、vLLM source map、V1 whole-block reclaim、
  logical/physical KV semantics、ownership/ACK/safe-free、V2 Triton compaction、
  V3 hardening。
- `docs/20-execution/`：M0-M7 Parent Task cards 与 Slice 模板；候选 Slice 不是授权。
- `docs/30-evaluation/`：correctness、multi-request reuse、capacity serving、
  Nsight/NCU 评测手册。
- `docs/40-agent/`：Codex 启停、Task 执行、风险/fallback 决策树。
- `docs/50-delivery/`：reference ownership、简历/面试 evidence 审计。
- `docs/90-reference/`：BASELINE-RESOLUTION、REFERENCE-PORTING-LEDGER 和完整
  v3 baseline snapshot；snapshot 是只读 reasoning reference。
- `docs/process/`：Web↔Codex workflow、execution contract。
- `learning/`、`decisions/`：项目学习卡和内部取舍入口。

### P1 技术边界

MVP 固定 single GPU、TP/PP/DP=1、dense full attention、BF16、eager、prefix
caching/spec decode/async/CUDA Graph off、one KV group。核心不变量是 logical
position 不回退、physical slot 可重建、attention 只消费 effective physical
length、worker ACK 后 scheduler 才能让 BlockPool reuse；free page 不代表 HBM
返还系统。

## 7. P2：Compressed KV Offload

路径：`01-projects/project2-compressed-kv-offload-lmcache/`，共 48 个文件。

### 文件分工

- `README.md`：P2 入口；明确 runtime blocked。
- `AGENTS.md`：P2 blocked policy、ABI gate、Serde/FS-L2/codec invariants。
- `PROJECT_STATE.md`：P2-M0-T1 blocked，Approved Slice none。
- `docs/00-overview/`：Master Plan、SOURCE-MANIFEST、support matrix、导航。
- `docs/10-design/`：vLLM/LMCache source map、V0 connector/FS-L2、V1
  object-domain Block-INT8 Serde、V2 Triton codec、V3 pipeline/resource/pre-D2H。
- `docs/20-execution/`：M0-M7 Parent Task cards 和 Slice 模板。
- `docs/30-evaluation/`：Serde bytes correctness、codec/IO crossover、filesystem
  page-cache/O_DIRECT regimes、Nsight/NCU。
- `docs/40-agent/`、`docs/50-delivery/`、`docs/process/`：Agent、交付和 runtime
  context 规范。
- `docs/90-reference/`：P2 baseline resolution、porting ledger、v3 historical
  snapshots。
- `learning/`、`decisions/`：项目学习和取舍入口。

### P2 技术边界

P2 的 `Block-INT8` 默认是 LMCache object-domain quantization block，不是
vLLM BlockPool physical page。V0 raw connector/FS-L2、V1 codec、V2 Triton 和
V3 pipeline 必须分层验证。Serialized bytes、L1 footprint 和 PCIe bytes 必须
分开；generic Serde 不自动证明 first D2H 已压缩。P2 runtime 只能在 P1
`V1_RUNTIME_CORE_DONE` accepted Gate 后解锁。

## 8. Evidence 与学习资产：`04-experiments/`

| 路径 | 文件数 | 语义 |
|---|---:|---|
| `project1_kv_reclaim/` | 7 | P1 raw/results/handoff 入口，当前无 runtime evidence |
| `project2_kv_offload/` | 7 | P2 raw/results/handoff 入口，当前 blocked |
| `vllm-bridge/` | 365 | vLLM 学习脚本、raw/processed bridge evidence；不是项目结论 |
| `qcache/` | 24 | 历史/前置 QCache evidence；保留但不属于新 P1 design |

正式实验必须记录 source commit、environment、model、workload、shape、batch、
budget、warmup/repeats、command、raw result、processed result 和 conclusion。
当前没有因本架构说明运行任何 experiment。

## 9. 历史层与可追溯性

### `archive/`

本次 migration 归档旧 `project1-qcache-vllm`、旧 `project2-kv-retention`、旧
roadmap/design-notes 和旧 project experiments。`archive/README.md` 与
`archive/legacy-projects/README.md` 明确标记 historical only。

### `99-archive/`

保留更早的 nano-vLLM、KIVI、LMCache reference、旧 roadmap、旧 profiler 和历史
manifest。它可能包含旧路径文字，这是历史记录，不是 active navigation。

## 10. Source / Environment 层

- `third_party/vllm/`：vLLM v0.26.0 pinned study tree，HEAD 是
  `568afb3...`；当前有用户 dirty changes，不能覆盖或回退。
- P1 implementation worktree、branch、editable import 和 environment：`TO_FREEZE`。
- P2 vLLM/LMCache worktree、import path、native ABI 和 filesystem root：`TO_FREEZE`。
- `.venvs/vllm-v026-torch211-cu129-py310/`：已有本地环境，但在 P1 state 中仍需
  通过 M0 identity gate 后才能成为项目 implementation environment。
- `.cache/`：运行时缓存，不是 source、evidence 或 design authority。

本轮 topology refactor 已建立 Repo A（workspace `main`）、Repo B P1 worktree
`worktrees/p1-vllm-reclaim`（branch `p1/physical-kv-reclaim-v026`），并将现有
vLLM precompiled `.so` 复制到该 worktree。Repo B study tree 的 stale 旧 QCache
worktree registration 已移除；归档文件本身仍保留。根 `activate` 现在只接受
`source ./activate p1`，不执行 Git mutation、pip install 或 native build。

## 11. 本次架构说明的修改边界

本次只新增：

- `WORKSPACE-ARCHITECTURE.md`：本说明；
- `WORKSPACE-FILE-MANIFEST.txt`：逐文件清单。

没有修改：

- `third_party/vllm/**` 或其他 runtime source；
- `CURRENT.md`、`REPO_PINS.md`、任何 `PROJECT_STATE.md`；
- P1/P2 Master Plan、support matrix、BASELINE-RESOLUTION 或 ADR；
- 任何 benchmark、profiler 或模型文件。

## 12. 给网页版 ChatGPT 的复核问题

1. `00-docs`、`01-projects`、`04-experiments`、`archive` 的职责边界是否清晰？
2. P1 是否确实是唯一 active runtime project，P2 是否仍被正确阻塞？
3. P1/P2 `PROJECT_STATE`、Master Plan、Task Card、handoff 的权威顺序是否一致？
4. `third_party/vllm` dirty study tree 与未来 implementation worktree 的身份是否
   足够清楚？
5. `99-archive` 中的旧路径文字是否只作为历史追溯，是否有任何 active 文件误引？
6. 当前是否仍满足 `REVIEW_ONLY`、Approved Slice `NONE`、不运行 benchmark 的门？

## 13. 完整文件清单

`WORKSPACE-FILE-MANIFEST.txt` 是本次生成时对所有 regular file 和 symlink 的
路径/字节数快照。它包括 `.cache`、`.venvs`、`third_party`、两个 archive 和
本文件所在 workspace；因此它是 inventory，不是 hash protection，也不改变任何
文档的可编辑性。
