# vLLM Bridge 实验资产

这是 Project 1/2 共用的 vLLM 定向学习和桥接实验区。它承接固定 vLLM v0.26.0 的 E0–E8 前置证据，不等同于 Project 1 M0–M11 或 Project 2 M0–M6 的正式结果。

## 目录

```text
scripts/   可运行 workload 和 probe 脚本
raw/       原始命令、日志、JSON/CSV、Nsight report、SQLite 和环境快照
processed/ 可复用汇总、表格和分析输出（当前为空时保持目录）
```

E0–E8 的报告位于 `00-docs/source-reading/vllm/experiments/`；Nsight/NVTX 学习资料位于 `00-docs/profiling-notes/vllm-bridge/`。正式项目实验必须分别写入 `04-experiments/project1_pkv/` 或 `04-experiments/project2_kv_retention/`，不把 bridge 结果冒充项目完成。

## 脚本规则

- 脚本默认只读 fixed study tree；不修改 `third_party/vllm` 的用户 dirty changes。
- 新脚本先写入 `scripts/`，运行产物写入对应 `raw/<experiment>/`。
- 脚本中的路径以工作区根目录为基准；不要恢复旧的 `repro/` 路径。
- 失败产物保留，后续用说明文档解释其状态，不删除“难看”的结果。
