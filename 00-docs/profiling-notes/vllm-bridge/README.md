# vLLM Bridge Profiling 学习区

这里保存 E6.5/E6.6 以及 Nsight、NVTX、SQLite、CUDA Graph、PyTorch Dispatcher 的 profiling 学习资料和结论。它们用于解释已有 bridge evidence，不直接替代 Project 1 的 `30-evaluation/` 评测方案。

## 文件分工

- `11-e6-5-e6-6-nsight-cudagraph.md`：从 Runtime 语义追踪到 host/framework submission bottleneck 与 CUDA Graph A/B。
- `12-nsight-nvtx-multiprocess-troubleshooting.md`：多进程 NVTX/Nsight 捕获与排错。
- `13-pytorch-dispatcher-native-so.md`：PyTorch dispatcher、native shared object 和执行包装层来源。
- `14-nsight-gui-eager-vs-cudagraph.md` 与图片：GUI/Eager/CUDA Graph 的逐步学习材料。
- `15-nsight-cli-sqlite-gui-workflow.md`：CLI、SQLite、GUI 三种分析方式的选择和配合。

对应脚本、capture、SQLite 和原始日志见 [`04-experiments/vllm-bridge`](../../../04-experiments/vllm-bridge/README.md)。正式 Project 1 性能结论仍须按项目 `30-evaluation/` 方案重新绑定模型、workload、环境和 Task。
