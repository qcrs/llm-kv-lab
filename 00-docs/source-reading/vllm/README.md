# vLLM 源码与动态桥接学习区

这里保存固定 vLLM v0.26.0 的前置学习资产，用于建立 Project 1/2 共同的 Runtime 心智模型。它不是 Project 1/2 的当前实现目录，也不改变各项目的 `PROJECT_STATE.md`。

## 子目录

- `10-vLLM-v0.26-V1-Engine-MRV2架构与KV主链.md`：当前 pinned source 的 V1 Engine / MRV2 主链锚点；明确区分 Engine V1、Model Runner V2 与 P1 feature V1/V2。

- `architecture/`：Frontend、EngineCore、Executor、Worker、Scheduler、KVCacheManager、BlockPool、ModelRunner、Paged KV 和 Prefix Cache 的源码阅读笔记。
- `experiments/`：E1–E8 动态实验报告，从 async scheduling、chunked prefill、continuous batching、KV pressure、Prefix Cache、Paged KV 到 BF16/INT4 attention bridge。

## 阅读方式

按 `architecture/00` → `01` → `02` → `03` → `04` → `05` → `06` → `07` → `08` → `09` → `10`，再按需要阅读 `experiments/01`–`08`。每个结论都要区分源码事实、动态证据和解释性推断，并绑定 fixed commit。

脚本与原始结果不放在本目录，统一位于 [`04-experiments/vllm-bridge`](../../../04-experiments/vllm-bridge/README.md)。
