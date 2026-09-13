# Source Reading Rules

源码笔记每条使用 SOURCE_FACT / OBSERVED / HYPOTHESIS / TO_VERIFY，并记录 exact commit、path、symbol、caller/callee、owner、state、shape/dtype、CPU/GPU boundary、side effect、tests 与 design implication。

source reading 不直接改变 PROJECT_STATE 或批准设计。一次精读 1–3 个 symbol；不复制大段源码。

## Final Contract Revalidation

`GITHUB-REVALIDATION-v1.2.md` 是当前 package 的 targeted public-source revalidation；`v1.1` 文件保留为历史记录。它们都不能替代 local M0 source audit。
