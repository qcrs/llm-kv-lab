# VB-00 / E0 最小 Smoke

结论：`PASS`。

- Qwen3-0.6B，BF16，TP=PP=1；
- V1 Engine，V2 Model Runner；
- eager，block size 16，prefix cache disabled；
- 1 GiB 固定 KV budget，可用 9,360 tokens；
- 自动选择 `FLASH_ATTN` / FlashAttention 2；
- 29 个 prompt tokens 完成 Prefill，生成 2 个 tokens `[4710, 785]`；
- EngineCore PID 为 `3259522`，完成后无残留进程。

`kv_cache_memory_bytes=1 GiB` 是共享服务器 smoke 的安全限制，不是正式性能配置。自动选择 `FLASH_ATTN` 只说明默认 BF16 路径可运行；后续 dense INT4 和 QCache Gate 仍需显式验证 Triton backend。

重跑：

```bash
bash 04-experiments/qcache/bridge/e0-smoke/run_e0_smoke.sh
```

`engine.log` 保存 runner/backend/容量和输出证据，`pid-tree.log` 保存主进程、EngineCore 与首次 FlashInfer JIT 的进程树，`result.json` 保存 token IDs。
