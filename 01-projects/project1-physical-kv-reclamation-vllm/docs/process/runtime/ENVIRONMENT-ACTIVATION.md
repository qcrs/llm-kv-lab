# P1 Environment Activation

标准 activation command：

```bash
source /home/qcrs/learning/llm-kv-lab/activate p1
```

This activates the existing vLLM v0.26 Python environment, sets
`CUDA_HOME=/usr/local/cuda-12.1`, cache roots, and prints P1 worktree/branch/HEAD
and Python/native identity. It never runs Git mutation, pip install, dependency
resolution, or native build.

M0 review 后记录：

- project root: `/home/qcrs/learning/llm-kv-lab`；
- explicit Python path: `.venvs/vllm-v026-torch211-cu129-py310/bin/python`；
- CUDA_HOME: `/usr/local/cuda-12.1`；
- cache dirs: `.cache/uv`, `.cache/triton-v026-sm80`；
- required env flags；
- minimal version/import probes；
- expected `vllm.__file__`: `worktrees/p1-vllm-reclaim/vllm/__init__.py`；
- do-not-change/install rules。

在此之前 Codex不得安装、卸载、rebuild或运行正式 workload。
