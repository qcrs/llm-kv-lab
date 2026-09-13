# Q1-00-002 安装与运行时验收

结论：`PASS`。

固定栈：

```text
vLLM: 0.26.0 @ 568afb3a13806beb53bb2e6bd518269357b237c0
Python: 3.10.20
torch: 2.11.0+cu129
Triton: 3.6.0
NCCL: 2.28.9
driver: 565.57.01
GPU: A100 80GB PCIe, SM80
system toolkit selected for nvcc/ncu/nsys: /usr/local/cuda-12.1
```

运行时证据：

- `import-smoke.json`：torch cu129、A100/SM80、NCCL 和 stable-libtorch extension；
- `vllm-collect-env.txt`：vLLM 官方环境报告；
- `source-pin.json`：源码、分支、tag、commit、环境和 wheel variant；
- `pip-freeze.txt`：最终 Python 依赖；
- `host-env-candidate.json`：安装前宿主机事实；
- `install_vllm_env.sh`：可重建安装脚本。

保留的失败证据：

1. `install-attempt-1-missing-tools.log`：初始 sparse checkout 缺少 `tools/build_rust.py`；
2. `install-attempt-2-existing-venv.log`：`uv` 拒绝覆盖第一次留下的空 venv；
3. `install-attempt-3-wrong-native-module-name.log`：依赖安装成功，但验收误用旧模块名 `vllm._C`。v0.26 CUDA 后端实际加载 `vllm._C_stable_libtorch`。

`vllm-collect-env-attempt-1-path-contamination.txt` 记录了上游工具误用旧环境 `pip3` 的现象。最终环境安装 `pip==26.2.1` 后，`vllm-collect-env.txt` 中的 pip 包版本已与当前 venv 一致。

激活：

```bash
cd /home/qcrs/learning/llm-kv-lab
source ./activate
```

`.venv` 是指向共享环境的短链接。激活入口选择 CUDA 12.1 工具链，但清除继承的 CUDA 12.1 `LD_LIBRARY_PATH`，避免覆盖 torch wheel 自带的 CUDA 12.9 用户态库。
