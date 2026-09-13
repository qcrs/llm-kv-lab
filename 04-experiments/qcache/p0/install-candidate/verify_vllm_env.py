#!/usr/bin/env python3
"""Verify the installed vLLM ABI and GPU runtime for Q1-00-002."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import triton
import vllm
import vllm._C_stable_libtorch
import vllm._moe_C_stable_libtorch


OUTPUT = Path(__file__).resolve().parent / "import-smoke.json"


def main() -> None:
    properties = torch.cuda.get_device_properties(0)
    tensor_sum = torch.arange(4, device="cuda").sum().item()
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    result = {
        "task": "Q1-00-002",
        "collected_at": datetime.now().astimezone().isoformat(),
        "status": "PASS",
        "vllm": vllm.__version__,
        "vllm_path": vllm.__file__,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "driver": driver,
        "cuda_home": os.environ.get("CUDA_HOME", "MISSING"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_device_count": torch.cuda.device_count(),
        "device": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "nccl_runtime": list(torch.cuda.nccl.version()),
        "cuda_tensor_sum": tensor_sum,
        "native_extension": vllm._C_stable_libtorch.__file__,
        "moe_native_extension": vllm._moe_C_stable_libtorch.__file__,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
