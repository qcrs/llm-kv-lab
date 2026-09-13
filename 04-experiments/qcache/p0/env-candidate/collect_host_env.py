#!/usr/bin/env python3
"""Collect the immutable host facts required by QCache Q1-00-001."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent
SELECTED_CUDA_HOME = Path("/usr/local/cuda-12.1")
VLLM_SOURCE = WORKSPACE / "third_party" / "vllm"
MODELS = {
    "smoke": Path("/data/models/Qwen3-0.6B"),
    "formal": Path("/data/models/Qwen3-4B-Instruct-2507"),
}


class CommandRecorder:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def run(self, name: str, argv: list[str], cwd: Path | None = None) -> str:
        record: dict[str, Any] = {
            "name": name,
            "command": shlex.join(argv),
            "cwd": str(cwd or WORKSPACE),
        }
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd or WORKSPACE,
                check=False,
                capture_output=True,
                text=True,
            )
            record.update(
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout.rstrip(),
                    "stderr": completed.stderr.rstrip(),
                }
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            record.update(
                {
                    "returncode": None,
                    "stdout": "",
                    "stderr": f"{type(exc).__name__}: {exc}",
                }
            )
        self.results.append(record)
        return record["stdout"] if record["returncode"] == 0 else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cuda_version(text: str) -> dict[str, str]:
    release = re.search(r"release\s+([0-9.]+)", text)
    build = re.search(r"\bV([0-9.]+)", text)
    return {
        "release": release.group(1) if release else "MISSING",
        "build": build.group(1) if build else "MISSING",
    }


def parse_gpu_rows(text: str) -> list[dict[str, str]]:
    keys = [
        "index",
        "name",
        "uuid",
        "driver_version",
        "memory_mib",
        "compute_capability",
    ]
    rows: list[dict[str, str]] = []
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        values = [value.strip() for value in row]
        if len(values) == len(keys):
            rows.append(dict(zip(keys, values)))
    return rows


def read_toolkit_version(cuda_home: Path) -> str:
    version_json = cuda_home / "version.json"
    if not version_json.is_file():
        return "MISSING"
    try:
        return json.loads(version_json.read_text())["cuda"]["version"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return "MISSING"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def collect_python_stack() -> dict[str, Any]:
    result: dict[str, Any] = {
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "virtual_env": os.environ.get("VIRTUAL_ENV", "MISSING"),
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda or "MISSING",
                "cuda_available": torch.cuda.is_available(),
                "cuda_visible_device_count": torch.cuda.device_count(),
                "nccl_available": torch.distributed.is_nccl_available(),
                "nccl_runtime": list(torch.cuda.nccl.version())
                if torch.distributed.is_nccl_available()
                else "MISSING",
            }
        )
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            result["visible_device_0"] = {
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": properties.total_memory,
            }
    except Exception as exc:  # The failure is evidence for the environment gate.
        result["torch_probe_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import triton

        result["triton"] = triton.__version__
    except Exception as exc:
        result["triton"] = "MISSING"
        result["triton_probe_error"] = f"{type(exc).__name__}: {exc}"

    result["nvidia_nccl_package"] = package_version("nvidia-nccl-cu12")
    return result


def collect_models(recorder: CommandRecorder) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for role, path in MODELS.items():
        config = path / "config.json"
        size_text = recorder.run(f"model_size_{role}", ["du", "-sb", str(path)])
        size_match = re.match(r"(\d+)", size_text)
        models[role] = {
            "path": str(path),
            "exists": path.is_dir(),
            "size_bytes": int(size_match.group(1)) if size_match else "MISSING",
            "config_path": str(config),
            "config_sha256": sha256(config) if config.is_file() else "MISSING",
        }
    return models


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recorder = CommandRecorder()

    gpu_text = recorder.run(
        "nvidia_smi_gpu_query",
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    )
    nvidia_smi = recorder.run("nvidia_smi", ["nvidia-smi"])
    nvcc_text = recorder.run(
        "selected_nvcc", [str(SELECTED_CUDA_HOME / "bin" / "nvcc"), "--version"]
    )
    ncu_text = recorder.run(
        "selected_ncu", [str(SELECTED_CUDA_HOME / "bin" / "ncu"), "--version"]
    )
    nsys_text = recorder.run(
        "selected_nsys", [str(SELECTED_CUDA_HOME / "bin" / "nsys"), "--version"]
    )
    gcc_text = recorder.run("gcc", ["gcc", "--version"])
    glibc_text = recorder.run("glibc", ["ldd", "--version"])
    disk_text = recorder.run("workspace_disk", ["df", "-B1", str(WORKSPACE)])

    vllm_commit = recorder.run(
        "vllm_commit", ["git", "rev-parse", "HEAD"], cwd=VLLM_SOURCE
    )
    vllm_branch = recorder.run(
        "vllm_branch", ["git", "branch", "--show-current"], cwd=VLLM_SOURCE
    )
    vllm_status = recorder.run(
        "vllm_status", ["git", "status", "--short"], cwd=VLLM_SOURCE
    )
    vllm_tags = recorder.run(
        "vllm_tags", ["git", "tag", "--points-at", "HEAD"], cwd=VLLM_SOURCE
    )

    cuda_link = Path("/usr/local/cuda")
    driver_cuda = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi)
    disk_lines = [line for line in disk_text.splitlines() if line.strip()]

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "Q1-00-001",
        "collected_at": datetime.now().astimezone().isoformat(),
        "status": "PASS" if gpu_text and nvcc_text else "INCOMPLETE",
        "host": {
            "kernel": platform.release(),
            "machine": platform.machine(),
            "glibc": platform.libc_ver()[1] or "MISSING",
            "glibc_command_first_line": glibc_text.splitlines()[0]
            if glibc_text
            else "MISSING",
            "gcc_first_line": gcc_text.splitlines()[0] if gcc_text else "MISSING",
            "workspace_disk_df_b1": disk_lines[-1] if len(disk_lines) >= 2 else "MISSING",
        },
        "gpu": {
            "physical_devices": parse_gpu_rows(gpu_text),
            "driver_reported_cuda_max": driver_cuda.group(1)
            if driver_cuda
            else "MISSING",
        },
        "cuda": {
            "floating_cuda_path": str(cuda_link),
            "floating_cuda_target": str(cuda_link.resolve())
            if cuda_link.exists()
            else "MISSING",
            "selected_cuda_home": str(SELECTED_CUDA_HOME),
            "selected_toolkit_version": read_toolkit_version(SELECTED_CUDA_HOME),
            "selected_nvcc": parse_cuda_version(nvcc_text),
            "path_nvcc": shutil.which("nvcc") or "MISSING",
            "ncu": ncu_text.splitlines()[-1] if ncu_text else "MISSING",
            "nsys": nsys_text.splitlines()[-1] if nsys_text else "MISSING",
        },
        "python_stack_before_install": collect_python_stack(),
        "models": collect_models(recorder),
        "vllm_candidate": {
            "path": str(VLLM_SOURCE),
            "commit": vllm_commit or "MISSING",
            "branch": vllm_branch or "MISSING",
            "tags_at_head": vllm_tags.splitlines() if vllm_tags else [],
            "worktree_clean": not bool(vllm_status),
        },
        "environment_selection": {
            "python": "/home/qcrs/.conda/envs/learning/bin/python3",
            "shared_venv": str(
                WORKSPACE / ".venvs" / "vllm-v026-torch211-cu129-py310"
            ),
            "short_venv_link": str(WORKSPACE / ".venv"),
            "uv_cache": str(WORKSPACE / ".cache" / "uv"),
            "wheel_variant_candidate": "cu129",
            "note": (
                "cu129 is the vLLM v0.26 CUDA 12 wheel variant; runtime compatibility "
                "must pass Q1-00-002 on the installed 565.57.01 driver."
            ),
        },
    }

    command_path = OUTPUT_DIR / "command-results.json"
    manifest_path = OUTPUT_DIR / "host-env-candidate.json"
    failure_path = OUTPUT_DIR / "failures.log"
    command_path.write_text(
        json.dumps(recorder.results, indent=2, ensure_ascii=True) + "\n"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")

    failures = [
        result
        for result in recorder.results
        if result["returncode"] not in (0, None) or result["stderr"]
    ]
    with failure_path.open("w") as handle:
        if not failures:
            handle.write("No command failures.\n")
        for result in failures:
            handle.write(f"[{result['name']}] {result['command']}\n")
            handle.write(f"returncode={result['returncode']}\n")
            handle.write(result["stderr"] + "\n\n")

    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    print(f"\nWrote {manifest_path}")
    print(f"Wrote {command_path}")
    print(f"Wrote {failure_path}")
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
