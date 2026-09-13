#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/home/qcrs/learning/llm-kv-lab"
VLLM_SOURCE="${WORKSPACE}/third_party/vllm"
ENV_DIR="${WORKSPACE}/.venvs/vllm-v026-torch211-cu129-py310"
OUTPUT_DIR="${WORKSPACE}/04-experiments/qcache/p0/install-candidate"
UV_BIN="/home/qcrs/.local/bin/uv"
BASE_PYTHON="/home/qcrs/.conda/envs/learning/bin/python3"
VLLM_COMMIT="568afb3a13806beb53bb2e6bd518269357b237c0"

mkdir -p "${WORKSPACE}/.venvs" "${WORKSPACE}/.cache/uv" "${OUTPUT_DIR}"
exec > >(tee "${OUTPUT_DIR}/install.log") 2>&1

export CUDA_HOME="/usr/local/cuda-12.1"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${PATH}"
export UV_CACHE_DIR="${WORKSPACE}/.cache/uv"
export UV_LINK_MODE="hardlink"
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_COMMIT}"
export VLLM_PRECOMPILED_WHEEL_VARIANT="cu129"

printf 'QCache Q1-00-002 vLLM install\n'
printf 'source=%s\n' "${VLLM_SOURCE}"
printf 'environment=%s\n' "${ENV_DIR}"
printf 'CUDA_HOME=%s\n' "${CUDA_HOME}"
printf 'VLLM_PRECOMPILED_WHEEL_COMMIT=%s\n' "${VLLM_PRECOMPILED_WHEEL_COMMIT}"
printf 'VLLM_PRECOMPILED_WHEEL_VARIANT=%s\n' "${VLLM_PRECOMPILED_WHEEL_VARIANT}"

actual_commit="$(git -C "${VLLM_SOURCE}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${VLLM_COMMIT}" ]]; then
    printf 'ERROR: expected vLLM commit %s, found %s\n' "${VLLM_COMMIT}" "${actual_commit}" >&2
    exit 2
fi

if [[ -n "$(git -C "${VLLM_SOURCE}" status --short)" ]]; then
    printf 'ERROR: vLLM worktree is not clean before installation.\n' >&2
    exit 3
fi

"${UV_BIN}" venv --clear --python "${BASE_PYTHON}" "${ENV_DIR}"

"${UV_BIN}" pip install --python "${ENV_DIR}/bin/python" pip

"${UV_BIN}" pip install \
    --python "${ENV_DIR}/bin/python" \
    --editable "${VLLM_SOURCE}" \
    --torch-backend=auto

"${UV_BIN}" pip freeze --python "${ENV_DIR}/bin/python" \
    > "${OUTPUT_DIR}/pip-freeze.txt"

jq -n \
    --arg repository "vllm-project/vllm" \
    --arg path "${VLLM_SOURCE}" \
    --arg branch "$(git -C "${VLLM_SOURCE}" branch --show-current)" \
    --arg tag "$(git -C "${VLLM_SOURCE}" tag --points-at HEAD | head -n 1)" \
    --arg commit "${actual_commit}" \
    --arg environment "${ENV_DIR}" \
    --arg cuda_home "${CUDA_HOME}" \
    --arg wheel_variant "${VLLM_PRECOMPILED_WHEEL_VARIANT}" \
    '{repository: $repository, path: $path, branch: $branch, tag: $tag, commit: $commit, environment: $environment, cuda_home: $cuda_home, precompiled_wheel_variant: $wheel_variant}' \
    > "${OUTPUT_DIR}/source-pin.json"

cp "${WORKSPACE}/04-experiments/qcache/p0/env-candidate/host-env-candidate.json" \
    "${OUTPUT_DIR}/host-env-candidate.json"

"${ENV_DIR}/bin/python" -c \
    'import json, torch, triton, vllm, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch; print(json.dumps({"vllm": vllm.__version__, "vllm_path": vllm.__file__, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "triton": triton.__version__, "native_extension": vllm._C_stable_libtorch.__file__, "moe_native_extension": vllm._moe_C_stable_libtorch.__file__}, indent=2))' \
    | tee "${OUTPUT_DIR}/import-smoke.json"

printf 'Install and import smoke completed.\n'
