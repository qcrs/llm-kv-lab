#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/home/qcrs/learning/llm-kv-lab"
OUTPUT_DIR="${WORKSPACE}/04-experiments/qcache/bridge/e0-smoke"

source "${WORKSPACE}/activate" >/dev/null
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO

: > "${OUTPUT_DIR}/engine.log"
: > "${OUTPUT_DIR}/pid-tree.log"

python "${OUTPUT_DIR}/e0_smoke.py" > "${OUTPUT_DIR}/engine.log" 2>&1 &
smoke_pid=$!

while kill -0 "${smoke_pid}" 2>/dev/null; do
    date '+%F %T %Z' >> "${OUTPUT_DIR}/pid-tree.log"
    pstree -alp "${smoke_pid}" >> "${OUTPUT_DIR}/pid-tree.log" 2>&1 || true
    printf '\n' >> "${OUTPUT_DIR}/pid-tree.log"
    sleep 1
done &
sampler_pid=$!

set +e
wait "${smoke_pid}"
status=$?
wait "${sampler_pid}"
set -e

printf 'exit_code=%s\n' "${status}" > "${OUTPUT_DIR}/exit-status.txt"
tail -n 120 "${OUTPUT_DIR}/engine.log"
exit "${status}"
