#!/usr/bin/env python3
"""VB-00 minimal offline generation smoke for Qwen3-0.6B."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams


OUTPUT_DIR = Path(__file__).resolve().parent
MODEL = "/data/models/Qwen3-0.6B"
PROMPT = (
    "A paged KV cache stores attention keys and values in fixed-size blocks. "
    "In one short sentence, state why block tables are useful during serving."
)


def main() -> None:
    config = {
        "model": MODEL,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "block_size": 16,
        "max_model_len": 2048,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 4,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "kv_cache_memory_bytes": 1024**3,
        "temperature": 0,
        "max_tokens": 2,
    }
    print(
        "VB00_CONFIG="
        + json.dumps(
            {
                **config,
                "pid": os.getpid(),
                "VLLM_USE_V2_MODEL_RUNNER": os.environ.get(
                    "VLLM_USE_V2_MODEL_RUNNER", "MISSING"
                ),
                "CUDA_VISIBLE_DEVICES": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "MISSING"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    llm = LLM(
        model=config["model"],
        tokenizer=config["model"],
        dtype=config["dtype"],
        tensor_parallel_size=config["tensor_parallel_size"],
        pipeline_parallel_size=config["pipeline_parallel_size"],
        block_size=config["block_size"],
        max_model_len=config["max_model_len"],
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_num_seqs=config["max_num_seqs"],
        enforce_eager=config["enforce_eager"],
        enable_prefix_caching=config["enable_prefix_caching"],
        kv_cache_memory_bytes=config["kv_cache_memory_bytes"],
        disable_log_stats=True,
        seed=0,
    )
    sampling = SamplingParams(
        temperature=config["temperature"], max_tokens=config["max_tokens"]
    )
    requests = llm.generate([PROMPT], sampling, use_tqdm=False)
    request = requests[0]
    completion = request.outputs[0]
    result = {
        "task": "VB-00/E0",
        "collected_at": datetime.now().astimezone().isoformat(),
        "status": "PASS",
        "config": config,
        "prompt": PROMPT,
        "prompt_token_ids": request.prompt_token_ids,
        "output_token_ids": list(completion.token_ids),
        "output_text": completion.text,
        "finish_reason": completion.finish_reason,
        "request_id": request.request_id,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    )
    print("VB00_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
