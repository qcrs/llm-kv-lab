#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams


MODEL = "/data/models/Qwen3-0.6B"

PROMPT = (
    "Explain how paged KV cache maps logical KV blocks to physical GPU "
    "blocks and why slot mappings are required during decoding."
)


def parse_bool(v: str) -> bool:
    v = v.lower()

    if v in {"1", "true", "on", "yes"}:
        return True

    if v in {"0", "false", "off", "no"}:
        return False

    raise argparse.ArgumentTypeError(v)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--async-scheduling",
        type=parse_bool,
        default=True,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    print("=" * 80, flush=True)
    print("E6.6 BENCH WORKLOAD", flush=True)
    print("=" * 80, flush=True)

    print(
        f"requested batch_size={args.batch_size}",
        flush=True,
    )

    print(
        f"requested async_scheduling={args.async_scheduling}",
        flush=True,
    )

    print(
        "Initializing LLM...",
        flush=True,
    )

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",

        max_model_len=512,
        block_size=16,

        enable_prefix_caching=False,

        gpu_memory_utilization=0.2,

        # Fixed capacity across all experiments.
        max_num_batched_tokens=1024,
        max_num_seqs=16,

        disable_log_stats=True,

        # Keep CUDA Graph OFF for E6.6.
        enforce_eager=True,

        # Experimental variable.
        async_scheduling=args.async_scheduling,
    )

    # ------------------------------------------------------------
    # Verify the actual config instead of trusting the argument.
    # ------------------------------------------------------------

    engine = llm.llm_engine

    config = getattr(
        engine,
        "vllm_config",
        None,
    )

    if config is None:
        raise RuntimeError(
            "Cannot access llm.llm_engine.vllm_config"
        )

    actual_async = (
        config.scheduler_config.async_scheduling
    )

    max_concurrent_batches = getattr(
        config,
        "max_concurrent_batches",
        None,
    )

    print(
        f"actual async_scheduling={actual_async}",
        flush=True,
    )

    print(
        f"actual max_concurrent_batches="
        f"{max_concurrent_batches}",
        flush=True,
    )

    if actual_async != args.async_scheduling:
        raise RuntimeError(
            "Requested async_scheduling="
            f"{args.async_scheduling}, "
            "but actual config="
            f"{actual_async}"
        )

    # With V2 + PP=1:
    # async on  -> expected 2
    # async off -> expected 1
    expected_concurrent = (
        2 if args.async_scheduling else 1
    )

    if max_concurrent_batches != expected_concurrent:
        raise RuntimeError(
            "Unexpected max_concurrent_batches: "
            f"expected={expected_concurrent}, "
            f"actual={max_concurrent_batches}"
        )

    prompts = [
        PROMPT
        for _ in range(args.batch_size)
    ]

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    # ------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------

    for i in range(args.warmup):
        print(
            f"warmup {i + 1}/{args.warmup}",
            flush=True,
        )

        outputs = llm.generate(
            prompts,
            sampling,
            use_tqdm=False,
        )

        if len(outputs) != args.batch_size:
            raise RuntimeError(
                f"warmup returned {len(outputs)} requests, "
                f"expected {args.batch_size}"
            )

    # ------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------

    records = []

    for i in range(args.repeats):

        start = time.perf_counter()

        outputs = llm.generate(
            prompts,
            sampling,
            use_tqdm=False,
        )

        elapsed = (
            time.perf_counter() - start
        )

        output_tokens = sum(
            len(req.outputs[0].token_ids)
            for req in outputs
        )

        expected_tokens = (
            args.batch_size
            * args.max_tokens
        )

        if output_tokens != expected_tokens:
            raise RuntimeError(
                "Unexpected output token count: "
                f"expected={expected_tokens}, "
                f"actual={output_tokens}"
            )

        output_tps = (
            output_tokens / elapsed
        )

        records.append(
            {
                "repeat": i,
                "elapsed_s": elapsed,
                "output_tokens": output_tokens,
                "output_tps": output_tps,
            }
        )

        print(
            f"repeat={i} "
            f"elapsed={elapsed:.6f}s "
            f"tokens={output_tokens} "
            f"output_tps={output_tps:.2f}",
            flush=True,
        )

    result = {
        "batch_size": args.batch_size,
        "requested_async": args.async_scheduling,
        "actual_async": actual_async,
        "max_concurrent_batches":
            max_concurrent_batches,
        "max_tokens": args.max_tokens,

        "median_elapsed_s":
            statistics.median(
                r["elapsed_s"]
                for r in records
            ),

        "median_output_tps":
            statistics.median(
                r["output_tps"]
                for r in records
            ),

        "records": records,
    }

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"RESULT_JSON={out}",
        flush=True,
    )

    print(
        json.dumps(
            result,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
