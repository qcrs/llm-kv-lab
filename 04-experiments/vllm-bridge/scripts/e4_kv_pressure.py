from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import argparse
import math
import os


MODEL = "/data/models/Qwen3-0.6B"

BLOCK_SIZE = 16
TARGET_PROMPT_TOKENS = 512


def build_prompt(tokenizer, target_tokens: int):
    unit = (
        "Large language model inference runtimes manage paged KV cache blocks "
        "under limited GPU memory while continuously scheduling requests. "
    )

    parts = []

    while True:
        parts.append(unit)
        prompt = "".join(parts)

        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        if len(token_ids) >= target_tokens:
            return prompt, len(token_ids)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["probe", "reserve", "overadmit"],
        required=True,
    )

    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if (
        args.mode != "probe"
        and args.num_gpu_blocks_override is None
    ):
        raise ValueError(
            "--num-gpu-blocks-override is required "
            "for reserve / overadmit mode"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
    )

    prompt, prompt_tokens_constructed = build_prompt(
        tokenizer,
        TARGET_PROMPT_TOKENS,
    )

    # ------------------------------------------------------------
    # E4 mode
    # ------------------------------------------------------------

    if args.mode == "probe":
        num_requests = 1

        # Probe 本身不研究 admission。
        scheduler_reserve_full_isl = True

    elif args.mode == "reserve":
        num_requests = 2

        # E4-A
        scheduler_reserve_full_isl = True

    else:
        num_requests = 2

        # E4-B
        scheduler_reserve_full_isl = False

    prompts = [prompt] * num_requests

    print("=" * 80)
    print("E4 vLLM KV pressure / admission / preemption")
    print("=" * 80)

    print("mode:", args.mode)
    print("model:", MODEL)
    print("frontend/main PID:", os.getpid())

    print(
        "QCACHE_BRIDGE_TRACE:",
        os.environ.get("QCACHE_BRIDGE_TRACE"),
    )

    print("num_requests:", num_requests)
    print(
        "constructed_prompt_tokens:",
        prompt_tokens_constructed,
    )

    print("block_size:", BLOCK_SIZE)
    print("max_num_batched_tokens:", 128)

    print(
        "scheduler_reserve_full_isl:",
        scheduler_reserve_full_isl,
    )

    print(
        "num_gpu_blocks_override:",
        args.num_gpu_blocks_override,
    )

    print(
        "theoretical_prompt_blocks:",
        math.ceil(
            prompt_tokens_constructed / BLOCK_SIZE
        ),
    )

    # ------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------

    llm_kwargs = dict(
        model=MODEL,

        dtype="bfloat16",

        tensor_parallel_size=1,

        block_size=BLOCK_SIZE,
        max_model_len=640,

        # 保持 Chunked Prefill，
        # 才能观察 over-admission。
        max_num_batched_tokens=128,

        max_num_seqs=4,

        # 与 E1/E2/E3 保持一致。
        gpu_memory_utilization=0.3,

        enforce_eager=True,

        # E4 绝对不研究 Prefix Cache。
        enable_prefix_caching=False,

        # E4-A / E4-B 的核心控制变量。
        scheduler_reserve_full_isl=(
            scheduler_reserve_full_isl
        ),
    )

    if args.num_gpu_blocks_override is not None:
        llm_kwargs["num_gpu_blocks_override"] = (
            args.num_gpu_blocks_override
        )

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,

        # 让 Prefill 结束后继续增长 KV，
        # 更容易形成真实 block pressure。
        max_tokens=8,

        ignore_eos=True,
    )

    # ------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
    )

    # ------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    for i, output in enumerate(outputs):
        print("-" * 80)

        print("index:", i)
        print("request_id:", output.request_id)

        print(
            "prompt_tokens:",
            len(output.prompt_token_ids),
        )

        for candidate in output.outputs:
            print(
                "output_token_ids:",
                candidate.token_ids,
            )

            print(
                "num_output_tokens:",
                len(candidate.token_ids),
            )


if __name__ == "__main__":
    main()