import argparse
import os

from vllm import LLM, SamplingParams


MODEL = "/data/models/Qwen3-0.6B"
BLOCK_SIZE = 16


def build_prompt(tokenizer):
    unit = (
        "Prefix cache bridge experiment. "
        "This repeated text is used to create a deterministic prompt "
        "for testing KV prefix reuse across sequential requests. "
    )

    prompt = ""
    while True:
        prompt += unit
        n = len(tokenizer.encode(prompt))

        # 控制在一个小而清晰的范围，
        # 同时尽量避免刚好落在 block boundary。
        if 96 <= n <= 112 and n % BLOCK_SIZE != 0:
            return prompt


def show_result(label, outputs):
    out = outputs[0]

    print("-" * 80)
    print("label:", label)
    print("request_id:", out.request_id)
    print("prompt_tokens:", len(out.prompt_token_ids))
    print("output_token_ids:", out.outputs[0].token_ids)
    print("num_output_tokens:", len(out.outputs[0].token_ids))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["off", "on"],
        required=True,
    )

    args = parser.parse_args()

    enable_prefix_caching = args.mode == "on"

    print("=" * 80)
    print("E5 vLLM Prefix Cache reuse")
    print("=" * 80)

    print("mode:", args.mode)
    print("enable_prefix_caching:", enable_prefix_caching)
    print("model:", MODEL)
    print("pid:", os.getpid())

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,

        block_size=BLOCK_SIZE,

        max_model_len=512,
        max_num_batched_tokens=256,
        max_num_seqs=2,

        gpu_memory_utilization=0.2,

        enforce_eager=True,
        enable_prefix_caching=enable_prefix_caching,
    )

    tokenizer = llm.get_tokenizer()

    prompt = build_prompt(tokenizer)
    prompt_tokens = tokenizer.encode(prompt)

    print("constructed_prompt_tokens:", len(prompt_tokens))
    print(
        "theoretical_prompt_blocks:",
        (len(prompt_tokens) + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )
    print(
        "prompt_remainder:",
        len(prompt_tokens) % BLOCK_SIZE,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2,
        ignore_eos=True,
    )

    print()
    print("=" * 80)
    print("REQUEST 0: CACHE SEED")
    print("=" * 80)

    first = llm.generate(
        [prompt],
        sampling_params,
    )

    show_result("seed", first)

    print()
    print("=" * 80)
    print("REQUEST 1: IDENTICAL PROMPT REPLAY")
    print("=" * 80)

    second = llm.generate(
        [prompt],
        sampling_params,
    )

    show_result("replay", second)


if __name__ == "__main__":
    main()