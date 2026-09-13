from vllm import LLM, SamplingParams


MODEL = "/data/models/Qwen3-0.6B"
TARGET_PROMPT_TOKENS = 31


def build_prompt(tokenizer):
    units = [
        " paged",
        " cache",
        " token",
        " bridge",
        " kv",
    ]

    # Find a deterministic text with exactly 31 tokenizer tokens.
    for unit in units:
        prompt = ""
        for _ in range(128):
            prompt += unit
            n = len(tokenizer.encode(prompt))

            if n == TARGET_PROMPT_TOKENS:
                return prompt

            if n > TARGET_PROMPT_TOKENS:
                break

    # Fallback: choose the closest prompt below 32 tokens.
    best_prompt = None
    best_len = -1

    unit = (
        " Paged KV address experiment validates "
        "block tables and slot mappings."
    )

    prompt = ""
    for _ in range(32):
        prompt += unit
        n = len(tokenizer.encode(prompt))

        if n < 32 and n > best_len:
            best_prompt = prompt
            best_len = n

        if n >= 32:
            break

    assert best_prompt is not None
    return best_prompt


def main():
    print("=" * 80)
    print("E6: PAGED KV ADDRESSING + DEFAULT ATTENTION + BF16 KV")
    print("=" * 80)

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        kv_cache_dtype="auto",

        tensor_parallel_size=1,

        block_size=16,

        max_model_len=512,
        max_num_batched_tokens=256,
        max_num_seqs=1,

        gpu_memory_utilization=0.2,

        enable_prefix_caching=False,
        enforce_eager=True,
    )

    tokenizer = llm.get_tokenizer()

    prompt = build_prompt(tokenizer)
    prompt_ids = tokenizer.encode(prompt)

    print("prompt_tokens:", len(prompt_ids))
    print("block_size:", 16)
    print(
        "prompt_blocks:",
        (len(prompt_ids) + 15) // 16,
    )
    print(
        "prompt_remainder:",
        len(prompt_ids) % 16,
    )

    params = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
    )

    outputs = llm.generate(
        [prompt],
        params,
    )

    out = outputs[0]

    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print("request_id:", out.request_id)
    print(
        "prompt_tokens:",
        len(out.prompt_token_ids),
    )
    print(
        "output_token_ids:",
        out.outputs[0].token_ids,
    )
    print(
        "num_output_tokens:",
        len(out.outputs[0].token_ids),
    )


if __name__ == "__main__":
    main()