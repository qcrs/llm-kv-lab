from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os


MODEL = "/data/models/Qwen3-0.6B"


def build_prompt(tokenizer, label: str, target_tokens: int):
    """
    构造约 target_tokens 长度的 Prompt。

    不追求绝对精确。
    E3 最终分析全部以实际 tokenizer 长度为准。
    """

    prefix = (
        f"Request {label}. "
        "This request is used to study concurrent large language model "
        "inference scheduling. "
    )

    unit = (
        "KV cache scheduling coordinates token budgets and paged memory "
        "during continuous batching. "
    )

    prompt = prefix

    while True:
        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        if len(token_ids) >= target_tokens:
            return prompt, len(token_ids)

        prompt += unit


def main():
    print("=" * 80)
    print("E3 vLLM continuous batching / concurrency trace")
    print("model:", MODEL)
    print("frontend/main PID:", os.getpid())
    print(
        "QCACHE_BRIDGE_TRACE:",
        os.environ.get("QCACHE_BRIDGE_TRACE"),
    )
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. 构造三个不同长度的 Prompt
    #
    # 手册目标：
    #
    # A ≈ 32
    # B ≈ 160
    # C ≈ 320
    #
    # ------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
    )

    prompt_a, len_a = build_prompt(
        tokenizer,
        label="A",
        target_tokens=32,
    )

    prompt_b, len_b = build_prompt(
        tokenizer,
        label="B",
        target_tokens=160,
    )

    prompt_c, len_c = build_prompt(
        tokenizer,
        label="C",
        target_tokens=320,
    )

    prompts = [
        prompt_a,
        prompt_b,
        prompt_c,
    ]

    print("\n" + "=" * 80)
    print("PROMPT CONSTRUCTION")
    print("=" * 80)

    print("A constructed tokens:", len_a)
    print("B constructed tokens:", len_b)
    print("C constructed tokens:", len_c)

    # ------------------------------------------------------------
    # 2. Engine
    #
    # E3 核心变量：
    #
    # 3 requests
    # max_num_batched_tokens = 128
    #
    # 其余尽量沿用 E1/E2。
    # ------------------------------------------------------------

    llm = LLM(
        model=MODEL,

        dtype="bfloat16",

        tensor_parallel_size=1,

        block_size=16,
        max_model_len=2048,

        # E3 核心：
        # 三个 request 共同竞争这一份 global step budget。
        max_num_batched_tokens=128,

        # 可以同时容纳三个 request。
        max_num_seqs=4,

        # 与 E1/E2 一致。
        # E3 不研究 KV pressure。
        gpu_memory_utilization=0.2,

        enforce_eager=True,

        # E3 不研究 Prefix Cache。
        enable_prefix_caching=False,
    )

    # ------------------------------------------------------------
    # 3. Sampling
    #
    # 手册 E3 固定 max_tokens=4。
    # ignore_eos 保证三个 request 都稳定地产生 4 个输出 token。
    # ------------------------------------------------------------

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
    )

    print("\n" + "=" * 80)
    print("REQUEST")
    print("=" * 80)

    print("num_requests: 3")
    print("max_num_batched_tokens: 128")
    print("max_num_seqs: 4")
    print("block_size: 16")
    print("gpu_memory_utilization: 0.2")
    print("max_tokens:", sampling_params.max_tokens)

    print("A approx tokens:", len_a)
    print("B approx tokens:", len_b)
    print("C approx tokens:", len_c)

    # ------------------------------------------------------------
    # 4. 一次 generate 提交三个 requests
    # ------------------------------------------------------------

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
    )

    # ------------------------------------------------------------
    # 5. Ground truth
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    labels = ["A", "B", "C"]

    for label, output in zip(labels, outputs):
        print("-" * 80)

        print("label:", label)
        print("request_id:", output.request_id)
        print(
            "prompt_tokens:",
            len(output.prompt_token_ids),
        )

        print(
            "prompt_token_ids_head:",
            output.prompt_token_ids[:8],
        )

        print(
            "prompt_token_ids_tail:",
            output.prompt_token_ids[-8:],
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

            print(
                "output_text:",
                repr(candidate.text),
            )


if __name__ == "__main__":
    main()