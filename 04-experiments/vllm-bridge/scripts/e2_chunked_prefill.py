from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os


MODEL = "/data/models/Qwen3-0.6B"


def build_long_prompt(tokenizer, target_tokens=350):
    """
    构造一个稳定的长 Prompt。

    target_tokens 只是构造阶段的目标，
    E2 最终分析必须以：

        len(output.prompt_token_ids)

    为真实 Prompt token 数。
    """

    unit = (
        "Large language model inference systems use continuous batching, "
        "paged KV cache management, scheduler token budgets, and GPU execution "
        "pipelines to efficiently serve requests. "
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


def main():
    print("=" * 80)
    print("E2 vLLM forced chunked-prefill trace")
    print("model:", MODEL)
    print("frontend/main PID:", os.getpid())
    print("QCACHE_BRIDGE_TRACE:", os.environ.get("QCACHE_BRIDGE_TRACE"))
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. 构造长 Prompt
    # ------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
    )

    prompt, constructed_prompt_tokens = build_long_prompt(
        tokenizer,
        target_tokens=350,
    )

    print("\n" + "=" * 80)
    print("PROMPT CONSTRUCTION")
    print("=" * 80)
    print(
        "constructed_prompt_tokens_no_special:",
        constructed_prompt_tokens,
    )

    # ------------------------------------------------------------
    # 2. Engine
    #
    # 与 E1 保持一致。
    #
    # 唯一关键修改：
    #
    # E1:
    # max_num_batched_tokens = 256
    #
    # E2:
    # max_num_batched_tokens = 64
    #
    # ------------------------------------------------------------

    llm = LLM(
        model=MODEL,

        # 与 E1 相同。
        dtype="bfloat16",

        # 单 GPU。
        tensor_parallel_size=1,

        # 与 E1 相同。
        block_size=16,
        max_model_len=2048,

        # E2 的核心控制变量。
        #
        # Prompt > 256 tokens，
        # 但是每个 Scheduler step
        # 最多只能安排 64 tokens。
        #
        # 因此强制出现 Chunked Prefill。
        max_num_batched_tokens=64,

        # 与 E1 相同。
        max_num_seqs=4,

        # 与 E1 相同。
        #
        # E2 不研究 KV pressure / preemption，
        # 不需要占用大量 GPU memory。
        gpu_memory_utilization=0.2,

        # 与 E1 相同。
        # 关闭 CUDA Graph，方便 trace/debug。
        enforce_eager=True,

        # E2 不研究 Prefix Cache。
        enable_prefix_caching=False,
    )

    # ------------------------------------------------------------
    # 3. Sampling
    #
    # 与 E1 完全保持一致。
    # ------------------------------------------------------------

    sampling_params = SamplingParams(
        temperature=0.0,

        # 保留 3 个输出 token，
        # 这样 Chunked Prefill 完成以后，
        # 还能继续观察 Prefill -> Decode 转换。
        max_tokens=3,

        # 防止模型提前 EOS，
        # 保证实验稳定地产生 3 个 sampled tokens。
        ignore_eos=True,
    )

    # ------------------------------------------------------------
    # 4. Request
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("REQUEST")
    print("=" * 80)

    print("num_requests: 1")
    print("max_num_batched_tokens: 64")
    print("block_size: 16")
    print("gpu_memory_utilization: 0.2")
    print("max_tokens:", sampling_params.max_tokens)
    print(
        "constructed_prompt_tokens_no_special:",
        constructed_prompt_tokens,
    )

    # Prompt 很长，不建议整个打印出来。
    print("prompt_preview:", repr(prompt[:200]) + "...")

    # ------------------------------------------------------------
    # 5. Generate
    # ------------------------------------------------------------

    outputs = llm.generate(
        [prompt],
        sampling_params,
        use_tqdm=False,
    )

    # ------------------------------------------------------------
    # 6. Ground truth
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    for output in outputs:
        print("request_id:", output.request_id)

        # 这个才是后续 E2 分析真正认的 Prompt 长度。
        print("prompt_tokens:", len(output.prompt_token_ids))

        print(
            "prompt_token_ids_head:",
            output.prompt_token_ids[:16],
        )

        print(
            "prompt_token_ids_tail:",
            output.prompt_token_ids[-16:],
        )

        for candidate in output.outputs:
            print("output_token_ids:", candidate.token_ids)
            print(
                "num_output_tokens:",
                len(candidate.token_ids),
            )
            print("output_text:", repr(candidate.text))


if __name__ == "__main__":
    main()