from vllm import LLM, SamplingParams
import os


MODEL = "/data/models/Qwen3-0.6B"


def main():
    print("=" * 80)
    print("E1 vLLM single-request Prefill -> Decode trace")
    print("model:", MODEL)
    print("frontend/main PID:", os.getpid())
    print("QCACHE_BRIDGE_TRACE:", os.environ.get("QCACHE_BRIDGE_TRACE"))
    print("=" * 80)

    llm = LLM(
        model=MODEL,

        # 固定 BF16。
        dtype="bfloat16",

        # 单 GPU，当前不引入 TP。
        tensor_parallel_size=1,

        # 与 E0 保持一致。
        block_size=16,
        max_model_len=2048,
        max_num_batched_tokens=256,
        max_num_seqs=4,

        gpu_memory_utilization=0.2,

        # 关闭 CUDA Graph，方便 trace / debugger。
        enforce_eager=True,

        # E1 不研究 Prefix Cache。
        enable_prefix_caching=False,
    )

    sampling_params = SamplingParams(
        temperature=0.0,

        # E1 需要看到：
        # Prefill
        # Decode
        # Decode
        #
        # 因此生成 3 个 token。
        max_tokens=3,

        # 实验目的是固定看到 3 个生成 token，
        # 避免模型提前 EOS 导致少一个 Decode step。
        ignore_eos=True,
    )

    prompt = (
        "Large language model inference systems use KV cache "
        "to avoid recomputing previous keys and values during "
        "autoregressive decoding. Explain briefly why paged KV "
        "cache is useful for serving multiple requests efficiently."
    )

    print("\n" + "=" * 80)
    print("REQUEST")
    print("=" * 80)
    print("num_requests: 1")
    print("max_tokens:", sampling_params.max_tokens)
    print("prompt:", repr(prompt))

    outputs = llm.generate(
        [prompt],
        sampling_params,
    )

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    for output in outputs:
        print("request_id:", output.request_id)
        print("prompt_token_ids:", output.prompt_token_ids)
        print("prompt_tokens:", len(output.prompt_token_ids))

        for candidate in output.outputs:
            print("output_token_ids:", candidate.token_ids)
            print("num_output_tokens:", len(candidate.token_ids))
            print("output_text:", repr(candidate.text))


if __name__ == "__main__":
    main()