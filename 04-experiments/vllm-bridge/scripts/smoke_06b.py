from vllm import LLM, SamplingParams
import os
import time


MODEL = "/data/models/Qwen3-0.6B"


def main():
    print("=" * 80)
    print("VB-00 vLLM bridge smoke")
    print("model:", MODEL)
    print("=" * 80)

    llm = LLM(
        model=MODEL,

        # 固定 BF16。
        # BF16 精度略差 范围大
        dtype="bfloat16",

        # 学习阶段固定单 GPU，不引入 TP。
        tensor_parallel_size=1,

        # 手册固定参数。
        block_size=16,
        max_model_len=2048,
        max_num_batched_tokens=256,
        max_num_seqs=4,

         gpu_memory_utilization=0.2,
        # 禁用 CUDA Graph，方便后面 trace / debugger。
        enforce_eager=True,

        # 当前阶段关闭 prefix caching。
        enable_prefix_caching=False,
    )
    print("\n" + "=" * 80)
    print("ENGINE INITIALIZED")
    print("frontend/main PID:", os.getpid())
    print("Sleeping 60 seconds for process inspection...")
    print("=" * 80, flush=True)

    time.sleep(60)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2,
    )

    prompt = (
        "Large language model inference systems use KV cache "
        "to avoid recomputing previous keys and values during "
        "autoregressive decoding. Explain briefly why paged KV "
        "cache is useful for serving multiple requests efficiently."
    )

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
