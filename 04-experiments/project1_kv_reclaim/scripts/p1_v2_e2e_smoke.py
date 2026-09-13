#!/usr/bin/env python3

"""
P1 V2 real Engine compaction and next-step append smoke.

这个脚本的目的：

1. 真正启动一个 vLLM LLMEngine。
2. 真正跑一个 32-token prompt。
3. 等 prompt KV 建好之后，人为向 Scheduler 注入一个 V2 compaction plan。
4. 下一次 engine.step()：
      Scheduler 正常 schedule
      → Worker 正常 model forward
      → forward 后真正执行 V2 KV compaction
      → Worker 返回 CompactionResultData
      → Scheduler 正常 update_from_output()
      → ownership truncate + block free
5. 后面继续 engine.step()，验证 compact 后 request 还能继续生成并 finish。

注意：
这里测试的是“V2 runtime mechanism 是否闭环”，
不是测试 keep_member_indices 这个 retention policy 是否合理。
"""

from __future__ import annotations

import json

# MethodType 用来把一个普通 Python 函数，
# 动态绑定成某个 object 的 method。
#
# 后面：
#
# scheduler.schedule = MethodType(schedule_trace, scheduler)
#
# 就是在运行时把：
#
# scheduler.schedule()
#
# 替换成我们自己的 schedule_trace()，
# 但 schedule_trace() 内部仍然会调用真正的 original_schedule()。
from types import MethodType

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.engine.llm_engine import LLMEngine


# 真正加载的模型。
MODEL = "/data/models/Qwen3-0.6B"

# 外部 request ID。
REQUEST = "compact-b"


def main() -> None:

    # ============================================================
    # 1. 启动一个真正的 vLLM Engine
    # ============================================================

    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=MODEL,
            dtype="bfloat16",

            # 最大 context 长度。
            max_model_len=128,

            # 测试只允许同时 1 个 sequence，
            # 把并发因素排除掉。
            max_num_seqs=1,

            # 一个 scheduler step 最多调度 32 tokens。
            #
            # prompt 正好也是 32 tokens，
            # 所以第一个主要 step 可以把 prompt 做完。
            max_num_batched_tokens=32,

            # 每一个 physical KV block 存 16 个 token/member。
            #
            # 所以：
            # 32 tokens = 2 blocks
            # 33 tokens = 3 blocks
            block_size=16,

            gpu_memory_utilization=0.25,

            # eager mode：
            # 不使用 CUDA Graph，
            # 减少当前 functional-core 测试变量。
            enforce_eager=True,

            # 当前 V2 baseline 不测 prefix cache。
            enable_prefix_caching=False,

            # 当前 V2 baseline 是同步 scheduler。
            async_scheduling=False,

            disable_log_stats=True,
        ),

        # 很关键：
        #
        # EngineCore 不放到另一个 process。
        #
        # 这样测试脚本可以直接拿到：
        # scheduler
        # model_runner
        # block_pool
        #
        # 并 monkey-patch scheduler method。
        enable_multiprocessing=False,
    )

    # ============================================================
    # 2. 从真实 Engine 内部拿到对象
    # ============================================================

    # LLMEngine
    #   ↓
    # EngineCore client
    #   ↓
    # in-process EngineCore
    core = engine.engine_core.engine_core

    # 真正的 Scheduler。
    scheduler = core.scheduler

    # 真正的 GPUModelRunner。
    #
    # 这个脚本后面其实没有直接调用 runner，
    # 它存在主要是确认我们拿到的是生产 runtime 对象。
    #
    # 真正 model forward 仍然通过 engine.step() 触发。
    runner = core.model_executor.driver_worker.model_runner

    # Scheduler 所使用的真实 KV BlockPool。
    #
    # 后面用它观察：
    #
    # free blocks 数量
    # 是否真的因为 compaction 被释放。
    pool = scheduler.kv_cache_manager.block_pool

    # ============================================================
    # 3. 测试自己的状态变量
    # ============================================================

    # engine.add_request() 之后会得到内部 request id。
    #
    # 定义在这里，是因为下面的 schedule_trace/update_trace
    # closure 也要访问它。
    internal_id: str | None = None

    # V2 Plan 是否已经人为注入过。
    #
    # 避免每一个 step 都重复注入。
    injected = False

    # Scheduler 是否已经成功收到 Worker compaction result
    # 并完成 reconciliation。
    committed = False

    # ============================================================
    # 4. trace 只是测试记录，不参与生产逻辑
    # ============================================================

    trace: dict[str, object] = {

        "runtime": {
            "scheduler_class": type(scheduler).__name__,

            "async_scheduling":
                scheduler.vllm_config.scheduler_config.async_scheduling,

            "use_v2_model_runner":
                scheduler.vllm_config.use_v2_model_runner,

            "pipeline_parallel_size":
                scheduler.vllm_config.parallel_config.pipeline_parallel_size,

            "tensor_parallel_size":
                scheduler.vllm_config.parallel_config.tensor_parallel_size,

            "data_parallel_size":
                scheduler.vllm_config.parallel_config.data_parallel_size,

            "decode_context_parallel_size":
                scheduler.vllm_config.parallel_config.decode_context_parallel_size,

            # 注意：
            # 这里字段名字写成 max_concurrent_batches，
            # 但实际拿的是 max_num_seqs。
            #
            # 这是 instrumentation label 不准确，
            # 不影响 runtime correctness。
            "max_concurrent_batches":
                scheduler.vllm_config.scheduler_config.max_num_seqs,

            "connector": (
                None
                if scheduler.vllm_config.kv_transfer_config is None
                else scheduler.vllm_config.kv_transfer_config.kv_connector
            ),
        },

        # 每一次 schedule() 后观察到的状态。
        "schedule_events": [],

        # V2 compaction commit 后的状态。
        "compaction": None,

        # compaction 后 continuation state。
        "next_step": None,
    }

    # ============================================================
    # 5. 保存真正的 Scheduler 方法
    # ============================================================

    # 很关键：
    #
    # 后面虽然会 monkey-patch scheduler.schedule，
    # 但我们仍然需要调用原始 Scheduler.schedule()。
    original_schedule = scheduler.schedule

    # 同理保存真正的 update_from_output()。
    original_update = scheduler.update_from_output

    # ============================================================
    # 6. 包装 Scheduler.schedule()
    # ============================================================

    def schedule_trace(self, *args, **kwargs):

        # --------------------------------------------------------
        # 最重要：
        #
        # 先调用真正的 Scheduler.schedule()
        #
        # 我们不是 fake schedule。
        # --------------------------------------------------------

        output = original_schedule(*args, **kwargs)

        # output 就是真实 SchedulerOutput。

        event: dict[str, object] = {

            # 当前 step 每个 request schedule 了几个 token。
            "num_scheduled_tokens":
                dict(output.num_scheduled_tokens),

            # 当前 cached request IDs。
            "cached_req_ids":
                list(output.scheduled_cached_reqs.req_ids),
        }

        # 如果目标 request 这一 step 真正进入 scheduled set，
        # 就记录它的 Scheduler canonical state。
        if (
            internal_id is not None
            and internal_id in output.num_scheduled_tokens
        ):

            # Scheduler canonical block ownership。
            #
            # 例如：
            #
            # [1, 2]
            #
            # 下一步 append 跨 block boundary 后：
            #
            # [1, 2, 3]
            row = self.kv_cache_manager.get_block_ids(
                internal_id
            )[0]

            event["canonical_row"] = list(row)

            # Scheduler persistent physical frontier E。
            event["effective_kv_len"] = (
                self.requests[internal_id].effective_kv_len
            )

            # Scheduler logical progress L。
            event["num_computed_tokens"] = (
                self.requests[internal_id].num_computed_tokens
            )

            # 当前 allocator 还有多少 free KV blocks。
            event["free_blocks"] = (
                pool.get_num_free_blocks()
            )

        # 如果当前 SchedulerOutput 真正携带了 V2 Plan，
        # 把它记录下来。
        #
        # 这能证明：
        #
        # _set_prepared_compaction_plan()
        #   ↓
        # _prepared_compaction_plans
        #   ↓
        # schedule()
        #   ↓
        # SchedulerOutput.compaction_plans
        #
        # 这条真实 transport 路走通了。
        if output.compaction_plans:
            event["compaction_plan"] = repr(
                output.compaction_plans
            )

        trace["schedule_events"].append(event)

        # 返回真正的 SchedulerOutput。
        return output

    # ============================================================
    # 7. 包装 Scheduler.update_from_output()
    # ============================================================

    def update_trace(
        self,
        scheduler_output,
        model_runner_output,
    ):

        # 因为下面要修改 main() 作用域中的：
        #
        # injected
        # committed
        #
        # 所以 Python 需要 nonlocal。
        nonlocal injected, committed

        # --------------------------------------------------------
        # 同样最重要：
        #
        # 先执行真正的 Scheduler.update_from_output()
        #
        # 所以后面观察到的 canonical row / E / free blocks
        # 都是 reconciliation COMMIT 之后的状态。
        # --------------------------------------------------------

        result = original_update(
            scheduler_output,
            model_runner_output,
        )

        # ========================================================
        # 7A. 如果 Worker 这一轮返回了 CompactionResultData
        # ========================================================

        if model_runner_output.compaction_results:

            # 当前测试只有一个 request，
            # 所以直接取第一个 compaction result。
            comp = (
                model_runner_output.compaction_results[0]
            )

            # update_from_output() 已经执行完成。
            #
            # 所以此时取得的 row 是 Scheduler commit 后的
            # canonical ownership。
            row = self.kv_cache_manager.get_block_ids(
                comp.request_id
            )[0]

            req = self.requests.get(comp.request_id)

            trace["compaction"] = {

                # Worker 报告：
                #
                # compact 前 post-forward physical E。
                "source_E":
                    comp.expected_source_effective_kv_len,

                # compact 后的新 E。
                #
                # 当前 keep 有 5 个 member，所以 K=5。
                "K":
                    comp.new_effective_kv_len,

                # compact 前 3 blocks。
                "old_num_blocks":
                    comp.expected_source_num_blocks,

                # compact 后只需要 1 block。
                "new_num_blocks":
                    comp.new_num_blocks,

                # Scheduler reconciliation 后真正的 canonical row。
                #
                # 期望：
                #
                # [1, 2, 3]
                #     ↓
                # [1]
                "canonical_row_after_commit":
                    list(row),

                # 两个 trailing blocks free 后的 allocator capacity。
                "free_blocks_after":
                    pool.get_num_free_blocks(),

                # Scheduler persistent E。
                #
                # 期望：
                # E = 5
                "effective_kv_len_after_commit":
                    (
                        None
                        if req is None
                        else req.effective_kv_len
                    ),

                "result_step_seq":
                    comp.step_seq,
            }

            # 标记：
            # V2 Worker→Scheduler transaction 已经成功 commit。
            committed = True

        # ========================================================
        # 7B. 在适当时机人为注入一次 V2 Plan
        # ========================================================

        if (
            # 只注入一次。
            not injected

            # request 还在 Scheduler 里。
            and internal_id in self.requests

            # prompt 的 32 token 已经至少执行完成。
            and self.requests[
                internal_id
            ].num_computed_tokens >= 32

            # 当前 step 还不是 compaction step。
            #
            # 防止刚处理完 compaction result 又再注入一次。
            and not model_runner_output.compaction_results
        ):

            # 当前 Scheduler canonical row。
            #
            # 正常此时：
            #
            # 32 token / block_size 16
            #
            # row ≈ [1, 2]
            row = self.kv_cache_manager.get_block_ids(
                internal_id
            )[0]

            req = self.requests[internal_id]

            # ----------------------------------------------------
            # 计算“下一次 post-forward”预期 source E。
            # ----------------------------------------------------
            #
            # 当前已经有：
            #
            # E = 32
            #
            # 下一 decode step 会 schedule q=1。
            #
            # V2 compaction 是 post-forward 执行，
            # 因而 source 包含这个新 token：
            #
            # source_E = 32 + 1 = 33
            #
            # 如果此前已经有 explicit physical E，
            # 使用 effective_kv_len。
            #
            # 否则 physical == logical，
            # fallback 到 num_computed_tokens。
            source_e = int(
                req.effective_kv_len
                if req.effective_kv_len is not None
                else req.num_computed_tokens
            ) + 1

            # ----------------------------------------------------
            # P1 自己增加的 explicit V2 plan ingress
            # ----------------------------------------------------
            #
            # 注意：
            #
            # 这不是 upstream 的公开用户 API。
            #
            # 它是我们 P1 在 Scheduler 中加入的
            # policy-neutral/private seam。
            #
            # 测试通过它模拟：
            #
            # “假设未来 retention policy 已经决定
            #  要保留哪些 KV member”
            self._set_prepared_compaction_plan(

                internal_id,

                # 这是人为选择的 keep set。
                #
                # 它们是：
                #
                # 当前 physical member indices
                #
                # 不是 logical token ID，
                # 也不是 physical block ID。
                [0, 8, 16, 24, 31],

                # 下一 forward 后预期 E=33。
                source_e,

                # ceil(source_e / 16)
                #
                # source_e=33：
                #
                # ceil(33/16)=3
                #
                # 即 Worker 应该看到 3 个 source pages。
                (source_e + 15) // 16,

                # 当前 MVP 手工指定一次 transaction seq。
                step_seq=1,
            )

            # 记录 Plan 注入时的状态。
            trace["plan"] = {

                # 33
                "source_E":
                    source_e,

                "keep_member_indices":
                    [0, 8, 16, 24, 31],

                # 注意：
                #
                # 注入 Plan 的这一刻，
                # row 还是当前 step 的：
                #
                # [1,2]
                #
                # Plan 描述的是“下一 step post-forward”
                # 预期 source：
                #
                # [1,2,3]
                #
                # 所以这里 old_row=2 blocks
                # 和 expected_source_num_blocks=3
                # 并不冲突。
                "old_row":
                    list(row),

                "free_blocks_before":
                    pool.get_num_free_blocks(),
            }

            injected = True

        return result

    # ============================================================
    # 8. Python monkey patch
    # ============================================================

    # 原来：
    #
    # scheduler.schedule → Scheduler.schedule
    #
    # 现在：
    #
    # scheduler.schedule → schedule_trace
    #
    # schedule_trace 内部仍然调用 original_schedule。
    scheduler.schedule = MethodType(
        schedule_trace,
        scheduler,
    )

    # 同理：
    #
    # Scheduler update_from_output
    # 外面套一层观察逻辑。
    scheduler.update_from_output = MethodType(
        update_trace,
        scheduler,
    )

    # ============================================================
    # 9. 添加一个真正的 request
    # ============================================================

    internal_id = engine.add_request(

        REQUEST,

        # 不经过 tokenizer，
        # 直接给 32 个 token IDs。
        #
        # 所以 prompt length 精确等于 32。
        {
            "prompt_token_ids":
                [100] * 32
        },

        SamplingParams(

            # greedy-ish deterministic generation。
            temperature=0.0,

            # 为什么是 3？
            #
            # 因为不能 compact 完马上 request 就结束。
            #
            # 我们还想让它继续跑后续 decode，
            # 验证 compacted KV state 能继续使用。
            max_tokens=3,

            # 不因为模型提前产生 EOS 而影响测试。
            ignore_eos=True,
        ),
    )

    # ============================================================
    # 10. 真正不断调用 Engine.step()
    # ============================================================

    outputs_seen = 0

    # 最多跑 16 step，防止出 bug 后无限循环。
    for step in range(16):

        # --------------------------------------------------------
        # 这是整个测试最重要的一句。
        # --------------------------------------------------------
        #
        # 它触发真正的 vLLM：
        #
        # Scheduler.schedule()
        #     ↓
        # ModelRunner execute_model()
        #     ↓
        # real Qwen forward
        #     ↓
        # V2 post-forward compaction
        #     ↓
        # sample_tokens()
        #     ↓
        # ModelRunnerOutput
        #     ↓
        # Scheduler.update_from_output()
        #     ↓
        # Engine output
        #
        # 测试没有自己手工调用这些函数。
        outputs = engine.step()

        outputs_seen += len(outputs)

        # 如果 compaction 已经 commit，
        # 记录 commit 后 request 的状态。
        #
        # 注意：
        # “next_step” 这个名字不是特别精确。
        #
        # 第一次进入这里时，
        # 更严格地说是：
        #
        # “compaction step 刚结束后的 persistent state”
        #
        # 不等价于直接观测到了
        # 下一 forward 的 cache_position。
        if committed and trace["next_step"] is None:

            req = scheduler.requests.get(internal_id)

            if req is not None:

                row = (
                    scheduler.kv_cache_manager
                    .get_block_ids(internal_id)[0]
                )

                trace["next_step"] = {

                    "step":
                        step + 1,

                    # 应为 5。
                    "effective_kv_len":
                        req.effective_kv_len,

                    # logical counter 不会被压成5。
                    "num_computed_tokens":
                        req.num_computed_tokens,

                    # canonical row 应为 [1]。
                    "canonical_row":
                        list(row),

                    "outputs_seen":
                        outputs_seen,
                }

        # request 真正完成以后退出。
        if outputs and outputs[0].finished:
            break

    # ============================================================
    # 11. 最终确认真实 request finish
    # ============================================================

    trace["finished"] = bool(
        outputs
        and outputs[0].finished
    )

    # 打印完整 trace，方便保存 raw evidence。
    print("P1_V2_E2E_TRACE")

    print(
        json.dumps(
            trace,
            indent=2,
            sort_keys=True,
        )
    )

    # ============================================================
    # 12. Acceptance Assertions
    # ============================================================

    # 确实注入过 Plan。
    assert injected

    # Worker 确实返回过 Result，
    # Scheduler 确实处理过 commit。
    assert committed

    comp = trace["compaction"]

    # forward 后真实 physical source = 33。
    assert comp["source_E"] == 33

    # 保留5个 member。
    assert comp["K"] == 5

    # 33 tokens 需要3个16-token pages。
    assert comp["old_num_blocks"] == 3

    # compact 后5 tokens只需要1页。
    assert comp["new_num_blocks"] == 1

    # Scheduler commit 后 persistent physical E=5。
    assert (
        comp["effective_kv_len_after_commit"]
        == 5
    )

    nxt = trace["next_step"]

    # compaction 后 persistent E 保持5。
    assert (
        nxt is not None
        and nxt["effective_kv_len"] == 5
    )

    # request 最后真的可以继续运行并结束。
    assert trace["finished"]

    print("P1_V2_E2E_PASS")


if __name__ == "__main__":
    main()