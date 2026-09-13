#!/usr/bin/env python3
"""P1-M1-T3 real Engine lifecycle integration smoke."""

from __future__ import annotations

import json
from types import MethodType

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.engine.llm_engine import LLMEngine


MODEL = "/data/models/Qwen3-0.6B"
REQUEST_A = "reclaim-a"
REQUEST_B = "reuse-b"


def main() -> None:
    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=MODEL,
            dtype="bfloat16",
            max_model_len=128,
            max_num_seqs=2,
            max_num_batched_tokens=94,
            block_size=16,
            gpu_memory_utilization=0.25,
            enforce_eager=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            disable_log_stats=True,
        ),
        enable_multiprocessing=False,
    )
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    runner = engine.model_executor.driver_worker.model_runner
    pool = scheduler.kv_cache_manager.block_pool

    trace: dict[str, object] = {
        "model": MODEL,
        "request_a_finished": False,
        "request_b_finished": False,
        "plan_injected": False,
        "reclaim_committed": False,
        "worker_transition": None,
        "scheduler_commit": None,
        "reuse": None,
        "schedule_events": [],
    }
    released_ids: set[int] = set()
    request_a_internal: str | None = None
    request_b_internal: str | None = None
    original_schedule = scheduler.schedule
    original_update = scheduler.update_from_output

    def schedule_with_trace(self, *args, **kwargs):
        output = original_schedule(*args, **kwargs)
        event: dict[str, object] = {
            "num_scheduled_tokens": dict(output.num_scheduled_tokens),
            "cached_req_ids": list(output.scheduled_cached_reqs.req_ids),
        }
        if request_b_internal is not None and request_b_internal in output.num_scheduled_tokens:
            b_row = self.kv_cache_manager.get_block_ids(request_b_internal)[0]
            event["request_b_row_after_allocation"] = list(b_row)
            if released_ids and set(b_row) & released_ids:
                trace["reuse"] = {
                    "released_ids": sorted(released_ids),
                    "request_b_row": list(b_row),
                    "reused_ids": sorted(set(b_row) & released_ids),
                }
        transitions = output.scheduled_cached_reqs.reclaim_transitions
        if any(item is not None for item in transitions):
            event["has_reclaim_transition"] = True
        trace["schedule_events"].append(event)
        return output

    def update_with_trace(self, scheduler_output, model_runner_output):
        nonlocal released_ids
        transitions = {
            req_id: transition
            for req_id, transition in zip(
                scheduler_output.scheduled_cached_reqs.req_ids,
                scheduler_output.scheduled_cached_reqs.reclaim_transitions,
            )
            if transition is not None
        }
        transition = transitions.get(request_a_internal)
        before: dict[str, object] | None = None
        if transition is not None:
            assert request_a_internal is not None
            req_index = runner.req_states.req_id_to_index[request_a_internal]
            worker_num_blocks = int(runner.block_tables.num_blocks.np[0, req_index])
            worker_row = (
                runner.block_tables.block_tables[0]
                .gpu[req_index, :worker_num_blocks]
                .cpu()
                .tolist()
            )
            worker_e = int(runner.req_states.effective_kv_len.gpu[req_index].item())
            scheduler_row = self.kv_cache_manager.get_block_ids(request_a_internal)[0]
            before = {
                "worker_row_after_forward": worker_row,
                "worker_effective_kv_len_after_forward": worker_e,
                "scheduler_row_before_commit": list(scheduler_row),
                "scheduler_free_blocks_before_commit": pool.get_num_free_blocks(),
                "scheduled_tokens": scheduler_output.num_scheduled_tokens[
                    request_a_internal
                ],
                "retained_block_ids": list(transition.retained_block_ids),
                "target_effective_kv_len": transition.new_effective_kv_len,
            }
            trace["worker_transition"] = before

        result = original_update(scheduler_output, model_runner_output)

        if transition is not None and request_a_internal in self.requests:
            assert request_a_internal is not None
            request = self.requests[request_a_internal]
            final_row = self.kv_cache_manager.get_block_ids(request_a_internal)[0]
            old_row = before["scheduler_row_before_commit"]
            retained = set(transition.retained_block_ids)
            expected_old = old_row[: transition.expected_old_num_blocks]
            released_ids = {block_id for block_id in expected_old if block_id not in retained}
            trace["scheduler_commit"] = {
                "canonical_row_after_commit": list(final_row),
                "effective_kv_len_after_commit": request.effective_kv_len,
                "free_blocks_after_commit": pool.get_num_free_blocks(),
                "released_ids": sorted(released_ids),
                "released_ref_cnt": {
                    str(block_id): pool.blocks[block_id].ref_cnt
                    for block_id in released_ids
                },
            }
            trace["reclaim_committed"] = True

        if (
            not trace["plan_injected"]
            and request_a_internal in self.requests
            and self.requests[request_a_internal].num_computed_tokens == 94
            and len(self.kv_cache_manager.get_block_ids(request_a_internal)[0]) == 6
        ):
            assert request_a_internal is not None
            old_row = self.kv_cache_manager.get_block_ids(request_a_internal)[0]
            self._set_prepared_reclaim_plan(
                request_a_internal, (0, 1, 4, 5), 62
            )
            trace["plan_injected"] = True
            trace["prepared_old_row"] = list(old_row)
        return result

    scheduler.schedule = MethodType(schedule_with_trace, scheduler)
    scheduler.update_from_output = MethodType(update_with_trace, scheduler)

    request_a_internal = engine.add_request(
        REQUEST_A,
        {"prompt_token_ids": [100] * 94},
        SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
    )

    reuse_request_added = False
    final_outputs: dict[str, object] = {}
    for step in range(128):
        outputs = engine.step()
        for output in outputs:
            final_outputs[output.request_id] = {
                "finished": output.finished,
                "num_output_tokens": len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason,
            }
            if output.request_id == REQUEST_A and output.finished:
                trace["request_a_finished"] = True
            if output.request_id == REQUEST_B and output.finished:
                trace["request_b_finished"] = True
        if trace["reclaim_committed"] and not reuse_request_added:
            request_b_internal = engine.add_request(
                REQUEST_B,
                {"prompt_token_ids": [101]},
                SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
            )
            reuse_request_added = True
        if reuse_request_added and not engine.has_unfinished_requests():
            trace["engine_steps"] = step + 1
            break
    else:
        trace["unfinished_request_ids"] = sorted(scheduler.requests)
        trace["scheduler_request_state"] = {
            req_id: {
                "status": str(request.status),
                "num_computed_tokens": request.num_computed_tokens,
                "num_tokens": request.num_tokens,
                "num_output_tokens": request.num_output_tokens,
                "effective_kv_len": request.effective_kv_len,
            }
            for req_id, request in scheduler.requests.items()
        }
        print("P1_M1_T3_INTEGRATION_CLOSURE_DIAGNOSTIC")
        print(json.dumps(trace, indent=2, sort_keys=True))
        raise RuntimeError("Engine smoke exceeded 128 steps")

    trace["final_outputs"] = final_outputs
    print("P1_M1_T3_INTEGRATION_CLOSURE_TRACE")
    print(json.dumps(trace, indent=2, sort_keys=True))
    assert trace["plan_injected"]
    assert trace["reclaim_committed"]
    worker = trace["worker_transition"]
    commit = trace["scheduler_commit"]
    reuse = trace["reuse"]
    assert worker is not None and commit is not None and reuse is not None
    assert worker["worker_row_after_forward"] == worker["retained_block_ids"]
    assert worker["target_effective_kv_len"] == 62
    assert worker["worker_effective_kv_len_after_forward"] == 63
    assert commit["canonical_row_after_commit"] == worker["retained_block_ids"]
    assert commit["effective_kv_len_after_commit"] == 63
    assert commit["free_blocks_after_commit"] == worker["scheduler_free_blocks_before_commit"] + 2
    assert all(value == 0 for value in commit["released_ref_cnt"].values())
    assert reuse["reused_ids"]
    assert trace["request_a_finished"] and trace["request_b_finished"]
    print("P1_M1_T3_INTEGRATION_CLOSURE_PASS")


if __name__ == "__main__":
    main()
