# P1 Source Identity

## Canonical V1 Checkpoint

```yaml
accepted_commit: cd444b72ceecb2496bf015baf8c18bf883151fa1
accepted_tag: p1-v1-core-accepted
accepted_branch: p1/physical-kv-reclaim-v026
checkpoint_worktree: CLEAN
upstream_parent: 568afb3a13806beb53bb2e6bd518269357b237c0
```

该 checkpoint 是当前 P1 V1/Core 的 immutable restore point。下方 baseline/topology
字段保留 upstream provenance；不应将 upstream HEAD 误认为完整 P1 implementation。

## Expected Baseline

Repository：vllm-project/vllm
Version/commit：v0.26.0 / 568afb3a13806beb53bb2e6bd518269357b237c0

## Current Topology

```yaml
implementation_worktree: /home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim
branch: p1/physical-kv-reclaim-v026
head: cd444b72ceecb2496bf015baf8c18bf883151fa1
study_tree: /home/qcrs/learning/llm-kv-lab/third_party/vllm
study_branch: study-vllm-0.26
editable_python_source: /home/qcrs/learning/llm-kv-lab/worktrees/p1-vllm-reclaim/vllm
native_mode: PRECOMPILED_REUSE
native_source_seed: /home/qcrs/learning/llm-kv-lab/third_party/vllm/vllm
```

P1 worktree is clean at creation. The study tree remains dirty/read-only for
implementation. See `NATIVE-ARTIFACTS.md` for copied binary provenance.

## Historical Pre-Checkpoint Notes

- project root；
- upstream clone；
- Last Good Commit：已由 `p1-v1-core-accepted` 固化。

## Roles

implementation worktree = only approved patch target。
study tree = read-only，保留用户修改。
Tangram/Sparse/newer vLLM = reference only，never implementation source。

## Before Any Patch

git HEAD/status/worktree list；import vllm.__file__；native module paths；compare PROJECT_STATE。任一不一致停止。

Python, Triton JIT, and native C++/CUDA artifacts have separate lifecycles:
Python changes require process restart; Triton changes may regenerate JIT cache;
C++/CUDA changes require `NATIVE_REBUILD_REQUIRED` and Web/User review.
