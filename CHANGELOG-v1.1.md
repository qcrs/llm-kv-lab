# v1.1 Hardening Changelog

## P0 fixes

1. Embedded complete P1/P2 v3 audited baseline snapshots.
2. Added `BASELINE-RESOLUTION.md` to resolve historical contradiction instead of silently rewriting source baseline.
3. Corrected P1 zero-copy row reconstruction: A/B reference-backed, not C reference gap.
4. Demoted explicit `reclaim_generation` from mandatory Core invariant to conditional design draft.
5. Added P1 Qwen3/RoPE positional-semantic gate.
6. Added P1 partial-tail valid-token / append-frontier invariant.

## P1 fixes

7. Renamed P1-G3 to `V1_RUNTIME_CORE_DONE`; clarified it is not full P1 completion.
8. Defined P1-G5 `P1_MAIN_PROJECT_STRONG_DONE` for V2 runtime/Triton/profiling.
9. P2 quantization granularity now `quant_group` until MemoryObj layout is frozen.
10. Scoped exact byte equality to fixed-size Serde + FSL2Adapter and pre-existing-key semantics.
11. Added Supported Configuration Matrix for both projects.
12. Expanded P1/P2 Core design docs substantially.
13. Normalized escaped `#Uxxxx` filenames to real Unicode.

## Control-plane invariants preserved

- P1 active, P2 runtime blocked.
- REVIEW_ONLY.
- Approved Implementation Slice = NONE.
- No vLLM/LMCache source changes.
- No formal benchmark/profiler run.
