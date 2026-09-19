# P1-V2-DESIGN-REVIEW-01 — Submission Pack

Recommended destination:

```text
01-projects/project1-physical-kv-reclamation-vllm/
docs/10-design/ragged-kv-runtime-v5/
```

Included files:

```text
22-R1-Exact-Engineering-Contract-v6-Architecture-Refreshed.md
38-Scheduler-Side-Ragged-Physical-State-and-Allocation-Contract-v2.md
39-Ragged-Step-Metadata-Transport-and-Attention-Contract-v2.md
41-v6-Implementation-Slices-File-Ownership-and-Dependency-DAG.md
43-P1-V2-DESIGN-REVIEW-01-Ragged-Runtime-Architecture-Freeze.md
44-R1-A2-Exact-Implementation-Contract.md
45-R1-B1-Exact-Implementation-Contract.md
46-R1-B2-Exact-Implementation-Contract.md
```

The repository `PROJECT_STATE.md` currently identifies `P1-V2-DESIGN-REVIEW-01` as the next allowed action and keeps V1/Core frozen. This pack intentionally does not rewrite project state.

After these design documents are accepted/committed, project state can be updated separately to record:

```text
P1-V2-DESIGN-REVIEW-01 = ACCEPTED / FROZEN
next implementation contracts = A2 / B1 / B2
production Ragged activation = still forbidden until R1-E
```

Suggested docs commit:

```text
docs(p1): freeze ragged runtime namespace ownership transport activation design
```
