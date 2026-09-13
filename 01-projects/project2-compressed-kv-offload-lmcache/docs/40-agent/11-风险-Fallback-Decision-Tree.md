# P2 Risk and Fallback Tree

P1 not unlocked → remain blocked。

ABI mismatch/silent fallback → BLOCKED_ENV；不换later commit强装。

Connector fail → tracker/config/chunk/hit provenance；codec untouched。

Raw FS fail → key/path/permission/controller/adapter；Serde untouched。

Codec unit fail → math/layout/offset/scale/estimator。

Unit pass but file bytes mismatch → temp allocation/padding/backend writes；do not benchmark。

File/tensor correct but generation corrupt → deserialize dtype/shape/slot mapping。

Triton CPU staging negative → PyTorch/CPU V1 default；Triton benchmark path。

Generic Serde no PCIe reduction → expected; report placement；pre-D2H staysV3。

V3 owner/lifetime/five gates fail → skip with reason，raw connector fallback。

Performance negative → preserve crossover/negative regime；project still valid。

