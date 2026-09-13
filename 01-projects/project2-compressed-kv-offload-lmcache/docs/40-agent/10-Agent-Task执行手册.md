# P2 Agent Task Manual

## Start

检查P1 unlock、P2 PROJECT_STATE、both source identities、torch/native ABI、current Parent/Slice。任何缺失保持REVIEW_ONLY。

## Source First

same-commit code与docs一起读；记录MemoryObj owner/device/layout、temp lifecycle、completion、actual file write。later snapshot/reference不能patch truth。

## Predict

在patch前预测payload/scale/metadata bytes、source/dst device/copies、file/report size、error/tolerance、failure signals。

## Minimal

connector、raw FS、codec unit、Serde integration、E2E、Triton、NSYS是不同Slices。不得重写async/controller或把V3顺手做。

## Verify

Source→ABI/reference→format oracle→object lifecycle→E2E→micro→profiler→conclusion。

## Stop

ABI silent fallback、wrong worktree、size invariant mismatch、temp leak、partial publication、slot corruption、generic Serde被误写PCIe、unapproved external backend、user dirty conflict。

## Handoff

facts/drafts/approved、files/tests/failures、exact bytes、device/copies、what not done、Web questions、Parent NOT CLOSED。

