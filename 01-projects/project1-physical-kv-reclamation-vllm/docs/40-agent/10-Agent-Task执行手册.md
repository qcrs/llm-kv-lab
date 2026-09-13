# P1 Agent Task Execution Manual

## Source First

每个 TO_VERIFY：rg symbol → definition → caller → callee → owner/state → shape/device → mutation/publication → tests → feature-off。exact local commit 是 patch truth。

## Predict Before Patch

落盘：

- 理解正确时 state/ID/position/output 会怎样；
- 最先出现的三个反例；
- 最小观察点；
- 哪个信号立即停止。

## Minimal Slice

只改 approved allowed files；不重命名大模块、不支持额外 backend、不中途做性能重构。source audit、design approval、patch、benchmark 是不同 Slice 类型。

## Verification Order

~~~text
Source A → Reference B → Correctness C → Runtime D → Evidence E
~~~

上层失败不进入下层。feature-off 等价 native。

## Evidence and State

保存完整 command/env/stdout/raw；结束 diff check、限定 tests、status；更新 artifact index/handoff/PROJECT_STATE 的事实字段。Parent Task 仍 NOT CLOSED，除非单独 Gate review。

## Failure

source mismatch、oracle失败、cross-request污染、duplicate free、freed page live、nonfinite/OOB、unexplained sync/memory growth、claim不被数据支持时停止。不得删失败或自行换 architecture。

## Output to User

Task/Slice status、facts、files、tests、prediction、failures、evidence、what not done、review questions、only recommended next（not executed）。

