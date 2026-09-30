<!--
SPDX-FileCopyrightText: 2026 Charles Durham
SPDX-License-Identifier: MIT

MIT License

Copyright (c) 2026 Charles Durham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# CPU FP32 results

**FP32 works well for moderate-accuracy scoring on these datasets. It does not eliminate the need for FP64 verification.** Native CPU tests on mac1 cover the same 134 supported noiseless known-equation systems, with original ICs, observation times and masks. Of these, 120 meet the original 1e-6 sample-MSE gate under accurate FP64 integration; the other 14 have existing source-data discrepancies.

FP32 here means float states, literals, RHS, Jacobians, integrator stages, defect arithmetic and LU solves. **Observation times, step-size control/error norms, observation data and MSE accumulation remain FP64.** There is no GPU execution, reduced-precision observation quantization, fast-math or FMA contraction. This is not a bitwise reproduction of the production GPU scorer.

## Actual benchmark-sample MSE

The main columns count how many of the original **120 FP64-qualified systems** still meet <=1e-6 MSE on every split. No system-specific best-setting selection is used in this table.

| Method and setting | FP64 retained /120 | FP32 retained /120 | FP32 completed /134 |
|---|---:|---:|---:|
| Rosenbrock23, rtol=0.0001 | 98 | 98 | 134 |
| Rosenbrock23, rtol=1e-05 | 106 | 106 | 134 |
| Rosenbrock23, rtol=1e-06 | 111 | 111 | 134 |
| Rosenbrock23, rtol=1e-07 | 117 | 117 | 134 |
| RK4, 1 substeps | 100 | 100 | 128 |
| RK4, 4 substeps | 118 | 118 | 132 |
| RK4, 16 substeps | 120 | 120 | 134 |
| RK4, 32 substeps | 120 | 120 | 134 |
| RK4, 128 substeps | 120 | 115 | 134 |
| RK4, 512 substeps | 120 | 107 | 134 |
| Rosenbrock23, rtol=1e-08 | 120 | 120 | 134 |
| Rosenbrock23, rtol=1e-09 | 120 | 120 | 134 |

**RK4 at 16 or 32 substeps preserves all 120 baseline passes in FP32.** Increasing to 128 loses five of them; 512 loses thirteen. FP64 preserves all 120 at these settings. The finest FP32 steps are demonstrably worse in this implementation because roundoff competes with truncation error.

For example, ODEBench 005 FP32 sample MSE is 4.58e-7 at 16 substeps, 4.77e-7 at 32, 6.49e-6 at 128, and 7.40e-4 at 512. ODEBench 012 goes from 1.90e-7 at 16 to 1.93e-4 at 512. These are errors against the unchanged observations, not against an artificially rounded target.

At RK4/128, ODEBench 015 accidentally crosses below the sample gate in FP32 even though its accurate known-equation MSE exceeds it. Total FP32 passes at that setting are 116, but only 115 are retained baseline passes. That extra pass is numerical error cancellation and must not be used to claim improved integration or requalify the source data.

The coarse RK4 failures occur at the same six systems with one substep and two systems with four substeps in both precisions. They are not additional failures caused by FP32. Rosenbrock23 completes all 134 cases at every tested tolerance, including the tighter diagnostic settings.

## Tight score reproduction and trajectory disagreement

Passing the 1e-6 sample gate is a weaker requirement than reproducing a reference score closely. Using the preceding audit criterion `abs(score-reference_score) <= 1e-8 + 1e-4*abs(reference_score)` on every split, FP32 RK4/32 reproduces 101/134 reference scores, versus 130/134 for FP64. At RK4/512, FP32 reproduces only 67/134, versus 134/134 for FP64.

We also retain pointwise trajectory MSE against FP64 RK4/512 and against a matched-setting FP64 run. The former is a numerical reference, not exact truth. It prevents a low sample MSE caused by cancellation from being mistaken for accurate integration. Neither the sample gate nor these diagnostics establish that FP32 preserves rankings among many nearly tied candidate equations; that search test remains to be done.

## Rosenbrock precision floor and CPU time

At rtol=1e-8, FP32 and FP64 both preserve all 120 baseline sample passes. At 1e-9, the pass count stays 120, but FP32 has 627,934 rejected attempts versus 697 for FP64 across the benchmark. Tighter requested tolerance no longer buys proportional accuracy. These two settings are a precision-floor investigation, not proposed universal defaults.

Same-setting aggregate native CPU integration time, excluding compilation, JSON and Python array preparation:

| Setting | FP64 CPU seconds | FP32 CPU seconds |
|---|---:|---:|
| Rosenbrock23/1e-06 | 0.0555 | 0.0516 |
| RK4/16 | 0.0550 | 0.0483 |
| RK4/32 | 0.1077 | 0.0942 |
| RK4/512 | 1.6453 | 1.4199 |
| Rosenbrock23/1e-08 | 0.1560 | 0.1573 |
| Rosenbrock23/1e-09 | 0.3294 | 0.4297 |

In this sweep FP32 RK4 saves roughly 12–14% CPU time at matched step counts. Rosenbrock23 is about even at 1e-8 and approximately 30% slower in FP32 at 1e-9 due to extra work. These are single-sweep aggregate CPU measurements, not a controlled peak-throughput benchmark or GPU speed prediction.

## True stiff controls

At rtol=1e-6, FP32 Rosenbrock23 reaches 2.47e-11 MSE against the analytic million-to-one decay trajectory with 464 accepted steps. Tightening to 1e-7 increases the error to 1.22e-10.

For Robertson kinetics through t=10,000, FP32 at 1e-6 reaches 2.64e-13 MSE with 620 accepted steps and maximum mass-conservation error 4.98e-7. Its comparison trajectory is native FP64 Rosenbrock23 at 1e-10/1e-14, the same algorithm/settings previously checked against independent Radau. At 1e-7, FP32 Robertson MSE improves to 7.93e-15. These controls support using FP32 for stiff problems at appropriate accuracy; stiffness by itself does not force FP64.

## Implementation and next use

The CPU Solver constructor now accepts `precision="FP32"`; JSON accepts `solver.precision: "FP32"`. The FP32 default is rtol=1e-5, atol=1e-7, selected conservatively and not claimed to pass all datasets. Explicit settings in the report show the accuracy/cost tradeoff. The FP64 default is unchanged. All 134 FP64 RK4/512 sample scores reproduce the previous CPU implementation under a strict regression check. Seven tests pass, including actual float arithmetic, float-state storage, stiff masked integration, large absolute observation times and the existing FP64 regressions.

Use FP32 as a screening option with a measured step/accuracy budget. Promote finalists to FP64, and compare independent step resolutions before accepting a fit. Next investigate compensated state updates and candidate-ranking retention, rather than automatically adding FP32 substeps. No production GPU kernel or search policy was changed.

## Evidence

- [Broad paired sweep](validation/fp32-v1/summary.json)
- [Tighter Rosenbrock diagnostic](validation/fp32-tight-v1/summary.json)
- [All per-system/split scores](validation/fp32-mse-by-system-and-split.csv)
- [Reconciled counts and timings](validation/fp32-reconciled.json)
- [True stiff controls](validation/fp32-stiff-controls.json)
- [Regression tests](validation/tests-fp32.log)
