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

# Smaller-state cooperative LM scaling

Measured on rack1 (RTX 5080), 2026-09-09. One thread becomes preferable on the large three-state/three-parameter control, but six states is not an automatic one-thread cutoff. At six states, two lanes win with three fitted coefficients and four lanes win with six coefficients on the 32,768-start bank.

## Matched kernel timings

Milliseconds, mean of the last two of three identical launches. Compilation, transfer and validation are excluded.

| Fits | States | Fitted coefficients | 1 lane | 2 lanes | 4 lanes | 8 lanes |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 3 | 3 | 19.488 | 7.542 | 5.346 | 5.152 |
| 1,024 | 3 | 6 | 213.862 | 126.234 * | 76.090 | 74.298 |
| 1,024 | 6 | 3 | 22.882 | 9.109 | 8.798 | 8.759 |
| 1,024 | 6 | 6 | 301.948 | 192.619 | 96.505 | 100.360 |
| 32,768 | 3 | 3 | 28.305 | 30.404 | 38.300 | 74.589 |
| 32,768 | 3 | 6 | 418.745 | 326.932 * | 354.262 | 358.278 |
| 32,768 | 6 | 3 | 54.100 | 47.346 | 65.860 | 121.973 |
| 32,768 | 6 | 6 | 740.964 | 559.193 | 475.942 | 517.647 |

* The two-lane three-state/six-parameter cells are non-equivalent optimizer paths; exclude them from a topology-only speed recommendation. Details below. All other within-case widths return exactly equal coefficients, initial/final MSE and all three optimizer counters for every start.

All 32 cells pass independent analytic derivative/primal probes and their training-best candidate passes independent FP64 training and validation checks (aggregate and every observed-state MSE <=1e-10, DOP853/Radau agreement <=1e-8). No final test was consumed.

## Six-state resources

| Coefficients | Lanes | Registers/thread | Local bytes/thread |
|---:|---:|---:|---:|
| 3 | 1 | 154 | 0 |
| 3 | 2 | 128 | 0 |
| 3 | 4 | 80 | 0 |
| 3 | 8 | 80 | 0 |
| 6 | 1 | 254 | 0 |
| 6 | 2 | 167 | 0 |
| 6 | 4 | 142 | 0 |
| 6 | 8 | 96 | 0 |

There is no local-memory allocation for any tested smaller-state shape. Cooperative improvements therefore cannot be attributed to eliminating spills here. One-lane six-state/six-parameter code nevertheless uses 254 registers/thread; extra lanes divide sensitivity/optimizer storage and work, and expose more independent scheduling opportunities.

## Workload and interpretation

These extend the same fixed nonlinear control used previously. For three states, state references are reduced modulo three; this changes dependencies, so cross-state timings are not a pure dimension-only experiment. In the three-parameter control the inner coefficients 0.7, -0.6 and 0.2 are known literals instead of fitting slots. Each state-count pair uses the same true dynamics/initial-condition seed, with different sets of coefficients fitted. Each width receives identical observations and starts; the 1,024-start bank is the prefix of the 32,768-start bank. Four training trajectories, 21 times each, 32 RK4 steps per interval and at most 32 LM iterations are unchanged.

At 1,024 starts, one lane launches only 32 CTAs; the cooperative versions launch 64/128/256. At 32,768 starts, one lane launches 1,024 CTAs, and the return on extra lanes changes substantially. This is evidence for batch-size-dependent dispatch, not proof that all shapes have reached peak throughput. No occupancy profiler or clock locking was used. Width order was ascending in the small bank and descending in the large bank; each width has two warm samples, not a statistical performance study. Small differences should be treated cautiously.

For these controls, prefer one lane for large three-state/three-coefficient banks, two for large six-state/three-coefficient banks, and four for six-state/six-coefficient banks. Retain one lane as a supported topology, but dispatch on state count, fitted coefficient count, expression resources and number of independent fits. Do not install a blanket six-state threshold. The separate plain-CUDA prototype remains distinct from production specialized LM; these measurements do not establish a production dispatch policy.

## Active deviation: three states, six coefficients, two lanes

The full-array audit found a difference despite all derivative probes and selected-candidate validation passing. At 1,024 starts, 1,007 final scores differ and 27 iteration counts differ. At 32,768 starts, 32,163 final scores differ and 954 iteration counts differ. Maximum score differences are 1.04e-5 and 1.08e-4 respectively; maximum coefficient differences reach 0.694 and 1.350. The differences are confined to width two in this matrix.

Independent FP64 replay confirms the most divergent pair represents different fitted solutions, not merely different reported GPU MSE. At the large-bank fit 31,134: one lane reports 5.819698e-5, replay 5.819678e-5; two lanes reports 1.658840e-4, replay 1.658841e-4. Both are unsuccessful starts even though the bank winner validates. Cause is not isolated; do not label this harmless roundoff or equal numerical work.

Mitigation: retain the outputs and timings, flag the affected cells, and exclude width two for this case from topology-only recommendations. Next action in the fitting trial is per-iteration sensitivity/normal-equation/proposal comparison before changing production or expanding that shape. No kernel or acceptance threshold was changed in response.

## Evidence

- [1,024-start results](../validation/plain-cuda-lm-small-1024/results.json)
- [1,024-start output audit](../validation/plain-cuda-lm-small-1024/audit.json)
- [32,768-start results](../validation/plain-cuda-lm-small-32768/results.json)
- [32,768-start audit and independent replay](../validation/plain-cuda-lm-small-32768/audit.json)

Raw sources, cubins, inputs and every returned fit remain in the corresponding rack1 folders under `/home/cdurham/odezza/scratch/fitting_batch_trial/validation/`. Only calibration generation/harness code changed; the tested CUDA kernel and production code were unchanged.
