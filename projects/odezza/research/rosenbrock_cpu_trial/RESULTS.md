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

# CPU benchmark MSE reproduction

Native Rosenbrock23 and RK4 run on **mac1 CPU**, with no CUDA dependency. Both reproduce the accurate-reference MSEs against the actual benchmark samples for **134/134 supported noiseless source systems** at tested settings. **120/134** also meet the original MSE <=1e-6 data-agreement gate; the same 14 remain above it.

Independent Radau, BDF and LSODA checks run on **rack1 CPUs**, using its existing SciPy installation. Each reproduces the accurate-reference sample MSEs for all 14 difficult systems at a tested setting. BDF needs a different tolerance on one case, detailed below. No GPU is used in these new runs.

“Reproduces” means every split satisfies `abs(MSE - reference_MSE) <= 1e-8 + 1e-4*abs(reference_MSE)`. This checks score reproduction, not exact trajectory equality. The separate original sample-fit gate remains 1e-6.

## MSE against the published observations

Every cell is **worst split MSE against samples**. Native RK4 uses 512 substeps; other columns use their tightest successfully completed tested tolerance. The CSV retains every split and the selected setting. All sweeps and failures remain in the raw reports.

| System | CPU RK4 | CPU Rosenbrock23 | CPU Radau | CPU BDF | CPU LSODA | Source-recipe LSODA |
|---|---:|---:|---:|---:|---:|---:|
| biological-metabol1 | 1.02573e-05 | 1.02573e-05 | 1.02573e-05 | 1.02573e-05 | 1.02573e-05 | not established |
| mdbench-008-clean | 0.000969715 | 0.000969689 | 0.000969715 | 0.000969713 | 0.000969715 | 0 |
| mdbench-015-clean | 1.69769e-06 | 1.69748e-06 | 1.69769e-06 | 1.69769e-06 | 1.69769e-06 | 0 |
| mdbench-055-clean | 0.0769172 | 0.0769151 | 0.0769172 | 0.0769169 | 0.0769172 | 0 |
| mdbench-056-clean | 30.354 | 30.3549 | 30.354 | 30.354 | 30.3539 | 0 |
| mdbench-059-clean | 2.58749e-06 | 2.58857e-06 | 2.58749e-06 | 2.5875e-06 | 2.58749e-06 | 0 |
| mdbench-061-clean | 0.641062 | 0.64107 | 0.641062 | 0.641061 | 0.641062 | 0 |
| odebench-008-c0 | 0.000936462 | 0.000936437 | 0.000936462 | 0.00093646 | 0.000936462 | 4.59619e-13 |
| odebench-015-c0 | 1.3036e-06 | 1.30344e-06 | 1.3036e-06 | 1.3036e-06 | 1.3036e-06 | 6.17898e-10 |
| odebench-032-c0 | 0.000196913 | 0.000196924 | 0.000196913 | 0.000196912 | 0.000196912 | 3.1152e-08 |
| odebench-055-c0 | 468.721 | 468.721 | 468.721 | 468.721 | 468.721 | 379.376 |
| odebench-056-c0 | 28.9349 | 28.9358 | 28.9349 | 28.9349 | 28.9349 | 0.000263962 |
| odebench-059-c0 | 4.46954e-05 | 4.46964e-05 | 4.46954e-05 | 4.46955e-05 | 4.46954e-05 | 1.15814e-13 |
| odebench-061-c0 | 0.667745 | 0.667747 | 0.667745 | 0.667745 | 0.667745 | 0.000914268 |

The fresh source-recipe replay reproduces all six flagged MDBench systems **exactly: MSE 0 on every split**. It uses the published loose LSODA settings and symbolic substitution recipe, rather than the tight canonical-model integrations in the other columns. The chaotic ODEBench replay still differs from the samples; historical generator provenance is unresolved. The metabol1 source/model discrepancy also remains unresolved.

## CPU cost and completion

The native mac1 audit covered 809 solver/settings combinations in **32.78 s elapsed**, including compilation and reporting. Native integration consumed **8.50 s CPU time** in total (6.12 s Rosenbrock23, 2.39 s RK4). All 809 native settings completed successfully. Compilation was 5.32 s.

The difficult Lorenz settings in this native audit took approximately 0.1–0.5 s with tight Rosenbrock23 and 0.003–0.011 s with RK4/512. These are actual dataset batches, not saturated throughput comparisons or an isolated same-machine CPU/GPU speed ratio.

The independent initial SciPy check took 46.05 s with two CPU processes on rack1, SciPy 1.18.1. Its Python RHS callbacks, host and algorithms differ from the native mac1 run.

BDF on ODEBench 055 fails at rtol=1e-12 and 1e-13 with “Required step size is less than spacing between numbers.” The recorded targeted retry succeeds at 1e-10 and 1e-11, reproducing the reference MSE on every split. This failure is retained, not counted as a successful 1e-12 integration.

Five native CPU regression tests pass: stiff analytic trajectories with masking, fourth-order RK4 convergence, analytic Jacobians, explicit failures/input rejection, and ragged score-only requests. The JSON entry point was exercised on mac1. Native scores are independently recomputed in Python.

## Evidence and implications

- [All 134 native CPU cases](validation/all-clean-v1/summary.json)
- [Per-system, per-split MSE CSV](validation/mse-by-system-and-split.csv)
- [Independent CPU Radau/BDF/LSODA](validation/scipy-v2/summary.json)
- [Explicit BDF retry and failures](validation/bdf055-v1/summary.json)
- [Fresh generator replay](validation/source-replay-v1.json)
- [Tests](validation/tests-final.log)
- [CPU-only implementation and commands](README.md)

The implementation can reproduce the benchmark MSEs. A high nonzero MSE for the known equation is a data-generation/model/objective issue that a more accurate integrator does not necessarily repair. Keep the original observations and qualify those cases separately before treating them as recovery failures. All new native arithmetic here is FP64; FP32 and a higher-order Rosenbrock method remain separate future experiments.
