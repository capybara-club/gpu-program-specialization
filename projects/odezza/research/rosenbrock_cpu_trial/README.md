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

# CPU integration and benchmark MSE reproduction

Latest precision study: [paired CPU FP32/FP64 results](FP32_RESULTS.md).

This is a native CPU implementation of Rosenbrock23 and RK4, using the safe
generated RHS and analytic Jacobian. It runs on mac1 with the system C++17
compiler and standard-library Python. No CUDA, GPU, NumPy, or SciPy is needed
for either native method, the JSON entry point, or the benchmark audit.

The shared Rosenbrock numerical header now uses empty host/device annotations
when compiled by a normal C++ compiler. Its FP64 equations are unchanged.
The previous GPU trial, build cache and results remain separate.

```sh
cd /Users/cdurham/code/odezza
python3 -B scratch/rosenbrock_cpu_trial/solver.py \
  scratch/rosenbrock_trial/examples/stiff-decay.json \
  --out scratch/rosenbrock_cpu_trial/validation/example.json
python3 -B scratch/rosenbrock_cpu_trial/test_cpu.py
python3 -B scratch/rosenbrock_cpu_trial/audit.py \
  --out scratch/rosenbrock_cpu_trial/validation/new-audit
```

Requests accept the previous `odezza.rosenbrock-trial.v1` schema, or
`odezza.cpu-solver-trial.v1`. The `solver` object accepts
`method: "Rosenbrock23"` with `rtol`, `atol`, `h0`, `max_attempts`, or
`method: "RK4"` with `substeps`. Both accept `predictions: false` for scores only.
Set `solver.precision` to `"FP32"` to use float state, RHS, Jacobian, stage and LU
arithmetic; times, error-control norms and MSE remain FP64. The default remains
`"FP64"`. Without explicit tolerances, FP32 uses rtol=1e-5 / atol=1e-7 and FP64
uses 1e-8 / 1e-10. Tolerances are not automatically changed after a failure.

The complete model, state names, observation times, full initial conditions and
observations are supplied in the JSON. Missing observations are null. The original
audit below is FP64; the separate linked precision report covers FP32.

## What is being verified

The primary number is **MSE against actual published benchmark observations**,
separately for training, validation and test. Initial samples are excluded;
missing observations and unscored temporal prefixes are masked. Hidden states
still integrate from the supplied complete initial condition. There is no
resetting to observations, derivative fitting or symbolic recovery.

The report distinguishes:

1. Whether integration completes.
2. Whether each resulting sample MSE reproduces the retained accurate FP64
   DOP853/Radau sample MSE. The declared numerical reproduction tolerance is
   `abs(new_mse-reference_mse) <= 1e-8 + 1e-4*abs(reference_mse)` on every split.
   This checks score reproduction, not pointwise trajectory equality.
3. Whether the actual sample MSE meets the original `1e-6` data-agreement gate.
   The reproduction tolerance does not change that gate or qualify a benchmark.

Rosenbrock is swept at rtol 1e-8, 1e-10, 1e-12 (atol rtol/100), with a recorded
1e-13 retry if it fails to reproduce the reference score. RK4 uses 32, 128 and
512 substeps per observation interval. All attempted settings, failure statuses,
sample MSEs, work counts, CPU time, input fingerprints and prediction hashes are
retained. Native score accumulation is independently checked in Python.

`scipy_check.py` independently runs installed Radau, BDF and LSODA on the 14
previously flagged source systems. Those checks use rack1's existing scientific
Python environment on its CPU only. They are labeled by host and are not a
timing comparison with native mac1 code. The historical source-solver replay is
also rerun separately, because matching the generator's numerical settings can
produce different results from solving the model more accurately.

See [RESULTS.md](RESULTS.md) for results. Published-data discrepancies are retained;
no input, benchmark qualification, search service or acceptance gate is changed.

To reproduce the precision experiment on mac1 with fresh output directories:

```sh
python3 -B scratch/rosenbrock_cpu_trial/precision_study.py --out scratch/rosenbrock_cpu_trial/validation/new-fp32
python3 -B scratch/rosenbrock_cpu_trial/precision_study.py --out scratch/rosenbrock_cpu_trial/validation/new-fp32-tight --ros-rtols 1e-8,1e-9 --rk-substeps ''
python3 -B scratch/rosenbrock_cpu_trial/precision_controls.py --out scratch/rosenbrock_cpu_trial/validation/new-fp32-controls.json
```
