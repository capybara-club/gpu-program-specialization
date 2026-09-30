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

# Separate Rosenbrock GPU trial

An autonomous FP64 Rosenbrock23 integrator, authored on mac1 and tested on rack1.
This folder does not replace Odezza's production scoring or search services.
For native mac1 execution without CUDA and verification of actual benchmark
sample MSE, use the separate [CPU implementation](../rosenbrock_cpu_trial/README.md).
See [RESULTS.md](RESULTS.md) for measured coverage, accuracy and limitations, and
[DEVIATIONS.md](DEVIATIONS.md) for execution differences from production.

## What is implemented

The restricted arithmetic compiler emits the RHS and analytic state Jacobian
with common-subexpression reuse. Each GPU lane integrates one trajectory, owns
its pivoted LU and three stage solves, adapts the step, lands on each observation
time, and accumulates observed-state MSE. Rejected steps reuse the current
Jacobian. Every accepted step refreshes it on the next attempt. There is no
Newton iteration and no search policy in the numerical kernel.

All states require initial values. Unobserved values are JSON `null`; hidden
states still participate in integration and numerical error control. Temporal
prefixes are integrated from the original initial state, never reset to an
observation. The initial point is not scored. Failure leaves MSE infinite in
memory (JSON null) and uncomputed prediction rows NaN (JSON null).

Supported: 1–16 states, autonomous arithmetic `+ - * / sin cos exp tanh` and
finite FP64 literals. A constant derivative such as `0` is supported. Unsupported
expressions fail compilation; there is no solver fallback. Time-dependent ODEs,
mass matrices/DAEs, discontinuities/events, and production RNG/toggle banks
are not implemented in this prototype.

Status values: `0` integration complete, `1` attempt budget exhausted,
`2` step underflow, `3` nonfinite RHS/Jacobian at the current state,
`4` nonfinite final score. A singular stage matrix rejects the step and shrinks
it; pivot failures and all attempted/accepted/rejected work are counted. A
completed integration is not itself a guarantee of accurate global trajectories.

## JSON entry point

On rack1, using the already-installed environment and CUDA compiler:

```sh
cd /home/cdurham/odezza/scratch/rosenbrock_trial
/home/cdurham/odezza/scratch/pysr_rollout_trial/.venv/bin/python run.py \
  examples/stiff-decay.json --out validation/stiff-decay-result.json
```

The request contains `schema: "odezza.rosenbrock-trial.v1"`, `states`, a complete
`rhs` array, `trajectories` with `times`, `initial_state`, and `states`, plus optional
`solver` settings: `rtol`, `atol`, `h0`, `max_attempts`, and `predictions`.
This evaluates supplied equations; it does not recover unknown equations.

`Solver` in `prototype.py` is the reusable Python interface. It caches compiled
models by source/compiler/flags/architecture fingerprint. Compilation emits a
shared CUDA library with CPU and GPU entry points. The runtime records events
around the kernel and copies, waits for the completion event, and reports whole
call time separately. Allocations remain per call; host memory is pageable.
Source compilation and consumer-GPU FP64 throughput are not comparable to
Odezza's SASS-specialized FP32 scoring throughput.

## Reproduce the audits

Run these from this folder with the rack1 scientific Python above. Output audit
folders must be new; previous runs are never overwritten.

```sh
python test_solver.py
python audit.py --out validation/new-flagged --cpu-parity
python audit.py --out validation/new-all-clean --all-clean --rtols 1e-8 --device 1
python audit.py --out validation/new-chaotic --case 055 --case 056 --case 061 --rtols 1e-12,1e-13
python stiff_study.py --out validation/new-stiff-controls.json --device 1
```

The private benchmark audit reconstructs known equations exclusively to check
integration. It preserves public/source/private fingerprints and observation
masks, and compares separately against observations and independent accurate
FP64 DOP853/Radau trajectories. It does not register search jobs, consume search
test seals, or requalify published-data discrepancies.

The scalar stiff-control test has an analytic solution and a million-to-one
decay-rate separation. Robertson kinetics is compared against independent Radau
with an independently written RHS. Scaling repetitions deliberately repeat one
identical problem; they measure available parallel capacity, not diverse search
performance or additional distinct ASTs/configurations.

## Method sources

The implementation uses the autonomous Rosenbrock23 stage equations with
`d = 1/(2+sqrt(2))`, `c32 = 6+sqrt(2)`, second-order accepted solution and the
embedded local error estimate. Reference documentation and coefficients:

- [SciML Rosenbrock methods](https://docs.sciml.ai/OrdinaryDiffEq/stable/semiimplicit/Rosenbrock/)
- [Rosenbrock23 coefficients](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl)
- [Rosenbrock stage reference](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl)
- [SciPy independent reference solvers](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

Next: compare a higher-order Rosenbrock method before adopting this in search;
measure realistic heterogeneous coefficient banks, Jacobian workspace/constant
state elimination, persistent allocations, and precision-specialized codegen.
