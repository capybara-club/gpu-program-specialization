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

# Benchmark strategy

## Decision

Use MDBench as the primary external model-discovery benchmark. Its ODE suite
contains 63 systems and its comparison framework already includes SINDy,
EWSINDy, EQL, uDSR, PySR, Operon, ODEFormer, and other methods. Its method
contract is small enough for a Secant adapter:

- `fit(t_train, u_train, u_dot_train)`;
- `predict(t_test, u_test)` for predicted time derivatives;
- `complexity()`; and
- `to_str()` for the recovered equations.

Secant should accept `u_dot_train` for interface compatibility but must not use
ground-truth derivatives during trajectory-based search. The report should
state this explicitly. Derivatives may be used only as an additional test
metric after search.

ODEBench and the seven-system Strogatz collection remain useful secondary
views because ODEFormer publishes results on them. Start with a few canonical
systems as integration gates, then run the exact externally supplied datasets
rather than locally regenerated approximations.

PEtab is a second benchmark lane. It specifies parameter estimation for known
SBML models, measurements, conditions, observables, noise models, bounds, and
shared parameters. It is useful for measuring the fused trajectory and LM
engine, but it is not evidence of free-form equation discovery. PEtab Select
tests selection among supplied model candidates and can be considered after a
basic PEtab importer exists.

## Claims and benchmarks

| Claim | Benchmark | Primary measurement |
|---|---|---|
| Discover missing ODE structure | MDBench ODE suite | exact/support recovery, derivative prediction, held-out rollout |
| Robustness to noisy observations | MDBench SNR variants | quality versus SNR and wall time |
| Compare with a trajectory-to-equation model | ODEBench and Strogatz | published protocol plus our rerun when practical |
| Fit constants in a known biological model | PEtab benchmark collection | objective, parameter error, and time to objective |
| Choose among enumerated biological models | PEtab Select | selected model and total wall time |
| Demonstrate the fed-batch use case | Astaxanthin reproduction | rate-law and held-out trajectory recovery |

## First supported regime

Keep the first external adapter inside the kernel regime the current engine is
designed to serve:

- one to four state variables;
- one to four missing right-hand-side expressions;
- autonomous systems, so time is not yet a dynamic leaf;
- dense, aligned observations;
- complete state observation;
- `float32` RK4 scoring; and
- arithmetic, square/cube through multiplication, and protected division.

Do not initially claim support for stiff systems, events, delays, algebraic
constraints, partial observation, controls, or unaligned trajectory times.
Those cases need different integration, data, or kernel regimes and should be
added deliberately.

Canonical implementation gates should be Lotka--Volterra, Lorenz-63, Van der
Pol, and Rössler. These gates validate different state counts and nonlinear
couplings; they are not substitutes for the exact MDBench data and splits.

## Generic recovery problem

The source generator is already substantially generic. `SystemModel`,
`KernelShape`, the direct template, and the packed template support variable
state counts and variable missing-site counts. The current hard-coded boundary
is concentrated in `cli.py`, `recovery.py`, and `trajectory_lm.py`.

Introduce a `RecoveryProblem` value with:

- a `SystemModel` and `KernelShape`;
- training and held-out trajectory groups;
- observation times or a fixed observation interval;
- initial states and flattened reference storage;
- loss normalization and state weights;
- optional ground-truth programs, constants, and bindings;
- state and missing-site names;
- expression/operator policy;
- benchmark identity, split identity, and data hash; and
- optional symbolic and vector-field assessment callbacks.

Then make the following existing fed-batch assumptions problem-owned:

1. training-design selection and reference generation;
2. the fixed 832-float trajectory-LM reference check;
3. `FED_BATCH_SHAPE` allocation strides in the LM queue;
4. held-out trajectory integration and rate-surface metrics;
5. planted structural matching and resolved state names; and
6. report schema labels and experiment description.

The C99 specializer, packed module format, eager loader, GP population ABI, and
SASS instruction writer should remain unchanged unless a failing test shows a
real shape dependency.

## MDBench adapter

The adapter should load the benchmark's existing `t`, `u`, and `du` arrays and
preserve its train/validation/test split. A Secant run should:

1. convert each training trajectory into the `RecoveryProblem` reference
   layout;
2. search by integrated trajectory loss without reading `du`;
3. retain the Pareto archive over loss and expression complexity;
4. select the final model using only the benchmark's training/validation data;
5. return derivative predictions through `predict`;
6. return the resolved equations and the benchmark's complexity convention;
7. integrate the selected equations from held-out initial conditions; and
8. write a self-contained report with hardware, seeds, budgets, data hash, and
   every failure or rejected candidate count.

Do not vendor or silently regenerate the external dataset. Keep an explicit
path to a user-provided MDBench data checkout and record its version/hash.

## Metrics

Report all of the following rather than reducing the result to one score:

- exact symbolic recovery when canonical equivalence can be established;
- term-support precision, recall, and F1 for polynomial/rational systems;
- coefficient relative error after canonicalization;
- derivative test R2/NRMSE using `du` only after search;
- integrated held-out trajectory R2/NRMSE;
- divergence/failure rate;
- expression complexity and Pareto frontier;
- wall-clock time to fixed quality thresholds;
- total search wall time and hardware;
- number of structures, settings, LM fits, and trajectory steps evaluated; and
- specialization rejections, fallbacks, invalid trajectories, and GPU health.

The primary comparison is wall-clock time to held-out quality. Also publish
the benchmark's native aggregate metrics so the result is directly comparable
with its tables.

## Fair comparison protocol

- Use identical data, splits, noise, and test trajectories.
- Give each method the same wall-clock limits and report its hardware.
- Report Secant as a GPU method; do not disguise the GPU/CPU difference with an
  artificial single-core equivalence.
- Separate published numbers from numbers reproduced locally.
- Include compilation, module loading, and final model selection in
  end-to-end time; separately report steady-state search time.
- Never select a model using test trajectories or test derivatives.
- Freeze operator sets and maximum complexity before examining test results.
- Run enough seeds to expose variance and retain every failed run.

## Implementation gates

1. Extract `RecoveryProblem` while leaving the fed-batch output byte-for-byte
   and numerically unchanged.
2. Generalize the CPU reference, held-out rollout, and trajectory-LM queue.
3. Add planted direct-kernel tests for the four canonical systems.
4. Add one short C99 GP recovery test for a two-state system.
5. Implement the MDBench `Regressor` adapter against an explicit external data
   path.
6. Reproduce one MDBench baseline system and compare CPU/GPU reference scores.
7. Run a small clean/noisy pilot before committing to all 63 systems.
8. Add PEtab parsing only after the structural-discovery path is producing
   comparable results.
