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

# Generic recovery problems and benchmark expansion

Status: local infrastructure ready; official benchmark search runs not yet executed.

## Recovery-problem boundary

`RecoveryProblem` is now the benchmark-aware input above the CUDA kernel shape. It records:

- whether the task is trajectory recovery, derivative regression, or known-model parameter fitting;
- the known CUDA system backbone and every missing structural site;
- named parameters, nominal values, bounds, and whether each value is estimated;
- operator grammar, node limit, constant-bank width, and patch capacity;
- integration interval, sampling, candidate method, and reference-solver provenance;
- train, validation, and test experiments with exact time grids, initial conditions, optional derivatives, noise semantics, and explicit missing values;
- ground-truth equations and an optional CPU right-hand-side function for validation;
- benchmark name, system ID, source, protocol, and qualifying notes.

The current dense trajectory kernel is allowed only when every selected trajectory has complete state observations, a uniform grid, and the same grid. `trajectory_kernel_blockers()` reports incompatibilities rather than silently interpolating or dropping data. This is important for later PEtab work, where partial observations and unaligned measurement times are normal.

The legacy fed-batch `reference_data()` path now builds a generic fed-batch recovery problem and uses its dense reference layout. Existing callers and numerical results remain compatible.

## Two protocols over one 63-system corpus

MDBench does not introduce a second set of 63 ODEs. Its ODE collection is derived from ODEBench. The definitions should therefore be shared while their evaluation protocols remain distinct.

| Protocol | Input to search | Primary prediction | Official target | Intended comparison |
|---|---|---|---|---|
| MDBench ODE | One 150-point trajectory, first 80% for fitting | Derivative at observed state | Exact clean derivative on final 20% | MDBench model-discovery methods |
| ODEBench / ODEFormer | Sampled trajectory, with defined noise/drop conditions | Reintegrated system trajectory | Held trajectory behavior under the paper's protocol | ODEFormer |
| PEtab | Known mechanistic model, experiments, observables, measurements, and parameter bounds | Observable trajectories | Measurement likelihood/objective | Parameter-fitting correctness and LM throughput |

PEtab results must never be presented as free-form equation discovery.

## Four-system validation set

These are copied from the ODEBench definitions, including its constants and initial conditions.

| ID | System | Equations | Constants | Initial conditions |
|---:|---|---|---|---|
| 27 | Lotka–Volterra | `x' = x(c0-c1y)`; `y' = -y(c2-c3x)` | `1.84, 1.45, 3.0, 1.62` | `(8.3, 3.4)`; `(0.4, 0.65)` |
| 37 | Van der Pol | `x' = v`; `v' = -x-c0(x²-1)v` | `0.43` | `(2.2, 0)`; `(0.1, 3.2)` |
| 56 | Lorenz-63 | `x' = c0(y-x)`; `y' = c1x-y-xz`; `z' = xy-c2z` | `10, 28, 8/3` | `(2.3, 8.1, 12.4)`; `(10, 20, 30)` |
| 59 | Rössler | `x' = c3(-y-z)`; `y' = c3(x+c0y)`; `z' = c3(c1+z(x-c2))` | `0.2, 0.2, 5.7, 5.0` | `(2.3, 1.1, 0.8)`; `(-0.1, 4.1, -2.1)` |

The Rössler definition deliberately includes ODEBench's `c3=5` time scale. Omitting it would produce a familiar Rössler system but not the benchmark system.

Local dependency-free RK4 trajectories use 150 samples over `[0, 10]`. A step-halving test compares maximum internal steps of `0.005` and `0.0025`; all four systems are finite and have aggregate state MSE below `1e-5`. This checks our equations and local integrator but is not a substitute for the official LSODA artifact.

## MDBench adapter and pilot

`mdbench.py` reads the official NPZ layout (`t`, `u`, and `du`) without requiring NumPy. It validates float32/float64 NPY members, dimensions, time ordering, finite values, the official 80/20 temporal split, and the official NMSE formula. It also exposes approximate derivatives separately from exact derivative targets.

MDBench's noisy ODE generator perturbs states multiplicatively while retaining the exact derivatives of the clean trajectory as `du`. The adapter preserves that distinction.

The local clean/noisy pilot evaluates the planted right-hand side on the held-out 20%. It is an adapter correctness test, not symbolic search:

| System | Clean oracle NMSE | 20 dB noisy-state oracle NMSE |
|---|---:|---:|
| Lotka–Volterra | 0 | 0.0219527 |
| Van der Pol | 0 | 0.0240183 |
| Lorenz-63 | 0 | 0.0597425 |
| Rössler | 0 | 0.0124831 |

The nonzero noisy values are expected: the predictor is evaluated at perturbed states while the target remains the clean trajectory derivative.

## Expansion sequence

### 1. Four-system Secant validation

Use the four definitions above to validate end-to-end source generation, specialization, trajectory scoring, structural recovery, and constants. Run clean data first, then declared noise conditions. Store every seed, wall-clock checkpoint, training score, held-out score, recovered equation, complexity, and failure/rejection counter.

### 2. MDBench pilot

Load official clean and noisy NPZ files rather than regenerating them. Add a Secant estimator wrapper with the official `fit`, `predict`, `complexity`, and `to_str` behavior. The wrapper must respect MDBench's finite-difference fitting target and exact-derivative test target. Compare time-to-quality and final NMSE under the same timeout and CPU/GPU accounting.

### 3. All 63 shared systems

`load_solutions_json()` imports official ODEBench output and requires exactly 63 unique systems by default. It preserves every parameter/initial-condition trajectory, noise amplitude, subsampling pattern, and time grid. The corresponding MDBench NPZ loader handles any system dimension without a hard-coded four-system list.

Before launching all systems, freeze a single grammar policy. The four-system polynomial-only grammar is intentionally not sufficient for the full corpus, which contains functions such as trigonometric, exponential, logarithmic, division, and non-integer powers. Report unsupported operators and excluded systems rather than silently reducing the denominator.

### 4. Direct ODEBench / ODEFormer comparison

Use the official `solutions.json` artifacts and paper conditions: 150 points over `[0, 10]`, multiplicative Gaussian noise amplitudes through 5%, and the declared point-dropping conditions. Preserve each trajectory independently so the exact ODEFormer fitting/evaluation grouping can be mirrored rather than assuming a joint multi-trajectory fit. Report both structural and trajectory metrics used by the reference evaluation.

### 5. Gennemark–Wedelin biological ODE identification

Evaluate the older rollout-oriented benchmark collection described in [the dedicated note](gennemark_wedelin_rollout_candidate.md). It contains more than 40 biological ODE identification problems with unknown structure and parameters, time-series experiments, parameter bounds, an allowed biochemical model space, and a simulation-based error function.

Treat this as a separate benchmark family. Its constrained reaction grammar is useful domain knowledge and should not be described as equivalent to ODEBench's relatively free-form expression search. Published wall times also come from much older hardware and are contextual rather than directly comparable; rerun an available baseline on current hardware when making speed claims.

### 6. PEtab parameter fitting

Add this only after the trajectory and derivative protocols are stable. Use an official PEtab library to parse the YAML problem, model file, parameter table, experiment/condition definitions, observables, noise models, and measurements. Normalize those into known equations plus estimated parameters and measurement metadata. A separate sparse/irregular observation kernel or an explicit preprocessing policy will be needed when measurements are partial or time grids differ.

Benchmark LM proposal throughput, accepted-step throughput, objective quality, parameter recovery, and end-to-end wall time on known models. Do not mix these results with equation-discovery recovery rates.

## Material deviations and prerequisites

1. No official MDBench NPZ or ODEBench `solutions.json` artifact was present locally, so the current pilot uses locally integrated reference trajectories. This blocks a publishable comparison but not adapter validation. Remedy: provide the official artifacts, then rerun the same loader and tests.
2. Local references use fixed-step RK4, whereas ODEBench/MDBench generation uses LSODA with `rtol=1e-5`, `atol=1e-7`, `first_step=1e-6`, and `min_step=1e-10`. Official scores must use official data.
3. The local noise helper matches MDBench's formula but uses Python's RNG rather than NumPy's RNG, so noise samples are not byte-identical. Official noisy NPZ files remove this difference.
4. The pilot evaluates known right-hand sides. It does not measure Secant search quality or time-to-recovery.
5. The all-63 loader is implemented and count-checked, but all 63 systems have not been loaded or searched locally.
6. No new package was installed. PEtab parsing should use the maintained PEtab implementation after the user installs the supported dependency; it should not be replaced with an incomplete handwritten SBML/YAML parser.

## Local verification

- 86 tests pass, including the original Secant System ID suite.
- The four reference systems pass finite/convergence checks.
- The fed-batch legacy and generic dense reference arrays match exactly.
- The MDBench NPZ parser, temporal split, noise target semantics, and pilot labeling are tested.
- The ODEBench JSON loader rejects incomplete corpora by default and accepts an explicitly marked partial fixture for unit testing.
