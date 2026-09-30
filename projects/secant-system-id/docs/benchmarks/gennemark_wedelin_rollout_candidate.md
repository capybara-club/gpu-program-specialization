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

# Gennemark–Wedelin rollout benchmark candidate

Status: selected for investigation after the four-system ODEBench pilot; not yet imported or run.

## Why it belongs in the plan

Gennemark and Wedelin's 2009 *Bioinformatics* benchmark collection contains more than 40 ODE identification problems drawn mainly from cellular biology. Each problem specifies time-series experiments, an initial model, an allowed model space, parameter ranges, and an error function. Both structure and parameters can be unknown.

This is unusually aligned with Secant System ID because candidate quality is defined through simulated system behavior rather than only pointwise regression against supplied derivatives. It can exercise:

- complete coupled-system rollout inside a specialized kernel;
- repeated integration across parameter settings and experimental conditions;
- structural search and constant optimization together;
- early rejection of invalid or divergent trajectories;
- trajectory-resident scoring without materializing every intermediate state;
- time-to-quality as the primary end-to-end measure.

Primary references:

- Paper: <https://doi.org/10.1093/bioinformatics/btp050>
- Benchmark site: <https://www.cse.chalmers.se/~dag/identification/Benchmarks/>

## Proposed evaluation sequence

1. Inventory the official problem files and confirm which remain reproducibly loadable.
2. Select a small pilot spanning low and moderate state counts, one and multiple experiments, and at least two kinetic/model-space families.
3. Translate each problem losslessly into `RecoveryProblem`: measured variables, missing observations, initial conditions, inputs, parameter bounds, permitted reactions, integration horizon, and original error function.
4. Validate the planted/published model against the official observations using a trusted CPU solver before generating GPU kernels.
5. Implement the benchmark's allowed reaction grammar directly instead of substituting Secant's unrestricted arithmetic GP grammar.
6. Run Secant using full-rollout fitness, then replay every winner with the trusted solver.
7. Report success, objective, structure, parameters, wall time, candidate rollouts, invalid/divergent fraction, and time-to-quality checkpoints.
8. Rerun at least one maintained baseline on the same modern hardware before making a speedup claim.

## Comparability constraints

- The model space is biologically structured. A result is not equivalent to free-form equation discovery and must be labeled as constrained biochemical structure search.
- Published runtime numbers are from substantially older hardware and software. They are historical context, not a fair speed baseline.
- Solver choice, tolerances, positivity constraints, observation weighting, parameter transformations, and failure penalties are part of the benchmark. Changing any of them creates a non-equivalent mode.
- Some problems may contain unobserved states, sparse measurements, external inputs, discontinuities, or multiple experimental conditions. The current dense aligned trajectory kernel may require a new observation/scoring topology; such cases must not be silently interpolated or excluded.
- Test or validation experiments must remain unavailable to GP selection, constant optimization, and final candidate selection.

## Go/no-go pilot criterion

Proceed to the larger collection if the pilot can reproduce the official objective for the known model, replay Secant winners with an independent solver, and complete searches without narrowing the declared model space. If official artifacts or semantics are incomplete, retain the benchmark as historical evidence and prioritize ODEBench's fully reproducible rollout protocol.
