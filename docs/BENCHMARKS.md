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

# Performance evidence and comparison rules

This document links retained historical reports. The export did not rerun their
GPU campaigns. Host validation of the copied source is recorded separately in
[VALIDATION.md](VALIDATION.md).

## Counts have different meanings

| Unit | Meaning |
|---|---|
| RHS program occurrence | One generated expression occupying one RHS slot; not necessarily mathematically unique |
| System | A complete vector of RHS expressions, including known components |
| Configuration | A system/program with a particular toggle and coefficient assignment |
| Row evaluation | One expression/configuration evaluated on one data row |
| RHS evaluation | Evaluation of a vector field at one state, e.g. an RK stage |
| RK4 step | Four coupled RHS-stage evaluations, plus state updates |
| Fit | An iterative parameter optimization involving many evaluations |
| Solve | A search and verification workflow; not a kernel-throughput unit |

## Generated multi-RHS throughput

Source: [heavy multi-RHS generator report](../projects/odezza/benchmarks/2026-09-03-heavy-multi-rhs-generator.md).

RTX 5090, 4,096 prepared systems, four generated RHS components, 256
configurations/system (five toggle bits × eight bank rows), three trajectories,
eight observations, two RK4 substeps. The timed scope includes specialization,
module load, execution and unload. It excludes corpus construction, initial
allocation/uploads, score downloads and cold template compilation.

| Exact generated depth | Systems/s | RHS occurrences/s | Configurations/s |
|---|---:|---:|---:|
| 2 | 283,445 | 1,133,781 | 72.56M |
| 4 | 251,041 | 1,004,165 | 64.27M |
| 6 | 219,397 | 877,588 | 56.17M |
| 8 | 175,041 | 700,165 | 44.81M |

The generator uses a specific operation distribution; this is not a uniform
sample of arbitrary ASTs. The report also records a numerical caveat at depth 8:
some randomly generated systems become explosive, and finite/nonfinite agreement
is not universal. Read its correctness section before using that row as evidence
for numerical reliability. Depths 2/4/6 passed the report's sampled CPU checks.

The same report gives about 9.58 s for a large cold template compilation versus
tens of milliseconds for warm work. A deployment benchmark must include cold
cache behavior separately.

## Full native request and service path

Source: [CUB integration report](../projects/odezza/benchmarks/score_topk/integration/README.md).

Two RTX 5080 GPUs; one million six-equation combinations × 2,048 coefficient
configurations = 2.048 billion evaluations. Three observations/four RK4 steps
per evaluation. Medians of three warm runs:

| Path | Wall time |
|---|---:|
| Earlier direct native, existing structure retention | 1.5833 s |
| Integrated CUB, same request/results | 1.2210 s |
| Integrated CUB, raw global/family top-16 | 1.2001 s |
| Client → orchestrator → worker, existing retention | 1.2654 s |
| Client → orchestrator → worker, raw top-16 | 1.2142 s |

The matched direct comparison is about 1.30× throughput. Raw top-k changes the
retention contract and is **not** an equivalent-output comparison. The generated
systems are combinations from a finite six-RHS construction; do not describe the
count as a million unrelated unrestricted trees. Two-GPU results are not rates
for one 5080.

## Long rollout batching repair

Source: [production long-rollout report](../projects/odezza/benchmarks/long_rollout_tiles/PRODUCTION.md).

317.19M configurations with 5,120 integration steps took 222.64 s before and
34.16 s after the documented repair on two RTX 5080s, about 6.52× faster. This is
an internal comparison. It demonstrates why adequate work per launch/module
matters; it is not a speedup over SciPy, PySR or another vendor.

## PTX compilation and row-engine evidence

The [PTX compilation benchmark](../projects/mm-stack-ptx-ptx-inject-bench/)
retains compact host/compiler tables and the generating source. Secant retains
[hardware validation](../projects/secant/docs/cubin_hardware_validation.md),
[module-loading measurements](../projects/secant/docs/module_loading_and_throughput.md)
and comparison-backend source under `bench/`. Those reports use different
expression distributions, output modes and historical APIs. Do not combine
their numbers into one purported matched comparison.

Secant-SR reports describe complete searches and refinement policies. They are
preserved for context, but search success rates are not execution-engine rates.

## How to make a defensible new comparison

Use identical prepared expressions, coefficients, data, precision, solver steps,
mathematical operations, output/reduction contracts and correctness tolerances.
Report preprocessing separately and end-to-end, with both cold and warm caches.
Compare at least:

- compile-per-program or batched PTX compilation, including compiler/module cost;
- a sensible runtime interpreter, including its per-operation execution cost;
- specialized native execution, including preparation, loading and retention.

Sweep reuse per structure. Native specialization is intended to improve the
compilation/runtime tradeoff, not to guarantee a constant multiplier on every
workload. Record failed programs, unsupported shapes, NaNs, padding and duplicate
evaluations rather than silently removing them from the denominator.
