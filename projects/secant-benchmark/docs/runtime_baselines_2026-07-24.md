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

# Runtime Expression Baselines

> These measurements predate the versioned portable corpus and used
> independently generated input data in each collector. Keep them as
> historical backend measurements. The comparable fixed-corpus replacement is
> [`runtime_baselines_2026-07-26.md`](runtime_baselines_2026-07-26.md).

These measurements use eight fixed, balanced ALU expressions. Each expression
has eight input leaves and seven binary operations selected from add,
multiply, minimum, and maximum. Repeating the eight cases changes the timed
batch size without changing expression complexity. Throughput counts one
complete AST evaluated for one row.

Hardware:

- CPU: AMD Ryzen 9 9900X, 12 cores and 24 threads
- GPU: NVIDIA GeForce RTX 5090, 32 GB
- Driver: 595.71.05

Software:

- Native C: GCC 15.2.0, C99, `-O3 -mavx2 -mfma`, OpenMP
- PySR 1.5.10 / SymbolicRegression.jl 1.11.3
- EvoGP commit `5538d75`, PyTorch 2.12.1+cu130
- Kozax 0.1.4, JAX/JAXlib 0.11.0, CUDA 13 plugin
- PyOperon 0.6.1, Operon revision `5a1c937`, Clang 20.1.8, `x86-64-v3`

## Best Materialize Throughput

| Rows | Native AVX2 | Operon | PySR | EvoGP | Kozax |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 4.442e10 | 1.117e9 | 2.297e9 | 4.212e9 | 7.819e8 |
| 16,384 | 6.252e10 | 2.391e9 | 1.777e9 | 4.389e9 | 1.661e8 |
| 262,144 | 8.812e9 | 2.538e9 | 5.213e8 | - | 1.540e8 |
| 1,048,576 | - | 2.601e9 | - | - | 1.596e8 |

## Best SSE Throughput

| Rows | Native AVX2 | Operon | PySR | EvoGP | Kozax |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 6.508e10 | 8.382e8 | 2.249e9 | 8.024e9 | 7.980e8 |
| 16,384 | 7.081e10 | 1.265e9 | 1.780e9 | 1.346e10 | 1.667e8 |
| 262,144 | 3.786e10 | 1.158e9 | 5.217e8 | - | 1.550e8 |
| 1,048,576 | - | 1.188e9 | - | - | 1.601e8 |

Values are row evaluations per second. Each table selects the best measured
AST batch, thread count, and evaluator mode for that row count. The source CSVs
retain every measured configuration.

## SECANT Context

The direct-CUBIN SECANT runtime sweep uses the same balanced eight-leaf ALU
family at 1,048,576 rows. Materialize uses 32 total ASTs and SSE uses 1,536;
the external backends use their best feasible measured batches at that row
count.

| Shape | SECANT CUBIN | Operon | Kozax | SECANT / Operon | SECANT / Kozax |
|---|---:|---:|---:|---:|---:|
| Materialize | 3.709e11 | 2.601e9 | 1.596e8 | 143x | 2,324x |
| SSE | 1.517e12 | 1.188e9 | 1.601e8 | 1,277x | 9,476x |

These ratios describe the measured backend surfaces, not equal implementation
contracts. SECANT materialize writes every result. SECANT SSE is fused and
keeps its reduction local; the available Operon SSE path materializes before
reducing. Kozax uses runtime tree dispatch inside a JAX program and could fit
only 16 ASTs at this row count under the collector's 28 GiB working-set guard.

## Timing Boundaries

The native baseline contains no Python or Julia in its timed region. AVX2
vectorizes rows in groups of eight, OpenMP distributes ASTs, materialize writes
all outputs, and SSE writes one reduction per AST. The collector binds OpenMP
workers to core places with a spread policy.

PySR timing runs directly inside Julia. Materialize calls
`eval_tree_array`; SSE calls `eval_loss`. Search, mutation, Python-to-Julia
calls, and process startup are excluded. Standard evaluation and
`LoopVectorization.@turbo` are preserved as separate CSV records.

EvoGP SSE uses the native fused `tree_SR_fitness` runtime-dispatch path with
hybrid parallel execution. EvoGP materialize uses its public
`Forest.batch_forward` API. Its timed GPU work includes expansion of tree
metadata and inputs before the dispatch kernel, so the materialize number is
an end-to-end public API baseline rather than kernel-only throughput.

Kozax materialize JIT-compiles its `tree_evaluator` vectorized over population
and rows. Kozax SSE calls its JIT-compiled symbolic-regression fitness path,
which computes mean squared error. JIT compilation is excluded; synchronized
wall-clock intervals include JAX dispatch. Kozax's fixed-width tree state is
vectorized across the population-row product. A 1,024-AST by 262,144-row probe
asked XLA for approximately 65 GiB and did not fit on the 32 GB GPU, so the
largest recorded row count uses 256 ASTs.

Operon materialize calls its native `EvaluateTrees` C++ API with preallocated
output and 1, 12, or 24 workers. That API creates a Taskflow executor on every
call. PyOperon does not expose a fused parallel batch-fitness API: the reported
Operon SSE path performs `EvaluateTrees`, then an in-place NumPy SSE reduction.
It should be interpreted as materialize-plus-reduce throughput rather than as
an optimized Operon fitness ceiling.

Raw data:

- [`baseline_native_avx_2026-07-24.csv`](baseline_native_avx_2026-07-24.csv)
- [`baseline_pysr_2026-07-24.csv`](baseline_pysr_2026-07-24.csv)
- [`baseline_evogp_2026-07-24.csv`](baseline_evogp_2026-07-24.csv)
- [`baseline_kozax_2026-07-24.csv`](baseline_kozax_2026-07-24.csv)
- [`baseline_operon_2026-07-24.csv`](baseline_operon_2026-07-24.csv)
- [`baseline_runtime_2026-07-24.svg`](baseline_runtime_2026-07-24.svg)
