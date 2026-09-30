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

# Fixed-Corpus Runtime Backend Comparison

- Corpus: `portable_alu_balanced_8x1024_seed1_v1`
- Corpus hash: `5f790c4e4d7bc60a`
- Input seed: `1`
- Workload: first 256 unique balanced eight-leaf ALU expressions
- Metric: complete AST row evaluations per second

![Runtime comparison](baseline_runtime_2026-07-26.svg)

## Materialize

| Rows | SECANT CUBIN | SECANT CUDA | SECANT PTX | Native AVX2 | PySR | EvoGP | Kozax | Operon |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 4.431e+10 | 6.826e+10 | 6.997e+10 | 6.555e+10 | 1.151e+09 | 4.203e+09 | 8.090e+08 | 7.113e+08 |
| 16,384 | 5.366e+11 | 7.385e+11 | 7.454e+11 | 5.788e+10 | 1.523e+09 | 4.390e+09 | 1.548e+08 | 1.925e+09 |
| 262,144 | 4.124e+11 | 4.124e+11 | 4.146e+11 | 8.770e+09 | 5.659e+08 | 4.828e+09 | - | 2.423e+09 |

Best measured configuration for each series and row count.

## SSE

| Rows | SECANT CUBIN | SECANT CUDA | SECANT PTX | Native AVX2 | PySR | EvoGP | Kozax | Operon |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 2.590e+10 | 2.565e+10 | 2.607e+10 | 6.269e+10 | 1.088e+09 | 8.031e+09 | 8.089e+08 | 6.023e+08 |
| 16,384 | 3.543e+11 | 4.109e+11 | 4.125e+11 | 7.724e+10 | 1.441e+09 | 1.344e+10 | 1.557e+08 | 1.226e+09 |
| 262,144 | 2.408e+12 | 2.536e+12 | 2.587e+12 | 3.753e+10 | 4.993e+08 | 1.430e+10 | - | 1.124e+09 |

Best measured configuration for each series and row count.

## SECANT Relative Throughput

| SECANT series | Rows | Fastest external backend | SECANT / external |
|---|---:|---|---:|
| materialize / SECANT CUBIN | 1,024 | Native AVX2 | 0.68x |
| materialize / SECANT CUDA | 1,024 | Native AVX2 | 1.04x |
| materialize / SECANT PTX | 1,024 | Native AVX2 | 1.07x |
| materialize / SECANT CUBIN | 16,384 | Native AVX2 | 9.27x |
| materialize / SECANT CUDA | 16,384 | Native AVX2 | 12.76x |
| materialize / SECANT PTX | 16,384 | Native AVX2 | 12.88x |
| materialize / SECANT CUBIN | 262,144 | Native AVX2 | 47.03x |
| materialize / SECANT CUDA | 262,144 | Native AVX2 | 47.03x |
| materialize / SECANT PTX | 262,144 | Native AVX2 | 47.28x |
| sse / SECANT CUBIN | 1,024 | Native AVX2 | 0.41x |
| sse / SECANT CUDA | 1,024 | Native AVX2 | 0.41x |
| sse / SECANT PTX | 1,024 | Native AVX2 | 0.42x |
| sse / SECANT CUBIN | 16,384 | Native AVX2 | 4.59x |
| sse / SECANT CUDA | 16,384 | Native AVX2 | 5.32x |
| sse / SECANT PTX | 16,384 | Native AVX2 | 5.34x |
| sse / SECANT CUBIN | 262,144 | Native AVX2 | 64.15x |
| sse / SECANT CUDA | 262,144 | Native AVX2 | 67.57x |
| sse / SECANT PTX | 262,144 | Native AVX2 | 68.94x |

## Low-row Interpretation

At 1,024 rows, Native AVX2 SSE is 2.40x faster than the best SECANT GPU SSE result. The complete batch contains only 262,144 row evaluations. GPU SSE still pays kernel scheduling, CTA shuffle/shared-memory reduction, and atomic accumulation costs, while the 12-core AVX2 path works from CPU cache. This is a low-workload crossover rather than evidence of timed host-to-device transfers.

Materialize does not pay the SSE reduction epilogue: at the same row count, the best SECANT GPU result is 1.07x faster than Native AVX2. The GPU SSE path overtakes Native AVX2 at the next measured row count.

## Materialize Selected Configurations

| Shape | Rows | System | ASTs | Workers | Mode |
|---|---:|---|---:|---:|---|
| materialize | 1,024 | EvoGP | 256 | 1 | `batch_forward_public` |
| materialize | 1,024 | Kozax | 256 | 1 | `jit_tree_evaluator_vmap` |
| materialize | 1,024 | Native AVX2 | 256 | 12 | `avx2_fma_openmp` |
| materialize | 1,024 | Operon | 256 | 12 | `evaluate_trees_preallocated` |
| materialize | 1,024 | PySR | 256 | 24 | `julia_threads_standard` |
| materialize | 1,024 | SECANT CUBIN | 256 | 1 | `kernel_only_k2_a128_streams2` |
| materialize | 1,024 | SECANT CUDA | 256 | 1 | `kernel_only_k2_a128_streams2` |
| materialize | 1,024 | SECANT PTX | 256 | 1 | `kernel_only_k2_a128_streams2` |
| materialize | 16,384 | EvoGP | 256 | 1 | `batch_forward_public` |
| materialize | 16,384 | Kozax | 256 | 1 | `jit_tree_evaluator_vmap` |
| materialize | 16,384 | Native AVX2 | 256 | 12 | `avx2_fma_openmp` |
| materialize | 16,384 | Operon | 256 | 12 | `evaluate_trees_preallocated` |
| materialize | 16,384 | PySR | 256 | 24 | `julia_threads_standard` |
| materialize | 16,384 | SECANT CUBIN | 256 | 1 | `kernel_only_k2_a128_streams2` |
| materialize | 16,384 | SECANT CUDA | 256 | 1 | `kernel_only_k2_a128_streams2` |
| materialize | 16,384 | SECANT PTX | 256 | 1 | `kernel_only_k2_a128_streams2` |
| materialize | 262,144 | EvoGP | 256 | 1 | `batch_forward_public` |
| materialize | 262,144 | Native AVX2 | 256 | 12 | `avx2_fma_openmp` |
| materialize | 262,144 | Operon | 256 | 24 | `evaluate_trees_preallocated` |
| materialize | 262,144 | PySR | 256 | 24 | `julia_threads_standard` |
| materialize | 262,144 | SECANT CUBIN | 256 | 1 | `kernel_only_k8_a32_streams2` |
| materialize | 262,144 | SECANT CUDA | 256 | 1 | `kernel_only_k8_a32_streams8` |
| materialize | 262,144 | SECANT PTX | 256 | 1 | `kernel_only_k8_a32_streams8` |

## SSE Selected Configurations

| Shape | Rows | System | ASTs | Workers | Mode |
|---|---:|---|---:|---:|---|
| sse | 1,024 | EvoGP | 256 | 1 | `hybrid_parallel` |
| sse | 1,024 | Kozax | 256 | 1 | `jit_symbolic_regression_fitness` |
| sse | 1,024 | Native AVX2 | 256 | 12 | `avx2_fma_openmp` |
| sse | 1,024 | Operon | 256 | 12 | `evaluate_trees_then_numpy_sse` |
| sse | 1,024 | PySR | 256 | 24 | `julia_threads_turbo` |
| sse | 1,024 | SECANT CUBIN | 256 | 1 | `kernel_only_k8_a32_tile128_streams8` |
| sse | 1,024 | SECANT CUDA | 256 | 1 | `kernel_only_k8_a32_tile128_streams8` |
| sse | 1,024 | SECANT PTX | 256 | 1 | `kernel_only_k8_a32_tile128_streams8` |
| sse | 16,384 | EvoGP | 256 | 1 | `hybrid_parallel` |
| sse | 16,384 | Kozax | 256 | 1 | `jit_symbolic_regression_fitness` |
| sse | 16,384 | Native AVX2 | 256 | 12 | `avx2_fma_openmp` |
| sse | 16,384 | Operon | 256 | 24 | `evaluate_trees_then_numpy_sse` |
| sse | 16,384 | PySR | 256 | 24 | `julia_threads_turbo` |
| sse | 16,384 | SECANT CUBIN | 256 | 1 | `kernel_only_k8_a32_tile1024_streams8` |
| sse | 16,384 | SECANT CUDA | 256 | 1 | `kernel_only_k8_a32_tile1024_streams8` |
| sse | 16,384 | SECANT PTX | 256 | 1 | `kernel_only_k8_a32_tile1024_streams8` |
| sse | 262,144 | EvoGP | 256 | 1 | `hybrid_parallel` |
| sse | 262,144 | Native AVX2 | 256 | 24 | `avx2_fma_openmp` |
| sse | 262,144 | Operon | 256 | 24 | `evaluate_trees_then_numpy_sse` |
| sse | 262,144 | PySR | 256 | 24 | `julia_threads_turbo` |
| sse | 262,144 | SECANT CUBIN | 256 | 1 | `kernel_only_k8_a32_tile1024_streams8` |
| sse | 262,144 | SECANT CUDA | 256 | 1 | `kernel_only_k8_a32_tile1024_streams8` |
| sse | 262,144 | SECANT PTX | 256 | 1 | `kernel_only_k8_a32_tile1024_streams8` |

## Timing Boundary

All programs and inputs are prepared before timing. SECANT uses CUDA events around kernel launches after CUBIN specialization, module load, allocation, host-to-device copies, and CPU verification. Input columns and SSE targets are already resident in device memory. Output clearing is ordered before the start event. External collectors use their warmed backend evaluation APIs and synchronize GPU work where applicable. Search, mutation, and expression construction are excluded.

The Native AVX2 collector calibrates multiple complete evaluations into each long timed sample. A compiler memory barrier follows every evaluation so repeated writes cannot be collapsed. Its CSV notes record the actual iterations per sample.

Every plotted marker is a measured row count. The graph does not connect markers by default, and therefore does not visually interpolate between measurements. SECANT throughput is calculated from the exact number of timed batches, kernel launches, and row evaluations reported by the runtime executable. Each timed batch executes the same fixed set of 256 unique expressions against the resident data, which isolates steady-state evaluation from AST specialization and module loading.

Materialize writes every AST-row result. SSE/MSE rows use each backend's available fused or materialize-then-reduce path; inspect each CSV's `execution_mode` and `notes` fields before treating reduction ratios as identical kernel contracts.

The SECANT raw audit contains 729 accepted timing samples, 15,428,916 executed kernel launches, and 10,127,764,094,976 executed row evaluations. The shortest accepted sample ran for 48.890 ms.

## Raw Data

- [`baseline_secant_backends_2026-07-26.csv`](baseline_secant_backends_2026-07-26.csv)
- [`baseline_native_avx_2026-07-26.csv`](baseline_native_avx_2026-07-26.csv)
- [`baseline_pysr_2026-07-26.csv`](baseline_pysr_2026-07-26.csv)
- [`baseline_evogp_2026-07-26.csv`](baseline_evogp_2026-07-26.csv)
- [`baseline_kozax_2026-07-26.csv`](baseline_kozax_2026-07-26.csv)
- [`baseline_operon_2026-07-26.csv`](baseline_operon_2026-07-26.csv)
- [`baseline_secant_samples_2026-07-26.csv`](baseline_secant_samples_2026-07-26.csv) (every SECANT probe, calibration run, and accepted timing sample)
