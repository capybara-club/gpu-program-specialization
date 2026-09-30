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

# Direct CUBIN Benchmark

## System

- GPU: NVIDIA GeForce RTX 5090, `sm_120`, driver 595.71.05
- CPU: AMD Ryzen 9 9900X, 12 cores / 24 threads
- Toolkit: CUDA 13.1
- Build: CMake `Release`
- ASTs: balanced eight-leaf trees
- ALU workload: seven randomly selected add, multiply, min, or max operations
- MUFU workload: per-leaf approximate math plus reductions and safe routines

## Compile

The measurement times only `secant_cubin_specialize_into`: AST lowering, direct
SASS generation, and writes into an initialized caller-owned CUBIN. CUDA
skeleton generation, external NVRTC compilation, CUBIN inspection, allocation,
copying, module loading, and correctness execution are outside the timer.

Best result in each row of the common module-shape matrix:

| Kernel | AST mode | Workers | Module shape | AST/s | us/AST |
|---|---:|---:|---:|---:|---:|
| Materialize | ALU | 1 | 64x64 | 7,614,826 | 0.131 |
| Materialize | MUFU | 1 | 64x16 | 2,856,238 | 0.350 |
| SSE | ALU | 1 | 32x24 | 7,728,456 | 0.129 |
| SSE | MUFU | 1 | 64x8 | 2,973,529 | 0.336 |
| Materialize | ALU | 24 | 64x64 | 50,260,843 | 0.020 |
| Materialize | MUFU | 24 | 64x64 | 19,437,419 | 0.051 |
| SSE | ALU | 24 | 64x64 | 53,518,537 | 0.019 |
| SSE | MUFU | 24 | 64x64 | 21,077,881 | 0.047 |

Module shape is `kernels/module x ASTs/kernel`. The complete matrix is
[`compile_matrix_2026-07-24_cubin.csv`](compile_matrix_2026-07-24_cubin.csv).

## Runtime

All runs use 16 kernels, 1,048,576 rows, eight input columns, one SSE target,
128 threads for SSE, and 100 timed iterations. Runtime includes kernel
execution only.

| Kernel | AST mode | ASTs/kernel | Tile rows | Row-evals/s |
|---|---:|---:|---:|---:|
| Materialize | ALU | 2 | n/a | 3.709e11 |
| Materialize | MUFU | 2 | n/a | 3.442e11 |
| SSE | ALU | 96 | 2,048 | 1.517e12 |
| SSE | MUFU | 96 | 2,048 | 5.474e11 |

Materialize writes every AST result and becomes output-bandwidth bound. SSE
keeps the error accumulation and reduction local. Its best measured point uses
96 ASTs/kernel; larger packings remain valid but reduce occupancy enough to
lose throughput at the best tile size.

The runtime data is recorded in:

- [`runtime_packing_sweep_2026-07-24_cubin.csv`](runtime_packing_sweep_2026-07-24_cubin.csv)
- [`runtime_tile_sweep_2026-07-24_cubin.csv`](runtime_tile_sweep_2026-07-24_cubin.csv)
- [`runtime_sse_surface_2026-07-24_cubin.csv`](runtime_sse_surface_2026-07-24_cubin.csv)

## Template Capacity

Capacity is selected during CUDA skeleton generation, validated during CUBIN
inspection, and distinct from the runtime optimum.

For the sm_120 CUDA 13.1 SSE template with eight inputs, one target, 128
threads, and 64 reserved SASS instructions per AST:

- 230 ASTs/kernel compiles and inspects successfully.
- At 231 ASTs/kernel, ptxas inserts a local-memory spill inside the protected
  patch island.
- The inspector rejects that image with
  `SECANT_ERROR_UNEXPECTED_INSTRUCTION`; the spill must not be overwritten.
- 96 ASTs/kernel with two targets patches successfully.

The 231 boundary is an empirical property of this template, architecture, and
toolkit. It is not a public AST-count limit. The parser validates the emitted
binary rather than assuming source-level register pressure or patch layout.
