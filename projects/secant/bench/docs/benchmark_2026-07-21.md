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

# Secant RTX 5090 benchmark

> Historical results from the generic lifecycle benchmark before the backend-specific API split. The measurements remain useful; the benchmark driver is awaiting migration to the independent backend contracts.

Measured on July 21-22, 2026 using an NVIDIA GeForce RTX 5090 (`sm_120`), CUDA 13.1, and an AMD Ryzen 9 9900X with 12 cores and 24 threads. The project was built in Release mode with `-O3 -DNDEBUG`.

This report contains only the best measured compilation and runtime configurations. It is intended as a baseline for comparison with direct SASS injection.

## AST workloads

| AST | Definition |
|---|---|
| Simple | One input, one constant, and one add or multiply |
| ALU | Balanced eight-leaf tree with seven add, multiply, min, or max reductions |
| MUFU | Eight-leaf tree with per-leaf sin, cos, ex2, safe sqrt, or safe reciprocal sqrt and occasional safe division |

## Compilation

**Cell format: aggregate AST/s (aggregate wall-clock us/AST).** The parallel values are directly measured aggregate throughput across all workers, not single-worker results multiplied by the worker count.

| Shape | AST | NVRTC, 1 worker | NVRTC, 12 workers | NVRTC, 24 workers | PTX, 1 worker | PTX, 24 workers |
|---|---|---:|---:|---:|---:|---:|
| Materialize | Simple | 2,218 (450.95) | 17,053 (58.64) | **19,431 (51.46)** | 7,152 (139.81) | **81,739 (12.23)** |
| Materialize | ALU | 1,439 (695.10) | 10,440 (95.78) | **11,812 (84.66)** | **8,426 (118.68)** | **98,501 (10.15)** |
| Materialize | MUFU | 701 (1,425.87) | 4,212 (237.41) | **4,780 (209.20)** | 2,918 (342.72) | **34,776 (28.76)** |
| SSE | Simple | 149 (6,693.56) | 1,334 (749.79) | **1,647 (607.04)** | **563 (1,775.50)** | **6,756 (148.02)** |
| SSE | ALU | 176 (5,670.68) | 1,512 (661.46) | **1,848 (541.26)** | **532 (1,881.36)** | **6,283 (159.16)** |
| SSE | MUFU | 150 (6,662.74) | 1,251 (799.26) | **1,504 (664.74)** | **472 (2,119.89)** | **5,637 (177.39)** |

NVRTC uses 128 kernels per module and 8 ASTs per kernel. Its 24-worker measurements use 24 warmup modules and 48 timed modules, giving two timed worker waves. PTX materialize Simple and MUFU use the same packing with `-O0`. The record PTX materialize ALU result uses 64 kernels per module, 56 ASTs per kernel, and `-O0`. PTX SSE uses 128 kernels per module, 8 ASTs per kernel, and `-O1`; optimization is faster to compile for the larger SSE-generated programs.

The compilation timer surrounds only `secant_compile`. Handle creation, template preparation, AST generation, worker startup, scratch allocation, warmups, module loading, and execution are excluded. Compiler caches are disabled and internal split compilation is limited to one thread.

SSE compilation is slower because the current generator repeats target accumulation, reduction, and atomic epilogue code for every AST. MUFU compilation is slower than ALU because intrinsic and safety operations expand into more native-compiler work. These are compile-time costs and do not imply proportional runtime penalties.

## Runtime

All results use 1,048,576 rows, 2 modules, 32 kernels per module, and one CUDA stream. The `-O0` and `-O1` columns use identical generated PTX, AST packing, and launch configurations; only the nvPTXCompiler optimization level changes. CUDA events surround only kernel launch and execution. Compilation, module loading, allocation, copies, output initialization, and correctness checks are excluded.

| Shape | AST | ASTs/kernel | Rows/CTA | PTX `-O0` row-evals/s | PTX `-O1` row-evals/s |
|---|---|---:|---:|---:|---:|
| Materialize | ALU | 10 | - | 5.823e11 | **7.602e11** |
| Materialize | MUFU | 10 | - | 4.988e11 | **7.583e11** |
| SSE | ALU | 48 | 4,096 | 1.544e11 | **1.610e12** |
| SSE | MUFU | 48 | 4,096 | 9.766e10 | **1.164e12** |

The record SSE configuration processes 4,096 rows per CTA, so each thread processes 32 rows before shuffle reduction, shared-memory reduction, and atomic accumulation. The 1,048,576-row input still exposes 256 CTAs per kernel, reaching 1.610e12 ALU row-evals/s at `-O1`.

At `-O1`, materialize MUFU is effectively tied with materialize ALU because materialized output traffic dominates the additional expression work. SSE removes that output stream and becomes more compute-sensitive; its MUFU workload is 27.7% slower than ALU while still exceeding 1.1e12 row-evals/s.

The SSE shape depends heavily on native optimization. Relative to `-O0`, `-O1` improves ALU throughput by 10.4x and MUFU throughput by 11.9x. Materialize improves by 1.3x for ALU and 1.5x for MUFU.
