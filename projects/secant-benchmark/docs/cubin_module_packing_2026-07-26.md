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

# CUBIN Module Packing

## Configuration

The packing sweep used an NVIDIA GeForce RTX 5090 with:

- 48 modules and 24 compiler workers
- 128 ALU ASTs/kernel
- 262,144 rows/AST
- 4,096 rows/CTA and 128 threads/CTA
- 8 pre-created CUDA streams
- one target and one timed execution per module
- explicit `cuFuncLoad` for every enumerated kernel

Inputs and targets were resident before timing. CUBIN template generation,
inspection, template copies, allocation, transfers, and correctness checks
were outside the pipeline timer.

## Results

| Kernels/module | ASTs/module | Module load | Kernel rows/s | Pipeline rows/s |
|---:|---:|---:|---:|---:|
| 8 | 1,024 | 14.1 ms | 1.318e12 | 3.016e11 |
| 16 | 2,048 | 28.7 ms | 1.744e12 | **3.378e11** |
| 32 | 4,096 | 65.4 ms | 2.042e12 | 3.366e11 |
| 64 | 8,192 | 149.1 ms | 2.239e12 | 3.193e11 |
| 128 | 16,384 | 315.4 ms | **2.310e12** | 3.203e11 |

Kernel throughput approaches its plateau between 64 and 128 kernels/module.
Module residency cost continues to scale with module size, so 128 kernels
does not improve complete-pipeline throughput. Sixteen kernels produced the
highest measured pipeline rate, while 64 kernels is the practical default
when prioritizing near-peak kernel throughput and enough AST work per module.

## Module Transition

The CUDA path performs:

1. Wait on one stop event that depends on all launch streams.
2. Read event timing.
3. Unload the completed module.
4. Load the next module.
5. Enumerate every function and map its numeric suffix to an array slot.
6. Call `cuFuncLoad` for every function.
7. Launch kernels on the pre-created streams.

There is no device-wide synchronization before the first load or between
modules. The event wait is required because the previous module cannot be
unloaded while its kernels are still executing. CUDA does not expose a
stream-ordered asynchronous module-load operation that can safely overlap this
transition with kernels in the same context.

`cuModuleEnumerateFunctions` has no loading flag. It discovers handles;
`cuFuncLoad` is the explicit residency operation.

## Embedded Template

The 64-kernel template can be generated during the build and embedded with
`incbin.h`. The build artifact and a fresh runtime NVRTC artifact were
byte-identical.

In separate process runs, the embedded-only path measured 318 ms of aggregate
module load while the process that first invoked NVRTC measured 149 ms. Kernel
throughput was unchanged at approximately 2.24e12 row-evals/s. This indicates
CUDA process-state warming rather than different generated code. Embedding
removes cold template compilation, but it should not currently be presented as
a module-load optimization.

## Tile Follow-up

A current-shape sweep with 64 kernels/module, 128 ASTs/kernel, 1,048,576 rows,
and 50 timed iterations found 8,192 rows/CTA to be the runtime peak:

| Rows/CTA | ALU rows/s | MUFU rows/s |
|---:|---:|---:|
| 1,024 | 1.647e12 | 6.082e11 |
| 2,048 | 2.178e12 | 6.509e11 |
| 4,096 | 2.601e12 | 6.782e11 |
| 8,192 | **2.875e12** | **6.875e11** |
| 16,384 | 2.769e12 | 6.559e11 |
| 32,768 | 1.559e12 | 3.731e11 |

The pipeline and embedded-template defaults therefore use 8,192 rows/CTA.
The earlier module-packing table remains a 4,096-row measurement and should
not be mixed with this runtime-only tile sweep.

The measured data and figure are
[`runtime_sse_tile_rtx5090_2026-07-26.csv`](runtime_sse_tile_rtx5090_2026-07-26.csv)
and
[`runtime_sse_tile_rtx5090_2026-07-26.svg`](runtime_sse_tile_rtx5090_2026-07-26.svg).
