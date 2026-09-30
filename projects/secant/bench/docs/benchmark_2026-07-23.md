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

# SECANT Benchmark: 2026-07-23

## System

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB
- Driver: 595.71.05
- CUDA toolkit: 13.1
- CPU: AMD Ryzen 9 9900X, 12 cores / 24 threads
- Build: Release

## Workloads

| AST mode | Expression |
|---|---|
| ALU | Balanced eight-leaf tree with seven add, multiply, min, or max reductions |
| MUFU | The same tree with per-leaf sin, cos, ex2, safe sqrt, or safe reciprocal sqrt and occasional safe division |

## Compile

The timer measures only the backend's AST-to-CUBIN call. CUDA includes CUDA C++
generation and NVRTC compilation. PTX includes AST lowering, PTX injection, and
nvPTXCompiler compilation. Handle creation, template creation, AST generation,
scratch allocation, module loading, and execution are excluded.

Each worker performs one untimed warmup compile and two timed compiles. The
1/12/24-worker rows therefore use 1/12/24 warmup modules and 2/24/48 timed
modules. Internal split compilation is limited to one thread and compiler
caches are disabled. Shape is `kernels/module x ASTs/kernel`. PTX uses virtual
`sm_80` and native `sm_120`.

For CUDA, O0/O1 selects the ptxas optimization level while the NVRTC front end
remains optimized. For PTX, it selects the nvPTXCompiler optimization level.

`bench/compile_matrix.py` measures this complete row cross product:

```text
(1, 12, 24 workers)
    x (PTX, CUDA)
    x (materialize, SSE)
    x (O0, O1)
    x (ALU, MUFU)
```

Every row is measured against the same module-shape columns: `8x24`, `16x24`,
`32x24`, `64x8`, `64x16`, and `64x64`. This keeps the matrix bounded while
covering small CUDA-oriented and larger PTX-oriented module packings. The
PTX/CUDA measurements are recorded in `compile_matrix_2026-07-23_ptx.csv` and
`compile_matrix_2026-07-23_cuda.csv`. The July 24 analysis also includes the
O1-only direct CUBIN matrix from `compile_matrix_2026-07-24_cubin.csv`.

## Runtime

Runtime uses one module with 16 kernels, 1,048,576 rows, 3 warmups, and 100
timed iterations. GPU events measure kernel execution only. Compilation,
module loading, allocation, copies, output clearing, warmups, and verification
are excluded.

SSE logical tile size and ASTs/kernel were swept jointly with 128 threads.
Each thread streams `tile rows / 128` rows before the shuffle reduction,
shared-memory reduction, and atomic accumulation.

| Path | Opt | ALU rows/CTA | ALU ASTs/kernel | ALU row-evals/s | MUFU rows/CTA | MUFU ASTs/kernel | MUFU row-evals/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| PTX SSE | O0 | 1,024 | 32 | 3.853e11 | 1,024 | 32 | 2.765e11 |
| PTX SSE | O1 | 4,096 | 128 | **2.795e12** | 4,096 | 96 | **1.703e12** |
| CUDA SSE | O0 | 2,048 | 48 | 4.589e11 | 1,024 | 16 | 2.888e11 |
| CUDA SSE | O1 | 4,096 | 128 | **2.827e12** | 4,096 | 96 | **1.704e12** |

O1 SSE peaks around 96-128 ASTs/kernel and 4,096 rows/CTA. O0 peaks at smaller
logical tiles and lower packing. It remains substantially slower because its
generated reduction and accumulation code is not optimized as effectively.

### Fixed-Packing Tile Sweep

The tile sweep uses 16 kernels, 128 threads, and 1,048,576 rows. PTX and CUDA
use 96 ASTs/kernel; CUBIN uses 64 because its compiler-generated SSE skeleton
reaches a register-allocation boundary above that packing. Each point uses
3 warmups and 100 timed iterations.

![SSE runtime throughput by logical tile size](runtime_tile_sweep_2026-07-24.svg)

All four O0/O1 and ALU/MUFU configurations peak at 4,096 rows per CTA. At O1,
CUDA and PTX are effectively coincident: the 4,096-row peaks are 2.679e12
versus 2.651e12 ALU row-evals/s and 1.704e12 versus 1.703e12 MUFU row-evals/s.
At O0, CUDA is faster because its NVRTC device optimizer simplifies the
generated program before the ptxas O0 stage; injected PTX compiled by
nvPTXCompiler O0 receives less cleanup.

The source measurements are in
[`runtime_tile_sweep_2026-07-24_ptx.csv`](runtime_tile_sweep_2026-07-24_ptx.csv),
[`runtime_tile_sweep_2026-07-24_cuda.csv`](runtime_tile_sweep_2026-07-24_cuda.csv),
and
[`runtime_tile_sweep_2026-07-24_cubin.csv`](runtime_tile_sweep_2026-07-24_cubin.csv).

### AST Packing and Tile Surface

The materialize packing sweep fixes 16 kernels, 128 threads, 1,048,576 rows,
3 warmups, and 100 timed iterations. Every materialize kernel writes to a
distinct output region. The SSE surface uses the same run configuration while
crossing packing with logical tile size.

![Materialize runtime by AST packing](runtime_materialize_packing_2026-07-24.svg)

Materialize ALU peaks at four ASTs/kernel under O0, while both O1 modes peak at
two. O0 MUFU instead continues rising through the measured 192 ASTs/kernel.
After the early peak and dip, dense packing recovers toward 3.4e11 to 3.5e11
row-evals/s. CUDA and PTX remain closely aligned throughout the sweep.

Materialize has no logical tile-size parameter: it launches fixed 256-thread
blocks with one thread per row. The `tile_rows` dimension shown for SSE
therefore has no materialize counterpart.

The SSE sweep crosses every AST packing with every logical tile size. Tile
sizes are separate lines, while backend, expression mode, and optimization
level have separate panels. The O0 and O1 columns use independent linear Y
scales.

![SSE runtime by AST packing and logical tile size](runtime_sse_surface_2026-07-24.svg)

SSE needs substantially more packing to amortize its reduction and atomic
epilogue. The useful packing range moves upward with logical tile size. O1 ALU
peaks at 128 ASTs/kernel, while O1 MUFU peaks at 96. MUFU turns over more
sharply because its larger expressions consume more code and register
resources. CUDA and PTX track closely through the useful O1 range.

The source measurements are split by backend:
[`runtime_packing_sweep_2026-07-24_ptx.csv`](runtime_packing_sweep_2026-07-24_ptx.csv),
[`runtime_packing_sweep_2026-07-24_cuda.csv`](runtime_packing_sweep_2026-07-24_cuda.csv),
[`runtime_packing_sweep_2026-07-24_cubin.csv`](runtime_packing_sweep_2026-07-24_cubin.csv),
[`runtime_sse_surface_2026-07-24_ptx.csv`](runtime_sse_surface_2026-07-24_ptx.csv),
[`runtime_sse_surface_2026-07-24_cuda.csv`](runtime_sse_surface_2026-07-24_cuda.csv),
and
[`runtime_sse_surface_2026-07-24_cubin.csv`](runtime_sse_surface_2026-07-24_cubin.csv).

## Reduction Launch Shape

The 4,096-row SSE tile is a logical work assignment, not a shared-memory tile.
With 128 threads, each thread streams 32 rows while retaining the same
register-resident accumulators. The reduction and atomic epilogue therefore run
once per 4,096 rows instead of once per 128 rows.

A future launcher can express this amortization without compiling a fixed tile
size. It can launch a caller-selected number of CTAs and use:

```cuda
for (size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
     row < num_rows;
     row += (size_t)gridDim.x * blockDim.x) {
    // Accumulate thread-local SSE.
}
```

For example, 256 CTAs over 1,048,576 rows still assign approximately 4,096 rows
per CTA. This permits runtime tuning through a direct maximum-CTA argument
without requiring an occupancy query. A true shared-memory tile remains a
separate, smaller physical unit that a CTA may visit repeatedly before its
final reduction.
