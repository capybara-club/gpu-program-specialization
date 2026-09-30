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

# RTX 5090 NVRTC Template Sweep

Date: July 13, 2026

Hardware and toolchain:

- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32 GiB
- Driver: 595.71.05
- CUDA toolkit: 13.1.115
- Build: Release
- CUDA cache disabled for compilation measurements

## Shape

NVRTC compiled the include-only tile-static MSE template. Each global kernel
specialization was requested with `nvrtcAddNameExpression`, and its exact
symbol was recovered with `nvrtcGetLoweredName`. No CUDA wrapper source was
generated.

The main sweep used capacity 32, 32 active ALU-reduction ASTs per kernel,
64-row tiles, and 128-thread CTAs. It held the total at 512 kernels and 16,384
ASTs by reducing module count as kernels/module increased. Runtime work was one
setting, one row, and one iteration so it did not dominate module-load timing.

## Results

| kernels/module | modules | cold NVRTC | cubin/module | total load | load/AST |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 0.162 s | 49,128 B | 16.950 ms | 1.035 us |
| 2 | 256 | 0.247 s | 95,416 B | 14.608 ms | 0.892 us |
| 4 | 128 | 0.479 s | 188,120 B | 8.397 ms | 0.512 us |
| 8 | 64 | 0.923 s | 373,400 B | 6.819 ms | 0.416 us |
| 16 | 32 | 1.832 s | 744,216 B | 5.873 ms | 0.358 us |
| 32 | 16 | 3.687 s | 1,485,720 B | 2.904 ms | 0.177 us |
| 64 | 8 | 7.433 s | 2,968,856 B | 2.716 ms | 0.166 us |
| 128 | 4 | 15.017 s | 5,935,512 B | 2.928 ms | 0.179 us |

The 8, 16, 32, and 64 points are medians of three independent processes. The
other points are single runs. Additional 36-, 40-, and 48-kernel probes landed
between 0.167 and 0.171 us/AST, confirming that steady-state load amortization
has flattened by the low 30s.

NVRTC cost is nearly linear after its fixed startup cost, at approximately
115 ms per capacity-32 kernel. Moving from 32 to 64 kernels doubles cold compile
latency and cubin size while improving median load cost by only about 6%. The
128-kernel point is slightly worse.

At 32 kernels/module, compile latency scales substantially with AST capacity:

| AST capacity | cold NVRTC | cubin/module |
|---:|---:|---:|
| 8 | 1.054 s | 567,184 B |
| 16 | 1.755 s | 875,928 B |
| 32 | 3.687 s | 1,485,720 B |

## Recommendation

Use 32 kernels/module when minimizing steady-state module-load cost matters.
For capacity 32 it reaches the practical load knee with a cold NVRTC latency of
about 3.7 seconds. Use 8 kernels/module when time to first execution matters:
it compiles in about 0.92 seconds and has already removed most single-kernel
module overhead. Sixteen is a reasonable intermediate point.

The wrapper-only CUDA generator did not improve either path and was removed.
The measured 8-, 32-, and 128-kernel shapes are now compiled once with NVCC and
embedded in the host executables. Noncanonical experiments use the C99 runtime
compiler handle, which embeds the same template and emits stable `extern "C"`
symbols without NVRTC name expressions.
