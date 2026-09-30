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

# HIP Validation: 2026-07-23

## Environment

- GPU: AMD Radeon RX 9070 XT
- Architecture: `gfx1201`
- ROCm: 7.1
- Host workers: 24
- Build: Release, HIP enabled, CUDA disabled

The HIP benchmark discovers `gfx1201` from the selected device when
`--hip-arch` is omitted.

## Correctness

The CPU interpreter, C++ header, HIP generator, materialize pipeline, and SSE
pipeline pass CTest. The GPU tests generate random postorder MUFU ASTs, compile
them through HIPRTC, load the resulting HSACO, execute every kernel, and compare
against the CPU interpreter.

The random benchmark's safe-divide routine uses a `1e-2` stabilization term.
The earlier `1e-8` term made deep random expressions ill-conditioned enough
that approximate GPU math diverged materially from the CPU oracle.

HIP SSE permits 64 through 512 threads per block in multiples of 64. This
portable limit keeps the fixed 16-entry partial-wave array valid on both wave32
and wave64 devices.

## AST-To-HSACO Compile

Configuration: static-column materialize, MUFU ASTs, 16 kernels per module,
8 ASTs per kernel, 128 ASTs per module.

| Workers | Modules | ASTs | Time | AST/s | us/AST |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 256 | 2.575 s | 99.4 | 10,059 |
| 24 | 24 | 3,072 | 31.743 s | 96.8 | 10,333 |

Concurrent HIPRTC calls did not scale on this host. The 24-worker rate is
effectively the one-worker rate, indicating serialization or an equivalent
global compiler bottleneck.

## Runtime

Configuration: 16 kernels, 8 ASTs per kernel, MUFU mode, 1,048,576 rows,
5 timed iterations.

| Shape | Time | Row evaluations | Row-evals/s |
|---|---:|---:|---:|
| Materialize | 0.014697 s | 671,088,640 | 4.566e10 |
| SSE | 0.341577 s | 671,088,640 | 1.965e9 |

These are baseline generated-HIP kernels, not direct AMDGPU ISA injection.
