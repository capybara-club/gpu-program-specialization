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

# Gram Kernels Per Module Sweep - 2026-06-17

This note records the hot compile throughput sweep for the patched AST compiler.
It measures how many generated Gram kernels should be packed into each cubin
module. The benchmark is compile-only; runtime is dominated by the Gram kernels
and should be evaluated separately for a search workload.

## Command Shape

~~~bash
./build-verify-fused-sm120/implicit_sindy_ast_compile_api_bench \
  --modules 512 \
  --kernels K \
  --workers 12 \
  --mode polynomial \
  --repeats 1 \
  --warmup 0 \
  --workspace-mb 256 \
  --sm sm_120

./build-verify-gh200-current/implicit_sindy_ast_compile_api_bench \
  --modules 512 \
  --kernels K \
  --workers 64 \
  --mode polynomial \
  --repeats 1 \
  --warmup 0 \
  --workspace-mb 256 \
  --sm sm_90
~~~

Each Gram kernel contains 32 AST feature sites, so `asts/s` is
`gram_kernels/s * 32`.

## Systems

| System | GPU | SM | Workers |
| --- | --- | ---: | ---: |
| Local workstation | NVIDIA GeForce RTX 5090 | 120 | 12 |
| Lambda GH200 | NVIDIA GH200 480GB | 90 | 64 |

## Results

| Gram kernels/module | RTX 5090 compile ms | RTX 5090 ASTs/s | GH200 compile ms | GH200 ASTs/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 710.228 | 23068.664 | 528.272 | 31014.347 |
| 2 | 1232.846 | 26579.150 | 651.885 | 50266.535 |
| 4 | 2272.145 | 28843.224 | 965.403 | 67884.627 |
| 8 | 4439.969 | 29520.930 | 1656.442 | 79128.633 |
| 16 | 8639.010 | 30344.218 | 3078.507 | 85152.965 |
| 32 | 17437.840 | 30066.109 | 5821.832 | 90055.500 |
| 64 | 34821.498 | 30112.891 | 11328.547 | 92560.501 |

## Interpretation

The large gain is from packing more than one Gram kernel per module. Moving from
1 to 4 kernels/module significantly improves hot compile throughput on both
systems. After that, the 5090 mostly flattens, while the GH200 continues to gain
from larger modules because it has many more CPU cores available to amortize the
per-module overhead.

Recommended defaults:

| System | Recommended kernels/module | Reason |
| --- | ---: | --- |
| RTX 5090 | 16 | Highest measured AST throughput; 8 is nearly as good with smaller modules. |
| GH200 | 64 | Highest measured AST throughput; 32 is a practical smaller-module option only about 2.8% slower. |

For interactive local search on the 5090, `8` or `16` is the practical range. For
large cloud runs on GH200-class systems, `32` or `64` is preferable when module
size and memory pressure are acceptable.
