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

# RTX 5090 Active AST Capacity

Measurements from July 12, 2026 for the prefix-only SASS patcher and the split
between generated AST capacity and active patched AST count.

```text
GPU             = NVIDIA GeForce RTX 5090
compute         = sm_120
driver          = 595.71.05
CUDA            = 13.1.115
build           = Release
tile rows       = 64
CTA threads     = 128
columns         = 32
settings        = 1,024
runtime rows    = 1,048,576
runtime iters   = 30
kernels/module  = 8
```

## Implementation

`cusr_ast_sass_generate` now lowers exactly one postorder AST. It emits neither
scoring instructions nor a terminating branch or padding. The cubin patcher
owns site composition:

1. Lower each active AST.
2. Append `error = prediction - target`.
3. Append `sse = error * error + sse`.
4. Reuse temporary registers for the next AST.
5. Append one branch over the unused site tail.
6. Write only the generated prefix and branch into the cubin.

Instructions after the branch remain unchanged. This makes patch work scale
with active ASTs and expression size instead of generated site capacity.

Inspection records `ast_capacity`. The patch call may provide any uniform,
positive active AST count up to that capacity. The kernel receives that active
count as `num_asts`, so inactive SSE outputs are not written.

## Runtime

All runs used 8 active ASTs per kernel. ALU programs were balanced depth-3
8-leaf trees. MUFU programs added one `sin`, `cos`, `ex2`, or protected
`rsqrt` operation to every leaf before the same reduction tree.

| AST mode | Capacity / active | Patched registers | Blocks / SM | Row evals/s |
|---|---:|---:|---:|---:|
| depth3 ALU | 8 / 8 | 49 | 9 | `3.268e12` |
| depth3 ALU | 32 / 8 | 70 | 7 | `3.514e12` |
| depth3 MUFU | 8 / 8 | 50 | 9 | `9.597e11` |
| depth3 MUFU | 32 / 8 | 71 | 7 | `9.662e11` |

Capacity 32 did not penalize runtime when only 8 ASTs were active. It was 7.5%
faster for this ALU run and 0.7% faster for MUFU. The reason is not extra work:
both capacities wrote exactly 584 SASS instructions for ALU and 1,723 for MUFU
across the 8 sites. The branch skips the inactive tail.

The CPU comparison passed for every run. Maximum absolute error was
`1.76e-4` or lower. The GPU test also runs capacity 8 and capacity 32 over the
same 8 ASTs and compares both results directly.

## Patch And Load

The patch sweep used 128 modules and 8 kernels per module, for 1,024 sites and
8,192 active ASTs. Inspection, allocation, cubin copying, and module loading
were outside the patch timer.

| AST mode | Capacity / active | Patch ASTs/s | Patch us/AST | Load us/AST | Cubin/module |
|---|---:|---:|---:|---:|---:|
| ALU reduce | 8 / 8 | `5.713e6` | `0.175` | `1.070` | 136,208 B |
| ALU reduce | 32 / 8 | `5.451e6` | `0.183` | `2.087` | 365,848 B |
| depth3 MUFU | 8 / 8 | `2.297e6` | `0.435` | `0.995` | 136,208 B |
| depth3 MUFU | 32 / 8 | `2.252e6` | `0.444` | `1.789` | 365,848 B |

The larger capacity mildly reduces patch locality for short ALU programs and
roughly doubles module-load cost per active AST because the complete skeleton
cubin is still larger. MUFU lowering dominates patch time, so patch throughput
is nearly unchanged.

With capacity 32 and all 32 ASTs active, the depth3 ALU patcher processed
32,768 ASTs at `5.22e6` to `5.33e6 ASTs/s`, or `0.188` to `0.192 us/AST`.
A detached build of the pre-refactor full-padding implementation measured
`2.49e6` and `3.17e6 ASTs/s` for the same shape. The new path is at least 1.65
times faster than the better historical run.

## Commands

Runtime capacity comparison:

```sh
./build-release/cusr_bench --preset runtime --variant tile_static_mse_template \
  --ast-mode depth3-alu --ast-capacity 32 --asts-per-kernel 8 \
  --run-rows 1048576 --run-iters 30 --check-rows 4096
```

Patch comparison:

```sh
./build-release/cusr_bench --preset patch --modules 128 --kernels-per-module 8 \
  --ast-mode alu-reduce --ast-capacity 32 --asts-per-kernel 8
```

## Recommendation

A capacity-32 skeleton can safely service smaller active cohorts without
running inactive ASTs. It is a useful cached general-purpose shape when runtime
throughput matters most. A capacity-8 or capacity-16 skeleton is still useful
when module-load latency, cubin size, or short-program patch locality matters.
