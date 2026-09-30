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

# Direct HSACO Validation: 2026-07-24

## Environment

- GPU: AMD Radeon RX 9070 XT
- Architecture: `gfx1201`
- ROCm: 7.1.52801
- Build: Release, HIP enabled, CUDA disabled
- Native backend: [`secant_hsaco.h`](../include/secant_hsaco.h)

## Pipeline

The generator creates one HIP template for static-column materialize or SSE.
The caller compiles it with HIPRTC, then the shape-specific inspect call parses
each kernel and patch island into an immutable plan in caller-owned storage.
The timed hot path is `secant_hsaco_specialize_into`: it lowers postorder ASTs
and writes directly into a caller-owned HSACO without allocation, scratch,
copying, HIPRTC, module loading, or synchronization.

Register expansion keeps all three HSACO representations consistent:

1. The kernel descriptor `compute_pgm_rsrc1` VGPR allocation.
2. The absolute `<kernel>.num_vgpr` ELF symbol.
3. The `.vgpr_count` field in the AMDGPU MsgPack metadata note.

A forced-expansion test changed one materialize kernel from 6 to 68 VGPRs. The
descriptor allocation changed from 8 to 72 registers, the absolute symbol
changed from 6 to 68, and the MsgPack value changed from `0x06` to `0x44`.
The patched kernel evaluated 131,072 rows and matched the CPU interpreter.

## Correctness

The GPU tests cover materialize and SSE, multiple kernels, multiple ASTs per
kernel, routines, constants, register expansion, and the following operations:

`add`, `sub`, `mul`, `div`, `neg`, `sqrt`, `rcp`, `rsqrt`, `abs`, `min`,
`max`, `fma`, `sin`, `cos`, `ex2`, `lg2`, and `tanh`.

All ten configured tests passed on `gfx1201`. Large benchmark verification also
passed for 1,048,576-row materialize and SSE runs.

`gfx1200` template compilation, ELF/MsgPack inspection, and direct ISA patching
also passed. Runtime execution on `gfx1200` hardware has not been tested.

## Patch Throughput

The module shape is 128 kernels with 8 ASTs per kernel. Inspection and HIPRTC
template compilation are outside the timer.

| Shape | AST mode | 1 worker | 24 workers |
|---|---:|---:|---:|
| Materialize | ALU | 9.35M AST/s | 63.3M AST/s |
| SSE, 2 targets | MUFU | 3.57M AST/s | 41.2M AST/s |

The measured 24-worker run used independent inspected plans, although the
current immutable plan contract permits one plan to be shared while workers
patch independent caller-owned HSACO buffers.

## Runtime

These runs use 8 kernels, 8 ASTs per kernel, 1,048,576 rows, and 10 timed
iterations.

| Shape | AST mode | Configuration | Row-evals/s |
|---|---:|---:|---:|
| Materialize | ALU | 256 threads | 1.375e11 |
| SSE, 2 targets | MUFU | 4,096 rows/CTA, 128 threads | 2.727e10 |

Row evaluations count one AST evaluated for one row. Targets do not multiply
the SSE count.

## Current Limits

- Direct ISA encoding is intentionally limited to RDNA 4 `gfx1200` and
  `gfx1201`.
- Patching preserves the original MsgPack integer width. If an expanded VGPR
  count does not fit that in-place representation, the operation returns
  `SECANT_ERROR_REGISTER_OVERFLOW` rather than emitting an inconsistent
  code object.
- Template creation and inspection are cold-path work. Specialization may
  partially modify the destination before returning an error; failed images
  must not be loaded.
