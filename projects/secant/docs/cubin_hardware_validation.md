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

# CUBIN Hardware Validation

This document records correctness validation of SECANT's static-column f32
materialize and SSE CUBIN shapes. These are execution results, not only source
generation, CUBIN inspection, or specialization results.

## Environment

Validation was performed on 2026-07-28 using a Release build of SECANT.
Generated CUDA templates were compiled for the GPU's native compute capability
with NVRTC and ptxas optimization level 1.

| GPU | Compute capability | Memory | Host CPU | Driver | CUDA toolkit |
|---|---:|---:|---|---:|---:|
| NVIDIA A100-SXM4-40GB | `sm_80` | 40,960 MiB | AMD EPYC 7J13, 30 vCPUs | 580.105.08 | 12.8, V12.8.93 |
| NVIDIA A10 | `sm_86` | 23,028 MiB | Intel Xeon Platinum 8358, 30 vCPUs | 580.105.08 | 12.8, V12.8.93 |

Both systems used GCC 13.3.0 on x86-64 Linux. The tested source was `master`
at `353affe` plus comment-only table formatting in `include/secant.h`; the
CUBIN implementation was unchanged.

## Coverage

Both systems passed the same validation sequence:

- All four configured CTest tests: CPU oracle, C++ public-header compilation,
  deterministic CUBIN execution, and CUBIN random stress smoke test.
- Full random stress execution of 4,096 ASTs using eight inputs, three targets,
  257 rows, and programs from 2 through 57 instructions.
- Materialize and SSE comparison against the independent CPU interpreter.
- Actual execution with 96 ASTs in one kernel and two SSE targets.
- Row-boundary execution at 1, 127, 128, 129, 255, 256, and 257 rows.
- Four deeper random campaigns per GPU, totaling 1,024 additional ASTs per
  GPU with up to 16 leaves and unary depth 3.
- Compute Sanitizer memcheck, initcheck, synccheck, and racecheck.

The deepest generated program contained 86 instructions on the A100 and 87
instructions on the A10.

## Numerical Results

The common 4,096-AST stress corpus produced the same maximum errors on both
GPUs:

| AST class and shape | Maximum absolute error | Maximum relative error |
|---|---:|---:|
| ALU materialize | 0 | 0 |
| ALU SSE | 0.000854492 | 9.30786e-7 |
| MUFU materialize | 1.35265e-5 | 0.116798 |
| MUFU SSE | 0.00213623 | 1.72498e-5 |

The large MUFU materialize relative error occurs for a result close to zero;
its absolute error remained 1.35265e-5. SSE differences include the expected
effect of a different floating-point reduction order.

All sanitizer runs completed without findings:

| Sanitizer | Result |
|---|---|
| memcheck | 0 errors |
| initcheck | 0 errors |
| synccheck | 0 errors |
| racecheck | 0 hazards, 0 errors, 0 warnings |

## SM8x Branch Encoding

CUBIN inspection recovered `sm_80` and `sm_86` from each ELF image and selected
the SM8x branch encoding during specialization. The random and high-capacity
ASTs consumed less than the complete patch island, so execution depended on
the injected branch skipping the remaining `BPT.TRAP` padding and reaching the
original epilogue.

Both GPUs therefore validate the SM8x byte-relative branch calculation in
actual patched kernels. A wrong target would have executed a trap, skipped the
epilogue, or produced an output mismatch.

## Scope

These results provide strong evidence for the current CUBIN contracts on
`sm_80` and `sm_86` with CUDA 12.8. They do not establish compatibility with
every CUDA toolkit release, driver release, or other supported compute
capability. Each additional architecture and materially different toolkit
should run the same deterministic, stress, boundary, and sanitizer sequence.
