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

# Cooperative LM CUDA shape study

This experiment separates the CUDA execution topology from Odezza's SASS specialization. It then applies the surviving topology to the real specialized-AST path on both development GPUs.

The fixed diagnostic system is `dx_i/dt = k_i*x_i + 0.01*x_((i+1) mod N)`. Every fit uses three 21-point trajectories, four RK4 substeps per interval, eight fitted constants, at most six LM iterations, and at most eight damping attempts. The 8x8 saturated runs average 3.9928 completed iterations and 4.0431 Cholesky attempts per fit. Reported fit rates therefore cover complete multi-iteration LM fits, not one trajectory evaluation or one LM iteration.

## CUDA-only topology result

The fixed CUDA control compiles the RHS and analytic partials normally. It proves correctness and measures ownership without involving marker inspection or patched SASS.

| Threads per fit | Replicated solve fits/s | Distributed solve fits/s | Registers, replicated | Stack |
|---:|---:|---:|---:|---:|
| 1 | 27.992M | 27.528M | 255 | 0 B |
| 2 | 13.502M | 13.435M | 168 | 32 B |
| 4 | 9.373M | 9.346M | 142 | 0 B |
| 8 | 3.272M | 3.301M | 114 | 0 B |
| 16 | 1.149M | 1.149M | 114 | 0 B |
| 32 | 0.574M | 0.572M | 112 | 0 B |

Distributing the 8x8 factorization changes throughput by less than approximately one percent. The worthwhile partition is therefore the forward-sensitivity and normal-equation state; replicating the small solve is simpler and effectively free in this workload.

Making only the subgroup leader evaluate the RHS and local partials is harmful. It reaches 2.821M fits/s with four threads and 0.984M with eight threads, versus 9.373M and 3.272M when every lane evaluates the identical RHS. The production topology consequently replicates specialized AST execution in each lane and uses shuffles only for the parameter-partitioned optimizer state.

## Specialized 8-state, 8-constant result

All widths recover the same constants, optimizer counters, and fit. The best GPU MSE is `4.33e-15`; independent CPU replay is `1.76e-15`. Across eight fixed spot checks, the largest absolute GPU-versus-CPU MSE difference is `7.04e-15`.

| GPU | Threads per fit | Fits/s | Registers | Stack | SASS instructions |
|---|---:|---:|---:|---:|---:|
| RTX 5090 | 1 | 1.030M | 255 | 304 B | 6,224 |
| RTX 5090 | 4 | 0.532M | 158 | 0 B | 6,560 |
| RTX 5090 | 8 | 0.172M | 134 | 0 B | 7,256 |
| RTX 5090 | 32 | 0.0496M | 127 | 0 B | 7,392 |
| RTX 4090 | 1 | 1.226M | 255 | 608 B | 6,080 |
| RTX 4090 | 4 | 0.879M | 194 | 0 B | 5,424 |
| RTX 4090 | 8 | 0.651M | 155 | 0 B | 5,432 |
| RTX 4090 | 32 | 0.170M | 128 | 0 B | 5,568 |

For 8x8, one-thread ownership is the throughput winner even though it spills. Four-thread ownership is the best spill-free fallback. The RTX 4090 unexpectedly wins this specialized control-heavy workload. The measured SM89 kernels contain materially fewer SASS instructions, especially at cooperative widths; that is evidence for an architecture/code-generation effect, although the instruction count alone does not prove the entire cause.

## Sixteen-state boundary

The 16-state system uses the same eight shared constants, repeated across the 16 right-hand sides.

| GPU | Threads per fit | Result | Fits/s | Registers | Stack |
|---|---:|---|---:|---:|---:|
| RTX 5090 | 1 | Physical inspection rejected aliased output registers | — | 255 | 1,264 B |
| RTX 5090 | 4 | Correct GPU fit and CPU replay | 47,413 | 236 | 0 B |
| RTX 5090 | 8 | Correct GPU fit and CPU replay | 22,521 | 198 | 0 B |
| RTX 4090 | 1 | Physical inspection rejected aliased output registers | — | 255 | 1,272 B |
| RTX 4090 | 4 | Correct GPU fit and CPU replay | 82,233 | 254 | 0 B |

This is the actual reason to retain subgroup ownership. It is not an 8x8 speed optimization. At 16x8, the thread-owned kernel is both heavily spilled and physically unsafe to specialize, while four-thread ownership executes correctly without stack traffic on both GPUs.

## Decision

- Keep one-thread LM as the default for shapes that pass physical inspection and a resource policy.
- Keep four-thread cooperative LM as the first larger-system fallback.
- Do not distribute the Cholesky solve for eight constants; it adds complexity without measured benefit.
- Do not make one lane own specialized RHS evaluation; replicated lane execution is substantially faster.
- Treat eight- and 32-thread ownership as capacity fallbacks to be selected from measured resource limits, not default performance shapes.
- Select the topology after actual target-architecture compilation and inspection. State/constant dimensions alone do not predict SM89 and SM120 code generation closely enough.

Raw data: [`raw/2026-09-02-cooperative-lm-cuda-shape.json`](raw/2026-09-02-cooperative-lm-cuda-shape.json)
