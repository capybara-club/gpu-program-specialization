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

# Two-site trajectory-LM layout study

> **Superseded:** These measurements used an incorrect in-place update for coupled RK4 sensitivity stages. Primal trajectory replay remains valid, but the LM gradients, optimizer path, work counters, and fit rates are not current correctness evidence. Use [`2026-09-02-corrected-coupled-lm.md`](2026-09-02-corrected-coupled-lm.md).

This study evaluates an experimental LM layout with two physical specialization sites:

1. One site emits the primal right-hand-side state derivatives into registers.
2. One site emits every local state/parameter partial into a per-thread shared-memory arena as the CUDA sensitivity code consumes it.

The existing grouped-v4 layout remains the production control. It emits primal values and local partials through operand-bounded register groups.

## Result

The two-site layout is correct and useful as a capacity escape hatch, but it is not a universal throughput replacement for grouped-v4.

| Workload | GPU | Grouped-v4 | Two-site-v5 | Relative | Grouped regs | Two-site regs |
|---|---:|---:|---:|---:|---:|---:|
| Fed-batch, 4 states / 6 constants | RTX 5090, SM120 | 87,636 fits/s | 89,225 fits/s | +1.8% | 217 | 217 |
| Fed-batch, 4 states / 6 constants | RTX 4090, SM89 | 78,865 fits/s | 64,997 fits/s | -17.6% | 223 | 225 |
| Synthetic, 2 states / 8 constants | RTX 5090, SM120 | 7.31M fits/s | 6.90M fits/s | -5.6% | 188 | 174 |
| Synthetic, 2 states / 8 constants | RTX 4090, SM89 | 5.72M fits/s | 5.47M fits/s | -4.3% | 196 | 174 |

Every reported comparison uses the same starts, toggles, trajectories, integration work, LM iteration cap, damping schedule, and loaded-kernel timing scope. Both layouts produced identical LM work counters and final numerical results. The fed-batch best MSE was `2.745058107417927e-14`, with CPU replay MSE `8.292809951403466e-15`.

The clean 4-state/6-constant two-site specialization emits 34 primal-site instructions and 165 partial-site instructions. It uses no local stack or spill memory. Static shared use is 6,144 bytes on SM120 and 5,120 bytes on SM89.

## Dependency defect found during development

The first shared-partial implementation copied the compiler's `STS` control word but did not honor its read-dependency barrier. It reused SASS source registers before the shared-store pipeline had consumed them. Small grids happened to work; saturated grids produced `NaN` gradients and Gram matrices. The resulting apparent 2.65M fits/s was invalid and is excluded.

The correct implementation assigns up to six outstanding stores to distinct read-dependency slots, pins their source registers, waits once for the batch, and emits a non-collective CTA memory fence before CUDA loads the partials. A conservative wait after every store was also correct but reached only 61,214 fits/s.

## Capacity result

Compiling one large state-capacity template and reusing it for smaller systems is not cheap under the current thread-owned sensitivity topology:

| Template capacity | RTX 5090 | RTX 4090 |
|---|---:|---:|
| 6 states / 6 constants | 255 regs, 24 B stack/thread | 255 regs, 32 B stack/thread |
| 8 states / 6 constants | 255 regs, 136 B stack/thread | 255 regs, 240 B stack/thread |

CUDA allocates the full `state_count × constant_count` sensitivity state before SASS specialization. Unused state registers therefore cannot simply be reclaimed afterward. Exact-size templates or a few small cached buckets are preferable. Six or more states with six LM constants should trigger a distinct LM topology investigation rather than silently using an 8-state universal template.

## Recommendation

- Keep grouped-v4 as the default when it fits the inline-assembly operand and register budgets.
- Retain two-site-v5 as an optional capacity layout and for architectures/shapes where measurement shows a win.
- Do not add a one-site shared-output variant yet. It would also route primal RHS values through shared memory and add stores, loads, and dependency bookkeeping without addressing the dominant sensitivity/Gram register state.
- Cache compiled exact-size templates first. If cache cardinality becomes a real service problem, use narrow state/constant buckets and benchmark each bucket's spill cost.
- Treat 6-state/6-constant and larger thread-owned LM shapes as a topology boundary, not ordinary overallocation.

Raw results are in `benchmarks/raw/2026-09-02-two-site-lm-layout.json`.
