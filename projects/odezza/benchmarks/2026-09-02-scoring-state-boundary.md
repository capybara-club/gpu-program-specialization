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

# Scoring state-capacity boundary on RTX 5090

This experiment maps the current thread-owned scoring kernel's state boundary on SM120. Every valid point uses eight runtime constants, eight packed systems per module, three trajectories with 21 points each, four RK4 substeps per interval, and the shallow coupled system `dx_i/dt = k_(i mod 8)*x_i + 0.01*x_((i+1) mod N)`.

## Validated capacity points

| State capacity | Result | Registers | Stack/local | Saturated configurations/s |
|---:|---|---:|---:|---:|
| 8 | Correct | 63 | 0 B | 149.875M |
| 16 | Correct | 95 | 0 B | not measured |
| 20 | Correct | 127 | 0 B | 37.379M |
| 21 | CUBIN inspection failure | — | — | — |
| 24 | CUBIN inspection failure | — | — | — |
| 32 | CUBIN inspection failure | — | — | — |
| 37 | CUBIN inspection failure | — | — | — |
| 38 | Correct | 231 | 0 B | 12.107M |
| 39 | Correct | 235 | 0 B | not measured |
| 40 | Correct | 241 | 0 B | not measured |
| 41 | Correct | 245 | 0 B | not measured |
| 42 | Correct | 251 | 0 B | not measured |
| 43-46 | Correct | 254 | 0 B | 9.426M at 46 |
| 47-56 sampled | CUBIN inspection failure | — | — | — |
| 64, 72, 80 | CUBIN inspection failure | — | — | — |

The exact-capacity failures are non-monotonic PTXAS/physical-inspection outcomes, not mathematical failures. A passing overallocated template bridges them: active widths 21 and 32 both execute correctly in a capacity-38 template, and active widths 24 and 32 both execute correctly in a capacity-46 template.

## Interpretation

- The highest validated active width is 46 states with eight constant inputs.
- Capacity 46 is not a comfortable general-purpose shape. It consumes 254 registers before a deeper specialized AST asks for additional scratch.
- Twenty states is the current conservative boundary: 127 registers, no stack traffic, and materially more specialization headroom.
- Capacities 21-38 can use the passing capacity-38 template, but it always carries 231 registers and executes the overallocated CUDA loops. This is correctness-preserving but materially slower.
- State count is not the only dimension. An initial square sweep that increased constants with states reached 254 registers at 39 states plus 39 constants and failed at 40 plus 40. That sweep is not used as the answer to the state-only question.
- A robust shape above approximately 20 states should split the live state/RHS interface rather than rely on a 231-254-register monolith.

The throughput numbers time C99 specialization, eager module load, launch, complete scoring, retirement, and unload. Handle creation/compilation is outside the timed scope. The one-system SM120 template defect is avoided by using the intended packed eight-system template.

## MSE replay coverage

The initial boundary pass checked only the exact first configuration. A strengthened run then compared 16 independently replayed `(system, constant-bank)` pairs at 20 active states, 32 active states in the capacity-38 template, and 46 active states. Maximum relative MSE disagreement was `3.44e-6`, `2.44e-6`, and `7.45e-6`, respectively. All checks passed the FP32 tolerance. The configuration-layout test is recorded separately because the saturated timing run contains more than two million outputs and was not exhaustively replayed on the CPU.

Raw summary: [`raw/2026-09-02-scoring-state-boundary-rohini.json`](raw/2026-09-02-scoring-state-boundary-rohini.json)
