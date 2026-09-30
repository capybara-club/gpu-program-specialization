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

# Corrected coupled trajectory-LM validation

This report replaces trajectory-LM correctness and throughput figures collected before the coupled RK4 sensitivity-stage repair. The defect, impact, and validation evidence are documented in [`../docs/incidents/2026-09-02-coupled-rk4-sensitivity-order.md`](../docs/incidents/2026-09-02-coupled-rk4-sensitivity-order.md).

## Result

Grouped-v4 and two-site-v5 now propagate every coupled sensitivity stage simultaneously. A direct GPU LM proposal agrees with an independent CPU augmented-RK4 calculation to `6.78e-8` in the optimized parameter. The full 4x6 fed-batch workload recovers all constants and independently replays on CPU.

| Workload | GPU | Grouped-v4 | Two-site-v5 | Two-site relative |
|---|---|---:|---:|---:|
| Fed-batch, 4 states / 6 constants | RTX 5090 | 85,821 fits/s | 88,767 fits/s | +3.43% |
| Fed-batch, 4 states / 6 constants | RTX 4090 | 79,563 fits/s | 64,543 fits/s | -18.88% |
| Synthetic, 2 states / 8 constants | RTX 5090 | 7.078M fits/s | 6.845M fits/s | -3.29% |
| Synthetic, 2 states / 8 constants | RTX 4090 | 5.473M fits/s | 5.529M fits/s | +1.04% |

The fed-batch workload uses 65,536 starts, 16 trajectories, 13 points per trajectory, 16 RK4 substeps per interval, at most 20 LM iterations, at most eight damping attempts, 32-thread CTAs, and seven timed repeats. The 2x8 workload evaluates 131,072 fits from 65,536 starts and two toggle permutations, with three 21-point trajectories and four RK4 substeps per interval.

## Numerical agreement

All four fed-batch runs produced the same best fit and work counters:

| Quantity | Result |
|---|---:|
| Best initial MSE | `8.400698006e-2` |
| Best GPU MSE | `2.622390627e-14` |
| CPU replay MSE | `9.650164829e-15` |
| Maximum scaled parameter error | `8.8437e-7` |
| Mean LM iterations | `9.3188934` |
| Mean accepted steps | `9.2438965` |
| Mean factorization attempts | `12.5062256` |

The recovered constants are `[0.42999962, 63.69994736, 5.79999733, 0.13199998, 3.68000007, -2.94e-9]`.

The corrected 2x8 runs also agree across layouts and devices: best GPU MSE `2.250052181e-15`, CPU replay MSE `3.010767193e-15`, mean iterations `5.9711456`, mean accepted steps `5.9467697`, and mean factorization attempts `6.3585052`.

## Physical resources

| Shape | GPU | Layout | Template regs | Specialized regs | Stack | Static shared |
|---|---|---|---:|---:|---:|---:|
| 4x6 | RTX 5090 | grouped-v4 | 224 | 224 | 0 B | 1,024 B |
| 4x6 | RTX 5090 | two-site-v5 | 201 | 216 | 0 B | 6,144 B |
| 4x6 | RTX 4090 | grouped-v4 | 225 | 225 | 0 B | 0 B |
| 4x6 | RTX 4090 | two-site-v5 | 200 | 215 | 0 B | 5,120 B |
| 2x8 | RTX 5090 | grouped-v4 | 188 | 188 | 0 B | 1,024 B |
| 2x8 | RTX 5090 | two-site-v5 | 174 | 174 | 0 B | 3,584 B |
| 2x8 | RTX 4090 | grouped-v4 | 196 | 196 | 0 B | 0 B |
| 2x8 | RTX 4090 | two-site-v5 | 174 | 174 | 0 B | 2,560 B |

The corrected 8x8 two-site template remains outside the clean thread-owned regime: it compiles at 255 registers and uses 512/536 stack bytes per thread on RTX 5090/4090. It has not been specialized or executed as a correctness result.

Raw measurements: [`raw/2026-09-02-corrected-coupled-lm.json`](raw/2026-09-02-corrected-coupled-lm.json)
