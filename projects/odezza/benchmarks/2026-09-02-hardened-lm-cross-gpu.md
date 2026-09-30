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

# Hardened trajectory-LM cross-GPU validation

> **Superseded for trajectory-LM gradients and throughput:** This report predates the coupled RK4 sensitivity-stage repair. Primal trajectory and materialized-winner CPU replay results remain valid. Use [`2026-09-02-corrected-coupled-lm.md`](2026-09-02-corrected-coupled-lm.md) for current LM correctness and performance.

The current production path is grouped specialization ABI v4. Each site has an inspected entry branch that skips its disposable marker scaffold, and one multi-output specialized program computes several related values with cross-output common-subexpression elimination. The inspector continues to expose the original CUDA input and output registers. If PTXAS assigns a CUDA result register to an input, the specializer computes into a verified spare register and relays the result only after all input uses are complete.

The earlier scalar ABI v2 correctness results remain valid, but its performance is superseded. ABI v3 productionized the entry branch and reproduced the scratch bypass. ABI v4 then reduced the fed-batch statistics stage from 44 scalar sites to six grouped sites.

This validation compiled, inspected, specialized, executed, and independently CPU-replayed both the two-state/eight-constant (`2x8`) and four-state/six-constant (`4x6`) shapes on Rohini and Ada. The final templates include runtime per-observation residual weights and an exact per-fit factorization-attempt counter.

## Compilation and physical specialization

| GPU | Shape | CUBIN | Template registers | Specialized registers | Stack | Spill loads/stores | Sites | Relayed sites |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX 5090 (`sm_120`) | 2x8 | 108,432 B | 189 | 189 | 0 B | 0 B / 0 B | 22 | 0 |
| RTX 5090 (`sm_120`) | 4x6 | 189,736 B | 218 | 218 | 0 B | 0 B / 0 B | 44 | 0 |
| RTX 4090 (`sm_89`) | 2x8 | 98,016 B | 194 | 194 | 0 B | 0 B / 0 B | 22 | 0 |
| RTX 4090 (`sm_89`) | 4x6 | 175,840 B | 216 | 216 | 0 B | 0 B / 0 B | 44 | 1 |

The one relayed Ada site is the exact physical collision that originally exposed the defect. Its specialized AST now consumes the input first and copies its result into the CUDA-owned output register afterward.

The current grouped 4x6 production templates are:

| GPU | ABI and reserve | Template registers | Stack | Spills | Sites | Specialized instructions including branches |
|---|---|---:|---:|---:|---:|---:|
| RTX 5090 (`sm_120`) | grouped v4, 192 | 219 | 0 B | 0 B | 6 | 228 |
| RTX 5090 (`sm_120`) | grouped v4, 256 | 219 | 0 B | 0 B | 6 | 228 |
| RTX 4090 (`sm_89`) | grouped v4, 192 | 224 | 0 B | 0 B | 6 | 228 |
| RTX 4090 (`sm_89`) | grouped v4, 256 | 223 | 0 B | 0 B | 6 | 228 |

## Two-state replay and factorization counts

This diagnostic uses 32,768 starts, two toggle permutations, three 21-point trajectories, four RK4 substeps per interval, and 20 LM iterations.

| GPU | Fits/s | Mean LM iterations | Mean accepted steps | Mean factorizations | Best GPU MSE | CPU replay MSE |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 382,576 | 5.9624 | 5.9229 | 6.3798 | `2.49e-15` | `3.68e-15` |
| RTX 4090 | 359,631 | 5.9624 | 5.9229 | 6.3798 | `2.49e-15` | `3.68e-15` |

Both devices select the same fit, recover the same constants, and produce identical iteration, acceptance, and factorization aggregates. The average fit performs 1.070 factorization attempts per completed LM iteration.

## Matched fed-batch workload

The matched workload uses the Secant System ID fed-batch contract: 16 initial-condition trajectories, 13 points from 0 through 96 hours, 16 RK4 steps per eight-hour interval, target-relative residual weights, six active constants, 65,536 deterministic starts, 20 LM iterations, and eight damping attempts. Odezza specializes all four right-hand sides and all 40 local partials. ABI v4 groups the four primals into one site and the 40 partials into five eight-output sites.

| GPU | Implementation | Fits/s | Mean iterations | Mean accepted | Mean factorizations |
|---|---|---:|---:|---:|---:|
| RTX 5090 | Odezza generic scalar sites, original path | 5,540 | 9.0365 | 8.9481 | 12.1219 |
| RTX 5090 | Odezza scalar sites, production entry bypass | 30,789 | 9.0365 | 8.9481 | 12.1219 |
| RTX 5090 | Odezza grouped sites and CSE, 192-instruction reserve | 88,565 | 9.0365 | 8.9481 | 12.1219 |
| RTX 5090 | Odezza grouped sites and CSE, production 256-instruction reserve | 88,842 | 9.0365 | 8.9481 | 12.1219 |
| RTX 5090 | Secant System ID split specialized | 63,671 | 9.325 | 9.250 | not instrumented |
| RTX 5090 | Secant System ID native CUDA | 119,249 | 9.313 | 9.238 | not instrumented |
| RTX 4090 | Odezza generic scalar sites, original path | 5,166 | 9.0365 | 8.9481 | 12.1219 |
| RTX 4090 | Odezza scalar sites, production entry bypass | 31,435 | 9.0365 | 8.9481 | 12.1219 |
| RTX 4090 | Odezza grouped sites and CSE, 192-instruction reserve | 78,891 | 9.0365 | 8.9481 | 12.1219 |
| RTX 4090 | Odezza grouped sites and CSE, production 256-instruction reserve | 78,710 | 9.0365 | 8.9481 | 12.1219 |
| RTX 4090 | Secant System ID split specialized | 81,307 | 9.325 | 9.250 | not instrumented |
| RTX 4090 | Secant System ID native CUDA | 121,632 | 9.313 | 9.238 | not instrumented |

Odezza recovers the six true constants on both GPUs with GPU MSE `2.75e-14`, independent CPU replay MSE `8.29e-15`, and maximum scaled parameter error `1.92e-6`. The average fit performs 1.341 factorization attempts per completed LM iteration.

## Production optimization result

The original 4x6 path executed 1,329 dead scaffold instructions across its 44 statistics-stage scalar sites before executing 621 useful specialized instructions. Each site averaged 30.2 dead instructions. The same pattern existed in the 2x8 template. A controlled scratch patch inserted an entry branch from each site's scaffold to its specialized patch region.

| Shape | GPU | Original | Scaffold bypass | Gain |
|---|---|---:|---:|---:|
| 2x8 | RTX 5090 | 382,576 fits/s | 3,288,368 fits/s | 8.60x |
| 2x8 | RTX 4090 | 359,631 fits/s | 2,962,242 fits/s | 8.24x |
| 4x6 fed-batch | RTX 5090 | 5,540 fits/s | 30,828 fits/s | 5.56x |
| 4x6 fed-batch | RTX 4090 | 5,166 fits/s | 31,463 fits/s | 6.09x |

Both devices retained the identical winning fit, recovered parameters, iteration counts, factorization counts, GPU MSE, and CPU replay result. ABI v4 emits 228 specialized instructions across six sites for this system, versus 621 useful instructions across 44 scalar sites before return branches and relays. This removes 38 site dispatch/return sequences and reduces useful specialized instruction count by 63.3%.

The robust grouped Odezza path is 1.40x faster than the previous Secant System ID split-specialized kernel on the RTX 5090. It reaches 96.8% of that prior kernel's fit rate on the RTX 4090. Odezza performs slightly fewer accepted LM iterations on this deterministic start bank, so the fit-rate table is the literal end-to-end comparison and should not be interpreted as identical floating-point instruction traces.

The grouped templates use 219 registers on SM120 and 224 on SM89, with zero stack and zero spills. This is substantially below the 255-register historical split template, but it is not lower than Odezza's scalar template. The speedup is therefore attributable to control flow and shared expression evaluation, not to a new occupancy class. A 32-thread CTA was fastest in a 32/64/128/256-thread sweep on both GPUs. Precomputing irregular interval step sizes once per CTA improved Ada by about 2.5% and was neutral within run noise on Rohini.

The remaining gap to native CUDA is 1.35x on Rohini and 1.54x on Ada. Native CUDA knows the equation at compile time; the generalized kernel still specializes all four right-hand sides and their 40 local partials, retains runtime ragged-trajectory handling, and uses generic sensitivity propagation. Structurally zero partial omission and smaller grouped-site capacities remain candidates for measurement, not assumed wins.

## Scalar ABI v2 random AST stress

Each GPU specialized 512 random systems per shape, for 1,024 systems per GPU and 2,048 total. Every generated system included repeated optimized-constant indices, two- and four-way toggles, protected division, and a mixture of `sin`, `cos`, `tanh`, `exp`, `log`, and `sqrt` expressions.

| GPU | Shape | Succeeded | Failed | Largest RHS | Largest specialized site | Maximum registers |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 2x8 | 512 | 0 | 105 instructions | 169 / 192 | 189 |
| RTX 5090 | 4x6 | 512 | 0 | 93 instructions | 158 / 192 | 218 |
| RTX 4090 | 2x8 | 512 | 0 | 105 instructions | 169 / 192 | 198 |
| RTX 4090 | 4x6 | 512 | 0 | 93 instructions | 158 / 192 | 216 |

The Python derivative suite also finite-difference checks 128 random smooth AST pairs, plus explicit tests for repeated constants, toggles, protected division, transcendental operators, and the output-relay regression.

## Grouped ABI v4 stress and reserve selection

The fastest 4x6 grouping uses eight outputs per partial site. With the original 192-instruction reserve, 84 of 512 deliberately large random systems were rejected on both GPUs because at least one grouped program crossed its continuation. This did not affect the fed-batch validation, but it made that template unsuitable as the production default.

On Ada, reducing the group capacity lowered rejection but also reduced throughput:

| Outputs per partial site | Sites | Fits/s | Random systems accepted | Rejection rate |
|---:|---:|---:|---:|---:|
| 4 | 11 | 70,689 | 504 / 512 | 1.6% |
| 6 | 8 | 74,043 | 496 / 512 | 3.1% |
| 8 | 6 | 78,891 | 428 / 512 | 16.4% |

Directly measuring every generated group showed a maximum requirement of 247 instructions. A 256-instruction reserve with eight-output groups accepted 512/512 systems and retained 78,710 fits/s, within 0.2% of the smaller reserve. The specialized stress maximum was 232 registers with no stack or spills. The production default is therefore 256 instructions: the entry branch makes the extra reserve unexecuted, so the tradeoff is a larger template and slightly more compile/module metadata rather than LM execution time.

The 256-instruction production template accepted 512/512 stress systems on both GPUs. Maximum specialized register counts were 228 on Rohini and 232 on Ada. A stress-discovered missing import in the register-count expansion path was fixed and covered by a regression test before these final results were recorded.

Raw data: [`raw/2026-09-02-hardened-lm-cross-gpu.json`](raw/2026-09-02-hardened-lm-cross-gpu.json)
