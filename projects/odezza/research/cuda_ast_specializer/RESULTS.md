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

# Direct CUDA AST specialization results

Measured on Rohini's RTX 5090 (`sm_120`) on 2026-09-02. The input was the existing one-million-program deterministic exact-depth-2 corpus. Each run scored 33,554,432 configurations so that every packing point had enough work to saturate the GPU. Kernel timing used CUDA events and excluded compilation, module loading, allocation, upload, and download.

The CUDA kernel retained the current four-state benchmark: three trajectories, eight observation intervals, two RK4 steps per interval, three fixed RHS expressions, one candidate RHS, runtime constant banks, toggle permutations, shared-memory trajectory staging, and MSE output. RHS realization was the intentional difference: the three known expressions were compiled directly as CUDA and each candidate AST was emitted as ordinary CUDA in a `switch` selected by `blockIdx.y`. The current engine instead specializes both fixed and candidate postorder programs as SASS, so PTXAS has a larger fusion and scheduling scope in this experiment.

## Fast-math CUDA

This mode most closely resembles the approximate reciprocal and transcendental instructions used by the SASS AST writer, but `--use_fast_math` also changes arithmetic in the surrounding CUDA scaffold. It is therefore performance evidence, not a strict numerical equivalence result.

| ASTs compiled into one kernel | Registers/thread | Thread occupancy | CUBIN | Cache-disabled compile | SASS instructions | Whole-kernel mean stall nibble | Candidate line-info mean stall nibble | Configurations/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 100.0% | 24.7 KB | 0.150 s | 240 | 4.78 | 2.00 | 7.45B |
| 2 | 39 | 100.0% | 25.1 KB | 0.148 s | 256 | 4.88 | 7.50 | 6.99B |
| 4 | 40 | 100.0% | 28.1 KB | 0.151 s | 275 | 4.77 | 6.22 | 6.38B |
| 8 | 47 | 83.3% | 30.1 KB | 0.152 s | 321 | 4.79 | 5.14 | 6.31B |
| 16 | 52 | 75.0% | 37.1 KB | 0.163 s | 407 | 4.89 | 4.78 | 5.85B |
| 32 | 67 | 58.3% | 48.4 KB | 0.174 s | 561 | 4.84 | 4.47 | 5.47B |
| 64 | 63 | 66.7% | 74.7 KB | 0.207 s | 936 | 5.25 | 4.60 | 5.16B |
| 128 | 110 | 33.3% | 122.3 KB | 0.272 s | 1,557 | 5.34 | 4.53 | 3.72B |
| 256 | 239 | 16.7% | 211.0 KB | 0.438 s | 2,683 | 5.51 | 4.69 | 1.98B |

## Default precise CUDA

Default CUDA preserves the scaffold's ordinary division semantics. Precise candidate division introduces reciprocal refinement and helper calls, making it operationally different from the current SASS AST path.

| ASTs compiled into one kernel | Registers/thread | Thread occupancy | CUBIN | Cache-disabled compile | SASS instructions | Whole-kernel mean stall nibble | Candidate line-info mean stall nibble | Configurations/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39 | 100.0% | 27.7 KB | 0.186 s | 401 | 5.47 | 4.94 | 5.87B |
| 2 | 42 | 83.3% | 28.0 KB | 0.153 s | 408 | 5.55 | 7.58 | 6.22B |
| 4 | 50 | 75.0% | 31.3 KB | 0.156 s | 449 | 5.46 | 6.09 | 4.90B |
| 8 | 53 | 75.0% | 35.6 KB | 0.162 s | 534 | 5.63 | 6.28 | 5.13B |
| 16 | 62 | 66.7% | 41.4 KB | 0.170 s | 667 | 5.67 | 6.28 | 4.66B |
| 32 | 79 | 50.0% | 55.8 KB | 0.190 s | 983 | 5.81 | 6.19 | 4.38B |
| 64 | 73 | 50.0% | 88.2 KB | 0.230 s | 1,655 | 6.02 | 6.14 | 3.68B |
| 128 | 118 | 33.3% | 147.8 KB | 0.330 s | 2,910 | 6.08 | 6.07 | 2.76B |
| 256 | 229 | 16.7% | 257.1 KB | 0.604 s | 5,144 | 6.08 | 6.06 | 1.46B |

## Interpretation

- Direct CUDA is excellent for a very small number of ASTs. PTXAS folds redundant toggles, fuses expressions, schedules candidate instructions at roughly 4.5--6.3 encoded stall cycles in the representative packed cases, and reaches 5--7.5B scored configurations/s.
- Packing many CUDA AST branches into one function causes a register-allocation phase change. Fast-math grows from 40 registers at one AST to 239 at 256; active thread occupancy falls from 100% to 16.7%. The precise path reaches 229 registers. The nonmonotonic 32/64 result reflects different selected AST branches and PTXAS allocation, not a guarantee that register use always rises monotonically.
- The current SASS-specialized depth-2 shape stays at 39 registers for 256 ASTs because each branch reuses an inspected register arena. Its previously measured execution-dominated rate was about 1.87B configurations/s. Direct fast-math CUDA at 256 ASTs reached 1.98B/s, only about 5% faster despite a much more aggressive schedule, because its 239-register footprint destroys occupancy.
- The SASS writer deliberately gives ordinary inserted ALU/select/branch instructions a 12-cycle stall and MUFU instructions a 6-cycle stall. PTXAS's direct-CUDA candidate line-info averages are much lower. This is evidence that the hand schedule is conservative, but it is not permission to lower stalls blindly: PTXAS can reorder across the complete RK4 body, while the SASS writer operates inside fixed patch boundaries, and the prior toggle-predicate defect demonstrated the correctness risk.
- Every candidate-attributed instruction in both sweeps carried raw scheduling-control bit 44 equal to zero. Direct CUDA therefore provides no evidence that changing the specializer's current raw yield bit is useful. The semantic polarity of that undocumented bit is intentionally not asserted.
- Repeated compilation is strongly cached. At 256 ASTs, warm calls fell to roughly 0.075--0.079 seconds. Cache-disabled compilation took 0.438 seconds in fast-math mode and 0.604 seconds in precise mode. Warm values must not be described as new-batch compilation costs.
- CUDA line information is not a perfect instruction ownership map. PTXAS can eliminate redundant toggles, fuse operations, and attribute moved instructions to surrounding lines. Candidate schedule counts are consequently a compiler-attribution subset; whole-kernel counts are complete.

The experiment supports keeping SASS specialization for large batches while treating small CUDA-compiled AST groups as a potentially useful separate regime. A next comparison would pack many low-register CUDA functions into one CUBIN and quantify the launch/module-function tradeoff without forcing all AST branches through one function's global register allocation.
