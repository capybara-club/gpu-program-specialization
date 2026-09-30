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

# Unique-AST throughput with minimum configurations on Rohini

Date: 2026-08-31
GPU: NVIDIA GeForce RTX 5090
Prototype: generated CUDA toggle kernel, dense aligned fed-batch rollout, full MSE

## Result

For a resident module, the best tested strict minimum-work shape was:

- 256 structurally distinct complete systems per module
- 2 distinct component RHS ASTs per system
- 8 systems per CUDA entry point
- 32 entry points launched concurrently on 32 streams
- 1 evaluated configuration per system from a five-bit, 32-permutation compiled AST
- 32 threads per CTA

In a 5,000-sweep matched comparison, 1 through 32 configurations all took 222.6 us per 256-system sweep to within 0.025%. The one-configuration result was:

| Metric | Result |
|---|---:|
| Complete unique systems/s | 1,150,067 |
| Distinct component RHS ASTs/s | 2,300,133 |
| System configurations/s | 1,150,067 |
| Time for all 256 systems and 256 configurations | 222.60 us |
| Registers/thread | 56 |
| Local memory/thread | 0 bytes |
| CUDA source size | 527,792 bytes |
| CUBIN size | 391,744 bytes |

The same topology reached 1.229 million systems/s in a shorter boosted run, so its observed range was approximately 1.15-1.23 million unique systems/s. Packing 4 systems per entry point was essentially tied at the higher clock state, but required 64 entry points and a substantially larger module. Eight is therefore the better practical choice.

The important result is that inactive warp lanes do not make the kernel faster. All 32 toggle permutations can occupy those lanes for effectively no additional wall time. Two constant banks, or 64 configurations per system, also took essentially the same time in the earlier sweep. Therefore:

- If the metric explicitly penalizes the number of configurations evaluated, **1 configuration/system** is the strict answer.
- For a real search, use **32 configurations/system**, because it provides 32 times the structural coverage without reducing unique-system throughput.
- **64 configurations/system** is also nearly free on this measured shape, although it adds a second constant bank rather than filling otherwise dark lanes in the first warp.

## Packing sweep at 32 configurations/system

Every small entry point was given its own concurrent stream so that the comparison did not serialize or underfill the GPU. This packing sweep used all 32 toggle configurations; the matched partial-configuration sweep below shows that 1-32 configurations have the same execution time for the winning topology.

| Systems/entry point | Entry points | Resident unique systems/s | Registers/thread | Source bytes | CUBIN bytes |
|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 440,885 | 47 | 2,131,116 | 2,043,408 |
| 2 | 128 | 870,790 | 48 | 1,214,892 | 1,111,544 |
| 4 | 64 | 1,225,400 | 56 | 756,816 | 641,552 |
| **8** | **32** | **1,228,148** | **56** | **527,792** | **391,744** |
| 16 | 16 | 1,132,634 | 72 | 413,408 | 268,896 |
| 32 | 8 | 1,030,934 | 72 | 356,218 | 206,512 |
| 64 | 4 | 986,082 | 72 | 327,626 | 175,184 |
| 128 | 2 | 1,057,062 | 121 | 313,390 | 158,440 |
| 256 | 1 | 868,646 | 96 | 306,340 | 150,776 |

Eight systems per entry point is the resident sweet spot. Packing further reduces entry-point and module overhead, but register pressure and less flexible scheduling begin to cost execution throughput.

## Partial-warp configuration sweep at 8 systems/entry point

All rows below used the same generated module, 32 threads/CTA, and 5,000 sweeps. Only the runtime configuration limit changed.

| Evaluated configurations/system | Seconds/256-system sweep | Resident unique systems/s | Configurations/s |
|---:|---:|---:|---:|
| **1** | **222.596 us** | **1,150,067** | 1,150,067 |
| 2 | 222.580 us | 1,150,148 | 2,300,297 |
| 4 | 222.623 us | 1,149,927 | 4,599,706 |
| 8 | 222.602 us | 1,150,036 | 9,200,289 |
| 16 | 222.602 us | 1,150,032 | 18,400,514 |
| 32 | 222.568 us | 1,150,211 | 36,806,757 |

The entire timing spread is only 0.055 us, or 0.025%. The tiny ordering differences are noise. One configuration is the least-work point with the same unique-AST rate; 32 is the throughput-efficient search point because the other 31 configurations are wall-clock free.

## Multi-warp configuration-count sweep at 8 systems/entry point

| Configurations/system | Threads/CTA | Resident unique systems/s | Configurations/s | ns/configuration |
|---:|---:|---:|---:|---:|
| **32** | **32** | **1,225,995** | 39,232,000 | 25.49 |
| 64 | 64 | 1,227,613 | 78,567,000 | 12.73 |
| 128 | 128 | 1,060,993 | 135,807,000 | 7.36 |
| 256 | 256 | 863,162 | 220,969,000 | 4.53 |
| 512 | 256 | 605,509 | 310,020,000 | 3.23 |
| 1,024 | 256 | 338,810 | 346,941,000 | 2.88 |
| 2,048 | 256 | 171,067 | 350,346,000 | 2.85 |

Raw configuration throughput saturates near 1,024-2,048 configurations/system, but unique-system throughput falls sharply. One through 32 configurations share the same single-warp execution cost. Sixty-four is a nearly-free coverage upgrade in this workload, while 128 and beyond begin to reduce unique-system throughput.

## Resident versus one-shot modules

The resident result above assumes function handles are cached and the loaded module is reused. For one module sweep, adding measured module load and function lookup time changes the provisional optimum:

| Systems/entry point | Load + lookup + one sweep | Estimated unique systems/s |
|---:|---:|---:|
| 4 | 758.04 us | 337,713 |
| 8 | 550.74 us | 464,827 |
| 16 | 452.42 us | 565,844 |
| 32 | 411.32 us | 622,389 |
| 64 | 393.74 us | 650,182 |
| **128** | **349.69 us** | **732,070** |
| 256 | 365.11 us | 701,167 |

Thus:

- Prefer **8 systems per entry point** when modules remain resident for multiple launches or module work is overlapped by the eager pipeline. Run 1 configuration only for a strict minimum-work measurement; run all 32 toggle permutations for actual search coverage.
- Prefer roughly **128 systems per entry point** provisionally when a module is loaded and executed only once and startup cannot be hidden.

The one-shot table is not yet a complete eager-pipeline benchmark. It excludes input upload, output download, module unload, CPU specialization, and overlap with neighboring modules.

## Interpretation

“Unique” means distinct generated opcode/postorder signatures, not copies of one AST with scaled literals. Each complete system contained two unique component ASTs. The 256-system module therefore contained 512 distinct component AST signatures.

Five binary binding toggles provide 32 available materialized structures per compiled system skeleton. The new runtime limit proves that the kernel may evaluate any prefix from 1 through 32, but doing less than a warp does not lower elapsed time. One constant bank was used throughout the partial-warp comparison.

The current practical default for the next SASS/eager-pipeline implementation should be 8 systems per entry point, 32 concurrent entry points, and 32 configurations per system. Sixty-four is reasonable when an extra constant bank has search value. The full pipeline should also test a heavier 128-systems-per-entry-point module because startup overhead can reverse the resident result.

## Material deviations and limits

- An initial packing sweep capped concurrency at 16 streams. Its 1-, 2-, 4-, and 8-system results serialized entry points and were invalid for comparing peak occupancy. Those values are excluded above; all reported small-packing results use one stream per entry point.
- This benchmark measures resident CUDA execution. It is not yet the complete Secant eager specialize/load/run/unload pipeline.
- ASTs were structurally distinct, fixed-cost synthetic programs with 18 instructions, 8 leaves, and 5 toggle bits per component AST. They do not reproduce the size distribution or operator distribution of a mature GP population.
- The systems share the same known fed-batch integration envelope. Only the candidate RHS structures and constant banks differ.
- Most sampled random systems produced invalid trajectories. The kernel still executed the full RK4 and scoring loops without an early-exit shortcut, but the numerical behavior is not representative of a quality-filtered population.
- ASTs are embedded into generated CUDA in this prototype. Direct CUDA compilation is outside the throughput metric; the intended Secant path specializes precompiled CUBINs.
- The workload uses dense, aligned observations and full-trajectory MSE. Sparse and irregular observation shapes may move the crossover.
- The measured one-shot estimate includes module load and function lookup but excludes transfers, unload, specialization, and pipeline overlap.
- Resident throughput varied by about 7% across separate short and long benchmark invocations. The 1-32 comparison is based on a matched long sweep in one sequence; its relative conclusion is much firmer than any isolated absolute peak.
