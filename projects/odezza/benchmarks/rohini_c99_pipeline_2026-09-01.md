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

# Rohini native packed scoring pipeline

Date: 2026-09-01 EDT  
GPU: NVIDIA GeForce RTX 5090 (`sm_120`)  
Timed scope: CUBIN reset, C99 AST validation/assembly, module load, function
lookup, real fused RK4 trajectory scoring, completion, and module unload

## Outcome

Odezza now reaches the useful-throughput range that motivated its native
pipeline. The best tested configuration-throughput shape processes 115.49
million complete trajectory configurations/s. The best unique-structure shape
processes 1.448 million distinct systems/s while evaluating 32 toggle
configurations for every system. Both use one C99 specialization thread and one
CUDA-owner thread.

The previously reported 113,200 configurations/s result was valid for its
two-system validation fixture, but it was not comparable to the packed Secant
System ID results: it attached only 64 configurations to each module. Packing
is the dominant difference.

## Final packing sweep

Every system contains one candidate RHS AST plus one common prespecialized RHS.
The candidate family rotates through eight structurally different postorder
programs and exercises toggles, literals, arithmetic, absolute value, and
division. Every configuration integrates three trajectories, 16 observations,
four RK4 substeps per observation, and four RK stages per substep. The planted
system replayed at MSE `5.06631785e-14` in every row.

| Systems/module | CUBIN | Configs/system | Queue depth | Modules/s | Unique systems/s | Configurations/s | C99 specialization/module |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 157,888 B | 512 | 2 | 1,762 | 225,575 | 115.49M | 65.3 us |
| 256 | 295,104 B | 256 | 2 | 1,730 | 442,789 | 113.35M | 130.2 us |
| 512 | 569,536 B | 128 | 2 | 1,640 | 839,747 | 107.49M | 257.7 us |
| 1,024 | 1,118,400 B | 64 | 1 | 1,404 | 1,437,295 | 91.99M | 528.1 us |
| 1,024 | 1,118,400 B | 32 | 1 | 1,414 | **1,448,366** | 46.35M | 523.8 us |

The 128-system shape maximizes complete configurations/s. The 1,024-system
shape maximizes unique systems/s and reaches the prior 1.15--1.23M resident
System ID unique-system range through a full native module lifecycle. That is a
pipeline comparison, not an equal-science comparison: the older fed-batch
prototype used four states and two candidate component ASTs per system, whereas
this validation uses two states and one candidate AST.

## Host topology

The original defensive public API repeated whole-CUBIN hashing, inspection
checks, AST validation, and patch-arena clearing on every ticket. At 512 systems
per module it spent about 2.78 ms in specialization and required many workers.
The pipeline now uses a trusted fresh-template path: the slot has just been
copied from the exact template validated at pipeline creation, so redundant
whole-image work is skipped while untrusted candidate ASTs remain validated.

After that change, specialization takes 65 us for 128 systems, 258 us for 512,
and 524 us for 1,024. At the largest shape, GPU completion takes about 529 us,
so one specialization thread naturally balances one CUDA-owner thread. A
matched one/two/four-worker sweep measured 1.444M, 1.421M, and 1.447M systems/s;
extra specializers did not improve throughput. Setting NVIDIA's
`CUDA_BINARY_LOADER_THREAD_COUNT` to 1, 2, 4, and 8 measured 1.448M, 1.448M,
1.444M, and 1.436M systems/s. One binary-loader thread is also sufficient.

This does not mean every future shape needs only two host threads. Smaller
configuration banks, cheaper trajectories, larger ASTs, or more expensive
autodiff can make specialization dominant. The correct capacity rule is to
compare measured specialization time per module with completion/lifecycle time
per module and add workers only when the ready queue runs dry.

## CUDA eager versus application eager

Two independent mechanisms must remain distinct:

1. CUDA driver loading policy controls whether code is materialized during
   `cuModuleLoadData` or deferred to `cuModuleGetFunction`.
2. Odezza queue depth controls whether a later module is loaded while an earlier
   kernel is resident or running.

CUDA eager mode is now requested and verified in every final row. On the
1.118 MB depth-one shape, two paired runs measured 1,408--1,417 modules/s in
eager mode and 1,418--1,423 in lazy mode. The total difference is under 1%; lazy
merely moves about 59 us from module load to function lookup. On the standalone
65,960-byte early-exit lifecycle fixture, eager measured 14,559 modules/s and
lazy measured 15,972 modules/s. Therefore explicit policy is essential for
reproducibility, but CUDA eager is not itself the source of the speedup.

Application queue depth has a larger and shape-dependent effect. Depth two is
about 5% faster than depth one for the 158--570 KB templates. At 1.118 MB,
depth one is about 21% faster: depth two expands mean load/unload calls from
approximately 145/27 us to 370/479 us while a neighboring kernel executes.
The production rule is to sweep depth using real modules, not to assume that a
deeper eager queue is always better.

## Lifecycle ceiling

The earlier standalone code-heavy 1,056,680-byte early-exit fixture reached
4,819 unique modules/s at depth one. The 1,118,400-byte Odezza module reaches
1,414 modules/s while specializing 1,024 diverse systems and running real RK4.
It is therefore at 29% of the matched ideal lifecycle rate; it is not purely
module-loader-bound. Its roughly 524 us specialization and 529 us GPU completion
times are intentionally balanced and dominate the approximately 145 us eager
load call.

The standalone fixture remains a ceiling, not a production predictor. BRKPT
padding does not reproduce a branch-heavy RK4 instruction image, and its kernel
returns after one global load.

## Reproducibility and limits

- Raw final rows are in
  `benchmarks/raw/2026-09-01-rohini-c99-packed-pipeline/`.
- Eager/lazy, depth, worker, and loader-thread diagnostics are in
  `benchmarks/raw/2026-09-01-rohini-c99-pipeline-diagnostics/`.
- Compilation, context creation, initial input upload, and final output download
  are excluded from the main timed interval. Final download is separately
  reported in every JSON record.
- Each system is byte-distinct, but the benchmark cycles through eight opcode
  structures rather than sampling a mature GP population. It is stronger than
  a literal-only cache control but remains a controlled execution benchmark.
- The final templates use one exported kernel function. Multi-entry-point
  packing can change function lookup, stream concurrency, and compilation cost.
- The same final C99 path also passed on Ada's RTX 4090 (`sm_89`) with an
  eight-system fixture that executes the division and toggle variants: zero
  spills, explicit eager mode observed, and planted MSE `5.06631785e-14`.
- The 1,024-system CUDA template took about 34.6 seconds to compile with NVRTC
  and PTXAS. Compilation is problem-shape setup, but it is material latency and
  should remain separately reported.
