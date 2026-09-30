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

# AST engine benchmark results

Measured 2026-09-01 with repository revision `cd0bb8ca5914d6439f9fe1e1c2b97afaf1fbe60d` plus the benchmark added alongside this report.

## Workload

- Corpus: 1,000,000 deterministic random exact-depth-2 ASTs, SHA-256 `fedb879288e8790da9b46d0b99ccf2ecb47cbf7ec0291498b9ca6485adb24f42`.
- Grammar: four states, four constant slots, literals `0`, `1`, `-1`, and `2`, five toggle bits, two-way toggles, and `add`, `sub`, `mul`, `div`, `neg`, `sin`, and `cos`.
- Scoring system: four states, three trajectories, eight observations, two RK4 steps per observation, three fixed RHS expressions, and one corpus AST as the fourth RHS.
- Every AST has 32 toggle permutations. A runtime constant-bank count multiplies that into the reported configurations per AST.
- Timed engine path: C99 specialization, eager CUDA module load, function lookup, fused RK4 execution, retirement, and module unload.
- Excluded from engine time: corpus parse, host/device allocation, input upload, result download, and CPU replay.

## Full one-million-AST confirmation

| GPU | Systems/module | Configurations/AST | Median engine time | Unique AST/s | Configurations/s |
|---|---:|---:|---:|---:|---:|
| RTX 5090 (SM 120) | 256 | 256 | 0.9771 s | 1.023M | 262.0M |
| RTX 5090 (SM 120) | 256 | 512 | 0.9911 s | 1.009M | 516.6M |
| RTX 5090 (SM 120) | 512 | 256 | 0.9740 s | 1.027M | 262.8M |
| RTX 4090 (SM 89) | 256 | 64 | 1.3809 s | 724.2k | 46.35M |
| RTX 4090 (SM 89) | 256 | 256 | 1.3865 s | 721.2k | 184.6M |

Each full run strictly replayed 1,024 complete RK4 scores through the independent C99 interpreter. All finite exact-operation results agreed exactly; two samples were mutually invalid on CPU and GPU. The selector skipped 1,405 approximate-math candidates while finding the 1,024 strict samples.

## Configuration-count sweep

These shorter runs use the same first 16,384 ASTs, 256 systems per module, one worker, two CUBIN slots, two loaded modules, and two execution streams.

| Configurations/AST | RTX 5090 AST/s | RTX 5090 configurations/s | RTX 4090 AST/s | RTX 4090 configurations/s |
|---:|---:|---:|---:|---:|
| 32 | 1.026M | 32.82M | 702.3k | 22.47M |
| 64 | 1.001M | 64.06M | 700.8k | 44.85M |
| 128 | 1.026M | 131.3M | 682.0k | 87.29M |
| 256 | 1.022M | 261.7M | 626.3k | 160.3M |
| 512 | 984.5k | 504.1M | 656.9k | 336.3M |
| 1,024 | 620.5k | 635.4M | 432.4k | 442.8M |
| 2,048 | 440.6k | 902.3M | 278.8k | 570.9M |
| 4,096 | 278.7k | 1.141B | 181.1k | 741.7M |
| 8,192 | 148.0k | 1.212B | 101.2k | 829.1M |
| 16,384 | not run | not run | 54.3k | 889.3M |

The full-corpus confirmation is more representative of sustained pipeline throughput than small differences in this short sweep. In particular, the RTX 4090 sustained essentially identical unique-AST throughput at 64 and 256 configurations per AST in the full run.

## Packing and queue observations

- On the RTX 5090, 128 systems/module reached 589.7k AST/s at 256 configurations/AST, while 256 reached 1.022M AST/s.
- A 512-system template reached 1.095M AST/s on the short prefix but only 1.027M AST/s on the full corpus, effectively tied with 256.
- Cold template creation on the RTX 5090 took 9.18 seconds at 256 systems/module and 35.35 seconds at 512. On the RTX 4090 it took 15.71 seconds at 256 and 66.89 seconds at 512. Re-creating the identical template hit the compiler cache and took about 35–75 ms; those warm times must not be presented as cold compile performance.
- At the measured revision, a one-system template failed CUBIN inspection because the CUDA toolchain removed the indirect dispatch shape when only one target existed. The later direct-dispatch implementation resolves this and has dedicated one-system and multi-system GPU regressions; the historical throughput table above is unchanged.
- One loaded module leaves throughput on the table. Two modules in flight are sufficient in this workload; increasing configured depth to four or eight still produced a measured peak of two.
- On the RTX 4090 short sweep, two workers and eight slots improved the median by only about 1.5% over one worker and two slots while using four times the workspace. Four workers did not help.

## CPU verification boundary

The specialized GPU division implementation uses a reciprocal approximation followed by multiplication, and device transcendental instructions do not exactly match host libm. One deliberately encountered expression divided by a state difference near zero; its CPU and GPU trajectory scores differed by 4.0%, and a different constant bank amplified this further. The GPU result was identical across independent 128- and 256-system packing templates, indicating deterministic device behavior rather than a packing error.

The benchmark therefore makes a narrow claim: it strictly verifies full trajectory scores for ASTs containing only operations whose CPU and injected-GPU semantics are expected to match. It reports how many approximate-math candidates were skipped. Approximate-math trajectory agreement needs a separate numerical-policy test rather than a silently relaxed universal tolerance.

## Recommended default

Use 256 systems/module, one specialization worker, two CUBIN slots, two loaded modules, and two execution streams.

For this workload, 256 configurations/AST is the balanced default: it preserves essentially all unique-AST throughput on both tested GPUs and yields 262.0M configurations/s on the RTX 5090 and 184.6M/s on the RTX 4090 over the full corpus. The 32 toggle permutations can therefore be paired with eight constant banks without a measurable sustained structural-throughput penalty.

Use 512 configurations/AST when the extra constant coverage is scientifically useful and output storage is acceptable. Writing every FP32 score costs 1.024 GB per million ASTs at 256 configurations and 2.048 GB at 512. Much larger configuration counts maximize raw configuration throughput but sacrifice structural breadth and grow output storage linearly.
