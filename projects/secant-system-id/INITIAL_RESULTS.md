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

# Initial dynamic-setting kernel results

The first standalone kernel was validated on rohini's RTX 5090 (`sm_120`) and
ada's RTX 4090 (`sm_89`). One thread owns one setting, runs all three published
fed-batch trajectories sequentially, and produces one aggregate MSE. Each RK4
stage writes four current states to shared memory and gathers sixteen AST leaf
inputs from a shared bank containing four states followed by eight constants.

The two missing rate laws are separate post-order ASTs specialized into one
marker site. Their leaf bindings may independently select any state or constant
bank entry.

## Correctness and resources

| GPU | Template registers | AST SASS instructions | Register growth | Planted MSE | Planted rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5090 | 67 | 9 + 9 | 0 | 2.144e-12 | 1 / 262,144 |
| RTX 4090 | 71 | 9 + 9 | 0 | 2.144e-12 | 1 / 262,144 |

The RTX 5090 CUBIN reports zero stack and zero local memory. SASS inspection
shows sixteen indexed shared loads at the site, followed by the specialized AST
instructions and one branch over the unused marker island. The planted setting
recovers both published rate laws and is the best member of the generated
population on both GPUs.

## Initial sustained timing

The timing uses 262,144 settings, 128 threads per CTA, 16 RK4 steps per
eight-hour observation interval, a warm-up launch, and 20 timed launches. It
excludes compilation, inspection, specialization, module loading, and the
warm-up from the per-launch boundary.

| GPU | ms/launch | Settings/s | Trajectories/s | ns/three-trajectory setting | ns/trajectory |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5090 | 1.802 | 1.455e8 | 4.364e8 | 6.875 | 2.292 |
| RTX 4090 | 2.361 | 1.110e8 | 3.331e8 | 9.006 | 3.002 |

These results are not yet a search benchmark. They establish that the intended
topology works, that two missing expressions can share one specialization site,
and that per-setting state/constant selection remains fast enough to justify
building the structural search around it.

## Control-plane timing

One initial measurement produced:

| GPU host | CUDA to CUBIN | Python inspection | Python specialization |
| --- | ---: | ---: | ---: |
| rohini | 128 ms | 0.248 ms | 0.224 ms |
| ada | 162 ms | 0.396 ms | 0.344 ms |

These single observations are implementation checks, not stable benchmark
statistics. The source compiler is the installed NVRTC library called through
Python. No Python CUDA package or new system dependency is required.

## Compact multi-genome dispatch

The packed variant uses one native branch table and one compact SASS arena. An
entry is a complete system genome containing both missing rate-law ASTs. With
eight entries, every CTA loops through eight genomes, and every thread runs all
three trajectories for each genome. PTXAS generated `BRXU` on the RTX 5090 and
`BRX` on the RTX 4090. Both templates and all specialized genomes used 80
registers with no specialization growth.

Eight distinct genomes were checked against the CPU trajectory scorer. The
planted genome produced an MSE of `2.119e-12`; the largest absolute CPU/GPU MSE
difference across all eight was `1.751e-6`. A separate three-missing-site CUBIN
also compiled and packed correctly with per-genome AST instruction counts of
`3 + 4 + 5` and no register growth.

The exact small launch requested for topology validation used 2,048 settings,
128 threads, and a `16 x 1` grid. These 16 CTAs are far below full-GPU
occupancy:

| GPU | ms/launch | `(genome, setting)` configurations/s | Trajectories/s |
| --- | ---: | ---: | ---: |
| RTX 5090 | 5.675 | 2.887e6 | 8.664e6 |
| RTX 4090 | 6.397 | 2.561e6 | 7.684e6 |

With 262,144 settings, the same eight-genomes-per-CTA body has enough settings
tiles to fill the GPU. One launch evaluates 2,097,152 genome/setting pairs:

| GPU | ms/launch | `(genome, setting)` configurations/s | Trajectories/s |
| --- | ---: | ---: | ---: |
| RTX 5090 | 18.212 | 1.152e8 | 3.455e8 |
| RTX 4090 | 22.161 | 9.463e7 | 2.839e8 |

On the RTX 5090, packed throughput is about 79% of the original one-genome
direct-site rate. The packed correctness genomes also contain one extra scale
multiply per missing AST. This is a useful initial result: compact dynamic
dispatch and serial genome reuse cost throughput, but do not approach the much
larger penalty of launching or loading a separate kernel per genome.

## Next measurement

Before LM, sweep CTA size, integration accuracy, state/constant bank size, AST
depth, and settings per launch. Then compare this shared-bank gather against a
four-state register-select variant. LM should first be added as a promotion
kernel with fixed winning bindings and a modest constant count; dynamic binding
inside every LM iteration should only be retained if the measured search policy
benefits from changing bindings during optimization.
