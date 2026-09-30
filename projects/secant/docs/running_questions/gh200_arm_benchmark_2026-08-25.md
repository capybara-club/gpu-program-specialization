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

# Lambda GH200 Arm benchmark

## Result

The Grace Hopper host is excellent for wide CPU specialization and compilation,
but it does not make concurrent CUDA module loading scale. Secant should keep
one CUDA context and one module-loader thread. Grace's 64 CPU cores are useful
for the independently parallel C99 SASS writers; they are not a reason to add
loader contexts.

The test host was a Lambda GH200 with 64 single-threaded Arm Neoverse-V2 cores,
an NVIDIA GH200 with 97,871 MiB reported GPU memory, driver 580.105.08, and CUDA
13.0. The control was Rohini's Ryzen 9 9900X and RTX 5090. No software was
installed for these tests.

## C99 SASS specialization

This is the raw, CPU-only specialization path added to `secant-system-id`. Each
module contained eight kernels, 64 genomes, and 128 ASTs. Every worker owned its
CUBIN-sized output buffer. The test used an `sm_120` template because the C99
writer supports that instruction format; the GH200 did not load or execute that
image. This isolates CPU patch-and-copy throughput and makes the result directly
comparable with Rohini.

| Grace workers | ASTs/s | CUBIN copy rate |
| ---: | ---: | ---: |
| 1 | 2.42 M | 5.49 GiB/s |
| 2 | 4.82 M | 10.92 GiB/s |
| 4 | 9.69 M | 21.97 GiB/s |
| 8 | 19.29 M | 43.73 GiB/s |
| 16 | 38.61 M | 87.52 GiB/s |
| 24 | 57.76 M | 130.94 GiB/s |
| 32 | 76.92 M | 174.37 GiB/s |
| 40 | 96.24 M | 218.18 GiB/s |
| 48 | 114.57 M | 259.74 GiB/s |
| 56 | 134.42 M | 304.73 GiB/s |
| 64 | **153.62 M** | **348.25 GiB/s** |

The 64-core result is 99.1% of ideal scaling from one core. A Grace core is
about 2.1 times slower than one physical Ryzen 9900X core on this loop, but the
64-core aggregate is about 2.18 times Rohini's previously measured 24-thread
maximum of 70.5 M AST/s.

Compiling the C99 runtime with `-mcpu=native` reduced throughput from 153.62 M
to 139.03 M AST/s at 64 workers, a 9.5% regression. The generic `-O3` build is
therefore the current winner.

### Ticketed production queue and module packing

The raw loop excludes task submission, ticket bookkeeping, queue wakeups, and
completion publication. The existing production-shaped C99 queue was therefore
tested separately. Small modules do not give each ticket enough work to feed all
64 cores efficiently, but packing more ASTs into each module moves the crossover
in a predictable direction.

| ASTs/module | Kernels/module | CUBIN size | Best tested workers | Best ASTs/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 8 | 311,576 B | 16 | 37.72 M |
| 256 | 16 | 620,152 B | 24 | 56.05 M |
| 512 | 32 | 1,237,560 B | 64 | 70.83 M |
| 1,024 | 64 | 2,472,248 B | 64 | **132.35 M** |

The 512-AST case reaches about 70.5 M AST/s by 32 workers and then plateaus.
The 1,024-AST case continues scaling through 64 workers: 101.76 M at 48 workers
and 132.35 M at 64. It retains about 86% of the 64-core raw-writer ceiling.

This does not prove that 1,024 ASTs/module is the end-to-end optimum. Larger
modules cost more to compile and load, and the GPU must run long enough for eager
loading to hide that work. It does establish that the writer and existing queue
can use the full Grace CPU when the GP batches roughly 1,024 ASTs per ticket.

The writer now supports `sm_90`, and all 24 `secant-system-id` tests pass on
AArch64. Python and C99 specialization are byte-identical. The planted fed-batch
model, 128 distinct genome variants, and a separate kernel exercising every
supported FP32 postorder operation were executed successfully on the GH200.
The all-operation kernel agreed with the CPU reference to 1.49e-7 relative
error.

## Specialized RK4 execution

The execution sweep used the astaxanthin fed-batch model: two specialized ASTs
per system configuration, three trajectories, 12 observations per trajectory,
16 RK4 steps per observation, and four RK4 stages. One complete configuration
therefore invokes specialized AST bodies 4,608 times.

With eight genomes per CTA, performance depends on the total resident CTA work,
not on whether that work comes from more settings or more AST groups. The
following shapes all create 512 CTAs and converge on the same throughput:

| Genomes in one kernel | Candidate ASTs | Settings/genome | Configurations/s |
| ---: | ---: | ---: | ---: |
| 8 | 16 | 65,536 | 57.84 M |
| 16 | 32 | 32,768 | 57.80 M |
| 32 | 64 | 16,384 | 57.78 M |
| 64 | 128 | 8,192 | 57.80 M |
| 128 | 256 | 4,096 | **57.81 M** |

This validates the compact branch-dispatch topology: `blockIdx.y` can supply
independent AST groups, so the search does not need tens of thousands of leaf
settings merely to occupy the GPU. About 512 CTAs saturate this kernel. Moving
to 1,024 CTAs does not improve throughput and slightly hurts the 128-genome
case.

At the winning 128-genome/4,096-setting point, the already-loaded kernel
achieves:

- 57.81 M complete system configurations/s;
- 115.61 M candidate AST-setting combinations/s;
- 173.42 M full trajectories/s; and
- 266.38 billion dynamic specialized-AST stage calls/s.

The one-kernel CUBIN is 405,224 bytes and uses 80 registers. One launch scores
524,288 complete configurations in 9.07 ms.

### Complete eager pipeline

The same shape was then run for 64 modules through the C99 specialize, load,
execute, event-track, and unload pipeline:

| Metric | Complete pipeline result |
| --- | ---: |
| Wall time | 0.611 s |
| Modules/s | 104.7 |
| Configurations/s | **54.90 M** |
| Candidate AST-settings/s | **109.79 M** |
| Trajectories/s | **164.69 M** |
| Dynamic AST stage calls/s | **252.96 B** |

The complete pipeline retains 95.0% of the already-loaded execution rate.
Mean C99 specialization was 0.107 ms/module and mean module loading was 4.625
ms/module, but eager overlap hides almost all of it behind the roughly 9 ms of
GPU work. One Grace specialization worker can produce about 9,492 of these
modules/s while the GPU consumes about 105/s, so this ODE shape does not need
wide CPU specialization. Extra Grace cores matter for much thinner kernels or
parallel campaign preparation, not for feeding this saturated RK4 workload.

### Exact RTX 5090 control

Rohini then ran the identical generator/runtime revision and topology compiled
for `sm_120`: one kernel, 128 genomes, 256 candidate ASTs, 4,096 settings, eight
genomes per CTA, 512 CTAs, and 64 eager-pipeline modules.

| Metric | GH200 | RTX 5090 |
| --- | ---: | ---: |
| CUBIN size | 405,224 B | 416,432 B |
| Registers | 80 | 80 |
| Resident launch | 9.070 ms | **6.126 ms** |
| Resident configurations/s | 57.81 M | **85.59 M** |
| Eager-pipeline wall time | 0.611 s | **0.397 s** |
| Eager configurations/s | 54.90 M | **84.59 M** |
| Candidate AST-settings/s | 109.79 M | **169.18 M** |
| Trajectories/s | 164.69 M | **253.78 M** |
| Dynamic AST stage calls/s | 252.96 B | **389.80 B** |
| Resident throughput retained | 95.0% | **98.8%** |

The RTX 5090 is 48.1% faster for the already-loaded kernel and 54.1% faster end
to end. This workload is compute-heavy and its live data fits comfortably, so
the GH200's memory-capacity advantages do not offset the 5090's execution
advantage here.

One Rohini specialization worker produces about 20,210 modules/s while the GPU
consumes about 161 modules/s. The module therefore needs only one CPU writer.
Larger modules and more workers remain useful for thin kernels, but are not
required to hide host work for this fused RK4 regime.

### Minimal RTX 5090 saturation shape

A follow-up Rohini sweep showed that the 512-CTA control above did not saturate
the larger RTX 5090. Reducing the sequential genome loop from eight genomes to
one genome per CTA both exposes more independent CTAs and reduces the compiled
register count from 80 to 73. With one genome per CTA, throughput approaches a
153.26 M configuration/s asymptote as the grid grows.

| Threads | Genomes/CTA | Settings | Total CTAs | Resident configurations/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 8 | 4,096 | 512 | 85.73 M |
| 128 | 4 | 4,096 | 1,024 | 87.31 M |
| 128 | 2 | 4,096 | 2,048 | 113.66 M |
| 128 | 1 | 4,096 | 4,096 | 129.46 M |
| 128 | 1 | 8,192 | 8,192 | 136--141 M |
| **256** | **1** | **16,384** | **8,192** | **148.63 M** |
| 256 | 1 | 32,768 | 16,384 | 152.20 M |
| 128 | 1 | 65,536 | 65,536 | 153.26 M |

The selected practical minimum is therefore one 416,224-byte kernel, 128
genomes (256 candidate ASTs), one genome per CTA, 16,384 settings, 256 threads,
and 8,192 CTAs per launch. It reaches 97.0% of the largest resident rate without
quadrupling the setting population merely to gain the remaining 3%.

The eager-depth sweep used 512 submitted modules. Two in-flight modules are
enough to hide loading; a third does not improve throughput:

| Maximum in-flight modules | End-to-end configurations/s |
| ---: | ---: |
| 1 | 140.51 M |
| **2** | **146.25 M** |
| 3 | 146.00 M |

One specialization worker remains sufficient: specialization takes about
0.100 ms/module while an overlapped module load takes about 7.1 ms and the
kernel consumes about 14 ms. A module-count sweep found that eight total
modules already reached about 99% of the 512-module rate, while 128--512 modules
converged within 0.2%. Use at least 128 modules for benchmark measurements, but
a continuous production queue only needs two in flight and a small amount of
ready/pending slack.

The final exact validation used 512 modules, one worker, two modules in flight,
and the selected 256-thread topology. It evaluated 1,073,741,824 complete
system configurations in 7.303 s:

- 147.03 M complete system configurations/s, or 6.80 effective ns/configuration;
- 294.05 M candidate AST-settings/s;
- 441.08 M complete trajectories/s; and
- 677.50 billion specialized AST stage calls/s.

This is 98.9% of the matching 148.63 M resident rate and 95.9% of the largest
153.26 M rate observed anywhere in the extended resident sweep.

### Unique-module settings and compact-winner validation

The earlier eager sweep intentionally reused one specialized batch. A stricter
Rohini run generated 512 deterministic, byte-distinct populations and verified
512 distinct specialized CUBIN SHA-256 hashes before starting each timed
pipeline. The timed region includes C99 specialization, module loading, kernel
execution, and result transfer, but excludes Python population construction and
the one-time hash audit.

With full per-setting MSE output and two modules in flight:

| Settings/genome | Configurations/s | Unique genomes/s |
| ---: | ---: | ---: |
| **256** | 39.12 M | **152,799** |
| **512** | **75.23 M** | **146,938** |
| 1,024 | 78.78 M | 76,933 |
| 2,048 | 106.17 M | 51,839 |
| 4,096 | 125.73 M | 30,696 |
| 8,192 | 139.52 M | 17,031 |
| 16,384 | 147.20 M | 8,985 |

This rules out identical-module caching as the explanation for the previous
result. It also separates the two useful objectives: 256 settings maximizes
new structures per second, while 512 is the practical search knee. Moving from
256 to 512 roughly doubles configuration throughput while sacrificing only
3.8% of genome throughput; moving from 512 to 1,024 gains only 4.7% more
configuration throughput and nearly halves genome throughput. The initial
score-only GP default is therefore **512 settings per genome**, subject to a
later time-to-quality experiment.

The production-shaped output path performs two GPU reductions: one winner per
CTA setting tile, followed by one `(score, setting_index)` winner per genome.
At 512 settings, 128 genomes/module, 256 threads, one genome/CTA, one C99
specialization worker, and two modules in flight, four independent 512-module
runs produced a median of:

| Metric | Compact winner path |
| --- | ---: |
| Complete configurations/s | **73.07 M** |
| Unique genomes/s | **142,713** |
| Unique candidate ASTs/s | **285,425** |
| Effective time/genome | **7.01 us** |

The compact path is only about 3% slower than full-MSE output at this setting.
It reduces final host-visible output from 262,144 bytes to 1,024 bytes per
module (256x), with 2,048 bytes of transient GPU tile-winner scratch. Hardware
validation selected exactly the same setting as the full GPU MSE surface for
all 128 genomes, including a separate 513-setting partial-tile test; GPU winner
scores were bit-identical and the CPU/GPU score comparison remained within
normal FP32 integration error. The specialized score kernel uses 81 registers
with no local memory or stack, and the fixed final reducer uses 16 registers
with no local memory or stack.

One worker remains sufficient at this shorter workload. It specializes near
10,000 modules/s while the selected path consumes about 1,115 modules/s. Two
resident/in-flight modules are the minimum eager depth; depths two through four
were within 0.3% in repeated 512-setting runs. Python generated these 512 test
populations at only about 85 modules/s, so population production must move into
the planned C99 GP loop before an end-to-end search can sustain this rate.

## Module-load concurrency

The module-loader benchmark used a 64-kernel Secant SSE source compiled for
`sm_90`. The resulting GH200 CUBIN was 5,017,688 bytes. Each entry is the median
of five runs with 101 measured waves after five warmups. Function enumeration
was included.

| Context mode | Loaders | Load + enumerate modules/s | Mean driver load | Full lifecycle modules/s |
| --- | ---: | ---: | ---: | ---: |
| shared | 1 | **703.9** | **1.397 ms** | 333.5 |
| shared | 2 | 640.9 | 2.239 ms | 364.3 |
| shared | 4 | 607.2 | 3.982 ms | 366.3 |
| shared | 8 | 603.4 | 7.350 ms | 376.6 |
| shared | 16 | 581.8 | 14.523 ms | 370.7 |
| separate | 1 | **712.9** | **1.379 ms** | 341.6 |
| separate | 2 | 617.6 | 2.317 ms | 364.9 |
| separate | 4 | 585.4 | 4.128 ms | 351.4 |
| separate | 8 | 569.8 | 7.791 ms | 347.6 |
| separate | 16 | 552.9 | 15.346 ms | 339.2 |

Both shared and separate contexts serialize the important part of module
loading. Adding loader threads makes aggregate load throughput worse and grows
per-module latency almost linearly. The small lifecycle-throughput increase in
some multi-loader cases comes from overlap elsewhere, especially unloading; it
does not justify the added contexts or loader workers.

## Controlled module-size sweep

Exact 8-, 16-, 32-, and 64-kernel prefixes were taken from the same generated
source. Both hosts compiled them through NVRTC with C++17 and ptxas optimization
level 1, then ran the same one-loader benchmark.

| Kernels | GH200 `sm_90` bytes | GH200 compile | GH200 load | RTX 5090 `sm_120` bytes | Rohini compile | Rohini load |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 628,504 | 2.72 s | 0.280 ms | 1,019,416 | 4.50 s | **0.207 ms** |
| 16 | 1,255,640 | 5.36 s | **0.375 ms** | 2,036,120 | 8.71 s | 0.429 ms |
| 32 | 2,509,656 | 10.85 s | **0.575 ms** | 4,069,400 | 17.53 s | 0.848 ms |
| 64 | 5,017,688 | 21.70 s | **1.410 ms** | 8,136,088 | 35.69 s | 1.851 ms |

For the same 64-kernel source, the GH200 platform compiles about 39% faster and
loads about 24% faster. That is a useful platform result, but its CUBIN is also
38% smaller. A linear fit against actual image size gives approximately 0.261
ms/MB on Grace and 0.231 ms/MB on Rohini, and Rohini wins the smallest-module
case despite its larger image. The data therefore does **not** support a claim
that ARM itself parses or loads equivalent ELF bytes faster. It supports the
narrower claim that this GH200/CUDA/`sm_90` platform produced smaller binaries
and lower practical latency for this source.

## Recommendation

- Use all 64 Grace cores for independent specialization or compilation work.
- Pack roughly 1,024 ASTs per queued module when the goal is maximum Grace CPU
  specialization throughput; retune this against end-to-end GPU execution and
  module-loading latency rather than treating it as a universal constant.
- Keep one CUDA context and one loader thread, with eager loading overlapped
  against useful GPU execution as in Secant.
- Do not select ARM solely to improve `cuModuleLoadData`; the measured advantage
  is explained by platform and binary-size differences, not demonstrated ISA
  efficiency.
- For the fused RK4 kernel, favor one packed entry point with 64 to 128 genomes.
  On GH200, eight genomes per CTA and roughly 512 total CTAs were sufficient;
  on the RTX 5090, one genome per CTA and at least several thousand CTAs gave
  materially better occupancy.
- Treat `sm_90` as a supported Secant System ID target; keep the hardware
  correctness check in release qualification because SASS is an internal ISA.

The raw result files remain on the Lambda host under
`/home/ubuntu/scratch/secant-arm-module-load` and
`/home/ubuntu/scratch/secant-arm-specializer`.

## C99 system-ID GP producer on Rohini

The next stage replaced the Python population producer with a complete C99
system-genome GP loop. Python compiles and inspects the fixed CUDA template once
and makes one search call. C99 owns both preallocated population arenas,
postorder mutation/crossover, module-sized batching, winner inheritance, and
the best archive. It feeds the same one-context eager specialization and module
pipeline measured above.

Settings are not materialized on the CPU. The GPU deterministically generates
each genome's settings around its incumbent constants and bindings and reduces
them to one winner. The host uploads about 12 KiB of incumbent state per 1024
genomes and reconstructs only the winning setting. The existing full-MSE and
ordinary materialized-settings mode remains available for diagnostics.

On the RTX 5090, a 1024-genome, 512-setting, 500-generation run used one
128-genome packed kernel per module, one genome per CTA, one specialization
worker, and two modules in flight. Eight modules cover each generation:

| Metric | End-to-end C99 GP |
| --- | ---: |
| One-time template compilation | 0.137 s |
| Search wall time | **4.596 s** |
| System genomes evaluated | 512,000 |
| Genome/setting configurations | 262,144,000 |
| System genomes/s | **111,412** |
| Configurations/s | **57.04 M** |
| Modules specialized, loaded, and run | 4,000 |

The search wall time excludes the one-time CUDA compilation shown separately.
Its timed path includes evolution, specialization, module lifecycle, GPU
execution/reduction, compact readback, winner inheritance, and next-generation
production. It retains 78.1% of the raw 512-setting winner-path genome rate
(`142,713` genomes/s), which excluded GP production. About 93.9% of wall time
remained in the eager pipeline and only 4.2% in evolution, so population
production is no longer starving the runtime.

The best observed MSE was `0.0028249` at generation 407. This is only a search
sanity result, not yet a structural fed-batch recovery claim. Replaying the
winning setting through the materialized GPU kernel produced the exact same
FP32 MSE; the independent CPU trajectory reference agreed to `8.9e-6` relative
error. The measured winner used 94 registers with no stack or local storage;
the final winner reducer used 16 registers with no stack or local storage.

The 100-generation setting sweep favored 512 settings as the current knee:

| Settings/genome | System genomes/s | Configurations/s |
| ---: | ---: | ---: |
| 256 | **112,468** | 28.79 M |
| 512 | 109,662 | **56.15 M** |
| 1,024 | 59,673 | 61.11 M |

Moving from 512 to 1,024 settings added only 9% configuration throughput while
roughly halving structure throughput. Eager depths two and three were tied;
depth one was about 11% slower. The immediate next experiment is controlled,
multi-seed recovery of the blinded rate laws, followed by selective resident LM
promotion of promising survivors.
