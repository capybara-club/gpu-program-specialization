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

# CPU and GPU throughput analysis

This comparison uses the same score-only astaxanthin workload on rohini's AMD
Ryzen 9 9900X and RTX 5090. Both implementations use FP32 fast math, load six
constants per setting, integrate one 96-hour trajectory with fixed-step RK4,
score four channels at 12 observation times, and write one trajectory MSE.

The current Riezzo et al. reproduction is a batch model, despite sometimes
being called fed-batch in discussion.

## Rohini CPU

The installed CPU is an AMD Ryzen 9 9900X:

- 12 Zen 5 physical cores and 24 SMT threads;
- 4.4 GHz base and up to 5.6 GHz boost;
- AVX-512 with a full 512-bit datapath;
- two 6-core CCDs, 12 MiB total L2, and 64 MiB total L3; and
- 120 W default TDP.

The benchmark is compiled with GCC 15.2 using
`-O3 -march=native -ffast-math -fopenmp`. It explicitly processes 16 settings
per ZMM vector. Disassembly confirms ZMM `vfmadd*`, `vmulps`, and `vrcp14ps`
instructions rather than scalar AVX encodings.

Build the native, scalar, and peak binaries with:

```sh
make build/score_only_rk4_cpu_native build/score_only_rk4_cpu_scalar
make cpu-peak-avx512
```

For physical-core and SMT measurements, use `OMP_PLACES=cores` and
`OMP_PLACES=threads`, respectively, together with `OMP_PROC_BIND=close`.

At 16 RK4 steps per eight-hour observation interval:

| Implementation | CPU threads | Trajectories/s | Time/trajectory | Time/3 trajectories | RHS evaluations/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scalar FP32 | 1 | 225,281 | 4.439 us | 13.317 us | 1.730e8 |
| AVX-512 FP32 | 1 | 2.813e6 | 355.4 ns | 1.066 us | 2.161e9 |
| AVX-512 FP32 | 12 physical | 3.324e7 | 30.08 ns | 90.25 ns | 2.553e10 |
| AVX-512 FP32 | 24 SMT | **5.642e7** | **17.73 ns** | **53.18 ns** | **4.333e10** |

AVX-512 is 12.5x faster than the deliberately non-vectorized scalar build on
one thread. Twelve physical cores scale the one-core vector result by 11.8x.
SMT then adds an unusually large 1.70x because the rate laws contain serial
reciprocal dependency chains between RK4 stages; the sibling thread can occupy
the core while another chain waits.

At 80 steps per observation interval, all 24 SMT threads sustain 9.914 million
trajectories/s, or 100.9 ns/trajectory and 302.6 ns per three-trajectory
setting. RHS throughput remains high at 3.807e10/s.

## Measured CPU FP32 ceiling

`avx512_fma_peak.c` runs 16 independent AVX-512 FMA chains per thread. Counting
each 16-lane FMA as 32 floating-point operations, the 9900X measured:

| Threads | Measured FP32 throughput |
| ---: | ---: |
| 1 | 277.6 GFLOP/s |
| 2 | 672.6 GFLOP/s |
| 4 | 1.408 TFLOP/s |
| 6 | 2.085 TFLOP/s |
| 12 physical | **4.074 TFLOP/s** |
| 24 SMT | 4.021 TFLOP/s |

Two 512-bit FMAs per cycle give 64 FP32 operations/cycle/core. At the stated
5.6 GHz maximum boost, the mathematical socket ceiling is therefore
`12 * 5.6e9 * 64 = 4.301 TFLOP/s`. The measured 4.074 TFLOP/s is 94.7% of that
upper bound and implies roughly 5.30 GHz average all-core frequency during the
short compute test. SMT cannot raise an already saturated FMA ceiling.

The emitted AVX-512 integration loop contains, per 16 settings and one RK4
step, 53 vector FMAs, 64 vector multiplies, 20 vector adds/subtracts, and eight
vector reciprocals. Counting FMA as two operations and reciprocal as one gives
49.5 executed arithmetic operations per RHS evaluation. At 4.333e10 RHS/s,
the full 24-thread ODE run therefore sustains about **2.145 TFLOP/s**, or 52.7%
of the measured FMA-only ceiling. Reciprocal latency and the dependency between
RK4 stages, rather than memory bandwidth, keep it below FMA peak.

## RTX 5090 comparison

The RTX 5090 has 21,760 CUDA cores and an official 2.41 GHz boost clock. The
usual two-FP32-operations-per-core-per-cycle calculation gives an official
FP32 FMA peak of approximately **104.9 TFLOP/s**. During a multi-second version
of this benchmark, the GPU held about 2.54 GHz at 100% SM utilization and
approximately 522-527 W, corresponding to a clock-specific mathematical peak
near 110.6 TFLOP/s.

The GPU RK4 loop's SASS contains, per trajectory and RK4 step, 53 `FFMA`, 48
`FMUL`, four `FADD`, and eight `MUFU.RCP` instructions. This is 41.5 executed
arithmetic operations per RHS evaluation under the same counting convention.
It is lower than the CPU's 49.5 because CUDA fast math uses the hardware
reciprocal directly, whereas GCC refines `vrcp14ps` with a Newton step.

| RTX 5090 regime | Trajectories/s | RHS/s | Emitted arithmetic/s | Reference peak fraction |
| --- | ---: | ---: | ---: | ---: |
| Best short run | 2.260e9 | 1.736e12 | 72.0 TFLOP/s | 68.7% of official boost peak |
| Sustained 3.8-second run | 2.060e9 | 1.582e12 | 65.6 TFLOP/s | 59.3% of 2.54-GHz peak |

The sustained run processed 10,000 consecutive launches, held 100% SM
utilization, and completed one 786,432-trajectory launch in 0.3818 ms. The best
short run completed the same launch in 0.3480 ms.

## What the comparison means

| Metric | Ryzen 9 9900X, 24 SMT | RTX 5090 | GPU/CPU |
| --- | ---: | ---: | ---: |
| Nominal/measured FP32 peak | 4.074 TFLOP/s measured | 104.9 TFLOP/s official | 25.8x |
| Best 192-step trajectory throughput | 5.642e7/s | 2.260e9/s | **40.1x** |
| Sustained 192-step trajectory throughput | 5.642e7/s | 2.060e9/s | **36.5x** |
| 960-step trajectory throughput | 9.914e6/s | 4.514e8/s | **45.5x** |

The application advantage exceeds the raw peak-FLOP ratio because the GPU has
far more independent warps available to hide reciprocal and RK4 dependency
latency, and its fast reciprocal path avoids the CPU compiler's refinement
instructions. Both implementations keep state on-chip and are compute-bound;
this is not primarily a DRAM-bandwidth comparison.

The sub-nanosecond GPU number remains a throughput-equivalent cost, not the
latency of one trajectory. Likewise, the CPU numbers are obtained from large
batches. The directly comparable wall times for 786,432 trajectories are
13.94 ms on the fully threaded CPU and 0.348-0.382 ms on the GPU.

## Architecture references

- AMD Ryzen 9 9900X specifications:
  <https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9900x.html>
- AMD Zen 5 full-width AVX-512 datapath description:
  <https://www.amd.com/content/dam/amd/en/documents/epyc-business-docs/white-papers/5th-gen-amd-epyc-processor-architecture-white-paper.pdf>
- NVIDIA RTX 5090 core count and clock specifications:
  <https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/>
