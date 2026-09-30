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

# Rohini ideal CUDA module lifecycle benchmark

Date: 2026-08-31
GPU: NVIDIA GeForce RTX 5090 (`sm_120`)
Runtime: one CUDA context, worker-owned CUBIN copies, eager module queue

## Outcome

The standalone benchmark works and exposes three distinct regimes:

1. Exact-image reuse can materially overstate module throughput, particularly
   for small modules and large initialized-data images.
2. Code-heavy modules cost substantially more to look up and unload than
   equally sized initialized-global-data modules.
3. Existing System ID runs are not purely module-lifecycle-bound: matched
   early-exit modules move 5.2--6.5 times as many modules per second as the real
   scoring runs.

The benchmark therefore supplies the intended diagnostic: if a future System
ID configuration approaches the matched early-exit module rate, module
lifecycle is the bottleneck. If it remains far below that rate, useful kernel
execution or another production stage is still dominant.

## Stable single-entry results

These longer measurements use byte-distinct modules, an early global-memory
gate exit, and the best tested eager depth for each shape.

| Padding | CUBIN bytes | Functions | Unique modules/s | Identical modules/s | Unique load | Unique lookup | Unique unload |
|---|---:|---:|---:|---:|---:|---:|---:|
| BRKPT code | 65,960 | 1 | 14,454 | 22,996 | 33.93 us | 16.56 us | 15.11 us |
| BRKPT code | 264,104 | 1 | 14,507 | 15,632 | 29.29 us | 20.94 us | 13.80 us |
| BRKPT code | 1,056,680 | 1 | 4,819 | 5,124 | 71.55 us | 65.14 us | 36.63 us |
| Global data | 263,144 | 1 | 16,899 | 16,770 | 45.69 us | 3.28 us | 6.01 us |
| Global data | 1,049,576 | 1 | 6,724 | 6,714 | 118.39 us | 3.46 us | 22.68 us |
| Global data | 3,501,000 | 1 | 1,884 | 2,679 | 472.97 us | 4.08 us | 46.45 us |
| Global data | 6,671,000 | 1 | 866 | 1,465 | 1,053.97 us | 6.05 us | 81.02 us |

The global-data controls demonstrate why CUBIN byte size alone is not a valid
predictor. At approximately 264 KB, the global-data module is faster overall
despite a slower load because its function lookup and unload are much cheaper.

## Matched System ID module shapes

The code-heavy fixtures below match both approximate CUBIN size and function
count. The timed kernels still return immediately.

| Shape | Synthetic CUBIN | Matched lifecycle configuration | Ideal unique modules/s | Real System ID modules/s | Lifecycle ceiling / real |
|---|---:|---|---:|---:|---:|
| 8 kernels/module | 3,366,592 bytes | 1 worker, 4 streams, depth 16 | 715.2 | 136.83 | 5.23x |
| 16 kernels/module | 6,726,176 bytes | 1 worker, 16 streams, depth 32 | 343.3 | 53.04 | 6.47x |

The 8-kernel real module contains 2,048 candidate RHS ASTs. Its matched
lifecycle ceiling is approximately 1.465 million ASTs/s, compared with the
measured 280,234 ASTs/s while scoring 512 settings over real trajectories.

The 16-kernel real module contains 4,096 candidate RHS ASTs. Its matched
lifecycle ceiling is approximately 1.406 million ASTs/s, compared with the
measured 217,269 ASTs/s.

These ratios do not mean module loading is irrelevant. In the real runs,
reported mean `cuModuleLoadData` time was 2.88 ms for the 3.34 MB module and
7.88 ms for the 6.67 MB module, with long p95 tails. Loading becomes slower
while meaningful kernels are executing, and it accounts for a material share
of the effective per-module wall time. It is not the sole limiter at 512
settings.

## Lifecycle-only topology ceiling

This sweep recreates the earlier 256-system module layouts. Every module
represents 256 complete systems and 512 component RHS ASTs, but all functions
exit after one global load. The table selects the best of 1, 4, 16, and 32
streams at loaded depth 32.

| Systems/entry point | Functions/module | Approximate CUBIN | Best streams | Modules/s | Implied component AST lifecycle ceiling |
|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 2,032,744 | 4 | 373 | 190,804 AST/s |
| 2 | 128 | 1,112,936 | 1 | 751 | 384,238 AST/s |
| 4 | 64 | 640,992 | 1 | 1,425 | 729,398 AST/s |
| 8 | 32 | 392,168 | 4 | 2,724 | 1,394,802 AST/s |
| 16 | 16 | 269,864 | 1 | 4,705 | 2,409,185 AST/s |
| 32 | 8 | 207,680 | 1 | 7,153 | 3,662,275 AST/s |
| 64 | 4 | 176,024 | 1 | 9,016 | 4,616,213 AST/s |
| 128 | 2 | 159,456 | 1 | 11,517 | 5,896,684 AST/s |
| 256 | 1 | 151,880 | 32 | 17,644 | 9,033,540 AST/s |

Module lifecycle alone strongly favors fewer entry points and more systems
packed behind each entry point. Resident useful execution favored approximately
4--8 systems per entry point. The actual end-to-end optimum is therefore a
crossover between these curves; this benchmark does not replace a real
specialize/load/execute/unload run of each useful kernel topology.

## Fixed-size function-count control

To separate function count from total image size, code-heavy fixtures targeted
1 MiB while varying only the number of exported kernel entry points. Actual
CUBIN sizes ranged from 1,052,904 to 1,056,680 bytes, a maximum spread of
3,776 bytes (0.36%). Each entry point was resolved and launched once per module
on one stream. Results below use byte-distinct modules, one loader worker,
loaded depth 32, busy polling, and 4,096 timed modules.

| Functions/module | Actual CUBIN bytes | Modules/s | Load us | Lookup us | Launch/record us | Unload us |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,056,680 | 2,380 | 151.6 | 88.5 | 2.1 | 176.2 |
| 2 | 1,056,512 | 2,472 | 138.1 | 87.9 | 3.3 | 173.7 |
| 4 | 1,056,336 | 2,539 | 118.0 | 100.4 | 5.5 | 168.5 |
| 8 | 1,055,936 | 1,904 | 93.7 | 253.9 | 10.3 | 165.7 |
| 16 | 1,056,296 | 1,885 | 78.2 | 269.9 | 19.9 | 161.1 |
| 32 | 1,052,904 | 1,594 | 106.4 | 309.8 | 38.8 | 170.5 |
| 64 | 1,053,672 | 1,167 | 168.7 | 390.0 | 77.9 | 218.6 |

The four-function fixture was the fastest in this synthetic shape. Increasing
from four to 64 functions reduced end-to-end lifecycle throughput by 54.0%,
despite nearly constant file size. The dominant increase was function lookup
and launch work. `cuModuleLoadData` time was not monotonic, suggesting that
splitting code among functions changes driver parsing and/or moves lazy
per-function preparation between module load and `cuModuleGetFunction`.

A four-worker control produced the same ordering and similar rates: 2,318,
2,496, 2,517, 1,878, 1,794, 1,585, and 1,154 modules/s for 1 through 64
functions. Therefore the curve is not explained by single-worker starvation.
It remains a synthetic early-exit measurement: real useful kernels must be
tested at the likely crossover between fewer functions and resident execution
efficiency.

## Longer-kernel overlap control

The generated kernel accepts a runtime `wait_clocks` argument. On the timed
path it reads the global gate, executes a uniform `clock64()` loop, and exits.
Disassembly confirms `SR_CLOCKLO` reads and a uniform backward branch. Changing
the delay therefore does not change the CUBIN image or specialization path.

The following control holds total simulated GPU clocks per module constant and
divides them across all exported functions. All functions execute serially on
one stream, so a module receives approximately the same total resident time at
every function count. Rohini reported a maximum SM clock of 3,135 MHz, making
the three nonzero totals approximately 160 us, 638 us, and 2.55 ms. Dynamic
clock behavior makes those durations approximate.

| Functions/module | No wait | 0.5M total clocks | 2M total clocks | 8M total clocks |
|---:|---:|---:|---:|---:|
| 1 | 2,243 modules/s | 2,245 | 1,065 | 337 |
| 2 | 2,341 | 2,291 | 1,066 | 337 |
| 4 | 2,528 | 2,253 | 1,052 | 336 |
| 8 | 1,856 | 1,673 | 911 | 320 |
| 16 | 1,813 | 1,627 | 892 | 318 |
| 32 | 1,576 | 1,464 | 842 | 312 |
| 64 | 1,139 | 1,159 | 745 | 297 |

Longer kernels make fixed module-management overhead a smaller fraction of
wall time. With no delay, 64 functions were 54.9% slower than the four-function
peak. At approximately 2.55 ms of total resident time, one through four
functions were indistinguishable and 64 functions were only 11.7% slower than
four. Function count still matters, but it stops dominating.

The 160 us wait was almost completely hidden for one function because it fit
under the roughly 400--450 us lifecycle. At 638 us and 2.55 ms, GPU residency
became the primary cost. The time attributed to `cuModuleLoadData` rose from
roughly 80--190 us without a wait to approximately 2.6 ms at the longest wait.
This means the driver call blocks or serializes behind active GPU work in this
single-context topology; loading and execution cannot be modeled as two fully
independent timelines. That behavior is consistent with the much higher module
load latency observed while real System ID kernels execute.

A second sweep gave every function the same wait and used 16 streams. At two
million clocks per function, throughput was 1,063, 1,057, 907, 889, 537, and
300 modules/s for 1, 4, 8, 16, 32, and 64 functions. Corresponding function
execution rates were approximately 1.1K, 4.2K, 7.3K, 14.2K, 17.2K, and 19.2K
per second. More entry points can therefore expose more concurrent work when
each function is long, but module throughput declines because each module now
contains proportionally more GPU work.

This remains a residency/overlap control. One warp per block spins on the
clock; it has none of the register pressure, memory traffic, math-pipeline use,
or multiple-CTA structure of a trajectory kernel. Real kernels remain the
decisive end-to-end comparison.

## Packed functions and loaded-module depth

The loaded-depth limit counts retained `CUmodule` objects, not functions. With
`D` loaded modules, `K` exported functions per module, and `B` blocks per
launch, the upper bounds represented by the queue are `D*K` function launches
and `D*K*B` blocks. Each in-flight module additionally retains one completion
event and `min(K, streams)` dependency events.

An initial sweep increased depth and stream count together. It showed 64 packed
functions rising from 5.64K function executions/s at depth/streams 1 to 49.66K
at 16 and 66.53K at 64. That result is valid as a joint depth-plus-stream
scaling measurement, but it is not a pure depth result. The benchmark also had
an unnecessary validation rule requiring depth to be at least stream count.
That rule was removed because the runtime structures support independent
values.

The corrected control holds 16 streams fixed and varies only loaded-module
depth. Every function launches one block and waits 500,000 clocks.

| Functions/module | Depth 1 | Depth 4 | Depth 16 | Depth 32 |
|---:|---:|---:|---:|---:|
| 1 | 2.56K functions/s | 2.25K | 2.27K | 2.26K |
| 4 | 11.52K | 8.54K | 9.01K | 8.97K |
| 16 | 27.97K | 24.72K | 25.74K | 25.59K |
| 64 | 50.95K | 48.39K | 49.78K | 48.31K |

One 64-function module already places four functions on each of 16 streams, so
depth one supplies 64 queued launches. Increasing to depth 16 creates 1,024
queued launches, 16 loaded module objects, and 256 dependency events without
improving throughput. Depth 32 was slightly slower. The same synthetic control
also favored depth one for smaller packs because its 500,000-clock wait fit
largely under module lifecycle time and deeper loading introduced driver
load/unload contention.

The practical rule is to size the queue by represented GPU work, not modules:

1. Compute `D*K*B` and ensure it provides enough runnable CTAs for the desired
   occupancy.
2. Ensure functions are spread over enough streams when launches are too small
   to saturate the device independently.
3. Stop increasing `D` once streams remain nonempty; additional depth only adds
   loaded modules, events, memory, and driver pressure.

For a real function that launches enough CTAs to saturate the GPU by itself,
depth one or two may be sufficient. For one-block functions, packing many into
a module can replace a deep module queue. These clock-wait results do not
determine the final depth for register-heavy RK4 kernels; the real pipeline
must be swept using the same `D*K*B` accounting.

## Completion polling

The current C99 pipeline uses a one-millisecond timed wait when no completion is
immediately visible. With loaded depth one, that policy capped the synthetic
early-exit path near 800 modules/s:

| Code-heavy size | Ideal depth-one rate | 1 ms wait depth-one rate |
|---:|---:|---:|
| 65,960 bytes | 8,182 modules/s | 843 modules/s |
| 1,056,680 bytes | 5,374 modules/s | 791 modules/s |

At depths 8--16, loading later modules usually gives earlier launches enough
time to complete, and the penalty becomes much smaller. Shallow queues should
use an adaptive spin/yield policy or a shorter wait if low-latency module
turnover matters.

## Correctness evidence

- Every fixture contains a 64-bit constant nonce at a uniquely located CUBIN
  offset.
- Unique mode patches a different nonce into every worker-owned image.
- A separate preflight launch reads that nonce on the GPU and must match the
  expected patched value before timing begins.
- Whole-image FNV hashes differ between the template and first timed unique
  image.
- Disassembly confirms `LDG.E.STRONG.SYS`, predicated early `EXIT`, and
  unreachable `BPT.TRAP` padding.
- Multiple-entry fixtures resolve and launch every function, record one
  dependency event per used stream, join them on a completion stream, and
  unload only after that completion event succeeds.

## Material deviations and limits

- Unique synthetic images differ in GPU-visible constant data but retain the
  same SASS instruction bytes. Identical-image mode is reported beside them,
  and real Secant unique-SASS runs remain the strongest cache control.
- BRKPT padding approximates code size and executable metadata but is not the
  same instruction distribution as an RK4/SSE kernel.
- Global-data padding is a deliberately non-equivalent image-size control.
- Kernel work is intentionally trivial. Reported AST ceilings are mappings from
  known ASTs/module, not evaluated symbolic expressions.
- Fixture compilation, context creation, stream creation, and buffer allocation
  are excluded. Worker copying, loading, lookup, launches, events, and unloading
  are included.
- Rohini's installed `nvcc` has the known host-header `rsqrt` conflict. Fixtures
  were compiled by the repository's existing NVRTC-to-PTX path and installed
  `ptxas`; compilation is outside all lifecycle timings.
- A first monolithic 3.5 MB BRKPT attempt caused pathological compiler memory
  growth and was stopped. Padding is now split across functions and guarded at
  65,536 BRKPT instructions per function; the successful 8- and 16-function
  fixtures stayed within that limit.
- Separate invocations showed ordinary clock/runtime variability. Stable tables
  use longer runs, but comparisons with earlier production sweeps are not
  simultaneous measurements.

Raw reports are in `generated/module_lifecycle*.json` locally and under
`~/secant-system-id/generated/module_lifecycle/` on Rohini.
