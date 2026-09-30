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

# System-19 scoring slowdown: confirmed tile-sizing defect

Measured on rack1's two RTX 5080s, 2026-09-13. The same four structural requests
completed in **41.57s instead of 222.64s: 5.36× faster, 81.3% less time**. The
historical service run took 223.08s, closely reproduced by the fresh baseline.
The core scoring library, RHSs, observations, RK4 step, RNG seeds, coefficient
ranges, work allocations and retention policy are unchanged. All retained
candidate identities, expressions, coefficients, scores, rankings and
valid/invalid counts match the original saved service reports exactly.

This diagnoses a runtime batching defect. It is not a deployed optimization or
a repeat of the complete blind-recovery workflow.

## Full-request comparison

| Screen | Configurations | Original-cap replay | Larger-tile replay | Speedup |
|---|---:|---:|---:|---:|
| Initial structural screen | 224,919,552 | 150.761s | 28.399s | 5.31× |
| Second structural screen | 53,084,160 | 46.462s | 8.672s | 5.36× |
| Quadratic additions | 33,947,648 | 22.030s | 3.898s | 5.65× |
| Polynomial fitting screen | 5,242,880 | 3.391s | 0.598s | 5.67× |
| **Total** | **317,194,240** | **222.643s** | **41.566s** | **5.36×** |

Timings are native request wall times, excluding context creation, network
transport and final report serialization. The baseline is a fresh single replay
per request. Larger-tile values are one unprofiled warm replay after a first run;
they are not many-run medians. The separate profiled replays corroborated the
timings (28.299s, 8.657s, 3.966s and 0.597s). Only the baseline second screen had
an NVRTC cache miss, costing 0.131s; that cannot explain a roughly 181s gap.

The isolated runtime raises only the aggregate tile-work ceiling from 2^26 to
2^32. The normal 50ms advisory controller, memory budgets and automatic packing
remain enabled. The compiled core library is byte-identical. Total tiles fell
from 25,621 to 2,445. Actual templates stayed at two ASTs except for a maximum
capacity of four in the second screen. A configured ceiling of 64 did not mean
64 active ASTs per launch.

## Why the old policy is slow

Each configuration schedules sixteen trajectories × twenty intervals × sixteen
RK4 steps = **5,120 RK4 steps**, or 20,480 evaluations of the complete RHS. The
runtime applies an aggregate-work limit before the measured-time controller can
choose a large batch:

```text
67,108,864 / 5,120 = 13,107 configurations at most
ceil(13,107 / 128 threads) = 103 CUDA blocks for one active AST
```

The normal launches really did have one active AST and about 103 blocks. That
small grid supplies few independent warps per SM to hide the dependencies in a
long RK4 rollout. Increasing the bank slice exposes many more independent
configurations while each thread still runs the same sequential integrator.

The original work cap appears in commit `385190f3`. The subsequent automatic
sizing validation (`dcc62a0`) covered 2 and 40 RK4 steps/configuration, and its
million-AST throughput fixture used four steps. Those tests did not exercise
the 5,120-step regime or establish its saturation. The hard aggregate cap
overrode the otherwise adaptive tile policy.

## Prepared-input and fixed-shape controls

A prepared-tile probe repeats the same already packed AST descriptors and
device constants/trajectories three times, checking every output score bit on
each replay. CUDA events surround scoring; parsing, generation and copying are
outside those intervals. Pipeline-call wall time still includes specialization
and module loading.

| Configurations/launch | Blocks | Median GPU time | Approximate rate |
|---:|---:|---:|---:|
| 13,107 | 103 | 16.352ms | 0.802M configurations/s |
| 524,288 | 4,096 | 100.086ms | 5.239M configurations/s |

Forty times the work costs about 6.12 times as much GPU time. That directly
locates the main loss in launch-size-dependent execution, rather than parsing or
module loading. Between the two tilings, complete retained outputs and coverage
also match; all raw scores were compared within each prepared replay, not
downloaded and compared for every original 317M evaluation.

A separate unprofiled control fixes the template at two AST slots, disables
automatic configuration sizing using explicit tile counts, and scores the same
complete 5,242,880-evaluation `polyfit` request on one GPU:

| Configurations/tile | Blocks for the active AST | Tiles | Warm request time |
|---:|---:|---:|---:|
| Original cap, approximately 13,107 | 103 | 410 | 6.820s |
| 65,536 | 512 | 80 | 1.379s |
| 131,072 | 1,024 | 40 | 1.360s |
| 262,144 | 2,048 | 20 | 1.040s |
| 524,288 | 4,096 | 10 | 1.029s |

All retained results and coverage agree exactly. The last doubling buys only
about 1% here, giving a measured saturation region. Even 65,536 configurations
recover most of the loss. The fixed sizes divide the bank evenly; automatic
sizing with the larger cap took 1.186s and made 28 tiles. Avoiding small tail
tiles is a secondary tuning opportunity, separate from the primary cap defect.

## Timing attribution and limits

For the profiled 28.299s initial screen, scoring GPU intervals total 28.111s on
GPU 0 and 28.022s on GPU 1. Their execution overlaps. Module loading took about
46ms on GPU 0; JSON parse/reservation took 13.7ms for the request. AST generation
took under 1ms. These counters have different scopes and must not be summed as
a disjoint wall-time breakdown. Kernel execution dominates the remaining time;
it need not imply full theoretical SM occupancy.

Nsight Compute is installed, but hardware counters returned `ERR_NVGPUCTRPERM`.
No privilege or driver-setting change was attempted. Achieved occupancy, issue
efficiency and specific stall-reason percentages are therefore **unmeasured**.
The failed counter run is excluded from performance comparisons. CUDA events,
recorded grids and fixed-input scaling establish the launch-size effect without
those counters.

The new observed aggregate rate is 7.63M configurations/s for these full
trajectories. The 768M/s short-workload claim is still not an interchangeable
denominator: every configuration here schedules 5,120 RK4 steps. Invalid
candidates can exit early. The original 223s was unnecessarily slow, but this
experiment does not support a subsecond time for all this work.

## Recommended runtime repair

1. Preserve per-configuration integration admission. A single thread's
   sequential work needs its own bound, independent of the number of parallel
   configurations that the GPU can execute.
2. Choose tiles using device/shape capacity and measured scoring duration, with
   enough blocks to reach the measured throughput plateau when enough work is
   available. Reconcile that with a hard operator work/memory envelope; do not
   make a customer-provided limit disable service admission.
3. Balance bank slices to avoid tiny final tiles. Report the binding limiter,
   actual active ASTs, blocks, bank rows, and maximum/percentile kernel duration.
4. Add a long-rollout fixed-input regression to the sizing suite, including
   mixed RHS complexity, invalid candidates, one/two GPUs and cancellation.

The diagnostic 2^32 cap is **not** a validated universal production default.
One measured cold large launch took about 100ms despite the advisory 50ms target;
maximum cancellation latency was not established across all shapes. No pipeline
overlap, kernel change, service deployment or new solve policy was introduced.
The live core and runtime hashes still match the original build manifest.

## Artifacts

- [Compact results and timing scopes](compact-results.json)
- [Full-screen numeric summary](summary.json)
- [Original service-result signatures](original-signatures.json)
- [Reproduction and isolation protocol](README.md)

Raw requests, reports, logs, the private diagnostic runtime and its manifest are
on rack1 at `/home/cdurham/experiments/recovery19-throughput-20260913`. The compact
results include SHA-256 hashes of those raw reports. The same validated signatures
were independently checked against the original mac1 reports after collection.
