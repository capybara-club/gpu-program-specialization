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

# Grammar overhead findings — 2026-09-11

**The ten-second job is dominated by host grammar processing and intermediate
candidate reconstruction/storage. The numerical GPU work is a small fraction.**

All measurements ran on rack1 GPU 0, RTX 5080, in the isolated checkout
`/home/cdurham/odezza/scratch/grammar_handoff_20260911`. Existing services remained
running but were idle in the before/after GPU checks. No production grammar,
retention policy, C99 code, SQLite durability setting or runtime fallback changed.

## Workload and correctness

- Supplied `adaptation_search.json`: 8,192 whole-system skeletons, each with 128
  coefficient configurations, for **1,048,576 configuration visits per run**.
- Three states; two trajectories, five observations each, duration 0.1, matching
  the earlier execution fixture. This is not a claim about realistic recovery
  quality, long-trajectory latency or peak throughput.
- Unchanged execution settings: batch variants 64, module systems 32, patch
  capacity 1,024, execution limit 1,200 seconds; native reducer k=16.
- 130 adapter batches and 258 specialized scoring modules per job.
- **13 complete jobs, 13,631,488 configurations, zero invalid scores.** All jobs
  have identical full global/family/tag winner records and provenance. Three
  winners per job independently pass CPU FP64 replay using exact FP32 values.
- Winner report hash across all jobs:
  `7e09ebc76fcc86902e8d90c6912e7b931b4bc896eed4df9c885a57bfc77afb0b`.

## Normal latency and instrumentation cost

Four warm, uninstrumented runs took **10.234, 10.263, 10.239 and 10.367 seconds**:
median **10.251 seconds**. The initial new-context run took 10.384 seconds. Disk
and driver caches were not flushed, so this is not a forced cold-NVRTC benchmark.

The two coarse instrumented runs averaged 10.489 seconds, about 2.3% above the
warm baseline median. Finer SQLite timing runs took 10.724 and 10.749 seconds.
cProfile took 22.002 seconds and must not be used as normal runtime. Its function
counts/hotspot locations remain useful. The warm CUDA trace took 10.465 seconds.
These modes are numerically equivalent, but their elapsed times are not equivalent.

## Where the worker time goes

Means from the two coarse timing runs. These rows use separate scopes and form a
wall-time partition; nested child times are not added twice.

| Work | Seconds |
|---|---:|
| Grammar expansion, validation, initial lowering and plan storage | 3.495 |
| Read plan records, form batches and lower programs again | 0.565 |
| Resolve intermediate candidates and build numeric/structural identities | 1.772 |
| Hash candidate records and execution identities | 0.422 |
| Serialize, insert, prune and commit intermediate survivors | 3.035 |
| Build final leaderboards and readable results | 0.371 |
| Native scoring wrapper, including specialization/module/integration | 0.168 |
| Remaining orchestration, setup, reducer, uploads and resource management | 0.662 |
| **Total** | **10.489** |

This explains the formerly unattributed 6.75 seconds: the largest omitted scopes
were survivor persistence, candidate identities/hashing and post-plan batching.
The native service report measures the call inside the Python wrapper, explaining
its slightly smaller ~0.14-second scoring time. It remains a combined host/native
pipeline measurement, not pure GPU kernel duration.

The request's tag policies retain five numeric candidates per structural binding.
The current controller conservatively keeps that many for every variant before
combining the final leaderboards: **40,960 intermediate candidates**, not merely
the final global top-20. Preserving family/tag coverage and distinct-unit semantics
is essential; replacing this with global raw top-k would be incorrect.

## Storage and repeated work

The finer timer measured survivor persistence as:

| SQLite retention operation | Calls | Mean seconds |
|---|---:|---:|
| Commit | 130 | 1.672 |
| Insert candidate | 40,960 | 0.835 |
| Lookup existing candidate | 40,960 | 0.095 |
| Prune a structural binding | 8,192 | 0.048 |

These times are components of persistence, not additions to its ~3-second total.
The separate CUDA/OS trace saw 678 `fdatasync` calls totaling 2.473 seconds across
the entire traced job, including other database/schema/journal activity. Do not
treat that separate trace's total as identical to the scoped retention commits.

The final SQLite plan-and-candidate file is **281,251,840 bytes**. Stored JSON
payloads account for approximately 16.76 MB of skeletons, 53.43 MB of variants,
5.62 MB of provenance and 95.71 MB of intermediate candidates; SQLite pages and
indexes account for additional space. This is database file size, not measured
physical bytes written to disk. The final JSON report is only about **214 KB**.

cProfile counted **294,912 calls to RHS lowering**:

- 8,192 systems × 3 RHS × two passes during preparation/dispatch = 49,152 calls.
- 40,960 intermediate candidates × 3 RHS × numeric and structural identity forms
  = 245,760 calls.

The structural part of identity construction is invariant across coefficient
configurations of a given binding. This offers a concrete caching opportunity.
The profile also exposes JSON encoding and recursive copying in grammar expansion.

## CUDA timeline

The already installed Nsight Systems captured only the warm job. Actual GPU
kernel durations, summed over all launches:

| Kernel work | Launches | Seconds |
|---|---:|---:|
| RK4 scoring | 258 | 0.049218 |
| First-stage score reduction | 130 | 0.003674 |
| Merge reduction | 130 | 0.002389 |
| Coefficient prelude | 130 | 0.001160 |
| Gather winner constants | 130 | 0.000153 |
| **All kernels** | **778** | **0.056593** |

There was no overlap between recorded kernel intervals, so their union is also
0.056593 seconds: approximately **0.54% of this traced job's elapsed time**. This
describes this process's kernel activity; it is not measured SM occupancy or a
system-wide utilization counter. The first kernel began after grammar planning.

Native driver API totals include 258 module loads (0.0514 seconds) and 258 unloads
(0.0693 seconds). Those are summed call durations across threads and can overlap
other pipeline work; they cannot simply be added to kernel time to predict latency.
There is no evidence that multi-second module loading explains this warm job.

The usual scoring grid is **32 CTAs of 128 threads**, with 95 registers/thread
(some simpler modules use 63). The GPU has 84 SMs. Those launches are too small to
occupy every SM individually, and the recorded kernels did not overlap. Larger
numeric banks/module batches merit a separate throughput sweep. However, reducing
49 milliseconds of scoring will barely change this ten-second host-bound job.

Device-to-host transfer was **11,550,720 bytes**, taking about 0.000472 seconds on
the copy engine. The reducer gathers 16 coefficient rows for each binding even
though only five survive locally. At this bank length the downloaded records and
coefficient rows exceed the **4,194,304 bytes** that the full scalar MSE population
alone would occupy. Reduction does not guarantee smaller transfer for every
geometry. This overfetch is real, but transfer bandwidth is not the multi-second
bottleneck. The complete reducer/gather/download wrapper took about 0.103 seconds
in coarse measurements. The next design should retain compact indices longer.

Trajectory packing/uploading was about 0.052 seconds across 130 batches. Reusable
trajectory assets and buffers remain useful, but would not alone remove seconds
from this particular job.

## Recommended changes, in order

1. **Compact intermediate survivors and batch durable updates.** Store addresses,
   scores and necessary identity/provenance references; materialize rich snapshots
   for retained output. Preserve correct uniqueness, late membership and partial
   results. Define checkpoint durability explicitly instead of silently disabling
   SQLite synchronization for a faster number.
2. **Cache per-binding structural identities and lowered programs.** Eliminate
   repeated preparation/dispatch lowering and the same structural identity work
   for every retained coefficient row. Preserve exact current identity semantics.
3. **Reuse prepared grammar populations when valid.** Separate immutable structure
   and pool plans from a particular evaluation, with checked state/known-RHS and
   numerical-context compatibility. Avoid re-expanding unchanged requests.
4. **Gather only required coefficients and normalize report objects.** Choose an
   appropriate supported reducer k or delay gather until survivors are known;
   duplicates may require additional passes. Reference shared candidates once in
   MCP leaderboards. Measure this after the larger persistence/identity costs.
5. **Then tune GPU work per module.** Sweep useful numeric-bank sizes, module
   sizes and longer trajectories using prepared ASTs. Report native execution,
   module overhead and achieved workload separately from end-to-end latency.

No speedup is claimed yet: this task measured and localized the overhead. The
unchanged kernel/core boundary is intact. The highest-priority open work is the
compact, correctness-preserving survivor path, followed by repeated-lowering
removal—not a different RK4 kernel shape.

## Evidence and reproduction

- [Normal/coarse/cProfile runs](results/20260911/overhead-01-summary.json)
- [Fine SQLite timings](results/20260911/overhead-02-summary.json)
- [Trace job and correctness](results/20260911/trace-summary.json)
- [CUDA/OS aggregates and shape data](results/20260911/trace-analysis.json)
- [Storage payload breakdown](results/20260911/storage-analysis.json)
- [CPU profile](results/20260911/worker.prof)
- [Nsight timeline](results/20260911/overhead-trace-01.nsys-rep)
- [Capture SQLite](results/20260911/overhead-trace-01.sqlite)
- [Harness and instructions](README.md)

Full requests, source hashes, job reports and replay data remain on rack1 in
`overhead-01`, `overhead-02` and `overhead-trace-run-01` under the isolated checkout.
Local environment JSON records pin source/native hashes; the measured adapter
and compiler files were verified byte-for-byte against mac1. All benchmark jobs
finished, and both rack1 GPUs returned to their pre-run idle state.
