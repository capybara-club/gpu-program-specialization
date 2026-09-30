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

# Native request runtime: integration and validation

Date: 2026-09-11. Implementation: [runtime/](../../runtime/README.md),
[thin Python transport](../../python/odezza/grammar/native_service.py).
Validation host: rack1, one NVIDIA RTX 5080, SM120, driver 595.91.07,
`CUDA_MODULE_LOADING=EAGER`, isolated checkout
`/home/cdurham/odezza/scratch/grammar_handoff_20260911`.
The second GPU was not used. No GP campaign or existing server was replaced.

## What is connected

The default direct-grammar CLI/MCP route now copies one JSON request into a queued
C job. Its worker prepares trajectory/static/grammar arenas, prespecializes fixed
equations, runs double-buffered family producers, binds resident numeric pools,
launches scoring and GPU reduction, and keeps compact winners in RAM. ASTs are
not serialized into intermediate JSON. Python performs transport/queuing, final
formatting and optional independent CPU replay only.

Family allocations are reserved before generation. A compatible configuration
tile, rather than an entire family, is the scheduling quantum. Numerical bank
rows can span tiles without changing their addresses. Generation overlaps scoring.
CUDA uploads feed the nonblocking stream through an explicit completion event.

Core changes from the previous frontend turn remain limited to the additive NVRTC
timing accessor. This integration adds no grammar, search or context ownership
to the core. The existing Python compiler/SQLite/LM adapter remains available via
`--backend python`; native scoring is the default direct-grammar backend.

## Measurements

These are synthetic **execution** workloads, not discovery success or time-to-solve
claims. “Structural systems” below means distinct complete RHS-vector/binding
candidates within a family, not unique individual RHS subtrees.

| Workload | Structural systems | Configurations | Warm elapsed |
|---|---:|---:|---:|
| Matched explicit grid, native | 512 | 1,048,576 | 23.71 ms median |
| Same explicit grid, previous Python backend | 512 | 1,048,576 | 1.039 s median |
| Expanded grid, native | 512 | 8,388,608 | 70.92 ms |
| Supplied adaptation grammar, native Philox profile | 8,192 | 1,048,576 | 246 ms |
| Six-equation structural combinations, native | 1,000,000 | 32,000,000 | 2.516 s |

The matched grid uses eight alternatives for each of three equations and 2,048
coefficient rows. Both routes use the same trajectories, explicit FP32 constants,
module capacity 32, patch capacity 384 and objective. All retained global program
bytes and FP32 MSE values match, not just the best candidate. Four native warm
runs at producer page sizes 128/256/512 take 22.1–25.4ms; two Python warm-context
runs take 1.037–1.040s. The measured warm ratio is about **44×**. Samples are few;
this is a workload-specific result, not a general speedup guarantee.

The first native grid request with cached template files but a new runtime takes
156.6ms including context/module initialization. Its warm jobs take about 24ms.
Do not compare that initialization cost to a warm legacy worker without labeling
it. [Raw matched runs](runtime-validation/odezza-runtime-benchmark.json).

The six-equation population uses ten expression alternatives per equation, giving
10^6 distinct full vectors, and 32 coefficient rows each. Individual equation
subtrees are heavily reused. There are 977 streamed tiles, all 32 million scores
are valid, and exact dedup reports no repeated full vectors. Three observations
per state on one short trajectory make this an inexpensive rollout workload.
The cold run is 4.564s, including 2.030s of actual NVRTC compilation; the cached
run is 2.516s with zero NVRTC time. Warm timing scopes:

| Scope | Seconds |
|---|---:|
| Parse/prepare job arenas | 0.024 |
| AST generation, concurrent with scoring | 0.650 |
| Native scoring pipeline | 1.780 |
| GPU prelude | 0.004 |
| Reduction/gather and transfer | 0.536 |
| Retention processing | 0.105 |
| Template preparation | 0.006 |

Timing scope clarification: the 2.516s scale figure is the C job duration with
cached templates. That measurement used a fresh service process: runtime
initialization added 93.7ms, giving 2.611s in the Python worker. The harness
checked completion once per second and observed 3.001s; that extra polling
latency is not execution time. The cold-template run used 4.564s in C and
4.699s in the worker including initialization. The matched-grid warm runs
instead reuse an already initialized runtime and are reported end to end.

These times are not additive: generation overlaps the worker, NVRTC is included
in template work, and some scopes exclude allocation/scheduling. Pipeline time
includes specialization and module loading. No peak GPU occupancy claim follows
from these numbers. Remaining host/result traffic and tile scheduling merit
profiling. After completion, this job retains about 12KB of prepared arenas and
9KB of compact winners; this excludes the copied JSON, Python objects, persistent
CUDA allocations and modules. Generation arenas are released, not retained with
the job. Peak process memory is not established by this post-completion figure.
[Cold](runtime-validation/scale-cold.json),
[warm and CPU replay](runtime-validation/odezza-runtime-scale.json).

## Correctness checks

- Strict C99 build with warnings as errors. Runtime exports only its eight public
  entry points; no internal grammar helper symbols are exposed.
- Fifteen focused runtime contracts pass: masked irregular observations, state
  toggles, grid/Philox products, chunk invariance, CPU Philox uniform reconstruction,
  exact winner/address replay, duplicate grid rows past the reducer's first 16,
  late tag aliases, unrequested metadata tags, family allocation prefixes,
  patch-capacity growth/cache reuse, invalid transform/memory rejection, queued
  cancellation, bounded retention timeout, context reuse, and MCP submission.
  Some checks are grouped in the same reported case.
- CPU replay validates every retained row in the small successful cases, and best
  rows in scale/benchmark cases. The million-system best CPU MSE is approximately
  8.75e-19. This is an execution fixture with a known generating formula.
- The frontend's measure/short-buffer/cursor contracts and 225 supplied native
  program-prefix comparisons pass under AddressSanitizer and UBSan on mac1.
  Its CPU archive still has no allocator, file/database or CUDA dependencies.
- The 83 original grammar tests, 19 adapter/transport tests and 60 main Python
  tests pass. CUDA memory checking reports **zero errors** for the focused native
  contract suite and for the mixed-layout sequence.
- The old aggregate test target's previously documented cubin-runner packing
  failure was not changed by this work. The new optional GPU CMake target was
  added but not configured on rack1, which lacks CMake; Make is the tested GPU
  build route. No missing tools were installed.

[Conformance records](runtime-validation/odezza-runtime-conformance.json),
[CUDA checker](runtime-validation/odezza-runtime-final-memcheck.log).
An attempted host-ASan build of the complete Python/CUDA runtime could not
initialize its CUDA runtime; it is not counted as a passing runtime sanitizer
check. The passing host sanitizers cover the CPU frontend, and the passing CUDA
checker covers device execution.

## Material deviations and repairs

**Repaired: missing upload-to-execution ordering.** During development, a sequence
of large grids followed by the supplied adaptation grammar produced incorrect
partial scores and then `numeric prelude failed`. The isolated instrumented run
passed because CUDA checking changed scheduling. Those early failed-job outputs
are invalid and must not be used as performance or recovery evidence. Pageable
synchronous HtoD calls may return after staging while DMA remains in progress;
the new nonblocking stream lacked a dependency on that work. The runtime now
records an upload event and waits on it before prelude/scoring. The original
uninstrumented sequence, independent CPU replay, repeated benchmarks and CUDA
memory checks pass. This uses an event, not a full stream/device synchronization.

**Repaired: repeated large-grid hashing.** The 8.39M grid job initially took about
0.672s because its 16,384-value grid was rehashed for each candidate/tile. A
job-local immutable-binding fingerprint now reuses the content key. The same
workload takes about 0.070s in the C worker; scored work is unchanged.

**Intentional profile difference.** Structural C sampling and numeric Philox stream
addressing use explicitly versioned native profiles. They do not reproduce the
old Python sampler/hash-derived numeric banks from the same JSON seed. Reports
record actual seed, stream, axis index, coefficients and resolved bytecode. The
supplied adaptation timing is consequently not a matched RNG-result comparison
against the historical 10.25s Python measurement. The explicit-grid comparison
above establishes matching work and full retained results without this ambiguity.

**Intentional retention identity difference.** Native `variant` identity is
family plus accepted AST ordinal, so cross-family duplicates remain separate
variant entries. Numeric/structural retention compares resolved program content.
Family/tag leaderboards may reference distinct candidate addresses even when the
resolved model matches. Native process-local job IDs and compact reports are not
compatible with persisted Python manifests or its LM replay IDs.

**Memory/cancellation scope.** Job arenas, numeric pools and tile buffers have
separate budgets. They are not a total process RSS/VRAM cap; driver/modules and
core workspaces are additional. Requested tag rankings retain per-AST/permutation
finalist archives while generation is active. If cancellation/deadline interrupts
extra uniqueness-reduction passes, `retention_complete=false` explicitly marks
partial retention, while already evaluated scores count toward completed work.

## Remaining work

1. Profile/batch further across execution tiles and add multi-GPU dispatch without
   changing candidate addresses or family allocation semantics.
2. Define per-skeleton diversity allowances and bounded checkpoint/seek semantics.
3. Add explicit reusable trajectory views and GPU per-trajectory diagnostics.
4. Run blind recovery and representative longer/sparser trajectory campaigns via
   this runtime before drawing discovery or peak-throughput conclusions.
5. Add stronger total-memory accounting and retain/release controls for a
   long-lived multi-client service. Current job handles are bounded and in-memory.

The live five-item tracker is [TODO.md](../../scratch/structural_search_trial/TODO.md).

## Follow-up: larger configuration banks

A matched three-repetition sweep now evaluates one million systems at 32, 1,024
and 2,048 coefficient rows. The 2.048B-evaluation job reaches 719M configs/s in
2.848s with a tile large enough for a complete producer page; the existing default
splits that work and takes 5.696s. All retained winners agree across tile settings.
See [configuration-scale measurements and next scheduling priority](C99_CONFIGURATION_SCALE.md).
