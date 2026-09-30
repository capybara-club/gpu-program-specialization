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

# Native JSON scoring runtime

This is the optional service layer around `frontend/` and `core/`. It owns CUDA
contexts, numeric pools, scheduling and compact results. The allocation-free parser
API remains in [frontend/](../frontend/README.md); kernel generation,
prespecialization, module loading and scoring remain in [core/](../core/README.md).
There is no GP controller or persisted customer data here. C owns a local SQLite
cache containing only compiled kernel templates.

## Run

On an existing CUDA development checkout:

```sh
make request-runtime
export CUDA_MODULE_LOADING=EAGER
export PYTHONPATH="$PWD/python"
python3 -m odezza.grammar --backend native --root /tmp/odezza-kernel-cache mcp
```

Or execute and export a combined request:

```sh
python3 -m odezza.grammar --root /tmp/odezza-kernel-cache run \
  examples/grammar/submit.json --output /tmp/result.json
```

The native backend is the CLI default and runs in a supervised child process.
Status distinguishes queued work from CUDA `initializing`; initialization has its
own watchdog, so a healthy control loop cannot hide a stalled CUDA initializer.
Use `--devices 0,1` before `mcp` or `run` to use both GPUs for one request.
C generates each AST once and assigns pages to independent context-owning workers.
`--in-process` is available for profiling or embedding without process supervision.

The native backend is now the CLI default. `--backend python` selects the previous
compiler/database adapter, including its planning and LM tools. Native missing
libraries fail explicitly; there is no silent backend fallback. Use
`--runtime-library` / `ODEZZA_RUNTIME_LIBRARY` for a nonstandard runtime build.
CMake users can select `-DODEZZA_BUILD_REQUEST_RUNTIME=ON`. SQLite is vendored; no system SQLite development package is required. Other
requirements are the repository's CUDA/C99 build dependencies and Python transport.

The MCP tools are capabilities, prepare, submit, status, results, cancel and
replay, plus `release`. Submit returns a queued job handle before C parsing or CUDA preparation.
Optional `prepare` stores raw input in memory; validation happens inside the
submitted C job. Completed job and prepared-input handles expire after one hour by default, or
can be released explicitly. `odezza_release` takes exactly one of `job_id` or
`problem_id`; active jobs must finish/cancel before release. Export results first.
Releasing a job also releases its idempotency key; reuse then submits a new job.
A worker restart invalidates its handles; failed work is never silently replayed. There is no result
database, AST JSON intermediate, per-candidate file or persisted problem asset.
Explicit `run --output` exports the final report. Only C writes the kernel cache.

## C boundary

[odezza_runtime.h](odezza_runtime.h) is the single service header:

```c
OdzRuntime *runtime = NULL;
OdzJob *job = NULL;
unsigned devices[] = {0, 1};
odz_runtime_create_devices(devices, 2, cache_directory, &runtime);
/* Or odz_runtime_create(device, cache_directory, &runtime) for one GPU. */
odz_job_create(json, json_bytes, &job);                /* queued, copies input */
odz_job_run(runtime, job);                            /* same worker/context */
odz_job_report(job, NULL, 0, &required_bytes);          /* measure; thread safe */
odz_job_report(job, output, output_bytes, &required_bytes);
odz_job_destroy(job);                                /* after run returns */
odz_runtime_destroy(runtime);                        /* on worker thread */
```

`odz_job_status` uses the same output-buffer signature as `odz_job_report` and
returns a small published snapshot: status/error, counts, elapsed time and retained
memory. It never walks winners or takes the C parsing/report mutex. It may
lag work by one producer page or tile. Detailed family coverage remains in report.

Check all return codes. Report returns `1` for a short output buffer; retry with
its updated required size. Cancellation is thread safe and cooperative between
bounded launches and reduction passes. A CUDA failure that cannot be safely
fenced quarantines the runtime and borrowed work; restart its process. Do not
reuse or concurrently run a runtime. Its implementation can allocate; the three
frontend parsers and AST producer still use only caller-provided arenas.

For a worker that pools bulk data, call `odz_runtime_reserve_buffers(runtime,
trajectory_device_bytes, tile_device_bytes, tile_host_bytes)` before its first
job. These are fixed per-GPU capacities, reserved on each GPU's owning thread.
Trajectory offsets/times/values, score/coefficient/reduction storage and CPU
reduction scratch are views into those slabs. They never grow during a request.
Tiles shrink to fit; a trajectory exceeding its device slab is rejected before
launch. Report `buffer_storage` exposes the policy and selected capacities.

Use `odz_job_create_borrowed(json, bytes, trajectory_arena, trajectory_capacity,
&job)` to avoid copying the request or allocating the CPU trajectory payload.
The caller keeps both buffers alive and unchanged (except writes made by the
trajectory parser) until job destruction. A short arena returns a failed job
with the required size, without falling back to allocation. The ordinary
`odz_job_create` ownership contract remains unchanged for legacy callers.

Only bulk buffers are covered. Handles, pipeline-owned streams/events,
compilation/grammar workspaces, numeric-bank storage and compact winner records
can still allocate. The user explicitly accepts per-job scoring handles and
pipeline-owned streams; a separate shared-stream API is not required.

## Execution

Population counts refer to accepted full candidate RHS vectors and their binding
layouts within each family. They are not counts of unique individual equation
subtrees across those vectors.

1. C validates and prepares trajectories, fixed equations and grammar. The
   trajectory layout is the existing FP32 state-major ragged RK4 representation.
2. Each family has its reserved allocation and independent producer. A host
   generator fills two pages per selected GPU per family while workers score leased pages.
   ASTs are generated once; bank tiling and template growth reuse their bytes.
3. Workers share a family round-robin cursor and lease separate pages per tile. It groups compatible
   coefficient/toggle layouts within each page, with whole toggle products and
   bounded coefficient-row ranges. It does not redistribute unused allocations.
4. Fixed equations are prespecialized into each needed scoring layout. Native
   SQLite template artifacts validate shape/source/checksum. Capacity growth caches the
   enlarged template. C file locks coalesce cold cache misses across workers/processes. Reports distinguish actual NVRTC time from template work.
5. Persistent GPU Philox pools feed a generic resident Cartesian prelude. Scaling
   and shifting happen once per configuration before RK4. Large constant grids
   are fingerprinted once per immutable job binding, not once per AST or tile.
6. CUB reduction returns indices and gathers exact coefficients only for winners.
   Raw family/global rankings without tag rankings reduce entire family tiles on
   the GPU; tag or distinctness policies use finer candidate/permutation groups.
   C merges finalists across tiles/devices by explicit family identity. The
   requested retention determines k. Additional CUB passes remain available for
   exact distinct numeric winners, and reports expose those continuation passes.

Pageable host uploads have an explicit completion event feeding the nonblocking
prelude stream. The scoring pipeline waits on its input-ready event. No device
synchronization was introduced. The current prelude and reducer also wait on
their own completion events. Different GPUs overlap independent tiles, module
loading and reduction. Within one GPU, tile-to-tile prelude/reduction overlap is
still not implemented; the existing core module-loading workers overlap their
own pipeline work.

## Configuration and limits

The [2026-09-12 request-path audit](../docs/grammar/C99_REQUEST_PATH_AUDIT.md)
records open failure-recovery and service-control defects, including numeric-cache
reuse after a failed upload. Successful throughput tests do not cover those
failure-continuation paths; consult the audit before long service campaigns.

`execution` supports these controls (positive numeric values, except the boolean profiler):

| Field | Default | Meaning |
|---|---:|---|
| `batch_variants` | automatic, at most 1,024 | ASTs per producer page per family |
| `module_systems` | automatic, at most 64 | Candidate systems per compiled scoring module |
| `patch_capacity` | 384 | Initial variable-program patch space; grows if needed |
| `worker_count` | 2 | Native pipeline module-loading workers |
| `cubin_slots_per_worker` | 2 | Native pipeline slots |
| `max_chunk_configurations` | automatic, ceiling 16,777,216 | Omit to fit compatible pages to memory and measured tile cost; explicit values retain a fixed cap |
| `target_tile_seconds` | 0.05 | Advisory auto-tile wall-time target; positive and at most 1 second |
| `max_device_bytes` | 268,435,456 | Score/coefficient/reducer buffers per tile |
| `max_bank_bytes` | 268,435,456 | Resident grid/Philox pool cache budget |
| `max_host_bytes` | 2,147,483,648 | Shared job reservation across all GPU workers: input, arenas, retained records, tile scratch and pipeline workspaces |
| `dedup_bytes_per_family` | 67,108,864 | Exact native-program key storage |
| `max_seconds` | 600 | Deadline after dequeuing, including preparation |
| `profile_timing` | false | Optional per-launch CUDA event profiling |

Numeric pools are trimmed to a smaller request budget before cache lookup, including
jobs with no numeric slots. Unpooled tile buffers are trimmed when retained
capacity would exceed a new tile budget; replacement releases fenced old storage
before allocating larger storage. Fixed startup slabs retain their advertised
capacity and are separate from the active tile-byte limit. Family dedup
also needs its index table; page sizes are conservatively measured. Generation
arenas and provenance archives are released at completion; only prepared data,
compact winners and the copied request remain with the job. The service limits retained job/problem handles to 32 each, unfinished jobs to 8,
and accounted request copies to 256 MiB by default. Release/expiration makes room
without restarting. Admission errors leave accepted jobs unchanged.

Host reservations happen before allocation under one shared lock, including
allocation headers and temporary overlap during resizing. Borrowed trajectory
payload and result scratch are charged for the bytes used by a job; their whole
startup capacities are reported separately. A failed reservation stops work with
an explicit error and releases safely fenced work. It does not silently reduce
retention or the requested search allocation. Report serialization now uses no
temporary candidate-pointer vector. Caller-owned output buffers are outside this
ledger; the C service reserves its bounded report storage at startup.

`odz_runtime_host_limit(runtime, bytes)` sets an operator ceiling before any job.
A larger requested `max_host_bytes` is rejected, with zero scored configurations.
The direct API has no operator ceiling unless configured. The private C worker
sets 2 GiB by default. `memory.classes` reports live/peak reservations;
`charged_peak_bytes` is the simultaneous shared peak, not the sum of class peaks.
Device reports distinguish this runtime's explicit allocations and numeric pools
from sampled device-wide usage, which includes other processes. Process RSS and
its sampled job peak are separate from the ledger; process lifetime RSS peak can
include earlier jobs. Samples marked unavailable must not be interpreted as zero.

Opaque core handles, compiler internals, loaded modules, driver and allocator
overhead are not charged to `max_host_bytes`. They are covered by process/device
telemetry and the service's external RSS guard. This is not a hard total-VRAM or
whole-process allocation limit. See the [memory contract and validation](../docs/grammar/C99_MEMORY_ACCOUNTING.md).

`NativeService` exposes `max_jobs`, `max_pending`, `max_request_bytes` and
`completed_ttl` constructor settings. `SupervisedNativeService` additionally checks
worker RSS (4 GiB default), combined parent/worker RSS (8 GiB), heartbeat (30 seconds) and job elapsed time (requested
limit plus 10 seconds cleanup grace). A breach kills only its worker; the next
submission creates a fresh worker. Lost jobs return `worker_lost`, never success. Failure tombstones expire and are bounded by `max_jobs`.
Process RSS includes CUDA driver/host buffers but not GPU VRAM. These are sampled
limits, so transient overshoot is possible. macOS reports peak RSS; execution and
supervision are validated on Linux/CUDA. In-process embedding has no kill/restart
supervisor and must provide its own process boundary for unfenced CUDA failures.

The MCP limit is **16 MiB per UTF-8 message** including its envelope; the direct C
request limit is **64 MiB**. An oversized MCP message is drained and rejected while
the connection stays usable. Tool errors include `unknown_handle`, `admission_limit`,
`job_active`, `invalid_request`, `worker_lost` or `execution_error`.

Auto sizing changes tile boundaries and partial coverage at a deadline, not the
requested population or full-run indices/RNG. Its 50 ms target is advisory: initial
compilation, a costly first tile, one indivisible toggle product, and distinct
winner reduction can exceed it. Explicit `max_chunk_configurations` overrides
adaptation. Under tight memory/time limits, automatic mode reduces the AST group
first to preserve numeric-bank work per module load; only then splits a single
system's bank when necessary. AST pages and module packing now have measured automatic defaults; explicit
`batch_variants` and `module_systems` remain overrides. See the
[timing/sizing contract](../benchmarks/request_sizing/README.md). `execution.devices` reports per-GPU work, largest tile and busy/wait
time. Multi-GPU stage times sum overlapping worker durations and can exceed wall
time; do not add them to obtain elapsed time.

The `parallel-work-v1` policy lives in the pure C host module
[`odz_sizing.c`](odz_sizing.c), outside the kernel core. The effective tile ceiling
is `max(67108864 / max(RK4_steps, points), nextpow2(2*SMs*threads_per_SM))`, further
limited by the request and available memory. Hardware parallel capacity is queried
at context creation; the parallel target cannot exceed 4,194,304 configurations.
At 5,120 steps, a 5080 can now use 262,144 configurations instead of 13,107.
Automatic bank slices are balanced, so a bank does not leave a tiny final tile.
No rows are added or omitted; global bank and toggle indices remain unchanged.
The cost estimate resets for each job and family/numeric layout and ignores
undersized tails and template growth. It measures prepared scoring calls and
cannot shrink automatic work below the parallel target unless a hard limit or
available work requires it. Thus the duration target can be exceeded.

`execution.devices[].sizing` reports hardware capacity, the effective work ceiling,
tile configurations/ASTs/block geometry, below-one-wave tiles and configurations,
memory/explicit/available-work limiting counts, balanced-bank counts, and maximum
plus histogram p50/p95 upper bounds for scoring-call wall time. Limiter counts can
overlap. Total blocks across modules are **not measured occupancy** or a promise of
simultaneous execution. GPU event timing remains separately opt-in; the controller
does not force profiling on. Small finite families can lack enough useful work;
the scheduler reports that scarcity rather than padding evaluations or changing
family budgets. Tests: `make -C runtime test-sizing-cpu` and
`tests/runtime/tile_sizing.py --library NEW --baseline OLD --output REPORT`.

Workers still lease whole producer pages. A single AST with a large numeric bank
currently executes on one GPU even in a two-GPU runtime; multiple available pages
permit both GPUs to work. Splitting one page's numeric ranges across devices is
separate scheduling work, and this policy does not claim to implement it. The
per-device configuration and timing counters expose that case. The user's
deferred overlap refactor remains deferred.

Numeric pools evict least-recently-used completed bindings on both byte-budget
and cache-entry pressure. The current binding's pools are protected. Reports
include `cache.numeric_pool_evictions`; an active binding that cannot fit still
fails explicitly.

A single toggle product must fit a tile. Per-family budgets, not traversal order,
determine admitted work. Optional per-skeleton configuration allowances are supported; configurable
weighted redistribution remains future scheduler work.

## Results and reproducibility

Reports contain family allocation/coverage, valid/invalid counts, generation stop
reasons, timing scopes, cache activity, and candidates once with leaderboard ID
references. Timings overlap: generation runs concurrently, and NVRTC is inside
template time. Pipeline time includes specialization/loading/execution, not just
GPU arithmetic. Runtime initialization is reported separately by the transport.

Each retained address is `(family_index, ast_index, bank_index, permutation)`.
AST index is the family's accepted, deduplicated index; derivation and variant
indices are also retained. `slots` records each axis index and actual Philox
seed/stream. `values` and resolved native programs preserve the winning FP32
coefficients; nine significant decimal digits round-trip to the exact FP32 bits.
`value_bits` additionally records the exact hex bits. If an extreme transform
overflows a coefficient, its JSON value is null and the bits remain authoritative;
the resolved bytecode still describes what was scored.
Replay resolves retained addresses without regenerating the population. Optional
CPU replay returns state and trajectory MSE breakdowns for that winner.

The C structural sampler uses
`odezza.c99-grammar.splitmix64-fisher-yates.v1`. Numeric banks use
`odezza.native.philox4x32-10.stream-v1`: the declared seed plus a named stream
address derived in `odz_bank_address`; skeleton-scoped streams include family and
derivation. These profiles **differ from the old Python sampler/address hashing**.
The same JSON seed does not imply the same sampled population across backends.
Uniform samples match the core CPU Philox primitive; normal values use GPU
Box–Muller. Exact gathered snapshots are authoritative for portable replay.

`evaluation_row` retains separate evaluation addresses, including repeated
expressions/constants. For example:

```json
"retain": {
  "global": {"k": 16, "unit": "evaluation_row"},
  "per_family": {"k": 16, "unit": "evaluation_row"}
}
```

This selects raw top-16 globally and per family. Every intermediate reduction
retains enough rows for all downstream requests; no expression-uniqueness rescans
are needed. Counts still cover all evaluated rows, including invalid scores.
Ties use `(mse, family_index, ast_index, bank_index, permutation)`. The core keeps
original flat indices, so AST/toggle/bank/RNG reconstruction remains exact.

Raw and distinct units may coexist. They have separate finalist sets and late-tag
archives: a duplicate expression does not erase a requested raw evaluation row.
Omitting `unit`, including bare integer retention, keeps the existing
`resolved_structure` default. Selecting raw rows is an explicit API choice.

The `reduction` report identifies `cub_block_hierarchy`, total reducer calls,
`continuation_passes`, CUDA-event `kernel_seconds`, and local row/distinct k.
A reducer call may use several hierarchy kernels. `runtime_initialization_seconds`
is the reusable reducer's one-time NVRTC/module/resource setup, summed across
workers; it is repeated as attribution in reports, not incurred on each request.
It is separate from the existing auxiliary numeric-kernel NVRTC measurement.

`numeric_candidate` uniqueness compares full resolved program bytes;
`resolved_structure` canonicalizes coefficient slot numbering and state choices.
`variant` currently means family plus accepted AST, so identical structures in two
families remain two variant entries. Retention `k` is limited to 256, with at most
64 requested tag rankings. All emitted tag names are retained as metadata. Late
aliases are merged without re-evaluating duplicate ASTs. Requested tag rankings
also keep bounded per-AST/permutation finalist archives until generation finishes;
that memory can be substantial and is charged to the job retention budget.

If cancellation/deadline interrupts extra distinct-winner reduction passes,
`retention_complete` is false: scores already evaluated count toward work, but
leaderboards may be incomplete. Partial/failed jobs never imply complete coverage.

Current scope: 1–8 selected GPUs per runtime, scoring only, in-memory forward producer
cursors. No automatic search/LM, arbitrary seek/checkpoint restore, server-side
trajectory views or GPU per-trajectory loss output. GPU groups cannot concurrently share one runtime; a service queues jobs serially. The previous backend retains its existing fitting path.

## Validation

```sh
make test-frontend test-grammar
make test-request-runtime
CUDA_MODULE_LOADING=EAGER PYTHONPATH=python python3 tests/runtime/benchmark.py \
  --root /tmp/odezza-benchmark-cache --output /tmp/runtime-benchmark.json --legacy
```

The benchmark uses explicit identical grids for cross-backend comparisons. The
supplied RNG workload is reported separately because the numeric profiles differ.
See [validation and deviation record](../docs/grammar/C99_RUNTIME_VALIDATION.md).

`make test-request-robustness` runs the broader synthetic grammar campaign,
including RNG reconstruction, coupled recovery calibration, supplied grammars,
large ASTs, cache pressure and rejection cases. See the
[robustness report](../docs/grammar/C99_ROBUSTNESS.md) for coverage and limits.

## Source packaging and release checks

`python3 runtime/package.py --output /tmp/odezza-native-trial.tar.gz` makes a
reproducible source snapshot with per-file SHA-256 in `SOURCE_MANIFEST.json`.
It includes untracked working-tree sources, C/Make build files, Python transport,
examples and tests; it excludes compiled binaries and experiment artifacts.
Build the extracted project with the existing CUDA toolchain using
`make request-runtime`; no git, package download or dependency installation is
needed. This is a reviewable trial snapshot, not a published or tagged release.

`make test-request-service` exercises release/expiration, admission, small status,
transport errors, a 1,000-job release soak, cold shared cache, exact single/dual GPU
rankings, matched 2.048B-configuration timing and controlled worker failure/restart.
It requires two GPUs. The source snapshot is validated on rack1's dual RTX 5080
with its installed CUDA toolkit; wider CUDA/GPU compatibility remains to be tested.

The [service and multi-GPU report](../docs/grammar/C99_SERVICE_AND_BATCHING.md)
records matched throughput, the low-memory scheduling repair and remaining limits.

## Attempt-owned numeric inputs (separate service trial)

Call `odz_runtime_attempt_inputs(runtime)` on the owner thread before its first
run to invalidate numeric content identities at every attempt boundary across all
devices. Allocation capacity can persist, but explicit grids are uploaded and
Philox pools generated again for the next job. The native report records
`numeric_input_scope: "attempt"`; the compatibility default is `"runtime_cache"`.
This switch does not implement fixed startup reservations or persistent scoring
sessions/streams. Those are separate lifecycle work.

Failed input uploads and uncertain CUDA ownership quarantine the runtime. Its
subsequent jobs are rejected without scoring; replace the worker process before
continuing. The [service trial](../service_trial/README.md) includes actual
fault-injection and recovery checks. C cancellation now publishes a small signal
under the snapshot mutex, so it need not wait for JSON/grammar preparation's
main job mutex. Execution observes that signal at its normal stop boundaries.


## Compiled-template cache and bounded retention

The runtime uses a host-local `templates.sqlite3` database for compiled
scoring templates and numeric-helper CUBINs. Its only owner is C. SQLite is a
vendored SQLite 3.53.4 dependency outside `core/`, with pinned source and checksums. The build has no
implicit download or system-library fallback. No trajectories, constants,
grammars, specialized customer RHS, results or jobs are persisted in this DB.

`ODEZZA_TEMPLATE_CACHE_BYTES` sets the artifact-payload budget (default 256 MiB;
1 MiB–8 GiB). On insertion/open, least-recently-used entries are removed until
payloads fit. Use timestamps have one-second resolution. The SQLite page ceiling
is separate: at most approximately twice the payload budget plus 16 MiB; WAL
can exceed its journal-size target while readers hold snapshots. FULL auto-vacuum
and checkpointing reclaim storage; the payload budget is not a hard total-directory
quota. A 200-write eviction test checks bounded WAL growth under the cache’s
normal short-reader behavior; externally held snapshots can delay reclamation. Legacy `.bin` files are ignored and
must be archived/removed during migration; there is no automatic trust/import
of their missing compiler-version identity. A cache checksum/source mismatch
causes a miss and recompile. Database errors fail explicitly. A scoring artifact
larger than the payload budget can execute uncached and is counted in
`template_cache.oversized_uncached`; repeated requests then compile again.
Numeric-helper artifacts must fit. Schema versions are checked before use.

The source/profile/shape validation remains in the existing template reader.
Fixed RHS prespecialization remains at pipeline creation; per-candidate workers
copy that prepared template before patching variable RHS. SQLite is not accessed
per configuration or CUDA launch. Sixty-four striped host-local file locks
coalesce concurrent compile misses without holding a SQLite write transaction
through NVRTC. Unrelated keys in the same stripe may serialize cold compilation.

`execution.max_report_bytes` optionally lowers the native 8 MiB report ceiling
(minimum 4096). After parsing, before AST generation/scoring, C computes a
conservative report bound from all requested leaderboards, maximum program size,
slot metadata, possible tags and trajectory metadata. It reports
`delivery.conservative_bound_bytes`. Exceeding the budget produces
`report_size_limit` with zero scored configurations. This can reject a request
whose eventual results would have been smaller; reduce top-k, tag output or the
grammar's `max_nodes`. No requested winners/tags are silently truncated.

Retained configurations share slot metadata and normalized structure by
AST/permutation. Tag provenance is shared by AST, including aliases discovered
before or after scoring. Raw/distinct late-tag archives retain only the depth
required by tag leaderboards. They remain bounded by the existing host budget,
and still grow with the searched population when late tags are possible.
Report identity deduplication uses a preflight-reserved hash table, avoiding the
former quadratic scan and allocations during report construction.

See [CPU/GPU validation and scope](../benchmarks/cache_retention/README.md).
