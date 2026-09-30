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

# Odezza

> **Collection category:** Current native execution engines. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

Odezza is a C99 GPU execution engine for scoring families of ordinary differential equation systems by integrating complete trajectories. One compiled kernel packs multiple candidate systems, and each candidate can expose compact bit-toggle and constant-bank permutations. This supplies enough useful work to amortize specialization, CUDA module loading, and short trajectory kernels.

> The native runtime now shares retained metadata, checks report size before
> scoring, and uses a C-owned SQLite compiled-template cache. See the
> [validation record](benchmarks/cache_retention/README.md).

## Current JSON service: request limits and failure behavior

This section describes the **mac3 → rack1 C service trial as of 2026-09-12**.
Send one JSON object containing `problem`, `grammar`, and optional `execution`;
poll the returned handle for status/results. The [service README](service_trial/README.md)
has client commands. These service limits are separate from the lower-level
[core API](core/README.md) and the older Python/stdio service's limits.

This is a trusted private trial, not yet a paid multi-tenant MCP endpoint.
[Commercial readiness work](scratch/structural_search_trial/TODO.md#paid-mcp-scope-clarification--2026-09-12)
explicitly tracks customer authorization/isolation, usage settlement, reliable
delivery and the remote MCP contract in addition to execution improvements.

**Queue acceptance is not full validation.** mac3 checks the envelope, deadline,
trajectory data and integration work before publishing the job. The GPU worker
then validates the complete grammar, execution options, memory requirements and
kernel compatibility. A later failure may therefore have an attempt number and
a valid partial prefix of evaluated work. Always inspect status, error, family
coverage and `retention_complete`; a returned handle does not promise a complete
search or a successful fit.

### Admission, transport and rollout ceilings

| Constraint | Current limit / behavior | Where it takes effect |
|---|---|---|
| JSON request body | 8 MiB; transfer chunks at most 256 KiB | Transport, before execution |
| Occupied job/result slots | 8, including completed results until release/expiry | A further submission receives `admission_full` |
| Distinct admitted job IDs | 4,096 per service generation; released IDs cannot be reused | `generation_id_limit` / `released_job` |
| Execution deadline | Default 600 s; effective value is the smaller of `execution.max_seconds` and a supplied `grammar.limits.max_seconds`; must be positive and at most 3,600 s | Admission; execution includes preparation |
| Total trajectory points | 65,536, including every initial point | Admission and native runtime |
| RK4 subdivisions per interval | 4,096 | Admission and native runtime |
| RK4 steps per configuration | 65,536, summed over all trajectories; a smaller `grammar.integration.max_steps` also applies | Admission and native runtime |
| Work per execution tile | Device-aware: `max(67,108,864 / max(R,P), parallel_target)` configurations; see below | Further bounded by explicit configuration and memory limits; rejects an indivisible toggle product that cannot fit |
| Native result report | 8 MiB; `execution.max_report_bytes` can lower it to 4,096–8,388,608 bytes | Worker checks a conservative bound after parsing and before generation/scoring; excess returns `report_size_limit` with zero evaluations. No winners/tags are silently truncated |

Integration counts use the actual FP32 layout:

```text
P = total points across trajectories
I = sum(points_in_trajectory - 1)
S = max(1, ceil(largest FP32 time gap / FP32 integration.dt))
R = S * I
tile_work = configurations_in_tile * max(R, P)
parallel_target = next_power_of_two(2 * GPU_SM_count * max_threads_per_SM)
```

The parallel target is bounded at 4,194,304 configurations. It permits enough
independent work for long trajectories without changing their integration steps.
On an RTX 5080 the target is 262,144. The former fixed aggregate cap allowed only
13,107 configurations at 5,120 RK4 steps, which severely underfilled the GPU.
The new effective work ceiling applies to both automatic and explicit tiles;
memory limits and a smaller `max_chunk_configurations` still win. This is a hard
work envelope, **not a wall-time guarantee**. The 50 ms automatic duration target
is advisory and yields to the parallel target. Cancellation waits for submitted
GPU work to finish. See `execution.devices[].sizing` for the effective ceiling,
underfilled work, memory/explicit/available-work limits, and scoring-call duration
statistics. Block totals span modules and do not establish achieved occupancy.

The same `S` is used for **every** interval, including shorter irregular gaps.
Masking observations does not remove intervals or state integration. The tile
cap also applies to explicit `execution.max_chunk_configurations`. The runtime
does not enlarge `dt`, discard trajectories, or omit configurations to fit it.
For example, 1,024 configurations with 65,536 steps each fill the work ceiling;
one configuration exceeding 65,536 steps is rejected rather than tiled in time.

`health.integration_limits` advertises the rollout ceilings; job status includes
`integration_work`, and native reports include `integration_limits`. Excessive
rollout work produces a terminal failed admission report with
`integration_work_limit` and zero attempts/evaluations. Invalid trajectory data
uses `invalid_trajectory_or_integration`; malformed envelopes or excessive
deadlines use `invalid_envelope_or_budget_above_3600s`. Counts that overflow their
representation are explicitly marked unavailable.

These limits bound work, **not kernel duration in seconds**. State count, AST
complexity and GPU contention still matter. Cancellation prevents new work at
safe boundaries; it cannot preempt a running CUDA kernel. Upload/queue waiting
expires after 600 s, terminal results after one hour, and lost workers have at
most three claims per job. Keep a copy of results: jobs are held in RAM and a
broker/orchestrator restart invalidates the service generation.

### Supported input and structural limits

| Area | Accepted contract / reasons for failure |
|---|---|
| JSON | Valid UTF-8 JSON, no duplicate keys or embedded NUL, bounded nesting (128); unknown schema fields are rejected |
| State definitions | 1–126 unique identifier names (up to 127 bytes; `t` is reserved); names/order must agree between problem and grammar. This is an encoding ceiling, not a promise every 126-state kernel fits |
| Trajectories | Nonempty `times`, a `values` row per time with one cell per state, and a separate complete finite `initial` vector for every trajectory |
| Times and observations | Finite after FP32 conversion; times strictly increasing after conversion. Initial observations must agree with `initial`. Later `null` or boolean masks may hide states; at least one noninitial observed scalar is required overall |
| Solver | Positive finite FP32 `integration.dt`, fixed-step `rk4` only. Other methods, `stiff: true`, `rtol` or `atol` are unsupported. No automatic FP64, adaptive or stiff fallback |
| Fixed RHS | Expression strings shorter than 32,768 bytes, using known state names, numeric literals and supported scoring operators. The JSON fixed section does not bind grammar constants or toggles |
| Variable RHS | Version-1 scoring grammar (`kind: "compile"` when supplied); provide each nonfixed RHS and omit fixed ones. Unknown rules/operators/states, explicit time dependence and incomplete/conflicting RHS assignments fail |
| Expression syntax | Production/RHS text must be shorter than 32,768 bytes and fit parser scratch/nesting bounds. Arithmetic, `sqrt`, `abs`, `sin`, `cos`, `tanh`, `exp`, `log` and literal integer powers in [-16, 16] are supported; arbitrary powers, conditionals and user code are not |
| Expansion settings | `max_nodes`: 1–4,096 (default 63); `max_depth`: 1–128 (default 12); `max_expansion_depth`: 1–64 (default 20). Out-of-range settings fail; trees exceeding valid expansion bounds are pruned during production |
| Native inputs | At most 32 toggle bits; active coefficient slots plus `2 * state_count` must be at most 253. Actual register/specialization requirements can fail below these encoding ceilings |
| Toggles | Grammar state-leaf choices have arity 2 or 4, with valid state groups. One numeric row's complete toggle product must fit the tile work, configuration and memory limits; partial toggle products are not executed |
| Constants and RNG | Finite fixed/grid values; nonempty grids and positive bank counts. Bases are `uniform01` or `normal01`, with supported transforms. Shared numeric axes require equal counts. Transform references must resolve to constants; uniform bounds must be ordered, log-uniform endpoints positive, normal standard deviations nonnegative |
| Cartesian indexing | Allocation counts and products must fit their native integer representations; overflow is an error, even if a requested execution prefix would be small |
| Retention | `k` is 0–256 for global/family/tag rankings; at most 64 requested tag rankings. Units are `evaluation_row`, `numeric_candidate`, `resolved_structure`, or `variant`. Raw-row rankings use CUB top-k; omitted units retain the existing `resolved_structure` default. The additional conservative report-size preflight must also pass |

See [the grammar reference](docs/grammar/README.md) for supported operators and
syntax. A mathematically invalid generated candidate (for example division by
zero, a nonfinite rollout, or FP32 MSE overflow) counts as invalid evaluated work;
it does not by itself reject the entire request. Grammar/schema/resource errors
are different and can fail the job.

### Memory and kernel fit

Worker startup currently reserves **64 MiB of CPU trajectory storage** and,
per GPU, **64 MiB trajectory storage, 256 MiB scoring/coefficient/reduction
storage, and 64 MiB CPU result scratch**. Trajectory overflow fails explicitly;
configuration tiles shrink to fit the scoring pools. A single row that still
cannot fit fails. Raising a JSON memory budget does not enlarge these startup
pools. Grammar/dedup/retention arenas, numeric banks and pipeline workspaces have
additional costs; the current accounting is not a complete RSS/VRAM guarantee.

The scoring kernel also stages the trajectory in **shared memory per CTA**:

```text
shared_bytes = 4 * ((state_count + 1) * P + trajectory_count + 1)
```

This must fit the selected GPU/kernel's shared-memory allowance. The 65,536-point
safety ceiling is therefore not a supported trajectory size for every device or
state count. This device-specific failure currently occurs on the worker; mac3
does not yet perform capability-aware admission. Sending fewer configurations
does not reduce the shared-memory requirement for that trajectory.

Larger ASTs can exhaust patch space or temporary registers. The bridge grows
patch capacities and caches the larger template (up to its 65,536 patch-capacity
ceiling), but this does not increase available registers or change solver shape.
NVRTC, inspection, specialization, module-load and CUDA failures are reported;
syntactically valid grammar is not a guarantee of kernel fit.

Execution options are positive integers unless stated otherwise:

| `execution` field | Default | Maximum accepted value |
|---|---:|---:|
| `batch_variants` | automatic, at most 1,024 | 4,096 |
| `module_systems` | automatic, at most 64 | 256 |
| `patch_capacity` | 384 | 4,096 initial capacity; later bridge growth is separate |
| `worker_count` | 2 | 32 |
| `cubin_slots_per_worker` | 2 | 32 |
| `max_chunk_configurations` | Automatic, ceiling 16,777,216 | Explicit cap 4,294,967,296; further reduced by work/pool limits |
| `profile_timing` | false | Boolean; optional CUDA profiling adds overhead |
| `target_tile_seconds` | 0.05 s | Positive float at most 1 s; advisory auto-sizing target |
| `max_device_bytes` | 256 MiB | 8 GiB; startup pools still apply |
| `max_bank_bytes` | 256 MiB | 2 GiB |
| `max_host_bytes` | 2 GiB | Parser maximum 16 GiB; current worker operator ceiling 2 GiB |
| `dedup_bytes_per_family` | 64 MiB | 2 GiB, plus index-table overhead |

Values inside these ranges can still be mutually incompatible or exceed physical
resources. Dedup/retention exhaustion fails with coverage of any completed prefix;
it does not silently disable deduplication or drop requested tag rankings. Explicit
small time/memory budgets can prevent completion of otherwise valid work.

The current C worker shares `max_host_bytes` across all GPUs processing a job.
It charges request data, parsed/producer arenas, retained records, tile scratch
and caller-provided pipeline workspaces before using them. A request above the
operator ceiling fails before scoring; exhausting its own budget can return a
failed report with completed-prefix coverage. The supervisor also checks worker
RSS every 100 ms and replaces a worker exceeding 4 GiB by default. RSS includes
compiler/driver overhead outside the exact job ledger, and this sampled guard
allows transient overshoot. Both limits are operator-configurable. Startup slabs
keep their declared sizes; smaller requests trim reusable numeric/unpooled tile
capacity. These are separate named memory limits, not a hard total-VRAM limit.
See [memory policy, report fields and verification](docs/grammar/C99_MEMORY_ACCOUNTING.md).

### Search budgets: stopping is different from rejection

These existing grammar limits allocate search effort. They are **not fixed
service-wide admission ceilings**:

| `grammar.limits` field | Default total when allocations are omitted | What is counted |
|---|---:|---|
| `max_skeletons` | 10,000 | Grammar derivations that contribute accepted work; not a global algebraically distinct AST count |
| `max_variants` | 100,000 | Accepted native candidates after family-local exact deduplication |
| `max_configurations` | 1,000,000,000 | Reserved candidate × numeric-bank-row × toggle-permutation evaluations |
| `max_derivations` | 1,000,000 | Attempted derivations, including unsuccessful/pruned work |
| `max_expansion_steps` | 100,000,000 | Grammar expansion work |

Each limit may also be allocated explicitly under `families[].limits`. For each
field, explicit family allocations are reserved first; a supplied request total
divides its remainder among unspecified families. Fully explicit family allocations
can derive the total without a request total. Mixed explicit/implicit allocations
need a request total. Overflow, conflicting totals, nonpositive limits or a zero
allocation for a family are rejected. With neither explicit form, the default
total is divided among families. Unused allocations stay unused; weighted
redistribution is not supported. Families execute in bounded interleaved batches.

Hitting an allocation stops that family's generation and is reported as a stop
reason; it is not an invalid-request error. A huge variant can receive a prefix
of numeric rows, always containing the complete toggle product per row. Leftover
budget smaller than one such row remains unused. Budgets are not successful-score
counts: invalid evaluations consume work, and a timeout can leave reserved work
unevaluated. Retention `k` controls returned winners, not search effort.

An optional `max_configurations_per_skeleton` in request or family `limits`
reserves at most that much configuration work for a structural derivation across
its native variants. A family value overrides the request value; omission adds no
cap. It keeps a prefix of numeric rows with complete toggle products. Remaining
variants of that skeleton are skipped after its allowance, then generation moves
to the next skeleton. A cap smaller than one toggle product skips that skeleton.
Reports expose the cap and `configuration_limited_derivations`; this is an
attempted-derivation count, not a distinct-AST count. Invalid GPU evaluations still
consume their allocated work. Configuration addresses and RNG pools are unchanged.

### Aggregate service admission

mac3 sums the same reserved family budgets as the C frontend **before publishing
the job**. It does not enumerate or compile expression ASTs. Operator defaults:

| Reserved resource per request | Service ceiling | Environment suffix |
|---|---:|---|
| Families | 64 | `FAMILIES` |
| Skeletons | 100,000,000 | `SKELETONS` |
| Native variants | 10,000,000 | `VARIANTS` |
| Configurations | 1,000,000,000,000 | `CONFIGURATIONS` |
| Derivations | 1,000,000,000 | `DERIVATIONS` |
| Expansion steps | 10,000,000,000 | `EXPANSION_STEPS` |
| Configurations × max(RK4 steps, trajectory points) | 1,000,000,000,000,000 | `ROLLOUT_WORK` |
| Requested shared host reservation | 2 GiB | `HOST_BYTES` |

Override with `ODZ_ADMISSION_MAX_<suffix>` in the orchestrator environment;
invalid/nonpositive/overflowing settings prevent startup. Worker pool and host
ceilings apply independently: raising an admission limit does not enlarge them.
`health.admission_limits` publishes effective settings; job status `allocation`
and rejected reports give reserved totals. Rejections have zero attempts and
zero completed configurations, with `invalid_search_allocation`,
`search_<resource>_limit`, `rollout_work_limit/overflow`, or a host-reservation
reason. No later family is silently truncated to make the request fit.

Admission counts declared reservations, not an inferred finite grammar size.
Unused family budgets remain unused. The host preflight rejects requests whose
JSON bytes plus family dedup arenas already exceed their reservation. This is
explicitly a **lower bound**: the worker still measures parsed grammar, producer,
page, retention and pipeline storage before scoring. Admission is not a promise
that every grammar fits, or a kernel-duration guarantee. The existing integration,
register, patch, shared-memory, pool and report limits still apply.

**1M variants × 2,048 configurations = 2.048B evaluations** remains supported.
For timing definitions and measured automatic sizing, see
[request sizing](benchmarks/request_sizing/README.md). Explicit execution settings
are preserved; `execution.profile_timing: true` adds optional CUDA event timing.
[Remaining work](scratch/structural_search_trial/TODO.md#remaining-grammar-execution-work-2026-09-12).

## Public boundary

[`core/odezza.h`](core/odezza.h) is the core's only public header (`src/` is a compatibility path). A caller provides:

- an explicit compilation target such as `sm_89` or `sm_120`;
- active state/constant/toggle counts within reusable compiled capacities and any right-hand sides shared by every candidate;
- compact postorder ASTs for the remaining right-hand sides;
- resident device buffers for constants, reference trajectories, and MSE output;
- one reusable host workspace sized by the handle.

The public API does not expose CUDA-source generation, NVRTC programs, CUBIN inspection, SASS patch sites, module tickets, or owned streams. A launch may supply an input-ready CUDA event to establish upload dependencies without a host wait. With the intended context current, handle creation performs generation, native CUBIN compilation, physical inspection, fixed-RHS specialization, CPU-worker creation, and one-time stream/event creation. A bulk run only divides candidate systems into packed modules, specializes worker-owned CUBIN slots in the caller's workspace, loads and launches them, waits for completion, and unloads them.

CUDA and NVRTC are required build dependencies. Odezza links directly to the CUDA Driver API and NVRTC; it has no no-CUDA backend, runtime `dlopen` probe, or source-build fallback.

## Build

On a machine with an installed CUDA toolkit:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Configuration fails when CUDA or NVRTC is unavailable. This is intentional. It keeps one supported runtime rather than a library that compiles successfully but cannot perform its core operation.

Tests also require Python 3. The CMake suite includes CUDA scoring regressions and independent checks of generated C99 manifests and hashes. With the Makefile, `make test` checks host code and artifacts; `make test-cuda` runs the scoring data and shape regressions on the current GPU.

## Lifecycle

CUDA's eager module-loading policy must be selected before the higher layer initializes CUDA. That layer makes the intended context current before handle creation. The resulting pipeline owns streams and events from that context, so the same context must be current for every run and for destruction and must outlive the handle. Odezza never creates, obtains, stores, pushes, pops, or sets a context.

```c
#include "odezza.h"
#include <assert.h>
#include <stdlib.h>

#ifdef ODEZZA_DEBUG
#define ODEZZA_TRY(expression) \
    do { \
        OdezzaResult result_ = (expression); \
        assert(result_ == ODEZZA_SUCCESS); \
        if (result_ != ODEZZA_SUCCESS) return result_; \
    } while (0)
#else
#define ODEZZA_TRY(expression) \
    do { \
        OdezzaResult result_ = (expression); \
        if (result_ != ODEZZA_SUCCESS) return result_; \
    } while (0)
#endif

static OdezzaResult run_scoring(
    const OdezzaScoringPipelineCreateInfo *create_info,
    const OdezzaScoringSystem *systems,
    size_t system_count,
    const OdezzaScoringLaunch *launch,
    OdezzaScoringPipeline **pipeline_ret,
    void **workspace_ret,
    OdezzaScoringRunReport *report_ret
) {
    size_t workspace_size;
    size_t workspace_alignment;
    ODEZZA_TRY(odezza_scoring_pipeline_create(create_info, pipeline_ret));
    ODEZZA_TRY(odezza_scoring_pipeline_workspace_requirements(*pipeline_ret, &workspace_size, &workspace_alignment));
    *workspace_ret = aligned_caller_allocation(workspace_alignment, workspace_size);
    if (*workspace_ret == NULL) return ODEZZA_ERROR_ALLOCATION;
    return odezza_scoring_pipeline_run(*pipeline_ret, systems, system_count, launch, *workspace_ret, workspace_size, report_ret);
}

OdezzaScoringPipeline *pipeline = NULL;
void *workspace = NULL;
context_owner_make_compatible_context_current(create_info.sm_version);
OdezzaResult result = run_scoring(&create_info, systems, system_count, &launch, &pipeline, &workspace, &report);
if (result != ODEZZA_SUCCESS && pipeline != NULL) {
    size_t error_bytes;
    char error[512];
    if (odezza_scoring_pipeline_write_error(pipeline, error, sizeof(error), &error_bytes) == ODEZZA_SUCCESS) {
        report_error(error);
    }
}
if (workspace != NULL) free(workspace);
if (pipeline != NULL) (void)odezza_scoring_pipeline_destroy(pipeline);
context_owner_release_current_context();
```

The handle owns its generated source, immutable prespecialized CUBIN, inspection metadata, worker threads, bounded host queues, streams, and events. The run workspace remains caller-owned and may be reused for subsequent runs on the same handle. A handle is bound to one caller-owned context and serves only one synchronous run at a time. Concurrent multi-device use therefore creates one pipeline and workspace per device while each device's context is current.

Creation may return a failed diagnostic handle. This lets the caller retrieve an NVRTC, inspection, specialization, CUDA-resource, or host-resource error through `odezza_scoring_pipeline_write_error`; the caller must then destroy that handle with the creation context current. Runtime context/device mistakes are reported from the first failing CUDA operation.

## Allocation and queue behavior

Creation may allocate because it constructs long-lived state and persistent CUDA resources. `odezza_scoring_pipeline_run` performs no Odezza allocation and creates no CUDA resources. The queried workspace contains an exact mutable CUBIN copy and specialization arena for every worker slot. Internal queues carry indices, not CUBIN bytes. A slot is reusable immediately after `cuModuleLoadData` consumes its image, while the resulting module can remain in flight on a retained internal stream.

The current measured knee for short scoring modules has been one specialization worker, two slots, two streams, and two simultaneously loaded modules. The public API exposes worker/slot depth because it sets workspace size; stream, event, queue, and resident-module policy are internal. Packing and CUBIN size can move the knee, so the private diagnostic API retains overrides for controlled sweeps. See [`docs/c99_scoring_pipeline.md`](docs/c99_scoring_pipeline.md) and the retained reports under [`benchmarks/`](benchmarks/).

## AST encoding

`OdezzaAstProgram` is a borrowed byte span using `odezza-postorder-f32-v2`:

| Instruction | Bytes after opcode | Stack behavior |
|---|---:|---:|
| state leaf | `u8 state_index` | push |
| indexed constant | `u8 constant_index` | push |
| literal | four little-endian IEEE-754 FP32 bytes | push |
| two-way toggle | `u8 toggle_bit_index` | pop 2, push 1 |
| four-way toggle | two distinct `u8` bit indices | pop 4, push 1 |
| unary/binary/FMA operator | none | operator arity |
| return | none | require exactly one value |

A toggle selects only direct state, indexed-constant, or literal leaves. The same bit may control multiple toggle sites across one or more right-hand sides, which expresses a consistent structural choice. Fixed RHS programs may use states, indexed constants, literals, and operators, but cannot contain toggles.

A state-leaf index and `OdezzaScoringRhs.state_index` identify the same positional state. State names do not enter generated source, template identity, inspection, or specialization. Fixed RHS indices must be unique. Every candidate system supplies every complementary active RHS exactly once and cannot overwrite a fixed derivative.

## Device data layout

Trajectories are supplied at run time as concatenated points. `trajectory_offsets[trajectory_count + 1]` partitions them, `trajectory_times[trajectory_point_count]` may be irregular, and reference values are `[active_state, concatenated_point]`. The first point in each trajectory is its complete initial condition; later points are scored. Trajectories may have different lengths. With `allow_missing_observations`, NaNs at later points mark unobserved states; the JSON frontend accepts `null` or masks and supplies this native representation.

Every supplied observation and time must be finite, except enabled missing-observation NaNs at noninitial points. Offsets must partition the entire buffer into nonempty trajectories, and times must increase within each trajectory. Invalid data, a step that underflows to zero, nonfinite integrated states, and FP32 squared-error overflow receive `FLT_MAX`. The MSE denominator counts observed scored state values, excluding initial points. Device buffers must cover the declared layouts; arithmetic span checks do not establish actual allocation sizes. The JSON frontend rejects invalid trajectory data before scoring.

Constant banks are `[system, bank, active_constant]`, and MSE output is `[system, configuration]`. The active toggle-bit count is a runtime uniform and does not create another compiled template shape. Toggle capacity is not a CUBIN dimension: the configuration ABI provides 32 bits to every template. Each AST toggle carries an immediate SASS bit mask that reads its assigned low configuration bit; the runtime uniform shifts the configuration index to select the constant bank. With one toggle bit and 16 banks, lanes `0,2,4,...` take choice zero, lanes `1,3,5,...` take choice one, and adjacent pairs share a bank: 32 configurations total. Twelve bits provide 4,096 toggle permutations. Reusing a bit index couples toggle sites. A bulk request can contain more systems than one module's `system_capacity`; Odezza partitions it without changing system-major output order.

The compiled CUDA template is identified by SM version, state and constant capacities, system capacity, and shared/system patch capacities. Active counts, state labels, constant values, toggle bit positions, permutation count, trajectory counts, offsets, times, observations, and RK4 steps do not require recompilation. During SASS specialization, physical input registers reserved for inactive states and constants are returned to the temporary-register allocator. Overallocating those remaining dimensions still affects the CUDA scaffold and can reduce performance, so capacities should be reusable buckets rather than unbounded maxima.

Patch budgets are especially expensive because their reserved instructions are physically present in every CUBIN. The internal tools default to 64 shared and 64 per-system instructions; larger explicit buckets should be selected from actual AST analysis rather than used as a universal maximum.

The current scoring shape is FP32 with ragged, irregularly timed observations, optional missing-state masks and fixed-step RK4. Adaptive/stiff integration is unsupported by this path; local optimization uses the separate LM API. The complete runtime trajectory is staged in dynamic shared memory per CTA, so very large trajectory batches can exceed the device's shared-memory limit.

## Internal diagnostics

The [unified search design](docs/unified_search_design.md) connects expression
families, constrained grammars, shared choices, parameter sampling and execution
planning. It is historical design context; the current native request API and
resident mac3/rack1 service are implemented as described above.

The [September 5 scoring review](docs/scoring_kernel_review_2026-09-05.md) records the latest correctness fixes, generator measurements, GPU regressions, and remaining compatibility gaps.

The [SASS writer guide](docs/sass_writer.md) describes the reusable C99 instruction encoders, buffer checks, caller responsibilities, and byte-for-byte regression coverage.

[`src/o_odezza_internal.h`](src/o_odezza_internal.h) exposes source generation, NVRTC results, CUBIN inspection, prespecialization, candidate specialization, retained artifacts, and detailed pipeline statistics only to repository tests and diagnostic executables. These interfaces are not part of the public compatibility contract.

The internal executables are named accordingly:

```bash
build/odezza-internal-generate-scoring-cuda --help
build/odezza-internal-pipeline-benchmark --help
```

The Python implementation under [`python/`](python/) contains two research paths. The original scoring reference predates the current C99 ragged runtime pipeline. The newer trajectory-LM path generates, inspects, and specializes a one-candidate-system, thread-owned LM kernel over ragged irregular trajectories. That research implementation remains available for reference; the validated service numerical path now also has a native C99 LM handle in [`core/`](core/README.md). See [`docs/python_trajectory_lm.md`](docs/python_trajectory_lm.md).

GPU score reduction and replay indices: [API, JSON options and measured results](docs/score_reduction.md).

## Trial coefficient pools and partial observations

The isolated recovery trial supports [Philox pools, missing observations and producer events](docs/philox_pool.md). Rebuild callers for generator ABI 5. The reusable pool and reducer APIs are declared in `core/odezza.h` (the old headers are compatibility includes); trial request examples and measured validation are in [the implementation record](scratch/philox_pool/RESULTS.md).

## Native C99 core

The authoritative kernel and execution implementation is in [core/](core/README.md).
[core/odezza.h](core/odezza.h) is the single public header for scoring, analytic LM,
Philox pools and score reduction. Shared internals and functional C99 compilation
units stay in that folder; GP, grammars and the service stay outside. Historical
`src/` paths forward to it for build compatibility.

Paired fitter calibration and blind recovery sweeps use the reusable
[LM tuning infrastructure](benchmarks/lm_tuning/README.md), with dependency-controlled
random generation, frozen JSON protocols, detached GPU queues and retained timings.

## Direct grammar jobs

The version-1 grammar compiler, native GPU adapter, and separate stdio MCP trial
are documented in [docs/grammar/README.md](docs/grammar/README.md). They support
full RHS vectors, binary/quad state toggles, GPU Philox banks and pre-RK4 numeric
transforms, grouped retention, replay and separate native LM. The existing GP
service and hardened C99 core remain separate.


The [C99 request frontend](frontend/README.md) now feeds a separate
[native scoring runtime](runtime/README.md), the default direct-grammar CLI
backend. C owns parsing, double-buffered AST production, interleaved family
execution, GPU Philox/prelude binding, reduction and compact retention. Fixed
RHS/template caching and capacity growth remain below the request layer. The old
Python compiler/LM adapter is preserved under `--backend python`; the GP service
and hardened core boundary are unchanged. See [execution validation](docs/grammar/C99_RUNTIME_VALIDATION.md).

Native requests now support automatic configuration tiling and a C worker group
for one job across GPUs (`--devices 0,1`). The default CLI/MCP process supervises
its CUDA worker and exposes release/expiration, bounded admission and cheap status
snapshots. See [service and batching validation](docs/grammar/C99_SERVICE_AND_BATCHING.md)
for timings, lifecycle limits and the reproducible source snapshot procedure.
