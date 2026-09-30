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

# C99 request preparation

This is the separate, allocation-free CPU frontend for Odezza scoring. It parses
JSON into caller-owned trajectory arrays, fixed RHS programs, and an immutable
prepared grammar. Producers emit native postorder bytes directly. There is no
Python, database, file I/O or CUDA dependency in `libodezza_request.a`.

The optional C scoring bridge owns template preparation, a C-owned SQLite compiled-template cache,
fixed-RHS prespecialization, and patch-capacity growth. The hardened kernel API
remains in `core/`. The direct-grammar service uses this frontend through the
separate native runtime; the original GP services retain their existing path.
See [frontend validation](../docs/grammar/C99_FRONTEND_VALIDATION.md) and the
[integrated runtime](../runtime/README.md).

## Build

From the repository root:

```sh
make frontend
make test-frontend
make benchmark-frontend
# On a CUDA development machine, using the repository's normal native setup:
make shared
make -C frontend scoring test-gpu
```

A CPU-only CMake build is also available with `cmake -S frontend -B build/frontend-cmake`.
The root CMake project includes both frontend libraries alongside the native core.
Link CPU consumers against `libodezza_request.a` and the math library. The optional
bridge additionally links `libodezza_scoring_request.a` and the native Odezza/CUDA libraries.

## Separate phases

Public CPU API: [odezza_request.h](odezza_request.h).
Optional native bridge: [odezza_scoring_request.h](odezza_scoring_request.h).

| Phase | Function | Output |
|---|---|---|
| Trajectories | `odr_trajectories_parse` | FP32 times, state-major values, offsets, spacing/observation metadata |
| Fixed system | `odr_static_parse` | State-indexed native postorder RHS programs |
| Grammar | `odr_grammar_parse` | Immutable rules, bound symbols, choices, slots, family allocations |
| Family cursor | `odr_producer_create` | Independent bounded cursor, expansion scratch, exact dedup storage |
| AST batch | `odr_producer_next` | Native programs, coefficient axes, indices, tags, allocated bank ranges |
| Native descriptors | `odr_scoring_pack_systems` | Pipeline descriptors borrowing the batch's instruction bytes |
| Scoring preparation | `odr_scoring_create` | Cached template with the fixed RHS prespecialized |
| Execution | `odr_scoring_run` | Native scores/report; grows patch capacity when necessary |

Each parser accepts its section independently. All three can also receive the
same full request: trajectory/static phases select `problem`; the grammar phase
selects `grammar`. For separate trajectory/static calls use `{states, trajectories}`
and `{states, known_rhs}` respectively. The grammar must omit fixed equations;
the static handle supplies them, and state order must match exactly.

The [GPU acceptance caller](../tests/frontend/gpu_test.c) consumes
[submit.json](../examples/grammar/submit.json) using these phases, including its
coefficient grid and RK4 settings, without the Python compiler.

## Arena and lifetime contract

Parsers and producers never call a heap allocator. Supply aligned memory using
`odr_arena_alignment()`. A NULL arena measures the required capacity and returns
`ODR_OK`; the output handle remains NULL. A short arena returns `ODR_BUFFER` and
the required capacity, without writing past the supplied extent. In-range writes
on failed calls are allowed. Outputs from a failed parser are invalid.

```c
size_t bytes;
const OdrTrajectories *trajectories = NULL;
OdrError error;
OdrResult result = odr_trajectories_parse(
    request, NULL, 0, &bytes, &trajectories, &error);
/* Check result; obtain bytes of aligned storage from your arena manager. */
result = odr_trajectories_parse(
    request, storage, capacity, &bytes, &trajectories, &error);
```

Measurement does not construct the persistent plan: the caller's subsequent
write call prepares it. Grammar preparation validates syntax and sizes first,
then binds names to direct references in the arena. It does not revisit JSON
while producing candidates. The input JSON can be released after preparation.

The grammar arena outlives its producers and batches. The producer arena contains
mutable cursor/scratch/dedup data; a batch owns its instruction bytes and output
descriptors but borrows immutable names, grids and tags from the grammar. Reusing
the producer never changes a prior batch. Keep fixed program memory alive for the
scoring session. Arenas/handles contain pointers and cannot be moved or serialized
by copying their bytes. There is no frontend destructor: release the arenas after
all borrowers finish.

The parser uses bounded stack scratch, including 256 KiB for each static RHS
parse; expressions are limited to 32,767 text bytes and 128 levels of nesting.
Large syntax trees can return a capacity error even when their emitted program
would be small. The frontend currently accepts 1–126 states and checks the native
coefficient/toggle operand limits separately.

CUDA session creation and cache I/O **may allocate**, including inside NVRTC and
the native pipeline. That is separate from the allocation-free parsing/generation
contract. The context, device buffers and input-ready event remain caller-owned.
Set `CUDA_MODULE_LOADING=EAGER` before CUDA initialization. A scoring session is
single-caller; keep its context current for execution and destruction.

## Trajectory layout

Only the current native FP32, ragged RK4 layout is accepted:

- `offsets[trajectory_count + 1]`, with concatenated FP32 `times[point_count]`.
- `values[state * point_count + point]`, with each trajectory's complete initial
  vector at its first point. Every trajectory must supply `initial` separately.
- Later `null`/masked observations become NaNs. Missing initial observations are
  filled from `initial`. Supplied initial observations must agree with it.
- Metadata records point/observed counts and first/last time, minimum/maximum gap,
  and whether spacing is approximately uniform. Irregular spacing retains every
  timestamp; no resampling occurs. `dt` is NaN when spacing is irregular.

Values and times must be finite after FP32 conversion; times must still increase
strictly. MSE excludes initial conditions and counts observed scalar residuals.
No observation-only derivative, interpolation, stiff-solver, or CSV conversion
path is silently selected.

`odr_trajectories_rk4` resolves a common subdivision count from a positive maximum
time step and enforces `max_steps` per configuration. `odr_grammar_info` exposes
those integration settings. It also returns the scheduler's deadline and a single
arena-owned retention-policy JSON slice; the AST producer does not run a clock
or execute top-k retention.

## Streaming, allocations and indices

A producer belongs to one family. Create a producer per family and interleave
bounded calls. `odr_grammar_family_info` exposes reserved allocations. Explicit
allocations are reserved first; a supplied request ceiling divides the remainder
equally among unspecified families, with the remainder assigned in declaration
order. Mixed explicit/implicit allocations require a request ceiling. Conflicts
are rejected during preparation. Unused allocations are not redistributed.

`start` is the next **accepted native candidate** index for that producer:

```c
uint64_t next = 0;
for (;;) {
    OdrBatch batch;
    OdrResult result = odr_producer_next(producer, next, 256,
        output_arena, output_capacity, &required, &batch, &error);
    /* Consume any completed prefix even when a later generation error occurs. */
    next = batch.next_index;
    /* Queue batch.count candidates; keep their arena alive through GPU completion. */
    if (result != ODR_OK || batch.exhausted) break;
}
```

Use two or more output arenas: while one is in a synchronous scoring call on an
execution thread, generate into another. One producer cannot be used concurrently;
independent producers can share the immutable grammar. Native pipeline worker/
module-slot settings remain selectable. This library supplies these components;
it does not create a background family scheduler or manage a job queue.

This is **forward cursor resumption**, not arbitrary random access. Asking for an
index other than the next one returns `ODR_INDEX`, without replaying or discarding
a prefix. General filtered/deduplicated grammars require stored checkpoints or an
index to support arbitrary seeks; that is not implemented.

`odr_batch_requirements` is a conservative bound and does not expand anything.
A NULL/short output arena never advances the cursor. Calls may return fewer than
requested, including zero, when their work quantum is reached. Continue at the
returned index until `exhausted`; distinguish `yielded` from completion. The
quantum bounds both derivations and native variant visits. Errors after generation
begins return a valid completed prefix; an exhausted dedup arena requires a larger
producer allocation for a new producer, not moving/resizing a live arena.

Budgets distinguish skeleton derivations with accepted work, emitted native
variants, expansion visits, and configurations. Each candidate reports the full
numeric Cartesian count plus an allocated `bank_start/bank_count`. A partial
allocation consists of whole bank rows across the complete toggle product; unused
remainders are reported. These are **reserved** configurations, not successful GPU
evaluations. Report actual submitted/valid/invalid work from execution separately.
Optional `max_configurations_per_skeleton` applies across the variants of one
structural derivation. The family setting overrides the request setting. Reaching
it skips that skeleton's remaining variants; a cap smaller than one complete
toggle product skips the skeleton. `configuration_limited_derivations` reports
such attempted derivations, including ones clipped before deduplication. Omission
preserves prior coverage. It is not a budget to redistribute among families.
The separate native runtime provides interleaved family execution.

`odr_request_allocations` reads the reserved totals using exactly the same
allocation helper as grammar preparation. It allocates nothing and does not
parse expression trees or emit ASTs. It is useful for prequeue policy checks;
it is not full grammar validation or a prediction of deduplicated population size.

Programs are emitted directly once per attempted variant; accepted bytes are
copied into the output arena and used for exact deduplication without a JSON AST
round trip. There is no algebraic equivalence deduplication. Bindings are part of
identity; distinct local coefficient occurrences remain independent. Dedup stays
within a family. Later duplicates can call `on_duplicate` with the original index
and their tags so retention can accumulate provenance without another AST payload.
Without this callback the batch carries only the first occurrence's tags.

The batch carries accepted index, raw derivation index, selected joint-variant
rank, family, tags and slot/axis layout. State toggle choices are embedded in
native postorder instructions. A configuration uses low bits for toggles and the
remaining index for a coefficient row. For a numeric axis, its index is
`(bank_index / numeric_stride) % count`; fixed values have stride zero.

RNG descriptors retain base distribution, count, seed, bank, stream, shared axis,
scope, local occurrence, and transform/dependency slots. This CPU-only frontend
prepares descriptors; the [native runtime](../runtime/README.md) now binds them to
C-owned GPU pools and applies Cartesian transforms before scoring. Direct bridge
callers remain responsible for binding actual coefficients or supported native
sampled descriptors.

## Templates, capacity and timing

The optional bridge uses an existing directory containing a C-owned SQLite
compiled-template cache. SQLite 3.53.4 is vendored; no system development package
is needed. Keys include compiler version/options, SM and template capacities;
the native reader validates shape, generated-source identity and checksum.
Transactions protect writes, and striped host-local locks coalesce cold
compilation. A configurable payload budget defaults to 256 MiB with LRU eviction.
The allocation-free parser/producer library and hardened core do not link SQLite.
No request data or specialized customer RHS are stored. See the
[runtime cache contract](../runtime/README.md#compiled-template-cache-and-bounded-retention).

Each session prespecializes fixed equations once for its current shape. On native
specialization-capacity failure it doubles shared/system patch capacities up to
`maximum_patch_capacity`, caching the larger template and reusing the existing
AST bytes. New workspace needs are returned to the caller. Only patch space grows;
state, coefficient, register or unsupported-instruction errors do not silently
change shapes or solver semantics.

A failed multi-module attempt may already have executed some modules. Discard its
scores; retries can repeat that execution, but never regenerate the ASTs. Stats
expose attempts and retry-requested configuration counts, which are an upper bound
on repeated work. An unfenced CUDA failure is preserved for the caller to handle;
resources are not freed while device work may still use them.

Stats separate actual time inside `nvrtcCompileProgram`, template preparation,
and pipeline preparation/prespecialization. Cache reads report zero NVRTC time.
The native run report remains an end-to-end pipeline time, including specialization
and loading; it is not a pure GPU-kernel timer.

Implementation files are split by JSON scanning, expression syntax, trajectories,
static RHS, grammar preparation, production, sampling, and optional scoring/cache
integration. Shared declarations are private. Keep service policy and GP outside
both this parser library and the hardened core.

## Queued runtime integration

The optional [native runtime](../runtime/README.md) now consumes these arenas and
producer batches directly. It provides the default direct-grammar CLI/MCP scoring
backend, with C numeric pools/prelude, interleaved execution and compact retention.
It may allocate and owns a CUDA context; those responsibilities remain outside
this allocation-free CPU library. [Validation](../docs/grammar/C99_RUNTIME_VALIDATION.md).
