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

# Previous Python grammar backend

**The CLI now defaults to the database-free [C99 runtime](../../runtime/README.md).**
This document describes the preserved `--backend python` adapter, including its
SQLite persistence, SHA-based sampling identities and LM tools. Its capabilities
and result schema differ from the native scoring backend.

The supplied version-1 grammar now connects to the public native scoring and LM
APIs. It is a separate Python package/service; it does not replace the existing
GP service and adds no search strategy or grammar code to `core/`.

The canonical compiler source is [python/odegrammar](../../python/odegrammar).
The adapter is [python/odezza/grammar](../../python/odezza/grammar). The imported
compiler is unchanged from `ode_grammar_handoff.zip`; its 83 original tests live
in [tests/grammar_compiler](../../tests/grammar_compiler). Historical handoff files
in this directory retain their original archive-relative paths. Use this README
for the repository layout and executable entry points.

Pending design work and the user's compact-result proposal are tracked in
[the results, trajectories and overhead review](RESULTS_AND_TRAJECTORIES_REVIEW.md).
That checklist distinguishes proposed interfaces from current behavior.

A separate [C99 frontend](../../frontend/README.md) now implements phased arena
parsing, prepared grammar cursors, native AST batches and C-owned template caching.
It now connects to the separate native runtime; see [runtime validation](C99_RUNTIME_VALIDATION.md).

## Request flow

1. Call `odezza_submit` once with `problem`, `grammar` and optional `execution`.
   The problem contains ordered states, trajectories, complete initial states
   and optional known RHS. See [a complete request](../../examples/grammar/submit.json).
   The service saves and queues the request, then returns a `job_id` before
   trajectory validation, grammar expansion, CUDA initialization or scoring.
   Receiving/decoding/snapshotting the JSON and storing the request still take
   time proportional to input size; "immediate" does not mean zero-cost upload.
2. The worker prepares an immutable problem and compiles the version-1 grammar.
   Known RHS can be omitted from the incoming grammar; the adapter fills them
   before invoking the canonical compiler. A supplied known RHS must match the
   prepared expression exactly. Placeholder `problem_id` fields in examples must
   be replaced with the returned handle or omitted.
3. Poll `odezza_status` or `odezza_results` with the job ID. Active statuses are
   `queued`, `preparing`, `generating` and `running`. Invalid observations or
   unsupported grammar/backend features become pollable failed jobs before GPU
   submission. Shallow malformed submission envelopes fail in the tool call.
   Generation and execution have separate completion statuses and counts.
   `running` includes GPU setup before the first completed batch; its `phase`
   distinguishes `gpu_setup` from `scoring` so cold kernel setup is not labeled
   as grammar generation.
4. Read global, family and tag winners. The default full report is preserved;
   `format: "compact"` returns candidates once and leaderboards as ID lists.
   A replay record contains the full vector,
   exact FP32 coefficient values, configuration address and execution provenance.
5. Optionally issue a separate LM request naming retained candidates and fitted
   `theta` parameters. State bindings, constants and sampled RNG values stay fixed.

No GP loop, automatic structural proposal, fitting or trajectory thinning occurs
implicitly. The calling LLM/controller decides the next request.

For reuse, `odezza_prepare` remains available and submission may use `problem_id`
instead of inline `problem`. Exactly one is required. The worker returns the
resolved `problem_id` in status/results for both routes. Idempotency keys deduplicate
retries of the same complete payload; changed payloads create different jobs.
The request is a snapshot: later caller-side object edits cannot affect the job.

## Run locally on a CUDA host

Use the repository's existing CUDA setup and `make shared`. No new Python
dependencies or package installation are necessary; Python 3.10+ is recommended
by the supplied compiler.

```sh
make shared
export PYTHONPATH="$PWD/python"
export ODEZZA_CORE_LIBRARY="$PWD/build/libodezza.so"
python3 -m odezza.grammar --backend python --root ./grammar-jobs run \
  examples/grammar/submit.json
```

The CLI `run` waits for completion; immediate submission/polling is provided by
the persistent Python service or MCP process. The previous split-file invocation,
`run grammar.json --problem observations.json`, remains supported.
`plan` accepts the same arguments and does not initialize CUDA. A plan traverses
the bounded grammar; it is not a constant-time count estimate. CPU references
are verification tools and never a fallback for an unavailable GPU.

For programmatic persistent use:

```python
from odezza.grammar.service import Service

service = Service("grammar-jobs", library="build/libodezza.so", device=0)
try:
    job = service.submit(problem=problem_json, grammar=grammar_json,
                         execution={"max_seconds": 120},
                         idempotency_key="my-first-screen")
    # Returns immediately. Poll service.status(job["job_id"]).
    report = service.jobs[job["job_id"]].result()  # local blocking convenience
finally:
    service.close()
```

One service owns one GPU execution thread and storage root. Use separate roots
and `--device` values for separate workers. Automatic multi-GPU partitioning is
not implemented in this adapter. The CUDA context stays alive across requests.
Native pipeline handles own their execution streams; uploads and prelude work
use an application stream and event dependencies. The worker establishes
`CUDA_MODULE_LOADING=EAGER` before initialization and rejects an explicit lazy
setting.

## Stdio MCP

```sh
python3 -m odezza.grammar --backend python --root ./grammar-jobs mcp
```

The dependency-free stdio transport negotiates MCP `2025-06-18` and exposes:
`odezza_capabilities`, `odezza_prepare`, `odezza_submit`, `odezza_status`,
`odezza_results`, `odezza_cancel`, `odezza_replay`, `odezza_fit`, and
`odezza_fit_results`. Scoring and fitting return handles promptly. This is an
SSH/subprocess service, not a public HTTP deployment or a multi-tenant server.
The storage root is exclusively locked. Restarted incomplete jobs are marked
`interrupted`; they are not automatically resumed. Start a fresh job to rerun.

For mac1, the MCP client can launch `ssh rack1` with the remote Python command,
absolute `PYTHONPATH`, library and storage paths. See [rack1.json](rack1.json) for
the isolated checkout used during integration. Each MCP process must own a
different storage root. No Telegram messages or external notifications are sent.

The transport follows the official [stdio framing](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
and [tool result](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
contracts. It does not claim optional MCP task/resumption extensions.

## Compact results and index replay

The same `odezza_results` tool accepts `{"job_id": "...", "format": "compact"}`.
Its version is `odezza.grammar-results.compact.v1`. The response retains job counts,
status, integration, timings and execution profile, and contains:

- `leaderboards.global`: candidate IDs in ranked order.
- `leaderboards.families` / `leaderboards.tags`: named lists of candidate IDs.
- `candidates`: one record per retained output candidate, including readable
  equations, named values, score and an `origin` address.
- `ranking`: the exact requested retention policy and uniqueness units.

Pass a candidate's `origin` unchanged to
`odezza_replay(job_id=..., address=origin, cpu=True)`. An origin has the form:

```json
{
  "version": "odezza.grammar-address.v1",
  "manifest_id": "<immutable manifest hash>",
  "variant_index": "271",
  "configuration_index": "137"
}
```

Both indices are decimal strings to preserve integer precision in MCP/JavaScript
clients. The variant ordinal is a persisted uint32 mapping, independent of batch
ordering, and references the exact skeleton and pool plan. The configuration
address is uint64 and follows the compiler's last-axis-fastest ordering. The
manifest pins the ordinal-table digest and the emitted request definitions.

Index replay directly looks up that variant, decodes the bank/permutation and
locates the retained coefficient snapshot. It rebuilds the resolved RHS from
the stored program and exact FP32 values, then checks its numerical/structural
identities and bytecode against the archived candidate. It does not enumerate
earlier ASTs, regenerate random banks, run LM or launch GPU work. Optional CPU
verification remains separately labeled. Wrong manifests, invalid/out-of-range
indices and configurations without retained snapshots produce explicit errors.

An address names a scored initialization, not a later LM result. Replay fitted
candidates by `candidate_id`, retaining their optimized values. Candidate-ID
replay and full reports remain available for jobs created before ordinal manifests
were introduced. Index replay requires a new-format job.

This adds compact delivery and addressed reconstruction. It does not yet change
the hot path's intermediate candidate materialization/storage or introduce a
final-winner-only GPU gather. Those optimizations remain separately tracked in
the [overhead report](../../benchmarks/grammar_overhead/REPORT.md).

## Observations and RK4

Prepared data has this shape:

```json
{
  "states": ["x0", "x1"],
  "known_rhs": {"x0": "x1"},
  "trajectories": [{
    "initial": [1, 0],
    "times": [0, 0.1, 0.23],
    "values": [[1, 0], [0.995, null], [0.974, null]]
  }]
}
```

Rows follow state order. `null` and an optional boolean `[time,state]` `mask`
exclude observations, never evolving states. Every trajectory requires a
complete initial vector. Initial observations must agree with it when supplied.
The loss is the sum of squared residuals divided by the number of observed
noninitial scalars, pooled across states and trajectories. Unequal weights,
observation functions, events and input tables are not implemented here.

The grammar's `dt` is resolved as a maximum step: choose a common subdivision
count `ceil(max_observation_interval / dt)` and land exactly on every observation.
Shorter intervals consequently get smaller steps. FP32 rounding can add one
subdivision near integer ratios. The effective count, initial-point exclusion,
precision and objective enter result identity. `max_steps` limits total RK4 steps
per configuration across all trajectories. There is no adaptive or stiff-solver
fallback. Native numerical failures are counted as invalid and never ranked.

## Constants, toggles and RNG

See [the compiler syntax](COMPILER_README.md) and
[constant_rng_product.json](../../examples/grammar/constant_rng_product.json).

- Numeric axes remain lazy Cartesian products. The adapter creates one native
  program per structural/toggle-group variant, not per coefficient trial.
- A named toggle reads its selected evolving state at every RHS evaluation.
  Repeated references share the selection. Exactly two or four distinct states
  form a group. Exhaustive groups overlap; counts are visits, not unique models.
- Constants referencing the same bank with different names remain independent
  axes. RNG bindings sharing an `axis` share the index; sharing bank and `stream`
  shares raw values. Those are separate relationships.
- A resident GPU kernel generates standard uniform/normal banks. A separate
  reusable GPU prelude resolves numeric indices, constant selections and RNG
  transforms into bounded device coefficient chunks. The existing scorer loads
  each row before RK4; those values remain fixed throughout the rollout and LM.
- Transforms `identity`, `affine`, `uniform`, `normal`, and `log_uniform` support
  finite numeric operands and `const.name` operands. Transform-only constants
  are allocated even when absent from direct RHS references.
- `rng_banks.<name>.count` declares pool length. `execution.max_bank_bytes` bounds
  resident bank storage; oversized individual or simultaneous batch banks fail
  explicitly. Extending a bank preserves the sample prefix.

Production profile: **`odezza.grammar.philox4x32-10.v1`**. Hash ASCII
`PROFILE + ':' + compiler_bank.key` with SHA-256; little-endian bytes 0..7 form
the Philox 64-bit seed, and bytes 8..15 masked to 63 bits form the stream.
The counter is `[block_lo,block_hi,domain_lo,domain_hi]`, `block=sample_index//4`,
`domain=(stream<<1)|normal`. Keys are `[seed_lo,seed_hi]`; ten rounds use the
standard Philox4x32 multipliers and Weyl increments. Uniforms use the open
23-bit midpoint mapping `((word >> 9) + 0.5) * 2^-23`. Normal pairs use FP32
Box–Muller with the same distribution-separated counter domain as native Odezza.
Prelude multiply/add operations are separately rounded, with no fused multiply
add and no implicit clamp. Uniform/log-uniform interpolation uses the compiler's
convex formula. GPU transcendental rounding is part of the production profile.

This intentionally differs from the handoff's `sha256_reference_v1` samples.
Compiler recipes are preserved, and result identity also includes the production
profile, native library hash, adapter/compiler source hashes, prepared problem,
resolved integration, driver/NVRTC versions and exact winner coefficients. Normal replay uses gathered
FP32 values for portable exact constants; it does not promise bitwise CPU libm
reconstruction. RNG represents sampled deterministic coefficients, not SDE noise.

## Retention and limits

The native reducer operates per system/permutation. Larger requested top-k uses
additional reduction passes; host code receives winners and gathered rows, not
all MSE scores. Repeated numeric rows are removed before filling top-k. Per-variant
survivors remain available for late family/tag provenance. Rankings are finalized
after provenance is known, without reevaluating those variants.

`numeric_candidate` equality is exact resolved FP32 postorder equality;
`resolved_structure` uses fixed state bindings and alpha-normalized coefficient
slots, preserving repeated-slot sharing; `variant` uses the compiler variant ID.
These are syntactic identities, not an algebraic-equivalence theorem. Ranking is
by measured MSE and stable candidate address. No complexity penalty or posterior
probability is implied. Per-family/tag `k` is limited to 256 in this adapter.

The grammar's mixed-radix address has the last axis fastest. Native scoring uses
`bank * permutations + permutation`; replay explicitly converts between these
orders and records the original grammar index. Numeric chunks never reset that
address or the RNG stream. Compiler generation limits are preserved. Separate
execution budgets bound wall time, module size, numeric chunks and device memory.
Cancellation is cooperative between records/launches; an in-flight native kernel
is allowed to finish. The stored generation summary does not prove GPU completion.

The native marker constraint is `2*states + coefficient_slots + 1 < 255`;
architecture/register/patch resources can impose lower limits. Unsupported `t`,
integer powers outside [-16,16], unsupported solvers and invalid coefficient
layouts are explicit compatibility errors. LM is separate, supports up to eight
states/eight active fitted `theta` slots, and supports explicit 1/2/4/8 lanes with
wider register-pressure fallback. Its `settings.tolerance` is interpreted as the
target observed MSE, not a gradient tolerance. Non-fitted values are literals in
the LM program. `abs`/other derivative limitations are enforced by native LM.

Prepared known equations are currently checked and included in each full-vector
candidate. The adapter does not yet hoist those common equations into the core's
fixed-RHS patch. The core itself supports that optimization.

## Validation and performance

```sh
make test-grammar
PYTHONPATH="$PWD/python" python3 tests/grammar/gpu_conformance.py \
  --library build/libodezza.so --root /tmp/grammar-conformance-new
PYTHONPATH="$PWD/python" python3 tests/grammar/gpu_examples.py \
  --library build/libodezza.so --root /tmp/grammar-examples-new
```

The native scoring report exposes combined specialization/module/integration
wall time, not separate device integration events. Reports label this limitation;
they separately record grammar planning, cold template setup, RNG/prelude/reducer
events and a residual host/setup category. Do not compare cold end-to-end times
with a prepared kernel's peak configurations/second.

See [VALIDATION.md](VALIDATION.md) for measured populations, conformance evidence,
current limitations and follow-ups.
