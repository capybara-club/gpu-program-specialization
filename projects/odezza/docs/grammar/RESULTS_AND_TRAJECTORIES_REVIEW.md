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

# Review: compact results, trajectory assets and execution overhead

Recorded 2026-09-11 at the user's request. This is a design and review backlog,
not an implementation announcement. The current executable contract remains in
[README.md](README.md). The user's supplied proposal is preserved verbatim in
[RESULT_ADDRESSING_PROPOSAL.md](RESULT_ADDRESSING_PROPOSAL.md); its examples are
proposed interfaces, not commands or additional authorization to change the core.

**2026-09-11 implementation update:** combined asynchronous submission, persisted
variant ordinals, string-encoded addresses, retained-snapshot reconstruction and
normalized compact result views are now executable. See [usage](README.md) and
[acceptance evidence](VALIDATION.md). The original design below records the starting
point; trajectory assets/views, native partial losses, pagination and the compact
intermediate-survivor performance work remain proposals.

## Decisions to carry forward

- **Latest user direction (2026-09-11): no database in the grammar execution
  path.** Parse static RHS into AST data once, load observations into float arrays,
  and prepare grammar-derived data for the native pipeline. Use bounded in-memory
  batches, compact winners and shared run metadata. This replaces the previous
  suggestion to batch durable SQLite writes; it includes plan/deduplication storage,
  not just the candidate table. Existing SQLite execution has not yet been removed.
  Define handle lifetimes and replay after process exit explicitly during this
  change; any requested export should be a plain artifact, not an implicit database.
- Keep GPU hits compact and tied to immutable execution manifests. Reconstruct
  only the candidates necessary for correct requested rankings and durable output.
- Preserve exact winner values even when indices can reproduce their RNG inputs.
  Store optimized LM parameters explicitly; an initialization index cannot recover
  the result of optimization.
- Return candidates once, with leaderboard entries referencing their IDs.
- Separate reusable observation data from model declarations and search requests.
- Default to one aggregate objective for bulk screening; offer per-trajectory
  diagnostics on frozen retained candidates. Searching independently per trajectory
  is a separate objective/retention mode, not a display option.
- Keep asset management, grammar expansion, identities and reporting outside
  `core/`. Any optional native output mode needs its own clean ABI review and
  performance comparison before a core change.

## Current implementation versus proposal

| Area | Present behavior | Review/action |
|---|---|---|
| GPU winner | Native reducer returns a 16-byte `(float mse, uint32 reserved, uint64 score_index)` record | Reuse the native decoder; do not replace the ABI just to put a variant ordinal in the reserved field |
| Address decoding | Layout decodes native system/bank/permutation; adapter maps back to the grammar configuration index | Persist a stable run-local ordinal map, including chunk/module addressing; the current SQLite plan has direct hash-ID lookups but no exported ordinal contract |
| RNG | Versioned Philox coordinates, fixed pre-RK4 transforms, exact gathered FP32 winner values | Compare gathering retained values with regenerating them in a finalization kernel; no bitwise CPU normal/transcendental promise |
| Retention | Per-binding survivors, repeated reduction when needed, distinct-unit global/family/tag rankings | Preserve correctness while postponing expensive reconstruction; a global raw shortlist is insufficient |
| Identity work | Host constructs identities and resolved programs for local survivors before final ranking | Measure survivor counts/cost; numeric deduplication may require resolving more than the final global k |
| MCP output | Full candidate records are repeated in global/family/tag arrays | Add a versioned candidate dictionary plus ID lists and bounded/paged retrieval |
| Large indices | Current JSON uses Python integer values | Encode uint64 addresses as decimal strings for clients that cannot exactly represent integers above 2^53 |
| Input data | `odezza_prepare` persists trajectories, state order and known RHS in one immutable problem | Grammar submissions already reuse this handle; independently reusable trajectory assets and selection views are not implemented |
| Diagnostic loss | `odezza_replay(cpu=true)` returns FP64 per-state/per-trajectory MSE for a retained candidate | Add explicit frozen-candidate evaluation across selected assets; GPU per-trajectory output is not currently exposed |

The proposal's example is correct: last-axis-fastest dimensions `2 * 3 * 32`
decode index 137 as positions `(1, 1, 9)`. An address is meaningful only with its
manifest, ordered axes and variant. Never derive draws from launch/thread order.
Run-scoped and skeleton-scoped bank sharing must survive chunking and scheduling.
Guard ordinal/address width overflow rather than silently truncating; verify any
new wire structure's size and field offsets explicitly.

## What the measured overhead means

**Follow-up measured:** [2026-09-11 profiling report](../../benchmarks/grammar_overhead/REPORT.md)
now attributes the previously unresolved time to intermediate candidate storage,
identities/hashing and repeated lowering/batching. The historical measurement
and initial hypotheses below are retained for provenance; use the new report
for current attribution and the ordered optimization priorities.

Evidence: [warm adaptation run](validation/examples-06/examples-summary.json).
The workload is 8,192 skeletons and 1,048,576 configuration visits on two short
synthetic trajectories, five observation points each. It is not a recovery test.

| Accounted category | Seconds |
|---|---:|
| Grammar planning, validation, lowering and plan storage | 3.376852 |
| Native scoring calls, including specialization/loading/integration | 0.140104 |
| Other accounted setup, reduction/transfer and RNG/prelude work | 0.160392 |
| Unattributed wall-time residual | 6.750536 |
| Worker job elapsed | 10.427883 |

Planning finishes before the first scoring batch is dispatched. The hot path then
reads plan records, lowers programs again, builds coefficient chunks, invokes the
native pipeline/reducer, constructs survivor identities, persists them and builds
the report. These are inspection-based candidates for the residual; their
individual costs have not been profiled. The residual is not a measurement of
Python alone. Native scoring time is not pure GPU kernel time. Event intervals
can overlap host work, so the arithmetic residual is not a causal profiler.

The timer begins in the server worker. It does not include the earlier trajectory
prepare request, queue wait or the mac1 round trip. Do not blame network transfer
for this measured ten seconds. GPU occupancy or utilization cannot be calculated
from these totals. Longer rollouts can change the ratio substantially.

First measure cold/warm contexts separately and instrument generation, validation,
each lowering pass, storage, allocations, uploads, native pipeline, reduction,
candidate construction and serialization. Record input/output bytes, module/batch
counts, survivor counts and actual completed work. Use a correlated native/device
timeline to separate specialization, module loading and integration. Keep queue
and client-visible latency as separate measurements.

Then test cached prepared populations, avoiding repeat lowering, batched metadata
work, compact survivor storage, common-known-RHS specialization and batch sizing.
Increasing numeric trials can amortize per-structure work, but must be justified
by useful search coverage. Do not infer constant runtime for a larger bank or a
kernel-only speedup from these totals. Overlapping planning with dispatch requires
an explicit change to today's full-preflight-before-GPU contract.

## Proposed trajectory and objective model

1. **Trajectory assets:** register each trajectory independently or in a batch.
   Store state schema/order, explicit full initial state, initial time, observation
   times, values, missing-data mask and optional descriptive metadata. Return an
   immutable content/revision ID. Validate compatibility when binding to a problem.
   A changed observation or initial value creates a new revision.
2. **Problem:** reference the state schema, known RHS and available trajectory IDs.
   Observation data can be reused while the known equations or unknown RHS change.
   The existing all-in-one prepare call should remain a convenience operation.
3. **Scoring view:** explicitly select trajectory IDs, scored states, observation
   points/time coverage, aggregation and integration policy. Persist an immutable
   view ID. Include it in every score identity; scores from different views are
   not interchangeable. Cache device data by content/view with bounded residency.
4. **Search job:** bind a grammar/population to one scoring view and retention
   policy. All trajectories share the same candidate coefficients, sampled values
   and state assignments by default. Experiment-specific parameters need a future
   explicit scope; never introduce them implicitly by splitting a job.
5. **Frozen-candidate evaluation:** name retained candidate IDs and a scoring view,
   then request aggregate, per-trajectory and optionally per-state diagnostics or
   predicted observation samples. Do not regenerate RNG, reselect toggles or fit
   coefficients. Evaluate full retained models even when only a subset of states
   is scored. Predicted trajectories are a separate optional output from MSE.

Illustrative future request, NOT an implemented MCP tool/schema:

```json
{
  "candidate_ids": ["candidate_42"],
  "scoring_view_id": "validation_view_3",
  "outputs": ["aggregate", "per_trajectory", "per_state"],
  "refit": false
}
```

Selection must not accidentally make integration easier. Removing observations
does not remove the physical time between them. Later scoring windows must evolve
from a complete initial condition, or require a separately supplied full restart
state. Never initialize a hidden state from unavailable observations. Changing
subsets can change the currently resolved common RK4 subdivision count; diagnostics
of an existing score should preserve that effective count. A new integration policy
or independent finer/FP64 check gets a separate score identity and label.

Held-out trajectory views must be explicitly chosen by the caller. Keep search,
validation and final-test use distinguishable; do not tune a search on the final
test while presenting it as unseen verification.

## MSE semantics and execution options

For trajectory j, record squared-error sum `SSE_j`, observed noninitial scalar
count `N_j`, `MSE_j = SSE_j / N_j`, and numerical validity/status. The current
pooled objective is `sum(SSE_j) / sum(N_j)`. This weights longer/more-observed
trajectories more heavily. Equal-trajectory weighting would instead average the
individual MSEs. Weighted/normalized objectives must be explicit future options;
retain raw components and do not change the current default silently.

Zero scored observations means no MSE (`null` plus count/status), not perfect fit.
Numerical failure on a required trajectory invalidates the aggregate according to
the declared policy; do not drop a difficult trajectory and average the remainder.
Do not confuse no observations with failed integration. FP32 partial sums and
different reduction orders may produce small aggregate differences; test with
declared tolerances and label CPU FP64 diagnostic scores separately.

Recommended modes:

- **Aggregate screen (default):** existing scalar per configuration, native
  reduction, requested structure/family/tag winners. Avoid an N-candidate by
  T-trajectory score array during a large search.
- **Retained diagnostics (first addition):** rescore K frozen winners across T
  trajectories, producing K*T MSE values. An initial implementation can use the
  existing scorer with one-trajectory launch inputs or the existing CPU reference.
  The current pipeline may specialize/load again per call, so measure this route;
  it is not a zero-overhead batched GPU diagnostic implementation.
- **Native trajectory partials (optional later):** a clean numerical output mode
  emitting candidate-by-trajectory SSE/count/status or bounded group partials.
  Compare it with retained rescoring; request only when the diagnostic/Pareto/
  robust objective needs it. Keep grammar tags and report policy outside the core.
- **Independent trajectory leaderboards (explicit):** preserve a top-k for each
  requested trajectory during scoring or run separate objective jobs. The global
  aggregate top-k cannot reconstruct these leaderboards afterward. Their best
  coefficients and even structures can differ; this does not establish a single
  model fitting every experiment.

For scale, one million candidates and 100 trajectories produce 400 MB of raw
FP32 MSEs alone; 20 retained candidates and 100 trajectories produce 8 KB.
This excludes all metadata/workspace. Optional population-wide trajectory output
needs explicit memory/output budgets. Independent uploads should still combine
into substantial native batches; one upload must not imply one GPU launch.

## Ordered review checklist

- [x] **P0: Attribute host and device time.** Produce a cold/warm stage profile on
  the same prepared population and trajectories, with counts/bytes and identical
  score/retention validation. Do not optimize an assumed 6.75-second cause.
  [Completed measurement](../../benchmarks/grammar_overhead/REPORT.md): new-context
  and warm runs, unchanged disk caches, scoped timings and CUDA trace; a forced
  cold-NVRTC cache purge was not performed. Optimization remains open.
- [ ] **P0: Specify compact identity and wire formats.** Persist stable ordinals,
  native-to-grammar address mapping, exact coefficient snapshots and LM results;
  add uint64 JSON strings and tests beyond 2^53. Benchmark gather versus regenerate.
  Ordinals, manifests, string addresses, exact-snapshot reconstruction and width
  tests are implemented; finalization/gather benchmarking remains open.
- [ ] **P0: Protect retention semantics.** Verify overlapping groups, duplicates,
  late tags, ties, partial budgets and k>native-k while reducing reconstruction
  cost. Keep independent family/tag floors before truncation.
- [ ] **P1: Normalize result delivery.** Version candidate dictionaries, leaderboard
  references, optional readable summaries and paged/full snapshot retrieval.
  Versioned compact dictionaries/leaderboard references and full snapshot replay
  are implemented. Pagination remains open.
- [ ] **P1: Add trajectory assets and immutable scoring views.** Retain the current
  prepare convenience call. Test missing observations, full IC requirements,
  changed state order, trajectory revisions, subsets and cross-view score identity.
- [ ] **P1: Add frozen-winner trajectory diagnostics.** Return SSE/count/MSE/status
  per trajectory and optional per state; test pooled reconciliation, failures,
  masks, unequal sample counts and preservation of exact scored coefficients/RK4.
- [ ] **P2: Benchmark optional native partial-loss output.** Vary candidates,
  trajectories, state counts and rollout length. Compare retained rescoring,
  full/blocked partials and aggregate-only throughput/memory before choosing ABI.
- [ ] **P2: Design independent-trajectory/robust ranking deliberately.** Specify
  per-trajectory top-k, alternative weighting and any experiment-specific parameter
  scope separately from diagnostic reporting. Benchmark useful search outcomes.
- [ ] **P2: Validate larger prepared structural populations.** Include the supplied
  million-structure example and realistic rollout lengths, reporting incomplete
  runs honestly; the existing million-configuration tests do not cover this case.

Acceptance requires unchanged address/RNG semantics under chunking, matching
retained winners under the requested uniqueness rule, honest partial/failure
accounting, and no grammar/GP policy added to the C99 numerical core.
