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

# Native request-path audit — 2026-09-12

Update, 2026-09-12: the separate [distributed trial](../../service_trial/VALIDATION.md)
now includes a tested upload-failure quarantine repair and explicit attempt-owned
numeric inputs. The historical reproduction below remains valid for its recorded
build. Older running services were not replaced. Legacy Python control/deadline/
completion defects remain open; the new C service has separate control and result
commit paths. See the [active tracker](../../scratch/structural_search_trial/TODO.md).


Scope: the recent native JSON → prepared grammar → generated AST → cached
scoring pipeline → GPU reduction → retained report path, including its Python
service/supervisor. This is a focused source review and failure-path audit, not
an exhaustive new audit of the core instruction specializer or numerical solver.
Runtime implementation was not changed during this audit.

The successful throughput measurements remain valid for their tested workloads.
The findings below restrict claims about recovery after errors and mixed-load
service robustness. In particular, fault tests that kill a whole worker did not
cover a recoverable upload failure followed by another job in the same worker.

## P1 — Failed grid uploads can poison subsequent requests (reproduced on GPU)

`runtime/odz_gpu.c:275–280` increments the pool allocation accounting and installs
its lookup key before copying the constant grid. If that copy fails, the job
fails, but the entry remains discoverable and the runtime is not quarantined.
The next job can hit that entry at lines 250–251 and use incomplete data.

An isolated rack1 process used the real CUDA runtime with a test-only preloaded
shim. It zeroed one 65-float destination and returned an upload error instead of
copying the requested grid. This deliberately makes reuse visible; no production
process, driver or data was modified.

| Request | Result | Best coefficient | MSE |
|---|---|---:|---:|
| First, injected grid upload failure | Failed; not quarantined | — | — |
| Same request, same runtime after failure | **Complete, wrong requested population** | **0** | 0.0114344805 |
| Same request, fresh runtime | Complete | 1 | 8.8817842e-15 |

The requested grid was `[0.5, 0.515625, …, 1.5]`; zero was not a permissible
coefficient. This is an input/provenance correctness defect even though the
reported equation accurately describes the wrong coefficient that was scored.
Only the injected test is known affected; no evidence indicates an upload error
preceded the earlier performance runs.

**Repair:** publish a pool entry only after successful initialization and an
appropriate completion dependency. On failure, retire/invalidate the provisional
entry, or quarantine when completion/ownership is uncertain. Audit neighboring
CUDA error paths too: buffer replacement ignores `cuMemFree`'s return code at
`runtime/odz_gpu.c:45`. Existing poisoned workers must be restarted after repair.
Until repaired, use a fresh worker after an upload failure; do not trust its
subsequent requests. Turn this reproducer into a passing recovery regression.

Evidence: [GPU reports](runtime-validation/odezza-request-audit-upload.json),
[driver-error shim](../../tests/runtime/audit_upload_failure.c),
[reproducer](../../tests/runtime/audit_failure_paths.py).

## P1 — Reporting can block service control and heartbeat

The lightweight C status snapshot is useful, but `NativeService._report` holds
one Python service lock across the complete C measure/write and JSON decode
(`python/odezza/grammar/native_service.py:224–247`). Status, cancellation,
submission, expiry and heartbeat collection need that same lock. C full reports
also take the parsing/retention mutex (`runtime/odz_report.c:127`) and construct
their union of candidates with a quadratic identity scan at lines 179–190.

The actual Python methods with a gated native report confirm that an otherwise
immediate status call cannot finish until the full report releases its lock.
This is a controlled concurrency reproduction, not a measured GPU stall.

In addition, the worker performs RPCs synchronously in the loop that emits
heartbeats (`python/odezza/grammar/native_supervisor.py:33–51`). Long full reports
or optional CPU replay can prevent heartbeat emission for the 30-second watchdog
period. That can cause the parent to kill healthy CUDA work and lose all resident
job handles. The false-watchdog consequence is established by source tracing;
this audit did not run a 30-second slow-replay kill experiment.

**Repair:** isolate heartbeat/control liveness from expensive report/replay work;
pin job lifetimes without holding the global registry lock across serialization.
Bound report work and test concurrent results/status/cancel/release. C parsing
also holds the job mutex and delays cancellation; avoid routing a blocking cancel
through the only heartbeat loop.

## P1 — Requests above ten minutes receive the wrong supervisor deadline

`SupervisedNativeService.submit` starts its deadline minimum with `[600.]`
(`python/odezza/grammar/native_supervisor.py:242`). C instead uses 600 as a
**default**, allows an explicit execution limit to replace it, and only then
applies a smaller grammar limit (`runtime/odz_job.c:78–83,185`).

The real Python submit/monitor methods, with a mocked RPC and clock progress,
confirm that an explicit 1,800-second request gets a 600-second parent deadline
and is killed at simulated elapsed time 611 seconds. A 30-minute GPU run was not
needed or performed. Earlier sub-600-second benchmarks are unaffected.

**Repair:** use the same effective deadline as C, preferably expose that parsed
value in status so policy is not implemented twice. Test omitted limits, larger
execution limits, smaller grammar limits and initialization separately.

## P2 — Terminal status and releasability disagree briefly

C publishes completion before Python finishes formatting the automatic final
report and completing the executor Future. Status can therefore say `complete`
while `release` returns `job_active` because it checks `Future.done()`
(`native_service.py:126,207–222`). The controlled reproduction exercises these
actual methods with a terminal native handle and an unfinished Future.

**Repair:** define a consistent finalization/release contract and test immediate
release after the documented wait/status interface. The Future also retains a
second full Python report while later result calls serialize the C report again;
cache/reuse or eliminate that duplication deliberately.

CPU evidence for the preceding three findings:
[control/lifetime results](runtime-validation/odezza-request-audit-controls.json).

## P2 — Request memory budgets do not reserve all live allocations

The documented limits are not total RSS/VRAM caps. Beyond that declared scope,
there are concrete accounting gaps worth fixing:

- Tile descriptor arrays, packing and pipeline host workspaces are allocated
  outside `max_host_bytes` accounting (`runtime/odz_execute.c:133–135,193,219`).
- Reduction scratch is checked against remaining host bytes but not reserved
  there (`odz_execute.c:244–255`). Concurrent GPU workers and growing retained
  candidates can each consume space the others still regard as available.
- Device buffers retain independent high-water capacities and replacement
  allocates the new buffer before freeing the old (`odz_gpu.c:38–48`). A current
  tile's size estimate is not the worker's actual resident/peak allocation.
- A smaller later `max_bank_bytes` does not evict existing cached entries on a
  cache hit. Eviction runs only when inserting a new entry (`odz_gpu.c:250–260`).

These are source-reviewed gaps; no OOM was deliberately induced. The parent RSS
watchdog provides coarse containment but does not measure device VRAM.

**Repair:** separate documented tile requirements, persistent cache budgets and
process limits; reserve aggregate host scratch under the job lock; account and
report resident/peak GPU bytes. Enforce a lowered bank budget even on cache hits.

## Performance compromises still used by normal requests

1. **Distinct top-k uses repeated top-16 reductions.** The resident reducer is
   created with k=16. When a request needs more distinct numeric winners, it
   removes those rows and scans again (`odz_execute.c:257–328`). With B identical
   finite bank rows and requested k=2, this can require approximately `ceil(B/16)`
   passes just to establish that a second distinct winner does not exist. Existing
   duplicate-bank tests cover correctness and partial timeout, not an efficient
   general solution. **Direction revised 2026-09-12:** the user prefers optimizing
   pure CUDA row top-k, keeping expression identity out of the kernel. The earlier
   uniqueness-aware-kernel recommendation is superseded by item 2 in the active
   TODO. The existing repeated-pass behavior remains until an explicit row
   retention contract is implemented; do not silently redefine existing units.
2. **Tag rankings can retain much more than final top-k.** Late duplicate tags
   require a finalist archive per accepted AST/permutation until generation ends.
   This is intentional and preserves semantics, but arbitrary rich tag requests
   can consume substantial host memory. It is not strictly final-top-k storage.
3. **Warm templates do not mean persistent scoring sessions.** Every job destroys
   scoring sessions (`odz_job.c:408–413`). Each new job recreates the pipeline and
   prespecializes fixed equations. Bank-split tiles can specialize/load the same
   ASTs again. Enlarged templates are cached, but a new request starts from its
   original requested patch size and can repeat smaller-capacity attempts before
   reaching an already cached larger shape (`frontend/odr_scoring.c:234–263`).
4. **GPU event ordering still includes host waits.** Numeric generation/prelude
   and reduction complete before the next tile proceeds on the same GPU. The
   normal path has event dependencies, but that alone does not create overlap
   across tile stages. Different GPUs and core module-loader workers do overlap.
5. **The auto scheduler is a heuristic.** It adjusts configuration tiles, not
   producer or module shapes. Its first tile has no measured cost and a 50ms
   target is advisory. No scoring-register-pressure shape fallback was added;
   patch-capacity growth solves a different problem.

These compromises generally preserve the requested scoring semantics on a
successful complete run. They can materially change latency, memory use and
coverage before a deadline. They should not be described as universal peak
utilization.

## What the successful path actually does

- No evidence of a per-request hand-edited AST population, Python expansion or
  CPU scoring fallback was found in the native scoring path. Python handles
  request transport, serialization, human-readable equations and explicitly
  requested CPU winner replay; C produces and scores ASTs and retains winners.
- Frontend arena phases use a measure pass followed by a build pass. “Parsed
  once” means prepared once for generation, not literally one traversal of JSON.
- Known-RHS specialization, C-owned binary template caching and capacity growth
  are intentional API behavior, not correctness bypasses. Growth can repeat work;
  reports expose growth/retry counts.
- The recent 715.6M/839.7M configurations-per-second comparisons measured warm
  **in-process C job time**, including parsing, generation, scoring and retention.
  They used one short trajectory, four RK4 steps/configuration and explicit
  `batch_variants=1024`, `module_systems=32` (defaults are 256/64). They exclude
  network/MCP, process/context startup and CPU replay; those rates are not a
  universal default-request throughput guarantee.
- Earlier 191-check GPU suites and short release soaks remain useful evidence.
  They did not establish the failure-continuation and mixed control/report
  properties exposed by this audit. No runtime repair is claimed here.

## Repair order

1. Transactional GPU pool initialization and CUDA error/lifetime review.
2. Independent heartbeat/control, unified effective deadline and terminal release.
3. Aggregate memory reservation plus failing-upload and mixed-load regression tests.
4. Reducer uniqueness, session/capacity reuse and deeper stage attribution.
5. Only then expand long campaigns and release qualification.
