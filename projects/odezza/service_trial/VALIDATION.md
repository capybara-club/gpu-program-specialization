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

# Service trial validation — 2026-09-12

The live path is mac1 standard-library client → authenticated SSH tunnels → mac3
C admission + memory-backed NATS work queue → rack1 C runtime → native report.
Python on GPU hosts is used by test harnesses and the process-only supervisor,
not for live request parsing, AST expansion, scoring or retention.

## Passed checks

| Check | Evidence / scope |
|---|---|
| Installed NATS versions interoperate; broker binds localhost | mac3 server 2.14.6 / C client 3.13.0; rack1 C client 3.12.0; project configuration validates |
| Chunked 1.3 MiB request preserves exact bytes; duplicate admission is idempotent | [Protocol tests](validation/protocol.json) |
| Wrong/stale attempts cannot write; expired lease permits replacement | Real broker and C service, synthetic worker claims in protocol tests |
| 2 MiB report can upload while status responds; completion permits immediate release | Largest observed status call 5.1 ms in the final protocol run |
| Effective budget above 600 seconds and smaller grammar limit | Protocol checks: 1,800 s remains 1,800 s; a 17 s grammar cap wins; 3,601 s is explicitly rejected |
| Ninth outstanding job rejected without displacing eight accepted jobs | Protocol tests; jobs released after the check |
| Queued and active cancellation | Protocol tests and [real GPU cancellation](validation/recovery-and-dual.json); active cancellation 0.281 s |
| Constants change between attempts without stale data | [Five real-CUDA runs](validation/input-lifetimes.json), including repeated and changed grids |
| A failed constant upload cannot poison the next request | [Injected CUDA failure](validation/upload-failure.json): first attempt quarantined, second same-runtime attempt rejected without scoring, fresh runtime recovers the correct coefficient |
| Existing native API still works | [15 native conformance reports](validation/native-conformance.json): masks, irregular times, toggles, RNG reconstruction, chunking, retention/replay and error contracts |
| Twelve concurrent queued jobs equal direct C | [Queued runs](validation/queued-gpu.json) versus [references](validation/direct-reference.json); exact candidate records, family counts and leaderboards |
| Million-system request equals direct C on one or two GPUs | [One GPU](validation/million-queued-single.json), [two GPUs](validation/recovery-and-dual.json), [final supervised run](validation/final-supervised-dual.json) |
| Actual killed worker replaced without stale result acceptance | [Another executor takes over](validation/recovery-and-dual.json): two attempts, ~18 s including lease expiration |
| Sole two-GPU worker automatically restarts after an in-flight kill | [Supervisor recovery](validation/supervised-recovery.json): fresh child/context, two attempts, exact winners, 17.76 s |

The protocol tests use synthetic terminal reports and are not numerical tests.
The GPU comparisons use controlled fixtures, not unknown-RHS recovery campaigns.
All requests/results used for comparisons are retained in this directory. The
service itself stores live customer requests/results only in memory.

## Timing interpretation

[Direct old cache policy](validation/million-direct.json),
[direct attempt-owned inputs](validation/million-direct-attempt.json),
[one-GPU service](validation/million-queued-single.json), and
[two-GPU service](validation/recovery-and-dual.json) use the unchanged
[million-system/2,048-bank JSON](validation/million-2048-request.json).

Median direct times: 2.836 s with old numeric cache policy, 2.824 s with fresh
attempt inputs. Median mac1-to-report times: 2.897 s on one 5080 and 1.612 s on
two 5080s. Three repetitions each; the first direct run's cold compilation is
excluded. The grouped worker's first client measurement includes some startup
queueing; later runs and the native timers are separately available.

The one-GPU queued median admission is 8.1 ms and result retrieval 4.7 ms.
Client completion polling is every 20 ms. The residual is not a pure network or
GPU-compute timer: it includes queueing, polling and run-to-run variation.
The native report still combines specialization/loading/execution in its pipeline
timer. No new claim of separate CUDA arithmetic timing is made here.

The bank is 8 KiB; re-upload cost is below these runs' variation. Large grids and
longer/more numerous trajectories require their own matched measurements. These
six-state systems score just four RK4 steps and 12 state values/configuration.

## Material deviation found and repaired

**Worker drain race.** During the first switch from individual GPU workers to
a two-GPU worker, a worker already sent SIGTERM consumed a delivery from its
outstanding pull. Its control thread then cancelled that new job. This produced
a valid cancelled report, not the intended dual-GPU completion. No dual-GPU
throughput conclusion uses that attempt.

- Evidence: [cancelled request](validation/draining-worker-failure.json) and
  [first test's earlier successful stages](validation/recovery-failure.json).
- Repair: check shutdown after pull returns and return that delivery; graceful
  shutdown finishes an already claimed attempt rather than cancelling it. The
  deployment/test helper waits for old workers to finish draining.
- Regression: the same cancellation/loss/drain/group test now passes, including
  three full two-GPU jobs with exact direct-C winners. [Rerun](validation/recovery-and-dual.json).

**Input ownership change.** The service selects the explicit `attempt` policy;
successful legacy callers default to `runtime_cache`. Results expose the policy.
Retaining raw capacity does not retain an input identity; explicit constants and
Philox banks are uploaded/generated afresh for the next attempt. Comparisons
above measure this change on the small benchmark bank. The failure-injection
repair is in updated source, not retroactively in older running service binaries.

**Pool contract clarified and implemented, later 2026-09-12.** The user requires
pooling for trajectory inputs and MSE/result buffers, and explicitly accepts
small handle allocations and pipeline-owned streams/events. The current trial
reserves those bulk slabs before accepting work. Grammar/compilation workspaces,
numeric-bank storage and retained candidate records may still allocate. The
earlier full-preallocation/cross-job-stream requirement is superseded. No core
kernel or native-core resource interface was modified.

## Still unproven

Public tenant isolation/authentication, all-worker memory reservations, huge-bank
transfer cost, complete accounting of work lost between heartbeats, long service
soaks, automatic capability routing, and reliable durable recovery after mac3
restarts. A memory-only service generation is intentionally lost on restart.
The current direct mac3–rack1 tunnel has passed an independent-client check with
both mac1 relays closed; mac1 is no longer a service dependency. rohini/ada have
not joined this topology. See the main [remaining-work tracker](../scratch/structural_search_trial/TODO.md).

## Timing follow-up: overhead attribution

Saved ordinary one-GPU queued runs spend 29.8–56.0 ms outside `odz_job_run`.
The paired middle run is 2.896916 s client elapsed versus 2.852505 s native,
leaving 44.4 ms for admission/transport, claim/fetch, report creation/commit and
completion polling/result retrieval. Admission (~8 ms) and retrieval (~5 ms)
are measured separately. The service `queue_seconds` timer begins at request
reservation and therefore overlaps admission; do not add it as an independent
stage. Benchmark polling is 20 ms; the CLI `wait` loop currently polls at 250 ms.

The final single two-GPU check's native time was 1.591998 s, versus the earlier
three-run median 1.549632 s. Its largest timer increase is the combined pipeline
stage (specialization/loading/execution). Those stage times sum across devices
and overlap other work. They cannot identify a pure GPU arithmetic regression,
and one slower sample is insufficient to assign a cause.

Historical finding, now repaired: the worker used to hold its lock during a
heartbeat RPC. The control/lease changes and validation below resolve that
coupling. The original inspection did not establish it as the cause of the
44 ms residual; the new measurements likewise are not a controlled attribution.

## Bulk pools and client-independent transport

- [Ten real-CUDA pool checks](validation/buffer-pools.json): one/two GPUs,
  repeated requests, exact legacy winners, CPU guard bytes, short CPU/GPU pools,
  and bounded tiling with preserved configuration counts. The literal-RHS tests
  measure zero explicit CUDA allocations after startup; this is not a claim
  about internal driver allocation or requests creating new numeric banks.
- [Twelve queued comparisons](validation/pooled-direct-network.json): exact
  prior reference candidates, family reports and leaderboards with the new pools.
- [Full million-system request](validation/pooled-million-dual.json): all
  2,048,000,000 configurations completed with exact reference winners on two GPUs.
  Native 1.551 s / client 1.579 s is one functional check, not a controlled
  repeated timing comparison during the simultaneous hardware-move backups.
- [Existing native conformance](validation/pooled-native-conformance.json): all
  15 checks passed, including masks, irregular times, replay and error recovery.
- [Pooled worker recovery](validation/pooled-supervised-recovery.json): kill an
  actively claimed worker locally on rack1; supervisor replacement completes the
  same job on attempt two with exact reference results in 16.64 s.
- [Cancellation and reuse](validation/pooled-cancel-reuse.json): cancel after
  444,596,224 configurations, then reuse the same worker pools for an exact-reference
  request. No quarantined runtime or stale-input result.
- [Client independence](validation/without-mac1-relay.json): both old mac1 SSH
  relay tunnels closed; a client on rack1 submits and retrieves a verified job
  through mac3's directly supervised connection. mac1 was used only to start the
  remote test command, not to transport any NATS request or GPU job traffic.
- Existing timing runs above precede these changes. Concurrent hardware-move
  backups use the network/disks; new functional timings during those transfers
  must not be presented as an isolated speed comparison.

The first new restart check sent its kill through mac1's SSH connection while
the short job was already finishing. The job committed successfully before the
kill arrived, so it did not exercise in-flight recovery. Its attempt-count check
failed and [that run](validation/pooled-recovery-missed-kill.json) is excluded from
recovery evidence. The passing repeat kills locally after observing a live claim.
A preliminary cancellation inspection also waited on the wrong status field and
observed a completed job; only the observed-work cancellation/reuse test above
is cited as cancellation evidence.

## Worker control and integration limits

Deployed to the current mac3/rack1 service on 2026-09-12, after verified rollback
archives. Live generation: `e2b73994485a30dd55a1561a81108f51`; mac3 orchestrator PID
5272, rack1 supervisor PID 419787 / initial child 419788. These PIDs are a snapshot,
not stable service identities. Core kernels and shapes were not changed.

| Check | Evidence/result |
|---|---|
| Pending RPC cannot hold up job attachment/destruction; stale replies cannot cancel new jobs; expiry cannot be revived; queue progress uses owned subject outside lock | [5 gated control cases](validation/control-races.json), ASAN-enabled harness, largest observed lock acquisition 1.06 microseconds |
| Final admission, exact work boundaries, oversized point count, result release, stale attempts, lease replacement, normal control concurrency | [Real NATS/C protocol tests](validation/control-protocol.json); largest status call 2.13 ms |
| 150 ms disconnect during 65.536B configurations | [GPU fault campaign](validation/control-gpu.json): same attempt/process, identical baseline results, 15.97 s vs 15.85 s baseline |
| 6 s disconnect while GPU job is active | Same campaign: retired process, attempt two, identical baseline results; 31.84 s including outage, server lease expiry and replay |
| Cancellation after observed GPU progress | Same campaign: 0.274 s to cancelled report; subsequent requests reuse pools correctly |
| Direct native API cannot bypass step/subdivision/point limits | [6 real-runtime cases](validation/native-work-limits.json): rejects before scoring/specialization, then boundary and ordinary requests succeed |
| Explicit huge configuration ceiling and many initial-only trajectories | [Final tile work tests](validation/work-gpu-final.json): all configurations scored, at least two chunks each; configs × max(steps, points) stays within 67,108,864 |
| Final grammar/RNG/toggle/family compatibility | [6 exact-reference queued cases](validation/control-final-conformance.json) |
| Deployed normal endpoint | [Live check](validation/control-live.json): oversized request rejected with zero attempts; three exact 2.048B-config runs in 1.620, 1.570, 1.577 s; native 1.590, 1.548, 1.539 s |

Frontend C99 tests also pass, including the new allocation-free preflight versus
prepared-layout test, masked/irregular trajectories, FP32 timestamp collision,
overflow, caller limits, existing 18,000 mutation checks and the allocator-symbol
audit. See [frontend test log](validation/control-frontend-tests.log).

The fault campaign deliberately repeats the original constant values 32 times
to keep work active long enough for lease expiry. Its 65.536B count is actual
configuration evaluations, not that many distinct numeric candidates or a blind
recovery result. Both baseline and fault requests are identical. Fault testing
preceded the additional point-visit guard; the worker control source is unchanged,
and the final guard was tested separately (including live exact-reference runs).

The defaults are explicit work ceilings, not a GPU wall-time proof. One global
subdivision count covers all intervals. Single-point trajectories and masks do
not bypass point-visit accounting. Smaller tiles can cause more preparation;
the short four-step throughput workload retains its prior effective shape.
The live timing is a functional/performance check, not a controlled old/new
speedup experiment. Broader campaigns, state/AST cost weighting and resource
isolation remain open.

[Test-harness corrections](validation/control-test-deviations.json) preserve the
initial missing NATS initialization, the too-short fault workload that recovered
on the same process, and a mistaken list lookup into the candidate-id mapping.
None of these preliminary checks is claimed as the final passing evidence.
