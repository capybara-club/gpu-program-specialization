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

# Shared memory accounting — 2026-09-12

Implemented in the optional C runtime and deployed to the current mac3/rack1
service. The hardened kernel generator, scoring kernel, reducer and core API were
not changed by this work. There is no customer-data database.

## Enforced contract

| Resource | Policy |
|---|---|
| Host memory controlled by a job | One reservation ledger across the producer and all GPU threads. Reserve before allocating request/arena/retained/tile/pipeline-workspace storage. Charge allocation headers and transient resize overlap. |
| Borrowed bulk CPU storage | Charge bytes used by the job; report complete startup capacities separately. A charge is not another allocation. |
| Operator host ceiling | `odz_runtime_host_limit`, called before any job. Current service default 2 GiB, configurable with `--job-host-limit-mib`. Larger requested budgets fail before scoring, without silent clamping. |
| Runtime numeric pools | Trim to `max_bank_bytes` at job entry and before binding, including cache hits and jobs with zero numeric slots. The cap applies per GPU. |
| Score/coefficient/reducer buffers | `max_device_bytes` bounds the selected tile. Unpooled retained capacity is trimmed for tighter limits; fenced old buffers are freed before growth. Service startup slabs remain fixed and separately advertised. |
| Opaque host overhead | External supervisor samples the worker's RSS every 100 ms. Default 4 GiB; configurable with `--max-worker-rss-mib`. Excess or unreadable RSS for a live child kills that process; existing attempt/replacement machinery handles failure. |

Small per-job handles and pipeline-owned streams/events remain allowed. The exact
ledger includes workspaces supplied by the runtime to the core, not allocations
inside the core/compiler/driver. Allocation failure stops the request, preserves
completed-prefix reporting and releases safely fenced work. Unfenced CUDA work
still quarantines the runtime until its process is replaced.

Host ceiling rejection currently happens in the worker before scoring, after
queue admission. It is not a new pre-publication resource scheduler. The direct
runtime has no operator ceiling unless its caller configures one; the C service
always configures it. Direct embeddings must supply their own process guard.

## Report interpretation

`memory.classes` contains live and peak reservations for request data, arenas,
retained records, tile scratch and pipeline workspace. `charged_peak_bytes` is
the peak simultaneous sum. **Do not add class peaks:** they need not coincide.
`denied_reservations` records budget failures. The small published status includes
the shared totals without walking retained candidates or acquiring the main job
lock. Report construction no longer allocates a candidate-pointer vector; ranking
semantics and deterministic IDs are unchanged.

Process RSS, sampled peak during the job and process lifetime high-water RSS are
separate measurements. Linux RSS sources are approximate and need not agree
exactly. Capacity reserved with malloc is not necessarily physically resident.
Per-device reports distinguish explicit allocations owned by this runtime from
sampled total device usage, which also includes driver/context/module storage and
other processes. Do not add these overlapping quantities. Failed early requests
may have unavailable samples; validity flags distinguish that from zero usage.

The RSS threshold permits transient overshoot between samples and is not a hard
whole-process reservation or a hard total-VRAM limit. Independent service worker
processes do not share this ledger or RSS allowance. Host-wide admission,
nonoverlapping GPU ownership and customer quotas remain separate service work.
This milestone does not establish public multi-tenant readiness.

## Verification

- `make -C runtime test-memory`: ASan/UBSan checks concurrent 48 MiB reservations
  against one 64 MiB cap, release, overflow, exact-cap borrowing, resize overlap
  and preservation of the original buffer when resizing fails.
- [19 runtime cases](../../service_trial/validation/memory-accounting.json): cached
  and attempt-owned banks, lower-bank-limit cache hits, zero-slot trimming,
  insufficient workspace budgets, successful reuse, operator rejection, unpooled
  large-to-small capacity changes, and four dual-GPU benchmark repetitions.
  The smaller unpooled request retained exactly the same 525,152 device bytes
  as a fresh runtime, down from 1,315,736 bytes after the larger request.
- [15 native conformance checks](../../service_trial/validation/memory-native-conformance.json)
  cover RNG reconstruction, toggles, late tags, distinct ranking, family budgets,
  fixed-program growth, cancellation and context reuse. Existing six integration
  work-limit and five control-lock checks also passed on the isolated build.
- [Injected upload failure](../../service_trial/validation/memory-upload-failure.json)
  quarantines the affected runtime; reuse rejects, and a fresh runtime succeeds.
- [RSS guard tests](../../service_trial/validation/memory-supervisor.json) cover
  normal exit, excess, unreadable RSS, and a real disposable child exceeding its
  test threshold. The test does not allocate gigabytes in the live worker.
- [Six live queued fixtures](../../service_trial/validation/memory-live-conformance.json)
  match prior candidate records, family coverage and leaderboards exactly.
  [Live memory failures and recovery](../../service_trial/validation/memory-live-limits.json)
  were submitted from mac1 through mac3: both failures scored zero configurations;
  the same rack1 worker then returned the exact reference result.

## Matched performance

The unchanged prepared workload is **1,000,000 candidate ASTs × 2,048 numeric
configurations = 2.048 billion evaluations**, using both RTX 5080s on rack1,
EAGER module loading, the same warm template cache, pools and reduction policy.
This is a short-rollout throughput fixture, not a general ODE search rate.

| Path | Measured time |
|---|---:|
| Previous direct runtime, median of three runs after one warmup | 1.5775 s |
| New direct runtime, same procedure | 1.5909 s |
| Deployed queue submission through result retrieval, median of three | 1.5975 s |

The direct difference is +0.85%, within the observed spread of these short runs;
this sample does not establish a meaningful regression or a speedup. Candidate,
leaderboard and family results match exactly. Sources:
[old baseline](../../service_trial/validation/memory-baseline.json),
[new direct runs](../../service_trial/validation/memory-accounting.json),
[live runs](../../service_trial/validation/memory-live.json).

The last live run peaked at **416,707,375 charged host bytes (397.4 MiB)**, including
up to 3,390,752 bytes of shared pipeline workspace. Its sampled process RSS peak
was 452,104,192 bytes (431.2 MiB). Each GPU had about 320 MiB of explicitly tracked
runtime allocations and about 1.39 GiB of sampled device-wide use. These are
different scopes, not missing bytes to add to the host budget.

## Deployment and rollback

The mac3 broker/orchestrator was left running. The idle rack1 worker was drained,
rebuilt and restarted under the memory-aware supervisor. Default limits are
2 GiB per-job charged host memory and a 4 GiB sampled worker RSS threshold.
Deployment source/binary hashes and process details are retained in
[the manifest](../../service_trial/validation/memory-deployment.json).

Rollback source and binaries are in
`rack1:/home/cdurham/odezza-checkpoints/20260912-memory-accounting/before-live.tar.gz`.
The final source/docs/receipts archive is copied to mac1, mac3 and rack1 with
SHA-256 verification. This remains a working-tree trial snapshot; a reviewed,
committed/tagged release is still outstanding.
