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

# Network client → mac3 → GPU service trial

Status, 2026-09-12: **running as a separate private trial**. The current deployment
includes aggregate admission, per-skeleton caps and measured automatic sizing.
[Version, paths, limits and live verification](../benchmarks/request_sizing/deployment.json). mac1 sends one JSON
request; mac3 admits it and queues a descriptor; the rack1 C executor fetches the
unchanged bytes, parses/expands in C, scores on CUDA, and returns the native report.
The live executor uses both rack1 5080s for each job. Separate one-GPU executors
sharing the queue also passed concurrent-job checks. Existing native/GP services
and the hardened kernel core were not replaced.

The [main README's request-limit reference](../README.md#current-json-service-request-limits-and-failure-behavior)
lists admission, grammar, trajectory, memory, kernel-fit and report failures,
including the distinction between rejecting a request and stopping at its search
allocation. It is the entry point for users preparing a request.

**Bulk buffers are reserved at worker startup.** Each GPU has a fixed 64 MiB
trajectory slab, 256 MiB scoring/coefficient/reduction slab and 64 MiB CPU result
scratch slab. The worker also owns one 64 MiB CPU trajectory arena plus reusable
request/report buffers. Jobs borrow the request and trajectory arena until job
destruction. Oversized trajectories fail explicitly; scoring tiles shrink to fit
fixed pools and retain all requested work. No fallback allocation enlarges these
buffers during a request.

Scoring handles, pipeline workspaces, streams/events, grammar arenas, numeric-bank
storage and compact retained records may still allocate. The user explicitly
accepted per-job scoring handles and pipeline-owned streams (2026-09-12); a shared
stream pool is no longer a requirement. This is bulk-buffer pooling, not a claim
of allocation-free execution. Contexts and helper modules/resources persist, and
numeric identities are invalidated between attempts.

## Integration work admission limits

The service rejects excessive rollout work **before queue publication or GPU
claim**. It validates the trajectory section and computes the same FP32 RK4
layout as the executor, without an arena allocation or grammar expansion:

```
subdivisions = max(1, ceil(largest FP32 observation gap / FP32 integration.dt))
steps_per_configuration = subdivisions * sum(points_per_trajectory - 1)
```

The current kernel uses that subdivision count for every interval, including
shorter irregular gaps. All trajectories and masked intervals still participate
in integration; masks only change scoring. Limits apply to actual loop work.

- At most **65,536 trajectory points per configuration**, including initial points.
- At most **4,096 subdivisions per observation interval**.
- At most **65,536 RK4 steps per configuration**, summed across trajectories.
- A **device-aware tile envelope**: the larger of 67,108,864 divided by
  `max(RK4 steps, total trajectory points)` and two hardware thread-capacity waves
  rounded to a power of two. The latter is bounded at 4,194,304 configurations
  (262,144 on RTX 5080). Counting points includes initial-only trajectory work.
  Explicit configuration limits and memory pools can further reduce the tile.
  The runtime never changes dt, truncates trajectories or omits configurations
  to satisfy this policy. An indivisible toggle product that cannot fit is rejected.
  This replaces the fixed aggregate cap that starved long-rollout launches.
  The 50 ms target remains advisory; submitted kernels are not preempted on cancel.

`health.integration_limits`, status `integration_work`, and native report
`integration_limits` expose the policy and counts. Rejected submissions return a
terminal failed handle/report with `integration_work_limit`, zero attempts/work,
required counts when representable, and the limits. Both raw RK4 steps and point visits must fit the advertised tile ceiling. Overflowed layouts explicitly
mark work counts unavailable. A smaller `grammar.integration.max_steps` also
applies. The native runtime independently enforces the same ceilings, including
callers that bypass the network service.

These are initial safety ceilings in `runtime/odezza_runtime.h`; requests cannot
raise them. They bound work, **not elapsed seconds**: state count, AST complexity,
GPU type and interference still affect duration. The tiny four-step throughput
fixture keeps its original configuration ceiling. Longer rollouts may use more
tiles/module preparations; that is a visible safety/performance tradeoff.

## Aggregate allocation admission

The orchestrator also checks summed family reservations, rollout work and host
reservations before publication. [The main limit table](../README.md#aggregate-service-admission)
lists defaults, environment overrides and failure codes. `health.admission_limits`
is authoritative for the running generation; status `allocation` describes this
request. The allocation helper is shared with the worker frontend. It does not
expand ASTs, and the GPU host still performs full grammar preparation.

Admission's host estimate is a lower bound (JSON plus family dedup arenas), not a
replacement for the worker's complete measured reservations. Declared ceilings
can exceed a finite grammar's actual size; unused allocations are not redistributed.
Changing a broker/orchestrator generation still loses its RAM-only jobs/results.
Drain and export them before restart.

## Worker control and cancellation

One control thread renews the current attempt lease and reads the small runtime
snapshot. It copies attempt identity and the queue acknowledgement subject under
a short lock, then performs heartbeat and progress-publication I/O with that lock
released. No job/delivery pointer is used after unlocking. Late replies are
applied only to the same still-active attempt. The runtime's existing small,
thread-safe cancellation signal remains independent of report/preparation locks;
no network or lease policy enters the kernel core.

Worker NATS connections use disconnect/reconnect/closed notifications, bounded
reconnect attempts and **no disconnected publish buffer**. New claims require a
connected transport. A transient disconnect may recover within the worker's
five-second application lease. An acknowledged continuation extends that lease
from the heartbeat's send time, not from a delayed response's arrival. Once
cancelled/retired, reconnect cannot revive the attempt. Heartbeat RPCs have a
250 ms timeout and run approximately every 250 ms plus RPC time.

Mac3 uses a 15-second lease and checks expiry on attempt operations as well as
its housekeeping timer. Expired attempts cannot renew or commit. Queue progress
uses the documented `+WPI` acknowledgement on the copied delivery reply subject;
it does not grant application ownership. The outer supervisor replaces a failed
worker. Broker/orchestrator generations remain separate; a broker restart is not
transparent recovery of the in-memory job store.

Cancellation stops new work at safe runtime boundaries and waits on completion
events for already submitted CUDA work. A host signal cannot preempt a running
kernel. A ten-second cancellation grace bounds worker retirement; execution
budgets remain enforced separately from healthy heartbeat traffic.

## Use from any machine with SSH access to mac3

Open a client connection directly to mac3 (current private address shown):

```sh
ssh -fNT -o ExitOnForwardFailure=yes -L 127.0.0.1:14222:127.0.0.1:4222 cdurham@192.168.3.185
```

This tunnel belongs only to that client. Closing it does not disconnect the GPU
worker or cancel an admitted job. Reconnect and use its handle to retrieve results.

From the repository root:

```sh
python3 service_trial/client.py health
python3 service_trial/client.py submit examples/grammar/submit.json
python3 service_trial/client.py jobs
python3 service_trial/client.py status GENERATION.JOB_ID
python3 service_trial/client.py wait GENERATION.JOB_ID
python3 service_trial/client.py result GENERATION.JOB_ID
python3 service_trial/client.py cancel GENERATION.JOB_ID
python3 service_trial/client.py release GENERATION.JOB_ID
```

`submit` returns a handle immediately after queue admission. The full request
contains `problem`, `grammar`, and optional `execution`, exactly as the existing
C frontend expects. The client does no grammar expansion or numerical work.
Reports contain native MSEs, candidates, configuration/permutation indices,
Philox provenance, family/tag leaderboards, counts and C stage timings. `status`
adds worker host/devices, queue time, attempts and observed work per attempt.
Observed retry work is a **lower bound**: a killed worker may have completed GPU
work since its last heartbeat. It is not a billing-grade count.

The Python client uses only the standard library and the basic NATS request/reply
wire protocol through the private SSH tunnel. Service admission and GPU execution
are C99 using the installed NATS C library. `supervise_worker.py` only restarts a
failed C process; it never parses jobs, expands ASTs or manages CUDA.

Python/library usage:

```python
from pathlib import Path
from service_trial.client import Client

client = Client()
try:
    submitted = client.submit(Path("request.json").read_bytes())
    print(submitted["handle"])
    print(client.status(submitted["handle"]))
finally:
    client.close()
```

One client connection is serial. Use separate clients for simultaneous calls.
A timeout is not confirmation of rejection. Retry admission with the **same**
job ID, generation and exact bytes to resolve an uncertain response. Changed
bytes under an existing ID are rejected. Released IDs cannot be reused within
that generation. A broker/orchestrator restart invalidates the generation; no
old result is silently replaced by a new job.

## Ownership and queue behavior

- mac3 has eight preallocated request/result slots (128 MiB of reserved virtual
  payload capacity), each accepting at most 8 MiB of JSON and 8 MiB of native
  report. Network chunks are at most 256 KiB. Admission rejects a ninth job until
  an existing slot is released/expired. There is no database or per-AST JSON.
- Request bodies and reports live in mac3 process RAM. JetStream uses a bounded
  **memory-backed** work stream containing only 32-byte job IDs. Jobs/result data
  are not written to a broker database. Local validation artifacts are test files,
  separate from the live service's storage contract.
- Each generation creates `ODZ_<generation>` and one shared `executors` pull
  consumer. Workers pull **one job** only when their C runtime is idle. A worker
  may own one GPU or a fixed device group; the C runtime splits a group's job.
  Workers must be configured with nonoverlapping physical device sets.
- Claims use independent attempt IDs. A 15-second expired lease allows a new
  claimant, up to three claims. A stale attempt cannot upload/commit a result.
  Progress acknowledgements keep a valid delivery alive. Terminal results are
  committed to mac3 before the worker acknowledges the queue message.
- A lost acknowledgement can cause another delivery, but a terminal job does
  not execute again. This is not exactly-once GPU execution: a lost worker can
  have performed work before the replacement starts over.
- Status/heartbeat handling has separate subscription threads from submission,
  upload, and report commit. GPU control reads the small C snapshot, not the
  full report. C cancellation no longer waits for the main preparation mutex.
- The service uses the same execution/grammar budget rule as C (execution default
  600 seconds, reduced by a smaller grammar limit). This trial explicitly rejects
  an effective budget above 3,600 seconds. Queue/upload waiting expires after
  600 seconds; the service adds ten seconds of execution cancellation grace.
- Completed slots expire after one hour or explicit release. A generation retains
  at most 4,096 admitted IDs as small tombstones. These limits are trial constants
  in `trial.h`/`orchestrator.c`, not a public configuration API yet.
- SIGTERM drains the C worker: finish the current attempt and stop pulling.
  A failed CUDA input upload quarantines the process; the supervisor replaces it
  with a fresh context. No failed device input is promoted to reusable data.
  Five short startup failures within 60 seconds stop the supervisor with an
  explicit circuit-breaker error.

The new runtime API `odz_runtime_attempt_inputs(runtime)` is selected before the
first run and applies to all devices. It reuses capacity without keeping numeric
input identities across requests. Native reports identify `numeric_input_scope`.
Successful legacy callers can still select the previous runtime-cache policy;
that lets the trial remain separate. The failed-upload repair also applies to
that mode in the updated source, but older running service binaries have not
been replaced.

## Deployment and processes

- mac3 checkout: `/Users/cdurham/code/odezza-service-trial`
- rack1 checkout: `/home/cdurham/odezza/scratch/service_trial_20260912/odezza-native-trial`
- Broker: mac3 `127.0.0.1:4222`; monitoring: `127.0.0.1:8222`.
- A mac3-owned SSH supervisor connects **directly to rack1** and forwards rack1
  `127.0.0.1:14223` to mac3's broker. The worker uses this port. mac1 has no relay
  role; the obsolete two-hop tunnels were closed and an independent rack1 client
  successfully submitted/retrieved an exact-reference job.
- The direct tunnel supervisor restarts a lost SSH connection. Existing SSH keys
  authenticate it, and rack1's verified host key is pinned in mac3 known_hosts.
  The broker and forwarded ports remain loopback-only. This is private SSH access,
  not an unauthenticated LAN/public NATS listener.
- mac3 tunnel log: `build/service_trial/tunnel.log`. Start on mac3 with
  `python3 service_trial/supervise_tunnel.py --worker cdurham@192.168.3.195`.
  mac1's optional client-only control socket is `/tmp/odezza-client-mac3.sock`.
- mac3 logs/PIDs: `build/service_trial/{broker,orchestrator}.{log,pid}`.
- rack1 supervisor log/PID: `build/service_trial/supervisor.{log,pid}`. Its log
  records current child PID and restart reason; the child handles devices `0,1`.
- No Homebrew launch-at-login service was registered. The processes were started
  independently with redirected logs. mac3 broker/orchestrator restart is manual
  and creates a new service generation. Rack1 child restart is automatic.

Build with installed tooling:

```sh
# mac3
make -C frontend
make -C service_trial service
/opt/homebrew/bin/nats-server -t -c service_trial/config/nats-mac3.conf
/opt/homebrew/bin/nats-server -c service_trial/config/nats-mac3.conf
# separate process, same checkout
build/service_trial/orchestrator nats://127.0.0.1:4222 build/service_trial/orchestrator.lock

# rack1, after normal Odezza core/runtime builds
make -C service_trial worker NATS_CFLAGS= NATS_LIBS=-lnats
LD_LIBRARY_PATH=/usr/local/cuda/lib64 python3 service_trial/supervise_worker.py --devices 0,1
```

mac3 has NATS server 2.14.6 / C client 3.13.0; rack1 has C client 3.12.0. Their
common APIs passed this trial. `config/nats-mac3.conf` is validated. The old
`config/work-stream.json` is a design example; the C orchestrator creates the
actual per-generation memory stream.

rohini and ada were checked but do not have NATS C headers installed, so neither
has joined this trial. System dependency installation remains a user action.
The broker is private to users who already have SSH access. Authentication/tenant
roles, public API limits, payload digests across untrusted transport and durable
recovery are not implemented paid-service guarantees.

## Current CUB reducer update (2026-09-12)

The live rack1 worker now uses only CUB reduction. Raw `evaluation_row` retention
can request global/per-family top-k and reduce whole family tiles before host
merging; existing distinct units/defaults preserve their meaning. Reports include
reducer backend, continuation calls and separate startup attribution. The C99
core remains independent of family identities and search policy.

The matched 1M AST × 2,048-configuration request improved from 1.5833 to 1.2210 s
warm direct native on both 5080s. Through mac1 → mac3 → rack1, the measured median
is 1.2654 s; raw global/family top-16 is 1.2142 s with a different output contract.
This fixture uses four RK4 steps, not long trajectory recovery. Full native,
family/RNG, memory and live-reference checks passed. See the
[CUB integration record](../benchmarks/score_topk/integration/README.md), including
microbenchmark regressions, the conservative one-system template repair and
verified deployment/backup identities. The tables below are earlier measurements.

## Validation and measured cost

[Validation report](VALIDATION.md) links complete evidence. No custom CUDA search
or Python AST expansion was used by the new service. The existing kernel shapes
and C scoring/reduction implementation perform the work.

| Warm workload / path | Median time | Configuration rate |
|---|---:|---:|
| 1M systems × 2,048 configurations, direct C, old numeric cache policy | 2.836 s | 722M/s |
| Same, direct C, attempt-owned inputs | 2.824 s | 725M/s |
| Same, mac1 → mac3 → one rack1 5080 → mac1 report | 2.897 s | 707M/s |
| Same, one job across both rack1 5080s, end to end | 1.612 s | 1.27B/s |

Three repetitions per timing row (excluding the direct cold compilation run).
The one-GPU queued median includes about 8.1 ms admission and 4.7 ms result fetch;
remaining differences include queueing, polling, preparation variance and network.
The small constant bank is only 8 KiB: its fresh-upload cost was below run-to-run
variation. This does not establish the cost of large banks/trajectory arrays.

This fixture has six independent RHSs, one very short trajectory, four RK4 steps
and 12 scored state values per configuration. These are throughput measurements,
not realistic long-rollout recovery rates. Twelve smaller concurrent requests
covering explicit constants, uniform/normal Philox, dependent RNG scaling, toggles,
and family/tag retention matched direct C exactly (63–110 ms end to end).

## Remaining implementation work

Use the [numbered active tracker](../scratch/structural_search_trial/TODO.md#remaining-grammar-execution-work-2026-09-12)
for current status and completion criteria. Its service-specific work includes
complete memory accounting (1), bounded retention/report costs (2–3), compatible
artifact reuse and scheduling measurements (4–6), aggregate work admission (7),
broader validation/release (10), and capability routing/operational readiness (11).
Trajectory/MSE pooling, direct mac3–rack1 transport and independent worker control
are implemented. Per-job handles and pipeline-owned streams are accepted. Legacy
deployment fixes are outside the current active scope.

## Parsing placement and CPU concurrency

Observed mac3: Apple arm64, 10 physical/logical cores, 16 GiB RAM; existing Python
3.9.6, C compiler and make. The existing C frontend builds and passes trajectory,
static, grammar, bounded-cursor and mutation tests, the allocation dependency audit,
and 225 native-program comparisons with the Python oracle on this host.

The user explicitly allowed parsing on GPU hosts after considering the binary
format cost. That is the selected initial route: raw JSON travels in a queue
message or bounded fetch, and C performs preparation at the executor. The request
still has a defined envelope and immutable identity; it is not a dump of a C
struct and does not claim to contain a prepared binary grammar.

The earlier million-system/2,048-bank measurements put the full native
parsing/preparation phase at about 26–31 ms (roughly 1–2% of those warm request
times). That timer also includes producer/page/retention setup, so it is an upper
bound on the part attributable solely to JSON parsing. It does not demonstrate
that binary encoding is never useful, especially for large trajectory arrays.

Use bounded parser workers on GPU hosts, each with reusable arenas, so preparation
can eventually overlap another request's execution. Start with measured concurrency
that leaves CPU capacity for specialization/module loading; do not automatically
occupy every core. mac3 can concurrently admit/dispatch requests without running
many expensive grammar compilers itself. One producer cursor remains single-owner.

`prepare_benchmark.c` measures the existing C trajectory/static/grammar phases with
separate reusable arenas per thread. It excludes AST generation, wire encoding,
queue/network, runtime setup and GPU work. It is evidence about parsing cost, not
a deployed service or an end-to-end throughput claim.

Measured on mac3, median of three runs per case:

| JSON workload | Input bytes | Prepared arena bytes/worker | One-thread time/request | Eight-thread aggregate requests/s |
|---|---:|---:|---:|---:|
| Small fixed/variable RHS example | 567 | 1,528 | 44.7 µs | 114,078 |
| 1M-system grammar with 2,048-value grid | 27,881 | 18,973 | 1.47 ms | 3,379 |

The arena byte count is not a portable serialization size: it contains pointers
and depends on native layout. The eight-thread rate is aggregate throughput, not
individual request latency. These cases justify deferring binary grammar
transport; they do not establish a universal parser cost or optimal thread count.
[Measurements](validation/mac3-preparation.json),
[CPU build and validation](validation/mac3-frontend-build.log).


## References and deferred binary transport

The initial route forwards JSON and parses on GPU hosts, following the user's
placement decision. [PROTOCOL.md](PROTOCOL.md) preserves the **deferred** prepared
binary payload sketch; this implementation does not claim a portable C grammar
codec. If profiling warrants it, compare that codec against this route using the
same requests, winners, coverage and end-to-end timings.

- [NATS pull consumers](https://docs.nats.io/learn/jetstream/pull-consumers)
- [NATS work-queue retention](https://docs.nats.io/learn/jetstream/retention-policies)
- [NATS server limits](https://docs.nats.io/reference/config/)

## Bulk buffer configuration and verification

`supervise_worker.py` accepts `--trajectory-pool-mib` (64 by default),
`--tile-device-pool-mib` (256) and `--tile-host-pool-mib` (64). The trajectory
setting reserves one CPU arena per worker plus one device slab per GPU; the other
two settings reserve per GPU. CPU slabs reserve address space through malloc;
this does not pin or pre-fault every physical host page. CUDA reservations are
made before pulling work. Each pool size must be 1..8192 MiB, subject to available
memory. A failed startup reservation prevents the worker from accepting jobs.

Native API: `odz_runtime_reserve_buffers` reserves device/result slabs;
`odz_job_create_borrowed` borrows the supplied request and trajectory arena until
destruction. Neither changes the hardened scoring core. `buffer_storage` in the
native report identifies the selected path and capacities per GPU.

The supervisor also accepts `--job-host-limit-mib` (2048, range 1..8192) and
`--max-worker-rss-mib` (4096, range 1..65536). The first sets the C runtime's
operator ceiling via `ODEZZA_JOB_HOST_LIMIT_MIB`: a larger requested host budget
fails before scoring. All device threads share the job reservation ledger;
the allowance is not multiplied by GPU count. The second checks the child RSS
every 100 ms, including while C/NVRTC is busy. Exceeding it kills that child and
the existing supervisor/attempt protocol handles replacement. Unreadable RSS for
a live worker fails closed. It requires Linux `/proc`. This is a sampled kill
threshold, with possible transient overshoot, not an allocation interceptor.

`worker_info.host_limit_bytes` and native `memory.worker_host_ceiling_bytes`
expose the job ceiling. Full reports include allocation classes, shared peaks,
denied reservations, process RSS, explicit device allocations and device-wide
samples. Fixed slab capacities remain separately reported. The guard is per
worker process; multiple independently launched workers need host-wide admission
and nonoverlapping GPU ownership. No customer data database was added. See
[scope and test evidence](../docs/grammar/C99_MEMORY_ACCOUNTING.md).

[Pool checks](validation/buffer-pools.json) compare legacy and pooled results,
exercise one/two GPUs, repeated inputs, short arenas and smaller tiles. A test-only
CUDA allocation probe observes zero explicit cuMemAlloc calls after startup for
the literal-RHS fixtures (numeric-bank allocations are outside that assertion).
[Queued fixture comparisons](validation/pooled-direct-network.json) cover constants,
Philox, toggles and family/tag reports. [Independent-client check](validation/without-mac1-relay.json)
ran with both mac1 relay tunnels closed. These checks establish the stated private
trial contract, not public multi-tenant readiness or reboot orchestration.
