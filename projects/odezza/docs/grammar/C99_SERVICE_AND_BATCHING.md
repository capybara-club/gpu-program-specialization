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

# Native service reliability and configuration batching — 2026-09-12

Implemented in the separate `runtime/` and grammar transport. Kernel shapes,
SASS generation and the hardened `core/` were not changed in this work. The
existing Python/SQLite/LM and GP backends remain separate.

## Service behavior

- `odz_job_status` reads an allocation-free published snapshot under its own short
  mutex. It does not traverse retained candidates or take the C parsing/report
  mutex. Detailed family coverage stays in `odz_job_report`.
- `odezza_release` releases a completed job or prepared input. Completed jobs and
  prepared inputs expire after one hour by default. Job release also removes its
  idempotency keys; replay requires a live handle or an exported report.
- Defaults: 32 retained job handles and 32 prepared inputs, 8 unfinished jobs,
  256 MiB of accounted request copies. Admission failure does not truncate jobs.
- CLI/MCP defaults to a supervised child process; C owns contexts, generation,
  pools, cache and scoring. The parent supervises worker RSS (4 GiB), combined RSS
  (8 GiB), heartbeat (30 s), and native job elapsed time plus 10 s cleanup grace.
  CUDA initialization is reported separately and has its own watchdog, even while
  the control loop remains responsive. Lost jobs fail explicitly. No automatic resubmission, database or recovery from
  a hidden seed. The next new submission starts a fresh worker when needed.
- `--in-process` retains the direct embedding/benchmark path. This mode cannot
  terminate a wedged CUDA worker; embedders must supply their own supervision.
- MCP rejects and drains oversized UTF-8 messages without closing the transport.
  Its 16 MiB envelope limit and C's 64 MiB request limit are both documented.
  Native schemas describe native replay/prepare behavior and expose stable error
  codes. Large RPC payloads are sent without holding the reply reader's lock.
- C file locks serialize shared cache misses across device workers/processes.
  Template files still validate shape/source/checksum and use atomic replacement.

## Scheduling

`odz_runtime_create_devices(devices, count, cache, &runtime)` supports 1–8 distinct
GPU indices. One parser and producer generate the population once. Each GPU owns
its context, numeric buffers, sessions and module-loading pipeline. Workers lease
independent pages using a shared family round-robin cursor. Candidate indices,
Philox addressing, allocations and retention identities are independent of which
GPU executes a page. Two pages per GPU per family provide generation buffering.

When `max_chunk_configurations` is omitted, the runtime considers compatible page
size, memory, and measured tile cost with a 50 ms advisory target and 16,777,216
configuration ceiling. The device-aware parallel-work envelope also applies:
`max(67108864/max(RK4_steps,points), nextpow2(2*SMs*threads_per_SM))` configurations.
The parallel target is bounded at 4,194,304 and is 262,144 on RTX 5080. Explicit
limits disable timing adaptation but still obey this envelope and memory bounds.
Under pressure, automatic mode shrinks the number of ASTs first, preserving more
coefficient rows per loaded module. It only splits a single system's rows when
necessary, balancing bank slices to avoid tiny tails. `target_tile_seconds` is
configurable in (0,1] and yields to the parallel target when enough work fits.
Producer page size and
module shape remain explicit; this is not a general kernel autotuner.

Reports expose per-device configuration/chunk counts, largest tile, busy time and
page-wait time. Stage timers are sums over workers; they overlap each other and
generation and cannot be added to obtain wall time. Full jobs preserve results;
deadline/cancellation coverage can change with scheduling.

## Matched throughput

Warm-cache in-process C request time on rack1, dual RTX 5080. Median of three runs
per mode. Fixture: 1,000,000 complete six-equation vectors × 2,048 explicit constant
rows = 2,048,000,000 configurations. Same finite grammar, banks, trajectories,
retention, 1,024-AST producer pages and 32-system modules in every mode. The short
trajectory has three observations, four RK4 steps, and 12 scored scalars/config.

| Execution | Seconds/request | Systems/s | Configurations/s | Total tiles |
|---|---:|---:|---:|---:|
| One GPU, old explicit 1,048,576 cap | 5.694 | 175,611 | 359.7M | 1,954 |
| One GPU, automatic tiling | 2.853 | 350,529 | 717.9M | 977 |
| Two GPUs, automatic tiling | 1.584 | 631,465 | 1.293B | 977 |

The two-GPU runs split configurations approximately 50/50. Complete retained
candidate records and all leaderboards match across modes and repetitions.
These are execution rates, not blind recovery rates; they exclude remote upload,
process/context startup and optional CPU verification. No NVRTC occurred in these
warm measurements. A separate cold two-GPU template test compiled once (one miss,
one hit): native request 7.307 s, of which 7.276 s was NVRTC for that shape. This
cold case has a different shape/workload and is not a cold version of the table.

[Clean-source-build measurements](runtime-validation/odezza-upgrade-final.json).
The subsequent bank-preserving repair changes constrained tiling; the default
full-page geometry in this table is unchanged. The final clean-package target
repeats these checks after the repair: [full target report](runtime-validation/odezza-final-release-checks.json),
[build/test log](runtime-validation/odezza-final-release-checks.log).

## Validation and consequential findings

- CPU frontend contracts, 18,000 parser-mutation measurement calls, allocator
  audit, 225 native/Python program comparisons, 83 compiler tests and 25 transport
  tests pass. Original 15 native execution contracts also passed after the initial
  lifecycle/group integration.
- Final dual-GPU robustness suite: 191 checks, 67,429,493 configurations, independent
  CPU/Philox verification and exact chunk-invariant leaderboards. CUDA memcheck
  reports zero errors. [Final report](runtime-validation/odezza-upgrade-final-memcheck.json),
  [memcheck log](runtime-validation/odezza-upgrade-final-memcheck.log). These include controlled finite-family recovery rather
  than unrestricted discovery.
- 1,000 sequential release cycles, following 50 warm-up jobs, complete in 21.03 s;
  observed process RSS growth is 241,664 bytes and retained job/request/key tables
  are empty. This is a short release soak, not an overnight reliability claim.
- 100 lightweight queued-status calls have median 3.46 microseconds through direct
  Python/ctypes. The test replaces full-report serialization with a failure to
  prove status does not invoke it. This is not remote MCP latency.
- Controlled worker freeze/watchdog and RSS-limit tests recover for a new job.
  A separate active-job process crash yields `worker_lost` and never auto-replays
  the failed job. Concurrent 2 MiB control messages complete without blocking the
  reply reader. Cooperative two-GPU cancellation takes about 7 ms in its test;
  this is not a worst-case bound.

Two findings were retained and repaired:

1. **Cache-pressure coverage changed under two GPUs.** The original 4,200 banks
   split into two caches with 4,096 entries each; no eviction was necessary. The
   initial run scored all work but had 190/191 passing assertions. The pressure
   case now uses 8,400 banks for two devices (42,000 evaluations), preserving the
   original single-GPU test and exceeding aggregate entry capacity. No failed
   numerical evaluations were hidden.
   [Original assertion](runtime-validation/odezza-upgrade-robustness.json).
2. **Automatic low-memory tiling repeated module work.** The first controller
   kept a large AST group and shrank rows, which then fed a low observed rate back
   into even smaller banks. Under a deliberate 1 MiB tile budget, it timed out at
   120.04 s after 4,561,610/8,388,608 evaluations. Those results are partial and
   must not be compared as a complete run. The repair shrinks AST count before
   bank length. The original request then completes all 8,388,608 evaluations in
   0.220 s with exact reference winners. The normal-memory fixed/automatic
   comparisons take 0.302/0.169 s. All three use four trajectories, 26 observations,
   six states and dt=0.005. These are single-run diagnostics, not timing medians.
   [Original timeout](runtime-validation/odezza-upgrade-memory-timeout.json),
   [repair and live cancellation/crash checks](runtime-validation/odezza-upgrade-scheduling-fixed.json).

## Reproduce and release boundary

```sh
make test-frontend test-grammar
make test-request-service
CUDA_MODULE_LOADING=EAGER PYTHONPATH=python /usr/local/cuda/bin/compute-sanitizer \
  --tool memcheck --error-exitcode 99 python3 tests/runtime/robustness.py \
  --devices 0 1 --seed 20260913 --output /tmp/robustness.json
python3 runtime/package.py --output /tmp/odezza-native-trial.tar.gz
```

`runtime/package.py` makes a reproducible working-tree source snapshot, including
untracked sources, with per-file SHA-256. The source snapshot was extracted into
an empty rack1 directory, all hashes verified, and `make request-runtime` succeeded
using installed tools. It is a trial artifact, not a committed/tagged public
release. [Clean build log](runtime-validation/odezza-release-build.log).

Still open: overlapping prelude/reduction across tiles on the same GPU; automatic
producer/module shape selection; producer seek/checkpoint restore; broader GPU/
CUDA compatibility, long mixed-load service campaigns and unrestricted recovery.
The service is ready for bounded internal trials, not certified as a public
multi-tenant service. Memory limits are sampled and native cancellation remains
cooperative; compilation and one expensive launch can exceed the advisory target.

**Subsequent audit:** [request failure-path findings](C99_REQUEST_PATH_AUDIT.md)
include a reproduced numeric-cache error after failed upload, an unintended
600-second supervisor ceiling and reporting/control coupling. These remain open;
the passing checks above should not be read as covering those cases.

## Matched RTX 5090 follow-up

Rohini was measured with the same checked source snapshot and the same million-
system/2,048-row automatic single-GPU request. One warm-up followed by three timed
runs; all retained candidate records and leaderboards match the 5080 reference
exactly, and the top three winners independently replay on CPU. NVRTC is 13.1,
driver 595.71.05. Different host CPUs/drivers remain part of this end-to-end
comparison; these are not measurements of GPU arithmetic alone.

| GPU | Median native request | Configurations/s | Pipeline | Reduction/gather |
|---|---:|---:|---:|---:|
| RTX 5080, rack1 | 2.8618 s | 715.6M | 1.8179 s | 0.8265 s |
| RTX 5090, rohini | 2.4390 s | 839.7M | 1.7531 s | 0.4538 s |

The 5090 achieves 17.3% higher end-to-end throughput. Reduction/gather improves
substantially, while the combined specialization/module-loading/execution timer
changes little. That motivates finer pipeline attribution and overlap; this timer
does not establish how much is CPU overhead versus GPU execution. Generation
(0.858 s on rohini) overlaps scoring and must not be added to wall time.

The cold warm-up takes 4.489 s in C, including 2.044 s NVRTC; each timed run has
zero scoring NVRTC. Context initialization is excluded from the C request timer.

[Raw 5090 report](runtime-validation/odezza-native-5090-timing.json),
[run log](runtime-validation/odezza-native-5090-timing.log),
[reusable single-host harness](../../tests/runtime/host_compare.py).
