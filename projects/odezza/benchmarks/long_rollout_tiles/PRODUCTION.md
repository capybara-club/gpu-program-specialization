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

# Production tile-sizing repair — 2026-09-13

The fixed aggregate-work limit starved long-rollout kernels. A separate pure C
runtime policy now accounts for the device's parallel capacity, balances bank
slices, and reports underfilled work. The native kernel core is unchanged.

## Matched evidence

Two RTX 5080s, original full requests and original numeric/trajectory inputs.
The previous-cap baselines are the completed diagnosis; no rerun of a slower
baseline was substituted for a different scientific workload. New direct times
are warm runs after an initial complete replay; polynomial uses a two-run median.

| Screen | Configurations | Previous C time | New C time | Old/new tiles |
|---|---:|---:|---:|---:|
| Initial | 224,919,552 | 150.761 s | 22.120 s | 18,099 / 867 |
| Second | 53,084,160 | 46.462 s | 8.157 s | 4,455 / 225 |
| Quadratic | 33,947,648 | 22.030 s | 3.344 s | 2,657 / 130 |
| Polynomial | 5,242,880 | 3.391 s | 0.535 s | 410 / 20 |
| Total | 317,194,240 | **222.643 s** | **34.156 s** | 25,621 / 1,242 |

**6.52× faster.** Candidate identities, scores, coefficient bits, reconstructed
programs, global/family rankings and valid/invalid counts match the original
service reports exactly. Not all raw MSE arrays were copied back and compared.
The earlier private larger-cap probe took 41.566 s; the balanced production
policy improves on that diagnostic result without shipping its environment knob.

After deployment, mac1 → mac3 → rack1 replayed all four complete requests:
**34.279 s worker time, 34.821 s including submission/polling/result retrieval**.
Every retained-result signature still matches. The polynomial row uses its second
live run; both runs matched. Service generation: `d21229170c59974cb127acd38e99326a`.

## Why the billion-per-second benchmark missed this

The previous reported **2.048 billion configurations / 1.612 s = 1.27 billion/s**
was a two-GPU end-to-end warm measurement. That fixture has one million complete
six-equation AST vectors, 2,048 explicit bank rows, three observations, **four
RK4 steps and twelve scored scalar values per configuration**.

At four steps, the old 67,108,864 work-unit cap allowed 16,777,216 configurations
per tile. Its 1,024-AST × 2,048-row pages fit comfortably. At system 19's **5,120
steps**, the same cap allowed only 13,107 configurations: ordinarily 103 blocks
with one AST, sometimes followed by a tiny remainder. The older synthetic sizing
suite used only 2/40 steps, so it missed this sequential-work regime.

A fresh matched short-fixture comparison gives **1.23230 s previous versus
1.22405 s new**, medians of three warm C runs, about **1.67 billion configs/s**.
Both execute 977 tiles, and all exact output signatures agree. Cold timing and
cache misses are retained separately. This is no material short-fixture
regression, and it is not a universal configuration rate for all ODE workloads.

## Runtime design

`runtime/odz_sizing.[ch]` is allocation-free host policy, with no grammar traversal
or CUDA handle ownership. Device SM/thread-capacity attributes are queried at
runtime creation. The policy is:

```
work_per_configuration = max(RK4_steps, trajectory_points)
parallel_target = nextpow2(2 * SM_count * max_threads_per_SM)
work_ceiling = max(67108864 / work_per_configuration, parallel_target)
```

Parallel target is bounded at 4,194,304 configurations. On a 5080 it is 262,144
(2,048 blocks for a single AST). Explicit configuration ceilings and memory pools
can reduce it. Complete toggle products cannot cross these hard ceilings.
Automatic sizing may shrink toward the measured 50 ms target, but preserves the
parallel target where enough work fits. The target is advisory. Per-configuration
point, subdivision and integration limits remain unchanged.

Banks are divided into balanced contiguous ranges rather than maximum-sized
chunks followed by a tiny tail. The existing bank indices and Philox counters
are unchanged, and no padding configurations are scored. Timing estimates are
scoped to the job/family/numeric layout; undersized tails and template growth do
not train them. The estimate uses prepared scoring-call wall time, excluding
JSON parsing, template creation and retention.

New per-device `sizing` diagnostics include the work ceiling, hardware target,
smallest tile, maximum AST count, block geometry and inactive lane capacity;
underfilled tile/configuration counts and scoring time; explicit, memory and
available-work constraints; balanced-bank counts; and maximum plus histogram
p50/p95 upper bounds for scoring-call duration. Constraint counts can overlap.
These are host-call and geometry metrics, **not measured SM occupancy**. CUDA
profiling remains optional. Hardware-counter profiling is still unavailable
under the driver permissions recorded in the original diagnosis.

## Validation and material limits

- CPU ASan/UBSan: sizing arithmetic, overflow, full bank coverage/no duplicate
  ranges, tiny-tail prevention, explicit limits and duration accounting; existing
  retention/report-bound sanitizer and admission tests also pass.
- One and two GPUs: eleven old/new matched scenarios each, including 5,120 and
  65,536 steps, uneven bank sizes, two/four-way state toggles, uniform/normal RNG,
  12 states with 24 constants, literal ASTs, scarce work, explicit small tiles and
  a 128 KiB device budget. Exact retained results and coverage agree.
- Separate new-runtime maximum-step test: 262,145 configurations × 65,536 steps
  in about 0.765 s, split into balanced calls of about 0.38 s. This is not a
  paired speedup result. The first old-runtime test at that population exceeded
  its 120 s budget; the matched maximum-step test was reduced to **8,193 rows on
  both runtimes**. Its final report was not retained by the initial test harness.
- Live cancellation after progress: ~29 ms at 5,120 steps and ~386 ms at 65,536
  steps, for one and two GPUs; subsequent jobs reuse the runtime successfully.
  Submitted CUDA work is not preempted. The 50 ms target is not a latency SLA.
- An initial cancellation test incorrectly assumed the measured status JSON
  size could not grow between calls. The harness now uses a bounded 8 KiB buffer;
  this was a concurrent test-client bug, not a runtime result-corruption failure.
- Live service checks pass for long integration and initial-only trajectory
  reset work; both respect the per-device envelope and retain all configurations.
- Workers still lease **whole producer pages**. One AST with a large bank uses
  one GPU; many available pages use both. Sharing one page's numeric ranges
  across devices remains separate work. No claim of universal saturation is
  made, and the user's deferred overlap refactor remains untouched.
- Very small requested populations and memory/explicit limits still underfill
  hardware; literal ASTs can waste lanes. These are visible in reports, not
  hidden by manufacturing extra configurations. Other GPU architectures still
  require calibration. Finishing whole blind recoveries also depends on proposal
  quality and the many small fitting requests; this does not claim 6.5× faster
  end-to-end discovery.

The health/report field `work_units_per_tile` was replaced by
`base_work_units_per_tile`, `maximum_parallel_target_configurations` and
`tile_policy`. The effective configuration ceiling is per device in `sizing`.
Report preflight reserves 2 KiB more metadata per supported GPU; the conservative
report bound can therefore reject a smaller explicit report-byte budget earlier.

## Artifacts and deployment

Compact evidence: [production-summary.json](production-summary.json). Full raw
reports, exact input requests, original/diagnostic binaries and the tested build:
`rack1:/home/cdurham/experiments/recovery19-throughput-20260913/`.

Live runtime:
`/home/cdurham/odezza/scratch/service_trial_20260912/timing-admission`.
Pre-deployment sources/binaries/manifest are in the experiment's
`pre-deploy-backup/`. mac3's equivalent backup is
`/Users/cdurham/code/odezza-parallel-sizing-validation/pre-deploy-backup/`.
No active jobs or retained results existed when restarting the service.
An immediate health check raced subscription startup and timed out; subsequent
health checks and all full requests passed. The worker supervisor restarted one
initial connection attempt while the orchestrator became ready.

Unchanged core SHA256:
`762c65871aff160b143aef223120c2821075bea7ea28902952d2f7cad4f831bd`.
Deployed runtime SHA256:
`dd1288edb0b146152356fc4da2abe587206acb76f7cb6333dd10559823ba63c1`.

Reproduce the policy tests with `make -C runtime test-sizing-cpu` and
`tests/runtime/tile_sizing.py --library NEW --baseline OLD --devices 0,1 --output FILE`.
The saved old binary is required for the matched baseline. `replay.py` also accepts
ordinary production libraries: its diagnostic cap environment variable has no
meaning to the production runtime. `build_probe.py` intentionally targets the
old fixed-cap checkout. Its original instrumented source and runnable build are
preserved under `probe/`; the pre-deploy backup also preserves the old runtime
sources, but is not a complete checkout by itself.

Final live short-fixture regression also passes: three complete 2.048-billion
requests and upfront rejection of an excessively small integration step. The
old archived report lacked newer family metadata fields; the test now compares
every historical field and separately checks the added allocation/coverage
fields. Exact candidates and rankings remain required. Client median: **1.283 s**.
