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

# Search latency attribution — 2026-09-10

The population integration checks are valid correctness checks, but their total
times do not isolate the effect of population filling. A first-run scoring
pipeline creation penalty explains most of the successful pairs' differences.
This corrects the previous interpretation of case-024 as a large population
regression. The separately warmed prepared-fitting curves remain valid.

## Measured phases

All four successful pairs used the same initial 65,536 seed ASTs, 1,024 coefficient
rows and four state-binding permutations within each pair: 268,435,456 scheduled
configurations. All succeeded in one search wave. The toggle-expanded global
distinct AST count is unknown. Each configuration schedules four trajectories,
six observation intervals and 32 RK4 steps/interval (768 RK4 steps total).

Seconds below are measured. Screen family time includes module dispatch/loading,
execution and reporting; it is not GPU kernel time alone. Fitting is the complete
controller fitting phase. The remaining time includes baselines, verification,
controller and generation work and is not separately attributed here.

| Host / case | Policy / order within case | Pipeline creation | Screen family | Fitting | Total |
|---|---|---:|---:|---:|---:|
| rack1 / 001 | baseline / first | 12.694 | 7.769 | 3.137 | 25.395 |
| rack1 / 001 | population / second | 0 | 7.792 | 2.653 | 12.020 |
| rohini / 008 | baseline / first | 13.145 | 7.619 | 5.522 | 28.954 |
| rohini / 008 | population / second | 0 | 7.631 | 6.230 | 16.431 |
| ada / 028 | baseline / first | 22.545 | 4.170 | 3.757 | 32.519 |
| ada / 028 | population / second | 0 | 4.171 | 1.388 | 6.988 |
| ada / 024 | population / first | 25.651 | 9.103 | 3.255 | 42.271 |
| ada / 024 | baseline / second | 0 | 9.088 | 2.348 | 14.789 |

Case-024's 27.482-second total difference contains 25.651 seconds of pipeline
creation; fitting differs by only 0.906 seconds. Its first-screen counts and
invalid counts match exactly. The original inference that extra fitting
substantially delayed structural exploration is not supported by this pair.
Likewise, the aggregate 223.62→198.89 seconds is descriptive, not a policy speedup.
Subtracting setup retrospectively is not a replacement for matched reruns:
setup consumed budget and can change subsequent search decisions.

Cases 004 and 032 time out under both policies. Three attempts have incomplete
work accounting because a scoring command was killed at the deadline: rack1
004 baseline and rohini 032 under both policies. Keep recorded work as a lower
bound rather than treating it as all executed work.

## Code boundary and remaining uncertainty

`scratch/fitting_batch_trial/runtime/o_run.c` times the complete
`odezza_scoring_pipeline_create` call. In `core/o_scoring_pipeline.c` that call
generates source, invokes NVRTC, inspects the cubin, specializes known RHSs and
creates pipeline resources. Those stages do not have separate retained timers;
do not label all 25.65 seconds as measured NVRTC compilation.

Generated source depends on state/constant/system/patch capacities and compiler
architecture/options, not the known equation bytes or RNG seed. Known equations
are specialized afterward. The adapter's four-entry resident pipeline cache
nevertheless checks known equations, RNG seed/stream/pool configuration and
reduction settings together. RNG stream changes across waves can therefore
recreate a pipeline unnecessarily. The observed later-wave recreations took
about 0.024–0.027 seconds, not the initial 13–26 seconds. Existing compiler reuse
appears to make them cheap; these logs do not separately prove its mechanism.

Retained module-load, unload and completion-wait durations overlap. They cannot
be added to estimate device idle time. Achieved hardware occupancy and a
correlated CPU/GPU timeline remain unmeasured. The prepared 4x-fit experiment
demonstrates inexpensive additional fitting work, not four times as much useful
search information or a guaranteed faster solve.

## Next controlled work

1. Report service readiness, cold template creation, actual screening, fitting
   and verification separately. Compare cold against cold and warm against warm.
2. Prepare supported scoring shapes before readiness; retain reusable compilation
   artifacts separately from known-RHS specialization and RNG/reducer resources.
   Do not hide preparation when reporting first-request latency.
3. Spend available fitting capacity on useful independent starts and compatible
   structural/state-binding variants. Measure verified solves per wall second,
   with unique work counts alongside configuration counts.
4. Give search timely results without starving launches: bounded substantial
   screens, batched refinement and early promotion on informative trajectories.
   Assess fresh paired recoveries after removing the cache-state confound.

Owner: search/benchmark runtime follow-up. Status: attribution corrected; cache
separation and a matched warm/cold policy rerun remain open. No core or runtime
behavior was changed during this review.

## Evidence

- [Prepared curves and corrected integration report](runs/occupancy-20260910a/REPORT.md).
- [Search records](runs/occupancy-20260910b/): original per-host `*-search-check.json`,
  retained `*-all-screen-attribution.jsonl` extracts with remote report paths and
  SHA-256 hashes, and `*-screen-attribution.jsonl` initial-screen records.
- [Persistent priorities](../../scratch/structural_search_trial/TODO.md).

The original setup-failure records and all worker results remain unchanged.

## Resolution

The subsequent [cache-controlled comparison](LATENCY_RESULTS.md) implements the
template/resource separation and separately records preparation costs. All 36
corrected-sampler attempts ran with scoring artifacts and LM templates prepared.
The original attribution above remains a review of the earlier implementation;
its cache-separation and matched-rerun follow-ups are now complete. Selective
fitting policy and fresh-system confirmation remain open.
