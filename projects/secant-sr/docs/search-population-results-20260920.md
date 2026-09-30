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

# Overnight SRBench population and variation results

All **1,475 measured fits completed**, with zero execution errors. All 1,304
current-toggle fits passed the CPU score audit and winner reconstruction; the
171 archived-settings fits passed that adapter's materialization guards. Worker
and result-copy exit codes were all zero. Local result hashes match the remote
files. No fit stopped at its generation ceiling.

| Worker | Fits | Elapsed time | Finished, EDT |
| --- | ---: | ---: | --- |
| Rack1 GPU 0 | 524 | 4 h 8 min | September 20, 00:32 |
| Rack1 GPU 1 | 518 | 3 h 44 min | September 20, 00:08 |
| Ada GPU 0 | 433 | 3 h 18 min | September 19, 23:42 |

Overall wall time was **4 h 8 min**, including retained warmups. The eight-hour
window was a maximum, not a requirement to keep rerunning finished work. All
three GPUs were idle at the status check. Results are copied to
`scratch/search-night-20260919/shard{0,1,2}/` on mac1 and remain on both hosts.
The current engine reported **2,294,116,515,840 configuration evaluations**;
archived settings logs do not provide comparable configuration counts. These
counts include duplicates and unused selector bits, not unique expressions.

## Broad panel

Each row below covers the same 116 datasets × two official seeds, with each
dataset/seed kept on the same GPU across policies. Every fit used 10,000 train
and 25,000 held-out rows. Success means **held-out R² > 0.999**, not exact symbolic
equivalence. Mean process time includes both successes and budget misses.

| Policy | Population | Budget | Successful fits | Success rate | Mean process time | Mean generations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing variation | 8,192 | 30 s | 169/232 | 72.8% | 15.57 s | 26.8 |
| Existing variation | 8,192 | 60 s | **186/232** | **80.2%** | **26.22 s** | 41.5 |
| Existing variation | 32,768 | 30 s | 148/232 | 63.8% | 20.24 s | 12.0 |
| Existing variation | 32,768 | 60 s | 166/232 | 71.6% | 33.36 s | 16.7 |
| Combined toggle variation | 32,768 | 60 s | 175/232 | 75.4% | 30.80 s | 16.0 |

At population 8,192, the longer budget gained 19 successful cases and lost two,
for a net gain of 17. These are independent timed searches; see the
reproducibility caveat below. Increasing population to 32,768 at 60 seconds
gained three and lost 23 relative to 8,192. It evaluated **61.5% more configurations**
but completed substantially fewer selection/breeding cycles and recovered fewer
cases. This supports prioritizing search feedback over population growth in
this regime; it does not establish the best population for every problem.

Combined toggle variation gained 13 and lost four compared with the 32,768-member
control. Mean runtime fell by 7.7%. This is promising at that population, but
still trails the ordinary 8,192-member search. Neither policy proves exact
structural recovery; under the stricter native train-and-holdout NMSE ≤ 1e-6
criterion, both the 8,192/60-second control and 32,768/60-second combined policy
have 149 successes.

## Diagnostic panel

The preselected 12 datasets × two seeds tested additional policies at 60 seconds.

| Policy | Population | Successes | Mean process time |
| --- | ---: | ---: | ---: |
| Existing variation | 8,192 | 15/24 | 40.78 s |
| Bit alignment only | 8,192 | 15/24 | 37.65 s |
| Targeted toggle mutation only | 8,192 | 15/24 | 39.17 s |
| Leaf mixing only | 8,192 | 15/24 | 38.70 s |
| Combined variation | 8,192 | 15/24 | 39.95 s |
| Existing variation | 65,536 | 12/24 | 49.34 s |
| Combined variation | 65,536 | 12/24 | 48.54 s |

The 65,536-member population is not supported by these results. The smaller
variation tests provide no numerical-success advantage on this panel; their
timing differences require repetitions before changing defaults.

## Fresh old-system comparison

Only the 155 rack1 dataset/seed cases have an archived-settings comparison.
Both policies below ran on matching data and GPUs with a 60-second budget.

| Complete search system | Successes | Success rate | Mean process time |
| --- | ---: | ---: | ---: |
| Current toggles, population 8,192 | 121/155 | 78.1% | 26.72 s |
| Old settings with random coefficient refinement | **133/155** | **85.8%** | **24.62 s** |

The old system gained 13 and lost one case, while averaging 7.9% less process
time. It remains the stronger complete search system on this matched subset.
This is **not a settings-versus-toggles kernel ablation**: the old policy has
coefficient refinement, different tree limits and caching; current fitting was
disabled for the population/variation comparison. LM was not used by either
system. Ada was excluded from the old-system arm because it lacks the archived
binary's NVRTC runtime dependency. No substitute backend was used.

## Where time went

For the 8,192/60-second control, mean process time was 26.22 seconds:

- Scoring stage: 25.60 s, **97.6%** of process time.
- Host AST generation/encoding/breeding timing: 0.38 s.
- Setup: 0.14 s, including 0.035 s reported NVRTC time.
- Device event time: 22.88 s and module-load timing: 1.21 s, both within the
  scoring work; these are not extra amounts to add to scoring.

The warm overnight measurements are dominated by evaluation, not seconds of
Python or repeated compilation. The cold validation's roughly 3.4-second NVRTC
measurement is not representative of the subsequent warmed campaign. These
timings alone do not prove full SM occupancy. Persistent template caching remains
useful for cold requests but is not the leading measured bottleneck here.

## Reproducibility issue: GP-REPRO-20260920

Two cases succeeded at 30 seconds but failed the numerical threshold at 60:
`feynman_I_40_1-s23654` and `feynman_III_19_51-s23654`. Commands differ only in the
time budget; binary, data, seed and GPU match. Generation logs show different
population counts/scores by generations 2 and 1 respectively, well before the
shorter cutoff. Therefore the longer runs are not exact extensions of the
shorter runs, and the losses should not be attributed to extra search time.

The scoring generator uses floating-point `atomicAdd` for tile SSE accumulation.
Order-dependent rounding followed by GP selection amplification is a plausible
cause; the exact cause has not been isolated by a deterministic-reduction test.
All final score audits remain accepted. Results are valid as independent
stochastic-search observations, but do not support identical-prefix timing
claims or small-effect conclusions without repetitions. No results were removed.

Next investigation: measure same-seed repeated-run variation and consider a
deterministic diagnostic reduction mode, preserving the fast path for ordinary
search. Do not change the kernel merely to make this experiment look cleaner.

## Recommendation and remaining work

Use **8,192 candidates and a 60-second maximum** as the next tested baseline;
retain early stopping. Leave new variation options experimental until repeated
tests establish a benefit at 8,192. Do not increase the default population based
on raw configurations/second alone.

2. Comparison complete. Investigate reproducibility and repeat promising policies;
   expand seeds/noise/black-box coverage. Cold-request caching remains lower priority.
3. Profile occupancy and packing within the dominant scoring stage. Larger
   populations improved measured work throughput but reduced search effectiveness.
4. Isolate the old system's fitting and selection advantages. Revisit coefficient
   capacity and starts; keep refinement opt-in and LM deferred for this migration.

[Machine-readable totals, paired gains/losses and deviation record](data/search-population-results-20260920.json).
[Original experiment design and implementation](search-population-night-20260919.md).

The follow-up [controlled comparison](controlled-settings-toggles-20260920.md)
now replays identical populations through both engines and runs the same GP with
fitting on/off. Keep those results separate from this historical-policy comparison.
