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

# Historical search versus the fresh-seed finalist trial

The complete historical search recovered substantially more of this diagnostic
panel than the current search. This is a real gap in the retained results, even
after selecting the same problems and seeds. The historical 5090 versus current
5080 and different timing boundaries prevent attributing the entire gap to
search policy or claiming a normalized speed ratio.

The comparison was recomputed from archived CSVs and the 336 completed trial
records. All three historical CSV hashes match the backup manifest. Every
current search winner reconstructs from its binding, all retained native score
audits pass, and the full four-policy/two-repeat matrix is present.

## Same problems and seeds

These rows cover **21 problems × seeds 860 and 5390 = 42 cases**. Source dataset
hashes, official protocol, row counts (10,000 train / 25,000 holdout), zero noise,
and unscaled inputs/targets match. Historical prepared-dataset byte hashes are
not recorded, so this is not a claim of newly verified byte-identical inputs.
Success means final held-out **R² > .999**, not exact symbolic recovery.

| Search | Budget per fit | Numerical successes | Mean recorded time |
|---|---|---:|---:|
| Historical LM, every fourth generation | 100 generations; no time cap | 23/42 (54.8%) | 10.64 s |
| Historical LM, every fourth generation | 600 generations; no time cap | 33/42 (78.6%) | 50.07 s |
| Historical best LM, every generation | 60 s maximum | **35/42 (83.3%)** | **32.55 s** |
| Current baseline | 60 s maximum | 39/84 (46.4%) | 48.33 s |
| Current power mutations | 60 s maximum | 44/84 (52.4%) | 46.66 s |
| Current final polishing | 50 s search + up to 10 s polishing | 41/84 (48.8%) | 40.53 s |
| Current power + final polishing | 50 s search + up to 10 s polishing | 43/84 (51.2%) | 41.78 s |

Historical rows have **one observation per case**; current rows have two
identical-seed repeats. The historical observations are not duplicated into
84 independent trials. Means include failures and early successes. Historical
runs used Rohini's RTX 5090; current runs used one rack1 RTX 5080 per fit.
Two concurrent 5080 workers reduce campaign wall time, not the time of one fit.

For the current power policy, separate repeat results were **20/42** and
**24/42**, versus historical LM's 35/42. Relative to that old observation,
repeat 0 gained two cases and lost 17; repeat 1 gained one and lost 12.
Twelve of 42 cases changed success classification between current power repeats.
This reinforces the need for repeated comparisons rather than trusting small
differences between current policy totals.

Some concrete examples (old two seeds / current power two seeds × two repeats):

| Problem | Old LM successes | Old mean | Current power successes | Current mean |
|---|---:|---:|---:|---:|
| III.8.54 | 2/2 | 7.78 s | 0/4 | 60.54 s |
| I.8.14 | 2/2 | 3.91 s | 1/4 | 52.05 s |
| I.24.6 | 2/2 | 4.18 s | 2/4 | 52.23 s |
| test 1 | 2/2 | 36.86 s | 0/4 | 60.54 s |
| test 17 | 2/2 | 60.13 s | 4/4 | 46.84 s |

The examples demonstrate losses and a current timing improvement; they are not
a hardware-normalized ranking. Per-case outcomes are retained in the JSON.

## What budget produced the historical 92.67%?

The August 15 best campaign used **116 datasets × 10 official seeds = 1,160 fits**,
8,192 population, 8,192 leaf settings, scientific operators and a 60-second
search limit. It stopped using training R² >= .999999 and assessed holdout only
after search. Every row terminated at the target or time limit; none at a
generation cap. Final CPU optimization was disabled.

- **1,050/1,160 individual fits succeeded: 90.52%.**
- **92.67%** is the mean of each dataset's median success indicator across ten
  seeds, a different aggregate from raw fit success.
- Mean recorded fit: **21.71 s**; median **2.04 s**.
- Sum recorded fit time: **25,181.83 s**, approximately **7 hours**. This is summed
  fit time, not a separately verified campaign wall-clock duration.

Thus 60 seconds was a maximum, not the cost of every fit. Many full-suite
problems finished quickly; the selected diagnostic panel is harder.

The saved per-row configuration identifies this fitting policy:

| Setting | Historical best | Latest finalist trial |
|---|---|---|
| Population | 8,192 | 8,192 |
| Broad binding/configuration work | 8,192 leaf settings for projected candidates | 64 banks × 64 permutations = 4,096 configurations per evaluated genome |
| Continuous fitting | Integrated GPU LM | None during evolution; optional final CPU LM |
| Candidate fitting allocation | Up to 512 selected candidates, every generation | Global winner + up to 16 retained structures at the end |
| LM states | 32 bindings × 4 coefficient starts | Up to 2 coefficient starts per final structure; binding fixed |
| LM iterations | 4 proposal iterations after initialization | Up to 200 function evaluations per start, subject to deadline |
| Refined-model feedback | Promoted into the evolving population after rescore | No feedback into evolution |
| Mixed leaf reopening | 4 holes, 25% refinement probability | Different toggle-genome search policy |
| Final CPU optimization | Disabled | Optional, included in elapsed time |

These are different allocations and policies, not equivalent configuration
counts. The old system materialized concrete candidates, reopened selected
leaves, and fitted candidates during evolution. The current trial did not port
that entire strategy. End-only LM cannot improve the descendants bred earlier.
Whether that feedback explains most of the gap needs a controlled test.

Historical timing came from persistent request execution with shared process
setup amortized; current timing includes each search subprocess and final-stage
dispatch/teardown. Cached compilation was used historically. Neither column
should be called a cold-service latency or a controlled GPU speed ratio.

## Independent same-hardware evidence

The [September 19/20 matched complete-system comparison](search-population-results-20260920.md)
used the same rack1 GPUs, prepared data and 60-second limit:

| Complete search | Successes | Mean process time |
|---|---:|---:|
| Old settings + random coefficient refinement | **133/155 (85.8%)** | **24.62 s** |
| New toggle baseline, fitting disabled | 121/155 (78.1%) | 26.72 s |

This is stronger evidence of a complete-search regression independent of the
5090/5080 difference. It used random refinement, not the strongest old LM policy,
and different GP/tree/refinement choices remain part of the comparison.

Conversely, the [controlled engine/GP comparison](controlled-settings-toggles-20260920.md)
favored the new engine. With the same current GP and fitting off, common-scheduler
toggles achieved 31/48 successes in 40.5 s mean versus settings' 24/48 in 48.2 s.
Matched real-population toggle batch evaluation was 1.85–5.36× faster. These
measurements argue for recovering the old search behavior on the new engine,
rather than treating the toggle kernel interface as the demonstrated problem.

## Budget utilization defect in the finalist experiment

Final polishing averaged **0.283 s/fit** without power mutations and **0.339 s/fit**
with them, including skipped stages. The experiment reserved ten seconds by
limiting search to 50 seconds but **did not return unused fitting time to search**.
Consequently the policies had the same maximum allowance, not equal useful time.
Their lower average elapsed time is not evidence of a faster route to equivalent
quality. All recorded elapsed times remain valid; quality conclusions apply to
this actual policy.

Polishing itself rescued **4/84** and **6/84** pre-polish search results across
the two polished policies, respectively, with no losses at R² > .999. This is
evidence of local benefit, not equivalence to the old per-generation LM strategy.
Some polished expressions improved training error without becoming successes.

This limitation affects both polished arms of `finalists-20260920`; it was
identified in the September 21 analysis. No stored results were modified.
Next repair: reserve a measured small allowance or resume GP with unused time,
record stage budgets explicitly, and compare on the same hardware. Keep the
current policy experimental until then.

## Recommendation and remaining work

The current complete search should not yet replace the historical best as the
quality baseline. Retain the new kernels. Recover and test the historical
candidate allocation, coefficient fitting during evolution, and concrete-model
promotion behavior on a bounded matched panel before another broad campaign.
Do not infer that switching selector encoding alone will recover the gap.

2. Historical comparison is complete; next establish matched-hardware parity
   against the strongest historical LM policy, then broaden SRBench coverage.
3. Achieved occupancy profiling remains outstanding; engine throughput alone
   has not translated into comparable search quality.
4. Fix unused final-stage allowance and test integrated fitting plus candidate
   retention. GPU LM is not newly implemented or enabled by this analysis.

Reproduce the numerical comparison without starting searches:

```sh
python3 bench/finalists/compare_history.py \
  --run scratch/finalists-20260920 \
  --audit scratch/benchmark-audit-20260918 \
  --output docs/data/finalists-history-comparison-20260921.json
```

[Recomputed results, identities and limitations](data/finalists-history-comparison-20260921.json).
