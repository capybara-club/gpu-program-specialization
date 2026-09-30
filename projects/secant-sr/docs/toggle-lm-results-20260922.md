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

# Toggle GPU LM diagnostic results — 2026-09-22

Both rack1 RTX 5080 workers completed all 84 fits with zero execution errors.
All 84 native-versus-CPU score audits passed and every fit reported zero local
bytes. Worker and result-copy exits were zero; the automatic start and finish
Telegram notifications were accepted. The register-only repair held up throughout.

The search-quality result is insufficient to replace the historical best search.
Longer runtime plus integrated LM raised numerical success on this panel, but
six datasets failed every attempt. This is a valid completed experiment, not
an equal-budget or hardware-normalized comparison.

## Accuracy and elapsed time

21 datasets × two data/search seeds (860, 5390) × two identical-seed repeats.
Each fit used 10,000 training rows and 25,000 held-out rows, population 8,192,
64 coefficient banks × 64 toggle assignments, and a 900-second maximum.
LM selected up to 512 genomes per generation, up to 32 bindings × four starts,
four proposal iterations, with at most eight parameters jointly fitted.
There was no CPU coefficient optimization or symbolic equivalence assessment.

| Measure | Result |
|---|---:|
| Held-out R² > .999 | 52/84 (61.9%) |
| Both train and held-out NMSE ≤ 10⁻⁶ | 29/84 (34.5%) |
| Reached training stopping threshold | 31/84 |
| Exhausted 900-second time allowance | 53/84 |
| Campaign elapsed, two concurrent GPUs | 7 h 3 min 47 s |
| Mean fit process elapsed | 600.44 s |
| Median fit process elapsed | 900.19 s |
| Sum of both workers' fit process times | 14.01 h |

Two training-threshold stops narrowly missed the strict holdout threshold:
`feynman_test_17-s5390`, both repeats, at holdout NMSE 1.0076e-6 and 1.0884e-6.
Twenty-one fits passed R² > .999 but still exhausted the allowance because the
training stopping criterion is stricter. Reported process overruns were at most
1.02 seconds. Repeat results were 23/42 and 29/42 numerical successes; those are
repeats, not independent additional data seeds.

## Where time went

Exclusive stage totals below are summed across both concurrent workers. They
are not campaign elapsed times. The residual contains generation, setup,
selection, validation, teardown, process overhead and other host work.

| Stage | Summed time | Share of fit process time |
|---|---:|---:|
| LM fitting, including proposal rescoring | 10.09 h | 72.02% |
| Broad scoring | 3.86 h | 27.53% |
| Everything else | 3.76 min | 0.45% |

The LM total contains 36,071.83 seconds measured with GPU events and 108.19 seconds
of native proposal rescoring. These are nested measurements, not additional time.
The recorded LM event interval accounts for about 99.3% of LM stage elapsed;
host setup and grammar generation are not the dominant cost here. This does not
measure achieved occupancy or prove the interpreter kernels are optimal.

Work counters: 132,276,224 AST occurrences, **541,803,413,504 broad configurations**,
528,580,608 additional LM proposal-rescore configurations, and
**50,838,505,720,000 attempted LM row/Jacobian evaluations**. AST occurrences are
not unique structures; LM attempted rows include invalid-domain early exits.
The latter is a separate unit and must not be added to configuration counts.

## Comparison with retained results

Same diagnostic problems and two seed identities; current rows have two repeats,
historical LM has one. These are observed complete-system results with materially
different budgets, implementation and hardware, not a controlled speed test.

| Search | Limit per fit | Numerical successes | Mean recorded fit |
|---|---:|---:|---:|
| Recent toggle baseline, fitting disabled | 60 s | 39/84 (46.4%) | 48.33 s |
| Recent toggle search, power mutations | 60 s | 44/84 (52.4%) | 46.66 s |
| This toggle GPU LM campaign | 900 s | 52/84 (61.9%) | 600.44 s |
| Historical settings + best integrated LM | 60 s | 35/42 (83.3%) | 32.55 s |

The current run has 13 more successful fits in aggregate than the recent baseline,
while average process time is 12.4 times greater. Both LM and the time limit changed,
so this does not isolate LM's contribution. Historical LM used a 5090, different
GP/fitting behavior and persistent execution; no hardware-normalized speed ratio
is justified. Its success rate remains substantially higher despite its smaller
recorded allowance. The selected panel cannot be compared directly with the old
full-suite 92.67% dataset-aggregated score.

[Historical comparison and provenance](finalists-history-comparison-20260921.md).

## Per-dataset outcomes

Four fits per dataset. Strict means both training and holdout NMSE ≤ 10⁻⁶.
Displayed R² values are rounded; a printed 1.000000 does not prove exact recovery.

| Dataset | R² > .999 | Strict | Mean fit | Holdout R² range |
|---|---:|---:|---:|---|
| III_7_38 | 4/4 | 4/4 | 0.99 s | 1.000000–1.000000 |
| III_8_54 | 4/4 | 2/4 | 553.65 s | 0.999832–0.999999 |
| II_11_27 | 4/4 | 4/4 | 31.46 s | 1.000000–1.000000 |
| II_35_18 | 2/4 | 1/4 | 764.83 s | 0.973339–1.000000 |
| II_6_11 | 4/4 | 4/4 | 10.69 s | 1.000000–1.000000 |
| II_6_15a | 3/4 | 0/4 | 900.50 s | 0.997973–0.999489 |
| I_12_11 | 4/4 | 4/4 | 19.34 s | 1.000000–1.000000 |
| I_24_6 | 4/4 | 2/4 | 507.89 s | 0.999404–1.000000 |
| I_32_17 | 2/4 | 0/4 | 900.16 s | 0.994481–0.999985 |
| I_39_11 | 4/4 | 4/4 | 0.80 s | 1.000000–1.000000 |
| I_40_1 | 0/4 | 0/4 | 900.56 s | 0.953289–0.995862 |
| I_6_2 | 4/4 | 1/4 | 683.12 s | 0.999942–1.000000 |
| I_8_14 | 0/4 | 0/4 | 900.32 s | 0.950348–0.993309 |
| test_1 | 0/4 | 0/4 | 900.19 s | 0.967673–0.991426 |
| test_11 | 0/4 | 0/4 | 900.35 s | 0.991678–0.996158 |
| test_13 | 2/4 | 0/4 | 900.34 s | 0.993480–0.999624 |
| test_14 | 0/4 | 0/4 | 900.51 s | 0.977504–0.989827 |
| test_16 | 0/4 | 0/4 | 900.29 s | 0.989514–0.995386 |
| test_17 | 4/4 | 1/4 | 519.32 s | 0.999999–0.999999 |
| test_18 | 3/4 | 2/4 | 513.39 s | 0.987216–1.000000 |
| test_5 | 4/4 | 0/4 | 900.64 s | 0.999721–0.999995 |

## What to do next

Keep the zero-local-memory kernels, but do not treat this as demonstrated search
parity. Fifteen of 21 datasets had at least one numerical success; six had none:
I.40.1, I.8.14, test 1, test 11, test 14 and test 16.

Before another broad campaign, use a small matched-budget panel to separate
candidate selection/diversity, coefficient parameterization and promotion behavior
from fitting throughput. Historical search reopened leaves on concrete models;
this controller retains whole toggle genomes and selects the best distinct genome
fingerprints. That is an actionable policy difference, not a proven explanation
for the quality gap. Compare shorter fitting allocations and restarts against the
current depth-first time allocation. Retain training-only decision making.

Remaining original work:

2. This campaign analysis is complete. Establish matched-hardware quality parity
   with the historical LM policy before expanding SRBench coverage.
3. Measure achieved occupancy and tune arbitrary AST packs and LM parameter shapes.
4. Tune fitting candidate diversity, budgets, starts and parameter handling;
   derivative SASS specialization remains separate future work. Current fitting
   still interprets operators, while broad scoring and proposal rescoring use SASS.

Raw results: `runs/lm-quality-20260921/shard{0,1}/results.json`. Executable, adapter,
runner and manifest identities are retained in each shard's `plan.json`. Local
notification receipt: `runs/lm-quality-20260921/telegram-receipts.json`.
[Compact metrics, per-fit equations and raw-result hashes](data/toggle-lm-results-20260922.json).
