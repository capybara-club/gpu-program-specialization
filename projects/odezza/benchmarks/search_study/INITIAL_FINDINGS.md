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

# Initial findings from the September 10 search-design study

This is an interim interpretation. The [live report](runs/20260910-robustness/REPORT.md)
updates as the frozen random searches and subsequent benchmark supplement finish.
Do not treat the faster early cases as the completed cohort's success rate.

The useful questions are: how much structural exploration to buy before fitting,
how many starts and bindings to spend on each candidate, which native fit width
to use for that population, and how to stop numerical or candidate failures from
disrupting otherwise useful searches.

## Completed fitting calibration

All 648 planned calibration trials have reported: 624 complete populations and
24 expected exact-shape resource rejections. There were no partial fitting
populations. This is an explicitly planted-structure calibration, not blind
recovery. Sixty completed trials rejected at least one candidate during CPU
replay; these records remain excluded from verified-winner selection.

The 108 comparisons below match the candidate bank and retained candidate output
hashes across the available widths. Each cell is one GPU × state count ×
coefficient count. Three GPUs cover nine 3/6/8-state × 1/3/6-coefficient cells.
The same input systems across machines are correlated observations.

| Starts per AST | Fastest 1 lane | Fastest 2 lanes | Fastest 4 lanes | Fastest 8 lanes | Median slowest / fastest |
|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 8 | 0 | 11 | 1.12× |
| 64 | 0 | 4 | 2 | 21 | 1.22× |
| 1,024 | 0 | 4 | 2 | 21 | 1.22× |
| 16,384 | 21 | 6 | 0 | 0 | 2.16× |

This supports testing a dispatch rule that uses population size as well as state
and coefficient counts. Register feasibility alone is insufficient. It does not
yet establish a production lookup table: these are single measurements, with
template/cache history and exact supported widths recorded in the raw profiles.
Use new structures and repeated accuracy-matched measurements to confirm a rule.

## Search scheduling still matters

At the 71-search interim snapshot, 2,794 of 2,795 native fitting calls contained
at most 128 fits. They accounted for about 99.96% of recorded native runner time.
That is a strong signal that this search controller is still issuing tiny fitting
populations. It is not a hardware occupancy measurement, and it does not prove
that arbitrary extra starts improve recovery.

The eight fixed random policies will distinguish spending on more ASTs, larger
coefficient banks, fewer bindings, more candidates, fewer candidates, more starts,
and curvature versus native LM. They share an initial budget of 268,435,456
configuration slots per case. Compare gained/lost recoveries and budget-penalized
time on matched cases before changing defaults. A longer descriptor queue alone
did not help the earlier two-stream pilot materially.

The highest-priority remaining performance measurement is a correlated loader/
device timeline. Module-load wall time and kernel event intervals overlap and
cannot be added to form an exclusive time breakdown. This study records their
profiles but does not claim that attribution has been resolved.

## Failure containment and numerical fidelity

The study already found a reproducible candidate-domain bug that stopped one GPU
queue. Its [incident and repair](../../docs/incidents/2026-09-10-singular-lm-proposals.md)
show that both candidate-level domain checks and process-level recovery deserve
priority. An invalid proposal should not consume the remainder of a machine's
campaign budget. Unreported trials must remain separate from unsuccessful
completed searches.

Keep independent trajectory replay. Some rational candidates can score well at
coarse resolution yet fail finer integration, while explicit zero divisors can
produce misleading finite IEEE results when nested. The former needs a measured
coarse/finer promotion policy; the latter can be rejected cheaply before fitting.
Neither finding warrants narrowing the grammar to a guessed answer.

## What the benchmark supplement will add

Twelve preselected qualified noiseless tasks from MDBench, ODEBench, and biological
models compare three fixed policies: native LM, curvature, and native LM with up
to 21 screening observations instead of seven. There are 36 trials, each with a
180-second budget and raw-unit MSE target 1e-6. The original source data and ICs
are used; one RHS is hidden and derivative targets are excluded.

This probes transfer beyond the random generator and the value of finer screening.
It is an adapted completion protocol, not an official leaderboard result. The
MDBench/ODEBench tasks share equation ancestry, and unresolved numerical cases,
noise, unsupported operators, and systems beyond eight states are outside this
pilot. A later stress matrix should deliberately cover those exclusions.

The benchmark version includes the candidate-domain repair, while the ongoing
random cohort remains frozen. Compare policies within each cohort, and use a
fresh random cohort before promoting any policy selected from these results.
