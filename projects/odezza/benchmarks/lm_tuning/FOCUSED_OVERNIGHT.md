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

# Focused overnight experiment for specialized Odezza

The next run separates search allocation from native fitting geometry. The broad
grammar, verification threshold, damping and step bounds stay fixed. Preparation
does not launch the overnight work. No native core code or production default is
changed by this experiment.

## What the previous results suggest

- The hour audit matched all completed screen counts, but 40% of native fitting
  calls had at most 128 fits across two kernels, consuming 27% of fitting time.
  Useful work per submission deserves attention before another broad random sweep.
- Only 180/480 old prepared candidate prefixes contained the planted structure.
  LM verified 173/180 and curvature 164/180 in that subset. Overall 253/480 versus
  240/480 mixed optimizer performance with missing structural coverage. These
  are historical observations from before the sensitivity correction.
- The toggle overnight recorded 498/570 baseline successes versus 477/570 after
  trading independent seed structures for more bindings. More configurations
  did not compensate for reduced structural diversity.
- The prepared scoring experiment found a useful 768→1,024 bank step but also
  redundant evaluations of coefficient-free structures. Raw counts are not the
  scientific objective.
- Ordinary-CUDA lane rankings changed with population. Those results do not
  rank the current specialized native implementation; measure it directly.

Evidence: [hour audit](runs/20260910-hour1-register-replay/UTILIZATION.md),
[toggle review](../../scratch/toggle_search_trial/validation/overnight-20260908a/TIMING_REVIEW.md),
[prepared scoring](../2026-09-07-first-pass-saturation.md),
[ordinary-CUDA limitations](../../scratch/fitting_batch_trial/plain_cuda_lm/OCCUPANCY_RESULTS.md).

## Fixed search policies

Every policy for a case uses the same public problem and search seed on the same
GPU. Order rotates. The private equation seed, RHS and true dependency subset
never enter search. All policies use the same supplied depth-three grammar.

| Policy | Initial seed ASTs | Bank rows | State bindings | Fitted candidates × starts |
|---|---:|---:|---:|---:|
| Native LM baseline | 65,536 | 1,024 | 4 | 64 × 4 |
| Curvature control | 65,536 | 1,024 | 4 | Existing screen-derived start semantics |
| More structures | 262,144 | 256 | 4 | 64 × 4 |
| More coefficients | 16,384 | 4,096 | 4 | 64 × 4 |
| Fewer bindings | 131,072 | 1,024 | 2 | 64 × 4 |
| Fit more candidates | 65,536 | 1,024 | 4 | 128 × 2 |
| Fit fewer candidates | 65,536 | 1,024 | 4 | 16 × 16 |
| More fitting starts | 65,536 | 1,024 | 4 | 64 × 16 |

Each initial screen requests **268,435,456 configuration slots**. The candidate
allocation contrasts preserve 256 nominal candidate/start fits. Binding padding
and concrete uniqueness are separate from independent AST diversity. Subsequent
waves use the corresponding initial AST scale, up to 64 nominal waves and the
API's 10M-offspring cap. Absolute offspring caps differ and are retained in JSON.
All policies stop at verified recovery or the same **240-second** wall budget.
Their adaptive candidate paths are not expected to stay identical.

Native LM uses 16 iterations, 20-second fitting calls, one preferred lane and
explicit wider fallback. Curvature keeps its existing start semantics; identical
explicit-bank fitter comparisons belong to the calibration below. No PySR work
is included: this run answers internal allocation questions first.

## Prepared native calibration

Nine cells: **3/6/8 states × 1/3/6 declared coefficients**, run on all three hosts.
Each uses eight fixed ASTs: the planted structure and neighbours, without labels
or true coefficients. This is explicitly coefficient-fitting calibration, not
blind discovery. The original larger candidate pools remain retained. Declared
coefficient count need not equal identifiable parameter dimension.

Each cell has 24 trials:

- Banks 4/64/1,024/16,384 × exact widths 1/2/4/8: 16 trials.
- Automatic width with two versus 16 submitted packs at banks 4/64/1,024: six.
- Bank 64 without sibling-toggle packing, and bank 64 curvature: two.

Each has a 20-second admission budget. The 131,072-fit soft buffer cap can be
exceeded by one indivisible pack, whose hard limit is one million fits. Actual
pack sizes, widths, cache state, allocation/transfer time and native timings are
reported. The core still has two execution streams/module slots: a longer queue
reduces drain boundaries but does not add arbitrary concurrent tiny kernels.

The first pilot used 64 ASTs and exposed incomplete large-bank trials. Eight
fixed ASTs retain a large per-module bank while reducing total calibration time.
This was an engineering sizing decision; search survivor policies stayed fixed.
Exact resource errors, fallback and partial work remain explicit. Only complete,
input/output-matched populations qualify for equivalent-work timing comparisons.
Do not treat overlapping kernel/load spans as exclusive timing categories.

## Cohort, schedule and questions for the morning

The design requests 36 independent random systems: nine N/P cells × four draws.
Unknown dependence alternates between one/two states; each known RHS uses two
states with random operators. Views are dense or sparse with x0 unobserved and
complete initial vectors. Six trajectories span duration 0.5. Generation allows
three bounded draws per slot, retains every rejection and never silently changes
the known background to linear. Performance is conditional on these generation
and numerical acceptance rules, not arbitrary or stiff ODEs.

The prepared `focused-night-20260910` cohort accepted **33/36 cases**, covers all
nine calibration cells and schedules **912 trials**: 648 fitting and 264 blind
search. Conservative reservations including drain margins are 6.68/5.72 hours
on rack1's two workers and 8.70 hours each on rohini and Ada. The hard window is
**nine hours from launch**; completion can be earlier. Whole policy blocks must
fit before admission. Unstarted work is distinct from timeout or solve failure.

The report should answer:

1. Does structural breadth, coefficient sampling or fewer correlated bindings
   give more verified recoveries within the same time budget?
2. At 256 nominal candidate/start fits, is 128×2, 64×4 or 16×16 most useful?
   Does increasing to 64×16 improve recoveries enough to justify its cost?
3. Which native widths and bank sizes work best for each measured N/P/RHS/GPU?
   Do longer submissions help at identical banks and actual shapes?
4. Where does time go, and how much useful work actually ran? Report scheduled
   RK4 work separately from LM evaluations, padding, invalids and discarded work.

Summaries retain case/GPU/state/parameter/view strata, verified outcomes and
outcomes within budget, all-outcome medians, and budget-penalized times (a miss
costs the full budget). Do not rank solely by time among successful solves.
Repeated calibration cases across hosts are correlated. This is exploratory:
freeze the next policy and confirm on a new cohort before claiming a tuned win
or superiority to PySR. Numerical replay rejects and incomplete cells stay visible.

## Operation

The frozen build passed native tests on all three hosts. The first integration
pilot completed 51 trials, including six verified blind searches. The resized
final-build check completed all work in 36/36 fitting trials, with independently
verified winners. All eight search policies passed API preparation for three
representative systems (24 checks, without GPU submission).

Verification rejected 24 nonwinning candidate records across the final checks.
Four distinct candidate/coefficient records reproduce their GPU MSE on CPU at
the same 32 steps, but change materially at finer steps. Their rational forms
have poles inside the observed state range. These are numerical search traps;
the accepted winners passed independent replay. Retain this qualification when
interpreting raw GPU rankings. See the [readiness report](runs/focused-night-20260910/readiness/REPORT.md).

**Status: prepared and deployed; the nine-hour campaign has not started.**

To prepare and deploy another cohort:

```sh
python3 -m benchmarks.lm_tuning.focused prepare --root benchmarks/lm_tuning/runs/my_focused_night
python3 -m benchmarks.lm_tuning.focused deploy --root benchmarks/lm_tuning/runs/my_focused_night
```

To start the already prepared run in mac1 tmux:

```sh
tmux new-session -d -s odezza-focused-night -c /Users/cdurham/code/odezza \
  'python3 -B -m benchmarks.lm_tuning.focused run --root /Users/cdurham/code/odezza/benchmarks/lm_tuning/runs/focused-night-20260910 > /Users/cdurham/code/odezza/benchmarks/lm_tuning/runs/focused-night-20260910/coordinator.log 2>&1'
```

Deployment freezes source and runs native tests/preflight in separate host
directories. The coordinator checks source/group identity, owns a supervisor
lock and resumes collection without rerunning completed groups. GPU workers
run as bounded user services. One Telegram notification follows terminal
completion. Unexpected unsafe worker failures stop that GPU queue and remain
visible while other workers continue. No silent numerical fallback or retry of
ambiguous running work. Results: `SUMMARY.md` and `summary.json` in the run folder.
