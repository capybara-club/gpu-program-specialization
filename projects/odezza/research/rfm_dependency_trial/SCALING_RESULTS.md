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

# RFM state-count scaling results — 2026-09-08

Completed: 84 fresh systems on rack1's two RTX 5080 GPUs, with six, twelve and
twenty-four states. Eight calibration and twenty held-out systems per dimension.
Every RHS has a depth-two AST; each system contains equal numbers of equations
with exactly one, two and four state inputs. Duration .5, sample spacing .05,
4/8/16/32/64 independent initial conditions. Fully observed and noiseless.

All 420 fitting cells completed with all three methods, producing 17,640 final
output-specific models. Seven tests passed. The original pilot, Odezza solver,
kernel shapes and search defaults are unchanged. Private ASTs and support sizes
were used only by generation, post-run grading and the post-freeze outlier audit.

## Main finding: more states require more initial-condition coverage

Mean dependency-ranking average precision (AP) for diagonal RFM, equally mixing
one-, two- and four-input RHSs. One means all true inputs outrank irrelevant inputs.

| Trajectories | 6 states | 12 states | 24 states |
|---:|---:|---:|---:|
| 4 | .8811 | .7242 | .5680 |
| 8 | .9817 | .8659 | .7209 |
| 16 | .9974 | .9536 | .8112 |
| 32 | 1.0000 | .9873 | .9229 |
| 64 | 1.0000 | .9984 | .9822 |

![State scaling](scaling-01/report/state_scaling.png)

Sixteen trajectories are already strong for six states. Twelve states benefit
from 32–64, while twenty-four need 64 to approach similar ranking quality. This
is an empirical scaling curve for the declared sparse grammar, not a sample-
complexity law. Chance AP changes with dimension/support; the report includes
its analytic expectation and chance-adjusted AP alongside raw AP.

## Four-input interactions are the remaining difficulty

Diagonal RFM at 64 trajectories:

| True inputs per RHS | 6 states | 12 states | 24 states |
|---:|---:|---:|---:|
| 1 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 1.0000 | .9889 |
| 4 | 1.0000 | .9953 | .9576 |

![Support scaling](scaling-01/report/support_scaling.png)

At 24 states, the three support strata each contain 160 held-out RHSs. One-input
dependencies are easy by 64 initial conditions. Four-input interactions remain
substantially harder. Pure four-input depth-two ASTs necessarily use binary
arithmetic; this cohort does not include a unary wrapper around a four-input AST.

## Recursive updates help, but full matrices do not win consistently

At 24 states:

| Trajectories | Fixed kernel AP | Diagonal RFM AP | Full RFM AP |
|---:|---:|---:|---:|
| 16 | .7335 | .8112 | .7451 |
| 32 | .8224 | .9229 | .8872 |
| 64 | .9470 | .9822 | .9698 |

The diagonal metric is a strong default candidate for further trials. The full
metric's additional degrees of freedom do not consistently improve recovery
under this fixed bandwidth/ridge/update protocol. None of these methods were
retuned separately for each dimension.

## High average ranking is insufficient for hard exclusions

An exact development-only cutoff targets 99% empirical all-input retention.
It is shared across all RHSs, without knowing a test RHS's support size.
Its test results at 64 trajectories are:

| States | Mean candidate states retained | RHSs retaining every true input | Systems retaining all inputs for every RHS |
|---:|---:|---:|---:|
| 6 | 2.317 | 118/120 (98.33%) | 18/20 (90%) |
| 12 | 2.429 | 239/240 (99.58%) | 19/20 (95%) |
| 24 | 3.625 | 471/480 (98.13%) | 13/20 (65%) |

The true mean support is 2.333 states. At 24 states, mean retained size is 1.00,
2.30 and 7.575 for true support 1/2/4 respectively. All-input retention in those
strata is 100%, 96.875% and 97.5%. Thus the small overall retained-state average
does not mean four-input equations can uniformly be reduced to four candidates.

The failure tail matters: for the four-input stratum at n=24, a fixed top-k
ranking cutoff would need k=15 to cover at least 95% of test RHSs, and k=21 for
99%. These are retrospective test diagnostics, not deployment thresholds.
Higher trajectory count improves ranking overall, but the separately calibrated
cutoff becomes more aggressive and can lower test coverage. More observations
and stricter safe-pruning guarantees are different objectives.

## An audited failure mechanism

After freezing predictions, we inspected all imperfect diagonal-RFM rankings at
64 trajectories: two of 240 RHSs at n=12 and 29 of 480 at n=24.

For n=24, replicate 010, RHS 6:

    dx6/dt = 0.3*x12*x19*(x1 + x15)

The actual standardized sensitivity score of x1 on the training observations is
about .565 relative to the strongest true input. The fixed kernel estimates .138;
diagonal RFM reduces it to .000682 and ranks this true input last among 24 states.
Observation-derived derivative labels have relative RMSE 5.63e-6 for this RHS.

For replicate 020, RHS 12:

    dx12/dt = 0.3*cos(x19)*tanh(x21)

The true relative sensitivity of x19 is .153, but diagonal RFM assigns .0000401
and ranks it last. Label relative RMSE is 1.33e-6. These failures show inaccurate
gradient estimation and reinforcement of underweighted inputs; small derivative
label errors alone do not explain them. This audit is diagnostic only: private
gradients were not fed back into models, thresholds or the experiment.

## Cost and validation

- Generation: 19.74 seconds; all 28 matched template replicates accepted, no
  integration rejection. Conditional AST generation used 43,333 raw draws.
- Actual summed fitting-cell elapsed time across both GPUs: 46.86 seconds. This
  includes the three methods, shared initialization, matrix extraction, prediction
  checks and per-cell preparation. It is not total experiment wall time: startup,
  file I/O outside cells, generation, grading and development are excluded.
- Diagonal RFM standalone-equivalent fitting at 64 trajectories, all RHSs:
  71 ms for six states, 143 ms for twelve, 283 ms for twenty-four.
- The largest cell has 576 derivative-training rows, exact dense float64 kernels;
  there were no CPU fallbacks, sample caps or failed cells.
- Tests cover exact support/depth after state binding, gradients versus autograd
  and finite differences, observation isolation, zero-target handling, ranking
  ties and calibration below the former threshold-grid floor.

## Reporting correction and comparability

The original logarithmic calibration grid had no positive thresholds below 1e-6.
At n=24 it unnecessarily selected zero in two diagonal-RFM cases. Replaced this
with the exact order statistic of development minimum true-input scores, with a
regression test. Fitted predictions, AP and timings did not change. Initial grid
reports remain in `scaling-01/report-grid-v1`; final reports record the exact
calibration rule and grader hash. See [correction record](SCALING_DEVIATIONS.md).

The balanced support distribution differs from the original pilot. Larger and
smaller systems share abstract template prefixes and initial-condition coordinate
prefixes, but their state bindings and coupled flows differ. Compare dimensions
within this cohort; do not treat these as the same flow with inert inputs added.

## Recommended next steps

1. Use diagonal-RFM rankings as soft initial state-binding priorities. Keep a
   broad exploration floor and per-permutation survivors in Odezza. Do not hard
   exclude states based on these calibration results.
2. Test conservative metric updates and agreement/stability across observation
   subsets. A relevant state must have a path back into the search after an
   early predictor underweights it. Measure worst-case retention and whole-system
   coverage, not just average AP or edge recall.
3. Use fresh calibration and held-out systems for larger-IC/noise/depth-three
   trials; preserve this cohort. Then benchmark end-to-end recovery with and
   without the proposal stage, including its cost. No search speedup is established
   merely by these feature rankings.

Artifacts: [configuration](scaling.json), [protocol and commands](SCALING.md),
[ranking](scaling-01/report/ranking.csv), [calibrated selection](scaling-01/report/selection.csv),
[top-k curves](scaling-01/report/topk.csv), [outlier audit](scaling-01/report/outlier-audit.json).
