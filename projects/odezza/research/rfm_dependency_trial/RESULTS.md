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

# RFM dependency pilot results — 2026-09-08

Completed on rack1's two RTX 5080 GPUs, using its existing PyTorch environment.
The observation-only learner never received private ASTs or exact derivatives.
Generation, prediction, grading and the post-freeze derivative audit are separate
programs. No Odezza numerical kernel, service or search default was changed.

## Coverage

- 24 six-state systems: four development/calibration and 20 held-out systems.
- Every RHS independently sampled at structural depth two, including constants
  and repeated state leaves; declared outer time-scale factor 0.3.
- All states observed, noiseless, uniformly sampled at spacing .05.
- 480 count/duration cells and 120 equal-observation-budget cells, each comparing
  fixed Gaussian kernel regression, diagonal RFM and full RFM across all six RHSs.
  Total: 10,800 output-specific fitted-model results. Recursive methods use three
  metric updates. Exact dense float64 kernels, no approximation or subsampling cap.
- 120 test RHSs: 11 with no state dependency, 52 with one, 45 with two, 11 with
  three, one with four. Empty support includes both zero and constant equations.
- 24 accepted systems in 25 attempts. The bounded-trajectory acceptance condition
  is part of the sampled distribution. This is predominantly a sparse cohort.

## Multiple initial conditions dominate additional duration here

Mean average precision (AP) over 109 test RHSs with nonempty support; 1 means
every actual dependency ranks above every irrelevant state. It does not mean a
threshold recovers the exact support or that the RHS expression is recovered.

| Trajectories | Duration | Fixed kernel AP | Diagonal RFM AP | Full RFM AP |
|---:|---:|---:|---:|---:|
| 1 | .5 | .4911 | .4919 | .4930 |
| 1 | 4 | .5665 | .5695 | .5840 |
| 4 | 1 | .8623 | .8955 | .8485 |
| 8 | 2 | .9850 | .9914 | .9796 |
| 16 | .5 | .9977 | .9977 | 1.0000 |
| 16 | 4 | .9946 | .9935 | .9939 |

![Dependency accuracy](report-01/dependency_accuracy.png)

The equal-budget experiment fixes 80 observed state vectors (480 scalar state
values) per system. Discarding spline endpoints gives fewer derivative training
rows as trajectory count grows, so this does not favor more trajectories through
a larger training matrix.

| Trajectories | Samples each | Duration | Fixed AP | Diagonal AP | Full AP |
|---:|---:|---:|---:|---:|---:|
| 1 | 80 | 3.95 | .5599 | .5664 | .5797 |
| 2 | 40 | 1.95 | .6541 | .6944 | .6647 |
| 4 | 20 | .95 | .8610 | .8963 | .8465 |
| 8 | 10 | .45 | .9607 | .9857 | .9481 |
| 16 | 5 | .20 | .9969 | .9931 | .9972 |

This supports a coverage explanation: independent initial conditions break state
correlations that a single orbit leaves ambiguous. The comparison deliberately
changes duration while holding total observations fixed; it is not a universal
claim that shorter trajectories are better for every system.

## Ranking is stronger than threshold calibration

Thresholds were selected on four separate development systems to retain all
inputs for at least 95% of nonempty-support RHSs. Test truth did not set thresholds.

At 16 trajectories and duration .5:

| Method | All true inputs retained, nonempty RHSs | Mean states retained, all RHSs | Exact support, all RHSs |
|---|---:|---:|---:|
| Fixed | 108/109 (99.08%) | 1.700 | 88.33% |
| Diagonal | 103/109 (94.50%) | 1.458 | 95.00% |
| Full | 106/109 (97.25%) | 1.533 | 93.33% |

Full RFM ranks perfectly in this cell but drops a true input for three equations
at the calibrated cutoff. Predictive relevance magnitude is not a probability
of presence. Four development systems cannot establish high-confidence exclusion
rules. Zero-support rows are excluded from AP and separately counted; exact
support rates above include them. JSON also reports nonempty-only exact recovery
and retained-state counts.

The full metric is not consistently better than the diagonal metric or fixed
kernel. For example, with four trajectories of duration one, diagonal improves
AP .8623 -> .8955 while full lowers it to .8485. With enough observations, all
three rank well. These fixed-hyperparameter results do not establish which model
would win a thorough independently tuned comparison.

## Speed and validation

Summed cell elapsed times: 24.70 seconds for the main sweep and 2.20 seconds for
equal-budget comparisons, across both workers. These sums include fitting all
three methods, matrix extraction, validation predictions and per-cell
preprocessing. They are not full experiment wall time: Python startup, file reads
outside cells, generation, report generation and development are excluded.
Generation took 7.31 seconds. A method's mean standalone-equivalent fitting time
includes its shared initial fit; those method times must not be summed as actual
execution time because the initial fit was reused.

- Full RFM, all six RHSs, 16 trajectories of duration .5: about 13.6 ms.
- Full RFM, all six RHSs, equal-budget 16 trajectories: about 7.3 ms.
- Full RFM, all six RHSs, 16 trajectories of duration 4: about 181 ms.

Five focused tests passed: symbolic cancellation, analytic gradients versus
autograd and finite differences, metric/zero-target handling, exclusion of future
and unselected observations, and tied ranking/false-negative scoring semantics.
All 600 cells completed without fitting errors. Atomic outputs include hashes
and configuration fingerprints; resume rejects a provenance mismatch.

The post-freeze audit compared observation-derived derivatives to private true
RHS evaluations, without changing any model or threshold. Across 120 RHSs:

| View | Median relative derivative RMSE | 95th percentile | Maximum |
|---|---:|---:|---:|
| 1 trajectory, duration 4 | 3.93e-8 | 1.17e-6 | 1.35e-5 |
| 16 trajectories, duration .5 | 1.53e-7 | 8.75e-7 | 2.02e-6 |
| 16 trajectories, duration 4 | 5.59e-8 | 8.80e-7 | 9.95e-5 |

These small errors support the interpretation that state coverage, rather than
derivative-label noise, explains much of the ranking gap in this cohort.

## Next decision

1. Keep this as a separate dependency-proposal stage. Use scores to allocate
   initial state-binding toggles with an exploration floor; do not prohibit
   low-scoring states based on this calibration.
2. Freeze a fresh support-stratified cohort with more 3/4-input interactions and
   larger state counts. Then test depth three, wider timescales, noise and sparse
   observations. Retain generation failures and clearly label changed cohorts.
3. Expand independent calibration and measure stability across trajectory subsets.
   Benchmark conservative all-input retention, not just attractive AP heatmaps.
4. Only then measure end-to-end recovery time with and without the proposal stage,
   including RFM time. Initial correlated state toggles and per-permutation winner
   retention remain the relevant unfinished Odezza API work.

Artifacts: [configuration](pilot.json), [method and commands](README.md),
[main summary](report-01/summary.csv), [equal-budget summary](equal-report-01/summary.csv),
[top-k curves](report-01/topk.json), [example matrices](report-01/example_matrices.png),
[derivative audit](report-01/derivative-audit.json), [limitations and corrections](DEVIATIONS.md).
