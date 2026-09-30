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

# State-count scaling protocol — 2026-09-08

This is a fresh, support-controlled experiment. The original pilot and its
predictions are unchanged. Use `scaling.json` to reproduce this run.

## Frozen design

- State counts: 6, 12, 24. Eight development systems and 20 held-out systems per
  dimension, for 84 systems total. Every RHS is unknown to the feature learner.
- Every system has equal numbers of RHSs with exactly 1, 2 and 4 state inputs.
  Sample depth-two templates with the original add/sub/mul/sin/cos/tanh grammar,
  conditional on effective support size. Verify symbolic partial derivatives,
  then bind abstract roles injectively to a uniform random subset of states.
  Store support labels and templates privately. Never provide true support size
  to the learner or use it to choose an operational threshold.
- Generate blocks of six templates with two of each support size. Larger systems
  reuse the smaller systems' abstract RHS templates for their shared output
  indices, but use new state bindings. Initial-condition coordinates are shared
  prefixes of a common 24-dimensional draw. These are matched template replicates,
  not the same six-state flow with inert variables appended: all dimensions evolve
  under their own coupled ODE system.
- At most two operator layers, with an outer 0.3 time-scale coefficient. Under
  this grammar, four-state support requires a full binary depth-two tree; its
  operators are necessarily add/sub/mul. Higher-order nested unary interactions
  require greater depth and are not represented by this stratum.
- Hold duration at .5 and spacing at .05, giving 11 samples per trajectory.
  Sweep 4, 8, 16, 32, 64 trajectories from nested initial-condition pools. Cubic
  spline derivatives use nine interior rows per trajectory: training sizes
  36, 72, 144, 288, 576. Four additional trajectories diagnose prediction error.
- All observations noiseless and complete. Initial conditions uniform [-1,1].
  DOP853 rtol=1e-10, atol=1e-12, max_step=.1, reject the entire matched replicate
  if any dimension/trajectory fails integration or exceeds |state|=20. Retain
  generation rejection counts and conditional AST proposal counts.
- Unchanged fixed, diagonal and full Gaussian RFM implementations. Float64 exact
  dense kernels, three updates, ridge 1e-4, bandwidth sqrt(n), metric trace n,
  identity floor 1e-4. No sample cap, CPU fallback or per-dimension tuning.

## Measures

- Dependency-ranking average precision (AP), by state count, trajectory count,
  method and true support size. Include random-ranking expected AP because
  chance performance changes with dimension and sparsity.
- System-bootstrap AP confidence intervals, 1,000 resamples of independent test
  systems. Coupled RHSs and time rows are not treated as independent systems.
- Select a single relative-score threshold per dimension/count/method using
  only development systems, targeting 95% and 99% all-input retention. Apply it
  to all held-out RHSs without knowing their support size. Report test retention,
  retained-state count, exact support, edge recall and edge precision, including
  support-stratified breakdowns. Calibration targets are not coverage guarantees.
  Use the exact order statistic of per-RHS minimum true-input scores. The initial
  logarithmic threshold grid from the earlier pilot had a 1e-6 floor and was
  repaired before final reporting; retained first-pass reports are separately
  labeled `report-grid-v1`. Predictions and their rankings are unchanged.
- Fixed top-k retention curves are descriptive test diagnostics. Never use private
  test support sizes or test-selected k as an operational selection rule.
- Standalone-equivalent per-method fitting times include common initial work;
  actual summed cell times reuse the common initial fit. Full experiment wall
  time, generation, startup and report production are distinct accounting scopes.

## Run

Use rack1's existing `/home/cdurham/odeformer-trial/.venv/bin/python` in this
directory. No installation is needed.

```sh
python -m unittest -v test_scaling test_trial
python scaling_generate.py scaling.json scaling-01
python learn.py scaling-01/n6 scaling-01/n6/results --device cuda:0
python learn.py scaling-01/n12 scaling-01/n12/results --device cuda:0
python learn.py scaling-01/n24 scaling-01/n24/results --device cuda:0
python scaling_report.py scaling-01
```

For two GPUs, run each learner with `--shards 2 --shard 0 --device cuda:0` and
`--shards 2 --shard 1 --device cuda:1` in separate processes. Grading starts only
after both shards finish. Existing learner/code/data fingerprint checks govern
resume; support-controlled generation and grading are separate additions.

## Interpretation boundary

The new support distribution is deliberately harder than the original pilot,
which had mostly one- and two-input equations. Compare dimensions within this
new cohort; do not attribute differences from the original pilot to state count
alone. Noise, missing observations, depth-three targets and downstream symbolic
recovery remain separate follow-ups. No search defaults or kernel shapes change.
