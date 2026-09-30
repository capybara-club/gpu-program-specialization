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

# Observation-only RFM dependency pilot

Separate research experiment; no Odezza solver/kernel modifications. Implements
Gaussian recursive feature machines (RFM), including full/diagonal AGOP metric
updates, alongside fixed Gaussian kernel regression. Mathematical background and
planned follow-ups: [research protocol](../../docs/rfm_dependency_screening_research_2026-09-08.md).

## Environment and commands

Runs using rack1's existing `/home/cdurham/odeformer-trial/.venv/bin/python`:
PyTorch 2.14.0+cu130, NumPy 1.26.4, SciPy 1.17.1, plus SymPy and Matplotlib.
No packages were installed. This is a local implementation of the RFM algorithm,
not a claim to benchmark the authors' software. Float64, exact dense Gaussian
kernels, fixed bandwidth/ridge and three metric updates; no sample cap.

From this directory, using a Python with those packages:

```sh
python -m unittest -v test_trial
python generate.py pilot.json pilot-01
python learn.py pilot-01 results-01 --shards 2 --shard 0 --device cuda:0
python learn.py pilot-01 results-01 --shards 2 --shard 1 --device cuda:1
python report.py pilot-01 results-01 report-01
python learn.py pilot-01 equal-01 --equal-budget 80 --device cuda:0
python report.py pilot-01 equal-01 equal-report-01 --equal-budget
```

The two main learner commands can run concurrently; each owns different cases.
Completed cells are atomic and resumable only with matching data/code/config
fingerprints. `--limit` is a development smoke option, not used for final runs.

## Frozen pilot choices and scope

- Six states; a separate random AST for every RHS. Operators add/sub/mul/sin/cos/
  tanh; constant leaves drawn uniformly from [-1,1] with probability 0.2, otherwise
  uniform state leaves. Every leaf starts at operator depth two; simplification
  can reduce depth/support. Unary/binary choice is uniform over the six operators.
- An outer coefficient 0.3 sets the time scale and does not count toward the
  declared structural depth. No stabilizing linear term is added. Max four state
  leaves before simplification. All states observed and all RHSs unknown to learner.
- Initial states independent uniform [-1,1]. DOP853 rtol=1e-10, atol=1e-12,
  max_step=0.1, reject any system failing integration or crossing |state|=20 in
  any of its 20 trajectories before t=4. Record rejections; this is a conditioned
  finite-horizon cohort, not an unbiased sample of all ODEs.
- Four development systems calibrate score thresholds; 20 independent systems
  evaluate them. The learner receives only public observations, times and generic
  configuration. It never imports generation/grading code or reads private ASTs,
  supports, exact derivatives or generation seed.
- Nested counts 1/2/4/8/16 and durations .5/1/2/4; spacing .05. Each cell estimates
  derivatives only from its own time prefix using CubicSpline, discarding both
  endpoints. Four additional ICs provide derivative prediction diagnostics but
  do not select model hyperparameters or matrix updates.
- Input/target standardization uses training rows only. Kernel bandwidth sqrt(n),
  ridge=1e-4 added to K's diagonal, metric trace normalized to n, identity floor
  1e-4. Final AGOP is recomputed after the final fit, before any metric floor.
- Equal-budget experiment: exactly 80 observed state vectors allocated across
  1/2/4/8/16 trajectories. Durations 3.95/1.95/.95/.45/.20 respectively. Derivative
  fitting rows differ because each trajectory loses two endpoints. This tests
  observation allocation, not identical training-matrix sizes.

## Reports and accounting

Every cell retains all six full feature matrices for each method, input hash,
code/config fingerprint, scales, prediction diagnostics and timing. Raw AGOPs
use standardized input/output coordinates. To recover physical-unit diagonal
sensitivities multiply each entry by target_scale^2 / input_scale^2.

For ranking/thresholding, each RHS's diagonal is divided by its largest entry.
Rows whose maximum is below 1e-20 are treated as zero score. These are relevance
scores, not probabilities. Thresholds are chosen separately per method/count/
duration on development systems to retain all true inputs in at least 95% of
nonzero RHSs. This is a small calibration sample, not a guarantee of 95% test
coverage. AP excludes zero-support RHSs; other metrics identify them explicitly.
Self-dependencies are included. Top-k curves break ties by state index; tied
scores receive grouped thresholds in average precision.

AP confidence intervals bootstrap whole test systems (1,000 resamples), not
individual time rows. Per-trajectory refit uncertainty, sparse/noisy views,
hidden states, polynomial baselines and downstream recovery speedups are future
experiments, not measured by this pilot.

Method timings include the common initial kernel fit plus that method's updates
and validation. Shared initial work is reused during execution, so summing the
three standalone-equivalent method times double-counts shared work. Full cell
elapsed time is retained separately. GPU events wait for the relevant work when
timing; the Odezza runtime is untouched.

## Validation

Focused tests check analytic RFM gradients against autograd and central finite
differences, positive metric regularization, zero-RHS behavior, symbolic
cancellation, and invariance to hidden future/unselected trajectory observations.
The final report checks all expected cells exist before reporting a complete run.
Private solutions enter only in the post-run grader.
