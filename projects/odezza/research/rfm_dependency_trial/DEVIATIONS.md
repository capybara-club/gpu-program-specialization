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

# Scope and interpretation record

2026-09-08 pilot-01. Owner: RFM experiment follow-up.

The implemented primary comparison follows the six-state/depth-two, observation-
only, noiseless pilot. Both GPU shards completed with no CPU fallback, sample cap
or failed fitting cells. Frozen code/input fingerprints accompany every result.

## Limited support diversity — valid pilot, limited generalization

The sampled 120 test RHSs have support sizes 0/1/2/3/4 in counts 11/52/45/11/1.
One of 25 system-generation attempts was rejected by the declared integration
rule. The cohort is sparse and mostly low-interaction; only one RHS has four
inputs. The grammar excludes exp/div and the outer 0.3 factor slows dynamics.
These were explicit generation choices, not a post-result modification. Results
do not establish arbitrary-operator, high-order, stiff, noisy or hidden-state
recovery. Next: freeze a separate support-stratified harder cohort; do not replace
this cohort or silently combine its results with a changed distribution.

## Threshold generalization — ranking is not safe exclusion

On sixteen trajectories of duration .5, full RFM has AP=1 across all 109 test
RHSs with nonempty support. The development-calibrated threshold still loses a
true input on three of them (106/109 complete support retention). Four development
systems are too few to claim high-confidence safe pruning. Results are valid as
ranking/selection measurements, not a guarantee of 95% or 99% future coverage.
Next: larger independent calibration set and conservative/uncertainty-aware soft
priors before any hard state exclusions.

## Initial report corrections before final publication

The first grading output called every zero-support equation a `zero_rhs`, although
constant nonzero RHSs also have empty support. Renamed to `zero_support_rhs` and
recorded the distinction. Also corrected the system-bootstrap AP calculation to
weight sampled systems by their nonempty RHS count, matching the reported macro
over RHSs. Predictions, calibration thresholds and point estimates are unchanged.
Added regression tests for tied ranking scores, zero-support rows and the case
where edge recall obscures an incomplete RHS support set.

## Explicit follow-ups not measured here

Per-trajectory model-refit uncertainty, a sparse polynomial baseline, noisy/sparse
observation sweeps, hidden-state inference and downstream Odezza recovery timing
remain unimplemented. System-level AP bootstrap intervals are provided. The
equal-budget experiment fixes observed state vectors, not derivative training
rows: endpoint removal costs two rows per trajectory. This distinction is shown
in reports and is not an equal-matrix-size performance comparison.
