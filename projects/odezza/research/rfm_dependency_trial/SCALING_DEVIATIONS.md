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

# Scaling experiment scope and reporting correction

## Controlled cohort — valid, distinct from original pilot

This is the explicitly requested support-controlled state-count expansion.
Equal counts of 1/2/4-input RHSs replace the original randomly varying support
distribution. Duration is fixed at .5 to isolate dimensions/counts. Eight
development and twenty held-out systems per dimension; 84 systems total. All
28 matched template replicates generated without integration rejection. Raw
template generation took 43,333 draws because exact support is conditioned on.
Four-input depth-two templates necessarily use binary arithmetic operators only.
There are no CPU fallbacks, reduced populations or missing fitting cells.

## Calibration grid floor — repaired after prediction freeze

The initial scaling grader reused the pilot's threshold candidates: zero plus
logarithmically spaced values from 1e-6 to one. At n=24, diagonal RFM development
true-input scores sometimes lie below this floor. For 16 trajectories, the
largest threshold meeting the 99% empirical coverage goal is 9.77e-10; for 32
trajectories it is 6.79e-7. The grid could choose only zero, retaining every state.

This is a selection/reporting limitation, not evidence that no useful cutoff
exists. Initial AP, fitted matrices and timings remain valid. Initial selection
reports are retained locally under `scaling-01/report-grid-v1`; they describe the
coarse-grid policy and are not interchangeable with the corrected policy.

Repair: select the largest threshold meeting development coverage exactly using
an order statistic of each development RHS's minimum true-input score. No test
truth chooses the threshold, and no models, observations, hyperparameters or
cohort members change. One threshold still applies to all RHSs regardless of
their private support size. The final report records the rule and grader hash.
Regression test covers tiny scores below the old floor, coverage at the chosen
threshold and failure to meet coverage at the next larger representable threshold.

Owner/next action: RFM screening follow-up. Revisit the original pilot's coarse
threshold policy before operational reuse; preserve its already published reports.
Finite development-set coverage remains an empirical calibration target, not
a guarantee of future retention. Hard exclusion remains unproven.
