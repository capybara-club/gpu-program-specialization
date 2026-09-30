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

# RFM screening for ODE state dependencies

Research and proposed experiment, 2026-09-08. No experiment has been run and no
solver defaults or numerical kernels have been changed for this proposal.

## Research assessment

The proposal has strong precedents in nonlinear variable selection and ODE
network reconstruction. Targeted searches did not find a direct validation of
the exact pipeline: recursive feature machines (RFMs), trained from trajectory
observations, estimating every RHS's state support before symbolic search.
This is a promising empirical question, not an established recovery guarantee.

- Radhakrishnan et al., [RFM paper](https://arxiv.org/abs/2212.13881), 2022/2023:
  alternate kernel regression with a metric update based on the average gradient
  outer product (AGOP). Demonstrates feature learning; does not establish ODE
  support recovery from temporally correlated observations. Its discussion of
  spurious features also cautions against equating predictor relevance with true
  physical dependency.
- Rosasco et al., [Nonparametric Sparsity and Regularization](https://jmlr.org/papers/v14/rosasco13a.html),
  JMLR 2013: nonlinear variable selection through partial-derivative norms, with
  prediction and selection consistency results under assumptions. No linear or
  additive restriction. These results concern that estimator, not arbitrary RFM
  training on trajectories.
- Dai and Li, [Kernel Ordinary Differential Equations](https://arxiv.org/abs/2008.02915),
  JASA 2022: kernel estimation and sparse network recovery from noisy observations,
  using smoothed trajectories and integral matching. Includes individual and
  pairwise interaction components, with selection consistency under assumptions.
  It is close to the proposed application, but its pairwise ANOVA model does not
  represent every nested AST with higher-order interactions.
- Qiu et al., [Identifiability Analysis of Linear Ordinary Differential Equation
  Systems with a Single Trajectory](https://arxiv.org/abs/2103.05660), 2021:
  even linear ODE identification depends on trajectory excitation and initial
  conditions. One trajectory can suffice in some settings, but is not a universal
  source of enough information.

## Define the object being estimated

Let dx_i/dt = f_i(x). Fit a separate scalar RFM predictor fhat_i for every state,
using only observed states and observation-derived derivative estimates. The
learner receives neither the private AST nor exact RHS evaluations/derivatives.

For standardized input coordinates z, recompute the final predictor's AGOP:

    M_i = mean_a [grad_z fhat_i(z_a) grad_z fhat_i(z_a)^T]
    S[i,j] = M_i[j,j]

Each M_i is n by n. Stack its diagonal into row i of the directed dependency score
matrix S. The sample Gram matrix is a different object, N by N. Off-diagonal
entries in M_i measure products of input sensitivities for output i; they are
not edges between their two input states, nor guaranteed interaction detectors.

Save the full M_i, raw and scaled diagonal scores, scaling transforms and
trajectory-resampled uncertainty. Compute gradients in the original state axes
after diagonal rescaling; do not report rotated latent coordinates as states.
Do not mistake an identity regularization floor for relevance or force a
zero-RHS row to have nonzero normalized importance. Scores are not calibrated
probabilities by default.

For the true smooth vector field, a missing input has identically zero partial
derivative. Squared derivatives avoid signed cancellation, but small average
sensitivity can also mean weak coefficients, poor excitation or saturation.
Finite trajectory data can admit multiple off-trajectory extensions with different
dependencies. Good prediction MSE alone does not validate the inferred support.

Do not train on finite-horizon next-state prediction and interpret its gradients
as direct RHS dependencies. For dx1/dt=x2, dx2/dt=x3, dx3/dt=0:

    x1(t+h) = x1(t) + h*x2(t) + h^2*x3(t)/2

The flow depends on x3, while f1 does not. Finite differences approximate the
derivative only as the interval becomes sufficiently small.

## Pilot protocol

1. Start with six observed states, a random depth-at-most-two AST for **every**
   RHS, and 20 accepted systems. This differs from earlier games with only one
   unknown RHS. Freeze unary/binary operators, coefficient distribution, depth
   convention, initial-condition distribution and integration acceptance rules.
   Ordinary unary/binary depth two admits at most four leaves; document this
   built-in sparsity. Expand later to 3/10 states and depth three.
2. Retain generation failures, cancellations, constant/zero equations and the
   acceptance rate. Keep syntactic and effective functional support separate.
   Grade effective support on the declared domain, auditing cancellations such
   as x-x; numerical probes alone cannot certify global absence. Freeze a
   common maximum-horizon eligibility rule so each duration sees the same cohort.
3. Generate a private high-accuracy master trajectory set. Build nested public
   views using the same systems and initial-condition pool. The learner only
   sees the requested view, never the denser master trajectory or generator seed.
4. Sweep trajectory counts 1, 2, 4, 8, 16 and durations 0.5, 1, 2, 4 in fixed
   declared model time units. Initially hold sample spacing at 0.05. Adjust these
   provisional ranges after a development-only timescale check, then freeze them.
   Independently vary spacing and include equal-total-observation comparisons.
   Longer duration, denser sampling and more initial conditions are different
   interventions. Record actual sample count and occupied state ranges.
5. Begin noiseless and fully observed. Estimate derivatives with an explicit
   spline/local-polynomial procedure, excluding unreliable endpoints. Select
   smoothing and kernel settings on separate development systems and validation
   trajectories. No random row splits across the same trajectory for validation.
6. Compare full-metric RFM, diagonal RFM and fixed isotropic kernel regression
   with the same final AGOP scoring. This measures whether recursion and learned
   off-diagonal geometry improve selection. Include a cheap sparse polynomial
   derivative baseline and label its representational limitations.
7. Use whole-trajectory resampling for stability when multiple trajectories
   exist; label single-trajectory block resampling separately. Aggregate
   uncertainty across independent systems, not as if every edge were independent.
8. Add noise and sparse sampling as separate phases. Integral or weak-form kernel
   learning is a justified follow-up, but integrating the unknown predictor in
   the loss requires a custom training path; ordinary RFM regression does not
   automatically provide it. Hidden states are another experiment entirely.

## Objective scores

Private truth A[i,j] means that f_i actually depends on x_j on the declared
domain. The learner never receives A or each row's true support size.

- Average precision / precision-recall curves for ranking states.
- Edge precision and recall, exact row-support recovery and false-positive /
  false-negative counts at thresholds frozen on development data.
- **Fraction of RHSs retaining every true input**, versus mean number of states
  retained. This is the primary safety/utility curve for symbolic search.
- Retained-state count at high recall, such as 95% and 99%, with uncertainty.
  Do not choose test thresholds using test truth. Oracle recall-at-true-k may be
  a labeled diagnostic, never an operational rule.
- Include self-dependencies and report off-diagonal edges separately. Report
  zero-support rows separately because ordinary positive-edge ranking metrics
  are undefined there. Avoid raw accuracy dominated by absent edges.
- Measure preprocessing, RFM fitting, matrix extraction and total time/memory.
  Exact dense kernels require quadratic sample storage; any subsampling or
  approximation must be explicit and use the same available observations.

Deliverables: per-system truth/score matrices, aggregate recall-versus-retained-
states curves, duration/count heatmaps, uncertainty, failures and full timing.
Derivative-label accuracy may be audited against private truth only after outputs
are frozen; it must not select the primary model or its hyperparameters.

## How to use a successful result in Odezza

Begin with soft state-binding priorities for the first toggle screen. Allocate
more work to well-supported states while preserving broad exploration; do not
silently turn uncertain low relevance into a forbidden state. Use a held-out
matched recovery comparison to establish net time saved, including RFM cost.

Dependency probabilities belong in the experimental search controller and request
metadata. The scorer still receives concrete ASTs, toggle layouts, constants and
observations. No kernel-shape change is implied by this experiment.

Working hypothesis: additional varied initial conditions often help support
recovery more than extending an already redundant trajectory. This is what the
matched observation-budget experiment tests; no monotonic improvement or speedup
is claimed in advance.
