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

# Coupled RHS benchmark proposal

Proposed 2026-09-14. Not generated, benchmarked, deployed, or demonstrated to favor
Odezza. Keep separate from existing grammars, services, and benchmark results.

## Purpose and order

Start with two mutually dependent unknown RHSs, then test three unknown RHSs in
a public feedback cycle. Retain modest coefficient counts and informative
independent experiments. Blinding extra equations alone does not prevent
independent regression when their state trajectories/derivatives are available.

Known equations are nonlinear. Measurements are endpoints, not a dense time
series or near-initial samples. All initial states are supplied; hidden initial
values are not additional fitting parameters. The construction removes the
specific linear-response moment inversion used in recovery31_cpu, not every
possible CPU inference method.

## Shared unknown-law grammar

For each unknown state j:

```
xj' = -d_j*xj + A_j*U_j(H_j(x))/(1 + c_j*x_r^2)
U_j in {sin, cos, tanh}
H_j in {
  a_j*x_p + b_j*x_q,
  a_j*x_p*x_q + b_j*x_q,
  a_j*x_p + sin(b_j*x_q)
}
```

The three roles p,q,r are distinct. There are four independent coefficients per
unknown RHS; sharing is not implied between RHSs. Amplitude magnitude is in
[0.4,1.0], a/b magnitudes in [0.3,1.2], all with independent signs; c is positive
in [0.3,1.2]. Damping is fixed and public. No extra terms or coefficients.

These counts are ordered structural assignments before mathematical equivalence,
not unique functions. The small operator/shape catalogue and state-role choices
are intended to permit grouped execution with toggles where compatible. No
lowering, occupancy, throughput, or coefficient-bank sufficiency has been tested.

## A: two mutually dependent hidden laws — first experiment

Six states. Unknown x4 and x5. Observe x0 and x1 at t=2 only, after supplying all
six initial values. Known equations:

```
x0' = -0.5*x0 + 0.8*tanh(x2 + 0.4*x4*x5) + 0.3*sin(x1)
x1' = -0.5*x1 + 0.7*tanh(x3 - 0.5*x2*x5) + 0.3*sin(x0)
x2' = -0.6*x2 + 0.8*tanh(x5 + 0.4*x0*x3)
x3' = -0.6*x3 + 0.7*tanh(x4 + 0.5*x1*x2)
```

Use d4=0.5 and d5=0.4. In H4, either p or q must be 5; in H5, either p or q
must be 4. All other roles can use any remaining state under distinctness.

Each law has 3 U choices x 3 H choices x 2 cross-state positions x 5 other
argument states x 4 denominator states = 360 assignments. The pair has
129,600 joint assignments and eight coefficients. For scale only, 8,192 sampled
coefficient vectors per pair would be 1,061,683,200 rollout configurations;
that is not a recommended or validated coverage budget.

## B: three hidden laws in a known cycle — second experiment

Six states. Unknown x3,x4,x5. Observe x0,x1,x2 at t=2 only. Known equations:

```
x0' = -0.5*x0 + 0.8*tanh(x3 + 0.4*x4*x5) + 0.3*sin(x1)
x1' = -0.5*x1 + 0.7*tanh(x4 - 0.5*x3*x5) + 0.3*sin(x0)
x2' = -0.6*x2 + 0.8*tanh(x5 + 0.4*x0*x4)
```

Use d3=0.5,d4=0.5,d5=0.4. Publish the mandatory cycle:
H3 uses p=4; H4 uses p=5; H5 uses p=3. Each q selects an observed state
from {0,1,2}; r selects any state different from p and q.

Each law has 3 x 3 x 3 x 4 = 108 assignments; the triple has 1,259,712
joint assignments and twelve coefficients. Publishing the mandatory cycle is
an explicit structural prior. This does not test discovery of arbitrary network
topology. Extra dependencies through r remain variable.

## Data and validity

Begin with 96 independent training ICs in [-1.5,1.5]^6 and 32 independent private
validation ICs. Each experiment supplies a single noninitial endpoint at t=2.
No derivatives, post-initial hidden states, or repeated stopping times for the
same IC. Start noiseless; noisy data is a separate robustness tier.

Use independently checked float64 reference integration. Reject nonfinite or
failed trajectories; verify explicit-step convergence before choosing Odezza's
screen/refinement step sizes. Bounded forcing does not by itself establish that
every draw is nonstiff or that a coarse RK4 step preserves candidate rankings.

Check each unknown law's influence on observed endpoints by ablation, and check
the endpoint Jacobian's numerical rank/sensitivity for all eight/twelve
coefficients. These are local informativeness diagnostics, not proofs of global
structural identifiability. If distinct models remain observationally
indistinguishable, label the case ambiguous and improve experimental coverage;
do not imply that either compute platform can recover absent information.

Use the same fresh data and public priors for CPU and hybrid workflows. Allow
CPU derivative/integral/latent-state proposals, multistart fitting, and symbolic
search. Do not select cases based on which solver wins. Calibrate cheap-screen
survivor recall before committing a large random coefficient campaign.

## Structural recovery grading

User direction: synthetic discovery should require structurally correct RHSs.
Historical `verified` flags based on held-out MSE remain prediction-verification
evidence unless separately audited; do not silently relabel old results.

Keep the private generating expressions and seed outside every search process.
Freeze a final candidate (or a predeclared timestamped candidate archive) before
the private grader reads it. Do not return ground-truth-guided corrections to
an ongoing solve.

Report separately:

1. Predictive success: observed trajectory error under independent integration.
2. Structural success: all unknown RHSs match the generating functions up to
   valid algebraic/trigonometric identities and parameter reparameterizations.
   State identities remain fixed; matching only the dependency graph or merely
   matching raw AST strings is insufficient.
3. Coefficient accuracy under a predeclared precision-aware tolerance, accounting
   for equivalent sign/parameter conventions.
4. Whole-system structural success versus how many individual RHSs are correct.

Use symbolic checks where supported. Independent private evaluations of the RHS
over a broad state domain help reject trajectory-only approximations but finite
point tests do not prove functional equivalence. Report an inconclusive symbolic
check as inconclusive, rather than silently converting small numeric error into
an exact-recovery claim. Zero/dead terms and numerical coefficient tolerances
need explicit grading rules.

For an offline timestamped archive, time to first structurally correct candidate
can be determined after the run. Otherwise report the frozen submission time;
do not invent an earlier discovery timestamp from the final formula.

Reference for separating exact-expression recovery from predictive accuracy:
[SRBench++](https://ir.cwi.nl/pub/34307/34307.pdf).
