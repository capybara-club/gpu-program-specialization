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

# Proposed next blind ODE challenge

Status: a generator/search specification, not an implemented JSON schema or a
measured Odezza speed advantage. This proposal follows challenge29's successful
CPU sparse-library recovery. Freeze this specification before generating the
next comparison datasets; keep generation seeds and solutions private.

## Objective

Test recovery across different outer structures as well as fitting several
coefficients inside nonlinear compositions. The user's follow-up explicitly
asks for more structural diversity: varying only the arguments of two fixed
trigonometric terms would leave a strong and unnecessary outer-template prior.
Preserve informative, noiseless observations in the first experiment.
Increasing missingness, noise, stiffness and dimension simultaneously would
make it hard to tell what caused a failure.

The previous four-template challenge admitted a small fixed feature catalogue.
Sparse discovery can exploit such a representation; see
[Brunton, Proctor and Kutz](https://arxiv.org/abs/1509.03580). The hypothesis here
is that a larger compositional space will make broad GPU proposal scoring more
valuable. It does not establish that CPU optimization, symbolic search or a
different continuous relaxation cannot solve it quickly.

## Public system definition

Use six states, with these two known equations:

```text
x0' = -0.20*x0 - 0.80*x1 + 0.35*x2
x1' =  0.80*x0 - 0.20*x1 + 0.35*x3
```

For each i in {2,3,4,5}, independently generate:

```text
xi' = -0.35*xi + sum(A[j]*term[j](x), j=1..m)
m in {1, 2, 3}
```

Terms use the following bounded, nonrecursive grammar. Distinct numbered
children are independently sampled; repeated references to the same numbered
child reuse exactly that expression and its coefficients. In particular, a
squared denominator must not sample a new argument or new coefficients on each
appearance. Coefficients are otherwise independent between children, terms and
equations. This is specification notation, not accepted Odezza request syntax.

```text
branch   := x[u]
          | x[u] + c*x[v]
          | x[u]*x[v]

argument := a*branch1
          | a*branch1 + b*branch2
          | a*branch1*branch2
          | a*branch1 + sin(b*branch2)

unit     := sin(argument0)
          | cos(argument0)
          | tanh(argument0)
          | argument0/(1 + argument0*argument0)

term     := unit1
          | unit1*unit2
          | unit1/(1 + argument2*argument2)
```

- State indices range over 0..5. Within two-state branches, u != v; states may
  repeat between branches and terms.
- Each whole unknown RHS must depend algebraically on at least three distinct
  states, not counting its fixed damping term. Individual terms may depend on
  fewer states. Reject duplicate terms within an RHS, including copies
  differing only by an overall sign, and exact cancellations.
- Across the whole system, require at least one term using a product of units,
  attenuated unit, or nested-sine argument. Do not force every RHS to have the
  same topology or to contain every operator.
- Sample m and production alternatives uniformly before applying these
  constraints. Sample state indices uniformly among the allowed choices.
  Rejection changes the accepted distribution: record rule frequencies and
  rejection reasons; do not describe it as uniform over distinct ASTs.
- Every A[j] has an independent equiprobable sign and magnitude uniform in
  [0.25, 0.65]. Every present internal coefficient a, b or c has an independent
  equiprobable sign and magnitude uniform in [0.30, 1.20].
- Enforce at most 12 independent coefficients per unknown RHS, including its
  amplitudes, and at most 96 postorder nodes in its variable portion (all terms,
  amplitudes and connecting sums). Count duplicated denominator expressions
  each time they occur in postorder, but count their shared coefficient slots
  only once. There are at most 48 free coefficients in the complete system.
  Record the actual counts. These are proposed generation limits; scoring
  shape/slot compatibility must still be checked before execution.
- The first coefficient of the affine branch is fixed at one intentionally:
  two freely scaled multiplied branches would introduce an avoidable continuous
  rescaling ambiguity. Other symbolic equivalences can still occur.
- Require a directed influence path from each hidden state to an observed
  state. Report the dependency graph privately. This is a necessary information
  check, not proof of structural identifiability.

An allowed illustrative term is:

```text
0.52*sin(0.8*x0*x4 + sin(1.1*(x2 - 0.6*x5)))
```

An allowed illustrative second term is:

```text
-0.41*cos(0.9*(x1 + 0.7*x3)*(x4 - 0.5*x0))
```

Other allowed outer structures include a product of two tanh/sine units, or a
sine unit attenuated by `1 + argument^2`. These illustrate syntax only; do not
insert these expressions into the generated truth or privilege their state
assignments in search. This admits additive, multiplicative, saturating and
rational responses, as well as mixtures within a single RHS. It is still a
declared finite grammar, not unrestricted symbolic dynamics.

All units and terms are bounded in magnitude by one, and denominators are at
least one. Together with positive damping, this prevents finite-time blowup of
the unknown states. The known subsystem is a stable linear system driven by
them. This boundedness does not guarantee easy identification or nonstiff
integration. The exclusions of singular divisions and unbounded exponentials
are deliberate for this first structural-search comparison.

## Data and numerical quality

For the first comparison, retain the previous observation design:

- Generate 32 independent training ICs uniformly from [-1.5,1.5]^6 and 16
  additional independent test ICs from the same box.
- Supply all six initial values for every trajectory. Observe only x0..x3
  after time zero; x4 and x5 observations remain hidden.
- Use times [0, 0.7, 1.4, 2.2, 3.1, 4.1, 5.2, 6.4, 8.0]. No added noise or
  derivative observations. Do not forbid solvers from estimating derivatives.
- Publish all 32 training observation trajectories. A solver can reserve a
  subset for its own validation. Publish test ICs/times, but keep their
  observations for independent evaluation after the solver freezes its answer.
- Generate truth using CPU float64 DOP853, initially rtol=1e-11, atol=1e-13;
  verify agreement with rtol=1e-12, atol=1e-14 to per-state normalized RMSE
  below 1e-8 on the training observations. Normalize by that state's standard
  deviation across all noninitial training observations, floored at 0.1.
- Predeclare a numerical gate: CPU float64 RK4 with maximum substep 0.025 must
  agree with the reference to the same normalized RMSE below 1e-5. Also check
  maximum substep 0.0125; require smaller error unless both are below 1e-10.
  Split steps to land exactly on observation times. Reject failures of this
  gate, preserving the rejected draw and reason privately. These gates isolate
  structural search from difficult numerical integration in this first trial.
- Preserve all generation attempts and acceptance reasons. Never reject a
  system because one competitor solved it quickly or Odezza failed it.

The generator may additionally report local trajectory-sensitivity singular
values and state/term variation as private diagnostics. These are not proofs
of a globally unique formula. Do not impose an unreported conditioning filter.

The public output is one JSON file containing this complete specification,
known RHSs, ICs, times, observed-state mask and training values. The generator
prints the solution privately and never embeds its seed, unknown ASTs, hidden
observations, derivative values or test observations in that public file.

## Evaluation

Start with ten fresh accepted systems, with independent private random seeds.
Report coverage of term counts, unit types, term types, state dependencies,
node counts and coefficient counts. For a later balanced structural sweep,
declare grammar strata in advance; do not cherry-pick solved or failed cases.
Run CPU-only and Odezza-plus-CPU on every identical public file, using frozen
search policies and the same wall-time budget. Record any policy changes and
rerun the comparison separately rather than mixing tuned and untuned outcomes.

Report time to a frozen candidate, independent test errors and solve fraction,
including all failures/timeouts. Give per-state normalized RMSE and absolute
MSE for observed test states; the generator can also score hidden test states
privately. A useful initial target is observed test NRMSE <= 1e-3 for prediction
and <= 1e-5 for high-accuracy prediction. Declare these targets before the run.
Report formula/functional equivalence separately: good predictions alone do
not prove the true structure was recovered, and textual AST equality is too
strict when algebraic or trigonometric equivalents exist.

Separate setup/LLM time, proposals, GPU scoring, coefficient fitting, validation
and canceled work. Match CPU and hybrid proposal/refinement policies where
possible so hardware benefit is not confused with a different algorithm.

## Odezza integration boundary

The mathematical operations are +, *, /, sin, cos and tanh. Candidate structures
can reuse state choices and coefficient banks, with coefficients prepared once
per rollout configuration. CPU coefficient refinement remains appropriate; the
intended hybrid is proposal -> bulk score -> fit -> validate. Preserve diversity
among term counts and outer structures in the retained candidate pool rather
than letting one prolific family consume all proposal/refinement slots.

The existing `examples/hidden_rhs_trial/` controller accepts the older fixed
four-template profile and will reject this one. It needs a separate profile,
grammar lowering, proposal/decoding logic and end-to-end numerical checks before
this can be a prepared benchmark. Do not claim that this document is already a
working request or that every resulting AST fits the chosen scoring shape.
Check AST/slot capacity and batch geometry before running the blind comparison.
Keep those changes outside the hardened kernel core unless an independently
verified core limitation requires a change.

If this remains easy for both methods, expand branch recursion by one bounded
level as the next single change. If neither succeeds, inspect identifiability,
proposal quality and fitting first. This is a synthetic structural-search
benchmark, not by itself evidence of demand or speed on scientific workloads.
