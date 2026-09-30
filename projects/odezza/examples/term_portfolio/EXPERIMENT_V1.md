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

# Version-one experiment: systems 19 and 20

This is the frozen experiment report. The current default is described in
[README.md](README.md); use `prepare.py --legacy` to reproduce this older profile.

This is a separate retrospective experiment in **useful initial feedback**.
It does not change core, runtime, service, or the preceding depth-tree example.
The same operator-only grammar was applied to both systems. Explicitly labeled
clue-assisted requests then used motifs from their public `knowns` files.
No prior recovered RHS was loaded by preparation, scoring or fitting. After all
search and fitting ended, the earlier frozen answers were inspected solely to
audit coverage. No further proposals were submitted after that inspection.

## What was tested

[grammar-v1.json](grammar-v1.json) is the native grammar, and [prepare.py](prepare.py)
constructs it without expanding the candidate population in Python:

```text
S = x | y | z
L = a*x + b*y + c*z
Q = d*S*S
A = e*S | e*S*S | e*S + f*S
U(A) = sin(A) | cos(A)

RHS = L
    | L + Q
    | L + optional(Q) + g*U(A)
    | L + optional(Q) + g*U(A)*V(B)
    | L + optional(Q) + g*U(A) + h*V(B)
```

The last three forms become separate families for sine/cosine combinations.
Argument occurrences have independent coefficient and state roles. The
background includes all three states; the initial coefficient sample is not
restricted to stable signs. A quadratic monomial is optional by family, not
forced to have an arbitrarily small random coefficient in the simpler family.

There are 18 families, 26,890 resolved syntax occurrences and 3,105 packed native
variants. Each occurrence receives 16,384 joint Philox coefficient vectors:
**440,565,760 evaluations**. State roles use disjoint two-way x/y toggles plus
the fixed z alternative. All active coefficient roles use one joint `trial`
axis with separate streams, not a Cartesian product of coefficient grids.

Initial ranges are [-1,1] for linear coefficients, [-0.5,0.5] for the quadratic
coefficient, [-1.5,1.5] for oscillation amplitudes and [-4,4] for argument weights.
These are explicit numerical priors, not unit-independent universal defaults.
The grammar uses only the operator classes declared by the public files.
No exact oscillation frequencies are used in this operator-only mode.

All screens use complete T=40 trajectories with original training-array indices
0,4,8,12 and RK4 hmax=0.125. They integrate 1,280 RK4 steps per configuration,
without teacher forcing, short windows or estimated derivatives. Fitting uses
the first 16 full trajectories. The other four are withheld during this search.

## Initial reports and the next proposal

| System | General screen | Best screen MSE, 4 trajectories | Best bounded fit MSE, 16 trajectories |
|---|---:|---:|---:|
| 19 | 14.836s / 440.6M evaluations | 0.102706 | 0.0404595 |
| 20 | 14.696s / 440.6M evaluations | 0.0646611 | 0.00412561 |

The general grammar gave **partial structural information, not recovery**:

- System19's fitted leaders used an x*y quadratic background, linear drain
  terms, and an oscillatory x*z argument. Its best fitted polynomial-only
  family had MSE 0.188383; adding nonlinear terms improved that substantially.
- System20's fitted leaders used an x*z background and sin(f*x*y), with a
  frequency near 2.86. The fitted polynomial baseline plateaued near 0.0280.
  This was a useful lead, but the missing cosine factor still mattered.

We then kept **the exact same structural language** and centered only the four
background coefficient distributions on the best fitted general candidate,
with half-width 0.15. The resulting screen MSE improved to 0.0438129 on system19
and 0.0148650 on system20, each again taking about 14.7s. This is a measured
example of using a first report to improve the next request without selecting
the answer's structure. It narrows numerical exploration and should coexist
with an unguided allocation in a longer controller; it does not justify
permanently excluding other coefficient regimes.

The earlier broad depth-tree grammar scored system20 at 0.0283512 in 18.10s,
better raw MSE than the new initial general grammar. Its leading expressions
were polynomial lookalikes. Thus **raw MSE alone is not structural evidence**.
That earlier grammar also has different operators and coefficient ranges, so
this is a workflow comparison, not a controlled grammar-only ablation.

## Public clues as explicit grammar primitives

The supplied knowns files disclose these motifs:

```text
System19: H = cos(2.4*x)
System20: H = sin(2.8*x*y)*cos(1.7*z)
```

The same generic wrapper is used for either supplied motif:

```text
RHS = L + optional(Q) + g*H
    | L + optional(Q) + g*H*sin(A)
    | L + optional(Q) + g*H*cos(A)
    | L + optional(Q) + g*H + h*sin(A)
    | L + optional(Q) + g*H + h*cos(A)
```

There are 12 families including polynomial baselines, 860 resolved syntax
occurrences, 210 packed native variants and 65,536 vectors per occurrence:
**56,360,960 evaluations**. The initial requests took 1.410s for system19 and
1.418s for system20. These are **clue-assisted**, not operator-only discoveries.

On system20, a candidate from the simple `H + linear + quadratic` family fitted
to the following equation (rounded for display):

```text
z' = 0.36*sin(2.8*x*y)*cos(1.7*z) + 0.19*x*z - 0.37*z - 0.13*y
```

The fit has five adjustable coefficients including a redundant x term that
converged near zero. It took **4.81s**, with 16-trajectory training MSE
1.91e-16. The frozen candidate's four held-out trajectories scored **2.16e-16**
under tight FP64 DOP853, maximum absolute error 6.25e-8. Tightening the solver
tolerance preserved the result. The exact validation used fitted coefficients
and native-exported FP32 motif constants, not the rounded display above.

The 1.418s screen + 4.81s winning fit is **not the measured full controller
time**. This experiment fitted a broader family-balanced shortlist: its
clue-assisted fitting sweep took 60.60s. The winning candidate was not selected
by an oracle before that sweep. The historical holdout datasets had been used
in earlier work; this is a retrospective replay, not new blind generalization.

## System19's coverage failure

The tested grammar did **not recover system19**. A final guided clue screen
took 1.407s; nine focused fits took 22.15s and reached only 0.0397111 MSE.

Only after those jobs ended, the coverage audit read the previous answer:

```text
z' = 0.40*sin(x*z + 0.35*y)*cos(2.4*x) + 0.20*x*y - 0.36*z - 0.14*y
```

The missing production is a **product plus a state inside the same unary
argument**: `x*z + c*y`. The tested `A` could represent a product or a sum of
states, but not their composition. Neither more coefficient rows nor better
local coefficient fitting repairs that structural omission. The x*z and x*y
hints were useful, but did not imply that the complete answer was representable.

A better next initial profile should use a sparse quadratic argument:

```text
M = 1 | state | state*state
A = a*M | a*M + b*M
```

This generalizes to phase offsets, product-plus-state, two-product and affine
arguments. It does not encode system19's particular state combination. Preserve
separate allocations for one-term versus two-term arguments and single versus
multiplied oscillations; otherwise the larger Cartesian family can dominate
the work. Two complex arguments can receive a bounded sample initially, followed
by explicit expansion when fitted evidence supports them. This richer profile
is a **recommendation from the audit, not an implemented or validated recovery
claim in this folder**. The checked-in grammar intentionally records what was
actually tested.

## Reporting, fitting and limitations

`run.py` from the preceding example retains 12 structures per family here and
saves authoritative results plus readable feedback. Structural support, family
scores and parameter counts help choose the next request. Poor random-bank
scores must not be used as proof that a family cannot fit.

[fit.py](fit.py) is an experimental CPU adapter using NumPy, SciPy DOP853 and
`least_squares` with generated analytic forward sensitivities. It fits rollout
residuals, never synthetic observation derivatives. The generated Jacobian and
rollout sensitivity were checked against finite perturbations. Five local
contract tests pass. No new dependency was installed; the existing SciPy
environment was reused. PySR and Odezza's native LM were not used.

The current service returns resolved programs. This adapter rebinds coefficients
by unique exact slot bits, rejects collisions with fixed motif literals, and
retains the complete source record. That is a constrained experimental bridge,
not a robust public parameter-identity API. TODO16 still needs an explicit
parameterized-program export. Direct tests cover the collision rejection.

Initial fitting selected up to two records per family. Some were algebraic
duplicates such as `(q*x)*y` and `(q*y)*x`; a later client-side shortlist repair
flattens and sorts +/* with the same named parameter roles. This changes only
selection among retained rows, not native top-k semantics. The final system19
follow-up selected three distinct candidates from each of its three leading
families. The earlier sweeps were not rerun under that policy, so fitting times
are not a matched comparison of shortlist strategies.

Each CPU fit had a 12s wall cap and 70 evaluations. Many hit their cap; their best
valid evaluated point is retained with the timeout flag. Those results are not
converged rejections of a structure. Initial CPU divergence is reported; invalid
optimizer proposals receive a large penalty. A small experiment's bounded fit
policy is not a globally reliable optimizer.

The general request reduced geometrically underfilled work to **3.1%**, versus
17.3% in the earlier large-bank tree screen; the clue requests had 1.4%.
Those diagnostics do not measure SM occupancy. Initial general requests had
roughly two thirds invalid configurations, largely because of broad background
coefficients. Guided requests increased valid work substantially. Invalid exits,
different expressions and module groupings complicate throughput comparisons.

Across the eight newly submitted screens: **2,446,131,200 evaluations in 81.26s
summed native wall**. The five fitting sweeps made 123 bounded fit attempts and
took **269.86s summed wall**. This excludes code development, analysis and
validation; it is not an elapsed time-to-solution claim. The prior system19 tree
screen was reused and is excluded from these new-work totals.

[summary.json](summary.json) preserves counts, family results, hashes, fitting
limits and validation. Bulk inputs/reports stay in ignored `runs/`.

## Reproduce

```sh
python3 examples/term_portfolio/prepare.py \
  --legacy \
  --knowns knowns.txt --trajectories trajectories.csv \
  --output examples/term_portfolio/runs/initial.json
python3 examples/search_portfolio/run.py \
  examples/term_portfolio/runs/initial.json \
  --output examples/term_portfolio/runs/initial-result
```

For an explicitly supplied public motif, add `--public-motif 'cos(2.4*x)'` and
`--rows 65536`. The metadata labels that altered prior information. This small
adapter expects x/y/z, one unknown z RHS, common observation times for CPU fits,
and at least 13 trajectories for its default subset. It does not auto-detect
an appropriate integration step or support arbitrary operators.

```sh
scratch/recovery20_cpu/.venv/bin/python -m unittest discover \
  -s examples/term_portfolio -p 'test_*.py' -v
python3 examples/term_portfolio/summarize.py
```

The summary command audits already saved results and requires local raw files.
