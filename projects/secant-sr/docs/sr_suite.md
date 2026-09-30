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

# Symbolic Regression Suite

> **Historical result:** these small-suite measurements predate training-only
> early stopping. Candidate selection used training fitness, but held-out R2
> could stop a campaign. The results remain useful integration history rather
> than held-out benchmark claims.

## Experimental contract

The suite checks whether `secant-sr` functions as a search system rather than
only as a fast expression evaluator. Every problem uses the same operator
vocabulary, fixed constant set, population mechanics, mutation probabilities,
and stopping rule. Only the number of available input columns and the sampled
input domain vary with the problem definition.

The search vocabulary is:

- unary: negation, absolute value, square root, reciprocal, sine, cosine, and
  hyperbolic tangent, natural exponential, and natural logarithm;
- binary: addition, subtraction, multiplication, division, minimum, and
  maximum;
- constants: `-3`, `-2`, `-1.5`, `-1`, `-0.5`, `-0.25`, `0.25`, `0.5`, `1`,
  `1.5`, `2`, `3`, and binary32 pi.

The built-in suite independently generates training and holdout coordinates
from deterministic seeds. Official SRBench campaigns instead consume explicit
train/test data prepared from PMLB; see
[srbench_feynman.md](srbench_feynman.md). Search code receives only
column-major inputs, targets, and scalar fitness results. It does not receive
the target expression or a problem-specific operator distribution.

Nguyen-1 and Nguyen-5 are recognizable symbolic-regression problems. The
remaining four are synthetic analytic stress problems; they must not be
presented as standard SRBench tasks.

## Measured results

This table reports a three-seed direct-CUBIN run with 8,192 individuals, 100
generations, 1,024 training rows, 4,096 holdout rows, and the same policy for
every problem. A numerical solve requires holdout `R2 > 0.999999`.

| Problem | Inputs | Numerical solves | Median holdout R2 | Holdout R2 range |
|---|---:|---:|---:|---:|
| Nguyen-1 | 1 | 3/3 | 1.000000000 | 1.000000000-1.000000000 |
| Nguyen-5 | 1 | 0/3 | 0.990364162 | 0.989818960-0.992562855 |
| Rational-2 | 2 | 0/3 | 0.998880598 | 0.998731605-0.999892353 |
| Distance-2 | 2 | 3/3 | 1.000000000 | 1.000000000-1.000000000 |
| Interaction-3 | 3 | 3/3 | 1.000000000 | 1.000000000-1.000000000 |
| Oscillator-2 | 2 | 3/3 | 1.000000000 | 1.000000000-1.000000000 |

The result is 12 numerical solves in 18 campaigns, with four of six problems
solved in every seed. It establishes that the current code performs population
generation, mutation/crossover, evaluation, selection, and holdout validation
as an end-to-end SR prototype.

It does not establish mature symbolic recovery. Some numerically exact
solutions contain redundant identities, while Nguyen-5 and Rational-2 converge
to high-fit trigonometric or hyperbolic surrogates instead of the generating
structure. Current missing search machinery includes algebraic simplification,
canonicalization and duplicate rejection, protected operator routines,
GPU-batched constant optimization, and stronger diversity or Pareto policy.
The CPU reference optimizer added after this campaign validates periodic local
constant refinement, but its results are not represented in this table. These
search limitations are now more important than evaluator throughput on the
small suite.

## Reproduction

```bash
CUDA_MODULE_LOADING=EAGER python3 python/run_suite.py \
  --backend cubin \
  --population 8192 \
  --generations 100 \
  --rows 1024 \
  --validation-rows 4096 \
  --output scratch/suite/results.csv
```

The CSV records the best train and holdout scores, complexity, depth,
generation, elapsed process time, and formatted expression for each seed.

## PySR comparison

PySR 1.5.10 was run on the same deterministic coordinates, input domains,
training rows, holdout rows, and broad operator vocabulary. Every problem used
the same PySR configuration: 24 Julia threads, 24 populations of 64 members, a
40-node maximum, a 16-level depth maximum, and a requested two-second search
timeout. PySR retained its normal ephemeral constants and constant optimizer;
forcing Secant's discrete constant pool onto PySR would not represent normal
PySR use.

| Problem | secant-sr solves | secant-sr median R2 | PySR solves | PySR median R2 |
|---|---:|---:|---:|---:|
| Nguyen-1 | 3/3 | 1.000000000 | 3/3 | 1.000000000 |
| Nguyen-5 | 0/3 | 0.990364162 | 0/3 | 0.995442601 |
| Rational-2 | 0/3 | 0.998880598 | 2/3 | 0.999999906 |
| Distance-2 | 3/3 | 1.000000000 | 2/3 | 1.000000000 |
| Interaction-3 | 3/3 | 1.000000000 | 3/3 | 1.000000000 |
| Oscillator-2 | 3/3 | 1.000000000 | 3/3 | 1.000000000 |
| **Total** | **12/18** | | **13/18** | |

PySR's median fit call took 2.240 seconds after warm-up. The first fit took
9.822 seconds because it included Julia/operator initialization; all 18 fit
calls totaled 48.118 seconds. The secant-sr suite launches one process per
campaign and totaled 17.704 seconds, with a 0.616-second median. These are
observed end-to-end campaign times, not an equal-expression-evaluation budget:
the search algorithms perform different amounts and kinds of work.

Extending only Nguyen-5, Rational-2, and Distance-2 to ten requested seconds
gave numerical solve rates of 1/3, 3/3, and 0/3 respectively. One Nguyen-5 run
and one Rational-2 run recovered the generating expression exactly; other
numerical solves were more complicated surrogates. PySR's multithreaded mode
is nondeterministic despite setting `random_state`, so the ten-second runs are
independent trials rather than monotonic continuations of the two-second runs.

The comparison therefore says that the current secant-sr prototype is already
competitive on this small numerical-recovery suite, while PySR is stronger on
continuous constant fitting and some rational approximations. PySR also shows
the same general failure mode as secant-sr under the broad grammar: `Min`,
`Max`, trigonometric functions, and optimized constants can construct accurate
but structurally unhelpful surrogates.

The adapter is reproducible with:

```bash
JULIA_NUM_THREADS=24 python3 python/run_pysr_suite.py \
  --seconds 2 \
  --rows 1024 \
  --validation-rows 4096 \
  --output scratch/suite/pysr_results.csv
```
