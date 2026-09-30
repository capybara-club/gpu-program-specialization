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

# Validation And Numerics

FusedSINDy makes feature search faster. It does not make sparse equation
discovery automatically reliable. Derivative quality, feature scaling,
collinearity, validation design, and model-selection discipline still matter.

## Engine Correctness

Correctness tests should compare the fused path against independent references:

| Area | Reference check |
| --- | --- |
| Feature columns | Materialized column kernels or C/Python reference evaluation. |
| Gram matrix | Explicit `X.T @ X` from materialized columns. |
| Cross terms | Explicit `X.T @ y`. |
| Sums | Independent reductions over generated features and targets. |
| Constants | Raw `f32` constant packing/unpacking. |
| Safe ops | Edge cases around zeros, negatives, infinities, and NaNs. |
| Solver | Ridge/STLSQ against an explicit Gram/stat path. |
| MSE | Explicit validation prediction from materialized features. |
| Patch path | Patched evaluator output against a slower reference when available. |

Expect small `float32` differences from reduction order.

## Train, Validation, Test

Recommended split hierarchy:

```text
train rows:
    choose coefficients and sparse masks

validation rows:
    choose alpha/threshold/cohort candidates during search

test rows or trajectories:
    final report only
```

The current API handles validation by launching the same Gram kernel on
validation tensors, then scoring MSE from validation stats. If k-fold behavior
is needed, run the same `gram -> solve -> mse` path for explicit splits.

## Normalization

The Gram kernel writes raw statistics. The solver computes feature means,
feature scales, and target means from those stats, then solves in standardized
feature space. The returned `beta_standardized` values are in that standardized
space unless a higher-level caller maps them back.

For equation reporting, store:

```text
x_mean
x_scale
y_mean
beta_standardized
active mask
primitive feature names
AST and leaf settings
```

Then decide how to print coefficients in original units.

## STLSQ

STLSQ is sensitive to:

- ridge alpha;
- threshold value;
- feature scaling;
- collinearity;
- noise;
- validation split;
- redundant generated expressions.

Report at least:

```text
alpha
threshold
active feature count
train MSE if computed
validation MSE
solve_info
iteration count
active expressions
```

## Collinearity And Equivalent Terms

Candidate libraries often contain near-equivalent terms:

```text
u*u_x      versus 0.5*(u^2)_x
u_xx       versus filtered derivative variants
constant   versus nearly constant generated expressions
safe ops   versus identity-like behavior in a narrow range
```

Low validation MSE does not guarantee unique terms. Check stability across
seeds, row subsamples, and validation splits.

## Safe And Approximate Ops

Safe operators are useful during search but can create pathological expressions.
For any accepted equation that uses safe/approx operators, document:

- the exact safe behavior or epsilon;
- whether the operator is approximate;
- whether the expression is meaningful in the physical domain;
- whether a stricter operator gives the same model.

## Final Report Checklist

- Fused Gram/stat output matches an explicit reference on small cases.
- Active terms are stable across several seeds or splits.
- Validation MSE is not only good for one split.
- A final untouched test score is reported.
- Feature scaling and thresholding are documented.
- Safe/approx operators in the final equation are justified.
- Primitive feature definitions and derivative methods are documented.

