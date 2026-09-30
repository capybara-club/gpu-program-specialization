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

# Kernel Shape Roadmap

## Static-column SSE

The implemented baseline specializes every input token to a fixed column.
Secant packs multiple ASTs in one kernel, evaluates row tiles, reduces squared
error, and atomically accumulates one result per AST and target. This is the
most direct fit for ordinary symbolic regression where a mutation can change
both topology and column references.

Its main weakness during early search is repeated compilation of structurally
identical expressions whose only difference is a leaf column or constant.

## Runtime-bound leaves

The implemented early-search path uses a topology-specialized, leaf-bound SSE
kernel:

- A specialized AST input token names a leaf slot rather than a physical
  column.
- A runtime setting maps each leaf slot to either one of up to 32 tiled columns
  or a runtime constant.
- A CTA loads the row tile once and evaluates many binding settings against the
  same AST topology.
- Output is `[topology][target][setting]` SSE.

This does not require a second expression language. The same postorder AST can
be compiled with a recipe that interprets input indices as leaf slots. It is
especially attractive for early search because column selection and constant
proposals can vary at runtime while topology changes still use Secant's fast
specialization.

The likely search unit becomes one topology plus a batch of bindings. The best
binding is promoted back into an ordinary fixed-column individual before
crossover. That avoids making every population operation aware of settings.

Binding batches must be budgeted uniformly or accounted for in selection. If
one topology receives many more column/constant trials than another, selecting
only its best result creates a multiple-testing advantage unrelated to model
quality. A first implementation should give every topology the same binding
count, retain a small number of distinct winners, and charge fixed constants
and leaf slots consistently in the complexity score.

The current default uses eight dynamic leaves, 4,096 settings, and 64-row
static tiles. Every evaluated winner is materialized into an ordinary concrete
AST before selection. This keeps the search representation independent of the
runtime setting table and avoids retaining hidden leaf state beside elites.

## 32-expression Gram cohort

A second high-value shape is a cohort kernel that materializes 32 expressions
for a row tile into shared memory and forms sufficient statistics:

- the symmetric 32 by 32 Gram matrix, with 528 unique entries;
- 32 feature-target products per target;
- target squared norm and optional feature means/norms.

Those statistics support correlation ranking, ridge regression, and iterative
STLSQ without retaining the full feature matrix in global memory. The same
cohort can be reused for multiple ridge values and threshold schedules in a
small post-Gram kernel.

The Gram shape is most useful after static SSE or runtime-bound leaves produce
a shortlist. Building a Gram matrix for every raw evolutionary candidate adds
quadratic work and atomic traffic that simple fitness ranking does not need.
It becomes compelling when the scientific objective is sparse library
construction, SINDy, residual modeling, or selecting complementary features
rather than merely finding the single lowest-SSE expression.

## Recommended sequence

1. Keep static-column SSE as the correctness and general-search baseline.
2. Add runtime-bound leaf settings for early topology search.
3. Promote winning bindings into fixed-column ASTs.
4. Pack selected expressions into 32-feature Gram cohorts.
5. Run ridge/STLSQ and use residual or coefficient information to seed the next
   campaign.

This sequence keeps each kernel shape narrow and lets measured campaign data,
not a speculative universal kernel, determine when to switch representations.
