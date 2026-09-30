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

# Dynamic-Leaf Settings and Feature Learning

## Current Secant-SR Settings

During the current search, Secant-SR evaluates a static population and then
temporarily opens leaves in selected ASTs as dynamic holes:

- Exploratory ASTs with at most eight leaves can have every leaf opened.
- More mature ASTs keep most of their structure fixed and open up to four
  leaves.
- Each hole may become either a specific input column or a specific constant.

A setting contains:

```text
column/constant mask
value for hole 0
value for hole 1
...
value for hole 7
```

For a column, the value is its column index. For a constant, it is its exact
`f32` encoding.

The current 4,096-setting table is generated once for each dataset and seed:

- Setting 0 is entirely columns.
- Setting 1 is entirely constants.
- Every remaining hole is independently 50% column and 50% constant.
- Columns are selected uniformly.
- Constants are selected uniformly from the 13-value static pool.
- Selection is with replacement, so duplicate columns are possible.
- The same table is reused by every AST and every generation.

With eight holes, the random settings follow a binomial distribution. Most
have approximately four columns and four constants. Very few are extremely
column-dense or constant-dense. The current table therefore contains an
implicit density prior even though one was not chosen explicitly.

For each AST, the kernel computes

\[
L_a(s) = \operatorname{SSE}(\text{AST } a \text{ under setting } s)
\]

for all 4,096 settings. The reducer chooses the setting with minimum finite
SSE. Secant-SR then converts the winning dynamic leaves into concrete
static-column and constant instructions, evaluates that fixed AST again, and
promotes it only if it improves upon its source. Later refinement can open
some of those fixed leaves again.

This gives two meanings to "fixed settings":

1. The setting bank itself is fixed for the entire trial.
2. A winning setting is fixed into a new static AST before entering the
   population.

The constant optimizer is separate. It selects up to 512 ASTs with constants
and runs 8,192 Philox perturbations for four iterations around each AST's
current constants.

## Learning From SSE

The AST-by-setting SSE surface is potentially a large feature-learning
dataset. A training record can contain:

```text
AST structure
operators above each hole
existing static columns
setting column/constant mask
selected column identities
constant values
source SSE
setting SSE
SSE improvement
promotion outcome
later archive survival
```

From this data, a search policy could estimate:

- Whether successful settings prefer columns or constants.
- The best column density for different AST maturities.
- Which columns repeatedly improve SSE.
- Which column pairs are useful together.
- Which columns are useful under multiplication, division, sine, and other
  contexts.
- Whether a setting produces a narrow lucky win or works across many ASTs and
  row subsets.

This is contextual feature learning rather than ordinary marginal
correlation. A column could receive high value specifically when inserted
under multiplication with another column.

All scores must be normalized by exposure. Useful first-order statistics
include:

```text
mean SSE gain per exposure
promotion probability per exposure
top-k frequency per exposure
archive survival after promotion
pairwise column gain
gain by parent operator and maturity
```

## Larger Setting Banks

A larger deterministic bank could contain, for example, 65,536 settings. It
could be uploaded once while the search evaluates a 4,096-setting window per
generation. The complete bank would require only a few megabytes.

Possible schedules include:

- **Disjoint chunks:** Each generation sees new settings.
- **Sliding window:** Some previous settings remain while new settings enter.
- **Round-robin:** The search eventually covers the complete bank.
- **Adaptive replacement:** Weak settings are retired and learned settings are
  appended.
- **Multiple launches:** Additional chunks are evaluated only for promising
  AST cohorts.

A structured bank would provide explicit coverage instead of relying only on
a binomial draw:

```text
settings 0-127:       all or nearly all columns
settings 128-255:     one constant
settings 256-383:     two constants
...
settings 1024-1535:   mostly constants
settings 1536-2047:   discrete scientific constants
settings 2048-3071:   random continuous constants
settings 3072-4095:   learned column and pair biases
```

For eight holes, the bank could allocate equal capacity to every possible
number of column leaves from zero through eight. This would test extreme
densities much more often than the current policy.

## Evolving Settings

Settings could also form their own population. Mutations could:

- Flip a column/constant bit.
- Replace a column.
- Perturb a constant.
- Swap hole assignments.
- Insert a column pair that has previously worked together.

Setting fitness should not simply be its best SSE, which would reward a
setting specialized to one AST. Better objectives include:

- Mean normalized SSE gain across an AST cohort.
- Promotion count.
- Diversity of ASTs improved.
- Performance across complexity and maturity classes.
- Stability across row subsamples.
- Novelty relative to existing settings.

Because hole meanings depend on AST context, a contextual sampler will
probably outperform one global evolutionary fitness.

## Recommended Policy

Learned settings should not replace broad exploration. A mixture can retain
coverage while exploiting accumulated evidence:

\[
P(s) =
\lambda_u P_{\text{uniform}} +
\lambda_d P_{\text{density}} +
\lambda_m P_{\text{learned}} +
\lambda_e P_{\text{evolved}}
\]

A reasonable initial allocation is:

- 25% uniform random exploration.
- 25% explicitly balanced density strata.
- 25% balanced column and column-pair coverage.
- 25% learned or evolved settings.

The first implementation should gather telemetry without changing search
selection. Once the signals are stable across development seeds, the search
can introduce a larger bank and rotate deterministic windows. Any learned
policy must use training data only and be frozen before an official test
campaign.
