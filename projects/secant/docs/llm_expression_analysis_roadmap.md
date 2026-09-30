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

# Secant Expression-Analysis Roadmap for LLM Campaigns

Secant becomes materially more useful when it compiles an expression directly
into the next numerical operation rather than stopping at materialization or
plain SSE. This document prioritizes kernel shapes and supporting statistics
that would let an informed LLM use Secant as a general native
expression-analysis backend.

## Recommended Priority

| Capability | Search impact | Broadens use cases | Priority |
|---|---:|---:|---:|
| Affine-calibrated SSE | Very high | Medium | 1 |
| GPU argmin/top-k over settings | High throughput | Low | 1 |
| Cohort Gram matrices | High | Very high | 2 |
| Per-AST constant settings | High | Medium | 2 |
| Error signatures | High | Medium | 3 |
| Multi-target and residual statistics | Medium | High | 3 |
| Derivative and constraint statistics | Specialized high | Very high | Later |
| More structural AST metadata | Low-medium | Low | Supporting |

## Affine-Calibrated SSE

This should probably be the next scoring shape. For each expression `f(x)`,
fit:

```text
prediction = a * f(x) + b
```

The kernel only needs to accumulate sufficient statistics:

```text
count
sum(f)
sum(f*f)
sum(f*y)
sum(y)
sum(y*y)
nonfinite_count
```

The host or a small fixed reduction kernel can derive `a`, `b`, and SSE.
Division to MSE should happen outside the expression kernel because SSE and MSE
rank expressions identically when all candidates use the same row count.

This materially improves search because an AST no longer needs to discover an
arbitrary output scale and offset. For example, searching for:

```text
sin(x^2) * cos(x) - 1
```

reduces to finding the nonlinear feature:

```text
sin(x^2) * cos(x)
```

and fitting `a = 1`, `b = -1`. The fitted coefficients must remain attached to
fitness and be applied unchanged to validation data. An LLM would likely make
affine-calibrated SSE its default Secant evaluation mode.

## GPU Argmin over Dynamic Settings

Dynamic-leaf evaluation currently returns the complete `[AST][setting]` SSE
matrix. At 4,096 settings, that produces:

| Population | Current SSE output | Best SSE plus setting index |
|---:|---:|---:|
| 8,192 | 128 MiB | 64 KiB |
| 32,768 | 512 MiB | 256 KiB |

A fixed reduction kernel should return one record per AST:

```c
typedef struct SecantBestSetting {
    float sse;
    uint32_t setting_idx;
} SecantBestSetting;
```

This reduces host output by 2,048x. It does not directly improve expression
quality, but it makes larger topology populations and more dynamic campaigns
practical. The first implementation can retain the full device SSE workspace;
removing that workspace would require a more invasive persistent or multi-stage
reduction design.

## Cohort Gram Matrices

For a cohort of perhaps 16 or 32 expressions, compile them into one kernel and
compute:

```text
G = Phi^T W Phi
c = Phi^T W y
target_norm = y^T W y
```

A 32 by 32 `f32` Gram matrix occupies only 4 KiB. These statistics support:

- ordinary least squares;
- ridge regression over many ridge values;
- sequentially thresholded least squares;
- forward feature selection;
- condition-number and redundancy analysis;
- multi-feature residual models;
- SINDy cohort evaluation.

This expands Secant from finding one expression to discovering and assembling
a compact equation library. Affine SSE is effectively the one-feature special
case, but it deserves a dedicated, cheaper kernel.

## Per-AST Constant Settings

The current dynamic-constant contract shares settings across ASTs. A local
refinement shape should instead provide independent proposals conceptually
equivalent to:

```text
constants[ast][setting][constant]
```

A dense host tensor should be avoided. The kernel can generate settings from:

```text
base_constants[ast]
mutation_seed[ast]
```

or process a small selected cohort at a time. This would let a campaign request
thousands of local constant proposals for each shortlisted expression and
replace many serial SciPy refinement calls while preserving independent
parameters.

## Error Signatures

Instead of returning only total SSE, an optional evaluator should return SSE
over 8-16 deterministic row buckets:

```text
error[ast][bucket]
```

This enables:

- lexicase-like selection;
- identification of expressions that fail in different regions;
- residual diversity;
- boundary- or singularity-focused search;
- interpolation versus extrapolation comparisons;
- behavioral duplicate detection.

Behavioral statistics are more useful to an LLM campaign than simply adding
more structural AST counters because they explain where a candidate succeeds
or fails.

## Multi-Target and Residual Statistics

One expression evaluation should be reusable against several right-hand sides:

```text
original target
current residual
validation regimes
previously selected residuals
derivative targets
```

This lets a campaign determine whether a feature explains the original target,
a residual, or a particular operating regime without reevaluating the AST for
each question. It is especially valuable for iterative additive modeling and
SINDy.

## Other Valuable Fused Shapes

Later additions could include:

- weighted SSE, Huber loss, and missing-value masks;
- prediction mean, variance, min/max, and finite count;
- correlation with targets and residuals;
- derivatives of an AST with respect to selected inputs;
- monotonicity, positivity, and bound-violation counts;
- PDE weak-form integrals;
- ASTs embedded directly in an ODE rollout kernel;
- classification losses for symbolic classifiers.

Secant's native compiler makes specialized fused kernels practical because the
numerical loop contains native expression instructions rather than a runtime
AST interpreter.

## Expected Effect on LLM Usage

The likely progression is:

- Today, an LLM uses Secant for candidate search and then leaves for
  PyTorch/SciPy analysis.
- With affine SSE, Secant performs calibrated candidate ranking directly,
  without materializing candidates for an external scale-and-intercept fit.
- With Gram matrices, it uses Secant for feature selection, sparse
  combinations, and SINDy.
- With per-AST settings, it uses Secant for constant refinement.
- With error signatures, it uses Secant for campaign steering and failure
  analysis.
- With multi-target support, it repeatedly searches residuals and related
  equations without separate expression evaluation passes.

These capabilities increase the amount of numerical work an LLM delegates to
Secant without necessarily increasing campaign wall time. They replace data
materialization, host transfers, and serial analysis with fused native kernels.

## Proposed Implementation Order

```text
1. GPU argmin for dynamic settings
2. Affine-calibrated SSE
3. 16/32-expression Gram kernel
4. Per-AST constant refinement
5. Bucketed error signatures
6. Multi-target and residual campaigns
```

This sequence would turn Secant from a fast symbolic evaluator into a general
native expression-analysis backend while preserving the existing postorder AST
and specialization machinery.
