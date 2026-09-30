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

# Search Quality and Kernel Priorities

## Measured Search Results

The following campaigns use 1,024 training rows, 4,096 validation rows, 4,096
dynamic-leaf settings, and no continuous constant optimization. A solve means
held-out `R2 > 0.999999`.

| Search policy | Population | Relevant solve rate |
|---|---:|---:|
| Static SSE, broad grammar | 8,192 | 12/18 core campaigns |
| Staged dynamic-leaf/static SSE, broad grammar | 8,192 | 14/18 core campaigns |
| Staged, trig grammar | 8,192 | 2/3 Nguyen-5 and 2/3 Distance-2 |
| Staged, trig grammar | 32,768 | 3/3 Nguyen-5 |

The staged broad grammar improved Rational-2 from 2/3 to 3/3 and Distance-2
from 1/3 to 2/3. The 32,768-member trig campaign recovered Nguyen-5 in every
seed. One result was the literal target:

```text
(sin(x0 * x0) * cos(x0)) + -1
```

These runs distinguish two failure modes:

- Dynamic leaf binding materially improves compact rational and geometric
  topologies.
- Nguyen-5 primarily needs more topology trials and a less distracting
  operator vocabulary. Increasing settings beyond 4,096 was not necessary;
  increasing population volume with the same settings solved the problem.

Operator profiles are explicit campaign inputs. `broad` preserves the full
grammar, `trig` keeps arithmetic plus `neg/abs/sqrt/rcp/sin/cos`, and
`algebraic` removes sine and cosine as well. Profiles must not be selected from
knowledge of a hidden target expression. Suitable sources include domain
constraints, dimensional analysis, or a fixed portfolio of grammar cohorts.

## Highest-Priority Kernel Change

### GPU argmin for dynamic settings

The dynamic-leaf runner currently returns one SSE for every
`[AST][setting]`. The staged controller copies that matrix to the host and
scans it to choose one setting per AST. At 4,096 settings this transfers:

| Population | Current SSE output | Best SSE + setting index |
|---:|---:|---:|
| 8,192 | 128 MiB | 64 KiB |
| 32,768 | 512 MiB | 256 KiB |

Add a fixed GPU reduction after dynamic-leaf SSE that emits one best finite SSE
and one setting index per AST. This preserves the current evaluator semantics
and the existing tile-static kernel while reducing host output by 2,048x. The
device SSE workspace can remain initially; removing it would require a more
invasive persistent or multi-stage reduction.

The search API should then materialize proposals from setting indices rather
than requiring the complete SSE matrix. This is the most direct way to spend
the saved bandwidth and host work on larger populations.

## Search-Quality Kernel Candidates

### Source-relative dynamic leaf settings

The current setting table is shared by every AST. It is effective for broad
initial binding discovery but cannot express local mutations around every
candidate's own columns and constants. A second dynamic-leaf mode should take
each AST's concrete base leaf bindings and compact mutation descriptors. Each
setting would preserve the source or alter one or two leaves.

Do not allocate a dense `[AST][setting][leaf]` host table. Generate mutations
from per-AST base bindings and deterministic seeds in the kernel, or process a
small selected cohort. Keep the existing global sweep for initial discovery.

### Affine-calibrated SSE

For each expression value `f(x)`, accumulate sufficient statistics for the
best training fit `a*f(x) + b`:

```text
sum(f), sum(f*f), sum(f*y), sum(y), sum(y*y), row_count
```

The host or a fixed reduction kernel can derive `a`, `b`, and the calibrated
SSE. This removes scale and offset from topology search without materializing
rows or introducing a general optimizer. Nguyen-5, for example, reduces from
searching `sin(x^2)*cos(x)-1` to searching the nonlinear feature
`sin(x^2)*cos(x)` plus two fitted coefficients.

Coefficients must be retained with fitness and applied unchanged to validation
data. This shape also becomes a useful bridge toward one-feature regression,
SINDy candidate screening, and later Gram builders.

### Error signatures

Scalar SSE hides where an expression fails. An optional evaluator can return
8-16 deterministic row-bucket SSE values per AST. That enables lexicase-like
selection, residual diversity, and adaptive row sampling without materializing
every prediction. It is more likely to improve structural recovery than tighter
SASS scheduling because it changes the information available to selection.

## Kernel and Campaign Efficiency

- Cache architecture- and recipe-specific template CUBINs. Fresh staged
  processes spend about 18-21 seconds preparing three templates in the current
  64-kernel configuration; this is outside the specialization hot path.
- At population 32,768, one dynamic generation spent roughly 0.30-0.33 seconds
  in module loading and 0.25-0.28 seconds in device runtime. Sweep 64 versus
  128 kernels per module after template caching. More ASTs per module may be
  more valuable than micro-optimizing instruction stalls.
- Keep 4,096 settings as the current default. The measured hard-case gain came
  from topology population and grammar selection with settings held constant.
- Run grammar cohorts through one prepared evaluator controller rather than
  separate fresh processes. Cohorts can exchange complexity-bucket elites
  after static rescoring.

## Useful Dataset Metadata

The backend only needs column-major values, but search policy benefits from:

- input and target scales, robust ranges, and missing-value masks;
- train, interpolation-validation, and extrapolation-validation row groups;
- sample weights or estimated noise variance;
- physical units and dimensional compatibility constraints;
- monotonicity, symmetry, positivity, conservation, or known boundary data;
- deterministic row buckets for error signatures;
- derivative columns and coordinate metadata for PDE/SINDy campaigns;
- operator-family permissions supplied before search, not inferred from the
  hidden target expression.

For large datasets, early generations should use a deterministic stratified
row subset and periodically rescore frontier candidates over all rows. More
rows are useful when they cover new regimes, singularities, or boundaries;
duplicating the same distribution is less valuable than increasing topology
or grammar diversity.
