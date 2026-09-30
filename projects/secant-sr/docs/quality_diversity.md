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

# Quality-Diversity Search

## Policy

The search can compute structural descriptors with a validated linear AST pass:

- a 128-bit mask of referenced static columns;
- unique and total static-column counts;
- fixed-constant count;
- unary, binary, and ternary operation counts;
- transcendental-operation count;
- a 64-bit set of operation types.

The persistent archive is a Cartesian map over complexity bucket, unique-column
bucket, and transcendental-count bucket. Counts saturate in the final bucket.
Each cell retains the configured number of highest-scoring distinct programs.
Variation can draw an occupied cell uniformly and use its best elite as a
parent. The staged evaluator disables archive-parent sampling when it expands
the active AST limit for static-column search; the archive continues preserving
elites while static search returns to tournament-selected live parents.

Structural descriptors are computed on demand for winners and diagnostics.
Multidimensional QD selection opts candidate creation into descriptor extraction
so it can retain the two scalar bucket indices; the complete descriptor is not
stored with the population. The default is one column bucket, one
transcendental bucket, and zero archive-parent probability, which reproduces
complexity-only behavior.

## Initial Controlled Comparison

The initial comparison used the same direct-CUBIN staged evaluator, broad grammar,
8,192 structures, 4,096 dynamic-leaf settings, five dynamic generations, 50
total generations, 1,024 training rows, 4,096 held-out rows, no CPU constant
optimization, and seeds 1 through 3. QD used five column-count buckets, four
transcendental-count buckets, and a 0.35 probability of drawing any variation
parent from the archive.

| Problem | Complexity-only solves | QD solves | Complexity-only mean R2 | QD mean R2 |
|---|---:|---:|---:|---:|
| Distance-2 | 1/3 | 2/3 | 0.996676 | 0.999052 |
| Feynman III.15.12 | 3/3 | 3/3 | 1.000000 | 1.000000 |
| Interaction-3 | 3/3 | 3/3 | 1.000000 | 1.000000 |
| Nguyen-5 | 0/3 | 0/3 | 0.971938 | 0.986644 |
| Oscillator-2 | 3/3 | 3/3 | 1.000000 | 1.000000 |
| Rational-2 | 3/3 | 3/3 | 1.000000 | 1.000000 |
| **Total** | **13/18** | **14/18** | **0.994769** | **0.997616** |

Mean generation construction time was 2.905 ms for complexity-only selection
and 2.872 ms for QD selection. QD occupied roughly 52 to 95 cells at generation
49, depending on the problem; the complexity-only archive occupied ten. This
result was encouraging but did not generalize to the larger benchmark.

At the end of the five-generation dynamic phase, mean held-out R2 was 0.945345
for complexity-only selection and 0.981582 for QD. The dynamic phase did
establish useful compact bases in this panel. The larger comparison below shows
that the tested archive-parent policy was not a reliable way to preserve and
refine those bases.

## Feynman-116 Comparison

> **Historical result:** these policy sweeps selected candidates by training
> fitness but used held-out R2 for early stopping. They remain useful relative
> policy measurements, but they are not protocol-correct SRBench results. See
> [srbench_v2_protocol.md](srbench_v2_protocol.md) for the corrected campaign.

The broader comparison used one seed on all 116 available Feynman problems,
8,192 structures, 100 generations, 10,000 training rows, full held-out
validation, 4,096 dynamic-leaf settings, five dynamic generations, and no CPU
constant optimization. Skeleton CUBINs came from the warm SQLite cache. These
are measured policy comparisons, not statistically conclusive multi-seed
results.

| Policy | Exact solves | Mean held-out R2 | Summed process time |
|---|---:|---:|---:|
| Broad grammar, complexity-only | 58/116 | 0.990799 | 320.96 s |
| Broad grammar, 5x4 QD, archive probability 0.35 | 57/116 | 0.958987 | 257.73 s |
| Broad grammar, 10x4 QD, archive probability 0.10 | 57/116 | 0.990324 | not retained |
| Scientific grammar, complexity-only | 59/116 | 0.984004 | 270.37 s |
| Trig grammar, complexity-only | 60/116 | 0.985410 | 253.67 s |
| Broad + scientific + trig result portfolio | 64/116 | n/a | three independent passes |

The multidimensional archive increased representation diversity without
improving exact recovery. It therefore remains an experimental selector rather
than the default. The stronger result came from rapidly rerunning the same
data-oriented engine with a focused operator family and retaining the best
result per problem. This uses Secant's low template and specialization costs as
a search dimension instead of committing every candidate to one broad grammar.
Trig was the strongest standalone profile, but it added no exact solve beyond
the existing broad-plus-scientific union. The current three-profile portfolio
has therefore reached a local plateau at 64/116.

`python/run_suite.py --resume` appends only missing `(problem, seed)` pairs to
an existing CSV after validating its schema. Named panels support short policy
rejection runs before a full pass. `--persistent-process` now keeps one CUDA
context and cache connection across all pending records. Dataset-dependent
plans, runners, streams, and allocations are still recreated; capacity-keyed
reuse of those objects remains a separate orchestration optimization.

## Kernel Pressure Exposed by the Search

The next high-value Secant shape is a mixed static/dynamic-leaf SSE kernel.
The current early phase turns every column and constant occurrence into a
dynamic leaf. A mixed shape would preserve columns already established by an
elite while sweeping only selected uncertain leaves. This is more useful than
making every leaf dynamic for every generation.

For wide data, the mixed shape should use a runtime gather list shared by a
cohort. Static AST inputs name gathered tile slots, while dynamic leaves choose
among gathered slots or constants. QD column masks provide the metadata needed
to pack candidates with overlapping column sets. This avoids loading every raw
dataset column and avoids dedicating a register to every possible input.

The next priorities are:

1. **Mixed gathered static/dynamic-leaf SSE.** Preserve established bindings,
   mutate a bounded number of leaf slots, and group ASTs by overlapping column
   masks.
2. **Dynamic affine statistics.** Rank `a*f(x)+b` for each topology and setting,
   then materialize the winning affine wrapper. This removes scale and offset
   from structural search.
3. **Device argmin for dynamic settings.** Return one SSE and setting index per
   AST instead of the complete `[AST][setting]` matrix. At 8,192 by 4,096 this
   reduces host result traffic from 128 MiB to about 64 KiB per generation.
4. **Bucketed error signatures.** Return 8 to 16 deterministic row-bucket errors
   for selected candidates, enabling semantic QD or lexicase selection. Static
   descriptors prevent structural collapse but cannot distinguish candidates
   that fail on different regions of the data.
5. **Per-AST local dynamic constants.** Use source-relative settings for a
   selected frontier cohort rather than applying one global constant table to
   unrelated expressions.

The existing static-column SSE remains the general second-stage evaluator.
Gram statistics belong after candidate shortlisting when the objective becomes
complementary feature selection or SINDY, not as the primary GP fitness path.
