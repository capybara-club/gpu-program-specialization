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

# Dynamic Search Policy

This note records the initial 2026-08-04 experiments with larger dynamic-leaf
expressions and GPU dynamic-constant sweeps. These are search-policy experiments,
not official SRBench results.

## Independent Node And Leaf Limits

The dynamic phase defaults to a 30-node limit and eight dynamic leaves. These
are independent constraints. The normal generator decides whether each internal
node is unary or binary; it does not force a unary operation after another node.
This permits unary-heavy expressions beyond the former 15-node envelope without
creating an expression that exceeds the kernel's eight dynamic-leaf slots.

`--dynamic-max-nodes` controls the initial dynamic node limit. `--dynamic-leaves`
controls both the initial leaf limit and the generated kernel capacity. Staged
search raises both limits to the static evaluator's capacity after the dynamic
phase.

A forced-unary prototype was tested while developing this policy. It reduced
recovery on both the generated and hard Feynman panels and was removed. Those
measurements should not be interpreted as a comparison against the current
larger-budget generator.

## Dynamic Constant Search

`--constant-settings N` considers ASTs with between one and `--dynamic-leaves`
fixed constants. An eligible AST has its complete constant set projected to
indexed dynamic constants; an over-capacity AST is excluded rather than
partially optimizing its first constants. Each generation independently
selects a bounded quality-diverse cohort with `--constant-optimize-budget`
(default `4,096`) every `--constant-optimize-interval` generations (default
`1`). The legacy probability selector remains available only for reproducing
older experiments.
Each selected AST receives its own Philox perturbations around its current
constant vector. The CUBIN runner loads a specialized module once, alternates
evaluation and reduction for every configured iteration while that module is
resident, multiplies the perturbation scale by `--constant-optimizer-decay`
between iterations, and returns one constant vector per selected AST for
static-column SSE rescoring. `--constant-optimizer-scale` sets the initial
additive radius. A decay of `1` keeps that radius constant; values in `(0, 1)`
geometrically anneal it.

The regular default uses 1,024 settings and one iteration. In a matched
8,192-candidate, 4,096-row, ten-generation Nguyen-5 probe, optimizer-off took
0.940 s, regular refinement took 1.179 s, two iterations took 1.306 s, and four
iterations took 1.529 s after moving the iteration loop inside the resident
module lifecycle. These are policy-path timings, not kernel-only peaks; the
iteration sweep used 1,024 settings throughout.
On a separate regular-default run, the one-time final CPU audit optimized two
constants in 1.81 ms and raised training R2 from 0.992565 to 0.992627 and held-out
R2 from 0.992366 to 0.992414. This is the intended use of CPU L-BFGS: quantify
the accuracy still available after GPU refinement without entering the
per-generation hot path.

`--dynamic-batch-asts` bounds the reusable device SSE surface. Dynamic-leaf
and dynamic-constant evaluations process successive AST chunks through this
same capacity. Each chunk is reduced on device before the surface is reused.
The host receives one compact winner per dynamic-leaf AST and one updated
constant vector per selected optimizer AST; it never receives the full
`[AST][setting]` output.

`--constant-sweep-phase each` is the current regular-refinement default. `early`
restricts optimization to dynamic-leaf generations, `final` runs only after the
last generation evaluation, and `both` runs during the dynamic phase plus the
last generation. Every proposal must improve its source training SSE before it
can alter subsequent parent selection.

`--constant-setting-credit-weight` can reward expressions that perform well for
many settings. Robustness is the mean clipped training R2 across every tested
constant setting, with invalid settings contributing zero. It is added to the
normal fitness score using the configured weight; it does not replace the best
setting's fitness.

The tables below describe earlier policy experiments that used the former
shared-setting/final-sweep implementation; they are retained as historical
evidence and are not timings for the current per-expression Philox optimizer.
The corrected quick panel
used six generated equations, seeds 1 through 3, 4,096 candidates, 100
generations, 1,024 training rows, 4,096 validation rows, five dynamic
generations, and 4,096 constant settings.

| Dynamic policy | Exact solves | Mean validation R2 | Minimum validation R2 | Summed process time |
|---|---:|---:|---:|---:|
| 15 nodes, 8 leaves | 12/18 | 0.997411 | 0.980895 | 11.27 s |
| 30 nodes, 8 leaves | 14/18 | 0.997442 | 0.971975 | 10.58 s |
| 30 nodes, 8 leaves + constant sweep | 12/18 | 0.997190 | 0.979457 | 16.94 s |

The independent 30-node budget recovered two additional exact expressions with
no measured runtime penalty in this small panel. Applying constant search to the
whole population raised the worst fit but lost two exact recoveries by changing
the evolutionary trajectory. Large constant sweeps are therefore better applied
to a selected frontier than indiscriminately to the complete population.

## Numerical Validation

One experimental constant run produced a nested reciprocal around approximate
MUFU `tanh`. Small CPU/GPU differences near a zero denominator were amplified
enough to move a poor candidate's R2 from about `-0.852` on the CPU to `-0.832`
on the GPU. The materialization kernel completed correctly; this was numerical
instability in the random expression, not a failed launch or corrupt result.

The CPU interpreter is therefore the canonical campaign validation score.
Materialization failures still abort, while numerical divergence is recorded as
telemetry. Dedicated CPU/GPU backend tests continue to enforce strict agreement
on controlled expressions and data.

## Expanded Feynman Panel

A broader pass used 24 PMLB Feynman datasets, seeds 1 through 3, 4,096
candidates, 100 generations, 4,096 training rows, 8,192 held-out rows, a
30-node/eight-leaf dynamic phase, and 4,096 settings for final polishing.

| Policy | Exact runs | Datasets solved in any seed | Mean held-out R2 | Median held-out R2 | Minimum held-out R2 |
|---|---:|---:|---:|---:|---:|
| 15 nodes, no constant sweep | 11/72 | 6/24 | 0.750329 | 0.999408 | -13.733503 |
| 30 nodes, no constant sweep | 14/72 | 9/24 | 0.969810 | 0.999747 | 0.540277 |
| 30 nodes, final eight-constant sweep | 14/72 | 8/24 | 0.965253 | 0.999629 | 0.552298 |
| Per-run diagnostic upper envelope | 17/72 | 10/24 | 0.973841 | 0.999887 | 0.553209 |

The 30-node policy beat the 15-node policy in 28 matched records, lost in 30,
and tied in 14. Despite that mixed per-run result, it recovered more exact
expressions and avoided the 15-node policy's severe unstable tail. Clipping R2
to `[-1, 1]` still raises the mean from `0.927184` to `0.969810`. The larger
node budget is therefore the stronger current default, not a uniform dominance
claim for every dataset.

Final polishing beat the unswept run in 30 matched `(dataset, seed)` records,
lost in 24, and tied in 18. The upper envelope demonstrates complementary
coverage, but it is not a valid test-set selection policy: a real portfolio must
choose by training fitness, cross-validation, or another predeclared selector.

Repeated identical GPU campaigns are not bitwise reproducible. Atomic SSE
accumulation can change low-order bits, and tournament selection amplifies those
differences into different search trajectories. Single-run policy deltas should
therefore not be interpreted causally. The final sweep itself compares the
source and polished proposal inside one run and only promotes lower training
SSE, which is the topology-preserving way to combine the mechanisms.

An early-sweep probe on one seed reached three exact solves but produced one
numerically unstable result with held-out R2 `-6.95`. Early sweeping remains an
explicit exploration policy rather than the default.

## 128K Population Probe

A larger stress probe used 131,072 candidates, 100 generations, one seed, 4,096
training rows, and 8,192 held-out rows on three difficult Feynman datasets.

| Dataset | 30 nodes | 30 nodes + final eight-constant sweep |
|---|---:|---:|
| Feynman III.19.51 | 0.996718 | 0.993364 |
| Feynman III.8.54 | 1.000000 | 0.969815 |
| Feynman test 9 | 0.881496 | 0.904935 |

The unswept runs took 23.0 to 38.4 seconds per dataset. Final-polish runs took
39.7 to 41.9 seconds and promoted between 69,417 and 108,955 constant proposals.
These are independent stochastic GPU trajectories, so the table measures policy
runs rather than before/after mutation of one fixed population. Inside a final
sweep, a proposal still cannot replace its source unless its training SSE is
lower.

## 1M Population Probe

The same three-dataset probe was repeated with 1,048,576 candidates. All other
search parameters remained unchanged: one seed, 100 generations, 4,096
training rows, 8,192 held-out rows, five dynamic generations, a 30-node/eight-
leaf dynamic limit, and 4,096 settings when final constant search was enabled.

| Dataset | 30 nodes | Time | 30 nodes + final eight-constant sweep | Time | Constant proposals promoted |
|---|---:|---:|---:|---:|---:|
| Feynman III.19.51 | 0.996864 | 312.7 s | 0.991245 | 344.7 s | 591,595 |
| Feynman III.8.54 | 1.000000 | 158.1 s | 1.000000 | 146.9 s | 0 |
| Feynman test 9 | 0.920115 | 318.0 s | 0.955739 | 341.1 s | 852,142 |
| Mean | 0.972326 | 262.9 s | 0.982328 | 277.6 s | - |

III.8.54 stopped at generations 29 and 24 respectively after exact recovery,
so neither run paid for a final constant sweep. The two complete 100-generation
final-sweep runs added about 23 to 32 seconds. As with the 128K probe, these are
independent GPU trajectories rather than a paired before/after evaluation of
one fixed population.

One million candidates with 4,096 settings logically produces 4.29 billion
scores, or 16 GiB of f32 output. That historical probe allocated and copied the
complete surface. The current bounded reducer instead uses
`dynamic_batch_asts * settings * sizeof(float)` device bytes and O(population)
compact host metadata. With the default 65,536-AST batch, the same 4,096-setting
search uses a 1 GiB reusable device score surface. Final-only constant search
also destroys the no-longer-needed dynamic-leaf evaluator before creating its
constant evaluator.

## Bounded Reducer Probe

A 131,072-candidate Nguyen-5 probe used 4,096 rows, 2,048 leaf settings, 2,048
constant settings, 32,768 ASTs per device batch, 32,768 sampled constant
candidates, and a 2,048-entry constant frontier. The dynamic SSE surface was
256 MiB instead of 1 GiB for a population-wide allocation.

The post-leaf population contained 113,851 eligible constant-bearing ASTs.
The policy sampled 32,768, returned 2,048 device-reduced winners, and promoted
1,825 after static rescoring. With all kernel templates cached, the generation
took 1.570 seconds and the complete process took 1.933 seconds. The app's
aggregate accounting reported `8.760e11` row evaluations/s. GPU kernel runtime
was 0.912 seconds; module loading remained a substantial 1.160 seconds, with
those phases overlapping inside the runner pipeline.

## Current Direction

1. Use the larger dynamic node budget while enforcing leaf capacity separately.
2. Independently sample only ASTs whose complete constant set fits the dynamic
   kernel, optimize bounded chunks on device, then materialize and statically
   rescore every sampled proposal.
3. Treat robustness as a quality-diversity descriptor or tie-breaker before
   adding it directly to fitness.
4. Measure optimizer cadence, setting count, and iteration count independently;
   their work scales approximately with sampled ASTs times settings times
   iterations times rows.

The raw development measurements are under `scratch/policy_20260804/` and are
intentionally not part of the repository.
