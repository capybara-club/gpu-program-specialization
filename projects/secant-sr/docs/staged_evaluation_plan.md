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

# Staged Secant Evaluation Plan

## Implementation Status

Phase A is implemented by the `cubin-staged` search backend. The controller
creates dynamic-leaf SSE, static SSE, and one-AST materialize runners in one
CUDA context before the generation loop. Early generations use static source
scores, dynamic binding selection, concrete proposal materialization, static
proposal rescoring, and strict compare-and-promote replacement. The transition
raises the active node limit without reallocating either population arena and
then uses static SSE for subsequent generations.

Held-out predictions for the current winner are produced by the materialize
runner and checked against the CPU interpreter every generation. The original
`cubin-dynamic-leaf` backend remains available as the unconditional projection
experiment; it is not the staged promotion policy.

Per-expression GPU constant optimization is integrated into the staged backend.
Eligible fixed-constant ASTs are independently sampled, assigned AST-local
Philox perturbations, reduced to updated centers, materialized as concrete ASTs,
and statically rescored before promotion. CPU L-BFGS is reserved for one final
best-expression audit after the generation loop.

## Objective

Use each production Secant kernel shape for the search work it performs well,
while keeping one canonical concrete AST representation throughout the GP
system:

1. Explore column and constant bindings for small topologies with dynamic-leaf
   SSE.
2. Promote the selected binding to an ordinary concrete AST.
3. Score growing and unrestricted ASTs with static-column SSE.
4. Use constant-optimizer SSE for regular expression-local refinement.
5. Materialize only shortlisted expressions that need row-wise values.
6. Keep the CPU interpreter as the independent semantic oracle.

The first implementation should use global search stages. It should not begin
with candidate-by-candidate routing inside one generation. Whole-population
transitions give us a simple correctness boundary. Mixed routing can be added
after the staged implementation provides measurements showing that it is
valuable.

## Core Invariants

- The population, archive, crossover, and mutation systems store only concrete
  ASTs containing static column inputs and fixed constants.
- Dynamic leaf and dynamic constant instructions are temporary lowering forms.
  They never enter an arena, archive, parent selection, or checkpoint.
- A dynamic binding is a search proposal, not a different expression type.
- The concrete source candidate remains available while dynamic proposals are
  evaluated. A binding sweep cannot discard it solely because the sampled
  settings were poor.
- Before ordinary evolution resumes, every selected dynamic binding is
  materialized into a concrete AST.
- Fitness used across search stages is produced by static SSE on the same rows.
  Dynamic SSE may select a proposal, but promoted candidates are statically
  rescored before cross-stage ranking or final reporting.
- Specialization, inspection, register, patch-space, and launch failures are
  hard errors. They must not silently route a candidate to another evaluator.
- Campaign accounting includes every topology-setting-row evaluation. A
  topology evaluated over 4,096 bindings consumed 4,096 times the row work of
  one static candidate even though it yields one promoted AST.
- Nonfinite results from raw operations such as `log(x)` are invalid fitness,
  not protected math. Protected semantics require an explicit operation or
  routine.

## Stage 1: Dynamic-Leaf Topology Search

Start with compact topologies and use
`secant_cubin_runner_run_dynamic_leaf_sse()` to evaluate many leaf bindings
without specializing a separate AST for every column and constant choice.

Initial policy:

- active maximum: 15 semantic nodes;
- bindable leaf occurrences: at most 8;
- settings per topology: start at 4,096;
- settings shared fairly across every topology in the population;
- setting zero: a deterministic all-column anchor;
- remaining settings: deterministic mixtures of column choices and constant
  values, including the normal constant pool and exact binary32 pi;
- result: one concrete winner per input topology for the first implementation.

Fifteen nodes is an initial campaign threshold, not an encoding limit. Eight
leaves form a 15-node full binary tree, but unary-heavy expressions can have
more than 15 nodes while retaining eight leaves. Benchmarks should determine
whether those expressions remain profitable in the dynamic stage.

Dynamic eligibility is not determined by AST size alone. The generated kernel
loads every runtime-selectable input column into its shared-memory row tile.
The policy must validate all of:

```text
bindable_leaves <= dynamic_leaf_recipe.num_dynamic_leaves
semantic_nodes <= active_dynamic_node_limit
estimated_lowered_instructions <= patch_budget_for_one_ast
(num_input_columns + num_targets) * tile_rows * sizeof(float) <= shared_memory_budget
```

The last condition makes dataset width part of evaluator selection. Dynamic
columns may be excellent for an eight-column problem and unattractive for a
wide derived-feature matrix at the same AST size.

The settings table is shared by all ASTs, so no one setting can preserve every
heterogeneous candidate's original leaves. The current setting generator makes
setting zero an all-column random binding, and the current
`secant_sr_search_dynamic_leaf_scores_apply()` always replaces each source
with its best sampled binding. That behavior is useful for the existing pure
dynamic-leaf experiment, but it is not a safe promotion contract for a staged
search.

The staged implementation must:

1. Score the concrete source population with static SSE.
2. Use `secant_sr_search_dynamic_leaf_programs_write()` to project temporary
   dynamic-leaf programs.
3. Evaluate the shared binding settings.
4. Materialize the best sampled binding for each topology.
5. Statically score those concrete proposals.
6. Copy the proposal into the alternate arena only when it improves the source;
   otherwise copy the original concrete source.

This compare-and-promote operation should replace, or be added alongside, the
current unconditional dynamic score apply function. Both source and proposal
fitness must come from static SSE on the same rows. Dynamic SSE chooses which
sampled binding becomes the proposal; it does not make the final acceptance
decision.

## Stage 2: Static-Column Structural Search

After the early binding phase:

1. Stop projecting the population into dynamic leaves.
2. Raise the active node, depth, and complexity limits up to the storage limits
   selected when the search was created.
3. Evaluate every generation with `secant_cubin_runner_run_sse()`.
4. Continue ordinary elite copying, crossover, subtree mutation, and point
   mutation on concrete ASTs.

No representation conversion is required at the transition because Stage 1
already ended with concrete programs. The static kernel naturally supports
larger trees, repeated columns, constants embedded directly in instructions,
and ASTs with more leaves than the dynamic recipe.

The search currently stores one immutable maximum in `SecantSRSearchConfig`.
Add active generation limits bounded by those allocation maxima. A setter such
as `secant_sr_search_active_limits_set()` should change generation policy
without reallocating either arena. It must reject limits above the values used
to measure search storage.

The initial transition policy should be deterministic and easy to compare:

```text
generations [0, dynamic_generations): dynamic-leaf topology search
generations [dynamic_generations, end): static-column structural search
```

Later, the transition can also trigger on stagnation or a row-evaluation
budget. Generation count alone is not a fair comparison when Stage 1 evaluates
thousands of settings per topology.

## Stage 3: Constant Refinement

Use two distinct constant mechanisms rather than treating them as equivalent.

### Per-expression GPU refinement

The constant-optimizer SSE shape packs one AST per kernel and gives each AST its
own current constants and stateless Philox perturbations. Its generated module
also contains a fixed reducer that regenerates the winning perturbation from
`(seed, generation, iteration, AST index)` and updates that AST's center. The
runner keeps each specialized module resident across all requested iterations,
then unloads it. The search samples eligible expressions independently, runs
bounded reusable chunks, writes winning values into fixed-constant ASTs, and
statically rescores before promotion.

### Per-expression local refinement

Use finite-difference L-BFGS once on the final best expression as an accuracy
audit. It is not a generation-stage search operator. Keeping it after search
prevents serial CPU optimization from invalidating GPU-throughput assumptions.

## Stage 4: Materialization and Scientific Outputs

`secant_cubin_runner_run_materialize()` should not participate in routine
SSE fitness evaluation. Materializing every candidate writes far more global
memory than reducing SSE in the kernel.

Use materialization for bounded, explicit work:

- validation predictions for the best expressions in each complexity bucket;
- train/test residual inspection;
- exporting discovered features;
- behavioral duplicate checks on a fixed probe set;
- constructing a later Gram/SINDy cohort;
- plotting and user-visible diagnostics.

The shortlist size and output memory must be explicit campaign parameters.
Materialization is the bridge from scalar search fitness to downstream
scientific analysis, not a competing evaluator for the whole population.

## CPU Oracle

Use `secant_cpu_run_sse()` and CPU materialization for:

- every backend correctness test;
- every newly added operation or routine;
- deterministic spot checks from each GPU generation;
- final validation of archived winners;
- diagnosing nonfinite or unexpectedly divergent scores.

The spot-check policy should sample candidates from multiple complexity
buckets, not only the current best. Tolerances should be operation-aware: ALU
expressions should match tightly, while MUFU-heavy expressions need the tested
approximation tolerance.

## Campaign Controller

Keep CUDA ownership and evaluator scheduling outside the population engine.
The controller owns prepared Secant plans and runners for:

- dynamic-leaf SSE;
- static-column SSE;
- dynamic-constant SSE;
- materialize.

Input columns, targets, and reusable settings remain device-resident across
stages. Template generation, NVRTC, inspection, plan allocation, and runner
creation occur before timed generations. Stage changes select an already
prepared runner; they do not compile a new CUDA template.

The controller records at least:

```text
topologies evaluated
bindings evaluated
concrete ASTs specialized
rows evaluated by each kernel shape
specialization, module load, and device runtime
promotions accepted
static rescore disagreements
nonfinite candidates
materialized rows and bytes
```

## Implementation Sequence

### Phase A: Global two-stage search

1. Add active generation limits bounded by the existing allocation limits.
2. Make the search app own both dynamic-leaf and static-SSE evaluators.
3. Statically score and retain each compact source population.
4. Run dynamic binding proposals and materialize their winners.
5. Statically rescore proposals and apply compare-and-promote.
6. Raise active limits and continue through static SSE.
7. Report row work separately for binding exploration and concrete scoring.

This phase exercises the most important transition without changing population
layout or introducing subset replacement.

### Phase B: Constant and materialize adapters

1. Add constant-only temporary projection and concrete materialization.
2. Independently sample eligible candidates for AST-local Philox refinement.
3. Statically rescore every optimized proposal and retain strict improvements.
4. Materialize complexity-bucket winners on validation rows.
5. Compare materialized GPU output against the CPU interpreter.

### Phase C: Optional mixed routing

Only after Phase A is benchmarked, consider routing newly born candidates by
their actual metadata:

- small, eligible candidates receive a dynamic binding sweep;
- large candidates go directly to static SSE;
- elite copies retain their concrete binding unless deliberately selected for
  rebinding.

Mixed routing requires indexed projection, subset materialization, and a safe
way to rebuild one concrete population from dynamic and static batches. It is
more complex and should earn its place through measured search quality or
runtime improvement.

## Required Tests

1. A dynamic sweep with no improving binding preserves the original AST and
   score byte-for-byte.
2. A promoted dynamic winner's static SSE improves the retained source.
3. Dynamic winner SSE matches static SSE after materialization within the
   backend tolerance.
4. The dynamic-to-static stage transition preserves every program and score.
5. Active limits can increase up to, but never beyond, allocation maxima.
6. Large ASTs that exceed the dynamic policy are accepted by static SSE.
7. Partial final modules work for every stage and do not write guard regions.
8. Shared dynamic-constant winners match fixed-constant static SSE.
9. Materialized GPU rows match CPU output for ALU, MUFU, exp, log, routines,
   and nonfinite cases.
10. Fixed seeds reproduce settings, promotions, and generation fingerprints.
11. Any CUBIN inspection or specialization error aborts the campaign visibly.

## Benchmark Matrix

Measure complete campaigns and isolated stages over:

- semantic nodes: 3, 7, 15, 31, and 63;
- bindable leaves: 2, 4, 8, 16, and 32;
- input columns: 4, 8, 16, 32, 64, and 128;
- settings: 128, 512, 2,048, and 4,096;
- rows: 1,024, 16,384, and 131,072;
- ALU, distinct MUFU, and exp/log-heavy corpora.

Record both row-evaluations per second and improvement per fixed row-evaluation
budget. The second metric determines whether dynamic binding is helping the
search rather than merely generating an impressive throughput number.
