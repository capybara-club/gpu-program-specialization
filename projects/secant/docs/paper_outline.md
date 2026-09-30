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

# Secant Paper Outline

## Paper Scope

The main paper should present Secant as a systems contribution and Secant-SR
as the end-to-end application that tests whether the systems mechanism changes
what is practical in symbolic regression. Iterated maps and ODE discovery are
optional extensions, not prerequisites for the central paper.

The narrative should remain centered on rapidly changing programs. Symbolic
regression supplies a demanding instance of that workload because candidate
programs change much faster than a conventional GPU compiler can comfortably
process them, while the useful scoring and optimization kernels are large
enough that interpretation and intermediate materialization are costly.

## Strongest Argument Hierarchy

### 1. A different point in the compilation/execution trade-off

Describe and measure the combination of:

- specialization latency per changing AST;
- specialization throughput per CPU core and with multiple producer cores;
- native GPU execution throughput;
- the effect of increasing the size of the surrounding fused kernel;
- all module, launch, synchronization, and lifecycle costs that occur during
  an end-to-end search.

The important comparison is not specialization latency alone or kernel speed
alone. The evidence should show both on the same workload. A useful primary
figure would place specialization or compilation latency on one axis and
resulting execution performance on the other for Secant, CUDA, PTX, and any
relevant interpreter or generic evaluator.

### 2. The advantage grows when the useful kernel becomes richer

Measure how conventional compiler latency changes as the kernel grows from a
small expression evaluator into fused SSE, mixed-leaf search, statistics, and
LM-related kernels. Compare this with the cost of specializing the AST-sized
portion of an already prepared Secant template.

This is stronger than reporting one unusually small kernel. It connects the
system design to workloads in which fusion avoids repeated global-memory
traffic, intermediate arrays, or separate evaluator passes.

### 3. Secant-SR converts the mechanism into end-to-end time to quality

Use complete search runs rather than evaluator-only measurements. Curves
should include population construction, genetic operations, specialization,
module handling, GPU execution, settings and bindings, constant optimization,
selection, and any scoring required during the run.

The main application evidence should include:

- held-out validation quality versus elapsed wall time;
- percentage of trials and datasets exceeding the SRBench accuracy threshold;
- time to first accurate or exact solution;
- final expression complexity;
- enough seeds to show run-to-run variability;
- full hardware and software configuration disclosure.

### 4. The result is not explained by one benchmark shortcut

Include ablations that separate the contributions of:

- fast native specialization;
- fused GPU evaluation;
- mixed static/dynamic leaf settings;
- binding search;
- LM constant optimization;
- population size and generation budget;
- operator language and maximum expression size.

Report AST skeletons, runtime settings, and fully instantiated configurations
as different counters. This prevents a settings-heavy search from being
presented as if every configuration required a unique compilation.

### 5. The binary specialization mechanism is controlled and verifiable

Provide correctness and portability evidence for the supported architectures:

- template inspection and patch-site validation;
- comparison against the portable CPU oracle and generated CUDA/PTX versions;
- randomized AST stress tests and boundary cases;
- sanitizer results;
- numerical-error characterization;
- explicit behavior on unsupported architectures or toolkit changes.

This section should explain the scope of the binary contract rather than imply
that arbitrary CUBIN rewriting is universally safe.

## Narrative Spine

The paper should move through the following sequence:

1. Rapidly changing programs create a mismatch with conventional GPU
   compilation.
2. Existing execution strategies trade compilation cost for interpretation,
   generic execution, or intermediate memory traffic.
3. Secant keeps the large kernel structure fixed and specializes only the
   changing typed program directly into validated native-code sites.
4. Microbenchmarks test specialization cost and native execution quality.
5. Secant-SR tests whether those properties improve complete scientific search
   workloads.
6. Ablations and limitations define where the method does and does not help.

## Main Paper

### Abstract

Include compact descriptions of:

- the rapidly changing GPU-program workload;
- the compile-versus-execute trade-off addressed by Secant;
- the direct-CUBIN template specialization mechanism;
- the Secant-SR application and its fused search kernels;
- the main categories of systems and end-to-end evidence;
- the supported scope and principal limitation.

Avoid filling the abstract with several unrelated benchmark percentages. Use
one systems result and one end-to-end application result after the final
experiments are selected.

### 1. Introduction

#### Workload motivation

Describe applications that repeatedly produce small typed programs embedded
inside much larger data-parallel kernels. Explain why symbolic regression is a
representative and difficult case: high program churn, large datasets, fused
reductions, runtime leaf choices, and continuous constant refinement.

#### Limitations of common execution paths

Summarize the costs associated with:

- compiling each candidate or candidate batch with a conventional toolchain;
- interpreting AST instructions on the GPU;
- evaluating a generic tensorized representation;
- materializing intermediate predictions or derivative arrays;
- grouping search unnaturally to amortize compilation.

#### Approach overview

Give a high-level picture of the template, validated patch islands, immutable
specialization plan, changing AST bytes, resulting CUBIN, resident module, and
fused evaluation kernel.

#### Contribution categories

List contribution types rather than detailed result prose:

- binary specialization design and API;
- typed AST and supported fused kernel families;
- correctness and architecture validation;
- Secant-SR integration;
- specialization, execution, and end-to-end evaluation methodology.

#### Recommended figure

An overview diagram showing one conventional per-program compilation path and
the Secant template-inspection/specialization path, ending at the same GPU
workload.

### 2. Workload Model and Design Requirements

#### Rapidly changing program model

Define the stable and changing parts of the workload:

- stable data layout, reduction, launch topology, and surrounding kernel;
- changing operators, leaves, constants, routines, or bindings;
- program batch size, rows, settings, targets, and specialization frequency.

#### Cost model

Break total time into:

- template generation and one-time compilation;
- template inspection;
- per-AST specialization;
- module loading and residency;
- kernel execution and synchronization;
- search-side CPU work;
- optional materialization and transfer costs.

State which terms are amortized and which occur for every candidate batch.

#### Design requirements

Describe requirements for native execution quality, bounded specialization
work, no allocation in the specialization path, concurrent specialization,
typed validation, explicit capacity limits, and failure on unsupported binary
contracts.

#### Comparison taxonomy

Classify competing execution approaches by when they compile, what they
interpret, whether they materialize intermediates, and how much kernel fusion
they permit. This establishes the comparison dimensions used later without
turning the section into full related work.

### 3. Secant Design

#### 3.1 Typed program representation

Describe the compact postorder instruction format, scalar types, indexed
leaves, constants, routines, validation limits, and relationship between the
portable CPU oracle and native backends.

Include a small AST-to-bytecode example and the corresponding expression
region in a generated kernel.

#### 3.2 Template generation

Describe how a recipe determines kernel capacities, data layout, launch
topology, patch-island size, register conventions, and the fixed fused work
around the changing program.

Distinguish source generation and conventional compilation of a reusable
template from per-AST specialization.

#### 3.3 CUBIN inspection and specialization plans

Describe discovery and validation of architecture-specific patch sites,
construction of immutable caller-owned plans, branch-to-epilogue handling,
immediate encoding, instruction placement, and specialization into independent
image copies.

Include pseudocode for inspection followed by repeated specialization.

#### 3.4 Safety and correctness boundary

Describe the conditions checked before specialization, supported compute
capabilities, capacity failures, invalid-image behavior after an error, and the
validation required for a new architecture or materially different toolkit.

#### 3.5 Runtime and module lifecycle

Describe producer threads, CUBIN copies, caching, eager module loading,
resident-module iteration, streams, launches, synchronization, and unloading.
Clarify which lifecycle choices belong to Secant and which belong to the host
application.

#### 3.6 Kernel families

Briefly describe why each implemented family is useful and what it avoids
materializing:

- materialization;
- fixed-column SSE;
- affine statistics;
- Gram statistics;
- dynamic-constant SSE;
- mixed dynamic-leaf SSE;
- LM-related evaluation and solve paths used by Secant-SR.

Detailed layouts should move to the appendix unless needed to interpret a
performance result.

### 4. Secant-SR: End-to-End Case Study

#### 4.1 Search representation

Describe the relationship among AST skeletons, static columns, immediate
constants, dynamic constants, dynamic column-or-constant leaves, bindings,
settings, and fully instantiated candidate configurations.

State the operator languages and complexity limits as experimental choices,
not universal properties of Secant.

#### 4.2 Evolutionary search loop

Describe population initialization, mutation, crossover, selection, maturity
or staging, deduplication, incumbent handling, and how generated ASTs are
grouped into templates and modules.

Include algorithm pseudocode showing where specialization and GPU evaluation
occur.

#### 4.3 Mixed-leaf settings and binding search

Describe how one skeleton represents multiple leaf assignments, how masks and
words choose columns or constants, how static leaves remain fixed, and how
winners are materialized back into individuals used by later generations.

Explain the accounting distinction between skeletons and configurations.

#### 4.4 Constant optimization with LM

Describe indexed dynamic constants, starts, binding/start combinations,
thread-owned state, per-AST kernels, accumulation of SSE, `J^T J`, and `J^T r`,
damping trials, acceptance, early exit, and reuse of a loaded module across
iterations.

Keep detailed register, shared-memory, and solver layouts in an appendix.

#### 4.5 Anytime measurement

Describe in-memory incumbent checkpoints, deferred held-out scoring, timer
boundaries, checkpoint overhead measurement, and reconstruction of validation
quality versus elapsed search time without perturbing every generation.

### 5. Experimental Methodology

#### 5.1 Research questions

Organize the experiments around questions such as:

- How quickly can Secant specialize changing programs?
- How does specialization scale with AST size and surrounding kernel size?
- How close is specialized execution to equivalent CUDA and PTX kernels?
- What is the complete cost after module lifecycle and synchronization?
- Does Secant-SR improve end-to-end time to held-out quality?
- Which search and kernel mechanisms account for the result?
- How stable are correctness and performance across supported GPUs?

#### 5.2 Hardware and software

Provide exact GPU, CPU, memory, driver, toolkit, compiler, library revision,
power mode, module-loading mode, and thread-affinity information. Separate
Rohini, Ada, and any additional validation hosts.

#### 5.3 Workloads and datasets

Describe the selected benchmark editions, sampling, train/validation/test
splits, target noise, seeds, row counts, input dimensions, and exclusions.

The main end-to-end set should include the complete 116-problem zero-noise
Feynman panel if feasible. SRBench 2025, SRSD, Strogatz, and other suites can
be added according to experiment capacity and relevance.

#### 5.4 Baselines

Describe the role and configuration of:

- generated CUDA kernels;
- generated PTX kernels;
- any generic interpreter or non-specialized evaluator;
- official single-core Operon reproduction;
- practical all-core Operon;
- same-GPU EvoGP where its workload is compatible;
- published PySR and wider SRBench results as external context;
- direct reruns used for shared time-to-quality curves.

Do not combine published timing from different hardware with direct speedup
claims.

#### 5.5 Metrics and accounting

Define:

- specialization latency and throughput;
- compile-plus-specialize latency;
- module lifecycle latency;
- row evaluations per second;
- AST skeletons, settings, and configurations evaluated;
- trial- and dataset-level `R^2 > 0.999` recovery;
- symbolic solution rate;
- expression complexity;
- time to quality and area or profiles over the anytime curve;
- confidence intervals and aggregation across seeds and datasets.

#### 5.6 Fairness and reproducibility

Describe timer boundaries, concurrency, CPU pinning, GPU exclusivity, stopping
criteria, common seeds and splits, operator-language differences, failed runs,
version pinning, and artifact availability.

### 6. Results

#### 6.1 Specialization latency

Present distributions of per-AST specialization time over AST size, operation
mix, routines, and producer-thread count. Include throughput scaling and tail
latency, not only the fastest mean.

#### 6.2 Sensitivity to kernel richness

Plot compilation or specialization time as fused functionality is added and as
template source or native code grows. Compare CUDA, PTX/ptxas, and Secant at
matched kernel shapes.

This should be one of the central systems figures.

#### 6.3 Native execution quality

Compare row throughput, launch topology, occupancy, registers, shared memory,
and output correctness for equivalent CUDA, PTX, and Secant kernels. Separate
simple ALU expressions, special functions, mixed leaves, and LM-related work.

#### 6.4 Complete pipeline cost

Show stacked or tabular timing for specialization, image copying, module load,
launch, execution, synchronization, and unload. Include resident-module
iteration separately from cold module lifecycle.

#### 6.5 End-to-end symbolic regression

Present held-out quality and recovery percentage against linear wall-clock
time. Include short-budget zooms and the full time range. Report both
trial-level and dataset-level percentages.

Use direct Operon comparisons for shared time-to-quality curves and the full
published SRBench table as clearly labelled external context.

#### 6.6 Search-mechanism ablations

Compare configurations that isolate:

- fixed leaves versus mixed leaves;
- binding count versus start count;
- legacy constant search versus LM;
- LM promotion probability and budget;
- settings per CTA and tile shape;
- population versus generation allocation;
- maximum node count and operator language;
- specialization and module caching choices.

Each ablation should keep total wall-clock accounting and report whether it
changes throughput, search diversity, or solution quality.

#### 6.7 Scaling and resource use

Report CPU specialization-producer scaling, GPU saturation behavior, memory
capacity, number of resident modules, and all-core Operon scaling. Include
energy or power only if measurement is reliable and consistently available.

#### 6.8 Correctness and portability

Summarize randomized cross-backend comparisons, supported GPU architectures,
sanitizer results, numerical tolerances, and failures on unsupported inputs.

### 7. Discussion

#### Where Secant should help

Discuss workload properties associated with a favorable result: high program
churn, a stable surrounding kernel, repeated execution over rows or settings,
valuable fusion, bounded typed programs, and a reusable binary contract.

#### Where Secant should not help

Discuss cases dominated by one long-lived kernel, large arbitrary generated
programs, unsupported operations, rapidly changing launch topology, tiny
execution workloads, or architectures without a validated backend.

#### Interpretation of settings-based search

Discuss the algorithmic benefit of evaluating many configurations per
skeleton and the corresponding reporting requirements. Separate compiler
throughput from total configuration throughput.

#### Generalization beyond Secant-SR

Describe candidate workloads by required kernel shape rather than presenting
unevaluated application claims. Iterated maps can appear here as a motivating
case for fused recurrence, with experimental results included only if the
implementation and baseline are complete.

### 8. Related Work

Organize related work by execution strategy rather than as a catalog:

- symbolic-regression systems using CPU GP and local constant optimization;
- GPU GP systems using interpreters, tensorized trees, levelized execution, or
  population packing;
- per-expression CUDA/PTX generation and JIT compilation;
- dynamic binary translation, template-based code generation, and binary
  rewriting;
- fused analytics, sufficient-statistics kernels, and GPU nonlinear least
  squares;
- symbolic and scientific-discovery benchmark methodology.

For every category, identify which costs occur when the program changes and
which fused operations are supported. Full related work can appear late in the
paper as long as the introduction contains enough early positioning for the
reader to understand the novelty.

### 9. Limitations and Broader Impact

Describe:

- NVIDIA- and architecture-specific binary contracts;
- maintenance across toolkit and ISA changes;
- bounded instruction and register capacities;
- the difference between numerical and exact symbolic recovery;
- benchmark-language and dataset-selection bias;
- CPU/GPU hardware asymmetry in wall-clock comparisons;
- risks of overcounting settings as independently compiled programs;
- reproducibility requirements for generated binaries and benchmark data.

Broader-impact content should stay proportional to the work. Relevant topics
include energy use, reproducible scientific-model discovery, and the risk of
interpreting high predictive accuracy as recovery of a physical law.

### 10. Conclusion

Summarize the workload, mechanism, evidence categories, practical boundary,
and the role of Secant-SR as an end-to-end demonstration. Avoid introducing
new application areas or unmeasured claims.

## Recommended Main-Paper Figures and Tables

### Figure 1: System overview

Pipeline from changing AST through template specialization, CUBIN/module
lifecycle, fused GPU execution, and search feedback, alongside a conventional
JIT path.

### Figure 2: Compile/specialize versus execute trade-off

Matched CUDA, PTX, Secant, and generic-evaluator points showing program-change
latency and resulting native execution throughput.

### Figure 3: Kernel-richness scaling

Compilation or specialization latency over increasing fused-kernel size and
functionality.

### Figure 4: End-to-end time to quality

Validation recovery percentage and mean/median validation quality against
linear elapsed wall time, including a short-time zoom.

### Figure 5: Ablation or cost breakdown

Either the search-mechanism ablation most explanatory of final quality or a
stacked end-to-end timing decomposition. The less central visualization can
move to the appendix.

### Table 1: System and kernel capabilities

Supported types, operations, kernel families, capacities, architectures, and
whether each path materializes intermediate predictions.

### Table 2: Native performance and lifecycle

Matched kernel throughput, specialization/compilation latency, module costs,
and numerical error.

### Table 3: End-to-end benchmark summary

Hardware, budget, operator language, seeds, trial success percentage, dataset
recovery percentage, symbolic recovery, expression complexity, and runtime for
Secant-SR and direct baselines.

## Appendix Plan

### A. Typed instruction format

Full opcode families, encodings, limits, validation rules, routines, and
examples.

### B. CUBIN template contracts

Patch-island layouts, instruction encodings, branch calculations, immediate
handling, architecture tables, and inspection invariants.

### C. Kernel layouts

Thread/warp/CTA ownership, row tiles, settings tiles, shared memory, registers,
atomics, output layouts, and launch parameters for every measured shape.

### D. LM derivation and solver

Gradient construction, sufficient statistics, damping schedule, linear solve,
acceptance rules, state layout, starts, and binding interaction.

### E. Secant-SR algorithm details

Complete pseudocode, mutation/crossover distributions, maturity stages,
promotion probabilities, population schedules, and materialization of winning
settings.

### F. Benchmark definitions

Dataset lists, splits, seeds, target noise, scaling, success criteria, operator
languages, time/evaluation limits, and baseline versions.

### G. Full results

Per-dataset and per-seed tables, failure cases, confidence intervals,
time-to-quality checkpoints, expressions, and complexity distributions.

### H. Additional ablations

Tile sizes, CTA sizes, AST packing, settings counts, starts versus bindings,
LM iterations, module residency, producer cores, and population/generation
trade-offs.

### I. Correctness and hardware validation

Randomized corpora, boundary cases, CPU/CUDA/PTX/CUBIN comparisons, sanitizer
output, supported GPUs, toolchain versions, and numerical tolerances.

### J. Optional recurrent demonstration

If implemented in time, include the iterated-map kernel, trajectory-level
splits, rollout metrics, materialization baseline, and known-map recovery. Keep
ODEBench out of the experimental claims unless continuous integration and
search are complete.

## Minimum Evidence for a Strong Submission

The main paper can remain focused on Secant and Secant-SR if it contains:

1. A matched specialization/compilation benchmark over increasing AST and
   fused-kernel complexity.
2. Native CUDA, PTX, and Secant execution comparisons with correctness checks.
3. Complete pipeline timings including module lifecycle.
4. End-to-end time-to-quality curves against at least one strong direct CPU
   baseline.
5. A broad benchmark result with prescribed seeds and held-out evaluation.
6. Ablations for settings/bindings and LM versus the legacy constant search.
7. A reproducible artifact with exact hardware, versions, datasets, and raw
   per-trial results.

Iterated-map or ODE results would broaden the application story, but they
should not displace these core experiments or appear as incomplete evidence.
