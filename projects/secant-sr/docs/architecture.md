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

# secant-sr Architecture

## Boundary

`secant-sr` owns search policy and population memory. Secant owns expression
bytecode semantics, CPU interpretation, native instruction generation, CUBIN
specialization, and the bulk GPU runner. The search engine never emits SASS or
duplicates Secant operations.

The evaluator boundary is deliberately narrow:

1. `secant_sr_search_asts_get()` exposes the current ordered AST pointer array.
2. An evaluator produces one SSE value per AST for the current target.
3. `secant_sr_search_scores_set()` computes goodness-of-fit metrics and updates
   the archive.

This allows static-column SSE, runtime-bound leaves, and future Gram builders to
share one evolutionary core.

## Population memory

Two caller-owned generation arenas alternate roles. Parent programs and
sidecars remain immutable while children are built in the other arena. At a
generation boundary the old child arena is reset in O(1); there is no
per-individual allocation or free.

Each individual has a postorder sidecar entry for every node:

- instruction and subtree byte offsets;
- subtree byte and node counts;
- first subtree node;
- depth, weighted complexity, instruction width, and arity.

Program-level descriptors are separate from the node sidecar. A validated
on-demand pass produces a 128-bit static-column mask, unique and total column
counts, constant count, operation-family counts, and a 64-bit operation mask.
The default complexity-only policy computes these descriptors only for the
generation winner. Enabling multidimensional QD opts candidate creation into a
second linear descriptor pass.

Subtree selection is therefore O(1). Crossover copies a parent prefix, one
donor subtree, and a parent suffix. Subtree mutation copies a prefix, generates
a replacement, and copies the suffix. Point mutation copies one parent and
rewrites one compatible token. Every candidate is decoded once before commit,
which independently verifies stack shape, return termination, depth,
complexity, and byte limits.

Storage measurement reserves worst-case program and node capacity for every
member in both arenas. This is simple and deterministic but can dominate memory
for million-member populations. A future version could expose separate total
arena capacities while preserving the same hot path.

## Selection

Fitness records SSE, MSE, RMSE, NMSE, R-squared, and a parsimony-adjusted score.
Elites are copied into fixed MAP-Elites cells. Complexity is always represented;
unique static-column count and transcendental-operation count are optional
dimensions whose counts saturate into their final configured bucket. Multiple
slots per cell preserve several exact programs, while fingerprint matching
removes exact duplicates. Each variation parent may be drawn by selecting an
occupied archive cell uniformly and using that cell's best elite, or by fitness
tournament from the current population. The measured default uses only the
complexity dimension and tournament parents.

New generations combine archive copies, tournament-selected crossover,
subtree mutation, point mutation, and random immigrants. Parent programs are
never modified. Lineage indices, origin, birth generation, fingerprint, depth,
and complexity remain attached to each child.

## Repeated Campaigns

Dataset sweeps keep policy variation outside the population hot path. The same
contiguous AST, node, individual, and archive layouts are reused for every
operator profile; no profile introduces a second object graph. Generated
skeleton CUBINs are keyed by Secant version, complete source recipe,
architecture, and compiler configuration in SQLite, so rerunning a dataset
normally pays only Secant specialization, module loading, and evaluation. A
warm lookup happens before CUDA source generation.

The suite driver supports resumable `(problem, seed)` records and small named
panels for rejecting weak policies before a full sweep. With
`--persistent-process`, pending records run through one process that retains the
CUDA context and SQLite connection. Shape-specific plans, runners, streams, and
device buffers are still rebuilt for each dataset because their capacities
depend on input count, row count, and population shape. Reusing those resources
requires a capacity-keyed evaluator cache; it is intentionally separate from
the already safe process-level reuse.

## Constant refinement

Generation-stage refinement is GPU-only. Every generation independently samples
at most 4,096 fixed-constant expressions every generation by default,
using an optimizer RNG stream that does not consume structural mutation state.
Each sampled expression gets its own Philox perturbations and center updates;
the complete fixed-constant set must fit the configured dynamic capacity.
Optimized values are written back into a concrete AST, statically rescored over
all training rows, and promoted only for lower training SSE.

`secant_sr_search_best_constants_optimize_cpu()` is deliberately outside the
generation loop. After search, it applies finite-difference L-BFGS once to the
best archived expression and reports before/after train and held-out scores.
The archived bytes are restored unless full-training SSE improves. This keeps
the CPU implementation as a final accuracy audit instead of allowing a serial
optimizer to dominate accelerated population evaluation.

The distinction is performance-critical. An earlier prototype refined roughly
60-80 population members on the CPU each generation and took 0.12-0.16 seconds
even on a 257-row prefix, versus roughly 1.1-1.3 ms for its direct-CUBIN dynamic
evaluation probe. That population-wide CPU path has been removed.

## Population-generation scaling

A matched warm-cache Feynman III.19.51 measurement used 100 generations, 4,096
training rows, 8,192 validation rows, no constant refinement, and the staged
backend. At 32,768 individuals, child construction consumed 1.501 s of a 9.615 s
run (`15.6%`, 15.0 ms/generation). At 1,048,576 individuals, it consumed 72.867 s
of a 290.617 s run (`25.1%`, 728.7 ms/generation). The population grew by 32x
while end-to-end time grew by 30.23x.

Population construction is therefore material at million-member scale but not
yet dominant. Parallel child construction has an Amdahl ceiling of about 1.33x
for that run before synchronization and memory-bandwidth costs. Any future
parallel implementation should partition deterministic child output ranges
after archive selection; it must not overlap dependent generations or alter RNG
consumption silently.

## Current benchmark

On the RTX 5090 host CPU, the current Release build generated ten
100,000-member generations with a 24-node limit at approximately 6.1 million
children per second on one core. Exact measured search storage was 157.4 MiB.
The default path stores no population-wide program-feature sidecar. This
benchmark uses deterministic synthetic fitness and excludes expression
evaluation.

The deterministic Nguyen-1 test uses 4,096 individuals and the Secant CPU
oracle. A separate test evaluates the same random 256-expression population
through CPU and direct-CUBIN SSE and compares every result. The campaign suite
uses independent training and holdout coordinates and applies one broad search
policy to six one-, two-, and three-input problems. See `docs/sr_suite.md` for
the measured recovery rates and limitations.

The dynamic-leaf campaign benchmark uses 8,192 structures, 4,096 settings,
eight runtime leaves, 64-row static tiles, and 1,048,576 rows. On the RTX 5090,
one deterministic Nguyen-1 generation measured 3.056e12 row-evaluations per
second over the complete 11.512-second evaluation stage. This includes AST
projection, specialization, module loading, GPU execution, synchronization,
copying 33,554,432 SSE values to the host, choosing each structure's best
setting, and materializing those settings back into concrete individuals. It
is an integrated search measurement rather than Secant's peak kernel-only
throughput.
