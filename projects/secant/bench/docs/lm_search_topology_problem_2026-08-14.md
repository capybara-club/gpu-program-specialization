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

# GPU LM topology and symbolic-search design problem

Please analyze the following GPU algorithm-design problem. Assume no knowledge
of the codebase that motivated it. The goal is to recommend both a GPU kernel
topology and a symbolic-regression search policy, not merely to optimize an
isolated microbenchmark.

## Goal

We are building a GPU symbolic-regression system. It searches expression trees
(ASTs) and may refine the numeric constants in each expression using
Levenberg-Marquardt (LM). We want LM to be fast enough to be a first-class part
of search rather than an occasional post-processing step.

An LM candidate is identified by:

1. an AST structure;
2. a discrete binding of each dynamic leaf to either a numeric constant or an
   input-data column;
3. a column index for every leaf bound to a column; and
4. an initial value, or "start," for every leaf that remains a constant.

Different starts for the same AST and binding are independent local LM solves.
Different bindings are also independent solves because they represent
different expressions.

Typical datasets contain roughly 10,000 to 16,384 rows and up to 32 input
columns. The current prototype uses eight dynamic leaves/parameters and eight
input columns. The GPU is an NVIDIA RTX 5090, although the design should also
work on a 4090.

## Per-candidate LM work

For every row, the specialized AST produces a prediction and the analytic
gradient with respect to the active constants. A leaf bound to a data column
has gradient zero. Repeated occurrences of one dynamic leaf refer to the same
parameter.

For `p` parameters, one LM iteration accumulates:

- scalar SSE;
- `p` entries of `J^T r`; and
- the packed upper triangle of `J^T J`.

The number of accumulated values is:

```text
1 + p + p(p + 1)/2
```

At `p = 8`, this is 45 floating-point accumulators per candidate. The current
thread-owned and warp-owned kernels compile to approximately 96 registers per
thread with no local-memory spill for the measured AST.

Each row evaluation in the performance numbers below means evaluating one AST
for one candidate setting on one data row and accumulating its LM statistics.
It is not equivalent in operation count to a simple SSE-only row evaluation.

## Existing tile-static execution

The input data is staged into shared memory one row tile at a time. The grid's
X dimension selects a row tile and its Y dimension selects a group of candidate
settings. Every row tile atomically contributes partial LM statistics to global
memory. Therefore a small number of settings does **not** necessarily imply a
small number of CTAs.

For example, 16,384 rows with a 64-row tile produces 256 row CTAs even when
there is only one setting group. With four starts, the thread-owned kernel can
therefore launch 256 CTAs, but only four threads in each CTA perform candidate
work. This distinction between "few CTAs" and "many mostly dark CTAs" is
important.

The kernel can be reused for several LM iterations. After each iteration, a
small solver updates each candidate's constants and the next iteration runs
against the same specialized AST code and discrete bindings.

## Candidate ownership topologies

### 1. Thread-owned, tile-static

One thread owns one candidate setting for one row tile. It keeps all 45 LM
values in registers, walks the tile serially, and atomically writes its 45
partials. There is no inter-thread reduction of the LM state.

Advantages:

- no reduction or synchronization for LM statistics;
- excellent throughput when a CTA has enough settings to occupy its threads;
- fixed, simple register state per active thread; and
- row tiles and setting groups create abundant independent CTAs.

Disadvantages:

- if an AST has only 4-8 starts and no binding enumeration, only 4-8 threads
  in a nominally 64-256-thread CTA do useful candidate work;
- every active thread independently reads the shared row tile; and
- creating fake or redundant starts only to fill lanes wastes search work.

### 2. Warp-owned, tile-static

One warp owns one candidate. Its lanes divide the rows in the tile. Every lane
keeps a partial 45-value LM state, after which the warp reduces all statistics
and lane zero performs the atomics.

At eight parameters, reducing 45 statistics with a five-stage warp shuffle
requires 225 shuffle-add steps per candidate per tile. The row-parallel work
also changes the best tile size.

Advantages:

- 4-8 settings can use 4-8 complete warps instead of 4-8 individual threads;
- no block-wide reduction is required; and
- it retains tile-static parallelism and bounded atomics.

Disadvantages:

- all 32 lanes carry the large LM state;
- every statistic requires a warp reduction; and
- when many independent settings already exist, the reduction overhead buys no
  occupancy and substantially lowers throughput.

### 3. CTA-owned, tile-static

This topology has not yet been implemented in the same benchmark. One CTA
would own one candidate on one row tile. Threads divide the rows, keep partial
LM states, reduce 45 statistics across the CTA, and atomically write one result
per statistic.

This is different from a persistent CTA that walks the entire dataset. A
tile-static CTA-owned grid contains approximately:

```text
candidate count * ceil(dataset rows / tile rows)
```

CTAs. Four candidates and 79 row tiles would therefore launch 316 CTAs, not
four. It may provide enough occupancy without traversing the full dataset in
one CTA.

Its costs are duplicated tile loads across candidates, a block-wide reduction
of 45 values, synchronization, and high per-thread register use. It might also
load only the columns required by its one binding, which could reduce shared
memory traffic for sparse expressions.

### 4. CTA-owned, full-dataset

One CTA owns one complete AST/start/binding candidate and strides through all
rows before writing once. This minimizes global atomics and can load only the
needed columns, but 4-8 starts for one AST create only 4-8 CTAs. We currently
believe this is too little parallelism and do not want this to be the primary
design.

### 5. Thread-owned, full-dataset

One thread owns a complete candidate and walks the full dataset. With only a
few candidates this can launch one mostly dark CTA. It avoids row-tile atomics
but appears even less suitable for the small-candidate regime.

## Measured ownership crossover

These are hot-kernel PTX measurements on an RTX 5090 using 16,384 rows, eight
parameters, eight columns, one square/cube-heavy AST, and mixed constant/column
bindings. Each reported cell is the best observed topology in the tested
thread/tile neighborhood.

| Total settings | Thread-owned | Warp-owned | Relative result |
|---:|---:|---:|---|
| 4 | 4.73B row evals/s | 9.40B row evals/s | warp is 1.99x faster |
| 8 | 9.78B row evals/s | 18.49B row evals/s | warp is 1.89x faster |
| 8,192 | 177.5B row evals/s | 120.9B row evals/s | thread is 1.47x faster |

The small-setting thread winner used a 64-row tile. The small-setting warp
winner used a 256-row tile. At 8,192 settings, thread-owned used 128 settings
per CTA, a 128-row tile, and 128 threads. Warp-owned peaked near 32 settings per
CTA, a 256-row tile, and 128 threads.

This establishes a real crossover:

- warp ownership recovers otherwise dark-lane throughput when only a few
  candidates exist;
- thread ownership wins decisively when there are enough candidates to fill
  its lanes; and
- the warp reduction is expensive, but it is still worthwhile when its
  alternative is severe thread masking.

## AST packing and register pressure

The SSE-only kernel can evaluate many ASTs in one row pass because it needs one
scalar SSE accumulator per AST. This packing was previously important for
reaching multi-trillion SSE row-evaluation rates.

The current generated LM kernel likewise allocates a separate 45-value state
for every AST packed into the kernel. That is not viable for substantial LM
packing: two or more simultaneously live LM states are likely to reduce
occupancy sharply or spill to local memory.

A proposed alternative is sequential AST packing:

```text
load one row tile into shared memory
load or hoist one candidate's binding/start data

for each AST in a small packed group:
    clear and reuse one 45-value LM register state
    walk the resident row tile for this AST
    atomically write this AST's partial statistics
```

AST-specific evaluation could be selected by a uniform branch or switch. The
AST index is identical for every thread in the CTA, so the branch does not
diverge. One version would branch inside the common row loop, produce a
prediction and gradient, and then join one shared `J^T r`/`J^T J` accumulation
block. Another version would branch once per AST into a duplicated row loop,
trading a larger instruction footprint for fewer dynamic branches.

Sequential packing reuses the staged tile, setting metadata, module, and
launch, while keeping only one LM state live. It does not reduce the AST
arithmetic or the 45 atomics per AST/tile, and unlike SSE packing it cannot
evaluate many ASTs in the same row pass while retaining only cheap state.

Other multi-AST options are:

- use `blockIdx.z` to assign one AST to each CTA and perform one uniform AST
  dispatch; this packs code and launches but does not reuse tiles across ASTs;
- assign a small sequential group of perhaps 2, 4, or 8 ASTs per CTA, with the
  group selected by `blockIdx.z`; or
- keep one AST per specialized kernel and use concurrent streams, accepting
  repeated tile loads and more launches.

## Search-policy fork: starts only versus bindings plus starts

The kernel topology depends strongly on whether LM is merely a constant
refiner or a first-class search operator.

### Starts-only LM

If the symbolic search has already fixed the AST and every column/constant
binding, LM may need only 4-8 starts. For one AST at a time this leaves the
thread-owned kernel heavily masked. Warp-owned or a new CTA-owned tile-static
kernel may be better, but both pay reductions that thread ownership avoids.

Launching many ASTs together creates more global work, but if each AST is in a
different generated kernel, individual kernels can still be small unless they
are run concurrently. Packing ASTs introduces the register/code-size problem
described above.

### Binding-and-start LM search

An alternative is to make LM a first-class search stage. For each AST, the
search deliberately enumerates or samples useful discrete configurations:

- which holes are constants versus data columns;
- which data columns occupy the column-bound holes;
- possibly permutations of column assignments; and
- several continuous starts for each binding.

This naturally produces hundreds to thousands of meaningful independent
candidate settings instead of manufacturing redundant starts for occupancy.
At that scale, the measured thread-owned topology is the throughput winner.
The same specialized AST kernel can be reused for several LM iterations while
the constants change in GPU memory.

This approach makes GPU occupancy and the search algorithm agree: discrete
bindings provide global exploration, while LM provides local continuous
optimization. It may also find better models than optimizing constants only
after the binding search has committed to one assignment.

However, binding enumeration has algorithmic costs:

- it can create a combinatorial candidate space;
- candidates with different numbers of active constants still pay for a fixed
  maximum-size gradient and 45-value state unless grouped and specialized by
  active parameter count;
- many column permutations may be redundant under AST symmetries;
- bindings should be generated because they are useful search hypotheses, not
  solely to inflate GPU utilization; and
- the system needs a policy for pruning candidates between LM iterations.

## Core design tension

There are two very different operating regimes:

1. **Few candidates per AST:** 4-8 starts, perhaps one fixed binding. The GPU
   needs row-parallel ownership or many concurrently scheduled ASTs to avoid
   dark lanes.
2. **Many candidates per AST:** hundreds or thousands of bindings, column
   permutations, and starts. Candidate-parallel thread ownership avoids
   reductions and achieves the highest measured row throughput.

A single fixed ownership model may therefore be the wrong abstraction. The
runtime could choose thread-, warp-, or CTA-owned execution from the number of
live candidates, row count, parameter count, and AST batch size. On the other
hand, making binding enumeration an intentional part of LM search may keep the
important production workload almost entirely in the many-candidate regime
and simplify the production kernel surface.

## Questions to resolve

Please reason about the following:

1. Should binding and column-assignment search be integrated with LM so that
   the normal production workload contains enough candidate settings for the
   thread-owned kernel?
2. Is 4-8 starts per binding a reasonable default, and how should starts be
   allocated across many bindings under a fixed time budget?
3. Should the implementation retain separate thread- and warp-owned kernels,
   add a tile-static CTA-owned kernel, or deliberately standardize on one
   topology?
4. For only 4-8 candidates, is warp-owned reduction likely better than a
   CTA-owned reduction once duplicated tile loading and atomics are included?
5. Can a CTA-owned tile-static kernel improve performance without ever using
   the low-occupancy full-dataset-per-CTA design?
6. For multi-AST LM, is a uniform per-row AST branch with one common LM
   accumulation block preferable to duplicated per-AST row loops?
7. Should ASTs be grouped into small sequential packs, assigned independently
   through a grid dimension, or kept in separate concurrently launched
   kernels?
8. Should candidates be bucketed by active constant count so kernels accumulate
   smaller `J^T J` systems rather than always paying for eight parameters?
9. How should the design account for repeated LM iterations, candidate pruning,
   global atomic contention, instruction-cache pressure, and specialization or
   module-launch overhead?
10. What benchmark matrix would distinguish the best architecture without
    conflating useful search work with artificial occupancy work?

The desired outcome is a concrete recommended production topology, an adaptive
dispatch policy if necessary, a binding/start search policy, and a short list
of decisive experiments.
