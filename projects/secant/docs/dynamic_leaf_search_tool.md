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

# Mixed-Leaf Search And Packed Constant Refinement

This document describes two complementary direct-CUBIN kernel shapes for
high-volume symbolic search:

1. Mixed static/dynamic-leaf SSE for discrete leaf assignment.
2. Packed shared-jitter SSE for continuous constant refinement.

Both shapes specialize AST operators directly into a reusable CUBIN skeleton.
Their timed runner measurements include SASS specialization, eager CUDA module
loading, kernel execution, synchronization, and unloading. They exclude the
one-time source generation, NVRTC skeleton compilation, CUBIN inspection,
runner creation, allocation, upload, warmup, and correctness checks.

A row-evaluation means evaluating one AST for one runtime setting and one input
row. All measurements below used one target and an RTX 5090.

## Mixed Static/Dynamic-Leaf SSE

### Purpose

The mixed-leaf kernel searches over discrete choices at selected AST leaves
without recompiling the AST topology. It can preserve known columns and
constants while leaving holes that each runtime setting fills with either a
column or an `f32` constant.

An AST may contain:

```text
static_column(i)
dynamic_column_or_constant(i)
constant(c)
```

Each setting provides:

```text
leaf_mask[setting]
leaf_words[setting][dynamic_slot]
```

For dynamic slot `i`:

- A set mask bit interprets `leaf_words[setting][i]` as a column index.
- A clear mask bit interprets it as the exact bits of an `f32` constant.

The kernel initializes each dynamic register with the constant interpretation,
then uses a predicated shared-memory load to replace it when the setting selects
a column. Static-only, dynamic-only, and mixed recipes use the same public shape
but generate specialized skeleton behavior for the enabled capacities.

For example:

```text
sin(hole0) + static_column(2) * hole1
```

can represent:

```text
sin(column(4)) + column(2) * 0.25
sin(1.7)       + column(2) * column(6)
sin(column(1)) + column(2) * -3.0
```

### Kernel Shape

- A CTA loads a 64-row tile, the active input-column prefix, and targets into
  shared memory once.
- Threads represent independent settings rather than rows.
- Each thread loops over the cached rows and evaluates every packed AST.
- Each AST has its own SSE accumulator.
- Tile partials are atomically accumulated into:

  ```text
  output[ast][target][setting]
  ```

- The tested recipes reserve up to eight dynamic slots and up to 32 static
  input columns.
- The runtime may activate fewer columns, ASTs, targets, or settings than the
  compiled capacities.
- A setting is shared across all ASTs in a packed kernel, so an algorithm should
  group ASTs for which a common leaf-assignment distribution is meaningful.

### Measured Throughput

| Workload | Representative shape | Runtime only | Full runner pipeline |
| --- | --- | ---: | ---: |
| Very simple upper bound | 32 ASTs/kernel, 8 kernels, 8,192 settings, 65,536 rows | About `1.04e13` | Up to `1.02e13` row-evals/s |
| Short mixed ALU | 16 ASTs/kernel, 8 kernels, 4 static columns, 4 dynamic sites, 8,192 settings, 65,536 rows | `2.737e12` | `2.713e12` row-evals/s |
| Distinct 31-node mixed ALU | 64 modules, 8 kernels/module, 32 ASTs/kernel, 8 static columns, 4 dynamic sites, 4,096 settings, 65,536 rows | `2.217e12` | `2.209e12` row-evals/s |
| Distinct mixed MUFU | 8 ASTs/kernel, 8 kernels, 4 static columns, 4 dynamic sites, 8,192 settings, 65,536 rows | `8.261e11` | `8.221e11` row-evals/s |

The 64-module, 31-node measurement evaluates 16,384 distinct ASTs per measured
iteration. Its stable full-pipeline result demonstrates that the approximately
`2.2e12` ALU rate is not merely a resident single-module number.

Settings generally reach diminishing throughput returns around 2,048 to 4,096,
although 8,192 settings retain essentially the same per-row throughput. Packing
roughly 8 to 32 ASTs per kernel works well depending on expression weight.

## Packed Shared-Jitter Constant Optimizer

### Purpose

The packed optimizer refines the existing constants of many unrelated ASTs.
Each AST has independent center, scale, velocity, incumbent SSE, and winning
proposal state, but ASTs share a generated normalized jitter vector for each
setting. This lets many ASTs reuse one loaded row tile without storing one
complete perturbation vector per AST and setting.

Incoming programs use indexed dynamic-constant instructions for optimizable
constants. Ordinary static columns and non-optimizable immediate constants are
also allowed. For AST `a`, dynamic constant `j`, setting `s`, and optimizer round
`t`, evaluation uses:

```text
mean[a][j] = center[a][j] + momentum * velocity[a][j]
candidate[a][j][s] = mean[a][j] + scale[a][j] * jitter[j][s]
```

Philox4x32-10 generates `jitter` from counters containing the seed, generation,
optimizer round, and setting. Setting zero evaluates the momentum proposal mean.
Winner mode accepts only strict incumbent improvements. Accepted steps update
the center and velocity and adapt each coordinate's scale toward twice its
accepted random-step magnitude. Rejected steps preserve the center and SSE,
damp velocity by momentum, and shrink scales by the configured failure decay.
This remains the default, with momentum disabled.

### Per-AST, Per-Constant Scaling

The scale is not global, per AST, or per constant index across the population.
It is an independent value for every `(AST, dynamic constant)` pair:

```text
center[ast][constant]
scale[ast][constant]
velocity[ast][constant]
incumbent_constant[ast][constant]
```

For example, the first constant of one AST can search with a radius of `100.0`
while its second constant searches with `0.001`; another AST packed into the
same kernel can use two entirely different radii. Only the dimensionless Philox
jitter is shared:

```text
jitter[constant][setting]
```

Specialization embeds each AST's current center and scale as SASS immediates.
At each indexed dynamic-constant instruction, the generated expression is
equivalent to:

```text
candidate = jitter_register * scale[ast][constant]
candidate = candidate + center[ast][constant]
```

This costs one `FMUL` and one `FADD` per dynamic-constant occurrence, but it
does not load per-AST constants or scales in the row loop. Reusing the same
normalized jitter registers therefore does not force unrelated ASTs or
constants to share units, magnitudes, or search radii.

In winner mode, an accepted step `delta` adapts only its corresponding scale:

```text
delta = old_scale * winning_jitter
target_scale = 2 * abs(delta)
new_scale = old_scale + scale_learning_rate * (target_scale - old_scale)
```

The result is clamped to the configured scale bounds. A rejected round instead
multiplies every scale for that AST by `scale_failure_decay`. Consequently a
poorly conditioned expression can independently contract a nearly solved
constant while retaining a wide search around another constant that is still
far from its optimum.

Elite-distribution mode is also available. It retains the best evaluated
proposal as the incumbent while updating a separate search center from the
uniformly weighted top-`k` mean. Its diagonal scale target combines elite
variance with winner-step expansion so a distribution at the edge of its
sampling box does not collapse before reaching a distant optimum.

### Kernel And Feedback Shape

- The measured recipe uses 64 kernels per module and 32 ASTs per kernel.
- A CTA loads a 128-row tile with four columns and one target into shared memory.
- Threads generate and evaluate 8,192 settings.
- The same raw jitter registers are reused by every AST, while each AST embeds
  its own proposal means and per-constant scales in the specialized SASS.
- SSE tile partials are accumulated independently for every `[ast][setting]`.
- A GPU reducer selects either one winner or the top `k` settings for each AST,
  regenerates their Philox perturbations, and updates the compact
  `[SSE, search centers, scales, velocities, incumbent constants]` state row.
- Reducer output is copied asynchronously into pinned worker storage.
- The next optimizer round respecializes each AST around its updated center and
  loads the resulting module. Worker queues and CUDA events overlap feedback,
  specialization, loading, and execution across modules.
- Every adaptive round therefore contributes one module load per input module;
  it is not a resident-module inner loop because the AST center immediates change.

### Measured Throughput

A fresh default benchmark on August 8, 2026 used:

```text
modules:                 4
workers / streams:       24 / 8
kernels per module:      64
ASTs per kernel:         32
total ASTs per round:    8,192
columns / constants:     4 / 4
settings:                8,192
rows:                    10,000
tile rows / CTA threads: 128 / 128
optimizer rounds:        4
measured runner calls:   3
initial scale:           1.0
momentum:                0.0
scale learning rate:     0.25
failure scale decay:     0.5
scale bounds:            1e-6 to 1e6
```

It processed `8.05306368e12` row-evaluations and measured:

```text
kernel and reducer runtime:  1.666e12 row-evals/s
complete runner pipeline:    1.642e12 row-evals/s
pipeline wall time:          4.905 s
actual specialization work:  0.032 s across workers
module loads:                48
```

The same shape with the MUFU corpus measured `1.130e12` kernel/reducer
row-evals/s and `1.118e12` complete-pipeline row-evals/s.

The reusable 64-kernel skeleton took 6.48 seconds to generate and compile with
NVRTC in this run. That is template preparation, not an AST-to-CUBIN hot-path
cost, and is excluded from both throughput rates. A deployment can prebuild or
cache this architecture-specific skeleton.

Winner-only and elite-8 reducers both measured approximately `1.67e12`
kernel/reducer row-evals/s and `1.64e12` complete-pipeline row-evals/s at this
shape. Elite selection therefore had no measurable throughput cost once row
evaluation dominated. Controlled convergence results and the current policy
recommendation are recorded in `constant_optimizer_convergence.md`.

## Intended Combined Use

The two kernels solve different search problems:

| Kernel | Search variable | Shared across packed ASTs | Per-AST state |
| --- | --- | --- | --- |
| Mixed-leaf SSE | Column/constant choice at AST holes | Leaf setting distribution | AST topology and SSE accumulator |
| Packed constant optimizer | Continuous offsets around existing constants | Normalized Philox jitter vectors | Per-constant centers, scales, velocities, incumbent constants, and incumbent SSE |

An intended staged search is:

1. Use mixed-leaf settings for broad column and coarse-constant discovery.
2. Reduce the `[ast][target][setting]` surface on the GPU.
3. Rewrite winning dynamic leaves into ordinary static AST instructions.
4. Select promising archive members for packed constant refinement.
5. Run several adaptive rounds and rewrite the winning constants into the ASTs.
6. Statically rescore promoted candidates before mutation or crossover.

These kernels are implemented, correctness-tested, and benchmarked in Secant.
The mixed-leaf stage and the packed optimizer are not yet composed into the
active `secant-sr` search policy, so these are execution measurements rather
than end-to-end equation-recovery results.
