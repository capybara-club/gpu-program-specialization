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

# Dynamic Settings Topology

## Current Contract

The current dynamic-constant and dynamic-leaf SSE kernels apply one runtime
setting environment to every AST packed into a kernel.

```text
dynamic constants: [constant][setting]
dynamic leaves:    [setting][leaf]
output:            [ast][target][setting]
```

For a fixed `setting`, every packed AST sees the same constant values or leaf
bindings. ASTs may use different subsets of those slots, but slot `i` has one
value for the entire packed cohort.

The dynamic-leaf generator has two register-bank modes. A dynamic-only recipe
sets `num_static_input_columns` to zero and rejects static-column instructions.
A mixed recipe sets it equal to `num_input_columns`; every loaded column then
has one fixed register per setting thread in addition to the dynamic leaf
registers. The static-only row-parallel SSE shape remains separate because it
does not need a settings dimension or predicated leaf loads.

The recipe column count is a capacity. Each run may provide a smaller nonzero
active prefix, while settings, rows, targets, and AST population size remain
runtime counts. Inactive compiled tile columns are zero-filled rather than read
from the caller's allocation.

This topology is useful when a cohort intentionally shares a parameter vector:

- evaluating several related expressions under the same experimental state;
- comparing structural variants with aligned parameter slots;
- sweeping global physical parameters across a family of expressions;
- preserving tile reuse while many ASTs consume the same loaded values.

It is not the natural topology for independently optimizing the constants or
leaf bindings of unrelated population members. In that case, sharing settings
couples candidates that should have independent parameter searches.

## Candidate Topologies

### Shared settings across packed ASTs

This is the existing implementation.

```text
settings[setting][slot]
ASTs[ast]
```

Advantages:

- one set of setting loads is reused by every packed AST;
- one patch island and one settings loop remain simple;
- many settings provide enough independent work to saturate the GPU;
- register and shared-memory requirements are predictable.

Disadvantages:

- unrelated ASTs cannot carry independent optimized constants;
- crossover and mutation must preserve shared slot meanings to benefit fully;
- a large population may need artificial grouping by parameter schema.

### One AST per kernel

Each kernel owns one AST and one independent settings table.

```text
settings[ast][setting][slot]
kernel[ast]
```

Advantages:

- direct and unambiguous constant-optimization contract;
- one set of AST accumulators minimizes register pressure;
- no need to split a packed SASS island into independently parameterized
  sections.

Disadvantages:

- fewer ASTs share each module launch and loaded row tile;
- more kernels and modules increase launch and module-management pressure;
- small setting counts may not provide enough work per kernel;
- input tiles are reloaded independently for every AST kernel.

This is the cleanest reference topology and should be benchmarked before a
more complicated packed design is built.

### Packed ASTs with AST-specific settings

The most general runtime layout is:

```text
dynamic constants: [ast][constant][setting]
dynamic leaves:    [setting][ast][leaf]
output:            [ast][target][setting]
```

This is close to the fused-SINDy feature-cohort model, where every feature has
its own leaf bindings for each setting. It preserves kernel and tile packing
without coupling candidate parameters.

The current patch ABI cannot adopt this layout as a superficial stride change.
All packed ASTs currently consume one shared set of input registers. Supporting
AST-specific values requires one of the following:

1. Reserve separate setting registers for every AST. This is simple but scales
   register pressure as `asts_per_kernel * slots`.
2. Split the patch island into per-AST sections and reload a reusable setting
   register bank before each AST. This keeps register pressure bounded but
   makes the patch ABI and branch layout more complex.
3. Let the SASS patcher emit fixed setting-load prologues between AST programs.
   This retains one physical island, but the inspector must expose enough
   address and register state for the patcher to encode those loads safely.

The second or third option is likely the useful packed implementation. It lets
the row tile remain resident while ASTs are evaluated sequentially with their
own settings, and it avoids keeping every AST's setting registers live at once.

## Affine Statistics

Both dynamic shapes can use affine statistics instead of raw SSE. For each
`[ast][target][setting]`, the kernel needs prediction sum, prediction square
sum, and prediction-target cross sum. Dataset-level target moments remain
separate and reusable.

Dynamic affine statistics would provide:

- scale- and offset-invariant ranking for every runtime setting;
- raw SSE reconstruction without rerunning the AST;
- fitted slope and intercept for constant optimization;
- a better screening score before materializing selected candidates.

The arithmetic is useful under either setting topology. The topology question
should be resolved first, because adding dynamic affine variants to the current
shared-setting contract would duplicate an API that may soon be superseded.

## Recommendation

Keep the existing shared-setting shapes stable while collecting measurements.
Do not silently reinterpret their layouts.

The next experiments should compare:

1. One AST per kernel with 256-4096 independent settings.
2. Four or eight ASTs per kernel with AST-specific settings.
3. The current shared-setting topology at the same total AST-setting count.
4. Raw SSE versus affine statistics for the winning topology.

Measure row evaluations per second, AST-setting evaluations per second,
register count, occupancy, module size, specialization speed, and module-load
cost. If AST-specific settings win, add them as a new shape rather than changing
the meaning of the existing descriptors.

For a future dynamic Gram kernel, AST-specific leaf bindings are the appropriate
default: one setting describes a complete cohort, but each feature in that
cohort owns its own leaf words.
