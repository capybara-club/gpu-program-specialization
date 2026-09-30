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

# Settings versus register toggles

See [the complete measured results](report-20260919.md).
The subsequent [selector-reuse experiment](report-selector-reuse-20260919.md)
improved shared-selector fixtures, but is **not the default writer**. The default
again uses transient selections for arbitrary ASTs; see the
[preserved experiment](selector-reuse-experiment/README.md).

This benchmark compares the original Secant 0.1 dynamic-leaf settings kernel
from commit `7c77c3e` with the current Secant 0.3 shared-bank toggle kernel.
It does not compare GP controllers, derivative estimation, or ODE rollouts.

Both builds compile **the same harness** against their respective headers and
libraries. The old checkout is an isolated export from the local Git repository;
no settings compatibility path is added to the current library.

## Matched work and correctness

- 128 AST occurrences, balanced depth-three arithmetic over eight leaves per AST.
  Each AST varies arithmetic operators and leaf ordering. One workload uses
  add/subtract/multiply; the other adds four sine operations per AST.
- Eight input columns, eight coefficient slots, and eight permutation bits.
  Choices include state/state, state/constant, constant/constant, and mixed
  four-way selections. Some bits are deliberately reused across positions.
- Every bank is crossed with all 256 permutations. The settings backend receives
  the explicitly expanded mask/word table for exactly those same choices.
  Both output layouts are `[ast][bank * 256 + permutation]`.
- Build with `--groups 1` for eight common choice slots, or `--groups 4` for
  32 distinct slots spread across four AST groups. The latter rotates bit IDs
  and state/constant indices by group; the underlying states and banks remain
  shared. Both backends compile from the same fixture definition.
- One row evaluation means **one AST, one configuration, one input row**. Rates
  include expression evaluation and squared-error accumulation. SSE is reduced
  over row tiles by the kernels' existing atomics.
- Data repeat a deterministic 64-row pattern. The CPU oracle evaluates those
  unique rows and accounts for their multiplicity; both GPUs still execute
  every requested row. This permits verification of every score without
  spending the benchmark budget on billions of CPU reference evaluations.
- Every score is checked against a common independently written evaluator.
  Single-row runs with targets zero and one additionally reconstruct the signed
  prediction as `(SSE0 - SSE1 + 1) / 2`, checking every AST/configuration.
  The sweep compares **all** returned settings/toggle scores directly.
- The scaled error limit is `2e-5`; failures stop a case and remain visible in
  `failures.json`. No invalid case is admitted into the comparison table.

## Timing and resources

The engine metric uses already specialized, resident kernels in a CUDA graph
with independent kernel nodes. Both backends use the same harness scheduler.
Output clearing precedes the start event and is excluded; kernel arithmetic,
row reads, and atomic SSE reduction are included. Seven samples follow two
warmups, and median/min/max are retained. No graph-capture setup, NVRTC,
specialization, module loading, transfers, or CPU validation are included in
this metric.

The original launch geometries are preserved: settings threads loop over all
settings within each row-tile CTA; toggle configurations also span grid.y.
The native settings baseline is the dynamic-only shape, avoiding unused static
column registers. This measures the actual original native implementation,
not a rewritten settings kernel with a new launch geometry.

A separate metric runs each version's real blocking C99 runner with three
workers, up to eight streams, resident inputs, and one module containing all
ASTs. Three timed calls follow one warmup. This includes clearing, native
specialization, loading, kernel execution, and unloading; it excludes initial
template compilation, runner creation, transfers, and JSON/GP work. Timers are
measured externally around the complete runner call. NVRTC and preparation
costs are reported separately.

Each function reserves `128 * asts_per_kernel` patch instructions in both
backends. Register counts and local bytes come from the **specialized** loaded
functions. Occupancy is CUDA's resource-based active-block limit, not a claim
of measured achieved occupancy. Packing and row counts vary to reveal launch
coverage and amortization effects.

## Running

Use the repository's CUDA installation and existing C compiler/Python. Given
an isolated original checkout and the current source tree:

```sh
python3 build.py --source /path/to/secant-7c77c3e --baseline --output ./settings
python3 build.py --source /path/to/current-secant --output ./toggles
python3 run.py --settings ./settings --toggles ./toggles --output ./results --gpu 1
# Repeat both builds with --groups 4 for the diverse-choice fixture.
```

The sweep alternates execution order and runs one backend at a time on the same
GPU. Raw score binaries, JSON metadata, per-case logs, and `pairs.csv` are kept.
`--quick` runs just the eight-AST arithmetic case. No additional packages are
needed. `compare.c` can also be called directly with positional arguments:

```
packed rows banks asts tile_rows threads transcendental repeats output_prefix
```

## Finding and follow-up

The first matched arithmetic case (8 ASTs/kernel, 8,192 rows, 4,096
configurations/AST, RTX 5080) passed correctness but showed a material execution
regression: roughly 1.30 trillion row evaluations/s with settings versus
0.623 trillion with toggles. Register counts were 55 versus 53 with zero local
bytes. This is a valid paired comparison, not the earlier 165-to-53 register
comparison between two toggle implementations.

The original settings kernel resolves eight shared leaf slots once per row and
reuses them across packed ASTs. The current toggle writer emits each AST's
selectors again, even when another AST just selected the identical leaves.
Each two-way selection emits a predicate test and FSEL; each four-way emits
three such pairs. This fixture therefore adds 28 selection instructions per
AST per row. Both native writers use the same conservative 12-cycle ALU stall;
the complete sweep and disassembly confirm 112 adjacent test/select pairs in
the eight-AST toggle patch. Repeated selection is the main compiler optimization
target, though the timings do not attribute an exact cycle cost to it. The
diverse-choice and short-row fixtures also show meaningful toggle advantages;
see the report for all 32 paired cases.

Do not treat lower register counts as proof of higher throughput. Cross-AST
selected-value reuse was tested and set aside: it did not improve the measured
random GP initial populations. Prioritize transient-register use and instruction
scheduling on arbitrary ASTs, without related-AST packing. Constant-only hoisting
is a possible future experiment, with its extra live registers measured explicitly.
No reduced safety checks are hidden in this benchmark.
