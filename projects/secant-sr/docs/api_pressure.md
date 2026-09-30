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

# Secant API Pressure Findings

The prototype intentionally uses public Secant APIs without adding a parallel
compiler implementation. The following constraints appeared in real use.

## Working cleanly

- Return-terminated variable-length bytecode is directly usable as arena data.
- Sidecar metadata can be derived in one pass without changing Secant's AST.
- `secant_cpu_run_sse()` is a useful independent oracle.
- `secant_cubin_runner_run_sse()` accepts the search's AST pointer array through a typed run descriptor
  directly. No serialization or lowering copy is required.
- `secant_sr_search_scores_set()` makes the search evaluator-independent.

## Constraints

### Partial module batches

The CUBIN runner accepts a partial final module. It specializes and launches
only the supplied ASTs, so population management does not need to pad a final
batch with inert programs or allocate output for those unused slots.

### Template preparation

The demo generates and compiles CUDA skeleton source at startup. A 64-kernel
template with 4,096 patch instructions per kernel took about 14.6 seconds to
prepare on a cold local run. A cold 1,536-instruction variant took about 7.7
seconds end to end, while repeating that exact recipe completed the process in
under 0.5 seconds. NVRTC and driver cache state therefore materially changes
template preparation. This does not measure Secant specialization, but it is
too large and too variable to include in a generation hot path. Templates
should be cached explicitly by compute capability and recipe, or embedded as
build artifacts.

The default demo island is now 1,536 instructions for 32 ASTs per kernel. Patch
capacity must still be selected from the admitted maximum AST shape and tested
under stress; silently truncating it is not acceptable.

### Population storage

The search measure reserves `population_size * max_program_bytes` and
`population_size * max_nodes` for each of two arenas. This guarantees that a
generation cannot fragment or allocate, but average programs are much smaller
than the cap. Separate total arena budgets would improve population density.

### Routine generation

Secant evaluators accept routine tables, but the first search generator only
emits native f32 unary and binary operations. Adding protected operations
requires routine index, arity, and complexity metadata in the search policy;
it should not duplicate routine bodies or backend lowering.

### Multiple targets and objectives

Secant already returns `[ast][target]` SSE, while the first search API accepts
one SSE per AST. Multi-target campaigns need an explicit aggregation policy or
vector fitness rather than silently flattening targets.

## Representation conclusion

No substantial divergent system was required. Population mechanics can remain
outside Secant. The strongest candidate for a new Secant-facing shape is
runtime-bound leaves, followed by a 32-feature Gram builder. Both can consume
the existing postorder bytecode while changing how input tokens and epilogues
are interpreted by a recipe.
