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

# SECANT

> **Collection category:** Current native execution engines. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

Secant specializes compact postorder ASTs directly into NVIDIA CUBIN templates.
Its C99 API includes a CPU reference evaluator, CUDA source generation, CUBIN
inspection and specialization, and a persistent bulk runner. Search strategy,
coefficient proposals, GP, and LM are outside this core.

**Version 0.3 is a breaking API revision.** Leaf settings, masks, and runtime
column-index tables have been removed. Expressions use explicit columns,
coefficient slots, literals, and two-/four-way toggles. The evaluation space is
**ASTs × coefficient banks × toggle permutations**. See [the API guide](docs/toggles.md)
for encoding, layouts, migration, and validation evidence.

## AST and scoring

```c
#include "secant.h"

/* (bit 0 ? column[1] : column[0]) * coefficient[0] */
static const SecantAstInstruction expression[] = {
    secant_ast_encode_column_f32(0),
    secant_ast_encode_column_f32(1),
    secant_ast_encode_toggle2_f32(0),
    secant_ast_encode_bank_constant_f32(0),
    secant_ast_encode_mul_f32,
    secant_ast_encode_return_f32
};
```

A toggle may select columns, coefficient slots, or literals in any combination.
The selected operands must be direct leaves. Reusing a bit couples choices.
Coefficient values load before the row loop; specialized instructions refer to
one shared set of fixed registers for all ASTs in a kernel. Each AST reads the
same permutation bits and keeps its own score accumulator. Columns load from fixed row addresses. Banks and permutations
are independent dimensions, with `configuration = bank * 2^toggle_bits + permutation`.

The active shapes are:

| Shape | Output |
| --- | --- |
| Static materialize | `[ast][row]` predictions |
| Static SSE | `[ast][target]` squared-error sums |
| Static affine statistics | Prediction moments and prediction–target cross moments |
| Static Gram statistics | Feature moments, Gram matrix, and feature–target cross moments |
| Toggle SSE | `[ast][target][configuration]` squared-error sums |

All data is float32 and column-major. SSE is a **sum**; divide by the row count
for MSE. The GPU runner clears active scoring/statistics outputs. The CPU toggle
path also clears its outputs; the older static CPU scoring/statistics paths
accumulate into caller-initialized output.

## Build and test

Host build, including source generation, binary inspection, specialization, and
CPU tests:

```sh
cmake -S . -B build -DSECANT_ENABLE_CUDA_INTEGRATION=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On a CUDA development machine, omit `SECANT_ENABLE_CUDA_INTEGRATION=OFF`.
CUDA tests require NVRTC and an available GPU; the runner requires
`CUDA_MODULE_LOADING=EAGER`. CUDA, streams, buffers, and template compilation are
not needed for the portable CPU evaluator.

`secant_generate_cubin` emits CUDA source for a chosen recipe; compile that source
with NVRTC and inspect the resulting CUBIN. Example:

```sh
./build/secant_generate_cubin toggle_sse --kernels 8 --asts-per-kernel 8 \
  --inputs 6 --constants 4 --targets 1 --tile-rows 128 --threads 128 \
  --patch-instructions 2048 --output scoring.cu
```

The runner retains CPU specialization workers and CUDA streams/events, eagerly
loads each specialized module, executes its work, waits for its completion events,
and unloads it. Multiple datasets can share one specialization/load through
`secant_cubin_runner_run_batch()`. GPU storage is caller-owned. Compilation and
module loading timings are separate from the device execution interval.
Normal completion uses events. Error recovery may synchronize an affected runner
stream; it never synchronizes the whole device. Failed destruction retains the
handle for retry. `SECANT_ERROR_COMPLETION_UNKNOWN` requires preserving submitted
device buffers until destruction succeeds. See the public header and the
[runner hardening report](docs/runner-hardening.md) for ownership and test coverage.

## Boundaries and current limitations

- `secant.h` is the public API; shared implementation headers remain private.
- The native input is trusted, return-terminated AST storage. Programs do not
  carry byte lengths; this is not a parser for untrusted network bytecode.
- Up to 1024 instructions/program, 128 stack values, 255 routines, and eight
  nested routine calls. Integer opcodes are reserved; current evaluators use f32.
- A toggle request evaluates its full Cartesian product. There is no sparse
  permutation list or configuration-range API yet. Output needs
  `ASTs * targets * banks * 2^toggle_bits` float32 values, plus caller padding.
- Packed toggle recipes use 1–32 ASTs/kernel. The sum of column registers and
  shared coefficient registers cannot exceed 128; input/target/output markers
  are capped at 192. Shared trajectory tiles are capped at 48,000 bytes.
  Actual register pressure or patch capacity can still reject a specialization.
- Unknown compiled scaffold instructions, including unexpected spill code,
  are rejected. Compiler local memory outside the patch island can still occur;
  the 32-AST resource test reports 16 local bytes/thread. There is no silent
  interpreter or recompilation fallback.
- The runner waits between loaded modules. Cross-module GPU stage overlap is a
  separate future optimization; CPU specialization already runs ahead.
- Legacy Python bindings, optimizer/LM adapters, and comparison backends
  still require migration. Their build options fail with an explicit migration
  diagnostic. Retired settings implementations and their tests are in
  [icebox/settings-v01](icebox/settings-v01/README.md); they are not compatibility code.
  The new [Secant-SR](../secant-sr/README.md) supports native toggles and optional
  random coefficient refinement through affine bank leaves.

The native generator/inspector recognizes supported SM 8.x, 9.0, 10.0, and 12.0
encodings. The current toggle and static execution tests passed on ada's RTX 4090
(SM 8.9) and both rack1 RTX 5080s (SM 12.0), using CUDA 13.1. Other architectures
and compiler versions have not been validated by this test round.
