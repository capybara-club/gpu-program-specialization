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

# Architecture and lineage

This map explains the progression of ideas. It is not a claim that the projects
have one compatible API, or that the latest implementation replaces every older
one for every workload.

## The progression

1. **Program representation:** Stack PTX made small programs and routines cheap
   to describe, mutate and emit. AST PTX adds an explicit postorder/arity model.
   Reference interpreters and multiple emitters help check those representations.
2. **Stable scaffold:** PTX Inject made the boundary between a CUDA kernel and
   a generated program explicit through annotated sites and register bindings.
3. **Compilation throughput:** compiler-worker and benchmark projects explored
   batching, parallel nvPTXCompiler calls, and caller-sized memory workspaces.
4. **Avoid repeated linking:** CUBIN Function Patch replaces compatible compiled
   function bodies in linked modules, within strict resource/metadata bounds.
5. **Avoid per-program PTX compilation:** Secant and Odezza specialize supported
   operations directly into inspected native instruction regions.
6. **Amortize each module:** register-backed toggles, coefficient banks, program
   packing, fused statistics and rollout loops provide more work per load.
7. **Make the engine usable:** prepared C99 request data, caches, pools, top-k,
   events and explicit ownership reduce overhead around the kernel.

The original commit IDs and dirty-working-tree status are in
[source-manifest.json](source-manifest.json). Its file-level hashes are more
precise than assigning this entire collection a single original version number.

## Current engine boundaries

| Layer | Secant | Odezza |
|---|---|---|
| Public native interface | `secant.h` | `core/odezza.h` |
| Supported program validation | `s_ast_internal.h`, `s_cpu.c` | `core/o_ast.c`, `core/o_lm_ast.c` |
| Stable CUDA generation | `s_cubin_generate_*.c` | `core/o_generate_scoring_cuda.c`, `core/o_generate_lm_cuda.c` |
| Binary discovery | `s_cubin_inspect.c` | `core/o_inspect_*_cubin.c`, `core/o_elf.c` |
| Native specialization | `s_cubin_specialize.c` | `core/o_specialize_*_cubin.c`, `core/o_sass_assembler.c` |
| Module/launch lifetime | `s_runner_pipeline.c`, `s_cubin_runner.c` | `core/o_scoring_pipeline.c`, `core/o_lm_pipeline.c` |
| Search policy | Secant-SR, outside Secant | Python/search experiments, outside core |
| Prepared user request | Consumer-specific | `frontend/` |
| Scheduling/retention/service | Consumer-specific | `runtime/`, `service_trial/` |

Do not put grammar expansion, family fairness, GP mutation, HTTP, retries,
tenant policy or model-selection heuristics inside the instruction writer.
The native interface should speak in bounded programs, layouts, buffers,
resources, completion and results.

## The three identities to keep separate

- **Template identity:** architecture, kernel shape, fixed work, compiler/codegen
  settings and supported instruction/register capacity.
- **Program identity:** the particular ASTs, literal values, bindings and any
  information baked into specialization.
- **Evaluation identity:** program + toggle assignment + coefficient vector +
  observations/initial conditions + integration settings + scoring contract.

A cache for the first is not a cache for the third. Reusing a template must not
reuse another request's constants, mutable state or result indices.

## Preserved implementation differences

- PTX Inject and Stack PTX produce PTX; Secant's current native path writes
  supported SASS into a template. Do not report both as compiler-free.
- A packed Secant row kernel maintains separate expression accumulators.
  Odezza's candidate-selecting scoring CTA integrates a whole candidate system.
- Odezza's native LM and Secant-SR's GPU LM are separate kernels and evaluators.
- Older settings-based Secant code is archived beside toggle-era code. Its
  adapters are not supported by the current Secant API merely because the files
  remain in the snapshot.
- Embedded header copies preserve the API that a historical consumer used.
  Deduplicating those copies requires differential tests, not just matching names.

## What broke, and what those failures teach

Read the retained [Odezza incidents](../projects/odezza/docs/incidents/) and
[Secant runner hardening notes](../projects/secant/docs/runner-hardening.md).
Relevant classes of failure include unsafe register/predicate assumptions,
insufficient instruction capacity, code/metadata disagreement, asynchronous
resource lifetimes, small underfilled packs, and stale per-job buffer data.

These incidents are useful engineering evidence. The presence of a historical
report does not mean the current source still has that exact defect, and a fix
on one GPU/toolchain does not prove every shape is safe.
