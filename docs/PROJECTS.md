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

# Project map

This is the navigation index for the collection. **Current engine** means the
primary implementation in this snapshot, not a production-support certification.
**Reference** and **historical** components preserve a technique or experiment;
their APIs and dependencies may differ from the current engines.

## Current native execution engines

| Project | Purpose | Start here |
|---|---|---|
| Secant | Row-expression evaluation, native specialization and fused statistics | [Public header](../projects/secant/secant.h), then [toggle semantics](../projects/secant/docs/toggles.md) |
| Odezza | Trajectory scoring and native LM specialization | [Core guide](../projects/odezza/core/README.md), then [public header](../projects/odezza/core/odezza.h) |

For Odezza, choose `core/` for instruction generation and kernel execution,
`frontend/` for prepared requests/grammar production, `runtime/` for job execution
and retention, and `service_trial/` for the service consumer. Optional numerical
and topology experiments are in `research/`; they are not the default scorer.

## Program construction and specialization tools

| Project | Purpose / status | Start here |
|---|---|---|
| MM PTX | Canonical C toolkit: Stack PTX, AST PTX and PTX Inject | [Toolkit README](../projects/mm-ptx/README.md); choose its three technique guides before reading implementation |
| Stack PTX Emit | Reference multi-language emission of related program representations | [README](../projects/stack-ptx-emit/README.md) |
| CUBIN Function Patch | Reusable constrained patcher for compiled device-function bodies | [README and ABI constraints](../projects/cubin-function-patch/README.md) |

PTX emission/injection still invokes a compiler. Function replacement still
compiles the replacement function. The Secant/Odezza native specialization path
is the one that avoids PTX compilation for each supported candidate.

## Bindings and execution infrastructure

| Project | Purpose / status | Start here |
|---|---|---|
| MM PTX Python | Versioned Python bindings with embedded headers | [README](../projects/mm-ptx-py/README.md) |
| Stack PTX compiler workers | Historical local/NNG compilation queue prototype | [API](../projects/mm-stack-ptx-compiler/stack_ptx_compiler.h) |

Embedded headers are deliberately versioned copies. Do not assume that replacing
them with `projects/mm-ptx/` is a compatible update.

## Fused statistics and fitting research

| Project | Purpose / status | Start here |
|---|---|---|
| FusedSINDy | Reference application fusing feature evaluation with Gram/moment accumulation and fitting | [README](../projects/fused-sindy/README.md) |
| Secant-SINDy | Historical raw-statistics-to-ridge/STLSQ solver with CPU oracle and CUDA source generation | [Statistics contract](../projects/secant-sindy/README.md), [API](../projects/secant-sindy/secant_sindy.h) |

Secant-SINDy's standalone numerical reference is distinct from its older Secant
integration tests. Adapting a legacy statistics consumer to a current API is
explicit work, not a reason to alter the core API to impersonate an older one.

## Reference consumers

| Project | Purpose / status | Start here |
|---|---|---|
| Secant-SR | Toggle-aware search/refinement consumer; separate GPU LM implementation | [README](../projects/secant-sr/README.md), then `toggle/` |
| Secant System ID | Historical bridge from row specialization to ODE scoring/fitting | [README](../projects/secant-system-id/README.md) |
| MM Kermac | Secondary application of Stack PTX and PTX Inject to kernel-matrix computation | [Compilation/injection path](../projects/mm-kermac/k_compiler.c), then [API](../projects/mm-kermac/kermac.h) |

These consumers explain how an engine can be used. An integration does not need
to adopt their GP policies, search budgets or application architecture.

MM Kermac shows the PTX technique outside symbolic regression and ODEs. Its
injected PTX still passes through NVIDIA's compiler; it is not direct SASS
specialization. Start at the compiler path rather than the surrounding tensor,
AGOP or visualization code.

## Historical implementations and measurement

| Project | Purpose / status | Start here |
|---|---|---|
| cuSR | Early CUDA-template-to-SASS prototype; settings-era ancestor of Secant | [Kernel contract](../projects/cuSR/README.md), [native lowering](../projects/cuSR/src/cusr_ast_sass.h) |
| Secant Benchmark | Earlier standalone compiler/runtime comparison suite, including archived HIP work | [README](../projects/secant-benchmark/README.md) |
| PTX compilation benchmark (C) | Parallel compilation, generated injection sites, reusable workspaces | [README](../projects/mm-stack-ptx-ptx-inject-bench/README.md) |

The separate Secant Benchmark tree overlaps the benchmarks in current Secant.
It preserves an earlier API state; it is not an interchangeable modern adapter.
See [benchmark boundaries](BENCHMARKS.md) before using any recorded rate.

## Selected Odezza research trials

| Trial | Question isolated | Status |
|---|---|---|
| [AST tools](../projects/odezza/research/ast_tools/README.md) | How to enumerate/sample complete systems and serialize postorder programs efficiently? | Standalone CPU tools and reference tests |
| [CUDA AST specializer](../projects/odezza/research/cuda_ast_specializer/README.md) | What does ordinary generated-CUDA specialization cost? | Compiler-based control; distinct from direct SASS |
| [Cooperative LM](../projects/odezza/research/cooperative_lm/README.md) | How should multiple lanes share one trajectory fit? | Fixed-system CUDA topology control |
| [Plain CUDA LM](../projects/odezza/research/fitting_batch_trial/plain_cuda_lm/README.md) | How do state/parameter counts, subgroup width, spills and occupancy interact? | Full-CUDA control; historical drivers may need integration work |
| [Rosenbrock GPU](../projects/odezza/research/rosenbrock_trial/README.md) | Can an adaptive linearly implicit solver handle difficult trajectories? | Standalone FP64 prototype, not production scoring |
| [Rosenbrock/RK4 CPU](../projects/odezza/research/rosenbrock_cpu_trial/README.md) | What happens to sample MSE and integration work in FP32 versus FP64? | Native CPU reference and precision study |
| [RFM dependencies](../projects/odezza/research/rfm_dependency_trial/README.md) | Can observation-only feature metrics suggest which states affect an RHS? | Separate pilot; relevance scores, not causal guarantees |

See [research coverage](RESEARCH_COVERAGE.md) for omissions and the scope of the
inventory review. See [validation](VALIDATION.md) for checks actually rerun.
