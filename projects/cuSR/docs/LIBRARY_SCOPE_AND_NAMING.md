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

# Library Scope And Naming

## Starting Question

> If I generated a library that just exposed kernels of certain shapes that
> consumed postorder ASTs and gave back CUDA kernels or HIP kernels as a
> library for developers, what should I call it?

The follow-up sharpened the scope:

> If I created a library that produces kernels extremely quickly, embeds ASTs
> in different kernel shapes, and supports symbolic regression, fused SINDy,
> SINDy library construction, parameter sweeps, and LLM-directed campaigns,
> would it still be a symbolic regression library? Or is it broader than
> symbolic regression?

The naming constraints then became:

> I do not like KernExpr. I think it should have something like SEC in it for
> Symbolic Expression Compiler. What about gSEC? I like SECANT as well.

## Short Answer

This is broader than a symbolic regression library. The most accurate category
is a **GPU symbolic expression compiler and kernel-specialization runtime**.

The core library does not need to decide how expressions are proposed,
mutated, selected, or assembled into a scientific model. It accepts postorder
symbolic expressions, specializes a chosen GPU kernel shape with those
expressions, and returns a native module that can be loaded and launched.
Symbolic regression is one important client of that capability, not the
definition of the capability itself.

SINDy, fused SINDy, equation-family screening, coefficient sweeps, and
LLM-directed scientific campaigns can all use the same compiler backend. The
search policy and scientific interpretation belong above it.

## System Boundary

The proposed library boundary is:

```text
packed postorder ASTs
        +
kernel shape and runtime-setting schema
        +
prepared NVIDIA or AMD kernel skeleton
        |
        v
symbolic expression compiler and patch runtime
        |
        v
loadable native GPU module plus launch metadata
```

For NVIDIA, the native result is a patched cubin. For AMD, it would be a
patched HSA code object. Calling these results "CUDA kernels" or "HIP kernels"
is understandable, but **native GPU kernel modules** is the more precise public
term because the output is not merely CUDA or HIP source.

The library can expose several kernel shapes without owning the algorithms
that consume them:

| Kernel shape | Result | Example consumers |
|---|---|---|
| Materialize | One expression value per row | Visualization, downstream solvers, feature generation |
| Fused SSE/MSE | One score per AST and setting | Symbolic regression, parameter search, screening |
| Affine score | Slope, intercept, and residual score | Scale-invariant symbolic regression |
| Gram/statistics | Cohort Gram matrix and target statistics | SINDy, ridge, STLSQ, active-set methods |
| Custom reduction | Application-defined sufficient statistics | Scientific campaigns and domain-specific discovery |

## Where Each Responsibility Belongs

| Capability | Proper layer | Specific to symbolic regression? |
|---|---|---:|
| Encode an f32 expression as a postorder AST | Expression ABI | No |
| Lower an AST into NVIDIA SASS or AMD ISA | Compiler backend | No |
| Patch many expressions into prepared kernel shapes | Specialization runtime | No |
| Load native modules and launch kernels in bulk | GPU runtime | No |
| Sweep runtime constants or leaf bindings | Kernel-shape/runtime layer | No |
| Materialize expression values | Kernel library | No |
| Compute fused SSE, MSE, or sufficient statistics | Kernel library | No |
| Build candidate features for SINDy | SINDy application | No |
| Mutate, cross over, rank, and select expressions | Symbolic regression engine | Yes |
| Select a sparse dynamical-system model | SINDy engine | No, but domain-specific |
| Ask an LLM to design and monitor campaigns | Orchestration layer | No |

This separation is useful because it lets the native compiler remain small,
deterministic, and reusable. A Python package can own AST mutation, campaigns,
module caching, launch scheduling, and scientific policy while a compact C99
core owns the performance-sensitive AST-to-machine-code path.

## Is It Still Symbolic Regression?

It is symbolic regression when a client searches expression structure and
uses the resulting scores to evolve or select formulas. It is not inherently
symbolic regression when it only compiles and evaluates supplied expressions.

The distinction is similar to the distinction between a tensor compiler and a
neural-network training framework. The compiler is enabling infrastructure.
The search system determines the scientific method performed with it.

An LLM-directed campaign does not change this boundary. The LLM may propose
families of expressions, parameter ranges, residual targets, or SINDy cohorts,
but the core still performs symbolic expression specialization and execution.
That makes the backend useful beyond any single search policy and avoids
locking the public API to today's cuSR topology.

## Why The Broader Positioning Matters

Calling the core an SR library would undersell the parts that are unusual and
reusable:

- expressions become native GPU instructions without interpreter dispatch;
- specialization is cheap enough to be part of an active search loop;
- kernel shapes can preserve tile locality and fuse scoring or statistics;
- runtime settings can test many bindings or constants against one resident
  data tile;
- the same AST ABI can target NVIDIA and AMD backends;
- higher-level systems can choose evolutionary search, sparse regression,
  residual modeling, exhaustive campaigns, or LLM-directed exploration.

The strongest product description is therefore:

> A compiler and runtime for specializing native GPU kernels with symbolic
> expressions at search-loop latency.

Symbolic regression and SINDy should be shown as flagship applications because
they make the performance benefit concrete. They should not define the lowest
level library contract.

## Name Candidates

| Name | Possible expansion | Strengths | Concerns |
|---|---|---|---|
| `SEC` | Symbolic Expression Compiler | Exact technical category; short | Too generic to search for or own as a package name |
| `gSEC` | GPU Symbolic Expression Compiler | Direct, compact, and clearly broader than SR | Existing uses make the bare command and package name crowded |
| `SECANT` | Symbolic Expression Compiler for Accelerated Native Targets | Memorable, mathematical, backend-neutral, and leaves room for NVIDIA and AMD | The expansion is constructed and the ordinary mathematical word is not unique |
| `cuSEC` | CUDA Symbolic Expression Compiler | Immediately clear for the current NVIDIA implementation | Incorrect as the umbrella name once HIP/AMD is supported |
| `KernExpr` | Kernel Expression Compiler | Describes the mechanism | Less precise, less distinctive, and already rejected as the preferred direction |

## gSEC

`gSEC`, expanded as **GPU Symbolic Expression Compiler**, is technically a very
good description. It is short, easy to say, and makes the compiler boundary
clear. It also fits the existing naming history around cuSR and mm-ptx.

Its main weakness is namespace collision rather than technical meaning:

- Firebird ships a command named [`gsec`](https://www.firebirdsql.org/file/documentation/html/en/firebirddocs/gsec/firebird-gsec.html).
- The [`gsec` name on PyPI](https://pypi.org/project/gsec/) is already occupied.
- [`GSEC`](https://www.giac.org/certifications/security-essentials-gsec) is a
  well-established cybersecurity certification.

Those collisions do not prevent using gSEC as a project name, but they make a
bare `gsec` command, package, and web search less clean. If gSEC is selected,
qualified distribution names would reduce the problem:

```text
project/brand:  gSEC
repository:     gsec-gpu
Python import:  gsec_gpu
C prefix:       gsec_
backends:       gsec_cuda, gsec_hip
```

## SECANT

`SECANT`, expanded as **Symbolic Expression Compiler for Accelerated Native
Targets**, is the stronger umbrella name.

It has three useful properties:

1. It contains SEC and therefore preserves the literal compiler identity.
2. "Accelerated Native Targets" describes direct machine-code backends without
   binding the project to CUDA, HIP, NVIDIA, AMD, or even GPUs forever.
3. Secant is already a mathematical term, so it sounds natural in numerical and
   scientific-computing contexts.

The project could then use a straightforward hierarchy:

```text
SECANT                         native symbolic expression compiler/runtime
SECANT CUDA backend            cubin inspection, SASS generation, patching
SECANT HIP backend             HSA code-object inspection and ISA generation
cuSR                           symbolic-regression system built on SECANT
fused-SINDy                    sparse-discovery system built on SECANT
Python campaign layer          search, caching, orchestration, and analysis
```

`gSEC` can still be used descriptively for the technology: SECANT is a GPU
symbolic expression compiler. It does not need to be the package's literal
name.

## Recommendation

Use **SECANT** for the architecture-neutral public project and describe it as a
**GPU Symbolic Expression Compiler**.

Keep cuSR as an application or reference search system built on SECANT rather
than stretching the cuSR name over HIP, SINDy, and generic expression-kernel
APIs. This yields a clean division:

| Layer | Recommended identity |
|---|---|
| Native compiler and patch runtime | SECANT |
| Technical category | GPU Symbolic Expression Compiler, or gSEC |
| NVIDIA implementation | SECANT CUDA backend |
| AMD implementation | SECANT HIP backend |
| Symbolic regression application | cuSR or a later application name |
| SINDy integration | Separate SECANT consumer/library |

If the compactness of gSEC matters more than namespace cleanliness, it remains
a defensible choice. In that case, use qualified package and executable names
from the beginning. Before publishing either name, perform a proper trademark,
repository, package-registry, and domain search; the discussion here evaluates
technical fit, not legal availability.
