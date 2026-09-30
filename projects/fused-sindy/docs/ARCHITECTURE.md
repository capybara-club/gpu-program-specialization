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

# Architecture

FusedSINDy is organized around one performance idea: compile symbolic feature
structure into CUDA code, evaluate many runtime leaf settings, and accumulate
Gram/statistics directly without materializing generated feature columns.

## Layer Model

```text
Layer 1: fixed-shape ASTs and runtime settings
Layer 2: AST lowering to Stack PTX and evaluator PTX
Layer 3: nvPTXCompiler plus cubin-function patching
Layer 4: fused Gram/stat kernels
Layer 5: ridge/STLSQ solve and validation MSE scoring
Layer 6: Python examples/search controllers
```

The lower layers are the engine. The examples are starting points for search
controllers, not the only intended search strategy.

## GPU Code Terms

A PySINDy user does not need to write PTX, but the terms explain the structure:

```text
CUDA source:
    C/C++-like GPU source code.

PTX:
    NVIDIA's portable intermediate GPU assembly. It is not final machine code.

SASS:
    Final machine instructions for a specific NVIDIA GPU architecture.

cubin:
    A binary container holding compiled GPU code and metadata.

CUDA module:
    A cubin loaded into a CUDA context by the driver API.
```

FusedSINDy generates and compiles code while the search program is running
because each new AST cohort creates a new evaluator.

## Data Model

The user provides primitive features and targets:

```text
primitive_features: float32 [primitive_columns, rows]
targets:            float32 [rhs, rows]
```

Primitive columns are row-contiguous. Typical PDE columns are `u`, `u_x`,
`u_xx`, `u_xxx`, `v`, and so on. FusedSINDy does not estimate derivatives for
you; derivative/stencil quality remains part of the scientific workflow.

A cohort contains 32 generated features. Each generated feature is a fixed
depth-3 binary AST with 8 leaves. The AST stores operation structure only.
Runtime settings choose primitive-column or constant bindings for leaves:

```text
leaf_masks: int32 [settings, 32]
leaf_words: int32 [settings, 32, 8]
```

For each leaf, a mask bit means "interpret the word as a primitive column
index"; a clear bit means "interpret the word as raw f32 constant bits".

## Compilation Path

The old full-inline strategy generated one large Gram kernel per cohort:

```text
load primitive tile
evaluate these exact 32 ASTs inline
accumulate Gram/statistics
```

That shape gives the best runtime kernel, but every new cohort forces the
compiler to rebuild the full Gram kernel.

The current strategy keeps the Gram kernel stable and replaces only the
cohort-specific evaluator:

```text
Cold path:
  compile/link a reserved Gram template with a maximal evaluator slot

Hot path:
  BinaryAST cohort
    -> Stack PTX evaluator program
    -> evaluator PTX
    -> nvPTXCompiler relocatable cubin
    -> cubin_function_patch copies evaluator text into reserved Gram cubin
    -> CUDA Driver module load
```

The patch path gives up some runtime throughput because the evaluator is an
uninlined device function, but it greatly reduces structural compile latency.
That tradeoff is useful when search mutates cohorts frequently.

## Stack PTX

Stack PTX is a project-local expression IR, not an NVIDIA standard. It is a
small stack-machine representation that lets the AST code emit operations
without generating full CUDA source:

```text
AST expression tree
    -> Stack PTX program
    -> injected PTX evaluator function
    -> compiled GPU function
```

Keeping this IR narrow makes it easier for C99 and LLM-generated code to produce
valid evaluator programs.

## Gram Kernel

For each runtime setting, a CTA loops over the rows in 128-row tiles:

```text
load primitive tile into shared memory
configure leaf pointers and strides
evaluate 32 generated features
accumulate X^T X, X^T y, sum(X), sum(y), and sum(y^2)
```

The generated 32-column feature matrix is never written to global memory. The
kernel writes only compact regression statistics for each setting.

## Solve And Score

The solver consumes raw Gram/stat outputs. It applies centering/scaling from
the accumulated statistics, then runs dense ridge or STLSQ sweeps over 32x32
systems. Validation or test data is scored by launching the Gram kernel on a
separate dataset and passing those validation stats to the MSE kernel.

The current core path is:

```text
train gram -> ridge/STLSQ solve -> validation gram -> validation MSE
```

There is no internal k-fold aggregation in the current API. Callers that need
k-fold behavior can run the same path for each explicit split.

## Python Binding Strategy

Python owns tensors, validates layout, and passes raw device pointers and stream
handles into the native C API. The native extension is built with nanobind but
does not need PyTorch headers for the core C API.

The preferred public import is:

```python
import fused_sindy as fsindy
```

The implementation namespace remains `implicit_sindy`, and the C symbols remain
`implicit_sindy_*` for ABI continuity.

## What This Optimizes

Optimized:

- large row counts;
- many settings per compiled cohort;
- avoiding generated feature materialization;
- changing AST structures during search;
- dense 32x32 ridge/STLSQ scoring.

Not optimized:

- tiny problems where a materialized library is cheap;
- arbitrary symbolic expression trees;
- non-NVIDIA accelerators;
- derivative estimation, denoising, plotting, and estimator ergonomics.

