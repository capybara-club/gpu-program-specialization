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

# C API Overview

The public C API is intended for runtimes that own a CUDA context and want to
use the FusedSINDy engine without including PyTorch, nanobind, or CUDA headers
from the public header.

The public header is:

```c
#include "implicit_sindy.h"
```

The C symbols still use the historical `implicit_sindy_*` prefix. That is an ABI
stability choice, not a project-name recommendation.

## Design Principles

- C99-compatible public header.
- Explicit result-code enums.
- Explicit shape and stride arguments.
- Raw device pointers passed by the caller.
- CUDA Driver objects passed as opaque `void*` handles.
- Caller-provided worker scratch memory where practical.
- Final cubins are library-allocated because their sizes are known only after
  compilation; free them with `implicit_sindy_free_cubins`.
- No PyTorch, nanobind, or CUDA headers are required by the public header.

## Main Stages

```text
1. Create AST compiler handle.
2. Compile BinaryAST cohorts to cubins on CPU worker threads.
3. Load cubins into CUDA Driver modules.
4. Create Gram or column module wrappers.
5. Launch Gram/column kernels on caller-owned streams.
6. Solve ridge/STLSQ and score validation MSE.
7. Unload modules, destroy handles, free cubins.
```

## AST Compilation

Conceptual flow:

```c
ImplicitSindyAstCompiler* compiler = NULL;

implicit_sindy_ast_compiler_create_with_gram_ptx(
    kernels_per_module,
    sm_major,
    sm_minor,
    gram_template_ptx,
    gram_template_ptx_bytes,
    nvptx_options,
    num_nvptx_options,
    &compiler
);

implicit_sindy_ast_compile_workspace_size(
    compiler,
    num_asts,
    worker_count,
    scratch_bytes_per_worker,
    &workspace_bytes
);

implicit_sindy_ast_compile_cubins(
    compiler,
    asts,
    num_asts,
    sizeof(asts[0]),
    worker_count,
    workspace,
    workspace_bytes,
    out_cubins,
    out_cubin_sizes
);
```

The worker memory is caller-owned. The final cubins are allocated by the library
and must be freed with `implicit_sindy_free_cubins`.

The Python path creates the Gram template PTX with `cuda.core`, then passes it
into the C compiler handle. A pure C integration can use the constructor that
builds the template through the C/CUDA build path.

## Module Loading

```c
implicit_sindy_load_cubin_modules(cubins, cubin_sizes, num_cubins, modules);
implicit_sindy_unload_modules(modules, num_modules);
```

The caller must make the intended CUDA context current before loading modules.
Streams passed to launch functions must belong to that context.

## Gram Launch

The Gram launch consumes one cohort and many runtime settings:

```c
implicit_sindy_gram_module_create(module, kernels_per_module, &gram_module);
implicit_sindy_gram_launch(...);
implicit_sindy_gram_module_destroy(gram_module);
```

The current fixed shape is:

```text
32 generated features per cohort
up to 32 primitive feature columns
up to 32 target RHS columns
leaf_masks: [settings, 32]
leaf_words: [settings, 32, 8]
```

The Gram kernel writes raw `X^T X`, `X^T y`, `sum(X)`, `sum(y)`, and `sum(y^2)`
statistics. It does not normalize or solve. The solve kernel applies centering
and scaling from these raw stats.

## Column Launches

For debugging or explicit materialization, the C API also exposes column
kernels:

```c
implicit_sindy_columns_launch(...);
implicit_sindy_ast_column_launch(...);
```

The full-column path writes all 32 generated features for each setting. The
single-AST path writes one AST column at a time. Tests use these paths to verify
the fused Gram kernel against a separate reference.

## Solve And Score

```c
implicit_feature_ridge_solve_create(&solver);
implicit_feature_ridge_solve_posv_sweep(...);
implicit_feature_ridge_solve_stlsq_sweep(...);
implicit_feature_ridge_score_validation_mse(...);
implicit_feature_ridge_solve_destroy(solver);
```

The solver operates on raw Gram/stat outputs. STLSQ is cheap enough in the
current 32x32 shape that it is intended for use during search, not only as a
final cleanup pass.

## Ownership Summary

| Object | Allocator | Owner | Free path |
| --- | --- | --- | --- |
| Worker scratch | caller | caller | caller frees |
| AST compiler handle | library | caller | destroy function |
| Cubin buffers | library | caller | `implicit_sindy_free_cubins` |
| CUDA modules | CUDA Driver/library helper | caller | `implicit_sindy_unload_modules` |
| Gram/stat tensors | caller/Python | caller | caller frees |
| Solver handle | library | caller | destroy function |

## Threading

The bulk compiler uses worker threads internally. The hot compile call divides
the AST/cubin range across workers and does not use a mutex per job. Each worker
uses its assigned slice of caller-provided scratch. Final cubins are allocated
because the final size is not known until compilation completes.

