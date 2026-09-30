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

# kermac

> **Collection category:** Reference consumers; secondary PTX-injection example. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

For this handoff, start with [k_compiler.c](k_compiler.c): it applies Stack PTX
and PTX Inject to kernel-matrix operations. The public compiler/injection
functions are in [kermac.h](kermac.h). This path compiles injected PTX with
NVIDIA's compiler; it does not use Secant/Odezza's direct SASS specialization.

## Overview

kermac is a CUDA-focused tensor library for launching kernels, building kernel matrices, and running semiring ops. It includes efficient stack allocators for host/device memory and utilities for elementwise ops, RNG, and solver helpers.

## CMake integration

`kermac` expects `mm-ptx` and (optionally) `fmnist-c` to be added first:

```cmake
add_subdirectory(thirdparty/fmnist-c) # needed for kermac_cpu + fmnist tests
add_subdirectory(thirdparty/mm-ptx)
add_subdirectory(thirdparty/kermac)
```

Targets:
- `kermac`: core CUDA library.
- `kermac_cpu`: CPU helpers (links `fmnist_c`).

Dependencies:
- CUDA Toolkit (cublas/cusolver/cutensor/nvrtc/nvptxcompiler).
- `mm-ptx` (stack_ptx/ptx_inject headers + codegen).
- Python3 (used by mm-ptx codegen).

## How fmnist-c is linked

`thirdparty/kermac/cpu/CMakeLists.txt` links `kermac_cpu` against `fmnist_c`. The loader wrappers in `kc_fmnist.c` are used by the fmnist tests (for example, `tests/test_fmnist.cpp`). The core `kermac` library itself does not depend on `fmnist_c`, so you only need it when building `kermac_cpu` or the tests.

## Entry points

- C API: `thirdparty/kermac/kermac.h`
- C++ API: `thirdparty/kermac/kermac.hpp`
