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

# cuSR Apps

These are narrow checks for reusable code in `src`.

## `ast_sass_demo.c`

Builds fixed postorder programs, lowers them with `cusr_ast_sass_generate`, and
prints the generated instruction words. It also exercises register expansion
and error contracts.

```sh
./build/cusr_ast_sass_demo
```

## `ast_sass_patch_demo.c`

Constructs minimal site-inspection metadata over a fake cubin, patches one
AST plus its SSE epilogue, and verifies the updated regcount. It tests the
patcher without requiring CUDA.

```sh
./build/cusr_ast_sass_patch_demo
```

## `ast_sass_cpp_check.cpp`

Runs a complete one-AST patch-site pipeline from C++: copy the embedded
8-kernel cubin, inspect and patch all kernels, launch one tile-static SSE
kernel, and compare the result against `cusr_ast_sass_cpu.h`.

```sh
./build/cusr_ast_sass_cpp_check
```
