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

# Source Generator Applications

The build creates one source generator per available backend:

| Executable | Output |
| --- | --- |
| `secant_generate_cuda` | CUDA with AST expressions already emitted |
| `secant_generate_ptx` | CUDA containing PTX Inject expression sites |
| `secant_generate_hip` | HIP with AST expressions already emitted |
| `secant_generate_cubin` | CUDA containing direct-SASS patch islands |
| `secant_generate_hsaco` | HIP containing direct-AMDGPU-ISA patch islands |

Every executable accepts the same shape names and source-shape options:

```sh
./build/secant_generate_cubin dynamic_constant_sse \
  --kernels 2 \
  --asts 8 \
  --inputs 4 \
  --constants 4 \
  --targets 2 \
  --tile-rows 128 \
  --threads 256 \
  --reduction workspace \
  --patch-instructions-per-ast 64 \
  -o generated.cu
```

The shapes are `materialize`, `sse`, and `dynamic_constant_sse`. Running a
generator with only the shape uses small, readable defaults. CUDA and HIP
accept `--ast-mode simple|alu|mufu` and `--seed`; PTX, CUBIN, and HSACO emit
reusable templates, so those options do not alter their source.

For `sse` and `dynamic_constant_sse`, `--reduction atomic` writes tile
partials directly into the result. `--reduction workspace` emits result-major
tile partials and an additional named kernel that reduces them into the same
final result layout.

Without `-o`, generated source is written to standard output.

The direct CUBIN and HSACO dynamic-constant templates emit
`#pragma unroll 1` on the row loop. Their binary contract requires exactly one
contiguous patch island per kernel; allowing the frontend to unroll that loop
duplicates the island and makes source-level patch capacity disagree with the
compiled binary.
