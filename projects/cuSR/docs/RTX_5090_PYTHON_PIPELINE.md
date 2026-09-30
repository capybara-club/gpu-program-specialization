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

# RTX 5090 Python Pipeline

Measured on 2026-07-13 using:

- NVIDIA GeForce RTX 5090, driver 595.71.05
- CUDA Toolkit 13.1 (`nvcc` 13.1.115)
- AMD Ryzen 9 9900X
- Python cuda.core template compilation, Python ELF inspection, nanobind C99
  AST-to-SASS patching, cuda.core module loading, and PyTorch CUDA storage

Command:

```sh
PYTHONPATH=build_python_native/python:python \
  .venv/bin/python python/bench/benchmark_pipeline.py
```

The MSE module contained 128 kernels with 32 ASTs per kernel. Each kernel ran
4096 settings over 4096 rows and 32 columns. Every random ALU AST reduced all
eight inputs through seven independently selected add, multiply, minimum, or
maximum nodes. ASTs, settings, tensors, streams, and events were prepared
before their relevant timed sections.

| Phase | Time | Rate | Per AST |
|---|---:|---:|---:|
| NVRTC CUDA-to-cubin, excluded from hot path | 15,861.926 ms | 258.2 AST/s | 3,872.54 us |
| Python cubin inspection, excluded from hot path | 179.331 ms | 22,840 AST/s | 43.782 us |
| In-place AST-to-SASS patch, median | 0.529 ms | 7,738,508 AST/s | 0.129 us |
| In-place AST-to-SASS patch, best | 0.525 ms | 7,809,312 AST/s | 0.128 us |
| Module load and 128 kernel lookups | 1.424 ms | 2,876,639 AST/s | 0.348 us |

The cubin was 5.850 MiB. Its output SSE allocation was 0.062 GiB. Patching
mutated the same already-inspected cubin allocation on every iteration; it did
not copy the cubin, rerun inspection, allocate AST scratch storage, or repack
the already-contiguous AST matrix.

An earlier Python result of about 476,000 AST/s was not the native patch hot
path. `patch_cubin_in_place` was calling `pack_programs` on every invocation,
which performed 4096 separate NumPy return-instruction checks and consumed
about 7.05 ms by itself. Passing a prepacked contiguous `uint64` matrix directly
to native validation removes that setup work. A direct Release C benchmark of
the same 4096 depth-3 AST shape measured 7.24M AST/s, consistent with the
corrected Python result.

| Streams | Python submit | Host us/kernel | GPU elapsed | Row evals/s |
|---:|---:|---:|---:|---:|
| 1 | 0.553 ms | 4.323 | 168.170 ms | 4.086e11 |
| 2 | 0.552 ms | 4.314 | 84.152 ms | 8.166e11 |
| 4 | 0.561 ms | 4.387 | 44.427 ms | 1.547e12 |
| 8 | 0.588 ms | 4.596 | 25.515 ms | 2.693e12 |
| 16 | 0.652 ms | 5.091 | 17.932 ms | 3.832e12 |
| 32 | 0.733 ms | 5.726 | 17.995 ms | 3.819e12 |

Sixteen streams were the measured runtime plateau for this shape. Python host
submission remained below 0.7 ms for 128 launches through 16 streams, while
the GPU work took about 18 ms. The settings and row count therefore make
Python launch overhead small relative to execution even though each kernel is
submitted individually.

The benchmark also checked two ASTs across four settings against the NumPy CPU
runner over all 4096 rows. Maximum absolute SSE error was `0.00390625`; maximum
relative error was `1.78e-7`. The direct-value eval family was separately
smoke-tested through the same inspector and patcher with the value epilogue and
matched every checked GPU result exactly for an ALU test shape.
