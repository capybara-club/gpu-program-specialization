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

# Benchmarks

The active benchmark surface compares the CUDA C++, PTX, and direct-CUBIN
paths on NVIDIA hardware.

## Executables

| Executable | Purpose |
| --- | --- |
| `secant_compile_bench` | AST-to-binary compile throughput |
| `secant_runtime_bench` | Resident-kernel runtime throughput |
| `secant_pipeline_bench` | Bulk compile, load, launch, and unload pipeline |
| `secant_dynamic_constant_sse_bench` | Dynamic-constant SSE kernel shape |
| `secant_runtime_sweep` | Row-count and AST-packing sweeps |
| `secant_native_avx_bench` | Static native AVX CPU baseline |

Build in Release mode:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The compile, runtime, and pipeline executables accept
`--backend cuda|ptx|cubin`. Use `--help` for each executable's complete
options.

The former HIP/HSACO benchmark implementation, campaign scripts, reports, and
raw results are archived under [`../icebox`](../icebox) and are excluded from
the active CMake build.
