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

# mm-ptx compile bench

This benchmark compares compile time to cubin for two generator paths:

1. PTX inject: CUDA -> PTX (NVRTC), Stack-PTX stubs injected, PTX -> cubin (nvPTXCompiler)
2. CUDA inline PTX: Stack-PTX stubs embedded as inline PTX, CUDA -> cubin (NVRTC)

The kernel layout matches the app_elites_nle tiling (num kernels, groups per kernel, tile size).
The benchmark does not load cubins. Stack-PTX stubs and per-module sources are precomputed outside the
timed region; timings cover only PTX->cubin and CUDA->cubin compilation. OpenMP is optional via `--cores`.

## Build

From the repo root:

```
cmake -S . -B build -DMMPTX_BUILD_BENCH=ON
cmake --build build --target mm_ptx_compile_bench
```

## Run

```
./build/mm-ptx/bench/mm_ptx_compile_bench --modules 32 --kernels 2 --groups-per-kernel 16 --ptx-instructions-per-program 32
```

Override SM targets with `--sm` (both) or `--sm-ptx`/`--sm-cubin` (e.g., `--sm 80`).

Additional options:

- `--sm-ptx` and `--sm-cubin` let you target NVRTC PTX and nvptxcompiler SMs independently (default `8.0`).
- `--cores` controls OpenMP threads (0 = all cores, default runtime value).
- `--workspace-bytes` controls per-thread scratch arena size (default: auto).
- `--ptx-instructions-per-program` controls stack-ptx instruction count per program (includes return).
- `--dump-ptx-cu` dumps the CUDA source used for the PTX-inject path (alias: `--dump-cu`).
- `--dump-ptx` dumps the base PTX emitted by NVRTC before injection.
- `--dump-module-ptx` / `--dump-module-cu` dump one module's generated sources for inspection.
- `--dump-module-index` selects which module to dump (default: 0).
