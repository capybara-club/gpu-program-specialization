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

# Python Pipeline Benchmark

`benchmark_pipeline.py` measures the Python-driven cuSR stages independently:

- CUDA template compilation and cubin inspection, outside the hot path
- repeated in-place AST-to-SASS patching of one already-inspected cubin
- cubin module loading and kernel lookup
- Python host submission time and GPU elapsed time for one or more streams

The default MSE shape uses 128 kernels, 32 ASTs per kernel, 4096 settings, and
4096 rows. AST programs, settings, CUDA tensors, streams, and events are all
created before their corresponding timed regions.

Build the nanobind extension with `-DCMAKE_BUILD_TYPE=Release`. The benchmark
passes its already-packed contiguous AST matrix directly to the native patch
layout; Python sequence conversion and initial AST packing are setup work and
are not part of the patch interval.

```sh
PYTHONPATH=build_python_native/python:python \
  .venv/bin/python python/bench/benchmark_pipeline.py
```

The direct-evaluation family materializes `[AST, setting, row]`, so it uses
smaller defaults and enforces an output allocation limit:

```sh
PYTHONPATH=build_python_native/python:python \
  .venv/bin/python python/bench/benchmark_pipeline.py \
  --family eval --kernels 8 --asts-per-kernel 8 --settings 16 --rows 1024
```

The default `--streams 1,2,4,8,16,32` sweep compares round-robin kernel
submission across pre-created PyTorch CUDA streams. `host_us_per_kernel` is the
Python-facing submission cost; `gpu_ms` includes required output initialization
for MSE and all kernels in the module.
