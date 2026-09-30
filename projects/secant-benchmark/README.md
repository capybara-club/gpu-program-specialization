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

# SECANT Benchmark

> **Collection category:** Historical implementations and measurement. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

Comparison compilers, runtime baselines, benchmark campaigns, papers, and
performance artifacts for [SECANT](../secant).

This repository contains the CUDA C++ and PTX compiler paths used to measure
native compiler throughput and runtime against SECANT's direct CUBIN
specialization. These paths are experimental baselines, not SECANT deployment
backends.

The repository also owns:

- compile, runtime, packing, and end-to-end benchmark executables;
- PySR, EvoGP, Kozax, Operon, and static AVX comparison drivers;
- portable expression corpora shared by those comparisons;
- benchmark reports, generated figures, and source papers.

Configure against a neighboring SECANT checkout:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Use `-DSECANT_SOURCE_DIR=/path/to/secant` when SECANT is elsewhere.

CUDA and PTX runner adapters reuse SECANT's bulk pipeline internals. SECANT
itself does not link or install these comparison compilers.

The suspended HIP/HSACO implementation, integration sources, generated
examples, reports, CSV data, and figures are preserved under [`icebox`](icebox).
Nothing under that directory participates in the active CMake build.
