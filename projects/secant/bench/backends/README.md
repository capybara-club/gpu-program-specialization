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

# Comparison Backends

The CUDA C++ and PTX implementations in this directory are benchmark-only
compiler baselines. They validate expression semantics and quantify native
compiler throughput and runtime against SECANT's direct CUBIN specialization.
They are built only when `SECANT_BUILD_BENCHMARKS=ON`.

- [`cuda`](cuda) contains the NVRTC source generator, compiler, runner, public
  benchmark-only headers, and CUDA-specific tests.
- [`ptx`](ptx) contains the PTX Inject generator, nvPTXCompiler path, runner,
  public benchmark-only headers, and PTX-specific tests.
- [`tests`](tests) contains cross-backend GPU tests and shared CUDA test
  support.
- [`thirdparty`](thirdparty) contains the vendored PTX Inject and AST PTX
  dependencies used by the PTX baseline.

Nothing in this directory is linked into SECANT's production CPU or CUBIN
libraries.
