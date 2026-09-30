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

# CUDA 12 Build Notes

`fused-sindy` has two CUDA 12 paths:

- PyTorch cu128 can be used from Python without replacing PyTorch's
  `cuda-bindings<13` package. Install PyTorch first, then install
  `fused-sindy`; the package's base dependencies intentionally do not pin a
  CUDA bindings major.
- A pure CUDA 12 native build still requires a full CUDA toolkit, `nvcc`,
  `libnvptxcompiler`, and a CUDA-12-compatible host compiler.

CUDA 12.9 works with MathDx/cuSolverDx `25.06.1`, which the CMake build can
download automatically. CUDA 12.9 does not support very new GCC releases such
as GCC 15/16 as the CUDA host compiler. Use GCC 14 or older.

Example native build for a GH200/H100 target:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip build scikit-build-core nanobind cmake ninja numpy

python -m pip wheel . -w dist --no-build-isolation \
  -Ccmake.args=-DCMAKE_BUILD_TYPE=Release \
  -Ccmake.args=-DBUILD_PYTHON_BINDINGS=ON \
  -Ccmake.args=-DIMPLICIT_SINDY_BUILD_TESTS=OFF \
  -Ccmake.args=-DIMPLICIT_SINDY_BUILD_SOLVER=ON \
  -Ccmake.args=-DCUDAToolkit_ROOT=/path/to/cuda-12.x \
  -Ccmake.args=-DCMAKE_CUDA_ARCHITECTURES=90
```

If the host has GCC newer than CUDA 12 supports, provide an older compiler:

```bash
python -m pip wheel . -w dist --no-build-isolation \
  -Ccmake.args=-DCMAKE_BUILD_TYPE=Release \
  -Ccmake.args=-DBUILD_PYTHON_BINDINGS=ON \
  -Ccmake.args=-DIMPLICIT_SINDY_BUILD_TESTS=OFF \
  -Ccmake.args=-DIMPLICIT_SINDY_BUILD_SOLVER=ON \
  -Ccmake.args=-DCUDAToolkit_ROOT=/path/to/cuda-12.x \
  -Ccmake.args=-DCMAKE_C_COMPILER=/path/to/gcc-14 \
  -Ccmake.args=-DCMAKE_CXX_COMPILER=/path/to/g++-14 \
  -Ccmake.args=-DCMAKE_CUDA_COMPILER=/path/to/cuda-12.x/bin/nvcc \
  -Ccmake.args=-DCMAKE_CUDA_HOST_COMPILER=/path/to/g++-14 \
  -Ccmake.args=-DCMAKE_CUDA_ARCHITECTURES=90
```

For a local RTX 5090 build with CUDA 12.8/12.9, use
`-DCMAKE_CUDA_ARCHITECTURES=120`. PyTorch cu128 has been smoke-tested on
`sm_120`; older CUDA 12 wheels may not support RTX 5090.
