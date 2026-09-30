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

# Installation

FusedSINDy is currently tested as a Linux/NVIDIA/PyTorch CUDA package. It is not
a CPU-only or portable accelerator package.

## Requirements

Tested target path:

```text
Linux
NVIDIA GPU
NVIDIA driver recent enough for the target GPU
PyTorch CUDA wheel
CUDA Python/JIT packages pinned by this project
CMake and a C/C++ compiler
```

Unsupported targets:

```text
CPU-only
AMD/ROCm
Apple Metal/MPS
generic OpenCL/SYCL backends
```

## Python Install

Create a clean environment, install PyTorch first, then install this repo:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
python -m pip install /path/to/fused-sindy
```

Import with:

```python
import fused_sindy as fsindy
```

The historical implementation import still works:

```python
import implicit_sindy
```

## CUDA Python/JIT Packages

The base package pins the runtime JIT packages used by the Python path:

```text
cuda-core==1.0.1
nvidia-cuda-nvrtc==13.2.78
nvidia-nvjitlink==13.2.78
```

The build-system dependencies pin the matching native build components:

```text
nvidia-cuda-runtime==13.2.75
nvidia-nvptxcompiler==13.2.78
```

PyTorch CUDA wheels normally install a matching `cuda-bindings` major. If you
install without PyTorch and want CUDA Python to choose a bindings stack, use:

```bash
python -m pip install '/path/to/fused-sindy[cu13]'
python -m pip install '/path/to/fused-sindy[cu12]'
```

Normal `pip install` build isolation installs build dependencies. If build
isolation is disabled, install `nvidia-nvptxcompiler==13.2.78` before building.

## Why Pins Matter

This package dynamically compiles and loads GPU code. The CMake build is strict
about mixing Python CUDA JIT libraries with a system nvPTXCompiler from a
different CUDA toolkit, because mismatched compiler/linker components can
produce cubins that fail at module load time or behave inconsistently.

Treat the CUDA JIT packages as part of the runtime ABI.

## Smoke Test

Run from the repository root:

```bash
tools/smoke_pip_install.sh
```

The smoke test:

1. creates a fresh virtual environment;
2. installs the expected PyTorch CUDA wheel;
3. installs this repo;
4. verifies the installed `fused_sindy` and `implicit_sindy` packages;
5. compiles a tiny relocatable CUDA cubin through `cuda.core`;
6. runs the PyTorch Gram test;
7. checks that the installed extension contains a CUDA 13 nvPTXCompiler string.

Useful variables:

```bash
PYTHON_BIN=python3.14 \
FUSED_SINDY_SMOKE_VENV=/tmp/fused-sindy-smoke \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 \
tools/smoke_pip_install.sh
```

The older `IMPLICIT_SINDY_SMOKE_VENV` variable is still accepted as a fallback.

## Native Development Build

For C/CUDA tests without Python bindings:

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DIMPLICIT_SINDY_BUILD_TESTS=ON \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DIMPLICIT_SINDY_BUILD_SOLVER=ON \
    -DIMPLICIT_SINDY_SOLVER_USE_NVRTC=ON

cmake --build build -j
ctest --test-dir build --output-on-failure
```

The CMake options still use the historical `IMPLICIT_SINDY` prefix.

## Troubleshooting

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| Build finds an unexpected system CUDA toolkit | CMake discovered system CUDA components before pinned Python packages | Use build isolation and check CMake logs. |
| Module load fails after compile succeeds | Driver/toolkit/target mismatch | Check driver version, `sm_*` target, and CUDA package versions. |
| Runtime compilation fails from Python | Missing or mismatched CUDA Python/JIT dependency | Reinstall in a clean env and run the smoke test. |
| Tensor validation fails | Shape, dtype, device, or stride mismatch | Use float32 CUDA tensors with row stride 1. |
| Search is slower than expected | Too few rows/settings per compiled cohort | Increase settings per cohort and compare against a materialized baseline. |

