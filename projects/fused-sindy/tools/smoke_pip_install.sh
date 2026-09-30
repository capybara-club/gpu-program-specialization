#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
venv_dir="${FUSED_SINDY_SMOKE_VENV:-${IMPLICIT_SINDY_SMOKE_VENV:-/tmp/fused-sindy-smoke}}"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu132}"

rm -rf "${venv_dir}"
"${python_bin}" -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install torch --index-url "${torch_index_url}"
"${venv_dir}/bin/python" -m pip install --no-cache-dir "${repo_root}"

(
    cd /tmp
    "${venv_dir}/bin/python" - <<'PY'
from importlib import metadata, resources

import torch
import fused_sindy
import implicit_sindy
from implicit_sindy._impl import _implicit_sindy as native
import cuda.bindings
import cuda.core

for package in ("torch", "fused-sindy", "cuda-core", "cuda-bindings", "nvidia-cuda-nvrtc", "nvidia-nvjitlink"):
    try:
        print(f"{package} {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package} not installed")
for package in ("nvidia-cuda-nvrtc-cu12", "nvidia-nvjitlink-cu12"):
    try:
        print(f"{package} {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        pass
print(f"torch cuda available {torch.cuda.is_available()}")
print(f"torch cuda version {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"device {torch.cuda.get_device_name(0)}")
print(f"fused_sindy {fused_sindy.__file__}")
print(f"implicit_sindy {implicit_sindy.__file__}")
print(f"native {native.__name__}")
header = resources.files("implicit_sindy") / "cuda" / "implicit_feature_gram.cuh"
print(f"header packaged {header.is_file()} {header}")
PY

    "${venv_dir}/bin/python" - <<'PY'
from cuda.core import Device, Program, ProgramOptions

code = 'extern "C" __global__ void noop(float* x) { if (threadIdx.x == 0) x[0] = 1.0f; }'
dev = Device()
dev.set_current()
options = ProgramOptions(
    name="implicit_sindy_smoke_noop",
    arch=f"sm_{dev.arch}",
    relocatable_device_code=True,
    std="c++17",
)
object_code = Program(code, code_type="c++", options=options).compile("cubin")
print(f"compiled cubin bytes {len(bytes(object_code.code))}")
PY

    "${venv_dir}/bin/python" "${repo_root}/test/python/test_pytorch_gram_nanobind.py"
)

strings -a "${venv_dir}"/lib/python*/site-packages/implicit_sindy/_impl/_implicit_sindy*.so \
    | rg 'Cuda compilation tools, release 13\.' \
    | sort -u
