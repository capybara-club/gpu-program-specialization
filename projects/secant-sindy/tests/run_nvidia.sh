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

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 <cuda-architecture> <mathdx-root> [build-directory]" >&2
    echo "example: $0 80 /opt/nvidia/mathdx/26.03 build-sm80" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
architecture=$1
mathdx_root=$2
build_dir=${3:-"${repo_root}/build-sm${architecture}"}
python=${SECANT_SINDY_PYTHON_EXECUTABLE:-"${repo_root}/.venv/bin/python"}

if [[ ! $architecture =~ ^[0-9]+$ ]]; then
    echo "error: CUDA architecture must contain digits only, such as 80, 86, 90, or 120" >&2
    exit 2
fi
if [[ ! -x $python ]] || ! "$python" -c 'import numpy' >/dev/null 2>&1; then
    echo "error: the NumPy test sandbox is missing" >&2
    echo "run: cd '${repo_root}' && uv venv .venv && uv pip install --python .venv/bin/python -r tests/requirements.txt" >&2
    exit 2
fi

cmake -S "$repo_root" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSECANT_SINDY_BUILD_BENCHMARKS=ON \
    -DSECANT_SINDY_CUDA_ARCHITECTURE="$architecture" \
    -DSECANT_SINDY_MATHDX_ROOT="$mathdx_root" \
    -DSECANT_SINDY_PYTHON_EXECUTABLE="$python"
cmake --build "$build_dir" -j "$(nproc)"
ctest --test-dir "$build_dir" --output-on-failure

fatbin="${build_dir}/generated_cuda_bench/secant_sindy_bench.fatbin"
CUDA_MODULE_LOADING=EAGER "$build_dir/secant_sindy_solver_bench" "$fatbin" ridge
CUDA_MODULE_LOADING=EAGER "$build_dir/secant_sindy_solver_bench" "$fatbin" stlsq
