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

# cuSR Python

The Python compiler reads the tile-static MSE or direct-evaluation CUDA
template and asks `cuda.core` to instantiate its global kernel template
directly. It does not generate `extern "C"` wrappers. The returned lowered
names belong to the same compilation as the cubin and should be passed to
cubin inspection and module kernel lookup.

Create the local environment for CUDA 13. The nanobind build also needs the
development headers matching the environment's Python interpreter:

```sh
sudo apt install python3.14-dev
python3 -m venv .venv
.venv/bin/python -m pip install 'cuda-core[cu13]' nanobind numpy pyelftools torch
```

Build the native AST/cubin patch module into the build tree:

```sh
cmake -S . -B build_python_native \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUSR_BUILD_PYTHON_BINDINGS=ON \
    -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
    -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build_python_native --target cusr_python_native -j
export PYTHONPATH="$PWD/build_python_native/python:$PWD/python"
```

Compile an explicit instantiation list:

```python
from pathlib import Path

from cusr import TileStaticMseInstantiation, compile_tile_static_mse_cubin

instantiations = (
    TileStaticMseInstantiation(0, ast_capacity=32, tile_rows=64, threads_per_cta=128),
    TileStaticMseInstantiation(1, ast_capacity=32, tile_rows=64, threads_per_cta=128),
)
result = compile_tile_static_mse_cubin(instantiations, arch="sm_120")

Path("build/python/template.cubin").write_bytes(result.cubin)
print(result.lowered_names)
```

Inspect the cubin without loading any C library or invoking the C inspector:

```python
from cusr import inspect_cubin, print_cubin_inspection

inspection = inspect_cubin(
    result.cubin,
    result.lowered_names,
    ast_capacity=32,
    expected_occurrences=1,
)
print_cubin_inspection(inspection)
```

`inspect_cubin` uses `pyelftools` for generic ELF64 sections and symbols, then
applies cuSR's CUDA-specific `.nv.info` and SASS patch-site checks. Its
structured result contains kernel and section names, symbol indices, code and
file spans, register-count records, patch-site spans, marker values,
input/target/output/temp registers, and incoming wait masks.
`print_cubin_inspection` emits that data as JSON.

## Packed ASTs And In-Place Patching

`cusr.ast` defines the same backend-neutral 8-byte instruction ABI as
`src/cusr_ast.h`. Programs are contiguous `numpy.uint64` matrices with one AST
per row and a fixed row stride. Every row must contain `RETURN`. The encoding
does not contain NVIDIA register or SASS details, so another backend can lower
the same packed programs later.

The Python inspector runs once for a skeleton cubin. `prepare_patch_layout`
copies its offsets and register metadata into a reusable native layout. The hot
patch call accepts packed ASTs and mutates a caller-owned `bytearray` directly.
AST generation, register allocation, wait handling, and branch emission are
identical for both kernel families; only the requested epilogue changes:

```python
from cusr import (
    ADD,
    PatchEpilogue,
    RETURN,
    compile_tile_static_mse_cubin,
    encode_input,
    inspect_cubin,
    make_tile_static_mse_instantiations,
    pack_programs,
    patch_cubin_in_place,
    prepare_patch_layout,
)

instantiations = make_tile_static_mse_instantiations(
    1,
    ast_capacity=8,
    tile_rows=64,
    threads_per_cta=128,
)
compiled = compile_tile_static_mse_cubin(instantiations, arch="sm_120")
inspection = inspect_cubin(
    compiled.cubin,
    compiled.lowered_names,
    ast_capacity=8,
    expected_occurrences=1,
)
layout = prepare_patch_layout(inspection)
programs = pack_programs(
    ((encode_input(0), encode_input(1), ADD, RETURN),)
)

patched_cubin = bytearray(compiled.cubin)
stats = patch_cubin_in_place(
    layout,
    programs,
    patched_cubin,
    epilogue=PatchEpilogue.SSE,
)
```

Use `PatchEpilogue.VALUE` with a tile-static eval cubin. It moves each AST
result to the inspected output register instead of appending target subtraction
and SSE accumulation. The eval template then stores those registers as
`[AST, setting, row]` values.

Patching is destructive. On failure, `patched_cubin` may contain a partial
patch. Keep the original skeleton bytes when rollback is required. Layout
construction and inspection are outside the patch hot path.

## NumPy Reference And PyTorch Storage

`evaluate_program`, `evaluate_programs`, `evaluate_values`, and `evaluate_sse`
are independent NumPy implementations of the postfix AST semantics. They are
useful for search validation and do not call the C interpreter or SASS
generator.

`TileStaticMseModule` loads the patched cubin with cuda.core and launches it on
PyTorch-owned CUDA allocations. cuda.core receives non-owning `Buffer` wrappers;
the tensors remain the storage owners and no input or output is copied:

```python
import torch

from cusr import TileStaticMseModule

module = TileStaticMseModule(
    patched_cubin,
    compiled.lowered_names,
    instantiations,
)
output_sse = module(
    x_columns,
    target,
    leaf_masks,
    leaf_words,
    asts_per_kernel=1,
)
```

`x_columns` is contiguous float32 `[column, leading_dim]`. `target` is
contiguous float32, `leaf_masks` is uint8 `[setting]`, and `leaf_words` is
uint32 `[setting, stride]` with at least eight words. A set mask bit interprets
the corresponding word as a column index; a clear bit interprets the same bits
as an f32 constant. The returned contiguous float32 tensor is
`[total_ast, setting]` SSE and is zeroed before launch.

`TileStaticEvalModule` has the same settings contract but does not take a
target. It returns contiguous float32 `[total_ast, setting, row]` values. Both
module classes accept either `stream=` or `streams=`. Multiple streams assign
kernels round-robin, and input/output tensor lifetimes are recorded on every
worker stream.

The standalone eval CUDA implementation is intentionally not factored through
the MSE template. This keeps both kernel shapes readable and lets their memory
and output contracts change independently while the patcher continues to vary
only the epilogue.

The CLI creates a consecutive instantiation list for convenience:

```sh
PYTHONPATH=python .venv/bin/python -m cusr \
    --kernels 8 \
    --ast-capacity 32 \
    --tile-rows 64 \
    --cta-threads 128 \
    --arch sm_120 \
    -o build/python/template.cubin
```

Run the compiler tests with:

```sh
PYTHONPATH=build_python_native/python:python \
    .venv/bin/python -m unittest discover -s python/tests -v
```

Run the Python patch/load/launch benchmark with:

```sh
PYTHONPATH=build_python_native/python:python \
    .venv/bin/python python/bench/benchmark_pipeline.py
```

It reports AST patch rate, module-load cost, Python host submission time per
kernel, GPU elapsed time, and row evaluations per second for pre-created stream
counts. See `python/bench/README.md` for the direct-evaluation variant.

The comparison test compiles two kernels at AST capacities 8, 16, and 32. It
checks the complete Python result against JSON emitted independently by the C99
inspector. The C executable must already exist under `build/` for that
cross-implementation test; otherwise only that comparison is skipped.
`test_native_pipeline.py` additionally checks native/Python instruction ABI,
the NumPy routine interpreter, both patch epilogues, cuda.core module loading,
multi-stream dispatch, and MSE/direct-value execution over PyTorch CUDA
tensors.
