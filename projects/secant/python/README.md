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

# Secant Python Prototype

This package exercises the complete direct-CUBIN path from Python:

1. Build a typed expression tree and encode it as Secant's variable-length postorder bytecode.
2. Generate a CUDA skeleton through the native Secant API.
3. Compile the skeleton to a CUBIN with `cuda.core`.
4. Inspect and specialize a caller-owned CUBIN copy through nanobind.
5. Load and launch the specialized kernels with `cuda.core`.
6. Compare GPU output against an independent NumPy interpreter.

The native extension owns immutable inspection-plan storage. Direct module wrappers leave specialized CUBINs, modules, device
allocations, and streams in Python. `SSEBulkRunner` instead owns Secant's persistent native worker and stream pipeline while
Python continues to own the device input and output buffers.

## Build

Create the local sandbox and build the extension:

```sh
cd /home/cdurham/code/secant
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python -r python/requirements.txt

cmake -S . -B build-python \
    -DCMAKE_BUILD_TYPE=Release \
    -DSECANT_BUILD_BENCHMARKS=OFF \
    -DSECANT_BUILD_PYTHON_BINDINGS=ON \
    -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-python -j 24
```

Run the Python tests:

```sh
CUDA_MODULE_LOADING=EAGER \
PYTHONPATH="$PWD/build-python/python:$PWD/python" \
    .venv/bin/python -m unittest discover -s python/tests -v
```

Run the standalone materialize example:

```sh
CUDA_MODULE_LOADING=EAGER \
PYTHONPATH="$PWD/build-python/python:$PWD/python" \
    .venv/bin/python python/example.py
```

Run the resident SSE benchmark with the RTX 5090 comparison shape:

```sh
CUDA_MODULE_LOADING=EAGER \
PYTHONPATH="$PWD/build-python/python:$PWD/python" \
    .venv/bin/python python/benchmark_sse.py
```

The benchmark uploads its inputs once, distributes kernels over eight CUDA streams, and uses CUDA events to report the GPU
makespan separately from resident host launch and output-copy time. Its default shape is 32 kernels with 128 ASTs each, one
target, 8192 rows per tile, and 1,048,576 rows. ALU and MUFU programs come from the same deterministic 1,024-expression portable
corpora used by the C benchmark suite.

Run a large packed AST byte array through the native worker pipeline:

```sh
CUDA_MODULE_LOADING=EAGER \
PYTHONPATH="$PWD/build-python/python:$PWD/python" \
    .venv/bin/python python/benchmark_runner_sse.py
```

The default run generates 524,288 structurally unique balanced ASTs by independently selecting each leaf input, unary
operation, and binary operation from a deterministic seeded permutation. It packs the variable-length programs into one
`bytearray` plus a `uint64` offset array. Nanobind passes NumPy views over those allocations to
the generic `secant_cubin_runner_run()` API through a typed SSE descriptor; the native runner specializes modules on 24 workers
while loading and executing completed
modules on eight CUDA streams. Pass `--ast-source portable` to repeat the fixed 1,024-expression corpus used for comparisons
with other evaluators.

## Example

```python
import numpy as np
import secant
from secant.routines import DEFAULT_ROUTINES

x0 = secant.input(0)
x1 = secant.input(1)
programs = (
    secant.Program(secant.sin(x0) * secant.cos(x1)),
    secant.Program(secant.safe_div(x0, x1)),
)

recipe = secant.MaterializeRecipe(
    num_kernels=1,
    asts_per_kernel=2,
    num_inputs=2,
    patch_capacity_instructions=64,
)
template = secant.compile_materialize(recipe)
specialized_cubin = template.specialize(programs, routines=DEFAULT_ROUTINES)

columns = np.random.default_rng(7).uniform(0.5, 3.0, size=(2, 4096)).astype(np.float32)
gpu = secant.MaterializeModule(specialized_cubin, recipe).run(columns)
cpu = secant.materialize(programs, columns, routines=DEFAULT_ROUTINES)
np.testing.assert_allclose(gpu, cpu, rtol=3.0e-4, atol=3.0e-4)
```

Expressions use normal Python arithmetic. Functions such as `sin`, `cos`, `sqrt`, `fma`, `minimum`, and `maximum` create AST
nodes. `Program` lowers a tree to the exact byte representation accepted by the C API. `secant.materialize`, `secant.sse`,
`secant.affine_stats`, `secant.gram_stats`, `secant.dynamic_constant_sse`, and `secant.dynamic_leaf_sse` execute that bytecode
with NumPy and preserve the corresponding Secant output layouts. Every shape has a matching recipe and direct GPU module
wrapper. `DynamicLeafSSERecipe.num_static_input_columns=0` selects dynamic-only leaves; setting it equal to
`num_input_columns` enables mixed fixed-column and dynamic column-or-constant leaves.

`patch_capacity_instructions` is the total size of the one shared patch island in each kernel, measured in 16-byte SASS
instructions. The C and Python recipe APIs and `secant_generate_cubin --patch-instructions` use this total directly.
