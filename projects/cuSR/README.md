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

# cuSR

> **Collection category:** Historical implementations and measurement. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

cuSR is a standalone CUDA-to-SASS prototype for evaluating many f32 postorder
ASTs on the GPU. It specializes precompiled CUDA kernel skeletons by writing AST
SASS directly into each cubin.

The current pipeline is:

1. Compile handwritten 8-, 32-, and 128-kernel template instantiations to cubin
   with NVCC.
2. Embed those cubins in the host executables with `incbin`.
3. Inspect each kernel's single patch site and register-count records.
4. Lower postorder ASTs to SASS and patch the site in place.
5. Load the cubin and score runtime settings against a shared row tile.

Patching is intentionally destructive. The same cubin allocation can be
patched repeatedly, but an error may leave it partially modified.

## Kernel Contract

Every generated kernel has one logical SASS patch site. The site exposes:

- 8 read-only expression input registers;
- 1 read-only target register;
- one read/write SSE accumulator register per AST capacity;
- one contiguous `brkpt` pad for all ASTs in the kernel.

Marker `FADD` instructions identify those registers. The inspector validates
the complete marker/store/`brkpt` region, combines incoming shared-load wait
masks, finds reusable temporary registers, and records every kernel regcount
field.

The patcher emits each AST followed immediately by:

```text
error = prediction - target
sse[ast] = error * error + sse[ast]
```

Temporary registers are reused between ASTs, and one branch skips the unused
tail of the site. Only the generated prefix and branch are written; stale
instructions after the branch are left untouched. If the skeleton's temporary
registers are insufficient, the patcher allocates above the original kernel
regcount and updates the cubin metadata.

The generated AST capacity and active patched AST count are separate. All
active ASTs in a kernel share one runtime settings stream. Each setting binds the
8 leaves to either input columns or f32 constants. A runtime `num_asts` controls
which final atomic SSE outputs are written, so a kernel may be patched with any
positive AST count up to its generated capacity without recompiling CUDA.

The output is raw `[ast][setting]` SSE. It must be cleared before launch. Search
can compare SSE directly; divide by row count only when MSE is required.

## AST Encoding

[`src/cusr_ast.h`](src/cusr_ast.h) defines the backend-neutral postorder AST
encoding. [`src/cusr_ast_sass.h`](src/cusr_ast_sass.h) lowers it to SASS, and
[`src/cusr_ast_sass_cpu.h`](src/cusr_ast_sass_cpu.h) provides the independent
CPU interpreter used for validation.

[`src/cusr_ast_routines.h`](src/cusr_ast_routines.h) includes composable
routines for `exp`, `log`, `log10`, `pow`, protected square root/reciprocal,
and protected division. Base operations include f32 ALU, min/max, `sin`, `cos`,
`tanh`, `ex2`, `lg2`, square root, and reciprocal square root.

The SASS backend uses approximate MUFU operations and FTZ arithmetic. CPU
results are numerical references with tolerances, not bitwise emulation.

## Kernel Template

[`kernels/tile_static_mse_template/cusr_tile_static_mse_template.cuh`](kernels/tile_static_mse_template/cusr_tile_static_mse_template.cuh)
contains the complete CUDA kernel and breakpoint site. The handwritten files in
`kernels/tile_static_mse_template/instantiations` expose stable `extern "C"`
symbols for modules containing 8, 32, or 128 kernels. CMake compiles all three
modules for `CUSR_SASS_GPU_NAME`, and
[`src/cusr_tile_static_mse_embedded.h`](src/cusr_tile_static_mse_embedded.h)
provides the embedded cubin bytes.

Run that path directly through the benchmark:

```sh
./build/cusr_bench --variant tile_static_mse_template
```

The embedded modules use AST capacity 32, 64-row tiles, and 128-thread CTAs.
The template also supports capacities 8 and 16 plus alternate tile and CTA
sizes. [`cusr_tile_static_mse_nvrtc.h`](kernels/tile_static_mse_template/cusr_tile_static_mse_nvrtc.h)
compiles those shapes at runtime. Its opaque handle owns the `nvrtcProgram` and
emits stable `extern "C"` kernel names, so callers do not manage source text,
NVRTC name expressions, or lowered C++ names.

The runtime compiler follows a create, measure, copy contract:

```c
CusrTileStaticMseNvrtcHandle* handle = NULL;
size_t cubin_size = 0;
void* cubin = NULL;

cusr_tile_static_mse_nvrtc_create(8, 16, 64, 128, 12, 0, &handle);
cusr_tile_static_mse_nvrtc_cubin_size(handle, &cubin_size);
cubin = malloc(cubin_size);
cusr_tile_static_mse_nvrtc_get_cubin(handle, cubin, cubin_size);
cusr_tile_static_mse_nvrtc_destroy(handle);
free(cubin);
```

`cubin` is caller-owned storage of at least `cubin_size` bytes. Compiler-log
size and copy functions remain available on the handle, including when NVRTC
compilation fails after program creation.

The [`python/cusr`](python/cusr) compiler takes a list of template
instantiations and passes their name expressions directly to `cuda.core`. It
does not append `extern "C"` wrappers. Its result contains the cubin bytes, the
original name expressions, and the exact lowered symbols returned for that
compilation. See [`python/README.md`](python/README.md) for setup and usage.
The same package includes an independent Python-native cubin inspector whose
structured output is tested field-for-field against the C99 inspector.
Its nanobind extension consumes packed backend-neutral `CusrAstInstruction`
matrices and patches caller-owned cubin bytearrays in place. The Python runtime
then loads those bytes with cuda.core and launches directly over PyTorch-owned
CUDA tensors without copying their storage. An independent NumPy postfix
interpreter supplies the numerical reference. The complete workflow and build
command are in [`python/README.md`](python/README.md).

`cusr_generate_native_cuda_reference` generates a benchmark-only native
CUDA/C++ reference for the same settings, tile geometry, and atomic-SSE
topology. It measures the ceiling from expressions compiled directly by
NVRTC; it is not a runtime AST backend.

## Build And Test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUSR_SASS_GPU_NAME=sm_120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The GPU tests instantiate alternate kernels with NVRTC, inspect and patch their cubins, run
64-, 128-, and 256-row variants, and compare against both the C CPU interpreter
and an independent Python evaluator. They also test 4 active ASTs in capacity
16 and compare 8 active ASTs in capacity 8 against the same 8 ASTs in capacity
32.

Run the benchmark with the performance-oriented defaults:

```sh
./build/cusr_bench --preset runtime
./build/cusr_bench --preset patch
```

The capacity-versus-active measurements and prefix-only patch results are in
[`docs/RTX_5090_ACTIVE_CAPACITY.md`](docs/RTX_5090_ACTIVE_CAPACITY.md).

Architecture-specific branch and `brkpt` probes cover `sm_80`, `sm_86`,
`sm_89`, `sm_90`, `sm_100`, and `sm_120`:

```sh
python3 tools/probes/branch_brkpt/verify_branch_brkpt.py
```
