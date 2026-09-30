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

# cuSR Bench

`cusr_bench` selects an embedded NVCC-built skeleton module, inspects it once,
copies it across the requested module count, creates random AST programs, and
times three independent phases:

- AST lowering and in-place SASS site patching;
- `cuModuleLoadData` for every patched module;
- GPU execution.

Inspection, allocation, and optional CPU validation are not included in the
patch timer. Noncanonical template shapes fall back to NVRTC and use ptxas
spill warnings as errors.

## Defaults

```sh
./build/cusr_bench
```

The runtime preset uses:

```text
variant          = tile_static_mse_template
AST mode         = depth3_alu
modules          = 1
kernels/module   = 8
active ASTs/kernel = 32
AST capacity       = 32
settings         = 1,024
columns          = 32
tile rows        = 64
CTA threads      = 128
rows             = 1,048,576
iterations       = 30
output           = raw atomic SSE
```

Each kernel has one SASS patch site containing all 32 ASTs. The ASTs share
one settings stream and one loaded row tile. The patcher emits each expression
and its error/SSE update sequentially, reusing temporary registers, then emits
one branch over the unused site tail. It writes only the generated prefix.

`--ast-capacity N` selects the generated kernel capacity.
`--asts-per-kernel N` selects the active AST count. Supplying only either option
sets both values, preserving the original CLI behavior. Supplying both permits
tests such as 8 active ASTs in a capacity-32 kernel.

Use `--check-rows N` to run the independent CPU comparison before timing.

## Patch Preset

```sh
./build/cusr_bench --preset patch
```

This uses 32 modules, 8 kernels/module, and 32 ASTs/kernel, for 8,192 ASTs. GPU
work is intentionally tiny so the command focuses on single-core patch rate.

Use a Release build for patch measurements:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DCUSR_SASS_GPU_NAME=sm_120
cmake --build build-release -j --target cusr_bench
./build-release/cusr_bench --preset patch
```

## Variants

- `tile_static_mse_template`: the AST-SASS backend. Its canonical 8-, 32-,
  and 128-kernel modules are compiled by NVCC and embedded; alternate template
  shapes use the `cusr_tile_static_mse_nvrtc` C99 handle API.
- `native_cuda_reference`: benchmark-only native CUDA/C++ ceiling for the same
  settings, tile geometry, and direct atomic-SSE topology. The expressions are
  compiled into the kernel and cannot use a smaller active count than capacity.

Both variants accept `--tile-rows 64|128|256` and
`--cta-threads 64|128|256`. Output storage is exactly
`ASTs * settings * sizeof(float)` and is cleared inside the timed run before
each launch.

Runtime timing launches generated kernels on separate nonblocking streams and
joins them with events.

For a noncanonical template shape, the benchmark creates a runtime compiler
handle, queries the cubin size, copies the cubin into its own allocation, and
destroys the handle. The API embeds the CUDA template with `incbin` and emits
stable `cusr_tile_static_mse_f32_%03d` symbols, which are passed directly to
both `cusr_sass_inspect` and `cuModuleGetFunction`.

## AST Modes

- `simple`: small f32 ALU programs;
- `mufu`: MUFU-heavy unary programs;
- `mixed`: mixed ALU and MUFU programs;
- `offset-minmax`: adds an AST-specific offset to each input, then reduces with
  min/max;
- `alu-reduce`: 8 leaves with 7 deterministic add/mul/min/max reductions;
- `depth3-alu`: balanced 8-leaf ALU trees;
- `depth3-mufu`: balanced trees with MUFU-family leaf operations.

Run `./build/cusr_bench --help` for all dimensions.

The capacity-versus-active benchmark and prefix-only patch measurements are in
[`docs/RTX_5090_ACTIVE_CAPACITY.md`](../docs/RTX_5090_ACTIVE_CAPACITY.md).
The direct-template NVRTC and module-load sweep is in
[`docs/RTX_5090_NVRTC_TEMPLATE_SWEEP.md`](../docs/RTX_5090_NVRTC_TEMPLATE_SWEEP.md).
