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

# cubin-function-patch

> **Collection category:** Program construction and specialization tools. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

`cubin-function-patch` is a single-header C99 API for replacing reserved CUDA
device-function bodies in a fully linked cubin/ELF with same-signature function
bodies from nvPTXCompiler `--compile-only` RDC ELF output.

The public header is intentionally at the repository root:

```text
cubin_function_patch.h
```

Copy it into a project, include it normally, and define
`CUBIN_FUNCTION_PATCH_IMPLEMENTATION` in exactly one C or C++ translation unit.

## What It Does

The normal CUDA dynamic path is:

```text
CUDA/PTX input -> ptxas/nvPTXCompiler/nvJitLink -> final cubin
```

That can be too expensive when one small device function changes but a large
global-kernel body stays the same. This library supports a narrower workflow:

```text
cold path:
    link a final cubin template that contains oversized reserved function bodies
    create a cubin_function_patch handle from that linked cubin

hot path:
    compile replacement device functions with nvPTXCompiler --compile-only
    copy replacement function text into the reserved function slots
    load the patched cubin with the CUDA Driver API
```

The patch handle scans the template cubin once, records function text ranges,
and later patches caller-owned output cubin buffers. The library does not hide
allocation: handle memory and output cubin memory are supplied by the caller.

## Why It Exists

This is useful when all of these are true:

- the global CUDA kernels are large and mostly stable;
- the changed code is isolated behind one or more noinline device functions;
- the replacement functions have a fixed ABI;
- compile latency matters enough that relinking the full cubin is too slow;
- inputs are controlled by your compiler pipeline, not arbitrary user uploads.

The main motivating use case is dynamic CUDA program generation where many
small evaluator functions are compiled and patched into a reserved Gram/kernel
template.

## Contract

This library is intentionally narrow. It is not a general CUDA linker.

The replacement function must:

- have the same symbol name as the reserved function;
- have the same calling convention and signature;
- target the same CUDA ELF machine architecture;
- fit inside the reserved linked function body;
- be self-contained, except for CUDA constant-bank relocations that can be
  matched to reserved side sections;
- not require new sections or metadata that the linked template did not reserve;
- not rely on branch targets in stale tail bytes when the replacement is smaller
  than the reserved body.

The linked template must:

- contain each patchable function as an `STT_FUNC` symbol;
- reserve worst-case function-body size;
- reserve conservative `.nv.info.*` metadata for the worst-case body. The
  patcher preserves this metadata instead of copying replacement `.nv.info.*`;
- reserve compatible `.nv.constant2.*` side sections and cap/merc mirror
  text sections when the target CUDA version emits them;
- be built for the same architecture and ABI assumptions as replacement RDC
  objects;
- keep metadata conservative enough for all replacement bodies.

If a replacement is smaller than the reserved function body, the default policy
leaves tail bytes unchanged. Use
`CUBIN_FUNCTION_PATCH_TAIL_REQUIRE_EXACT_SIZE` if exact body size is required.

## Non-Goals

This library does not try to:

- patch arbitrary untrusted cubins;
- validate every CUDA ABI or metadata invariant NVIDIA's linker would validate;
- grow function bodies;
- add new sections;
- resolve general relocations;
- patch host code;
- support non-NVIDIA GPU formats;
- abstract over AMD, Apple, CPU, or generic accelerator backends.

It is a low-level CUDA tool for controlled compiler pipelines.

## Supported Platform

The implementation parses little-endian 64-bit CUDA ELF/cubin data and is
intended for Linux CUDA toolchains on:

```text
x86_64
ARM SBSA / aarch64
```

The tests require an NVIDIA driver and CUDA development tools.

## Tested Systems

The test suite is not a parser-only check. It compiles CUDA fixture kernels,
produces RDC and linked cubins, patches device-function bodies, loads the
patched cubins through the CUDA Driver API, launches the kernels, and compares
device outputs against normally linked references.

The current code has been tested on:

| System | CPU architecture | GPU | Target |
| --- | --- | --- | --- |
| Local workstation | `x86_64` | NVIDIA GeForce RTX 5090 | `sm_120` |
| Ada workstation | `x86_64` | NVIDIA GeForce RTX 4090 | `sm_89` |
| Lambda Labs GH200 | ARM SBSA / `aarch64` | NVIDIA GH200 480GB | `sm_90` |

## Quick Start

```c
#define CUBIN_FUNCTION_PATCH_IMPLEMENTATION
#include "cubin_function_patch.h"
```

Typical use:

```c
const char* symbols[] = {"patch_site"};
size_t handle_bytes = 0;
void* handle_memory = NULL;
CubinFunctionPatchHandle* handle = NULL;
size_t output_bytes = 0;
void* output_cubin = NULL;
size_t written = 0;

cubin_function_patch_handle_size(symbols, 1, &handle_bytes);
handle_memory = malloc(handle_bytes);

cubin_function_patch_create(
    template_cubin,
    template_cubin_bytes,
    symbols,
    1,
    handle_memory,
    handle_bytes,
    &handle
);

cubin_function_patch_output_size(handle, &output_bytes);
output_cubin = malloc(output_bytes);

cubin_function_patch_apply_all(
    handle,
    replacement_rdc_elf,
    replacement_rdc_elf_bytes,
    output_cubin,
    output_bytes,
    &written,
    NULL,
    0
);

/* output_cubin now contains a linked cubin-sized image with patched text. */
```

Production code should check every `CubinFunctionPatchResult` and free/destroy
state according to the ownership rules below.

## API Shape

The API has three phases.

1. Measure and create a handle:

```c
cubin_function_patch_handle_size(...);
cubin_function_patch_create(...);
```

2. Query template/output information:

```c
cubin_function_patch_num_sites(...);
cubin_function_patch_site_symbol(...);
cubin_function_patch_site_reserved_size(...);
cubin_function_patch_output_size(...);
```

3. Patch one or more replacement functions:

```c
cubin_function_patch_begin(...);
cubin_function_patch_apply_one_in_place(...);
cubin_function_patch_apply_all_in_place(...);
cubin_function_patch_apply_one(...);
cubin_function_patch_apply_all(...);
```

The `_ex` variants accept an explicit `CubinFunctionPatchTailPolicy`.

## Ownership

- `cubin_function_patch_create` does not copy the linked template cubin. Keep
  the template bytes alive and immutable while the handle exists.
- Handle memory is caller-owned.
- Output cubin memory is caller-owned.
- The library performs no internal allocation.
- `cubin_function_patch_destroy` currently performs no deallocation, but should
  still be called for API symmetry.
- The handle is immutable after creation. Multiple threads may patch through the
  same handle when each call writes to a distinct output buffer.

## Build And Test

Build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Override the CUDA architecture used by build-time tests:

```sh
cmake -S . -B build -DCUBIN_FUNCTION_PATCH_TEST_SM=90
```

The tests use two paths:

- `test/runtime_compile`: embeds CUDA source with `thirdparty/incbin`, compiles
  it at test runtime with NVRTC, nvPTXCompiler, and nvJitLink, patches cubins,
  launches them through the CUDA Driver API, and compares against normally
  linked references.
- `test/cmake_rdc`: uses CMake custom commands to compile CUDA files to RDC
  cubins and linked reference/template cubins before the test runs. The binary
  embeds those artifacts with `INCBIN` and then uses only the CUDA Driver API
  plus `cubin_function_patch.h`.

Shared CUDA fixtures live in:

```text
test/common/cuda
```

## Benchmark

The `bench/` directory contains a standalone multicore benchmark for the
compile/link paths this library is meant to replace:

```sh
cmake -S . -B build-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCUBIN_FUNCTION_PATCH_BUILD_BENCHMARKS=ON
cmake --build build-bench --parallel

./build-bench/bench/cubin_function_patch_compile_scaling_bench \
  --modules 120 \
  --workers 1,2,4,8,12 \
  --repeats 1 \
  --warmup 0
```

The benchmark checks correctness first by compiling representative cubins,
loading them with the CUDA Driver API, launching the kernel, and comparing
against a CPU reference. It also performs a final sanity pass across a few
compiled module indices after the timed rows. Use `--skip-correctness` and
`--skip-final-sanity` to omit those checks during pure timing runs.

The timed rows are:

| Mode | Hot-path work |
| --- | --- |
| `full_compile` | NVRTC compiles the full CUDA source and nvPTXCompiler compiles the full cubin. |
| `full_ptx_inject` | A precompiled PTX template containing the global kernel and noinline device function is patched at the PTX text level, then nvPTXCompiler compiles the whole injected PTX directly to a final cubin. |
| `partial_nvjitlink` | NVRTC/nvPTXCompiler compile a replacement noinline device function, then nvJitLink links it with a prebuilt caller RDC. |
| `partial_patch` | NVRTC/nvPTXCompiler compile a replacement noinline device function, then `cubin_function_patch` patches it into a prelinked template cubin. |
| `nvjitlink_only` | Replacement RDCs are precompiled before timing; the hot path only runs nvJitLink. |
| `patch_only` | Replacement RDCs are precompiled before timing; the hot path only copies and patches cubin bytes. |

`nvJitLink` is created with `-O0 -no-cache` so the link-only rows do not report
cache hits as linker throughput.

On the local RTX 5090 workstation (`x86_64`, CUDA 13.3, `sm_120`), this command:

```sh
./build-bench/bench/cubin_function_patch_compile_scaling_bench \
  --modules 120 \
  --workers 1,2,4,8,12 \
  --repeats 1 \
  --warmup 0 \
  --skip-correctness \
  --skip-final-sanity
```

produced:

```text
mode                  workers      best_ms      modules/s   nvptx_sum_ms      inject_ms   nvjitlink_ms       patch_ms
full_compile                1     2686.246         44.672       1978.775          0.000          0.000          0.000
full_compile               12      797.842        150.406       2124.971          0.000          0.000          0.000
full_ptx_inject             1     1933.417         62.066       1932.856          0.279          0.000          0.000
full_ptx_inject            12      224.036        535.629       2603.814          0.588          0.000          0.000
partial_nvjitlink           1     1024.719        117.105        278.462          0.000         72.846          0.000
partial_nvjitlink          12      732.781        163.760        296.122          0.000         76.823          0.000
partial_patch               1      939.842        127.681        273.373          0.000          0.000          0.370
partial_patch              12      746.391        160.774        296.851          0.000          0.000          0.496
nvjitlink_only              1       60.108       1996.415          0.000          0.000         60.017          0.000
nvjitlink_only             12       37.854       3170.104          0.000          0.000        332.153          0.000
patch_only                  1        0.155     772106.355          0.000          0.000          0.000          0.125
patch_only                 12        0.152     787489.420          0.000          0.000          0.000          0.136
```

On the Ada workstation (`x86_64`, CUDA 13.1, RTX 4090, `sm_89`), the same
120-module command produced:

```text
mode                  workers      best_ms      modules/s   nvptx_sum_ms      inject_ms   nvjitlink_ms       patch_ms
full_compile                1     3293.147         36.439       1708.950          0.000          0.000          0.000
full_compile               12      793.777        151.176       1805.477          0.000          0.000          0.000
full_ptx_inject             1     1588.421         75.547       1587.574          0.312          0.000          0.000
full_ptx_inject            12      207.552        578.170       2099.767          0.832          0.000          0.000
partial_nvjitlink           1      985.561        121.758        273.287          0.000         12.228          0.000
partial_nvjitlink          12      774.067        155.025        354.581          0.000         12.696          0.000
partial_patch               1      969.649        123.756        274.591          0.000          0.000          0.309
partial_patch              12      767.767        156.297        359.584          0.000          0.000          0.432
nvjitlink_only              1        9.123      13152.882          0.000          0.000          9.035          0.000
nvjitlink_only             12       12.135       9888.523          0.000          0.000        124.076          0.000
patch_only                  1        0.174     688784.298          0.000          0.000          0.000          0.139
patch_only                 12        0.240     499679.372          0.000          0.000          0.000          0.190
```

The `*_only` rows are the cleanest final-link comparison: nvJitLink improves
only modestly or regresses under many worker threads, while patching stays close
to a memory copy of the prelinked template.

## Install

The repository is primarily single-header, but CMake also installs the header
and license:

```sh
cmake --install build --prefix /tmp/cubin-function-patch-install
```

Installed files:

```text
include/cubin_function_patch.h
share/cubin-function-patch/LICENSE
```

## Third Party

The tests vendor `incbin` under `thirdparty/incbin`. It keeps its own license in
`thirdparty/incbin/UNLICENSE`.

## License

`cubin-function-patch` is licensed under the MIT license.

```text
Copyright (c) 2026 Charles Durham
```

See `LICENSE`.
