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

# PTX Inject

## Overview
PTX Inject is a single-file, header-only C library for modifying compiled GPU kernels by injecting custom PTX at user-defined sites. The sites are declared in CUDA code with macros that emit PTX markers and stable register names. The library parses those markers, lets you query register names, and renders a final PTX string with your injected stubs.

The main idea is to keep CUDA in charge of the full kernel structure, while giving you precise PTX insertion points with stable registers. This makes fast, repeatable kernel variants practical without rebuilding full CUDA code for every tweak.

## Usage
Include `ptx_inject.h` in exactly one translation unit with the implementation define:

```c
#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>
```

For debug builds, turn errors into asserts:

```c
#define PTX_INJECT_DEBUG
#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>
```

### CUDA-side markers
In CUDA code, use `PTX_INJECT` plus operand descriptors:

```c++
#include <ptx_inject.h>

extern "C" __global__ void kernel(float* out) {
    float x = 5;
    float y = 3;
    float z;
    PTX_INJECT("func",
        PTX_OUT(F32, v_z, z),
        PTX_MOD(F32, v_x, x),
        PTX_IN (F32, v_y, y)
    );
    *out = z;
}
```

Notes:
- `PTX_INJECT("func", ...)` declares an inject site named `func`.
- `PTX_IN/OUT/MOD` accept `(type, name)` or `(type, name, expr)`.
- `type` is a token with a `PTX_TYPE_INFO_<TOKEN>` entry. Defaults include `F16`, `F16X2`, `S32`, `U32`, `F32`, and `B32`.
- `name` is a label used for queries (`v_x`, `v_y`, `v_z` in the example).

### Host-side flow
A typical host-side flow mirrors `examples/ptx_inject/00_simple/main.c`:

1) Compile the CUDA source to PTX with `nvcc` or `nvrtc`. The `PTX_INJECT` macro emits marker comments in the PTX.
2) Parse the PTX:

```c
PtxInjectHandle ptx_inject;
ptx_inject_create(&ptx_inject, annotated_ptx);
```

3) Find the inject index and stable register names:

```c
size_t inject_idx;
ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_idx, NULL, NULL);

const char* reg_x;
const char* reg_y;
const char* reg_z;
ptx_inject_variable_info_by_name(ptx_inject, inject_idx, "v_x", NULL, &reg_x, NULL, NULL, NULL);
ptx_inject_variable_info_by_name(ptx_inject, inject_idx, "v_y", NULL, &reg_y, NULL, NULL, NULL);
ptx_inject_variable_info_by_name(ptx_inject, inject_idx, "v_z", NULL, &reg_z, NULL, NULL, NULL);
```

4) Build a PTX stub using the stable register names and render the final PTX:

```c
snprintf(stub, stub_size,
    "\tadd.ftz.f32 %%%3$s, %%%2$s, %%%1$s;",
    reg_x, reg_y, reg_z);

const char* stubs[1];
stubs[inject_idx] = stub;

ptx_inject_render_ptx(ptx_inject, stubs, 1, out_buffer, out_size, &bytes_written);
```

A helper that does measure-and-allocate for you lives at `common/ptx_inject_helper.h`.

## What PTX Inject emits
The CUDA macro expands to an inline asm block that declares stable temporary registers, moves operand values in and out, and emits marker lines that are ignored by the PTX assembler but parsed by PTX Inject. The marker format looks like this inside PTX:

```
// PTX_INJECT_START func
// _x0 o f32 F32 v_z
// _x1 m f32 F32 v_x
// _x2 i f32 F32 v_y
// PTX_INJECT_END
```

The `ptx_inject_variable_info_*` APIs expose:
- The stable register name (for example `_x0`).
- The mutation kind (out/mod/in).
- The PTX register type (`f32`) and the type token (`F32`).
- The variable name label (`v_z`).

If the inject site is duplicated by inlining or unrolling, the stable register names remain consistent across all sites, so you can inject the same stub everywhere.

## Examples
This repo includes several PTX Inject examples.

### 00_simple
Path: `examples/ptx_inject/00_simple`

- Uses a single inject site named `func`.
- Queries stable register names and injects an `add.ftz.f32`.
- Compiles the injected PTX to SASS and runs the kernel.

### 01_gemm
Path: `examples/ptx_inject/01_gemm`

- Uses two inject sites: `mma` and `epilogue`.
- Retrieves register names for each site and injects a fused multiply-add for the MMA path plus a move in the epilogue.
- Demonstrates multiple stubs ordered by inject index.

### 02_custom_types
Path: `examples/ptx_inject/02_custom_types`

- Demonstrates custom type tokens. In `kernel.cu`:

```c++
#define PTX_TYPE_INFO_DOG PTX_TYPES_DESC(f32, f32, f, ID)

PTX_INJECT("func",
    PTX_OUT(DOG, z, z),
    PTX_MOD(DOG, x, x),
    PTX_IN (DOG, y, y)
);
```

Here `DOG` is a custom token that maps to an `f32` register type and the `f` constraint. This lets you tag variables with domain-specific type names while still using standard PTX types.

### Stack PTX integration
If you want to generate PTX programmatically, see `thirdparty/mm-ptx/examples/stack_ptx_inject` for combined examples that feed Stack PTX output into PTX Inject.

## Custom types
The default type tokens are defined in `ptx_inject.h` inside the `__CUDACC__` block:

- `F16`, `F16X2`, `S32`, `U32`, `F32`, `B32`

To add your own, define `PTX_TYPE_INFO_<TOKEN>` before using the token in `PTX_INJECT`. The macro form is:

```c
#define PTX_TYPE_INFO_MYTYPE PTX_TYPES_DESC(reg_suffix, mov_postfix, constraint, bind_kind)
```

- `reg_suffix`: PTX register type suffix (for example `f32`, `b32`, `s32`).
- `mov_postfix`: PTX `mov` suffix (often the same as `reg_suffix`).
- `constraint`: inline PTX constraint character (for example `f`, `r`, `h`).
- `bind_kind`: one of `ID`, `U16`, `U32` to control how the C variable is bound.

To replace all defaults, define `PTX_INJECT_NO_DEFAULT_TYPES` before including `ptx_inject.h` in CUDA code, then declare the types you need.

## License
This project is licensed under the MIT License. See `thirdparty/mm-ptx/LICENSE` for details.
