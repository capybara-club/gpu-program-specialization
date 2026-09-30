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

# Stack PTX / PTX Inject Benchmark

> **Collection category:** Historical implementations and measurement. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

Standalone benchmark that measures stack-ptx stub generation plus PTX-to-cubin
compilation. It generates large CUDA kernels, injects stack-ptx stubs, and
compiles a cubin per module in parallel. Each thread has its own bump allocator
and reuses the same memory pool inside the hot loop.

This version embeds `ptx_inject.h`, `stack_ptx.h`, and the kernel generator. The `stack_ptx.h` and
`ptx_inject.h` headers are copied from the mm-ptx repo: https://github.com/MetaMachines/mm-ptx

## Why this is useful

- Measures end-to-end compile throughput (stub render + PTX inject + cubin).
- Stress tests nvPTXCompiler with large generated kernels and many inject sites.
- Compares compile scaling across OpenMP threads without extra runtime overhead.
- Reproducible: all program generation happens locally with a fixed seed.

## How it works (in depth)

1) Kernel generation

- The tool emits `num_kernels` CUDA kernels. Each kernel contains
  `groups_per_kernel` inject sites, selected by `blockIdx.x` (the “group”).
- The total number of inject sites (programs) per module is
  `num_kernels * groups_per_kernel`. Each inject site is named `func_<id>`.

2) Program generation (random stack-ptx)

- For each program, it builds a stack-ptx instruction list of length
  `ptx-instructions-per-program`.
- The last instruction is always `return`.
- Earlier instructions are randomly sampled from a pool of simple F32 ops
  (add, mul, fma, div, sin, etc.), plus random input reads or random constants.
- If the input type is U32, the generator inserts a cast to F32 when needed.
- This means each program is a different random instruction sequence up to the
  requested length, and every module is filled with unique programs.

3) Stub compilation and injection

- Each program is compiled into a PTX stub via `stack_ptx_compile` from
  `stack_ptx.h`.
- Stubs are injected into the kernel PTX using `ptx_inject` before cubin
  compilation.
- The benchmark measures the full time for stub compilation + injection +
  nvPTXCompiler cubin generation.

4) Kernel execution model (what a CTA does)

- Each CTA (CUDA block) is tied to a specific program via `blockIdx.x`.
  That program’s injected PTX executes for every element in the tile handled
  by that CTA.
- `blockIdx.y` selects a tile in the training dimension. Threads in the block
  iterate over `i` in `[tile_start, tile_end)` with stride `blockDim.x`.
- For each `i`, the kernel loads an input vector from `data` and runs the
  injected PTX to produce `embed_dims` outputs.

## Inject site example

One stack-ptx instruction list that produces the injected stub below (input 0
is x0, input 1 is x1; output pops are y0 then y1; looks like:

```c
static const StackPtxInstruction kExampleProgram[] = {
    stack_ptx_encode_input(1),
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,
    stack_ptx_encode_constant_f32(-6.126209259f),
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_input(1),
    stack_ptx_encode_constant_f32(4.942284107f),
    stack_ptx_encode_ptx_instruction_tanh_approx_f32,
    stack_ptx_encode_input(1),
    stack_ptx_encode_ptx_instruction_add_ftz_f32,
    stack_ptx_encode_ptx_instruction_sqrt_approx_ftz_f32,
    stack_ptx_encode_ptx_instruction_div_approx_ftz_f32,
    stack_ptx_encode_ptx_instruction_cos_approx_ftz_f32,
    stack_ptx_encode_constant_f32(1.441460252f),
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_constant_f32(3.308667660f),
    stack_ptx_encode_ptx_instruction_fma_rn_ftz_f32,
    stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_ptx_instruction_div_approx_ftz_f32,
    stack_ptx_encode_ptx_instruction_abs_ftz_f32,
    stack_ptx_encode_constant_f32(-2.771166325f),
    stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_ptx_instruction_sub_ftz_f32,
    stack_ptx_encode_constant_f32(4.709908485f),
    stack_ptx_encode_ptx_instruction_rcp_approx_ftz_f32,
    stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_ptx_instruction_max_ftz_f32,
    stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_ptx_instruction_copysign_f32,
    stack_ptx_encode_input(1),
    stack_ptx_encode_ptx_instruction_ex2_approx_ftz_f32,
    stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_ptx_instruction_div_approx_ftz_f32,
    stack_ptx_encode_input(0),
    stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_F32),
    stack_ptx_encode_ptx_instruction_sub_ftz_f32,
    stack_ptx_encode_constant_f32(-5.611264706f),
    stack_ptx_encode_return
};
```

The kernel PTX contains inline-asm markers (`PTX_INJECT_START`/`PTX_INJECT_END`)
that `ptx_inject` replaces with the compiled stack-ptx stub. For example, a
site like:

```ptx
$L__BB0_22:
	// begin inline asm
	{
	.reg .f32 %_x0;
	.reg .f32 %_x1;
	.reg .f32 %_x2;
	.reg .f32 %_x3;
	.reg .f32 %_x4;
	.reg .f32 %_x5;
	mov.f32 %_x4, %f5;
	mov.f32 %_x5, %f6;
	// PTX_INJECT_START func_9
	// _x0 o f32 F32 y0
	// _x1 o f32 F32 y1
	// _x2 o f32 F32 y2
	// _x3 o f32 F32 y3
	// _x4 i f32 F32 x0
	// _x5 i f32 F32 x1
	// PTX_INJECT_END
	mov.f32 %f652, %_x0;
	mov.f32 %f653, %_x1;
	mov.f32 %f654, %_x2;
	mov.f32 %f655, %_x3;
	}
	// end inline asm
	mov.u64 	%rd106, 9;
	bra.uni 	$L__BB0_159;
```

gets injected as something like:

```ptx
$L__BB0_22:
	// begin inline asm
	{
	.reg .f32 %_x0;
	.reg .f32 %_x1;
	.reg .f32 %_x2;
	.reg .f32 %_x3;
	.reg .f32 %_x4;
	.reg .f32 %_x5;
	mov.f32 %_x4, %f5;
	mov.f32 %_x5, %f6;
	{
	.reg .f32 %_a<19>;
	abs.ftz.f32 %_a0, %_x5;
	ex2.approx.ftz.f32 %_a1, %_a0;
	abs.ftz.f32 %_a2, 0fC0C409E8;
	tanh.approx.f32 %_a3, 0f409E2731;
	add.ftz.f32 %_a4, %_x5, %_a3;
	sqrt.approx.ftz.f32 %_a5, %_a4;
	div.approx.ftz.f32 %_a6, %_a5, %_x5;
	cos.approx.ftz.f32 %_a7, %_a6;
	abs.ftz.f32 %_a8, 0f3FB881C5;
	fma.rn.ftz.f32 %_a9, 0f4053C136, %_a7, %_a8;
	div.approx.ftz.f32 %_a10, %_a2, %_a9;
	abs.ftz.f32 %_a11, %_a10;
	sub.ftz.f32 %_a12, %_a11, 0fC0315ACA;
	rcp.approx.ftz.f32 %_a13, 0f4096B792;
	max.ftz.f32 %_a14, %_a12, %_a13;
	copysign.f32 %_a15, %_a1, %_a14;
	ex2.approx.ftz.f32 %_a16, %_x5;
	div.approx.ftz.f32 %_a17, %_a15, %_a16;
	sub.ftz.f32 %_a18, %_a17, %_x4;
	mov.f32 %_x0, 0fC0B38F7B;
	mov.f32 %_x1, %_a18;
	}
	mov.f32 %f652, %_x0;
	mov.f32 %f653, %_x1;
	mov.f32 %f654, %_x2;
	mov.f32 %f655, %_x3;
	}
	// end inline asm
	mov.u64 	%rd106, 9;
	bra.uni 	$L__BB0_159;
```

## Tensor layout

Inputs are passed as:

- `data`: a 2D tensor with shape `[input_dims, num_train]`, stored with stride
  `ld_input` between input dimensions. Indexing is:
  `data[dim * ld_input + i]` for `dim in [0, input_dims)`.

Outputs are written as:

- `embed`: a 3D tensor with shape `[programs, embed_dims, num_train]`.
  The program index is the inject site id (`local_idx`), and the layout is:
  `embed[local_idx * batch_stride + dim * ld_embed + i]`.

So each CTA executes its own random program and writes a slice of the output
tensor for its program id.

## Dependencies

- CUDA Toolkit (nvrtc + nvptxcompiler).
- A C toolchain and Make or CMake.

Beyond a standard C compiler, the only external dependency is the CUDA Toolkit.

OpenMP is optional. If it is not found or disabled, the benchmark runs single-core.

## Build

From the repo root:

```bash
cmake -S . -B build
cmake --build build --target stack_ptx_ptx_inject_bench
```

### Makefile build

From this folder:

```bash
make
```

Override CUDA path or disable OpenMP:

```bash
make CUDA_HOME=/opt/cuda OPENMP=0
```

If your toolkit only ships `libnvptxcompiler_static.a`, the Makefile auto-detects
it. You can also override explicitly:

```bash
make NVPTXCOMPILER_LIB=nvptxcompiler_static
```

## Usage

```bash
./build/stack_ptx_ptx_inject_bench \
  --sm sm_80 --modules 128 --kernels 16 --groups-per-kernel 128
```

## Benchmarks

Collected runs live in [benchmarks/](benchmarks/).

## Options (exhaustive)

- `-h`, `--help` Show usage.
- `--modules N` Total modules to compile (default: 128).
- `--kernels N` Kernels per module (default: 16).
- `--groups-per-kernel N` Inject sites per kernel (default: 128).
- `--tile-size N` Kernel tile size (default: 256).
- `--embed-dims N` Output dims per inject (default: 1).
- `--input-dims N` Input dims per inject (default: 1).
- `--input-type STR` Input type: `F32` or `U32` (default: `F32`).
- `--ptx-instructions-per-program N` Instructions per program, includes return (default: 32, must be >= 2).
- `--program-execution-limit N` Stack-ptx execution limit (default: 100).
- `--workspace-bytes N` Per-thread workspace bytes (default: auto).
- `--cores N` OpenMP threads (default: OpenMP runtime).
- `--seed N` RNG seed (default: 1).
- `--sm SM` GPU SM for both stages (e.g. `sm_90`, `90`, `9.0`).
- `--sm-ptx SM` GPU SM for CUDA -> PTX (NVRTC).
- `--sm-cubin SM` GPU SM for PTX -> cubin (nvPTXCompiler).
- `--dump-cu PATH` Write generated CUDA source to PATH.
- `--dump-ptx PATH` Write NVRTC PTX to PATH.
- `--dump-module-ptx PATH` Write injected PTX for one module to PATH.
- `--dump-module-index N` Module index to dump (default: 0).
- `--verbose` Print NVRTC info logs.

## Output

The benchmark prints a compact summary including PTX version, nvPTXCompiler API
version, SMs for both stages, CPU label, configuration, wall time, modules/sec
(and per thread), and full compile throughput in us/prog and progs/sec (plus
per-thread equivalents).

## Notes

- The cubin is compiled for `--sm-cubin` (or `--sm`); PTX is generated for `--sm-ptx` (or `--sm`).
- SM fallback for unset stages uses `STACK_PTX_COMPILER_SM`, then `STACK_PTX_NNG_SM`, else `8.0`.
- If you see failures, increase `--workspace-bytes` or lower `groups-per-kernel`.

## Further documentation

- mm-ptx repo: https://github.com/MetaMachines/mm-ptx
- mm-ptx README: https://github.com/MetaMachines/mm-ptx/blob/master/README.md
- stack-ptx docs: https://github.com/MetaMachines/mm-ptx/blob/master/STACK_PTX.md
- ptx-inject docs: https://github.com/MetaMachines/mm-ptx/blob/master/PTX_INJECT.md
