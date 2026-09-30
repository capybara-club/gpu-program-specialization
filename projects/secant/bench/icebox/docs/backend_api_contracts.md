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

# Backend API Contracts

SECANT shares only the postorder f32 AST representation. Each backend exposes
the lifecycle that matches its actual work.

## Backend Roles

- CUBIN and HSACO are the primary native specialization backends intended for
  application deployment.
- CUDA C++, PTX, and HIP C++ are compiler baselines retained for correctness
  validation and compile/runtime benchmarking.
- CPU is the portable semantic oracle.

The compiler baselines are deliberately complete enough to produce and execute
the same kernel shapes. Their presence does not imply that applications should
select them instead of native specialization.

## CPU

The CPU API is declared in `secant_cpu.h`. It is a direct interpreter and has no
create, compile, load, or destroy lifecycle.

- ASTs, routines, input, targets, and output remain caller-owned.
- Materialize writes `[ast][row]`.
- SSE adds to existing `[ast][target]`; the caller clears it before a fresh run.
- Calls allocate and retain nothing.

The CPU backend is the correctness oracle for generated GPU kernels.

## CUDA Compiler Baseline

`secant_cuda.h` directly generates CUDA C++ and invokes NVRTC.

- There is no reusable create handle.
- A compile call includes AST rendering and NVRTC-to-CUBIN compilation.
- Caller scratch holds option pointers and generated source.
- The successful result owns one internally allocated final CUBIN.
- Module loading, function resolution, execution, and CUDA runtime state are
  outside this backend.

The canonical CUDA compile timer surrounds only
`secant_cuda_*_compile`.

## PTX Compiler Baseline

`secant_ptx.h` has a reusable handle because template preparation is meaningful
cold-path work.

- Create generates the CUDA template, runs NVRTC to virtual PTX, parses PTX
  Inject sites, and retains immutable template metadata.
- Compile lowers ASTs, renders PTX stubs, injects them, and invokes
  nvPTXCompiler for the native CUBIN.
- Concurrent compile calls may share one handle when each call has independent
  ASTs, scratch, log storage, and output.
- Module loading and execution of the resulting CUBIN belong to the caller.

The canonical PTX compile timer excludes create and surrounds only
`secant_ptx_compile`.

## HIP Compiler Baseline

`secant_hip.h` directly generates HIP C++ and invokes HIPRTC.

- There is no reusable create handle.
- A compile call includes AST rendering and HIPRTC-to-HSACO compilation.
- Caller scratch holds option pointers and generated source.
- The successful result owns one internally allocated final HSACO.
- Module loading, function resolution, execution, and HIP runtime state are
  outside this backend.

The canonical HIP compile timer surrounds only
`secant_hip_*_compile`.

## HSACO

HIP source generation, native HSACO inspection, and HIP execution are separate
operations exposed through `secant_hsaco.h`.

- `secant_hsaco_materialize_source_generate` and
  `secant_hsaco_sse_source_generate` render HIP from shape-specific recipes
  into caller-owned measure/write buffers.
- The caller compiles that source to HSACO using HIPRTC, hipcc, or another
  compatible compiler path.
- `secant_hsaco_materialize_inspect` and `secant_hsaco_sse_inspect` each take
  only their shape's generation arguments plus the caller's HSACO.
- Null plan storage measures the exact required storage. The write pass places
  the opaque immutable plan and parsed metadata in caller memory. Inspection
  validates the GFX architecture and every expected patch island but does not
  copy or retain the HSACO.
- Patch capacity is one total instruction count per kernel island. It does not
  imply fixed per-AST slots.
- `secant_hsaco_specialize_into` lowers ASTs and patches a caller-owned HSACO
  copy directly without allocation, scratch, HIPRTC, module loading, or
  synchronization. An error may leave the destination image invalid.
- Register expansion updates the kernel descriptor, absolute `.num_vgpr`
  symbol, and `.vgpr_count` in the AMDGPU MsgPack metadata note.
- One plan may be shared by concurrent calls that specialize independent
  HSACO copies.
- Loading and launching specialized HSACO images belong to the caller.

The current native encoder and binary contracts support `gfx1200` and
`gfx1201`. If an expanded VGPR count cannot fit the metadata note's existing
in-place integer representation, patching returns
`SECANT_ERROR_REGISTER_OVERFLOW`.

The canonical direct-HSACO compile timer surrounds only
`secant_hsaco_specialize_into`.

## CUBIN

CUDA source generation and native CUBIN inspection are separate operations
exposed through `secant_cubin.h`.

- `secant_cubin_materialize_source_generate` and
  `secant_cubin_sse_source_generate` render CUDA from shape-specific recipes
  into caller-owned measure/write buffers.
- The caller compiles that source to CUBIN using NVRTC, nvcc, cuda-python, or
  another compatible compiler path.
- `secant_cubin_materialize_inspect` and `secant_cubin_sse_inspect` each take
  only their shape's generation arguments plus the caller's CUBIN.
- Null plan storage measures the exact required storage. The write pass places
  the opaque immutable plan and parsed metadata in caller memory. Inspection
  validates the SM architecture and every expected patch island but does not
  copy or retain the CUBIN.
- Patch capacity is one total instruction count per kernel island. It does not
  imply fixed per-AST slots.
- `secant_cubin_specialize_into` lowers ASTs and patches a caller-owned CUBIN
  copy directly without allocation, scratch, compilation, module loading, or
  synchronization. An error may leave the destination image invalid.
- One plan may be shared by concurrent calls that specialize independent
  CUBIN copies.
- Loading and launching specialized CUBIN images belong to the caller.

The canonical direct-CUBIN compile timer surrounds only
`secant_cubin_specialize_into`.

## Binary And Runtime Ownership

CUDA, PTX, and HIP compile objects own their native binary bytes until their
compiled-destroy call. Native CUBIN and HSACO specialization writes directly
into caller-owned images.

The core compile and specialization APIs do not wrap module or function
handles. Optional backend runner APIs provide bulk compile/specialize, module
streaming, launch, synchronization, and unload. The caller still owns the
active context or device, input/output device buffers, and any external
streams.

All compile APIs currently require caller-sized scratch and return
`*_ERROR_INSUFFICIENT_BUFFER` when it is too small. There is not yet one exact
compile-workspace measurement call covering the foreign compiler's complete
hot path. Benchmarks allocate reusable scratch before timing and never retry or
allocate inside the timed interval.
