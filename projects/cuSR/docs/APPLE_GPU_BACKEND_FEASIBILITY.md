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

# Apple GPU Backend Feasibility

This note sketches whether the cuSR AST-patching model could exist on Apple
GPUs. The short version is: the tile-static, multi-setting evaluation idea maps
well to Apple GPU hardware, but the current low-level patching toolchain does
not map cleanly to public Metal APIs. A supported Apple backend is plausible as
a source-level Metal generator. A true SASS-style machine-code patch backend is
theoretically possible only through reverse-engineered/private surfaces and
should not be treated as a normal product path.

## Core Read

cuSR's breakthrough is cheap AST specialization combined with tile-local
evaluation. On NVIDIA, that is implemented by compiling a patchable CUDA
skeleton, inspecting the cubin, and writing machine-code AST bodies into SASS
islands.

Apple exposes a very different stack:

- Metal Shading Language (MSL) source;
- Metal libraries (`metallib`);
- compute pipeline state objects;
- function constants for specialization;
- binary archives for pipeline caching;
- threadgroup memory for workgroup-local storage.

What Apple does not expose publicly is the equivalent of a stable ELF code
object with documented machine-code sections, kernel descriptors, register
counts, and supported in-place machine-code patching. That is the major
difference from AMD and NVIDIA.

## Public Metal Path

The supported Apple backend would look like this:

1. Generate MSL source for a tile-static scoring kernel.
2. Use `threadgroup` memory for the row/column tile.
3. Encode runtime settings as buffers.
4. Generate one or more specialized AST bodies directly in MSL.
5. Compile with Metal runtime or offline tools.
6. Cache pipeline state using Metal binary archives.
7. Dispatch many settings per tile to amortize threadgroup-memory loads.

This could preserve the most important runtime idea: keep data local and score
many AST+setting combinations against it. It would not preserve the near-zero
dynamic AST patching cost unless Apple exposes a supported lower-level binary
patch surface.

Function constants can specialize branches, sizes, and mode switches, but they
are not a replacement for arbitrary postorder AST machine-code injection. A
function-constant backend would still require pipeline creation for each
meaningfully different AST shape, and pipeline creation is the cost cuSR is
trying to remove from the hot path.

## Unsupported Binary Path

There is reverse-engineering work around Apple AGX GPUs and the Asahi Linux
driver stack. In principle, one could imagine an AGX backend analogous to
`cusr_ast_sass.h`:

1. Generate or capture a Metal/AGX skeleton.
2. Disassemble AGX shader code.
3. Locate marker islands.
4. Generate AGX machine instructions for postorder ASTs.
5. Patch the binary shader body.
6. Submit the patched shader through a reverse-engineered loader path.

That may be technically possible in a research environment, especially under
Asahi/Mesa where more of the stack is open or reverse-engineered. It is not the
same as a supported macOS Metal product path. On macOS, `metallib` and pipeline
state internals are opaque implementation details. Relying on private driver
formats would be brittle across macOS versions and Apple GPU generations, and
would likely be unsuitable for normal distribution.

## Threadgroup Memory

The Apple equivalent of CUDA shared memory and AMD LDS is Metal `threadgroup`
memory. In MSL this is the address space used for data shared by threads in the
same threadgroup. On the host side, dynamic threadgroup memory can be assigned
with `setThreadgroupMemoryLength`.

This maps well to cuSR's tile-static idea:

- load a row tile and active columns from device memory;
- store the tile in `threadgroup` memory;
- synchronize the threadgroup;
- evaluate many AST+setting combinations using the resident tile;
- reduce directly to SSE/MSE;
- avoid materializing prediction vectors.

Apple GPUs also have a tile-based graphics architecture and imageblock/tile
memory in render/tile-shading paths. For cuSR compute kernels, the relevant
public abstraction is still `threadgroup` memory. Render tile memory may be
interesting for graphics-style workloads, but it is not the natural first target
for symbolic-regression scoring.

The same caution as AMD applies: threadgroup memory is local and fast, but
banking and exact access behavior are not as publicly documented as CUDA shared
memory. A Metal backend would need empirical layout sweeps:

- row-major `[row][column]`;
- padded row-major;
- column-major;
- small swizzles for arbitrary setting-driven column reads;
- bypassing threadgroup memory for low-reuse shapes and relying on Apple GPU
  caches.

Because Apple CPUs and GPUs share unified memory, host-to-device copies can be
cheaper to manage than on discrete GPUs. That does not remove the value of
threadgroup memory. The speedup cuSR is after comes from keeping a small tile in
on-chip memory while many AST+setting scores are computed, not merely from
sharing the same physical DRAM.

## What Can Be Reused

The high-level cuSR design ports cleanly:

- postorder AST representation;
- CPU interpreter for validation;
- settings buffers;
- tile-static MSE scoring;
- packed AST sites per kernel;
- benchmark structure;
- row-evals/s and AST+settings/s metrics.

The machine-code pieces do not port directly:

- no public AGX instruction writer target;
- no public `cubin`/`hsaco` equivalent with stable patchable sections;
- no public register-count metadata patch;
- no supported way to load a mutated shader binary as a Metal pipeline.

## Potential Apple Backend Modes

### Mode 1: Supported Source Generator

Generate MSL for batches of ASTs and compile/cache with Metal.

Pros:

- supported by Apple;
- likely robust across macOS and GPU generations;
- can still exploit threadgroup memory and settings;
- suitable for production distribution.

Cons:

- AST specialization cost depends on Metal compilation and pipeline creation;
- no microsecond AST patching;
- dynamic search may need batching to amortize pipeline creation;
- less compelling if ASTs change constantly.

### Mode 2: Function-Constant Skeleton

Use function constants to specialize operation choices or enable/disable fixed
AST slots.

Pros:

- supported by Metal;
- better than a fully interpreted runtime-dispatch kernel;
- can prebuild/cache common shapes.

Cons:

- function constants still produce specialized functions/pipelines;
- arbitrary ASTs are awkward;
- likely far slower to specialize than cuSR's machine-code patch path;
- may collapse into a template explosion.

### Mode 3: Runtime Interpreter Kernel

Pass AST bytecode to a generic Metal kernel and interpret it per row/setting.

Pros:

- easy and supported;
- no compile latency;
- portable across Apple GPUs.

Cons:

- loses the central cuSR advantage;
- dynamic dispatch inside the hot loop;
- likely much closer to EvoGP-style runtime evaluation than cuSR's fused
  specialized path.

### Mode 4: Reverse-Engineered AGX Patch Backend

Write AGX instructions directly and patch shader binaries.

Pros:

- closest to the NVIDIA AST-SASS architecture;
- in theory could recover near-zero AST specialization;
- could preserve tile-static multi-setting scoring.

Cons:

- unsupported on macOS;
- tied to reverse-engineered AGX ISA and driver formats;
- brittle across OS/GPU revisions;
- unclear how to update resource metadata safely;
- unsuitable for App Store-style distribution.

## Doable Product Path

For a normal user-facing Apple backend, the practical path is:

1. Build a Metal source generator for tile-static MSE kernels.
2. Use many settings per threadgroup to exploit threadgroup-memory reuse.
3. Batch many ASTs per generated Metal source file.
4. Use Metal binary archives to reduce repeat pipeline cost.
5. Treat Apple as a strong runtime backend, not a near-zero compile backend.

This could still be useful. Apple GPUs have high memory bandwidth, unified
memory, efficient threadgroup memory, and excellent laptop/workstation
availability. But it would be a different performance story from NVIDIA
AST-SASS: probably strong throughput after batching, weaker dynamic AST
specialization latency.

## Research Path

If the goal is to reproduce the exact cuSR patching model on Apple hardware, the
research route would be:

1. Study Asahi/AGX disassembly and Mesa's Apple GPU compiler work.
2. Generate very small Metal compute kernels and inspect the resulting AGX code.
3. Identify whether marker constants survive in a stable way.
4. Determine whether patched shader binaries can be loaded through any available
   user-space path.
5. Build a tiny AGX AST writer for `add`, `mul`, `min`, and `max`.
6. Validate on one Apple GPU generation only.

That is feasible as a reverse-engineering project, but it is not the same level
of product feasibility as AMD ROCm. AMD has documented code objects and
machine-readable ISA descriptions. Apple does not publicly provide the same
level of low-level compute ABI.

## Feasibility Read

Apple can absolutely support the cache-local, tile-static, multi-setting part of
cuSR through public Metal. That is the most important runtime idea.

Apple probably cannot support the current near-zero AST machine-code patching
workflow through public APIs. A same-shape toolchain could exist in theory only
as a reverse-engineered AGX backend or if Apple exposed a lower-level shader
binary interface in the future.

So the practical answer is:

- public Metal backend: doable, useful, but not the same compile-latency
  breakthrough;
- private/reverse-engineered AGX backend: theoretically possible, high risk,
  not a normal product path;
- full cuSR-style patching on Apple via official APIs: currently unlikely.

## References

- Metal function constants:
  https://developer.apple.com/documentation/metal/mtlfunctionconstantvalues
- Metal threadgroup memory:
  https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength%28_%3Aindex%3A%29
- Metal binary archives:
  https://developer.apple.com/documentation/metal/mtlbinaryarchive
- WWDC20 "Build GPU binaries with Metal":
  https://developer.apple.com/videos/play/wwdc2020/10615/
- Apple Metal Shading Language Specification:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Asahi AGX notes:
  https://asahilinux.org/docs/hw/soc/agx/
- Dougall Johnson's Apple GPU reverse-engineering notes:
  https://github.com/dougallj/applegpu
