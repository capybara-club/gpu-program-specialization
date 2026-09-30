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

# Secant: Direct Native Specialization of Embedded GPU Programs

## Summary

Secant is a technique for changing a small arithmetic program inside a much
larger GPU kernel without interpreting the program or invoking a compiler for
every change. The surrounding kernel is compiled normally once. Secant then
lowers compact typed ASTs directly to native GPU instructions and writes them
into validated instruction regions in copies of that compiled binary.

The result combines very low specialization latency with execution close to a
native fused kernel. Data loading, tiling, reductions, and solver state remain
compiler-optimized, while the changing expression executes from registers and
does not require intermediate arrays in device memory.

## The problem

A workload may have a stable numerical structure but millions of changing
expressions. Symbolic search is one example: every candidate uses the same data
layout and scoring reduction, but has different arithmetic.

Common execution strategies each impose a cost:

- An interpreter decodes the expression and dispatches every operation at run
  time.
- Source or PTX generation produces native code, but repeatedly invokes a
  comparatively expensive compiler.
- Tensorized execution turns operations into separate kernels or materialized
  intermediates, adding launches and global-memory traffic.

Secant removes these costs by treating the expression as a small native patch
inside a stable fused kernel.

## Technique

### 1. Generate and compile a kernel skeleton

A recipe describes a fixed kernel shape: its inputs, outputs, tiling, reduction
topology, program capacity, and number of expression sites. Secant emits CUDA
source for that shape. Each expression site is an inline-assembly scaffold
containing:

- `BRKPT` instructions that reserve a contiguous region of fixed-width SASS;
- arithmetic markers with unique immediate values that expose the physical
  registers assigned to inputs and outputs; and
- keepalive arithmetic and a shared-memory store that prevent the compiler
  from deleting or coalescing the scaffold.

The ordinary CUDA toolchain compiles this skeleton once for a target GPU
architecture. It remains free to optimize the surrounding kernel and choose
physical registers.

### 2. Inspect the native binary once

Secant parses the resulting CUBIN and locates each expected kernel and patch
site. It verifies the ELF structure, architecture, function bounds, register
metadata, exact breakpoint layout, marker immediates, physical register
relationships, and the absence of unexpected instructions or spills inside the
scaffold.

Inspection produces an immutable plan containing file offsets, patch capacity,
input and output registers, reusable scratch registers, incoming dependency
barriers, and register-count metadata. The plan refers to offsets rather than
one mutable binary, so it can specialize independent CUBIN copies concurrently.

### 3. Lower each AST directly to SASS

A Secant program is a compact, return-terminated, typed postorder AST. The
specializer validates and lowers it in one traversal. A small stack allocator
maps AST values to immediate operands, scaffold registers, or temporary
registers. The native encoder emits the required arithmetic instructions and
their scheduling, stall, wait-barrier, and functional-unit control fields.

If the program uses less than the reserved region, Secant emits a native branch
over the unused instructions. It replaces the leading breakpoint with a no-op
and updates CUBIN register-count metadata when temporary registers extend the
original allocation. The binary is patched in place without allocation,
linking, relocation, or a compiler invocation.

### 4. Load and execute the specialized module

The specialized CUBIN is loaded through the normal GPU driver. The expression
now executes as straight-line native instructions inside the fused kernel. The
surrounding kernel can consume its result immediately for materialization, SSE,
affine or Gram statistics, or optimization state.

Specialization and module loading are distinct costs: Secant removes native
compilation from the hot path, but it does not make driver loading free. Its
bulk runners therefore pipeline CPU specialization, reuse resident data and
worker resources, and keep a specialized module loaded across settings,
datasets, or optimizer iterations whenever the workload permits.

## Runtime settings

One specialized AST can represent many concrete functions. Runtime leaf slots
may select different input columns or constants, allowing a search to evaluate
many bindings and constant starts without producing another binary. Settings
both expand the search represented by one AST and provide enough homogeneous
work to amortize specialization, module loading, and data-tile loads.

## Why execution remains fast

Secant changes only the arithmetic that must vary. The compiler still owns the
large kernel structure, memory access, synchronization, and reduction code.
The specialized expression uses native instructions and register-local
intermediates, so there is no interpreter dispatch and no feature tensor to
write to and reread from global memory. CUDA and PTX generators plus a portable
CPU interpreter provide matched correctness and performance controls.

## Scope and limitations

Secant is not a general replacement for CUDA or a general-purpose binary
rewriter. It requires a controlled expression instruction set, a fixed kernel
ABI, bounded patch capacity, and validated encodings for each supported GPU
architecture. A new kernel topology or architectural family requires a new
compiled skeleton and binary contract. Driver module lifecycle can still
matter for very short kernels.

Within that domain, the central claim is:

> Compile the stable kernel once, specialize only its changing arithmetic
> directly in the native binary, and obtain both near-native execution and
> specialization cheap enough for high-volume program search.
