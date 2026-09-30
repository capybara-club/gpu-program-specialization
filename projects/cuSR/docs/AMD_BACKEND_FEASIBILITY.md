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

# AMD Backend Feasibility

## Current gfx1201 Proof

The first executable AMD proof now lives under [`amd`](../amd/README.md). It was
compiled and run on an RX 9070 XT with ROCm/HIP 7.1 and Ubuntu LLVM 21.

The HIP compiler produced a contiguous wave32 expression island delimited by
`s_nop imm16` sentinels. An explicit `s_wait_loadcnt 0` completed the global
input loads before the island, and no compiler-generated instruction appeared
inside the sentinel range. A C99 ELF parser recovered the compiler-selected
input and output VGPRs, patched either `v_add_f32_e32` or `v_mul_f32_e32` into
the HSACO, loaded the modified code object, and matched 1,024 CPU reference
results for both operations.

This is evidence for the core code-object mutation strategy on RDNA4, not yet a
complete AST backend. In particular, it does not yet expand VGPR metadata or
emit arbitrary postorder programs.

This note sketches how the cuSR AST patching architecture could map to the AMD
ROCm/HIP ecosystem. The short version is: it looks doable, especially if the
first target is ROCm/CDNA, but it should be treated as a separate backend rather
than a small port of the NVIDIA SASS path.

## Core Idea

cuSR no longer depends on PTX as the dynamic expression format. The important
system boundary is now:

1. Generate a tile-static GPU kernel skeleton.
2. Compile that skeleton once to a loadable device code object.
3. Inspect the compiled code object and record patchable AST islands.
4. Generate machine-code AST bodies directly.
5. Patch those bodies into the code object, update resource metadata, load, and
   run.

That architecture can exist on AMD. The equivalent NVIDIA `cubin` object would
be an AMD HSA code object (`hsaco`) produced by HIPRTC/clang/COMGR. The
equivalent SASS writer would be an AMDGCN/RDNA/CDNA ISA writer.

The strongest reason this is plausible is that AMD documents much more of this
surface. LLVM documents AMDGPU code-object structure, ELF notes, kernel
metadata, kernel descriptors, and register-count fields. AMD also publishes
machine-readable ISA XML files for recent CDNA and RDNA architectures.

## Likely AMD Pipeline

The first AMD version should mirror the current cuSR shape:

1. Generate HIP source for a tile-static MSE kernel.
2. Use HIPRTC to compile source to an AMD code object.
3. Parse the ELF code object.
4. Find each kernel symbol and its code range.
5. Find each kernel descriptor and resource fields.
6. Decode the AMDGPU metadata note.
7. Locate AST skeleton islands using marker constants and a strict instruction
   pattern.
8. Generate AMD ISA for each postorder AST.
9. Patch the island bytes in place.
10. Patch the kernel descriptor and metadata if VGPR/SGPR counts increase.
11. Load the patched code object through HIP module APIs and launch.

HIPRTC is the right analogue to NVRTC. ROCm documents that HIPRTC accepts
kernel source strings and uses COMGR internally for compiling, linking, and
inspecting code objects.

## Code Object Pieces To Parse

The AMD code object is an ELF file. For the ROCm/HSA path, the important pieces
are:

- ELF header and `e_flags`, which identify the AMDGPU machine/target.
- Symbol table entries for kernels and kernel descriptors.
- `.text` sections containing executable machine code.
- Kernel descriptor data, typically referenced by symbols such as
  `<kernel>.kd`.
- `.note` records containing `NT_AMDGPU_METADATA`.
- MessagePack metadata for code object V3+.

LLVM documents that AMDHSA code object V5 is the default if not specified, and
V6 is available through `-mcode-object-version=6`. The parser should therefore
key behavior from the actual code-object version rather than hard-coding one
format.

The NVIDIA path patches an `EIATTR_REGCOUNT`-style register count. The AMD path
will need to patch the kernel descriptor fields that feed `COMPUTE_PGM_RSRC1`
and probably the metadata `.vgpr_count` / `.sgpr_count` fields as well. LLVM
documents `.amdhsa_next_free_vgpr` and `.amdhsa_next_free_sgpr` as the assembler
inputs used to calculate the hardware resource fields. In a patcher, we would
not have the assembler doing that final calculation, so cuSR needs a small
resource-field encoder per supported `gfx` generation.

## Skeleton Strategy

The skeleton should be generated from HIP/C++ rather than from an AMD assembly
template at first. That keeps the tile-static kernel readable and lets the
compiler handle address calculation, LDS setup, launch ABI, and normal control
flow.

Inside the kernel, each AST site should reserve an island that is easy to parse:

- Load the eight runtime leaf values into VGPRs.
- Emit a known marker operation for each input register.
- Emit an explicit wait/synchronization boundary before the AST island.
- Emit a long no-op or dummy arithmetic region sized for the maximum AST body.
- Emit a result marker that identifies the final output register.

On NVIDIA we had to fight hidden scheduler/control bits. AMD should be cleaner
because memory dependencies are more explicitly represented through wait
instructions such as `s_waitcnt`, and instruction encodings are documented. The
safe first design should still be conservative:

- put an explicit wait after global/LDS loads before the AST island;
- keep AST values in VGPRs;
- avoid adding new SGPR usage in the AST writer unless the descriptor patcher
  supports it;
- use conservative NOP/delay sequences around special-function instructions
  until per-architecture probes show what can be removed.

The site discovery logic should remain strict. If the patcher sees an
unexpected instruction inside an island, it should return an error rather than
guessing. That is the same lesson as the NVIDIA backend: the patcher should
accept only compiler output that matches a known skeleton contract.

## AMD Shared Memory: LDS Versus SMEM

The AMD equivalent of CUDA shared memory is Local Data Share, usually called
LDS. In HIP source this is the memory used for workgroup-local storage, and in
ISA it is accessed with `ds_*` instructions. AMD documentation and tools may
also use "SMEM" to mean scalar memory instructions such as `s_load_*`; that is
not the same thing as CUDA shared memory. For cuSR, the tile-static data cache
maps to LDS, not to scalar memory.

The AMD tile-static kernel should therefore think in these terms:

- global column data is loaded by work-items from HBM through the vector memory
  path;
- the row tile is staged into LDS using compiler-generated `ds_write*`
  instructions;
- AST evaluation reads tile values from LDS using `ds_read*` instructions;
- uniform kernel arguments, pointers, and loop bounds may arrive through SGPRs
  and scalar memory loads, but the AST writer should avoid touching those.

LDS is banked, so the NVIDIA shared-memory layout assumptions cannot be copied
blindly. Bank count, wave mode, and access phasing vary by architecture and
instruction width. A layout that is conflict-free for a 32-lane NVIDIA warp is
not automatically conflict-free for an AMD wave32 or wave64. AMD's CK-Tile
documentation explicitly treats LDS bank conflicts as a first-order performance
issue and uses padding or XOR-style index transformations to avoid pathological
access patterns.

For cuSR, the access pattern is different from a GEMM tile. Each work-item owns
a runtime setting and may read arbitrary columns for each leaf. That means the
LDS layout should be chosen empirically, not assumed. Candidate layouts:

- row-major `[row][column]` with padded column stride;
- row-major with an XOR swizzle on column index;
- column-major `[column][row]` if coalesced LDS reads matter more than simple
  stores;
- bypass LDS for low-reuse shapes and rely on vector L0/L1 cache instead.

The first AMD implementation should keep the current cuSR principle but avoid
overfitting:

1. Load a `rows x active_columns` tile into LDS.
2. Use a normal HIP workgroup barrier after the tile load.
3. Preserve or emit the required wait sequence before entering the AST island.
4. Patch only the AST island, not the LDS load/store setup.
5. Benchmark row-major padding versus XOR/swizzled indexing per `gfx` target.

The main correctness rule is that the patchable AST island should start only
after all LDS writes needed by the tile are visible to the workgroup. The main
performance rule is that random setting-driven column reads can still create
bank conflicts, so settings amplify tile reuse but do not automatically solve
LDS banking. The AMD backend will need the same kind of tile-layout sweep that
cuSR did on NVIDIA.

## AST ISA Writer

An AMD AST writer would be the peer of `cusr_ast_sass.h`, not a replacement for
the existing CPU AST representation.

Expected first operations:

- `input(i)`
- `constant(f32)`
- `add`
- `mul`
- `fma` if useful
- `min`
- `max`
- `abs`
- `neg`
- `rcp`
- `rsqrt`
- `sqrt`
- `exp2` / `log2` where available
- `sin` / `cos` where available
- safe routines such as `safe_div` and `safe_sqrt`

The writer should initially be VGPR-only. Inputs are already VGPR values, and
intermediate AST results naturally live in VGPRs per work-item. If immediate
constants are awkward because of literal-constant encoding limits, the backend
can use one of these policies:

- reserve literal slots inside the AST island;
- materialize constants into temporary VGPRs;
- use a small per-site constant table loaded into VGPRs before the island;
- restrict the first backend to constants that encode cleanly.

The register allocator is similar to the NVIDIA AST-SASS allocator:

- consume input VGPRs without owning them;
- use reusable skeleton VGPRs first;
- allocate overflow VGPRs beyond the original kernel count;
- report the new max VGPR count so the code-object resource fields can be
  patched.

SGPRs are the riskier path because AMD uses SGPRs for kernel arguments, dispatch
state, pointers, masks, and scalar control. A first backend should avoid new
SGPR allocation inside AST islands.

## Hazard And Wait Model

This is the main technical investigation area.

AMD should be less opaque than NVIDIA because there is no NVIDIA-style hidden
per-instruction control word that must be inferred from `ptxas`. But "less
opaque" does not mean "free." A raw machine-code writer still has to respect:

- VMEM and LDS wait counts;
- VALU-to-special-function latency;
- generation-specific hazards;
- wave32 versus wave64 behavior;
- branch encodings and relative offsets;
- instruction-size differences from literal constants.

The first robust strategy should over-synchronize:

1. Let the compiled HIP skeleton do all global and LDS address work.
2. Insert or preserve a known `s_waitcnt` / barrier sequence before the AST
   island.
3. Generate the AST island assuming all input VGPRs are ready.
4. Emit conservative waits or NOPs for special-function sequences.
5. Patch the tail so the next compiled instruction sees the final result VGPR.

Then build probes per `gfx` architecture to reduce stalls. This is similar to
what the NVIDIA backend did, except AMD's public ISA should make the probes more
about performance and less about discovering undocumented bitfields.

## Doable Parts

These pieces are straightforward engineering:

- generating HIP source for tile-static kernels;
- compiling with HIPRTC;
- parsing ELF section headers and symbol tables;
- locating `.text` code ranges;
- locating kernel descriptors;
- locating marker constants in machine code;
- building a postorder AST interpreter for CPU validation;
- comparing patched GPU output against CPU output;
- adding a benchmark harness parallel to the CUDA path.

The current cuSR architecture already proves the shape of the problem. The AMD
version would reuse the same high-level concepts: tile-static kernels, runtime
settings, AST sites, postorder programs, CPU validation, patch statistics, and
module-load benchmarks.

## Hard Parts

The hard parts are AMD-specific, not conceptual:

- encoding and decoding the exact ISA subset for each target `gfx` generation;
- handling literal constants and instruction-size variation;
- patching VGPR/SGPR resource fields in kernel descriptors correctly;
- keeping metadata consistent enough for ROCm loaders and tools;
- proving wait/hazard safety for CDNA and RDNA targets;
- supporting both wave32 and wave64 without corrupting the settings topology;
- avoiding accidental SGPR clobbering;
- deciding whether to target only ROCm/HSA or also PAL/Mesa code-object paths.

The best initial scope is ROCm/HSA only, CDNA first, with one wave mode and one
code-object version. Consumer RDNA can come later once the machinery is proven.

## Why AMD Might Be Cleaner Than NVIDIA

The NVIDIA path worked, but it required reverse-engineering enough scheduling
control behavior to emit correct SASS. AMD has two advantages:

- official LLVM documentation for code objects, metadata, kernel descriptors,
  and resource fields;
- official machine-readable ISA XML and decoder tooling for recent AMD GPU
  architectures.

That does not eliminate backend work, but it changes the work from "infer hidden
encoding behavior" toward "implement documented encodings and validate
generation-specific hazards."

## Recommended Prototype Plan

1. Generate a tiny HIP kernel with one patch island and eight VGPR inputs.
2. Compile with HIPRTC for one target, probably `gfx90a`, `gfx942`, or the
   available local AMD GPU.
3. Dump and decode the code object.
4. Find the kernel symbol, descriptor, metadata, and island markers.
5. Patch a simple `add/mul/min/max` AST into the island.
6. Update VGPR count in the kernel descriptor and metadata.
7. Load the code object and compare GPU output against CPU.
8. Add one special operation such as `rsqrt` or `rcp`.
9. Add tile-static MSE and settings.
10. Only then optimize waits, branches, and packing.

The prototype should deliberately avoid full symbolic-regression complexity. A
single patched AST returning `x0 * x1 + x2` is enough to prove the code-object
patching and resource-update path.

## Feasibility Read

This is doable. It is not a weekend wrapper around HIPRTC, but it is also not a
research project from scratch. The CUDA path already established the important
systems idea: patchable specialized AST islands inside tile-static multi-setting
kernels. AMD's ecosystem appears to expose enough official binary and ISA
surface to build an analogous backend.

The riskiest part is not "can we generate AMD instructions?" It is keeping the
patched code object acceptable to the ROCm loader while updating all resource
counts correctly and handling hazards across `gfx` generations. If we constrain
the first version to ROCm/HSA, one code-object version, one or two CDNA targets,
VGPR-only AST code, and conservative waits, the path looks very plausible.

## References

- LLVM AMDGPU backend usage and code-object documentation:
  https://llvm.org/docs/AMDGPUUsage.html
- ROCm HIPRTC documentation:
  https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html
- AMD machine-readable GPU ISA documentation:
  https://gpuopen.com/machine-readable-isa/
- ROCm CK-Tile LDS bank conflict documentation:
  https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html
