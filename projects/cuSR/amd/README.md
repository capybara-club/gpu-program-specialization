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

# cuSR AMD Probe

This subproject investigates a ROCm/HIP backend for cuSR without requiring the
CUDA-only top-level build. The first target is the Radeon RX 9070 XT
(`gfx1201`, wave32).

The initial probe compiles a standalone HSA code object containing:

- a compiler-isolated expression island delimited by `s_nop imm16` sentinels;
- fixed-width `s_nop` patch space;
- explicit `v_add_f32` and `v_mul_f32` donor instructions.

Configure and build on the Radeon host with:

```bash
cmake -S amd -B build-amd -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-amd
```

Generated code and disassembly are written under `build-amd/generated`.

Run the automated in-place patch validation with:

```bash
ctest --test-dir build-amd --output-on-failure
```

## RX 9070 XT Result

The first probe passed on ROCm/HIP 7.1 targeting `gfx1201`:

```text
device=0 name=AMD Radeon RX 9070 XT arch=gfx1201
patch op=add island_file_offset=0xfb0 input_regs={v3,v2} result_reg=v4 passed=yes
patch op=mul island_file_offset=0xfb0 input_regs={v3,v2} result_reg=v4 passed=yes
```

The generated code object reports wave32, 6 VGPRs, 10 SGPRs, and no spills for
the probe kernel. LLVM emitted an explicit `s_wait_loadcnt 0` immediately before
the start sentinel. The sentinel, input anchors, 32 `s_nop` instructions, result
anchor, and end sentinel remained one contiguous region with no interleaved
address or control instructions.

The C99 test performs the full proof automatically:

1. Parse the little-endian AMDGPU ELF image.
2. Resolve `cusr_amd_island_probe` through its symbol table.
3. Find and validate the unique sentinel-delimited island.
4. Decode the compiler-selected input and result VGPRs.
5. Encode either `v_add_f32_e32` or `v_mul_f32_e32` with those registers.
6. Patch a private copy of the HSACO in place.
7. Load the modified image with `hipModuleLoadData`.
8. Compare 1,024 GPU outputs against the corresponding CPU operation.

This establishes same-register compact VALU patching only. A general AMD
backend still needs postorder AST generation, temporary VGPR allocation,
arbitrary constant handling, kernel resource-field updates when VGPR use grows,
the complete f32 operation set, and a production tile-static kernel skeleton.
