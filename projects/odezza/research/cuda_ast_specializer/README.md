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

# Direct CUDA AST specialization scratch experiment

This non-production experiment reads the same self-delimiting postorder AST corpus used by the Odezza engine benchmark, generates concrete CUDA source for every selected AST, compiles it with NVRTC, inspects the resulting CUBIN, and optionally runs it through the CUDA Driver API.

The generated kernel deliberately retains the current comparison shape:

- four states and four constants;
- three runtime trajectories with eight observation intervals each;
- two fixed RK4 steps per observation interval;
- the same three fixed RHS expressions compiled directly as CUDA and one candidate AST as the fourth RHS;
- `blockIdx.y` selecting a candidate and each thread owning one configuration;
- runtime constant banks and toggle permutations;
- dense ragged trajectory input staged in shared memory; and
- one MSE value per candidate configuration.

The intended structural difference is RHS realization. Odezza normally compiles a reusable branch arena and writes both fixed and candidate postorder AST instructions directly into its CUBIN. This experiment writes the three known benchmark RHS expressions as CUDA, emits a CUDA `switch` containing every candidate AST, and lets NVRTC/PTXAS allocate registers, fuse expressions, and choose scheduling controls across the complete RHS/RK4 body.

Run on a CUDA host:

```bash
python3 scratch/cuda_ast_specializer/cuda_ast_specializer.py \
    --corpus scratch/ast_tools/build/ast-corpus-self-delimiting-random-depth2-1m.oast \
    --system-count 32 \
    --constant-banks 64 \
    --toggle-bits 5 \
    --runs 7 \
    --artifact-directory scratch/cuda_ast_specializer/build/32
```

The JSON report includes CUDA and CUBIN size, compilation time, register count, total SASS instructions, encoded stall-nibble and raw yield-control-bit distributions, common opcodes, and kernel-only throughput. The raw yield bit is intentionally reported without assigning a semantic polarity because NVIDIA documents the disassembler but not the scheduling-control encoding.

Direct CUDA division and transcendental semantics are not automatically identical to the hand-written SASS implementations. `--fast-math` is an explicit, separately reported mode. Performance or numerical comparisons must keep this distinction visible.

NVRTC and the driver cache compiled programs. Use `--disable-cache` when measuring a new-AST cold compilation. A repeated source without that flag is a warm-cache measurement and cannot be reported as the cost of compiling a new candidate batch. `--compile-tag` changes the generated source comment for provenance, but comments alone do not reliably defeat the compiler cache.

The candidate scheduling subsection is based on NVRTC line information. PTXAS may fold, fuse, move, or attribute instructions to surrounding source lines, so it is a useful view of compiler decisions rather than a complete one-to-one accounting of declared postorder instructions.

Measured results and comparison boundaries are recorded in [RESULTS.md](RESULTS.md).
