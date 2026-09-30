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

# Reusable SASS writer

The C99 scoring specializer now emits instructions through [`src/o_sass.h`](../src/o_sass.h). This internal header owns the supported instruction encodings, their control fields, bounded instruction emission, and little-endian byte serialization. It requires the existing Odezza/CUDA headers but has no CUDA calls, allocations, AST dependency, CUBIN inspection dependency, or mutable global state. Functions are inline so the host compiler can fold checks for known operands.

The current boundary is:

```text
AST lowering + register/barrier allocation
                  |
           checked encoders
                  |
    bounded OSassWriter instruction buffer
                  |
     preflighted CUBIN patch destinations
                  |
       bounded little-endian stores
```

The scoring allocator and AST compiler remain in `o_specialize_scoring_cubin.c`. Another C99 kernel or a future C99 LM backend can use the same encoders with its own allocation and lowering policy. The existing Python LM implementation has not been migrated.

## Using it

The API uses ordinary C99 functions. An encoder returns `OSassEncoding`, carrying either two instruction words or an error. `o_sass_writer_emit` accepts only successful encodings, merges pending incoming waits into the next instruction, and advances the buffer only on success.

For example, with valid allocated registers, ready inputs, and dependency barrier zero available. The scratch register must not overwrite the live factor:

```c
#include "o_sass.h"

static OdezzaResult emit_reciprocal_product(
    OSassWriter *writer,
    uint32_t destination,
    uint32_t scratch,
    uint32_t denominator,
    uint32_t factor
) {
    OdezzaResult result = o_sass_writer_emit(
        writer,
        o_sass_mufu(scratch, denominator, 0u, 0u, O_SASS_MUFU_RCP)
    );
    if (result != ODEZZA_SUCCESS) return result;

    /* Wait on barrier zero before reading the MUFU result. */
    return o_sass_writer_emit(
        writer,
        o_sass_fmul_register(destination, scratch, factor, 1u)
    );
}
```

Initialize a writer over caller-owned, 8-byte-aligned `OSassInstruction` storage using `o_sass_writer_init`. The storage capacity is measured in instructions. `writer.count` is the committed instruction count. A failed individual emission preserves that count, the instruction buffer, and pending waits. A multi-instruction helper like the example can have emitted its first instruction when a later emission fails; compile into scratch storage and commit a complete expression only after success.

`o_sass_instruction` extracts a successful encoding into an instruction value. `o_sass_store` writes an already encoded instruction to a checked byte span. It checks the complete 16-byte destination before writing either word, accepts unaligned byte offsets, and always writes little-endian data. It does not decode or validate arbitrary raw SASS words.

## What is protected

- Register parameters use `uint32_t` rather than accepting truncation to `uint8_t`. The current arithmetic subset requires destinations R0–R254 and permits RZ as a source.
- Wait masks, predicate indices, barrier slots, multiplication modes, and MUFU functions are checked before their bits are packed. Invalid control fields produce an error instead of being silently masked.
- Toggle emission validates the compiler-observed predicate-test form and its predicate register. Bit indices must be below 32, so a bad index cannot cause an invalid shift. The captured form stays separate from scoring inspection objects.
- Forward branches support the validated SM89, SM90, and SM120 forms. This writer deliberately limits the accepted displacement to a nonnegative signed-32-bit byte offset and rejects unsupported architectures. The distance argument counts 16-byte instructions from the branch itself. The previous implementation could pass a 64-bit arithmetic check and then truncate the encoded displacement.
- Buffer limits use `count >= capacity`, and failed emission cannot consume a pending dependency wait. Byte stores validate subtraction-based bounds before offset arithmetic.
- The normal specializer preflights every patch destination, including target-table and register-count metadata, before modifying the CUBIN. Malformed metadata cannot make one of the later stores write past the buffer after earlier patches have already happened. Internal callers still own the validity and lengths of metadata arrays.
- The allocator explicitly reserves the permutation and final-output registers. The trusted fresh-copy pipeline still consumes an already validated inspection and preserves its existing fast path.

The scheduling policy is unchanged: the existing ALU and predicate stalls are 12 cycles, and MUFU uses the same barrier encoding and 6-cycle issue stall. The writer does not infer dataflow, allocate registers or barriers, calculate occupancy, or prove that a caller supplied all required waits. Those decisions belong to the layer that knows value lifetimes. Adding an instruction or another architecture requires encoding evidence, boundary tests, and device validation. The current subset does not yet contain the integer arithmetic needed to implement a complete Philox generator.

## Validation, 2026-09-05

- **83 frozen instruction vectors** cover the existing instruction forms, all six MUFU barrier slots, predicate endpoints, RZ sources, toggle bit 31, and branch packing boundaries.
- **3,003 deterministic AST assemblies** cover SM89/90/120 encodings, arithmetic and transcendental lowering, nested expressions, four-way toggles, and a seven-live-MUFU case that exhausts six barrier slots. Every complete synthetic CUBIN is byte-identical to the pre-refactor version. The aggregate SHA-256 is frozen in `tests/o_sass_corpus_test.c`.
- A **native SM120 CUBIN**, compiled with the production NVRTC options (`--std=c++17`, `--gpu-architecture=sm_120`), is byte-identical after specialization with either version: `468ede8359823409b326adcce9bf7c549deb77e6f9d8dbbc0d625b785ac69542`. Both report 48 registers and 17 candidate-body instructions for that fixture. An additional comparison using the older fixture's fast-math flags also matched; it is a separate fixture.
- **20/20 CMake tests pass on Rohini/SM120 and Ada/SM89.** Makefile C and artifact/CLI targets also pass. No SM90 device was tested; its instruction encodings are covered by host tests.
- AddressSanitizer and UndefinedBehaviorSanitizer pass for the writer and the AST/patch-metadata tests. Bad offsets, control fields, oversized branches, and capacity failures have explicit rejection tests, including unchanged-buffer assertions.

The full historical `make test` still has the separate legacy scratch-packer incompatibility recorded in [the scoring review](scoring_kernel_review_2026-09-05.md). This change does not remove that check or claim to repair that older runner.

## Performance

These measurements compare against the code **after the preceding scoring-kernel corrections**. CUDA generation, the RK4 topology, and valid specialized instruction bytes are unchanged by this refactor.

Rohini, five alternating repetitions per version:

| Fixture | Before | After |
|---|---:|---:|
| Many constant banks, end-to-end scoring | 435.122 M configurations/s | 435.226 M/s |
| Many candidate systems, end-to-end scoring | 505,213 configurations/s | 505,063/s |
| CPU fresh-copy specialization, seven-MUFU fixture | 236.40 ns | 216.27 ns |

The two scoring differences are below 0.03%; these measurements show no meaningful end-to-end throughput change. The CPU fixture improved by about 8.5%, which should not be generalized to every AST. CPU timing includes reset of an 8 KiB synthetic CUBIN and specialization; it excludes CUDA execution.

[Raw measurements and provenance](../benchmarks/raw/2026-09-05-sass-writer.json) and [reproduction details](../benchmarks/raw/2026-09-05-sass-writer-method.txt) retain the exact workloads. Reuse this boundary first; measure any future scheduling or register-allocation change separately from encoding cleanup.
