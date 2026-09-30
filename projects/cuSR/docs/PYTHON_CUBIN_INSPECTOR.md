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

# Python Cubin Inspector

## Conclusion

The Python cubin inspector is materially simpler than the C99 implementation,
although it does not make the underlying inspection problem trivial.

`pyelftools` owns generic ELF64 header, section, string-table, and symbol-table
parsing. cuSR keeps the NVIDIA `.nv.info` and SASS patch-site logic because
those records are part of cuSR's architecture-specific correctness boundary.
The exact line count is less important than the ownership and allocation
machinery removed from the Python path.

Inspection is performed once for a compiled kernel skeleton and is not part of
the AST patch hot path. Python performance is therefore not a concern here.

## What Python Removes

Python eliminates or substantially reduces:

- the workspace-size pass;
- arena allocation and alignment calculations;
- caller-owned workspace and pointer-lifetime contracts;
- checked integer addition and multiplication helpers;
- manual little-endian integer readers;
- most C range and allocation bookkeeping;
- separate count and fill traversals;
- fixed-capacity arrays for kernels, sites, registers, and regcount records;
- C string-table and section-name pointer management;
- a large result-code and cleanup surface.

Python integers avoid arithmetic overflow, while `pyelftools` provides bounded
ELF decoding. `struct` remains useful for the small set of fixed-width SASS
words cuSR decodes. Lists and dataclasses can grow as kernels and site
occurrences are discovered, so inspection only needs one logical traversal.

## What Python Does Not Remove

The architecture-sensitive work remains:

- validating a 64-bit little-endian ELF cubin;
- locating the symbol table, string table, and kernel text sections;
- resolving each generated kernel symbol and its instruction range;
- finding every `EIATTR_MAXREG_COUNT` record in NVIDIA info sections;
- decoding the small set of SASS fields used by the skeleton ABI;
- locating marker `FADD` instructions and their registers;
- collecting incoming wait masks;
- identifying target and SSE accumulator registers;
- proving that the complete marker/store/`BPT` region has the expected shape;
- rejecting interleaved, missing, duplicated, or malformed sites;
- recording exact cubin offsets for patching and register-count updates.

This site-validation logic is the real correctness boundary. It should remain
fail-closed and should not be weakened merely because Python is memory-safe.

## Python Result

The Python inspector returns an immutable logical patch plan containing:

- SASS architecture;
- cubin size or a template identity hash;
- kernel text ranges;
- register-count value offsets for every kernel;
- site start and end offsets;
- eight input registers per site;
- one target register per site;
- all SSE accumulator registers up to kernel capacity;
- reusable skeleton registers;
- the incoming wait mask;
- the number and ordering of physical site occurrences.

The plan uses Python dataclasses for inspection and diagnostics. Before the hot
path, it is converted once into native records owned by the extension.

## Native Boundary

The nanobind extension should not parse ELF or manage CUDA. Its narrow job
should be:

1. accept a validated, packed inspection plan;
2. accept contiguous AST programs and routines;
3. lower ASTs to SASS;
4. write generated prefixes into mutable cubin buffers;
5. update kernel register-count records;
6. return patch statistics.

The bulk patch call should release the GIL and process all modules and kernels
inside one native invocation. Python should not loop over individual ASTs or
individual SASS instructions.

The native patcher must retain inexpensive cubin range checks even when the
plan came from the package's Python inspector. This prevents stale metadata or
the wrong cubin template from becoming an out-of-bounds write.

## Validation Strategy

The Python inspector is kept alongside the C inspector as a differential
reference while this path is under development.

For every existing cubin in the architecture corpus, compare:

- architecture;
- kernel names and text ranges;
- site counts and occurrence ordering;
- site start and end offsets;
- input, target, SSE, and reusable registers;
- incoming wait masks;
- register-count offsets and values.

The corpus should include skeletons compiled for `sm_80`, `sm_86`, `sm_89`,
`sm_90`, `sm_100`, and `sm_120`, multiple kernel counts, multiple AST
capacities, and any intentionally unrolled-site probes.

Truncated and deliberately corrupted cubins should also be tested. Both
inspectors should reject malformed input, although their exact error messages
do not need to match.

After the Python inspector reaches parity across the corpus and the complete
GPU test suite passes with its metadata, the C inspector should be deleted.
Maintaining two production parsers would create more complexity than the
Python rewrite removes.

## Implemented Boundary

Use `pyelftools` for generic ELF parsing and keep NVIDIA info records and
SASS-site validation in cuSR. The C99 inspector remains an independent
differential oracle while the Python path is still under development.

This gives the Python package a clean division:

- Python generates CUDA, compiles and loads cubins, inspects templates, and
  orchestrates launches.
- Native code performs only the high-throughput AST-to-SASS patch operation.
