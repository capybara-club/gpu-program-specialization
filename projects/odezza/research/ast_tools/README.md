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

# Temporary AST-space tools

This directory contains non-production C99 helpers for exploring the Odezza postorder AST space. It is deliberately separate from the public library and does not depend on CUDA.

The iterator treats states, constant slots, inline literals, and optional two-way or four-way toggles as terminal-like expressions. A terminal has tree depth zero. Every selected unary, binary, or ternary operator adds one level above its deepest child. Enumeration is syntactic: commutative operand permutations and algebraically equivalent expressions remain distinct.

Selection and depth are independent:

- `--depth-mode up-to` includes every expression no deeper than `--depth`.
- `--depth-mode exact` includes only expressions whose depth equals `--depth`.
- `--mode exhaustive` visits the complete space.
- `--mode truncate --count N` visits its deterministic first `N` ranks.
- `--mode random --count N --seed S` visits `N` unique ranks through a deterministic seeded permutation.
- `--mode structural-random --count N --seed S` directly samples trees without requiring the expression-space count to fit in 64 bits.

The iterator writes one AST into reusable stack storage and calls the visitor before reusing that storage. It does not allocate per AST. Exhaustive and random selection require the complete space to fit in a 64-bit count. Prefix truncation can still operate on a saturated depth-at-most space; exact-depth traversal requires an exact count so exclusion of shallower trees remains correct.

Structural-random mode is intended for heavier spaces where ranked selection overflows. It chooses an operator family and child depths directly while guaranteeing the requested exact depth. It is deterministic for a given seed, but it may contain duplicates and is deliberately not presented as a uniform sample over all syntactic expressions. Terminal families (direct leaf, two-way toggle, and four-way toggle) and operators are sampled uniformly by family before their operands are sampled.

## Complete systems

`OAstSystemSpaceConfig` assembles independently configured RHS spaces into complete candidate systems. Every RHS can have its own depth mode, depth, operator set, literal set, and toggle policy, while all RHS expressions share the state, constant, and toggle register-bank dimensions required by one executable system. State-output indices must be unique.

The system iterator is allocation-free. The caller first queries `o_ast_system_space_workspace_size`, supplies that reusable arena, and receives one temporary array of RHS program views per callback. The command-line helper applies one expression-space configuration to every RHS selected by `--rhs-count`:

```bash
scratch/ast_tools/build/odezza-ast-tools \
    --states 4 \
    --rhs-count 4 \
    --constants 4 \
    --literal 0 \
    --literal 1 \
    --operators add,sub,mul,div,neg,sin,cos \
    --toggle-bits 5 \
    --include-toggle2 \
    --depth 4 \
    --depth-mode exact \
    --mode structural-random \
    --count 1000 \
    --seed 20260903 \
    --no-evaluate
```

Each output line is one JSON system containing an ordered `rhs` array. Structural-random records use `rank: null` because they are not selected by a combinatorial-space rank.

The independent CPU interpreter accepts an explicit toggle permutation. Bit `i` of `--permutation` selects the corresponding `TOGGLE2` choice; `TOGGLE4` uses its first declared bit as the low selection bit and its second as the high bit. Basic FP32 arithmetic is suitable for exact spot checks. Transcendental GPU instructions may differ slightly from the host math library and should be compared with a tolerance.

Build and test:

```bash
make -C scratch/ast_tools test
```

Example:

```bash
scratch/ast_tools/build/odezza-ast-tools \
    --states 3 \
    --constants 2 \
    --operators add,sub,mul,div,neg \
    --toggle-bits 3 \
    --include-toggle2 \
    --depth 2 \
    --depth-mode exact \
    --mode random \
    --count 100 \
    --seed 42 \
    --state-values 1,2,3 \
    --constant-values 0.5,-1 \
    --permutation 5
```

Each stdout line is one JSON record containing the rank, actual depth, encoded program bytes, and CPU result. The final count summary is written to stderr so stdout remains pipeable.

## Binary files

Use `--binary-output` to stream generated programs into a binary file without materializing the corpus in memory:

```bash
scratch/ast_tools/build/odezza-ast-tools \
    --states 4 \
    --constants 4 \
    --literal 0 \
    --literal 1 \
    --operators add,sub,mul,div,neg \
    --toggle-bits 5 \
    --include-toggle2 \
    --depth 2 \
    --depth-mode exact \
    --mode random \
    --count 1000000 \
    --seed 20260901 \
    --binary-output programs.oast
```

`--verify-binary programs.oast` streams through every AST and checks the complete file structure without printing the programs. `--binary-input programs.oast` rereads the programs and emits the same JSON representation used by direct generation. State values, constant values, toggle-bit count, and permutation are supplied with the existing evaluation options when CPU results are wanted. Serialized files preserve AST order, not enumeration ranks; the reread JSON `rank` is the zero-based file position.

The version-1 file format is deliberately simple and independently parseable:

- All integers are little-endian.
- The 24-byte file header contains only the eight-byte `ODEZZAAS` magic, a `u32` version, a `u32` maximum AST byte length, and a `u64` AST count.
- The remaining bytes are variable-width postorder ASTs concatenated directly with no per-AST metadata or padding.
- Opcodes determine their immediate widths and `RETURN_F32` terminates each AST. The reader therefore reconstructs every boundary and tree depth from the instruction stream.
- The reader rejects bad magic, unsupported versions, malformed programs, truncation, unexpected maximum lengths, and trailing bytes.

The format is a scratch interchange format rather than a committed production or network protocol.

With `--rhs-count N`, binary output writes the RHS programs for each system consecutively in the same minimal AST format. The file intentionally does not add system metadata: consumers must be given the matching RHS count, and full systems are reconstructed from fixed groups of `N` programs. This preserves the previously chosen 24-byte file header.

See [COMPRESSION_RESULTS.md](COMPRESSION_RESULTS.md) for a reproducible one-million-program size experiment.

## Engine benchmark

The CUDA-dependent benchmark sends a binary AST corpus through the complete scoring pipeline and independently replays sampled trajectory scores with the C99 interpreter:

```bash
make -C scratch/ast_tools engine-benchmark
CUDA_MODULE_LOADING=EAGER scratch/ast_tools/build/odezza-ast-engine-benchmark \
    --corpus scratch/ast_tools/build/programs.oast \
    --rhs-per-system 1 \
    --system-count 1000000 \
    --systems-per-module 256 \
    --constant-banks 8 \
    --workers 1 \
    --slots-per-worker 2 \
    --loaded-modules 2 \
    --streams 2 \
    --runs 3 \
    --spot-checks 1024
```

Its timed scope includes C99 specialization, eager module load, function lookup, fused RK4 execution, retirement, and module unload. Corpus parsing, host/device allocation, input upload, score download, and CPU replay are reported outside the engine timing. Exact-operation ASTs receive strict complete-trajectory CPU checks. ASTs containing device-approximate division or transcendental operations are skipped by that strict check because their rollout can legitimately diverge from host libm near singularities.

The current engine fixture has four states. `--rhs-per-system 1..4` interprets consecutive programs as one candidate system and leaves the earlier state derivatives fixed only when fewer than four are generated. It reports complete systems/s, RHS ASTs/s, and configurations/s separately.

See [ENGINE_BENCHMARK_RESULTS.md](ENGINE_BENCHMARK_RESULTS.md) for the first tuning sweep and recommended defaults.
