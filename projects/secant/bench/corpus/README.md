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

# Portable Benchmark Corpora

The JSON files in this directory are deterministic expression workloads shared
by SECANT, native AVX2, SymbolicRegression.jl, EvoGP, Kozax, and Operon. A
benchmark comparison is valid only when `corpus`, `corpus_hash`, and `seed`
match.

## Checked-In Profiles

| File | Profile | Postorder instructions including return | Hash |
|---|---|---:|---|
| `portable_alu_v1.json` | `balanced-alu` | 16 | `5f790c4e4d7bc60a` |
| `portable_mufu_v1.json` | `balanced-mufu` | 24 | `862c74ba3da644b7` |
| `portable_mixed_depth_v1.json` | `mixed-depth` | 4, 6, 8, 10, 12, 14, or 16 | `ad9404796983ee68` |
| `portable_constants_v1.json` | `constants` | 16 | `b662f9e1be41e33f` |
| `portable_safe_math_v1.json` | `safe-math` | 24 | `f5302a03febf2de3` |

Every checked-in corpus contains 1,024 unique expressions, uses eight available
input columns, selects expression zero as its target, and uses the
`secant_hash32_f32_v1` input generator.

- `balanced-alu` reduces inputs 0 through 7 exactly once with a balanced tree
  of add, multiply, minimum, and maximum operations.
- `balanced-mufu` applies one of sin, cos, tanh, safe square root, or safe
  reciprocal square root to each input before the balanced ALU reduction.
- `mixed-depth` contains ordered full binary trees with two through eight
  leaves. Inputs may repeat, and subtraction joins the ALU operation set.
- `constants` uses balanced eight-leaf trees containing at least one input and
  one exact dyadic Float32 constant from `{-2, -1, -0.5, -0.125, 0.125, 0.5,
  1, 2}`.
- `safe-math` applies abs, tanh, safe square root, or safe reciprocal square
  root to every input and reduces with add, multiply, minimum, maximum, and
  safe division.

Protected operations use one shared epsilon:

```text
safe_div(x, y)   = x * y / (y * y + 0.01)
safe_sqrt(x)     = sqrt(abs(x) + 0.01)
safe_rsqrt(x)    = 1 / sqrt(abs(x) + 0.01)
```

## Representation

Expressions are recursive JSON arrays:

```text
input i:          i
constant c:       ["constant", c]
unary op:         ["op", argument]
binary op:        ["op", left, right]
```

`corpus.py` validates the grammar, profile-specific shape, operation set,
uniqueness policy, generation metadata, and Float32 constant range. It also
contains the NumPy oracle and translators shared by the external benchmark
collectors.

## Generation

`generate_corpus.py` uses SplitMix64 to select deterministic affine
permutations of each profile's finite expression space. Balanced commutative
profiles canonicalize operand swaps. The mixed-depth profile deliberately
cycles over leaf counts: uniform sampling from the complete space would be
overwhelmingly dominated by eight-leaf trees and would not exercise depth
variation.

Generate a profile with:

```sh
python3 bench/corpus/generate_corpus.py \
  --profile safe-math \
  --count 1000000 \
  --seed 1 \
  --output scratch/portable_safe_math_1m.json
```

Generation streams JSON to disk with bounded memory. A one-million-expression
balanced ALU corpus occupies about 77 MB.

`generate_header.py` derives SECANT postorder programs and native
AVX2/scalar expressions from any profile:

```sh
python3 bench/corpus/generate_header.py \
  --corpus bench/corpus/portable_safe_math_v1.json \
  --output scratch/portable_safe_math_v1.h
```

SECANT and native AVX consume a generated header at build time, so selecting a
different corpus for those binaries requires regenerating
`portable_alu_v1.h` and rebuilding. PySR, EvoGP, Kozax, and Operon select JSON
at runtime with `--corpus`.

Verify the canonical corpus and header with:

```sh
python3 bench/corpus/generate_corpus.py \
  --check \
  --profile balanced-alu \
  --count 1024 \
  --seed 1 \
  --name portable_alu_balanced_8x1024_seed1_v1 \
  --output bench/corpus/portable_alu_v1.json

python3 bench/corpus/generate_header.py \
  --check \
  --output bench/corpus/portable_alu_v1.h
```

CTest also verifies deterministic regeneration of every checked-in profile.

Materialize is the strictest runtime comparison because every backend returns
the same expression values. Reduction APIs are less uniform: SECANT, native
AVX2, and the available Operon path return SSE, while EvoGP, Kozax, and
SymbolicRegression.jl expose native fitness paths that compute MSE. The exact
reduction boundary is recorded in each CSV row's `notes`.
