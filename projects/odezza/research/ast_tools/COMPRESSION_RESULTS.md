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

# AST binary compression experiment

Date: 2026-09-01 EDT

Host: mac1, arm64
Corpus: one million unique seeded-random ASTs

## Corpus configuration

- Exact tree depth: 2
- States: 4
- Constant slots: 4
- Inline literals: `0`, `1`, `-1`, `2`
- Operators: `add`, `sub`, `mul`, `div`, `neg`, `sin`, `cos`
- Two-way toggle leaves: enabled with 5 toggle bits
- Selection: 1,000,000 unique ranks from seed `20260901`
- Largest encoded AST: 52 bytes

The complete command was:

```bash
scratch/ast_tools/build/odezza-ast-tools \
    --states 4 \
    --constants 4 \
    --literal 0 \
    --literal 1 \
    --literal -1 \
    --literal 2 \
    --operators add,sub,mul,div,neg,sin,cos \
    --toggle-bits 5 \
    --include-toggle2 \
    --depth 2 \
    --depth-mode exact \
    --mode random \
    --count 1000000 \
    --seed 20260901 \
    --binary-output scratch/ast_tools/build/ast-corpus-self-delimiting-random-depth2-1m.oast
```

## Results

| Representation | Bytes | Relative to raw | Reduction | Compression time |
|---|---:|---:|---:|---:|
| Raw `.oast` | 35,649,787 | 1.000 | 0.0% | — |
| LZ4 1.10.0 | 18,766,663 | 0.526 | 47.4% | 0.02 s |
| Zstandard 1.5.7, level 3 | 10,772,730 | 0.302 | 69.8% | 0.05 s |
| Apple gzip 479, level 9 | 9,959,143 | 0.279 | 72.1% | 8.88 s |
| Zstandard 1.5.7, level 19 | 6,688,247 | 0.188 | 81.2% | 12.02 s |
| XZ 5.8.3, level 6 | 6,460,776 | 0.181 | 81.9% | 10.38 s |

The raw file contains 35,649,763 bytes of concatenated self-delimiting postorder programs and one 24-byte file header. There is no per-AST metadata or padding. Zstandard level 3 is the practical default in this sample: it saves 24.9 MB in 0.05 seconds. XZ and Zstandard level 19 save another 4.1–4.3 MB but take over 200 times longer.

Serialization took 0.16 seconds. A complete structural reread took 0.47 seconds. The Zstandard level-3 output was decompressed, compared byte-for-byte with the original, and then accepted by the AST file reader as all 1,000,000 programs. The raw AST format intentionally has no checksum; when corruption detection is required, it belongs in the selected compression or transport layer.

## Material issue found and resolved

The first large random-generation attempt exposed exact-depth tuple unranking that walked candidate child ranks linearly, making random access depend on the size of the child space. The command also remained alive after its tool session stopped returning output. It later wrote through its old file descriptor into a reused benchmark filename. The reader rejected that mixed file. All partial, old-format, and contested-filename outputs are invalid and excluded from the table.

The stale process was stopped and confirmed absent. The unranking routine now chooses each tuple component arithmetically in constant time. Existing exhaustive/reference tests and new binary round-trip tests pass after the change. The final corpus was generated under a fresh filename, reread in full, compressed, decompressed byte-identically, and reread again. The complete million-program generation exercised the repaired path and took 0.16 seconds.

Compression ratios are specific to this randomized syntax distribution. Prefix-truncated enumeration is more repetitive and would likely compress better, so it should not be presented as equivalent to this result.
