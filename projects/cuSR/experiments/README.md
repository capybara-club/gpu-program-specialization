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

# cuSR Experiments

This directory contains standalone CUDA/C++ experiments that explore kernel
shapes before they become part of the AST-SASS runtime.

## AST And Settings

`ast_settings` compares four fixed expression classes:

- `simple-alu`: a small three-input ALU expression;
- `complex-alu`: an eight-leaf balanced add/mul/min/max tree;
- `light-mufu`: the complex tree with one approximate MUFU leaf operation;
- `heavy-mufu`: the complex tree with a MUFU operation on every leaf.

The expression bodies are written directly in CUDA/C++ templates. The benchmark
can instantiate 4, 8, 16, or 32 ASTs per kernel. Every row-tile CTA directly
adds its AST-setting SSEs to the final output, and timing includes clearing that
output.

Build and run the default 8-AST, 4096-setting experiment:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUSR_SASS_GPU_NAME=sm_120
cmake --build build -j --target cusr_ast_settings_experiment
./build/cusr_ast_settings_experiment
```

Sweep an expression or packing shape explicitly:

```bash
./build/cusr_ast_settings_experiment --asts 16 --expr heavy-mufu --settings 4096
./build/cusr_ast_settings_experiment --asts 8 --expr complex-alu --settings 1024
```

See [`ast_settings/README.md`](ast_settings/README.md) for the current RTX 5090
measurements and interpretation.

## Global-load SASS Probe

`gmem_load_sass` contains a small RTX 5090 disassembly probe that forces eight
column-major global loads before a `brkpt`, adds their register values, and
stores one result per row.
