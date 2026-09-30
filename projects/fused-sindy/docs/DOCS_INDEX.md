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

# Documentation Index

Start here:

1. [README](../README.md) - project overview, install sketch, first example, and performance snapshot.
2. [Install](INSTALL.md) - supported environment, dependency pins, smoke test, and troubleshooting.
3. [PyTorch Usage](PYTORCH_USAGE.md) - tensor layouts and the Python workflow.
4. [Architecture](ARCHITECTURE.md) - why the compiler/runtime pipeline exists and how it fits together.
5. [Validation And Numerics](VALIDATION_AND_NUMERICS.md) - how to score models without fooling yourself.

Focused references:

- [Operators And Shapes](OPERATORS_AND_SHAPES.md) - fixed AST shape, leaves, and supported ops.
- [Search Strategy](SEARCH_STRATEGY.md) - LLM-assisted generation, settings, and MAP-Elites style loops.
- [Performance](PERFORMANCE.md) - compile/runtime tradeoffs and measured benchmark records.
- [C API](C_API.md) - native integration concepts and ownership rules.
- [Name And Positioning](NAME_AND_POSITIONING.md) - why this is called FusedSINDy and where it fits.

Related repositories are listed in the README:

- `https://github.com/MetaMachines/fused-sindy`
- `https://github.com/MetaMachines/cubin-function-patch`
- `https://github.com/MetaMachines/mm-ptx`

Historical benchmark records remain in this directory so performance claims can
be traced back to concrete runs:

- [RTX 5090 compilation pipeline](RTX5090_COMPILATION_PIPELINE_2026_06_16.md)
- [Combined compile scaling](COMBINED_COMPILE_SCALING_512_MODULES.md)
- [GH200 compile scaling](GH200_COMPILE_SCALING_512_MODULES.md)
- [GH200 Lambda test and bench run](GH200_LAMBDA_TEST_BENCH_2026_06_17.md)
