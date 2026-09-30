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

# stack-ptx-emit

> **Collection category:** Program construction and specialization tools. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

Unified backend emission for Stack PTX.

This repo keeps the multi-backend emitter layer outside `mm-ptx`. It consumes
`stack_ptx.h` from `mm-ptx` and emits code through `stack_ptx_emit.h` and the CLI
tools. Supported outputs include PTX, C, CUDA, NumPy, LaTeX, and Markdown.

It also includes `stack_ptx_emit_linear_regression.h`, a single-header wrapper
that emits a fully hard-coded linear-regression inference function by inlining a
set of Stack PTX feature programs into the selected output format.
The NumPy linear-regression backend emits batched inference over `x.shape == (M, D)`
and returns a length-`M` result.

There is also a lightweight Flask app under `webapp/` for browsing randomly
generated linear-regression models in a browser and calling the same generator
through a JSON API. The generator CLI it wraps is
`stack_ptx_emit_linear_regression_cli`.
