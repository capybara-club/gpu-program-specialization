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

# Contributing

Contributions that make the techniques easier to understand, verify or transfer
are welcome: minimal examples, new supported operations/shapes, architecture
validation, clearer ownership contracts, differential tests, and reproducible
engine benchmarks.

Keep search policy separate from execution. Explain the mathematical contract,
hardware/toolchain, before/after behavior, relevant tests, and known limits of a
change. Record unsupported paths and performance regressions explicitly.

Do not commit datasets, generated modules, logs, credentials, private paths as
configuration defaults, or large experiment artifacts. Compact human-readable
benchmark summaries are useful when they state workload and timing boundaries.

Run `python3 tools/audit.py` and the relevant checks from docs/BUILDING.md. Adding
an instruction encoding requires GPU verification on the claimed architecture;
CPU tests alone are not sufficient.

Existing Charles Durham/Meta Machines work is MIT. Preserve third-party notices.
Future contributors should retain their own copyright attribution and license
contributions under MIT; do not relabel someone else's work as Charles Durham's.
