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

# Historical settings-based Secant-SR

The root build now targets Secant 0.3's explicit toggles and shared banks.
`CMakeLists.txt`, `secant_sr.h`, and `README-settings.md` in this directory preserve
the former build, public API, and documentation. They are reference copies, not
a standalone build against current Secant.

The former `src/`, most of `app/`, historical Python campaign scripts, and
historical tests/docs remain in place. Only `app/dataset.c`, its test, and the
static expression test are reused by the new build. This preserves earlier
research without compatibility aliases or settings behavior in the new API.

Use a matching historical Secant/Secant-SR revision to reproduce those experiments.
Do not point `python/run_suite.py`, `python/benchmark_dynamic_leaf.py`, or archived
SRbench campaign instructions at the new executable: their flags and report
schemas have not been ported. No new results should be labeled historical
maturity/QD/LM/SRbench results.

SRBench now has a separate [toggle-native campaign adapter](../docs/srbench_toggle.md)
that reuses the frozen data protocol; it does not emulate the old search policy.

Current implementation: [../toggle/README.md](../toggle/README.md).
