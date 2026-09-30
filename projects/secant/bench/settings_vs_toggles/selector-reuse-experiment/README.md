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

# Selected-value reuse experiment (not the default)

The default writer assumes arbitrary, unrelated ASTs. State and bank source
registers are shared, but selector results are ordinary temporary operands;
they are released or overwritten when consumed. There is no similarity-based
packing or cross-AST retention of selected values.

On 2026-09-19 the user requested this baseline after an experiment retained up
to eight repeated selector results. The experiment improved deliberately shared
selector fixtures but showed no meaningful GPU scoring improvement for the three
measured random GP initial populations. Its instruction savings do not establish
a benefit for unrelated ASTs. Removing it restores the earlier default writer
exactly; it also gives up the reported shared-fixture speedup.

`experiment.patch` preserves the writer change, its white-box test, and CMake
registration. It is not part of the normal build. `manifest.json` records the
two writer hashes. To reproduce, use an isolated copy of the corresponding
Secant source, check `git apply --check experiment.patch`, then apply the patch
there and use the normal host and GPU validation commands. Do not apply it to
the default checkout during a benchmark.

The [measurements and limitations](../report-selector-reuse-20260919.md) remain
valid for the experimental implementation. The already-running comparison uses
frozen binaries on rack1; restoring the default source does not change those
jobs. The before arm matches the restored writer; the after arm is experimental.

Restoration validation: the default writer matches the pre-experiment snapshot
byte for byte, and the preserved patch passes `git apply --check`. All six Secant
host tests and all six Secant-SR host tests pass in both normal and ASan/UBSan
builds. GPU benchmarks were not rerun for this restoration; the earlier default
writer results and the frozen experimental results remain separate measurements.
