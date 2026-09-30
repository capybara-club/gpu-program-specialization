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

# Cooperative trajectory LM CUDA control

This scratch experiment isolates subgroup-owned trajectory LM from CUDA-template inspection and SASS specialization. The fixed eight-state system is:

```text
dx_i/dt = k_i x_i + 0.01 x_((i + 1) mod 8)
```

Each subgroup owns one complete fit. Its lanes divide the forward sensitivities and packed normal equations and exchange values only with warp shuffles. The default control independently executes the small Cholesky solve in every lane. Passing `--distributed-solve` to `compile.py` instead distributes the factor and solve vectors across those lanes. Dynamic shared memory stages only the immutable trajectory table; it is not used for lane-to-lane optimizer communication.

Passing `--leader-evaluation` makes only lane zero compute the fixed right-hand sides and analytic local partials before broadcasting them with shuffles. This isolates the execution topology intended for a leader-only specialized AST site without involving inline assembly, physical patch sites, or SASS branches.

Compile `cooperative_lm.cu` repeatedly with `ODEZZA_FIT_THREAD_COUNT` set to 1, 2, 4, 8, 16, or 32. Correctness requires the same initial MSE, accepted proposal, optimized constants, final MSE, and optimizer counters for identical starts at every width, allowing ordinary FP32 rounding differences.
