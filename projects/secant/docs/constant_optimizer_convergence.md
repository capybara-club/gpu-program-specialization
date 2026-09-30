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

# Packed Constant Optimizer Convergence

This experiment isolates constant fitting after an AST structure has already
been selected. It gives every policy the same rows, proposal count, Philox seed,
and starting state. The policies differ only in their scale and momentum rules.

The three exact targets are:

```text
affine:      y = 96 x + 0.015625
polynomial:  y = 0.03125 x^2 - 18 x + 240
local sine:  y = sin(2.75 x) - 0.125
```

The affine and polynomial cases deliberately combine constants with very
different magnitudes. The sine case begins near the correct frequency basin;
this is a local constant-refinement test, not a global frequency search.

Each result below is the geometric-mean RMSE across eight deterministic Philox
seeds using 97 rows, 1,024 settings, and 20 optimizer iterations. `fixed` uses
the initial radius throughout. Every adaptive policy uses a scale learning rate
of 0.35 and rejection decay of 0.5.

| Target | Fixed | Adaptive winner | Elite 4 | Elite 8 | Elite 16 | Winner momentum 0.8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Affine | `6.797e-2` | `2.081e-4` | `1.004e-3` | `1.083e-3` | `1.081e-3` | `9.610e-1` |
| Polynomial | `1.038e0` | `1.157e-2` | `3.052e-2` | `2.944e-2` | `5.641e-2` | `1.079e0` |
| Local sine | `2.046e-3` | `9.434e-7` | `4.570e-6` | `1.009e-5` | `1.325e-5` | `3.780e-6` |

With equal proposal counts, per-coordinate scale adaptation improved geometric
mean RMSE by about 327x for affine, 90x for polynomial, and 2,169x for the local
sine fit. These are intentionally favorable diagnostic cases, not expected
end-to-end symbolic-regression speedups. They demonstrate that a fixed radius
can find the correct region while remaining unable to refine all coordinates.

Momentum did not improve the aggregate result. Momentum 0.8 was unstable on the
mixed-scale cases because an accepted random step was extrapolated after the
scale had already contracted. Secant therefore keeps momentum configurable but
defaults it to zero. The supported default is adaptive independent scales
without momentum.

Elite-distribution mode keeps a distinct search center and evaluated incumbent.
It recombines the top settings using their mean and diagonal variance. A naive
variance-only update failed when a distant optimum put every elite at the edge
of the sampling box: the variance collapsed after moving only partway toward
the target. The implemented rule prevents this by taking the larger of the
elite-variance scale and twice the winner-step magnitude.

That corrected method converged across every seed and substantially beat a
fixed radius, but it did not beat the adaptive single winner on these
low-dimensional noiseless fits. Elite 4 was consistently the strongest elite
configuration. The mode remains available for noisy or higher-dimensional
experiments, while adaptive winner remains the default.

The convergence executable uses the CPU oracle so it can cheaply repeat exact
policies and seeds. `secant_packed_constant_optimizer_test` separately compares
both reducer modes' complete CPU and CUBIN SSE surfaces, search centers, scales,
velocities, incumbent constants, and incumbent SSE over multiple iterations.
The measured CUBIN throughput cost
of scaling each dynamic constant is about 17.4%, so adaptation needs roughly
17.5% fewer proposals to break even. The reductions in this controlled test are
well beyond that threshold, but broader search experiments remain necessary.
