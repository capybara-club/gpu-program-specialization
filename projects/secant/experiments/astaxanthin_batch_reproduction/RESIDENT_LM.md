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

# Resident LM for the known rate-law skeleton

This experiment tests the execution shape proposed for fused dynamical-system
identification. It is intentionally narrower than a general search system: the
two rate-law structures are known, while their six numerical constants are
blinded and recovered by Levenberg-Marquardt.

## Kernel topology

- One CUDA kernel contains one specialized pair of rate-law ASTs.
- `blockIdx.y` selects a tile of settings; one thread owns one setting.
- Every CTA cooperatively loads the complete reference dataset once.
- The CTA remains in one kernel launch for every LM iteration and damping retry.
- Each thread privately integrates all three experiments with RK4.
- Four states, 24 forward sensitivities, 21 unique `J^T J` terms, six `J^T r`
  terms, the damping value, and the 6x6 Cholesky solve remain thread-local.
- No trajectory, Jacobian, Gram matrix, or proposed parameter vector is written
  to global memory. Only final constants, MSE, and counters are returned.

The shared reference array contains 12 initial-condition values and 144 target
values: 156 floats, or 624 bytes of user-declared shared memory.

The specialized rate laws are:

```text
mu1 = c0*S1 / ((S1 + c1*X) * (1 + c2*S2))
mu2 = c3*S2 / ((S2 + c4*X) * (1 + c5*S1))
```

The expected parameters are:

```text
c0=0.43, c1=63.7, c2=5.8, c3=0.132, c4=3.68, c5=0
```

The generated targets use the double-precision CPU reference with a maximum
RK4 step of 0.01 hours. GPU starts are deterministic but deliberately broad:
they do not include a planted ground-truth start.

Residual channels are scaled by `[1, 0.05, 0.2, 1]` for `X`, `S1`, `S2`, and
`P` before normal-equation accumulation so the larger substrate concentrations
do not determine the conditioning alone. Reported MSE values are therefore
weighted MSE, not an unweighted paper metric.

## Build and run

On a CUDA machine, from the Secant repository root:

```sh
cmake -S . -B build -DSECANT_BUILD_TESTS=OFF -DSECANT_BUILD_BENCHMARKS=OFF
cmake --build build --target \
  astaxanthin_resident_lm_compile \
  astaxanthin_resident_lm_specialize \
  astaxanthin_resident_lm_runner

build/astaxanthin_resident_lm_compile \
  sm_89 \
  experiments/astaxanthin_batch_reproduction/resident_lm_known_skeleton.cu \
  build/resident_lm_template.cubin

build/astaxanthin_resident_lm_specialize \
  build/resident_lm_template.cubin \
  build/resident_lm_specialized.cubin

build/astaxanthin_resident_lm_runner \
  build/resident_lm_specialized.cubin 128 64 16 20 8 3
```

Use `sm_120` for the RTX 5090. Runner arguments after the CUBIN are settings,
CTA threads, RK4 steps per eight-hour observation interval, maximum LM
iterations, maximum damping attempts, and timing repetitions.

The compile tool can produce a fully native comparison CUBIN by appending
`--direct`. That CUBIN is launched directly and must not be passed through the
Secant specializer.

## Recovery results

The primary check used 128 broad starts, 64 threads per CTA, 20 LM iterations,
and up to eight damping attempts. With 16 RK4 steps per observation interval:

| GPU | Kernel time | Starts within 1% on every parameter | Best weighted MSE |
| --- | ---: | ---: | ---: |
| RTX 4090 (`sm_89`) | 32.01 ms | 98 / 128 | 2.45e-13 |
| RTX 5090 (`sm_120`) | 34.57 ms | 98 / 128 | 2.45e-13 |

The best recovered vector was:

| Parameter | Fitted | Truth | Absolute error |
| --- | ---: | ---: | ---: |
| `mu_m1` | 0.4300277 | 0.43 | 2.77e-5 |
| `K_c1` | 63.70399 | 63.7 | 3.99e-3 |
| `k_1` | 5.800067 | 5.8 | 6.68e-5 |
| `mu_m2` | 0.1320000 | 0.132 | 0 |
| `K_c2` | 3.679993 | 3.68 | 7.15e-6 |
| `k_2` | 9.99e-8 | 0 | 9.99e-8 |

At the more accurate 80 steps per observation interval, the best 4090 fit had
weighted MSE 8.48e-13 and recovered `K_c1` within 0.00179, `mu_m1` within 1.09e-5, and
the true-zero `k_2` within 2.42e-8. It took 162.50 ms for the deliberately
underfilled two-CTA launch.

## Residency and resources

| Target | Registers/thread | Stack | Spill loads/stores | Local memory | Shared memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 4090 `sm_89` | 167 | 0 | 0 / 0 | 0 | 624 B |
| RTX 5090 `sm_120` | 168 | 0 | 0 / 0 | 0 | 624 B declared |

Blackwell's `cuobjdump` reports 1,648 bytes of total shared allocation, which is
the 624-byte array plus 1,024 bytes of architecture/runtime overhead. The
compiler itself reports the 624-byte user allocation.

The module is loaded once before timing. Each timed launch performs the full
resident optimizer; there are no conditional kernel launches inside LM.

With 65,536 starts, enough CTAs exist to occupy the machines more fully:

| GPU | CTA threads | CTAs | Time per full optimization launch | Recovered within 1% |
| --- | ---: | ---: | ---: | ---: |
| RTX 4090 | 64 | 1,024 | 85.31 ms | 45,861 / 65,536 |
| RTX 5090 | 64 | 1,024 | 72.34 ms | 45,861 / 65,536 |

On the 5090, 32, 64, 96, and 128 CTA threads took 70.36, 72.34, 71.90, and
72.45 ms respectively at this setting count. The topology is therefore not
particularly sensitive to CTA size in this range; 32 threads was narrowly best
in this measurement.

## Current specialization boundary

This is a genuine CUBIN specialization: the template contains one 128-SASS
instruction patch island, and Secant inserts the two known ASTs before the
module is loaded. The integration, sensitivity propagation, Gram accumulation,
solve, and acceptance logic remain native CUDA around the patch.

This first proof emits only the two primal rate values from the patch. Analytic
partials are native code coupled to the known unblinded skeleton. Consequently,
the patched AST recomputes denominator products that fully native CUDA can
common-subexpression-eliminate with the derivative calculation. On the 4090:

| Shape | 8,192 settings | 65,536 settings |
| --- | ---: | ---: |
| Fully native known expressions | 20.05 ms | 58.80 ms |
| Secant primal-only specialization | 35.52 ms | 85.31 ms |

The specialized proof is therefore correct, resident, and spill-free, but not
yet a native-performance result. The next specialization ABI should emit each
rate's primal value and its state/parameter partial derivatives from the same
specialized expression evaluation. That will remove duplicated reciprocal and
denominator work and will also make the kernel applicable to arbitrary smooth
candidate skeletons rather than this one known derivative shape.
