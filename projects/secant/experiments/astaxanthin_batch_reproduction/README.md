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

# Astaxanthin batch-fermentation reproduction

This folder reproduces the ground-truth, dual-substrate batch model used by
Riezzo et al. (2026). It is deliberately independent of Secant's GPU search
path: the immediate goal is to establish a small, auditable CPU reference for
the published dynamical system.

## Published model reproduced here

The state is:

- `X`: biomass concentration in g/L
- `S1`: glucose concentration in g/L
- `S2`: sucrose concentration in g/L
- `P`: astaxanthin concentration in mg/L

The implementation follows Equations 6a-6d:

```text
glucose_growth = mu_m1 * S1 * X / ((S1 + K_c1 * X) * (1 + k_1 * S2))
sucrose_growth = mu_m2 * S2 * X / ((S2 + K_c2 * X) * (1 + k_2 * S1))

dX/dt  = glucose_growth + sucrose_growth - mu_d * X
dS1/dt = -Y_S1 * glucose_growth
dS2/dt = -Y_S2 * sucrose_growth
dP/dt  = alpha_1 * glucose_growth + alpha_2 * sucrose_growth + beta * X - k_d * X^2
```

Ground-truth parameter values from Table 4:

| Parameter | Value |
| --- | ---: |
| `mu_m1` | 0.43 h^-1 |
| `mu_m2` | 0.132 h^-1 |
| `K_c1` | 63.7 |
| `K_c2` | 3.68 |
| `k_1` | 5.8 L/g |
| `k_2` | 0 L/g |
| `mu_d` | 0.0055 h^-1 |
| `Y_S1` | 2.58 g/g |
| `Y_S2` | 1.71 g/g |
| `alpha_1` | 0 mg/g |
| `alpha_2` | 0 mg/g |
| `beta` | 0.21 mg/(g h) |
| `k_d` | 0.0466 mg L/(g^2 h) |

The paper defines three experiments:

| Experiment | Initial X | Initial glucose | Initial sucrose | Initial P |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1 | 10 | 5 | 0 |
| 2 | 0.1 | 20 | 5 | 0 |
| 3 | 0.2 | 5 | 2.5 | 0 |

Each batch runs from 0 to 96 hours, with measurements every 8 hours. The paper
calls these 12 measurements because the initial condition at hour 0 is not
counted as a measurement; the generated CSV includes hour 0 and therefore has
13 rows per experiment.

## Run

```sh
make test
make data
```

`make test` compares the inline, one-RHS-site C99 RK4 implementation against an
independently implemented adaptive Dormand-Prince 5(4) integrator using only the
Python standard library. It also verifies the expected fourth-order improvement
when the RK4 step is halved.

`make data` writes `generated/equation_6_ground_truth.csv`. Generated files and
compiled binaries are ignored by Git.

See [PERFORMANCE.md](PERFORMANCE.md) for the paper's reported 120-hour PySR run,
the measured CPU cost of this reference, and a clearly labeled GPU projection.

## Score-only CUDA benchmark

[`score_only_rk4_kernel.cu`](score_only_rk4_kernel.cu) and
[`score_only_rk4_runner.c`](score_only_rk4_runner.c) measure the
native-CUDA search shape without LM. One thread owns one constant setting and
one experimental trajectory, loads six constants once, runs fixed-step RK4 for
the complete 96-hour experiment, scores all 12 observations, and writes one
MSE. The launch's `y` dimension covers the three experimental trajectories.
Setting zero contains the published constants and is verified against the
independent double-precision CPU implementation.

The CUDA source is compiled to CUBIN through the same NVRTC path used by the
resident LM experiment, then loaded once through the driver API. On a CUDA
machine, select the appropriate architecture and run:

```sh
make CUDA_ARCH=sm_89 cuda-bench
```

Use `sm_120` for an RTX 5090. The executable accepts setting count, CTA thread
count, RK4 steps per eight-hour observation interval, and timing repetitions.
It reports trajectories/second, time/trajectory, equivalent RHS evaluations/s,
registers, local memory, and planted-truth verification.

See [`SCORE_ONLY_RK4.md`](SCORE_ONLY_RK4.md) for the exact timing boundary,
resource use, correctness checks, saturation sweep, and measured RTX 4090 and
RTX 5090 results.

See [`CPU_GPU_THROUGHPUT.md`](CPU_GPU_THROUGHPUT.md) for the `-march=native`
AVX-512 implementation, scalar/SIMD/core/SMT scaling, a measured FP32 FMA
ceiling, emitted-instruction accounting, and a peak-normalized comparison with
rohini's RTX 5090.

See [`DYNAMIC_LEAF_RK4.md`](DYNAMIC_LEAF_RK4.md) for the shared-memory dynamic
leaf-binding variant, its planted correct setting, SASS/resource inspection,
and the fixed-versus-dynamic throughput comparison on both GPUs.

## Secant specialization sketch

[`secant_specialization_template.cu`](secant_specialization_template.cu) shows
the intended fused GPU shape. One thread owns one constant setting, with all four
state variables and RK4 intermediates intended to remain in registers. A single
Secant marker island accepts `X`, `S1`, `S2`, `P`, and eight dynamic constants
and returns two specialized expressions, `mu_1` and `mu_2`. The known
mass-balance equations, integration, and trajectory loss remain native CUDA
around that island. The compiled CUBIN must be checked for register spills.

The file deliberately uses the existing materialize-site marker ABI: 12 input
registers, two AST output registers, and 128 reserved SASS instructions. It is a
template for the next integration step, not a runnable unspecialized kernel;
the `BRKPT` island must first be inspected and replaced by Secant.

## Resident LM proof

[`resident_lm_known_skeleton.cu`](resident_lm_known_skeleton.cu) implements the
next experiment: a complete thread-owned LM optimization remains inside one
kernel launch while each CTA caches the full dataset. The companion compile,
specialize, and runner programs provide a reproducible CUDA-machine build and a
constant-recovery regression test. See [`RESIDENT_LM.md`](RESIDENT_LM.md) for
the topology, commands, recovered constants, resource inspection, timings, and
the current primal-only specialization boundary.

## Compilation-stage benchmark

[`benchmark_compile_stages.py`](benchmark_compile_stages.py) separates NVRTC
CUDA-to-CUBIN, CUDA-to-PTX, and standalone `ptxas` time.
[`benchmark_nvptxcompiler.c`](benchmark_nvptxcompiler.c) measures the
in-process PTX-to-CUBIN path used by PTX Inject. The measured results,
equivalence checks, and proposed per-problem compilation architecture are in
[`docs/running_questions/ode_compilation_pipeline.md`](../../docs/running_questions/ode_compilation_pipeline.md).

## Reproduction boundary

This currently reproduces the paper's declared data-generating equations,
parameters, initial conditions, time horizon, and sampling schedule. A direct
row-for-row check against the publisher's Supporting Information S1 should be
added once that DOCX is available locally; the publisher endpoint currently
rejects automated retrieval with a Cloudflare challenge. The generated CSV is
therefore labeled as Equation 6 ground truth and is not represented as an
independently downloaded copy of the authors' table.

## Sources

- Luca Riezzo, Alexander Rogers, Harry Kay, and Dongda Zhang, “Automated
  Data-Efficient Symbolic Regression for Interpretable Bioprocess Model
  Development,” *Biotechnology and Bioengineering* (2026),
  <https://doi.org/10.1002/bit.70328>.
- Fernando Vega-Ramon et al., “Kinetic and Hybrid Modeling for Yeast
  Astaxanthin Production Under Uncertainty,” *Biotechnology and Bioengineering*
  118 (2021), <https://doi.org/10.1002/bit.27950>.
