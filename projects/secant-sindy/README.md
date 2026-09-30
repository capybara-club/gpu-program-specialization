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

# Secant-SINDy

> **Collection category:** Fused statistics and fitting research. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

Secant-SINDy consumes sufficient statistics produced by Secant expression cohorts and solves sparse regression
problems without materializing every feature column. The initial implementation provides:

- A target-moment CUDA kernel for dataset-level `sum(y)` and `sum(y^2)` preprocessing.
- Generated cuSolverDx CUDA source for normalized ridge and STLSQ solves.
- A C99 CPU oracle that uses the same raw-statistics layout and returns raw-coordinate coefficients, intercepts,
  SSE values, solve status, and STLSQ active sets.

The raw Secant Gram matrix is never modified. Solvers center and scale it into local workspace, solve in standardized
coordinates, convert coefficients back to the original feature units, and score against the untouched raw moments.
One CTA owns one feature cohort, normalizes its Gram matrix once, and reuses that normalized matrix across every ridge
or STLSQ sweep and active target before the CTA exits.

## Statistics Contract

For feature capacity `A` and active target count `T`, each Secant cohort produces:

```text
[0, A)                       feature sums
[A, A + A*A)               full row-major raw Gram matrix
[A + A*A, A + A*A + A*T)  feature-target sums [feature][target]
```

Target preprocessing produces `[target][sum, sum_squared]`. These values are reusable across every cohort,
regularization sweep, and search generation for the same dataset or trajectory.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Generate the preprocessing kernel or a solver specialization:

```bash
./build/secant_sindy_generate_cuda target-stats -o build/target_stats.cu
./build/secant_sindy_generate_cuda solver 8 0 32 4 32 -o build/solver_32x4_sm80.cu
./build/secant_sindy_generate_cuda solver 12 0 32 4 32 -o build/solver_32x4_sm120.cu
```

The target-moment source is ordinary CUDA and can be compiled directly by NVRTC, cuda-python, or NVCC. Solver source
requires C++17 and cuSolverDx. With current MathDx distributions, an NVRTC integration should compile the solver to
LTO IR and use nvJitLink with `libcusolverdx.fatbin`; an NVCC integration can use relocatable device LTO followed by a
device link. The caller owns compilation, module caching, loading, streams, launches, and device allocations.
The solver target is always explicit; Secant-SINDy does not select or assume a default GPU architecture.

A practical cache key is `(compute capability, feature capacity, padded target count, STLSQ iteration capacity)`.
Target moments should be computed once per dataset or trajectory, not once per expression cohort.

## Verification

The portable suite includes deterministic STLSQ edge cases, seeded comparison against an independent NumPy
implementation, and an end-to-end Secant AST-to-Gram-to-STLSQ PDE recovery test. Install the test dependency once:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r tests/requirements.txt
```

Run the full NVIDIA suite on a machine by supplying its architecture and MathDx root explicitly:

```bash
./tests/run_nvidia.sh 80 /path/to/nvidia/mathdx/26.03
./tests/run_nvidia.sh 120 /path/to/nvidia/mathdx/26.03
```

The command builds and runs capacity-4 integration coverage, capacity-32 STLSQ coverage, a direct specialized Secant
CUBIN Gram-to-cuSolverDx pipeline, and solver-only ridge/STLSQ benchmarks. It does not install dependencies or infer a
GPU target. Benchmark timing excludes statistics construction, target preprocessing, allocations, transfers, and module
loading.
