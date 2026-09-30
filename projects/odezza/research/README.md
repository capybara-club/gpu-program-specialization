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

# Selected Odezza research references

These small source snapshots preserve experiments that were originally under
`scratch/`. They are separate from the [current core](../core/README.md).

| Directory | What it isolates |
|---|---|
| `ast_tools/` | AST/system enumeration, sampling, compact serialization and CPU reference evaluation |
| `cuda_ast_specializer/` | Generated-CUDA compilation as an alternative to direct SASS specialization |
| `cooperative_lm/` | One fit shared across lanes, isolated on a fixed ODE |
| `fitting_batch_trial/plain_cuda_lm/` | Full-CUDA fitting controls for states, constants, subgroup widths, spills and occupancy |
| `rosenbrock_trial/` | Adaptive FP64 Rosenbrock23 GPU prototype and analytic Jacobian generation |
| `rosenbrock_cpu_trial/` | CPU RK4/Rosenbrock implementation and FP32/FP64 precision study |
| `rfm_dependency_trial/` | Observation-only feature-metric estimates of RHS dependencies |

Start with the [project map](../../../docs/PROJECTS.md) and
[coverage notes](../../../docs/RESEARCH_COVERAGE.md). Original READMEs retain
historical command paths and links to omitted run output. The source layout here
preserves sibling references, but not every old campaign's external environment.
See [portable checks](../../../docs/BUILDING.md) and
[actual validation](../../../docs/VALIDATION.md).

These trials do not add an adaptive/stiff solver or RFM preprocessing to the
current scoring API. Their results should not be mixed with its throughput claims.
