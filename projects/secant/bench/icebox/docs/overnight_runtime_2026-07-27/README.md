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

# Overnight Runtime Report

This folder summarizes the completed July 27, 2026 overnight campaigns on:

- NVIDIA GeForce RTX 5090 with AMD Ryzen 9 9900X
- AMD Radeon RX 9070 XT with AMD Ryzen 9 7900X

Every result uses the balanced eight-input ALU corpus
`portable_alu_balanced_8x1024_seed1_v1`, hash
`5f790c4e4d7bc60a`. The measured row counts are 1,024, 4,096, 16,384,
65,536, 131,072, and 262,144.

## Reading the Graphs

### Runtime Overview

[Combined runtime overview](runtime_overview.svg),
[RTX 5090 runtime overview](runtime_overview_rtx5090.svg), and
[RX 9070 XT runtime overview](runtime_overview_rx9070xt.svg) compare all
available execution systems. These graphs use logarithmic throughput axes
because the measured rates span several orders of magnitude.

For CUDA, PTX, CUBIN, HIP, and HSACO:

- Solid lines use the fully saturated 24-worker compile pipeline, including
  module loading, execution, and unloading.
- Dashed lines show runtime-only GPU event throughput.

External systems use dotted warm-runtime lines and are conservatively assigned
zero compilation cost.

![Combined runtime overview](runtime_overview.svg)

![RTX 5090 runtime overview](runtime_overview_rtx5090.svg)

![RX 9070 XT runtime overview](runtime_overview_rx9070xt.svg)

The external baseline line at each row count is the fastest measured execution
mode for that system. This can select a different worker count or execution
mode at different row counts.

EvoGP was measured only on the NVIDIA system. This is a limitation of the
current benchmark adapter, which rejects non-NVIDIA platforms and is hardcoded
around its CUDA environment. EvoGP itself documents ROCm/HIP support, so the
missing AMD line must not be interpreted as an EvoGP hardware limitation.

### GPU Runtime Detail

[GPU runtime detail](runtime_gpu_detail.svg) uses independent linear axes.
This makes differences among GPU implementations visible without flattening
them against the slower CPU and runtime-dispatch systems.

![GPU runtime detail](runtime_gpu_detail.svg)

Peak runtime-only rates:

| System | Materialize | SSE |
|---|---:|---:|
| CUDA O1, RTX 5090 | `4.200e11` | `2.072e12` |
| PTX O1, RTX 5090 | `4.149e11` | `2.400e12` |
| CUBIN, RTX 5090 | `4.205e11` | `1.443e12` |
| HIP O1, RX 9070 XT | `1.011e11` | `1.997e11` |
| HSACO, RX 9070 XT | `1.343e11` | `1.854e11` |

The direct binary paths retain conservative scheduling. Consequently, CUBIN
matches the native compiler paths closely for materialization but does not
match optimized PTX or CUDA scheduling for SSE.

### Full Pipeline

[RTX 5090 pipeline throughput](pipeline_frontier_rtx5090.svg) and
[RX 9070 XT pipeline throughput](pipeline_frontier_rx9070xt.svg) show two
rates for each SECANT backend:

- Solid: measured wall time from 24-worker concurrent compilation start
  through module loading, execution, and unloading.
- Dashed: GPU event time with inputs already resident.

![RTX 5090 full pipeline](pipeline_frontier_rtx5090.svg)

![RX 9070 XT full pipeline](pipeline_frontier_rx9070xt.svg)

Peak compile-inclusive rates:

| System | Materialize | SSE |
|---|---:|---:|
| CUBIN, RTX 5090 | `2.172e11` | `1.932e11` |
| HSACO, RX 9070 XT | `9.915e10` | `1.212e11` |

Native CUDA, PTX, and HIP runtime can be faster after compilation, but their
compiler wall time dominates this workload. Direct CUBIN and HSACO generation
therefore define the compile-inclusive frontier.

### Compilation Throughput

[Compilation throughput](compile_throughput.svg) reports the median measured
24-worker compile rate across the six row-count runs.

![Compilation throughput](compile_throughput.svg)

Approximate rates across these repeated measurements:

| Backend | Materialize AST/s | SSE AST/s |
|---|---:|---:|
| CUBIN | `6.90e7` | `6.45e7` |
| PTX O1 | `6.92e4` | `9.08e3` |
| CUDA O1 | `5.35e3` | `1.69e3` |
| HSACO | `3.39e7` | `3.45e7` |
| HIP O1 | `1.92e3` | `2.16e2` |

Rows do not affect AST compilation. The six row-count campaigns serve as
repeated measurements with different generated AST seeds.

### Module Lifecycle

[Direct binary lifecycle](direct_binary_lifecycle.svg) partitions full
pipeline wall time into runtime, module load, module unload, and residual
pipeline overhead.

![Direct binary lifecycle](direct_binary_lifecycle.svg)

For the RTX 5090 SSE campaign at 262,144 rows, CUBIN runtime was `0.357 s`
while module loading and unloading consumed `2.185 s`. The specialization
window was only about `0.030 s`. Module residency therefore explains most of
the difference between runtime-only and full-pipeline throughput.

The RX 9070 XT has a different profile. HSACO loading is still visible, but
unloading is negligible and execution becomes the largest component at high
row counts.

### Speedup Over External Runtimes

[Direct binary speedup](direct_binary_speedup.svg) compares CUBIN or HSACO
against the fastest available external runtime at each row count. External
compilation cost is conservatively treated as zero. Solid lines include the
direct backend's compilation and module lifecycle, while dashed lines show
runtime-only speedup.

![Direct binary speedup](direct_binary_speedup.svg)

At 262,144 rows:

| Direct backend | Shape | Runtime-only speedup | Full-pipeline speedup |
|---|---|---:|---:|
| CUBIN | Materialize | `40.34x` | `25.49x` |
| CUBIN | SSE | `38.61x` | `5.17x` |
| HSACO | Materialize | `12.79x` | `10.62x` |
| HSACO | SSE | `5.20x` | `3.40x` |

## Measurement Contract

- Materialize writes every AST-row result.
- SSE reduces every AST over rows to a scalar sum of squared errors.
- A row evaluation means one AST applied to one row for both shapes.
- SECANT runtime-only values exclude compilation, module transitions, and
  transfers.
- SECANT full-pipeline values include concurrent compile, module load,
  execution, and module unload wall time.
- External baseline values are warm runtime measurements and are assigned zero
  compilation cost in compile-inclusive comparisons.
- Inputs and targets are resident before timed GPU execution.
- These are backend throughput measurements, not complete symbolic-regression
  search timings.

The normalized source values used by every graph are in
[`overnight_points.csv`](overnight_points.csv).

## Regeneration

From the repository root:

```bash
./.venv/bin/python bench/overnight_runtime_report.py \
  --nvidia-dir scratch/comprehensive_rtx5090_overnight_20260727 \
  --amd-dir scratch/comprehensive_rx9070xt_overnight_20260727 \
  --output-dir docs/overnight_runtime_2026-07-27
```

The command writes both SVG and PNG versions of every graph.
