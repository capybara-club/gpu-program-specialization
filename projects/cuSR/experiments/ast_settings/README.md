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

# Fixed AST And Settings Experiment

This experiment measures the native CUDA ceiling for the production atomic-SSE
topology. Each row-tile CTA directly adds one SSE per AST-setting.

The code is split into:

- `cusr_atomic_kernel_example.cu`: an explicit row-atomic kernel without
  templates or device functions;
- `cusr_ast_settings_kernels.cuh`: fixed CUDA/C++ expression templates;
- `cusr_ast_settings_bench.cu`: deterministic data generation, validation,
  occupancy selection, memory accounting, and event-based timing.

The kernel writes raw SSE to an `[ast][setting]` array. Timings include clearing
that array. It allocates no row-tile workspace and uses no cooperative launch.

## RTX 5090 Snapshot

Measured July 11, 2026 with CUDA 13.1, `sm_120`, 1,048,576 rows, 32 columns,
4,096 settings, and fixed hashed ASTs. Rates are AST-setting row evaluations per
second. These retained results motivated the production atomic-only contract.

### Expression Complexity At Eight ASTs

| expression | row atomic |
|---|---:|
| simple ALU | `6.371e12` |
| complex ALU | `4.442e12` |
| light MUFU | `3.431e12` |
| heavy MUFU | `1.962e12` |

### Complex ALU AST Packing

| ASTs/kernel | row atomic | final output |
|---:|---:|---:|
| 4 | `3.092e12` | 0.0625 MiB |
| 8 | `4.401e12` | 0.125 MiB |
| 16 | `5.357e12` | 0.25 MiB |
| 32 | `7.919e12` | 0.5 MiB |

Eight ASTs is the first packing knee, sixteen is a stronger general-purpose ALU
compromise, and thirty-two is the native CUDA reference throughput ceiling measured
for this shape.

At only 128 settings, row atomic still measured `3.821e12` row evals/s with 8
ASTs/kernel and `7.906e12` with 32 ASTs/kernel. Each result received 8,192 CTA
contributions, so this was a direct contention stress test.

These kernels are native CUDA reference ceilings, not AST-SASS measurements. The
project GPU tests independently validate the generated atomic-SSE kernels
against the CPU evaluator.

## Search Output

For a fixed evaluation dataset, SSE ranks candidates identically to MSE, RMSE,
and R2:

```text
MSE  = SSE / num_rows
RMSE = sqrt(SSE / num_rows)
R2   = 1 - SSE / fixed_target_variance
```

The kernel therefore writes final SSE directly:

```cuda
atomicAdd(output_sse + ast_setting, row_tile_sse);
```

Output storage is exactly `ASTs * settings * sizeof(float)`. Selection can
compare SSE directly and normalize only results exported to the user.
