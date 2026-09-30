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

# Dynamic-leaf SSE AST packing, 2026-08-14

This sweep measures how dynamic-leaf SSE throughput changes as more ASTs are
specialized into one kernel. Every result passed the CPU/GPU comparison.

## Shape

- One module and one kernel
- Distinct ALU ASTs with eight dynamic leaves/sites
- Eight active input columns
- 16,384 rows
- Thread-owned settings
- Hot-kernel runtime; compilation and module-management time are excluded
- Row evaluations count `ASTs * settings * rows`

The initial packing curves hold the best one-AST Rohini topology fixed at 128
settings per CTA, 128 tile rows, and 128 threads. A later sweep retunes these
dimensions around the packing plateau.

## Rohini RTX 5090

| ASTs/kernel | PTX, 4,096 settings | PTX, 8,192 settings | CUDA, 4,096 settings | CUDA, 8,192 settings |
|---:|---:|---:|---:|---:|
| 1 | 0.722T | 0.772T | 0.699T | 0.758T |
| 2 | 1.409T | 1.480T | 1.382T | 1.474T |
| 4 | 2.090T | 2.146T | 2.100T | 2.239T |
| 8 | 2.592T | 2.711T | 2.582T | 2.776T |
| 16 | 2.925T | 3.079T | 2.954T | 3.120T |
| 32 | 3.156T | 3.309T | 3.213T | 3.370T |
| 64 | 3.311T | 3.445T | 3.222T | 3.342T |
| 128 | 3.356T | 3.466T | 3.053T | 3.189T |

The repeated PTX medians at 8,192 settings were 3.311T, 3.445T, and
3.465T row evals/s for 32, 64, and 128 ASTs respectively. The retuned winner
remained 128 ASTs/kernel, 128 settings/CTA, 128 tile rows, and 128 threads.

Packing eight ASTs is already sufficient to exceed the earlier 2.4T target.
The curve has largely plateaued by 32-64 ASTs. Relative to one AST, 128 ASTs
increase normalized throughput by approximately 4.49x.

At 8,192 settings, normalized hot-kernel latency is approximately:

| ASTs/kernel | PTX throughput | Kernel time |
|---:|---:|---:|
| 32 | 3.311T row evals/s | 1.30 ms |
| 64 | 3.445T row evals/s | 2.49 ms |
| 128 | 3.465T row evals/s | 4.96 ms |

Thus 64 ASTs delivers 99.4% of the maximum measured rate with half the AST
batch and roughly half the kernel duration. It is the better default when
specialization cost, code size, or scheduling latency matters; 128 is the pure
hot-throughput choice.

## Ada RTX 4090

The initial curve used Ada's one-AST winner: 256 settings per CTA, 256 tile
rows, and 256 threads.

| ASTs/kernel | PTX throughput |
|---:|---:|
| 1 | 0.555T row evals/s |
| 2 | 1.081T row evals/s |
| 4 | 1.587T row evals/s |
| 8 | 1.970T row evals/s |
| 16 | 2.232T row evals/s |
| 32 | 2.385T row evals/s |
| 64 | 2.463T row evals/s |
| 128 | 2.188T row evals/s |

Retuning 128 ASTs to 128 settings per CTA, 256 tile rows, and 128 threads
recovered it to 2.451T row evals/s. Repeated medians were:

| ASTs/kernel | Retuned PTX throughput | Kernel time |
|---:|---:|---:|
| 32 | 2.378T row evals/s | 1.81 ms |
| 64 | 2.458T row evals/s | 3.50 ms |
| 128 | 2.451T row evals/s | 7.01 ms |

Ada therefore has a clear practical winner at 64 ASTs/kernel. Doubling to 128
does not improve normalized throughput and doubles batch latency.

## Recommendations

- Use 64 ASTs/kernel as the portable default for packed dynamic-leaf SSE.
- On Rohini, use 128 ASTs only when maximum resident hot-kernel throughput is
  more important than specialization size and per-batch latency.
- Keep host-specific CTA presets: Rohini uses 128 settings/CTA, 128 tile rows,
  and 128 threads; Ada uses 256 settings/CTA, 256 tile rows, and 256 threads at
  64 ASTs.
- Include compilation, module load, and end-to-end search latency before making
  128 ASTs the production default. This sweep measures resident kernel runtime.

## Reproduction

```sh
python3 bench/dynamic_leaf_cta_sweep.py \
  --output /tmp/dynamic-leaf-sse-packing.csv \
  --workloads sse --backends ptx \
  --settings 4096 8192 \
  --settings-per-cta 128 --tile-rows 128 --threads 128 \
  --sse-asts-per-kernel 1 2 4 8 16 32 64 128
```

Retune the plateau with multiple values for `--settings-per-cta`,
`--tile-rows`, and `--threads`, then repeat finalists with `--trials 3`.
