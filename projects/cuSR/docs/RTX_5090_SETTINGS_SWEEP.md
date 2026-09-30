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

# RTX 5090 Settings Saturation Sweep

Date: July 10, 2026

> Historical benchmark: this sweep predates the atomic-only SSE contract. The
> score-plus-reduce columns document the removed implementation and are retained
> only as decision evidence. Current commands always use direct atomic SSE.

Hardware and toolchain:

- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32 GiB
- Driver: 595.71.05
- CUDA toolkit: 13.1.115
- Build: Release, `sm_120`

## Benchmark Shape

The primary sweep used the AST-SASS tile-static MSE path:

```text
variant             = tile_static_mse
ast_mode            = depth3_alu
modules             = 1
kernels/module      = 8
ASTs/kernel         = 32
ASTs total          = 256
rows                = 1,048,576
columns             = 32
check rows          = 0
```

Each depth-3 ALU AST is a balanced eight-leaf tree with seven hash-selected
`add`, `mul`, `min`, and `max` reductions. The setting count is runtime
configured and does not change the patched ASTs.

An example run is:

```bash
./build/cusr_bench --preset runtime --ast-mode depth3-alu --settings 512 --run-iters 15
```

The primary points below are medians of three runs performed in shuffled order
with 15 timed iterations per run. `Score` measures only the kernels that emit
partial SSE. `Score + reduce` also includes the SSE-to-MSE reduction kernels.

## Results

| settings | score rows/s | score + reduce rows/s | score vs 1024 | score + reduce vs 1024 |
|---:|---:|---:|---:|---:|
| 128 | 3.005e12 | 2.485e12 | 95.9% | 83.2% |
| 256 | 3.087e12 | 2.799e12 | 98.6% | 93.8% |
| 384 | 3.104e12 | 2.896e12 | 99.1% | 97.0% |
| 512 | 3.118e12 | 2.952e12 | 99.6% | 98.9% |
| 768 | 3.128e12 | 2.981e12 | 99.9% | 99.9% |
| 1024 | 3.132e12 | 2.985e12 | 100.0% | 100.0% |

The marginal gains make the knee clearer:

| transition | score gain | score + reduce gain |
|---:|---:|---:|
| 128 to 256 | 2.7% | 12.6% |
| 256 to 384 | 0.6% | 3.5% |
| 384 to 512 | 0.5% | 1.9% |
| 512 to 768 | 0.3% | 1.0% |
| 768 to 1024 | 0.1% | 0.1% |

Score-kernel returns therefore start diminishing after the first 128 settings.
At 256 settings, score throughput is already within 1.4% of the measured
maximum. The reduction path needs more work to amortize its overhead, but reaches
within 3.0% at 384 settings and within 1.1% at 512 settings.

## Setting Batch Alignment

The kernel assigns one setting to each of its 128 threads and advances the
settings loop in batches of 128. Setting counts that are not multiples of 128
leave part of the final batch inactive. A broad single-run sweep demonstrated
the effect:

| settings | active batches | score rows/s | score + reduce rows/s |
|---:|---:|---:|---:|
| 64 | 0.5 of 1 | 1.603e12 | 1.303e12 |
| 96 | 0.75 of 1 | 2.343e12 | 1.918e12 |
| 128 | 1 of 1 | 2.991e12 | 2.486e12 |
| 192 | 1.5 of 2 | 2.428e12 | 2.179e12 |
| 256 | 2 of 2 | 3.076e12 | 2.795e12 |

In particular, 192 settings is slower per row evaluation than 128 because it
executes two setting-loop passes while only half of the second pass is active.
Production setting counts should be multiples of 128.

## MUFU-Heavy Check

A second sweep used `depth3_mufu`, which applies hash-selected `sin`, `cos`,
`ex2`, and protected reciprocal-square-root leaf operations before the same
depth-3 reduction. These are single runs with 10 timed iterations:

| settings | score rows/s | score + reduce rows/s |
|---:|---:|---:|
| 64 | 6.066e11 | 5.610e11 |
| 128 | 9.238e11 | 8.701e11 |
| 256 | 9.268e11 | 9.025e11 |
| 384 | 9.282e11 | 9.125e11 |
| 512 | 9.290e11 | 9.182e11 |
| 768 | 9.298e11 | 9.218e11 |

The heavier AST reaches its score plateau by 128 settings. At 256 settings it
has 99.7% of the 768-setting score rate and 97.9% of the score-plus-reduce rate.
The practical setting threshold therefore does not increase for heavier ASTs.

## Literal One-Kernel Control

The sweep was also repeated with one module containing exactly one kernel and
32 ASTs. One million rows still supply enough CTAs to saturate AST scoring:

| settings | score rows/s | score + reduce rows/s |
|---:|---:|---:|
| 128 | 2.899e12 | 1.037e12 |
| 256 | 3.020e12 | 1.577e12 |
| 384 | 3.054e12 | 1.903e12 |
| 512 | 3.066e12 | 2.109e12 |
| 768 | 3.076e12 | 2.376e12 |
| 1024 | 3.078e12 | 2.529e12 |

The one-kernel score rate has the same knee: 128 settings delivers 94.2% of the
1024-setting rate and 256 delivers 98.1%. The combined rate keeps increasing
because the lone reduction launch adds approximately 2.4 to 2.7 ms per timed
iteration. With several kernels launched concurrently, this overhead overlaps
and is much less significant. It should not be interpreted as additional AST
score throughput from larger setting batches.

## Recommendation

- Use setting counts that are multiples of 128.
- Use 256 settings when score-only throughput and smaller batches matter most.
- Use 384 settings when accepting about a 3% end-to-end throughput gap is useful.
- Use 512 settings as the general default for the multi-kernel MSE path. It is
  within about 1% of the measured 1024-setting rate at half the batch size.
- Increase beyond 512 primarily when a larger search batch is useful in its own
  right, not to obtain materially higher AST evaluation throughput.

## Pooled Patch Site Full-Padding Baseline

The production generator now emits one pooled patch island per kernel. The
island exposes the 8 shared expression inputs, the target, and one live SSE
accumulator per generated AST capacity. The SASS patcher emits each AST and its
error/SSE update immediately, then reuses its prediction and error temporaries
for the next AST. SSE registers remain live because the CUDA epilogue atomically
adds them after the row loop.

This is the lower-register-pressure alternative described in the original
design note. It avoids keeping 32 prediction outputs live and has these concrete
properties:

- one combined shared-load wait per row;
- one patch span and one optional branch per kernel;
- pooled instruction capacity across all ASTs;
- temporary register growth based on the largest live AST state, not the sum of
  all AST temporaries;
- runtime `num_asts` masking of final atomic outputs.

A July 12, 2026 RTX 5090 run used 8 kernels, 32 depth-3 ALU ASTs/kernel, 1,024
settings, 32 columns, 64-row tiles, 1,048,576 rows, and 10 iterations:

```text
row evals/s             = 4.720e12
patch ASTs/s/core       = 2.097e6
patch time/AST          = 0.477 us
module load time/AST    = 0.238 us
original registers      = 66
expanded registers      = 68
patched registers       = 70
```

The patch-focused preset reached `3.344e6 ASTs/s/core` (`0.299 us/AST`) over
4,096 ALU-reduction ASTs. These numbers supersede the per-AST-site patch rates
from the earlier separate-site implementation. The prefix-only patcher that
supersedes this full-padding implementation is measured in
[`RTX_5090_ACTIVE_CAPACITY.md`](RTX_5090_ACTIVE_CAPACITY.md). The earlier
setting sweeps remain useful as historical runtime-shape evidence.

## 32-AST Module Load Sweep

One pooled site per kernel does not require one kernel per module. Module
load was measured with the current separate-site skeleton while holding the
total at 16,384 ASTs and using 32 ASTs/kernel. Runtime work was minimized with
one setting, one row, and one timed iteration. The ranges below come from
distinct depth-3 ALU and MUFU AST instruction mixes.

| kernels/module | ASTs/module | modules | cubin/module | total load | load/AST |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 512 | 0.09 MiB | 29.4-35.2 ms | 1.79-2.15 us |
| 2 | 64 | 256 | 0.17 MiB | 12.7-16.7 ms | 0.77-1.02 us |
| 4 | 128 | 128 | 0.34 MiB | 11.4-12.1 ms | 0.70-0.74 us |
| 8 | 256 | 64 | 0.67 MiB | 9.9-10.4 ms | 0.61-0.64 us |
| 16 | 512 | 32 | 1.33 MiB | 9.6-9.7 ms | 0.58-0.59 us |
| 32 | 1,024 | 16 | 2.65 MiB | 8.2-8.9 ms | 0.50-0.54 us |
| 64 | 2,048 | 8 | 5.30 MiB | 9.9 ms | 0.60 us |
| 128 | 4,096 | 4 | 10.59 MiB | 4.1-4.3 ms | 0.25-0.26 us |

The first load of the 128-kernel shape took 10.96 ms. Loads with different
patched ALU and MUFU instruction mixes took 4.1-4.3 ms after the loader was
warm, so both first-use latency and steady-state throughput should be retained
when interpreting that point.

Relative to one kernel/module:

- 8 kernels/module loads about 3.0-3.4 times faster;
- 32 kernels/module loads about 3.6-4.0 times faster;
- 128 kernels/module loads about 7.1-8.1 times faster after warm-up.

The practical initial shape is one island with 32 ASTs per kernel and 8-32
kernels per module. Eight kernels already remove most fixed module overhead.
Thirty-two improve load amortization further without creating the approximately
10.6 MiB skeleton produced by 128 kernels/module. The largest shape is useful
when the CUDA-to-cubin skeleton compilation is cached and minimum steady-state
module-load cost matters more than time to first template.
