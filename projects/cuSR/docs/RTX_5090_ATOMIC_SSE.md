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

# RTX 5090 Atomic SSE Output

Production AST-SASS benchmarks measured July 11, 2026 on the local RTX 5090
with CUDA 13.1 and `sm_120`. The generated tile-static MSE kernel has one output
contract: every CTA atomically adds its row-tile SSE directly to the final
`[ast][setting]` value. Timing includes clearing that output before every launch.
The kernel does not divide by the row count because search can rank candidates
by raw SSE.

## One-Site Full-Padding Baseline

On July 12, 2026 the dynamic kernel changed from one independently padded SASS
site per AST to one pooled site per kernel. The patcher now emits every AST and
its error/SSE accumulation into that site, reuses temporary registers between
ASTs, and branches over unused padding once.

With 8 kernels, 32 depth-3 ALU ASTs/kernel, 1,024 settings, 32 columns,
64-row tiles, 1,048,576 rows, and 10 iterations, the new shape measured:

| metric | result |
|---|---:|
| row evaluations/s | `4.720e12` |
| patch ASTs/s/core | `2.097e6` |
| patch time/AST | `0.477 us` |
| module load time/AST | `0.238 us` |
| original / expanded / patched registers | `66 / 68 / 70` |

The patch-focused 4,096-AST preset measured `3.344e6 ASTs/s/core`, or
`0.299 us/AST`. Earlier tables below predate pooled site patching and are
retained as evidence for tile, CTA, packing, and atomic-output decisions.
The current prefix-only patcher no longer writes unused site padding; see
[`RTX_5090_ACTIVE_CAPACITY.md`](RTX_5090_ACTIVE_CAPACITY.md).

## Shape

```text
modules             = 1
kernels/module      = 8
ASTs/kernel         = 32
ASTs total          = 256
settings            = 1,024
rows                = 1,048,576
columns             = 32
tile rows           = 128
timed iterations    = 30
AST mode            = depth3_alu
```

The initial decision measurement used the median of three independent Release
build runs. Each timed run evaluated `8,246,337,208,320` AST-setting rows.

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DCUSR_SASS_GPU_NAME=sm_120
cmake --build build-release -j --target cusr_bench
./build-release/cusr_bench --preset runtime --tile-rows 128 --check-rows 0 --run-iters 30
```

The initial decision sweep measured direct atomic SSE at `3.171e12` row evals/s
with 128-row tiles. It was approximately 5.2% faster than the removed
score-plus-reduction implementation while reducing score storage from 8 GiB of
row-tile scratch plus 1 MiB of final output to only the 1 MiB final output.

This initial output-policy comparison used 128-row tiles for both paths. The
later tile sweep below found that the 64-row atomic path reaches `3.695e12`
row evals/s at 32 columns, so 64 rows is now the atomic default.

## Correctness

The GPU test independently generates, inspects, and patches 64-, 128-, and
256-row kernels with the same nine AST programs. Its 257-row input exercises
full and tail tiles, and all three outputs are checked against the standalone
CPU AST interpreter and against one another.

Atomic accumulation order is not deterministic, so bitwise reproducibility is
not expected.

## Row Tile Sweep

A follow-up sweep kept the CTA at 128 threads and varied the static row tile.
A 64-row tile uses only the first 64 threads while loading shared memory, then
all 128 threads evaluate settings. A 256-row tile has every thread perform two
coalesced load passes. Smaller tiles increase the number of CTAs and final
atomic additions but reduce shared memory per CTA.

The table reports three-run median atomic-SSE throughput with 1,024 settings,
32 ASTs/kernel, 256 total ASTs, 1,048,576 rows, and depth-3 ALU ASTs.

| columns | 64 rows | 128 rows | 256 rows | best resident blocks/SM |
|---:|---:|---:|---:|---:|
| 8 | `3.856e12` | `3.863e12` | `3.826e12` | 7 |
| 16 | `3.833e12` | `3.852e12` | `3.200e12` | 7 |
| 24 | `3.816e12` | `3.840e12` | `2.061e12` | 7 |
| 26 | `3.710e12` | `3.500e12` | not measured | 7 versus 6 |
| 28 | `3.698e12` | `3.494e12` | not measured | 7 versus 6 |
| 30 | `3.688e12` | `3.136e12` | not measured | 7 versus 5 |
| 32 | `3.695e12` | `3.150e12` | `1.403e12` | 7 versus 5 versus 2 |

The corresponding per-CTA dynamic shared allocations are:

| columns | 64 rows | 128 rows | 256 rows |
|---:|---:|---:|---:|
| 8 | 2,560 bytes | 5,120 bytes | 10,240 bytes |
| 16 | 4,608 bytes | 9,216 bytes | 18,432 bytes |
| 24 | 6,656 bytes | 13,312 bytes | 26,624 bytes |
| 32 | 8,704 bytes | 17,408 bytes | 34,816 bytes |

At 24 or fewer columns, 64 and 128 rows both retain seven blocks per SM. The
128-row tile is only 0.2% to 0.6% faster there because it performs half as many
tile loads and atomics. At 26 columns, the 128-row tile drops to six resident
blocks and 64 rows becomes about 6% faster. At 30 and 32 columns it drops to
five blocks, making 64 rows approximately 17% faster.

The 32-column result remains stable with fewer settings. At 128 settings,
64 rows measured `3.434e12` versus `2.963e12` for 128 rows. At 256 settings it
measured `3.552e12` versus `3.042e12`. Depth-3 MUFU ASTs also favored 64 rows,
measuring `9.671e11` versus `9.305e11`.

The 257-row correctness case exercises a one-row tail for all three tile sizes.
The 64-row kernel's maximum absolute MSE error versus the CPU interpreter was
`2.38419e-07`; its maximum difference from the 128-row atomic path was
`1.78814e-07`.

The 64-row tile is now the default because its low-column penalty is less than
1% while its high-column gain is substantial. All three tile sizes remain
available for architecture-specific tuning.

### ASTs Per Kernel

Reducing packing from 32 to 8 ASTs lowers the patched register count from 69 to
45. With 64-row tiles this raises residency from 7 to 10 blocks per SM. The
following three-run medians use 8 kernels with 8 ASTs each, 1,024 settings,
1,048,576 rows, and depth-3 ALU ASTs:

| columns | 64 rows | 128 rows | 64-row change | resident blocks/SM |
|---:|---:|---:|---:|---:|
| 8 | `3.313e12` | `3.336e12` | `-0.7%` | 10 versus 10 |
| 32 | `3.140e12` | `2.601e12` | `+20.7%` | 10 versus 5 |

Repeating the test with 32 kernels and 8 ASTs each, preserving the original 256
total ASTs, produced rates within approximately 0.5% of this table. The result
therefore reflects AST packing rather than an insufficient number of kernels.

Despite its higher residency, the 8-AST shape is slower than packing 32 ASTs.
At 8 columns, 32 packed ASTs with 64-row tiles reached `3.856e12`, about 16%
above the 8-AST result. At 32 columns it reached `3.695e12`, about 18% higher.
Packing 32 ASTs amortizes each tile load and settings setup across four times as
many expression bodies. Eight ASTs remains a reasonable smaller-cohort option
when independent search settings are worth an approximately 15% lower row
evaluation rate.

A fixed-total sweep used 256 ASTs, 1,024 settings, 32 columns, 64-row tiles,
and adjusted the kernel count to keep the total AST count unchanged. The ALU
program has 8 leaves and 7 hash-selected binary operations. The MUFU program
adds one `sin`, `cos`, `ex2`, or protected `rsqrt` operation to every leaf
before the same 7 binary reductions.

| ASTs/kernel | ALU row evals/s | MUFU row evals/s | patched registers | blocks/SM |
|---:|---:|---:|---:|---:|
| 8 | `3.138e12` | `9.560e11` | 45 / 46 | 10 |
| 16 | `3.641e12` | `9.649e11` | 53 / 54 | 9 |
| 32 | `3.701e12` | `9.671e11` | 69 / 70 | 7 |

Sixteen ASTs retains 98.4% of the 32-AST ALU throughput and 99.8% of its MUFU
throughput. It also halves the settings cohort, keeps two additional CTAs
resident, and leaves more register headroom for larger ASTs. It is therefore
the recommended single general-purpose kernel shape. Thirty-two ASTs remains
the light-ALU throughput ceiling when sharing one settings stream across the
larger cohort is acceptable.

## Atomic SSE As The Default

For 1,048,576 rows, every AST-setting result receives this many row-tile
contributions:

| tile rows | atomic additions per result |
|---:|---:|
| 64 | 16,384 |
| 128 | 8,192 |
| 256 | 4,096 |

The 64-row kernel therefore performs twice as many final atomic additions as
the 128-row kernel and four times as many as the 256-row kernel. It still wins
at high column counts. The 8-column case also shows that changing the atomic
count by 2x changes throughput by less than 1% when occupancy is unchanged.
Together with the 128- and 256-setting sweeps, this indicates that final FP32
SSE atomics are not the limiting resource in the current kernel.

Within a warp, lanes own different settings and issue atomics to adjacent,
distinct output addresses. Contention occurs only when CTAs processing different
row tiles contribute to the same AST-setting result.

Direct atomic SSE is therefore the only production search path. The result
allocation is exactly `ASTs * settings * sizeof(float)` and must be cleared
before scoring. Atomic performance still needs validation on each supported
architecture.

## CTA Size

The CTA-size sweep held the recommended 64-row, 16-AST shape fixed and varied
only the generated CTA width. It used 16 kernels, 256 total ASTs, 1,024 settings,
1,048,576 rows, depth-3 ALU ASTs, and direct atomic SSE. The 32-column timings
below are from 20 iterations; the 8-column timings are from 10 iterations.

| columns | threads/CTA | blocks/SM | warps/SM | row evals/s |
|---:|---:|---:|---:|---:|
| 8 | 64 | 18 | 36 | `3.851e12` |
| 8 | 128 | 9 | 36 | `3.848e12` |
| 8 | 256 | 4 | 32 | `3.750e12` |
| 32 | 64 | 10 | 20 | `2.929e12` |
| 32 | 128 | 9 | 36 | `3.652e12` |
| 32 | 256 | 4 | 32 | `3.505e12` |

At 8 columns, 64 and 128 threads are effectively tied because both expose 36
resident warps. At 32 columns, the fixed 8,704-byte shared allocation limits the
64-thread CTA to 20 resident warps, making it 19.8% slower than 128 threads.
The 256-thread CTA reduces settings-loop batches and reaches 32 resident warps,
but remains 4.0% slower than 128 threads in the 32-column case.

CPU-checked runs also covered non-multiple setting counts at both non-default
CTA widths. The 64-thread run used 129 settings and reported `1.526e-5` maximum
absolute SSE error; the 256-thread run used 257 settings and reported
`1.144e-5`, both against the CPU evaluator.

The 128-thread CTA is the strongest one-shape default because it balances
settings parallelism, register allocation, shared memory per resident warp, and
CTA scheduling. The complete recommended general-purpose shape is:

```text
64 rows/tile
128 threads/CTA
16 ASTs/kernel
16 kernels/module
256 ASTs/module
1,024 settings
atomic SSE output
```

The measured sweep supports 128 threads as the general-purpose default. A
64-thread CTA is viable for narrow input tiles but degrades sharply once shared
memory limits resident warps. A 256-thread CTA is competitive, but did not beat
128 threads in either measured column shape.
