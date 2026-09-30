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

# Performance

FusedSINDy is optimized for search loops that change symbolic feature structures
often and evaluate many runtime settings per compiled structure.

The performance thesis is:

```text
avoid materializing generated feature matrices
+ compile feature structure into CUDA evaluator code
+ reuse each compiled structure across many settings
+ accumulate Gram/statistics directly
= better throughput for row-heavy, search-heavy workloads
```

These numbers are local measurements, not portable performance claims. Use them
to understand the shape of the tradeoffs and to sanity-check new machines.

## What To Remember

The current patch path is deliberately asymmetric:

- it reduces hot structural compile latency by avoiding full Gram-kernel
  recompilation;
- it gives up some GPU runtime throughput compared with fully inlined generated
  kernels;
- it is usually better when cohorts mutate frequently;
- full-inline kernels can win when one cohort is reused for extremely many
  rows/settings.

Always benchmark your workload against a simple explicit-materialization
baseline when possible.

## Compile Pipeline Matrix

```text
cold path: setup paid when the compiler/template/cache state is created
hot path:  work paid for every new AST cohort batch during search
```

| Pipeline | Gram compilation | Evaluator PTX generation | Evaluator PTX compilation | Linking/final cubin | Hot-path summary |
| --- | --- | --- | --- | --- | --- |
| Full dynamic Gram injection | Hot: full Gram/evaluator PTX compiled per module | Hot | Included in full-module compile | Hot direct cubin emit | Best runtime shape, but recompiles the full Gram kernel every cohort. |
| nvJitLink object-link prototype | Cold: Gram object prepared once | Hot | Hot: nvJitLink consumes evaluator PTX and does PTX-to-SASS work | Hot link | Avoids Gram recompilation, but linker work remains expensive. |
| Current cubin-function patch path | Cold: reserved Gram cubin/template prepared once | Hot | Hot: nvPTXCompiler compile-only evaluator cubin | Hot binary patch into reserved cubin | Removes Gram compile/link from hot path; only evaluator compile plus patch remains. |

## Previous Strategy Comparison

This direct local comparison used 240 modules, 12 workers, and 4 Gram kernels per
module on:

```text
CPU: AMD Ryzen 9 9900X, 12 cores / 24 hardware threads
GPU: NVIDIA GeForce RTX 5090
Driver: 610.43.02
Target: sm_120
```

| Pipeline | Modules | Workers | Wall ms | Gram kernels/s | AST features/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full dynamic Gram injection | 240 | 12 | 27400.022 | 35.036 | 1121.167 |
| nvJitLink object-link, no-cache random ASTs | 240 | 12 | 10060.998 | 95.416 | 3053.375 |
| Current cubin-function patch path | 240 | 12 | 1644.552 | 583.744 | 18679.860 |

In this matched comparison, the patch path is about `16.7x` faster than full
dynamic Gram injection in Gram kernels/sec and about `6.1x` faster than the
nvJitLink object-link prototype. This is why the current package uses patching
as the default compile strategy.

## Current Package Compile Scaling

This run uses the current package benchmark with 512 modules and 4 Gram kernels
per module. It should not be mixed directly with the 240-module strategy table
above.

| System | Workers | Compile ms | AST features/s | Gram kernels/s |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 + Ryzen 9900X | 1 | 23758.462 | 2758.428 | 86.201 |
| RTX 5090 + Ryzen 9900X | 2 | 12450.930 | 5263.543 | 164.486 |
| RTX 5090 + Ryzen 9900X | 4 | 6262.121 | 10465.464 | 327.046 |
| RTX 5090 + Ryzen 9900X | 8 | 3262.994 | 20084.623 | 627.644 |
| RTX 5090 + Ryzen 9900X | 12 | 2275.620 | 28799.190 | 899.975 |
| RTX 5090 + Ryzen 9900X | 24 | 1897.167 | 34544.147 | 1079.505 |
| GH200 | 1 | 40158.894 | 1631.917 | 50.997 |
| GH200 | 2 | 20016.975 | 3274.021 | 102.313 |
| GH200 | 4 | 9928.076 | 6601.078 | 206.284 |
| GH200 | 8 | 4957.284 | 13220.142 | 413.129 |
| GH200 | 16 | 2546.196 | 25738.788 | 804.337 |
| GH200 | 32 | 1409.292 | 46502.770 | 1453.212 |
| GH200 | 64 | 941.264 | 69625.533 | 2175.798 |

Raw files:

- `combined_compile_scaling_512_modules.csv`
- `combined_compile_asts_per_sec_512_modules.svg`
- `combined_compile_time_512_modules.svg`

## AST Density Effects

On the same RTX 5090 workstation, a 1024-module compile sweep showed that AST
operator density matters:

| Mode | Modules | Workers | Time ms | AST features/s |
| --- | ---: | ---: | ---: | ---: |
| simple | 1024 | 12 | 9125.669 | 28726.004 |
| polynomial | 1024 | 12 | 13564.553 | 19325.665 |
| random | 1024 | 12 | 21031.824 | 12464.159 |
| maximal | 256 | 12 | 14323.106 | 4575.544 |

Simple ASTs compile much faster than maximal/safe-transcendental-heavy ASTs.
This is expected: the generated evaluator PTX and final code are denser.

## Runtime Pipeline

RTX 5090 runtime pipeline, 4 cohorts, 2048 settings per cohort, 131072 train
rows, 65536 validation rows, 1 RHS, 4 solver sweeps:

| Solve | Total best ms | Train Gram ms | Validation Gram ms | Solve ms | MSE ms | Settings/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| STLSQ | 633.107 | 421.541 | 211.074 | 0.344 | 0.147 | 12939.365 |
| POSV | 633.153 | 421.572 | 211.211 | 0.256 | 0.115 | 12938.412 |

The solve and MSE phases are tiny compared with Gram construction at this row
count. STLSQ is cheap enough to use during search.

Large-module runtime runs preserve almost the same throughput:

| Modules | Cohorts | Total best ms | Settings/s | Row-settings/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 633.107 | 12939.365 | 2.546e9 |
| 4 | 16 | 2540.147 | 12900.041 | 2.538e9 |
| 16 | 64 | 10158.873 | 12902.219 | 2.539e9 |
| 64 | 256 | 40728.848 | 12872.645 | 2.533e9 |
| 128 | 512 | 81624.898 | 12846.276 | 2.528e9 |

These numbers include train Gram, validation Gram, solve, and MSE. The useful
summary for this RTX 5090 run is about `12.85k` to `12.94k` settings/s for the
full train/validation/STLSQ/MSE pipeline at the listed row counts.

## Runtime Penalty Of Patching

Matched local runtime comparison, 2048 settings, 131072 rows, 32 primitive
columns:

| Runtime path | Avg launch ms | Rows/s | Feature-rows/s | Penalty vs full-inline |
| --- | ---: | ---: | ---: | ---: |
| Full-inline generated Gram | 95.748 | 2.804B | 89.714B | baseline |
| Device-call generated Gram | 123.341 | 2.176B | 69.644B | +28.8% launch time |
| Current AST-patch Gram | 111.347 | 2.411B | 77.146B | +16.3% launch time |

The patch path pays a runtime penalty, but the compile-time reduction is usually
more important for search loops that mutate AST structure often.

## Memory Traffic Argument

For 1024 settings, 65536 rows, and 32 primitive columns:

```text
setting-rows = 1024 * 65536 = 67,108,864
primitive read lower bound = setting_rows * 32 * sizeof(float) = 8.00 GiB
target read lower bound    = setting_rows * sizeof(float)      = 0.25 GiB
```

If a system materializes the final 32 generated feature columns before Gram
construction, the final feature matrix alone is:

```text
1024 * 65536 * 32 * sizeof(float) = 8.00 GiB
```

That final-feature path pays at least an extra 8 GiB write plus 8 GiB read
beyond primitive reads, before expression intermediates. A generic array
expression workflow that materializes unary/binary intermediates can add far
more traffic and many extra launches.

This is not a benchmark of PySINDy, PySR, or PDE-FIND. It is the lower-bound
cost of implementing the same dynamic search in a materialized array style.

## Reporting Rules

For useful performance reports, include:

```text
CPU model and core count
GPU model
driver version
target sm
CUDA Python/package versions
module count
kernels per module
worker count
settings per cohort
rows
primitive feature count
RHS count
warmup and timed launches
cold create time vs hot compile time
Gram time
solve time
MSE time
```

Do not compare AST features/s or modules/s across different module shapes
without stating kernels/module and features/kernel.

