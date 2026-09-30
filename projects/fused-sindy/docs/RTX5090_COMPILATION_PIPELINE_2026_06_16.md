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

# RTX 5090 Compilation Pipeline Benchmarks

Date: 2026-06-16

Host:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 610.43.02
- Target: `sm_120`
- Repo commit: `756140c`

These runs measure the current patched AST compiler path and the current Gram
runtime path after the native CUDA reference-test merge. The compile benchmark
reports AST features/s, where one Gram kernel contains 32 AST-backed implicit
features.

## Compile Hot Path, 1024 Modules

Command shape:

```sh
./build/implicit_sindy_ast_compile_api_bench \
    --modules 1024 \
    --kernels 8 \
    --workers 1,2,4,8,12,24 \
    --mode MODE \
    --repeats 1 \
    --warmup 0
```

The `maximal` mode used 256 modules because the 1024-module single-worker row is
intentionally very slow. That row is useful as an upper-bound AST-density
calibration, not as a normal search configuration.

| Mode | Modules | Workers | Time ms | Modules/s | Gram kernels/s | AST features/s |
|---|---:|---:|---:|---:|---:|---:|
| simple | 1024 | 1 | 62045.492 | 16.504 | 132.032 | 4225.029 |
| simple | 1024 | 2 | 32804.057 | 31.216 | 249.725 | 7991.207 |
| simple | 1024 | 4 | 24214.836 | 42.288 | 338.305 | 10825.760 |
| simple | 1024 | 8 | 13178.901 | 77.700 | 621.600 | 19891.188 |
| simple | 1024 | 12 | 9125.669 | 112.211 | 897.688 | 28726.004 |
| simple | 1024 | 24 | 4966.066 | 206.199 | 1649.596 | 52787.057 |
| polynomial | 1024 | 1 | 92651.142 | 11.052 | 88.418 | 2829.366 |
| polynomial | 1024 | 2 | 48403.282 | 21.156 | 169.245 | 5415.831 |
| polynomial | 1024 | 4 | 35318.013 | 28.994 | 231.950 | 7422.388 |
| polynomial | 1024 | 8 | 19344.492 | 52.935 | 423.480 | 13551.351 |
| polynomial | 1024 | 12 | 13564.553 | 75.491 | 603.927 | 19325.665 |
| polynomial | 1024 | 24 | 7276.435 | 140.728 | 1125.826 | 36026.434 |
| random | 1024 | 1 | 141604.070 | 7.231 | 57.851 | 1851.246 |
| random | 1024 | 2 | 76041.785 | 13.466 | 107.730 | 3447.368 |
| random | 1024 | 4 | 54907.228 | 18.650 | 149.197 | 4774.308 |
| random | 1024 | 8 | 30199.899 | 33.907 | 271.259 | 8680.294 |
| random | 1024 | 12 | 21031.824 | 48.688 | 389.505 | 12464.159 |
| random | 1024 | 24 | 11155.350 | 91.795 | 734.356 | 23499.397 |
| maximal | 256 | 1 | 88415.765 | 2.895 | 23.163 | 741.225 |
| maximal | 256 | 2 | 46890.362 | 5.460 | 43.676 | 1397.643 |
| maximal | 256 | 4 | 36146.429 | 7.082 | 56.658 | 1813.070 |
| maximal | 256 | 8 | 19912.224 | 12.856 | 102.851 | 3291.245 |
| maximal | 256 | 12 | 14323.106 | 17.873 | 142.986 | 4575.544 |
| maximal | 256 | 24 | 7681.340 | 33.328 | 266.620 | 8531.845 |

## Runtime Pipeline

Command shape:

```sh
./build/implicit_sindy_runtime_pipeline_bench \
    --modules 1 \
    --kernels 4 \
    --settings 2048 \
    --train-rows 131072 \
    --validation-rows 65536 \
    --rhs 1 \
    --sweeps 4 \
    --solve SOLVE \
    --mode polynomial \
    --compile-workers 12 \
    --warmup 1 \
    --repeats 5
```

| Solve | Total best ms | Train Gram ms | Validation Gram ms | Solve ms | MSE ms | Settings/s |
|---|---:|---:|---:|---:|---:|---:|
| STLSQ | 633.107 | 421.541 | 211.074 | 0.344 | 0.147 | 12939.365 |
| POSV | 633.153 | 421.572 | 211.211 | 0.256 | 0.115 | 12938.412 |

The solve and MSE phases are very small compared with Gram construction at this
row count. STLSQ has almost no impact on total runtime here.

## Reserved Cubin Runtime

Command shape:

```sh
./build/implicit_sindy_reserved_cubin_runtime_bench \
    --kernels 4 \
    --settings 2048 \
    --rows 131072 \
    --rhs 1 \
    --mode MODE \
    --warmup 1 \
    --repeats 5
```

| Mode | Compile ms | Run best ms | Settings/s | Row-settings/s | Feature-row-settings/s |
|---|---:|---:|---:|---:|---:|
| simple | 32.534 | 417.277 | 19632.040 | 2.573e9 | 8.234e10 |
| polynomial | 46.531 | 420.462 | 19483.351 | 2.554e9 | 8.172e10 |
| maximal | 185.325 | 555.430 | 14748.925 | 1.933e9 | 6.186e10 |

The patched device-function path gives a large compile-time win over full Gram
recompilation, but it gives up some runtime throughput because the AST evaluator
is an uninlined device function. For current search workloads, the compile-time
reduction is the more important tradeoff.

## Large Module Count Runtime Pipeline Runs

After the 1024-module compile sweep, the larger-module request was clarified as
runtime-focused. The long compile sweep was stopped and these runs measure the
GPU pipeline with larger module counts.

Command shape:

```sh
./build/implicit_sindy_runtime_pipeline_bench \
    --modules MODULES \
    --kernels 4 \
    --settings 2048 \
    --train-rows 131072 \
    --validation-rows 65536 \
    --rhs 1 \
    --sweeps 4 \
    --solve stlsq \
    --mode polynomial \
    --compile-workers 12
```

The benchmark still compiles the requested modules during setup, but the table
below reports the timed GPU phases. The 128-module row used `warmup=0` and
`repeats=1` to keep the wall-clock time reasonable; the smaller rows used a
warmup and repeated timed passes.

| Modules | Cohorts | Warmup | Repeats | Total best ms | Train Gram ms | Validation Gram ms | Solve ms | MSE ms | Settings/s | Row-settings/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 1 | 5 | 633.107 | 421.541 | 211.074 | 0.344 | 0.147 | 12939.365 | 2.546e9 |
| 4 | 16 | 1 | 3 | 2540.147 | 1691.860 | 846.140 | 1.502 | 0.645 | 12900.041 | 2.538e9 |
| 16 | 64 | 1 | 3 | 10158.873 | 6766.066 | 3384.235 | 6.006 | 2.567 | 12902.219 | 2.539e9 |
| 64 | 256 | 1 | 2 | 40728.848 | 27129.576 | 13564.925 | 24.085 | 10.264 | 12872.645 | 2.533e9 |
| 128 | 512 | 0 | 1 | 81624.898 | 54313.492 | 27243.008 | 47.895 | 20.505 | 12846.276 | 2.528e9 |

Runtime scales almost exactly linearly with module count while preserving
throughput. The useful summary is that this 5090 sustains about `12.85k` to
`12.94k` settings/s for the full train-Gram + validation-Gram + STLSQ + MSE
pipeline at `131072` train rows and `65536` validation rows.

The solve and MSE phases also scale linearly, but they remain tiny compared with
Gram construction. At 128 modules and 512 cohorts, the combined solve and MSE
time is about `68.4 ms` out of an `81.6 s` pipeline pass.
