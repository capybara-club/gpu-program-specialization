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

# Request sizing results — 2026-09-12

Two RTX 5080s on rack1; one- and two-GPU modes. Every comparison used the same
4,096 finite ASTs. Positive constant counts share 256 Philox rows (1,048,576
evaluations); zero constants means 4,096 evaluations. The full reports verify
exact retained addresses, coefficients, programs, MSEs and population counts.
A winner in each mode was independently replayed on CPU.

Warm medians below use four unprofiled runs after one warm-up. Short/long AST
refers to whether the common expression includes sixteen additional nonlinear
state products. The RK4 column is actual FP32 subdivision work.

| GPUs | States | Constants | AST | RK4 steps | Previous defaults | Automatic |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 2 | 0 | shorter | 2 | 30.70 ms | 25.71 ms |
| 1 | 2 | 1 | shorter | 2 | 31.61 ms | 28.48 ms |
| 1 | 6 | 8 | shorter | 2 | 38.85 ms | 35.39 ms |
| 1 | 6 | 8 | longer | 2 | 40.45 ms | 38.32 ms |
| 1 | 12 | 24 | shorter | 2 | 58.15 ms | 53.19 ms |
| 1 | 12 | 24 | longer | 2 | 61.47 ms | 55.64 ms |
| 1 | 6 | 8 | shorter | 40 | 48.10 ms | 47.75 ms |
| 1 | 12 | 24 | longer | 40 | 81.91 ms | 82.47 ms |
| 2 | 2 | 0 | shorter | 2 | 34.86 ms | 25.06 ms |
| 2 | 2 | 1 | shorter | 2 | 35.00 ms | 26.26 ms |
| 2 | 6 | 8 | shorter | 2 | 38.24 ms | 29.74 ms |
| 2 | 6 | 8 | longer | 2 | 39.38 ms | 31.38 ms |
| 2 | 12 | 24 | shorter | 2 | 50.24 ms | 40.84 ms |
| 2 | 12 | 24 | longer | 2 | 52.81 ms | 44.05 ms |
| 2 | 6 | 8 | shorter | 40 | 43.06 ms | 43.28 ms |
| 2 | 12 | 24 | longer | 40 | 63.08 ms | 62.46 ms |

Short-rollout requests improve by 5–16% on one GPU and 17–28% on two. Longer
rollouts are within about 1% of the previous settings after correcting module
and page selection. These small jobs can be faster on one GPU: per-device setup
is a significant part of their cost. This is a starting policy for these GPUs,
not evidence that two GPUs or automatic sizing always wins.

## Full request throughput

One million six-equation AST combinations × 2,048 coefficient configurations;
three observations, four RK4 steps per configuration, existing distinct retention.

| Mode | Warm median | ASTs/page | Tiles |
|---|---:|---:|---:|
| explicit | 1.2294 s | 1024 | 977 |
| automatic | 1.2432 s | 1024 | 977 |
| profiled | 1.2507 s | 1024 | 977 |

Automatic throughput is **1.65 billion configurations/s** across both GPUs.
Optional profiling added about **1.7%** against the explicit unprofiled median.
Report times exclude client initialization/transport and final serialization.
The first client call additionally initializes contexts and helper modules.

## Correctness and limitations

- 159/159 broader runtime robustness cases, totaling 67,428,341 configurations.
- Ten cap/profile/chunk/device invariance cases with CPU replay.
- Allocation-free parser audit, 18,000 mutation calls, retention ASan/UBSan,
  SQLite cache checks and 108 Python grammar tests passed.
- Native public-export, template, scoring, CUB reducer and Philox tests passed.
- Eight existing queue protocol cases and ten new allocation-admission cases
  passed on an isolated mac3 broker without GPU claim.
- Ten combined mac3→rack1 requests passed, including all six retained fixtures,
  family caps and 2.048B evaluations in explicit, automatic and profiled modes.
- [Rejected policies and the tiny-template restriction](DEVIATIONS.md) are
  preserved explicitly. Automatic selection starts at two systems; the core
  inspector is not claimed to support every explicit one-system template.
- No GPU-stage overlap, new solver, search-space expansion, or recovery claim.
  Small-k CUB hierarchy tuning and coverage on other GPUs remain follow-ups.

Raw results stay outside Git; `summary.json` is the compact numeric record.

Final live generation `f97820eb98636a74c57be2108e891e23` passed ten complete
requests after a fresh worker restart and on-disk/mapped-library identity checks.
The automatic 2.048B request took 1.2456 s in the C worker. See
[deployment identities and checks](deployment.json). The first pre-restart live
run is excluded as documented in DEVIATIONS.md.
