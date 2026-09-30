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

# Oversubscribed LM shape comparison

Measured on rack1 RTX 5080 device 0, 84 SMs. Rates are thousands of complete LM fits per second across the entire GPU, using median kernel event time from three launches. All shapes use the ordinary CUDA prototype, including one lane; none uses the SASS-specialized fitter. Compilation, module loading, transfers and search generation are excluded.

Each row uses the largest population measured for all four widths. The 3-state/3-constant row uses the larger completed follow-up. Larger populations repeat the same 1,024 starts. Constants means fitted coefficients. Each workload has four training trajectories, 21 samples per trajectory, 32 RK4 substeps per interval and at most 32 LM iterations. Convergence work differs between systems, so cross-row ratios are not pure state-count scaling.

| States | Constants | Fits/launch | 1 lane | 2 lanes | 4 lanes | 8 lanes | Best eligible lanes | Minimum resident waves |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 786,432 | **1,388.84** | 1,334.02 | 938.26 | 460.92 | 1 | 14.63 |
| 3 | 6 | 262,144 | 110.09 | 135.56‡ | **113.97** | 105.53 | 4 | 8.13 |
| 3 | 8 | 262,144 | **53.05** | 52.72 | 45.83 | 34.82 | 1 | 12.19 |
| 6 | 3 | 131,072 | **876.17** | 790.75 | 537.31 | 277.79 | 1 | 4.06 |
| 6 | 6 | 262,144 | 64.07 | 76.70 | **80.45** | 76.97 | 4 | 12.19 |
| 6 | 8 | 262,144 | 22.64† | 26.14 | **26.91** | 25.63 | 4 | 12.19 |
| 12 | 3 | 131,072 | **505.10** | 415.27 | 302.75 | 153.21 | 1 | 6.10 |
| 12 | 6 | 262,144 | 17.11† | **42.60**† | 41.74 | 39.10 | 2 | 12.19 |
| 12 | 8 | 131,072 | 4.22† | 10.46† | **12.31** | 9.54 | 4 | 6.10 |
| 16 | 3 | 131,072 | 248.63† | **328.34** | 232.04 | 117.94 | 2 | 6.10 |
| 16 | 6 | 262,144 | 5.66† | 10.80† | **16.72** | 16.22 | 4 | 12.19 |
| 16 | 8 | 131,072 | 2.31† | 4.07† | **9.39**† | 7.09 | 4 | 6.10 |

† Compiler-reported spilling. ‡ Known different optimizer path; excluded from best-shape selection. Its tiled reference checks pass, but its work/results are not equivalent to the one-lane baseline.

All displayed launches supply at least four predicted resident waves. Oversubscription does not imply 100% achieved warp occupancy or a confirmed asymptotic throughput plateau for every cell. Hardware occupancy counters were unavailable. Bold marks the highest measured eligible rate at this population; small differences should be treated as near ties.

The single-lane prototype is fastest in several smaller cases, while cooperative shapes win on larger fitted systems. This does not establish the fastest Odezza implementation: a matched, oversubscribed comparison against the SASS-specialized single-thread fitter is still needed, with equivalent equations, starts, observations, bounds and stopping behavior.

Sources: [main raw measurements](../validation/occupancy-sweep-01/results.json), [follow-up raw measurements](../validation/occupancy-small-followup-01/results.json), [prototype implementation boundary](README.md), [full scaling analysis](../validation/occupancy-sweep-01/OCCUPANCY.md).
