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

# Long-rollout tile diagnosis

**Completed diagnosis:** [full report](REPORT.md). The four original requests
improved from 222.64s to 41.57s in the isolated larger-tile runtime, with exact
retained-result and coverage agreement. The live service is unchanged.

This benchmark investigates the system-19 structural screens: 317,194,240
configuration evaluations in 223.078 seconds on two RTX 5080s. Every evaluation
schedules 5,120 RK4 steps. The live runtime's 67,108,864 aggregate-work ceiling
limits scoring tiles to 13,107 configurations, about 103 CUDA blocks.

The earlier automatic-sizing validation covered 2 and 40 RK4 steps/configuration;
the million-AST throughput fixture used four. System 19's work is 128 times the
longest configuration in that sizing grid. Those measurements did not establish
saturation for long scientific trajectories.

## Isolation and comparison

- `build_probe.py` copies the existing runtime to an isolated directory and links
  the existing core library. It changes no live source or installed library.
- The diagnostic header reads `ODZ_BENCH_TILE_WORK_UNITS`, set and bounded by the
  replay driver. It is an operator benchmark control, not a supported request
  field or proposed production setting. Reports expose the actual probe cap.
- `replay.py` runs the native C API directly with the same per-GPU pool sizes and
  attempt-owned numeric inputs. NATS transport is excluded. Grammar and numeric
  seeds, observations, integration step, family allocations and retention stay
  fixed within a comparison. An optional explicitly labelled family/configuration
  prefix permits quick pilots before full-request replays.
- Request timings include C parsing and one-time AST generation, reported
  separately. GPU scoring event intervals begin after ASTs, constants and
  trajectories are prepared. Optional `--prepared` repeats the first already
  packed scoring call three times without regeneration and compares every raw
  score bit. These extra repetitions are diagnostic work, excluded from ordinary
  request timing comparisons. They still include specialization/module loading.
- Exact retained candidate IDs, programs, coefficient bits, scores, leaderboards,
  valid/invalid counts and evaluation counts must agree. Chunk counts and launch
  shapes may change. Optional tracing records the first eight module-containing
  calls per GPU; it is not a complete timeline or a maximum-duration metric.
- Warm unprofiled repetitions establish throughput; CUDA-event and Nsight runs
  establish timing/occupancy attribution and are labelled separately. Host wait
  time is not achieved occupancy. Request speedups are not blind-recovery speedups.

## Experiment sequence

1. Establish a diagnostic-build baseline against the deployed library at the
   unchanged work cap, including exact retained-result equality.
2. Sweep work caps with all other input data fixed, first on one GPU; inspect
   actual tile sizes, module sizes, CUDA durations and warm request time.
3. Check the chosen region with fixed module/configuration settings to separate
   work-cap effects from the automatic packing/controller effects.
4. Profile representative small and larger launches. Existing installed Nsight
   tools may be used; do not change driver permissions to obtain counters.
5. Replay the full four original structural requests on two GPUs and compare
   exact results to the saved service reports. Preserve the original scientific
   objective and all original work. Report any deviation before drawing a speed
   conclusion.

Request fixtures contain the supplied observations and are research artifacts,
not generic public examples. Raw requests/results and profiler captures should
not be committed. Keep compact findings and reproduction scripts in the repo.

## Initial measurements (superseded by the full report)

One RTX 5080, complete original `polyfit` request (5,242,880 configurations),
all sixteen full trajectories. The deployed library and isolated probe at the
original cap took 6.820180s and 6.820797s respectively in their warm runs; exact
retained candidates, rankings, valid/invalid counts and evaluation counts agree.
The cap sweep completed two warm repetitions after one first run per setting.
The last warm runs were 1.864966s at a 268,435,456-work-unit cap, 1.357847s at
1,073,741,824, and 1.186268s at 4,294,967,296. These are individual run times;
the final report will tabulate medians from the retained raw reports.

Prepared-call diagnostics (one original family, same bank prefix):

| Configurations in launch | CUDA blocks | Median scoring GPU time, 3 repeats |
|---:|---:|---:|
| 13,107 | 103 | 16.35184 ms |
| 524,288 | 4,096 | 100.08551 ms |

The larger launch does 40 times the work in about 6.12 times the GPU duration.
All raw score bits matched each call's prepared replays; complete retained
results also match between the two tilings. Preparation is outside those CUDA
intervals. This establishes a major launch-size effect without changing kernels.
It does not measure achieved SM occupancy or prove a universal tile optimum.

Nsight Compute 2025.4.1 is installed, but its hardware-counter attempt returned
`ERR_NVGPUCTRPERM`. No driver settings or privileges were changed. The failed
counter run is excluded from performance comparisons. CUDA-event profiling
works and supplies the durations above. Full two-GPU request comparisons and
fixed-shape controls are now complete and documented in REPORT.md.

## Authorization record

Automatic approval review initially rejected transferring the four saved JSON
requests and diagnostic scripts to rack1 and building the isolated probe because
explicit payload/destination authorization was required. The transfer was not
retried or routed indirectly. An approval question naming the payload and
`/home/cdurham/experiments/recovery19-throughput-20260913` was presented. The user
then explicitly answered **yes**, authorizing those transfers, the isolated
build and benchmark execution. Work resumed under that approval.
