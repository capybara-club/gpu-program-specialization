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

# Python trajectory-LM validation on Rohini

> **Superseded:** These measurements predate both liveness hardening and the coupled RK4 sensitivity-stage repair. The materialized winner's primal CPU replay remains valid, but that replay did not validate the LM gradient or proposal path. Do not use the correctness or throughput claims below as evidence for the current implementation. Use [`2026-09-02-corrected-coupled-lm.md`](2026-09-02-corrected-coupled-lm.md).

This file remains as a historical record only.

The first Python-generated trajectory-LM template was compiled, physically inspected, specialized, and executed on Rohini's RTX 5090 (`sm_120`). The validation system has two states, eight optimized constants, three 21-point trajectories, four RK4 substeps per interval, one toggle bit, and optimized-constant index 2 reused across both right-hand sides.

## Correctness

| Check | Result |
|---|---:|
| Exact-start initial MSE | `3.81e-15` |
| Exact-start final MSE | `2.05e-15` |
| Perturbed best initial MSE | `1.96e-2` |
| Perturbed best final GPU MSE | `2.93e-15` |
| Independent CPU replay MSE | `3.24e-15` |
| Best correct-toggle MSE | `2.93e-15` |
| Best wrong-toggle MSE | `5.37e-6` |

The best fit recovered the correct toggle branch. Its eight fitted constants were within roughly `1.1e-5` absolute error of the generating values. Reusing one optimized-constant index in two right-hand sides produced one consistent fitted value and one accumulated derivative dimension.

A separate 258-fit launch deliberately left the final 128-thread CTA partially occupied. It completed and replayed correctly, verifying that inactive lanes participate in cooperative trajectory staging before exiting.

## Warmed throughput

The saturated measurement used 32,768 starts, two toggle permutations, and five timed repetitions after warmup:

| Metric | Result |
|---|---:|
| Fits per launch | 65,536 |
| Mean launch time | 150.84 ms |
| Complete fits/s | 434,465 |
| Mean LM iterations/fit | 5.963 |
| Mean accepted steps/fit | 5.923 |

This is a workload-specific rate for a small synthetic two-state system. It is not directly comparable to the earlier four-state fed-batch or hand-written native-control rates because their trajectory counts, observation counts, right-hand sides, and iteration behavior differ.

The initial Python derivative/SASS specializer processed about 352 systems/s on Mac1 over 100 repeated specializations of the validated system. That is prototype-control-plane throughput only; it is not the intended production hot path and does not measure GPU fitting.

## Register-pressure boundary

| States | Optimized constants | Registers | Stack | Spill stores | Spill loads |
|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 192 | 0 B | 0 B | 0 B |
| 4 | 6 | 215 | 0 B | 0 B | 0 B |
| 4 | 8 | 255 | 40 B | 52 B | 44 B |

The four-state/six-constant shape remains spill-free and matches the dimensional scale of the earlier fed-batch fitting kernel. The four-state/eight-constant shape is supported functionally but is a materially different, spilled execution regime. It should not be treated as the preferred thread-owned shape without a direct performance study; a smaller parameter bucket or split/subgroup topology is the likely next shape.

## Compilation deviation

The installed CUDA 13.1 `nvcc` path on Rohini currently encounters the known `rsqrt` declaration conflict with the host glibc headers. This validation used the installed NVRTC library to compile the same generated CUDA directly to a native CUBIN. No headers were patched and no alternate software was downloaded or built. The emitted PTXAS report for the validated two-state shape showed 192 registers, no stack, no spills, and 351.1 ms PTXAS time.

Raw data: [`raw/2026-09-02-rohini-python-lm-validation.json`](raw/2026-09-02-rohini-python-lm-validation.json)

## SM89 portability

Ada's RTX 4090 compiled the same two-state/eight-constant source to a 96,480-byte SM89 CUBIN using 178 registers with no stack or spills. Physical inspection, specialization, execution, and independent replay all passed. The same 65,536-fit saturated workload reached 401,557 complete fits/s.

The portability check exposed two physical differences that are now explicit in the inspector: SM89 uses its `IADD3` encoding for the site-specific toggle salt and PTXAS may reorder logical input markers. The inspector accepts the verified SM89 and SM120 integer-add encodings and derives the leading scaffold boundary from the earliest physical input marker.
