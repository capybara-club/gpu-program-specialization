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

# Rohini warp-per-system SSE benchmark

Date: 2026-08-31 EDT
GPU: NVIDIA GeForce RTX 5090 (`sm_120`)
Purpose: determine whether assigning one 32-setting system to each warp is a
better low-setting topology than assigning one system to an entire CTA.

## Result

Warp-per-system ownership is the new default candidate when a system has no
more than 32 settings. It processed the same ordered AST workload **2.39x**
faster than the saturated CTA-owned baseline.

| Ownership | Best streams | Concurrent CTA supply | Unique AST/s | M system configurations/s | Time/system over 32 settings | Registers |
|---|---:|---:|---:|---:|---:|---:|
| CTA-owned | 8 | 1,024 | 493,877 | 7.902 | 4.05 us | 81 |
| Warp-owned | 32 | 512 | **1,179,377** | **18.870** | **1.70 us** | 80 |

Each system contains two independent candidate RHS ASTs. Configuration
throughput counts one complete system genome under one setting, not two
individual AST/settings.

This does not replace the 512-setting CTA-owned shape. That shape reached 71.74
million configurations/s because it exposes much more independent setting work,
although it consumes ASTs more slowly at 280,234 AST/s. The crossover is:

- up to 32 settings: use warp-per-system ownership;
- hundreds or thousands of settings: use CTA-owned, one setting per thread;
- between 33 and roughly 256 settings: measure a multi-tile warp variant before
  choosing; that topology is not implemented yet.

## Matched workload

Both modes evaluated exactly:

- 131,072 byte-distinct system genomes;
- 262,144 byte-distinct candidate ASTs;
- 32 materialized settings per system;
- 4,194,304 system/settings configurations;
- two missing RHS expressions, three trajectories, and the same dense aligned
  fed-batch reference; and
- winner output with device-side reduction.

The ordered AST stream SHA-256 was
`d9cdcf91d08b54487b74436c723f51f1e2fdc1a7e6d81e964cbb5f9726c307ee`
for both modes. Both also required exactly 108,998 duplicate retries. AST
generation was made a function of global system index so changing module
boundaries cannot change the workload.

The timed region includes C99 SASS specialization into private CUBIN copies,
module load, function lookup, concurrent scoring launches, device winner
reduction, completion events, unload, and final readback. CUDA compilation,
AST generation, hash verification, resident dataset/settings upload, and
correctness replay are outside timing.

## CTA-owned stream sweep

Configuration: one system/CTA, 256 threads, eight kernels/module, 128 systems
per kernel, 1,024 systems/module, 128 modules, and maximum loaded depth 16.

| Streams | Concurrent CTA supply | Wall time | Unique AST/s | M system configurations/s | Exact winners |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1.2055 s | 217,449 | 3.479 | 1,024/1,024 |
| 2 | 256 | 0.6626 s | 395,611 | 6.330 | 1,024/1,024 |
| 4 | 512 | 0.5325 s | 492,297 | 7.877 | 1,024/1,024 |
| 8 | 1,024 | 0.5308 s | 493,877 | 7.902 | 1,024/1,024 |

Four streams are practically sufficient: eight streams improve throughput by
only 0.32%.

## Warp-owned stream sweep

Configuration: one system/warp, eight systems/CTA, 256 threads, 32
kernels/module, 128 systems per kernel, 4,096 systems/module, 32 modules, and
maximum loaded depth 32. Lane ID selects the setting; warp ID selects the
system. Winner reduction uses warp shuffles rather than CTA shared-memory
reduction.

| Streams | Concurrent CTA supply | Wall time | Unique AST/s | M system configurations/s | Exact winners |
|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 0.2925 s | 896,153 | 14.338 | 4,096/4,096 |
| 16 | 256 | 0.2323 s | 1,128,262 | 18.052 | 4,096/4,096 |
| 24 | 384 | 0.2295 s | 1,142,399 | 18.278 | 4,096/4,096 |
| 32 | 512 | 0.2223 s | **1,179,377** | **18.870** | 4,096/4,096 |

Thirty-two streams is the best measured point. The 24-to-32-stream gain is
3.24%, so 24 is usable but not yet a demonstrated plateau.

## Correctness

Warp winners were checked against a separately generated CTA-owned full-MSE
kernel, not against another warp-owned execution. All 4,096 per-system winners
at every stream point exactly matched that independent GPU reference, with zero
score error. The earlier same-topology smoke replay was therefore superseded.

Five of the 4,096 checked random systems differed in invalid/overflow
classification between the FP32 GPU result and the higher-precision CPU
reference. The exact GPU cross-topology match shows these are the already-known
pathological overflow-reference cases rather than a warp dispatch race.

PTXAS uses a five-instruction compact branch dispatch when different warps in a
CTA select different systems; the CTA-uniform form uses four. The inspector now
accepts these two observed forms, while the specializer records and restores
every dispatch instruction verbatim. The specialized warp workload used at
most 80 registers, compared with 81 for the CTA-owned workload.

## Setup cost and limitations

The speedup requires enough different systems resident at once. The warp
template therefore has 32 kernels and is much larger:

| Ownership | CUDA source | CUBIN | Cold compile | Modules in timed batch | Total timed CUBIN bytes |
|---|---:|---:|---:|---:|---:|
| CTA-owned | 8.33 MB | 3.34 MB | approximately 56.8 s | 128 | 427.7 MB |
| Warp-owned | 33.31 MB | 13.25 MB | 131.0 s | 32 | 423.9 MB |

Thus total module bytes through the timed pipeline differ by less than 1%, but
the warp template has materially higher one-time compilation and inspection
latency. A future loader could pre-load several smaller modules and then launch
their kernels together, potentially retaining warp occupancy without one
33 MB CUDA template.

Current limitations are intentional and material:

- at most 32 settings;
- materialized settings only in this benchmark;
- not selected by the C99 GP campaign path yet;
- dense, aligned fed-batch trajectory layout; and
- 32 kernels/module are used to supply enough independent systems on the 5090.

## Superseded attempts

The first smoke compile exposed PTXAS's five-instruction warp dispatch and
stopped before timing because the inspector only accepted four. After that was
fixed, a smoke replay used the same warp topology as its full-MSE reference;
that result was not accepted as sufficient validation. The first large A/B
attempt was also stopped because module-dependent random chunking produced
statistically similar but non-identical AST streams. Run 2 resolves all three
issues with five-instruction regression coverage, independent CTA correctness,
and matching workload hashes.

## Raw reports

Final copied reports are under
`generated/warp_per_system_comparison_2026-08-31/`. Originals remain on Rohini
under
`~/secant-system-id/generated/warp_per_system_comparison_2026-08-31_run2/`.
The excluded attempts remain under the corresponding `run1` directory with
their abort reasons in `progress.log`.
