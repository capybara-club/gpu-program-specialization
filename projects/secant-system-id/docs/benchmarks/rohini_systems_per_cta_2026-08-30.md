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

# Rohini packed SSE systems-per-CTA sweep

> **Superseded for modest setting counts.** The 128/512-setting pipeline rows
> below launched one packed kernel at a time. Increasing systems/CTA therefore
> reduced the available CTA count without replacing it with concurrent kernel
> streams. The occupancy-matched rerun is recorded in
> `rohini_multistream_occupancy_2026-08-31.md`. The high-setting resident and
> 8,192/16,384-setting pipeline measurements remain useful because each launch
> already exposed enough CTAs; they still show a real serial-packing penalty.

Date: 2026-08-30 EDT
GPU: NVIDIA GeForce RTX 5090 (`sm_120`)
Purpose: compare 1, 2, 4, and 8 sequential system genomes per CTA while
evaluating byte-distinct postorder AST programs drawn from a structurally
diverse random family, with enough setting tiles to saturate the GPU.

## Result

For the current dense aligned fed-batch SSE shape, **one system per CTA is the
best default**. Two systems per CTA nearly ties it in the warm resident kernel
at 16,384 settings, but remains slower end to end. Four and eight systems per
CTA are consistently slower.

The recommendation is based on the full per-AST module pipeline below, not on
resident-kernel launch speed. The resident measurements are retained only to
separate kernel execution behavior from loader-pipeline behavior.

### End-to-end unique-module path at modest setting counts

Each point below passes 512 unique specialized modules through the production
C99 eager path. Timing includes SASS specialization into a private CUBIN copy,
module load, function lookup, launch, device-side winner reduction, completion
events, unload, and final result readback. Each CTA topology therefore processes
65,536 unique systems and 131,072 unique candidate ASTs per point. The one-time
CUDA template compilation, deterministic benchmark-batch construction, CUBIN
hash audit, and resident reference/settings upload are outside the timed region;
those are problem setup rather than per-AST hot-path work.

| Settings/system | Systems/CTA | Wall time | Unique systems/s | Unique AST/s | M configurations/s | System-throughput loss vs 1 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 0.6011 s | 109,030 | 218,061 | 13.956 | 0.00% |
| 128 | 2 | 1.1001 s | 59,575 | 119,149 | 7.626 | 45.36% |
| 128 | 4 | 2.0511 s | 31,952 | 63,905 | 4.090 | 70.69% |
| 128 | 8 | 3.9239 s | 16,702 | 33,404 | 2.138 | 84.68% |
| 512 | 1 | 0.6348 s | 103,233 | 206,466 | 52.855 | 0.00% |
| 512 | 2 | 1.1153 s | 58,762 | 117,523 | 30.086 | 43.08% |
| 512 | 4 | 2.0780 s | 31,539 | 63,077 | 16.148 | 69.45% |
| 512 | 8 | 3.9725 s | 16,497 | 32,995 | 8.447 | 84.02% |
| 2,048 | 1 | 1.8412 s | 35,595 | 71,190 | 72.898 | 0.00% |
| 2,048 | 2 | 2.1059 s | 31,121 | 62,241 | 63.735 | 12.57% |
| 2,048 | 4 | 2.1879 s | 29,954 | 59,907 | 61.345 | 15.85% |
| 2,048 | 8 | 3.9796 s | 16,468 | 32,936 | 33.726 | 53.74% |

At modest settings, serially assigning multiple systems to each CTA reduces
not only configuration throughput but the rate at which the complete module
pipeline can consume new ASTs. The penalty narrows at 2,048 settings because
execution becomes a larger fraction of the ticket lifetime, but one system per
CTA still wins.

### 1,024-module full-pipeline confirmation

Because the 512-module low-setting points finish quickly, the representative
128, 512, and 2,048-setting measurements were repeated with 1,024 distinct
modules per topology. Each row processes 131,072 unique systems and 262,144
unique candidate ASTs. Every topology produced 1,024/1,024 unique CUBIN hashes;
all 128 replayed winners at every setting count exactly matched the full-MSE
GPU reference, with zero invalid-status mismatches.

| Settings/system | Systems/CTA | Wall time | Modules/s | Unique systems/s | Unique AST/s | M configurations/s | System-throughput loss vs 1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 1.2001 s | 853.29 | 109,222 | 218,443 | 13.980 | 0.00% |
| 128 | 2 | 2.2004 s | 465.36 | 59,566 | 119,133 | 7.624 | 45.46% |
| 128 | 4 | 4.1081 s | 249.27 | 31,906 | 63,812 | 4.084 | 70.79% |
| 128 | 8 | 7.8676 s | 130.15 | 16,660 | 33,319 | 2.132 | 84.75% |
| 512 | 1 | 1.2712 s | 805.52 | 103,106 | 206,213 | 52.790 | 0.00% |
| 512 | 2 | 2.2321 s | 458.75 | 58,720 | 117,440 | 30.065 | 43.05% |
| 512 | 4 | 4.1657 s | 245.82 | 31,464 | 62,929 | 16.110 | 69.48% |
| 512 | 8 | 7.9672 s | 128.53 | 16,451 | 32,903 | 8.423 | 84.04% |
| 2,048 | 1 | 3.6915 s | 277.40 | 35,507 | 71,014 | 72.718 | 0.00% |
| 2,048 | 2 | 4.2253 s | 242.35 | 31,021 | 62,042 | 63.531 | 12.63% |
| 2,048 | 4 | 4.3878 s | 233.38 | 29,872 | 59,744 | 61.178 | 15.87% |
| 2,048 | 8 | 7.9792 s | 128.33 | 16,427 | 32,853 | 33.642 | 53.74% |

The one-system results differ from the corresponding 512-module measurements
by only +0.18%, -0.12%, and -0.25% at 128, 512, and 2,048 settings. The longer
run therefore confirms that the low-setting conclusion is not a short-sample
artifact.

### Warm resident full-MSE kernel (diagnostic only)

This boundary loads one specialized module, warms it, and times 50 launches.
It excludes CUDA compilation, specialization, module loading, transfers, and
correctness replay. Each launch contains 128 distinct systems and two distinct
AST programs per system.

| Systems/CTA | CTAs at 8,192 settings | 8,192 settings M config/s | Loss vs 1 | CTAs at 16,384 settings | 16,384 settings M config/s | Loss vs 1 | Registers |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,096 | 124.164 | 0.00% | 8,192 | 128.062 | 0.00% | 73 |
| 2 | 2,048 | 118.056 | 4.92% | 4,096 | 127.615 | 0.35% | 80 |
| 4 | 1,024 | 102.889 | 17.13% | 2,048 | 118.063 | 7.81% | 80 |
| 8 | 512 | 77.360 | 37.70% | 1,024 | 99.567 | 22.25% | 80 |

All full-MSE output arrays were bit-identical to the one-system-per-CTA
baseline: zero differing FP32 scores for both setting counts and every tested
topology.

### End-to-end unique-module winner path

This is the search-relevant boundary. For every point it generated and
specialized 512 byte-distinct modules with 128 distinct systems per module:

- 65,536 distinct system genomes;
- 131,072 distinct candidate ASTs;
- 512/512 unique specialized CUBIN hashes;
- eager module depth 2;
- 256 threads per CTA;
- compact per-genome winning-score and setting reduction; and
- 8,192 and 16,384 settings per system.

The deterministic generator rejected both duplicate system genomes and any
duplicate individual AST within the same RHS site. It needed 44,166 retries to
produce 65,536 distinct systems, 65,536 distinct first-site ASTs, and 65,536
distinct second-site ASTs. The same resulting AST batches and settings were
used for all four CTA topologies.

| Systems/CTA | 8,192 settings M config/s | Unique AST/s | Loss vs 1 | 16,384 settings M config/s | Unique AST/s | Loss vs 1 | Registers |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 87.157 | 21,279 | 0.00% | 89.150 | 10,883 | 0.00% | 81 |
| 2 | 79.337 | 19,369 | 8.97% | 84.488 | 10,314 | 5.23% | 96 |
| 4 | 70.187 | 17,136 | 19.47% | 79.035 | 9,648 | 11.35% | 94 |
| 8 | 66.583 | 16,256 | 23.61% | 69.552 | 8,490 | 21.98% | 94 |

Every topology and setting count selected exactly the same GPU winner as its
full-MSE GPU replay for all 128 replayed systems. There were zero invalid-status
mismatches. Some random expressions produced extremely large finite SSEs, so
the report's maximum absolute CPU/GPU error is not a useful accuracy metric;
winner identity and invalid classification are the relevant checks here.

## Interpretation

The reference trajectory is small, so reusing its shared-memory copy across
several systems does not recover the cost of retaining a CTA for serial system
work. One-system ownership exposes the most independent CTAs and also lets
PTXAS use fewer registers in both measured output modes. At 16,384 settings,
two-system ownership comes within 0.35% in the resident full-MSE kernel, showing
that sufficiently many setting tiles can mostly hide its reduction in CTA
count, but it does not surpass one system.

This conclusion applies to the current dense aligned four-state, three-
trajectory, 12-observation, 16-RK4-steps-per-observation fed-batch shape. A much
larger reference dataset or a topology with expensive CTA initialization could
move the crossover and should be measured separately.

## Provenance and deviations

The first tmux harness attempt was aborted before yielding a usable point
because shell expansion collapsed all result paths to `cta_`. A subsequent
complete sweep guaranteed unique system pairs but not globally unique
individual RHS ASTs; it found 101,900 unique site-tagged ASTs in 131,072 AST
slots and was superseded. Those reports are retained under
`cta_diverse_sweep_system_unique_only` and excluded from every table. The final
sweep records each CTA topology separately and enforces uniqueness at both the
system and individual-AST levels.

Rohini began at Git commit `a6cbe65`, while the local checkout was `09d8f3f`.
The latter adds generic recovery-problem and benchmark adapters. An audit found
that every CUDA generation, SASS specialization, runtime, AST, shape, and C99
specializer source used by this benchmark has the same SHA-256 on both
checkouts. The fed-batch reference-data implementation was refactored between
the commits, but both paths produced the same 156 FP32 values with SHA-256
`2603d0de73db6761516fa8411788d47a35b8ea69d2b168e96655bb775bd0d866`.
No rerun is required for the current SSE execution path.

The benchmark-only CLI addition that generates and verifies structurally
distinct systems was uncommitted during the run. The existing C99 specializer
also performs one temporary `malloc`/`free` for packed body offsets per kernel
specialization. That allocator path is common to every end-to-end point and is
included in the timing; it does not affect the resident-kernel table.

The controlled sweep uses a materialized settings/bindings table uploaded once
before timing. Current GP campaigns instead use the hashed-incumbent kernel
mode, which derives each setting in the kernel from an incumbent and setting
index. Thus, this is a full measurement of the unique-AST module lifecycle and
packed-system topology, but not an exact measurement of a complete GP
generation. The resident upload is excluded in both intended regimes; the
setting-mode distinction can change kernel throughput and must remain explicit
when these numbers are compared with campaign measurements.

## Raw reports

The copied reports are under
`generated/cta_diverse_sweep_2026-08-30/`:

- `resident_full_mse.json`
- `cta_1/unique_pipeline_sweep.json`
- `cta_2/unique_pipeline_sweep.json`
- `cta_4/unique_pipeline_sweep.json`
- `cta_8/unique_pipeline_sweep.json`
- `progress.log`

The original reports remain on Rohini under
`~/secant-system-id/generated/cta_diverse_sweep/`.

The modest-setting reports are under
`generated/cta_diverse_low_settings_2026-08-30/`. The 1,024-module confirmation
is under `generated/cta_diverse_pipeline_confirm_2026-08-30/`, with originals
on Rohini under `~/secant-system-id/generated/cta_diverse_pipeline_confirm/`.
