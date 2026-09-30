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

# Rohini occupancy-matched packed SSE pipeline

Date: 2026-08-31 EDT
GPU: NVIDIA GeForce RTX 5090 (`sm_120`)
Purpose: repeat the modest-setting systems-per-CTA comparison without reducing
the total independent CTA supply as more systems are evaluated serially by each
CTA.

## Result

Launching multiple packed kernels concurrently fixes most of the apparent
low-setting penalty reported by the original one-stream sweep. At 512 settings,
the best measured full-pipeline rates are:

| Systems/CTA | Best streams | Aggregate scoring CTAs/module | Unique AST/s | M system configurations/s | Loss vs 1 system/CTA |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 1,024 | 280,234 | 71.740 | 0.00% |
| 2 | 4 | 512 | 247,067 | 63.249 | 11.84% |
| 4 | 8 | 512 | 222,031 | 56.840 | 20.77% |
| 8 | 16 | 512 | 217,269 | 55.621 | 22.47% |

One system per CTA remains the fastest measured shape, but the earlier 84%
penalty at eight systems/CTA was primarily an under-occupancy artifact. The
remaining penalty is the actual cost of serially retaining a CTA across several
systems, including the higher register allocation (81 registers for one system
versus 96 for the packed alternatives).

## Matched eight-kernel sweep

Every timed point evaluates:

- 128 byte-distinct specialized modules;
- eight packed scoring kernels per module;
- 128 system genomes per kernel and 1,024 systems per module;
- two independent candidate RHS ASTs per system;
- 512 settings per system;
- 131,072 systems, 262,144 candidate ASTs, and 67,108,864 system/settings
  configurations in total; and
- 256 threads per CTA with 16 modules permitted in the eager resident queue.

The benchmark changes only systems/CTA and execution-stream count. Compilation,
deterministic AST construction, hash auditing, resident dataset/settings upload,
and correctness replay are excluded. The timed region includes C99 SASS
specialization into private CUBIN copies, module loading, function lookup,
scoring launches, device winner reduction, event completion, unloading, and
result readback.

| Systems/CTA | Streams | Aggregate scoring CTAs/module | Wall time | Unique AST/s | M system configurations/s | Exact GPU winners |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 256 | 1.2912 s | 203,020 | 51.973 | 1,024/1,024 |
| 1 | 2 | 512 | 0.9727 s | 269,515 | 68.996 | 1,024/1,024 |
| 1 | 4 | 1,024 | 0.9354 s | 280,234 | 71.740 | 1,024/1,024 |
| 1 | 8 | 2,048 | 0.9391 s | 279,131 | 71.458 | 1,024/1,024 |
| 2 | 1 | 128 | 2.2107 s | 118,581 | 30.357 | 1,024/1,024 |
| 2 | 2 | 256 | 1.2693 s | 206,526 | 52.871 | 1,024/1,024 |
| 2 | 4 | 512 | 1.0610 s | 247,067 | 63.249 | 1,024/1,024 |
| 2 | 8 | 1,024 | 1.0672 s | 245,626 | 62.880 | 1,024/1,024 |
| 4 | 1 | 64 | 4.0945 s | 64,024 | 16.390 | 1,024/1,024 |
| 4 | 2 | 128 | 2.1157 s | 123,901 | 31.719 | 1,024/1,024 |
| 4 | 4 | 256 | 1.2348 s | 212,295 | 54.347 | 1,024/1,024 |
| 4 | 8 | 512 | 1.1807 s | 222,031 | 56.840 | 1,024/1,024 |
| 8 | 1 | 32 | 7.8121 s | 33,556 | 8.590 | 1,024/1,024 |
| 8 | 2 | 64 | 3.9776 s | 65,906 | 16.872 | 1,024/1,024 |
| 8 | 4 | 128 | 2.0594 s | 127,291 | 32.586 | 1,024/1,024 |
| 8 | 8 | 256 | 1.2185 s | 215,141 | 55.076 | 1,024/1,024 |

The one-stream rows closely reproduce the original pipeline sweep. This makes
the stream-scaling change, rather than a different AST workload, the cause of
the corrected result.

## Eight-system saturation extension

This extension keeps the same total 262,144 candidate ASTs and 67,108,864
configurations, halves submissions from 128 to 64, and doubles kernels per
module from 8 to 16. It tests 256, 384, and 512 aggregate scoring CTAs at 8, 12,
and 16 streams.

| Systems/CTA | Kernels/module | Streams | Aggregate scoring CTAs/module | Wall time | Unique AST/s | M system configurations/s | Exact GPU winners |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 8 | 256 | 1.2580 s | 208,375 | 53.344 | 2,048/2,048 |
| 8 | 16 | 12 | 384 | 1.2147 s | 215,810 | 55.247 | 2,048/2,048 |
| 8 | 16 | 16 | 512 | 1.2065 s | 217,269 | 55.621 | 2,048/2,048 |

The final 256-to-512-CTA gain is only 4.27%, and the 384-to-512-CTA gain is
0.68%. The topology has reached its practical plateau. Its best rate is 22.47%
below one system/CTA, not the 84.04% loss from the original under-occupied
one-stream comparison.

## Runtime change required by the benchmark

The original C99 eager loader owned one CUDA context and one stream. Merely
assigning consecutive modules to different streams did not increase throughput:
`cuModuleLoadData` calls interleaved between those module launches prevented the
intended execution overlap. A one-kernel smoke test stayed at approximately
33.5k AST/s for both one and eight streams.

The effective topology loads one module containing several packed kernels,
launches its kernels round-robin over a configurable nonblocking stream pool,
records dependency events for every used scoring stream, and joins them before
the module's winner reduction and completion event. An eight-kernel G=8 smoke
test improved from 33,906 AST/s on one stream to 214,062 AST/s on eight streams
(6.31x), with identical GPU winners.

This means production campaigns only benefit from the new stream option when a
module contains multiple packed scoring kernels. A one-kernel module still has
only one scoring launch to distribute.

## Correctness and deviations

Every point selected exactly the same GPU winner as a separate full-MSE GPU
replay for all 1,024 checked systems, and the maximum GPU score error was zero.
Three expressions overflowed differently in the FP32 GPU and higher-precision
CPU references, causing three invalid-status classification mismatches. The
same three mismatches occurred at every stream count, so they are not a
multi-stream race; they remain a reference-validation issue for randomly
generated pathological expressions.

The controlled sweep uses materialized settings/bindings uploaded before
timing. Production GP campaigns currently use hashed-incumbent settings derived
inside the kernel. These results characterize the module lifecycle and packed
SSE topology, not an exact complete GP generation.

The eight-kernel CUDA templates are approximately 8.33 MB of source and 3.34 MB
of CUBIN. Initial CUDA compilation took 42.7--57.0 seconds per distinct template.
The 16-kernel extension is 16.66 MB of source and 6.67 MB of CUBIN and took 85.0
seconds to compile. Compilation is excluded from the hot-path rates. This is
valid for repeated SASS specialization of a compiled template, but it is
material setup latency for a new problem shape.

`ssid_specialize_kernel` still performs one temporary `malloc`/`free` per packed
kernel specialization. That known allocator cost is included uniformly in all
pipeline measurements.

## Raw reports

The copied reports are under
`generated/cta_multistream_fair_2026-08-31/` and
`generated/cta_multistream_fair_g8_16kernel_2026-08-31/`. Originals remain on
Rohini under `~/secant-system-id/generated/cta_multistream_fair/` and
`~/secant-system-id/generated/cta_multistream_fair_g8_16kernel/`.
