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

# Prepared-score CUB comparison

This records the isolated comparison. The subsequent [CUB-only integration and
live service validation](integration/README.md) is now complete.

An isolated trial of NVIDIA CUB against Odezza's current public C reducer.
It does not change the core, runtime, service, grammar or retention contract.

## Result on rack1, 2026-09-12

Both RTX 5080s show a large improvement for **top-16 over many small groups**.
The current reducer measured faster for top-1/top-4 on the common layout and
for a single very large global group. The user's subsequent direction is to use
CUB alone in production. That requires selecting suitable CUB primitives for
these cases; this trial does not establish that full sorting wins everywhere.

For **1,024 groups × 2,048 scores = 2,097,152 scores**, GPU 0 median over 21 calls:

| Method | Top-1 GPU µs | Top-4 GPU µs | Top-16 GPU µs | Top-16 complete operation µs |
|---|---:|---:|---:|---:|
| Current Odezza | 14.72 | 43.42 | 811.49 | 866.60 |
| CUB segmented stable sort | 159.74 | 159.65 | 159.97 | 215.49 |
| CUB segmented radix sort | 307.81 | 307.74 | 310.02 | 366.98 |
| CUB block radix sort | 95.20 | 96.99 | **98.85** | **153.61** |

The block wrapper improves the top-16 GPU operation **8.21×**, or **5.64×** including
coefficient gathering and host downloads. These are comparisons at the same k,
with the same score/index/count results. They are not full-request or solve-time
speedups. GPU 1 independently measured 794.94 → 96.80 µs for the same top-16 case.

Other top-16 GPU results from GPU 0:

| Workload | Current µs | Segmented stable µs | Block radix µs |
|---|---:|---:|---:|
| 2,048 groups × 2,048, strided by 16, mixed invalid values | 1,613.15 | 335.52 | 197.22 |
| 1,024 × 2,048, all scores equal | 695.74 | 152.22 | 91.33 |
| 1,024 × 2,048, descending scores | 1,037.50 | 158.40 | 91.14 |
| 1,024 × 192, mixed invalid values | 599.74 | 140.96 | 33.57 |
| 8,192 × 2,048 (64 MiB input) | 6,262.40 | 1,356.45 | 706.43 |
| One global group of 2,097,152 | **327.33** | 15,517.41 | Unsupported by this block wrapper |

Full results: [GPU 0 matrix](results/topk-full.jsonl),
[GPU 1 cross-check](results/topk-device1.jsonl),
[build resources](results/topk-build.log).

## What is measured

All AST generation and trajectory integration are absent: the inputs are synthetic
scores already on the device. Compilation, module loading, allocation, input
uploads and CPU verification happen outside the timed samples. The benchmark
rotates algorithm order, warms each method three times and reports medians.
Repeated inputs may benefit from cache residency; these are not cold-DRAM bandwidth
measurements. The 64 MiB case adds a larger-buffer check. GPU clocks are not locked.

Every method produces the same ABI: `[group,k]` MSE/index winners and valid/invalid/
negative counts. Ties prefer the smaller original score index. NaN, infinities,
negative scores and FLT_MAX are invalid; an unfilled rank has FLT_MAX/UINT64_MAX.
Signed-zero input bits are preserved in the reported MSE. Both CUB sorts carry
64-bit original indices as values. No expression deduplication occurs.

- **Current:** calls the unchanged C API and uses its CUDA-event kernel time.
- **Segmented:** includes a GPU packing/normalization/counting pass, CUB sorting,
  and gathering the first k winners. Strided permutation groups are packed into
  contiguous segments; their rearrangement cost is included. Stable ordering
  preserves original-index ties because each segment is packed in ascending
  original-index order. Sorted keys/indices are not downloaded in full.
- **Block:** one block loads a group into CUB's blocked layout, computes counts,
  performs stable BlockRadixSort over score/index pairs, then emits k winners.
  Normalization, counts and winner gathering are included in that kernel.
- **Complete operation:** host duration including reduction, the existing
  Odezza coefficient-gather operation and winner/count/coefficient downloads.
  It excludes host retention, request handling and scoring. Individual median
  stage values need not add to the median total.

The common 2,048-row block uses **80 registers/thread**, **33,792 bytes of dynamic
shared memory** and no spills in this build. It uses no additional global scratch
beyond common outputs. The segmented stable path needs about **72.0 MiB additional
device storage** for the 2,097,152-score fixture, versus 280 KiB of workspace for
the current top-16 reducer. Size queries and all allocations are outside timing;
production would need to account for these buffers in the existing pools.

## Correctness and coverage

The full matrix contains **120 measured algorithm/layout/k cases**, each checked
against independently CPU-ranked winners and counts. Checks run before and after
timed repetitions and verify coefficient gathering from each returned index.
GPU 1 adds 24 checked cases. The sweep covers k=1/4/16, tiny and uneven groups,
global grouping, contiguous and strided layouts, random/equal/descending scores,
signed zeros, nonfinite/negative/sentinel values and all-invalid groups.

Twelve block-only cases are explicitly skipped. The trial instantiates capacities
up to 8,192 rows, but its 8,192-row load needs more shared memory than the 5080
allows; larger rows/global groups also cannot use it. They are still checked with
both segmented sorts and the current reducer. No fallback timing is substituted
for a skipped block result. The block shape has not been tuned for these larger
groups. No speed claim extends to skipped or untested shapes.

## Build and run

Uses already installed CUDA/CUB and a built `libodezza.so`; no extra dependencies:

```sh
make -C benchmarks/score_topk SM=120
LD_LIBRARY_PATH=/usr/local/cuda/lib64 CUDA_MODULE_LOADING=EAGER \
  build/score_topk/benchmark 0 full > results.jsonl
```

`0` selects the GPU. `quick` runs the common contiguous and strided cases with
11 repetitions; otherwise the full matrix uses 21. `ODEZZA_BUILD` and `BUILD`
can point to isolated existing build/output directories.

**Build exception on the measured host:** CUDA 13.1.115 conflicts with glibc 2.43
math declarations. The normal nvcc build failed. The measured build explicitly
used `make -C benchmarks/score_topk GLIBC_COMPAT=1`, disabling GNU feature exposure
and adding two matching declarations from installed pthread.h in
[glibc_compat.h](glibc_compat.h). This adapter is confined to the benchmark;
system and CUB headers are untouched and no software was installed. It does not
alter the GPU sorting algorithm. The full comparisons passed under that build.
See [deviation record](deviations.json). These results do not establish that a
normal nvcc production build works on this host. Production integration should
use a compatible host toolchain or validate the existing NVRTC route before
adopting the device primitive, rather than inheriting this compatibility header.

## Next step

Integrate a CUB-only backend behind the existing reducer ABI, following the user's
direction after this trial. Use block radix sorting for supported group sizes;
validate CUB segmented or hierarchical selection for larger groups and measure
appropriate CUB primitives for small k. Do not silently dispatch to the old custom
reducer. Preserve exact ties, counts, index reconstruction, event ownership and
pool budgeting. Choose local k from the actual downstream retention requirements.
Rerun complete JSON-request comparisons before claiming a service-level speedup.
The service's row-versus-distinct retention decision remains explicit API work.

## Family reduction boundary

This trial ranks score groups defined by the launch layout, such as one candidate
system/toggle permutation across its coefficient bank. It does not receive family
identities or directly return a final top-k for a grammar family. The live runtime
still uses the existing reducer and merges retained rows into C-side global,
per-family and tag rankings, routing them by explicit candidate family indices.

For exact raw top-k per family, each contributing subgroup must retain up to k
rows before merging (or use an equivalent exact hierarchical algorithm). Keeping
only one row per AST can lose family winners when several best configurations
belong to that AST. Apply the same requirement across bank tiles and GPUs, using
one consistent tie order. Family/search semantics stay outside the kernel core;
numerical segments and original indices suffice for the GPU operation.

The existing benchmark verifies exact group winners, counts and coefficient
gathers. It does **not** test an end-to-end CUB-backed family report. Integration
needs separate tests for interleaved families, a dominant subgroup, uneven banks,
ties/invalids and multiple tiles/devices, plus explicit raw-row retention semantics.
