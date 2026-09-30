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

# CUDA Eager Module Streaming

## Recommendation

For a backlog of independently compiled CUDA modules, use
`CUDA_MODULE_LOADING=EAGER` and keep one executing module plus one staging
module:

1. Load the first module.
2. Launch every kernel from the current module on the prepared streams.
3. Call `cuModuleLoadData` for the next module while the current module is
   executing.
4. Wait for the current module's joined completion event.
5. Unload the current module and launch the staged module.
6. Repeat.

`cuModuleLoadData` is still a synchronous host call. The important measured
behavior is that already-enqueued GPU work continues while the host call is
blocked. Eager mode prepares the module functions during that one call, so a
separate `cuFuncLoad` pass is unnecessary.

This is the default NVIDIA transition in `secant_pipeline_bench` for the CUDA,
PTX, and CUBIN backends. The benchmark sets `CUDA_MODULE_LOADING=EAGER` before
CUDA initialization and verifies the resulting driver mode.

## Measurement

The strategy probe prepared all CUBIN buffers before the timed pipeline and
used independently specialized modules.

- GPU: NVIDIA GeForce RTX 5090, SM 12.0
- Driver: 595.71.05
- CUDA toolkit: 13.1
- Modules: 128
- Kernels/module: 64
- ASTs/kernel: 128
- Total ASTs: 1,048,576
- Rows: 131,072
- Streams: 8
- Kernel: static-column SSE, MUFU corpus
- Samples: three fresh processes per strategy

Median wall time:

| Strategy | One pass/module | Eight passes/module |
|---|---:|---:|
| Lazy serial | 1,830.0 ms | 6,843.8 ms |
| Lazy load overlap | 1,195.6 ms | 6,207.5 ms |
| Lazy preregister all | 1,343.8 ms | 6,363.4 ms |
| Lazy split-work overlap | 946.3 ms | 5,957.1 ms |
| **Eager streaming** | **878.5 ms** | **5,894.2 ms** |
| GPU event time | 719.0 ms | 5,734.4 ms |

Eager streaming was 52.0% faster than lazy serial for one pass and 13.9%
faster for eight passes. Its residual wall time above GPU execution was about
160 ms across all 128 modules. This is approximately 1.25 ms/module or
0.15 microseconds/AST.

The lazy split-work strategy remains useful if eager loading cannot be
enabled. It divides useful work into two batches so `cuModuleLoadData` and
`cuFuncLoad` can each block against a different GPU batch. It was within 1.1%
of eager streaming for the longer workload, but 7.7% slower for one pass.

## Interpretation

The loader call durations overlap GPU execution and therefore are not
additive phase times. Overall pipeline wall time and summed GPU-event time are
the valid comparison. Preregistering all modules is not preferred: it was
slower and retained all 128 modules simultaneously. Eager streaming peaked at
two loaded modules.

These results verify the behavior on the configuration above. They are an
empirical driver/toolkit result, not a CUDA ABI guarantee for every future
driver, architecture, or operating system. The benchmark deliberately checks
the active loading mode and fails rather than silently reverting to a
different transition.

## Improvement Over The Overnight Benchmark

The July 27 RTX 5090 overnight campaign used the previous serialized
transition with 240 modules, 64 kernels/module, 128 ASTs/kernel, 1,966,080
ASTs, 24 compiler workers, and eight streams. The eager scheduler was rerun
with the same dimensions, ALU corpus, row counts, and per-case seeds.

Compile-inclusive row-evaluation throughput:

| Rows | Materialize before | Materialize eager | Speedup | SSE before | SSE eager | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 2.371e9 | 4.318e9 | 1.82x | 8.516e8 | 2.440e9 | 2.87x |
| 4,096 | 8.875e9 | 1.725e10 | 1.94x | 3.277e9 | 9.858e9 | 3.01x |
| 16,384 | 3.330e10 | 7.002e10 | 2.10x | 1.227e10 | 3.942e10 | 3.21x |
| 65,536 | 1.075e11 | 2.512e11 | 2.34x | 4.944e10 | 1.581e11 | 3.20x |
| 131,072 | 1.728e11 | 3.132e11 | 1.81x | 9.777e10 | 3.156e11 | 3.23x |
| 262,144 | 2.172e11 | 3.024e11 | 1.39x | 1.932e11 | 6.246e11 | 3.23x |

The overnight report's peak CUBIN pipeline rates therefore change from
2.172e11 to 3.132e11 row-evals/s for materialize and from 1.932e11 to
6.246e11 row-evals/s for SSE. These are 1.44x and 3.23x improvements,
respectively. Runtime-only kernel throughput is effectively unchanged; the
gain comes from eliminating exposed module lifecycle time.

At the 262,144-row SSE endpoint:

- GPU-event runtime remained effectively unchanged:
  0.3572 seconds before and 0.3586 seconds with eager streaming.
- Pipeline wall time fell from 2.6680 seconds to 0.8251 seconds.
- Explicit completion-event waiting fell from 0.3343 seconds to
  0.0016 seconds.
- Module unloading fell from 0.6542 seconds to 0.0366 seconds.
- Pipeline throughput increased from 1.932e11 to 6.246e11 row-evals/s.

Holding the overnight external-runtime measurements fixed, the reported
262,144-row full-pipeline speedup against the fastest external baseline
changes approximately as follows:

- CUBIN materialize: 25.49x becomes 35.5x.
- CUBIN SSE: 5.17x becomes 16.7x.

For SSE, that fastest baseline was the 24-thread native AVX2 runner at
3.737e10 row-evals/s, not EvoGP. Against EvoGP's measured
1.440e10 row-evals/s, the CUBIN SSE full-pipeline advantage changes from
13.4x to 43.4x. The CUBIN runtime-only rate is about 100x EvoGP; eager
streaming closes much of the gap between that kernel-only rate and the full
compile/load/run pipeline.

The CUDA and PTX compile-inclusive overnight points are less affected because
native compilation dominates their wall time. For example, the 262,144-row
PTX O1 SSE compile window was 217.2 seconds and the CUDA O1 SSE compile window
was 1,165.6 seconds. Eager streaming improves their module scheduling, but it
cannot materially move those compile-bound pipeline rates.

## Loading Strategy Probe

The original pipeline treated each module transition as a serialized
lifecycle:

1. Wait for every stream using the current module.
2. Unload the current module.
3. Load and enumerate the next module.
4. Force every function resident with `cuFuncLoad`.
5. Launch the next module.

That left the GPU idle during most module-management work. Across 128 modules,
one pass took 1,830.0 ms even though GPU-event execution totaled only
718.3 ms. The non-GPU wall-time difference was about 1,112 ms, or
1.06 microseconds/AST.

Loading the next module while the current module ran improved wall time to
1,195.6 ms, but lazy loading still left the separate `cuFuncLoad` phase
exposed. Splitting each module's useful work into two batches allowed the load
and function-residency phases to overlap separately and reached 946.3 ms, but
required extra launches, events, and scheduler complexity. Preregistering all
128 modules was slower at 1,343.8 ms and retained every module simultaneously.

Eager streaming combines module loading and function residency into one host
call that overlaps the current module. It reached 878.5 ms while retaining at
most two modules. Compared with the original serialized transition, this is:

- 52.0% lower total wall time, or 2.08x throughput, for one pass/module.
- 85.7% less non-GPU lifecycle overhead: about 1,112 ms became 160 ms.
- About 0.15 microseconds of residual pipeline overhead per AST instead of
  1.06 microseconds/AST.
- 81.8% GPU-time utilization instead of 39.3% for the short workload.

For eight passes/module, total wall time improved from 6,843.8 ms to
5,894.2 ms, a 13.9% reduction. The absolute lifecycle saving remained about
950 ms, but it represented less of the total because GPU execution itself took
5,734.4 ms. Eager streaming reached 97.3% GPU-time utilization in that case.

The production benchmark now applies this transition uniformly to CUDA, PTX,
and CUBIN artifacts after compilation. CPU workers continue publishing
completed artifacts to the ready queue, while the GPU consumer holds one
active module and one eager staging module. The kernel, AST representation,
and compilation backend do not need different module schedulers.
