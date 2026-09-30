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

# Request timing and sizing

The native runtime reports timings without changing the scoring algorithm,
configuration addresses, RNG values or tile execution dependencies. GPU stage
overlap remains deferred.

`execution.profile_timing: true` enables CUDA timing events around each scoring
launch. The pipeline owns these reusable events; runs do not allocate them.
Profiling adds launch overhead, so benchmark it separately. Without profiling,
GPU interval fields are `null`, rather than estimates from host waits.

## Reading a report

- `critical_path`: serial wall phases for parsing/reservation, followed by
  execution/finalization. Generation and GPU workers overlap within execution.
- `timing`: existing stage totals. Worker totals overlap and do not sum to elapsed
  time. NVRTC compilation is separated from total template/cache preparation.
- `execution.devices[].pipeline_breakdown`: module counts, queue wait, cubin reset,
  specialization, eager module load, function lookup, launch submission,
  completion wait and module unload. These are per-module host sums for successful
  scoring calls, not a disjoint wall-time decomposition. Failed capacity attempts
  remain visible in cache growth/retry counts and host worker elapsed time.
- Optional `gpu_sum_seconds` sums individual kernel intervals, counting concurrent
  launches more than once. `gpu_active_seconds` unions those intervals. `gpu_span_seconds`
  is the first start to last end within each scoring call; `gpu_gap_seconds` is
  span minus active. Per-device reports sum those disjoint scoring calls.
  These exclude preparation before the first launch, inter-tile gaps, reduction,
  RNG/prelude and transfers. They are **not SM occupancy or total GPU utilization**.
- Actual maximum specialized register count and dynamic shared-memory bytes are
  reported independently of module and patch capacity. Packing more ASTs into a
  module does not reduce per-thread register pressure. Existing inspection and
  specialization failures still apply; no spilling fallback was added.
- `transfers`: successfully transferred trajectory/grid/descriptor and returned
  winner/count/coefficient bytes. Trajectory-upload and result-copy host durations
  are provided; the result-copy duration is contained in reduction/gather time.
  RNG banks generated on the GPU are not counted as host transfers. CUB reduction
  reports GPU time and continuation passes separately.
- `busy_seconds` is retained for compatibility and explicitly means host tile
  execution including waits. `worker_wall_seconds` and `page_wait_seconds` provide
  worker context; neither measures GPU occupancy.

`total_seconds` ends before client transport and final report serialization.
The service/client elapsed time can therefore exceed it.

## Automatic choices

Explicit `batch_variants`, `module_systems`, and configuration caps retain their
meaning. When omitted:

1. AST pages start with a ceiling of 1,024 variants, bounded by that family's
   variant allocation. The producer's allocation-free measurement shrinks each
   family's page to its share of one eighth of the requested host budget, capped
   at 64 MiB for families reserving at least 65,536 variants; smaller populations
   use 16 MiB for short rollouts or 32 MiB for at least 32 RK4 steps. No ASTs are generated to perform this measurement.
   The chosen family `batch_variants` appears in the report. A one-variant page
   still must fit the normal host reservation.
2. Module packing uses powers of two starting at two, up to 32 systems for short screens. When
   each configuration integrates at least 32 steps and the tile supplies at least
   128 configurations per skeleton, the ceiling is 64. Small compatible tiles
   use fewer systems per module while preserving work for the available pipeline
   workers. A two-system minimum keeps NVRTC from eliding the dispatch needed by
   inspection; a tile can still contain just one real candidate. This minimum
   applies only to automatic selection, not explicit requested shapes. Device reports expose the smallest/largest module actually used.
3. Existing configuration sizing adapts from measured tile wall time, while work
   and pool limits remain hard ceilings. The complete bank is preferred before
   splitting it across repeated module loads. Patch growth reuses emitted ASTs
   and caches the expanded compiled shape; it is independent of register limits.

This is a measured RTX 5080 policy, not a universal optimum or a duration promise.
See [validation and measurements](SUMMARY.md) and the [rejected initial policy](DEVIATIONS.md).

## Reproduce

From a built checkout with installed CUDA/CUB:

```sh
CUDA_MODULE_LOADING=EAGER PYTHONPATH=python python3 benchmarks/request_sizing/sweep.py \
  --devices 0,1 --output /tmp/odezza-sizing.json
```

Use `--devices 0` for the single-GPU comparison. Each finite grammar emits 4,096
ASTs; positive slot counts share a 256-row Philox axis with independent streams.
The sweep checks exact retained IDs, coefficients, MSEs, programs and counts
across settings, then independently replays a winner on CPU. It records cold,
warm and profiled runs separately. Short/long fixture labels are not step counts;
the summary uses the actual FP32 RK4 subdivision count from the report.
