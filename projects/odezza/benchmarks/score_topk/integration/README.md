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

# CUB integration — 2026-09-12

CUB is the only GPU score-selection backend in the current core and rack1
service. It compiles through NVRTC using the installed CUDA/CCCL headers, without
nvcc, a host C++ adapter or the isolated benchmark's glibc compatibility header.
The C99 core still owns only numerical layouts, kernels, indices and execution;
family identity and retention policy stay in the C runtime.

## Behavior

- `odezza_score_reducer_create(sm,k,...)` accepts k=1..256. The additive
  `odezza_score_reducer_set_k` changes k between completed operations without
  allocation, compilation or module loading. Existing descriptor layouts remain
  unchanged. Query buffer requirements for the selected k before running.
- Top-1 uses CUB BlockReduce. Top-2..4 uses repeated CUB argmin over resident tile
  values. Larger k uses stable CUB BlockRadixSort. Larger groups use a hierarchy
  of bounded 2,048-element tiles, with caller-owned ping-pong scratch. A one-tile
  group writes final outputs directly, avoiding a redundant merge launch.
- Raw scores remain unchanged. Original indices, deterministic ties, exact
  signed-zero bits, invalid/negative counts, and explicit/Philox coefficient
  gathering are verified. No custom list-insertion reducer remains.
- The runtime sizes local k from requested output. An explicit `evaluation_row`
  unit preserves separate evaluation addresses even when expressions/constants
  repeat. Existing default/explicit distinct units keep their contracts.
- Raw-only family/global ranking without requested tag rankings selects top-k
  over a complete family tile on GPU. Tag and distinctness policies use finer
  candidate/permutation groups, retaining enough rows for downstream rankings.
  The runtime checks family and ascending AST order before selecting whole-tile
  grouping. Merges across tiles and devices use explicit family identities.
- Mixed raw/distinct policies have separate finalist sets and late-tag archives.
  Raw-only jobs have no expression-uniqueness continuation. Existing distinct
  policies may require additional CUB passes; reports expose their count.
- The reducer handle's stream/events persist; bulk buffers continue to come from
  the worker's reserved pools. Larger k/scratch requirements can reduce tile size
  under existing budgets, without losing requested work.

Request example:

```json
"retain": {
  "global": {"k": 16, "unit": "evaluation_row"},
  "per_family": {"k": 16, "unit": "evaluation_row"}
}
```

`reduction` reports backend, calls, continuation calls, CUDA-event kernel time,
local k requirements, and one-time reducer initialization attribution. A call
may contain several hierarchy kernels. Initialization includes NVRTC/module/
resource creation and is reported again as attribution, not repeated per job.
The final direct dual-GPU run records approximately 1.46 s total reducer setup
across workers. This is separate from numeric-helper NVRTC timing.

## Measured performance

Both devices are RTX 5080s. Prepared inputs: 1,024 groups × 2,048 scores;
21 timed repetitions after warm-up, same k/output, identical CPU reference.

| k | Previous GPU reduction | Integrated CUB | Speedup |
|---|---:|---:|---:|
| 1 | 14.72 µs | 11.23 µs | 1.31× |
| 4 | 43.42 µs | 28.83 µs | 1.51× |
| 16 | 811.49 µs | 94.75 µs | 8.56× |

A single global group of 2,097,152 scores at k=16 now takes 113.47 µs, versus
327.33 µs previously; it does not perform the expensive full-device sort from the
original trial. All 33 integrated layout/k cases matched the CPU reference.
The reused comparison executable labels its loaded public API `odezza_current`;
in [this run](results/cub-integrated-prepared.jsonl) that label refers to the new
CUB library selected by `LD_LIBRARY_PATH`. Standalone trial CUB methods in the
same executable retain their original benchmark build exception and explicit
capacity skips. The integrated method has no skipped cases or host build adapter.

**Remaining microbenchmark regressions:** 128 groups of 8,192/8,193 scores take
9.98 µs at k=1 versus 7.81/7.90 µs previously, and 22.18/23.90 µs at k=4 versus
19.65/19.84 µs. These are hierarchy overheads to tune within CUB, not grounds for
an unreported custom fallback. Hardware scope is these two SM120 devices.

Complete JSON requests generate **1,000,000 six-equation AST combinations ×
2,048 constant configurations = 2.048 billion evaluations**. These synthetic
trajectories contain three observations and four RK4 steps per evaluation; this
is a throughput benchmark, not blind recovery or representative long-rollout cost.
Medians of three warm runs:

| Request/path | Time |
|---|---:|
| Previous direct native, existing 20-structure retention | 1.5833 s |
| Integrated direct native, identical request/results | 1.2210 s |
| Integrated direct native, raw global/family top-16 | 1.2001 s |
| mac1 → mac3 → rack1, existing retention | 1.2654 s |
| mac1 → mac3 → rack1, raw global/family top-16 | 1.2142 s |

The matched direct request is **22.9% faster / 1.30× throughput**, approximately
1.68 billion configurations/s across both GPUs. The raw-row request has a
*different retention contract* and is not an equivalent-output speedup. Its
host retention drops to about 2.34 ms summed across workers, versus 116 ms for
the existing structure-retention request.

For the matched request, reduction/gather host time falls from 837 ms to 58 ms
summed across GPUs. Generation (656 ms), pipeline work (2.075 s), and other
reported stages overlap; do not add them to explain the 1.221 s wall time.
Cold runs include runtime creation and potentially cold templates and are saved
separately in [before](request-before.json), [after](request-after.json), and
[raw](request-raw.json). They are not used for the warm comparison.

## Validation

- 113 native score layouts/patterns/k combinations per GPU, including k=1/3/4/16/
  17/64/255/256, partial tiles, multiple hierarchy levels, invalid groups, ties,
  signed zeros, counts and coefficient gathering: **226 checks across two GPUs**.
- [Compute Sanitizer](results/cub-memcheck.log): **zero errors** on the reducer
  matrix. Sanitizer timings are instrumented and excluded from performance data.
- [Native suite](results/cub-native-suite-final.log): public export boundary,
  generator/inspection/specialization, scoring, Philox, and template round-trip.
- [Eight family/RNG cases](family-checks.json): independent analytic rankings,
  different tile sizes, both single GPUs and dual GPU, dominant ASTs, mixed units,
  late tags, raw k=256 and exact indexed/CPU RNG replay.
- [Existing conformance](results/cub-conformance.json): 15 cases.
- [Memory checks](results/cub-memory-checks.json): pools, budgets, shared accounting,
  shrink/reuse, error recovery and post-job live-byte checks.
- [Robustness campaign](results/cub-robustness.json): **159 passed, zero failed**,
  67,428,341 configurations, supplied grammar/RNG cases and eight random systems.
- [Six live service fixtures](live-conformance.json) match the existing direct-C
  reference. [Six live million-AST requests](live-million.json) match all native
  candidate fields, leaderboards and family counters; both GPUs do the work.
  `NativeService` adds formatted `equations` in Python. The C wire report returns
  authoritative resolved programs; only that derived string field is excluded
  from this comparison. The initial unprojected comparison failure is retained
  in `live-first-comparison.json`; no score or index discrepancy was found.
- mac1 frontend arena/grammar tests, 18,000 mutation measurement calls, allocation
  audit and eight supplied-grammar comparisons passed.

The broader suite exposed an existing single-system template rejection on the
unchanged baseline: NVRTC elides a final branch, producing 127 physical patch
instructions from a request for 128. The repair accepts this specific one-slot
shortfall and preserves the smaller inspected capacity for specialization.
Cold creation/artifact read have a regression test; Philox sampled-vs-explicit
scores are bitwise equal, and one-system scoring checks now pass. The 32-system
benchmark shape and scoring math are unchanged. See [deviations](deviations.json).

## Reproduction and deployment

Build normally with installed tooling; no installation is required:

```sh
make shared request-runtime test-public-api
make test-c test-cuda test-scoring-template
PYTHONPATH=python CUDA_MODULE_LOADING=EAGER python3 tests/runtime/cub_retention.py --output /tmp/cub-families.json
```

Run `request_benchmark.py` from the build under test. It saves all inputs, source
binary hashes, cold/warm runs and independent winner replays. `network_check.py`
checks the current service against those direct reports through the client tunnel.

The tested source was rebuilt in the live rack1 checkout, with the existing
supervisor, both GPUs, original pool sizes and host/RSS ceilings. mac3's generation
and queue were preserved. No Python search policy or grammar changes were needed.
Deployment/binary identities and verified backup locations are in
[deployment.json](deployment.json). No repository-wide commit is claimed.
