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

# Runner failure handling and hardware coverage — 2026-09-19

Secant 0.3's scoring runner now proves GPU completion before unloading a module,
retains resources when teardown fails, and rejects reuse when completion or
module cleanup remains unresolved. These changes affect the runner and its
public error contract. AST semantics, CUDA source generation, specialization,
packing, and the bank × permutation product are unchanged.

## Defects and fixes

| Previous behavior | Current behavior |
| --- | --- |
| A failed completion event could lead directly to module unload. | Try independent completion events on every owned stream; use stream synchronization only where the recovery event fails. Unload only after completion is established. |
| A failed recovery wait was ignored and the runner could be reused. | Return `SECANT_ERROR_COMPLETION_UNKNOWN`, retain the module and streams, and reject new runs. |
| A failed worker join still allowed its shared memory to be freed. | Retain storage and retry only workers not successfully joined. |
| CUDA teardown errors discarded handles regardless of whether release succeeded. | Clear only successfully released handles; failed handles remain owned for another destroy attempt. |
| Construction ignored errors while cleaning up partially initialized resources. | Return a destroy-only handle when cleanup itself fails, so ownership is not lost. Ordinary construction failures still return NULL. |
| `total_seconds` excluded request normalization and validation. | Successful/work-started calls include those stages and host cleanup; preflight-only failures still return zero counters. |

The issues existed in the rewritten 0.3 runner before this hardening pass.
Ordinary successful numerical results remain valid. The previous failure paths
had no established safe lifetime/reuse guarantee; outputs from failed calls
must not be used. The regression tests below cover the repaired transitions.

## Caller contract

- Successful run: device buffers are available for reuse; the runner is reusable.
- Failed validation or specialization: discard the failed call's scores. The
  runner is reusable once the blocking call returns.
- Recoverable CUDA operation failure: discard scores. If completion and module
  cleanup succeeded, subsequent runs are allowed.
- Module unload failure: buffers are no longer in use by the GPU, but the runner
  is destroy-only. Subsequent run calls return `SECANT_ERROR_INVALID_STATE`.
- `SECANT_ERROR_COMPLETION_UNKNOWN`: keep **all device buffers** submitted to
  that call valid. Retry destroy in the same current CUDA context. The retained
  module is not unloaded until completion is proven. Host AST storage is no
  longer used after the run returns because specialization workers are drained.
- Destroy invalidates the handle **only on success**. On teardown error, retain
  and retry it. A wrong context or a currently active run rejects destruction
  without starting teardown. Externally serialize destruction against all use.
- Create failure normally returns NULL. If it also encounters a cleanup failure,
  it returns a non-NULL destroy-only handle. Always inspect that output, even on
  failure, and keep the borrowed plan alive until destroy succeeds.

No device-wide synchronization, new production dispatch table, fault flags, or
GPU allocation was added. Error recovery may wait on individual owned streams;
normal execution continues to use events. A permanently lost CUDA context/device
is not repaired by this API. A worker process may need to be restarted; buffers
cannot be assumed safe merely because a driver call returned an error. Explicit
context teardown also ends their device lifetime, but there is no separate API
for abandoning the retained runner after external context destruction.

## Deterministic failure tests

`tests/runner_failure_test.c` compiles the actual public runner and pipeline
against a test-only CUDA driver model, using real pthread workers. The library
build has no test hooks. Only the test translation units redirect allocation and
thread setup/teardown calls. The model tracks stream/event dependencies and
asserts that a module is never unloaded with unfinished kernel work.

The 122 scenarios include a baseline plus every operation reached in successful
create/run/destroy, failed individually, and compound cases:

- Every host allocation, mutex/condition initialization, and worker creation.
- Partial stream/event setup and cleanup.
- Output clearing, module loading, function lookup, launch, event recording,
  stream/event ordering, event completion/timing, and module unload.
- Failed joins and stream/event/mutex/condition destruction, then cleanup retry.
- Reuse after recoverable failures, specialization failure, and wrong context.
- Persistent completion failures, including failure of the stream fallback;
  retained module/buffers and rejection of subsequent work.
- Construction failure followed by cleanup failure, without losing ownership.
- Concurrent run rejection and rejection of destruction during an active call.
- Exact error-code checks and zero outstanding modeled resources after cleanup.

The model tests ownership and ordering, not CUDA arithmetic or real device-loss
behavior. Those distinctions matter; the actual GPU tests below are separate.

| Validation | Result |
| --- | --- |
| mac1 release host suite | 6/6 passed |
| mac1 AddressSanitizer + UndefinedBehaviorSanitizer | 6/6 passed |
| mac1 ThreadSanitizer runner suite | 122 scenarios passed, no reported races |
| rack1 GPU 1, RTX 5080, SM 12.0 | Toggle and static CUDA suite passed |
| rack1 GPU 0, RTX 5080, SM 12.0 | Same suite under compute-sanitizer memcheck; 0 errors |
| ada GPU 0, RTX 4090, SM 8.9 | Toggle and static CUDA suite passed |

All GPU runs used eager loading, CUDA 13.1 NVRTC and driver 595.91.07. Ada lacked
build tools/toolkit, so the repository test executable was built on rack1 using
its installed static NVRTC/compiler libraries and copied to ada. It generates,
compiles, inspects and runs SM 8.9 templates on ada; it does not execute SM 12.0
binaries there. No software installation was needed. Rohini did not answer SSH;
no test ran there. SM 9.0/10.0 and other compiler versions remain unverified.

GPU tests include independent formula checks, CPU-reference comparison,
packing 1/3/8/32, shared banks, padded layouts, changed bank values, repeated
runs and batches, partial modules, invalid-AST recovery, zero-input/constant/bit
cases, bit 31 encoding, random ASTs, and all four retained static shapes.
Each of the three random-packing cases scores 129 ASTs × 5 banks × 256
permutations × 97 rows = 16,016,640 row evaluations. Comparisons use the test's
existing `2e-4 * (1 + abs(reference))` tolerance.

Observed specialized resources (8 shared coefficient slots):

| GPU | ASTs/kernel | Max registers/thread | Local bytes/thread |
| --- | ---: | ---: | ---: |
| RTX 5080 | 1 | 47 | 0 |
| RTX 5080 | 8 | 53 | 0 |
| RTX 5080 | 32 | 168 | 16 |
| RTX 4090 | 1 | 43 | 0 |
| RTX 4090 | 8 | 50 | 0 |
| RTX 4090 | 32 | 72 | 0 |

These are correctness/resource checks, not a controlled throughput comparison.
The [packed-selector performance regression](../bench/settings_vs_toggles/report-20260919.md)
is still open. In particular, this work does not remove the 5080's local-memory
use at packing 32 or eliminate repeated shared selections inside the row loop.

## Reproducing the host checks

```sh
cmake -S . -B build-toggle -DSECANT_ENABLE_CUDA_INTEGRATION=OFF \
  -DSECANT_BUILD_BENCHMARKS=OFF -DSECANT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-toggle -j8
ctest --test-dir build-toggle --output-on-failure
```

For ASan/UBSan, use a separate build directory with
`-fsanitize=address,undefined -fno-omit-frame-pointer` in both C/C++ flags and
`-fsanitize=address,undefined` in executable linker flags. For ThreadSanitizer,
use another directory with `-fsanitize=thread`, build
`secant_runner_failure_test`, and run that test alone.

Normal CUDA-enabled CMake builds register `secant_toggle_cuda_test` automatically
when CUDA is available. The portable executable used here was built with:

```sh
cc -std=c99 -D_POSIX_C_SOURCE=200809L -O2 -Wall -Wextra -Werror \
  -I. -I/usr/local/cuda/include \
  s_result.c s_cpu.c s_cubin_api.c s_cubin_generate_materialize.c \
  s_cubin_generate_sse.c s_cubin_generate_affine_stats.c \
  s_cubin_generate_gram_stats.c s_cubin_generate_toggle_sse.c \
  s_cubin_inspect.c s_cubin_specialize.c s_cubin_runner.c s_runner_pipeline.c \
  tests/toggle_cuda_test.c -L/usr/local/cuda/lib64 -lcuda \
  -lnvrtc_static -lnvrtc-builtins_static -lnvptxcompiler_static \
  -ldl -lpthread -lrt -lm -o toggle_cuda_test
CUDA_MODULE_LOADING=EAGER ./toggle_cuda_test
CUDA_MODULE_LOADING=EAGER /usr/local/cuda/bin/compute-sanitizer \
  --tool memcheck --error-exitcode=99 ./toggle_cuda_test
```

Raw GPU logs are in ignored `scratch/runner-hardening/`. Remote test artifacts
are under `/home/cdurham/experiments/secant-runner-hardening-20260919/` on rack1
and ada. The test executable and logs are not included in source control.

## Matched throughput check after hardening

The user requested a full-runner throughput recheck. The existing pre-hardening
0.3 toggle binary and the hardened binary ran on the same RTX 5080 (GPU 1).
Each case used three paired passes with alternating order; each reported pass
was the median of three externally timed runner calls after warmup. All
paired score files had identical SHA-256 hashes. The harness also retained
its independent CPU-oracle and signed-prediction checks.

All cases use 128 AST occurrences, eight ASTs/kernel, 16 coefficient banks
× 256 permutations = 4,096 configurations/AST. These are full blocking C99
runner calls with resident inputs and a prepared template: output clearing,
native specialization, module loading, scoring/SSE reduction, completion and
unloading are included. Initial NVRTC, runner construction, data transfers,
and JSON/GP work are excluded.

| Workload | Before | After | Time change | After, billion row evaluations/s |
| --- | ---: | ---: | ---: | ---: |
| arith-8192 | 7.288 ms | 7.286 ms | -0.03% | 589.5 |
| sin-8192 | 8.277 ms | 8.287 ms | +0.12% | 518.3 |
| arith-256 | 0.599 ms | 0.624 ms | +4.16% | 215.1 |

The two large cases differ by less than 0.2%. The short case measured 25 µs
(4.2%) higher median latency: pre-hardening pass medians ranged 0.599–0.607 ms,
and post-hardening medians ranged 0.602–0.657 ms. That small, variable sample
does not isolate the cause; it is retained rather than declared equivalent.
The added error-recovery path is not used in these successful calls.

This preserves bulk throughput of roughly 0.59 trillion arithmetic row
evaluations/s through the runner. It does **not** resolve the earlier regression
against the original settings implementation. One row evaluation is one AST
× one bank/permutation configuration × one input row, including squared-error
accumulation and reduction.

Raw measurements and binary hashes: ignored
`scratch/runner-hardening/throughput-comparison.json`; reproducible check script
`scratch/runner-hardening/check_throughput.py`. Remote copies and full per-pass
scores/logs are in the existing rack1 experiment directory under `throughput/`.
