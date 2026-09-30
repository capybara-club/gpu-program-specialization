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

# Scoring kernel review — 2026-09-05

The current C99 scoring path is substantially better defined than its surrounding research tools suggest. Its central risks were template register liveness, insufficient validation of trajectory data, and stale checks that no longer matched the generated kernel. This review changes the generator and launch validation while retaining the existing thread-per-configuration RK4 topology and SASS scheduling policy.

## Implemented corrections

- **Permutation liveness:** an eight-state/eight-constant, one-system SM120 template failed inspection because the compiler reused the permutation input register for an output. The inline PTX now keeps that input live through output materialization. The inspector still rejects overlaps; no inspection check was relaxed. The one-system shape runs directly on SM120 and SM89.
- **Smallest shape:** with one state and zero constants, the scaffold's select read operand 1 before it had been initialized; that operand was an RHS output. It now selects between initialized inputs. A GPU test covers this shape, including actual toggles.
- **Nonfinite observations:** a NaN initial observation in a singleton trajectory could be ignored, yielding a finite score. The block now checks reference values and times cooperatively during shared-memory loading and rejects nonfinite observations, including singleton points. A positive interval whose FP32 step underflows to zero is also rejected.
- **Launch preflight:** grid and reference-layout limits are checked before any specialization is queued. Full byte spans are checked for overflow and alignment, including accesses inside the last module. Previously, checking only module starting offsets missed those cases. Bulk module counting uses overflow-safe ceiling division. These checks cannot establish actual device allocation sizes or context ownership; those remain caller responsibilities.
- **Artifact identity:** the canonical manifest used for hashing and the printed manifest used different configuration descriptions. They now share that text. Independent Python checks recompute both the payload hash and template identity from the actual C99 artifact. C99 generator ABI is now 4, distinguishing it from the historical Python scorer's ABI 3. The public C API signature and AST encoding are unchanged.
- **Generation cost:** sizing requests no longer hash the payload, because encoded digest lengths are fixed. Reserved breakpoint instructions use a constant string instead of repeated formatting.

The public header documents the observation contract, `FLT_MAX` penalty, and the fact that failed bulk runs may have partially written outputs. A returned error means that run's scores should be discarded.

## Measured result

Rohini, RTX 5090, CUDA 13.1.115. Both versions use isolated snapshots of the actual dirty working tree; the existing machine checkouts were preserved.

| Measurement | Before | After | Change |
|---|---:|---:|---:|
| Source sizing + writing, 8 states / 8 constants / 128 systems / 384-slot patches | 21.515 ms | 9.226 ms | 2.33× faster |
| Mixed candidate scoring | 441.65 M configurations/s | 435.10 M/s | 1.48% lower |
| Valid-only scoring control | 445.32 M configurations/s | 438.56 M/s | 1.52% lower |

Generation timings average 20 calls per operation and exclude NVRTC. This is a source-generation improvement, not a claim that large fused kernels compile 2.33× faster.

Scoring timings are medians of five alternating process invocations per version and workload. Each invocation runs five synchronous calls with 64 candidate systems, packed eight per module, 65,536 banks, and one toggle bit. Timing includes specialization, module loading, execution, retirement, and unloading; handle creation and downloads are outside it.

All **8,388,608 outputs per workload** are bitwise identical before and after. The mixed workload has 6.25% penalty scores; the valid-only control has none. The control forces the benchmark's ordinary Lotka–Volterra candidate variant while retaining its bank values and per-system biases. It is explicitly a separate workload. These tests establish equivalence for those fixtures, not all possible ASTs or GPU architectures.

The final measured shape uses 48 registers versus 47 before. The approximately 1.5% throughput cost is retained with the correctness fixes and is not described as a scoring speedup.

### Experiments discarded

Early exit after invalid integrated states or squared-error overflow increased register use to 55 and reduced throughput by about 17%, despite identical results. Moving all offset/time validation into a separate cooperative pass also used 55 registers and remained slow. Neither experiment is in the final code. Existing per-observation state checks and final penalty behavior remain; divergent candidates do not gain a new early-termination path.

[Raw results and source hashes](../benchmarks/raw/2026-09-05-scoring-review-rohini.json) retain both final measurements and the discarded experiment. [Measurement details](../benchmarks/raw/2026-09-05-scoring-review-method.txt) specify the workload modifications and generation harness.

## Validation and known gaps

- **18/18 CMake tests pass on both Rohini/SM120 and Ada/SM89.** New cases cover ragged analytic trajectories, every system/bank/toggle output in the fixture, scalar zero-constant and eight-state shapes, bad offsets, nonfinite data, step underflow, overflowing/misaligned launch spans, zero submissions for preflight rejection, malformed AST recovery, and division-by-zero penalties.
- **41/41 Python tests pass.** These cover the historical scorer and LM reference; they are not a claim of C/Python runtime-kernel parity.
- **CUDA Compute Sanitizer:** the final packed eight-state regression reports zero memory errors and zero race hazards on Rohini.
- The repaired Makefile artifact/CLI target and new `make test-cuda` target pass. The scratch AST tool's default-directory test also passes (including its 10 Python tests).
- **`make test` is not fully green:** the unchanged legacy `scratch/c99_cubin_runner/pack_input.py` still reads the removed `KernelShape.toggle_permutation_count` property. The same failure was confirmed against the baseline. Its older launch/input protocol needs a separate compatibility repair; the failing check was retained.
- Running the scratch AST test with inherited `BUILD_DIR=build-make` failed because its file round-trip test assumes `build/`. Re-running the documented default-directory target passed. That test portability issue is separate from kernel correctness.
- No Rack1 deployment, SM90 validation, new RNG implementation, LM topology change, or broad high-state-capacity sweep was performed.

## What should happen next

1. **Treat the C99 runtime artifact and GPU regressions as the scoring contract.** Keep the historical Python scorer explicitly separate. Restore the legacy scratch runner only with matching launch-ABI and end-to-end GPU tests; fixing its packer alone would not establish compatibility.
2. **Measure the next optimization against actual request shapes.** The results above do not justify changing the scoring topology. Track register count, patch capacity, compile time, and complete request time together. Module-load/unload wall timings can include waiting for GPU work; they are not isolated driver overhead measurements.
3. **Add explicit numerical result statuses before service-level top-k.** `FLT_MAX` currently conflates bad observations, candidate divergence, and FP32 accumulation overflow. In particular, a sum of squares may overflow even when its mathematical mean is representable. Wider or scaled accumulation needs a separate numerical/performance decision and reference tests.
4. **Keep expression authoring separate from physical generation.** A Python AST/family builder with explicit tags and filters remains the useful authoring layer. The kernel generator should compile reusable capacities and expose measured limits. This review improves CUDA generation; it does not add a new family grammar or search policy.
5. **Benchmark resident-module reuse and bounded output reduction next.** Repeated evaluations of the same specialized family may benefit more from reusing loaded modules and returning compact top-k/statistics than from another unmeasured integration-loop change. Measure that against the persistent-service workloads in [the service plan](scoring_service_plan.md).
