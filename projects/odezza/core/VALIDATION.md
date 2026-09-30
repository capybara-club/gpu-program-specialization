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

# Native core validation

## 2026-09-10 cooperative shape integration

Native LM now supports explicit 1/2/4/8-lane shapes and wider register fallback.
The original public structs and exact-shape entry point retain their layout;
two additive functions expose fallback creation and shape reports (26 exports).
All three hosts pass the native numerical/coupled-proposal/failure suite and the
shape/toggle CPU replay checks (4,284 fits, complete arrays equal across widths).
All 72 host/shape preflight entries pass with fallback. The search adapter adds
168 independent CPU replay checks across widths and hosts; Ada memcheck reports
zero errors. Real 8x8 resource fallback selects four lanes.
See [the current API](README.md), [retained results](../scratch/multilane_core_validation/)
and [correctness findings](../docs/incidents/2026-09-10-native-cooperative-lm.md).

This integration also repairs simultaneous coupled sensitivities. Earlier native
LM Jacobian/optimization-path and timing claims below are historical; independent
primal MSE replay remains valid. Rebaseline before selecting tuning defaults.


## 2026-09-10 resource rejection correction

The overnight tuning run exposed a valid 255-register template incorrectly
reported as FORMAT. LM inspection now returns REGISTER_PRESSURE for that existing
rejection and creation diagnostics include states, parameters and register count.
The accepted kernel shapes, fitting mathematics and output layouts are unchanged.
Malformed or absent register metadata still returns FORMAT.

Separate repair builds pass native numerical, failure cleanup and public ABI
tests on rack1/rohini (SM120) and ada (SM89). The public C99 regression rejects
eight states/six parameters, destroys the failed creation and then runs the
existing 201-fit numerical test. End-to-end worker tests on each host record
the original unsupported shape, finish paired curvature work, and verify the
next control with both methods in the same process. Held-out MSEs are 1.49e-17
(curvature) and 2.58e-15 (LM). No fallback or larger-shape support was introduced.

Evidence: [failure audit](../benchmarks/lm_tuning/runs/20260909-night/FAILURES.md)
and [isolated repair checks](../benchmarks/lm_tuning/runs/20260909-night/diagnostics/fix/).
The original campaign's frozen source and failed outcomes remain intact.

## Original API validation

Validated 2026-09-09 on mac1 and rack1 (RTX 5080, SM 120). The isolated rack1 build
is `/home/cdurham/odezza/scratch/core_api_validation/source`; the deployment and
historical research snapshots were not overwritten. Retained local evidence is
under [scratch/core_api_validation](../scratch/core_api_validation/).

## Boundary and numerical checks

| Check | Result |
|---|---|
| Strict C99 Make build | Passed with warnings treated as errors; static/shared libraries and tools |
| Public ABI exports | Exactly 24 declared functions; ELF/SASS/compiler helpers remain private |
| Existing native scoring contracts | Generation, NVRTC diagnostics, inspection, specialization and lifecycle passed |
| Shared SASS writer | 3,003 assemblies across three architecture encodings; no capacity rejections |
| CPU LM expression checks | 132 analytic-partial comparisons against finite differences; bounded writer and invalid ELF checks passed |
| Public-header-only GPU LM caller | 201 fits in three packs, partial CTAs; all coefficient errors below 1e-4 and MSE below 1e-10 |
| LM rejection/ownership | Workspace, aliases and counter-overflow guards passed; eight injected load/launch/event failures drained, cleaned up and allowed reuse |
| CUDA memory checker | Zero errors, including the eight injected failure cases |
| Scoring GPU regression | 1-state and 8-state, scalar and packed cases passed |
| Philox/reduction | Sampled/explicit scoring bitwise equal; reducer exactness checks passed |
| Thin binding ownership | Three tests cover failed creation, workspace query/allocation and failed cleanup with retained diagnostic handle |

CPU-only AST tests on mac1 use a tiny test header containing CUDA opaque types,
with no driver functions. This permits testing mathematical lowering locally;
it is not a CUDA simulation and is never included in the native library build.
GPU claims above come from the real CUDA driver on rack1.

The shape replay controls used 67 starts each with nonlinear prepared RHSs:

| States | Parameters | Best native GPU MSE | Independent CPU MSE | Registers |
|---:|---:|---:|---:|---:|
| 3 | 3 | 1.0783e-14 | 6.7725e-15 | 105 |
| 3 | 6 | 6.0821e-15 | 7.3795e-15 | 177 |
| 6 | 3 | 9.4351e-15 | 3.0542e-16 | 151 |
| 6 | 6 | 9.8496e-15 | 2.3503e-16 | 240 |

Both DOP853 and Radau verification passed. These intentionally supplied correct
structures to test coefficient fitting; they are not blind recovery experiments.
The population is small and does not establish saturated throughput.

## Downstream search

The existing fitting trial was run through an isolated overlay selecting the
new shared library and a scoring worker linked to the current C99 library.
The native fitting path uses C99 generation, inspection and specialization;
there is no fallback to Python JIT or a CPU fitter.

- Four candidate families, leaf toggles, eight starts, dense and hidden-state
  observations: 14 winning records matched independent FP64 RK4 replay.
- Unused coefficients, fixed coefficients, bounds, template reuse and cancelled
  work passed. Fully fixed candidates were scored once by the C scoring worker.
- The unchanged campaign controller passed training/validation/test verification.
- The unchanged four-island search ran on both GPUs: 448 ASTs, persisted
  checkpoints, then 32 additional ASTs after resume. Cancellation passed.

See retained [final integration](../scratch/core_api_validation/integration-03/checks.json),
[GP controller checks](../scratch/core_api_validation/search-01/checks.json),
[shape replays](../scratch/core_api_validation/shapes-checks.json),
[native regressions](../scratch/core_api_validation/checked-build.log) and
[memory checking](../scratch/core_api_validation/checked-memcheck.log).
The [source identity check](../scratch/core_api_validation/source-identity-checks.json)
confirms the 35 source/build files in the manifest match mac1, and the static
library imports no CUDA context-management symbols. Final counter-overflow and Python initialization-cleanup
changes were followed by native fault/memory regressions and focused integration
replay; they do not change fitting mathematics or GP policy.

## Explicit limits and next measurements

Only the one-lane specialized LM shape is supported in the new public handle.
Cooperative shapes remain experimental. Resource limits can reject larger or
more complex fits even within nominal state/parameter limits. Oversized shared
trajectory staging is rejected; data contents and allocation validity remain
the caller's responsibility under the documented buffer contract.

Native templates persist, but specialized modules are retired after completion,
matching scoring's bounded lifecycle. The previous Python trial retained modules.
The native adapter records this difference in every report. Correctness and
search compatibility are validated; comparative warm throughput remains
unmeasured. The next performance check should match equations, trajectories,
starts, stopping rules, cache state and occupancy before choosing a default.

The Linux Make build was executed. CMake source/install wiring was updated but
not built: rack1 has no CMake, and mac1 has no CUDA toolkit. No new tools were
installed. GPU execution was validated on SM 120 only; the three-architecture
encoding corpus is not a substitute for running LM on the other GPUs.
