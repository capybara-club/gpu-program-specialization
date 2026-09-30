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

# Rosenbrock GPU prototype results — 2026-09-08

The separate FP64 Rosenbrock23 prototype integrates all 134 supported noiseless
source systems successfully on rack1, including biological models with 13 states.
Every system reaches **worst-split MSE <=1e-8 against an independent accurate FP64
reference** at a tested tolerance. This required a tolerance sweep; it is not a
claim that one default tolerance meets that bound for arbitrary systems.

Code was authored on mac1 and executed with rack1's installed CUDA 13.1 and two
RTX 5080 GPUs. No production kernel, search policy, service, benchmark data or
qualification threshold was changed. [Run the prototype](README.md).

## Dataset coverage and accuracy

| Collection | Source systems integrated | Pass reference MSE gate at rtol=1e-8 | Pass after tighter settings |
|---|---:|---:|---:|
| MDBench, noiseless | 60 | 54 | 60 |
| ODEBench | 60 | 53 | 60 |
| Biological archive | 14 | 14 | 14 |
| Total | **134** | **121** | **134** |

The broad pass comprises 492 trajectory rows across training, validation and
test, including original initial conditions and masked temporal prefixes.
Each full source system is integrated once per setting, rather than repeating
the same integration for every choice of blinded RHS. This represents the 300
supported clean completion tasks; unsupported source models, noisy tasks and
unavailable Dynobench trajectories were not added to this numerical experiment.

Starting from rtol=1e-8 (atol=rtol/100), seven of the 13 remaining systems pass at
1e-10, four at 1e-12, and two Lorenz cases at 1e-13. The acceptance check is the
maximum raw observed-state MSE across all three splits, not a pooled average
that can hide a bad split. Tolerances were selected using private integration
references for this audit; this is not a deployed search selection policy.

| Difficult example | Tested passing rtol | Worst-split MSE vs accurate reference | GPU kernel time |
|---|---:|---:|---:|
| MDBench Lorenz chaotic (056) | 1e-13 | 9.08e-10 | 10.42 s |
| ODEBench Lorenz chaotic (056) | 1e-13 | 7.55e-10 | 13.19 s |
| ODEBench Lorenz complex periodic (055) | 1e-12 | 5.40e-9 | 10.60 s |
| Biological metabol1 | 1e-8 | 1.61e-10 | 140 ms |

These are tiny batches of actual dataset trajectories, not saturated throughput
measurements. Higher-order explicit integration already handles several of these
nonstiff/chaotic references efficiently. Rosenbrock23's low order makes very tight
long-horizon accuracy expensive; its largest tight-tolerance runs used roughly
1.67 million attempted steps in a lane. Stability alone does not solve chaotic
error amplification.

The 14 previously flagged systems were also run through CPU and GPU builds of
the same algorithm at three tolerances. Maximum split CPU/GPU disagreement was
1.68e-18 MSE across 42 settings. This is implementation parity; the independent
DOP853/Radau comparison is the accuracy check. Fused score residual counts were
reconciled against host recomputation across all retained sweeps. The final
seven-test regression suite and standalone JSON entry point also pass.

**Published-data discrepancies remain.** For example, ODEBench Lorenz 055's
accurate model trajectory still disagrees with the published observations by
roughly 469 worst-split MSE. The prior source-generator investigation already
explains why tighter integration does not erase such discrepancies. No source
agreement counts were upgraded, and this work does not claim symbolic recovery.

## True stiff controls and batching

| Control | Solver tolerance | Accepted steps | MSE vs independent Radau | Maximum absolute error |
|---|---:|---:|---:|---:|
| Decay rates 1 and 1,000,000, through t=1 | 1e-8 / 1e-12 | 2,043 | 6.21e-14 | 1.17e-6 |
| Robertson kinetics, through t=10,000 | 1e-8 / 1e-12 | 4,974 | 8.62e-17 | 5.79e-8 |

The decay case also agrees with its analytic solution. For that fast decay mode,
explicit RK4's stability interval alone implies roughly 359,067 steps across
the horizon; this is a stability calculation, not a measured RK4 speedup.
Robertson uses a separately written reference RHS, checks mass conservation,
and is not substituted for the real benchmark results.

Identical-work scaling at rtol=1e-6, atol=1e-10, median of three warm launches:

| Control | 1 lane | 128 lanes | 4,096 lanes |
|---|---:|---:|---:|
| Million-to-one decay | 3.227 ms | 3.565 ms | 3.569 ms |
| Robertson | 6.873 ms | 7.603 ms | 7.608 ms |

4,096 duplicate trajectories cost about 11% more kernel time than one. This
demonstrates unused parallel capacity in tiny validation jobs. It does **not**
measure 4,096 unique equations or parameter configurations: all lanes repeat
the same trajectory and have identical adaptive behavior. Heterogeneous banks
may diverge and need their own measurement. There is no claimed peak throughput.

## Time and kernel layout

The broad 134-system run took 133.65 seconds elapsed. Summed components:
29.63 seconds compiling 59 uncached models, 95.70 seconds obtaining independent
references, and 6.30 seconds of GPU kernels. Complete GPU calls totaled 6.68
seconds. The rest includes Python processing, reports and loading. This run used
some already cached models and two audit processes overlapped on different GPUs;
it is not a cold end-to-end comparison with production scoring.

| Generated kernel | Registers/thread | Stack/thread | Compiler spill stores / loads |
|---|---:|---:|---:|
| Lorenz, 3 states | 128 | 144 B | 0 / 0 B |
| Biological 3genes, 10 states | 255 | 1,792 B | 1,084 / 1,860 B |
| Biological 4genes, 13 states | 255 | 2,528 B | 1,120 / 1,828 B |

These are compiler resource reports, not measured aggregate memory traffic.
The small-system fused layout works; larger dense FP64 workspaces are already
spilling. The current API also allocates per call and uses pageable copies. It
waits for completion events on the successful path, but is not yet a persistent,
fully overlapped resident worker. FP64 arithmetic and full CUDA compilation
are deliberate prototype differences from Odezza's FP32 SASS specialization.

## Recommended next work

1. Keep Rosenbrock23 as a tested reference and compare a higher-order Rodas4-class
   method at equal independent accuracy. Retain RK4 for suitable nonstiff jobs.
2. Benchmark distinct coefficient/IC banks, their adaptive-step divergence, and
   true cold/warm costs before integrating a search-facing stiff option.
3. Reduce workspace liveness and eliminate known constant-state dimensions from
   the linear solve. Compare a cooperative small-matrix layout for larger systems.
4. Reuse the production specialization and bank interfaces, plus resident buffers,
   without importing grammar or search-controller policy into the kernel layer.

## Retained evidence

- [All 134 systems](validation/all-clean-v1/summary.json)
- [14 flagged systems with CPU parity](validation/flagged-v1/summary.json)
- [Tight chaotic sweep](validation/chaotic-tight-v1/summary.json)
- [Two additional convergence checks](validation/rossler058-tight-v1/summary.json)
- [Reconciled per-system tolerances and metrics](validation/reconciled.json)
- [Stiff controls and scaling](validation/stiff-controls-v1.json)
- [Compiler metadata and spill reports](validation/compiler-metadata.json)
- [Final tests](validation/tests-final.log)
- [Standalone JSON request result](validation/stiff-decay-result.json)

`report.py` reconciles fingerprints, successful statuses, residual counts and
the stated per-split reference gate from the retained sweeps. All artifacts
are retained on mac1; GPU build caches remain on rack1.
