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

# Secant-SR toggle migration: 2026-09-19

Subsequent SRBench campaigns and the scale-aware final score-audit repair are
recorded in [srbench_toggle.md](srbench_toggle.md). The smoke timings below remain
the original pre-repair measurements, not measurements of that later executable.
The later native [selector-reuse experiment](../../secant/bench/settings_vs_toggles/report-selector-reuse-20260919.md)
is archived, not enabled by default. The default writer again uses temporary
selection results and assumes unrelated ASTs; the original regression remains
relevant for shared-selector fixtures.

## Delivered scope

The active C99 search uses Secant 0.3 native toggle ASTs and a shared immutable
coefficient bank. It retains selectors in the GP genome, evaluates banks × bit
permutations, and breeds those genomes after GPU argmin reduction. There are
separate host search, CPU reference, CUDA adapter, CLI, and Python data/codec
interfaces. No GP policy or compatibility shim was added to Secant's kernels.

The old build/header/README are preserved under `legacy/`; old search source and
campaign scripts remain in place but are excluded from the new build. See
[the new architecture](../toggle/README.md) and [compatibility boundary](../legacy/README.md).

## Executed checks

- mac1: four host/CLI/Python tests passed normally and with AddressSanitizer and
  UndefinedBehaviorSanitizer. They cover postorder lowering, all configurations
  of mixed selectors, fixed-AST replay, deterministic CPU evolution, transactional
  score validation, invalid call order, custom/constant targets, and malformed CLI.
- rack1, RTX 5080, CUDA 13.1, driver 595.91.07: GPU adapter agrees with the CPU
  oracle for 129 ASTs, 5 banks × 8 permutations, 3 columns, 2 coefficient slots,
  31 rows, chunks of 19 ASTs, 3 ASTs/kernel and 2 kernels/module. Tests include
  exact ties, invalid log domains, partial kernels/modules, rejected-AST recovery,
  and repeated use of one scorer.
- The same GPU test passed Compute Sanitizer memcheck with **0 errors** on
  rack1 GPU 0. Normal correctness used GPU 1.
- ada, RTX 4090 / SM 8.9, driver 595.91.07: the same adapter test passed. The test
  executable was built on rack1 against its already-installed static NVRTC
  libraries and copied to ada; no toolkit or compiler was installed there.
- The Python client also fitted an independently supplied two-column dataset on
  rack1's CUDA backend, returning `x0*x0+x1` with held-out MSE 1.16e-15 after
  12,582,912 configurations. Resolving its toggle AST using its returned vector
  and permutation exactly matched the returned fixed AST.

The core runner's earlier fault-injection campaign is separate evidence, not
fault coverage of every new adapter/search branch. This task did not change the
core runner or its failure contract.

## Search smoke panel

One RTX 5080, `CUDA_MODULE_LOADING=EAGER`, search seed 42 (bank seed derived by the CLI), population
1,024, 64 banks, 4 coefficient slots, 6 toggle bits, 512 training and 512 held-out
rows, maximum 100 generations/60 seconds. Operators: add/subtract/multiply/sine/
cosine. No truth expression was seeded. Packed shape: 8 ASTs/kernel, 16 kernels/
module, 128-row tile, 128 threads, batch 256, 3 workers and 8 streams.

These final runs used **warm compilation caches but a new process/context/scorer
for each problem**. Process wall time includes startup and teardown. Each
configuration evaluates all 512 training rows. These are occurrence counts,
including duplicate/unselected configurations, not unique expressions.

| Problem | Process wall | Scoring adapter | AST occurrences | Configuration evaluations | Held-out MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nguyen-1 | 0.227 s | 46.63 ms | 5,120 | 20,971,520 | 3.16e-15 |
| Nguyen-5 | 0.265 s | 92.61 ms | 10,240 | 41,943,040 | 8.95e-14 |
| Oscillator-2 | 0.181 s | 17.84 ms | 2,048 | 8,388,608 | 0 |
| Interaction-3 | 0.212 s | 46.50 ms | 5,120 | 20,971,520 | 0 |

Nguyen-1 simplifies to x³+x²+x. Oscillator-2 returned `cos(x1)+sin(x0)`;
Interaction-3 returned `x2*x2+x1*x0-x2`. Nguyen-5 returned
`cos(x0)*sin(x0*x0)-cos(cos(1.57154667))`: an excellent approximation to the
target's -1 term, **not the exact symbolic expression**. All four passed the
declared train/holdout NMSE threshold, and final JSON explicitly reports symbolic
equivalence as unchecked. This distinction must remain in future campaign tables.

Detailed Nguyen-1 attribution:

| Measurement | Time | Boundary |
| --- | ---: | --- |
| Initial setup | 125.36 ms | Includes context, dataset, template, scorer |
| NVRTC | 30.84 ms | Included in setup, warm internal compilation cache |
| GP generation/lowering | 2.73 ms | All scored generations |
| Scoring adapter | 46.63 ms | Includes core pipeline, argmin and gather |
| Core pipeline | 45.58 ms | Included in scoring; CPU/GPU stages overlap |
| Scoring GPU events | 8.36 ms | Included in core pipeline |
| Module loading | 15.15 ms | Included in core pipeline |
| GPU argmin events | 0.82 ms | Included in adapter |
| D2H event intervals | 0.22 ms | Included in adapter |
| Selection | 0.013 ms | Host GP state/selection reporting |
| Winner CPU audit/holdout | 0.025 ms | Separate fixed-AST evaluation |
| Scorer teardown | 44.44 ms | Included in reported total |
| Result ready | 174.79 ms | Before teardown |
| Reported total | 219.24 ms | Through scorer teardown |
| External process wall | 227.32 ms | Also includes process startup/exit |

Nested/overlapping rows must not be added together. These tiny early-converging
searches are dominated by setup/teardown and module handling, not steady-state
GPU execution. They do not measure maximum occupancy or demonstrate GP superiority
over PySR. Early development probes had cold NVRTC around 3.3–3.4 seconds and
roughly 3.5 seconds to a result; those used earlier controller revisions/timing
boundaries and are only a cold-start warning, not paired search benchmarks.

The final reproducible artifact is
[data/toggle-smoke-20260919.json](data/toggle-smoke-20260919.json), containing
commands, source/executable hashes, every attempt and complete winning records.
Re-run with `python3 toggle/run_smoke.py ./build-toggle/secant_sr_search NEW_OUTPUT_DIR`.
It refuses to overwrite an existing directory and retains failed attempts.
Raw JSONL and logs remain under ignored `scratch/toggle-search-20260919/` and
`rack1:/home/cdurham/experiments/secant-sr-toggle-20260919/`. The Secant runtime
snapshot used for manual GPU builds is
`rack1:/home/cdurham/experiments/secant-runner-hardening-20260919/`.

## Material differences and remaining limits

| Difference | Effect and disposition |
| --- | --- |
| New controller, not a port of all old policies | No maturity/QD archive, learned setting distribution, historical campaign protocol or LM. Legacy claims do not transfer. Broad quality validation and campaign migration are next. |
| Fixed sampled banks; no local coefficient fit | Precision can be limited or approximations can pass. Exact caller banks are supported in C. Continuous fitting remains deferred as requested. Host RNG is SplitMix64/Box–Muller, not Philox. |
| No ported persistent CUBIN/SQLite cache | Each scorer invokes NVRTC once for scoring and once for reduction; generations do not compile templates. Cold starts cost seconds. Add template persistence/reuse before interpreting short-search latency as steady state. No dependency installation workaround was used. |
| Existing packed-selector throughput regression | Still present in Secant relative to the old settings path on shared-choice workloads. This GP migration does not resolve it. Keep original item 3 open. |
| Full configuration grid remains on GPU | Bounded, reused pool; only one 24-byte winner record/AST is copied. Large single-AST products that cannot fit are rejected, not silently reduced. Fused reduction/overlap are future optimizations. |
| Numerical success is not structural identity | Explicit separate status; Nguyen-5 demonstrates why. Preserve exact returned AST and coefficients. |
| Fixed seed does not imply GPU bitwise search reproducibility | Floating-point accumulation and close fitness ties can change later selection. CPU deterministic tests and exact winner replay are covered; cross-device identical populations are not promised. |

An initial reducer build failed because NVRTC could not find an unnecessary
`math.h` include. Removing that include and using supported CUDA builtins fixed
the compilation; subsequent tests ran on the GPU without fallback. A probe using
unsupported built-in `nguyen9` was rejected before search and is not included as
a successful attempt. The final panel above is explicitly the supported four-case
panel, not a filtered SRbench run.

Original remaining items: **2** complete historical adapters and broader quality
validation/template persistence; **3** shared-selector/packing throughput tuning;
**4** native-toggle coefficient refinement is now implemented in the
[follow-up report](toggle-refinement-20260919.md); fitting policy tuning remains,
and LM is deferred. The new C99 GP, GPU adapter and
Python custom-data/AST path are implemented and validated within the scope above.
