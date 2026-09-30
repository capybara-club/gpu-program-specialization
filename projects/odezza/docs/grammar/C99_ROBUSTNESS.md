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

# Native grammar robustness campaign

2026-09-11 local time (2026-09-12 UTC), isolated rack1 checkout, two RTX 5080s.
This exercises the native scoring runtime. Existing GP services and the core
kernel implementation were not changed.

## Results

| Check | Result |
|---|---|
| Main campaign, seed 20260911, device 0 | 191/191 checks pass |
| Fresh seed 20260912, device 1, CUDA memcheck | 191/191 checks pass; zero device-memory errors |
| Configuration evaluations across those two campaigns | 134,816,986, including deliberately invalid math candidates |
| Controlled coupled recovery | 24 systems per seed; all 48 have a retained solution below MSE 1e-10 |
| Chunk/page invariance | 79 paired comparisons per seed; exact IDs, FP32 scores, programs, coefficient bits, tags and leaderboards agree |
| Retained-candidate independent FP64 rollout checks | 2,364 across the two campaigns |
| Independent Philox/transform reconstructions | 5,258 retained RNG slot values checked |
| Existing native runtime regression suite | 15 contracts pass, including cancellation, bounded retention timeout, late tags, context reuse and MCP |
| CPU parser mutation test | 18,000 measurement calls; 817 successful measurements and 798 complete builds, with guarded short/full arenas |
| CPU ASan/UBSan | Parser contracts, mutation tests and 225 Python-oracle program comparisons pass |
| Existing Python grammar and transport tests | 83 + 19 pass |

The main campaign has 29 intentional rejection cases; these count as passing
checks only when a failed job is returned without quarantining its CUDA runtime.
Completion of a rejected request is not scored work. Deliberately undefined
operators are checked separately as invalid numerical evaluations.

### Coverage

- Fixed scalar constants, named grid banks, Cartesian grid products, unused
  declarations, parameter initial values and independent production-local slots.
- Uniform and normal bases; identity, affine (including negative scale), uniform,
  normal (including zero standard deviation), and log-uniform transforms.
  Transform operands include fixed/grid constants absent from the RHS itself.
- Shared and independent RNG axes, separate streams, run/skeleton scope, local
  occurrence identity, non-multiple-of-four pool lengths, and a full-width seed.
- Binary/quad state toggles with all, explicit, sampled and joint sampled groups.
- Recursive sampling, parameterized productions, shared shape choices, pruning,
  duplicate structures, family rule overrides and late Unicode tag memberships.
- Numeric retention at k=0/1/17/32, duplicate bank rows beyond the first 16 GPU
  winners, structure retention and global/family/tag rankings.
- All 16 supplied scoring grammars, including the four biology examples and
  family-only adaptation/enzyme/oscillator grammars. Their operators, banks,
  transforms and retention policies are preserved. Work is explicitly bounded
  to 128 skeletons/variants and 262,144 configurations per request. This is
  execution of bounded prefixes, not exhaustive execution of every supplied space.
- Fine FP64 synthetic trajectories for coupled systems with 2/3/6/8 states,
  2–4 initial conditions, 3–7 irregular observations, missing observations and
  fixed-RHS prespecialization. Fully known systems with no variable RHS also run.
- Balanced expressions with 8/32/96 nonlinear terms and 1/3/6/12 states;
  forced patch growth from capacity 16; cold/warm template reuse.
- A larger mixed workload: 4,096 complete six-equation combinations, 2,048
  shared random rows × two explicit offsets × four state permutations =
  67,108,864 configurations. Two trajectories and four timestamps per trajectory.
- Host/device/pool/dedup limits, invalid time/IC layouts, unknown fields,
  invalid transform domains, unequal shared-axis lengths and unsupported solvers.

The controlled recoveries choose a truth from the finite candidate family and
generate its observations independently in FP64 with much smaller RK4 steps.
The request contains observations, known equations and the candidate grammar;
the expected answer is only used by the test. This establishes coverage/scoring/
retention correctness on controlled problems. It is **not** a benchmark of an
LLM or GP discovering an unrestricted hidden expression. Likewise, CPU replay
checks retained candidates, not every one of the 134.8 million evaluations.

RNG verification uses an independent Python integer Philox implementation and
scalar Box–Muller/transform calculations from the returned address. It does not
call the native Philox helper to calculate its expected value. Normal and
log-uniform checks allow small differences in GPU transcendental arithmetic;
chunk invariance requires exact retained FP32 results.

## Repaired defects

### Numeric pool cache could exhaust its entry table

The original runtime evicted old banks only when its byte budget was exceeded.
With many small skeleton-scoped banks, the 4,096-entry table filled while most
entries belonged to finished tiles and memory was available. The first campaign
passed 151 checks but stopped this case at 20,160 of 21,000 requested evaluations.
That initial job is a valid partial failure, not a complete result.

The runtime now evicts the least-recently-used completed binding on either entry
or byte pressure. Entries referenced by the binding currently being constructed
are protected. Failed device-memory retirement quarantines the runtime. Reports
expose `cache.numeric_pool_evictions` so this behavior is visible.

The repaired entry-pressure and byte-pressure tests each finish all 4,200 ASTs ×
five configurations. They pass ordinary execution and CUDA memcheck, including
subsequent jobs in the same persistent context. An active binding exceeding the
available cache still receives an explicit resource error; there is no hidden
CPU fallback or changed sample set.

Earlier completed configuration-scale benchmarks did not hit this entry limit
and remain valid. The repair preserves the numeric address profile and sample
values; it changes when unused GPU pool allocations are retired.

### Extra fields beside `choices` were ignored

An object such as `{"choices":["-x"],"weights":[1]}` was accepted while discarding
the unsupported field. This could give a false impression that a sampling policy
or annotation was honored. Rule and shape choice wrappers now accept only the
`choices` field. Unsupported siblings fail explicitly; valid wrappers and normal
production tags remain supported. Positive and negative CPU contracts and a
runtime rejection case cover the change.

No scoring mathematics, register allocation, specialization implementation or
kernel shape was modified. The repairs are in the frontend and service runtime.

## Reproduce

On an already configured CUDA checkout:

```sh
make test-request-robustness
make test-request-robustness ROBUSTNESS_DEVICE=1 ROBUSTNESS_SEED=20260912 \
  ROBUSTNESS_OUTPUT=/tmp/odezza-robustness-second.json

CUDA_MODULE_LOADING=EAGER PYTHONPATH=python compute-sanitizer \
  --tool memcheck --error-exitcode 99 \
  python3 tests/runtime/robustness.py --device 1 --seed 20260912 \
  --output /tmp/odezza-robustness-memcheck.json
```

Use `--only pool_` for the cache regression or `--only coupled_recovery` for the
controlled recoveries. `--random-count` changes the number of synthetic systems.
All requests and reports are saved incrementally, including failed cases. The
standard `make test-frontend` and CMake/CTest frontend targets include mutation
testing. No additional Python packages are required by the new harness.

## Limits and next work

Instrumentation, warm/cold cache history and simultaneous independent jobs make
these campaign durations unsuitable for hardware speed rankings. GPU 1 runs an
independent runtime; this is not multi-GPU scheduling of one job. The existing
whole-repository legacy cubin-packer failure remains outside these focused checks.

Next: automatic tile sizing without changing coverage; configurable reducer
width; resource-aware batching; broader uncontrolled recovery and longer/stiffer
trajectory cases. The tests do not establish that every syntactically legal AST
fits every scoring shape or that all malformed inputs have been exhausted.

Evidence: [main records](runtime-validation/odezza-robustness-final.json),
[fresh-seed instrumented records](runtime-validation/odezza-robustness-memcheck.json),
[CUDA checker](runtime-validation/odezza-robustness-memcheck.log),
[original partial failure](runtime-validation/robustness-initial.json),
[original ignored field](runtime-validation/choices-unknown-field-initial.json),
[existing regression](runtime-validation/odezza-robustness-conformance.json),
[verified Make target and final binary hashes](runtime-validation/odezza-robustness-target.json),
[CPU sanitizer log](runtime-validation/robustness-cpu-sanitizers.log),
[CPU regression log](runtime-validation/robustness-cpu-regression.log),
[CMake/CTest log](runtime-validation/robustness-cmake.log),
[harness](../../tests/runtime/robustness.py).

The final Make-target verification repeats the main seed and passes all 191
checks. Its repeated work is excluded from the two-campaign totals above.
