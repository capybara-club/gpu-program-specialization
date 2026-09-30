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

# Singular proposals aborted an LM search

The frozen `focused-night-20260910` random campaign stopped rack1 GPU 1 on
`case-009 / more_coefficients`, group `7c7589f747ee4a3a4c65cf65`. The fitting
campaign reported `division by zero` and the worker stopped its queue. Other
queues retained their original source and continued. The failed trial and its
unstarted successors must remain visible; this is not a search timeout.

The failing request is retained under
`benchmarks/search_study/runs/20260910-robustness/incident/failed-native-request.json`.
Its source command is `0008-refit-0000-gpu1` in campaign
`4105a9e4ad83de4e5098141cf0a6f424` on the original worker. The source runtime is
`/home/cdurham/odezza/scratch/focused_focused_night_20260910`.

## Cause and repair

An automatically proposed expression contained a literal zero denominator.
Python's symbolic derivative constant folding raised `ZeroDivisionError` before
packing the other candidates. Merely preserving that operation in Python was
insufficient: native LM correctly rejected its nonfinite derivative as AST error
10. Native validation remains unchanged.

An additional diagnostic showed why scoring the expression once is insufficient:
IEEE arithmetic can make a nested singularity look finite, as in `1/(x/0)`.
The independent CPU verifier rejects this undefined real arithmetic, but waiting
until verification wastes fitting work and lets it disrupt a batch earlier.

The search adapter now recognizes explicit zero divisors, including signed zero,
and records those candidates as `invalid_ast`, with reason `literal_zero_divisor`.
It excludes them before GPU fitting, reports `invalid_candidate_asts` separately,
and assigns them zero GPU evaluations. A zero coefficient start or a state-valued
denominator does not trigger rejection. A toggle is rejected by this check only
when all choices are invalid. Ordinary candidates still use native LM.

The defensive Python derivative change also preserves literal-zero division
instead of evaluating it on the host. It does not relax native AST validation.
No C99 API, kernel shape, native specialization, or native binary changed.

## Validation and retained intermediate failures

- 24 derivative/template unit tests pass, including zero and negative-zero
  denominators. Nine packing/domain tests and five existing service tests pass
  in rack1's existing Python environment. mac1 lacks `nltk`; no dependency was
  installed to run the service tests.
- The original failed fitting request completes in the isolated `runtime-v5`:
  two rejected ASTs, 20 LM starts, 307 actual coefficient evaluations, and nine
  finite candidate results independently replayed with CPU FP64 at 128 steps.
  The diagnostic reports `work_complete=true` and all replay checks pass.
  Measured adapter wall time is 0.537 seconds. This uses a separate 30-second
  diagnostic drain budget; it is not a paired speedup versus the aborted search.
- Version 2 exposed native AST rejection after the first Python-only fix.
  Version 3 exposed a scoring-slot decoding issue in the new domain check.
  Version 4 exposed finite GPU scores for nested undefined arithmetic. Those
  attempts and logs remain under the supplement's incident and remote regression
  directories. Version 5 is the validated, frozen supplement runtime.
- All 36 benchmark case/policy API preparations pass on the repaired snapshot.
  That is admission validation, not evidence of benchmark recovery.

Detailed validation is in the supplement's `repair-validation.json`,
`repair-deployment-v5.json`, and `incident/` directory. The deployment receipt
checks that native binaries equal the originals on each host.

## Experiment boundary and follow-ups

The original random cohort is not rewritten or silently resumed with new code.
The benchmark supplement uses a separate runtime and records this change; it
cannot establish a paired speedup over the original random cohort. Each fixed
benchmark policy retains independent inputs, budget, and held-out ledger.

Next: distinguish candidate-domain errors from unsafe device failures throughout
the controller, retain exception tracebacks instead of only their messages, and
resume explicitly reconciled unstarted policy blocks in separate attempts.
Do not broadly swallow native AST errors or relax CPU verification. Extend
domain checks beyond explicit zero divisors only with sound semantics and
candidate-level accounting.
