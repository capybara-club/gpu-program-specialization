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

# Prepared six-state hidden-RHS trial

An experimental controller outside Odezza core, built from challenges 27–28.
It accepts the published six-state/two-hidden-state, four-template challenge
profile. It does **not** yet accept arbitrary known subsystems, grammars,
observation layouts or state counts; incompatible profiles are rejected upfront.
No recovered equations are embedded in the search. Only the public template
catalogue, known linear subsystem, damping and coefficient ranges are fixed.

Use the existing numerical environment from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  scratch/recovery20_cpu/.venv/bin/python examples/hidden_rhs_trial/run.py \
  /absolute/path/public.json --out scratch/new_hidden_rhs_run --timeout 900
```

The output directory must be new. `--check-only` validates the supported input
profile without a run. The existing private mac3/rack1 service must be reachable
by `service_trial.client.Client`; this is not a public MCP endpoint.

The controller prepares observed-state proposals, then starts the native GPU
screen and two CPU adjoint proposal fits concurrently. The first screen samples
128 two-term choices for each hidden RHS under four observed-RHS hypotheses,
using 8,192 Philox coefficient rows. It reserves 16 MiB of dedup workspace per
family, avoiding the 1 MiB allocation failure encountered during challenge28.

As soon as the first survivor fits finish, automatic structural-edit rounds
start. Each round tests the public 105-term catalogue at each of eight term
positions with 16,384 coefficient rows, then fits a diverse shortlist on full
trajectories. Edits can change states and operators; they are not limited to
operators on a fixed state support. If auxiliary fits finish first, their
focused pair screens also enter the candidate pool. A sufficiently accurate
candidate triggers coefficient refinement and independent scalar-evaluator
validation on eight reserved trajectories. Successful validation stops remaining
work. CPU process groups are terminated and retained service handles are
canceled/released on exit; shutdown events are recorded.

`controller-events.json` records starts, exits, accurate-candidate detection,
validation and cancellation. Requests, native reports, fitted candidates,
coefficients, full-precision submission and official-holdout predictions remain
in the output directory. Six states are always integrated from their complete
initial values. Screening uses prefixes; final fitting/validation uses T=8.
Official holdout observations are absent in these challenge files, so exported
predictions are not an official holdout score.

Limits and interpretation:

- This is a reusable trial, not the completed production search/MCP API.
- Search is not exhaustive. Initial derivative proposals bias four observed-RHS
  hypotheses; the full public catalogue remains available in structural edits.
- Exploration allows amplitudes outside the final public range and near zero;
  final refinement enforces the public amplitude/frequency bounds.
- A short-rollout score is only a screening objective. It is never reported as
  full-trajectory validation.
- The adaptive controller is separate from core; no kernel or pipeline changes
  are required.
- Checkpoints, restart after service-generation changes and robust continuation
  after arbitrary process failures remain future work.
- The three-second live smoke test and prepared replay use an already-solved
  dataset. Neither is a fresh blind-recovery measurement.

Evidence: `scratch/recovery28_odezza/`, `scratch/recovery28_controller_smoke/`,
`scratch/recovery28_prepared_replay/`.

The initial/focused survivor selector retains eight candidates per family,
interleaves them and then appends additional global winners. The initial fit
budget is 32, enough to cover all four families without thinning their shortlists. `test_selection.py` checks this through the real native
report decoder using the retained challenge28 fixture. This corrects the earlier
first-two-families-only fitting cap; blind and replay coverage remain labeled
separately in the experiment report.

The converter reads returned syntax without algebraic folding: coefficients or
frequencies equal to zero/one must not erase search slots. `test_decode.py`
checks retained reports numerically, including the coefficient-one regression.
The core scorer is unchanged. The final prepared replay is retained separately
from the first replay and the intermediate replay that exposed this defect.

A 16-fit intermediate policy split across all four families lost useful deeper
survivors and entered a slower search path on the retrospective fixture. That
result is preserved rather than conflated with the successful first replay.
The final policy uses 32 initial fits, eight per family. Its measured replay is
`scratch/recovery28_prepared_balanced_replay/`.

Final balanced replay: **66.78s** to recovery, **70.18s** through validation,
MSE 1.1169e-28 on the eight reserved trajectories. This remains retrospective.
A multiprocessing semaphore-cleanup warning occurred when the remaining fit
pool was terminated. No search processes/service jobs remained. Graceful pool
cancellation is an open hardening item; this wrapper is an experimental client.
