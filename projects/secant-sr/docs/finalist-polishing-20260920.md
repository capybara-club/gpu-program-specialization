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

# Fresh-seed finalist-polishing trial

**Completed:** all 336 fits finished without execution errors. The
[September 21 historical comparison](finalists-history-comparison-20260921.md)
finds a substantial gap versus the strongest old search on these same problems
and seeds, and documents unused reserved polishing time in this trial.

This experiment tests a separate final CPU fitting phase while preserving the
unrefined GPU search as the control. It follows the
[completed five-policy experiment](policy-repair-20260920.md), which found modest
gains from coefficient blocks and square/cube mutations but substantial repeat
variability. The CUDA core, specializer and module-loading pipeline are unchanged.

## Candidate retention and fitting

The optional `--finalists 16` switch enables a passive C99 archive across
generations. It has fixed capacity and performs no allocations after creation.
It copies the best winning binding for each retained key, preserving the chosen
states, fixed literals and adjustable-parameter sharing. Adjustable coefficient
values and slot names are excluded from the key. Keys are compared exactly;
hash collisions cannot merge unrelated expressions. The best training SSE is
kept within a key, and a full archive evicts its highest-SSE entry.

This definition of diversity distinguishes **ordered structures, input bindings,
fixed literals and parameter sharing**. It does not merge algebraically equivalent
forms, measure prediction diversity, or guarantee representation of every family.
The 16 entries are distinct under that definition, not a claim of 16 fundamentally
different mechanisms. Models without active adjustable coefficients are not
archived for fitting; the global search winner is always preserved separately.

The archive never changes GP selection, mutation or RNG state. Each archived
model records its generation, GPU training SSE, concrete binding, fitted centers,
parameter identities and replayable AST. All trial arms enable the same archive
so its host overhead is accounted for on both sides.

The optional Python final stage tries the global winner plus up to 16 archive
entries, removing identical resolved-expression bytes. It uses existing SciPy
float64 LM with an analytic Jacobian, at most two starts and 200 function
evaluations per start, and divides remaining time among candidates. All rankings
use **training data only**, with float32 parameters and predictions. Fixed literals
stay fixed and tied coefficients stay tied. The incumbent is retained unless a
candidate improves training error.

The selected expression is replayed with Secant's native CPU evaluator before
acceptance. Native training error must improve; the numerical agreement check
uses the existing 2e-5 scaled-RMSE tolerance. Holdout metrics are only reported
after training selects the expression and do not affect acceptance. Original
search results and the final model have separate provenance records; old GPU
configuration indices are never attached to a changed polished expression.

Final CPU candidate re-scoring and archive selection are part of the treatment,
so this trial does not isolate LM from those other changes. Reports distinguish
whether fitting improved the selected candidate's own initial training error.
CPU fitting evaluations are not added to the GPU configuration counter.

## Trial design and budget

The same 21 diagnostic problems use **fresh official seeds 860 and 5390**, replacing
23654 and 15795. This changes both the official data split and search seed. Two
identical-seed repeats expose search variability. The case set remains selected
from previous diagnostic outcomes, so it is not an unbiased full-SRBench sample.
Two short preflight cases test integration without changing the frozen policies.

**42 cases × two repeats × four policies = 336 fits.** All use 8,192 population,
64 banks × 64 permutations, four constant slots, 31 logical nodes and native
pack-8 CUDA scoring. In-search coefficient fitting is disabled.

| Policy | GPU search budget | Final CPU fitting | Power mutation probability |
|---|---:|---|---:|
| baseline | 60 s | Off | 0 |
| power | 60 s | Off | 0.25 |
| polish | 50 s | Up to 10 s within 60 s total | 0 |
| power-polish | 50 s | Up to 10 s within 60 s total | 0.25 |

The deadline starts before search dispatch and covers data validation, process
startup, search, archive export, final fitting and native replay. GPU search
limits are cooperative at generation boundaries; fitting checks its deadline
at objective evaluation boundaries. Actual end-to-end runtime and budget overrun
are recorded. Ten seconds is reserved for polishing; it cannot consume unused
early-search time beyond that maximum. Early training-threshold success skips
polishing. No extra unreported optimization time is added to a fit.
One-time worker imports and per-case warmups are outside the measured fits and
remain included in campaign wall time.

The campaign uses **rack1's two RTX 5080s** and its existing NumPy/SciPy environment,
with one CPU linear algebra thread per worker. Ada is left available; this design
keeps the fitting software and GPU class matched without installing dependencies.
Expect roughly two to three hours depending on early stopping. Matched arms for
a case stay on the same GPU, with rotated execution order. Manifests, prepared
dataset hashes, source and binary hashes, dependency versions and warmups are
retained. Failures stop their worker with evidence; there is no silent retry or
backend fallback.

- mac1 run directory: `scratch/finalists-20260920/`.
- rack1 experiment: `/home/cdurham/experiments/secant-finalists-20260920`.
- mac1 tmux workers: `secant-finalists-0`, `secant-finalists-1`.
- Background Telegram notifier: `secant-finalists-notify`; one start and one
  finish message, with failure messages if needed. mac1 must remain online.

The campaign is not a port of historical GP/LM policy, and numerical recovery
at held-out R² > .999 is not symbolic equivalence. Evaluate paired gains/losses,
strict 1e-6 NMSE success, timing and repeat consistency before changing defaults.

## Validation

- Ten CPU test targets pass normally and under address/undefined sanitizers.
- Archive tests cover owned snapshots, exact binding replay, coefficient-value
  deduplication, state-binding diversity, fixed capacity and preserved sharing.
- Fixed-generation CPU comparisons preserve winner bytes, scores and work counts
  when passive collection is enabled.
- Final-stage tests alter only holdout targets and obtain the same fitted model;
  they also check expired budgets, invalid replay and separated provenance.
- Campaign tests verify four arms, matched budgets, resume identities and
  stopping on failures.
- Initial CUDA preflight exercised all four policies. A polished result selected
  archive entry 2 and passed native replay; fitting took 0.35 s. This validates
  integration, not general search-quality improvement.
- A second, harder case passed all four paths and both final native audits.
  The 12-second search-only preflights overran by 0.68–0.88 seconds at cooperative
  boundaries; this is retained explicitly, not reported as exactly 12 seconds.

## Remaining work

2. Finish and analyze this fresh-seed comparison before full SRBench.
3. Profile achieved occupancy on real populations; unchanged by this experiment.
4. Determine whether final polishing and archive selection improve reliability
   within the same time budget. Defaults remain opt-in; GPU LM stays deferred.
