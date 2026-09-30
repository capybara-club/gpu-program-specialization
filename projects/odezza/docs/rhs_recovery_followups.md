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

# Recovery controller follow-ups after RHS6

Recorded before RHS7. These are priorities, not claims of implemented features.

Latest search-policy note (September 6, after RHS10):
[broad initial search and explicit structural expansion](explicit_search_expansion.md).
Prefer a transparent grammar-based expansion loop before adding a GP controller;
preserve the user's concern about controller complexity and tuning.

1. Include direct simple structures alongside broader compositions. Avoid forcing
   a single sine to emerge by zeroing an overparameterized two-sine model.
2. Separate a plateau/simplification proposal trigger from strict verification
   acceptance. Try bounded coefficient rounding and term removal, then rescore
   and refit; never accept a structural change without numerical checks.
3. Own promotion, per-family diversity, simplification, freezing and sealed final
   testing in one server workflow. Keep final-test observations outside adaptation.
4. After removing redundant parameters, add residual-based LM and measure it
   against native scalar-curvature fitting; work-count savings are not speedups.
5. Deduplicate same-objective frontiers and show dataset/time scope and actionable
   plateau/verification events. Profile controller, admission, queues and native
   execution separately before attributing unaccounted wall time to transport.

RHS6 evidence: ../scratch/recovery6/RESULTS.md. RHS7 will use the existing tested
production API and record controller workarounds; this task does not silently
change production policy during a blind search.

## September 6: staged trial preparation

The isolated [v2 recovery trial](../scratch/recovery_trial/STAGED_WORKFLOW.md)
now integrates native grammar screening, progressively longer training scopes,
bounded refinement, full-data promotion and verification, with a live LLM revision
window and Mac1 CLI. [Measured results](../scratch/recovery_trial/validation/STAGED_RESULTS.md)
include the public RHS8 starter regression and explicitly distinguish it from a
fresh blind benchmark. Production kernels and service remain unchanged. Residual
LM, native family pause/resume, and validation of a 3–5 minute fresh recovery
target remain open.

## September 6: RHS11 coverage and admission lessons

[RHS11](../scratch/recovery11/RESULTS.md) recovered
`0.18*sin(y)+0.47*cos(x*z)+0.21*x*y-0.12*y-0.4*z` after adding quadratic tails.
The first portfolio allowed products inside waves/prefactors but lacked additive
nonlinear terms. Treat additive polynomial support as its own coverage dimension;
more coefficients or nesting alone do not fix that omission. This is a future
expansion-policy item, not a newly implemented automatic controller.

Division-free prepared profiles are now supported and tested (`broad-v3`). The
10,000-candidate campaign cap rejected the first larger revision; show current
admission, proposed addition and remaining capacity before revision compilation.
The cap remains unchanged. Record preparation repairs outside the prepared-entry
clock separately rather than reporting the shorter clock as complete task time.

RHS12 reused that generic quadratic-tail revision without new family construction
and verified in 71.930 seconds from prepared-entry launch; its successful revision
executed through verification in 8.363 seconds. See
[the measured record](../scratch/recovery12/RESULTS.md). This remains an explicit
revision, not an automatic default expansion. Also distinguish the numerical
leaderboard winner from `verification.frozen.candidate`; completion reports now
show the latter alongside verification metrics.

## September 6: pooled constants, partial observations and events

Implemented and deployed to the isolated Rack1 trial: reusable Philox uniform/normal pools, per-family CTA scaling, exact gathered winner replay, hidden-state observations with required complete initial values, and event dependencies for uploads/scoring/reduction. Prepared CSV runs accept blank cells plus `--initial-states`. See [protocol](philox_pool.md) and [validation/performance](../scratch/philox_pool/RESULTS.md). A verified Mac1-to-Rack1 synthetic campaign exercises both sampling families and scoring only x/z while integrating hidden y. Pool caching currently belongs to pipeline cache entries; independent server pool handles and sampled explicit-candidate JSON remain follow-ups.

## RHS13: prefactor plus independent wave, with hidden observations

[RHS13](../scratch/recovery13/RESULTS.md) verified
`0.44*z*cos(x+0.33*y)+0.23*sin(x*y)-0.12*y-0.40*z` while y was entirely unobserved after initialization. The missing reusable portfolio combination was `prefactor * wave(phase) + independent_wave(phase) + tail`. Keep prefactor choice and both phases independent; do not hardcode this recovered coefficient vector. This generic family is a follow-up, not yet installed as a default.

Cold creation of the 512-system scoring pipeline consumed about 52 seconds; warmed scoring of the 59-million-configuration first screen took about 0.45 pipeline seconds per GPU. Prewarming/capacity selection and finer creation-time diagnostics are priorities. The prepared CSV adapter also needs to accept missing state columns when a complete state declaration and initial vectors are provided.

Total time was 14m46s including approval and handoff gaps; recorded search/fitting stages were 157.7 seconds. Explicit bounded adaptive authorization was followed by verification in 206.6 seconds. Keep workflow notifications separate from individual campaign STOPPED events. See the linked report for trial counts, held-out metrics, limitations and exact timing definitions.

## RHS14: first-pass combined portfolio

[RHS14](../scratch/recovery14/RESULTS.md) verified in 65.006 seconds from prepared-entry launch, with hidden y and observations every two time units. The initial `combined-waves-v1` wrapper in `scratch/recovery14/start.py` admitted 8,046 ordered ASTs with four starts apiece; the winning product-of-waves plus quadratic-tail structure required no revision. Its exact 185,815,903 fitting/scoring trials reconcile with 264 native reports.

Preparation/preflight cost 23.5 seconds, versus 41.4 seconds in the campaign. Add preparation sub-timings and avoid unnecessary repeated admission when shrinking budgets. Package the combined portfolio under its own reusable profile identifier: the wrapper currently overrides the request while retaining the base `broad-v3` label, and unmodified `recover.py` has not acquired this portfolio as its default. See the results for cache/capacity comparison limits and verification metrics.

## RHS15: independent rational terms and campaign capacity

[RHS15](../scratch/recovery15/RESULTS.md) verified
`0.38*sin(z+0.26*x)*cos(x*y)+0.21*x/(1+y*y)-0.13*y-0.41*z`
with hidden y observations. The first 9,909-AST portfolio covered wave products
and rational terms, but its product-plus-rational combination restricted the
cosine phase to a state. A generic expansion independently varying both phases,
numerator state, and squared denominator state admitted another 6,561 ASTs and
verified. Treat rational terms as composable additions to existing cores; merely
including division somewhere in the portfolio does not establish coverage.

The recorded clock was 392.229 seconds, starting after initial attachment
inspection; exact file-receipt time was not instrumented. Numerical campaign
stages took 142.790 seconds. The successful expansion campaign took 60.577
seconds including admission. All 397,276,899 coefficient trials reconcile against
604 native reports. Both runs used explicit starts and native curvature, not a
Philox screen. The new portfolio remains a scratch wrapper.

The previously documented 10,000-AST cumulative campaign cap caused another
late revision rejection because the wrapper did not check that only 91 slots
remained. Expose remaining admission capacity and reject oversized finite
revisions before expansion; longer term, support multiple bounded admissions
without forcing a new campaign. A second wrapper error shrank the total budget
without its phase budgets; using prepared_recovery.shrink_budget fixed the
rejected retry. Reuse this helper rather than duplicating budget arithmetic.

The staged workflow's resolved screening iterations were 16, while the request's
separate policy screen_iterations field said 8. Make effective settings explicit
and avoid two apparent controls for one operation. Retain recovery-level
notification state across campaigns so a cancelled intermediate campaign is not
mistaken for overall failure. The blank-y CSV adapter requirement also remains.

## RHS16: eight-state recovery and semantic coverage

[RHS16](../scratch/recovery16/RESULTS.md) verified the eighth RHS of an eight-state
system with all states observed. The generic campaign service/scorer already
supported this dimension; the prepared entry point and stock grammar still
required three states. A direct client and finite portfolio were used without
kernel/service changes. Generalize the prepared interface independently of
portfolio selection so unsupported presets do not imply unsupported backends.

Recorded elapsed was 604.297 seconds, missing the ten-minute target by 4.297
seconds. Search/fitting stages took 248.409 seconds; initial setup took 147.859
seconds; remaining admission/waits/controller gaps took 207.622 seconds. The
478,880,747 trials reconcile with 720 native reports. The final nine-structure
compact campaign verified in 9.024 seconds after two broader campaigns.

The second expansion selected six cores that represented only five phase states
plus a phase-free expression. A manual completion revision added x1 and x6 but
still omitted x7, which the recovered formula required. Derive uncovered roles
from actual semantic choices, not the number of selected expressions. Keep cheap
complete state-role sweeps around promising cores even when retaining a small
beam. The compact campaign included every phase state and a sparse linear
background; it also changed initialization and fitting scope, so its success
does not isolate sparsity or an optimizer speedup. No stock default was changed.

Run sparse and dense backgrounds alongside one another. Wrong or missing phase
terms can be partially absorbed by extra linear coefficients, producing a good
short-prefix approximation that stalls on longer trajectories. Expose exhausted
phase allocations separately from total remaining time; the second campaign
waited after its refinement allocation was spent. Continue to preserve sealed
whole-trajectory tests, per-state verification errors, and recovery-level
notification intent across component campaigns.
