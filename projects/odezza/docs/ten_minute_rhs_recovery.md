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

# A ten-minute RHS recovery loop

Later design direction, recorded after RHS10:
[broad initial search and explicit structural expansion](explicit_search_expansion.md).
This captures the preference for an explicit expansion policy before introducing
a general genetic-programming controller; it is not an implemented API extension.

Status: the first campaign implementation now exists; see
[the implemented API and limits](../scratch/grammar_search/CAMPAIGNS.md). Native
batched scalar-curvature fitting, finite weighted families, multi-GPU scheduling,
progress, additive revisions, cancellation and numerical verification are
implemented. The broader policies and illustrative fields below remain a roadmap;
they are not all accepted API fields. The ten-minute blind-recovery target remains
unproven. This extends unified_search_design.md and preserves the working scorer.

## Objective and evidence

Target: recover and independently verify a comparably difficult noiseless RHS
within 600 seconds, with an LLM choosing search directions and Odezza executing
the numerical search. The end-to-end clock starts when the problem becomes
available to the controller, including LLM latency, planning, compilation,
transport, fitting, and verification. The service gets the remaining deadline.

RHS5 is a regression case, not sufficient evidence for this target. The
successful final model was proposed after substantial earlier work and the
user's exponent-depth clarification. Replaying it from its favorable starting
coefficients is an optimizer test, not blind recovery.

Observed second-attempt work:

| Stage | Completed jobs | Trials | Wall span | Summed worker round trips |
|---|---:|---:|---:|---:|
| Weighted exponent search and fitting | 848 | 44,902,826 | 477.9 s | 250.8 s |
| Partial outer sine expansion | 609 | 9,326,016 | 358.1 s | 70.2 s |
| Quadratic containing models and partial rollout fitting | 396 | 967,500 | 159.1 s | 41.9 s |

The entire second attempt comprised 18,572 jobs and 619,801,062 coefficient
trials over 104.7 minutes. Worker times sum across two GPUs; stages overlapped.
They include worker orchestration, not isolated kernel time. Subtracting the
last two columns does not measure Python, network, or GPU-idle time. Those
components need proper instrumentation.

One start in the final containing-model stage reached local-flow MSE 3.98e-11
after four curvature/proposal rounds, 6.73 seconds after that stage's first job.
This supports prioritizing structural admission and automatic fitting, but
does not prove an end-to-end ten-minute result.

Evidence: ../scratch/recovery5_v2/RESULTS.md and
../scratch/recovery5_v2/server-accounting.json.

## Search rules to follow

The main RHS5 failure was overly focused families, with a secondary depth-
accounting problem. The target does not need nested transcendental functions:
both sin and exp have function-nesting depth one. Under leaf-depth-zero counting
with coefficient scaling excluded, the exponential argument has depth two,
exp(argument) has depth three, and multiplying by y makes that term depth four.
Thus a total-tree depth-three cutoff can still miss it, even with all operators
enabled. Increasing function nesting alone would not admit the missing outer
compositions. The hint's argument depth must not become the entire RHS budget.

1. **Start from a reusable composition portfolio.** Do not spend the first
   several minutes inventing one large bespoke family. Independently vary the
   outer prefactor, exponential argument, sine argument, and additive terms.
   A complete argument grammar inside the wrong outer template is insufficient.
2. **Separate coefficient placement from structural depth.** Weighted edges
   represent amplitudes, relative weights, and function-input scales. Report
   state-bearing depth, function nesting, actual bytecode nodes, and active
   parameter count separately. Define whether unary functions consume each
   depth budget. Do not silently reinterpret the user's convention.
3. **Fit containing models and simplify them.** For example, fit
   `a*x*x+b*y*y+c*z*z+d*x*y+e*x*z+f*y*z` inside an exponential, then propose
   dropping small coefficients and refit. This can discover several sparse
   supports without a separate poorly initialized search for each support.
   Removing a term is a tested model change; coefficients near zero do not
   establish that it is absent. Also search smaller direct structures: a dense
   containing model can be ill-conditioned or exceed parameter capacity.
4. **Cover interactions between choices.** Search `state*exp(Q)`, `sin(P)`,
   `sin(P)*exp(Q)`, and `sin(P*exp(Q))`; permit products inside P and Q and sums
   of nonlinear terms. Vary more than one hole jointly in part of the budget.
   Alternating one-hole improvements alone can miss a jointly good combination.
   These are portfolio examples, not a promise to exhaust arbitrary RHSs.
5. **Give fitting a budget before treating a structure as bad.** A poor random
   row is not a poor structure. Use multiple starts, automatic active-parameter
   detection, per-family fitting floors, and exploration/rescue slots. Size the
   fitted cohort to the time budget. If millions of structures only receive a
   cheap screen, report them as screened, not adequately fitted or disproved.
6. **Change structure promptly on a plateau.** Spend part of each round on new
   compositions and simpler alternatives. Increasing coefficient draws on the
   same inadequate grammar must not be the default response to stagnation.
7. **Fit short observed-state flows; certify full trajectories.** For noiseless
   data, all training intervals can supply independent local flows, exposing
   late-time states without long-rollout conditioning. Then promote candidates
   to uninterrupted training and validation. Keep rankings objective-specific.
   For noisy data, restarting exactly from noisy observations needs a different
   formulation and a separate benchmark; do not reuse the noiseless policy
   under the same accuracy claim.
8. **Stop and verify automatically.** Near-zero fitting error should trigger
   simplification, numerical convergence checks, and a frozen-formula test.
   Do not keep running 80 configured iterations because the LLM has not polled.
   Hold final test trajectories outside the adaptive controller's reports.

For RHS5, these rules would admit both `y*exp(Q(x,z))` and `sin(P(x,y,z))` with
`P=x+y*z` early. They would not require guessing those exact roles initially:
state permutations and weighted linear/bilinear arguments belong in the
standard portfolio. Whether this actually recovers RHS5 cold in ten minutes
must be measured with the policy frozen before the timed run.

## API changes, in priority order

### P0: a persistent campaign that owns the numerical loop

Add a campaign resource above existing immutable sessions and score/search
requests. One campaign owns family admission, candidate pools, coefficient
starts, fitting, objective promotions, simplification, and a wall-clock budget.
The LLM receives compact progress and submits occasional steering revisions.
It does not create a script, upload a bank, and poll for every optimization step.

Proposed operations:

- `POST /sessions/{id}/campaigns`: start one budgeted multi-GPU search.
- `GET /campaigns/{id}/plan`: compiled coverage, capability checks, resource
  estimates, and the exact immutable search revision.
- `GET /campaigns/{id}/events?after={cursor}`: bounded progress deltas.
- `POST /campaigns/{id}/revisions`: add families, expand selected holes, adjust
  allocation, or promote retained model references. Revisions preserve lineage.
- `POST /campaigns/{id}/cancel`: stop new submissions, finish or safely interrupt
  a bounded work chunk, and publish completed versus unfinished work.
- `GET /campaigns/{id}/result`: authoritative final rankings and replay records.

Planning can run automatically; it is not a user approval step. Revisions need
idempotency keys and an expected revision number to avoid applying stale LLM
decisions. Recovery after disconnect must not depend on client-local files.

First implement campaign orchestration on Rack1 around the existing worker.
Move the current scalar-MSE curvature loop beside the scorer, preferably as a
native batched refinement command, so its intermediate banks and score vectors
never make a Mac1 round trip. Do not require a new LM kernel to ship this step.
Merely moving the current scripts onto Rack1 while retaining one durable JSON
job per probe is an intermediate implementation, not the final hot path.

At p active coefficients the current central scalar-MSE Hessian uses
`1+2*p*p` rollout configurations before its proposal search: 513 at p=16.
There are up to 33 more proposal rows per iteration. Detect unused slots and
inactive choices automatically. Separate logical active parameters from
physical kernel slots so numerical savings do not force constant handle churn.

### P0: reusable weighted families and preflight

Promote weighted_arguments.py from a scratch helper into a versioned family
builder/IR. Preserve parameter sharing, separate free parameters from fixed
literals, and keep proposal ranges separate from optimizer hard bounds.
The current scratch fitter's unconditional [-4,4] clamp must not become an
implicit public search constraint. Report bound hits and let policies revise
proposal ranges without changing user-imposed hard bounds.

Add native/library constructors for weighted sums, products, polynomial bases,
function composition, prefactors, and additive RHS assembly. Use the same
operator support for string expressions, grammar generation, and encoded ASTs;
an LLM should not need private bytecode helpers to express supported `exp`.
Keep an explicit-AST escape hatch.

Preflight returns:

- examples covering each declared role and selected joint combinations;
- raw derivations, admitted models, duplicates, and excluded structures where
  exact; bounds/estimates/unknown where exact counts are unavailable;
- separate coverage for each hole and the outer composition;
- reasons for every enforced restriction and any family that cannot compile;
- node, parameter, patch, memory, and nonterminal requirements;
- expected fitting cohort size and time estimates from a small calibration,
  clearly distinguished from measured full-job throughput.

Provide a membership/witness query for known diagnostic structures. Such
queries can catch an accidentally missing `state*exp(sum(products))` family;
passing a few witnesses is not proof of complete language coverage. Do not
ask the blinded controller to supply the hidden target as a witness.

The 96-node repair already shipped. Preflight and appropriate resource buckets
should prevent another middle-of-search limit surprise. Further raising every
limit indiscriminately is not a search strategy.

### P0: progress reports that support decisions

Return a small global frontier plus winners and effort by outer composition,
argument structure, prefactor, and complexity. Include:

- ASTs generated/admitted/scored, distinct count scope, starts tried, fit steps,
  and models that have not yet received the minimum fitting budget;
- per-family floors fulfilled, starvation, and explicit pruning reasons;
- local-flow, full-training, and validation errors in separate fields with
  objective/dataset/integrator IDs;
- stagnation, bound hits, invalid fraction, and convergence status;
- fitted expressions and named constants, with exact winner bytes;
- candidate simplification proposals: near-zero terms, possible parameter
  redundancy, or a small sine argument worth testing as a linear alternative;
- timings for admission/generation, queueing, compilation, patching, module
  loading, execution, reduction, fitting, serialization, storage, and transport,
  with scopes and overlap specified.

Distinguish measured diagnostics from hypotheses. A small sine frequency and
large amplitude suggest a linear alternative; they do not authorize replacing
sin(u) by u without refitting and validating that alternative. Likewise an
all-invalid sampled bank does not prove that no valid parameterization exists.

Keep full probe vectors server-side and optional. Default to bounded
checkpoints, deterministic proposal metadata, compact step summaries, and exact
winner records. Do not erase existing artifacts to implement a new retention
policy. This experiment accumulated about 8.9 GiB locally; the LLM needed only a
small fraction of that information.

### P1: residual fitting, proposal generation, and resource scheduling

After the orchestration slice is measured:

1. Add batched residual/Jacobian or normal-equation accumulation for trajectory
   fitting, suitable damping, and parameter scaling. Finite-difference residual
   Jacobians need p+1 forward or 2p+1 central rollout evaluations instead of the
   current quadratic number of scalar-curvature probes; this is a work-count
   comparison, not a promised speedup. Compare end-to-end costs and fit quality.
   Rank-deficient containing models need robust handling and tested pruning.
2. Generate proposal banks on Rack1. Add the reusable separate Philox kernel
   when materializing large banks warrants it. Stream bounded tiles rather than
   allocating a global billion-row bank. Support explicit rows, fixed values,
   zero/unit atoms, continuous samples, and fitted centers in one clear policy.
3. Let one campaign schedule across GPUs and mix differently sized families
   into batches. Family tags must not force singleton execution. Add a bounded
   handle/data cache keyed by actual execution compatibility, with memory
   limits and visible hit/miss/compile/load statistics.

Philox is useful for throughput and reproducible search, but is not the first
fix for RHS5. Address draws by seed, algorithm version, logical bank/round, row,
and semantic parameter ID; scheduling/GPU assignment must not change them.
Reuse draws across structures only when parameter meanings match, and hold
constants fixed throughout each trajectory evaluation. Refined coefficients
must be retained as exact values: an RNG counter cannot reconstruct an LM
update. Even for sampled winners, a dozen FP32 values are cheap insurance for
replay; bulk banks are what should stay off the wire.

## Proposed LLM-facing request

This is a contract sketch, not JSON accepted by today's service. The named
portfolio and policy must resolve to versioned, inspectable definitions.
The initial session already supplies known equations, training observations,
split policy, and declared operators.

```json
{
  "schema": "odezza.campaign.proposal.v1",
  "request_id": "blind-recovery-run-1",
  "budget": {
    "wall_seconds": 570,
    "verification_reserve_seconds": 60
  },
  "devices": [0, 1],
  "space": {
    "portfolio": "weighted_compositions_v1",
    "depth": {
      "state_bearing": 2,
      "coefficient_edges_count": false,
      "unary_functions_count": true
    },
    "additive_nonlinear_terms": [1, 2, 3],
    "include_containing_models": true,
    "generic_tree_exploration_fraction": 0.15
  },
  "policy": "noiseless_fit_then_expand_v1",
  "report": {
    "interval_seconds": 20,
    "top_models": 12,
    "group_by": ["outer_composition", "prefactor", "argument_structure"],
    "include_probe_scores": false
  }
}
```

The example allows 30 seconds of the total ten minutes for initial problem
inspection; pass the actual remaining budget. The 15% exploration fraction and
other policy allocations are initial tunable choices, not measured optima.
The policy includes multistart/family floors, local-flow fitting, full-rollout
promotion, proposal domains, stopping rules, and frozen-test gates. Return its
resolved contents so a convenient preset cannot hide important constraints.

An LLM intervention should be as small as: “expand the prefactor and second
phase of these retained models; preserve the current envelope family; give
the new joint combinations 25% of the remaining exploration budget.” The
server resolves the candidate IDs and performs the work without regenerated
AST encoders, giant banks, or a new Python optimizer.

## Initial ten-minute operating schedule

This is a proposed budget, not a prediction from measured throughput. Stages
can overlap, and a near-exact candidate jumps directly to verification.

| End-to-end clock | Work |
|---|---|
| 0–30 s | Inspect problem, establish noise/depth conventions, resolve portfolio and preflight |
| 30–150 s | Screen diverse compositions and begin multistart fitting on both GPUs |
| 150–420 s | Expand productive structures and containing models; maintain exploration and family floors |
| 420–540 s | Simplify/refit and compare uninterrupted training/validation trajectories |
| 540–600 s | Freeze candidate and perform independent precision/step-size/final-test checks |

The LLM should make roughly 3–5 substantive steering decisions, not thousands
of micro-submissions. Search continues while it reasons. Time budget and
verification reserve override trial-count ambitions. Return an explicit
budget-exhausted best approximation if verification does not pass.

## Build sequence and acceptance

1. **First vertical slice:** campaign lifecycle, native batched scalar-curvature
   refinement, progress/cancellation, and automatic verification promotion.
   Compare the same fixed candidates/starts against today's controller: winner
   bytes/scores within declared numerical tolerances, complete accounting,
   fewer external calls, measured client wall time and bounded storage.
2. **Coverage slice:** weighted builders, joint composition portfolio, preflight,
   containing-model simplification, and family fitting floors. Exhaustively
   compare tiny populations against an independent enumerator and check
   parameter-sharing/zero-coefficient cases. Test that a deliberately omitted
   family is reported, not silently counted as explored.
3. **Ten-minute evaluation:** freeze these policies and run a blinded problem
   suite with end-to-end LLM timing. Include RHS5 as a public regression, plus
   held-out structures, rotated/mixed states, varied coefficients and starts,
   shifted envelopes, factored/quartic exponents, and genuinely nested functions.
   Separate runs with the depth hint from runs given only the original operators.
   Do not let the controller inspect hidden formulas or final test observations.
4. **Optimize the measured remaining bottleneck:** residual LM, Philox, dataset
   residency, module packing/cache, or reduction. Do not select a new kernel
   topology on uninstrumented end-to-end gaps alone.

Suggested initial release criterion: at least 90% independently verified
recoveries within 600 seconds on a declared, frozen suite of at least 30 held-out
noiseless systems across repeated search seeds. Report failures, success
fraction, time-to-verification distribution, and cold/warm setup separately;
do not compute latency only on successful runs. This is a proposed engineering
target, not a demonstrated guarantee or a guarantee for arbitrary RHSs.

For each run retain the inputs/hints exposed to the LLM, decisions and timestamps,
policy/backend versions, coverage and fitting effort, winner provenance, and
the independent validation. Compare generic defaults against LLM steering to
learn which decisions the LLM actually improves.
