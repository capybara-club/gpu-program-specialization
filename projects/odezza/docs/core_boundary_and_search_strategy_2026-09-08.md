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

# Core boundary review and a stronger search strategy

Reviewed 2026-09-08. This is a source/interface review and proposed architecture,
not a new throughput benchmark or a certification of every kernel. No kernel,
search default, running experiment, or deployed binary was changed for this review.

## Assessment

Odezza's main C scoring interface remains substantially clean. Its distinctive
capability is inexpensive evaluation of many specialized systems and parameter
configurations through a reusable compiled shape. Search should exploit that
capability through a planner, rather than add family or GP concepts to the kernel.

The strongest immediate opportunity is better toggle packing and survivor
retention. It is not simply increasing every Cartesian product. Spend module
loads on useful alternatives, retain the evidence needed to learn from them,
and measure verified recovery per unit time as well as evaluator throughput.

No finite-data system can promise recovery of arbitrary dynamics. Unobserved
states, unknown inputs, unexcited regimes and nonidentifiability can leave multiple
models consistent with observations. Return competing models and diagnostics
when that happens. Structural identifiability is a property of the model and
observations, not something brute-force evaluation alone resolves.
[Identifiability and observability review](https://arxiv.org/abs/1812.04525).

## What is clean today

- `src/odezza.h`: immutable handle shape, fixed RHSs, caller-owned context and
  buffers, candidate RHS bytecode, runtime trajectory layout, coefficient banks
  or sampled descriptors, explicit output layout. It has no GP, family,
  stagnation, island or widening options. Each candidate can supply multiple
  unfixed derivatives even though the current search service handles one unknown.
- `src/o_generate_scoring_cuda.c`: grid Y selects the candidate system; grid X
  and lane select a configuration. Low bits choose toggle leaves and higher bits
  choose a bank. The kernel implements RK4, observation handling, coefficient
  loading/transformation and MSE. Those are numerical operations, not search policy.
- `src/o_scoring_pipeline.c`: CPU specialization workers prepare reusable CUBIN
  slots; the caller thread under its caller-owned CUDA context loads, launches,
  observes completion events and unloads. Separate per-device owners belong in
  the runtime. Current successful execution does not use a whole-context sync.
- `src/o_sass.h`: instruction encoding and scheduling fields, separated from AST
  assembly, register allocation and CUDA/module ownership. Fixed conservative
  stalls are a performance tuning topic, not a reason to move search into SASS.
- `src/o_philox.h` and `src/o_score_reduce.h`: reusable numerical helpers with
  explicit lifecycle and layout contracts. Shared sample descriptors reproduce
  the scorer's scale/shift/clamp behavior during winner gathering. Normal draws
  should retain gathered coefficients when portable bitwise replay matters.

The main scorer, pipeline, pool, reducer and public header were compared by hash
with both the local trial copies and the live rohini trial: the five checked
files match. Embedded CUDA sources passed both repository consistency checks.
That does not prove the entire legacy static archive matches current sources.

## Findings and boundaries to strengthen

### 1. Expose existing per-permutation retention in the search adapter

`src/o_score_reduce.h` already supports system, system/permutation and global
groups. `scratch/robust_search_trial/runtime/run_search.py` limits grammar
reduction to candidate top-1, and `runtime/o_run.c` selects
`ODEZZA_SCORE_BY_SYSTEM`. Consequently a toggle template retains one winning
configuration across its permutations. Increasing toggle width without changing
retention can discard promising concrete structures and coefficient basins.

Expose system/permutation top-k through the execution protocol; budget a small
number of coefficient starts per concrete variant. Update buffer sizing,
report decoding and CPU/GPU parity together. This requires adapter work, not a
new scoring shape. Memberships, family floors and parent selection remain host
operations over the returned candidate IDs.

### 2. The current leaf planner is too narrow

`scratch/robust_search_trial/leaf_toggles.py` and `runtime/o_leaf_toggles.h` select
one leaf using ordinal modulo leaf count, then select two/four alternatives from
a rotated state/parameter domain. This is reproducible but is neither general
multi-site exploration nor a guarantee of uniformly covering assignments.

Build a planner with two explicit modes:

1. Exact packing: group already-requested candidates with compatible operator
   topology and coefficient-slot semantics. Shared selector bits at every
   differing leaf enumerate exactly those two/four candidates.
2. Neighborhood expansion: assign independent selectors to selected sites when
   their Cartesian product is intentionally requested. Rotate/cycle the sites
   and domains; measure coverage rather than assuming it from random ordinals.

The LM planner already demonstrates correlated multi-leaf packing in
`lm_toggle/planner.py`. It also explicitly crosses the union of starts with all
member candidates, which can add coefficient fits; do not describe that as exact
work equivalence unless the supplied banks/starts match.

Core toggles select direct leaves. Changing an operator or subtree topology
still emits another AST. That restriction is a useful specialization boundary.

### 3. LM has a service-specific generator variant

The live trial's `runtime/odezza/lm_template.py` differs from
`python/odezza/lm_template.py`. Its `service_mode` changes the launch signature,
bounds, maximum step, masking, stopping tolerance, target and evaluation counters.
The trial manifest already labels this `bounded-observed-counted-lm-v1`, so this
is not evidence of an untagged ABI collision. But a service flag is the wrong
long-term abstraction for a numerical primitive.

Promote the validated behavior into an explicitly versioned numerical LM
interface. Put observation weights, bounds, stopping controls and counters in
that contract; keep candidate packing, start proposals, family retention and
search budgets in the adapter. Keep legacy numerical behavior as an explicit
compatibility variant where required, rather than silently changing old results.
The current service also explicitly rejects unsupported state/parameter and
shared-memory sizes instead of silently falling back; preserve that behavior.

### 4. Deadline handling needs completed execution chunks

`odezza_scoring_pipeline_run` is synchronous and its public failure contract
requires discarding the run's output. The search worker invokes large prepared
streams; a watchdog at the wall deadline can kill the resident worker and lose
the final command's results. The recent frontend fix classifies that correctly,
but still cannot recover unknown in-flight work or avoid worker recreation.

Add policy-neutral completion chunks and cooperative stopping at a safe host
batch/module boundary. The runtime decides whether to submit another chunk and
drains its completion event. Reports distinguish submitted, completed and
unsubmitted ranges. If a core extension is needed, make it a generic execution
control/report contract, not a GP-generation or stagnation callback. Preserve
unscored ranges in checkpoints. Chunk sizing must be benchmarked so cancellation
responsiveness does not sacrifice steady-state module throughput.

### 5. Operational telemetry and build provenance need a stable surface

The public run report contains counts and total time, while useful module load,
specialization, wait, unload and in-flight statistics are private in
`src/o_odezza_internal.h`. Expose a versioned read-only telemetry snapshot to
the planner without exposing internal tickets, patch sites or worker internals.
CPU wall phase sums are not GPU busy time when work overlaps.

The trial Makefile links explicit core/reducer objects alongside a retained
static archive and copies Python LM sources. Keep frozen experiments, but ship
new experiments with an explicit core artifact/ABI manifest and verified header,
source, generator, specialization and binary identities. Avoid treating an
unchanged archive filename as proof of unchanged effective core code.

Future public C ABI growth should include explicit version/structure-size
negotiation. The current launch's added fields require recompiling callers;
that is documented, but it is not a durable independent-client compatibility
mechanism. Raw device pointers also retain caller-owned allocation/context
preconditions, which is reasonable for this low-level API.

## Proposed search architecture

Follow-up user requirement: the new profile should use state-assignment toggles
from the first screen, with correlated bindings of repeated abstract state roles.
System dimension, observed-state coverage and per-RHS dependency bounds are
separate. A six-state rollout can search an RHS using any subset of at most three
states; enforce that constraint after toggle materialization and during GP too.
See the [dependency design](workbench_design_2026-09-08.md#initial-screen-toggles-and-sparse-rhs-dependencies)
for proposed JSON, assignment coverage and current API limitations. This remains
a planner/search-adapter requirement, not a new numerical kernel responsibility.

| Layer | Owns | Must not own |
|---|---|---|
| Search controller | Priors, grammar/growth, candidate neighborhoods, islands, budget allocation, full/short ranking, experiment selection | CUDA pointer layout, register allocation, patch offsets |
| Work planner/runtime | Exact candidate identity, toggle packing, sampling descriptors, per-device caches, batching, result reduction and replay | Hidden changes to the requested candidate set, verification policy |
| Numerical core | AST validation/specialization, compiled shapes, SASS, loading, numerical scoring/fitting, layout-based reduction | GP families, grammar quotas, parent selection, LLM decisions |

The LLM should provide priors, constraints and useful search revisions when
reports show a reason. It should not be required to schedule every scoring wave.
The automatic controller should keep the GPU supplied while preserving an
exploration budget for structures outside the best current family.

Start with inexpensive broad screens and several independent search populations.
Probe diverse shortlisted candidates over all training trajectories; retain both
short-view and full-view winners. Allocate measured GPU-time budgets to full-span
fitting instead of irreversibly widening every evaluation. Compare rankings of
the same expression/coefficients; a large full/short MSE ratio alone is weak
evidence that short views misrank candidates.

Use role-aware coefficient scales and a mixture of local and wider samples.
Keep raw Philox pools separate from specialized RHSs. Common raw banks are useful
for matched structural comparisons; distinct streams and additional starts are
still needed for exploration. Independent state choices use distinct selector
bits; correlated replacements intentionally reuse bits. All that mapping is
explicit in the planner, not inferred by the kernel from a family name.

## Experiments to prioritize

1. First compare exact prepared candidates with identical coefficient banks,
   explicit versus correlated toggle packing. Require per-variant score/replay
   agreement; measure patch/load, kernel, reduction, download and end-to-end time.
2. Enable per-permutation retention, then test two independent four-way sites.
   At an equal 1,024-configuration budget, compare 1 permutation x 1,024 banks,
   4 x 256 and 16 x 64. Those explore different structure/coefficient allocations,
   so report them as search experiments, not identical work. Track aliases and
   active coefficients; a parameter-slot swap is not automatically a new structure.
3. Test fixed broad screens with bounded full-training reranking against fixed
   screens alone. Freeze case/seed budgets and keep failures. Extend unresolved
   cases under a separate predeclared continuation protocol, counting all work.

Track verified recovery rate, cumulative time-to-recovery, unresolved outcomes,
completed configurations, unique concrete ASTs where actually known, fitting
time, numerical invalids, load amortization and discarded/pending work. Neither
high GPU utilization nor a large configuration counter demonstrates useful search.

## Real-world scope

Random depth-three games are an engineering regression suite. Broader evaluation
needs additive/saturating/rational mechanisms, coupled unknown equations, more
states, unequal units, forcing inputs, measurement maps, noise, missing states,
unknown initial conditions and stiffness, varied separately. Label simplifiable
or trivial generated targets after completion; do not let them dominate the
success headline. Generation rejection rates are part of the dataset definition.

Allow domain-informed proposers alongside general GP: sparse/weak-form methods
when observations support them, mechanistic templates, and learned or LLM
proposals. Weak-form methods avoid relying on pointwise noisy derivatives;
SINDy-PI offers a distinct route for rational/implicit structure. These are
proposal sources, not evidence that they solve arbitrary sparse hidden-state
cases. Final comparison remains against independent trajectory predictions.
[Weak SINDy](https://epubs.siam.org/doi/abs/10.1137/20M1343166),
[SINDy-PI](https://arxiv.org/abs/2004.02322).

For noisy observations, state/noise weighting and appropriate likelihoods should
be explicit numerical objectives. For sparse/hidden observations, avoid treating
unknown intermediate states as measured restart points. Multiple shooting, if
added, must fit those states with continuity constraints. For stiff dynamics,
use a separately validated integrator capability; adding RHSs to an explicit
RK4 benchmark does not establish stiff-solver coverage.

When candidate models disagree outside the observed regime, recommend informative
initial conditions or forcing experiments if the user can acquire more data.
When the data cannot distinguish candidates, report that ambiguity instead of
claiming an exact recovered physical law.
