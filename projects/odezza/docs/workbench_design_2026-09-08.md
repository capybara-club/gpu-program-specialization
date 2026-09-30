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

# Odezza workbench: proposed web interface

Design proposal, 2026-09-08. No service, search policy, numerical kernel, running
experiment, or deployment is changed. The accompanying interactive concept uses
illustrative data and does not connect to the machines.

## Product direction

Build a shared experimental workspace for the human, LLM and automatic search
controller. Its primary object is a **problem with competing candidate models**.
Machines, grammars and runs are linked evidence about how those models were found.
The browser and an LLM should use the same planning, submission and report APIs.
Closing the browser must not stop a search.

The four working views are Runs, Fits, Grammars and Performance. Open on active
runs during development; a saved run opens directly on its fit evidence. Avoid a
permanent wall of GPU gauges or an embedded terminal as the main experience.

## Precedents and what to borrow

| Product | Relevant established behavior | Odezza adaptation |
|---|---|---|
| Certara Pirana | Model editing and management, list/tree histories, local/cluster execution, intermediate output and fit reports | A model/run workbench with explicit ancestry, experiment notes and per-host work |
| Monolix | Individual predicted curves over observations, prediction-vs-observation and residual diagnostics; simulation-based visual predictive checks | A fit workspace where aggregate loss leads to trajectory, state and regime-specific evidence |
| Phoenix NLME | Graphical model editor and model library alongside textual PML; automatic diagnostics | A JSON source with structured assistance and a readable preview, rather than a second incompatible grammar language |
| MATLAB System Identification | Import/preprocess data, estimate several models, compare responses and validate against independent data | Persistent competing models, distinct fitting/validation data and overlays; later input-response diagnostics |

Sources: [Pirana overview](https://onlinehelp.certara.com/pirana/25.7/Pirana_User_Guide/Pirana_Overview/Pirana_Overview.htm),
[Monolix plots](https://monolixsuite.slp-software.com/monolix/2024R1/generating-and-exporting-plots),
[Monolix VPC](https://monolixsuite.slp-software.com/monolix/2024R1/visual-predictive-check),
[Phoenix NLME](https://www.certara.com/software/phoenix-nlme/),
[System Identification app](https://www.mathworks.com/help/ident/ref/systemidentification-app.html).

These are interface precedents, not a claim that Odezza currently implements
population pharmacometrics, regulatory workflows, or their statistical models.
In particular, a collection of surviving AST curves is model disagreement, not a
calibrated confidence/prediction interval or a pharmacometric VPC. VPCs require
an explicit statistical model and repeated simulated datasets under the study design.

## 1. Runs: what is happening, and where?

One row per host, expandable to services and GPU devices. Include rack1, rohini,
ada, mac1 and mac3 as configured hosts; discover their actual capabilities rather
than assigning GPUs or inventing health from a machine name. mac1 can run the
workbench and registry while the GPU servers keep their current execution roles.

Show heartbeat age; online/stale/unreachable state; active run and service
variant; stage; elapsed time; remaining execution budget; last completed work;
and last improvement measured on a fixed objective. Expand for GPU utilization,
VRAM, CPU preparation, device ownership and effective core/controller build IDs.
Utilization is not occupancy. Occupancy belongs in a measured profiler report.
Loss values from different datasets or scoring views must not be ranked together.

Separate queue wait, preparation, scoring, fitting and verification. Show a budget
countdown, not an asserted time-to-solution. For a finite batch, an ETA can be a
range based on observed completion rates, explicitly conditional on the budget.
An unavailable heartbeat must not turn an active run into a failed or successful run.

A selected run exposes its public problem, grammar revision, resolved plan,
top candidates, progress events and checkpoint/artifact references. A revision
creates a recorded branch or applies at a reported safe boundary. Never silently
edit the meaning of already-scored work. Where exact resume or pause is unsupported,
show that limitation; do not rename cancellation as pause.

## 2. Fits: the equation and its evidence

The left side is a shortlist of distinct models. Use a compact equation renderer
with normal precedence and named coefficients. Keep exact AST and full-precision
coefficients available for replay; display rounding must not change evaluation.
Show parameter count and structural size, training and selection-validation
loss, and verification state. Do not equate a trajectory threshold with unique
recovery of the physical law.

The main area shows observations as points and integrated predictions as lines,
faceted by state with trajectory and data-view selectors. Indicate the exact
intervals/samples used for the cheap screen. Hidden states get predicted curves
labelled unobserved, without invented observation points or measured-state errors.
Units, any normalization and initial conditions remain inspectable.

Below the overlay, show error by time, state and trajectory. Let users inspect
the worst trajectory directly. Add observation-vs-prediction plots and residuals
against time/prediction. Parameter bound hits, alternative fitted basins and
integration-step agreement are useful existing or incremental diagnostics.

When inputs and an appropriate prediction/noise model exist, add residual/input
correlations and relevant industrial response views. MATLAB's whiteness and
independence diagnostics refer to prediction residuals with model-specific
qualifications; do not apply their pass/fail rules blindly to Odezza's free
rollout errors, sparse samples or autonomous systems.
[MATLAB residual analysis](https://www.mathworks.com/help/ident/ug/what-is-residual-analysis.html).

A loss-versus-complexity frontier and a comparison of the same candidates on short
and full training views help diagnose search decisions. Candidate ancestry is
available on selection, not rendered as a million-node graph. Group summaries
show coverage, invalid fraction, retention and best/quantile losses on the same
objective. Tags can overlap: their counts are not additive, and groups absent
from retained top-k cannot be declared unexplored or poor-performing without
population-level evidence. Quantiles of survivors must be labelled as such.

Selection validation may guide search. The sealed final test stays out of the
interactive feedback loop until its declared terminal evaluation. Private
synthetic target equations and generation seeds remain outside the solver-facing
catalog and LLM tools; post-run truth audits are separate.

## 3. Grammars: files first, with a useful preview

Yes: an LLM can add a JSON file and the interface should discover it. Use one
configured grammar directory on mac1, initially `grammars/` under the repository.
The path is proposed, not created or deployed by this design. Keep UI metadata
in the registry so existing strict grammar parsers do not receive unknown fields.

Workflow:

1. A file is atomically saved by an editor, the user, or an LLM.
2. The registry validates the document with its declared schema/compiler adapter.
   An invalid revision is visible with diagnostics; it cannot replace the last
   valid runnable revision. Watch events are backed by reconciliation scans.
3. The UI shows its filename, format, revision hash, modified time and parse/
   compile status. Discovering a file does not launch a search or allocate CUDA.
4. The user selects a public problem and grammar. Server preflight resolves the
   plan for a chosen backend and returns its constraints and estimated work.
5. Submission pins immutable copies/hashes of problem, grammar, plan and runtime
   identity. Later edits create new revisions; current runs retain their snapshot.

Keep the current grammar syntax. `examples/grammar_game/depth3.json` is a
`odezza.grammar_game.v1` **game configuration** including generation and trajectory
settings, not the same document as a fully resolved native search grammar. Label
formats explicitly and reuse the existing import/compiler path. The public
search projection must not expose private generation seeds or target recipes.
Separate initial grammar, permitted GP growth, sampling and execution settings.

The editor has a JSON source pane and a structured preview: available operators,
states, depth/node/parameter limits, constraints and small rendered samples.
Schema completion and field controls edit the same source. Preserve unsupported
extensions or show them read-only rather than dropping them during a round trip.
Use revision checks to prevent browser saves overwriting concurrent LLM edits.

Preflight reports accepted/rejected samples with reasons; unreachable productions;
current backend limits; toggle feasibility; and AST/template/configuration counts
with exact, estimated, upper-bound or unknown labels. It must be bounded and
cancellable: fully enumerating a huge grammar just to populate a UI is not useful.
Finite pilot samples do not establish exhaustive grammar coverage.

The toggle preview should show selected leaf sites and coefficient-slot aliases,
shared versus independent selector bits, materialized example variants, bank
allocation and retention. Four correlated alternatives and two independent
four-way sites are different requests. Show planned templates, permutations and
coefficients separately, then report actual distinct structures where known.
Some of this planner/retention behavior is proposed, not available in the current
one-site trial; capability discovery must make that distinction visible.

From a fit, “Prepare a neighborhood” creates a reviewable draft with parent IDs,
intended structural changes and a preflight. An LLM can propose the same draft
through the same API. Automatic controllers can submit within a previously
authorized experiment envelope without a click for every wave.

### Initial-screen toggles and sparse RHS dependencies

User requirement, 2026-09-08: state-assignment toggles are a first-screen priority,
not merely a neighborhood optimization after fitting. The proposed new search
profile should enable them from the initial structural screen; explicit no-toggle
benchmarks remain available. Record actual packing and cases without compatible
variants. This document does not change defaults in the current services.

Keep three quantities separate: total simulated states, states observed in the
data, and the states each unknown RHS is allowed to mention. A six-state system
may have an unknown derivative depending on only three states, including or
excluding its own state. Known coupled derivatives still use their complete
state vectors, and all six states remain integrated. Sparse direct RHS dependency
does not imply that the full trajectories are independent of the other states.

The current structural grammar already admits ASTs mentioning fewer than all
declared states. Its `structures.grammar` interface has no allowed-state or
distinct-state cap, and `leaf_toggles` currently selects alternatives from the
whole state domain. The separate explicit CFG frontend can name a chosen subset
in its leaf productions, but is not the same as automatic dependency discovery
in the structural-search path. The proposal below is new API design, not syntax
accepted by the current service:

```json
{
  "rhs_dependencies": {
    "x2": {
      "allowed_states": ["x0", "x1", "x2", "x3", "x4", "x5"],
      "max_distinct_states": 3
    }
  }
}
```

Here any subset of up to three states is legal. Naming only `["x0", "x3", "x5"]`
instead limits the admissible domain to those states without requiring all three.
An optional `min_distinct_states` can express exactly three when set to three.
The default minimum is zero unless another explicit grammar rule requires state
use. Repeated occurrences count once. These are syntactic state-use constraints;
algebraic cancellation or a fitted zero coefficient can reduce effective dependence.
A hard cap is a user constraint. When sparsity is merely a suspicion, prefer sparse
models while retaining a broader search allocation instead of imposing a hidden cap.

Enumerate abstract expression skeletons with a bounded number of distinct state
roles. For example `a*sin(u*v) + b*u*w` has three roles, even though `u` appears
twice. Preserve every occurrence of a role under a binding. For six physical
states and three distinct roles there are 20 possible unordered three-state sets,
and 120 injective ordered role assignments before expression symmetries or
equivalences. Searching only 20 bindings would miss different placements of the
same states in an asymmetric expression. Also include skeletons using zero, one
and two roles for an at-most-three constraint when otherwise grammar-compatible.

Pack compatible ordered assignments into groups of four or two. One shared
selector across all affected leaves selects a complete binding; 120 assignments
can form 30 four-way packs for one three-role skeleton. This is a work-layout
count, not a promised speedup or a guarantee of 120 algebraically distinct RHSs.
It uses existing direct-leaf selectors and requires planner work, not a new
kernel operator. Compatible packs also need matching coefficient-slot semantics.

Independent site toggles remain useful for changing the equality pattern of
roles, replacing states by parameters and exploring additional neighborhoods.
Every materialized variant must obey the same RHS dependency constraint. Reject
or avoid invalid combinations before evaluation; they cannot count as accepted
population. Apply the constraint to baselines, fresh generation, toggles, GP
mutation/crossover, imported parents and checkpoint compatibility. Parent mutation
may change the chosen subset while remaining inside the declared domain and cap.

Preflight and reports should expose role count, binding coverage, actual distinct
state masks, per-permutation retention, duplicate/invalid bindings and coefficient
trials per variant. In the workbench, add per-RHS controls for allowed states and
the dependency cap, then show which subsets were evaluated and survived. A state
usage view must distinguish syntactic inclusion from inferred causal relevance.
For larger systems, report sampled assignment coverage; do not promise exhaustive
coverage of a combinatorially growing binding space. Preserve a coefficient budget
per concrete variant and measure both equal-work throughput and recovery quality.

## 4. Performance: make throughput explainable

Two distinct experiment types:

- Prepared-work comparison: pin explicit AST stream, coefficient banks or pool
  mapping, trajectories, integrator, output/retention contract and work identity.
  Compare explicit versus packed execution or different core builds/devices.
- Recovery comparison: pin public cases, policy versions, independent search
  seeds and total budgets. Compare success rate and cumulative time-to-criterion,
  keeping unresolved, failed and interrupted outcomes.

Use CPU and GPU timeline lanes on a common axis for generation, packing,
specialization, loading, scoring, reduction, fitting and transfer. Asynchronous
phases overlap: do not stack their durations as a purported wall-time breakdown.
Include cold preparation, warm runs, queue time and replayed/discarded work.
Report completed work independently from planned/submitted work and mark
incomplete accounting. Use CUDA event timing for numerical GPU work when available;
use trace correlation IDs, not invented precision, for cross-clock alignment.

The central developer question should be answerable in one comparison:
“With these same candidates and banks, did toggles reduce module work, and did
the saved time survive reduction and fitting?” In recovery experiments ask
whether that translated into faster verification across all outcomes.

## Backend boundary and existing foundations

Suggested deployment:

`browser / LLM -> workbench API on mac1 -> existing per-host search services`

The workbench maintains a small indexed metadata database and durable artifacts;
large candidate/score streams stay on workers. Poll compact status and event
pages initially; a workbench event stream can update connected browsers. Use
pagination and lazy trajectory loading. A retained candidate can reconstruct
its AST, toggle permutation and sampled coefficients through a stable replay
record; normal replay uses retained gathered values where required for exactness.

Existing trial endpoints already cover `/health`, `/searches/plan`, `/searches`,
`/searches/{id}`, revisions/cancellation, datasets/problems, archives/checkpoints
and cursor-based campaign events. Their scopes and schemas differ: normalize
them through explicit adapters instead of pretending every service has the same
features. The workbench needs a global run index (including imported tmux batch
manifests), a grammar registry, host/device telemetry, common diagnostics and
versioned runtime reports. A tmux session alone is not an authoritative run state.

For the local trial, keep services on their current loopback boundaries and let
the mac1 backend own configured SSH tunnels. Browser clients do not hold SSH or
Telegram credentials. Remote/multi-user hosting is a separate deployment scope.
One existing notification path should consume terminal run events with an
idempotency key, so reconnects do not send duplicate Telegram messages.

GPU telemetry collectors and search adapters live outside `src/`. The core may
gain a versioned numerical timing/completion report, as already recommended in
the core review. It should never need a grammar editor concept, dashboard ID,
family name, browser session, or database connection. The UI must work with
frozen older runs and show unsupported fields as unavailable.

## Build order and acceptance

1. Read-only Runs + Fits over existing artifacts/services. Import old Odezza and
   PySR runs. Make stalled work, error locations and trajectory mismatch easy to
   inspect. Show metric definitions and exact objective identity.
2. File-backed grammar registry/editor + server preflight + explicit submission.
   Prove an LLM-written valid file appears, a running job pins its revision, and
   concurrent edits do not lose changes. Reuse existing grammar validation.
3. Toggle/retention inspector + prepared-work comparisons. Connect this to the
   per-permutation retention and exact-packing priorities, without changing core
   numerical behavior for UI convenience.
4. Rich fit diagnostics, population/noise-aware uncertainty where implemented,
   experiment design and domain-specific views based on actual user needs.

The first usable slice should answer: what is each machine doing; why did this
run stop; where does this model fail; what changed between runs; and what exact
work would this grammar revision submit?
