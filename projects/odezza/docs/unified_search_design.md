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

# Unified expression-family search design

September 5, 2026. Proposed architecture, with an executable finite example.
This document unifies the constrained grammar work, existing family-search
tools, and the suggested variation language. It is the design reference for
future authoring and planning work; the current C scoring API remains the
execution contract. Names such as `FamilyIR` and the draft JSON schemas below
are proposals, not installed public APIs.

Implementation update: the [restricted grammar runner](../scratch/grammar_search/RUNNER.md)
now connects actual C enumeration to trajectory scoring with explicit parameter
rows, global/per-family top-k and winner replay. Its state-dependent CUDA fixture
is validated. It does not yet execute FamilyIR bindings or native choice planning;
the finite unified example below remains a separate encoding fixture.

## Decision

An LLM describes **families of complete ODE models**, using templates with
bounded grammar-generated subexpressions, named choices, shared bindings,
parameter policies, constraints, and metadata. A planner turns the admitted
models into concrete work the current scorer can execute.

```text
Python builder / declarative JSON / optional later text syntax
                              |
                           FamilyIR
        bounded grammars + templates + choices + bindings + policies
                              |
             constrained enumeration and candidate admission
                              |
                         ExecutionPlan
       concrete ASTs + native toggle groups + joint constant rows
                              |
                   existing C99 scoring pipeline
                              |
           global / grouped results + exact replay records
                              |
          LLM revises families or promotes models to fitting
```

The grammar supplies structural variety. Named choices coordinate variation.
Constraints define the accepted population. The plan explains which programs,
parameter assignments, and hardware work realize it. None of these requires
putting grammar logic inside the SASS writer.

| Retain | Role in the unified design |
| --- | --- |
| Existing motif and family constructors | Convenient templates, augmentation, term libraries, provenance |
| C99 constrained enumeration | Bounded generation, early pruning, accepted quotas, streaming callbacks |
| Named grammar productions | Open structural search, including bounded recursive nesting |
| Named dimensions from the variation proposal | One labelled decision reused at several sites |
| Explicit expression and parameter bindings | Shared subexpressions and joint multi-RHS models |
| Constant banks and separate RNG generation | Many parameter trials per structure with reproducible policies |
| Tags, descriptors, grouped reports | Feedback about both winners and coverage |
| A separate execution plan | Honest cardinality, native toggle mapping, resource and timing accounting |

## Authoring and meaning

Use a Python builder that produces data, with JSON as the service boundary.
Do not execute submitted Python in the service. A later text syntax can produce
the same IR. The initial API does not need a new parser for mathematical text:
the existing restricted expression parser and grammar frontend are useful
adapters, with their supported subsets stated explicitly.

The semantic expression nodes are `state`, finite FP32 `literal`, `param`,
`op(args)`, `ref(binding)`, and `choose(dimension, labelled_cases)`. A binding
may contain an expression or a bounded `generate(grammar, start, limits)`.
Recursive grammar rules must consume finite node/depth budgets. Bindings form
an acyclic graph; recursive references outside the bounded grammar are errors.
Validate all choice cases, references, operators, and arities, even when a case
is later excluded. Unsupported fields or semantics are errors, not ignored hints.

| Construct | Meaning |
| --- | --- |
| Two occurrences of a grammar nonterminal | Independent expansions |
| Two references to one generated binding | The same selected ordered subtree |
| Two sites using one named choice | The same label at both sites |
| Two occurrences of one parameter ID | One value, shared across the complete ODE system |
| Two independent parameters with the same proposal | Distinct parameter identities and draws |
| Two parameters explicitly sharing a base RNG draw | Correlated proposals; potentially different transforms and fit identities |
| One bound subtree used twice | Structural sharing; numerical common-subexpression elimination is a later compiler decision |

Each family emits a map of unknown state names to RHS expressions. Combining
that map with the known RHSs must define each state exactly once. Parameter IDs
and choice IDs are family-scoped and may be referenced across those RHSs.
Independent RHS spaces are combined by an explicit Cartesian, paired, or joint
construction. Reusing a reaction-rate binding can express coupled equations;
independently expanding two copies cannot promise conservation.

A dimension has labels, not physical bit positions. All sites referencing it
provide exactly those labels. Dimensions and parameters that become unreachable
after structural selection are inactive. Enumeration visits active dimensions
in declared order and gives each inactive dimension one `inactive` value; it
does not multiply models by irrelevant choices. Proven branch equivalence may
remove more duplicates later. Explicit input rows retain their identities;
deduplicating identical active parameter projections is a separate reported
execution optimization that preserves aliases and sample multiplicities.

Grammar generation and choice enumeration have a declared order or sampling
policy. Uniform production selection is not uniform sampling of distinct ASTs.
Exact unique sampling requires its own algorithm; rejection sampling reports
attempts, collisions, admitted counts, and termination status.

## A concrete combined example

The example uses the known `x/y` equations from the conversation, but a small
illustrative `z` family. It is neither the revealed answer nor a discovery run.

```text
Phase -> X | 'mul' X Z | 'cos' Product
Product -> 'mul' X Y
X -> 'x'
Y -> 'y'
Z -> 'z'

phase := bind one expansion of Phase
direction := {forward, reverse}
envelope := {unit, state}

u := choose direction {forward: x, reverse: y}
v := choose direction {forward: y, reverse: x}
wave := sin(phase) * cos(phase)

dz := a * choose envelope {unit: wave, state: z*wave} + b*(u-v)

reject nested_state_envelope:
    phase contains cos AND envelope == state
```

This block explains the semantics; it is not another parser to implement.
The concrete JSON uses explicit nodes. The ordinary Python helpers in
[unified_example.py](../scratch/grammar_search/unified_example.py) construct
those nodes, for example:

```python
wave = op("mul", op("sin", ref("phase")), op("cos", ref("phase")))
u = choose("direction", forward=state("x"), reverse=state("y"))
v = choose("direction", forward=state("y"), reverse=state("x"))
```

Both uses of `phase` select the same expression; both direction sites select
the same label. Nested trigonometry remains admitted for the unit envelope.
The exclusion is an explicit family prior, not a claim that nested envelopes
are physically invalid. Descriptors refer to the selected expression/binding,
not the union of all unselected source branches.

The parameter plan supplies three **paired rows**, reused across structures:

| Row | a | b |
| --- | --- | --- |
| 0 | 0.25 | -0.5 |
| 1 | 0.5 | 0.25 |
| 2 | 1.0 | 0.5 |

This is three parameter vectors, not a nine-element independent product.

| Quantity | Exact value in this fixture |
| --- | --- |
| Grammar expansions for `phase` | 3: `x`, `x*z`, `cos(x*y)` |
| Raw structural assignments | 3 × 2 envelopes × 2 directions = 12 |
| Excluded structural assignments | 2 |
| Admitted distinct ordered models | 10 |
| Native system records | 5, each representing two directions |
| Parameter vectors per model | 3 |
| Planned scoring configurations | 30 |
| Evaluated configurations | 0 |

Resolve the grammar binding and envelope choice on the host. Lower `u` and `v`
to `toggle2(0,x,y)` and `toggle2(0,y,x)` in each remaining program. Both read bit
zero. Each program therefore evaluates exactly the two admitted directions.
Its configuration index is `2*bank_row + direction_bit`. Excluded models never
enter those five records. CUDA block padding and module packing are separate
quantities; five records does not mean five modules or thirty physical threads.

Run from the repository root with the existing Python 3 installation:

```sh
python3 scratch/grammar_search/unified_example.py --output /tmp/odezza-unified-example
python3 -m unittest -v scratch/grammar_search/test_unified_example.py
```

The output directory must be new. It contains a draft `request.json`, the
worked `plan.json` with existing-format AST bytes, FP32 bank bytes and replay
IDs, and a `report.json` explicitly marked `not_evaluated`. There are no invented
scores. Checked-in copies are available as
[request](../scratch/grammar_search/examples/unified/request.json),
[plan](../scratch/grammar_search/examples/unified/plan.json), and
[report](../scratch/grammar_search/examples/unified/report.json).
The script uses a manually exhaustive three-expression grammar oracle;
it does not run NLTK or the C enumerator. It reuses the current family-search AST
encoder and validator. Tests compare all native words with materialized programs
and a separate mathematical oracle, and check exclusions and bank indexing.
This is CPU expression validation, not FP32 GPU rollout or SASS-capacity validation.
Validation on the local Python 3.9.6 environment passed six example tests and
seven existing family-generator tests, with no new dependencies installed.

These artifacts specify a proposed contract. Neither the current grammar CLI nor
the current C99 family-screen executable accepts the unified request/plan yet.
In particular, handing the old runner the ASTs alone would lose the declared
parameter rows and metadata; it would not execute this experiment as specified.

## FamilyIR and ExecutionPlan

Keep two representations with separate versions and hashes:

| FamilyIR: what is being searched | ExecutionPlan: how a finite selection will run |
| --- | --- |
| State order, known and joint unknown RHSs | Encoded postorder programs and supported operator checks |
| Named grammar rules, bindings, expression graph | Resolved variants and native toggle groups |
| Labelled choices and parameter identities | Bit packing and constant-slot maps |
| Constraints, tags, descriptor definitions | Admitted IDs, aliases, parameter rows or materialization tasks |
| Proposal distributions and hard domains | Concrete sample ranges, seeds, transforms and FP32 values |
| Requested structural and numerical budgets | Capacities, batches, device assignments, memory requirements |
| Selection and identity policies | Exact counts/bounds, unsupported features and partial status |

Postorder spans, register allocations, and module boundaries are not logical
identity. Logical `let` bindings may be expanded during postorder lowering.
Numeric CSE must preserve evaluation semantics and account for register liveness;
state-dependent subtrees must be recomputed at each RK4 stage's state.

The planner first validates and admits selected models, then groups only exact
sets into native two/four-leaf toggle spaces. Arbitrary subtree alternatives
become resolved ASTs before specialization. Full-dimensional native groups must
represent only admitted configurations. Irregular admitted subsets require exact
regrouping or materialization under the current ABI. An arbitrary-ID execution
mode would be a separate kernel extension. An early return in a GPU thread does
not count as omitting that thread.

Plans expose potentially expensive choices: number of resolved models, native
records, constant rows, active counts, capacity buckets, estimated patch use,
module packing, and cold versus reusable compilation work. Estimates are labelled;
the actual specializer and device limits remain authoritative. No silent change
of precision, objective, grammar limits, or sample policy to fit a kernel.

## Constraints, population, and identity

Three phases handle exclusions:

1. **Construction and partial pruning:** generate distinct-state pairs directly;
   propagate node budgets, function nesting, parameter identities, required-state
   possibilities and operator contexts. Prune only when every completion fails.
2. **Completed model admission:** inspect resolved ordered trees, choice labels,
   bindings and descriptors. Named pattern/predicate failures are excluded before
   serialization and scoring and do not consume accepted-model quotas.
3. **Parameter or trajectory checks:** value constraints require assigned values;
   trajectory conditions require numerical work. Rejected parameter draws consume
   proposal effort. Divergent scored configurations consume evaluation budgets.

Provide a small native predicate IR: boolean composition, selected choice tests,
operator/subtree occurrence, node/parameter counts, state occurrence, operator
context, and repeated-wildcard structural patterns. `?q - ?q` requires identical
bound ordered subtrees; two different wildcard names are independent matches.
These are declared search rules, not automatic algebraic rewrites.

Arbitrary predicates remain possible in trusted client-side code or registered
extensions. They operate before submission, carry a version/hash and named reason,
and report CPU cost and rejected proposals. They cannot generally be compiled
into early pruning. Exceptions abort or return an explicit error. The service
does not accept an opaque predicate and pretend it can execute it.

Exact cardinality is available only where computed or proved. Report count kinds
as `exact`, `bound`, `estimate`, or `unknown`. A pruned branch is not an exact
number of excluded ASTs without a known completion count. Ordered, first-failure
rejection counters sum to rejected proposals; optional overlapping diagnostics
are labelled separately. Every search has work/time/memory limits and returns
`complete`, `target_reached`, `exhausted`, or a specific partial/error status.

Default model identity preserves ordered AST topology, state references, FP32
literal bits, and parameter-sharing relationships. Canonical parameter numbering
preserves aliases across all RHSs. Distinct proposal/hard-bound policies remain
distinct search admissions even if their executable model can be shared. Tags
do not change the model but all memberships survive merging. Operator multisets
and unqualified algebraic hashes are not suitable identities.

Start with structural union/intersection/difference against finite indexed model
sets and explicit predicate exclusion. General grammar-language intersection,
semantic equivalence, and e-graph rewriting are later projects with separate
correctness and cost contracts. Do not implicitly enable FP32 reassociation.

Keep per-family admission quotas and global deduplication separate. A model shared
by two families may count toward both family exposures, while it is executed
once for identical numerical work. Record requesting memberships, canonical model
IDs and the number of globally unique models. If the user asks for one million
globally unique AST models, keep filling after duplicates or return explicit
exhaustion; do not count one million derivations or bank configurations instead.

## Parameter sampling and RNG

Each parameter declares identity, role (`fixed` or `fit`), optional physical
units/hard bounds, and a proposal policy. An initial uniform range is not an LM
bound. Sampling and optimization domains are separate. Literal specialization,
user-supplied bank values, grid choices, and RNG-backed slots have explicit source
modes: a supplied constant must never be silently replaced or transformed twice.

Parameter plans distinguish explicit rows, Cartesian grids, zipped/paired grids,
and sampled joint distributions. A mixture declares weights, components, sample
count and counter addressing. Repeated parameter IDs share a value; independent
IDs remain independent unless the joint distribution or draw identity says
otherwise. Numeric sampling never redraws a parameter inside an RK4 step.

Use a separate reusable Philox pool/materialization path. A planned sample is
addressed by seed, round, bank row and a stable semantic draw ID, not GPU number,
thread order or temporary constant-slot position. Record the Philox variant,
counter/key encoding, uniform/normal conversion and implementation versions.
Replay across architectures is guaranteed by saved FP32 bank rows; seed-only
cross-device replay needs separately validated conversion behavior.

Reuse the same rows across structures when parameter meanings and joint policies
match. For example, different oscillatory mechanisms can share amplitude draws
without equating an amplitude to an unrelated decay coefficient. Independent
replicates remain useful. Changing structures must not accidentally permute the
RNG streams by renumbering physical slots.

The first implementation materializes final FP32 rows, preserving the current
kernel contract. A later scoring setup path can transform reusable base draws
with scale/offset/distribution selection. That requires explicit source flags
and numerical tests; it is not implemented by today's scorer. Measure extra
memory traffic and setup cost against specialization/module-load amortization.
Decide from end-to-end measurements before committing a large campaign.

## Results that direct the next search

A winner carries a model ID and family memberships, the joint resolved RHSs,
selected bindings/choice labels, named parameter values and FP32 bytes, objective
and dataset hashes, numerical policy, plan/work IDs, and score status. Persist
these records for replay without re-enumerating a changed grammar. Physical
configuration indices are useful diagnostics, not sufficient model identities.

Report both best configurations and best models. The latter chooses the best
scored parameter vector for each model, so fifty draws of one structure do not
fill every global top-k slot. Per-family and descriptor results include counts,
draw budgets, invalid fraction, and retained winners. Overlapping groups have
overlapping counts. Descriptor values refer to the selected model; any descriptor
dependent on fitted parameters or trajectories is labelled with its evaluation stage.

Scores from different windows, trajectories, objectives, or precision policies
are separate rankings. A short segment screen can promote structures; an
uninterrupted rollout validates them under a different named objective. More draws
give a model more opportunities to win: display that effort and use independent
validation before final selection. Grouped top-k can inform adaptive allocation;
it does not itself implement MAP-Elites.

The current invalid-score sentinel supplies an invalid-score count. Detailed
NaN/overflow/positivity classifications and trajectory metrics need additional
instrumentation. Unsupported diagnostics return capability errors or explicit
unavailable values, never inferred reason counts. Reductions exclude invalid
scores from winning and report an empty winner set if all candidates are invalid.
Stable `(score, model_id, work_id)` ordering resolves ties and supports exact
chunk/device merges. Repeated chunks are merged once.

Expose `inspect`, `plan`, `evaluate`, `report` and `replay` operations. Planning can
be a preview or run automatically as part of evaluation; it is not a mandatory
human approval step. A resident service later owns CUDA contexts and pipeline
handles per device. Multi-GPU sharding changes physical execution, not model IDs,
rows, objective or merge rules. Use deterministic round barriers for adaptive
decisions when reproducibility matters.

## Implementation sequence and acceptance criteria

| Step | Extend existing work | Completion criterion |
| --- | --- | --- |
| 1. Family semantics | Add versioned bindings, labelled choices, parameter IDs and multi-RHS family records beside the existing grammar frontend | Tiny exhaustive oracle checks shared versus independent expansion, inactive choices, rejected combinations and alias retention |
| 2. Generation bridge | Feed supported grammar holes into existing C production tables; implement required binding/choice coordination explicitly | Streaming admission preserves the accepted set and quotas across chunking; unsupported correlated productions fail clearly |
| 3. Current-backend planner | Resolve complex choices, pack exact leaf-toggle spaces, materialize joint FP32 rows, use the existing AST encoder | Packed and explicitly materialized models produce matching real GPU scores; exclusions consume no logical scoring slots |
| 4. Useful reports and replay | Extend current scorer consumers with global/per-family top-k, model deduplication and work IDs | Winner replay and per-device/chunk report merging match a full reference reduction |
| 5. Resident service and RNG | Build the Rack1 request loop, device workers, resource buckets and separate Philox setup | Fixed-bank one/two-GPU runs agree; seeded materialization and retry preserve work; cold and steady timings are measured separately |
| 6. Aggressive search and fitting | Add measured structural sampling, adaptive budgets and LM promotion using the same model/parameter identities | Complete search budgets and independent rollout validation are recorded; no unsupported throughput extrapolation |

The first production vertical slice should execute the worked example through
the C99 scorer with a synthetic trajectory fixture, then extend it to a bounded
recursive grammar and two jointly unknown RHSs. The checked example in this
change establishes the model-to-native-AST mapping only.

For one million models, stream programs and compact provenance in bounded
batches, keep finite work limits, and retain only requested reports/artifacts.
Measure generation, filtering, specialization, module loading, kernels, reduction
and total time separately. Report models/s and configurations/s alongside search
quality per fixed budget. The earlier million-AST corpus result measures
generation; it is not a measured end-to-end score for this unified design.

## Relationship to existing documents

- [Grammar design review](grammar_design_review.md) remains the coverage and
  pruning analysis. Its proposed next steps are consolidated above.
- [Scoring service plan](scoring_service_plan.md) retains the numerical objective,
  resident-service and experiment plan. This document refines its authoring,
  identity, constraints and planning contract.
- [Grammar experiment](../scratch/grammar_search/README.md) is the implemented
  restricted NLTK-to-C adapter, with its own measured results and limitations.
- [Family-search tooling](../scratch/family_search/README.md) supplies existing
  expression parsing, templates and a C99 consumer; it does not yet preserve
  all aliases or implement the joint request semantics above.

The [choice calculus](https://web.engr.oregonstate.edu/~erwig/papers/ChoiceCalculus_TOSEM11.pdf)
motivates shared labelled dimensions. [LLM-SR](https://arxiv.org/abs/2404.18400)
motivates skeletons with numerical parameters. These are precedents for useful
abstractions, not claims of novelty or evidence for Odezza performance.
