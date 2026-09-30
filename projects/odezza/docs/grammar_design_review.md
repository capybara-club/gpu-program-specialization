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

# Grammar design review

September 5, 2026. This reviews the grammar proposed in the conversation: quadratic monomials; trigonometric functions of weighted sums of those monomials; products of two such functions; a restricted rational family; and a weighted sum of one to six motifs. The Python grammar API, constraint API, shared RNG pool, and parameter-setup specialization are proposals, not implemented capabilities. This document does not change the scoring kernel or launch a search.

The [unified search design](unified_search_design.md) now consolidates the next
steps with named choices, shared bindings, parameter policies, explicit execution
plans and grouped results. It includes an executable finite example. The coverage
and pruning analysis below remains applicable; its earlier implementation sequence
is superseded by that consolidated sequence.

## Assessment and coverage

That grammar is a useful narrow hypothesis family. It is not a general grammar for nested ODE laws. Its ability to express the revealed three-state example is a post-reveal coverage observation, not evidence of blind discovery or an adequate sampling probability.

| Example | Limitation of the proposed grammar | Where to address it |
|---|---|---|
| `x*x*x`, `x*x*y` | Polynomial motifs stop at degree two | Grammar |
| `x*sin(y)`, `z*cos(x*y)` | There is no product of a polynomial and a nonlinear motif | Grammar |
| `sin(tanh(x*z))`, `sin(sin(x))` | Trigonometric arguments contain only polynomials | Bounded recursive grammar |
| `exp(-x)`, `log(1+x*x)`, `sqrt(1+x*x)` | Those operators are absent from the grammar although the C AST supports them | Grammar plus domain policy |
| `x/(K+x)`, `x^4/(K^4+x^4)` | Denominator is restricted to `1 + rho*state^2` | Rational families and appropriate domain constraints |
| `sin(omega*x)*cos(omega*y)` with one shared `omega` | Fresh parameters can take equal values, but do not encode the intended equality constraint | Explicit binding and parameter identity |
| Coupled equations using the same reaction rate | Independent expression expansion does not enforce shared rates or conservation | Joint system grammar and bindings |
| Constant forcing or a zero RHS | The suggested unconditional state-dependence filter excludes them | Remove that default; enable only as an explicit prior |
| `sin(omega*t)-gamma*x` | No explicit time leaf in the current AST | Kernel/AST change, or a declared time-state augmentation |
| Delayed states, state-triggered discontinuities, stochastic forcing | Expression grammar alone does not supply history, event handling, or stochastic integration | Execution model, beyond this grammar |

Finite bounded trigonometric sums can approximate omitted functions on restricted data ranges. Approximation is not exact representability. More sampled ASTs do not repair a missing production.

## Other holes in the original proposal

- A coefficient's proposal distribution and its admissible optimization domain were conflated. `Uniform(-2, 2)` must explicitly mean an initial proposal, a hard bound, or both. A change in measurement units should not silently exclude the relevant parameters.
- `weighted_sum` needs defined term ordering, repetition rules, parameter allocation, and sampling probabilities. Repeated terms, repeated constants, and expressions such as `a*(b*x)` can produce redundant or unidentifiable parameterizations.
- Grammar derivations, ordered ASTs, parameterized models, and mathematically equivalent functions are different notions of uniqueness. The default should deduplicate identical ordered programs after deterministic parameter naming, preserving parameter aliasing, priors, and toggle coupling. Do not silently reassociate FP32 arithmetic. Optional stronger equivalence policies need their own specification.
- Uniform choice among grammar productions is not uniform choice among completed ASTs. Deduplication, filtering, and different acceptance rates change family exposure. Record the selection method and use explicit family budgets when exposure matters.
- Sharing the same rule name does not share parameters. Explicit bindings should express shared subexpressions and shared parameters. Sharing underlying RNG draws between candidates is a separate choice from tying parameters within one system.
- A syntactic state-use bitset records that a state occurs in the tree. It does not prove the resulting function depends on that state: cancellations and parameter values matter. Name those capabilities honestly.
- Poor scores or invalid trajectories for a few parameter draws do not prove a structure is invalid for every admissible parameter vector. Such observations belong to search policy, not an unconditional grammar rejection.

## Recommended representation

Keep Python as the authoring interface. Compile a small set of constructs into an explicit grammar representation: terminals, alternatives, operator applications, bounded repetition/sums, references, named bindings, and tags. Coefficient declarations carry roles, proposal distributions, optional hard bounds, units when known, and identity rules.

Add a bounded general expression family alongside the specialized motif families. A conceptual definition is:

```text
E(depth, budget) = leaf
                | unary(E(depth-1, ...))
                | binary(E(depth-1, ...), E(depth-1, ...))
```

Every operator consumes a node budget; sums distribute a remaining budget among their children. Reject cycles that can recur without consuming a finite bound. Limit total nodes, parameters, function nesting, and supported emitted code, rather than relying solely on tree depth. Special-purpose polynomial, rational, and trigonometric families remain useful priors with explicit budgets.

Attach compact attributes to partial trees and holes: minimum completion size, possible/used state sets, operator counts, enclosing operator context, parameter identities/counts, units, and conservative sign/domain information. This enables early constraint checks. Resource estimates guide batching; the real assembler remains authoritative about register and patch capacity.

## Pruning and filtering are distinct operations

1. **Construct only allowed productions.** For a family of cross-state interactions, ordinary Python `itertools.combinations(states, 2)` can emit `x*y`, `x*z`, and `y*z`. Squared terms and the opposite operand orders never enter that family. This defines its grammar; it is not a global algebraic rewrite.
2. **Prune partial derivations using supported constraints.** If even the smallest completion exceeds the node budget, discard that branch. If a required state cannot occur in any remaining hole, discard it. If a denominator forbids trigonometry, do not expand trigonometric productions there.
3. **Apply arbitrary predicates to completed symbolic trees.** A Python callback can inspect any available tree or metadata property. Rejected trees are not pretty-printed, serialized for execution, specialized, or counted as accepted population members. Constructing a tree sufficiently to run that callback still costs CPU time.

An arbitrary Python callback cannot generally be converted into an early pruning rule. Standard constraints should have native implementations; custom callbacks remain an explicit path whose cost is reported. Callback failure is an error, not a rejected candidate.

Partial checks need at least two outcomes: proven impossible and not yet ruled out. Absence of `z` in a partial expression is not grounds for rejection if a remaining hole can introduce it. Interval analysis that includes zero does not by itself prove a denominator has a root. A policy requiring provable safety is stronger than merely rejecting proven invalidity, and must be named separately.

Do not bake every plausible heuristic into hard exclusions. For example, banning `sin(sin(x))` removes a valid nested law; requiring every RHS to depend on its own state removes legitimate coupled systems. Family-local restrictions and soft sampling preferences provide less destructive alternatives when the evidence is weak.

## Examples of useful constraints

| Intention | Mechanism | Caveat |
|---|---|---|
| Only cross-state products in a specified family | Construct pairs of distinct states | Keep self-interactions available elsewhere when appropriate |
| No trigonometry in denominators | Propagate an operator-context restriction | An explicit prior; bounded oscillatory denominators may be valid physics |
| At most two nonlinear-function levels | Track inherited nesting depth | Separate this from total arithmetic tree depth |
| At most twelve independent parameters | Track distinct parameter identities | Repeated use of a shared parameter counts once |
| Trigonometric arguments must be dimensionless | Propagate units through the grammar | Coefficients may carry units; unconstrained coefficient units weaken pruning |
| Odd response in one state, or exchange symmetry | Construct symmetric/antisymmetric expressions or prove parity | Sampling a few points cannot prove the property globally |
| A conservation relation across unknown RHSs | Reuse named rates in a joint construction | Independent RHS filtering may miss or break the relation |
| No duplicate terms or redundant scale chains | Restricted constructors and declared identity rules | Preserve priors, aliases, evaluation order, and provenance |
| Reject a custom domain-specific arrangement | Named Python predicate with bounded rejection examples | May require a complete symbolic tree |

Operator-argument size and nesting constraints have precedent in the [PySR API](https://ai.damtp.cam.ac.uk/pysr/v2.0.0a2/api). This supports their practicality, not a claim that PySR or the current Odezza iterator implements the particular pruning contract proposed here.

## Toggles require a separate contract

Grammar alternatives describe possible models. They should not automatically become GPU toggle bits. Validate allowed resolved alternatives first, then optionally group compatible models into toggles while preserving exactly the declared work.

For example, two independent bits selecting `u in {x,y}` and `v in {x,y}` generate `x*x`, `x*y`, `y*x`, and `y*y`. A rule excluding equal states leaves two cases. These can be encoded using one shared bit with opposite choice ordering: `u=choose(bit,x,y)`, `v=choose(bit,y,x)`.

More general filters can leave, for example, six admissible words out of sixteen. The current scorer enumerates every word of a power-of-two toggle space; it has no arbitrary admissible-word table. Options are to materialize allowed variants as ordinary ASTs, repartition them into exact representable toggle groups, or introduce a separately validated index-table execution mode. A runtime skip/mask still incurs some GPU work and is not equivalent to omitting a branch during generation.

Report accepted systems, represented structural variants, submitted configurations, and physical padding separately. Parameter draws that deactivate a term can produce repeated realized models; they do not automatically create new unique structures.

## Population and accounting

Define a requested population as **accepted unique candidate systems after declared structural constraints, deduplication, and executable-shape validation**. The definition must state whether toggle variants are grouped under a system or counted individually. With multiple unknown RHSs, uniqueness applies to the complete system and its parameter-sharing graph.

Generation continues until that target, a maximum proposal/work budget, a time limit, or exhaustion is reached. Rejected candidates consume work and rejection counters, but not the accepted-population quota. Assembly failures that merely exceed a supported shape bucket should trigger declared rebatching where possible; unrecoverable errors and unsupported candidates are reported explicitly.

Keep counters for:

- partial derivation nodes visited and branches pruned;
- completed structural proposals;
- complete-tree rejection counts with named reasons;
- duplicate derivations/programs and retained metadata aliases;
- accepted unique candidate systems;
- admitted/excluded toggle variants;
- submitted/completed configuration counts and invalid numerical scores;
- time spent in generation, callbacks, encoding, compilation, scoring, and reporting.

Do not report a pruned branch as an exact number of excluded ASTs unless the completion count is actually known. Supported finite constraints may permit dynamic-programming counts; arbitrary predicates and semantic equivalence generally do not. Label exact counts, estimates, and unknown quantities.

For rejection reasons, specify whether counters are first-failure counts or overlapping diagnostic counts. Provide bounded rejected examples, including a derivation path and an explanation. Constraint contradictions, reference cycles, and rules with no valid completions should be caught in preview when possible.

A run that reaches its work limit before filling the population returns partial coverage. Numerical divergence after an admitted structure is scored remains a scored invalid configuration; it must not disappear from the throughput or coverage denominator.

## Metadata and reproducibility

Retain grammar and constraint hashes, generator version, proposal policy and seed, parameter policies and identities, accepted-system hashes, derivation provenance, tags, and all sampling/replay identifiers. Tags are not part of model identity: merging duplicates must preserve all memberships.

Track family budgets independently of incidental generation order. If groups overlap, their counts overlap; define how quotas are assigned. Changing a pruning rule creates a new search manifest, even when the GPU template can be reused. Retained numerical scores are reusable only under the same candidate, objective, dataset, and parameter configuration.

## Implementation status and next steps

A concrete [C99 constrained enumerator](../scratch/constrained_search/README.md)
now demonstrates node/parameter/context pruning, complete-tree predicates,
accepted-quota accounting, and a callback into the real scoring pipeline. It
includes an independent exhaustive oracle and a synthetic GPU scoring check.
This is a bounded eight-operator example, not the named-production grammar or
family scheduler described above. Its structural admission also precedes
executable-shape validation; specialization failures stop rather than silently
excluding candidates. Canonical exchangeable parameters and externally named
parameter slots are explicit separate modes.

The subsequent [NLTK grammar experiment](../scratch/grammar_search/README.md)
adds actual named productions, family quotas/tags, and normalized C grammar
tables. It generated and checked a million 13-node ASTs across four families,
including an explicitly nested family, and separately tested the grammar table
with the GPU scoring callback. Its restrictions on correlated alternatives,
family overlap, and deterministic ordering are documented; it is not the full
general grammar language proposed here.

The current [C99 iterator](../scratch/ast_tools/o_ast_space.c) already supports bounded opcode/depth spaces, ranked generation when counts are exact, structural sampling, and reusable program buffers. The [configuration header](../scratch/ast_tools/o_ast_tools.h) has no named-production or pruning-attribute representation. Its current visitor return contract aborts on failure; it does not provide an accepted-population refill contract. The scratch Python family generator also materializes expanded candidates.

Implement in this order:

1. Define and lower finite named productions, bindings, coefficient identities, and tags to the existing AST format. Keep general grammar metadata outside the SASS writer.
2. Add node/parameter/state-use/context constraints with early pruning, accepted-population accounting, deterministic bounded generation, and preview/rejection reports.
3. Add optional complete-tree Python predicates and measure their overhead independently. Do not weaken them automatically for throughput.
4. Add bounded recursion and selected type/domain attributes, using an exhaustive tiny-space oracle to verify that pruning preserves exactly the accepted set.
5. Add exact toggle grouping for filtered populations, plus replay tests and benchmarks against explicitly materialized variants.

Verification should include positive and negative reachability fixtures, fresh versus shared parameters, nested-function acceptance, constructor pruning versus post-filter equivalence, unsatisfiable constraints, partial-budget termination, metadata merging, and population invariance across chunking. Semantic or sample-based heuristics must not be presented as sound structural pruning.
