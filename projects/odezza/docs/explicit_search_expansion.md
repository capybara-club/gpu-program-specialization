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

# Broad initial search and explicit structural expansion

Recorded September 6, 2026, after RHS10 and the discussion of genetic programming
(GP). **Design direction only: this note does not implement or change the recovery
controller.** Read alongside [recovery follow-ups](rhs_recovery_followups.md) and
[the ten-minute recovery roadmap](ten_minute_rhs_recovery.md).

## Preference to preserve

The user wants to exploit Odezza's bulk throughput without losing control over
the search or introducing a branch-heavy, difficult-to-tune controller. The
discussion initially suggested asynchronous GP populations, mutation, migration,
coefficient inheritance and diversity policies. After the user challenged the
complexity, the recommendation changed:

**Build an explicit, deterministic grammar-based neighborhood expansion loop
first. Retain GP as a later alternative to benchmark, not the default next step.**

Do not interpret this discussion as authorization to implement a GP subsystem.
Deterministic enumeration does not eliminate tuning: expansion selection,
coefficient budgets and promotion policy must remain visible and measurable.

## Proposed workflow

1. Validate the supplied problem and constraints, then immediately launch a
   reusable, diverse structural portfolio. Include simple direct expressions,
   outer prefactors, rational forms and nested compositions. Avoid waiting for
   the LLM to write a bespoke search before useful GPU work begins.
2. Screen on inexpensive training scopes, preserve diversity, and automatically
   refine/promote survivors to longer training scopes. A poor sampled coefficient
   row is not evidence that its structure is wrong; reserve fitting opportunities
   across families and complexity ranges.
3. Expand selected expressions using a small, declared menu of transformations.
   Enumerate bounded alternatives through the grammar, apply exclusions before
   accepted-population accounting, and submit packed native AST batches.
4. Continue fresh broad exploration alongside local expansion. Local improvement
   alone can miss a needed composition or a jointly beneficial change.
5. Report useful results to the LLM while work continues. The LLM can introduce
   missing transformations or change allocations; it should not be responsible
   for keeping GPUs occupied between every batch.

For a selected subtree `sin(u)`, illustrative expansions are:

```text
sin(u + a*v)
sin(u * v)
v * sin(u)
sin(u) / (1 + a*v*v)
```

Here `v` comes from a bounded grammar, and `a` is a parameter with an explicit
initialization policy. These four examples are not an exhaustive transformation
set or a claim about which structures will recover a new system. Apply operator,
depth, parameter and domain constraints consistently to the expanded result.

Preserve coefficient bindings for unchanged subexpressions when useful, while
also allowing fresh starts and refitting. New coefficients need declared ranges;
do not silently reuse parameter slots with different meanings.

## Throughput and reporting

- Generate and pack work in native batches rather than per-AST Python/JSON calls.
  Use multiple ASTs per module and multiple coefficient configurations per AST.
  Existing compatible toggles may encode choices; arbitrary subtree changes
  still require separate ASTs.
- Use bounded batches/checkpoints so the controller can respond to results.
  Choose batch and bank sizes from actual trajectory-work measurements, not a
  universal AST-count target. Tiny final fits need separate latency measurements.
- Keep explicit budgets for each expansion and for survivor promotion. Avoid
  adding mutation probabilities, migration schedules or other GP controls before
  evidence shows that they help.
- Retain parent ID, transformation ID/version, grammar choices, coefficient
  provenance, seed, objective/scope ID and work counts. Stable enumeration and
  recorded inputs should make proposals replayable; GPU timing/order should not
  be confused with deterministic numerical results.
- Report accepted and duplicate/pruned structures, coefficient trials, fitting
  effort, complexity, family coverage, improvement after fitting and failures on
  longer scopes. Compare scores only under comparable objectives and budgets.
- Attribute discoveries to expansions: for example, whether a state prefactor
  helped after comparable fitting. Expose uncertainty when unequal fitting
  budgets prevent a clean comparison.

GPU-side result reduction and resident analytic LM are useful candidate
improvements, not prerequisites for recording or testing the expansion policy.
Do not describe them as already integrated into this recovery path.

## Evidence and comparison boundary

[RHS10 timing and LM analysis](../scratch/recovery10/LM_ESTIMATE.md) records
79.6 million broad-screen configurations in 2.954 seconds, 44.439 seconds of
fitting and 254.863 seconds to observed verification. Recorded revision waiting
plus remaining controller/client gaps total approximately 204 seconds; this is
not a measurement of GPU idle time or a separate profile of LLM/Python/transport.
The successful revision's 13-second suffix is conditioned on the already-found
search direction, not a fresh-system recovery benchmark.

Odezza's evaluator and its search policy are separate. GP could use the same
scorer, and controller branching does not inherently make specialized scoring
kernels branchy. The concern is implementation complexity, tuning, generation
throughput and predictable coverage. We have not established superiority over
GP or established that GP can sustain the desired AST supply rate here.

If comparing later, run explicit expansion and GP through the same Odezza scorer
and fitter with matched operators, training data, hardware and wall budgets.
Compare recovery success and time to independently verified solutions across
fresh blinded systems and repeated seeds. Preserve the sealed-test boundary.
Configuration counts alone do not measure search quality.
