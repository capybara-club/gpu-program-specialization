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

# Using the ODE grammar compiler as an agent

Use this compiler to turn a scientifically motivated search space into postorder programs and numeric-pool plans. It does not prepare observations, integrate trajectories, score models, or run LM. Prepare state ordering, trajectories, masks, initial conditions, and known dynamics through the user's engine separately. Keep the compiler's state ordering identical to the prepared problem.

## Start from the scientific question

1. Keep known RHS components and terms fixed.
2. Identify uncertain mechanisms: response laws, interactions, source states, and coefficient ranges.
3. Write structural alternatives as rules and independent `hole(...)` occurrences.
4. Use named `shape.*` when one structural choice must be shared across equations.
5. Use `leaf.*` for binary or quad state choices and numeric pools for values fixed during a solve.
6. Choose explicit structural and numeric budgets, validate, and inspect a bounded plan before scaling up.

Every declared state needs an RHS. State names select evolving quantities; a toggle does not freeze a measurement. The grammar should represent smooth ODE hypotheses compatible with the engine's supported operations and solver assumptions.

## Common requests

| User intent | Compiler construct |
|---|---|
| “Try these ten response laws independently in six equations.” | Six holes in a ten-production rule: up to one million whole-system programs |
| “Keep the same selected force term in two equations.” | A named shape used twice |
| “Explore all pairs of these five source states.” | A leaf with `arity: 2`, `coverage: "all"`, and five states |
| “Try seven combinations of pairings across three leaves.” | Top-level `toggle_sampling: {"count": 7, "seed": 321}` with comprehensive leaf domains |
| “Pick seven possible pairs independently for each leaf.” | Per-leaf `coverage: "sample", samples: 7`; their group choices form a product |
| “Sweep these stiffness and damping values.” | Two active constant slots with value lists; independent axes form a Cartesian product |
| “Use 32 random parameter vectors.” | Give their RNG bindings the same `axis`; use separate streams for independent components |
| “Use common raw draws to compare two transforms.” | Same bank and explicit stream; same axis if the indices must also be paired |
| “Use fresh but reproducible draws per structure.” | RNG-bank `scope: "skeleton"` |
| “Fit selected finalists.” | A separate `kind: "lm"` request with candidate IDs and named fitted parameters |

Example commands:

```bash
python -m odegrammar validate examples/oscillator.json
python -m odegrammar plan examples/million.json --max-skeletons 100
python -m odegrammar compile examples/toggles_joint_sample.json --compact -o candidates.jsonl
```

## Be explicit about workload

Set `limits.max_skeletons`, `max_variants`, `max_configurations`, `max_derivations`, `max_expansion_steps`, and a positive `max_seconds`. Bound recursive generation with `expansion.max_nodes`, `max_depth`, and `max_expansion_depth`. `plan` traverses the grammar under these limits; it is not a free closed-form estimate.

Count three different things:

- A **skeleton** is the lowered whole-system operator program.
- A **variant** provides particular toggle groups and numeric-pool definitions.
- A **configuration visit** selects one value on every active axis inside a variant.

Exhaustive pair and quad groups overlap. Report configuration visits honestly rather than calling every visit a distinct model. No three-way runtime toggle is available: use binary or quad groups, or a direct fixed state. A group must contain exactly two or four distinct candidate states.

For large output, use `--compact` and a new `--dedup-db` SQLite path. Compact transport shares repeated banks and plans while preserving identities and exact reconstruction. Do not load a million-record stream into the LLM context; retain summaries and inspect selected candidates through the engine.

## Local choices, tags, and parameter sharing

Tag rule alternatives with scientific labels such as `linear_damping`, `saturation`, or `cross_coupling`. Use family IDs for larger hypotheses. `retain.global`, `retain.per_family`, and `retain.by_tag` specify downstream top-k policies; the compiler only forwards them.

Put a binding under an alternative's `locals` when each independent occurrence should own it. Local names are hygienically renamed. Leave a binding global when all references should share one value or toggle choice. Use an explicit RNG stream to share raw randomness between otherwise local bindings.

Keep named fitted scalars in `parameters` and reference them as `theta.k`. Repeated references consume one fitted dimension. Avoid redundant parameterizations that multiply an unconstrained fitted amplitude by another freely fitted amplitude. LM allows at most eight states and eight distinct fitted RHS parameters in this version. Non-fitted constants and RNG values do not consume fitted dimensions. Initial-condition fitting is not implemented.

## Random values are prepared constants

Use standard `uniform01` or `normal01` banks. A binding's transformation runs before the integration loop, and `rng.name` reads that prepared value during RHS evaluation. Uniform/log-uniform transforms require a uniform bank; normal transforms require a normal bank. Transform parameters may reference constant slots, activating those axes.

Prefer `scope: "run"` for controlled comparisons across structures. Use `scope: "skeleton"` when each program should receive a separate reproducible bank. Same `axis` zips draw indices; same bank and `stream` share the raw values. Omitting an axis creates an independent numeric axis and can multiply the workload substantially.

These are deterministic ODEs with sampled constants. Resampling noise during integration would be a different contract. The optional reference RNG is not Philox; the engine must implement and version its production bank generator.

## Solvers and unsupported requests

RK4 is the only supported executable integration-method declaration. Do not present it as a stiff solver or silently substitute it when the user requests another method. A future method can be compiled only as an explicitly unsupported annotation using `--allow-unsupported-integrator`; the result has `integration.backend_supported: false` and cannot be treated as an executable solve request.

A successful compile does not establish numerical stability, physical consistency, identifiability, or fit quality. Ask the engine to assess generated candidates using the prepared problem and its scoring policy. Preserve provenance when equivalent programs appear under several tags or families. Use scored results to narrow scientifically meaningful alternatives before making a separate LM request.
