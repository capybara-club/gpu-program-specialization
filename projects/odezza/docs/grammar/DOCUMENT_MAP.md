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

# Document relevance and authority

The Markdown is still relevant. The current compiler documentation describes
implemented behavior; the original MCP design describes a broader product.
Those roles must remain distinct when implementing the adapter.

| Document | Status and intended use |
|---|---|
| `START_HERE.md` | Current handoff entry point and known user constraints. |
| `INTEGRATION_HANDOFF.md` | Current compiler-to-engine boundary plus proposed integration work. Statements about existing behavior are grounded in code; future steps are explicitly labeled. |
| `CODEX_TASK.md` | Task brief for the next Codex instance. |
| `ode_grammar_compiler/README.md` | Current public request syntax, CLI and output behavior. |
| `ode_grammar_compiler/LLM_GUIDE.md` | Current instructions for LLM-authored search requests. |
| `ode_grammar_compiler/IMPLEMENTATION_CONTRACT.md` | Current implementation/API notes, corrected in this handoff copy. |
| `ode_grammar_compiler/VALIDATION.md` | Earlier validation record: 83 tests and a million-program host planning run. Timing is historical and excludes GPU work. |
| `ode_grammar_use_cases/README.md` | Current executable examples proving the three large biological search spaces. |
| `ode_grammar_use_cases/evidence/*` | Exact request hashes, expansion counts, representative postorder records and semantic checks. The compact toggle JSONL is a full small stream. |
| `biology_grammar_examples/README.md` | Earlier, still useful examples: feedback, hidden reporter-maturation state, exhaustive overlapping toggle groups, sampled group combinations, and separate LM. |
| `design/ode_search_mcp_design.md` | Historical architecture/product proposal, retained with a warning banner. Its JSON is not the implemented schema. |

When a document conflicts with an executable fixture or test, inspect the code
and reconcile the documentation before wiring the adapter. This archive
contains one canonical compiler copy so changes do not diverge between example
bundles. Use-case check scripts were adjusted only to import that copy.

## Differences from the historical design

| Historical proposal | Current implementation |
|---|---|
| `dsl_version: "1.0"`, `mode: "score"` | `version: 1`; compiler kind is `compile` or omitted. LM uses a separate request and compiler function. |
| Partial `rhs_patch` against a prepared problem | Every declared state needs a full RHS. An adapter can assemble known and mutable pieces before compilation. |
| `leaf_axes`, `constant_axes` | `leaves`, `constants`, optional `constant_banks`. |
| `parameters.*.initial_distribution`, `parameter_draws`, `draw_groups` | `rng_banks`/`rng` describe sampled fixed values. Named parameters accept finite numerical `initial` values. Examples bridge to LM using `rng.a*exp(theta.log_a)`. |
| Nested `shape.generate` | Rules and independent `hole(...)`; named `shape.*` for shared choices. A shape can be an expression, tagged alternative, alternatives list, or `{"choices": [...]}`. |
| Structural sampling without replacement | `expansion.strategy: sample` samples derivations with replacement and deduplicates outputs. Toggle-group sampling is without replacement. |
| Padded three-choice quad leaf | Unsupported. Every runtime group contains exactly 2 or 4 distinct states. |
| General correlated tuple axes | Unsupported. Named leaves form products; explicit whole-system variants can encode correlations. |
| `retain.pareto` | Unsupported. `global`, `per_family`, and `by_tag` are validated and forwarded; ranking is not implemented. |
| Tagged JSON AST input | Public expressions are safe infix strings. Internal Python `Node` objects are not an additional public request format. |
| `input.name`, time-series input tables | Unsupported. The compiler accepts `t` and lowers it to `TIME`; the engine must support it. |
| Unit constraints, mutation/rewrite service, experimental-design endpoints | Proposed future features. |
| `ode.prepare`, `ode.search`, `ode.fit`, result tools | Proposed server operations; no MCP implementation is included. |
| Production Philox | Emitted bank descriptors name `sha256_reference_v1`. A production profile needs explicit versioned translation. |

Limited historical aliases (`family_id`, selected `budget` and top-level
`generate` fields) are normalized by the compiler. They do not make the entire
old example request valid. Prefer the documented current syntax.

## Corrections made while preparing this handoff

- Unsupported integrator annotations use `integration.backend_supported=false`
  and `integration.annotation_only=true`. There is no `executable` field.
- Added `group_sampling` and the monotonic `deadline` to the abbreviated API
  signatures; documented the supported shared-shape forms and `TIME` behavior.
- Clarified that grammar parameter declarations carry numerical initial values;
  LM requests carry candidate IDs and fitted parameter names, not initial states.
- Clarified whole-system node bounds and inlined shared-shape accounting.
- Marked old architecture syntax as historical, and directed use-case scripts
  and commands to the single canonical compiler directory.

No compiler implementation changes or grammar-version changes were needed.
