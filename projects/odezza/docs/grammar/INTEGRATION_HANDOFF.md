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

# Integration handoff: existing grammar to GPU engine and MCP

This document distinguishes the implemented compiler contract from the work
still required in the engine adapter. The engine's source and binary ABI were
not available while preparing this archive. Function names proposed below are
integration responsibilities, not claims that those functions already exist.

## 1. Establish the actual backend capabilities

Inspect the destination engine before selecting layouts or exposing tools.
The user reports one ODE per GPU thread, a common executable skeleton within
each warp, binary and quad state-leaf toggles, Cartesian constant/RNG
configurations, and separate LM kernels supporting at most eight states and
eight fitted scalars. Additional non-fitted constants are allowed. Only RK4 is
currently known to be supported. The scoring state limit has not been specified.

Create a capability profile from actual engine code: scoring/LM state limits,
fitted and non-fitted slot limits, supported operations and powers, time-leaf
support, instruction and stack limits, toggle counts/arities, dtype, device
index width, solver settings, observation modes and derivative coverage.
Ordinary compiler validation does not enforce these device limits.

The implemented compiler checks only its own syntax/declarations and the
separate LM metadata limit. A grammar-valid model can still be backend-invalid.
Reject incompatibilities explicitly before launch.

## 2. Start with ordinary compiler records

The core entry points are in `ode_grammar_compiler/odegrammar/compiler.py`:

```python
from odegrammar.compiler import Compiler, compile_lm_request

compiler = Compiler(request)       # One instance per traversal.
for record in compiler.records():
    handle_record(record)          # Implement this in the adapter.
```

`compile_request(request)` is a convenience iterator. A `Compiler` is single
use. `validate` checks declarations without expanding; `plan` actually walks
the bounded grammar and constructs pools. Do not present planning as a free
closed-form count query.

| Record type | Implemented content | Adapter responsibility |
|---|---|---|
| `manifest` | Format, request ID, optional problem ID, ordered states, integration metadata, retention, generation limits/families | Validate prepared context and effective execution profile. |
| `skeleton` | `id`, one postorder array per RHS, direct `active_slots`, node and parameter counts | Translate/cache device code and slot layout. |
| `variant` | `id` and identical `variant_id`, `skeleton_id`, `pools`, family/tags and annotations | Queue configuration-index ranges using the pool plan. |
| `provenance` | Additional membership for an existing `variant_id` | Merge family/tag/span information without re-executing identical work. |
| `summary` | Counts, `complete`, `stop_reason`, host elapsed time and generation statistics | Finalize generation status; execution may still have queued work. |
| `lm_request` | Separate validated candidate IDs, fitted names and optimization metadata | Resolve candidates and execute the selected LM kernel. |

The normal format is `odegrammar.postorder.v1`. The final summary proves only
generation status, not completed GPU evaluation. A limit can yield a valid
partial stream (`complete: false`). A runtime error can follow earlier valid
records; require a terminal status before claiming a complete job.

Bounds apply at whole-variant granularity: if the next variant would exceed
the configuration budget, the compiler does not emit a partial variant.
There is no resumable generation cursor. A SQLite deduplication path must be
fresh for each stream; reusing one would omit required declarations.

## 3. Lower postorder instructions without changing semantics

Each state has a separate scalar instruction array. Preserve manifest state
order when assembling the full vector for the device.

| Instructions | Meaning |
|---|---|
| `LITERAL {value}` | Finite embedded number. |
| `STATE {index}` | Read the current evolving state at that index. |
| `TIME` | Read integration time; compiler accepts it, engine support must be checked. |
| `TOGGLE {slot,arity}` | Read the current state selected by this slot's group and configuration choice. |
| `CONSTANT {slot}` | Fixed numeric slot for this candidate. |
| `RNG_VALUE {slot}` | Prepared transformed sample, fixed during integration. |
| `PARAMETER {slot}` | Named numerical parameter; fixed during one integration, updated between LM trial evaluations. |
| `ADD`, `SUB`, `MUL`, `DIV` | Binary stack operators; pop the right operand first. |
| `NEG`, `SIN`, `COS`, `TANH`, `EXP`, `LOG`, `SQRT`, `ABS` | Unary operators. |
| `POWI {exponent}` | Literal integer power. |

Use `evaluate_postorder` as the reference evaluator. Do not silently substitute
protected division, clipped logarithms, altered power semantics, or saturated
overflow: those change the proposed equations.

Named `shape.*` chooses one grammar expansion per whole system and inlines
that expansion consistently wherever used. It is not a runtime DAG or
common-subexpression register. For reaction fluxes, this shared choice is
essential to preserve consistent consumption/production equations. Any later
CSE/hoisting optimization must retain that sharing and its derivatives.

Tags are not numerical instructions. Variant annotations contain state-specific
`[start,end)` instruction spans and tag lists. Keep their original source
meaning if backend lowering changes instruction positions.

Schedule compatible executable programs uniformly within a warp. Start with
exact skeleton grouping and contiguous ranges within a variant; combine more
aggressively only after defining an explicit argument remapping. Never schedule
different operator sequences in lanes merely because their node counts match.

## 4. Decode pools lazily and preserve tied slots

The authoritative allocation contract is `variant.pools`:

```python
from odegrammar.pools import configuration_indices, evaluate_prelude

indices = configuration_indices(pools, linear_index)
prepared = evaluate_prelude(pools, indices)  # CPU reference, not GPU setup.
```

`pool_axes` supplies the canonical mixed-radix order; the **last listed axis
varies fastest**. `configuration_count` is the product of its cardinalities.
The compiler uses arbitrary-size Python integers. Check the device's index
width and split launches without changing the global configuration address.
Do not allocate one AST or host object for each of a million configurations.

Rules to preserve:

- A leaf group contains exactly two or four distinct state indices. There is
  no three-way toggle, duplicate padding, or invalid fourth lane-code feature.
- Choose a group position once per configuration, but read the selected
  state's current value on every RHS evaluation. A toggle does not freeze an
  observation or initial state.
- Repeated references to a leaf, constant or fitted parameter name are tied.
  Different constant-slot names using the same bank are independent axes.
- RNG bindings with the same `axis` share their draw index. Bindings with the
  same bank and `stream` share raw values. These are different relationships.
- Exhaustive binary/quad groups overlap. Counts are configuration visits, not
  necessarily unique resolved numeric candidates.
- `skeleton.active_slots` lists direct RHS references only. A constant used
  exclusively in an RNG transform can be absent there but present in
  `pools.constants` and `pool_axes`. Allocate from the complete pool plan.

## 5. Make the production RNG choice explicit

Current `rng_bank_requests` carry `algorithm: sha256_reference_v1`. The Python
`bank_value` and `materialize_bank` helpers implement that reference algorithm;
they are not Philox. Preserve this fact in tests, execution metadata and replay.

The emitted address contains version, bank name, seed, scope, stream, and base
distribution. Skeleton scope additionally includes the exact skeleton ID.
Bank length is not part of its raw identity: extending the bank extends the
same sample prefix. Run scope contains no request ID, so the same recipe across
separate requests also reproduces the same reference draws.

For production, define a versioned profile containing the actual Philox
variant/rounds, counter and key packing, distribution transformation, endpoints,
precision and normal-generation algorithm. Record both the compiler recipe and
the production profile in replay/score identity. Never silently reinterpret a
SHA-reference bank as Philox while claiming identical numerical candidates.

Generate unit uniform/normal banks in the separate backend phase requested by
the user. Per configuration, bind constant values, read the chosen bank entry,
apply the emitted `RNG_BIND` transformation, and store the result for
`RNG_VALUE`. Supported transforms are identity, affine, uniform, normal, and
log-uniform. Transform operands can be numbers or constant-slot references;
they cannot be arbitrary state expressions or fitted parameters.

The transformed values remain fixed through all RK4 stages and timesteps and
through a subsequent LM trial. This is random sampling of deterministic ODE
coefficients, not an SDE or transcriptional-noise simulator.

The examples express a positive fitted coefficient as
`rng.a*exp(theta.log_a)` with initial `log_a=0`. Its RNG part is prepared once;
the exponential currently remains in the RHS. If hoisted, recompute it whenever
LM changes theta, including rejected trial steps, and preserve derivatives.

For initial GPU conformance, small precomputed reference banks can be supplied
to both evaluators so RNG differences do not obscure opcode or integration
errors. Test the production Philox profile separately with known vectors and
launch-order/chunking invariance.

## 6. Prepared data and loss are engine responsibilities

The compiler only carries `problem_id` as metadata. It does not resolve a
prepared handle, patch known RHS fragments, upload observations, or calculate
loss. If the engine exposes mutable RHS patches, assemble the complete vector
before calling this compiler and preserve the known parts explicitly.

The prepared problem needs a stable revision for ordered state meanings,
initial conditions, times, observation values/masks, weights, known inputs,
known dynamics and objective semantics. Entire unobserved states are still
integrated and need initial conditions. Missing observations do not imply
zero state values or zero initial conditions.

Connect the actual regular/irregular observation path and define whether RK4
lands on observation times, subdivides intervals, or uses a declared
interpolation policy. Mask residuals only at observed entries. Do not put
missing entries into the MSE denominator or silently fill them with zeros.
Decide whether loss is pooled over observed entries, balanced by state, or
balanced by experiment, and whether weights encode scale or noise. A proposed
weighted pooled objective is `sum(mask*w*residual^2)/sum(mask*w)`; it is not
automatically the objective implemented by the compiler or backend.

Handle empty observed sets, incomplete trajectories, domain errors, nonfinite
states, exceeded step budgets and solver failures explicitly. A successful
trajectory prefix cannot receive a full-trajectory score. Use finite scores for
eligible results and an explicit failure status; avoid NaN/Infinity JSON.

The adaptation examples use a known constant post-step input. Candidate-specific
pre-step steady-state preparation, arbitrary input tables/events, observation
functions, and joint multi-experiment fitting are additional engine features.
A Cartesian input axis selects alternative solves; it does not aggregate
several experiments under shared coefficients. Do not advertise those features
until the execution contract implements them.

## 7. Build complete execution and replay identities

The current compiler hashes:

```python
skeleton_id = content_id({"format": FORMAT, "states": states, "rhs": lowered_rhs})
variant_id = content_id({"skeleton_id": skeleton_id, "pools": pools,
                         "integration": integration["settings"]})
```

The skeleton key includes ordered state names, fixed state indices, global
slot names, literals, parameter-sharing layout and toggle arities. It excludes
prepared context, solver/backend profile and tags. It is not an opcode-only
kernel shape key. Generated local names are normalized; arbitrary user slot
names are not.

Variant identity additionally includes pools and declared integration settings,
but still excludes observations/objective, actual backend profile, dtype and
production RNG profile. `(variant_id, linear_index)` identifies a visit, not a
globally valid score-cache entry.

Persist at least the prepared-problem revision, compiler/ABI version, variant
and configuration address, effective solver settings including resolved dt,
precision/backend and RNG versions, and parameter values. An LM result needs
its parent candidate, fixed bindings, fit settings and fitted values. Define a
separate resolved-candidate key if deduplicating overlaps between toggle groups.

A backend may reuse a more general opcode-only compilation key, but must supply
correct argument/state/slot remapping. Never discard scientific execution
context just to improve cache hit rates.

## 8. Implement top-k and late provenance correctly

`retain` carries `global`, `per_family`, and `by_tag` policies. Supported unit
names are `numeric_candidate`, `resolved_structure`, and `variant`; the compiler
does not implement their ranking or canonical grouping. Define/version what
counts as the same resolved structure, how values are profiled, and tie rules.
Keep measured score and any complexity-adjusted selection metric distinct.

Tags are exact memberships, not Boolean expressions. Use composite labels such
as `gene_0:single_repressor` for scoped questions. One candidate can belong to
several families and tags.

A later `provenance` record can add membership to an already scored variant.
Keeping only global winners would lose candidates needed for a newly attached
family/tag leaderboard. Retain enough per-variant results for all requested
units or a suitable result store, and update memberships without re-solving
identical candidates. Deduplicate resolved candidates before allowing overlap
to fill top-k with repetitions.

Return compact results to the LLM: counts and completion status, failures,
global/family/tag winners, resolved equations, parameter values, loss breakdown
and replay IDs. Store large streams/trajectories outside the model context.

## 9. Keep LM as a separate operation

`compile_lm_request` validates `kind: lm`, a nonempty distinct list of candidate
IDs, one to eight distinct fitted parameter names, at most eight states, and
optimization settings. It returns metadata only. Initial numerical values are
in the scored candidate's pool; the LM request does not carry initial states.

The adapter must resolve every candidate and verify that requested names are
active and supported by that candidate's derivative kernel. Freeze its operator
structure, state choices, constant selections and RNG draws. Fit only selected
theta slots; preserve other parameter values. Unknown initial states, bounds,
observation parameters and general parameter promotion are not implemented by
this compiler. The positive-log-correction examples are an existing-syntax
bridge, not an implicit `rng`-to-parameter promotion feature.

Return actual optimizer status, iterations, initial/final loss and fitted
physical coefficients. Do not equate a shallow random screen with successful
fitting or local LM convergence with a globally best structure.

## 10. Compact transport is optional

After ordinary records work, support `odegrammar.compact.v1` resources with:

```python
from odegrammar.stream import expand_records
ordinary = expand_records(parsed_jsonl_records)
```

Compact variants replace `pools` with `numeric_plan_id` plus `toggles`.
`pool_resource` kinds are `constant_bank`, `rng_bank`, and `numeric_plan`.
Their resource IDs differ from raw RNG descriptor IDs/keys. Resources precede
references and may be repeated identically after producer cache eviction.
RNG resource identity excludes requested count; cache the largest required
prefix. The reference reader checks hashes and reconstructs ordinary records.
It retains resource payloads in memory; large production readers can use disk.

## 11. MCP and integration acceptance plan

First connect the engine through a small Python/host API. Then expose logical
operations for capabilities, preparation, bounded search, result inspection,
candidate replay and separate LM. Names in the historical MCP design are
proposals. Choose transport/schema details using the destination project's
current MCP stack and negotiate optional protocol features rather than assuming
all clients implement them.

Preserve compiler generation limits and add separate execution, LM-iteration,
output and job-time budgets. Provide cancellation, idempotent request handling,
partial status and owner-scoped prepared/job handles. Do not use an unsupported
integrator annotation as permission to execute a fallback.

Acceptance checks for the actual integration, beyond the bundled parser tests:

1. Compare tiny candidate RHS values against `evaluate_postorder`, including
   operand order, powers, tied slots and shared reaction fluxes.
2. Compare regular and irregular trajectory losses against a trusted reference,
   including sparse masks and one completely unobserved state.
3. Compare binary/quad group decoding and constant/RNG products; verify unchanged
   results after chunking, reordering, cache reuse and replay.
4. Test RNG transform-only constant dependencies and separately versioned Philox.
5. Test duplicates and late provenance with global/per-family/per-tag retention.
6. Test separate LM parameter membership, frozen selections, positive-transform
   derivatives, and 8-state/8-fit limits.
7. Test unsupported methods, missing dt resolution, bad handles, nonfinite
   trajectories, exhausted budgets and partial streams.
8. Run the supplied 10k/1M requests on the backend and measure host expansion,
   compilation/preparation, transfer, integration, reduction and serialization
   separately. Current evidence contains no GPU timing.

Integration flags are `backend_supported`, `annotation_only`, `reasons`,
`implemented_methods`, `runtime_dt_required`, and nested `settings`. There is no
`executable` flag. RK4 with `stiff:true` or tolerance controls is rejected unless
annotation-only output is explicitly requested. If dt is omitted, resolve it
from prepared settings and include the effective value in result identity.
