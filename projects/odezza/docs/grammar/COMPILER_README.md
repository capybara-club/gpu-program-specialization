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

# ODE grammar compiler

A standalone Python 3.10+ compiler that turns a compact JSON grammar into a stream of whole-system postorder programs and configuration-pool descriptors. It uses only the Python standard library. It does not integrate ODEs, run GPU kernels, score trajectories, or execute Levenberg–Marquardt fitting.

The interface keeps four decisions separate:

1. **Structural generation:** rule productions choose operators and expression structure.
2. **Toggle grouping:** binary or quad leaves select evolving states within a fixed program.
3. **Numeric configuration:** constant banks and RNG draw axes form Cartesian products.
4. **Execution metadata:** integration policy, result-retention tags, and a separate LM request accompany the generated programs.

An LLM can therefore describe a large scientific hypothesis space without emitting every candidate equation or every numeric configuration.

See `LLM_GUIDE.md` for concise instructions an agent can use when preparing requests.

## Quick start

Run these commands from the directory containing this README:

```bash
python -m odegrammar validate examples/oscillator.json
python -m odegrammar plan examples/oscillator.json
python -m odegrammar compile examples/oscillator.json -o oscillator.jsonl
python -m odegrammar compile examples/constant_rng_product.json -o product.jsonl
python -m odegrammar plan examples/million.json --max-skeletons 100
python -m odegrammar lm examples/lm_request.json
```

The LM example contains an explicitly labeled placeholder candidate ID; replace it with an ID from your execution engine before submitting that metadata to an engine. The compiler itself does not look up candidates.

`compile` streams JSONL. Structural programs and toggle variants are separate records, so all pairings of a leaf do not require repeating the instruction arrays. A `manifest` record and a final `summary` record describe the request and actual emitted workload. Duplicate structures can produce additional provenance records when they introduce new family or rule-tag memberships.

The convenience launcher accepts the same commands:

```bash
python compile_grammar.py compile examples/oscillator.json -o oscillator.jsonl
```

For a large run, move deduplication metadata to a fresh SQLite file:

```bash
python -m odegrammar compile examples/million.json --dedup-db million_dedup.sqlite -o million.jsonl
```

The database path must be new. This option bounds the amount of deduplication metadata retained in Python memory; it does not remove output size, disk cost, or other host-runtime limits. For a bounded inspection, add `--max-skeletons 100`. `plan` actually traverses the grammar and builds pool plans under the same limits, discarding program output and reporting its summary; `validate` only checks syntax and declarations without expanding the grammar.

Add `--compact` to share repeated constant banks, RNG descriptors, and numeric plans through content-addressed resource records:

```bash
python -m odegrammar compile examples/toggles_all.json --compact -o toggles.compact.jsonl
```

Compact variants carry `numeric_plan_id` and their toggle groups instead of repeating the full `pools` object. `odegrammar.stream.expand_records` restores ordinary records exactly, apart from dictionary key order. Resource records appear before use; readers must accept repeated identical declarations. Structural program and variant identities stay unchanged. The reference expander caches resources in memory; an engine handling very large streams can store those resources on disk.

Use the CLI's help for available overrides:

```bash
python -m odegrammar --help
python -m odegrammar compile --help
```

## Examples

| File | Demonstrates |
|---|---|
| `examples/oscillator.json` | Known kinematics, three tagged damping laws, two named parameters, and alternative-local RNG ranges |
| `examples/toggles_all.json` | Every binary and quad grouping over five candidate states |
| `examples/toggles_sample.json` | Reproducible sampling of unique groups; repeated leaf references share a selection |
| `examples/toggles_joint_sample.json` | Exactly seven sampled joint group combinations across three leaves |
| `examples/constant_rng_product.json` | Named constant banks, zipped RNG indices, independent RNG axes, and a constant-dependent transform |
| `examples/recursive.json` | Bounded recursive structural generation with sampling |
| `examples/million.json` | Ten independent responses across six derivatives: one million structural combinations |
| `examples/future_stiff.json` | Explicit annotation of an unsupported future stiff integrator |
| `examples/lm_request.json` | Separate, validated LM metadata request |

The million example describes a million whole-system programs, each with one numeric configuration. It does not demonstrate a million different expressions for one equation. The recursive example explores changes inside an expression. A full expansion emits both skeleton and variant records and can generate a large file and consume substantial host time and deduplication storage; use smaller limits for an initial inspection. The example permits one hour of work; that is a budget, not a runtime prediction. This Python reference compiler makes no GPU or compilation-throughput claim.

## Request shape

```json
{
  "version": 1,
  "states": ["x0", "x1"],
  "integration": {"method": "rk4", "dt": 0.01},
  "rules": {
    "Force(z)": ["z", "sin(z)", "pow(z,3)"]
  },
  "parameters": {"k": {"initial": 1.0}},
  "rhs": {"x0": "x1", "x1": "-theta.k*hole(Force,x0)"},
  "expansion": {"strategy": "enumerate", "seed": 42},
  "limits": {"max_skeletons": 100, "max_variants": 1000, "max_configurations": 100000}
}
```

Every declared state requires an RHS expression. The compiler handles equations and configuration data; observation grids, missing-data masks, solver-preparation handles, and trajectories belong to the execution-engine layer and are not part of this prototype's input.

Use top-level `families` instead of `rhs` to submit multiple scientific families. Each family has an `id`, optional `tags`, and an `rhs`. It inherits top-level definitions and can supply its own `rules`, `shapes`, `leaves`, `constants`, `rng`, and `parameters`.

## Expression language

Expressions are parsed as a restricted language; they are never passed to Python `eval`.

| Form | Meaning |
|---|---|
| `x0`, `x1`, … | A declared, evolving state |
| `t` | Integration time |
| `1`, `0.25`, `1e-3` | Finite literal |
| `a+b`, `a-b`, `a*b`, `a/b`, `-a` | Arithmetic with parentheses and ordinary precedence |
| `pow(a,2)` | Integer power; exponent is a literal integer |
| `sin(a)`, `cos(a)`, `tanh(a)`, `exp(a)`, `log(a)`, `sqrt(a)`, `abs(a)` | Supported unary operations |
| `leaf.source` | State selected by a named binary or quad toggle |
| `const.rate` | A named fixed value or constant-bank entry |
| `rng.amplitude` | Value prepared from an RNG bank before integration |
| `theta.k` | Named scalar parameter with an initial value |
| `hole(Response,x0)` | Fresh independent expansion of a rule |
| `shape.force` | A shared named structural expansion |

The compiler validates syntax and bindings. It cannot generally establish that every proposed RHS stays finite throughout integration: for example, `log(x0)` still requires an appropriate trajectory domain in the engine.

### Independent holes and shared shapes

```text
hole(Response,x0) + hole(Response,x1)
```

Each occurrence independently chooses a production. A three-alternative rule gives nine structural combinations before deduplication.

```json
{
  "shapes": {"force": "hole(Response,x0)"},
  "rhs": {"x0": "x1", "x1": "shape.force+shape.force"}
}
```

A named shape is selected once per generated system and reused wherever referenced. In this fragment, there are three choices, not nine. A bare zero-argument rule reference is also supported; `hole(R)` is usually clearer to an LLM.

Rule arguments are expression substitutions. For independence inside a recursive production, write an explicit hole for each child:

```json
{
  "E(z)": ["z", "sin(hole(E,z))", "hole(E,z)+hole(E,z)"]
}
```

### Tags and alternative-local declarations

A production may be a string or an object with `expr`, `tags`, and `locals`:

```json
{
  "expr": "rng.a*tanh(v)",
  "tags": ["saturating_damping"],
  "locals": {
    "rng": {
      "a": {"bank": "unit_draws", "transform": {"kind": "uniform", "low": 0.1, "high": 2.0}}
    }
  }
}
```

Local `leaves`, `constants`, `rng`, and `parameters` are hygienically renamed for each production occurrence. This permits two branches to use `rng.a` with different ranges, and two independent occurrences to own distinct local values. Definitions outside `locals` remain shared by name.

Tags annotate the expanded subtree and are preserved as spans over its postorder instruction array. Spans include their first instruction and exclude their ending index. Family tags also accompany provenance. Deduplication preserves scientific memberships even when several derivations lower to the same program.

`retain` describes a downstream scoring policy:

```json
{
  "global": {"k": 20, "unit": "resolved_structure"},
  "per_family": {"k": 5, "unit": "resolved_structure"},
  "by_tag": {"saturating_damping": {"k": 3, "unit": "numeric_candidate"}}
}
```

Accepted units are `numeric_candidate`, `resolved_structure`, and `variant`. This compiler carries and validates the policy; it cannot select top-k without engine scores. A model may belong to several tag groups. Tags denote membership rather than probabilities or mutually exclusive classes.

## Binary and quad toggle coverage

```json
{
  "source": {
    "states": ["x0", "x1", "x2", "x3", "x4"],
    "arity": 2,
    "coverage": "all"
  }
}
```

`coverage: "all"` emits every unordered group of the specified size. Five candidate states produce ten binary groups or five quad groups. Each emitted group contains exactly two or four distinct states. There is no three-way opcode, duplicated-state padding, or implicit truncation. A fixed state should be written as `x0` directly.

Independent named leaves form a product of group choices. In `toggles_all.json`, each of two structures gets:

- `C(5,2) × C(5,4) = 50` toggle-group variants.
- `2 × 4 = 8` state assignments within each variant.
- `400` configuration visits, covering `25` distinct source/partner assignments repeatedly.

Exhaustive grouping deliberately overlaps: a state occurs in several pairs or quads. This is comprehensive coverage of the requested **groups**, not a partition of scalar candidates. Configuration counts report visits, including those repetitions; a scoring engine can separately deduplicate resolved candidates.

Repeating `leaf.source` anywhere, including across RHS components, uses the same state choice. A different name creates an independent axis.

For sampled group coverage:

```json
{
  "states": ["x0", "x1", "x2", "x3", "x4"],
  "arity": 2,
  "coverage": "sample",
  "samples": 4,
  "seed": 101
}
```

This samples four unique groups without replacement. It does not mean four individual state assignments. Sampling more groups than exist caps to exhaustive coverage. Each axis is sampled separately, then the selected axis groups are combined by Cartesian product.

To request **N joint combinations across all active leaves**, set a top-level or family-level option instead:

```json
{
  "toggle_sampling": {"count": 7, "seed": 321}
}
```

This samples up to seven distinct combinations from the Cartesian product of the active leaves' allowed groups. It does not sample seven groups per leaf. With three binary leaves, each choosing among the ten possible pairs of five states, the full group space has `10³ = 1,000` variants. `toggles_joint_sample.json` emits seven of those variants, each containing `2³ = 8` assignments, for `56` configuration visits. The seven-group target applies per expanded system and its active pool definitions; ordinary global workload limits still apply.

Joint sampling is uniform without replacement over group combinations. It does not guarantee seven disjoint sets of resolved scalar candidates, and it does not alter constant or RNG axes inside a variant. Omit `toggle_sampling` for comprehensive group enumeration. Leave leaf `coverage` as `all` when sampling directly from all possible groups; combining joint sampling with per-leaf `coverage: "sample"` first restricts the domains being sampled.

For deliberate correlations or selected group coverage, use `coverage: "explicit"` and `groups`, each containing exactly `arity` distinct states from that leaf's candidate set. Correlated selections across different leaf axes are not a general tuple-axis feature in this version; submit explicit whole-system family variants if you need that relationship.

## Constant banks and configuration axes

```json
{
  "constant_banks": {"rates": [0.1, 1.0, 10.0]},
  "constants": {
    "a": {"bank": "rates"},
    "b": {"values": [0.25, 0.5]},
    "c": {"value": 2.0}
  }
}
```

`const.a` and `const.b` create a `3 × 2` numeric product when both are active. Repeated references to one slot share its value. Two differently named slots drawing from the same constant bank are independent axes; reference the same slot to share a choice. A single fixed value consumes no additional configurations.

Only bindings used by the lowered program or its RNG transforms contribute to the product. Unused optional branches do not inflate counts. Literal constants are embedded in the program and can change its structural identity; bank values remain runtime configuration data.

## RNG banks, streams, and setup before integration

Declare standard banks independently from the expressions that use them:

```json
{
  "rng_banks": {
    "u": {"base": "uniform01", "count": 32, "seed": 17, "scope": "run"}
  },
  "rng": {
    "a": {"bank": "u", "axis": "trial", "stream": "shared_u", "transform": {"kind": "uniform", "low": 1.0, "high": 3.0}},
    "b": {"bank": "u", "axis": "trial", "stream": "shared_u", "transform": {"kind": "uniform", "low": 10.0, "high": 20.0}}
  }
}
```

There are two separate sharing decisions:

- **Same `axis`:** use the same bank index, yielding 32 paired draws rather than `32²` combinations. Shared axes require equal bank counts.
- **Same bank and `stream`:** use the same raw random bank. In the example, both transforms receive the same unit draw at a given index. Different streams provide separate draws even when their indices are zipped.

The default RNG axis and stream are slot-specific. Independent axes form a Cartesian product. A bank name describes a generation recipe; different streams of that bank denote separate generated arrays.

`scope: "run"` is the default: the same bank recipe, seed, stream, and index produce the same raw value across skeletons. This supports comparing alternative structures with common random numbers. `scope: "skeleton"` also keys generation by the structural program ID, giving reproducible values per skeleton. Scheduling order and toggle-group enumeration order do not alter the bank's addressing contract. Changes to the program can change a skeleton-scoped bank.

The compiler emits bank-generation requests and a per-configuration **prelude**:

1. Bind constant-pool values and parameter initial values.
2. Read the chosen entry from each needed standard bank.
3. Apply that slot's transformation once.
4. Store the result in a slot read by `RNG_VALUE` instructions during integration.

These are sampled constants in an ODE, not noise resampled at every RHS evaluation. An SDE would require a separate execution contract.

| Transform | Input bank | Prepared value |
|---|---|---|
| `identity` | Either | Raw bank entry |
| `affine` | Either | `scale × raw + shift` |
| `uniform` | `uniform01` | `low + (high − low) × raw` |
| `normal` | `normal01` | `mean + std × raw` |
| `log_uniform` | `uniform01` | `exp(log(low) + raw × (log(high) − log(low)))` |

Transform parameters can be finite numbers or `const.slot` references. Such references activate that constant axis even when it does not occur directly in the RHS. Log-uniform bounds must be positive; normal standard deviations must be valid. Distribution-specific transforms require the matching standard base.

In `constant_rng_product.json`, three stiffness values, two transform-scale values, sixteen zipped trial indices, and four independent offset draws give `3 × 2 × 16 × 4 = 384` configuration visits. The two trial bindings do not create `16²` combinations.

The optional Python reference bank materializer is for inspection and testing. It is explicitly **not Philox** and does not promise bitwise agreement with a GPU bank generator. An engine adapter should implement and version a specific Philox counter/key mapping and distribution transform if cross-backend reproducibility is required. Preserve the emitted bank/stream/scope identity when doing so.

## Postorder output

For an equation such as `const.k*x0+sin(leaf.source)`, the instruction order is:

```json
[
  {"op": "CONSTANT", "slot": "k"},
  {"op": "STATE", "index": 0},
  {"op": "MUL"},
  {"op": "TOGGLE", "slot": "source", "arity": 2},
  {"op": "SIN"},
  {"op": "ADD"}
]
```

Operands appear before the operator that consumes them. Each RHS component has its own instruction array. The supported families include `LITERAL`, `STATE`, `TIME`, `TOGGLE`, `CONSTANT`, `RNG_VALUE`, `PARAMETER`, arithmetic operations, `POWI`, and the unary functions listed above.

Skeleton identity includes operator structure, state references, slot-sharing layout, and toggle arities. It excludes family tags, concrete toggle groups, and values supplied through runtime pools. Numeric and pool differences can therefore be represented as variants of one structural program. This is structural deduplication after lowering, not an algebraic-equivalence theorem prover: for example, commutative reorderings need not collapse.

An adapter can retain these instructions, translate them to your opcode encoding, or use the slot annotations to lay out register and constant arrays. It must also combine provenance memberships when returning global, family, or tag results.

### Reference configuration and RHS evaluation

`configuration_indices` converts a flat index to one index per emitted `pool_axes` entry. The **last listed axis varies fastest**. Use the emitted axis order rather than inventing one in an adapter. The helper uses arbitrary-size Python integers; a GPU backend must separately enforce its supported index width.

The following evaluates one RHS at a chosen state without performing integration:

```python
import json
from odegrammar.compiler import compile_request, evaluate_postorder
from odegrammar.pools import configuration_indices, evaluate_prelude

with open("examples/constant_rng_product.json") as source:
    request = json.load(source)

programs = {}
for record in compile_request(request):
    if record["type"] == "skeleton":
        programs[record["id"]] = record
    elif record["type"] == "variant":
        variant = record
        break

pools = variant["pools"]
indices = configuration_indices(pools, 0)
prepared = evaluate_prelude(pools, indices)
choices = {slot: indices["leaf:" + slot] for slot in pools["toggles"]}
derivative = evaluate_postorder(
    programs[variant["skeleton_id"]]["rhs"]["x1"],
    [1.0, 0.0],
    constants=prepared["const"],
    random_values=prepared["rng"],
    parameters=prepared["param"],
    toggles=pools["toggles"],
    choices=choices,
    t=0.0,
)
print(derivative)
```

`evaluate_prelude` prepares constant bindings and transformed RNG values once. Reuse its result as the states evolve; repeated RHS evaluation should not generate new random values. It also returns the selected state index per leaf in `prepared["leaf"]` for direct backend setup. Reference evaluation uses the documented test RNG rather than Philox.

For compact input, pass parsed records through `expand_records` before using the same inspection code:

```python
from odegrammar.stream import expand_records

with open("toggles.compact.jsonl") as source:
    ordinary_records = expand_records(json.loads(line) for line in source)
    for record in ordinary_records:
        # Process ordinary manifest/skeleton/variant/provenance/summary records.
        pass
```

## Bounds, enumeration, and sampling

`expansion.strategy` is `enumerate` or `sample`. Enumeration is deterministic and lazy over rule choices; it is not guaranteed to visit models in increasing scientific complexity. Grammar sampling uses a request seed and uniform production-level random choices with replacement; it is not uniform over unique ASTs and is not size-stratified. Duplicate programs are subsequently deduplicated. This differs from toggle-group sampling, which chooses unique groups without replacement. Both recursive and nonrecursive grammars are supported.

Use `expansion.max_nodes`, `max_depth`, and `max_expansion_depth` to bound structure. Use separate workload limits:

| Limit | Controls |
|---|---|
| `max_skeletons` | New structural programs |
| `max_variants` | Emitted pool/toggle variants |
| `max_configurations` | Sum of configuration visits |
| `max_derivations` | Expanded candidate systems considered |
| `max_expansion_steps` | Internal grammar-expansion work |
| `max_seconds` | Wall-clock work budget |

The final summary reports actual counts and why generation stopped. A target or limit of one million cannot guarantee a million valid unique programs: grammar exhaustion, duplicates, rejection, or an earlier work limit can produce fewer. The output is streamed, but deduplication metadata still grows with the number of distinct programs; `--dedup-db` moves that metadata to disk. Wall-clock limits must be positive; use a sufficiently large `max_seconds` for a long run rather than `null`.

For useful LLM iteration, start with a small compile, inspect the resolved instructions and pool sizes, then increase breadth. Retain distinct structures when the scientific question concerns alternative mechanisms; retain numeric candidates when tuning one mechanism. This prototype exposes the primitives for such a loop without claiming to choose scientifically appropriate priors or validate identifiability.

## Integration policy and separate LM requests

RK4 is the only currently supported executable integration-method declaration. The compiler still emits metadata rather than implementing RK4 itself. `stiff: true` with RK4 is rejected under normal validation.

Future methods can be preserved explicitly as non-executable annotations:

```bash
python -m odegrammar compile examples/future_stiff.json --allow-unsupported-integrator -o future.jsonl
```

Such output carries `integration.backend_supported: false` and `integration.annotation_only: true`; there is no `executable` field. The compiler never silently changes a requested stiff integrator to RK4. The Python API exposes this as `annotation_only=True`.

LM is a separate request with `kind: "lm"`, candidate IDs, at most eight states, and at most eight distinct named fitted parameters. Multiple occurrences of one parameter count once. Non-fitted constant and RNG slots do not consume fitted dimensions. This version supports fitting named RHS parameters only; fitting initial conditions is not implemented. LM controls are validated and returned as metadata; optimizer execution and checking that candidate IDs contain the requested parameters belong to the engine adapter.

## Prototype boundaries

The deliverable is a grammar compiler and inspectable intermediate representation. It does not include MCP transport, a prepared-observation store, numerical integration, GPU compilation, ranking, LM execution, or a production Philox implementation. It does not enforce dimensional consistency, conservation laws, or numerical stability of generated models. Those concerns can be added around this representation without changing the distinction between structure, toggle variants, numeric pools, and setup instructions.

### Handoff clarification

`expansion.max_nodes` applies to the sum across the full RHS vector; `max_depth` applies per RHS. Shared named shapes are inlined and count at each occurrence. Compiler skeleton and variant IDs are not complete score-cache or replay identities; read `../INTEGRATION_HANDOFF.md` before connecting an execution backend.
