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

# A shared-grammar ODE game

The standalone [ode_game.py](../ode_game.py) and editable
[depth2.json](../examples/grammar_game/depth2.json) define a reproducible game.
The generator samples equations from the same bounded AST language that the
compiler submits to Odezza. There are no hand-selected wave, rational or
polynomial templates.

For depth 3, use [depth3.json](../examples/grammar_game/depth3.json), or change
`grammar.max_depth` to `3` in your existing configuration:

```sh
python3 ode_game.py generate examples/grammar_game/depth3.json
```

This writes one `depth3-<UTC timestamp>.json` file and prints the solution to stdout. Depth
is a maximum: smaller trees remain in the language, and generated trees can
simplify algebraically. Depth 3 supports structures such as
`sin(x0*x1) * cos(x2+x0)` and `sin(cos(x0+x1))`. All arithmetic and unary
operators count toward depth, including coefficient multiplication.

With the preset's three states and six operators, depth 3 contains
**2,164,156,924 ordered AST skeletons**, uses up to eight independent constant
slots and 15 nodes per tree, and requests **554,024,172,544 screening evaluations**
at 256 coefficient rows. It fits the existing frontend limits without changing
the scoring kernels. Compilation requests full coverage; campaign time limits
may stop work earlier. This population has not been exhaustively benchmarked.

## Play

Copy `ode_game.py` and the example JSON to any machine with Python 3.9 or newer.
Generation and grammar compilation use only the standard library. No Odezza,
NumPy, SciPy or CUDA installation is needed to generate a challenge.

```sh
python3 ode_game.py generate depth2.json
```

The default creates **one JSON file in the current directory**, with no
extra output folders. Its name combines the input configuration filename (without
its extension) and a UTC timestamp including microseconds, for example
`depth2-20260907T153012.123456Z.json`. Each invocation gets a fresh name. It automatically chooses a 64-bit seed using
operating-system randomness and prints the seed and all solution equations to
stdout. Share the generated JSON file with the solver; keep the terminal solution for
yourself. The file contains these sections:

```text
  depth2-20260907T153012.123456Z.json
    grammar               agreed game configuration; no generation seed
    trajectories          columns and numeric rows; null for hidden observations
    ics                   columns and numeric rows; complete initial states
    knowns                revealed equations and ??? for blinded equations
    problem               machine-readable known/unknown equations
    splits                fixed, disjoint train/validation/test trajectory IDs
    manifest              SHA256 of canonical public content
```

The public file excludes the solution and seed. The generator refuses to
overwrite existing output. To save the terminal solution yourself, redirect
stdout, for example `python3 ode_game.py generate depth2.json > solution.txt`.
The output-path message goes to stderr. Knowing the generation seed permits
reconstructing the answer.

For a deliberate replay, supply `--seed 928417`. To retain the old unpacked
public directory, supply `--public-format directory`; ZIP is also available
explicitly with `--public-format zip`.
`--out challenge` explicitly selects a different output directory. The optional
`--save-private` writes `private/solution.json`, clean trajectories and a generator
snapshot for archival replay; these are not created by default.
Explicit-seed runs with the same configuration,
generator and Python environment produce identical public JSON contents; timestamped
filenames differ. The timestamp and source filename are not embedded in the
challenge, so they do not change its content hash.

The JSON contains ordinary nested objects and numeric arrays, not embedded file
blobs or escaped CSV strings. Its schema is `odezza.grammar_game.public.v1`.
Each table has `columns` and `rows`; trajectory IDs remain strings. Initial
conditions are complete; only hidden trajectory observations use `null`.

On Mac1, from the Odezza repository:

```sh
python3 -B ode_game.py submit depth2-20260907T153012.123456Z.json \
  --banks 256 --search-seed 0 --seconds 300 --out search-run
```

`submit` uploads the three splits, registers the problem and starts a native
grammar campaign on Rack1's existing isolated trial service. It returns a
campaign ID and preserves the actual request and registration. It does not
read the private solution. Use `--plan-only` to register/preflight without GPU
submission. These commands require the Odezza repository client and an active
trial service; compilation remains standalone.

Submission accepts the public JSON directly and checks its content hash and
table format before uploading. It also accepts the old ZIP or public directory.
ZIP contents and manifest hashes are checked; archives may contain only the
seven expected public files (Finder metadata is ignored) and at most 1 GiB of
uncompressed public data. No manual extraction is needed. For a standalone
grammar preview, use the original game configuration:

```sh
python3 -B ode_game.py compile depth2.json \
  --banks 256 --search-seed 0 --out search-preview
```

Follow the returned ID with the existing client:

```sh
python3 -B scratch/recovery_trial/console.py watch CAMPAIGN_ID
```

The campaign owns screening, coefficient fitting, promotion and verification.
Search seed is independent of the private generation seed. The generated grammar
replaces the usual starter families; it is not added to them. Submission returns
immediately; watching or subsequent LLM interaction is separate.

## Grammar semantics

With the example configuration, the conceptual grammar is:

```text
E0 := x0 | x1 | x2 | constant
E1 := E0 | sin(E0) | cos(E0)
         | E0+E0 | E0-E0 | E0*E0 | E0/E0
E2 := E0 | sin(E1) | cos(E1)
         | E1+E1 | E1-E1 | E1*E1 | E1/E1
RHS := E2
```

Depth is the maximum number of operators on a root-to-leaf path. All operators,
including multiplication by a coefficient, count. States and signed numeric
constants are leaves. Thus `sin(x*y)` has depth 2, while `a*sin(x*y)` has depth 3.
There is no extra amplitude or damping term outside the grammar. A negative
constant is one leaf, not an implicit negation operator.

Each RHS is independently sampled **uniformly over ordered symbolic ASTs** in
this language, followed by independent uniform values at each constant leaf.
This does not mean uniform depth, uniform root operator category or algebraic
equivalence class: deeper binary trees are far more numerous. `x+y` and `y+x`
are separate ordered trees. Algebraic cancellations and constant-only expressions
are allowed. Constant values are not part of the finite AST count.

For S state leaves, one constant-leaf category, U unary and B binary operators:

```text
T(0) = S + 1
T(d) = S + 1 + U*T(d-1) + B*T(d-1)^2
```

Three states, two unary operators and four binary operators give T(0)=4,
T(1)=76, T(2)=23,260. The compiler uses independent parameter slots for
simultaneously present constant leaves: at depth 2, at most four slots. Slots
can be reused between mutually exclusive tree positions without tying constants
inside a realized expression.

The native search uses a reusable Philox uniform pool with per-slot scaling to
the configured constant interval. With 256 rows the default requests
**23,260 AST occurrences × 256 rows = 5,954,560 screening evaluations**.
ASTs without constants still repeat across rows; this is a work count, not a
count of mathematically distinct functions. Iterative fitting adds further trials.
Generation uses Python's seeded RNG; the search uses an independently seeded
Philox bank. Shared grammar means identical structural support, not matching RNG
sequences or a promise that a sampled bank contains the exact true constants.

Root choices form separate families so the native scheduler can distribute work.
Enumeration requests the whole finite language by default. `--quota N` instead
caps each root family at an enumeration prefix; it is explicitly reported as
partial coverage, not random sampling. Actual service work limits, campaign
budgets and early success can stop enumeration before all requested ASTs run.
Consult service grammar-coverage reports for achieved work.

## Settings

- `states`: 1–32, named x0 through x(N-1).
- `grammar.max_depth`: 0–4. Native compilation additionally checks the actual
  reachable 64-nonterminal/512-production limits and the available 16 coefficient
  slots. Large populations require a finite quota or smaller grammar.
- `grammar.unary`: any subset of `sin`, `cos`, `tanh`, `exp`.
- `grammar.binary`: any subset of `add`, `sub`, `mul`, `div`.
- `grammar.constants`: independent uniform `min`/`max` at constant leaves.
  These control generation and initial search sampling, not hard constraints on
  the recovery service's later coefficient fitting.
- `blinded_rhs`: names whose equations are withheld. This is separate from
  whether their state trajectories are observed.
- `trajectories.count`, `duration`, `samples`: number and temporal sparsity.
  `sampling` may be `uniform` or `irregular`; both retain t=0 and the endpoint.
- `trajectories.observed_states`: `"all"` or a nonempty list. Other state
  columns contain blanks; `ics.csv` always includes complete exact initial states.
- `noise_stddev`: absolute additive Gaussian observation noise. Initial rows
  remain exact; private clean trajectories are retained.

The standalone generator can hide multiple RHS equations. **The current recovery
service supports exactly one unknown RHS per problem**, so the compiler refuses
multi-blind submissions rather than revealing equations or pretending to search
them jointly. The default example uses one. This restriction does not limit
the number of integrated or unobserved states.

## Short first search for LLM feedback

For a new public challenge, use the [bounded probe entry point](grammar_game_probe.md):
`python3 -B ode_game_probe.py public.json --out scratch/new-game-probe`.
It preserves the supplied grammar, sizes one initial campaign, returns grouped
results, and saves a proposed next request without running it. This is the
preferred interactive starting point; full-language submission remains available.

## Acceptance and numerical limits

Random ODEs can blow up or hit division singularities. Generation uses unprotected
real arithmetic and bounded RK4 integration. It rejects the entire system if any
requested trajectory fails, crosses `max_abs_state`, lacks the requested minimum
observed span, or fails step-halving convergence tolerances. Rejection reasons,
attempt count and the chosen AST ranks are private; the public configuration
states all acceptance rules. It never repairs a sampled RHS or adds damping.

These filters condition the sampled population on numerical acceptance. Accepted
systems are therefore not an unconditional sample from the grammar. The span
check requires some observed variation per trajectory, not full observability or
identifiability of every term. Multiple equations can fit finite trajectories.
No solve-time or unique-recovery guarantee is implied.

The emitted observations use the finer RK4 integration. Step-halving is a
numerical consistency check, not a certified error bound or stiff solver.
Changing sparsity/noise does not consume the equation RNG stream, but a changed
acceptance outcome can select a different attempt. Pin configuration, script and
Python version for exact replay; private provenance records their identity.

The submission template defaults to a noiseless target MSE of 1e-10. Noisy games
require an explicit `--target-mse` on compile/submit; metadata alone does not
configure noise-aware recovery. A threshold controls trajectory acceptance, not
proof that a noisy system's exact symbolic structure has been identified.

## Validation performed

The tests in `tests/test_game.py` compare the sampler's complete finite language
with expansion of the compiled grammar, check counts and parameter slots,
verify integration against an analytic solution, check deterministic replay and
public/private separation, automatic CLI seed selection, byte-identical JSON
replay, JSON submission input, observation equivalence with legacy formats,
and rejection of tampered JSON/unsafe archives.
They also exercise hidden observations and unsupported multi-RHS search rejection.

An end-to-end Rack1 smoke test used the example grammar, 16 Philox rows and a
60-second campaign budget. All 23,260 ordered structures were evaluated:
372,160 screen trials and 372,254 total trials including fitting. The service
reached independent trajectory verification in 15.318 seconds. Its private
solution was inspected only after the result was frozen. The sampled RHS
contains `sin(constant)` and simplifies to an affine expression, so this tests
the pipeline rather than demonstrating hard structural discovery.

Retained evidence: `scratch/grammar_game_plan_active/{request,plan,latest}.json`;
campaign `f35ccbdcbd06992a9f261a37da5cf52e`. This test's 16-row bank differs from
the documented 256-row default; the 5,954,560 default configurations were not
benchmarked in this run. The initial connection failure was resolved by starting
the existing inactive trial service. No scoring kernels or default recovery
families were modified.
