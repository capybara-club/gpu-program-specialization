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

# Sparse-quadratic term portfolio, version 2

The default prepared grammar now supports one- and two-monomial arguments:

```text
M = 1 | state | state*state
A_simple   = a*M
A_compound = a*M1 + b*M2     (distinct monomials)
F = sin(A) | cos(A)

RHS = linear background + optional quadratic term
    + one oscillation, a product of two, or a sum of two
```

This includes `x*z + c*y`, phase offsets such as `x*y + c`, and two quadratic
terms such as `x*x + c*z*z`. No recovered system's particular argument is
hard-coded. C expands the compact grammar directly to postorder ASTs; Python
does not produce one JSON object per candidate. Core/runtime/service are unchanged.

For three states there are ten canonical monomials and 45 distinct unordered
two-monomial pairs. Repeated pairs such as `a*x+b*x` are represented by the
single-monomial class. Coefficient bounds apply to the canonical representation,
not the old redundant parameterization's effective ranges.

[grammar.json](grammar.json) is the current native grammar. The old profile and
results remain in [grammar-v1.json](grammar-v1.json) and
[EXPERIMENT_V1.md](EXPERIMENT_V1.md). `prepare.py --legacy` reproduces the old
18-family profile without silently expanding its search language.

## Explicit work allocations

There are 58 families without a supplied motif. Mechanism, background and
argument complexity have separate family identities and budgets. Every family
can exhaust its finite structural language; larger families cannot consume
the simpler families' allocations.

Edit [sparse-profile.json](sparse-profile.json) to change these defaults:

| Argument class | Joint coefficient vectors per structure |
|---|---:|
| Polynomial baselines | 8,192 |
| Simple arguments | 8,192 |
| One compound argument | 4,096 |
| Mixed simple/compound pair | 4,096 |
| Two compound arguments | 2,048 |

The three-state default evaluates **367,202,304 configurations** across 127,827
resolved syntax occurrences. These are not algebraically unique functions.
Structural allocations and configuration ceilings are derived per family;
small extra reservations allow the producer to report exhaustion. Unused
allocations are not redistributed. A one-billion-configuration local preflight
rejects oversized profiles before submission. More states can require explicit
budget changes; these defaults are not universal.

Canonical quadratic monomials and argument tails retain disjoint 2-/4-way state
toggles where possible. Named Philox streams share one `trial` axis within each
family. Class-specific bank sizes do not become Cartesian products across
parameter slots. A supplied fixed motif remains unchanged during follow-up.

## Prepare and submit

From the repository root, using the existing private mac3/rack1 connection:

```sh
python3 examples/term_portfolio/prepare.py \
  --knowns knowns.txt --trajectories trajectories.csv \
  --dt 0.125 --indices 0,4,8,12 --seed 12345 \
  --output examples/term_portfolio/runs/initial.json
python3 examples/search_portfolio/run.py \
  examples/term_portfolio/runs/initial.json \
  --output examples/term_portfolio/runs/initial-result
```

Omit `--indices` to preserve **all** trajectories. Version two no longer silently
selects four. Reserve validation data before choosing training inputs. Selected
trajectories retain their whole horizon, times and initial conditions. Choose
`--dt` for the problem; this script does not infer stiffness or a safe step.

`--profile path.json` selects another version-two budget/range profile.
`--rows N` explicitly overrides every class's coefficient-vector count. Add
`--public-motif 'cos(2.4*x)'` only when that motif was supplied by the researcher.
Motif mode has 20 families and is labeled clue-assisted in the plan; it does
not discover the supplied motif.

The CSV/knowns adapter expects x/y/z and one missing RHS. The Python
`build(problem, ...)` API supports general unique state names and one unknown
RHS, with the native numerical payload validated by the service. The CPU fitter
remains specific to x/y known, z unknown, common times and full observations.
This change is not a generalized public fitting API.
The example fitter also accepts native natural-exponential nodes and generates
their parameter/state derivatives, with direct-value and rollout-sensitivity
tests. System22 used that extension with an explicitly adapted `sin`/`exp`
grammar; the default profile remains the documented `sin`/`cos` language.

## Grouped feedback and fitting

The native report is saved verbatim. Feedback includes per-family allocations
and groupings by **argument class**, **mechanism**, and **background**: member
families, evaluated/valid/invalid counts and best retained MSE. Each grouping
separately partitions families; do not sum counts across dimensions. Best-winner
statistics do not establish population-average error or converged fitting.

Eight structures per family plus a global board are retained by default; the
display shows three per family. Exact slots, RNG addresses and programs remain
in the authoritative report. Every complexity class has its own survivors.

Use the existing experimental SciPy sensitivity fitter:

```sh
scratch/recovery20_cpu/.venv/bin/python examples/term_portfolio/fit.py \
  --report examples/term_portfolio/runs/initial-result/report.json \
  --knowns knowns.txt --trajectories trajectories.csv \
  --output examples/term_portfolio/runs/fit.json \
  --per-family 2 --seconds 12
```

Select specific families with `--families id1,id2,...` when fitting all would
be excessive. The fitter uses the first 16 provided trajectories, or all if
fewer, so provide the intended training file. It records source-report/input
hashes and actual training indices. For a public motif, pass its numeric literals
with `--fixed-literals` so coefficient collisions are rejected. Literal 1 is
automatically protected. This restricted literal-rebinding adapter remains
experimental; TODO16 still needs a supported parameterized-program export.

## Guided follow-up

```sh
python3 examples/term_portfolio/followup.py \
  --request examples/term_portfolio/runs/initial.json \
  --report examples/term_portfolio/runs/initial-result/report.json \
  --fit examples/term_portfolio/runs/fit.json \
  --half-width 0.15 \
  --output examples/term_portfolio/runs/guided.json
```

The helper verifies the request hash against the submission, the report hash
against the fit, and candidate origin, source values and parameter identities.
Old unbound fit files are rejected. `--candidate-id` overrides selection of the
best available fitted training MSE. A timed-out fit can contribute its best
valid point; the stopping reason is preserved and does not imply convergence.

Only fitted linear/quadratic ranges change. Structures, fixed motifs,
trajectories, seeds, bank lengths and evaluation allocations are preserved.
The plan records effective ranges and source hashes. The helper creates files,
does not submit them, and does not overwrite the original request. Keep the
original broad request available as exploration. This is not an autonomous
controller and does not automatically choose an exploration fraction.

## Validation and limits

[validation-v2.json](validation-v2.json) records five live native requests:
**368,249,904 evaluations**, exact family coverage, a four-state 4-way-toggle
fixture, and a fresh synthetic score -> fit -> guided request -> score round
trip. The default profile evaluated 367.2M configurations in **7.12s**, using
about **317 MB** charged host memory on two RTX 5080s. That fixture has four
short trajectories and 128 RK4 steps/configuration. It is not a system19
recovery, a long-rollout prediction, or measured peak occupancy.

The eight-vector smoke and single-family four-state case deliberately test
functionality, not throughput. Twenty local tests cover full monomial-pair
support, derivatives, budgets, grouped counts, provenance rejection and unchanged
structure during guided preparation.

The new language closes system19's representational gap. It does not establish
reliable recovery with sampled coefficients or globally converged fitting.
Growing state counts, compatible module packing and survivor selection remain
performance concerns. Sparse arguments are not arbitrary-depth trees; multiple
unknown RHS, additional operators and more background terms remain outside scope.

```sh
scratch/recovery20_cpu/.venv/bin/python -m unittest discover \
  -s examples/term_portfolio -p 'test_*.py' -v
python3 -m unittest discover -s examples/search_portfolio -p 'test_*.py' -v
python3 examples/term_portfolio/validate_v2.py
```

The last command audits saved fixtures without submitting work. Raw data and
logs stay under ignored `runs/`; source, profiles and compact validation evidence
are suitable for version control.
