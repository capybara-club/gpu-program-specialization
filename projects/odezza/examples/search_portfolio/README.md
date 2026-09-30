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

# Prepared structural screening portfolio

This separate example prepares a compact native request, submits it from mac1
through mac3 to rack1, and returns family-grouped feedback for the next search
decision. C performs grammar expansion and Philox sampling; Python does not
materialize the AST population. Core, runtime and service code are unchanged by
this example. It is a screening entry point, **not an autonomous recovery or
coefficient-fitting implementation**.

## Run it

From the repository root, using the existing private service connection:

```sh
python3 examples/search_portfolio/prepare.py problem.json \
  --dt 0.125 --rows 2048 --seed 12345 \
  --trajectory-indices 0,4,8,12 \
  --output examples/search_portfolio/runs/first.json
python3 examples/search_portfolio/run.py \
  examples/search_portfolio/runs/first.json \
  --output examples/search_portfolio/runs/first-result
```

Input is the native `problem` object, or an object containing it: `states`,
`known_rhs`, and `trajectories` with `initial`, `times`, `values`. This controller
requires exactly one unknown RHS. The service validates the numerical payload.
The explicit trajectory indices refer to the input array, not external CSV IDs.
Omit that option to preserve every trajectory. Selected trajectories retain
their entire duration, sampling times and initial conditions. This is ordinary
client-side subsetting, not the reusable trajectory-view service in TODO 9.

Use only training trajectories as input. Reserve validation/test trajectories
before constructing the request. A supplied seed controls structural sampling
and the numeric bank in this example; it is not a new native master-seed field
or a continuation cursor. Omit it for a system-generated seed saved in the plan.

The default is 2,048 coefficient vectors for up to three states, 512 for four to
six, and 128 above six. These are editable starting budgets, not a universal
autotuner. The larger state cases have not been calibrated here. Use `--rows`
to override them. `--coefficient-scale` defaults to 2, giving uniform [-2,2].
Scale and integration step must be chosen from the public problem's units and
dynamics; the script does not silently normalize the system or infer stiffness.

[example-problem.json](example-problem.json) is an intentionally trivial
three-state decay fixture. The known equations are x0'=-0.1*x0 and x1'=-0.2*x1;
the omitted fixture equation is x2'=-0.5*x2. Its four initial states are scaled
versions of one another, so it is unsuitable as an identifiability benchmark.
[example-request.json](example-request.json) is the complete prepared native
request, with [its allocation plan](example-request.plan.json).

## The finite language

Edit [operators.json](operators.json). The default operators are
`sin`, `cos`, `exp`, `tanh`, `square`, and binary `+`, `-`, `*`, `/`.
Conceptually:

```text
T0 = c * (any state or 1)
Td = T0
   | c * unary(T[d-1])
   | binary(T[d-1], T[d-1])
RHS = T2                     [10 root families]
    | T2 + c * target_state  [10 more root families]
```

Each leaf occurrence and unary amplitude has its own coefficient role. Repeated
uses within a square share its operand and parameters. Depth counts the abstract
operators above, excluding coefficient weights; the native postorder can be
deeper because coefficients and square lowering add arithmetic instructions.
The optional self term is outside the core depth bound. Use `--backgrounds none`
to omit it. This does not cover every possible depth convention or unrestricted
combinations of independently parameterized additive terms.

States are packed into **disjoint** groups of four, then two, then a fixed
singleton, plus the constant alternative. Every state assignment is covered
without padding three-way groups or forcing every state into an RHS. A six-state
system can therefore have an unknown RHS depending on only one, two or three
states. Toggle assignments and numeric coefficient rows are separate axes.

All coefficient roles use the same `trial` axis and distinct named Philox
streams. With 2,048 rows, a five-parameter AST receives 2,048 joint vectors,
not 2,048^5 Cartesian combinations. The same named role/row is reproducible
across structures. This does not mean all parameter roles receive equal values.
Scaling occurs through the native RNG transform path before integration.

For N states, five unary and four binary operators, the number of ordered,
state/constant-labelled core trees obeys:

```text
T(0) = N + 1
T(d+1) = N + 1 + 5*T(d) + 4*T(d)^2
```

| States | Depth-two core trees | With both backgrounds | Rows | Evaluations |
|---|---:|---:|---:|---:|
| 3 | 31,420 | 62,840 | 2,048 | 128,696,320 |
| 6 | 227,773 | 455,546 | 512 | 233,239,552 |

These are abstract syntax counts, **not algebraically distinct functions**.
Packing states into toggles yielded 23,874 generated native variants for each
of these two fixtures. Family boundaries, equivalent expressions and optional
backgrounds can overlap semantically. Continuous coefficients are sampled;
exhausting the structure language does not guarantee a good fitted coefficient
vector or successful recovery. Singular divisions and overflowing exponentials
are allowed to fail numerically and count against the evaluation budget.

Three-state depth three already contains **3,949,022,704 core trees** before the
second background or any coefficient trials. Full enumeration is rejected by
the example's one-billion-configuration ceiling. To exercise bounded depth
three, explicitly sample:

```sh
python3 examples/search_portfolio/prepare.py problem.json --dt 0.125 \
  --depth 3 --sample-asts 128 --rows 512 --seed 12345 \
  --output examples/search_portfolio/runs/deeper.json
```

Each root/background family receives its own reserved structural and numerical
limits. The total is derived; no later family silently loses its allocation to
an earlier one. Enumeration reserves slight slack to let the producer report
exhaustion. Sample-mode reservations are conservative maxima for state-toggle
products; they are not guaranteed work counts or uniform draws over resolved
mathematical functions. No allocation is automatically redistributed.

## What the LLM reads and does next

`run.py` saves the authoritative `report.json`, submission identity/hash, and
`feedback.json`. The native request retains eight resolved structures per family
plus a global board. Feedback displays three per family with readable unknown
equations, MSE, state support, parameter count, coefficient values/bits, and
candidate origin. The full report retains every winner and its exact programs,
slots, RNG addresses, coverage and timings. Feedback does not synthesize new
provenance or replace service scores with later fits.

The feedback explicitly distinguishes exhaustion from pruning, allocation limits
and sampled expansion. A low raw MSE is a useful proposal, not a structural
verdict: a needlessly complex expression can outscore the correct simple one
because its random coefficients happened to fit better.

The next workflow should:

1. Fit a diverse shortlist, keeping family/state-support/complexity diversity.
   Use full-rollout residuals and parameter sensitivities/Jacobians with SciPy,
   or compare the native LM path when it has a supported service operation.
   Avoid hundreds of tiny scalar-MSE finite-difference requests.
2. Compare fitted survivors on broader training observations. Use their
   successful subexpressions to propose operator replacements, additive terms,
   products, new state assignments and nesting. Keep an exploratory allocation
   for different branches; do not freeze the first winning tree.
3. Expand trajectory coverage before aggressive elimination. Preserve a final
   untouched validation set, check numerical convergence, and distinguish a
   predictive fit from an identifiable symbolic recovery.

This example implements the initial request and feedback, not those fitting and
adaptive proposal steps. A generic retained-AST-to-parameterized-fit adapter is
still TODO 16. Coefficient identities should come from structured slots, not
guessing parameter locations by matching floating-point literals.

## Measured calibration: 2026-09-13

All measurements use the live mac1 -> mac3 -> two RTX 5080 rack1 service. Every
depth-two family exhausted without pruning/truncation; expected per-family
configuration counts were checked against native reports. Larger same-seed
banks never worsened any family's best MSE. Twelve requests completed,
**2,712,615,936 evaluations** total. Seven local contract tests pass.

Warm native wall times below include parsing, generation and retained scoring.
These are single observations, not distributions or measured SM occupancy.

| Workload | Trajectories | RK4 steps/config | Rows | Configurations | Native wall | Client wall |
|---|---:|---:|---:|---:|---:|---:|
| 3-state decay fixture | 4 | 128 | 512 | 32,174,080 | 1.430s | 1.524s |
| Same | 4 | 128 | 2,048 | 128,696,320 | 1.414s | 1.531s |
| Same | 4 | 128 | 8,192 | 514,785,280 | 2.044s | 2.163s |
| 6-state decay fixture | 4 | 128 | 512 | 233,239,552 | 2.994s | 3.126s |
| Same | 4 | 128 | 2,048 | 932,958,208 | 5.018s | 5.154s |
| System19 training data | 16 | 5,120 | 512 | 32,174,080 | 41.897s | 42.033s |
| Same | 16 | 5,120 | 2,048 | 128,696,320 | 42.634s | 42.786s |
| System19 indices 0,4,8,12 | 4 | 1,280 | 2,048 | 128,696,320 | 10.911s | 11.027s |
| Same subset | 4 | 1,280 | 8,192 | 514,785,280 | 18.105s | 18.226s |

The first 128-row synthetic runs also spent about 0.40s in NVRTC and are recorded
separately in [validation.json](validation.json). Do not compare those cold wall
times as matched warm baselines. System19 here is a **retrospective throughput
fixture**, with its original 16 training trajectories; held-out data was not
used. No fit or new recovery was attempted. Full/subset MSE objectives differ.
The long full-data 2,048-row run had 58.0M valid and 70.7M invalid evaluations;
invalid early exits also affect comparisons.

**Material remaining bottleneck:** the multi-family arena budget selected small
producer pages, with mixed slot/toggle layouts yielding at most 22 compatible
ASTs per tile in this test. At 512 rows, all configurations were in underfilled
tiles. The full-data run increased its numeric work 4x for only **1.8% more
time** at 2,048 rows, but 92% of configurations still fell below the hardware
parallel-work target. At 8,192 rows on the four-trajectory view this dropped to
17%. These are geometry diagnostics, not GPU occupancy counters.

The conservative page budget divides host memory across all families and pages;
batch requirements reserve worst-case axis storage per AST. Improve page arena
sizing and compatible packing under TODO 6 rather than removing memory limits.
No workaround that raises the service's host limit was used. Load/unload/wait
timings can overlap or block on GPU work; their host sums cannot be added and
called independent overhead. Profiling actual CUDA intervals is still useful.

This suggests 2,048 rows and four selected full-horizon training trajectories as
a roughly 11s first response for this particular workload, with an 8,192-row
option for greater numerical coverage. It does **not** establish a universal
10-second solve or a win over PySR/SciPy.

## Why 34 seconds differs from the billion/s benchmark

The [deployed system19 repair](../../benchmarks/long_rollout_tiles/PRODUCTION.md)
replayed four original focused screens: 317,194,240 evaluations in **34.156s**,
down from 222.643s. That sum is not the full recovery time or one original job.

| Original workload | States | Trajectories | Points/trajectory | RK4 steps/config | Rate |
|---|---:|---:|---:|---:|---:|
| Historic short peak | 6 | 1 | 3 | 4 | 1.27B configs/s |
| Four focused system19 screens | 3 | 16 | 21 | 5,120 | 9.29M configs/s |

System19 has 1,280x as many RK4 steps per configuration. Scaling the historic
1.27B rate solely by that count predicts about 320s for its 317M evaluations.
The actual 34.16s is plausible, not suspiciously slow on that crude scale. It
corresponds to 47.55B scheduled RK4 steps/s, or at least 39.57B completed steps/s
from valid configurations alone. The historic short run is only about 5.08B
steps/s. These are not equal-FLOP comparisons: states, RHS instructions, early
invalid exits and fixed-overhead amortization differ. Current matched short
controls remain near 1.67B configs/s in direct C.

The four old focused screens generated 1,421 packed AST variants; numeric rows
and toggle permutations made up most of their count. The broad portfolio above
generates 23,874 variants and many more module groups, so dividing its work by
the old focused rate is also inappropriate. This is why calibration must use
the proposed grammar as well as the actual trajectories.

## CPU comparison and a defensible next experiment

[System20's SciPy workflow](../../scratch/recovery20_cpu/README.md) fit nine
clue-guided models with two starts each in 75.49s wall; the winning four-parameter
fit alone took 3.81s. First recovery including development was 4m00s, independent
validation 5m20s. This is a different system and prior information, not a matched
ratio against system19's 34s of screening. The original system19 recovery also
spent 291.55 native seconds on 398 curvature requests. Fixing its broad screens
does not remove those fitting costs.

SciPy's [least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
can exploit a supplied Jacobian for coefficient fitting. [PySR](https://github.com/astroautomata/PySR)
also searches expression structure; grammars and operator restrictions are not
an Odezza-exclusive advantage. A strong CPU baseline can exploit public known
dynamics and conditional linearity, as the
[previous reduced CPU demonstration](../../scratch/researcher_scipy_trial/README.md)
did in 1.10s.

To test a discovery advantage, freeze this portfolio/controller before new
datasets arrive. Give both workflows the same known equations, operator priors,
training/validation splits and success tolerances. Include CPU sensitivity fits
and useful mathematical reductions in the baseline. Report time to independently
validated recovery, failures under a fixed budget, and end-to-end time including
fitting/LLM/transport. Evaluation counts alone are not discovery speed.

## Validation and artifacts

```sh
python3 -m unittest discover -s examples/search_portfolio -p 'test_*.py' -v
python3 examples/search_portfolio/verify_runs.py
```

The second command audits saved `runs/` artifacts; it does not submit work and
requires the local calibration files. Raw requests, reports and logs stay in
ignored `runs/`; the code, small example and compact validation summary are
suitable for version control. No changes to core/runtime or long campaign were
needed for this example.
