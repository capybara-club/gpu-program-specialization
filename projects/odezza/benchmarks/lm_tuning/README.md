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

# Odezza fitting and recovery tuning

The [scoring readiness and fitting latency results](LATENCY_RESULTS.md) separate
template preparation from timed recovery and compare baseline, expanded fitting,
and state-binding neighbors on six retained systems. Reproduce those checks with
`latency_compare.py`; regenerate the report with `latency_report.py`.

The [dependency-breadth stress test](DEPENDENCY_BREADTH.md) varies the number of
distinct state inputs in every RHS while holding total states and grammar depth
fixed. Its private generation records stay separate from the public search plans.

For the fixed comparisons prepared after the native LM correction, see the
[focused overnight design and commands](FOCUSED_OVERNIGHT.md). It pairs eight
search policies per public case and separates native geometry calibration from
blind recovery. The original random-sweep protocol below remains available.

Reusable experiment infrastructure, separate from the native core and the
deployed search service. This folder owns protocols, generation, paired jobs,
detached queues, telemetry and summaries. It reuses the fitting trial's search
controllers and the public C99 core. It contains no CUDA kernel implementation.

## Dependency breadth is independent of system size

`generate.py` samples ordinary ordered ASTs with randomly selected unary/binary
operators. Each RHS chooses a random subset of states. The unknown RHS can have
an exact requested number of distinct state dependencies and coefficient leaves.
Known RHS dependency breadth is configured separately. The private record stores
all actual matrices, coefficients, seeds, rejection reasons and AST depths.

```json
{
  "states": 6,
  "depth": 3,
  "dependencies": 2,
  "known_dependencies": 3,
  "parameters": 3,
  "background": "random",
  "view": "sparse_hidden",
  "duration": 0.5
}
```

Here the unknown RHS uses two randomly selected states out of six. Depth counts
ordinary operators, including coefficient multiplication. Impossible leaf/depth
requests fail explicitly. Finite-difference probes reject obvious cancelled
dependencies; this is a numerical diagnostic, not a proof of identifiability.
Several coefficient leaves can remain functionally redundant. Analysis must
distinguish declared parameter count from identifiable parameter dimension.

The public challenge contains the broad state/operator/depth grammar, known RHSs,
complete initial values and observations. It contains no private equation seed,
unknown dependency matrix or coefficient values. Search can use every state;
the true dependency subset does not become an oracle restriction.

Dense, seven-sample and seven-sample/hidden-x0 views are supported. All runs blind
the last RHS. `linear_control` is an explicitly different known-background
distribution; it is never substituted after random generation fails. Failed
generation slots remain in the records. References use portable FP64 RK4 with
step-doubling checks; this generator does not require SciPy or claim an adaptive
solver reference. Generation remains on mac1, outside GPU timing.

## Two complementary experiments

**Prepared fitting:** a frozen pool contains a planted structure and structural
neighbours, with the coefficient values withheld. A requested prefix of that
pool may exclude the planted structure. Both methods receive exactly the same
candidate programs, trajectories, coefficient rows, bounds and stopping target.
This measures fitter calibration; it is not blind symbolic recovery. Candidate
count, explicit starts, iterations, LM toggle width, damping, damping retries,
step bound and curvature batch size are recorded. Banks range up to 65,536 rows
and four million requested candidate/start fits. Programs and a compact binary
bank are prepared before timing. Native LM uses the public C99 handle directly
through the existing planner and buffer helper, with at most two live packs;
it does not materialize millions of JSON candidate rows. Curvature consumes the
identical bank in chunks within its 4,096-total-start adapter limit. Each method
records admitted work and deadline truncation; partially evaluated banks are not
equivalent-work throughput comparisons. This is an intentional execution-layout
comparison as well as an optimizer comparison. No kernels or mathematics change.
Report actual candidate count when a finite neighbour pool has fewer members.

CPU training replay rejects inconsistent candidates. Validation selects among
at most four training winners before one held-out test
is consumed. Independent FP64 RK4 replay is timed separately from the fitter.
The kernel event sum may overlap and is not elapsed time. Native reports retain
template preparation, specialization, load and register counts. Native LM
retires modules; its lifecycle differs from the old Python module cache.

**Blind recovery:** both methods receive the same public problem and search seed.
Initial AST budget, coefficient banks, parent archive size, offspring per wave,
maximum offspring, fitted survivors, starts and fitting iterations are tunable.
`generation_limit` means the nominal maximum number of offspring waves, with
`offspring = wave * generation_limit` (capped at the API's 10M limit); wall time,
early success and adaptive coverage may terminate sooner. Actual waves are saved.
State-assignment toggles are enabled from the first screen, including repeated
state assignments. Concrete toggle AST uniqueness is not inferred by multiplying
seed AST counts by four.

Recovery is a whole-policy comparison: adaptive offspring and fitter work differ.
In particular, the legacy curvature search starts from retained screen rows;
`starts_per_candidate` adds LM restarts. Prepared fitting supplies identical
explicit rows to both methods to isolate this effect. Initial simple-baseline
passes retain their existing small fitting policy; zero-wave solves are visible
in the reports and should be analyzed separately from GP search.

## Durable execution

The JSON [example protocol](examples/overnight.json) fixes the design seed, sampling
axes, host list and absolute UTC deadline. That design seed is independent of
private equation-generation seeds. Settings are sampled before results. Repeated
settings use independent trial stores and method order alternates by case.

```sh
python3 -m benchmarks.lm_tuning.deploy --protocol benchmarks/lm_tuning/examples/overnight.json --root /absolute/new/run
python3 -m benchmarks.lm_tuning.coordinator --root /absolute/new/run
python3 -m benchmarks.lm_tuning.summarize /absolute/new/run
```

Run the coordinator in mac1 tmux. Deployment freezes source hashes and builds
existing project code in a separate directory on each CUDA host, then runs native
API and failure tests. No installed service or kernel is overwritten. The host
uses its existing Python environment; no packages are installed by these tools.
Deployment also checks creation of every current adapter template for the
requested state counts and three through eight compiled coefficients. This
covers coefficient counts introduced during search. Resource rejections remain
recorded outcomes; unexpected creation errors stop deployment before timed work.
Passing creation does not guarantee that every expression can specialize.

Each GPU gets a resident worker in a separate user systemd unit, with a process
group watchdog and hard deadline. rack1 has two workers; rohini and ada have one
each. A paired group always uses the same GPU. Multiple workers can contend for
host CPU resources; whole-host timings are not isolated single-process peaks.

The coordinator first supplies ten unchanged depth-three grammar-game cases to
each host, with curvature/native-LM pairs. The `RELEASE_OVERNIGHT` marker is the
explicit operational gate after pilot inspection. Source protocol and deployment
must be frozen before releasing it. Overnight jobs queue behind remaining pilot
work and use fresh dependency-controlled systems.

`inbox → running → done` uses atomic renames and a process lock. Completed groups
are never resubmitted. On restart, an ambiguous running group blocks that worker
until its saved trial results are reconciled. The supervisor resumes delivery
receipts and its generation counter. Infrastructure errors stop the affected
queue; unsupported native shapes are recorded and the paired curvature trial
continues. There is no automatic fitter fallback. Whole groups must fit before
the deadline including drain/verification reserve; unstarted groups are distinct
from timeouts and solve failures. A single Telegram completion notice follows.

## Reading results

`summary.json` and `SUMMARY.md` report completed pairs, errors, unsupported cases,
verified counts and all-outcome elapsed times. Detailed per-trial records retain
settings, input identities, timings and configuration counts. `tuning.csv` gives
one row per trial with state/parameter and hyperparameter axes; `paired-results.json`
records matched-bank identity and accuracy-qualified timing ratios. Cluster repeated
settings/views by source system; reserve new systems for confirmation before
promoting a tuning winner. This random sweep identifies promising regions, not
a globally optimal configuration. It cannot establish a PySR advantage without
a matched PySR recovery comparison on the same public inputs and accuracy gates.
