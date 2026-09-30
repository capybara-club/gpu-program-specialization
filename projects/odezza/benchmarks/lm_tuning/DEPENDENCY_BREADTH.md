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

# Dependencies per RHS

`coupling_sweep.py` prepares fresh random systems using the existing generator's
independent `dependencies` and `known_dependencies` controls. The default
[design](examples/dependency_breadth.json) holds six states, depth three, two
coefficient leaves in the blinded RHS, and observation coverage fixed. Every RHS
uses exactly 1, 2, 3 or 4 distinct states at the corresponding level. Known
equations draw coefficients naturally from the same grammar; their coefficient
counts are not fixed. Two coefficient leaves need not be two identifiable
parameters. All actual RHS depths and numerical dependency probes are retained.

```sh
python3 -m benchmarks.lm_tuning.coupling_sweep prepare \
  --spec benchmarks/lm_tuning/examples/dependency_breadth.json \
  --out benchmarks/lm_tuning/runs/NEW_RUN
```

Generation runs on mac1 with operating-system equation seeds. Only `plan.json`
goes to the search machines: it contains public challenges and predeclared search
settings. Private solutions, equation seeds, calibration candidates and the
dependency-level catalog remain local. The solver receives the same broad
depth-three grammar at every level and can use all six states.

The existing `latency_compare` worker accepts `modes: ["occupied"]` for this
single-policy study. Without an explicit scoring shape list it uses the scorer's
public-input validation to size the known-equation patches. Scoring artifacts
and LM templates are prepared before search timing, with costs recorded. Each
timed attempt restarts the scoring worker against those artifacts and rejects
unprepared template work. Runs belong in detached mac1 tmux sessions, using
isolated remote source trees and existing environments. No installed service or
core kernel needs modification.

Each machine receives the same cases, with rotated order and the same search
seed per case. Comparing dependency levels uses different random systems;
depth and leaf requirements also constrain their operator distributions. This
is a small conditional stress test, not a causal estimate of coupling alone.
Repeated attempts on three machines are correlated observations of the same
systems. Verification establishes predictive fit on the retained trajectories,
not a proof of symbolic uniqueness.

Keep failed generation slots and failed searches visible. On September 10 the
first 200-draw pass accepted only 6/12 slots. The separate 2,000-draw pass accepted
11/12 (3/3/3/2 by dependency level). No cancellation, trajectory-span, state-bound
or FP64 reference-convergence check was weakened. The first pass submitted no GPU
searches. Records live under `runs/coupling-20260910` and
`runs/coupling-20260910b`.

`collect` reads each isolated worker's retained status, results, cold preparation,
restart and LM readiness into `HOST-results.json`. It uses the saved `launch.json`
host/directory manifest and never sends a search revision. Generate the report with:

```sh
python3 -m benchmarks.lm_tuning.coupling_sweep collect \
  benchmarks/lm_tuning/runs/NEW_RUN
python3 -m benchmarks.lm_tuning.coupling_sweep report \
  benchmarks/lm_tuning/runs/NEW_RUN
```
