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

# Short first search with the supplied grammar

For the next public grammar-game challenge, run from the repository root:

```sh
python3 -B ode_game_probe.py /absolute/path/to/public.json --out scratch/new-game-probe
```

This submits one bounded campaign to the Rack1 trial service, waits for its
report, and returns control to the LLM. It does not submit another campaign or
change the supplied grammar. Telegram start/report notifications are enabled;
`--no-notify` disables them. The output directory must be new.

For an authorized challenge solve, the LLM explains and executes subsequent
requests autonomously. “Propose the next request” means using the report to make
a reasoned choice, not asking the user to approve every campaign. The script
itself still submits only one campaign per invocation.

The measured starting budget is **65,536 ASTs × 768 Philox rows**, with screen
kernel capacity explicitly fixed at 256. Override `--asts`, `--banks`,
`--search-seed`, or `--seconds` as needed. The 90-second default is a maximum
campaign allowance, not the target latency; an exhausted bounded search returns
promptly. Noisy inputs require an explicit `--target-mse`.

The subsequent [prepared saturation benchmark](../benchmarks/2026-09-07-first-pass-saturation.md)
found that **`--banks 1024` is a better measured choice for this workload**:
33% more configurations for about 3% more native prepared-work time, with similar
end-to-end feedback latency in the synthetic service check. The CLI default
remains 768. The larger coefficient-aware optimization is benchmark-only so far;
do not claim it is already used by this entry point.

## What stays fixed

The compiled grammar text, operators, state terminals, depth, parameter slots,
and constant distributions match `ode_game.compile_search`. The entry point
also applies the game's constant range as hard coefficient-refinement bounds.
This was corrected after the first live probe: previously that range controlled
sampling only, so unsuccessful refinement winners could drift outside it.
The entry point
partitions the original root families by node count and allocates the AST budget
across those disjoint strata. Small exhausted strata donate their unused quotas.
This makes large trees eligible in the first request while preserving the full
language. No heuristic expression templates are added.

Within each stratum the native enumerator still takes a deterministic prefix.
This is **not uniform random sampling**, nor coverage of every child-depth shape
or state assignment. The default depth-3 language has 2,164,156,924 ordered ASTs;
65,536 is a small probe of it. `--search-seed` changes coefficient sampling, not
the structural prefix. Depth 0–3 is supported by this entry point; native family
and grammar limits are checked before submission.

## Read the feedback before proposing more work

1. Read `coverage.json` and `latest.json`: compare every requested and evaluated
   quota, accounting completeness, invalid trials, and phase timings. Work-limit
   or timeout results cannot establish the throughput of the requested search.
2. Read `grouped-report.json`: repeated fits of the same ordered structure are
   collapsed, preserving parameter sharing. Each group retains the complete best
   record, family memberships, root, node count, depth, child depths, and states.
   Scores from different objectives stay separate. Family summaries and coverage
   are retained alongside the groups. Algebraically equivalent expressions may
   still occupy different groups.
3. Use `recommendation.json` as a timing suggestion, then explain the next request
   from the results. A complete screen can suggest an AST quota targeting about
   five screen seconds, with growth capped at 4×. This is local extrapolation,
   not an established optimum or evidence that broader search will improve MSE.
4. If expansion is suggested, `proposed-next-request.json` contains an unsubmitted
   request with the same grammar and bank settings. Review it first. If good
   structures need coefficient fitting, propose that instead using their retained
   bytecode, coefficients, and provenance. Do not infer structural evidence just
   from the global winner or compare short/full-trajectory MSEs as one ranking.

There is no native resume cursor: larger quotas repeat earlier prefixes. A
smaller/equal prefix with the same seed is not proposed as useful new work.
Continuing fitted candidates requires an explicit follow-up using those records;
the saved expansion request does not automatically warm-start them. A verified
result ends the search and uses the existing independent acceptance gates.

## Calibration and limitations

Rack1, both GPUs, fresh synthetic three-state depth-3 game. The screen uses six
training trajectories, seven observations each through t=0.6, and 32 RK4 steps
per interval. Same AST quotas, seed, training objective, and 256 capacity:

| Bank rows | Evaluated ASTs | Screen configurations | Screen seconds | Input to grouped feedback |
| --- | ---: | ---: | ---: | ---: |
| 256 | 65,536 | 16,777,216 | 1.451 | 3.385 s |
| 768 | 65,536 | 50,331,648 | 2.721 | 4.257 s |

These are single-run measurements; feedback includes refinement and transport.
The calibration table predates the hard-refinement-bound correction; its screen
work is unchanged, but its refinement timing should not be treated as a measured
bound-constrained fit timing.
They establish a practical starting budget for this workload, not peak GPU
throughput, a universal optimum, cold-start timing, or time to recover an RHS.
Different trajectory lengths, stiffness, missing observations, kernel cache
state, and numerical invalidity can change the cost. The synthetic benchmark
uses an intentionally tiny 1e-30 target to exercise screening/refinement without
claiming recovery; verification was not disabled. It is not a blind accuracy test.

The first baseline starved the 13–15-node strata. A trial-native enumeration fix
now prunes partial trees whose remaining nonterminals cannot grow large enough.
C ordered-output comparisons, quota regressions, recursive-grammar contracts,
and representative full-quota service runs passed. Eight retained screen winners,
including 13/14/15-node trees, were independently replayed in FP64 on the same
training scope and agreed within the recorded tolerances.

An unpinned 768-row request selected capacity 512 and timed out after 40 seconds
with no usable scores. Its underlying cause is unresolved; it is not included
in the table. The new entry point pins capacity 256. An older challenge could
not be reused because its test data were already consumed; a proposed training
resplit was also rejected by the dataset-role seals. Both protections were
respected; the calibration uses fresh synthetic data instead.

Evidence and explicit deviations: `scratch/game_probe_validation/validation.json`,
the four retained synthetic run directories, `enum-validation.txt`, and
`independent-replay.json`. Only the isolated trial enumerator was deployed;
its old source and backend are backed up on Rack1 at
`scratch/recovery_trial/backups/game-probe-1788794499042904620`.

The first live trial is documented in
[`scratch/game_recovery4/POSTMORTEM.md`](../scratch/game_recovery4/POSTMORTEM.md).
Its first report took 4.62 seconds. A larger prefix did not improve the fits;
training-only diagnostics led to six in-grammar candidates and verified recovery.
For fully observed noiseless inputs, prepare those diagnostics alongside the
first screen; do not keep increasing prefixes after equivalent behavior stalls.
