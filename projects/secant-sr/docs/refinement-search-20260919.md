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

# Secant-SR refinement search pilot — 2026-09-19

Completed 72 GPU searches in **440.1 seconds elapsed**, across two RTX 5080s on
rack1 and one RTX 4090 on ada. Every execution, CPU score audit and exact AST
replay passed. Search misses remain in the results; no failures were excluded.

## Protocol

The same 12 datasets and two official seeds (23654, 15795) selected for the
previous comparison were fixed before this run. Each dataset/seed stayed on one
GPU for all policies. Arm order rotated. All 10,000 training and 25,000 held-out
rows were retained. Population 8,192, 64 bank rows, 64 permutations, four bank
slots, 31 logical nodes, scientific operators, 20-second cooperative budget and
10,000-generation safety cap. Warmups occurred immediately before each group and
are retained separately. Each policy still pays its scorer setup. No formulas,
known expressions or hand-authored initial genomes entered a search.

## Results

Numerical success here means held-out R² > 0.999. The stricter native success
requires train and held-out NMSE ≤ 1e-6. Symbolic equivalence was not assessed.

| Fitting rounds | Numerical successes | Strict successes | Total process time | Fitting phase time |
| --- | ---: | ---: | ---: | ---: |
| 0 | 12/24 | 9/24 | 354.40 s | 0.00 s |
| 1 | 13/24 | 9/24 | 365.36 s | 6.74 s |
| 4 | 12/24 | 8/24 | 382.41 s | 24.54 s |

One round gained one numerical success, with three individual cases changing
success status between policies. Four rounds did not improve overall recovery.
The small sample does **not** establish a general gain; the new default remains
unchanged. We selected one round for the larger follow-up because it was the
strongest tested fitting arm, and retained a fresh no-fitting control.

| Fitting rounds | Broad configurations | Fitting configurations | Skipped fitting selection visits |
| --- | ---: | ---: | ---: |
| 0 | 21,541,945,344 | 0 | 0 |
| 1 | 20,602,421,248 | 311,951,360 | 508,162 |
| 4 | 20,602,421,248 | 1,245,708,288 | 469,444 |

These are configuration evaluations, not unique ASTs. Fitting skips mean the
genome had more independent parameters than four bank slots; those candidates
still received broad scoring. Counts are repeated selection visits, not distinct
models. This is a substantive remaining fitting-policy limitation. The refiner
uses host Philox, one center per selected AST, per-round specialization/module
loading and no LM. All policies use the same native-toggle scorer.

## Full SRBench follow-up

**Completed:** 232/232 fits, zero execution/replay errors. Numerical successes:
88/116 without fitting, 84/116 with one round. Elapsed: 21 min 51.5 s.
[Final comparison](srbench-refinement-20260919.md). The launch description below
is historical; its tmux workers have exited normally.

The full **116-dataset Feynman portion** ran with seed 23654, comparing
zero versus one fitting round on identical prepared data: **232 fits**, 30 seconds
per fit. This is one official seed, not the full ten-seed SRBench protocol and
not the separate black-box dataset suite. All dataset choices and policy settings
are frozen; 12 datasets overlap the policy-selection pilot.

Work runs in mac1 tmux sessions `secant-srbench-refine-0`, `-1`, and `-2`.
mac1 must remain online. Each worker copies its full results back at completion.
Local status: `scratch/srbench-refinement-20260919/`; remote status:
`/home/cdurham/experiments/secant-srbench-refinement-20260919/shard{0,1,2}/`
(shards 0 and 1 on rack1; shard 2 on ada). A search/audit error stops its shard
and is retained. No CPU scoring fallback or per-dataset budget change is used.

[`python/compare_refinement.py`](../python/compare_refinement.py) is the reusable
paired runner. The ordinary SRBench adapter now accepts refinement options and
reports both broad and fitting configurations/times; its four campaign tests pass.
Pilot [compact results, source/data identities and plans](data/refinement-search-20260919.json)
include every trial. Full logs remain under `scratch/refinement-search-20260919/`.

## Remaining original work

2. Finish and analyze the full SRBench comparison; broader seeds/noise/black-box
   validation and persistent template caching remain.
3. Tune arbitrary-AST registers/instructions/packing and full request throughput.
4. Validate fitting-policy gains; address coefficient capacity and starts. LM deferred.
