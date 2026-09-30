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

# SRBench refinement comparison — completed 2026-09-19

All **232 fits completed**, with zero execution/replay errors and all 232 independent
CPU/GPU score audits accepted. All three worker exit codes are zero. Results have
been copied back to mac1. Campaign elapsed time, including warmups: **21 min 51.5 s**.

This is the 116-dataset Feynman portion of SRBench, official seed 23654, 10,000
training and 25,000 held-out rows per dataset, and a 30-second cooperative fit
budget. Every dataset stayed on the same GPU for its paired policies. The two
RTX 5080s on rack1 and the RTX 4090 on ada ran independent shards. This is not
the ten-seed SRBench aggregate or the separate black-box suite.

| Policy | Held-out R² > 0.999 | Train and holdout NMSE ≤ 1e-6 | Mean process time | Configuration evaluations |
| --- | ---: | ---: | ---: | ---: |
| Refinement disabled | 88/116 (75.9%) | 71/116 | 15.61 s | 105,864,232,960 |
| One refinement round | 84/116 (72.4%) | 65/116 | 16.36 s | 102,321,094,656 |

Mean time includes all fits, including time-budget misses. Configuration counts
include broad scoring and fitting, duplicates and unused toggle dimensions; they
are not unique structures. Numerical success does not establish symbolic equivalence.

One-round refinement gained two successes and lost six compared with disabled
refinement. It was about 4.8% slower in mean process time. The small pilot advantage
did not persist on this larger single-seed panel. This supports leaving refinement
opt-in while investigating the policy; it does not establish that coefficient
optimization generally hurts search.

The fitting phase itself totaled 35.57 seconds across the 116 refined fits. Fitting
also changes which genomes survive and therefore the subsequent search; runtime
changes cannot be attributed solely to optimizer overhead. There were 3,191,075
recorded selection visits skipped for exceeding the four-parameter fitting capacity.
These are repeated visits, not distinct genomes. Those genomes still received
ordinary scoring. Host Philox, one center per selected AST, per-round module
specialization/loading and no LM remain the configured behavior.

[Summary and changed cases](data/srbench-refinement-summary-20260919.json).
Full per-case outputs, expressions, plans, hashes and timing breakdowns are under
`scratch/srbench-refinement-20260919/shard{0,1,2}/` on mac1. Remote copies remain at
`/home/cdurham/experiments/secant-srbench-refinement-20260919/` on rack1 and ada.
The tmux workers have exited normally; this campaign is no longer running.

## Remaining original work

2. Full comparison complete. Deeper per-case analysis, additional seeds/noise and
   black-box validation, and persistent template caching remain.
3. Arbitrary-AST register/instruction/packing throughput tuning.
4. Investigate fitting-policy regressions, parameter-capacity skips and starts;
   keep refinement opt-in. LM remains deferred.
