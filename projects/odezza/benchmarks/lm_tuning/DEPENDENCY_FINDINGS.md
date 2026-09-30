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

# More states referenced in every RHS — September 10, 2026

Increasing dependency breadth exposed a weakness in the current search policy.
All 33 attempts completed without execution errors or unprepared template work,
but only ten verified within the common 60-second search budget. Four of eleven
fresh systems were verified on at least one host. The five systems with three or
four distinct inputs per RHS were not recovered within that budget.

The [design](DEPENDENCY_BREADTH.md) holds six states, syntactic AST depth three,
two coefficient leaves in the blinded RHS, and observation coverage fixed.
**Every RHS**, including all five known equations, references the requested
number of distinct states. All state observations are available. The solver
receives the same broad grammar at every level; it receives no hidden dependency
mask, private equation seed or planted calibration candidate.

| Distinct inputs in each RHS | Generated systems | Verified attempts / three hosts | Systems verified on any host | Successful time range |
|---:|---:|---:|---:|---:|
| 1 | 3 | 4/9 | 2/3 | 1.76–50.97 s |
| 2 | 3 | 6/9 | 2/3 | 12.12–50.35 s |
| 3 | 3 | 0/9 | 0/3 | None within 60 s |
| 4 | 2 | 0/6 | 0/2 | None within 60 s |

rack1 verified 4/11; rohini and ada each verified 3/11. The machines repeat the
same systems and seeds, so 33 attempts are not 33 independent datasets. Different
levels contain different random structures, and dependency requirements change
which operator/leaf combinations fit the depth bound. This small cohort does not
isolate a causal effect of coupling alone. A timeout does not establish that a
system cannot be recovered with more time or a different search policy.

## What the examples reveal

The fastest case, case-008, has the generated RHS

```
(x3 + x3) * (x3 - x3) - ((-0.7416901554719879 * 0.9483446340444315) * x3)
```

It is exactly a linear term, approximately `0.70337788 * x3`. The simple-baseline
pass verified it in 1.76–1.80 seconds. The generator correctly detects one active
state dependency, but its syntactic depth and node count overstate mathematical
complexity. A depth-three random-tree benchmark needs a separate effective-
complexity diagnostic; blindly treating every depth-three draw as equally hard
would flatter recovery speed.

Conversely, missed case-007 simplifies to

```
dx5/dt = x2 * (1 - x1 + x4) + x5 + 0.21678265
```

The displayed constant is rounded. This is a four-input quadratic polynomial,
and every host exhausted the minute. The remaining search weakness includes
ordinary algebraic structure; it is not restricted to exotic functions. These
facts suggest auditing proposal, retention and fitting separately. They do not
prove that LM alone, or structure proposal alone, caused the failure. A labeled
calibration with the correct structure supplied is the next discriminating test.
These equations were inspected only for analysis of completed searches and were
never passed back into their search processes.

## Cost and accounting

The three hosts ran concurrently and finished in 10.79 minutes, including
preparation. Summed search-attempt times per host were 549.94 seconds on rack1,
541.72 on rohini and 549.41 on ada. Scoring preparation for six required shapes
took 41.27/42.30/80.93 seconds; restarting from artifacts took
0.15/0.26/0.22 seconds. LM preparation took 7.74/7.88/15.72 seconds. Preparation
costs are outside the per-search budgets and are not free work.

Recorded work totals 3,735,552 seed AST occurrences and 15,354,978,310
configurations. Twenty-one attempts have incomplete accounting for interrupted
commands, so those counts are lower bounds. Global toggle-expanded distinct AST
counts are unknown. No hardware occupancy measurement is implied.

For the three-/four-input groups, median reported controller screen time was
about 22–23 seconds and fitting time about 34–36 seconds per attempt. These are
controller phase times, not exclusive device-kernel measurements. Removing cold
compilation did not make this policy reliably recover the denser systems.

## Generation and retained evidence

The first 200-draw-per-slot pass accepted 6/12 slots and submitted no GPU work.
A separately retained pass allowed 2,000 whole-system draws per slot and accepted
11/12; one four-input slot remained a generation failure. The cancellation,
state-bound, trajectory-span and FP64 step-doubling checks were unchanged.
Whole-system draws per group were 98/1,139/2,324/2,418. Accepted systems are
conditional on those acceptance rules and are not an unbiased sample of all
possible ODEs. Generation took about 33 seconds in the second pass.

- [Per-case timings and held-out MSEs](runs/coupling-20260910b/REPORT.md).
- [Complete collected metrics](runs/coupling-20260910b/summary.json).
- [Generation rejection counts](runs/coupling-20260910b/generation-summary.json).
- [Structural audit](runs/coupling-20260910b/structure-audit.json).

The source and public execution plans were frozen before searches. Collection
and reporting were extended locally afterward; execution policy was unchanged.
Full worker reports, coefficient rows and replay artifacts remain under
`/home/cdurham/odezza/scratch/coupling_20260910b` on each host. Private generation
records are excluded from exported execution plans and GPU-host backups. This
experiment changes benchmark tooling only, with no
core kernel or installed-service changes.
