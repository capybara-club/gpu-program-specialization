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

# Coefficient rescue diagnostic and search-policy trial — 2026-09-20

**Trial complete:** all 350 fits passed execution and replay checks in **1 hour
40 minutes 40 seconds**. Square/cube mutations scored 36/70 numerical successes
versus 31/70 for baseline. Block fitting scored 35/70 and had the shortest mean
runtime. Improvements are modest, repeat variability remains substantial, and
defaults have not changed. See the final results below.

The completed coefficient diagnostic rescued **3 of 13** cases that the previous
complete random-refinement search solved and the new baseline missed. It took
**6.68 seconds** for both diagnostic parameter policies across all 13 cases.
The GPU selector comparison already favored toggles; this work addresses search
and fitting policy rather than reverting the kernel interface to settings.

## What the diagnostic establishes

Only each retained final winner was polished. Operators, tree structure and the
winning toggle binding stayed fixed. Eight coefficient starts used CPU float64
SciPy LM with an analytic Jacobian, up to 200 function evaluations per start and
a 15-second limit per case/policy. Training data alone selected coefficients.
Every candidate was quantized to float32 and ranked with float32 predictions;
the holdout was evaluated after selecting the final training winner.

Two policies were tested: preserve adjustable-parameter sharing and fixed
literals (`tied`), or independently optimize every literal occurrence
(`all_literals`). Both rescued the same three cases. The latter is a diagnostic
expansion of parameter freedom, not the production fitting policy.

| Case | Previous held-out R² | After tied fitting | Tied fitting time |
|---|---:|---:|---:|
| III.8.54, seed 23654 | 0.976825 | 0.999999999999 | 0.281 s |
| I.6.2, seed 15795 | 0.996185 | 0.99999999999995 | 0.056 s |
| test 5, seed 23654 | 0.998672 | 0.999043 | 0.048 s |

All three passed subsequent native CPU materialization, GPU materialization and
GPU scoring on training and held-out rows. The last is a numerical success at
R² > .999, not a claim of an exact symbolic solution. The other ten failures do
not prove their structures are impossible to fit; this was a bounded local
diagnostic of final winners, not every structure visited during search.

The source comparison used the old **random-refinement** search. The historical
**92.67%** figure used a different maturity/QD/LM policy and a median-of-ten-seeds
dataset aggregate. Neither this panel nor raw per-fit success rates are directly
comparable to that number. This diagnostic's extra fitting time was also outside
the original search budgets.

- [Compact diagnostic results](data/coefficient-rescue-20260920.json).
- Local raw diagnostic: `scratch/polish-20260920/`, including `native-replay.json`.
- Remote diagnostic: `/home/cdurham/experiments/secant-polish-20260920` on rack1.
- The diagnostic used existing NumPy/SciPy packages; no new software was installed.

## Experimental repair

All implementation changes are in **Secant-SR's host search/refinement layer**.
Secant's scoring kernels, specializer, runner and kernel ABI were not changed.

`--refine-parameters active-block` removes the requirement that *all* parameter
identities across a toggle genome fit simultaneously into the runtime bank.
Instead, it finds coefficients used by the winning binding and selects a block
of at most `num_constants`. If there are more active parameters, successive
rounds/generations rotate blocks. Unselected coefficients remain at their
centers, every toggle alternative remains available, and shared parameters stay
shared. The CLI skips genomes with no active adjustable parameters and reports
those skips separately. It still chooses up to 128 top-scoring eligible genomes;
it does not yet implement a diversity archive or multiple starts per genome.

Persistent fitted-parameter IDs are host metadata, independent of runtime bank
slots. Ordinary scoring encodes fitted values as zero-scale affine leaves using
slot zero. Fitting maps only the chosen block to actual GPU bank slots. Thus AST
bytes replay predictions, while **`fitted_leaves` metadata is required to recover
parameter identity for future fitting**. Host API version is 0.3.3; rebuild
clients using `SRConfig` or `SRRefineOptions`. The `all` policy remains the default
and still rejects over-capacity parameter sets.

`--power-mutation-probability 0.25` proposes squares/cubes by copying a subtree
with its existing toggle bits and coefficients and inserting ordinary multiply
nodes. This recreates a useful proposal available to the historical search,
without adding kernel instructions or exceeding the existing node/depth limits.
The probability applies within the mutation branch. It defaults to zero.

This is **random Philox coefficient refinement, not LM**. Fixed literals remain
fixed in production. The CPU LM diagnostic motivated the investigation but is
not inserted into these search arms.

## Matched trial

The panel is the union of the previous 24 diagnostic cases and all 13 old-only
wins, with two overlaps: **35 dataset/seed cases × 2 repeats × 5 policies = 350
fits**. It is selected using previous outcomes and cannot estimate unbiased
full-suite performance. No known formulas or previous winning ASTs enter search.

| Policy | Fitting | Square/cube mutation probability |
|---|---|---:|
| baseline | Off | 0 |
| all-fit | Existing all-parameter policy | 0 |
| block-fit | Active coefficient block | 0 |
| power | Off | 0.25 |
| block-power | Active coefficient block | 0.25 |

Every arm uses the same new executable, 8,192 population, 64 banks × 64 toggle
permutations, four runtime coefficient slots, 31 logical nodes and native pack-8
scoring. Enabled fitting selects up to 128 genomes for one round per generation.
All use a 60-second cooperative budget including setup, with generation-boundary
overruns and teardown recorded. There is no CPU polishing in this trial.

All arms/repeats for a case stay on the same GPU; arm order rotates. Three
workers use rack1's two RTX 5080s and ada's RTX 4090. Frozen manifests, source and
binary hashes, separate warmups and fail-fast score/replay audits are retained.
GPU float32 accumulation can alter close fitness ties, so identical seeds do not
guarantee identical search prefixes. Two repeats expose some of that variability.

- Harness: `bench/policy/campaign.py`.
- Local run: `scratch/policy-repair-20260920/`.
- Remote root on both hosts: `/home/cdurham/experiments/secant-policy-repair-20260920`.
- mac1 tmux workers: `secant-policy-0`, `secant-policy-1`, `secant-policy-2`.
- Independent notifier: `secant-policy-notify`. One campaign start/end pair;
  failures are reported separately. mac1 must remain online for supervision.

Compare paired gains/losses and time-to-threshold, particularly the old-only
misses; inspect parameter-capacity/inactive skips, fitting time, generation count
and accepted power mutations. Do not promote a policy from the small smoke test
or its raw total configuration count. Defaults remain unchanged pending results.

## Final results

The campaign ran September 20, **11:09:34–12:50:14 EDT**. All three workers and
result copies exited successfully; all 350 final CPU/GPU numerical audits and
index/AST replay checks passed. The completion Telegram was accepted at
12:50:21 EDT. Worker durations were 100.7 and 79.4 minutes on rack1 and 87.1 on
ada; their assigned cases differ, so those durations do not compare GPU speeds.
Workers have exited and no campaign fits remain running.

Each row is the same **35 dataset/seed cases × two repeats**. Success is held-out
R² > .999; mean elapsed includes unsuccessful fits, setup and teardown.

| Policy | Numerical successes | Mean elapsed | Paired gains/losses vs baseline | Configurations |
|---|---:|---:|---:|---:|
| Baseline, no fitting | 31/70 | 47.2 s | — | 167.50 B |
| Existing all-parameter fitting | 34/70 | 46.6 s | +7 / −4 | 155.06 B |
| Active block fitting | 35/70 | **44.6 s** | +7 / −3 | 145.76 B |
| Square/cube mutations, no fitting | **36/70** | 45.0 s | +7 / −2 | 152.20 B |
| Active blocks + square/cube mutations | 35/70 | 45.2 s | +6 / −2 | 139.93 B |

Total work was **760,462,966,784 configurations**. These count evaluations,
including repeated genomes and configurations; they are not distinct models.
Lower work totals can reflect earlier stopping, changed expressions and fewer
generations. They are not evidence of an engine throughput regression.

At the stricter native requirement of **both training and held-out NMSE ≤ 1e-6**,
the corresponding successes are **19, 21, 25, 23 and 23**. Block fitting leads at
that threshold; the mutation-only policy leads at the SRBench reporting threshold.

### What improved, and what remains uncertain

1. **Capacity handling is repaired.** Existing all-parameter fitting recorded
   **6,425,514 parameter-capacity skip visits**; both block policies recorded zero.
   These are repeated visits, not that many unique excluded expressions. Block
   fitting processed 541,184 selected starts versus 574,976 for all-parameter
   fitting; it changed eligibility rather than simply increasing fitting volume.
   Block and combined policies explicitly skipped 6,772 and 7,727 visits with no
   adjustable parameters active in the incumbent binding.
2. **Power proposals are promising.** They gain seven successful case/repeat pairs
   and lose two versus baseline. On test 5 / seed 23654, baseline missed both
   repeats while power-only and combined policies succeeded in both. In contrast,
   adding block fitting to power mutation gains one pair and loses two. Combining
   useful mechanisms does not automatically yield the strongest search policy.
3. **Fitting adds little direct runtime.** The entire fitting phase averages
   0.90 seconds per fit for all-parameter fitting, 0.97 for blocks and 0.97 for the
   combined policy: about 1.9–2.2% of process time. Broad scoring accounts for
   95.9–98.0%. Search-quality regressions therefore need investigation of
   selection, diversity and proposals, alongside their small time cost.
4. **Reliability is still weak.** Success classifications differ between the two
   same-seed repeats in **5, 8, 9, 4 and 3 of 35 cases**, respectively. Float32
   reduction differences and time-budgeted search can change the evolutionary
   path. Those disagreements are comparable in scale to the net policy gains;
   the ranking is provisional, not a statistically established improvement.

Among the **13 historical old-only wins × two repeats**, successes were
**3/26 baseline, 6/26 all-fit, 7/26 block-fit, 7/26 power and 6/26 combined**.
Ten of the 35 cases never crossed .999 under any of the five policies in either
repeat: both seeds of I.8.14, test 11, test 13 and test 16, plus test 1 / seed
15795 and test 18 / seed 23654. Their best held-out R² values span roughly
0.9503–0.99885. This motivates work on retained structure diversity and fitting;
it does not establish that every remaining failure is structural.

### Timing limits and next experiment

Across all fits, process time sums to 15,998 seconds because three workers run
concurrently. Broad scoring totals 15,491 seconds, fitting 199 seconds, population
generation 223 seconds and setup 49 seconds. NVRTC is 12 seconds **inside** setup.
Recorded device time totals 14,206 seconds and module-load time 649 seconds;
these are nested/overlapping pipeline measurements, not extra additive stages.
No achieved-occupancy measurements were collected by this campaign.

Keep both changes opt-in. The next quality experiment should preserve the
unrefined search as a control, add training-only coefficient polishing of a
small structurally diverse set of finalists, and evaluate on fresh seeds. The
earlier CPU diagnostic rescued 3/13 misses quickly, making this a useful test of
fitting quality without repeatedly perturbing GP selection. Its extra work must
be included in the next experiment's time budget. Do not move directly to a full
ten-seed campaign on the basis of this outcome-selected panel.

- [Machine-readable results, paired outcomes and every case's R²](data/policy-repair-results-20260920.json).
- Reproduce the aggregation with `bench/policy/report.py`; the report validates
  the complete matched matrix, frozen identities, device/data matching, work
  totals and all retained index/score audits before writing output.

## Validation

- Nine CPU test targets pass, both normally and under address/undefined sanitizers.
- New regressions cover block rotation beyond bank width, distinct equal-valued
  parameter IDs, unchanged inactive alternatives, incumbent trial preservation,
  alias preservation and square/cube toggle semantics.
- Native CUDA refinement tests pass on rack1 and ada, including base-bank restore.
- A three-seed, fixed-generation CPU comparison matches the frozen previous
  no-refinement baseline's winner bytes, scores and work counts exactly.
- Representative new CUDA searches pass existing CPU score and index replay
  audits. Audit tolerances were not relaxed.
- Campaign tests cover the five-arm matrix, identical resume requirements and
  stopping on a failed fit; notification tests also pass.
- Native replay accepts all six rescued model/split checks (three cases × two splits).

## Remaining work

2. This matched trial is complete and analyzed. Validate promising policies on
   fresh seeds before expanding noise/black-box coverage or full SRBench.
3. Profile achieved occupancy and production packing on real GP populations;
   no kernel topology changes were made in this repair.
4. Block fitting removes capacity skips; search reliability still needs work.
   Test budgeted final polishing and diversity-aware selection. Defaults stay
   opt-in; multiple starts and integrated LM remain open.
