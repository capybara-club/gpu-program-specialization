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

# SRBench on Secant 0.3: restart on 2026-09-19

The active entry point is `python/run_srbench_toggle.py`. It supports the frozen
116-dataset Feynman and 122-dataset black-box groups, official trial seeds and
target-noise levels. The old `run_suite.py` targets the retired settings API and
must not be pointed at the new executable.

The next experiment compares larger populations, 60-second fits, optional
toggle-aware variation and the archived settings binary. See the
[September 19 overnight plan](search-population-night-20260919.md).

## Data and assessment contract

Preparation reuses the tested v2.0 adapter: seeded 75/25 split, 10,000 training
samples drawn with replacement where specified by v2.0, and the full held-out
test partition. Clean Feynman jobs use 10,000 train / 25,000 test rows. Only
training targets receive requested noise. Black-box scaling is fitted on training
data; its test scores remain in the equivalent standardized coordinates.

Frozen name manifests are checked by count and SHA256 before preparing data.
Each prepared binary retains its source hash, prepared-data hash, dimensions,
protocol, split seed and target variances. Search receives only binary f32 data
and its random seed—not the dataset's equation, metadata formula, or known AST.
Preparation is serial because the frozen adapter uses NumPy's global RNG.

Fitness and stopping use training data only. The native CLI audits the winner
on CPU and evaluates the held-out fold at the end. The campaign checks that the
returned toggle AST, permutation and coefficients resolve to the reported fixed
AST. It computes R² from the native MSE and the prepared target variance; f32
transport/SSE is an intentional difference from estimators using float64.

For Feynman, `accuracy_solution` means **test R² > 0.999**, independent of the
native CLI's stricter `solved` field. Symbolic assessment stays unassessed, with
empty symbolic-solution fields. Numerical accuracy is not exact symbolic recovery.
Black-box jobs report R², not a symbolic-recovery success percentage.

## Commands

Preparation needs the already-documented NumPy/pandas/scikit-learn environment.
Execution and reports use only the Python standard library.

```bash
python python/run_srbench_toggle.py prepare \
  --suite feynman --pmlb-root /path/to/pmlb \
  --srbench-root /path/to/srbench-v2 \
  --seeds 23654 --noise 0 --output scratch/feynman-prepared

python python/run_srbench_toggle.py run \
  --manifest scratch/feynman-prepared/manifest.json \
  --executable build-toggle/secant_sr_search \
  --config toggle/srbench-baseline.json --gpus 0 1 \
  --output scratch/feynman-baseline

python python/run_srbench_toggle.py status scratch/feynman-baseline
python python/srbench_toggle_report.py scratch/feynman-baseline \
  --output scratch/feynman-baseline/report.md
```

`prepare --suite blackbox` uses the black-box manifest and corresponding frozen
PMLB payloads. It rejects nonzero noise for that track. Passing `--problems`
selects a named subset only after validating the entire official manifest; the
subset remains explicit in the plan. Pass all ten official seeds for a full
trial panel. The initial restart uses **one** official seed, not a leaderboard-
complete evaluation across seeds/noise levels.

## Scheduling, budgets, and retained results

One worker per selected GPU pulls complete fit requests from a shared queue.
Each process sees only its assigned GPU. Both GPUs can evaluate different
datasets concurrently; a single fit is not split between them. There is no CPU
fallback, automatic grammar narrowing, input-column selection, or bank truncation.

The baseline uses 8,192 members × 4,096 configurations/AST, up to 100 generations
or 30 seconds, with the full protocol row counts. Scoring chunks contain up to
4,096 ASTs within a 256 MiB ceiling. The 30-second native budget includes setup
and is checked between generations; a separate process deadline defaults to
twice that budget plus 120 seconds. Deadline failures preserve partial logs but
do not accept partial winners. Initialization/NVRTC costs are measured per fit;
the old persistent CUBIN cache/process has not been restored.

`campaign.json` freezes the effective configuration, executable hash, runner/
adapter hashes, prepared-manifest hash, hardware, and known deviations.
`trials/ID/attempt-N.jsonl` and `.stderr` retain every attempt; `result.json` is an
atomic terminal record. `progress.json` and `results.csv` are rebuilt after every
completed trial, including failures and pending rows. Interrupted attempts get a
new log number. An exclusive process lock prevents two runners owning one output.

`--resume` requires identical code, data manifest, executable, GPU selection and
configuration. It skips all terminal attempts by default, including failures.
`--resume --retry-failed` retries only failed/time-limited requests and preserves
their old logs. Use a new output directory when changing search settings. The
new report supports partial/error-containing campaigns; the legacy CSV reporter
assumes complete successful rows and should not consume these live CSV files.

## Restart validation and live run

All five active host tests passed normally and under ASan/UBSan. The six existing
SRBench protocol tests passed in rack1's existing OdeFormer Python environment;
they check exact prepared arrays against the frozen reference, including noise
and black-box scaling. New tests exercise the actual CPU CLI, winner decoding,
safe resume, changed-configuration rejection, tampered-data rejection, timeout
retention, and constant-target R².

Official dataset payloads and manifest CSVs were selectively restored from the
verified Ada hardware-move backup into an isolated rack1 experiment directory.
No tools, dependencies, or source builds were installed as a restoration workaround.

The one-generation integration pass completed **116/116 Feynman jobs, zero errors**
on rack1's two RTX 5080 GPUs. Every job retained full protocol rows and passed
native CPU audit plus coefficient/permutation replay. It evaluated 29,696 AST
occurrences / 7,602,176 configurations. Four reached test R² > 0.999, but this was
an integration check, not a recovery-quality measurement. Summed process time was
52.29 seconds across concurrent workers.

The repaired baselines run independently of the interactive assistant:

- tmux host: `mac1`; sessions `secant-srbench-rack1-v2` and `secant-srbench-ada-v2`.
- Rack1: two RTX 5080s share a queue, each running a complete independent dataset/
  seed search. Seed 23654, 116 jobs, root
  `/home/cdurham/experiments/secant-sr-toggle-20260919/srbench-baseline-feynman-v2/`.
- Ada: one RTX 4090, seed 15795, 116 additional jobs, root
  `/home/cdurham/experiments/secant-srbench-toggle-20260919/baseline-feynman-seed15795/`.
- The machines use the same native executable and search configuration. The
  seeds/splits differ; this adds quality coverage, not a paired GPU speed comparison.
- This is **parallelism across full searches**. One population is not split over
  GPUs, and the hosts do not share a cross-machine dynamic queue.
- Local log/results: `scratch/srbench-toggle-20260919/`, including `rack1-v2.log`
  and `ada-v2.log`. Each tmux script copies results to mac1 at completion and
  writes an exit-status file. The current tmux/SSH connections need mac1 online.
- Feynman and black-box prepared manifests exist on rack1. The 122 black-box jobs
  have **not** been launched or GPU-validated by this restart.

## Score-audit repair during the first recovery pass

The initial rack1 campaign was stopped and preserved at 58 terminal requests:
56 scored and two rejected by the CPU replay audit. Two more in-flight native
processes could finish into their raw logs while dispatch was paused; they are
not counted as completed campaign records. The original rejected results remain
rejected, and all jobs are rerun in fresh v2 campaign directories. No old and new
records are silently combined.

The original check compared `abs(gpu_sse - cpu_sse)` with
`2e-4 * (1 + gpu_sse)`. Near an accurate fit, a tiny f32 prediction difference can
be a relatively large fraction of the remaining error. Repeated searches produced
concrete rejected exp/log expressions; independent CPU and native GPU materialization
on all 10,000 training rows established:

| Reproduced case | CPU prediction SSE (f64 sum) | GPU prediction SSE (f64 sum) | Max scaled row difference | Prediction RMSE / target RMS |
| --- | ---: | ---: | ---: | ---: |
| Feynman III.7.38 | 0.110065821 | 0.109371405 | 9.02e-7 | 3.08e-7 |
| Feynman II.34.29b | 441.397527 | 441.516198 | 3.84e-7 | 1.21e-7 |

GPU prediction SSE also matched the independent fixed-AST scoring kernel.
These reproduced rejections came from f32 transcendental differences, not
incorrect winner reconstruction. Original failed candidates lacked retained
AST diagnostics, so they were not retroactively accepted.

The repaired final audit compares the RMSE gap on
`max(1, target RMS, CPU residual RMSE, GPU residual RMSE)`, with tolerance 2e-5.
It rejects nonfinite scores and material numerical discrepancies. This is a
numerical score check, not a proof of per-row identity; AST/index reconstruction
is checked separately. CPU train/test metrics remain authoritative, and every
accepted report now retains both SSEs, the measured gap, scale and tolerance in
`score_audit`. GPU arithmetic, scoring and GP selection were unchanged. Failed
audits now retain the expression, resolved/genotype bytes and coefficients.

The recorded cases and pure-helper rejection/acceptance tests are regression
coverage. All five active tests passed normally and under ASan/UBSan after the
repair. The full rack1 one-generation integration pass was repeated: 116/116,
zero errors. Ada's second-seed integration also passed 116/116 with zero errors
before its baseline started (three initial-population numerical successes).
The diagnostic is reusable with `toggle/build_cuda.py --entry toggle/tests/score_probe.c`;
it materializes predictions using both evaluators and compares their SSEs.
[Captured expressions, data hashes and row evidence](data/srbench-score-audit-20260919.json).

## Material differences and remaining work

The subsequent native-toggle coefficient-refinement pilot completed 72 fits with
zero execution/replay errors. Held-out numerical successes were 12/24 without
fitting, 13/24 with one round, and 12/24 with four rounds. The fresh 116-dataset,
seed-23654 comparison is complete: 88/116 numerical successes without fitting
versus 84/116 with one round, zero execution errors across 232 fits. See the
[completed comparison](srbench-refinement-20260919.md).
These new runs are separate from the historical baselines below.

Both initial v2 campaigns have now completed without errors: rack1 seed 23654
has **88/116** held-out numerical successes at **15.75 s mean process time**;
Ada seed 15795 has **85/116** at **15.17 s**. Combined: 173/232 (74.6%), with
symbolic equivalence unassessed. These runs precede the selector-reuse compiler
change. The higher partial-run percentages were biased by dataset ordering: the
harder final test group had not completed yet.

### Correction: later historical campaigns

The first comparison quoted the August 3–4 staged-search baselines (61.2% and
73.9% individual-trial success), overlooking later archived campaigns. That was
an incomplete account of the previous system's quality. Restored result CSVs show:

| Historical settings campaign | Successful trials | Mean recorded fit time | Budget / policy |
| --- | ---: | ---: | --- |
| August 9 scientific maturity | 942/1,160 (81.2%) | 9.17 s | 20 s cap, random coefficient refinement |
| August 15 scientific maturity + LM | 1,050/1,160 (90.5%) | 21.71 s | 60 s cap, LM |

Those more developed policies outperformed the new toggle baseline in numerical
recovery. Different hardware, trial counts, budgets and coefficient optimization
prevent attributing that difference to settings versus toggles. [Archived summary
records and source paths](data/settings-later-baselines-20260919.json) retain the
individual-trial counts separately from the official median-of-ten aggregate.

### Matched search campaign including the selector-reuse experiment

`python/compare_search_policies.py` selects 12 datasets randomly, independent of
their results, and uses both prepared official seeds (24 dataset/seed cases).
All three arms run sequentially on rack1 GPU 1, alternating their order, with
the same data bytes, population 8,192, scientific operators, training NMSE target
1e-6 and nominal 30-second limit. The 1,000-generation safety cap keeps time the
principal budget. Full 10,000 training and 25,000 held-out rows are retained.

Arms: archived maturity/settings with random coefficient refinement (no LM),
the prior toggle executable, and the identical toggle controller with native
selector reuse. The reuse writer is now an archived experiment; default source
was restored to the before arm without changing any frozen campaign executable.
Only the two toggle arms isolate the compiler change. The
settings policy has different literals, tree limits, refinement and batching;
this is a matched-time comparison of systems, not a claim of equivalent genomes.

One-generation warmups per input dimension/arm are separately retained. Settings
uses its existing SQLite template cache; toggles uses the existing NVRTC internal
cache. Time and outputs for all warmups are excluded from scored-trial totals but
remain visible. The archived executable was restored from the verified Ada backup;
no settings compatibility layer was added to current Secant.

The background tmux session on mac1 is `secant-search-comparison`; keep mac1 online.
Remote results live under
`rack1:/home/cdurham/experiments/secant-selector-reuse-20260919/policy-comparison/`.
The local log is `scratch/selector-comparison-20260919/run.log`. The wrapper copies
results to mac1 at completion. `plan.json` freezes binaries, data and selection;
`progress.json` and `results.json` retain partial status. Failures stop the panel
and remain recorded; they are never silently excluded or accepted as partial fits.

This is the new toggle GP, not the historical maturity/QD/LM search. It uses fixed
sampled banks, four slots, a 31-logical-node cap, and the explicitly recorded
operator set. Some historical formulas need larger trees or more precise
coefficients; no claim is made that every truth is exactly reachable. All 116
datasets remain in the evaluation rather than filtering by representability.
The source/binary hashes, budgets and row counts make this a reproducible new
baseline, not an equal-policy timing comparison with older SRbench results.

Bounded compiler reuse improved the shared-selector fixture, but was removed
from the default at the user's request. Its [paired measurements](../../secant/bench/settings_vs_toggles/report-selector-reuse-20260919.md)
describe the archived experiment, not the restored default; random initial GP
populations showed no meaningful GPU scoring improvement. Default tuning should
assume unrelated ASTs and keep selected values temporary.
The runner records timings without treating raw configuration count as unique
models or GPU saturation. Full configuration grids are still written within the
bounded device pool, then reduced to compact winners.

Remaining original items:

2. The first two SRBench baselines are complete. Analyze the matched search
   comparison, expand to official multi-seed/noise panels, validate black-box
   hardware limits, isolate symbolic assessment, and restore persistent template
   reuse. Other legacy Python bindings remain separate work if needed.
3. Tune arbitrary AST packs and transient selector registers, without similarity
   packing or retained cross-AST selections. Keep the shared-selector slowdown
   and larger-packing register/local-memory costs visible; the reuse experiment
   is archived, not enabled. Measure full search throughput as well as kernels.
4. Native-toggle random refinement and GPU LM are implemented. The new
   [register-only LM path](toggle-lm-20260921.md) feeds fitted toggle genomes into
   evolution after native rescoring. Its 84-fit, 900-second diagnostic campaign
   completed: [52/84 numerical successes, 29/84 strict successes](toggle-lm-results-20260922.md),
   with zero execution errors and zero local bytes throughout. Quality remains
   below the historical LM panel despite a larger allowance; establish matched
   policy parity. Tune fitting diversity, budgets, starts, and parameter capacity;
   SASS derivative specialization remains future work.
