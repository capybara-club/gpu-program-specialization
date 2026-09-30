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

# Population, time budget and toggle variation trial — 2026-09-19

**Completed:** all 1,475 fits, zero execution errors, in 4 h 8 min.
See the [September 20 results and recommendations](search-population-results-20260920.md).

The experiment uses the same prepared 116 Feynman datasets with two official
seeds (23654 and 15795): 232 dataset/seed cases, each with 10,000 training and
25,000 held-out rows. No formula is supplied to search. Numerical success means
held-out R² > 0.999; symbolic equivalence is not assessed.

## Questions and controls

1. Does doubling the fit budget from 30 to 60 seconds improve recovery?
2. Does a population of 32,768 or 65,536 beat the 8,192-member baseline at the
   same time budget, despite allowing fewer evolutionary generations?
3. Do toggle-aware edits improve search relative to ordinary subtree crossover
   and point/subtree mutations?
4. How does the archived settings system perform on the exact same prepared
   data and GPUs?

| Coverage | Policies | Population | Budget |
| --- | --- | ---: | ---: |
| All 232 cases | Existing variation | 8,192 and 32,768 | 30 and 60 s |
| All 232 cases | Combined new variation | 32,768 | 60 s |
| Diagnostic 24 cases | Combined new variation | 8,192 | 60 s |
| Diagnostic 24 cases | Existing and combined new variation | 65,536 | 60 s |
| Diagnostic 24 cases | Alignment only, targeted mutation only, leaf mixing only | 8,192 | 60 s |
| Rack1 cases only | Archived settings + random coefficient refinement | 8,192 | 60 s; also 30 s on diagnostic cases |

The diagnostic panel reuses the previously sampled 12 datasets at both seeds;
it is not selected using this experiment's outcomes. Diagnostic cases run first.
Every case's policies stay on one GPU; policy order rotates. Both rack1 RTX 5080s
and Ada's RTX 4090 run independent fits. Planned work is 524, 518 and 433 fits,
respectively: **1,475 measured fits**, excluding retained warmups. Each worker
has an eight-hour dispatch window and may finish sooner. It reserves a complete
case's nominal budget plus margin before starting it; unstarted work is reported.
Generation-boundary budget checks can overrun. A hard subprocess watchdog
prevents an individual fit from hanging indefinitely.

All current arms retain 64 constant-bank rows × 64 toggle configurations, four
coefficient slots, the scientific operator set, 31 logical nodes, and a 10,000
generation safety ceiling. Refinement is disabled to isolate population/time/
variation effects. There is no LM. Configurations per generation are 33,554,432,
134,217,728 and 268,435,456; these include duplicates and unused selector bits.
They are not counts of unique mathematical expressions. Full row counts remain
unchanged. GPU scoring chunks remain bounded to 4,096 ASTs and a 256 MiB score
buffer; population growth does not mean increasing the kernel's register demand.

## New optional variation

The existing search already mutates alternatives inside toggles and their bit
assignments. Its defaults remain unchanged, including RNG consumption when the
new options are zero. Version 0.3.2 adds three host GP controls:

- `align_crossover_bits`: remap a transplanted subtree's selector bits, prefer
  bits unused outside the transplant, and permute alternatives to retain the
  donor's selected leaf identities under the recipient's winning permutation.
  Mapping remains injective, so donor internal correlations and combinations
  are preserved. With no free bits, external coupling is unavoidable and counted.
- `toggle_mutation_probability`: probability within mutation events of replacing
  an inactive alternative, changing selector width among 1/2/4, or rewiring bits.
  These edits retain the recipient's selected configuration. Other configurations
  can improve or disappear; ordinary exploratory mutations remain available.
- `leaf_mix_probability`: probability within crossover events of placing a donor's
  selected leaf into an inactive alternative of a recipient leaf, promoting a
  fixed leaf to a toggle when needed. This keeps the recipient winner available.

The combined policy uses alignment on, targeted mutation probability 0.5 and
leaf mixing probability 0.25. These are conditional probabilities, not fractions
of the entire population. The ordinary 0.45 crossover / 0.45 mutation / 0.10 new
random tree branch split remains. Counters report accepted valid offspring;
neutral edits and duplicates are possible. No deduplication claim is made.

Alignment preserves selected donor **leaf identities**, not donor fitness or
coefficient values when recipient and donor use different bank rows. Recipient
preservation tests use its original bank row and permutation. No related-AST
packing, cross-AST selector caching, evaluator changes or kernel changes were
introduced for this trial.

Research motivating the trial includes [GP-GOMEA's learned dependencies and
substructure mixing](https://arxiv.org/abs/1904.02050), [coefficient mutation
studied alongside mixing](https://arxiv.org/abs/2204.12159), and the independent
feature/operator/constant/structural mutation controls in
[SymbolicRegression.jl](https://ai.damtp.cam.ac.uk/symbolicregression/dev/api).
The engineering inference is to preserve useful selector combinations while
retaining structural exploration, then measure each operator separately. This
implementation is **not** GOMEA, learned linkage, or semantic crossover; those
methods would require additional fitness sampling and accounting.

## Validation before launch

- Nine local test targets pass normally and with ASan/UBSan.
- New tests exhaustively check selector mapping and available-bit exhaustion,
  donor combination preservation, and CPU fitness at the retained parent
  configuration after targeted mutations and leaf mixing.
- A four-generation CPU control matches the frozen 0.3.1 executable exactly in
  genotype, resolved AST, coefficients, permutation, scores and work counts.
- Both GPU types pass the 65,536-member, two-generation combined-policy test,
  including independent CPU score audit and exact winner reconstruction.
- The archived settings binary passes its CPU/GPU materialization guard with
  final CPU optimization disabled.

The large-population smoke test on Feynman I.39.11 evaluated 536,870,912
configurations over 10,000 rows/configuration:

| GPU | Scoring wall time | Process time including setup | NVRTC |
| --- | ---: | ---: | ---: |
| Rack1 RTX 5080 | 3.279 s | 7.023 s | 3.394 s |
| Ada RTX 4090 | 2.539 s | 6.472 s | 3.397 s |

These are two-generation integration checks, not a population-size performance
comparison or an occupancy measurement. Winner scores validate on both GPUs;
floating-point differences can change subsequent parent selection.

## Rough previous-system comparison

| Run | Held-out successes | Mean recorded time | Qualification |
| --- | ---: | ---: | --- |
| Current toggle, no refinement, seed 23654 | 88/116 | 15.61 s process | 30 s budget, mixed GPU shards |
| Archived August random refinement, seed 23654 | 96/116 | 9.06 s | 20 s budget, different hardware/timing boundaries |
| Archived August LM, seed 23654 | 104/116 | 22.86 s | 60 s budget, different hardware/timing boundaries |
| Previous matched 24-case settings trial | 17/24 | 15.97 s process | Same data/GPU and nominal 30 s as next row |
| Previous matched 24-case toggle trial | 13/24 | 21.05 s process | Earlier toggle policy, before this change |

The historical settings policies currently have better numerical recovery.
The full-suite historical times are **not** evidence of a matched speed ratio.
The 24-case comparison matches data and hardware but still compares different GP
and fitting policies. Original references are recorded in
[the historical audit](data/settings-later-baselines-20260919.json) and
[the completed current refinement comparison](srbench-refinement-20260919.md).

The fresh old-system arm is random refinement, not the archived LM campaign.
It has different tree bounds, constant fitting and persistent SQLite template
caching. It retains its archived 1,000-generation ceiling; current runs use
10,000. Stop reasons and actual generation counts are retained, so any fits
that hit that ceiling must be separated from time-budget comparisons.
Native toggle runs compile templates per fit; that real setup cost is
included in measured process time. Warmups are retained separately and excluded
for both engines; they do not eliminate native per-fit NVRTC. The cold archived
one-generation validation took 32.45 seconds; campaign warmup precedes measured
old-system fits. Old logs do not provide comparable configuration counts, so the
report leaves those unknown rather than inferring them.

Ada lacks the archived binary's `libnvrtc.so.13` dependency. Settings comparisons
run only on rack1; Ada uses the self-contained current binary. No dependencies
were installed and there is no alternate evaluator fallback.

## Operation and retained evidence

Runner: `python/search_overnight.py`. It freezes executable, manifest and adapter
hashes, uses a worker lock, atomically checkpoints every trial, retains stdout/
stderr and winner records, and stops the affected shard on validation or execution
failure. Identical-hash resume skips finished work; failures require investigation.

Remote root on both hosts:
`/home/cdurham/experiments/secant-search-night-20260919/`.
Local root: `scratch/search-night-20260919/`.
Worker folders are `shard0`, `shard1`, `shard2`; inspect their `progress.json` and
`plan.json`. Local tmux sessions are `secant-search-night-rack1-0`,
`secant-search-night-rack1-1`, and `secant-search-night-ada-0`.
Tmux runs on **mac1**, where it is installed, with SSH workers on the GPU hosts.
**Mac1 must remain online.** Completion scripts copy results back to mac1 and
retain exit codes. The GPUs share host CPU resources, so this measures search
under the intended concurrent deployment, not isolated kernel peak throughput.

All three workers launched at 20:24 EDT on September 19. The initial check
confirmed 14 measured fits completed, zero errors, with every shard running.
This is a startup snapshot, not an outcome comparison. The source snapshot and
binary are retained on both GPU hosts; hashes and the preflight timing/audit
evidence are recorded in [validation results](data/search-population-night-20260919.json)
and the local `launch.json`.

## Remaining work

2. Analyze matched quality/time comparisons; extend to additional seeds, noise
   and black-box data. Add persistent native template caching.
3. Measure arbitrary-AST register pressure, packing, occupancy and full pipeline
   throughput. Separate population/search benefits from kernel throughput.
4. Use ablations to select a toggle variation policy; revisit coefficient capacity,
   starts and fitting policy after structural-search effects are understood.
   Refinement remains opt-in and LM remains deferred.
