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

# Controlled settings / toggles comparison — 2026-09-20

All nine real-population replays passed: **301,989,888 score pairs**, zero finite/invalid classification disagreements, maximum scaled RMSE gap **8.84e-07** (tolerance `2e-5`).

The original settings kernels and current toggle kernels receive exactly the same effective ASTs, bank values, permutations and rows. This measures engine behavior independently of GP decisions. The search pilot below is complete.

## Engine replay results

RTX 5080 on rack1. Each population has **8,192 AST occurrences × 4,096 configurations × 10,000 rows**. These are real captures at generations 0, 8 and 32 of three searches, not necessarily unique ASTs. Resident timings sum three-sample per-module medians; full batch timings are medians of three fresh replays, excluding setup/NVRTC and subsequent validation.

| Population | Settings kernels (s) | Toggle kernels (s) | Settings full batch (s) | Toggle full batch (s) | Production pack-8 full batch (s) |
|---|---:|---:|---:|---:|---:|
| III_7_38 · gen 000 | 0.452 | 0.214 | 0.694 | 0.265 | 0.201 |
| III_7_38 · gen 008 | 1.693 | 0.522 | 3.046 | 0.576 | 0.524 |
| III_7_38 · gen 032 | 1.910 | 0.850 | 3.897 | 0.906 | 0.851 |
| I_39_11 · gen 000 | 0.454 | 0.215 | 0.698 | 0.266 | 0.201 |
| I_39_11 · gen 008 | 1.687 | 0.511 | 3.021 | 0.564 | 0.514 |
| I_39_11 · gen 032 | 1.760 | 0.599 | 3.503 | 0.653 | 0.596 |
| I_40_1 · gen 000 | 0.455 | 0.326 | 0.699 | 0.377 | 0.206 |
| I_40_1 · gen 008 | 1.644 | 0.541 | 2.904 | 0.598 | 0.527 |
| I_40_1 · gen 032 | 1.791 | 0.791 | 3.929 | 0.851 | 0.781 |

**Matched pack-2 toggles are 1.40–3.30× faster in resident execution and 1.85–5.36× faster for a full batch.** This is evidence for these populations, not a universal speed ratio or a search-quality conclusion.

Settings used 76–140 registers/thread versus 32–37 for toggles, with no local-memory allocation reported for either. Resource-derived minimum active blocks/SM were 3–6 versus 12. These are occupancy limits, not measured achieved occupancy. Settings also constructs and uploads per-kernel slot tables; those costs are included only in full-batch timings.

## Comparison boundaries and validation

- Each backend retains its own kernel topology and input representation. This compares the old and new evaluation engines with one scheduler, not a single indexing instruction against a toggle instruction.
- The common scheduler uses serial modules and independent graph kernels within a module. Both engines use pack 2, the same reducer and event waits. Old templates adapt across 4/8/16/32 slots. Pack 2 avoids the old 32-slot ceiling without discarding or regrouping arbitrary candidates.
- Production pack 8 is a separate control, not the pure selector ablation. The old host API was adapted to provide private settings tables per kernel; its CUDA scoring kernels were preserved.
- Sampled CPU checks flagged 82 configurations per backend across the nine populations. Every flagged case passed independent GPU materialization of its fully resolved expression, with host float64 score summation. CPU/GPU math differences remain documented; this is not a claim of complete CPU agreement. The complete settings/toggle grids pass the unchanged tolerance.
- The last two I.40.1 replays overlapped another search worker on the other rack1 GPU. The replay GPU itself had no competing job. Treat full-batch timings as representative rather than isolated-host microbenchmarks.
- Search winners retain the existing production CPU audit and exact reconstruction from genotype, permutation and coefficients. Failed checks stop their shard and retain evidence; no silent retries.

## Matched search pilot — complete

**288 fits:** 12 diagnostic Feynman datasets × two official data/GP seeds × two identical-seed repeats × six arms. This is a pilot, not full official SRBench. Each case stays on one GPU for all arms. The three workers use both rack1 5080s and ada’s 4090.

| Engine | Packing | Refinement | Purpose |
|---|---:|---|---|
| Settings, common scheduler | 2 | Off / on | Old kernels with the same current GP |
| Toggles, common scheduler | 2 | Off / on | Controlled selector comparison |
| Toggles, production pipeline | 8 | Off / on | Real deployed scheduling/packing control |

Every arm uses the current GP with 8,192 candidates, 64 banks × 64 permutations, four coefficient slots, 31 logical nodes and a 60-second generation-boundary budget. Enabled refinement uses the same current Philox proposals and acceptance policy, 128 selected starts and one round. This is **not a reproduction of the old GP or old constant optimizer**. Fitting skips due to parameter capacity remain reported. Setup and deadline overruns are included in process time.

Warmups are retained separately. Binary, plugin, source and data hashes are frozen; resume rejects changes and retained failures. Repeated seeds measure sensitivity to float32 accumulation and divergent GP selection. Success means held-out R² > .999; symbolic equivalence is not checked.

- Remote root on rack1/ada: `/home/cdurham/experiments/secant-controlled-20260920`.
- mac1 tmux sessions: `secant-controlled-0`, `secant-controlled-1`, `secant-controlled-2`. mac1 must remain online.
- Local logs/checkpoints: `scratch/controlled-20260920/`; full worker output copies back on completion.
- [Harness design and commands](../bench/controlled/README.md).
- [All replay metrics and deviations](data/controlled-settings-toggles-20260920.json).

## Final search results

All **288 fits completed without execution errors** in **78 minutes 57 seconds**, finishing at **10:17:39 EDT on September 20**. The campaign evaluated **444,325,167,104 configurations**. All worker and result-copy exits were zero, and Telegram accepted the final summary at 10:17:57 EDT.

Each row covers the same 48 case/repeat combinations: 12 datasets × two official seeds × two repeats. Mean elapsed includes unsuccessful fits and per-fit setup; each fit has a 60-second generation-boundary budget.

| Backend | Fitting | Numerical successes | Mean elapsed | Configurations |
|---|---|---:|---:|---:|
| Settings, common scheduler | Off | 24/48 | 48.2 s | 34.02 B |
| Settings, common scheduler | On | 23/48 | 48.2 s | 34.57 B |
| Toggles, common scheduler | Off | 31/48 | 40.5 s | 92.14 B |
| Toggles, common scheduler | On | 25/48 | 42.4 s | 89.26 B |
| Toggles, production pipeline | Off | 30/48 | 38.6 s | 98.18 B |
| Toggles, production pipeline | On | 28/48 | 41.2 s | 96.15 B |

With fitting off, the common toggle backend gains seven successful case/repeat combinations over settings and loses none. With fitting on, toggles gain two and lose none. Under the current policy, enabling fitting loses six combinations for common-scheduler toggles and two for production toggles, with no compensating gains. This supports retaining toggles and keeping the current fitting policy opt-in while investigating the lost cases. It is not a general result that coefficient fitting is harmful.

Success classifications differ between identical-seed repeats in 0–2 of the 24 pairs per arm; resolved expression bytes differ in every pair. Those byte differences can reflect coefficient rounding or structurally different winners and are not themselves proof of different symbolic answers. The experiment does not establish deterministic search prefixes.

The old kernel backend uses the current GP/refiner here; this does not establish that the new GP beats the complete historical settings search policy. The diagnostic dataset selection and two seeds limit broader generalization.

## Remaining work

2. Matched pilot complete. Investigate the remaining repeatability differences and expand seeds/noise/black-box coverage after reviewing these results.
3. Real-population replay is complete. Profile achieved occupancy and production packing where the measurements justify it; do not infer occupancy from the register limit alone.
4. Diagnose the fitting regressions before changing its default; then test toggle-specific mutation/crossover separately. LM stays deferred.
