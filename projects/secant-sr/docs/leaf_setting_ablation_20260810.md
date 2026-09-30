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

# Leaf-Setting Distribution and Persistence Ablation

## Question

The earlier comparison changed both the setting distribution and whether exact
settings persisted across generations. This paired 2x2 ablation separates those
effects:

| Distribution | Fixed for trial | Rotated each generation |
|---|---|---|
| Legacy 50/50 | `legacy` | `legacy-rotating` |
| Structured virtual | `virtual-bank-fixed` | `virtual-bank` |

All four policies preserve the mandatory static baseline. Rotating policies
regenerate host setting tables and update existing device allocations between
blocking generations. They do not rebuild evaluators, runners, CUBINs, modules,
or allocations.

## Workload

- RTX 5090, Release build, eager CUDA module loading.
- Six generated problems: Nguyen-1, Nguyen-5, Rational-2, Distance-2,
  Interaction-3, and Oscillator-2.
- Seeds: 23654, 15795, and 860; 18 trials per policy.
- 20-second search limit, population 8,192, and 8,192 leaf settings.
- Eight full-dynamic leaves, four mixed holes, and 25% mixed selection.
- Scientific operator profile and no final CPU optimization.
- Identical search, constant-optimizer, kernel, and runner configuration for all
  policies.

## Results

| Policy | Solved | Mean train R2 | Solved by 2 s | Solved by 5 s | Mean normalized SSE at 20 s | Setting update per generation |
|---|---:|---:|---:|---:|---:|---:|
| `legacy` | 14/18 | 0.99915443 | 12/18 | 14/18 | 8.456e-4 | 0 ms |
| `legacy-rotating` | 16/18 | 0.99972926 | 14/18 | 15/18 | 2.823e-4 | 0.327 ms |
| `virtual-bank-fixed` | 16/18 | 0.99942511 | 12/18 | 15/18 | 5.749e-4 | 0 ms |
| `virtual-bank` | 15/18 | 0.99927342 | 13/18 | 14/18 | 7.266e-4 | 1.013 ms |

The difference is concentrated in the harder problems. Every policy solved all
Nguyen-1, Rational-2, Interaction-3, and Oscillator-2 trials. Nguyen-5 and
Distance-2 separated the policies:

| Policy | Nguyen-5 | Distance-2 |
|---|---:|---:|
| `legacy` | 2/3 | 0/3 |
| `legacy-rotating` | 2/3 | 2/3 |
| `virtual-bank-fixed` | 3/3 | 1/3 |
| `virtual-bank` | 3/3 | 0/3 |

## Interpretation

This run does not support the claim that rotation alone caused the earlier
regression. Rotation improved the legacy distribution but weakened the
structured distribution. Changing the distribution helped when settings were
fixed but hurt slightly when settings rotated. The observed interaction means
that persistence cannot be selected independently of setting composition.

The rotating update cost is not large enough to explain the quality changes.
The structured generator and two device copies consumed about 1.01 ms per
generation; the legacy rotating update consumed about 0.33 ms per generation.
Across the complete campaigns, updates accounted for approximately 0.44% and
0.14% of measured search wall time, respectively.

There is also material run-to-run variance. A fresh `legacy` run produced 14/18
solves, whereas an earlier nominally equivalent run produced 17/18. The legacy
setting table itself is unchanged. Likely contributors include floating-point
atomic ordering and the way small numerical differences alter subsequent
evolutionary choices, compounded by a wall-time generation cutoff. This source
of variance has not yet been isolated.

Therefore:

1. Keep `legacy` as the default.
2. Retain all four policies as explicit ablation modes.
3. Replicate the 2x2 with more seeds before assigning a causal effect.
4. Add a fixed-generation-budget comparison alongside wall-time trajectories to
   separate search variance from throughput variance.
5. Test a mostly persistent hybrid only after replication, starting with a
   permanent legacy control core and a small rotating exploration tranche.
6. Defer ticket scheduling and learned setting replay until the distribution and
   persistence baseline is stable.

## Artifacts

The raw outputs are under:

```text
scratch/leaf_setting_ablation_2x2_20260810/
```

The directory includes final trial results, per-generation stage SSE values,
paired trial rows, full trajectories, checkpoint summaries, and aggregate
results. Reproduce the experiment with
`python/benchmark_leaf_setting_ablation.py`.

## 40-Second Follow-Up

A separate campaign doubled the per-trial limit to 40 seconds while preserving
the same seeds and all other parameters:

| Policy | Solved at 20 s within run | Solved at 40 s | Mean normalized SSE at 20 s | Mean normalized SSE at 40 s |
|---|---:|---:|---:|---:|
| `legacy` | 13/18 | 13/18 | 7.964e-4 | 6.031e-4 |
| `legacy-rotating` | 16/18 | 16/18 | 4.206e-4 | 2.850e-4 |
| `virtual-bank-fixed` | 15/18 | 15/18 | 8.572e-4 | 8.514e-4 |
| `virtual-bank` | 13/18 | 14/18 | 8.896e-4 | 7.848e-4 |

Only rotating structured settings converted an additional trial after the
20-second checkpoint. The other policies reduced error without crossing the
solve threshold. Doubling this search schedule therefore produced little solve
rate improvement; additional compute alone does not address the observed
`distance2` stagnation.

The 40-second campaign's final counts must not be compared directly with the
separate 20-second campaign as though it were a continuation. For example,
fixed legacy reached 13/18 in this execution versus 14/18 in the separate
20-second execution. The archive is monotonic within one execution, so this
reversal confirms materially different run-to-run trajectories. The valid
budget comparison is the 20- and 40-second checkpoints extracted from the same
40-second run.

The follow-up artifacts are under:

```text
scratch/leaf_setting_ablation_2x2_40s_20260810/
```
