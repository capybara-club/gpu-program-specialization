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

# SRBench v2.0 Ground-Truth Protocol

`python/run_suite.py --protocol srbench-v2-groundtruth` reproduces the
published SRBench v2.0 ground-truth data and assessment path. The contract is
frozen from tag `v2.0` (`e5ded4715ed5721703353d3500a2fdb99004faf1`) rather
than inferred from the changing `master` branch.

## Data Contract

For every Feynman dataset and trial, the compatibility path:

1. Seeds NumPy and the search with the same official trial seed.
2. Creates the seeded 75/25 train/test split.
3. Samples 10,000 training rows with replacement when the complete dataset
   contains more than 10,000 rows. This intentionally preserves v2.0's
   replacement-sampling behavior.
4. Leaves the complete 25,000-row Feynman test split held out.
5. Does not standardize inputs or targets because the published command uses
   `-sym_data`.
6. Adds Gaussian noise only to the training target, with standard deviation
   `target_noise * rms(training_target)` after subsampling.

The official seeds are:

```text
23654 15795 860 5390 16850 29910 4426 21962 14423 28020
```

The official target-noise levels are `0`, `0.001`, `0.01`, and `0.1`.
The all-Feynman mode also verifies the 116-name v2.0 manifest against SHA-256
`754fb2a58a1bde27ef2f144f12aca7904b38746cfcc981626947f48a480e2988`.

Secant evaluates binary32 data because its public AST and kernels are f32.
SRBench's Python estimators ordinarily receive binary64 NumPy arrays. This is
an intentional backend characteristic and the remaining numerical difference
from the original data path.

## Search And Test Isolation

Candidate fitness, selection, and early stopping use only the training rows.
The held-out test fold is evaluated for reporting and is not used to choose a
candidate or stop a campaign. Before this protocol was added, the Secant-SR
application selected on training fitness but stopped when held-out R2 crossed
the threshold. Results produced by that older behavior are not
protocol-correct SRBench measurements.

The CUBIN skeleton cache may be warm. Skeleton construction is treated as
backend initialization in the same way that an installed native library or a
warmed Julia environment is not rebuilt for every fit. Runtime SASS
specialization, module loading, and kernel execution remain inside the search
request. Dataset preparation and symbolic post-processing remain outside the
recorded fit time, matching SRBench's separation of `fit()` from assessment.

## Metrics

Numerical accuracy uses SRBench's clean-data threshold:

```text
test R2 > 0.999
```

Symbolic assessment follows the released v2.0 implementation:

1. Parse the candidate and ground truth with SymPy.
2. Round floating constants to three decimal places; values below `1e-4`
   become zero.
3. Simplify the candidate.
4. For candidates with test R2 above `0.5`, compute candidate error and the
   candidate/ground-truth fraction.
5. Count a symbolic solution when the error is zero or constant, or the
   fraction is constant.

The released scorer does not explicitly simplify the fraction before asking
whether it is constant. The compatibility implementation preserves that
behavior rather than silently changing published solution semantics.

The released symbolic assessor has no internal simplification timeout. The
official scheduler invocation isolates assessment jobs with a one-hour limit
and 8 GiB memory limit instead. Pass `--skip-symbolic-assessment` to checkpoint
all numerical fit results without running SymPy inline. This leaves the
symbolic columns empty rather than counting unassessed candidates as failures.
When all Secant candidates were assessed sequentially in one Python process,
the unmodified symbolic path consumed one CPU-hour and approximately 8.7 GB of
resident memory without terminating. Symbolic recovery therefore remains
unreported until assessment is run with equivalent per-trial isolation.

## Clean Feynman Campaign

```bash
CUDA_MODULE_LOADING=EAGER PYTHONPATH=python \
scratch/benchmark-venv/bin/python python/run_suite.py \
  --protocol srbench-v2-groundtruth \
  --backend cubin-staged \
  --suite feynman-all \
  --pmlb-root scratch/pmlb-source \
  --srbench-root scratch/srbench-source \
  --target-noise 0 \
  --population 8192 \
  --generations 100 \
  --leaf-settings 4096 \
  --dynamic-leaves 8 \
  --dynamic-generations 5 \
  --kernels 64 \
  --asts-per-kernel 32 \
  --static-tile-rows 1024 \
  --patch-instructions-per-ast 64 \
  --constant-settings 0 \
  --final-cpu-optimize 0 \
  --operator-profile broad \
  --persistent-process \
  --skip-symbolic-assessment \
  --resume \
  --output scratch/suite/srbench_v2_feynman_noise0.csv
```

Omitting `--seeds` selects all ten official seeds. A shorter protocol smoke
may pass a subset explicitly. Run the same command separately for the other
three target-noise levels so each persistent batch key remains unambiguous.

Summarize one or more result files with:

```bash
PYTHONPATH=python scratch/benchmark-venv/bin/python \
  python/srbench_v2_report.py scratch/suite/srbench_v2_feynman_noise*.csv
```

The search policy is Secant-SR's current algorithm-specific configuration;
SRBench does not impose one common population or expression-evaluation budget
on every method. An apples-to-apples claim therefore requires reporting both
the SRBench metrics and observed fit time, not only comparing recovery rates.

## Clean Numerical Result, 2026-08-03

The command above completed all 1,160 clean-data fits: 116 equations by ten
official seeds. Every trial used 10,000 sampled training rows and the complete
25,000-row held-out test split.

| Measure | Result |
|---|---:|
| Median-trial `R2 > 0.999` across datasets | 59.91% |
| Individual trials with `R2 > 0.999` | 710 / 1,160 (61.21%) |
| Datasets with majority trial success | 69 |
| Datasets tied at five successful trials | 1 |
| Total recorded fit time | 2,399.383 s |
| Mean recorded fit time per trial | 2.068 s |
| Median recorded fit time per trial | 3.077 s |

SRBench computes the published numerical aggregate by taking the median of
the ten per-trial accuracy indicators for each dataset, then averaging those
116 dataset values. Applying that same aggregation gives 59.91% for Secant-SR.
The published clean-data values include AFP_FE at 59.05%, GP-GOMEA at 71.55%,
AIFeynman at 78.51% over 114 reported datasets, Operon at 86.21%, and MRGP at
93.10%. These are protocol-compatible recovery values, not equal-time results;
the published algorithms use method-specific search configurations. Exact
symbolic recovery is not reported for this Secant-SR campaign because the
released v2.0 SymPy assessment did not terminate.

## Larger Search Result, 2026-08-04

The larger campaign retained the same protocol and search policy but increased
the population to 131,072 and the generation limit to 200:

```bash
CUDA_MODULE_LOADING=EAGER PYTHONPATH=python \
scratch/benchmark-venv/bin/python python/run_suite.py \
  --protocol srbench-v2-groundtruth \
  --backend cubin-staged \
  --suite feynman-all \
  --pmlb-root scratch/pmlb-source \
  --srbench-root scratch/srbench-source \
  --target-noise 0 \
  --population 131072 \
  --generations 200 \
  --leaf-settings 4096 \
  --dynamic-leaves 8 \
  --dynamic-generations 5 \
  --kernels 64 \
  --asts-per-kernel 32 \
  --static-tile-rows 1024 \
  --patch-instructions-per-ast 64 \
  --constant-settings 0 \
  --final-cpu-optimize 0 \
  --operator-profile broad \
  --persistent-process \
  --skip-symbolic-assessment \
  --resume \
  --output scratch/suite/srbench_v2_feynman_noise0_pop131072_gen200_20260804.csv
```

| Measure | 8,192 x 100 | 131,072 x 200 |
|---|---:|---:|
| Median-trial `R2 > 0.999` across datasets | 59.91% | 71.98% |
| Individual trials with `R2 > 0.999` | 710 / 1,160 (61.21%) | 857 / 1,160 (73.88%) |
| Datasets with majority trial success | 69 | 83 |
| Datasets tied at five successful trials | 1 | 1 |
| Datasets with no successful trials | 33 | 19 |
| Datasets with all ten trials successful | 61 | 76 |
| Total recorded fit time | 2,399.383 s | 39,455.620 s |
| Mean recorded fit time per trial | 2.068 s | 34.013 s |
| Median recorded fit time per trial | 3.077 s | 21.594 s |

The larger search improved the per-dataset success rate on 35 datasets and
left it unchanged on 81; no dataset lost threshold successes. Under SRBench's
median-of-ten aggregation, 15 datasets improved category and none regressed.
This raises numerical recovery by 12.07 percentage points at 16.44 times the
recorded fit cost. It slightly exceeds the published GP-GOMEA clean numerical
aggregate of 71.55%, while remaining below SBP-GP at 73.71%, AIFeynman at
78.51% over 114 datasets, Operon at 86.21%, and MRGP at 93.10%. These remain
protocol-compatible recovery comparisons, not equal-time comparisons.

The final `feynman_test_*` group remains a clear weakness. The larger budget
improves several members substantially, but many still consume all 200
generations without crossing the numerical threshold. Exact symbolic recovery
was intentionally deferred and is not reported for this campaign.
