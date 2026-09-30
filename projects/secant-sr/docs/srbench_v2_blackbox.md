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

# SRBench v2.0 Black-Box Protocol

The black-box suite contains 122 PMLB regression datasets. They are prediction
problems, not 122 equations with known ground truth. The primary measurements
are held-out test R2, expression size, and fit time. Exact symbolic recovery
and the ground-truth-suite success percentage do not apply to this track.

## Frozen data and preprocessing

`python/run_suite.py --protocol srbench-v2-blackbox` reproduces the data path
from SRBench tag `v2.0` (`e5ded4715ed5721703353d3500a2fdb99004faf1`).
It uses the data payloads from PMLB `v1.0.1.post3` and verifies the sorted
122-name manifest against SHA-256
`af646147140f294128bc19be002f6c39016d3325466808ff056301547125cbed`.

For each dataset and each of the ten official seeds, the runner:

1. creates SRBench's seeded 75/25 train/test split;
2. when the complete dataset has more than 10,000 rows, draws 10,000 training
   indices with replacement, preserving the released v2.0 behavior;
3. fits `StandardScaler` independently to the training inputs and training
   target, then transforms the input partitions;
4. scores the held-out target in standardized coordinates. This is exactly
   R2-equivalent to SRBench's inverse-transforming predictions before scoring;
5. keeps the test fold out of selection, early stopping, and final optimization.

Secant transports the prepared arrays as binary32. The released Python
estimators ordinarily use binary64 arrays; this is an intentional property of
the GPU backend and should be disclosed with the results.

The dataset payload is about 135.7 MiB compressed. The widest dataset has 124
inputs, within Secant-SR's 128-input limit. Seventeen datasets have more than
32 inputs. The current LM promotion stage is disabled for those datasets, so
their result rows must be analyzed separately when attributing gains to LM.

The mixed kernel keeps every static input available to each thread. Its source
generator uses two inline operands per static/dynamic input and target, one SSE
operand per packed AST, and one keepalive operand. The runner conservatively
reserves another 32 registers for dynamic-leaf addresses and masks, loop state,
and generated-kernel control values. It routes a dataset through the
dynamic-leaf backend when the resulting requirement exceeds the 256-register
kernel budget, or independently when static inputs plus the requested dynamic
leaves exceed the 128-input AST encoding limit. The dynamic-leaf backend still
selects from every source column at runtime but does not run the mixed or LM
promotion stages. For the campaign shape below, the capacity rule applies to
the 100-input `588_fri_c4_1000_100` and the 117-input
`4544_GeographicalOriginalofMusic`; the encoding rule applies to the 124-input
`505_tecator`. The actual backend and constant settings are recorded in every
result row.

This capacity rule was added after the first full campaign deterministically
failed while creating the 117-input mixed kernel for
`4544_GeographicalOriginalofMusic` on both tested GPU architectures; a
100-input boundary test also failed before generation zero. Completed trials
from before the amendment have at most 48 inputs and remain
protocol-compatible, so the campaign can resume without repeating them.

## Campaign shape

Use the scientific grammar for the first preregistered campaign. It closely
matches the arithmetic, trigonometric, exponential, logarithmic, square-root,
constant, and variable vocabulary in tuned Operon. Multiplication can express
square and cube. A 50-node maximum is a reasonable general black-box limit and
matches the larger model class already used in Secant-SR comparisons.

Use a fixed generation budget on both GPUs. This gives every trial the same
population and generation count; Ada will take longer rather than receiving
less search. Split the ten official seeds six on Rohini and four on Ada, then
merge the result CSVs. The full campaign contains 1,220 fits.

```bash
CUDA_MODULE_LOADING=EAGER PYTHONPATH=python \
scratch/benchmark-venv/bin/python python/run_suite.py \
  --protocol srbench-v2-blackbox \
  --backend cubin-maturity \
  --suite srbench-blackbox-all \
  --pmlb-root scratch/pmlb-srbench-v2 \
  --srbench-root scratch/srbench-source \
  --population 8192 \
  --generations 600 \
  --leaf-settings 8192 \
  --leaf-setting-policy virtual-bank \
  --dynamic-leaves 8 \
  --mixed-dynamic-leaves 4 \
  --mixed-refine-probability 0.25 \
  --dynamic-max-nodes 50 \
  --constant-optimizer lm \
  --constant-settings 128 \
  --lm-settings-per-cta 128 \
  --lm-starts-per-binding 4 \
  --constant-optimizer-iterations 4 \
  --constant-optimize-budget 512 \
  --final-cpu-optimize 0 \
  --operator-profile scientific \
  --validation-mode final \
  --stop-metric train --stop-r2 1.1 \
  --time-limit-seconds 0 \
  --persistent-process \
  --checkpoint-output scratch/suite/srbench_v2_blackbox_checkpoints.csv \
  --output scratch/suite/srbench_v2_blackbox.csv
```

Pass six official seeds to the Rohini command and the remaining four to the
Ada command. Do not run both commands against the same output or checkpoint
file. The impossible training stop threshold and zero time limit ensure every
trial receives all 600 generations.

Before the full campaign, use `--suite srbench-blackbox-smoke` and a small
generation limit to verify the complete path on each GPU. Estimate the full
runtime from a representative mix of narrow, wide, small, and 10,000-row
training problems; a six-dataset smoke alone should not be extrapolated as if
all 122 datasets had the same shape.

Summarize the merged result files with:

```bash
PYTHONPATH=python scratch/benchmark-venv/bin/python \
  python/srbench_v2_blackbox_report.py \
  scratch/suite/srbench_v2_blackbox_rohini.csv \
  scratch/suite/srbench_v2_blackbox_ada.csv \
  --official-results scratch/srbench-source/docs/csv/blackbox_results.csv
```

The `nodes` column is Secant's unsimplified AST node count. Published SRBench
`model_size` is computed after conversion through its symbolic/SymPy path, so
the two size values must not be described as perfectly identical until a
Secant expression is passed through that same postprocessor.

## Current 32-input subset

For experiments focused on the common mixed/LM regime, pass
`--dataset-max-inputs 32` to the suite runner. This verifies the complete
122-dataset frozen manifest first, then skips the 17 datasets with more than 32
input columns. The resulting scope is 105 datasets and 1,050 official-seed
fits. It is an explicitly labeled SRBench v2.0 subset, not the official
122-dataset aggregate, and wider datasets must not be included in its score or
denominator.

When reporting an output that may also contain earlier wider rows, pass
`--max-inputs 32` to `python/srbench_v2_blackbox_report.py`. The report excludes
those Secant rows and restricts published comparator aggregates to the same
represented dataset names.
