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

# SRBench and Feynman Experiments

> **Protocol note:** the historical recovery runs below selected candidates by
> training fitness but used held-out R2 for early stopping. They are evaluator
> and search-policy experiments, not apples-to-apples SRBench measurements.
> The corrected v2.0 protocol and current reproduction command are documented
> in [srbench_v2_protocol.md](srbench_v2_protocol.md).

## Data contract

These experiments use the official PMLB Git LFS payloads rather than
independently resampling the published equations. `python/srbench_data.py`
uses the same pandas ingestion and seeded 75/25 `train_test_split` calls as
SRBench. When a training-row limit is requested, it also reproduces SRBench's
seeded `numpy.random.choice` subsample. A validation-row limit, when nonzero,
takes a deterministic prefix of the already shuffled test split.

The adapter writes a versioned little-endian file containing column-major f32
training inputs, training targets, validation inputs, and validation targets.
The C runner validates the magic, version, dimensions, finite values, exact
file length, and problem input count before evaluation. The f64 PMLB values
are intentionally converted to f32 because every Secant operation in this
prototype is f32.

Feynman formula-recovery runs do not standardize inputs or targets, preserving
the published symbolic relationship. This differs from SRBench's default
black-box preprocessing. The black-box runs below use `--scale-x --scale-y`;
validation targets are transformed with the training target scaler because
Secant scores in scaled space. R-squared is invariant to that affine target
transformation.

## Search without local optimization

The no-optimizer path is not a constant-free search:

1. GP creates a concrete postorder structure using the fixed constant
   vocabulary and input columns.
2. Dynamic-leaf evaluation projects each structure into eight runtime leaf
   slots and scores it over a shared batch of column-or-constant settings.
3. The best setting for each structure is written back into a concrete AST.
4. Fitness, complexity-bucket elites, crossover, and mutation operate on those
   concrete programs.

No continuous constant fitting occurs when both `--constant-settings 0` and
`--final-cpu-optimize 0` are used.
The current dynamic-leaf campaign is limited to eight leaves and 15 total
nodes. Its constants are `-3`, `-2`, `-1.5`, `-1`, `-0.5`, `-0.25`, `0.25`,
`0.5`, `1`, `1.5`, `2`, `3`, and binary32 pi, although larger rational values
can be built from arithmetic nodes.

## Selection bias and Feynman coverage

The original ten-problem subset was intentionally selected after checking that
the equations fit the available operations and size limits. It is therefore a
useful evaluator and search smoke set, but it is biased toward compact,
representable equations and must not be presented as a random Feynman sample.
The 43-problem sweep below removes hand selection within the old dynamic-leaf
grammar, but remains conditioned on that grammar and its 15-node limit.

SRBench's current ground-truth results contain 116 Feynman datasets, so this
group is larger than the historical shorthand "Feynman-100". The reproducible
grammar audit reports:

| Classification | Datasets |
|---|---:|
| Semantic lowering uses at most 15 nodes | 56 |
| Semantic lowering uses 16-72 nodes | 60 |
| Unsupported by the current exact grammar | 0 |
| Semantic lowering uses more than 72 nodes | 0 |

The initial audit classified 42 equations as unsupported because they required
literal pi, `exp`, or `log`. Secant now represents pi as an exact f32 constant
and lowers natural `exp` and `log` through native base-2 operations. With those
extensions, the audit can lower all 116 formulas. This does not imply that the
search will recover all 116, only that none is rejected by the semantic grammar
or the 72-node limit. The audit does not yet reproduce the search policy's
weighted-complexity accounting, so its categories are structural node-count
classes rather than a guarantee that every formula is reachable under the
current generation policy. Regenerate the current coverage table with:

```bash
scratch/benchmark-venv/bin/python python/audit_feynman.py \
  --pmlb-root scratch/pmlb-source \
  --srbench-root scratch/srbench-source \
  --output scratch/suite/feynman_coverage.csv
```

A one-generation integration sweep over all 116 datasets completed through
PMLB ingestion, binary loading, CUBIN specialization, module execution, and
fitness reporting. This validates the data/backend path across the group; one
generation is not a recovery benchmark.

## No-optimizer recovery

Before the `exp`, `log`, and pi extension, a 43-problem smoke run covered every
formula then classified as dynamic-leaf eligible. It used one seed, 1,024
structures, three generations, 256 leaf settings, 1,024 training rows, and
4,096 validation rows. These measurements have not been extrapolated to the 13
newly eligible formulas.

| Metric | Result |
|---|---:|
| Numerical solves (`R2 > 0.999999`) | 23/43 |
| Median validation R2 | 1.000000 |
| Validation R2 at least 0.99 | 25/43 |

The same smoke configuration was then run on the 13 equations newly admitted
to the dynamic-leaf set by `exp`, `log`, and pi. It numerically solved 2/13,
reached validation R2 of at least 0.99 on 4/13, and had median validation R2 of
0.963338. Generated candidates exercised all three additions. This establishes
that the lowering and search integration work; it does not make the added
equations easy to recover.

On the original ten-problem supported subset, the same three-generation smoke
solved 7/10. A larger run with 8,192 structures, 4,096 settings, 10,000
training rows, and the complete 25,000-row test split exactly recovered the
two previously missed cosine equations. Across those configurations, recovery
has been demonstrated for 9/10 selected equations.

The remaining equation is relativistic velocity addition:

```text
(u + v) / (1 + u*v/(c*c))
```

It requires only supported operations, constant `1`, seven leaves, and 15
lowered nodes. Three 100-generation no-optimizer runs produced validation R2
values of `0.969232`, `0.971715`, and `0.969789`, converging to simpler
`min`/`tanh` surrogates. This is a search-policy failure rather than an
evaluator, grammar, constant, or size limitation.

These are numerical solves. Exact symbolic-equivalence scoring with SymPy is
not yet integrated, and some exact numerical expressions contain redundant
identities such as `max(x, negative_constant)` on a positive input domain.

## Black-box smoke

Six datasets from SRBench's 122-dataset black-box result set were run with
standardized inputs and targets, one seed, 8,192 structures, 50 generations,
4,096 leaf settings, up to 10,000 training rows, and the complete test split.

| Dataset | Inputs | Train rows | Test rows | Validation R2 |
|---|---:|---:|---:|---:|
| 1027_ESL | 4 | 366 | 122 | 0.848187 |
| 1029_LEV | 4 | 750 | 250 | 0.577646 |
| 201_pol | 48 | 10,000 | 3,750 | 0.614879 |
| 227_cpu_small | 12 | 6,144 | 2,048 | 0.934699 |
| 503_wind | 14 | 4,930 | 1,644 | 0.703345 |
| 505_tecator | 124 | 180 | 60 | 0.975385 |

These runs establish that the current dynamic-leaf path works from 4 through
124 input columns and performs useful search. They are not leaderboard-complete
SRBench results: the time budget, number of trials, model simplification,
symbolic scoring, and hyperparameter protocol have not yet been matched.

## Reproduction

```bash
# Exact-row supported Feynman suite, no local constant optimizer.
CUDA_MODULE_LOADING=EAGER scratch/benchmark-venv/bin/python python/run_suite.py \
  --backend cubin-dynamic-leaf \
  --suite feynman-supported \
  --pmlb-root scratch/pmlb-source \
  --population 8192 \
  --generations 100 \
  --rows 10000 \
  --validation-rows 0 \
  --leaf-settings 4096 \
  --constant-settings 0 --final-cpu-optimize 0

# Current SRBench Feynman-group integration sweep.
CUDA_MODULE_LOADING=EAGER scratch/benchmark-venv/bin/python python/run_suite.py \
  --backend cubin-dynamic-leaf \
  --suite feynman-all \
  --pmlb-root scratch/pmlb-source \
  --srbench-root scratch/srbench-source \
  --population 256 --generations 1 --rows 257 --validation-rows 257 \
  --leaf-settings 64 --constant-settings 0 --final-cpu-optimize 0

# Frozen SRBench v2.0 black-box sample.
CUDA_MODULE_LOADING=EAGER scratch/benchmark-venv/bin/python python/run_suite.py \
  --protocol srbench-v2-blackbox \
  --backend cubin-maturity \
  --suite srbench-blackbox-smoke \
  --pmlb-root scratch/pmlb-srbench-v2 \
  --srbench-root scratch/srbench-source \
  --population 8192 --generations 50 \
  --leaf-settings 8192 --constant-settings 128 \
  --constant-optimizer lm --final-cpu-optimize 0
```

See `docs/srbench_v2_blackbox.md` for the full protocol and campaign command.
