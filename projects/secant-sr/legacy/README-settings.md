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

# secant-sr

`secant-sr` is an experimental, high-volume symbolic-regression search engine
built around Secant's postorder AST and bulk evaluator contracts. Its purpose
is to exercise the API under realistic population mutation, specialization,
module loading, and GPU execution pressure.

The C99 core uses two resettable generation arenas. Every individual carries a
sidecar postorder node table containing subtree byte ranges, node counts,
depth, and weighted complexity. Crossover and mutation always create a child
by copying selected parent ranges into the next generation arena; parent
programs are immutable.

Fitness tracks SSE, MSE, RMSE, normalized MSE, R-squared, and a parsimony
adjusted score. The default archive partitions by complexity. Full structural
descriptors, including a 128-bit column mask and operation counts, are computed
on demand for winners and diagnostics. Multidimensional quality-diversity
selection computes the descriptors for candidates as an experimental opt-in
policy.

The campaign driver applies one target-independent search policy to built-in
analytic problems or externally prepared tabular data. Built-in training and
holdout coordinates come from separate deterministic generators. The SRBench
adapter reads official PMLB LFS payloads, reproduces SRBench's seeded 75/25
split, and passes an explicit versioned f32 train/test image to the C runner.
Secant's CPU interpreter is the oracle; when CUDA integration is available,
the same populations can be evaluated through the direct-CUBIN SSE runner.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 24
ctest --test-dir build --output-on-failure
./build/secant_sr_search --backend cpu --population 8192 --generations 50
CUDA_MODULE_LOADING=EAGER ./build/secant_sr_search --backend cubin --population 8192
CUDA_MODULE_LOADING=EAGER ./build/secant_sr_search \
  --backend cubin-dynamic-leaf --population 8192 --leaf-settings 4096 \
  --dynamic-leaves 8 --tile-rows 64
CUDA_MODULE_LOADING=EAGER ./build/secant_sr_search \
  --backend cubin-staged --population 8192 --leaf-settings 4096 \
  --dynamic-leaves 8 --dynamic-generations 5 --tile-rows 64 \
  --static-tile-rows 1024 --operator-profile scientific
CUDA_MODULE_LOADING=EAGER ./build/secant_sr_search \
  --backend cubin-maturity --population 8192 --leaf-settings 4096 \
  --dynamic-leaves 8 --mixed-dynamic-leaves 4 \
  --mixed-refine-probability 0.25 --constant-settings 8192 --streams 64
python3 python/benchmark_dynamic_leaf.py
CUDA_MODULE_LOADING=EAGER python3 python/run_suite.py --backend cubin --persistent-process
CUDA_MODULE_LOADING=EAGER scratch/benchmark-venv/bin/python python/run_suite.py \
  --backend cubin-dynamic-leaf --suite feynman-supported \
  --pmlb-root scratch/pmlb-source --constant-settings 0 --final-cpu-optimize 0 --resume
```

The `cubin-dynamic-leaf` early-search mode projects each concrete structure's
fixed column and constant occurrences into postorder dynamic slots. Every
structure is evaluated over a shared setting batch, its best setting is then
materialized back into a concrete AST, and normal elite selection, crossover,
and mutation continue on that concrete population. This keeps the stored
individual and its fitness consistent instead of retaining a hidden runtime
setting beside the AST. Without the optional local optimizer, constants come
only from the fixed vocabulary or arithmetic compositions of those values;
there is no continuous constant fitting after materialization.

The `cubin-staged` backend uses dynamic-leaf SSE only during the configured
early generations. It statically scores each source, evaluates dynamic
bindings, materializes and statically rescores each proposal, and promotes
only strict improvements. It then raises the active node limit and continues
with static SSE. One CUDA context owns both prepared evaluators plus a one-AST
materialize runner that validates each generation's held-out winner against
the CPU interpreter.

The `cubin-maturity` backend is the current search-policy development path.

Dynamic-leaf settings retain the original fixed-table policy by default. The
setting distribution and exact-setting persistence can be ablated independently
with `legacy`, `legacy-rotating`, `virtual-bank-fixed`, and `virtual-bank`.
The optional structured policies add stable mask/density controls, column-pair
coverage, and exploratory constants without rebuilding evaluators or modules.
The implemented ranges and current learned-sampler boundary are documented in
[`docs/virtual_setting_bank.md`](docs/virtual_setting_bank.md).
New random structures first resolve all eligible terminal choices through the
dynamic-only kernel. Mature structures retain their concrete columns and
constants while a sampled cohort reopens selected leaves through the mixed
static/dynamic kernel. Winning settings are always materialized and statically
rescored before promotion. Its constant stage is selectable:
`--constant-optimizer legacy` retains the packed shared-jitter baseline, while
`--constant-optimizer lm` promotes the same selected AST cohort into the
fixed-eight, thread-owned native-SASS LM runner. Structural children restart
as resolved structures, while unchanged elites preserve their maturity and
refinement history. See
`docs/maturity_search.md` for the exact flow and the frozen legacy baseline.
Pass `--generation-output <csv>` to `python/run_suite.py` to enable and record
per-generation stage attribution. The additional proposal scans and archive
snapshots are disabled when that output is not requested.

The early-search defaults use eight dynamic leaves and a 64-row static tile,
matching the high-throughput cuSR topology. Limiting the dynamic phase to eight
leaves bounds generated programs to 15 nodes; later concrete search can use the
larger ordinary-search limits. `--dynamic-leaves` can raise the bound through
32 when wider early structures are worth the additional per-setting setup.
The patch-space option is expressed as `--patch-instructions-per-ast`; the app
multiplies it by the number of ASTs packed into each kernel before constructing
the Secant recipe.

The staged search regularly refines fixed constants on the GPU. Every
generation, the default policy selects at most 4,096 eligible expressions. It
uses 75% of that budget for the highest-scoring expressions distributed across
complexity buckets and samples the remaining 25% uniformly from other eligible
expressions. `--constant-optimize-budget`, `--constant-optimize-interval`, and
`--constant-optimize-random-fraction` control this bounded policy. The legacy
`--constant-optimize-probability` option remains available for reproducing older
independent-sampling runs and overrides bounded selection when supplied.

Selected expressions receive expression-local Philox perturbations over
`--constant-settings` (default `8,192`) and are statically rescored before
promotion. The regular default performs one perturb-and-reduce iteration per
selected expression; deeper local searches remain available through
`--constant-optimizer-iterations`. The initial additive radius is set by
`--constant-optimizer-scale`. Each resident iteration multiplies that radius by
`--constant-optimizer-decay`; a decay of `1` keeps the radius constant, while the
default `0.5` geometrically anneals it. The reducer is embedded in each optimizer
skeleton. The runner specializes and loads a module
once, executes every requested evaluate/reduce/update iteration while it is
resident, and only then unloads it. Optimizer skeleton CUBINs are cached in
SQLite. This is the only generation-stage continuous optimizer.

In LM mode, `--constant-settings` is the total number of thread-owned states per
promoted AST. States are grouped as mixed column bindings times deterministic
constant starts. `--lm-starts-per-binding` defaults to four, so 128 settings
mean 32 bindings x 4 starts. Binding zero keeps every hole constant; later
bindings substitute deterministic input columns into one or more constant
holes. Start zero within every binding uses the AST's incumbent constant vector,
and the remaining starts use deterministic, multi-scale additive offsets, so
the structural search RNG is unchanged. Setting
`--lm-starts-per-binding 128` with 128 settings reproduces the earlier
one-binding, 128-start regime. `--constant-optimizer-iterations` is the number
of LM proposal evaluations. All states and solver data remain in one loaded
module for the complete schedule. `--lm-batch-asts`,
`--lm-settings-per-cta`, `--lm-tile-rows`, and `--lm-threads` control module
packing and the independent settings/row/thread topology. The winning binding
and constants are materialized together and must pass the same strict
static-SSE rescore as the legacy optimizer before entering the population.

The CPU reference L-BFGS implementation runs once on the final best expression.
It reports train and held-out scores before and after refinement, and commits
only a full-training-SSE improvement. `--final-cpu-optimize 0` disables this
final audit; the `--final-cpu-optimizer-*` options control its bounded work.

`python/benchmark_dynamic_leaf.py` exercises the complete early-search
evaluation path rather than only the Secant kernel. Its defaults evaluate one
8,192-member generation over 1,048,576 rows, 4,096 leaf settings, eight dynamic
leaves, and 64-row static tiles, then write the generation metrics to
`scratch/dynamic_leaf_1m.csv`.

`secant_sr_bench` isolates generation construction with deterministic synthetic
fitness. Its default 100,000-member run measures arena memory and single-core
crossover/mutation throughput without including evaluation.

`python/run_suite.py` persists generated skeleton CUBINs in
`scratch/cache/secant_sr_cubin.sqlite3` by default. Cache keys include the
Secant version, complete source-generator recipe, compute capability, NVRTC
version, and compiler options. Neither source nor a source digest is stored,
and a warm hit occurs before CUDA source generation.
Use `--no-cubin-cache` for cold compilation measurements or `--cubin-cache`
to select another database. Only Secant specialization belongs in the
generation hot path. `--persistent-process` sends every pending suite record
through one search process, retaining the CUDA context and SQLite connection
while rebuilding dataset-dependent evaluators and buffers. Its per-record wall
times amortize the one-time process setup across the batch. `--resume`
validates an existing output CSV and runs only missing `(problem, seed)` pairs,
making focused panels and interrupted suite runs cheap to repeat.

See [docs/sr_suite.md](docs/sr_suite.md),
[docs/srbench_feynman.md](docs/srbench_feynman.md),
[docs/architecture.md](docs/architecture.md),
[docs/staged_evaluation_plan.md](docs/staged_evaluation_plan.md),
[docs/search_quality_next_steps.md](docs/search_quality_next_steps.md),
[docs/kernel_shapes.md](docs/kernel_shapes.md), and
[docs/api_pressure.md](docs/api_pressure.md). The cache contract and measured
cold/warm impact are in [docs/cubin_cache.md](docs/cubin_cache.md).
