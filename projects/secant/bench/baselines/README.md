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

# Runtime Baselines

These collectors compare expression evaluation after each system has already
constructed its programs. Every output uses the same normalized CSV schema and
the versioned corpus in
[`../corpus/portable_alu_v1.json`](../corpus/portable_alu_v1.json), so new
interpreters and dispatch engines can be added without duplicating expression
definitions.

The default workload contains 1,024 unique balanced, eight-leaf ALU
expressions. Additional checked-in profiles cover unary special functions,
variable tree sizes, constants, and protected math. Cases are consumed in
order and repeat only when a run requests more ASTs than the selected corpus
contains. Every collector uses the same stateless `(column, row, seed)` input
generator and corpus expression 0 as the target. CSV records include
`corpus`, `corpus_hash`, and `seed`; `comparison_graph.py` refuses to combine
mismatched records.

Every Python collector accepts `--corpus`. See
[`../corpus/README.md`](../corpus/README.md) for generating much larger JSON
corpora and for the additional header-generation step used by SECANT and the
native AVX baseline.

`row_evals_per_second` counts one complete AST evaluated for one row.
`asts_per_second` is the number of complete AST evaluations per second, not
search mutations or compilations.

## SECANT

`secant_sweep.py` runs the direct-CUBIN, CUDA, and PTX backends through
`secant_runtime_bench`. The executable verifies every kernel against the CPU
interpreter before CUDA events time kernel launches only. Streams and events
are created before timing. The collector adapts the number of iterations to
each row count and retains every requested backend, AST-packing, SSE-tile, and
stream configuration in the CSV.

```sh
python3 bench/baselines/secant_sweep.py \
  --backends cubin cuda ptx \
  --streams 1 2 8 16 \
  --output docs/baseline_secant.csv \
  --samples-output docs/baseline_secant_samples.csv
```

The aggregate CSV stores one median per configuration. The optional sample CSV
stores every probe, calibration retry, and accepted timed process run, including
the exact timed batch, kernel-launch, and row-evaluation counts. The collector
rejects a result if these counts disagree with the requested shape, if inputs
were not resident on the GPU, or if transfers entered the timed interval.

## Native AVX2

`native_avx.c` is a C99 AVX2/FMA implementation with no Python or Julia in its
timed region. It vectorizes rows in groups of eight and uses OpenMP across ASTs.
The materialize path writes every AST result. The SSE path computes one
reduction per AST. A probe calibrates enough complete evaluations to make each
timed sample long, and a compiler memory barrier after every evaluation prevents
repeated output writes from being collapsed. The CSV notes record the actual
iterations per sample.

```sh
cmake --build /home/cdurham/code/build --target secant_native_avx_bench -j 24

python3 bench/baselines/native_avx_sweep.py \
  --output docs/baseline_native_avx.csv
```

## PySR

PySR delegates expression execution to Julia's `SymbolicRegression.jl`.
`pysr_worker.jl` calls its low-level `eval_tree_array` and `eval_loss`
functions directly, excluding evolutionary search and Python-to-Julia call
overhead. This is specifically a benchmark of the Julia execution backend, not
PySR search. The Python collector translates the shared corpus to Julia
expressions, starts one Julia process per requested thread count, and extracts
its CSV output.

Use a dedicated Python environment because the first PySR import provisions a
Julia environment:

```sh
python3 -m venv .baseline-pysr
.baseline-pysr/bin/pip install -r bench/baselines/requirements.txt
.baseline-pysr/bin/python -c 'import pysr'

JULIA=.baseline-pysr/julia_env/pyjuliapkg/install/bin/julia
"$JULIA" --project=.baseline-pysr/julia_env \
  -e 'using Pkg; Pkg.add("LoopVectorization"); Pkg.precompile()'

.baseline-pysr/bin/python bench/baselines/pysr_sweep.py \
  --output docs/baseline_pysr.csv
```

The collector invokes Julia as a subprocess. This avoids making Python process
shutdown part of the benchmark and isolates Julia's requested thread count.
Standard and `LoopVectorization.@turbo` evaluation are recorded as distinct
execution modes.

Thread counts can also be collected in isolated invocations and merged. This
is useful when runs are distributed across hosts or Python environments:

```sh
for workers in 1 12 24; do
  .baseline-pysr/bin/python bench/baselines/pysr_sweep.py \
    --workers "$workers" \
    --output "scratch/pysr_${workers}.csv"
done

python3 bench/baselines/merge_csv.py \
  scratch/pysr_1.csv scratch/pysr_12.csv scratch/pysr_24.csv \
  --output docs/baseline_pysr.csv
```

## EvoGP

The SSE path calls EvoGP's native `tree_SR_fitness` runtime-dispatch kernel.
The materialize path calls its public `Forest.batch_forward` API, whose timed
GPU work includes expanding tree metadata and input rows before dispatch. That
materialize result is useful as an end-to-end public API baseline, but it is
not a kernel-only measurement.

Run from an environment containing CUDA PyTorch and EvoGP's compiled extension:

```sh
PYTHONPATH=/path/to/evogp/src /path/to/evogp-venv/bin/python \
  bench/baselines/evogp_sweep.py \
  --evogp-repo /path/to/evogp \
  --output docs/baseline_evogp.csv
```

The collector builds EvoGP's packed tree tensors explicitly rather than
relying on random tree generation. Protected operations are lowered to
EvoGP's primitive tree operations. CUDA events time GPU work. The `safe-math`
profile uses a `2e-3` normalized validation tolerance because EvoGP's CUDA
runtime reached `1.785e-3` against the Float32 oracle over all 1,024 corpus
expressions and 257 deterministic rows; other profiles retain `2e-4`.

### EvoGP population interoperability

`evogp_secant_interop.py` exercises the opposite direction. It runs an
unmodified EvoGP genetic-programming loop, decodes every tree in the final live
population, maps the exact prefix tree to return-terminated Secant postorder
bytecode, and compares EvoGP materialization against both Secant CPU and direct
CUBIN execution. Exact f32 constant bits are preserved. Unsupported EvoGP node
types or functions terminate the run instead of being approximated.

Build the C interoperability runner and execute the default 128-tree campaign:

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECANT_BUILD_BENCHMARKS=ON
cmake --build build --target secant_evogp_interop -j

/path/to/evogp-venv/bin/python \
  bench/baselines/evogp_secant_interop.py \
  --evogp-repo /path/to/evogp \
  --output-dir scratch/evogp_secant_interop
```

`population_map.json` records the raw EvoGP node values, types, subtree sizes,
decoded expression, fitness, and Secant bytecode hex for every mapped AST. The
adjacent binary files contain the packed programs, offsets, input columns, and
EvoGP outputs consumed by the C runner. They are scratch artifacts rather than
benchmark throughput records.

## Kozax

The Kozax collector encodes the selected trees in Kozax's native array format and
times JIT-warmed JAX execution. Materialize uses Kozax's `tree_evaluator`
vectorized over rows and population. SSE uses Kozax's native symbolic
regression fitness path, which computes mean squared error. JIT compilation is
excluded, and every timed call is synchronized before the wall-clock interval
ends.

Kozax vectorizes the complete fixed-width tree state over both population and
rows. This can require substantially more temporary storage than the final
output. The collector uses a conservative working-set estimate and refuses
shapes above `--max-estimated-working-bytes` before asking XLA to compile them.
On the RTX 5090, 1,024 ASTs by 262,144 rows caused XLA to request approximately
65 GiB without this guard.

Install Kozax and a JAX build appropriate for the local accelerator in a
dedicated environment. The following command targets CUDA 13:

```sh
uv venv --python 3.12 .baseline-backends
uv pip install --python .baseline-backends/bin/python \
  -r bench/baselines/requirements-backends.txt \
  'jax[cuda13]'

.baseline-backends/bin/python bench/baselines/kozax_sweep.py \
  --output docs/baseline_kozax.csv
```

## Operon

The Operon collector uses PyOperon's native `EvaluateTrees` batch API with a
caller-owned output buffer and configurable C++ worker count. PyOperon does not
expose a fused parallel batch-fitness function, so the SSE mode is explicitly
recorded as `EvaluateTrees` followed by an in-place NumPy reduction. It is a
materialize-plus-reduce backend baseline, not Operon's evolutionary search.

PyOperon 0.6.1 exposes min and max with binding labels that are reversed
relative to their observed behavior. The collector probes both operators,
selects them by evaluated semantics, and validates every fixed tree before
timing. Its postfix format requires binary children in right-then-left order;
the adapter performs that conversion explicitly for noncommutative operators.

```sh
.baseline-backends/bin/python bench/baselines/operon_sweep.py \
  --output docs/baseline_operon.csv
```

## Graphs

Pass any number of normalized baseline CSVs. For duplicate
system/backend/row combinations, the graph keeps the highest measured
throughput, allowing a sweep to include several AST batch sizes, worker counts,
and execution modes. Markers are not connected by default, so the graph displays
only measured row counts. Pass `--connect-points` to draw connecting lines or
`--all-configurations` to draw each configuration separately.

```sh
.venv/bin/python bench/baselines/comparison_graph.py \
  docs/baseline_secant.csv \
  docs/baseline_native_avx.csv \
  docs/baseline_pysr.csv \
  docs/baseline_evogp.csv \
  docs/baseline_kozax.csv \
  docs/baseline_operon.csv \
  --output docs/baseline_runtime.svg
```

The horizontal axis is logarithmic by default because row counts usually span
several powers of two. Use `--linear-x` for a linear row axis and `--log-y` for
a logarithmic throughput axis.

Write the corresponding compact Markdown tables with:

```sh
python3 bench/baselines/comparison_report.py \
  docs/baseline_secant.csv \
  docs/baseline_native_avx.csv \
  docs/baseline_pysr.csv \
  docs/baseline_evogp.csv \
  docs/baseline_kozax.csv \
  docs/baseline_operon.csv \
  --graph docs/baseline_runtime.svg \
  --secant-samples docs/baseline_secant_samples.csv \
  --output docs/runtime_baselines.md
```
