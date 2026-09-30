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

# Secant-SR 0.3.4

> **Collection category:** Reference consumers. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

The active implementation is a C99 symbolic-regression search built on Secant's
explicit **toggles × shared coefficient banks**. Leaf choices remain part of the
genome during crossover and mutation. Optional GPU LM fits coefficients before
breeding. There are no settings tables, and no GP policy has been added to the
Secant kernel library.

Each generation produces native postorder ASTs, scores their Cartesian products
in bounded GPU batches, reduces to one winning configuration per AST on the GPU,
and uses those scores for selection. The final result includes a resolved AST,
the original toggle AST, bank/permutation indices, coefficients, train/holdout
metrics, and stage timings. A separate CPU evaluation checks the winning model.

## Build

Use the sibling Secant **0.3.1** checkout. Existing settings-based binaries and
templates are incompatible; use a new build directory.

```bash
cmake -S . -B build-toggle -DCMAKE_BUILD_TYPE=Release
cmake --build build-toggle -j 8
ctest --test-dir build-toggle --output-on-failure
```

CUDA requires an installed toolkit and NVIDIA driver. On a CPU-only machine,
configure with `-DSECANT_SR_ENABLE_CUDA=OFF` and explicitly select `--backend cpu`.
The default backend is CUDA; a missing or failed CUDA backend never silently
falls back to CPU.

On rack1, which has an installed CUDA toolkit and C compiler but no CMake:

```bash
python3 toggle/build_cuda.py --secant ../secant --output ./secant_sr_search
```

This compiles repository sources using existing tools; it installs nothing.

## Run

```bash
CUDA_MODULE_LOADING=EAGER ./build-toggle/secant_sr_search \
  --backend cuda --problem nguyen1 --population 1024 --generations 100 \
  --banks 64 --constants 4 --toggle-bits 6 --seconds 60
```

That evaluates 4,096 configurations per AST: 64 shared bank vectors × 64
permutations. Every packed AST sees the same vector and bit pattern, but its
own leaf alternatives determine their meaning. The CLI emits JSON Lines to
stdout and diagnostics to stderr; `--help` lists the controls.

Add `--refine-rounds 4 --refine-budget 128` to fit coefficient values before
breeding. This runs packed Philox proposals crossed with the existing toggle
permutations, keeps fixed literals unchanged, and inherits fitted parameters.
It uses the ordinary scoring pipeline without legacy settings tables. Refinement
defaults to off so existing search configurations retain their meaning.
See [the fitting API, validation, and limits](docs/toggle-refinement-20260919.md).

Alternatively, enable GPU LM with `--lm-iterations 4 --lm-budget 512
--lm-bindings 32 --lm-starts 4`. It retains the toggle genome, fits tied
coefficients on training rows, and rescores proposals through the ordinary
Secant pipeline before promotion. It is mutually exclusive with random
refinement. The reusable CUDA evaluator uses register-only 1/2/4/8-parameter
shapes and rejects compiled kernels with local storage; it does not yet use
Secant's SASS specializer for derivatives. See [the API and measured limits](docs/toggle-lm-20260921.md).

The experimental `--refine-parameters active-block` policy fits a bounded block
of the winning binding's coefficients while preserving other centers and all
toggle alternatives. `--power-mutation-probability 0.25` optionally proposes
subtree squares/cubes through ordinary multiplication. Both are opt-in;
see the [diagnostic and matched trial](docs/policy-repair-20260920.md).

`--finalists 16` optionally retains distinct concrete bindings across generations
without changing evolution. The separate [final-polishing experiment](docs/finalist-polishing-20260920.md)
uses that archive for budgeted training-only CPU coefficient fitting. It is not
enabled by default and does not change the CUDA core.

Optional GP experiments: `--align-crossover-bits 1`,
`--toggle-mutation-probability 0.5`, and `--leaf-mix-probability 0.25`.
All default to zero. They preserve useful toggle alternatives during variation;
the evaluator and packing contract are unchanged. See the
[population/time/variation trial](docs/search-population-night-20260919.md)
for semantics, limitations and the matched overnight comparison.

A small CPU functionality check:

```bash
./build-toggle/secant_sr_search --backend cpu --problem nguyen1 \
  --population 64 --elites 2 --banks 4 --toggle-bits 2 --rows 32 \
  --validation-rows 32 --generations 10 --seconds 10
```

## Python and custom data

Add `python/` to `PYTHONPATH`. The client needs only the Python standard library:

```python
from secant_sr import fit
from secant_sr_ast import Expression

X = [[i / 100, (i % 17) / 17] for i in range(200)]
y = [a * a + b for a, b in X]
result = fit(X, y, backend="cuda", population=1024, generations=100,
             banks=64, constants=4, toggle_bits=6, seconds=60)
print(result["expression"], result["validation_mse"])
model = Expression.decode(bytes.fromhex(result["resolved_ast_hex"]))
print(model.evaluate([0.5, 0.2]))
```

`fit` accepts an explicit `validation=(X_validation, y_validation)` pair; otherwise
it reserves a deterministic 20% holdout. Data sent to C is column-major f32,
without formulas or generator seeds. `Expression.evaluate` is a convenience
Python evaluation, not a bitwise f32 oracle. The native CPU scorer is the oracle.

## Interfaces and scope

- [secant_sr.h](secant_sr.h): host GP, immutable banks, ask/tell, and model replay.
- [toggle/include/secant_sr_cuda.h](toggle/include/secant_sr_cuda.h): persistent
  dataset/scorer, bounded score pool, reduction, and timings.
- [toggle/README.md](toggle/README.md): architecture, ownership, configuration
  semantics, limits, and remaining work.
- [docs/toggle-search-20260919.md](docs/toggle-search-20260919.md): validation and
  initial measurements, including cold-start and search-quality limitations.
- [docs/srbench_toggle.md](docs/srbench_toggle.md): current SRBench preparation,
  multi-GPU campaigns, safe resume and reporting. Use `python/run_srbench_toggle.py`,
  not the historical `python/run_suite.py`, with the new executable.
- [bench/controlled/README.md](bench/controlled/README.md): isolated settings/toggle
  replay and matched GP/refinement experiments; see the
  [2026-09-20 measurements](docs/controlled-settings-toggles-20260920.md).

This is a new search controller, not a claim of parity with the historical
maturity/QD/LM system or its SRbench results. Constants start from a fixed bank
and literal vocabulary; optional random coefficient refinement adjusts bank
parameters. Fixed literals remain fixed. GPU LM is opt-in; CPU polishing remains
a separate experiment.
`solved` means the train **and** holdout normalized-MSE thresholds passed, not
that symbolic equivalence was established. Work counters count occurrences,
not unique expressions.

The previous implementation is preserved but excluded from the current build.
Its documentation follows in [legacy/README-settings.md](legacy/README-settings.md),
with the compatibility boundary explained in [legacy/README.md](legacy/README.md).
