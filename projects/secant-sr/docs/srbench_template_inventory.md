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

# SRBench CUBIN Template Inventory

The template counts and compilation timings in this inventory remain valid.
Its recorded recovery counts predate training-only early stopping and must not
be interpreted as protocol-correct SRBench recovery rates.

This inventory records the skeleton CUBINs generated during a single-seed
Secant-SR search over 122 official SRBench datasets on 2026-08-03. It is meant
to size an architecture-specific precompiled template set, not to establish an
SRBench leaderboard result.

## Campaign

The run used an RTX 5090 (`sm_120`), NVRTC 13.1, ptxas optimization level 1,
and NVRTC's no-cache option. It covered all 116 Feynman datasets and six
black-box datasets. The common search configuration was:

- 8,192 individuals and 100 generations.
- 4,096 dynamic-leaf settings and eight dynamic leaves.
- Ten dynamic generations before staged static-SSE evaluation.
- 64 kernels per module and 32 ASTs per kernel.
- 64 patch instructions per AST, giving 2,048 instructions per patch island.
- 128 threads, 64-row dynamic tiles, and 1,024-row static tiles.
- 24 specialization workers and eight CUDA streams.
- Up to 10,000 training rows and the complete holdout split.
- Broad operators with local continuous constant optimization disabled.

Five black-box datasets used the staged backend. `505_tecator` has 124 input
columns and used only dynamic-leaf SSE because its static 32-AST skeleton is
known to spill and fail inspection.

## Inventory

| Shape | Generation events | Unique templates | Input widths | Stored CUBIN size |
|---|---:|---:|---|---:|
| Static SSE | 121 | 12 | 1-9, 12, 14, 48 | 77.45 MiB |
| Static materialize | 121 | 12 | 1-9, 12, 14, 48 | 0.46 MiB |
| Dynamic-leaf SSE | 122 | 1 | Runtime selected | 3.46 MiB |
| **Total** | **364** | **25** | | **81.36 MiB** |

The campaign generated 1.176 GiB of CUBIN data cumulatively. Deduplication
removes 339 of 364 generation events, or 93.1%. Feynman alone needs 19
templates: static SSE and materialize for each input width from one through
nine, plus one dynamic-leaf template. The wider black-box datasets add six
templates for widths 12, 14, and 48.

The 25-template result applies to this exact recipe family. A cache key must
include at least:

- Secant major, minor, and patch version.
- Kernel shape, recipe version, and every concrete recipe parameter.
- Compute capability.
- NVRTC version and compilation options.
- Kernel and AST packing.
- Input and target capacities.
- Dynamic-leaf capacity where applicable.
- Tile size, threads per block, and patch capacity.

Row count, population size, generation count, leaf-setting count, worker
count, and stream count do not change these skeleton CUBINs.

## Search Result

The repeated Feynman search numerically solved 56 of 116 datasets at holdout
`R2 > 0.999999`; 88 reached `R2 >= 0.99`. The median holdout score was
0.9999751785 and the summed per-campaign elapsed time was 40.58 minutes. None
of the six black-box datasets crossed the numerical-solve threshold. These
figures use one seed and are included only to confirm that the inventory came
from complete searches rather than source-generation probes.

## Recorded Data

- `data/srbench_template_events_sm120_cuda13_1_20260803.csv` contains every
  generated template occurrence from the original source-hash instrumentation,
  with its problem, seed, recipe, architecture, compiler settings, source size,
  and CUBIN size.
- `data/srbench_template_summary_sm120_cuda13_1_20260803.csv` contains the 25
  deduplicated cache entries and their occurrence counts.
- `data/srbench_search_sm120_20260803.csv` contains the 122 search outcomes.

Future suite runs can produce the same event and summary files by passing
`--template-output PATH` to `python/run_suite.py`. New files use the explicit
Secant-version and recipe-parameter identity rather than a source digest.
