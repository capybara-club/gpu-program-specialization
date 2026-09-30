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

# Skeleton CUBIN Cache

Secant-SR stores generated skeleton CUBINs in SQLite so repeated searches do
not pay NVRTC template-compilation cost. The cache belongs to the search
application rather than Secant's AST-to-CUBIN specialization API.

## Identity And Validation

Each entry is keyed by:

- Secant major, minor, and patch version.
- The recipe shape, recipe version, flags, and every concrete source-generator parameter.
- Compute-capability major and minor.
- NVRTC major and minor version.
- PTXAS optimization level and NVRTC cache setting.

Neither the generated source nor a source digest is stored. A cache hit can
therefore occur before source generation, eliminating that work from the warm
path. A cached CUBIN still passes through Secant inspection before use. A
cached binary that fails inspection is recompiled from newly generated source
for the same recipe and replaced.

The fixed MSE reducer is stored in a separate artifact table keyed by artifact
name, explicit source version, compute capability, NVRTC version, PTXAS
optimization level, and NVRTC cache mode. Its CUBIN is validated by loading it
and resolving every required function. A failed load or symbol lookup
recompiles and replaces the entry. The constant-optimizer reducer is part of
the generated optimizer skeleton and therefore shares that recipe's cache
entry and binary validation.

Secant's public version is the compatibility boundary for persisted native
templates. Any source-generation or binary-inspection change that can affect a
template requires a Secant version change. Recipe fields, target architecture,
toolkit version, and compiler options remain independent key dimensions.

SQLite uses WAL mode and a 30-second busy timeout. The schema has an explicit
version and rejects unknown versions. Each row stores the measured NVRTC
compile time alongside the explicit parameter key and CUBIN. Source-based
schema-v1 and digest-based schema-v2 rows cannot be mapped back to complete
recipes, so opening either old schema resets the template table once.

## Timing Output

The C application prints one `template_cache` record and one
`mse_reducer_cache` record with cache hits, misses, database overhead, actual
NVRTC time, and the historical compile time recovered for cache hits.
`python/run_suite.py` writes both sets of cache values to its result CSV and
computes `estimated_uncached_elapsed_seconds` by replacing template and reducer
cache overhead with their stored NVRTC timings.

The Python suite uses `scratch/cache/secant_sr_cubin.sqlite3` by default:

```bash
CUDA_MODULE_LOADING=EAGER python3 python/run_suite.py --backend cubin-staged
```

Use `--no-cubin-cache` for a fresh-compilation baseline or `--cubin-cache PATH`
to select an isolated cache.

## RTX 5090 Measurement

The following Release-mode measurement used the canonical staged shape: 64
kernels, 32 ASTs per kernel, 8,192 individuals, 4,096 dynamic-leaf settings,
64-row dynamic tiles, and 1,024-row static tiles.

| Run | Wall time | Actual NVRTC | Estimated uncached wall time |
|---|---:|---:|---:|
| Cache disabled, one generation | 18.695 s | 18.163 s | 18.695 s |
| Cold cache, one generation | 18.827 s | 18.175 s | 18.716 s |
| Warm cache, one generation | 0.474 s | 0 s | 18.634 s |
| Warm cache, 100 generations | 3.083 s | 0 s | 21.239 s |

The warm one-generation run was 39.5 times faster end to end. The 100-generation
run was about 6.9 times faster than its estimated uncached time. The warm
uncached estimate was within roughly 0.3% of the separately measured no-cache
run, showing that storing the compile time gives a useful campaign-cost
estimate rather than only a hit counter.

## Complete SRBench Rerun

The campaigns in this section predate training-only early stopping: selection
used training fitness, but held-out R2 could stop a run. Their cache and timing
measurements remain valid; their solve counts are not protocol-correct SRBench
recovery rates.

The 122-dataset campaign recorded in `srbench_template_inventory.md` was rerun
with the same search configuration. It contains 116 staged Feynman searches,
five staged black-box searches, and one dynamic-leaf-only Tecator search.

| Campaign | Summed search time | Driver wall time | Cache hits | Templates compiled | Actual NVRTC |
|---|---:|---:|---:|---:|---:|
| Previous, cache disabled | 42.732 min | Not recorded | 0 | 364 | ~37.1 min |
| Fresh persistent cache | 8.103 min | 8.232 min | 339 | 25 | 2.294 min |
| Fully warm cache | 5.834 min | 5.963 min | 364 | 0 | 0 min |

On the directly comparable summed search time, the fresh-cache campaign was
5.27 times faster and the warm-cache campaign was 7.32 times faster. The fresh
database contained 25 unique entries and 81.36 MiB of CUBIN data. Parameter-key
metadata is small relative to those binaries. No cached entry failed
inspection.

The warm run estimated 42.901 minutes without caching from the historical
compile timings. The previous measured cache-disabled campaign took 42.732
minutes, a difference of about 0.4%.

Fixed-seed search winners are not bitwise reproducible across complete GPU
runs because parallel reductions and completion ordering can perturb close
fitness rankings. Repeating one problem twice with an already warm cache still
produced different winners, so this is not specific to fresh versus cached
CUBINs. Aggregate quality remained comparable: both cached campaigns solved 58
of 116 Feynman problems at holdout `R2 > 0.999999`, compared with 56 in the
previous campaign.

The complete per-problem records are in:

- `data/srbench_cache_cold_sm120_cuda13_1_20260803.csv`
- `data/srbench_cache_warm_sm120_cuda13_1_20260803.csv`
