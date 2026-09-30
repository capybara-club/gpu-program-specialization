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

# Dynamic-leaf setting CTA sweep, 2026-08-13

Commit `d7ff89e5d9d9c132b55a0d0be3226a64e0925315` adds independent
settings-per-CTA, row-tile, and thread-count sweeps for the PTX thread-owned
SSE and LM kernels.

## Shape

- 16,384 rows, 8 dynamic parameters, and 8 input columns
- SSE: one ALU AST
- LM: one square/cube AST with mixed constant/column bindings
- 4,096 and 8,192 total settings on Rohini; 8,192 on Ada
- Hot-kernel CUDA-event time; compilation and verification are excluded
- LM verifies eight settings spread across the full result, including the
  first and last setting, against the double-precision CPU oracle

The rates compare launch topologies within each workload. SSE and LM rates are
not direct operation-throughput comparisons because the ASTs and accumulated
outputs differ.

## Results

| Host | GPU | Workload | Settings | Settings/CTA | Tile rows | Threads | Row evals/s |
|---|---|---:|---:|---:|---:|---:|---:|
| rohini | RTX 5090 | SSE | 4,096 | 128 | 64 | 128 | 724.4B |
| rohini | RTX 5090 | LM | 4,096 | 128 | 64 | 128 | 175.4B |
| rohini | RTX 5090 | SSE | 8,192 | 128 | 128 | 128 | 775.0B |
| rohini | RTX 5090 | LM | 8,192 | 128 | 128 | 128 | 177.5B |
| ada | RTX 4090 | SSE | 8,192 | 256 | 256 | 256 | 556.4B |
| ada | RTX 4090 | LM | 8,192 | 128 | 256 | 128 | 115.5B |

The Rohini 8,192-setting entries and Ada entries are medians of three trials.
The Rohini 4,096-setting entries are from the coarse sweep. Both hosts used
driver 595.71.05.

On Rohini, one setting per thread is the useful ridge. Moving from 128 to the
entire 8,192-setting range in one CTA reduced SSE from 771.7B to 570.0B row
evals/s and LM from 176.5B to 128.9B. Tile sizes 64 and 128 are nearly tied.

Ada benefits from more SSE amortization. Its separate winners are shown above,
but a common setting of 256 settings/CTA, 256 tile rows, and 128 threads reaches
550.8B SSE and 115.4B LM row evals/s, within about one percent of both winners.

## Reproduction

```sh
python3 bench/dynamic_leaf_cta_sweep.py \
  --output /tmp/dynamic-leaf-cta.csv \
  --settings 4096 8192 \
  --settings-per-cta 32 64 128 256 512 1024 2048 4096 8192 \
  --tile-rows 32 64 128 256 512 \
  --threads 128
```

Repeat the winner neighborhood with `--threads 64 128 256 --trials 3` before
selecting a preset for a new GPU.
