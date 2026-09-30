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

# RTX 5090 Patch And Module Load Sweep

The newer 32-AST/kernel module-load measurements and pooled-site design are
in [`RTX_5090_SETTINGS_SWEEP.md`](RTX_5090_SETTINGS_SWEEP.md).

Date: July 8, 2026

GPU:

```text
NVIDIA GeForce RTX 5090, compute capability 12.0, driver 595.71.05
```

These sweeps use the AST-SASS `tile_static_mse` path with:

```text
ast_mode          = alu_reduce
sites_per_kernel  = 16
site_instructions = 256
program_stride    = 64
columns           = 32
```

`sites_per_kernel=16` means 16 ASTs/kernel for this shape. The point of these
runs is patch/load amortization, not runtime throughput, so the row workload is
kept tiny.

Raw logs:

```text
docs/bench_logs/bench_5090_module_sweep_20260708_183852.txt
docs/bench_logs/bench_5090_load_sweep_20260708_183942.txt
```

## Patch Sweep

This keeps total ASTs fixed at 1024 and varies ASTs/module by changing
kernels/module. Runtime uses 128 settings, 128 rows, and 1 timed iteration.

| ASTs/module | Modules | Kernels/module | NVRTC s | Patch us/AST | Patch AST/s | Load us/AST |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 1 | 0.187 | 1.362 | 734,372 | 1.859 |
| 32 | 32 | 2 | 0.283 | 1.358 | 736,494 | 1.587 |
| 64 | 16 | 4 | 0.546 | 1.367 | 731,492 | 0.519 |
| 128 | 8 | 8 | 1.047 | 1.371 | 729,544 | 0.350 |
| 256 | 4 | 16 | 0.045 | 1.373 | 728,291 | 0.238 |
| 512 | 2 | 32 | 4.194 | 1.389 | 719,737 | 0.316 |
| 1024 | 1 | 64 | 0.260 | 1.376 | 726,779 | 0.285 |

Read: the patcher is already on its plateau. For this AST shape, packing beyond
about 64 ASTs/module does not materially improve patch throughput. The
single-core patch rate stays around 720k-736k AST/s, or about 1.36-1.39 us/AST.

## Load Sweep

This keeps total ASTs fixed at 4096 and uses `settings=1`, `run_rows=1`,
`check_rows=0`, and `run_iters=1` so module loading is less dominated by
sub-millisecond noise.

| ASTs/module | Modules | Kernels/module | NVRTC s | Patch us/AST | Patch AST/s | Load us/AST |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 256 | 1 | 0.013 | 1.746 | 572,787 | 2.855 |
| 32 | 128 | 2 | 0.015 | 1.744 | 573,305 | 2.319 |
| 64 | 64 | 4 | 0.019 | 1.366 | 731,823 | 0.869 |
| 128 | 32 | 8 | 0.025 | 1.357 | 736,896 | 0.742 |
| 256 | 16 | 16 | 0.045 | 1.387 | 720,739 | 0.694 |
| 512 | 8 | 32 | 0.097 | 1.368 | 731,214 | 0.589 |
| 1024 | 4 | 64 | 0.258 | 1.384 | 722,621 | 0.270 |

Read: module-load amortization is different from patching. The biggest gain is
from moving away from tiny 16-32 AST modules. By 64 ASTs/module, load cost falls
below 1 us/AST. It still improves with larger modules in this run, reaching
0.270 us/AST at 1024 ASTs/module.

Practical take:

- For patching alone, 64 ASTs/module is enough for this shape.
- For module load, prefer at least 128-256 ASTs/module.
- If time-to-first-module and CUBIN size are acceptable, 512-1024 ASTs/module
  gives better load amortization.
- NVRTC timings here are warm-path observations from this benchmark and are not
  the target metric for the SASS patch hot path.
