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

# GH200 Compile Scaling

This run measures the patched AST compiler path on a Lambda Labs GH200 system.

- GPU: NVIDIA GH200, `sm_90`
- CUDA: 12.8
- Modules: 512
- Gram kernels per module: 4
- AST features per gram kernel: 32
- Total gram kernels per point: 2048
- Total AST features per point: 65536
- Worker scratch: 256 MiB per worker

| Workers | Compile ms | Modules/s | Gram kernels/s | ASTs/s |
|---:|---:|---:|---:|---:|
| 1 | 40158.894 | 12.749 | 50.997 | 1631.917 |
| 2 | 20016.975 | 25.578 | 102.313 | 3274.021 |
| 4 | 9928.076 | 51.571 | 206.284 | 6601.078 |
| 8 | 4957.284 | 103.282 | 413.129 | 13220.142 |
| 16 | 2546.196 | 201.084 | 804.337 | 25738.788 |
| 32 | 1409.292 | 363.303 | 1453.212 | 46502.770 |
| 64 | 941.264 | 543.949 | 2175.798 | 69625.533 |

The 1 through 16 worker points are nearly linear. Scaling remains strong at 32
and 64 workers, but the per-worker speed drops as all cores are loaded.

Files:

- `gh200_compile_scaling_512_modules.txt`: raw benchmark output
- `gh200_compile_scaling_512_modules.csv`: parsed table
- `gh200_compile_time_512_modules.svg`: compile time graph
- `gh200_compile_throughput_512_modules.svg`: throughput graph
