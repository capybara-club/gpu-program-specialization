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

# Combined Compile Scaling

This compares the patched AST compiler path using 512 modules and 4 gram kernels per module.

| System | Workers | Compile ms | ASTs/s | Gram kernels/s |
|---|---:|---:|---:|---:|
| RTX 5090 + Ryzen 9900X | 1 | 23758.462 | 2758.428 | 86.201 |
| RTX 5090 + Ryzen 9900X | 2 | 12450.930 | 5263.543 | 164.486 |
| RTX 5090 + Ryzen 9900X | 4 | 6262.121 | 10465.464 | 327.046 |
| RTX 5090 + Ryzen 9900X | 8 | 3262.994 | 20084.623 | 627.644 |
| RTX 5090 + Ryzen 9900X | 12 | 2275.620 | 28799.190 | 899.975 |
| RTX 5090 + Ryzen 9900X | 24 | 1897.167 | 34544.147 | 1079.505 |
| GH200 | 1 | 40158.894 | 1631.917 | 50.997 |
| GH200 | 2 | 20016.975 | 3274.021 | 102.313 |
| GH200 | 4 | 9928.076 | 6601.078 | 206.284 |
| GH200 | 8 | 4957.284 | 13220.142 | 413.129 |
| GH200 | 16 | 2546.196 | 25738.788 | 804.337 |
| GH200 | 32 | 1409.292 | 46502.770 | 1453.212 |
| GH200 | 64 | 941.264 | 69625.533 | 2175.798 |

Files:

- `combined_compile_scaling_512_modules.csv`
- `combined_compile_asts_per_sec_512_modules.svg`
- `combined_compile_time_512_modules.svg`
- `local_5090_compile_scaling_512_modules.txt`
- `gh200_compile_scaling_512_modules.txt`
