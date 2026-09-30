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

# RX 9070 XT Backend Snapshot

Fresh matched-backend measurements captured before HIP/HSACO development was
placed in the icebox. Raw command output is stored beside this file.

Runtime measurements used a Release build, 64 kernels, 128 ASTs per kernel,
262,144 rows, and 8 streams.

| Shape | Backend | AST mode | Best row-evals/s | Tile rows |
| --- | --- | --- | ---: | ---: |
| Static SSE | CUDA, RTX 5090 | ALU | 4.296e12 | matched sweep |
| Static SSE | CUBIN, RTX 5090 | ALU | 2.387e12 | matched sweep |
| Static SSE | HIP, RX 9070 XT | ALU | 2.511e11 | 8192 |
| Static SSE | HSACO, RX 9070 XT | ALU | 3.940e11 | 4096 |
| Static SSE | CUDA, RTX 5090 | MUFU | 1.838e12 | matched sweep |
| Static SSE | CUBIN, RTX 5090 | MUFU | 6.334e11 | matched sweep |
| Static SSE | HIP, RX 9070 XT | MUFU | 1.544e11 | 4096 |
| Static SSE | HSACO, RX 9070 XT | MUFU | 9.236e10 | 4096 |
| Materialize | CUDA, RTX 5090 | ALU | 3.293e11 | matched sweep |
| Materialize | CUBIN, RTX 5090 | ALU | 3.451e11 | matched sweep |
| Materialize | HIP, RX 9070 XT | ALU | 1.181e11 | matched sweep |
| Materialize | HSACO, RX 9070 XT | ALU | 1.177e11 | matched sweep |
| Materialize | CUDA, RTX 5090 | MUFU | 3.369e11 | matched sweep |
| Materialize | CUBIN, RTX 5090 | MUFU | 3.129e11 | matched sweep |
| Materialize | HIP, RX 9070 XT | MUFU | 1.231e11 | matched sweep |
| Materialize | HSACO, RX 9070 XT | MUFU | 9.725e10 | matched sweep |

One-core static-SSE ALU compilation throughput was 181 AST/s for CUDA,
6.97e6 AST/s for CUBIN, 215 AST/s for HIP, and 9.84e6 AST/s for HSACO.

These results are an archival engineering snapshot, not an active support
claim.
