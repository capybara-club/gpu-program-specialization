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

# SECANT Compile-to-Execution Frontier

## Protocol

The benchmark specializes 256 MUFU ASTs as eight kernels with 32 ASTs per
kernel, loads the resulting module, and launches all eight kernels over each
measured row count. Kernels use the static-column SSE shape with a 1,024-row
logical tile, 128 threads, one target, and eight streams.

Every subprocess and its compiler children are pinned to one CPU. Each point
is the median of three independently seeded samples. Backend order rotates
between samples to distribute thermal and clock-order effects. Inputs and
targets are copied to the device before module loading and remain resident.

The solid-line throughput is:

```text
row evaluations / (AST compile or specialization + module load + kernel run)
```

The dotted-line throughput is kernel execution alone. Cold template
generation, PTX/CUBIN/HSACO template inspection, template copying, allocation,
host-to-device transfer, correctness comparison, and teardown are outside both
measurements.

CUDA O0/O1 select final ptxas optimization. PTX O0/O1 select nvPTXCompiler
optimization. HIP O0/O1 select HIPRTC optimization. CUBIN and HSACO have one
specialization path and use O1 skeletons.

## 16M Rows per AST

Each row in this table represents 4,294,967,296 actual row evaluations.

| System | Backend | Compile | Load | Run | Runtime-only rows/s | Compile + load + run rows/s |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 | CUDA O0 | 766.481 ms | 0.347 ms | 14.943 ms | 2.874e11 | 5.494e9 |
| RTX 5090 | CUDA O1 | 800.461 ms | 0.186 ms | 2.952 ms | 1.455e12 | 5.345e9 |
| RTX 5090 | PTX O0 | 324.262 ms | 0.390 ms | 13.390 ms | 3.208e11 | 1.271e10 |
| RTX 5090 | PTX O1 | 303.561 ms | 0.176 ms | 2.944 ms | 1.459e12 | 1.400e10 |
| RTX 5090 | CUBIN | 0.098 ms | 0.375 ms | 5.011 ms | 8.571e11 | 7.832e11 |
| RX 9070 XT | HIP O0 | 1,735.253 ms | 3.412 ms | 2,499.107 ms | 1.719e9 | 1.013e9 |
| RX 9070 XT | HIP O1 | 952.308 ms | 0.996 ms | 72.022 ms | 5.963e10 | 4.189e9 |
| RX 9070 XT | HSACO | 0.072 ms | 0.834 ms | 83.408 ms | 5.149e10 | 5.094e10 |

Native O1 scheduling is faster at runtime on both systems: PTX/CUDA O1 reaches
about 1.46e12 rows/s on the RTX 5090, while CUBIN reaches 8.57e11; HIP O1
reaches 5.96e10 rows/s on the RX 9070 XT, while HSACO reaches 5.15e10. Direct
binary specialization nevertheless has the highest combined throughput at
every measured row count because its compile cost is approximately four
orders of magnitude smaller.

At 1,024 rows per AST, CUBIN sustains 6.08e8 combined rows/s versus 8.70e5 for
PTX O1. HSACO sustains 1.73e8 combined rows/s versus 2.75e5 for HIP O1. At
16,777,216 rows per AST, the corresponding advantages remain approximately
56x on the RTX 5090 and 12x on the RX 9070 XT.

## Artifacts

- [RTX 5090 summary CSV](end_to_end_rtx5090_mufu_2026-07-26.csv)
- [RTX 5090 raw samples](end_to_end_rtx5090_mufu_2026-07-26_samples.csv)
- [RTX 5090 graph](end_to_end_rtx5090_mufu_2026-07-26.svg)
- [RX 9070 XT summary CSV](end_to_end_rx9070xt_mufu_2026-07-26.csv)
- [RX 9070 XT raw samples](end_to_end_rx9070xt_mufu_2026-07-26_samples.csv)
- [RX 9070 XT graph](end_to_end_rx9070xt_mufu_2026-07-26.svg)
