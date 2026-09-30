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

# Packed toggle-SSE throughput on Rohini

## Result

The benchmark activated all eight candidate-system branches in one packed kernel function. One shared postorder program computes the prey derivative, and eight distinct branch programs compute candidate predator derivatives. The specialized `sm_120` kernel uses 37 registers and has no reported spills.

The exact Lotka–Volterra candidate retained a best MSE of `8.90300881e-14` in every run. The default runner-output regression also retained all 128 MSE values for each of the eight systems; summary mode only suppresses JSON rendering.

Peak sustained, module-lifecycle-inclusive throughput was **354.771 million complete trajectory configurations/s**, or **2.819 ns/configuration** as aggregate throughput. Each configuration integrates one complete 64-interval trajectory using four RK4 substeps per interval, so this corresponds to **363.3 billion complete system-RHS stage evaluations/s**. It is throughput across the GPU, not the latency of one trajectory.

Throughput was effectively at its plateau by 16,384 constant banks: 352.654 million configurations/s, 99.4% of the largest measured result. Doubling to 32,768 banks improved throughput by only 0.6%.

## Workload

- GPU: NVIDIA GeForce RTX 5090, driver 595.71.05
- Host: AMD Ryzen 9 9900X, 12 cores / 24 threads
- CUDA target: `sm_120`
- Kernel: packed toggle-SSE, 128 threads/CTA
- Active systems: 8 of 8 branch slots
- State dimension: 2
- Shared RHS outputs: 1
- Branch RHS outputs: 1 per system
- Branch instruction counts including the continuation branch: 7, 6, 6, 7, 9, 8, 11, 11
- Toggle bits: 5, giving 32 permutations per constant bank
- Trajectories per configuration: 1
- Observation intervals: 64
- RK4 substeps per interval: 4
- RK4 stages per configuration: 1,024
- Specialized CUBIN size: 67,904 bytes

The eight branch programs are different, but some toggle permutations intentionally select leaves that reproduce the same mathematical RHS. This benchmark measures execution topology and dispatch, not eight independent scientific datasets.

## Cold module-lifecycle sweep

Each row below is one kernel launch in a fresh runner process. The timer includes CUBIN read, module load, identity verification, output allocation, launch, final result copy, synchronization, and module unload. Values are medians of three repetitions.

| Constant banks | Configurations/system | Total configurations | CTAs | Pipeline time | Pipeline throughput |
|---:|---:|---:|---:|---:|---:|
| 4 | 128 | 1,024 | 8 | 1.383 ms | 0.740 M/s |
| 16 | 512 | 4,096 | 32 | 1.362 ms | 3.006 M/s |
| 64 | 2,048 | 16,384 | 128 | 1.370 ms | 11.958 M/s |
| 256 | 8,192 | 65,536 | 512 | 1.360 ms | 48.205 M/s |
| 1,024 | 32,768 | 262,144 | 2,048 | 2.052 ms | 127.750 M/s |
| 2,048 | 65,536 | 524,288 | 4,096 | 2.830 ms | 185.288 M/s |
| 4,096 | 131,072 | 1,048,576 | 8,192 | 4.553 ms | 230.282 M/s |
| 8,192 | 262,144 | 2,097,152 | 16,384 | 8.127 ms | 258.034 M/s |
| 16,384 | 524,288 | 4,194,304 | 32,768 | 14.840 ms | 282.644 M/s |
| 32,768 | 1,048,576 | 8,388,608 | 65,536 | 29.212 ms | 287.165 M/s |

The nearly flat 1.36–1.38 ms time in the first four rows shows the cold module/host path dominating small launches. Packing eight systems helps, but eight to 512 CTAs still do not provide enough work to amortize that boundary.

## Sustained sweep

These cases each execute approximately 50.3 million configurations inside one loaded module. Repeated launches reuse the same configurations and exist only to measure sustained execution. The lifecycle timer still includes one module load and unload.

| Constant banks | Configurations/system/launch | CTAs/launch | Launches | Pipeline time | Sustained throughput | Fraction of peak |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 8,192 | 512 | 768 | 533.396 ms | 94.361 M/s | 26.6% |
| 1,024 | 32,768 | 2,048 | 192 | 263.187 ms | 191.239 M/s | 53.9% |
| 2,048 | 65,536 | 4,096 | 96 | 197.738 ms | 254.537 M/s | 71.7% |
| 4,096 | 131,072 | 8,192 | 48 | 165.148 ms | 304.767 M/s | 85.9% |
| 8,192 | 262,144 | 16,384 | 24 | 149.477 ms | 336.718 M/s | 94.9% |
| 16,384 | 524,288 | 32,768 | 12 | 142.722 ms | 352.654 M/s | 99.4% |
| 32,768 | 1,048,576 | 65,536 | 6 | 141.871 ms | 354.771 M/s | 100.0% |

The three peak-case repetitions were 354.218, 354.771, and 355.129 million configurations/s.

## Interpretation

For this two-state, one-trajectory shape, eight packed systems do not eliminate the need for substantial configuration work. A practical knee depends on the API's goal:

- 1,024 banks (32,768 configurations/system) reaches 54% of peak.
- 4,096 banks (131,072 configurations/system) reaches 86% of peak.
- 8,192 banks (262,144 configurations/system) reaches 95% of peak.
- 16,384 banks (524,288 configurations/system) is effectively saturated.

More candidate systems per function should reduce the required configurations per system by increasing total CTAs, but that is an inference, not a result from this eight-system benchmark. Larger branch tables may also change module loading and instruction-cache behavior and must be measured directly.

## Measurement boundaries and deviations

- These are configuration-evaluation rates, not unique-AST specialization rates. Sustained cases deliberately rerun the same specialized module and configurations.
- Process startup, CUDA context initialization, and the shared constants/reference-data allocation and host-to-device copies occur before the timer. Kernel compilation is also excluded.
- `total_seconds` in the raw reports ends after result synchronization. `module_pipeline_seconds`, used in the tables, additionally includes module unload.
- The final output copy and scan remain real. Summary mode suppresses only the large JSON MSE arrays.
- The run uses the temporary C99 Driver API runner. It validates the GPU execution and module lifecycle but is not a measurement of a future Python orchestration path.
- The template CUBIN was compiled earlier with the existing NVRTC-to-PTXAS path because direct `nvcc` on this host encounters the installed CUDA/Ubuntu header conflict. No headers were patched and no tooling was installed. Compile time is excluded, so execution results are valid; they are not a direct-`nvcc` compilation comparison.

## Raw reports

- [Initial cold and short repeated sweep](raw/2026-08-31-rohini-packed-8-system-initial.json)
- [High-occupancy extension](raw/2026-08-31-rohini-packed-8-system-extended.json)
- [Single-launch plateau extension](raw/2026-08-31-rohini-packed-8-system-final.json)
- [Normalized sustained sweep](raw/2026-08-31-rohini-packed-8-system-sustained.json)
