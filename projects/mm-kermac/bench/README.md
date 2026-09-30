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

# Kernel Matrix Benchmark Strategies

This folder contains `kermac_kernel_matrix_bench` (see `bench_kernel_matrix.cpp`).
It also contains `kermac_solve_bench` (see `bench_solve.cpp`), which generates a
batched symmetric Laplace kernel matrix and times the Cholesky solve.
`kermac_solve_only_bench` (see `bench_solve_only.cpp`) skips the Laplace kernel
generation entirely and benchmarks only the solve path on a generated SPD matrix.
`bench_solve_only_pytorch.py` matches that solve-only setup in PyTorch on CUDA.
The benchmark prints a labeled line for each strategy. The labels below match
the output so you can tie performance back to the implementation choice.
`kermac_philox_bench` (see `bench_philox.cu`) is a standalone Philox RNG throughput
benchmark that only measures Philox generation and device fills.

## Shapes and kernel definition

- A: M x K x L
- B: N x K x L
- Kernel matrix: M x N x L
- Kernel matrix gradient: M x N x L
- Gradient output: M x K x C x L
- AGOP output: D x D x L (full) or D x L (diag), where D == K in this bench

The Laplace kernel uses an L2 distance:

- dist(a, b) = sqrt(sum_k (a_k - b_k)^2)
- K(a, b) = exp(-dist / bandwidth)

Symmetric Laplace variants also clamp very small distances (epsilon) and
regularize the diagonal by setting it to 1.0 + regularizer.

## Non-symmetric kernel matrix (M x N)

### semiring_laplace_l2
Computes the Laplace kernel inside the `cute_semiring` tiled kernel via PTX
injection (see `cuda/semiring/base_semiring.cuh` and `k_semiring.c`). The injected
"mma" step accumulates (a - b)^2 into the accumulator fragment, and the epilogue
applies sqrt and exp(-dist / bandwidth).
- Slow when: M, N, or K are small so launch overhead and epilogue work dominate.
- Slow when: shapes are awkward for the internal tiling and reduce occupancy.
- Note: the injected math is scalar PTX ops (sub/mul/add/sqrt/exp); this
  implementation does not invoke tensor-core MMA.
- Fast when: M and N are large enough to amortize launch and keep the kernel busy.
- Fast when: memory bandwidth is tight, because it avoids extra intermediate tensors.

### legacy_dot+laplace
Separates the computation into multiple steps: three cuTensor contractions
(`kermac_contraction`) for A * B^T and row-wise sum-of-squares for A/B, then a
legacy Laplace kernel (`kermac_legacy_laplace`) combines those outputs.
- Slow when: M and N are large because it writes and rereads multiple large tensors.
- Slow when: L is large, since all intermediate tensors scale with batch count.
- Fast when: problem sizes are small and the overhead of the fused semiring kernel
  is not amortized.
- Fast when: you need the dot products or norms for other work and can reuse them.

## L2 gradient (M x K x C)

### semiring_grad_l2
Uses the `cute_semiring_gradient` kernel with PTX-injected multiply/accumulate
(see `cuda/semiring/base_semiring_grad.cuh` and `k_semiring_grad.c`). The injected
math is: diff = (data_m - data_n) * kernel_matrix, then gradient += solution * diff.
This fuses difference, scaling, and accumulation into one kernel.
- Slow when: N or C are small, since parallelism is limited and overhead dominates.
- Slow when: K is tiny and the fused kernel cannot hide latency.
- Fast when: N and C are large enough to keep the fused kernel busy and reduce
  memory traffic vs multi-step approaches.

### cutensor_grad_l2
Uses cuTensor contractions to form the same gradient. It relies on library
planning and may allocate workspace for the contraction plan.
- Slow when: M, N, K, or C are small and plan/launch overhead dominates.
- Slow when: memory is tight and workspace allocation becomes the bottleneck.
- Fast when: sizes are large enough for cuTensor to amortize planning and use a
  tuned contraction plan; tensor core usage depends on the selected TensorCoreMode.

## AGOP (D x D) and AGOP (D)

AGOP in this bench computes a feature matrix from the per-sample gradients.
It first forms the gradient (M x D x C), then contracts grad with itself and
scales by 1/M. The "full" output is D x D; "diag" keeps only the diagonal.

### agop_full_cutensor_f32
Computes the gradient with cuTensor (F32 compute descriptor) and then runs
`kermac_contraction` on grad x grad to get the full D x D AGOP matrix.
- Slow when: D is large, since the output is O(D^2) and memory traffic is high.
- Slow when: M or C are small, because the contraction is underutilized.
- Fast when: D and C are large enough to benefit from cuTensor optimizations.

### agop_full_fused
Uses the semiring-gradient kernel for the gradient stage; the final grad x grad
contraction still goes through `kermac_contraction` (cuTensor).
- Slow when: M, N, or C are small and the fused gradient does not amortize overhead.
- Slow when: D is large and the final D x D contraction dominates.
- Fast when: the fused gradient path is faster than cuTensor for your shapes.

### agop_diag_cutensor_f32
Same as the full cutensor path, but only computes the diagonal of the D x D
feature matrix, which is O(D).
- Slow when: M or C are very small and contraction overhead dominates.
- Fast when: D is large and you only need the diagonal (avoids O(D^2) work).

### agop_diag_fused
Uses the semiring-gradient path; the final diagonal contraction still goes
through `kermac_contraction` (cuTensor).
- Slow when: M, N, or C are small and the fused gradient overhead dominates.
- Fast when: D is large and you only need the diagonal, and the fused gradient
  beats the cuTensor gradient for your shapes.

## Symmetric kernel matrix (M x M)

### semiring_laplace_symm_full
Semiring Laplace kernel for symmetric inputs. It computes the full matrix, and
the epilogue clamps small distances (epsilon) and writes the diagonal as
1.0 + regularizer.
- Slow when: you do not need the full matrix, because it computes all entries.
- Slow when: M is small, since the extra epilogue logic can dominate.
- Fast when: M is large and you need the full dense output.

### semiring_laplace_symm_lower
Same fused semiring kernel but only writes the lower triangle (packed type
lower). This saves roughly half the work and output for large M.
- Slow when: you ultimately need the full matrix (you will have to mirror it).
- Slow when: M is small and triangular masking overhead dominates.
- Fast when: M is large and a triangular packed matrix is sufficient downstream.

### semiring_laplace_symm_upper
Upper-triangle version of the semiring kernel. Performance tradeoffs match the
lower-triangle case, but the stored triangle is different.
- Slow when: you need the opposite triangle and must post-process.
- Fast when: you can consume upper-triangular storage directly.

### legacy_laplace_symm_full
Computes A * A^T via contraction, then uses the legacy symmetric Laplace kernel.
The legacy path copies the diagonal into a temporary norm tensor before applying
the Laplace transform and diagonal regularization.
- Slow when: M is large due to the extra tensor and extra kernel launches.
- Slow when: memory bandwidth is the limiting factor.
- Fast when: M is small and the simple kernel sequence is sufficient.

### legacy_laplace_symm_lower
Legacy symmetric Laplace kernel that only writes the lower triangle.
- Slow when: you need the full matrix, because you must mirror it later.
- Slow when: M is large and intermediate memory traffic dominates.
- Fast when: M is large and you can stay in packed lower-triangular form.

### legacy_laplace_symm_upper
Legacy symmetric Laplace kernel that only writes the upper triangle.
- Slow when: you need the opposite triangle.
- Fast when: packed upper-triangular storage is the natural input to the next step.
