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

# Kernel-Matrix Benchmark Strategy Analysis

This folder contains `bench_kernel_matrix.cpp`, a CUDA benchmark that compares multiple implementation *strategies* for:

- Building a Laplace (L2) kernel matrix (non-symmetric `M×N` and symmetric `M×M`)
- Computing an “L2 gradient” tensor (`M×D×C`)
- Computing an AGOP matrix (`D×D`) or diagonal (`D`)

The benchmark prints one line per strategy, e.g. `semiring_laplace_l2`, `legacy_dot+laplace`, `cutensor_grad_l2`, etc. This document explains what each strategy is doing and when it is likely to be slow/fast as you vary problem size.

---

## Notation (matches `bench_kernel_matrix.cpp`)

**Dimensions**

- `M`: number of “rows” / samples in set A (`d_a`)
- `N`: number of “cols” / samples in set B (`d_b`)
- `D` (called `K` in the benchmark): feature dimension per sample
- `C`: number of “channels” (think: number of right-hand-sides / output dims)
- `L`: batch count

**Tensors (batched case shown; without batching drop the `l` index)**

- `A[m,d,l]` = `d_a` with shape `(M, D, L)`
- `B[n,d,l]` = `d_b` with shape `(N, D, L)`
- `K[m,n,l]` = kernel matrix with shape `(M, N, L)` (or `(M, M, L)` for symmetric)
- `S[n,c,l]` = “solution” with shape `(N, C, L)`
- `G[m,d,c,l]` = gradient with shape `(M, D, C, L)`

**Laplace-L2 kernel**

The legacy CUDA implementation (`mm-kermac/cuda/k_legacy.cu`) makes the intent explicit:

- First compute squared distance:
  - `d2(m,n,l) = ||A[m,:,l]||² + ||B[n,:,l]||² - 2·(A[m,:,l]·B[n,:,l])`
  - clamped as `d2 = max(0, d2)` for numerical safety
- Then compute:
  - `dist = sqrt(d2)`
  - `K = exp(-dist / bandwidth)`
  - if `dist < epsilon`, treat as distance `0` → `K = 1` (stability)
- In the symmetric case `m==n`, set `K = 1 + regularizer` (diagonal regularization)

Note: `epsilon` is intended as a “near zero distance” threshold (see `compute_expected_laplace_symm` in `mm-kermac/tests/test_semiring_gold.cpp`). The **semiring** symmetric path compares `epsilon` against `dist`, while the **legacy** Laplace kernel compares `epsilon` against `d2` (squared distance). With the benchmark default (`1e-5`) this rarely matters; with larger `epsilon` it can.

---

## Strategy family 1: “Semiring” (fused, custom kernel)

### What “semiring” means here

`kermac::Semiring` and `kermac::SemiringGradient` are *runtime-compiled* CUDA kernels built by injecting small PTX instruction sequences into pre-written tiled kernels (`cute_semiring` / `cute_semiring_gradient`). Conceptually, they implement a generalized GEMM:

- “multiply” step: custom per-element function of `a` and `b`
- “accumulate” step: custom reduction into an accumulator
- “epilogue” step: final per-output transform (e.g. `sqrt`, `exp`, diagonal fixups)

In this benchmark the key point is: **the semiring path is designed to avoid extra passes over giant intermediate tensors** (dot-products, per-row norms, temporary gradients).

### `semiring_laplace_l2` (non-symmetric `M×N`)

**Purpose:** compute the Laplace-L2 kernel matrix directly.

**How it works (high level):**

- The injected “MMA” instruction sequence for L2 uses the identity:
  - accumulate `d2 += (a - b)²` over `d=0..D-1`
  - (see `mma_l2_norm_instructions` in `mm-kermac/k_semiring.c`)
- The epilogue then:
  - `dist = sqrt(d2)`
  - `K = exp(-dist / bandwidth)` (approximate `ex2.approx` path)

**Why it can be slow:**

- When `D` is large and the kernel becomes compute-bound, the per-`d` work is “sub + mul + add” (for `(a-b)²`) rather than a single FMA; depending on how the tiled kernel maps to hardware, this can lose to highly tuned GEMM-based approaches.
- When `M` and/or `N` are small, the kernel may not launch enough blocks to fully occupy the GPU (underutilization); you pay launch overhead without enough parallel work.
- If you recreate the semiring object frequently (instead of caching it), the one-time PTX injection / compilation cost can dominate. The benchmark creates semirings once and does *not* time this cost.

**When it should be faster:**

- When `M×N` is large enough that the extra memory traffic in multi-pass approaches (read/write of intermediates) becomes a bottleneck.
- When `D` is modest and the workload is bandwidth/latency sensitive (fewer kernels, fewer global reads/writes).
- When you repeatedly run the same semiring (same op + hyperparameters), amortizing its creation cost.

### `semiring_laplace_symm_{full,lower,upper}` (symmetric `M×M`)

**Purpose:** compute a symmetric Laplace-L2 kernel matrix with diagonal regularization.

**How it works:**

- Same fused L2 accumulation and Laplace epilogue as above, plus:
  - clamp very small distances to `0` before `exp` (via `epsilon`)
  - if `m==n`, write `1 + regularizer`
- The `{lower,upper}` variants use `MatrixPackedType` to only compute one triangle of the matrix.

**Why it can be slow:**

- For `{lower,upper}`, you run fewer thread blocks (roughly half), which can reduce occupancy and make small/medium `M` cases slower than `full`.
- If downstream code actually needs the *full* matrix, you still have to mirror the other triangle (extra pass), which can erase some of the benefit.

**When it should be faster:**

- Large `M` where avoiding redundant work on the second triangle is a big win *and* downstream can consume a packed/triangular matrix (or a mirrored fill is cheap relative to full compute).

### `semiring_grad_l2` (gradient `M×D×C`)

**Purpose:** compute a gradient tensor without large intermediate temporaries.

The cutensor implementation makes the mathematical shape clear (see `kermac_cutensor_gradient_norm_l2` in `mm-kermac/k_cutensor.c`):

- `G[m,d,c,l] = (1/bandwidth) * Σ_n ( S[n,c,l] * K[m,n,l] * (A[m,d,l] - B[n,d,l]) )`

**How it works:**

- A single custom kernel reads `K`, `S`, `A`, `B` and accumulates directly into `G`.
- No “term_x” intermediate is materialized.

**Why it can be slow:**

- If `N` is small, the reduction is shallow and the kernel can become dominated by overhead (launch + address math).
- If `C` is very small (e.g., `C=1`), there is less independent work per `m,d` and performance may fall off due to reduced reuse.

**When it should be faster:**

- Large `M×D×C` where a two-pass method would write/read a massive intermediate tensor.
- Shapes where the memory footprint of an intermediate (`M×D×C×L`) would thrash caches and saturate HBM.

---

## Strategy family 2: “Legacy” (cuTENSOR contractions + custom Laplace kernel)

The “legacy” strategies decompose the computation into familiar building blocks:

1. Use `kermac::contraction(...)` (implemented via cuTENSOR in `mm-kermac/k_cutensor.c`) to compute dot-products and norms.
2. Use a purpose-built CUDA kernel (`kernel_laplace_*` in `mm-kermac/cuda/k_legacy.cu`) to transform dot-products + norms into Laplace kernel values (and optionally mirror triangles / regularize the diagonal).

### `legacy_dot+laplace` (non-symmetric `M×N`)

**Purpose:** compute the same Laplace-L2 kernel matrix as `semiring_laplace_l2`, but via dot-products.

**How it works:**

- Compute dot-products: `P[m,n,l] = Σ_d A[m,d,l] * B[n,d,l]`
- Compute norms:
  - `na[m,l] = Σ_d A[m,d,l]²`
  - `nb[n,l] = Σ_d B[n,d,l]²`
- Transform in-place (overwriting `P`):
  - `d2 = na[m,l] + nb[n,l] - 2·P[m,n,l]`
  - `K = exp(-sqrt(max(0,d2)) / bandwidth)` with `epsilon` stability (note: legacy checks `d2 < epsilon`)

**Why it can be slow:**

- More kernel launches: 3 contractions + 1 Laplace kernel (vs. 1 semiring kernel).
- More global memory traffic: dot-product matrix is produced, then read+written again by the Laplace kernel.
- cuTENSOR planning overhead: `kermac_contraction` creates/destroys cuTENSOR descriptors and plans *each call*. For smaller problems, this overhead can dominate.

**When it should be faster:**

- Very large `D` where the dot-product GEMM is the dominant cost and cuTENSOR’s highly tuned kernels outperform a fused custom kernel.
- Cases where you *already* need dot-products and/or norms for other work (so the “extra” steps aren’t truly extra).

### `legacy_laplace_symm_{full,lower,upper}` (symmetric `M×M`)

**Purpose:** symmetric Laplace-L2 kernel matrix, with diagonal regularization and optional triangle mirroring.

**How it works:**

- Compute dot-products: `P[m,n,l] = Σ_d A[m,d,l] * A[n,d,l]`
- Inside `kermac_legacy_laplace_symmetric`:
  - Extract diagonal `P[m,m,l]` as norms (cheap)
  - Apply Laplace transform
  - For `{lower,upper}`, the kernel also mirrors the computed triangle into the other half (see `SymmetricCopy` path in `mm-kermac/cuda/k_legacy.cu`)

**Why it can be slow:**

- Same cuTENSOR planning overhead concerns as above for small/medium sizes.
- For `{lower,upper}` you add a “copy/mirror” cost (extra shared-memory staging and extra writes) that can blunt the savings from skipping half the compute, especially if you’re bandwidth-bound.

**When it should be faster:**

- Large `M` where computing a single triangle + mirroring is still much cheaper than full compute, and you truly need a full symmetric matrix in memory.

---

## Strategy family 3: “cuTENSOR gradient” (two trinary contractions)

### `cutensor_grad_l2`

**Purpose:** compute the same gradient tensor as `semiring_grad_l2`, but using cuTENSOR contractions.

**How it works (from `mm-kermac/k_cutensor.c`):**

1. Materialize an intermediate:
   - `term_x[m,d,c,l] = Σ_n ( S[n,c,l] * K[m,n,l] * B[n,d,l] )`
2. Finish the gradient:
   - `G[m,d,c,l] = (1/bandwidth) * ( A[m,d,l] * Σ_n(S[n,c,l] * K[m,n,l]) - term_x[m,d,c,l] )`

**Why it can be slow:**

- **The intermediate `term_x` is enormous** (`M×D×C×L` floats). It must be written once and read again, which can dominate runtime for large problems.
- Two trinary contractions → more cuTENSOR planning overhead per iteration.
- For small sizes, cuTENSOR setup overhead can outweigh compute.

**When it should be faster:**

- If the shapes align perfectly with cuTENSOR’s fast paths and the workload is compute-bound (rather than bandwidth-bound).
- If you can reuse plans/descriptors across iterations (not what this benchmark currently does), cuTENSOR overhead can drop dramatically.

---

## Strategy family 4: AGOP (gradient + outer-product accumulation)

AGOP in this codebase is:

1. Compute `G[m,d,c,l]` as above (either semiring-gradient or cuTENSOR-gradient)
2. Accumulate a feature matrix via contraction:
   - **Full:** `F[d,e,l] = (1/M) * Σ_{m,c} G[m,d,c,l] * G[m,e,c,l]`  → shape `(D,D,L)`
   - **Diag:** `f[d,l]   = (1/M) * Σ_{m,c} G[m,d,c,l]²`             → shape `(D,L)`

### `agop_full_cutensor_f32` vs `agop_full_fused`

- Both compute a full `D×D` AGOP matrix.
- Difference is only how `G` is computed:
  - `*_cutensor_f32`: cuTENSOR trinary contractions (and a `term_x` intermediate)
  - `*_fused`: semiring gradient kernel (no `term_x`)

**Why `*_full_*` can be slow:**

- The `D×D` accumulation itself is expensive: work scales like `O(M·C·D²)`; for large `D`, this dominates and the choice of gradient backend matters less.

**When `*_fused` should help:**

- When `D` is modest but `M×C` is large enough that the gradient step is still significant, avoiding `term_x` reduces memory pressure.

### `agop_diag_cutensor_f32` vs `agop_diag_fused`

- Both compute only the diagonal, which is much cheaper (`O(M·C·D)`).
- This makes the gradient computation a larger fraction of total time, so backend differences are usually more visible than in the full-matrix case.

---

## Quick “rules of thumb” by problem shape

- **Small `M`/`N` (few thousand elements total):** expect semiring variants to look better because cuTENSOR plan/descriptor overhead is a fixed cost per call.
- **Large `M×N` with moderate `D`:** semiring Laplace can win by avoiding extra full-matrix read/write passes.
- **Very large `D`:** dot-product-based approaches can win because they turn most work into a best-in-class GEMM (then pay a relatively small epilogue cost).
- **Large `M×D×C`:** prefer `semiring_grad_l2` / `agop_*_fused` to avoid materializing and re-reading `term_x`.
- **Symmetric kernels:** if you can consume a triangle/packed representation, `{lower,upper}` variants can be nearly 2× cheaper; if you must materialize full symmetry, include the mirror/copy cost in your expectations.
