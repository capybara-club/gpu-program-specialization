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

# Coupled RK4 sensitivity-stage ordering defect

Date: 2026-09-02  
Status: resolved and directly validated on RTX 5090 and RTX 4090  
Affected generator: trajectory LM through commit `f91444e`  
Corrected generator ABI: 2 (`simultaneous-stage-rk4-v2`)

## Summary

The trajectory-LM CUDA generator updated one output state's RK4 stage sensitivity before computing the sensitivity derivative for the next output state. In a coupled system, a later right-hand side could therefore read a mixture of old and newly updated stage sensitivities. This was a Gauss-Seidel-like update, not the simultaneous augmented-system RK4 update required by forward sensitivities.

The primal state trajectory, final candidate MSE, and independent CPU replay of a returned winner were unaffected. The LM Jacobian, proposal direction, damping path, acceptance history, and search trajectory were not generally correct for coupled equations. Previous LM throughput measurements also describe a slightly different amount of optimizer work and are superseded by the corrected measurements below.

## Intended and observed behavior

For each RK4 stage and optimized parameter, every output derivative must be computed from the same unchanged stage sensitivity vector:

```text
dS_i/dt = df_i/dp + sum_j (df_i/dx_j) S_j
```

Only after all `dS_i/dt` values are available may the next stage sensitivity vector be written. The affected generator instead computed `dS_i/dt` and immediately replaced `S_i`, allowing subsequent outputs to observe the replacement.

A two-state coupled linear diagnostic,

```text
x' = p*x + 2*y
y' = 3*x - y
```

measured a maximum sensitivity error of `8.580325e-3` for the old in-place update. Simultaneous augmented RK4 agreed with a centered finite difference to `3.69e-12`.

## Repair

Both grouped-v4 and two-site-v5 templates now compute every output's stage derivative into a short-lived `stage_derivative[state]` array before updating any stage sensitivity. The persistent `sensitivities[state,parameter]` array remains the unchanged base of the current RK4 step, so the separate full-size `base_sensitivity` array was removed.

The generator ABI was increased from 1 to 2 and the manifest now records `simultaneous-stage-rk4-v2`. Old generated source is therefore rejected by the current manifest parser rather than silently entering a current cache.

## Validation

The local suite contains a coupled finite-difference RK4 regression and a source-structure regression for both physical layouts. All 35 tests pass.

A focused GPU test used the coupled system above, three trajectories, nine observations, four RK4 substeps per interval, and one LM iteration. An independent CPU augmented-RK4 implementation proposed `p = 0.7786147079923798`; corrected GPU kernels proposed `0.7786146402359009`, an absolute difference of `6.78e-8`. CPU and GPU proposal MSEs were `1.38841615e-2` and `1.38841774e-2`. Two-site specialization passed on both GPUs; grouped specialization passed on Rohini. The Ada grouped microshape encountered the separate marker-uniqueness limitation described below.

The full fed-batch 4-state/6-constant workload produced identical results and optimizer counters across both layouts and both GPUs:

- Best GPU MSE: `2.6223906265899696e-14`
- Independent CPU replay MSE: `9.650164828615946e-15`
- Maximum scaled parameter error: `8.8437e-7`
- Mean iterations / accepted steps / factorization attempts: `9.3188934 / 9.2438965 / 12.5062256`

No corrected 4x6 or 2x8 template has stack or spill traffic.

## Corrected performance and resources

| Workload | GPU | Layout | Fits/s | Registers | Stack | Static shared |
|---|---|---|---:|---:|---:|---:|
| Fed-batch 4x6 | RTX 5090 | grouped-v4 | 85,821 | 224 | 0 B | 1,024 B |
| Fed-batch 4x6 | RTX 5090 | two-site-v5 | 88,767 | 216 specialized | 0 B | 6,144 B |
| Fed-batch 4x6 | RTX 4090 | grouped-v4 | 79,563 | 225 | 0 B | 0 B |
| Fed-batch 4x6 | RTX 4090 | two-site-v5 | 64,543 | 215 specialized | 0 B | 5,120 B |
| Synthetic 2x8 | RTX 5090 | grouped-v4 | 7.078M | 188 | 0 B | 1,024 B |
| Synthetic 2x8 | RTX 5090 | two-site-v5 | 6.845M | 174 | 0 B | 3,584 B |
| Synthetic 2x8 | RTX 4090 | grouped-v4 | 5.473M | 196 | 0 B | 0 B |
| Synthetic 2x8 | RTX 4090 | two-site-v5 | 5.529M | 174 | 0 B | 2,560 B |

An 8-state/8-constant two-site compile still spills: 512 stack bytes per thread on Rohini and 536 on Ada, both at 255 registers. This is an active topology boundary, not a resolved capacity claim.

## Scope of prior results

All pre-repair trajectory-LM gradient correctness and fit-rate claims are superseded. Primal trajectory scoring and the CPU replay MSEs of materialized winners remain valid. Relative layout measurements remain useful only as historical implementation evidence because corrected LM follows a different proposal and damping path.

## Separate active inspector limitation

Very small 2x1 templates fail physical inspection on both GPUs because PTXAS duplicates a scalar marker and the inspector requires one physical match. A 2x3 grouped template also shows this collision on Ada; its two-site form works. This limitation predates and is independent of the sensitivity-order repair. It remains open and is not silently bypassed.
