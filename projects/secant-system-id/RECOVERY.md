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

# Controlled fed-batch recovery

## Question and protocol

The recovery experiment blinds both nonlinear rate laws in the four-state
fed-batch astaxanthin system while leaving its mass-balance equations fixed.
Each system genome owns two independent postorder ASTs, a shared constant bank,
and dynamic leaf bindings. GP varies structure, constants, and state/constant
leaf choices together. No planted term or binding is injected into the search.

Search quality is evaluated at three levels:

1. training trajectory loss, used by GP;
2. held-out trajectory MSE on six unseen initial conditions; and
3. normalized rate-surface RMSE against the planted rate laws on a grid of
   biomass, glucose, and sucrose states.

Exact structural recovery is also reported after resolving dynamic bindings,
anonymizing fitted constants, and canonicalizing commutative expressions. A
numerically equivalent simplified second law is accepted when its planted
inhibition coefficient is zero. Numerical rate recovery is the main metric;
trajectory loss alone can hide incorrect but compensating dynamics.

Checkpoint analysis is outside the timed search. Every run records best-so-far
programs, constants, bindings, objective, and elapsed search time at a chosen
generation stride. Raw checkpoint JSONL is written durably after each seed,
and `progress.json` reports the current seed and all completed seeds. The final
report adds recovery metrics and an independent materialized GPU replay.

## Training design and loss

The first diverse design used 12 trajectories spanning low/high biomass,
glucose, and sucrose, all with product equal to zero. It admitted candidates
with spurious product dependence: those candidates could fit training yet
become singular or inaccurate away from the training surface.

The current `product_paired16` design uses the eight corners of the same
biomass/glucose/sucrose box and repeats every corner at product levels 0 and 4.
This exposes false product dependence without banning product from the search
language. Relative state-space MSE with denominator floor 1 is used so the
largest-magnitude state does not dominate selection.

The search language is deliberately modest: `+`, `*`, and protected `/`, with
at most 30 nodes per AST, initial trees of at most 15 nodes, depth at most 8,
and a small parsimony penalty. Subtraction and unary operators are opt-in.

## Pilot evidence

The 12-trajectory pilot ran ten seeds for 2,000 generations on an RTX 5090.
All ten reached training MSE below `1e-3`, but only two reached worst-site rate
NRMSE below 3%, and three produced invalid off-surface rate laws. The two good
seeds reached joint rate NRMSE of 0.52% and 0.92%. This established that the
kernel and GP can recover accurate laws, while also revealing that the original
training design did not identify product invariance.

A three-seed, 1,000-generation calibration of `product_paired16` with relative
loss produced one strong RTX 5090 recovery: training MSE `1.50e-6`, held-out
trajectory MSE `2.96e-5`, and per-site rate NRMSE 1.38% and 0.31%. It crossed
worst-site rate NRMSE 1% at checkpoint generation 560. The sample is too small
to estimate recovery probability; it only justifies the longer campaign.

An RTX 4090 calibration used 256 settings, full-MSE output, and C-materialized
settings. Its three final GPU scores reproduced exactly through the independent
materialized replay. One seed reached training MSE `3.65e-6`, held-out MSE
`1.24e-4`, and per-site rate NRMSE 2.81% and 0.11%.

## Architecture-specific execution

On `sm_120` (RTX 5090), the validated fast path uses stateless hashed settings
and the fused GPU winner reduction. The product-paired calibration sustained
about 13.8 million genome/setting configurations per second.

Long-run validation exposed one reducer boundary: finite trajectory states can
still accumulate an infinite FP32 SSE. Previously, a genome whose every tile
had this result left the second-stage setting index at `UINT_MAX`. Scoring now
maps every non-finite accumulated SSE to `FLT_MAX` before reduction, and the
reducer defensively emits setting zero if no usable tile remains. The original
seed reproduced the failure at generation 13,508; with the fix it completed
14,000 generations and its fused result matched materialized replay exactly.

On `sm_89` (RTX 4090), replay testing found that the fused winner score/index
pair could disagree with a materialized replay. Exact-zero score rejection
catches one symptom but does not repair the reducer. Until that path is fixed,
controlled results use C-materialized settings, full GPU MSE output, and a C
argmin. The validated fallback sustained about 4.3 million configurations per
second at 256 settings. Results from the faulty fused path are not used.

## Current long campaign

The longer comparison fixes population, grammar, training design, generation
count, and recovery metrics across machines. Rohini uses 512 settings and 15
seeds on the validated fused path; Ada uses 256 settings and 10 seeds on the
validated materialized/full-MSE path. Each seed receives 50,000 generations.
This is a recovery-probability and time-to-quality experiment, not yet a claim
of structural reliability or superiority to another system.
