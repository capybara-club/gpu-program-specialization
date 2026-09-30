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

# Plain CUDA cooperative LM results

Completed on rack1 (RTX 5080), 2026-09-09. The cooperative prototype runs the saved eight-state/six-parameter case without specialization and fits separate nonlinear 12/16-state controls successfully. One-thread ordinary CUDA also compiles, but uses local memory; cooperation removes that allocation for these shapes.

## 67 identical starts per width

| System | Lanes per fit | Registers/thread | Local bytes/thread | Warm kernel ms | Validation |
|---|---:|---:|---:|---:|---|
| n8-d3-p6-r1-dense | 1 | 255 | 64 | 2761.289 | fail |
| n8-d3-p6-r1-dense | 2 | 198 | 0 | 886.389 | fail |
| n8-d3-p6-r1-dense | 4 | 167 | 0 | 513.682 | fail |
| n8-d3-p6-r1-dense | 8 | 128 | 0 | 280.425 | fail |
| nonlinear-n12-p6 | 1 | 255 | 672 | 694.621 | pass |
| nonlinear-n12-p6 | 2 | 255 | 48 | 251.337 | pass |
| nonlinear-n12-p6 | 4 | 219 | 0 | 145.467 | pass |
| nonlinear-n12-p6 | 8 | 146 | 0 | 171.310 | pass |
| nonlinear-n16-p6 | 1 | 255 | 1152 | 2402.787 | pass |
| nonlinear-n16-p6 | 2 | 255 | 456 | 514.294 | pass |
| nonlinear-n16-p6 | 4 | 255 | 0 | 190.777 | pass |
| nonlinear-n16-p6 | 8 | 168 | 0 | 173.296 | pass |

## 1,024-start check on the same 16-state control

| Lanes per fit | Registers/thread | Local bytes/thread | Warm kernel ms | Relative to one lane |
|---|---:|---:|---:|---:|
| 1 | 255 | 1152 | 3002.495 | 1.00x |
| 4 | 255 | 0 | 272.623 | 11.01x |
| 8 | 168 | 0 | 237.663 | 12.63x |

All 1,024 starts have exactly equal returned coefficients, initial/final MSE, iteration, accepted-step and factorization-attempt counters across widths. The 67-start runs also match exactly across all four widths on each system. These comparisons use the full returned arrays, not just the winner.

Analytic state/parameter derivative and primal probes pass on every compiled shape (maximum normalized difference from CPU FP64 references below 7e-8; threshold 1e-4). Independent DOP853/Radau trajectory validation passes for the 12- and 16-state controls, with worst observed-state MSEs of 1.17e-13 and 5.39e-14 in the 67-start runs. The saved eight-state case passes training but fails validation: worst-state MSE 3.27e-10 versus 1e-10. We have demonstrated execution and equivalent fitting, not recovery of that case.

CUDA compute-sanitizer memcheck reports zero errors on a 16-state/four-lane nine-start run, including a partial final CTA. Sanitizer timings are excluded. No final-test data was consumed.

## Interpretation and boundaries

These results support continuing with subgroup-owned sensitivities and distributed optimizer storage. Four lanes remove local memory in all three tested cases; eight lanes further reduce register demand. Eight lanes are not universally fastest: four lanes win on the 12-state small batch. A dispatcher should consider state count, coefficient count and available independent fits.

Kernel times are CUDA event measurements averaged over the last two of three launches. Compile, module load, CPU checking and transfers are excluded. NVRTC options/version and compilation times are preserved per cell. Compilation was subsecond in the first matrix; later identical-source compiles can benefit from compiler caching. No inference about Odezza specialization/module-load performance follows from those times.

Even 1,024 starts is not a saturated-throughput proof: the one-lane grid has only 32 CTAs, while four/eight lanes launch 128/256. More distributed parallelism, removal of local memory and changed instruction costs jointly affect speed. The speed ratios apply only to this fixed-CUDA prototype and workload, not production LM, curvature or whole recovery.

The 12/16-state controls use six coefficients and one known nonlinear structure with a stable known background. This does not establish all-RHS fitting, larger coefficient counts, arbitrary dense coupling or stiffness. Starts are independent of truth. The reused eight-state validation set is diagnostic, not untouched evidence.

The old control was copied into a separate folder. New nonlinear directional differentiation, bounded proposals, a tighter progress criterion and zero-weight residual counting were added. None of the source-specialization machinery or production numerical kernels were changed.

The installed nvcc failed on CUDA/system-header exception-specification conflicts. Installed NVRTC successfully compiles the same ordinary device CUDA via existing driver helpers; no installation or header modification was performed. The first failed build remains on rack1 under validation/plain-cuda-lm-01.

## Artifacts

- [Kernel source](kernel.cuh)
- [Complete fixed 16-state/four-lane CUDA source](../validation/plain-cuda-lm-03/nonlinear-n16-p6/kernel-w4.cu)
- [Protocol, ownership and build details](README.md)
- [Small-matrix results](../validation/plain-cuda-lm-03/results.json)
- [Larger-batch results](../validation/plain-cuda-lm-1024/results.json)
- [CUDA memory-check log](../validation/plain-cuda-lm-memcheck.log)
