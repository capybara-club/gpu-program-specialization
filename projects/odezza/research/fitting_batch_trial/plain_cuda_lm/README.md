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

# Plain CUDA cooperative LM trial

This trial compiles a complete CUDA kernel with fixed nonlinear RHS expressions.
There are no patch sites, SASS rewriting, template inspection, runtime AST
interpretation or calls into the Odezza specialization pipeline. Changing an RHS
requires ordinary CUDA recompilation. This is an isolated topology prototype,
not a replacement for production fitting.

`kernel.cuh` extends a preserved copy of
`scratch/cooperative_lm/cooperative_lm.cu`. `experiment.py` renders the small fixed
RHS function into `model.cuh` and produces the complete inspectable `.cu` files.
`dual.cuh` implements analytic directional forward differentiation. The kernel
itself integrates the state and its parameter sensitivities using RK4 and runs
damped Gauss–Newton/LM iterations entirely on the GPU.

## Ownership and interface

One subgroup of 1, 2, 4 or 8 lanes owns one parameter start and one candidate RHS.
Each lane stores complete sensitivity columns for its assigned parameters;
normal-equation entries and the packed Cholesky factor/solve vectors are also
distributed. Warp shuffles exchange the values required for accumulation and
factorization. The primal state is replicated and RHS values are broadcast from
the subgroup leader to preserve consistent control flow. RHS directional
evaluation still repeats some primal work per owned parameter.

Immutable trajectory data is staged in shared memory. Mutable sensitivities
and optimizer data stay in lane-local arrays, which the compiler can allocate
to registers or local memory; both resource metrics are reported. No promise of
spill-free execution is made for arbitrary shapes. There is one 32-thread CTA
per group of independent fits; there are no cross-CTA optimizer dependencies.

The kernel takes starts/count, trajectory offsets/times, state-major observations
and weights, integration/iteration controls, and output pointers for coefficients,
initial/final MSE and optimizer counters. Complete initial states are supplied
in each trajectory's first reference row and excluded from scoring. Zero-weight
observations are excluded from the MSE denominator. Tail subgroups exit together
after shared-data staging. The test count of 67 intentionally leaves a partial CTA.

This prototype fixes coefficient bounds to [-1.5,1.5], target MSE to 1e-13,
and has no production JSON adapter. Relative-progress stopping was tightened
from the old control's absolute stopping floor. It does not implement every
production stopping/bounds/status option. Comparisons here hold this prototype's
own algorithm and starts constant across subgroup widths; they are not a fair
end-to-end speed comparison against the production fitter.

## Build and execution

The attempted installed `nvcc` build failed on incompatible `rsqrt`/`rsqrtf`
exception specifications in CUDA and system headers, including device-only
compilation. That failed attempt is retained in `validation/plain-cuda-lm-01`.
The working build uses installed NVRTC via the repository's existing
`lm_toggle/gpu.py` bindings. No dependency was installed or system header changed.
This remains full CUDA source compilation; only the compiler entry point changed.
NVRTC version, options (including the existing helper's `--use_fast_math`),
compilation time, cubin and source are saved for every width.

`run_gpu.py` reuses the existing CUDA driver resource helpers but supplies the
correct subgroup launch grid. It waits on CUDA events after kernels and copies;
there is no device or full-stream synchronize. Compilation and module load are
reported separately from CUDA event timings. Three identical launches are timed,
with the last two used as warm observations.

Example on rack1, from `scratch/fitting_batch_trial`:

```sh
/home/cdurham/odezza/scratch/pysr_rollout_trial/.venv/bin/python -B \
  plain_cuda_lm/experiment.py \
  --inputs validation/random-fit-inputs-01 \
  --out validation/plain-cuda-lm-new \
  --widths 1 2 4 8
```

The output directory must be new; existing experiment results are not overwritten.

## Validation scope

- Saved eight-state/six-parameter nonlinear case whose specialized LM exhausted
  registers. All four public candidate structures are compiled; private truth
  and generation seed are not read. Its previously consumed validation data is
  a diagnostic, not a fresh held-out benchmark. No final test is opened.
- Separate 12/16-state nonlinear six-parameter calibration controls, with
  independently generated DOP853 trajectories checked against Radau. These use
  a stable known background, a fixed correct nonlinear structure, and starts
  independent of the true coefficients. They are not random recovery problems.
- GPU analytic local derivatives with respect to every state and coefficient
  are checked against independent CPU FP64 central differences, along with RHS
  values. The probe uses public independent values, not true coefficients.
- The training-best candidate at every width is independently integrated on
  training and validation with FP64 DOP853 and Radau. Aggregate and every observed
  state must have MSE <=1e-10, and solver predictions must agree within 1e-8.
- All starts' coefficients, initial/final MSEs and optimizer counters are retained
  and compared across widths. Different floating-point layouts may produce small
  differences; these must be reported rather than hidden by comparing only winners.

The 67-fit experiments primarily measure feasibility and lane ownership. They
use too few CTAs to establish saturated throughput. Register savings, changes in
available parallelism and instruction costs all contribute to timing differences.
No core kernel generator, specialization limit or production service is modified.

The [smaller-state follow-up](SMALL_STATE_SCALING.md) adds `--cases 3 6` and
`--parameters 3 6`, with 1,024 and 32,768 starts. Three-state controls reuse state
indices modulo three. Three-parameter controls hold three inner coefficients
fixed. Its three-state/six-parameter two-lane output discrepancy is explicitly
flagged; do not extend the original matrix's exact-equivalence claim to that cell.
