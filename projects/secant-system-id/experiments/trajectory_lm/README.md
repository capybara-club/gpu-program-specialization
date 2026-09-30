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

# Dense trajectory LM and postorder-gradient control

This experiment is the first trajectory-optimization shape in
`secant-system-id`. It evaluates useful GP leaf settings while retaining a
thread-owned, fully resident LM fit for the dense `product_paired16` fed-batch
design and relative trajectory loss.

## Contract

- one fixed two-rate rational AST shape;
- 16 dynamic leaf bindings per setting, indexing the shared bank
  `[X, S1, S2, P, c0, ..., c7]`;
- one independent `(setting, start)` fit per CUDA thread;
- every thread integrates and scores all 16 trajectories;
- a setting's leaf bindings remain fixed throughout LM;
- constants `c0` through `c5` are active LM variables, while `c6` and `c7`
  remain fixed during each fit;
- 12 observations, four states, and 16 RK4 steps per eight-hour interval;
- local rate partials are accumulated according to each dynamic binding;
- forward sensitivities through RK4;
- thread-local `J^T J`, `J^T r`, damped 6x6 Cholesky solve, proposal, and
  acceptance state;
- complete reference data cached once per CTA in shared memory;
- the per-thread state/constant bank stored in dynamic shared memory; and
- only final constants, MSE, iteration count, and accepted-step count written
  to global memory.

The setting layout matches the GP evaluator: state slots are encoded as `0..3`
and constant slots as `4..11`. The flattened output fit index is
`setting * starts_per_setting + start`. Invalid binding indices produce the
invalid-score sentinel instead of leaving output memory unwritten.

The default CUDA target contains the fixed rational AST shape and its local
derivative calculation. It remains the native correctness/performance control.
The specialized target uses two independently inspectable SASS sites lowered
from the same postorder AD representation: a two-rate primal site on every
trajectory evaluation and a 16-partial site only during LM statistics.

## Build and run on Ada

```sh
make -C experiments/trajectory_lm CUDA_ARCH=sm_89
make -C experiments/trajectory_lm run CUDA_ARCH=sm_89 SETTINGS=8192 STARTS=4 THREADS=256
make -C experiments/trajectory_lm specialized CUDA_ARCH=sm_89
make -C experiments/trajectory_lm run-specialized CUDA_ARCH=sm_89 SETTINGS=8192 STARTS=4 THREADS=256
```

The build uses the installed CUDA toolkit and repository-local sources. The
`-U_GNU_SOURCE` host compiler option is the established CUDA 13.1/glibc
compatibility flag for Ada.

The runner always includes the planted setting, generates deterministic broad
starts without planting the answer, and generates the remaining bindings with
the GP's 90% per-leaf keep probability.

## RTX 4090 result

CUDA 13.1 reports 237 registers per thread, 3,328 bytes of static shared
memory, `12 * blockDim.x * sizeof(float)` bytes of dynamic shared memory, no
stack, and no spill loads or stores. At 256 threads, the complete shared-memory
allocation is 15,616 bytes and the register allocation permits one CTA per SM.

With four starts per dynamic binding setting:

| Settings | Fits | CTAs at 256 threads | Time per launch | Fits/s |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 16,384 | 64 | 334.90 ms | 48,922 |
| 8,192 | 32,768 | 128 | 343.00 ms | 95,532 |
| 16,384 | 65,536 | 256 | 673.56 ms | 97,298 |

The knee is about 32,768 total fits: 128 resident CTAs give the RTX 4090 one
CTA per SM. More work improves throughput only slightly. The relevant
production quantity is therefore
`promoted genomes * binding settings * starts`, not thousands of starts for
one genome.

A 65,536-fit launch containing one planted binding and 65,536 starts reaches
121,339 fits/s and averages 9.31 LM iterations. The earlier fixed native-rate
control reached about 216,000 fits/s, so dynamic leaf loads and generalized
local partials retain roughly 56% of that deliberately narrow native control's
throughput. Randomized bindings average about 11.4 iterations and reach about
95,000--97,000 fits/s once saturated.

The planted setting still recovers all six parameters. In the 128-setting,
four-start correctness run, its best fit reached relative MSE `3.85e-14` and a
largest scaled parameter error of `5.55e-7`.

## Inspectable postorder derivative specialization

The original AST remains the source of truth. Python and C99 independently
build the same versioned, byte-identical postorder AD tape. The first SASS
backend derives one postorder primal and one postorder partial per local leaf,
then hash-conses outputs within a site so shared subexpressions are emitted
only once. The two planted 15-node rate laws use 18 SASS instructions in the
primal site and 96 in the statistics-only partial site. An independent scalar
lowering would use 334 instructions.

No derivative scratch is allocated for this shape. The specialized CUBIN uses
255 registers per thread, 3,328 bytes of static shared memory, the same dynamic
bank as the native control, and no local memory, stack, or spills. At the
32,768-fit saturation point on the RTX 4090:

| Gradient implementation | Time per launch | Fits/s | Relative throughput |
| --- | ---: | ---: | ---: |
| Fixed hand-written CUDA | 342.70 ms | 95,618 | 1.00x |
| Combined primal/partial SASS site | 937.22 ms | 34,963 | 0.37x |
| Split postorder-specialized SASS sites | 528.54 ms | 61,998 | 0.65x |

All three recover the planted parameters at approximately machine precision.
The split removes partial evaluation from proposal-only phases without adding
a kernel launch or unloading the module. Statistics phases execute the primal
site followed by the partial site; proposal phases execute only the primal.

At 255 registers per thread, 256-thread CTAs remain the best measured topology:

| CTA threads | Split fits/s | Native fits/s |
| ---: | ---: | ---: |
| 64 | 57,409 | 90,450 |
| 128 | 57,246 | 91,597 |
| 256 | 61,998 | 95,618 |

The register ceiling therefore has not made subgroup ownership preferable for
this shape. All three CTA sizes have enough blocks to cover the RTX 4090, and
the 256-thread thread-owned kernel is fastest.

### Register-pressure fallback

The fast derivative writer shares subexpressions across all eight partials for
one missing site. A complex primal can keep that shared graph live while each
completed partial permanently pins another output register. This is especially
restrictive on the sm_89 LM template: its partial site exposes 12 usable
scratch/output registers, compared with 16 on the current sm_120 template.

Specialization now retries only register-pressure failures with a bounded
fallback. The fallback compiles partials independently in descending symbolic
size, so the hardest derivative runs before completed outputs shrink the
scratch pool. Normal genomes retain the smaller and faster cross-output CSE
schedule. Promotion telemetry records the AST sites that required the fallback
and records the complete program and node counts if the fallback also fails.

The regression genome came from completed Ada seed 2003. It has a 21-node,
depth-5 first AST and a 9-node, depth-4 second AST: exactly the campaign's
30-node complexity limit, not an out-of-policy tree. The first AST's symbolic
partials contain 33, 62, 35, 71, 2, 59, 89, and 74 postorder instructions. The
old writer exhausted registers while producing the third output. The fallback
uses 234 SASS instructions across both sites, below the 1,024-instruction
partial patch capacity, while retaining the template's 252-register count.

The exact fallback CUBIN was validated on the RTX 4090. From a deliberately
perturbed `0.00341275` CPU trajectory MSE, four accepted LM steps reached
`6.81315e-8`; independent CPU replay gave `6.81404e-8`. With the production
8,192 settings and four starts, the same complex genome sustained 33,370 fits/s
and found `2.87e-14`. A 512-genome specialization stress set, including random
systems up to 99 total AST nodes, had no failures on either the sm_89 or sm_120
template.

### Campaign diagnostic contract

Every recovery report contains per-seed `lm_diagnostics` and a campaign-wide
`trajectory_lm_diagnostics` aggregate. The aggregate keeps selection,
successful specialization, GPU evaluation, and GP acceptance as separate
counts. It reports rejection rates and categorized reasons, bounded examples
with candidate complexity and per-site node counts, pressure-fallback use,
register and emitted-instruction distributions, non-finite fit scores,
iteration-cap hits, accepted LM steps, GP population snapshots (including
invalid-score rates and complexity), and batch-level timing. Batch timing is
stored once at the selection boundary rather than copied and summed from each
candidate event.

Machine-readable health flags classify complete specialization loss as
critical and partial rejection, non-finite fits, or zero accepted promotions as
warnings. Pressure fallback use is informational because it represents a
recovered candidate. The campaign notifier includes accepted/evaluated and
rejected/selected counts so an optimizer stage that silently did no work cannot
be mistaken for a successful run. Search reports also break wall time into GP
evaluation, data movement, evolution, trajectory LM, and unattributed control
time to expose an unexpectedly expensive stage. The same aggregate is written
atomically to `lm_diagnostics.live.json` after every promotion boundary and to
`seed_<seed>/lm_diagnostics.json`, so monitoring does not have to wait for a
seed or campaign to finish.

This result keeps thread ownership as the preferred topology for the current
four-state, six-active-constant regime. A shared-memory reverse-tape backend is
still the appropriate next fallback if one individual partial cannot fit even
when scheduled first, or if larger state/parameter dimensions make the
thread-local LM state itself exceed the register budget.

The same split CUBIN builds and recovers the planted constants on sm_120 with
no stack, local memory, or spills. With 65,536 fits to cover the larger RTX
5090, the split path reaches 51,813 fits/s versus 92,312 fits/s for the native
control. This is a compatibility result rather than evidence for changing
ownership: the thread-owned kernel remains correct on both architectures.

All generated includes, CUBINs, and inspection JSON are written beneath the
explicit generated/trajectory_lm build directory, which is ignored by Git.
The command-line generator requires an output path and otherwise emits
inspection JSON to stdout, so it does not silently create scratch files.

## GP promotion status

The split ABI, randomized GPU differential gate, candidate selection, and
strictly improving GP writeback are now implemented. The recovery command keeps
this path disabled unless `--lm-promotion-interval` is set. Each promotion
specializes one complete smooth system genome, tries dynamic binding settings
and constant starts, runs resident LM, and returns the winning six active
constants plus its leaf bindings to the same C99 population member.

A 128-genome Ada smoke with 512 SSE settings and two 32,768-fit promotions
selected non-incumbent settings/starts and improved the final MSE from
`0.0868577` without LM to `0.0864534` with LM. Its materialized GPU replay
agreed with the promoted score within `7.5e-9`. This is an integration result,
not a search-quality comparison; the six-generation baseline is too short for
that claim.

Larger ASTs that do not fit either direct schedule should use the bounded
two-plane shared scratch layout described in `docs/postorder_autodiff.md`.

After that:

1. form concurrent promotion batches from selected GP frontier members;
2. compare one, two, four, and eight starts per promoted genome;
3. port the stable split specialization ABI to the C99 eager-loader pipeline;
4. add deterministic in-kernel start generation only if host preparation is
   measurable; and
5. use subgroup ownership only when the promotion queue cannot provide enough
   independent fits or larger state/constant counts cause spilling.
