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

# Maturity-Guided Search

## Status

`cubin-maturity` is the experimental development path for Secant-SR. The
older `cubin-staged` policy remains available as a frozen comparison, but new
search-policy work should target maturity-guided search.

This change does not discard the earlier SRBench evidence. Those campaigns
remain useful fixed baselines:

| Policy budget | SRBench numerical recovery | Successful trials | Recorded fit time |
|---|---:|---:|---:|
| 8,192 population x 100 generations | 59.91% | 710 / 1,160 | 2,399.383 s |
| 131,072 population x 200 generations | 71.98% | 857 / 1,160 | 39,455.620 s |

Both rows used the frozen SRBench v2.0 Feynman protocol documented in
`docs/srbench_v2_protocol.md`. They measure the legacy policy and must not be
silently relabeled as maturity-search results.

## Search Model

Every population member is a concrete, directly executable AST. Dynamic
instructions exist only in temporary projected programs sent to a dynamic
Secant kernel. A winning setting is immediately materialized back into a
concrete static-column and fixed-constant AST before it enters the population.

Candidates move through four search-local states:

1. **Exploratory**: a newly random expression whose terminal choices have not
   been resolved by a full dynamic search.
2. **Resolved**: a concrete expression produced by full terminal resolution or
   structural variation.
3. **Mixed refined**: a resolved expression improved by reopening selected
   leaves as dynamic holes while preserving the rest of the expression.
4. **Constant refined**: a concrete expression improved by the selected
   continuous optimizer: the packed shared-jitter baseline or multi-start LM.

Random candidates begin as exploratory. Crossover, subtree mutation, and point
mutation create new resolved structures with fresh refinement counters because
the child is not the same expression as either parent. Elites preserve their
maturity and counters exactly.

## Per-Generation Flow

1. Run static-column SSE over the complete concrete population. This is the
   baseline score and ensures a failed proposal cannot worsen an individual.
2. Fully project every eligible exploratory expression whose terminal count
   fits the configured dynamic-leaf capacity. All static columns and fixed f32
   constants become dynamic constant-or-column holes.
3. Evaluate all settings, reduce the best setting on the GPU, materialize the
   selected columns/constants into a concrete AST, statically rescore it, and
   promote it only when SSE improves.
4. Deterministically sample a configurable fraction of the remaining resolved
   population for mixed refinement. Reopen at most the configured number of
   leaves while preserving at least one concrete leaf. The projection seed and
   AST fingerprint select the holes without consuming the structural GP RNG.
5. Select a constant-bearing cohort by either independent probability or the
   bounded score/complexity/random policy. `legacy` runs the packed
   shared-jitter optimizer. `lm` rewrites up to eight fixed constants as dense
   dynamic constant-or-column inputs. Its setting axis is a Cartesian product
   of column bindings and deterministic constant-vector starts evaluated by the
   thread-owned native-SASS LM statistics and solve kernels. The winning
   binding and constants are written back into a concrete AST only after a
   strict static rescore.
6. Build the next generation through the existing quality-diversity archive,
   crossover, mutation, elite retention, and random injection.

The default experimental policy uses eight fully dynamic leaves, four mixed
holes, a 25% mixed-refinement probability, and the existing 14% constant-
optimization probability. These are policy parameters, not established
optima.

## Kernel Use

The search prepares separate dynamic-only and mixed Secant skeletons:

- The dynamic-only recipe has zero static input registers and accepts only
  dynamic constant-or-column leaves.
- The mixed recipe loads all configured static input columns into registers and
  also accepts dynamic holes.
- Both use the same `[setting][dynamic leaf]` masks and words table. The table
  stride is the compiled dynamic-leaf capacity even when a mixed projection
  activates fewer holes.

Static rescoring and both continuous optimizers use their existing Secant
kernel shapes. The LM path preserves the incumbent as start zero within every
binding, holds one AST per specialized function, keeps every state's
fixed-eight data thread-owned, and leaves the module loaded for the initial
statistics pass plus every LM proposal/solve iteration. Column-bound holes have
zero derivative; repeated constant indices still share a parameter. Permanent
tests cover mixed projection and materialization, binding/start construction,
known-binding LM convergence, static rescoring, and promotion. The frozen
`cubin-staged` backend retains the older one-AST-per-kernel constant optimizer
for comparison.

## LM Campaign Mode

`--constant-optimizer lm` is deliberately opt-in so historical `legacy`
results remain reproducible. `--constant-optimize-probability` supplies the
stochastic elevation policy; omitting it retains bounded diverse selection.
In LM mode, `--constant-settings` is the total state count rather than random
perturbations per round. `--lm-starts-per-binding` determines the inner start
count and must divide the total; the quotient is the binding count. With the
recommended 128 total states and four starts, lanes map as
`binding = lane / 4`, `start = lane % 4`. `--constant-optimizer-iterations`
means LM proposal evaluations after the initial statistics pass.

The first matched engineering panel uses the same maturity search, population,
operators, 4,096 settings-SSE table, 14% promotion probability, seeds, and
stopping rule as the packed Nguyen-5/Distance-2 panel below. Only the continuous
optimizer changes: the legacy baseline uses 8,192 settings x 4 rounds. The
first LM panel used one binding x 128 starts x 8 proposal evaluations; it is the
historical multistart baseline. The mixed LM regime reuses the same 128 lanes as
32 bindings x 4 starts x 8 proposal evaluations. This comparison measures the
new optimizer as a replacement policy, including its reduced evaluation budget,
rather than claiming equal row-evaluation work.

The mixed integration exposed a cache-key defect: dynamic-leaf template keys
did not include total or static input-column capacity, so dynamic-only and
mixed CUBINs could collide. The key now distinguishes both values. In a direct
two-process check, the first process had two expected misses and no
invalidations; the second had five hits, no misses, no invalidations, and
reduced template preparation from 14.84 seconds to 0.042 seconds.

### Binding/Start Ablation

Before adding column bindings, a preliminary fixed-binding LM panel over three
Nguyen-5 and three Distance-2 trials solved 6 / 6 in 182.060 seconds. A broader
seeds-4-through-23 replacement-policy panel then compared the 128-start LM path
with the legacy packed optimizer. Legacy solved 36 / 40 in 603.677 seconds;
fixed-binding LM solved 37 / 40 in 981.022 seconds. The single additional solve
did not justify a 62.5% wall-time increase, which motivated reallocating the
existing LM lanes from redundant constant starts to structural bindings.

A matched RTX 5090 engineering check compared the historical allocation of
128 starts for one fixed binding with 32 bindings and four starts per binding.
Both configurations used the same 128 thread-owned states, eight LM proposal
evaluations, 32,768-individual population, 200-generation cap, 1,024 training
rows, and Nguyen-5/Distance-2 seeds 24 through 26.

| LM state allocation | Solved | Generations | Search time | LM time | LM time/generation |
|---|---:|---:|---:|---:|---:|
| 1 binding x 128 starts | 6 / 6 | 132 | 145.190 s | 124.221 s | 0.941 s |
| 32 bindings x 4 starts | 5 / 6 | 45 | 49.966 s | 42.233 s | 0.938 s |

Reallocating the same state budget therefore did not measurably increase the
per-generation LM cost and reduced aggregate search time by 65.6% in this small
panel. The mixed run's nominal miss was Distance-2 seed 26 at validation R2
0.999990761 after the train score triggered early stopping. A targeted repeat
using validation stopping solved it in seven generations and 7.944 seconds.
That instrumented repeat selected 11,604 column-bound winners and promoted
11,323 of them, confirming that the binding lanes were active rather than
redundant. Floating atomic reduction makes this an exploratory engineering
result, not yet a statistically established recovery-rate comparison.

A follow-up held the total state budget at 128 and varied only its allocation
across bindings and starts. It used validation-based stopping and eight seeds
each for Nguyen-5 and Distance-2. Generation counts are summed across all 16
trials.

| LM state allocation | Solved | Generations | Search time | LM time | LM time/generation |
|---|---:|---:|---:|---:|---:|
| 32 bindings x 4 starts | 15 / 16 | 416 | 455.063 s | 388.903 s | 0.935 s |
| 64 bindings x 2 starts | 16 / 16 | 428 | 473.954 s | 405.836 s | 0.948 s |
| 128 bindings x 1 start | 15 / 16 | 481 | 526.367 s | 450.802 s | 0.937 s |

Per-generation LM cost remained within 1.4%, as expected for the fixed state
budget. The 64 x 2 allocation recovered every trial for 4.2% more aggregate
search time than 32 x 4. Relying entirely on the inherited incumbent with
128 x 1 did not improve recovery and took 15.7% longer than 32 x 4. This makes
64 x 2 the candidate from this small sweep: evolutionary inheritance removes
much of the need for repeated starts, but retaining a second start appears to
be useful insurance for nonlinear basins. Atomic-reduction nondeterminism and
the small problem panel preclude treating 16 / 16 as a seeded guarantee or
changing the default from this result alone.

### Exhaustive Split-Host Panel

The follow-up split seeds 4 through 23 between the RTX 5090 and RTX 4090 while
keeping both allocations for every problem/seed pair on the same host. Rohini
ran 13 seeds and Ada ran seven to compensate for their measured throughput
difference. The 64 x 2 campaign covered all six generated core problems; the
32 x 4 control covered the two hard problems. Both used validation stopping,
the same 128-state budget, and a 200-generation cap.

The exhaustive 64 x 2 core campaign solved all 120 trials:

| Problem | Solved | Aggregate search time | Minimum validation R2 |
|---|---:|---:|---:|
| Nguyen-1 | 20 / 20 | 41.924 s | 1.000000000 |
| Nguyen-5 | 20 / 20 | 691.849 s | 0.999999993 |
| Rational-2 | 20 / 20 | 43.123 s | 1.000000000 |
| Distance-2 | 20 / 20 | 94.953 s | 0.999999024 |
| Interaction-3 | 20 / 20 | 71.650 s | 0.999999964 |
| Oscillator-2 | 20 / 20 | 32.628 s | 1.000000000 |
| **Total** | **120 / 120** | **976.128 s** | **0.999999024** |

The matched hard-problem control reverses the small-sweep allocation ranking:

| LM state allocation | Solved | Generations | Search time | LM time |
|---|---:|---:|---:|---:|
| 32 bindings x 4 starts | 40 / 40 | 422 | 583.533 s | 491.099 s |
| 64 bindings x 2 starts | 40 / 40 | 585 | 786.801 s | 666.363 s |

Both allocations recovered every hard trial, but 32 x 4 used 25.8% less search
time and 27.9% fewer generations. The larger result therefore does not support
changing the four-start default. Inherited constants are useful, but the extra
bindings did not compensate for removing two starts in this panel.

Normalized LM throughput also confirms the expected host asymmetry. Rohini
processed 5.28e9 effective LM row-evaluations/s for 64 x 2 versus Ada's 3.07e9;
the 32 x 4 measurements were 5.53e9 and 3.39e9 respectively. Ada consequently
required about 1.6x to 1.7x as much time for equivalent LM work. The partition
assigned 104 trials to Rohini and 56 to Ada and completed from 5:43:01 PM to
5:56:29 PM EDT.

Detailed records are stored in:

- `docs/data/lm_optimizer_panel_sm120_20260814.csv`
- `docs/data/legacy_optimizer_panel_sm120_seeds4_23_20260814.csv`
- `docs/data/lm_optimizer_panel_sm120_seeds4_23_20260814.csv`
- `docs/data/lm_one_binding_128_starts_seeds24_26_sm120_20260814.csv`
- `docs/data/lm_32_bindings_4_starts_seeds24_26_sm120_20260814.csv`
- `docs/data/lm_32_bindings_4_starts_distance2_seed26_validation_stop_sm120_20260814.csv`
- `docs/data/lm_allocation_32b_4s_seeds24_31_validation_sm120_20260814.csv`
- `docs/data/lm_allocation_64b_2s_seeds24_31_validation_sm120_20260814.csv`
- `docs/data/lm_allocation_128b_1s_seeds24_31_validation_sm120_20260814.csv`
- `docs/data/lm_exhaustive_64b_2s_rohini_sm120_seeds4_16_20260814.csv`
- `docs/data/lm_exhaustive_64b_2s_ada_sm89_seeds17_23_20260814.csv`
- `docs/data/lm_exhaustive_32b_4s_control_rohini_sm120_seeds4_16_20260814.csv`
- `docs/data/lm_exhaustive_32b_4s_control_ada_sm89_seeds17_23_20260814.csv`

## Initial Panel

This is a one-seed engineering panel over the six generated problems, not an
SRBench result. It used 8,192 candidates, 100 generations, 1,024 training rows,
4,096 leaf settings, 8,192 constant settings, warmed skeleton caches, and no
final CPU constant optimization.

| Problem | Maturity validation R2 | Maturity time | Legacy validation R2 | Legacy time |
|---|---:|---:|---:|---:|
| Nguyen-1 | 1.000000 | 0.366 s | 1.000000 | 0.339 s |
| Nguyen-5 | 0.996442 | 17.817 s | 0.996107 | 13.021 s |
| Rational-2 | 1.000000 | 0.358 s | 1.000000 | 0.542 s |
| Distance-2 | 1.000000 | 1.357 s | 1.000000 | 1.352 s |
| Interaction-3 | 1.000000 | 1.361 s | 1.000000 | 0.789 s |
| Oscillator-2 | 1.000000 | 0.356 s | 1.000000 | 0.329 s |
| **Total / solved** | **5 / 6** | **21.615 s** | **5 / 6** | **16.372 s** |

The result establishes correctness and feasibility, not superiority. The new
policy spent more work reopening mature candidates and was 32% slower on this
panel. Its slightly better Nguyen-5 score is one seed and is not statistically
meaningful.

## Generation Telemetry

`python/run_suite.py --generation-output <path>` writes one record per completed
generation. Requesting this output enables the additional C-side attribution
work; normal runs do not scan proposals for these statistics. Each record
contains:

- Population counts before and after the three refinement stages.
- Eligible, selected, promoted, finite-gain, and invalid-to-finite counts.
- LM winners and promotions whose materialized AST uses at least one column
  binding.
- Promotion rates and normalized finite-baseline R2 gains.
- Static, full-leaf, mixed-leaf, constant, and complete evaluation times.
- Best archived R2 after static scoring and after each refinement stage.
- Final winner origin, maturity, complexity, depth, and validation R2.

This is intended to answer whether a stage changes the campaign best, rather
than treating a high count of arbitrarily small per-individual improvements as
evidence that the stage is useful.

## Historical AST-Local Panel

The 2026-08-08 engineering panel used 32,768 individuals, up to 200 generations,
three seeds, 1,024 training rows, 4,096 leaf settings, and 8,192 constant settings
over four optimizer iterations. Mixed refinement selected 25% of eligible
candidates and constant optimization selected 14%. This is the generated six-
problem core panel, not an SRBench result.

All 18 trials crossed the strict `R2 > 0.999999` stop threshold. No final
CPU/GPU materialization validation diverged. The run completed 276 generations
in 473.407 seconds; median trial time was 2.643 seconds and the longest trial
was 213.604 seconds.

| Stage | Selected ASTs | Promoted ASTs | Promotion rate | Time | Effective row-evals/s |
|---|---:|---:|---:|---:|---:|
| Full dynamic leaf | 887,516 | 647,892 | 73.0% | 13.544 s | 2.75e11 |
| Mixed dynamic leaf | 2,040,027 | 1,343,828 | 65.9% | 34.212 s | 2.50e11 |
| AST-local constant | 1,188,286 | 986,167 | 83.0% | 404.961 s | 9.85e10 |

Static scoring consumed 14.974 seconds. Constant optimization consumed 86.6%
of the complete 467.738-second evaluation time. The best archived R2 improved
after full dynamic-leaf evaluation in 17 generations, after mixed refinement in
10, and after constant optimization in 57. At termination, 14 of 18 winners
originated from a dynamic-leaf proposal, two from constant optimization, and
two directly from random generation.

The settings counts are therefore large enough to exercise both search modes:
each selected leaf proposal sees 4,096 settings, and each selected constant
proposal sees 32,768 perturbation trials. They are not an efficient final
policy. The constant stage is the first budget to sweep downward because it
dominates wall time despite changing the campaign best in only 57 of 276
generations.

The detailed records are:

- `docs/data/maturity_core_sm120_pop32768_gen200_seeds123_20260808.csv`
- `docs/data/maturity_core_sm120_pop32768_gen200_seeds123_20260808_generations.csv`

## Packed Optimizer Integration

`cubin-maturity` now uses Secant's packed shared-jitter constant optimizer.
The older `cubin-staged` path remains unchanged. A permanent CPU/GPU oracle
test covers two packed iterations, five ASTs, two ASTs per kernel, two modules,
and a partial final module. It compares the complete SSE surface and all final
centers, scales, velocities, and incumbent SSE values. The existing Secant test
also covers the packed reducer policies directly.

Historical execution detail, verified against the pre-toggle implementation on
2026-09-19: the packed kernel generates shared normalized Philox jitter before
the row loop. The native specializer embeds each AST's coefficient center and
scale as immediate multiply/add instructions at dynamic-constant references
inside that loop. Thus packed ASTs share jitter, not their final coefficients.
The packed runner reads back updated state and re-specializes/reloads each
module for the next refinement round; this is native patching, not fresh NVRTC.
Do not attribute the older one-AST optimizer's resident multi-round lifecycle
to this packed path. The current toggle GP has not ported either optimizer;
its scoring kernel reads already-valued shared banks. A proposed resident
refinement path would therefore have different coefficient-update mechanics,
whose register cost and throughput must be measured separately.

The packed shape is sensitive to launch concurrency at SRBench-scale row
counts. With 4,096 MUFU ASTs, 8,192 settings, 1,024 rows, four optimizer rounds,
64 kernels per module, and 32 ASTs per kernel, measured pipeline throughput was:

| Streams | Pipeline row-evals/s |
|---:|---:|
| 8 | 6.38e10 |
| 16 | 1.27e11 |
| 32 | 2.40e11 |
| 64 | 3.99e11 |

The module is approximately 3.45 MB. At eight streams, module loading and
underfilled concurrent launches dominate the packed pipeline. Search workloads
should currently use 64 streams with the 64-kernel template on the RTX 5090.
This is a measured tuning point for SM120, not a portable architectural rule.

A matched Nguyen-5 and Distance-2 panel used 32,768 individuals, up to 200
generations, three seeds, 1,024 training rows, 4,096 leaf settings, 8,192
constant settings, four optimizer rounds, and 64 streams. It solved 5 / 6
trials in 116.824 seconds. The one miss, Distance-2 seed 1, ended at validation
R2 0.999780784 after 200 generations.

| Stage | Selected ASTs | Promoted ASTs | Promotion rate | Time | Effective row-evals/s |
|---|---:|---:|---:|---:|---:|
| Full dynamic leaf | 553,977 | 409,884 | 74.0% | 5.660 s | 4.01e11 |
| Mixed dynamic leaf | 2,107,472 | 1,510,268 | 71.7% | 16.425 s | 5.25e11 |
| Packed constant | 1,191,089 | 1,043,370 | 87.6% | 74.019 s | 5.40e11 |

The prior successful AST-local panel solved 6 / 6 in 441.691 seconds and spent
381.402 seconds in constant optimization. Its exact reproduction solved 4 / 6
in 746.165 seconds. The packed result is therefore 3.78x faster than the
successful historical panel and 6.39x faster than the reproduction, but its
5 / 6 recovery must be interpreted within the atomic-reduction stochasticity
described below rather than as a statistically established quality change.

Detailed packed records are stored in:

- `docs/data/packed_optimizer_panel_sm120_20260808.csv`
- `docs/data/packed_optimizer_panel_sm120_20260808_generations.csv`

### Constant Budget Sweep

A follow-up held the population, leaf settings, policy probabilities, and first
100 generations fixed for Nguyen-5 and Distance-2 across three seeds. Only the
number of constant settings and optimizer iterations changed. Evaluation time
is the sum of generation telemetry and excludes final CPU optimization.

| Settings x iterations | Perturbations per selected AST | Solved | Generations run | Evaluation time | Constant-stage time | Minimum terminal R2 |
|---|---:|---:|---:|---:|---:|---:|
| 8,192 x 4 | 32,768 | 5 / 6 | 229 | 392.076 s | 341.616 s | 0.998958 |
| 4,096 x 4 | 16,384 | 4 / 6 | 224 | 223.419 s | 173.561 s | 0.999193 |
| 2,048 x 4 | 8,192 | 3 / 6 | 332 | 210.510 s | 137.502 s | 0.996770 |
| 4,096 x 8 | 32,768 | 4 / 6 | 216 | 383.585 s | 334.866 s | 0.999744 |

`4,096 x 4` is the current engineering frontier: it reduced evaluation time by
43% relative to `8,192 x 4` while losing one solve in this small panel.
`2,048 x 4` saved little additional total time because failed trials consumed
the full generation budget. Redistributing the baseline sample count over eight
shrinking-radius iterations did not recover baseline reliability and retained
nearly all of its cost. This suggests that broad per-iteration coverage matters
more than adding late iterations at very small radii under the current 0.5
decay schedule.

These are single stochastic panels, not a statistically sufficient default-
selection result. The floating atomic qualification below means the next policy
decision should compare `4,096 x 4` and `8,192 x 4` over more independent
repetitions. The detailed sweep files are stored as
`docs/data/constant_budget_sweep_*_sm120_20260808*.csv`.

## Reproducibility Qualification

Two controlled Nguyen-5 repeats used the same executable, seed, and search
arguments. They solved in 9 generations / 15.300 seconds and 8 generations /
13.588 seconds. Static and full-leaf best scores matched exactly in generation
zero, while constant-promotion counts differed slightly (3,204 versus 3,237).

A focused repeat probe localized the divergence to constant optimization.
Across ten identical generation-zero runs, static, full-dynamic, and mixed-
dynamic scores and promotion counts matched exactly. Constant promotions ranged
from 3,178 to 3,266, and the post-constant best R2 took three different values:
0.95341806, 0.98037495, and 0.999961398.

The constant-optimizer kernel uses one CTA per row tile and atomically adds each
tile's SSE into the setting score. With 1,024 rows and 128-row tiles, a direct
16-repeat evaluator probe changed thousands of score bit patterns per repeat,
with maximum absolute differences of 0.0078125 to 0.015625. The same probe with
one 128-row tile was bit-for-bit stable in all repeats. Host constant packing,
AST promotion order, module offsets, output clearing, stream joins, and reducer
indexing were independently checked and remained consistent. This establishes
unordered floating-point atomic accumulation as the source of score
nondeterminism; near-tied winners amplify it into different constant proposals,
archive updates, and later populations.

Results must therefore be reported over repeated trials even when the nominal
seed is fixed. The controlled traces are stored as
`docs/data/maturity_nguyen5_repeat_{a,b}_sm120_20260808_generations.csv`.

An exact reproduction attempt then repeated the six Nguyen-5 and Distance-2
trials with the original `8,192 x 4` constant budget and a 200-generation cap.
It recovered only 4 / 6 rather than the previously observed 6 / 6. Nguyen-5
seed 2 stopped at R2 0.999277897 and Distance-2 seed 3 at R2 0.999981981 after
200 generations. No CPU/GPU materialization validation diverged. The result is
stored in `docs/data/baseline_reproduction_s8192_i4_gen200_sm120_20260808*.csv`.

Therefore 6 / 6 is an observed successful campaign, not a reproducible seeded
guarantee. Constant-budget rankings from one six-trial panel are exploratory
until the reduction path is deterministic or each configuration has enough
independent repetitions to estimate its recovery distribution.

## Next Policy Work

Before further policy tuning, constant optimization needs an explicit choice:
retain the faster atomic reduction and treat nominal seeds as stochastic
replicates, or add a deterministic reduction/evaluation path and measure its
cost. The integrated packed optimizer also uses floating-point atomics, so the
same qualification applies to its setting selection.

After that, the next policy step is subtree-aware reopening: select a mutation
site, preserve the mature context, and make only the new subtree terminals
dynamic. Mixed and constant cohort budgets should then be adapted using
promotion rate, archive novelty, stagnation, and prior refinement count instead
of one global probability.

A full 116-problem run should wait until those cohort rules are stable. The
legacy 1,160-trial files provide enough evidence to compare the new policy
without rerunning the old search.
