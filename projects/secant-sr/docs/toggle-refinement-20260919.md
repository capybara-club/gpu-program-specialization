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

# Native-toggle coefficient refinement — 2026-09-19

Secant-SR 0.3.1 restores packed random coefficient fitting through Secant's native
toggle scorer. There are no settings tables, column-index settings, optimizer
kernel ABI shims, or GP policy inside Secant. Legacy settings code remains archived.
This is an adaptation of the old packed random search, not a port of LM or a
claim of identical random proposals/search results.

## Usage

```sh
CUDA_MODULE_LOADING=EAGER ./build-toggle/secant_sr_search \
  --backend cuda --problem nguyen5 --banks 64 --constants 4 --toggle-bits 6 \
  --refine-rounds 4 --refine-budget 128 --refine-scale 1
```

Python `fit()` accepts `refine_rounds`, `refine_budget`, and `refine_scale`.
The default is **zero refinement rounds**. Existing search configurations and
frozen SRBench baselines therefore retain their original policy. No long campaign
has been restarted with new defaults.

## Responsibilities and API

- **Secant:** new direct leaf `affine_bank(slot, scale, offset)`, opcode `0xbf`,
  10-byte encoding: opcode, slot, little-endian f32 scale, little-endian f32 offset.
  Use `secant_ast_affine_bank_write`. It can be a toggle operand. Arithmetic is
  separately rounded multiply/add, not FMA. Scale and offset must be finite.
- **Secant-SR:** parameter identity, proposal generation, adaptive scales, winner
  selection, population integration and reports. The refiner owns fixed-capacity
  storage allocated at creation and allocates nothing during seed/ask/tell.
- **CUDA adapter:** accepts replacement banks through `secant_sr_cuda_score_banks`,
  uses existing device buffers/events/streams, and restores its copied base bank
  before the next ordinary `secant_sr_cuda_score` call. No per-round CUDA allocation.

[`secant_sr_refine.h`](../toggle/include/secant_sr_refine.h) provides this lifecycle:

```text
create(config, options)
seed(models, source_banks, sequence)
repeat:
    ask() -> packed AST pointers + shared Philox bank
    score ASTs × bank rows × all toggle permutations
    tell(best score/configuration per AST)
model(index) -> fitted genotype + retained permutation
```

`sequence` identifies the batch/generation; seed and round identify the normalized
bank. One model entry is one start; callers can supply the same structure with
multiple starting configurations. Each entry can be an unrelated AST. All entries
share raw bank registers and permutation bits, with local selector operands and
per-AST affine transforms. There is no similarity packing or retained cross-AST
selected-value cache.

The CLI ranks finite candidates, selects up to its fitting budget, fits them,
imports improved genomes, then breeds. It chooses **one start per selected AST**.
A failed `seed` invalidates that batch. `tell` validates the complete report before
changing any model; rejected reports may be corrected and resubmitted. Input data
must remain unchanged across rounds. Copy borrowed models before reseeding.

## Coefficients, toggles, and fixed literals

Philox4x32-10 produces a shared uniform `[-1, 1)` proposal bank. Row zero is all
zeros to retain the current center. A trial coefficient is
`center[AST, parameter] + scale[AST, parameter] * bank[trial, parameter]`.
The scorer evaluates the full toggle Cartesian product for every trial; native
two-way and four-way selectors remain in the genotype throughout fitting.

Only parameters selected by the retained permutation update. Accepted moves
adjust scale toward twice the absolute step (learning rate 0.25); failures halve
the active scales. Default scale bounds are 1e-6 to 1e6. These are an initial
search policy, not validated optimal hyperparameters or a curvature estimate.

Fixed literals are never fitted. Raw references sharing a slot are one parameter.
Fitted references sharing both slot and exact value bits remain one parameter;
different fitted values brought together by crossover remain independent. Import
rejects changes that split or merge existing parameter identities or alter fixed
leaves, operators, or selectors. Fitted identities survive breeding. During normal
scoring, they lower to affine leaves with zero scale and their stored value.

## Performance and correctness limits

1. **RNG placement differs from the old packed kernel.** This first implementation
   generates Philox on the C host and uploads the shared normalized bank. The old
   packed kernel generated its normalized draws on device. Scale/offset are still
   embedded by native specialization. This is a measured, explicit port difference;
   no old/new throughput equivalence is claimed. Moving Philox to a reusable GPU
   producer is future tuning, not required for the current interface.
2. **Rounds reload modules.** Each round repatches and loads its packed ASTs through
   the ordinary scorer, as the old packed refiner did. It does not rerun NVRTC per
   round. Affine multiply/add instructions occur at each leaf use inside the row
   loop; they are not hoisted into a permanent per-AST coefficient register bank.
   Resident adaptive-round kernels are a separate possible optimization.
3. **Parameter capacity is explicit.** A crossover result can have more independent
   parameters than bank slots. Ordinary scoring still works, but CLI fitting skips
   it and counts `skipped_parameter_capacity`. It never silently ties coefficients.
   Increase configured slots if appropriate; adaptive capacity/subset-fitting policy
   is not implemented. Skips count selection visits, not distinct models.
4. **One start can get trapped.** In a local test for `2.75*x - 0.125`, a center could
   settle on a wrong toggle branch and shrink too much to reach the correct branch.
   The API supports multiple starts; the CLI does not maintain separate coefficient
   centers per permutation. The convergence unit test deliberately supplies a good
   local start. It tests fitting mechanics, not blind structural recovery.
5. **No universal fitting win.** Extra rounds consume time and alter GP selection.
   Trials/budget/rounds and starts require broader quality tests. LM is deferred.
6. **Cold compilation still matters.** The three-second initial comparison included
   3.46–3.51 s cold setup, leaving only the required first generation in affected
   arms. Those are valid end-to-end outcomes but not comparable steady-state fitting
   measurements. Repeating the entire sequence did not prime each shape reliably;
   the reported comparison explicitly primes each shape immediately before its
   paired runs. Persistent CUBIN caching remains item 2.
7. Finite SSE does not always imply finite intermediate coefficients (e.g. min/max
   can mask overflow). Fitting rejects such a winning report transactionally instead
   of storing a nonfinite coefficient. Float32, fast math and parallel reductions
   can perturb ties and subsequent search evolution across repeat runs/hardware.

## Replay and timing

`coefficients` is the raw base-bank vector, now explicitly labeled by
`coefficient_semantics: "raw_bank_vector"`. `fitted_leaves` reports every fitted
alternative by logical node, alternative, slot and value. It may include inactive
alternatives. `genotype_hex` embeds fitted values; `resolved_ast_hex` is the fixed
winning expression. Retained refined scores reference bank zero and the retained
permutation, not a discarded random-trial index. Python can reconstruct the exact
resolved bytecode from genotype, raw bank and permutation.

Broad work remains in top-level `configurations`; fitting work is separately in
`refinement.configurations`. Sum them for total evaluations. Counts include
repeats, unused toggle dimensions, and invalid evaluations; they are not unique
ASTs or unique models. `refinement.seconds` includes selection, bank generation,
lowering, scoring, imports and round handling. `timing.scoring_seconds` covers
broad scoring only. Pipeline/device/load/reduction/transfer totals include both
paths and overlap; do not add them together. Setup includes NVRTC. Full process
wall time includes startup/exit; `result_ready_seconds` excludes scorer teardown.

## Validation

- mac1: all **6 Secant tests + 7 Secant-SR tests**, normal and ASan/UBSan builds.
- CPU/GPU fitter oracle: three entries, 64 bank rows, eight permutations, 31 rows,
  24 rounds; two-/four-way selectors, shared parameters, fixed literals, partial
  packs/modules, failed-report transactionality and base-bank restoration.
  Local convergence SSE: about 6.23e-9 on both RTX 5080 and RTX 4090.
- Python: affine encoding/decoding, truncated bytecode rejection, native-toggle
  resolution, CLI fitting and byte-exact replay of the reported winner.
- GPU core: 129 random ASTs including affine leaves, 1,280 configurations and
  97 rows, packs 1/8/32, plus static-path regression. Pack 32 still reports the
  existing scaffold's 16 local bytes/thread; no claim of spill-free large packs.
- RTX 5080 CUDA adapter regression and Compute Sanitizer memcheck: passed,
  zero reported errors. Use `CUDA_MODULE_LOADING=EAGER` for these tests. The initial
  sanitizer invocation without it failed at scorer creation and was not a pass.
- RTX 4090 full search with fitting: final CPU/GPU audit accepted; held-out NMSE
  about 4.93e-13. Its first cold three-second run did not reach fitting; the warm
  rerun did. No fallback to CPU scoring.

The compact comparison data below records all successful warm paired runs.
Numerical train/holdout success is **not** proof of symbolic equivalence, and these
simple datasets are not a sufficient search-quality benchmark.

## Short warm comparison

RTX 5080, 1,024 ASTs/generation, 64 bank rows × 64 permutations, 512 training
and 512 held-out rows. Three-second cap includes setup; max 10,000 generations.
Four fitting rounds select up to 128 ASTs. All 12 runs passed train/holdout NMSE
≤1e-10 and the independent CPU/GPU audit. Times are full process wall time.

| Problem | Seed | Fitting off | Four rounds | Fitting phase |
| --- | ---: | ---: | ---: | ---: |
| nguyen5 | 41 | 0.233 s | 0.747 s | 0.191 s |
| nguyen5 | 42 | 0.393 s | 0.472 s | 0.103 s |
| nguyen5 | 43 | 1.269 s | 0.933 s | 0.250 s |
| distance2 | 41 | 0.353 s | 0.244 s | 0.026 s |
| distance2 | 42 | 0.227 s | 0.242 s | 0.025 s |
| distance2 | 43 | 0.251 s | 0.305 s | 0.045 s |

This tests integration on easy built-in problems. Fitting changes the search path;
it is not an isolated kernel-speed comparison and does not establish a general
quality benefit. Both modes solve these examples. Repeated warm runs changed
generation counts, consistent with floating-point tie sensitivity. The initial
unprimed runs are retained remotely and excluded from this table. Compact
[data and source hashes](data/toggle-refinement-20260919.json) include counters,
coefficient-capacity skips, and stage timings. Full logs remain in
`rack1:/home/cdurham/experiments/secant-toggle-refinement-20260919/`.

## Remaining original work

2. Broader search/SRBench validation and analysis; persistent template caching.
3. Arbitrary-AST register, instruction and packing throughput tuning.
4. Toggle-aware random fitting is implemented. Tune rounds, budgets, starts,
   parameter capacity and RNG placement. LM remains deferred.
