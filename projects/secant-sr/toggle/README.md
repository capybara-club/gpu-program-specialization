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

# Toggle-native symbolic regression

## Separation from the evaluator

Secant owns bytecode, code generation, inspection, native specialization, module
loading, and SSE execution. Secant-SR owns populations, random proposals,
selection, fitting policy, and the dataset lifecycle. The optional refiner uses
Secant's affine bank leaf and the ordinary toggle SSE pipeline; it adds no GP
policy or optimizer state to the evaluator.

| Component | Responsibility |
| --- | --- |
| `../secant_sr.h` | Public C99 search and model API |
| `src/internal.h` | Search-only shared structures and checked helpers |
| `src/program.c` | Validation, postorder lowering, resolution, formatting |
| `src/search.c` | Bank generation, population, crossover/mutation, selection |
| `src/cpu.c` | Explicit CPU scoring and reference reduction |
| `include/secant_sr_refine.h`, `src/refine.c` | Bounded Philox coefficient fitting, preserving toggle genomes |
| `include/secant_sr_cuda.h`, `src/cuda.c` | Persistent CUDA scoring adapter |
| `include/secant_sr_lm.h`, `src/lm_model.c`, `src/lm_cuda.c`, `kernels/lm.cu` | Persistent register-only GPU LM, separate from GP policy and the scoring core |
| `src/main.c` | CLI, data loading, JSONL reports, CPU winner audit |
| `../python/secant_sr.py` | Standard-library custom-data client |
| `../python/secant_sr_ast.py` | Expression construction, codec, winner replay |

## Genome and Cartesian product

A logical GP leaf contains one, two, or four direct alternatives. Each alternative
is a fixed input column, a coefficient-bank slot, a fitted coefficient, or a literal. A two-way leaf
specifies one bit; a four-way leaf specifies two distinct bits. Lowering writes
the alternatives followed by Secant's selector opcode. An arithmetic node is
an ordinary Secant opcode following its arguments in postorder.

The whole leaf group is indivisible for GP subtree operations. Crossover therefore
cannot turn selector operands into arithmetic subtrees. Mutation can change its
alternatives or bit IDs. Arithmetic subtrees can grow around leaf groups.

Point mutation can replace a state/slot/literal alternative or change a two-way
bit ID / four-way bit pair. It does not change the leaf group's size; subtree
mutation can replace it with a fixed, two-way or four-way group, or a larger tree.
The base bank stays fixed. Mutating a coefficient leaf can replace its slot reference;
optional refinement updates parameter centers retained in the genome. The winning permutation is evaluated, not bred as
a separate bit string: every child is scored over the full configured product.

Crossover preserves the donor subtree's bit IDs without renaming. Donor and
recipient leaves that use the same bit therefore become coupled. For example,
combining `toggle2(x0,x1,bit0)` and `toggle2(x2,c0,bit0)` allows the paired choices
`(x0,x2)` and `(x1,c0)`, not their full four-way product. Different bit IDs provide
independent choices. This is an explicit current policy, not a promise to preserve
each parent's winning configuration. Bit mutation can change these couplings;
automatic crossover bit remapping remains a search-policy experiment.

Every AST in a call uses the **same** coefficient bank table and permutation bits.
Alternative lists remain local to each AST. Reusing a bit deliberately couples
leaf positions, including between packed ASTs. Different bits vary independently.

```
configurations_per_ast = bank_count * 2^toggle_bits
configuration         = bank_index * 2^toggle_bits + permutation
constant_address      = bank_index * constant_slots + slot
```

For example, `toggle2(column(0), bank(0), bit=0)` can select x0 or coefficient
slot 0. Another AST can use that same bit for `toggle2(bank(1), column(2), bit=0)`.
Their branch meanings differ; their bit value and coefficients are shared.

No setting is materialized away before breeding. Selection scores the best
configuration of each toggle genome; offspring inherit and mutate that genome.
Only the reported winner is also materialized into a fixed AST.

## Ask/tell contract

1. Fill `SRConfig`, supply a bank table `[bank][slot]`, and call
   `secant_sr_create`. The object copies the table and operator list.
2. Call `secant_sr_ask` to get the generation's native AST pointers. Lowering is
   cached within that generation. Do not modify the borrowed programs.
3. Score through `secant_sr_cuda_score`, `secant_sr_cpu_score`, or another adapter
   returning the documented `SRScore` records. A failed score call may have
   partial output: discard it and do not call `tell` with it.
4. Call `secant_sr_tell` with all scores, training-row count and target SSD. The
   whole report is validated before any archive or fitness changes. Repeated
   tell, tell without ask, or changing rows/SSD between generations is rejected.
   The caller must also keep the actual dataset unchanged; equal metadata cannot
   detect a changed dataset.
5. Inspect `best`/`progress`, then `advance` for the next generation if desired.
   No per-individual or per-generation allocations occur inside this GP object.

`ask` storage is owned by the search and can be overwritten after `advance`/`seed`.
`best` nodes are borrowed and can change at the next successful `tell`. Copy them
if retaining a snapshot. `seed` optionally replaces an initial candidate with
a validated genome; normal CLI search does not seed a known solution.

`program_write`, `resolve`, and `format` support size measurement with a null
output and reject insufficient capacity before emitting output. `resolve` uses
the actual bank values, not RNG regeneration. The final JSON carries the winning
vector, index, permutation, original AST, and fixed AST for independent replay.

The search uses tournament selection with a small node-count penalty, a global
best archive ranked by raw SSE, subtree crossover, subtree/point mutation, and
random immigrants. The first survivor is the global incumbent; additional
survivors come from tournaments. Structural retries are bounded and counted.
These are initial policies, not a tuned replacement for the old QD controller.

## GPU lifecycle and memory

`secant_sr_cuda_create` retains a primary context, makes it current, uploads the
immutable dataset/banks, prepares a Secant template, and creates persistent
runner workers/streams, reducer buffers, pinned host records, and events. Calls
must retain that current CUDA context. Each handle is single-caller; do not run
or destroy it concurrently. Multiple independent handles can be used by callers
that manage their CUDA contexts correctly.

The grid is bounded by both `ast_batch` and `score_bytes`. Each chunk reuses it.
One reducer CTA per AST returns best finite nonnegative SSE, lowest configuration
index on exact ties, and valid-configuration count. An all-invalid AST returns
infinity, `UINT64_MAX`, and zero. Only **24 bytes per AST** cross back to the host.
The full SSE grid still exists on the device; this is reduction, not fused scoring
and argmin. There is no per-generation device allocation for data or score grids.

The pipeline blocks until scoring finishes, then the adapter reduces and gathers.
Events guard reduction and D2H completion. There is no device-wide synchronization;
owned-stream synchronization is only an error-recovery fallback. Cross-stage
overlap and a single combined reduction/transfer wait remain tuning opportunities.

Creation may return a non-null cleanup-only handle if cleanup itself fails.
Failed destruction retains ownership for retry. Preserve all resources if
completion cannot be established. Driver/cleanup failures block reuse; a rejected
AST can be retried after the core confirms cleanup. A successful destroy invalidates
the handle. The CLI exits nonzero on any CUDA failure; it never retries on CPU.

Each scorer creates one scoring template and one reduction module. NVRTC is
called at creation, not per generation. ASTs are specialized/loaded per scoring
call using the existing Secant pipeline. The old SQLite CUBIN cache and persistent
multi-dataset process are **not ported**; a warm NVRTC internal cache can shorten
compilation but is not a Secant-SR persistent CUBIN cache.

## Defaults, limits, and reporting

Defaults: 1,024 members, 64 banks, 4 coefficient slots, 6 bits, 31 logical nodes,
initial depth 3, maximum depth 7, operators add/subtract/multiply/sine/cosine,
256 ASTs per scoring chunk, 256 MiB score ceiling, 8 ASTs/kernel, 16 kernels/module,
128-row tile, 128 threads, 3 specialization workers, and 8 runner streams.

The CLI samples 64 four-value coefficient vectors uniformly from `[-2, 2]`
at startup by default. It evaluates each AST against these same vectors crossed
with all 64 bit permutations: 4,096 configurations per AST. These are 64 vectors,
not 64 independent choices for each of four coefficients. GP can mutate slot
references and literal leaves, but does not refine or resample bank values.
Literal mutations use `-2, -1, -0.5, 0, 0.5, 1, 2`. The bank distribution, size,
and range are configurable. Optional `--refine-rounds N --refine-budget K`
adds adaptive packed coefficient fitting after broad scoring and before breeding.
Defaults are zero rounds, 128 selected ASTs, and initial scale 1. Each round uses
the configured number of bank rows crossed with all permutations. See the
[refinement contract and measured limits](../docs/toggle-refinement-20260919.md).

- One target and at least one input. Input columns plus bank slots must be ≤128.
- At most 63 logical GP nodes, depth 32, and 16 toggle bits in this controller.
  Secant itself has a wider bit-count contract; this is a search-layer limit.
- The bytecode expands leaf alternatives, so a logical GP node is not one native
  AST instruction. The GPU patch budget conservatively follows the logical cap.
- At least one bank, including constant-free expressions. A complete AST's
  configuration scores must fit the pool; otherwise creation fails. No hidden
  truncation of banks, bits, population, or operators occurs.
- Templates/large input counts can exceed hardware/compiler/register constraints.
  Such failures are explicit; there is no automatic shape fallback in this layer.
- Two fixed-capacity population buffers and AST storage are allocated at creation.
  Memory scales with population and the 63-node storage capacity. A million-member
  population is not a low-memory default; allocation can fail.
- `--seconds` is cooperative between generations, includes setup, and permits at
  least one generation. It is not a hard deadline for compilation or a GPU launch.
- Arithmetic has ordinary domains: invalid divisions/logs/overflow produce invalid
  scores. They are counted as work and excluded from winning, not silently protected.
- The base bank uses deterministic SplitMix64 plus uniform or Box–Muller
  transforms. Refinement uses a separate host Philox4x32-10 bank, uniform [-1,1),
  with per-AST scale/offset embedded by the specializer. The separate opt-in
  GPU LM path uses Philox starts; see the [LM contract](../docs/toggle-lm-20260921.md).
- A refinement model is one start. The CLI selects one start per chosen AST.
  The default `all` policy skips fitting models with more independent parameters
  than bank slots and reports `skipped_parameter_capacity`. The opt-in
  `active-block` policy rotates bounded blocks of active parameters and reports
  `skipped_inactive` for genomes without adjustable parameters in the incumbent
  binding. All toggle alternatives remain available. See the
  [matched policy trial](../docs/policy-repair-20260920.md).

Every scored occurrence is counted, including survivors, duplicate expressions,
unused-bit combinations, and invalid configurations. Never label these counters
unique ASTs or distinct models. Fingerprints break ties; they are not deduplication.
CPU generation/replay is deterministic for fixed inputs and seed. GPU floating-point
reduction order can perturb close fitness ties and subsequent GP selection, so a
seed alone is not a bitwise cross-device search-reproduction guarantee.

Final `solved` requires both training and held-out NMSE ≤ `--stop-nmse` (default
1e-10). With constant targets NMSE falls back to MSE. Held-out rows are used only
for final reporting, not to select structures. The result explicitly reports
`symbolic_equivalence: "not_checked"`. A close numerical approximation can pass.

The final CPU/GPU score audit uses `scaled_rmse_v1`: RMSE gap divided by
`max(1, target RMS, GPU RMSE, CPU RMSE)` must be at most 2e-5. It preserves both
scores in `score_audit`; reported training/test metrics use the CPU evaluation.
This checks numerical score agreement, independently of the AST/index replay
checks. A relative-SSE-only audit falsely rejected near-fit exp/log expressions
whose row predictions agreed within one ppm; the [SRBench report](../docs/srbench_toggle.md)
records the diagnosis and repaired validation policy.

`scoring_seconds` covers broad-search adapter calls; `refinement.seconds` covers
the entire fitting phase. Pipeline/device/load/reduction/transfer totals include
both. A whole adapter call includes pipeline, reduction and
gather; device/load/reduction components are nested or overlapping measurements,
not numbers to sum. `setup_seconds` contains `nvrtc_seconds`. `total_seconds`
includes scorer teardown; `result_ready_seconds` precedes it. External process
wall time additionally includes process startup/exit and output handling.

## Validation and next work

Host tests cover encoding, replay, score reduction, deterministic evolution,
transactional errors, custom datasets and the Python client. The CUDA test covers
129 ASTs, ragged chunks/modules, mixed bank/state toggles, invalid domains, rejected
AST recovery, ties and repeated calls against the CPU reference. See the dated
[report](../docs/toggle-search-20260919.md) for executed hardware and timings.

Remaining original work:

2. The C99 search, Python fit/codec and [SRBench campaign adapter](../docs/srbench_toggle.md)
   work. The 84-fit, 900-second GPU LM diagnostic campaign is running on both
   rack1 GPUs (2026-09-21). Analyze it, expand multi-seed quality coverage and
   validate black-box limits. Add persistent template reuse for short searches;
   port legacy Secant Python bindings separately if needed.
3. Tune arbitrary ASTs with shared source registers and temporary selector results.
   No similarity-based packing or cross-AST selected-value retention is enabled.
   The reuse experiment is archived separately; its shared-fixture speedup does
   not apply to the restored default. Measure instruction/register cost and actual
   GP throughput, including long-row and larger-packing cases.
4. Native-toggle random refinement and GPU LM are implemented and validated.
   Tune rounds, selected population, starts, parameter capacity, and RNG placement.
   LM uses zero-local-memory CUDA register shapes; SASS derivative specialization
   remains future work. See the [LM report](../docs/toggle-lm-20260921.md).
