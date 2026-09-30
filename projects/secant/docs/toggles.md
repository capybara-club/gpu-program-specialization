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

# Secant 0.3: explicit toggles and coefficient banks

## The contract

The AST specifies **what can vary**; a run supplies coefficient values and the
Cartesian dimensions. No leaf masks, column-index tables, settings words,
optimizer state, or compatibility aliases are part of this API.

| Instruction | Bytes | Meaning |
| --- | --- | --- |
| `column_f32(i)` | 2 | The current row's value in fixed column i |
| `bank_constant_f32(i)` | 2 | Fixed slot i in the selected bank vector |
| `affine_bank_f32(i, scale, offset)` | 10 | `scale * bank[i] + offset`, with separate f32 rounding |
| `constant_f32_bits(bits)` | 5 | An inline IEEE-754 literal |
| `toggle2_f32(bit)` | 2 | Select one of the preceding two leaves |
| `toggle4_f32(low_bit, high_bit)` | 3 | Select one of the preceding four leaves |

For two leaves `[a,b]`, bit 0 selects `a`, bit 1 selects `b`. For four leaves
`[a,b,c,d]`, choice is `permutation[low_bit] + 2*permutation[high_bit]`.
The two bit indices must differ. Every selected operand must be a direct column,
bank slot, affine bank leaf, or literal. Arithmetic subtrees and nested selectors are not valid
selector operands. Arithmetic can freely combine selector results afterward.

Repeated use of a bit deliberately couples positions. Use different bits for
independent choices. A run with T bits enumerates all `2^T` permutations,
including unused-bit combinations; it does not deduplicate them. Every AST reads
the same permutation bit pattern for a configuration. Bit IDs have the same value
across ASTs; the leaves selected by those bits are specified by each AST. T can be 0–32,
subject to checked configuration counts and buffer extents.

## Example expression

```c
static const SecantAstInstruction expression[] = {
    /* bit 0: column 0 or column 1 */
    secant_ast_encode_column_f32(0),
    secant_ast_encode_column_f32(1),
    secant_ast_encode_toggle2_f32(0),

    /* bit 1: coefficient slot 0 or column 2 */
    secant_ast_encode_bank_constant_f32(0),
    secant_ast_encode_column_f32(2),
    secant_ast_encode_toggle2_f32(1),

    secant_ast_encode_mul_f32,
    secant_ast_encode_bank_constant_f32(1),
    secant_ast_encode_add_f32,
    secant_ast_encode_return_f32
};
```

With 1,024 bank vectors and two bits, **each AST gets 4,096 configurations**.
Slots 0 and 1 retain the same values across all four permutations of one bank.
Literals remain fixed across every bank and permutation.

## Recipe and run

```c
SecantCubinToggleSSERecipe recipe = secant_cubin_toggle_sse_recipe_init();
recipe.num_kernels = 8;
recipe.asts_per_kernel = 8;
recipe.num_inputs = 3;
recipe.num_constants = 2;       /* shared slots for every AST */
recipe.num_targets = 1;
recipe.tile_rows = 128;
recipe.threads_per_block = 128;
recipe.patch_capacity_instructions = 2048;

/* Generate source, compile with NVRTC, inspect, create the runner.
   The run assumes caller-owned device allocations have already been uploaded. */
SecantCubinToggleSSERun run = secant_cubin_toggle_sse_run_init();
run.programs.asts.items = ast_pointers;
run.programs.asts.count = ast_count;
run.input = input_matrix;       /* [column][row] */
run.targets = target_matrix;    /* [target][row] */
run.num_rows = row_count;
run.num_targets = 1;            /* must match this toggle recipe */
run.num_banks = 1024;
run.toggle_bits = 2;
run.constants.address = banks_device_address;
run.constants.num_elements = 1024 * 2;
run.constants.bank_stride = 2;
run.output = scores_matrix;    /* [ast][target][4096], leading dimension >= 4096 */
SecantRunnerStats stats = secant_runner_stats_init();
SecantResult result = secant_cubin_runner_run_toggle_sse(runner, &run, &stats);
```

Coefficient addresses, in float elements:

```
bank * bank_stride + constant_slot
```

One bank table is shared across every AST in the run, including all kernels and
modules. For a given configuration, every AST sees the same coefficient values
and permutation bits. There is no per-AST bank stride. Use separate runs or batch
entries to evaluate different bank tables. Each kernel loads only K coefficient
registers for K slots, regardless of the number of packed ASTs.
`bank_stride >= num_constants`; padding is allowed. With zero constants, the
bank pointer may be null and the stride must be zero. At least one bank is
required even for a constant-free expression.

Output offset:

```
(ast * num_targets + target) * output.leading_dimension + configuration
configuration = bank * 2^toggle_bits + permutation
bank = configuration >> toggle_bits
permutation = configuration & ((UINT64_C(1) << toggle_bits) - 1)
```

The CPU equivalent is `SecantCpuToggleSSERun`, with host pointers and explicit
`num_inputs`/`num_constants`. Both paths overwrite active SSE bins and preserve
padding. Division by `num_rows` converts SSE to MSE. NaN/Inf results from invalid
expression domains are not silently repaired.

## Specialization and execution

Only the template recipe goes through NVRTC. AST specialization patches native
instructions into copies of that template; changing coefficient values does not
change the generated instructions. Current `run()` calls re-specialize/load the
AST modules; `run_batch()` shares that work across data runs within one call.
There is no hidden cross-call AST/module cache.

The template loads the shared bank vector once before its row loop. Row
columns use fixed shared-memory offsets. Native toggles test permutation bits
and select registers. The inspector verifies the compiler's predicate-test and
selection forms, records their register bindings, and the writer reserves the
permutation register while emitting AST code. It uses the inspected predicate
encoding and conservative ALU dependency stalls.

`asts_per_kernel` explicitly controls packing. More packed ASTs amortize module
loads and row reads but keep more accumulator registers live. Coefficient
registers are shared; arithmetic temporaries are recycled between ASTs.
Start with modest packing and measure. The default is one AST/kernel; the caller
can request up to 32, subject to register/scaffold limits. The final kernel and
module can be partial; inactive ASTs never write output bins. Bank reads do not
depend on the AST index. Configuration blocks are spread across grid.y, with a bounded
grid-stride loop for larger products.

The runner keeps a bounded specialization queue, persistent CPU workers, and
its own persistent CUDA streams/events. Specialization runs ahead of GPU work.
A module is launched on those streams, completed through events, then unloaded.
The runner does not perform a device-wide synchronization. If event completion
fails, recovery tries events on every owned stream and then synchronizes only
streams whose event path also failed. No stream synchronization is used on the
successful path. Failed partial
specialization images are discarded; every slot starts from the immutable
original template on reuse.

Buffer extents, products, strides, output overlaps across batch entries, leaf
indices, and active toggle bits are checked before clearing output or launching.
Full structural AST validation is fused with specialization. If that later pass
fails, output may already have been cleared or partially written. The returned
error and partial counters are authoritative; do not consume scores from a failed
call. The runner is reusable after a failed call only when completion and module
cleanup succeeded. `SECANT_ERROR_COMPLETION_UNKNOWN` explicitly reports that
completion could not be established: preserve the submitted device buffers,
and retry destruction. The module stays owned until it is safe to unload.
An unload failure also blocks new runs. Destroy errors retain the handle for
retry; only successful destruction invalidates it. Creation normally returns
NULL on failure, but returns a destroy-only handle if its cleanup also failed.
Run/destroy calls require the same current CUDA context as creation.

## Migration and scope

- Library version is `0.3.1`; recipe, CPU run, and CUBIN run descriptor versions
  are 3. Version-1 and version-2 descriptors are rejected instead of reinterpreted.
  Recompile callers and rebuild cached templates. The version is part of the
  source/binary cache identity; the 0.2 bank field and kernel argument were removed.
- Use `column_f32` for fixed row columns and `bank_constant_f32` for coefficient
  slots. Old dynamic leaf opcodes 0xb7–0xb9 are invalid.
- AST/routine lists contain only programs. Optimizer centers/scales/velocities
  are absent. LM and coefficient proposal generation are outside this change.
- The optional legacy Python and benchmark adapters are not migrated. CMake
  reports this explicitly when those options are requested. The new
  [Secant-SR path](../../secant-sr/toggle/README.md) constructs native toggle ASTs,
  supplies shared banks, and includes a standard-library Python fit/codec client,
  SRBench campaign adapter, and optional packed random coefficient refinement.
  The historical maturity/LM controller is not ported.
- Static materialize, SSE, affine, and Gram paths remain active. Their packed
  AST layouts have not changed. Old mixed settings tests are archived; active
  static tests and random-program stress coverage verify the retained paths.

## Validation and measured tradeoffs — 2026-09-19

Host: five C/C++/CPU tests on mac1. GPU: rack1, RTX 5080 / SM 12.0, CUDA 13.1.
The CUDA test independently compares with the CPU reference, whose selectors
are checked against hand-written formulas. Tests cover mixed two-/four-way
leaves, coupled bits, shared banks, changed bank values, multi-run
batches, padded outputs, partial modules/kernels, rejected AST recovery, routines,
zero columns/constants/toggles, and native bit-31 mask encoding.

Full enumeration of 32 bits was **not** executed: even one AST/target requires
16 GiB of score storage. Its count/overflow behavior and specialized mask are
tested separately.

The randomized test uses 129 AST occurrences, 8 columns, 8 constants, 5 banks,
8 toggle bits, and 97 rows: 165,120 configurations and 16,016,640 row evaluations.
The historical 0.2 measurements below use exactly the same expressions and data, with 2,048 reserved patch instructions per function in both cases (deliberately not tuned for the single-AST case). They are a
correctness/workflow smoke test, not an occupancy-saturated benchmark.

| ASTs/kernel | Functions/module | GPU event time | Whole runner call | Module loading | Specialization worker time |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 0.469 ms | 3.394 ms | 1.878 ms | 0.393 ms |
| 8 | 8 | 0.216 ms | 0.862 ms | 0.386 ms | 0.176 ms |

These numbers exclude initial source generation, NVRTC, plan inspection, runner
creation, and input transfers. They must not be compared as equivalent to the
historical row-parallel engine peak. Single-AST packing was the first working
implementation; configurable packing was added before completion to avoid
forcing its increased module count on users. Larger banks can amortize this
further, but a representative throughput sweep remains follow-up work.

The 0.2 template reserved `A * K` coefficient bindings for A packed ASTs with K
slots, even when bank data was shared. The 0.3 shared-bank contract removes that
avoidable duplication: all ASTs now refer to the same K bindings. Arithmetic
temporaries recycle between ASTs; each AST/target retains its own accumulator.

Resource comparison after specialization on the same deterministic random test:

| ASTs/kernel | 0.2 registers/thread, max | 0.3 registers/thread, max | 0.3 local bytes/thread, max |
| --- | --- | --- | --- |
| 1 | 47 | 47 | 0 |
| 8 | 165 | 53 | 0 |
| 32 | Unsupported at 8 columns + 8 constants | 168 | 16 |

The register numbers are driver-reported across active kernels in the first
specialized module, including scaffold and temporaries, not just coefficients.
The baseline used rack1 GPU 0 and 0.3 used GPU 1, both RTX 5080 / CUDA 13.1.
All compared scores passed the CPU oracle. The 32-AST case reserves 8,192 patch
instructions/function versus 2,048 for the 1-/8-AST cases; it is a capacity and
correctness check, not a matched throughput comparison. Its local-memory use
occurs outside the inspected patch island; no inspection checks were weakened.
Do not infer that maximum packing is optimal from these small tests. The
8-AST case removes the duplicated constants without introducing local memory.

Logs are retained under `scratch/toggle-refactor/` and rack1's isolated
`/home/cdurham/experiments/secant-shared-banks-20260919/`; the previous template
and baseline resource probe remain in `secant-toggle-20260919/`.

An early test exposed NVRTC unrolling the patch island seven times. Explicit
non-unrolling now keeps one inspected island per function; the ragged 31-row
case remains in the regression suite. Unexpected scaffold code is rejected
rather than accepted by weakening inspection.

The 0.3 validation passed on rack1 GPU 1 and under CUDA Compute Sanitizer memcheck
on GPU 0 with **0 errors**. It covers packing 1, 3, 8, and 32, exact-sized shared
bank allocations, partial modules/kernels, changed bank values, and all four
retained static shapes. All five host tests passed with AddressSanitizer and
UndefinedBehaviorSanitizer. The earlier 0.2 static random-program suite passed
216 ASTs across arithmetic and transcendental paths. No claim is made that every
archived 0.1 settings/optimizer test was ported.

The runner's failure paths were subsequently hardened and tested in
[runner-hardening.md](runner-hardening.md): 122 deterministic fault/concurrency
scenarios, host sanitizers, a second architecture (RTX 4090 / SM 8.9), and another
zero-error CUDA memcheck run on SM 12.0. This completes original item 1 within the
available hardware; it does not validate SM 9.0/10.0 or older compiler versions.

## Remaining work

Original item 1 (failure-path hardening and expanded hardware coverage) is complete.
The original numbering is retained for the remaining items:

2. **New Secant-SR/Python search path implemented:** C99 GP retaining local
   toggles, persistent bounded CUDA scoring/reduction, and a Python custom-data
   client with winner replay. See its [validation report](../../secant-sr/docs/toggle-search-20260919.md).
   The [SRBench campaign adapter](../../secant-sr/docs/srbench_toggle.md) is now
   restored and all 116 Feynman integration jobs pass. Both first recovery baselines
   are complete; a matched-data/time settings-versus-toggle campaign is running.
   Broader multi-seed/black-box validation, persistent template reuse
   and legacy Secant Python bindings where needed remain; old search results do
   not transfer.
3. Tune arbitrary AST packs, bank sizes, row tiles, and coefficient counts;
   measure occupancy/register pressure and full request time separately.
   State and bank source registers are shared; selector outputs are ordinary
   temporary operands, released or overwritten when consumed. There is no
   similarity-based packing or retained cross-AST selector cache. The
   [reuse experiment](../bench/settings_vs_toggles/selector-reuse-experiment/README.md)
   improved shared fixtures but not the measured random GP initial populations;
   it was removed from the default at the user's request. The default therefore
   retains the [original measured shared-choice slowdown](../bench/settings_vs_toggles/report-20260919.md).
   Prioritize instruction scheduling and temporary liveness for independent ASTs;
   the 32-AST scaffold's local-memory use also remains open.
4. Native-toggle random coefficient refinement is implemented in
   [Secant-SR](../../secant-sr/docs/toggle-refinement-20260919.md).
   Secant adds only an affine bank leaf; proposal/selection policy stays outside
   the evaluator. Tune fitting budgets and starts; LM remains deferred.
