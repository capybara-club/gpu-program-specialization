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

# Python trajectory-LM research kernel

> **Current validation:** The coupled RK4 sensitivity-stage ordering defect was repaired on 2026-09-02. Older rates and optimizer counters later in this development record are historical. Current correctness, performance, and resource results are in [`../benchmarks/2026-09-02-corrected-coupled-lm.md`](../benchmarks/2026-09-02-corrected-coupled-lm.md).

This path fits indexed FP32 constants while integrating complete ODE trajectories. It generates a CUDA template, verifies physical patch sites in the compiled CUBIN, symbolically derives local right-hand-side partials, and specializes one complete candidate system plus its derivatives directly into SASS.

## Execution model

One CUDA thread owns one complete LM fit. Its low fit-index bits select the AST toggle permutation and its remaining bits select a row of starting constants:

```text
permutation_count = 2^active_toggle_count
fit_count         = start_count * permutation_count
permutation       = fit_index & (permutation_count - 1)
start_index       = fit_index >> active_toggle_count
starts            = float[start_count][optimized_constant_count]
```

Toggle instructions use the same immediate bit-index ABI as the scoring kernel. They select direct states, literals, or optimized constants without a runtime binding table. Reusing one toggle index couples multiple sites to the same structural decision.

`OPTIMIZED_CONSTANT_F32(index)` is a two-byte postorder instruction. Every occurrence of an index reads the same parameter register, and every derivative occurrence contributes to that parameter's one gradient and Gram-matrix row. The validated example deliberately uses optimized-constant index 2 in both right-hand sides.

Each thread keeps its state, parameters, forward sensitivities, gradient, packed upper-triangular Gram matrix, proposal, and Cholesky solve state resident. A kernel invocation performs fixed-step RK4, builds `JtJ` and `Jtr`, tries a bounded damping schedule, evaluates proposals, accepts improvements, and exits without host-controlled rejection launches.

## Specialization layout

The grouped v4 template exposes several scalar outputs through each physical site:

1. The primal right-hand sides occupy one or more groups and are never mixed into a partial group.
2. Local partials are packed into consecutive groups up to the inline-assembly operand limit.

For the validated four-state/six-constant shape, this is one four-output primal site followed by five eight-output partial sites: six sites for 44 scalar values. The group specializer interns the postorder expressions into one DAG and retains shared intermediates once in registers within each site. Equal outputs and derivative subexpressions therefore reuse work rather than compiling independently.

The inspector identifies physical input, result, predicate, permutation, temporary, entry, and patch registers for every group. Site-specific salt instructions prevent PTXAS from merging equal toggle probes. The specializer replaces the inspected entry with a direct branch to the patch arena, so marker and keepalive scaffolding are never executed. If PTXAS assigns a CUDA output register to an input register, the program computes into a safe marker register and relays the result after all input uses are complete.

Marker ABI v4 uses finite marker immediates. A prior NaN-valued marker let PTXAS prove away the grouped toggle probe through NaN propagation; the finite marker preserves the physical predicate and select instructions required for inspection. This marker choice does not alter specialized execution because the entry branch skips the disposable scaffold.

The derivative backend symbolically computes forward local partials. It simplifies literal zeros and ones and performs cross-output CSE inside each group; it does not build a reverse-mode tape.

## Runtime data

The current runtime ABI accepts ragged, irregular trajectories:

```text
trajectory_offsets[trajectory_count + 1]
trajectory_times[trajectory_point_count]
reference[state_count][trajectory_point_count]
reference_weights[state_count][trajectory_point_count]
```

Each trajectory's first point is its initial state. Every later residual and sensitivity is multiplied by its runtime weight before accumulating MSE, gradient, and Gram matrix. All-one weights recover ordinary MSE. The CTA cooperatively stages offsets, times, weights, and dense-state observations in dynamic shared memory; every fitting thread then integrates all trajectories independently.

Outputs are:

```text
constants_out[fit_count][optimized_constant_count]
initial_mse_out[fit_count]
mse_out[fit_count]
iterations_out[fit_count]
accepted_steps_out[fit_count]
factorization_attempts_out[fit_count]
```

## Python workflow

After installing the package in a normal project environment, the command-line workflow is:

```bash
odezza-lm-generate --shape examples/lm_shape.json -o lm_template.cu
odezza-compile lm_template.cu -o lm_template.cubin --architecture sm_120
odezza-lm-inspect lm_template.cu lm_template.cubin -o lm_inspection.json
python3 examples/make_lm_system.py -o lm_system.json
odezza-lm-specialize lm_template.cu lm_template.cubin lm_system.json -o lm_specialized.cubin --report lm_specialization.json
```

The source manifest binds source, CUBIN, dimensions, site ordering, and specialization ABI with SHA-256 identities. Inspection rejects a mismatched source/CUBIN pair. The specialization report records the template identity, system identity, toggle count, register count, per-site instruction counts, and decoded derivative programs.

## Validated behavior

The liveness-hardened scalar ABI and current grouped ABI have been compiled, inspected, specialized, executed, and independently CPU-replayed on RTX 5090 (`sm_120`) and RTX 4090 (`sm_89`). The validated templates have no stack frame or spills.

The initial SM120 validation used two states, eight optimized constants, one reused constant index, one toggle bit, three trajectories, 21 points per trajectory, and four RK4 substeps per interval.

- Templates: 189 registers on SM120 and 194 registers on SM89, with no stack or spills.
- Perturbed starts: best initial MSE `7.70e-2`; final GPU MSE `2.49e-15`.
- Independent CPU replay of that winner: MSE `3.68e-15`.
- The structurally correct toggle branch won.
- The original warmed diagnostic reached approximately 382,600 complete fits/s on the RTX 5090 and 359,600 fits/s on the RTX 4090. An execution audit found that each scalar site still executed its NOP-filled scaffold. The production entry branch raises this two-state diagnostic to approximately 3.29M/2.96M fits/s without changing any result.
- Mean completed LM iterations, accepted steps, and actual Cholesky factorization attempts are `5.9624`, `5.9229`, and `6.3798` respectively on both devices.

The matched fed-batch validation also passes on both GPUs and independently replays on CPU. It recovers the six constants to maximum scaled error `1.92e-6`. The original generic 4x6 scalar-site path reached only about 5.54k fits/s on the RTX 5090 and 5.17k fits/s on the RTX 4090 because it executed 1,329 dead scaffold instructions around 621 useful specialized instructions. The scalar production entry branch reaches 30.8k/31.4k fits/s. Grouped sites and cross-output CSE reach 88.8k/78.7k fits/s with the robust 256-instruction reserve. The winning fit, MSE, recovered constants, iteration counts, accepted steps, and factorization attempts are identical.

The grouped fed-batch templates use 219 registers on SM120 and 223 registers on SM89, with no stack or spills. They emit 228 specialized instructions including site-return branches, versus 621 useful scalar instructions before the old scalar returns. A 32-thread CTA is fastest on both devices. Against the older split specialized implementation on the identical workload, production Odezza is 1.40x faster on Rohini and reaches 96.8% of its rate on Ada. Both production templates accepted all 512 deliberately large randomized stress systems; their maximum specialized register counts were 228 and 232 respectively.

The throughput number is workload-specific. Fits exit after different iteration counts, and this two-state synthetic problem is much smaller than the fed-batch model, so it must not be compared directly with earlier native or fed-batch fit rates.

## Current boundaries

These are explicit research-shape limits, not silent fallbacks:

- One candidate system is specialized per module. Starts and toggle permutations provide parallel work; multi-system LM packing is not implemented.
- The state count and optimized-constant count are compile-time template dimensions. State names are excluded from template identity.
- The grouped marker ABI chooses an output-group capacity that keeps each inline-assembly site within 30 operands. The sum of state and optimized-constant inputs must leave at least one physical output operand.
- The measured four-state/six-constant grouped template is spill-free at 219 registers on SM120 and 224 on SM89. The earlier four-state/eight-constant scalar compile reached 255 registers and spilled; grouped 2x8 validation and broader pressure stress remain required before claiming that regime is resolved.
- Indexed scoring constants are rejected. LM programs currently contain states, optimized constants, literals, toggles, and smooth operators.
- `ABS`, `MIN`, and `MAX` are rejected until a nonsmooth derivative policy is selected.
- Every active state is observed at every point. Runtime per-observation weights are supported, but sparse observations, event dosing, and structured residual-error models are not yet represented.
- The whole runtime trajectory table must fit the launch's dynamic shared-memory budget.
- Integration is fixed-step FP32 RK4. There is no adaptive or stiff solver and no FP64 kernel.
- The derivative specializer is Python research code. It is not yet in the allocation-free C99 execution hot path.
