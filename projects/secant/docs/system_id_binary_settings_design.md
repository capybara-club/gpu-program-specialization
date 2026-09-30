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

# Binary Settings for Specialized Trajectory Search

## Purpose

This document describes a design question for a GPU system-identification
engine that specializes postorder expression trees directly into native GPU
instructions and scores them by integrating complete ODE trajectories.

The current engine gives every dynamic AST leaf an arbitrary runtime binding
into a shared bank of state values and constants. That is expressive, but it
requires per-leaf indices, indexed shared-memory reads, and a relatively large
runtime setting space. The proposed alternative gives each dynamic leaf two
specialized choices and uses one bit per leaf to select between them. The lost
within-kernel generality may be recovered by specializing many different
two-choice menus, because AST-to-SASS specialization is extremely fast.

The immediate decision is not simply "arbitrary bindings versus bits." It is:

> Which combination of discrete leaf choices, explicit constant slots,
> specialization frequency, and SSE/LM staging produces the best
> wall-clock time to a useful recovered equation?

The binary design has not yet been benchmarked. Any expected speed advantage
below is a hypothesis that must be measured against the implemented arbitrary-
binding path.

## Workload and terminology

The engine evaluates candidate right-hand-side expressions inside a fused ODE
integrator. A candidate evaluation therefore includes the expression, RK4
stages, trajectory state updates, and comparison with observations. It does
not materialize every intermediate state to global GPU memory.

The important terms are:

- **AST skeleton:** the postorder operators and leaf sites specialized into a
  CUDA template's SASS patch region.
- **Dynamic leaf:** an AST input site whose value source changes across runtime
  settings without changing the AST operators.
- **Setting:** one complete assignment of all dynamic leaf choices and any
  runtime constants for one AST or system genome.
- **Bank:** the runtime values visible to dynamic leaves. The current logical
  layout is `[state_0, ..., state_(S-1), constant_0, ..., constant_(K-1)]`.
- **SSE evaluation:** one forward trajectory score for one AST/system setting.
- **LM fit:** one setting and parameter start optimized through repeated
  rollouts, forward sensitivities, `J^T J`/`J^T r` accumulation, damped solves,
  proposal evaluation, and acceptance. One LM fit is much more work than one
  SSE setting evaluation.
- **Incumbent:** the winning bindings and constants retained by a genome from
  the previous generation.

## Current implementation

### Thread-owned trajectory evaluation

One CUDA thread owns one runtime setting. It integrates the required
trajectories and retains its evolving state and score. A CTA caches invariant
reference data in shared memory. Many settings, genomes, and modules supply the
independent work needed to occupy the GPU.

The postorder AST contains fixed dynamic-leaf slot numbers. For dynamic leaf
`i`, the runtime setting supplies an integer binding:

```text
binding[i] in [0, S + K)

binding[i] < S       -> current RK4 stage state[binding[i]]
binding[i] >= S      -> setting constant[binding[i] - S]
```

At every RK4 stage, the state values and per-thread constants are made visible
through a shared-memory bank. The leaf reads `bank[binding[i]]`. This lets any
dynamic leaf become any state or any constant without respecializing the AST.

This is highly expressive, but the generality has costs:

- one full-range binding index per dynamic leaf and setting;
- deterministic random generation or materialized storage for those indices;
- bounds handling and an indexed shared-memory load at every dynamic leaf;
- shared-memory storage and traffic for the per-thread state/constant bank;
- less information in the kernel about which states a leaf can legally use;
- more complicated local partials and more register pressure in LM; and
- a large search space containing many physically or dimensionally implausible
  assignments.

### Two current SSE setting paths

The two paths represent the same search distribution but differ in where the
settings are constructed.

#### Materialized settings

The host constructs and uploads full arrays of constants and leaf bindings for
every genome and setting:

```text
constants[genome][constant_slot][setting]
bindings[genome][leaf_slot][setting]
```

Setting zero is the incumbent. Other constant values are sampled around the
incumbent. Other leaf bindings either retain the incumbent or select an
arbitrary bank entry.

This path is simple to replay and inspect, but it uses more host work, memory,
upload bandwidth, and GPU memory.

#### Hashed-incumbent settings

The host uploads only each genome's incumbent constants and bindings. The GPU
regenerates a setting deterministically from:

```text
seed, generation, genome, leaf-or-constant slot, setting index
```

The generator is a small deterministic 64-bit hash/mixer. It is **not Philox**.
Setting zero remains the incumbent. With the current defaults, every other SSE
leaf retains its incumbent binding with probability `0.5`; otherwise it picks
uniformly from the complete state-plus-constant bank. A constant is perturbed
inside a radius

```text
(abs(incumbent) + 1) * constant_mutation_scale
```

where the current default scale is `0.5`.

The winning setting index is enough for the CPU to regenerate and retain its
constants and bindings. A materialized replay verifies the final winner.

This mode removes the full settings arrays but does not remove arbitrary bank
selection, indexed leaf loads, or the state/constant shared-memory bank.

### Current LM setting path

The measured dense trajectory-LM kernel is also thread-owned. A thread owns
one `(binding setting, parameter start)` fit. The binding assignment remains
fixed throughout that fit; LM updates only its active continuous constants.

The established fed-batch control has:

- four states;
- 16 dynamic leaves selecting from four states and eight constant slots;
- six active LM constants;
- four starts per binding setting in the main benchmark;
- forward sensitivities through RK4; and
- thread-local Gram matrix, gradient vector, damping, solve, and proposal
  state.

The current LM setting constructor retains each incumbent leaf binding with
probability `0.75` by default and otherwise selects from the full bank. Starts
perturb the active incumbent constants. This is a materialized setting/start
path, separate from the SSE hashed-incumbent implementation.

## Proposed binary-choice design

### Core representation

Give each dynamic leaf `i` two sources selected when the AST is specialized:

```text
choice_a[i]
choice_b[i]
```

At runtime, bit `i` of the setting chooses the source:

```text
leaf_i = ((setting_mask >> i) & 1) ? choice_b[i] : choice_a[i]
```

If the number of binary leaves is at most 32, the complete setting can be one
`uint32_t`. When settings enumerate masks in order, the setting index itself is
the mask and no settings array is necessary.

The setting counts are exact:

| Binary leaf choices | Exhaustive masks |
| ---: | ---: |
| 8 | 256 |
| 10 | 1,024 |
| 12 | 4,096 |
| 13 | 8,192 |
| 14 | 16,384 |
| 16 | 65,536 |

For example, 4,096 exhaustive settings encode **12**, not 14, independent
binary choices.

When both choices are known states, the kernel can select between two known
stage-state registers. It no longer needs a full-range binding, modulo
selection, or arbitrary shared-memory bank lookup for that leaf. The exact
SASS may use a predicate and select instruction or an equivalent mask
operation.

### How binary choices recover useful breadth

One binary menu covers only `2^L` assignments for `L` leaves, whereas a full
bank of `B` sources represents `B^L` possible assignments. Binary menus must
therefore be treated as specialized search neighborhoods, not as an exhaustive
replacement for the full Cartesian bank.

A new menu can be specialized for the same AST with different pairs, such as:

- incumbent state versus a newly sampled state;
- a state inherited from parent A versus one inherited from parent B;
- two unit-compatible or domain-allowed states;
- incumbent binding versus a residual-informed proposal;
- state versus a small fixed literal; or
- state versus an explicit optimizable constant slot.

If `M` different menus are specialized, they expose `M * 2^L` configurations.
They do not necessarily cover all `B^L` assignments, but they can concentrate
work around plausible and inherited neighborhoods instead of spending most
settings on arbitrary full-bank combinations.

Secant has measured approximately 7.5 million simple AST specializations per
CPU core per second, or roughly 100--130 ns per AST. This makes generating many
pair menus plausible. It does **not** make a complete new GPU evaluation free:
module loading, launch scheduling, and full trajectory rollout still have to be
amortized. Menus should therefore be packed into modules and should expose
enough masks, genomes, or trajectories per loaded module to keep the GPU busy.

### Occupancy arithmetic

With one setting per thread and 256 threads per CTA:

```text
CTAs per genome = binary masks / 256
```

Thus 4,096 masks provide 16 CTAs per genome. Several genomes or menu variants
can be in flight together to occupy a large GPU.

For the measured thread-owned LM shape, saturation occurred at roughly:

- 32,768 total fits on the RTX 4090; and
- 65,536 total fits on the RTX 5090.

With 12 binary leaves and four starts, one candidate supplies 16,384 fits.
Roughly two concurrent candidates cover the RTX 4090 saturation point and four
cover the RTX 5090 point. The useful quantity is:

```text
promoted candidates * masks per candidate * starts per mask
```

Thousands of starts for one binding are unnecessary when masks and candidate
batches already provide the independent work.

## Constant designs

State selection and continuous parameter fitting should not automatically be
the same mechanism. There are three credible designs.

### Design A: binary state choices plus explicit constant AST nodes

Dynamic state leaves choose only between two states. Optimizable constants are
explicit indexed constant instructions in the AST:

```text
state_switch(A, B, bit_i)
dynamic_constant(j)
```

Ordinary AST operators express how a constant modifies a state:

```text
dynamic_constant(0) * state_switch(X, S1, bit_0)
state_switch(S1, S2, bit_1) + dynamic_constant(1)
```

Advantages:

- the discrete and continuous search variables have clear meanings;
- LM differentiates only explicit continuous constants;
- state leaves do not need to masquerade as parameters;
- allowed state pairs can differ by missing equation or leaf role;
- repeated references to constant `j` naturally share one LM variable; and
- the kernel may avoid the general state-plus-constant shared-memory bank.

The GP can still mutate an explicit constant node into a state-switch node, or
change the two specialized states, between generations.

This is the recommended first binary implementation.

### Design B: one binary side may be a constant

A leaf can specialize to pairs such as:

```text
(state_X, constant_j)
(state_S1, literal_0)
(literal_1, literal_minus_1)
```

This retains one bit per leaf and is more flexible than state-versus-state
alone. A constant side may be:

- an explicit small literal such as `0`, `1`, `-1`, or `2` during SSE search;
- an incumbent-plus-random-jitter value; or
- an LM parameter slot during a promoted fit.

The drawback is that the semantic role of a leaf changes with its bit. LM must
predicate the derivative for a constant that is active in only some masks, and
the expression may become discontinuous across the discrete setting space.
This is workable but less clean than an explicit constant node.

### Design C: state pair plus a separate constant-replacement bit

Use one bit to choose between states and another bit to replace that result
with a constant:

```text
selected_state = state_bit ? state_B : state_A
leaf = constant_bit ? constant_j : selected_state
```

This supplies four logical configurations per leaf, but it costs two bits per
leaf. A 4,096-setting exhaustive mask then covers six such leaves rather than
12. It also adds another runtime select and more complicated LM partials.

This design should be tested only if experiments show that state-versus-
constant switching inside one structural leaf is important. Otherwise explicit
constant nodes provide a cleaner search grammar.

### Orthogonal constant banks

Discrete masks and continuous/coarse constants can use separate launch axes:

```text
blockIdx.x or an outer fit index -> constant bank or start
thread/setting bits              -> binary leaf mask
```

Total configurations then become:

```text
constant banks * binary masks * starts
```

For SSE, constant banks could contain small literals or deterministic
incumbent-centered samples. Per-AST scale and offset can be specialized while
a counter-based generator supplies normalized samples. For LM, each binary
mask can have one or a few starts and LM performs the continuous refinement.

This avoids using arbitrary state/constant bindings merely to create enough
parallel work.

## Expected kernel effects

### SSE

The binary state design can remove or reduce:

- materialized binding arrays;
- hashed arbitrary binding generation;
- full-range modulo operations;
- per-leaf shared-memory indexed loads;
- the need to expose all constants as interchangeable bank entries; and
- some shared-memory footprint and address registers.

The evolving RK4 state already exists in registers. A specialized pair lets a
leaf select directly from two known values. This should lower the cost of a
dynamic leaf, but the complete trajectory integration and scoring may still
dominate. The speedup cannot be inferred from leaf instruction counts alone.

### LM

For LM, explicit constant slots are especially attractive. The discrete state
mask is fixed during one fit, while LM differentiates only the active explicit
constants. Compared with a fully arbitrary bank, this may reduce:

- binding data and dynamic bank loads;
- generalized local-partial machinery;
- derivative live ranges;
- shared memory for the per-thread bank; and
- register-pressure fallback frequency.

LM will still retain expensive per-thread state: sensitivities, `J^T J`,
`J^T r`, damping, solve scratch, proposal constants, and trajectory state.
Binary bindings therefore cannot be assumed to close the entire gap between
the generalized and hand-written native LM kernels.

## Measured speed of the current designs

These measurements come from different controlled workloads. They establish
orders of magnitude, but they are **not one apples-to-apples SSE-versus-LM
benchmark**. Dataset size, trajectory count, RK4 steps, AST shape, and LM
iteration count differ.

### Raw setting/fit throughput

| Path | GPU | Measured throughput | Meaning |
| --- | --- | ---: | --- |
| Generic explicit-ODE SSE, common cases | RTX 4090 | 57.36--91.21 million settings/s | One complete forward score per setting in the isolated evaluator |
| Heavier case from the same SSE sweep | RTX 4090 | 7.906 million settings/s | Demonstrates strong dependence on system/trajectory shape |
| Specialized postorder LM | RTX 4090 | 61,998 fits/s | Saturated dense fed-batch fit with dynamic binding settings |
| Hand-written native LM control | RTX 4090 | 95,618 fits/s | Same measured LM problem with fixed native primal/partials |
| Specialized postorder LM | RTX 5090 | 51,813 fits/s | Compatibility measurement of the same dense shape |
| Hand-written native LM control | RTX 5090 | 92,312 fits/s | Native control on the same RTX 5090 workload |

Using the common SSE range, the current engine processes roughly 620--990
times as many SSE settings per second as native LM fits, or roughly
1,100--1,760 times as many as the generalized specialized LM fits on the RTX
5090. This ratio is expected: one LM fit contains multiple full rollouts and a
matrix optimization loop.

### Hashed versus materialized SSE search

A matched 100-generation RTX 4090 control used the same seed, population of
1,024, and 512 settings per genome:

| Current SSE settings mode | Effective configurations/s | Wall time |
| --- | ---: | ---: |
| Repaired hashed incumbent | 13.52 million/s | 3.877 s |
| Materialized | 5.81 million/s | 9.030 s |

Both controls passed materialized winner replay. The hashed path was 2.33 times
faster for this workload. This comparison shows that removing settings storage
and host construction matters. It does not isolate arbitrary shared-memory leaf
loads, which both semantic designs still require.

### Status of the binary design

There is no measured binary-setting throughput yet. It must not be assigned
the current static-kernel rate or the native LM rate by assumption. Its
potential benefits are fewer runtime setting operations and cheaper data
movement; its risks are reduced useful coverage and additional specialization/
module turnover.

## Search-policy options

### SSE screen followed by LM

1. Enumerate all binary masks for many specialized menus.
2. Score them with the fast SSE rollout kernel.
3. Retain winners in horizon/complexity archives.
4. Promote selected structures and masks to LM.
5. Optimize explicit constants with one to four starts.
6. Rewrite the winning mask and constants into the incumbent genome.

This exploits the roughly three-order-of-magnitude settings-rate difference
between SSE and LM. It is the safest initial policy.

### LM for every binary mask

For small binary spaces, run LM on every mask. This gives the GP the most
faithful ranking of each structural/binding choice after continuous constants
have been optimized. It may make better evolutionary choices than SSE around
poor initial constants, but it spends tens of thousands rather than tens of
millions of settings per second.

This policy is plausible when:

- the number of masks is small;
- enough genomes are batched to saturate the GPU;
- constants strongly determine whether a structure looks useful; and
- fewer GP generations compensate for the greater cost per setting.

It should be judged by time to recovery, not fits per second alone.

### SSE-only binary search

SSE-only maximizes raw specialized rollout throughput. It is attractive for
coarse discrete discovery and fixed-literal searches, but it may discard a
correct structure whose constants begin far from useful values. It should not
be the only policy for constant-sensitive models unless constant banks or an
equivalent refinement stage are strong enough.

## Recommended first implementation

Add binary settings as a new kernel/settings mode. Do not reinterpret or remove
the current arbitrary-binding modes until a matched quality comparison exists.

The first version should use:

1. one specialized state-versus-state pair per dynamic state leaf;
2. explicit indexed dynamic-constant AST instructions;
3. the setting index itself as a packed binary mask;
4. one thread per SSE setting or LM `(mask, start)` fit;
5. several independently specialized menus for each promising AST;
6. packed genomes/modules so module loads remain amortized;
7. SSE screening followed by LM on selected masks;
8. one to four LM starts, with candidate batching used for occupancy; and
9. final static/materialized replay of every reported winner.

Small literal choices can initially be represented as ordinary literal AST
nodes. A state-versus-constant leaf or separate replacement mask should be
added only if search evidence shows that explicit constants plus GP mutation
cannot express the useful neighborhood efficiently.

## Required benchmark

The comparison must hold ASTs, data, loss, integration steps, and hardware
constant. It should test:

| Variant | Discrete leaf mechanism | Constants |
| --- | --- | --- |
| Current materialized | Arbitrary bank index arrays | Materialized perturbations |
| Current hashed | Arbitrary bank index regenerated in kernel | Hashed incumbent perturbations |
| Binary | Specialized two-state menu and mask bits | Explicit slots plus matched perturbations |
| Binary + LM | Specialized two-state menu and mask bits | Explicit slots optimized by LM |

For each variant report:

- raw SSE settings/s;
- complete LM fits/s and mean LM iterations;
- useful configurations evaluated per loaded module;
- specialization time and ASTs specialized per second;
- module count, load time, launch time, and unload time;
- registers, local memory/spills, shared memory, and achieved occupancy;
- invalid trajectory rate;
- unique leaf assignments actually visited;
- best training and held-out loss versus wall-clock time;
- exact/structural recovery rate across seeds; and
- final materialized replay agreement.

Use both equal-evaluation and equal-wall-clock comparisons. An equal-setting
comparison favors the more expressive arbitrary mode; an equal-time comparison
tests whether cheap specialization and faster binary evaluation compensate for
that restriction.

## Questions an implementation decision must answer

1. How much faster is direct two-register selection than the current arbitrary
   shared-bank load in the complete RK4 kernel?
2. Does the binary design reduce LM register count or increase fits/s
   materially, or do sensitivities and the solve dominate?
3. How many different pair menus are needed before binary search matches the
   arbitrary path's recovery rate?
4. Should menu pairs be random, inherited, unit-constrained, residual-informed,
   or a mixture?
5. Is an explicit constant node sufficient, or does state-versus-constant at
   the same leaf materially improve search?
6. Is SSE ranking reliable enough to select LM promotions, especially for
   oscillatory, sensitive, or chaotic trajectories?
7. Does recompiling more binary menus increase module-loading pressure enough
   to erase the execution gain?
8. Should one module contain repeated versions of the same AST with different
   menus, different ASTs, or both?
9. At what state count, constant count, and number of blinded equations should
   the engine switch to a different LM ownership topology?

## Provisional recommendation

Binary state menus with explicit constant slots are the strongest next
experiment. They align the runtime representation with the actual optimization
problem: discrete state choice is handled by compact masks, continuous
parameters are handled explicitly, and fast specialization supplies many
different local choice menus.

They should not replace arbitrary bindings merely because they appear cheaper
at the instruction level. The success criterion is that the increased rate of
specialized, constrained neighborhoods produces equal or better recovery per
wall-clock second after module management and LM are included.
