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

# Packed trajectory kernel topology

## Work mapping

The packed score kernel uses three independent coordinates:

| Coordinate | Owner | Meaning |
|---|---|---|
| setting tile | `blockIdx.x` | `blockDim.x` settings, one per thread |
| genome group | `blockIdx.y` | a consecutive group of packed system genomes |
| genome within group | CTA-local loop | each CTA evaluates up to `genomes_per_cta` genomes in turn |

For the initial benchmark, 2,048 settings, 128 threads, eight genomes, and
eight genomes per CTA produce a `16 x 1` grid. All 16 CTAs therefore execute
the same two outer logical loops: eight genomes and, within each genome, all
three trajectories. RK4 observations, steps, and stages remain inside those
loops.

Each branch-table entry denotes one **system genome**, not one missing site. A
system genome contains one independent post-order AST for every missing
mathematical site and writes all missing outputs before returning to the common
RK4 continuation. This preserves the grouping required by GP crossover and
mutation.

The generated PTX contains one `brx.idx.uni` site and one native target table.
The exact four-instruction lowering is PTXAS-version dependent rather than an
architecture invariant: retained artifacts include both general-register
`LDC + BRX` and uniform-register `ULDC`/`LDCU + BRXU` forms, and the current
`sm_89` artifact uses the latter. Inspection therefore records and restores the
compiled dispatch instructions verbatim. The Python specializer packs
variable-length multi-output genome bodies into one
contiguous SASS arena and rewrites each 32-bit target-table entry in the final
CUBIN. The RK4 body is not duplicated for each genome.

Settings and bindings are genome-major:

```text
settings[(genome * constant_count + constant) * settings_ld + setting]
bindings[(genome * input_count    + leaf)     * bindings_ld + setting]
scores[genome * num_settings + setting]
```

No atomics are needed because every `(genome, setting)` pair has exactly one
owner.

## Experimental warp-per-system ownership

The low-setting variant assigns one complete system genome to each warp rather
than retaining a whole CTA for one system. For the current experimental shape:

| Coordinate | Owner | Meaning |
|---|---|---|
| setting tile | `blockIdx.x` | 32 settings |
| genome group | `blockIdx.y` | `genomes_per_cta` consecutive genomes |
| genome | `threadIdx.x >> 5` | one genome per active warp |
| setting | `threadIdx.x & 31` | one setting per lane |

A 256-thread CTA therefore evaluates eight independent systems over 32 settings
each. Each lane retains its own RK4 state and indexed shared-memory leaf bank;
the reference trajectory remains shared by the CTA. Search-output reduction is
warp-local via shuffles, followed by the existing per-genome reduction kernel.

The initial implementation deliberately accepts at most 32 settings. It also
requires a whole number of warps per CTA and `genomes_per_cta <= blockDim.x / 32`.
The C99 GP path does not select this topology yet; it is exposed through the
unique-module benchmark while its throughput and crossover are measured.

On `sm_120`, PTXAS lowers the warp-uniform branch-table dispatch to five SASS
instructions rather than the four used by the CTA-uniform form. Inspection
accepts these two validated sequences, and specialization restores every
recorded dispatch instruction verbatim before patching target offsets.

## Winner-output topology

Search mode avoids materializing the full MSE surface. The packed scoring
kernel adds one shared score and setting-index slot per thread, reduces each
setting tile inside its owning CTA, and writes one pair per
`(genome, setting_tile)`. A fixed, unspecialized CUDA reduction kernel then
emits one winning pair per genome in the same stream.

For `G` genomes, `S` settings, and `T` threads, the intermediate allocation is
`2 * G * ceil(S / T)` four-byte values and the final output is `2 * G`
four-byte values. The completion event is recorded after the second reduction,
so the eager loader can still treat scoring and result selection as one ticket.
The reduction kernel is ordinary CUDA rather than another SASS site because it
does not depend on the candidate expression.

Full-MSE mode remains available for correctness checks and analysis. Ties in
both reductions select the lowest setting index, and inactive lanes in a
partial final tile receive sentinel scores.

## Multiple packed kernels per module

A module may contain several copies of the packed kernel body. Each entry point
owns a contiguous global genome range but uses a local index for its own native
branch table. With two eight-genome kernels:

| Entry point | Global genomes | Local dispatch indices |
|---|---:|---:|
| `ssid_score_packed_0` | 0–7 | 0–7 |
| `ssid_score_packed_1` | 8–15 | 0–7 |

The CUDA body, target table, SASS arena, register count, and inspection plan are
independent for each entry point. The source and CUBIN identity are shared by
the module. Specialization patches every populated arena into the same CUBIN,
and the runtime loads that CUBIN once before resolving and launching its kernel
symbols.

For one homogeneous shape, increasing one kernel's capacity and using
`blockIdx.y` minimizes launches. Multiple entry points are useful when keeping
arena size bounded, mixing resource shapes later, or making several kernels
resident behind one module load.

## Current trajectory-data contract

The first kernel intentionally implements only the dense, aligned fast path.
Trajectories may have different initial states and different observed values,
but they must share all of the following:

- the same state vector and state order;
- the same observation count;
- the same observation times, currently represented by one fixed interval;
- the same fixed number of RK4 steps between observations; and
- a complete target for every state at every observation.

It does not yet support missing state measurements, masks, irregular time
gaps, different observation schedules per trajectory, or ragged trajectory
lengths. Those should be a separate sparse/unaligned kernel mode with explicit
time and observation metadata so the dense path does not pay for branches and
indirection it does not need.
