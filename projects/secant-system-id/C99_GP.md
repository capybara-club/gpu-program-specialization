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

# Efficient C99 GP search loop

## Scope

The stage-4 implementation joins system-genome evolution to the existing C99
specialize/load/execute pipeline. In the ordinary path, Python compiles and
inspects the immutable CUDA template once and makes one `ssid_gp_run` call;
C99 then owns every timed generation. The explicitly enabled trajectory-LM
experiment returns to Python only at promotion intervals until that module
path prepares a batch of specialized LM modules. Their loading, launch,
completion tracking, and unloading then stay inside a persistent C99 queue.

One system genome contains one postorder AST for every missing equation. Its
incumbent constant bank and leaf-binding vector travel with those structures.
The two population arenas, AST slots, annotations, scratch storage, and best
archive are allocated once at GP creation. Generations reuse those fixed
buffers rather than allocating per individual or per variation operation.

## Generation boundary

Each generation performs the following work:

1. Encode the current system genomes into module-sized flat AST batches.
2. Submit those batches to the eager C99 specialization pipeline.
3. Upload only each genome's incumbent constants and leaf bindings.
4. Generate the surrounding settings deterministically inside the scoring
   kernel, score them, and reduce to one winner per genome on the GPU.
5. Download the compact winners, reconstruct only those winning settings, and
   attach them to their genomes.
6. Preserve explicit elites and produce the next fixed population arena with
   tournament selection, subtree and whole-site crossover, subtree and point
   mutation, and random immigrants.

Subtree annotations cache byte and node ranges, arity, and depth so crossover
and mutation do not repeatedly parse program structure. The best-so-far
archive owns its program bytes, constants, and bindings independently of the
two rotating population arenas.

## Compact settings mode

Materializing `population x settings x constants` and the corresponding leaf
bindings on the CPU made setting construction the bottleneck. Search templates
can therefore use an explicit `hashed_incumbent` ABI:

- setting zero is the exact incumbent;
- later settings deterministically perturb incumbent constants;
- each leaf either retains its incumbent binding or selects another permitted
  state/constant bank entry; and
- the random stream is keyed by seed, generation, global genome index, setting,
  and field index.

Only the incumbent arrays are uploaded. At population 1024 this is roughly 12
KiB per generation, instead of materializing tens of MiB of setting rows. The
host reconstructs only the winning row with the same stateless function. The
ordinary `materialized` ABI and `SSID_OUTPUT_FULL_MSE` remain available for
exhaustive score capture and debugging.

The template descriptor records its settings ABI. The GP runner supports both
layouts explicitly and rejects any other value rather than interpreting its
pointers incorrectly.

The materialized mode constructs the same settings in C, uploads the complete
setting and binding surfaces, downloads full MSE, and performs the per-genome
argmin in C. It is slower and transfers more data, but it provides a direct
correctness path independent of the fused GPU reducer. Full-MSE capture remains
selectable, as does the compact GPU-winner output.

## Experimental local-search boundary

The scored current population now exposes a narrow borrowed candidate view for
selective local search. An external optimizer may read one genome, its
incumbent constants/bindings, MSE, objective, and complexity. It may then offer
new constants and bindings for that same index. C99 accepts the replacement
only when the MSE is finite and strictly lower, all constants are finite, and
all bindings are inside the state/constant bank. The AST is never replaced
through this API, the objective is recomputed at unchanged complexity, and an
accepted member is immediately reconsidered for the owned best-so-far archive.

The experimental fed-batch command uses this boundary for trajectory LM.
Ordinary SSE-only runs do not call it.

The LM queue is created once per campaign. It owns one dense reference upload
and fixed starts, bindings, constants, MSE, iteration-count, and accepted-step
buffers sized for the maximum promotion batch. At each promotion boundary,
Python specializes the eligible smooth genomes, uploads the batch's starts and
bindings once, and submits all same-sized CUBINs. The C loader workers keep one
writable image each, and the CUDA-owner thread uses event-governed module
lifetimes. Only improved constants and bindings cross back into the C99 GP.

## Correctness checks

The implementation has CPU-only tests for deterministic multi-site generation,
winning-setting inheritance, strictly improving external promotion, rejection
of non-finite promoted constants, valid variation over 50 generations, and
the hashed source contract. On the RTX 5090, the selected score reconstructed by
the stateless kernel was replayed through the ordinary materialized-settings
kernel with exactly the same FP32 MSE. The independent CPU trajectory scorer
agreed with the final 500-generation winner to `8.9e-6` relative error.

The same replay test exposed a correctness defect in the fused `sm_89` winner
path: selected scores and setting indices did not reliably reconstruct on the
RTX 4090. The search policy therefore uses full-MSE output with materialized
settings on that architecture until the reducer is repaired. This is an
explicit correctness fallback, not a performance claim about the faulty path.

The winner path treats a non-finite accumulated SSE as `FLT_MAX` before CTA
reduction. The module-level reducer also validates every candidate setting and
falls back to setting zero when no usable tile exists. This prevents an
all-overflow genome from leaking the internal `UINT_MAX` reduction sentinel
into the GP setting materializer.

## RTX 5090 steady-state result

The following search ran on Rohini with 1024 system genomes, 512 settings per
genome, 500 generations, 128 genomes per single-kernel module, one
specialization worker, and at most two loaded modules:

| Measurement | Result |
|---|---:|
| One-time CUDA template compilation | 0.137 s |
| End-to-end search wall time | 4.596 s |
| System genomes scored | 512,000 |
| Genome/setting configurations scored | 262,144,000 |
| End-to-end genome rate | 111,412/s |
| End-to-end configuration rate | 57.04 million/s |
| Best observed MSE | 0.0028249 |
| Best generation | 407 |
| Best total node complexity | 60 |
| Score-kernel registers for the measured winner | 94 |
| Score-kernel local/stack storage | 0 / 0 bytes |
| Winner-reducer registers | 16 |

The search wall time excludes the one-time template compilation shown
separately. The timed path includes GP production, CUBIN copying and specialization, eager
module loading, GPU execution and winner reduction, compact readback, setting
inheritance, and creation of the next population. It retained about 79.8% of
the earlier 512-setting raw unique-module winner benchmark (`142,713`
genomes/s and `73.07` million configurations/s), which intentionally excluded
GP production.

The final confirmation retained 78.1% of that raw genome rate. Its wall-time
split was approximately 93.9% eager pipeline, 4.2% evolution, 1.5% winner
materialization, 0.3% downloads, and 0.2% uploads. This indicates
that the C99 GP producer is feeding the GPU/module pipeline quickly enough and
is not the dominant stage.

### Settings and eager-depth knees

At 1024 genomes and 100 generations:

| Settings/genome | Genomes/s | Configurations/s | Observation |
|---:|---:|---:|---|
| 256 | 112,468 | 28.79 million | Highest structure rate, half the settings surface |
| 512 | 109,662 | 56.15 million | Current balance |
| 1024 | 59,673 | 61.11 million | Only 9% more configurations/s, about half the structure rate |

At 512 settings, eager module depths two and three were effectively tied at
about `109,700` genomes/s; depth one reached `97,632` genomes/s. Two is
therefore the current minimum useful in-flight depth on Rohini.

## What this result does not establish

The low MSE is a pipeline and search sanity result, not yet a claim that the
blinded fed-batch laws were recovered reliably or structurally. That requires
the controlled multi-seed recovery experiment in stage 5.
