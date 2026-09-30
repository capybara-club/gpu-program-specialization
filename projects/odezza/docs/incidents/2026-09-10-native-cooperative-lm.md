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

# Native cooperative LM integration and correctness repairs

Date: 2026-09-10. Owner: native LM pipeline. Changes remain in isolated validation
builds on rack1, rohini and ada; the one-hour campaign used its frozen source.

## Intended path

C99 generation, NVRTC, physical inspection and SASS specialization for 1/2/4/8
lanes per fit. A caller can request an exact width or permit wider resource
fallback. The same two execution streams, event fences, caller data and output
indexing apply to all shapes. No ordinary-CUDA, CPU or curvature substitution.

## Findings and disposition

1. **Incomplete initial cooperative port.** The first local two-lane build used
   distributed sensitivities with scalar optimizer indexing. It failed the
   numerical gate and was never used for search. The final body distributes the
   sensitivities, gradient, packed Gram matrix, triangular factor and solve.
   Initial experimental results are invalid.

2. **Coupled-stage sensitivity ordering in the existing native generator.**
   Derivative rows advanced their sensitivity state before later rows finished
   reading the same RK4 stage. Primal trajectories and replayed winner MSEs remain
   valid; the Jacobian, optimizer path and historical fit-rate comparisons are
   non-equivalent. All stage derivatives are now computed before any sensitivity
   row advances. A C99 regression compares the first GPU LM proposal against an
   independent double-precision CPU finite-difference Jacobian and linear solve.
   On the coupled three-state fixture, proposal errors are below 2e-7 in the
   retained cross-host checks. Older frozen campaigns are not modified or claimed
   to have used this repair. Rebaseline LM/curvature tuning before promotion.

3. **Compiler duplication of physical sites.** NVRTC unrolled the four-stage RK4
   loop in small four/eight-lane templates, producing four copies of each marker.
   Inspection rejected those builds. An explicit `#pragma unroll 1` now gives one
   unique physical copy. The strict uniqueness check remains enabled.

4. **Compiler-inserted predicate preservation inside a patch site.** On Ada's
   eight-state/eight-parameter four-lane template, PTXAS saved a live lane predicate
   with `P2R`, used that predicate for inline-asm scaffolding, then restored it with
   `ISETP`. Those instructions lay inside the replaced region. Dropping them made
   one coefficient receive zero sensitivity for permutations 2 and 3. A fixture
   stalled at MSE 7.20698e-6 while seven other coefficients converged. The local
   first-stage diagnostic measured zero instead of 4.2 for the missing derivative.
   The inspector now recognizes and validates the save/restore pair. The
   specializer reserves one additional register, moves the save/restore around
   the AST patch and accounts for both instructions in patch capacity. A private
   register is necessary because PTXAS may reuse its original save register as a
   later output. Unrecognized save/restore encodings return FORMAT. The repaired
   Ada fixture passes all four toggle optima and CPU replay.

5. **Incoming register lifetime.** A read/write operand's source register is no
   longer automatically recycled after materialization: the compiler can retain
   another live value there. Only known asm-local scratch is reused. Disabling
   register recycling alone did not repair the Ada predicate defect; these are
   separate changes. Additional register demand remains visible in reports.

## Validation and comparability

The strict C99 build, public export check, coupled first-proposal test, numerical
recovery and eight injected CUDA load/launch/event failure cases passed on all
three hosts at widths 1/2/4/8. The shape suite exercises 3x3, 3x6, 6x6, 8x6 and
8x8 templates with 17 starts, four toggle permutations and two ragged trajectories.
Every returned row is replayed independently. Expected register rejections are
recorded rather than discarded. Wider fallback recovers the 6x6 and 8x6 cases at
two lanes and 8x8 at four lanes. Invalid ASTs do not trigger a shape retry.

Linker injection additionally exercises a specialization-time resource failure
inside a multi-system batch: one system advances to two lanes, later systems use
one, outputs retain their layout, and exactly two streams remain owned. This
fault injection tests control flow; the larger-shape fallbacks above arise from
real compiler resource limits.

Evidence is retained under
[`scratch/multilane_core_validation`](../../scratch/multilane_core_validation/),
with source hashes, per-host logs and complete coefficient/MSE/counter arrays.
These small correctness fixtures are not saturation benchmarks. Wider fallback
adds cold template preparation and may alter occupancy and kernel time. Do not
promote a width as universally fastest from these tests.

Final integration checks add 168 CPU replays through the native search adapter
at all four widths on all three hosts. Full per-fit arrays agree exactly across
available widths and fallback for the retained shape fixtures. Ada CUDA memcheck
reports zero errors. The 24-entry service creation preflight passes on each host.
