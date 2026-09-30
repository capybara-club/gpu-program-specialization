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

# Adapting the technology to another domain

## First decide whether specialization addresses your bottleneck

Measure time spent preparing programs, compiling, loading modules, transferring
inputs, executing, reducing results and waiting between stages. Determine how
often a structure changes and how many evaluations reuse it. Compare an ordinary
compiled implementation and a sensible interpreter under the same numerical and
output contract.

A useful cost model is:

`time ≈ template setup + program preparation/specialization + module overhead + execution + results`.

Some stages overlap. Do not add overlapping profiler durations and call the sum
wall time. A cold-template result and a warm-cache result answer different questions.

## A practical implementation sequence

1. **Define the mathematical contract.** Specify precision, exceptional values,
   supported operations, protected/unprotected division, score normalization,
   and tolerances. Write a small CPU reference that a new backend can be tested
   against. For an ODE, specify the solver and stage semantics as well as the RHS.
2. **Define a bounded representation.** Give operations explicit arity, typed
   leaves, constant slots and maximum sizes. Keep the encoding independent of
   the strategy producing programs. Do not reinterpret one project's AST bytes
   as another project's format.
3. **Find the stable scaffold.** Keep data loading, looping, reductions and
   output layout stable. Make inputs/outputs and register lifetimes at the
   changing region explicit. Begin with one supported shape.
4. **Choose the least complex backend that helps.** PTX injection is easier to
   adapt than native encoding. Native specialization is worth considering when
   repeated PTX compilation remains a measured bottleneck.
5. **Validate before scaling.** Cover every operation, boundary-size programs,
   coupled and independent toggles, constants, NaNs/infinities, malformed inputs,
   repeated launches and resource cleanup after errors. Validate gradients or
   sensitivities separately from forward scores when fitting.
6. **Increase work per module.** Pack compatible programs, sweep coefficient
   vectors, or use leaf toggles. Record actual distinct programs and evaluations;
   padding and redundant bit assignments are not additional scientific coverage.
7. **Tune the measured system.** Sweep program capacity, pack size, configurations,
   streams/modules in flight, block/warp shape, and register use. Oversubscribe
   enough blocks to observe sustained behavior rather than one underfilled launch.
8. **Keep policy separate.** Add grammar, LLM, optimizer or evolutionary control
   outside the core once execution is correct and measured.

## Prompt for an LLM integrating this repository

> Read README.md, docs/ARCHITECTURE.md, docs/BENCHMARKS.md and the relevant public
> header before editing. My domain is [domain]; each candidate is [program];
> stable work is [loop/scaffold]; each structure is reused [count] times; inputs
> and required outputs are [contract]. Identify which components apply, and
> distinguish measured results from proposed applications. Implement a minimal
> CPU-checked example with a declared GPU/toolchain and a complete timing
> breakdown. Keep domain/search policy outside the native specializer. If the
> program does not fit a supported shape or opcode set, report that explicitly.
> Do not copy the symbolic-regression search system unless the integration
> actually needs it. Do not silently substitute an interpreter or reduced workload
> and report the result as equivalent native-specialization throughput.

## Concrete transfer examples

| Domain | Stable scaffold | Changing program | Natural fused result |
|---|---|---|---|
| Model-structure comparison | Time stepping and observation loss | One or more RHS components | Per-model trajectory score |
| Feature discovery | Row traversal and target loading | Feature expressions | SSE, moments or Gram statistics |
| Repeated likelihood evaluation | Dataset/batch traversal | Parametric likelihood fragment | Summed loss and possibly derivatives |
| Simulation ensembles | State update loop | Local rule or constitutive relationship | Aggregate objective and diagnostics |

These are design examples. No benchmark in this handoff establishes a general
speedup for all of these domains. Stability, identifiability, precision needs and
the amount of structure reuse can dominate the outcome.

## Handoff usability validation still to do

The reading order and category map are designed for a human or LLM arriving
without this project's conversation history. That design has not yet been
validated by an independent integration exercise. A useful next test is to give
a new reader only this repository and a small new-domain scoring task, then
check whether they can select the right API, implement CPU parity, identify the
GPU/toolchain requirements and report accurate timing boundaries. Record missing
information and wrong assumptions before calling the handoff self-sufficient.
