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

# LM read/write input-materialization defect

Date: 2026-09-02  
Status: resolved and directly validated on RTX 5090 and RTX 4090  
Corrected generator ABI: 4  
Corrected marker ABI: 8 (`odezza-lm-replicated-sites-v8`)

## Summary

The first cooperative specialized kernels produced width-dependent initial MSE values and incorrect LM proposals even though an independently compiled CUDA control produced identical results at every subgroup width. The defect was in the physical inline-assembly handoff, not in the cooperative sensitivity or solve algorithm.

Each streamed LM site exposes its state and constant inputs as read/write (`+f`) inline-assembly operands. PTXAS may place an operand's incoming value in one physical register and its post-assembly value in another. The marker instruction is then the required identity materialization from the source register to the destination register.

The previous experimental branch entered the specialized patch before those materializations. The patched AST read the incoming source registers, but CUDA considered the read/write values to live in the destination registers after the inline assembly. Because the same C inputs feed subsequent derivative sites, those sites could consume stale destination registers. Register allocation made the symptom subgroup-width and architecture dependent.

## Repair

The generated scaffold now has one unambiguous breakpoint after all input markers and before the output markers. Inspection records, for every input:

- Marker instruction offset.
- Incoming physical source register.
- Post-materialization physical destination register.
- Original dependency wait mask.

Specialization replaces every input marker with an explicit identity move carrying the original wait mask, branches from the post-input breakpoint into the AST patch, and makes the AST consume the destination registers. The input source registers become eligible scratch only after these moves. Output expressions write directly to the CUDA-owned output registers.

The specializer never guesses around physical aliasing. A 16-state one-thread template currently produces aliased output registers on both target architectures and is rejected explicitly.

## Impact and superseded evidence

All preliminary performance and optimizer results from the incorrect cooperative specialized kernels are invalid. The standalone fixed CUDA topology measurements were not affected. Earlier committed one-thread/grouped LM ABIs used a different handoff and are not silently treated as ABI 8 templates.

## Validation

- The local suite has 40 passing tests, including a regression that verifies every inspected input materialization is reinstated with its source, destination, and wait mask.
- RTX 5090 and RTX 4090 8x8 templates with one, four, eight, and 32 threads per fit agree on the recovered parameters, iteration counts, accepted steps, factorization attempts, GPU MSE, and independent CPU replay.
- Four- and eight-thread 16x8 templates on RTX 5090 and the four-thread 16x8 template on RTX 4090 recover the expected constants and pass independent CPU replay.
- Specialized 8x8 kernels report no stack traffic at four, eight, or 32 threads per fit on either GPU.

Performance and resource results are recorded in [`../../benchmarks/2026-09-02-cooperative-lm-cuda-shape.md`](../../benchmarks/2026-09-02-cooperative-lm-cuda-shape.md).
