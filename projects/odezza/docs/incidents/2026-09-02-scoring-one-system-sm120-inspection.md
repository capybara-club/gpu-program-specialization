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

# One-system SM120 scoring-template inspection failure

Date: 2026-09-02  
Status: resolved for the reproduced eight-state/eight-constant one-system shape, 2026-09-05  
Affected path: scoring pipeline with `system_capacity = 1` on SM120

## Observation

An eight-state scoring correctness run on the RTX 5090 failed during pipeline creation with `ODEZZA_ERROR_FORMAT` and the diagnostic `compiled CUBIN inspection measurement failed`. The identical source harness and one-system shape passed on the RTX 4090. On the RTX 5090, otherwise identical packed templates with capacities of eight and 128 systems compiled, inspected, specialized, executed, and reproduced the CPU reference at `8.64e-16` MSE.

## Impact

The requested packed-system throughput measurements are valid and do not use the failing shape. A caller requesting a one-system SM120 template cannot currently create the pipeline. This is an architecture- and shape-specific template/inspection defect, not evidence of a mathematical scoring error.

## Required follow-up

Inspect the SM120 one-system scaffold and compare its final fallthrough branch, arena boundary, and marker layout with both the passing SM89 one-system CUBIN and passing SM120 multi-system CUBIN. Add a direct SM120 regression once the physical difference is identified.

## Resolution, 2026-09-05

The baseline still fails creation for eight state slots, eight constant slots, one system, and 64/128 shared/system patch slots. Instrumenting the inspector identified a permutation/output register overlap. The template consumed the permutation input near the start of the inline PTX and did not keep it live through output materialization, allowing the compiler to reuse its register. Injected toggle instructions need that register throughout the specialization arena.

The C99 generator now includes the permutation in the final keepalive chain. The inspector's overlap rejection remains intact. `scoring_cuda_eight_state_one_system` creates and runs the requested one-system shape directly, checks all scores against an analytic linear ODE, and exercises toggles across all eight RHS outputs. It passes on Rohini (SM120) and Ada (SM89). The corresponding packed eight-system regression also passes. No larger-capacity workaround is used.

This resolves the reproduced incident, not every possible capacity/compiler combination. Other historical state-capacity holes have not been reclassified. See [the scoring review](../scoring_kernel_review_2026-09-05.md) for measurements and remaining work.
