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

# Running questions and concerns

This directory holds active design questions that should survive individual
experiments and conversations. Each note states the current recommendation,
the uncertainty that remains, and the measurement that would resolve it.

## Active notes

- [Cloud GPU and CPU choices](cloud_compute.md)
- [CUDA module-load concurrency](module_load_concurrency.md)
- [Lambda GH200 Arm benchmark](gh200_arm_benchmark_2026-08-25.md)
- [NVIDIA Arm and CUDA module loading](nvidia_arm_module_loading.md)
- [ODE template compilation pipeline](ode_compilation_pipeline.md)
- [Secant System ID hardware, memory, settings, and job queue](secant_system_id_hardware_memory_and_queue.md)
- [Strategic buyer landscape for system identification](strategic_buyer_landscape.md)

## Related established documents

- [Module loading and sustained throughput](../module_loading_and_throughput.md)
- [Secant specialization technique](../secant_technique.md)
- [High-value system-identification target](../system_identification_high_value_target.md)
- [System-identification product scope](../system_identification_product_scope.md)
- [CUBIN specialization domains](../cubin_specialization_domains.md)

Move a note out of this directory once the question has a measured answer and
has become a stable part of the design.
