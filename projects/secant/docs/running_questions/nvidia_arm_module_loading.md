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

# NVIDIA Arm and CUDA module loading

## Question

Does an NVIDIA-supported Arm host load many generated CUDA modules faster than
an x86 host, and would that improve Secant's end-to-end pipeline throughput?

## Current judgment

The controlled Lambda GH200 experiment is complete. Grace is valuable for its
64-way CPU specialization and compilation throughput, but it does not remove
CUDA's module-load serialization. Shared and separate-context loader threads
both regressed aggregate load throughput as their count increased. Keep one
context and one loader thread.

For the same 64-kernel source, the GH200 platform loaded its module about 24%
faster than Rohini, but the `sm_90` CUBIN was also 38% smaller than the
`sm_120` CUBIN. Byte-normalized load cost was slightly worse on Grace, and
Rohini won the smallest-module case. This is evidence for a useful complete
GH200 platform result, not for faster ARM ELF parsing. Full measurements are in
[Lambda GH200 Arm benchmark](gh200_arm_benchmark_2026-08-25.md).

`cuModuleLoadData` performs CUBIN parsing, validation, relocation, driver data
structure allocation, function residency, and code upload. Secant modules are
small enough that raw host-to-device transfer bandwidth should be a minor part
of the cost. The dominant work is more likely driver bookkeeping and
synchronization. Changing the host instruction set therefore has no obvious
reason to produce a large win by itself.

Grace-based systems can nevertheless differ materially from ordinary x86
hosts:

- NVIDIA controls more of the CPU, interconnect, firmware, and driver stack.
- Grace Hopper and Grace Blackwell use a coherent high-bandwidth CPU/GPU
  interconnect rather than an ordinary discrete-GPU PCIe topology.
- NUMA behavior, page management, kernel launch latency, and driver locks can
  differ even when the public CUDA API is the same.
- An Arm result can also be affected by CPU frequency, core count, Linux
  configuration, CUDA version, and GPU architecture. Those effects must not be
  attributed to Arm without controls.

The expectation is therefore: a small or negligible difference from host ISA
alone, with a possibility of a useful platform-level difference on Grace.

## Evidence already in the repositories

The earlier PTX Inject compile-scaling benchmark includes one Grace Hopper
result and provides useful context, but it does **not** measure CUDA module
loading. For the same 262,144 generated programs, its recorded aggregate
PTX-to-CUBIN throughput was:

| Host | Compiler threads | Programs/s total | Programs/s/thread |
| --- | ---: | ---: | ---: |
| GH200 Arm | 64 | 202,741 | 3,168 |
| Ryzen 9 9900X, physical cores | 12 | 67,136 | 5,595 |
| Ryzen 9 9900X, SMT | 24 | 77,520 | 3,230 |

The GH200 was about 3.0x faster in aggregate than the 12-core Ryzen because it
could run many more compiler workers, while one physical Ryzen core was about
1.77x faster than one GH200 core. The GH200 and Ryzen-SMT per-thread figures
were nearly equal. This is evidence that NVIDIA Arm can be attractive for
highly parallel *compilation*, but it is not evidence that Arm executes
`cuModuleLoadData` faster. It also reinforces the need to report both aggregate
throughput and per-core latency.

The 2026-08-24 standalone concurrency probe adds a second constraint. On
production-shaped Secant CUBINs, concurrent `cuModuleLoadData` calls did not
scale on either Rohini or Ada. Shared-context throughput regressed, while
separate contexts were flat within a few percent. A metadata-heavy CUBIN with
generated line information did parallelize across contexts, but that advantage
disappeared when line information was removed. See
[CUDA module-load concurrency](module_load_concurrency.md).

This makes a large Arm advantage less likely for the production workload. An
Arm test is still useful because the Grace driver and platform may have a
different serialization boundary, but the prior should now be "similar" rather
than "faster."

## Cheapest useful experiment

First compare an AWS Graviton/T4G instance with an x86/T4 instance using an
identical `sm_75` CUBIN corpus and CUDA version where possible. This is not a
Secant performance comparison; it is a driver-lifecycle probe that reduces the
GPU-architecture confound.

If that result is interesting, compare Grace Hopper with an x86 H100 system.
That second comparison is more relevant to NVIDIA's modern Arm platform but is
confounded by the CPU/GPU interconnect and platform integration. It answers
whether the complete Grace system helps, not whether Arm instructions help.

If access to the prior GH200 box is still available, run the module lifecycle
probe there first. It will not isolate CPU ISA because Rohini has a different
GPU and interconnect, but it can cheaply determine whether the possible effect
is large enough to justify a controlled rental.

## Required measurements

Use the same corpus shape as the eager-streaming benchmark and report:

- eager `cuModuleLoadData` latency and modules per second;
- lazy module load plus explicit `cuFuncLoad` latency;
- first-launch latency;
- module unload latency;
- cold and warm distributions, not only a mean;
- one active module versus active-plus-staging modules;
- serialized wall time and eager-overlapped pipeline wall time;
- CPU utilization, core frequency, CUDA driver/toolkit versions, CUBIN size,
  kernels per module, and ASTs per kernel.

The decision metric is end-to-end campaign time. Faster isolated module loads
have little product value if eager streaming already hides them behind GPU
execution.

## Decision rule

Do not maintain an Arm production build unless it improves compile-inclusive
campaign throughput by at least 10 percent, or it provides a separate capacity
or deployment advantage. A smaller difference is useful paper evidence about
portability but does not justify operational complexity.
