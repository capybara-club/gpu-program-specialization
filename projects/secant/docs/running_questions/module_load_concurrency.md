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

# CUDA module-load concurrency

## Question

Can several host loader threads increase Secant module throughput, either by
sharing one CUDA context or by giving every thread an independent context?

## Answer

Not for production-shaped Secant CUBINs on the tested x86 systems. The driver
allows some portions of concurrent calls to overlap, but total load throughput
is flat or lower. Multiple contexts do not provide enough additional loading
throughput to justify duplicating execution contexts, streams, events, and
dataset state.

Keep one CUDA context and one eager loader in the Secant runner. Continue to
hide that loader behind useful GPU execution. Do not implement a loader pool
unless a future driver, Arm platform, or materially different CUBIN shape
changes this result.

## Standalone benchmark

`bench/cuda_module_load_scaling_bench.c` is deliberately independent of the
Secant runner. It accepts one CUBIN, several CUBIN paths, or a directory corpus
and measures persistent loader threads in two modes:

- `shared`: every thread makes the same context current;
- `separate`: every thread owns an independent context on the same GPU.

Every wave loads one module per thread, optionally enumerates function handles
and names, waits with the modules resident, and then unloads them. The program
reports driver-load call time, enumeration time, unload time, phase wall time,
and aggregate throughput. CUDA eager loading is required and verified.

The benchmark is a standalone C11 CUDA Driver API target so the same source can
be run on x86 and Arm without bringing up an SR campaign.

## Production-shaped results

Measurements used 31 waves after three warmups, 64 functions per module,
optimization level 1, no generated line information, and eager loading. The
Rohini CUBIN came from the repository's normal template compiler and occupied
26.1 MB. The Ada `sm_89` image occupied 7.54 MB.

### Load plus function enumeration throughput

| Host / contexts | 1 loader | 2 loaders | 4 loaders | 8 loaders |
| --- | ---: | ---: | ---: | ---: |
| Rohini, shared | 198.5 modules/s | 166.0 | 165.8 | 169.6 |
| Rohini, separate | 201.2 modules/s | 195.4 | 194.3 | 211.9 |
| Ada, shared | 186.2 modules/s | 174.2 | 172.9 | 167.5 |
| Ada, separate | 186.3 modules/s | 186.2 | 178.4 | 183.2 |

Eight separate Rohini contexts were only 5.3% above its one-context result;
that small endpoint difference did not represent useful lifecycle scaling.
Ada's best concurrent loading result merely matched one loader.

### Complete load-enumerate-unload throughput

| Host / contexts | 1 loader | 2 loaders | 4 loaders | 8 loaders |
| --- | ---: | ---: | ---: | ---: |
| Rohini, shared | 116.8 modules/s | 105.8 | 105.0 | 106.5 |
| Rohini, separate | 118.2 modules/s | 116.3 | 116.3 | 120.7 |
| Ada, shared | 127.2 modules/s | 133.1 | 138.1 | 136.9 |
| Ada, separate | 127.4 modules/s | 151.2 | 150.0 | 154.2 |

Ada can unload independent contexts concurrently, which improves this
artificial load-then-unload lifecycle figure. That is not a module-loading
speedup, and it does not justify executing one Secant campaign across several
contexts. Production eager streaming already overlaps loading and avoids
making unload the primary throughput gate.

Function discovery is not the bottleneck. Retrieving all 64 handles and names
took roughly 0.001--0.004 ms per module. `cuModuleLoadData` itself took about
5.0 ms for one Rohini loader and 5.3 ms for one Ada loader in these cases.
Concurrent calls increased individual call latency nearly in proportion to the
number of loaders, which is evidence of substantial driver serialization.

## Why the first diagnostic result looked promising

A Rohini diagnostic CUBIN containing generated line information occupied
36.5 MB. For that metadata-heavy image, shared-context load throughput improved
1.25x with eight threads, and eight separate contexts improved it 3.06x.

That experiment demonstrates that some ELF/debug-metadata processing can run
in parallel. It is not representative of a production Secant module. Removing
line information reduced the image and eliminated the scaling. This is also a
reason to keep generated line metadata out of campaign CUBINs.

## Caveats and future trigger

The current sweep repeatedly loads a valid template image; the tool also
supports a directory of distinct specialized CUBINs for a future corpus test.
The single-loader latency agrees closely with the earlier Secant pipeline's
roughly 2.9 ms/module measurement for its smaller template, which argues
against a large exact-image cache artifact.

Repeat the test before changing the runner only when:

- a new driver or GPU architecture materially changes single-loader latency;
- production modules begin carrying substantially more metadata;
- module loading becomes exposed after eager overlap;
- or an Arm result shows unexpectedly different behavior.

For the Arm experiment, use the exact same production-style `sm_90` corpus on
an x86 H100 and an Arm GH200. Do not use a line-information-heavy CUBIN, because
that tests parallel debug-metadata parsing rather than the intended Secant
pipeline.
