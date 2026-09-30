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

# Cloud GPU and CPU choices

## Current recommendation

Keep the service control plane durable and independent of the GPU workers.
Start with NVIDIA workers because Secant already supports the relevant CUBIN
architectures, then add another GPU vendor only after measured cost per
successful search justifies the backend work.

The first non-AWS worker worth benchmarking is an RTX 6000 Ada instance from a
provider with an uptime commitment. It exposes the same `sm_89` architecture as
the existing RTX 4090 path and offers roughly L40S-class ordinary FP32
throughput. AWS G6e/L40S remains the conservative fallback. Google G2/L4 is a
lower-hourly-cost but substantially slower worker.

The service should use on-demand workers for customer jobs. Spot workers are
appropriate for internal experiments or resumable jobs once generation-level
checkpointing is reliable.

## CPU shape

For one Secant GPU, begin with 8 to 16 fast x86 CPU threads and 32 to 64 GiB of
system memory. Candidate generation is already fast enough that very large
host CPUs are unlikely to help one GPU. The remaining host work is evolution,
specialization, module loading, settings construction, and checkpointing.

- The 10 host threads bundled with a Verda RTX 6000 Ada worker are a reasonable
  starting point.
- A Verda L40S worker supplies 20 host threads for a modest additional hourly
  cost if measurement shows host starvation.
- On AWS, prefer `g6e.2xlarge` with 8 vCPUs over `g6e.xlarge` with 4 vCPUs for
  a production worker. Do not pay for `g6e.4xlarge` until a profile shows that
  the extra host cores improve GPU-active time.
- For CPU-only batch work, the AWS C8a family is attractive: fifth-generation
  AMD EPYC, high clocks, x86-64, and one physical core per advertised vCPU.
- Avoid burstable instances and CPU-flex instances for sustained campaigns.
  Avoid Arm initially unless a specific Arm experiment justifies the port.

## Measurement gate

For every candidate instance, report:

1. GPU-active time divided by campaign wall time.
2. Candidate generation and specialization time.
3. Eager module-load time, exposed load time, and load time hidden by kernels.
4. Search quality and completed searches per dollar, not peak FLOPS per dollar.
5. Cold-start and capacity-acquisition latency.

If GPU activity remains above 90 to 95 percent, more attached CPU is unlikely
to be valuable. If it does not, compare a larger attached CPU allocation before
changing the compiler or search topology.

## Reliability shape

Run the MCP endpoint, authentication, durable queue, billing state, checkpoints,
and final results on the primary cloud. Treat GPU machines as replaceable
workers. A campaign must survive a client disconnect or worker loss and resume
from its last generation checkpoint on another compatible GPU.
