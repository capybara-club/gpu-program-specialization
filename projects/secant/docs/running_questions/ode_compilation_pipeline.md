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

# ODE template compilation pipeline

## Goal

When a new ODE problem contains known equations and one or more unknown
expression sites, preserve native compiler optimization for the known system
while retaining a stable Secant specialization island for rapid candidate
search.

Three compilation costs must be reported separately:

1. full CUDA source to CUBIN;
2. CUDA source to PTX;
3. PTX to CUBIN with `ptxas`.

The current benchmark targets are:

- `experiments/astaxanthin_batch_reproduction/secant_specialization_template.cu`;
- `experiments/astaxanthin_batch_reproduction/resident_lm_known_skeleton.cu`.

The first represents fused RK4 scoring with one unknown rate-law site. The
second includes the resident sensitivity, Gram, damping, and solve path and is
a deliberately fat compilation case.

## Candidate pipelines

### Full CUDA compilation per submitted system

Generate one CUDA translation unit containing all known equations, the
integration and score path, and the Secant marker island. Compile it directly
to CUBIN, inspect the marker registers, and cache it as that problem's base
module.

This gives NVVM and `ptxas` maximum freedom to optimize the known expressions
together with RK4, sensitivity propagation, scoring, and control flow. Its cost
is paid once per submitted problem, not once per candidate AST.

### Cached PTX template plus PTX injection

Compile a structural CUDA template to PTX ahead of time. For a submitted
problem, inject generated PTX for its known equations and marker contract, run
`ptxas` over the completed PTX, inspect the resulting CUBIN, and cache it.

This skips the CUDA/NVVM front end for each new problem but still lets `ptxas`
schedule the complete kernel, assign registers, set yield and stall control,
and optimize machine instructions around the final structure. It is more
delicate because PTX injection must preserve types, address spaces, SSA-like
temporary naming, control-flow labels, calling conventions, and the marker
island's liveness contract.

## Current recommendation

Advance the cached-PTX/injection design as the intended service path, while
retaining full CUDA-to-CUBIN compilation as its correctness oracle and
fallback. The measured split path is materially faster for a new ODE and
produces the same scheduled machine code as direct compilation.

The proposed flow is:

1. Select a precompiled PTX template by topology: target architecture, state
   count, observation layout, integrator, precision, and LM/sensitivity width.
2. Generate typed PTX fragments for the submitted known laws and their required
   derivatives. Keep unknown laws as validated Secant marker/skeleton sites.
3. Insert those fragments with the existing PTX Inject infrastructure, then
   compile the complete PTX in-process with `nvPTXCompiler`.
4. Inspect and validate the marker register contract in the base CUBIN, cache
   that CUBIN by problem hash, and use direct Secant SASS specialization for the
   high-volume candidate expressions.

If a submitted problem requires a topology for which no PTX template exists,
compile the CUDA template to PTX once and add it to the topology cache. If PTX
generation or validation fails, compile the complete generated CUDA source as
the safe fallback.

Regardless of which front end is used, always run `ptxas` after the known
system has been inserted. Do not patch known equations into an already
assembled CUBIN: doing so prevents `ptxas` from assigning registers and control
codes with visibility into those equations. Reserve direct CUBIN/SASS
specialization for the high-volume unknown candidate expressions.

This distinction is important: `ptxas` performs final register allocation,
instruction scheduling, and yield/stall control, but it does not replace every
high-level NVVM optimization. The PTX expression generator must preserve a good
DAG, common-subexpression reuse, and fused primal/derivative computation. A
hand-generated known-law fragment still needs runtime, resource, and SASS
comparison against the full-CUDA oracle before the injection path is declared
equivalent.

## Benchmark results

Measured 2026-08-24 with CUDA 13.1, optimization level 3, fast math, and line
information enabled. Each reported value is the median of 11 semantically
distinct source inputs after two warmups. A live ODE constant changed for every
sample so the results represent new submitted systems rather than exact-source
compiler-cache hits.

`nvPTXCompiler` is the in-process PTX-to-CUBIN interface already used by PTX
Inject. The standalone `ptxas` column includes process launch; the in-process
column is the relevant service measurement.

| Host / target | Kernel shape | CUDA to CUBIN | CUDA to PTX | Standalone `ptxas` | In-process PTX to CUBIN | Split total, in-process | Direct / split |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rohini / `sm_120` | compact fused RK4 score | 29.57 ms | 5.80 ms | 13.02 ms | 12.64 ms | 18.43 ms | 1.60x |
| Ada / `sm_89` | compact fused RK4 score | 51.68 ms | 12.11 ms | 23.96 ms | 15.80 ms | 27.91 ms | 1.85x |
| Rohini / `sm_120` | resident sensitivity + Gram + LM solve | 104.53 ms | 7.66 ms | 36.60 ms | 35.33 ms | 43.00 ms | 2.43x |
| Ada / `sm_89` | resident sensitivity + Gram + LM solve | 198.39 ms | 15.59 ms | 57.92 ms | 58.40 ms | 73.98 ms | 2.68x |

With a structural PTX template already cached, the CUDA-to-PTX column leaves
the per-problem critical path. Before adding the small injection and validation
costs, the full resident kernel is therefore about **35.3 ms on Rohini** and
**58.4 ms on Ada** to assemble to a base CUBIN. That is about 2.96x and 3.40x
faster, respectively, than recompiling the complete CUDA source. Candidate AST
specialization remains a separate nanosecond-scale operation after this
once-per-problem setup.

The conclusion also holds when the resident kernel uses its fully native known
rate expressions instead of marker calls. Standalone staged compilation was
39.76 ms versus 99.94 ms direct on Rohini (2.51x), and 71.42 ms versus
203.79 ms direct on Ada (2.85x).

## Code quality evidence

For both measured kernel shapes and both GPUs, disassembly of the direct CUBIN,
standalone-staged CUBIN, and in-process-staged CUBIN was byte-for-byte
identical. This includes the SASS control encodings that carry yield and stall
decisions. The resident kernel used 168 registers on `sm_120` and 167 on
`sm_89`, with zero stack and local memory and no spills in either path.

The CUBIN containers themselves need not be byte-identical because compiler
metadata can differ; the executable instruction stream and kernel resources
are the relevant equivalence checks.

Repeatedly compiling the exact same source in one NVRTC process produced much
smaller apparent times—for example about 7.7 ms for the resident direct path on
Rohini—because of compiler caching. Those warm-cache figures must not be used
as new-problem latency.

The measurements can be reproduced with:

- `experiments/astaxanthin_batch_reproduction/benchmark_compile_stages.py`;
- `experiments/astaxanthin_batch_reproduction/benchmark_nvptxcompiler.c`.

The installed `nvcc` command currently conflicts with the hosts' system math
headers, so the direct stage used NVRTC, which is also the supported service
compiler path. No headers, packages, or system configuration were changed to
work around that host-toolkit mismatch.

## Remaining proof before adopting injection

The stage split is proven; arbitrary ODE-law PTX injection is not yet proven by
this timing test. The next gate is to emit the astaxanthin known laws and their
sensitivity expressions through typed PTX Inject sites, then require:

- identical numerical output and recovered constants;
- no spills and no material register increase;
- no meaningful trajectory-scoring throughput regression;
- expected marker registers and control-flow shape;
- stable diagnostics for invalid types, labels, address spaces, and ABI use.
