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

# Research coverage and scope

## Included research

The [project map](PROJECTS.md) describes the collection's 15 components and
seven Odezza trial directories under `projects/odezza/research/`. They cover
program representation, PTX injection, native specialization, fused evaluation,
coefficient fitting and execution-pipeline measurements.

MM Kermac provides a secondary application of Stack PTX and PTX Inject outside
the main symbolic-regression and ODE engines. cuSR, Secant-SINDy and Secant
Benchmark preserve earlier implementations and numerical reference paths.

The collection contains source, small input specifications and explanatory
reports. Original per-file provenance is retained in the source manifest.
The [validation record](VALIDATION.md) distinguishes checks run against this
snapshot from historical benchmark results.

## Research paths and where they live

| Research question | Preserved path | Boundary / lesson |
|---|---|---|
| Can a compact mutable program become GPU code? | Stack PTX, AST PTX and C-based emitters | Representation and math are separate; GPU interpretation is not required |
| Can generated math be inserted into a stable kernel? | PTX Inject and the C/C++ injection examples in MM PTX | Explicit register bindings; PTX compilation still remains |
| Can compiler/linker overhead be amortized? | C compiler benchmark, NNG workers, CUBIN Function Patch | Parallel compilation and function patching address different costs |
| Can native specialization avoid compiling each AST? | cuSR → Secant → Secant System ID/Odezza | Architecture-specific inspection, encoding and resource contracts |
| Can leaves/coefficients vary without a new module? | Archived settings kernels, current toggles, Philox and coefficient banks | Count structures, assignments and duplicate work separately |
| Can computation be fused with statistics and solves? | Secant, FusedSINDy, Secant-SINDy | Avoid full feature materialization; reuse moments/Grams across parameter sweeps |
| Can the PTX tools serve a different domain? | MM Kermac's compiler/injection path | Specialize kernel-matrix computations; PTX compilation still remains |
| How do batching, loading and occupancy affect end-to-end rates? | Engine pipelines, benchmarks and cooperative/plain-CUDA LM controls | A fast inner loop is insufficient if launch/module work underfills the GPU |
| How should iterative coefficient fitting be laid out? | Native Odezza LM, Secant-SR refinement/LM and CUDA controls | Distinct implementations; register pressure, communication and fit quality matter |
| How can inputs/ASTs/results stay compact? | C99 frontend, AST enumeration/serialization tools, reduction and retention | Prepare once; preserve stable identities through chunking and top-k |
| Does a different ODE solver or precision help? | CPU/GPU Rosenbrock trials and FP32/FP64 study | Standalone experiments; production request scoring remains fixed-step RK4 |
| Can observed trajectories guide state selection? | RFM/AGOP dependency pilot and Kermac-related code | Sampling bias is a possible application; dependency scores are not proofs |
| Do fast evaluations produce fast discovery? | Secant-SR/reference consumers and retained timing/postmortem reports | Search coverage, identifiability, fitting and verification remain separate costs |
| How do other backends compare? | Secant/standalone benchmark sources, cuSR AMD and HIP archives | Archived experiments are not claims of a currently supported AMD engine |

The [project map](PROJECTS.md) links implementations. The retained Markdown
reports contain measured findings and caveats; omitted raw data is not silently
recreated or presented as if available.

## Deliberately outside this handoff

- Full MAP-Elites/neuroevolution applications (`mm-elites-*`, `mm-nle`,
  `mm-machines`, `mm-neuro`, and feature-application variants). Their role is
  adjacent search/learning policy; the central compilation and execution
  mechanisms are represented by the included toolkit and compiler workers.
- Every older kernel-learning library/binding, visualization/UI application,
  toy CUDA setup demo or dataset loader. MM Kermac and the RFM pilot preserve
  relevant kernel/feature-metric ideas without claiming to archive all that work.
- Every individual RHS-recovery controller, fitted equation, generated benchmark
  campaign and external package installation. The public generators/consumers
  and selected methodology reports are the transferable part.
- Remote-only files, every historical branch, deleted versions and private
  external benchmark corpora. The scope is the checked-out local sources
  inspected for this handoff, not a forensic backup of every machine.
- Third-party learned proposal tools such as OdeFormer, their model weights and
  installed environments. Those are adjacent ways to propose candidate models,
  not implementations of this repository's execution techniques.

Accordingly, the defensible claim is **coverage of the identified execution
lineage and its principal numerical/measurement branches**, rather than a
guarantee that every experiment ever performed is present.

## Reading and reproducibility notes

The archived trial READMEs retain their original `scratch/...` command examples.
Their source now lives under `projects/odezza/research/...`; relative sibling
layout is preserved. Use the portable commands in [BUILDING.md](BUILDING.md).
Reports that audit external benchmark data still need that separately acquired
data. Some historical drivers reference omitted experiments or former machine
environments; those paths are evidence, not automatic deployment instructions.

The sources receive the MIT/third-party audit described in the validation record.
Only the specifically listed CPU checks were rerun; inclusion does not certify
GPU execution or legacy integration against today's API.
