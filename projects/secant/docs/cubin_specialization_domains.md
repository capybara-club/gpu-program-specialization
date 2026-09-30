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

# Domains for CUBIN Specialization

CUBIN specialization is best suited to workloads with a stable GPU kernel for
loading, tiling, synchronization, and output, but many short-lived
straight-line programs operating on registers inside that kernel. It is most
valuable when conventional CUDA or PTX compilation would take longer than the
specialized program will execute.

## Candidate Domains

| Domain | Specialized portion | Why CUBIN specialization helps |
|---|---|---|
| GPU query engines | Filters, projections, expressions, and aggregations | Queries may execute too briefly to amortize CUDA or PTX compilation. |
| Program synthesis | Candidate arithmetic, Boolean, or bitwise programs | Millions of candidates can be compiled, tested, and discarded. |
| Cellular automata | Per-cell transition rules | Neighborhood loading remains fixed while evolved rules change. |
| Rendering and signed-distance fields | Material graphs, procedural textures, and distance functions | Interactive systems generate many related expression kernels. |
| Scientific model discovery | Constitutive laws, closures, reaction terms, and flux functions | Candidate equations run over the same resident simulation data. |
| Monte Carlo simulation | Payoffs, transition models, and scoring functions | Path generation remains fixed while each contract or model is specialized. |
| Robotics and model-predictive control | Dynamics, cost, and constraint expressions | Candidate controllers are evaluated across many parallel rollouts. |
| GPU dataframe processing | User-defined column expressions | Native expressions can be inserted into an established scan kernel. |
| Tensor epilogues | Activations, normalization, loss, and quantization | Matrix multiplication remains fixed while custom elementwise logic changes. |
| Computational geometry | Collision predicates, CSG trees, and implicit surfaces | Traversal is stable while object-specific equations vary. |
| Signal processing | Filter graphs and nonlinear transforms | Shared tiled input can feed many rapidly generated transforms. |
| Reaction-network search | Rate laws and interaction terms | Candidate kinetics can be evaluated against the same trajectories. |
| Cryptographic and Boolean search | S-boxes, hashes, mixers, and PRNG transitions | Dense integer and bitwise candidates are naturally register-local programs. |
| Agent simulations | Policies and local interaction rules | Environment loading stays fixed while many policies are tested. |
| Search heuristics | Candidate scoring and ranking functions | An LLM, GP, or RL system can rapidly test generated heuristics. |

## Especially Strong Fits

### GPU Query Expressions

A scan kernel already knows how to read columns and write selections or
aggregates. The query expression is a natural specialization island. Many
interactive queries are too short to repay seconds of PTX compilation, making
this a technically clean demonstration of low-latency native specialization.

### Cellular Automata and Local-Rule Discovery

A fixed kernel can load a tile and halo into shared memory, execute a
specialized transition rule, and write the next state. The rule can use
floating-point, integer, bitwise, or fixed-point instructions. This is a clean
way to demonstrate that Secant is broader than symbolic regression while
retaining a simple and inspectable kernel skeleton.

### Procedural Rendering and Implicit Geometry

Signed-distance functions, material graphs, and procedural textures are
expression programs. Rapid specialization could allow an editor, optimizer,
or generative system to test many variations without normal shader-compiler
latency. The fixed traversal and output machinery provides the skeleton while
the generated object or material occupies the patch island.

### Program Synthesis and Superoptimization

This is the most direct generalization of Secant. Candidate instruction
sequences can be specialized into a correctness or performance harness,
executed briefly, and discarded. Native compilation latency normally forces
such searches to evaluate fewer candidates or spend substantially more work on
each candidate than its quality warrants.

### Constitutive-Law and Closure Discovery

In CFD, plasma physics, turbulence, material science, and combustion, the
expensive discretization and data movement can remain in a fixed skeleton.
Secant changes the local constitutive law, source term, flux, or closure being
evaluated. This combines a strong scientific use case with the expression and
sufficient-statistics kernels already developed for Secant.

## Applicability Boundary

The technique is less compelling when each variant changes global-memory
layout, synchronization topology, launch geometry, or an unbounded register
requirement. It is also less valuable for long-lived kernels that readily
amortize conventional compilation, or for workloads whose meaningful compute
already resides in fixed vendor-library kernels.

The broader interpretation of Secant is therefore:

> An ephemeral native GPU program-specialization system for register-local
> computations embedded in precompiled execution skeletons.

## Commercial Assessment

The most credible route to a business worth a few million dollars is not a
general-purpose compiler sold directly to individual developers. It is an
enterprise or OEM runtime for **runtime-generated local models inside GPU
simulation and optimization systems**.

The first product wedge would specialize expressions such as:

- Constitutive and material laws.
- Contact, friction, actuator, and sensor models.
- PDE source terms, fluxes, and closure models.
- Controller objectives and rollout scoring functions.
- Locally generated geometric or collision predicates.

These expressions would execute inside fixed CUDA skeletons that already own
the mesh, tile, neighborhood, trajectory, or environment traversal. An
optimizer, scientist, application user, GP system, RL policy, or LLM could
produce new local programs without asking NVRTC or the PTX compiler to rebuild
the complete simulation kernel for every candidate.

### Why This Is the Strongest Wedge

1. **It matches the specialization boundary.** Simulation kernels have
   substantial stable machinery around relatively small user-defined local
   laws. Secant can preserve the machinery and replace only the register-local
   computation.
2. **The surrounding ecosystem already uses runtime compilation.** NVIDIA Warp
   is explicitly positioned for simulation, robotics, geometry, optimization,
   and machine learning. Its documentation shows dynamically changed modules
   taking roughly 150-170 ms to compile and load in a small example, while also
   emphasizing caching and ahead-of-time compilation as ways to avoid that
   cost.
3. **Caching does not solve genuinely new programs.** A cache is effective when
   an identical model returns. It does not help an optimizer or generative
   system producing thousands of unique laws that may each run only briefly.
4. **The buyer can measure economic value.** Simulation and optimization teams
   can measure shorter design loops, more candidates evaluated per GPU-hour,
   better parameter searches, or less idle GPU time.
5. **It supports an OEM business model.** A small number of simulation,
   robotics, engineering, or scientific-software vendors could license the
   runtime and support rather than requiring a mass-market developer tool.

NVIDIA describes Warp as a JIT-compiled framework for simulation, robotics,
geometry processing, and optimization, with interoperability for machine
learning systems. Its code-generation documentation also demonstrates that a
new runtime-generated kernel can force approximately 150-170 ms of compilation
and module loading, compared with sub-5-ms cached loading in the example. These
are not direct Secant benchmark comparisons, but they establish that dynamic
GPU compilation latency is a recognized cost in the intended ecosystem.

- [NVIDIA Warp overview](https://developer.nvidia.com/warp-python)
- [Warp code generation and module caching](https://nvidia.github.io/warp/latest/user_guide/programming_model/code_generation.html)
- [Warp FEM API](https://nvidia.github.io/warp/latest/api_reference/warp_fem.html)

### Product Shape

The initial product should be a narrow CUDA SDK with C++ and Python bindings:

1. A typed, bounded expression language for f32, s32, u32, predicates, and
   reusable routines.
2. A small library of production skeletons for pointwise fields, tiled
   neighborhoods, reductions, trajectories, and material or constitutive-law
   evaluation.
3. Sub-millisecond bulk specialization into precompiled architecture-specific
   CUBIN templates.
4. A runner that batches many programs per module and accounts for module load,
   synchronization, execution, and errors.
5. PyTorch and Warp device-buffer and stream interoperability.
6. Persistent template caching keyed by Secant version, recipe, toolkit, and
   compute capability.
7. CPU reference execution, binary validation, deterministic testing, and
   explicit numerical policies.

The business would sell the complete integration and supported runtime, not
the AST encoder by itself.

### Revenue Model

A plausible model is annual enterprise or OEM licensing plus integration and
support. Reaching a few million dollars in annual revenue would require a small
number of substantial customers, for example approximately 10 customers at
$200,000-$300,000 per year or 20 customers near $100,000 per year. These are
planning targets, not validated prices. They become credible only after the
runtime demonstrably removes a blocking cost in a customer's production
workflow.

The likely buyers are:

- Robotics and simulation-platform vendors.
- CAE, digital-twin, and industrial-optimization vendors.
- Scientific-computing companies supporting custom physical models.
- GPU infrastructure teams building internal model-search systems.
- Companies allowing customers to author or generate local GPU behavior.

### Alternatives

| Commercial wedge | Technical fit | Commercial concern |
|---|---|---|
| Runtime-generated simulation laws | Excellent | Must prove that customers generate enough unique laws for compilation to matter. |
| GPU dataframe and query UDFs | Excellent | GPU database market is narrower, and many long queries already amortize compilation. |
| AI inference epilogues and plugins | Good | Very large market, but TensorRT, Triton, PyTorch, and NVIDIA compiler teams make it highly competitive. |
| Scientific equation-discovery application | Excellent | Requires substantially more search quality, domain validation, and workflow development than the compiler itself. |
| Quantitative-finance Monte Carlo | Good | High willingness to pay, but long sales cycles and repeated contracts often amortize ordinary compilation. |
| Quantum architecture search | Promising but unproven | Small current market and the patchable boundary for changing circuit topology still needs validation. |
| Cellular-automata rule search | Excellent demonstration | Weak standalone purchasing market. |

PyTorch documents compilation cold starts ranging from seconds to minutes in
common workloads, and demonstrates reducing one example from 11.4 seconds to
1.02 seconds by compiling and reusing a smaller repeated region. TensorRT-RTX
similarly uses runtime caches to avoid repeated device-specific JIT work. This
supports the general need, but those mature AI systems also make generic AI
kernel compilation a difficult initial market for Secant.

- [PyTorch regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)
- [PyTorch compiler troubleshooting](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
- [TensorRT-RTX runtime caching](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/work-with-runtime-cache.html)

## Commercial Validation Plan

Before treating this as a business rather than a technical hypothesis:

1. Implement one representative tiled simulation or FEM skeleton with a
   replaceable local law.
2. Generate at least 10,000 unique laws and compare Secant end to end against
   Warp or NVRTC, including template preparation, specialization, module load,
   execution, and cache behavior.
3. Demonstrate that specialized runtime remains close to native CUDA while the
   candidates execute too briefly for conventional compilation to amortize.
4. Integrate directly with existing PyTorch or Warp allocations and streams.
5. Interview at least ten simulation, robotics, or CAE teams and identify three
   design partners whose current workflows are materially constrained by
   runtime compilation or by an inability to generate native GPU behavior.
6. Convert at least one design partner into a paid pilot before broadening the
   instruction language or adding more kernel shapes.

The decisive falsification criterion is straightforward: if target customers
mostly reuse a small stable set of kernels, ordinary ahead-of-time compilation
and caching are sufficient, and Secant's specialization speed is not a product.
