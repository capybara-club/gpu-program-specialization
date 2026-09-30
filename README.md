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

# GPU Program Specialization

**Techniques for executing large populations of different mathematical programs on NVIDIA GPUs, by Charles Durham.**

This is a source and knowledge handoff. It collects Secant, Odezza, PTX Inject,
Stack PTX, AST PTX, and the experiments that connect them. The intended reader is
an engineer—or an LLM working with one—adapting the execution techniques to a new
domain. Secant-SR is included as an example consumer, not as the product this
repository is trying to ship.

The central question is: **how do you evaluate many changing programs without
paying for a complete GPU compilation for every program, or interpreting every
operation on the GPU?** The progression here goes from generating and injecting
PTX, through patching compiled functions, to specializing reserved native
instruction regions in compiled CUDA kernels.

These approaches trade compiler work for explicit control over instruction
encoding, registers, scheduling, kernel shapes, and module lifetimes. The code
and its limitations are both part of the handoff.

## The idea in one minute

Suppose you want to test many different mathematical expressions against the
same data or inside the same simulator. Most of the GPU program stays the same:
load inputs, run a loop, accumulate an error. Only the expression changes.

This work explores making that changing part cheap. Early versions insert
generated GPU assembly into a stable kernel and invoke NVIDIA's compiler.
Later versions compile a reusable kernel template once and write supported
native instructions directly into its reserved expression regions. Toggles and
coefficient banks give each loaded program more useful work, while fused scoring
and pipelining reduce the cost of moving data and coordinating launches.

**The reusable result is an execution technique:** a way to run many changing
programs efficiently. An LLM, enumerator, optimizer or search algorithm can be
the caller. The search strategy is a separate choice.

| Term | Meaning here |
|---|---|
| AST | A structured representation of a mathematical expression |
| PTX | NVIDIA's virtual assembly; it still needs compilation for the target GPU |
| SASS | Native instructions for a particular NVIDIA GPU architecture |
| CUBIN | A compiled GPU module containing native code and its metadata |

## Start here

| Your goal | Read first |
|---|---|
| Understand the techniques and their relationships | This README, then [architecture and history](docs/ARCHITECTURE.md) |
| Find a project's category, status and first file | [Project map](docs/PROJECTS.md) |
| Check which research paths are included | [Research coverage and scope](docs/RESEARCH_COVERAGE.md) |
| Give an LLM a useful entry point | [llms.txt](llms.txt), [adaptation guide](docs/ADAPTATION.md), [AGENTS.md](AGENTS.md) |
| Specialize row-wise mathematical expressions | [Secant API](projects/secant/secant.h), [Secant README](projects/secant/README.md) |
| Specialize a model inside an ODE rollout or fit | [Odezza core](projects/odezza/core/README.md), [public API](projects/odezza/core/odezza.h) |
| Use PTX without writing a native instruction encoder | [PTX Inject](projects/mm-ptx/PTX_INJECT.md), [Stack PTX](projects/mm-ptx/STACK_PTX.md), [AST PTX](projects/mm-ptx/AST_PTX.md) |
| Patch compiled device functions | [CUBIN Function Patch](projects/cubin-function-patch/README.md) |
| Understand measured performance and its limits | [Evidence and benchmark boundaries](docs/BENCHMARKS.md) |
| Build or inspect the handoff | [Build guide](docs/BUILDING.md), [validation record](docs/VALIDATION.md) |

## Project categories

The [project map](docs/PROJECTS.md) classifies all **15 component snapshots** and
gives a first file to read for each. Small, selected Odezza experiments live under
`projects/odezza/research/` with their own status and limitations.

| Category | Projects | What to learn |
|---|---|---|
| **Current native execution engines** | Secant, Odezza | Direct instruction specialization, fused scoring/fitting, module pipelines |
| **Program construction and specialization tools** | MM PTX, Stack PTX Emit, CUBIN Function Patch | Compact programs, PTX injection, code emission, compiled-function replacement |
| **Bindings and execution infrastructure** | MM PTX Python, compiler workers | Language integration, register/site contracts, bounded compiler queues |
| **Fused statistics and fitting research** | FusedSINDy, Secant-SINDy | Fusing feature evaluation with sufficient statistics and ridge/STLSQ fitting |
| **Reference consumers** | Secant-SR, Secant System ID, MM Kermac | Search/refinement consumers and a secondary PTX-injection application in kernel-matrix computation |
| **Historical implementations and measurement** | cuSR, Secant Benchmark, C PTX compilation benchmark; Odezza research trials | Why the designs changed, baseline implementations, topology and numerical experiments |

Start with a current engine when adapting native execution. Use the older
implementations to understand a technique or compare an approach. A component's
presence does not imply API compatibility with the other components.

This is a snapshot of working trees, including relevant uncommitted source—not
a claim that each original HEAD exactly describes every included file. The
[source manifest](docs/source-manifest.json) records original commits, original
file hashes, exported hashes, licenses, and exclusions. Original Git histories
are not imported. Embedded historical header versions remain with their
consumers; they must not automatically be replaced with the newest header.

## 1. Separate the stable kernel from the changing program

The stable work often dominates the shape of a GPU computation:

- loading a row, trajectory, initial condition, or parameter vector;
- looping over observations or integration steps;
- maintaining states, partial sums, or Jacobian information;
- reducing scores and writing results.

The changing work is a relatively small mathematical program embedded inside
that computation. Compile the surrounding kernel as a reusable template, then
specialize the program region for each group of candidates. Keep input data
separate from the compiled template whenever the data can be supplied at launch.

This is most useful when the program changes often enough that compilation is
expensive, yet each specialized program does enough work to justify module
loading and execution. A permanently fixed model may be better served by an
ordinary compiler. A tiny evaluation may still be dominated by dispatch overhead.

## 2. Stack PTX and AST PTX: structured programs become PTX

[Stack PTX](projects/mm-ptx/stack_ptx.h) represents programs with fixed-width
instructions and a configurable typed stack/routine description. It emits PTX
on the host. The device does **not** execute a stack-bytecode interpreter simply
because the source representation uses a stack. The compiler can eliminate
unused work and applies the validity rules documented in the header.

[AST PTX](projects/mm-ptx/ast_ptx.h) provides postorder expression programs,
explicit operator arity and routine arguments, with a
[CPU interpreter](projects/mm-ptx/tools/ast_ptx_interpreter.h) useful for checking
semantics. The representation and operation descriptions are kept separate.

The transferable idea is to have a compact, bounded, mechanically checkable
program representation that an enumerator, GP, grammar, human, or LLM can
produce. Do not make the GPU execution engine depend on who produced it.

## 3. PTX Inject: insert programs into a compiled scaffold

[PTX Inject](projects/mm-ptx/ptx_inject.h) locates annotated sites in PTX and
inserts generated code with explicit input, output, and modified-register
bindings. The marker generators copy values into stable named registers before
the replacement region and copy outputs back afterward.

That makes the program interface explicit while CUDA supplies the surrounding
kernel. It also permits multiple sites and reusable routine descriptions. The
API uses bare register names; PTX emission adds the `%` syntax.

**Injected PTX still needs to be compiled into native instructions.** This route
avoids rebuilding all of the source-level scaffolding, but it does not eliminate
the PTX compiler. The parallel compilation benchmarks explore how far batching
and multiple host workers can take this approach.

## 4. Patch native code instead of recompiling every candidate

There are two materially different native paths here:

1. [CUBIN Function Patch](projects/cubin-function-patch/cubin_function_patch.h)
   replaces a reserved device-function body in a linked template. The replacement
   function is still compiled. Matching signatures, code capacity, architecture,
   register resources and relocation constraints matter. This is a constrained
   patcher, not a general replacement for a linker.
2. Secant and Odezza translate supported programs directly into SASS instruction
   regions discovered in compiled CUBIN templates. This removes a PTX compilation
   for each supported candidate once the appropriate template exists.

The native path includes more than writing opcode bytes: inspection discovers
the template layout and register bindings; specialization respects instruction
capacity, predicate use, dependency scheduling, branch/control metadata and
architecture-specific encodings. See
[Secant inspection](projects/secant/s_cubin_inspect.c),
[Secant specialization](projects/secant/s_cubin_specialize.c), and
[Odezza core](projects/odezza/core/).

This relies on NVIDIA binary details that are not a portable public ISA contract.
A new GPU generation or compiler version needs inspection and numerical tests.
An architecture enum or an encoder table alone is not proof of support.

## 5. Register-backed leaf toggles and coefficient banks

A leaf can select between a small number of alternatives: state/column values,
literal constants, or runtime coefficient slots, according to the particular
API's supported leaf kinds. Two-way and four-way choices let one native program
cover related bindings without compiling each binding as a different AST.

Coefficient vectors are supplied as a bank. The kernel loads the selected
vector into designated registers and reuses it during the inner loop. A
specialized AST refers to those registers rather than chasing an arbitrary
pointer for every coefficient occurrence.

The search space can be a Cartesian product:

`program structures × toggle assignments × coefficient-bank vectors`.

Keep those counts separate. A million parameter assignments is not a million
structurally different expressions. Unused toggle bits and algebraically
equivalent expressions can also produce duplicate mathematical evaluations.

In current Secant packed execution, ASTs in a pack share the selected toggle
bits and coefficient vector, with separate score accumulators. A bit can select
different local alternatives in different ASTs; sharing bits does not require
the ASTs to have the same structure. Sharing a bit between two sites within an
AST **couples** those choices, so independent choices need independent bits.

See [Secant toggle semantics](projects/secant/docs/toggles.md) and the exact
[Odezza AST/API definitions](projects/odezza/core/odezza.h). These are related
designs, not interchangeable binary formats.

## 6. Fuse evaluation with the quantity the caller needs

Secant can evaluate expressions while accumulating SSE or other sufficient
statistics. Affine moments and Gram statistics support downstream fitting
without necessarily materializing a full feature matrix. FusedSINDy explores
the same memory-traffic principle in feature-library workflows.

Odezza specializes expressions inside the integration loop: state values remain
in registers while RK4 stages evaluate the candidate RHS. The scored output is
a trajectory error, not an error against numerically estimated derivatives.
Known RHS components can be specialized into the stable portion of the template.

In the scoring topology, a CTA selects a compatible candidate system and threads
evaluate its configurations. Candidate packing, parameter reuse, the number of
trajectories, observations and substeps all affect useful work and occupancy.
The row engine and rollout engine do not have identical thread mappings.

Full trajectory output, SSE, normalized MSE and moments are different output
contracts. Preserve normalization and observed-state masks when comparing them.
Missing observations are supported on the appropriate Odezza path; complete
initial state vectors are still required there.

## 7. RNG pools and reproducible result indices

Odezza includes a Philox pool path. Uniform and normal samples can be generated
separately from scoring, reused by multiple launches, and transformed with
family-specific scale/offset before the rollout loop. The pool represents
candidate constants; it is not automatically an SDE noise process.

Avoid transferring every sampled vector back just to identify winners. A retained
index plus the exact layout, grammar identity, bank/pool specification, seed,
distribution and transform can identify the selected values. Reconstruction
must use the same indexing and generation semantics; an index alone is not a
portable candidate description.

Changing precision, RNG mapping, normal-transform implementation or generation
order can change results. Preserve explicit values when exact cross-version
replay cannot otherwise be guaranteed.

## 8. Reduce on the GPU and retain only useful results

Odezza's current GPU score selection uses CUB-based kernels. Small-k selection
and larger radix-sort/hierarchical paths return indices as well as scores. This
reduces transfer and host selection work when only a small set of winners is
needed. Family-level retention belongs to the request/scheduling layer; a family
is not an intrinsic property of a mathematical instruction.

Result indices must survive chunking and remapping into the global candidate
space. Tie-breaking, nonfinite scores, family boundaries and partial work all
need explicit contracts. Keep allocation budgets separate from top-k retention:
keeping ten winners says nothing about how many candidates were evaluated.

## 9. Pipeline module loading, execution and lifetime management

Avoid a sequence of tiny compile/load/launch/synchronize/unload requests. The
pipeline creates capacity, specializes compatible groups, loads modules, launches
work, and tracks completion using streams and events. Reuse input and result
storage where possible; reclaim modules and buffers only after their users finish.

The hardened APIs expose their own ownership rules. In particular, do not infer
that an API accepts caller-owned CUDA streams just because another prototype
does. The current Odezza core owns its pipeline streams/events while the caller
supplies the current context and device-data lifetime.

Packs must contain enough work to amortize module overhead and populate the GPU.
Block size, configurations per candidate, packs in flight, register use and
trajectory length all interact. A fast kernel is not sufficient if the producer
or module-loading pipeline leaves it idle.

Use events when an individual stage needs completion, rather than synchronizing
unrelated work. Never remove a synchronization without establishing which
resources it protects. Failure-path tests matter as much as successful launches.

## 10. Cache compiled templates with complete identities

Odezza includes a C-side SQLite compiled-template cache. The expensive initial
template compilation can be reused across requests; supported ASTs are then
specialized into that template. A larger expression may need a larger template
capacity and a separate cache entry.

A cache key needs the inputs that change generated code and ABI: architecture,
kernel shape, fixed structure, relevant generator/toolchain identity and compile
options. Request trajectories and coefficient buffers are per-job data, not
mutable contents of a supposedly immutable cached template.

SQLite here stores compilation artifacts, not a requirement to put every
trajectory, candidate or score into a database. Cache correctness and buffer
ownership are separate concerns.

## 11. Parse once, prepare once, stream compact ASTs

[Odezza's C99 frontend](projects/odezza/frontend/) separates trajectory parsing,
static structure preparation and grammar preparation. Caller-provided arenas
support measurement before allocation; insufficient capacity has an explicit
result. A prepared grammar produces postorder AST payloads directly.

This removes the need to serialize every expanded AST into Python-generated
JSON. Bounded producer batches can feed the GPU pipeline while another batch is
prepared. Resume uses the prepared producer's documented cursor semantics; do
not assume an arbitrary accepted-AST index is a constant-time random-access key
through a heavily filtered grammar.

Families have resource allocations and generation-effort limits. Interleaving
bounded batches prevents traversal order from consuming all the work before
later families run. The report must distinguish attempted, accepted, scored,
failed and retained items. These policies stay outside the native instruction
specializer.

## 12. Fitting coefficients and sharing work across lanes

Odezza's native LM path specializes the model and sensitivity computation into
fitting kernels. RK4 sensitivities must follow the same coupled stage equations
as the state rollout. Normal equations, damping, acceptance and stopping rules
then update the candidate's coefficients.

One lane per fit can be effective for small systems. Larger state/parameter
counts create register pressure; cooperative shapes distribute a fit across
more lanes. Spills, occupancy and communication overhead determine the best
shape, not a universal rule that more lanes or fewer registers is always faster.
The core provides explicit shape controls and fallback/inspection machinery.

Secant-SR also contains random/curvature-style refinement and a newer GPU LM
path. Its register-only expression evaluator is a separate implementation;
do not confuse it with Odezza's direct SASS LM specialization or treat historical
refinement comparisons as universal conclusions about LM.

## Measured examples, with scope

These are historical measurements retained in the component reports, not new
measurements of this export and not matched speedups against competing systems.

| Workload | Hardware | Observed result | What it means |
|---|---|---:|---|
| Four generated depth-4 RHS programs per system; 4,096 systems; 256 configs/system; 3 trajectories, 8 observations, 2 RK4 substeps | RTX 5090 | 251,041 systems/s; 1,004,165 RHS program occurrences/s; 64.27M configs/s | Prepared-corpus specialize/load/run/unload scope; excludes corpus construction and initial setup |
| One million six-RHS vectors × 2,048 configurations; short four-step scoring | 2 × RTX 5080 | 1.2210 s native; 1.2654 s service path | Finite combinatorial vectors, not one million unrelated arbitrary trees |
| 317.19M configurations, long 5,120-step screening workload | 2 × RTX 5080 | 222.64 s → 34.16 s | Internal batching/pipeline repair, about 6.52× on that workload |

See [the evidence guide](docs/BENCHMARKS.md) for source reports, setup boundaries
and numerical caveats. A configuration can mean four integration steps or
thousands; a single headline configs/s rate cannot describe both workloads.

## What to reuse in another domain

Good candidates have many changing mathematical programs, a substantial common
execution scaffold, and enough evaluations per program to amortize loading.
Examples to investigate include simulation ensembles, model-structure
comparison, feature libraries, repeated likelihood/scoring functions, and
domain-specific program search. These are proposed applications, not validated
performance claims for every field.

The adaptation sequence is: define semantics and a CPU oracle; define a bounded
program format; establish a stable kernel interface; implement PTX or native
specialization; validate numerical and failure behavior; then measure the entire
producer-to-result path. See [the detailed guide](docs/ADAPTATION.md), including
a prompt to give an LLM.

You do not need this repository's GP policy, grammars, web service or evolutionary
representation to use its specialization techniques. Keep policy out of the
execution core, and prefer the smallest component that addresses your bottleneck.

## Boundaries and known limitations

- This is research/engineering source, not a security-hardened public execution
  service or a promise of turnkey builds for every historical prototype.
- Direct SASS specialization requires architecture/compiler validation. The
  recorded Ada/Blackwell results do not imply all NVIDIA GPUs are covered.
- The native scoring request path is currently based on FP32 fixed-step RK4.
  It is not a production stiff/adaptive ODE solver. High throughput cannot repair
  integration error or an unidentifiable inverse problem.
- Untrusted JSON, arithmetic overflow, expression capacity, pathological rollout
  lengths and nonfinite scores need bounded handling before expensive work.
- Register allocation, stack/local memory, spills and occupancy must be measured
  for the actual shape and model. Historical peak throughput is not a guarantee.
- Some archived adapters use older AST/settings APIs; current build entry points
  intentionally do not silently emulate those APIs.
- Bulk datasets, generated binaries, papers, logs, private operations and original
  repository histories are excluded. Historical reports may refer to omitted
  artifacts; see [omissions](docs/OMISSIONS.md).

## Build, contribute, and hand off

Start with `python3 tools/check_host.py` after installing the documented host
prerequisites. This runs selected CPU-side tests without requiring CUDA or
downloading dependencies. GPU execution needs a supported NVIDIA/CUDA environment
and the component-specific checks in [BUILDING.md](docs/BUILDING.md).

Run `python3 tools/audit.py` before publishing source changes. Read
[CONTRIBUTING.md](CONTRIBUTING.md) and [the validation record](docs/VALIDATION.md).
The snapshot manifest preserves where files came from; new changes should have
their own ordinary commits and validation evidence.

## License

Charles Durham's code and documentation are released under the [MIT
License](LICENSE). This includes his Meta Machines / MetaMachines work, PTX
Inject and Stack PTX. Commentable owned files carry MIT notices; formats such as
JSON and golden-data files use adjacent `.license` notices so their syntax stays
valid. Each component also has a standalone MIT license.

Third-party code keeps its original terms. Boost, SQLite and incbin are explicitly
identified in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The MIT license does
not relicense CUDA, third-party dependencies, or external benchmark datasets.
