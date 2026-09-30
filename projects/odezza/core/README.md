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

# Odezza native core

This is the authoritative C99 implementation of Odezza's kernel generation,
CUBIN inspection, SASS specialization and bounded CUDA execution. It contains
no GP, grammar enumeration, search controller, JSON protocol or dataset tooling.
Include **`odezza.h`**. All other headers here are private implementation details.

## Change notice

Before changing this folder for a task that did not request a core change,
notify the user and explain the affected interface and reason. When the user
requests the core change, do not issue an extra notice or ask for confirmation.
This agreement is enforced as working instructions in [AGENTS.md](AGENTS.md).

## Ownership and lifecycle

Both scoring and LM use an opaque handle and a synchronous bulk-run interface:

1. The application makes its CUDA context current and keeps it alive.
2. Create a handle for an explicit SM and compiled shape. The handle generates,
   compiles and inspects its template, and creates persistent streams/events.
3. Query and allocate the handle's host workspace. Allocate/upload device data
   through the application's CUDA layer.
4. Run batches. The handle handles specialization, bounded module loading,
   argument packing, launch, event completion and module retirement internally.
5. Destroy the handle before destroying the context. Free caller-owned data and
   workspace after completion.

The library never creates, selects, pushes or pops a CUDA context. The public API
intentionally does not expose internal tickets, stream counts or module queues.
Internally owned streams isolate ordering from other users of the same context.
`CUDA_MODULE_LOADING=EAGER` is application policy, set before CUDA initialization.
Normal completion uses events. A failed event-record operation may require a
stream fence solely to establish safe cleanup after an error.

The LM input-ready event permits dependence on an external upload/producer without
waiting for unrelated later work on that producer's stream. A completed bulk call
is the output-ready boundary. One host caller at a time may use each handle.

## Prepared scoring templates

Scoring can prepare its immutable compiled template separately from a particular
problem or CUDA context:

```c
OdezzaScoringTemplateInfo shape = {120, 6, 8, 256, 96, 128};
OdezzaScoringTemplate *compiled = NULL;
/* Check every return value; failed creation/read may retain a diagnostic handle. */
odezza_scoring_template_create(&shape, &compiled);
/* Optional: template_write measures/writes an opaque artifact into caller storage.
 * A later process uses template_read with the expected shape to load it. */
/* With the caller's CUDA context current, and matching create_info: */
odezza_scoring_pipeline_create_with_template(&create_info, compiled, &pipeline);
odezza_scoring_template_destroy(compiled); /* pipeline owns its independent copy */
```

The shape fields are SM, state capacity, constant capacity, systems per module,
shared patch capacity and per-system patch capacity. Active state/constant counts
and fixed RHS programs belong to pipeline creation. RNG seeds, sample pools,
reducer settings, trajectories and search strategy do not belong to a template.

The template owns generated code, the compiled cubin and its inspection. Creation
uses NVRTC but creates no GPU resources. Serialization exposes an opaque artifact,
not SASS metadata or a compiler interface. Reading checks the expected shape,
generator-source identity, length, checksum and cubin inspection without NVRTC.
Malformed/stale artifacts return errors. Error details use
`odezza_scoring_template_write_error`; destroy failed diagnostic handles too.

Immutable templates can serve multiple independently owned pipelines. Keep a
template alive while a creation/write call borrows it; it may be destroyed once
those calls return. Disk paths, build namespaces, atomic file publication,
readiness, eviction and cache policy remain in the application. The original
`odezza_scoring_pipeline_create` still prepares a temporary template internally.
Existing public descriptor layouts and scoring math are unchanged.

## Scoring and fitting concepts

Scoring keeps the existing `OdezzaScoringPipelineCreateInfo`, `OdezzaScoringLaunch`
and bulk system descriptors. Its public layout is preserved.

LM uses `OdezzaLmPipelineCreateInfo`, `OdezzaLmShape` and `OdezzaLmFit`.
A fit descriptor contains all RHS programs, initial parameter rows, trajectory
buffers, bounds, iteration settings and output buffers. The public C99 LM AST
uses the same postorder encoding as scoring: `CONSTANT_F32` is an optimized
parameter and `LITERAL_F32` is fixed. Python's legacy optimized-constant bytecode
must be translated at its adapter boundary, not interpreted by the native core.

A logical fit is one parameter start and one toggle permutation. Output order is
`[start, permutation]`; parameter output adds a final parameter dimension. States
and observations use `[state, concatenated trajectory point]`. Offsets delimit
ragged trajectories. Every trajectory must contain complete initial states at its
first point, which is excluded from scoring. Zero observation weight hides a
state/point. Weights multiply residuals and sensitivities before squared-error
accumulation, preserving the existing service LM convention.

The native LM generator implements RK4, analytic
sensitivities, bounded proposals and in-kernel Cholesky/damping retries. Native
symbolic differentiation and grouped common-subexpression lowering produce the
primal and local-partial SASS. Non-smooth ABS/MIN/MAX are rejected until an
explicit derivative policy is implemented. Unsupported shapes and resource
failures return errors; there is no hidden CPU or ordinary-CUDA fitting fallback.

Shapes support 1, 2, 4 or 8 cooperating lanes per fit. Wider shapes distribute
sensitivities, the normal equations and the Cholesky solve across lanes; primal
RHS and derivative sites still use native SASS specialization. CTAs contain 32
threads, giving `32 / lanes_per_fit` independent fits per CTA.

The native bounds are at most
16 states, 16 fitted parameters, 24 combined inputs, and 8,192 interned expression
nodes; physical register/patch limits may reject a smaller shape or expression.
These are implementation limits, not claims that all combinations fit on a GPU.

## LM shape selection and fallback

The original `odezza_lm_pipeline_create` requests exactly the specified width.
Use the additive fallback API to permit wider shapes:

```c
OdezzaLmPipelineCreateInfo info = {120, {8, 8, 256, 1}};
OdezzaLmPipeline *pipeline = NULL;
OdezzaResult result = odezza_lm_pipeline_create_with_fallback(
    &info, 2u | 4u | 8u, &pipeline);
/* Check result. On failure, query the diagnostic and destroy the handle. */
```

Lane values are also the mask bits. A fallback mask can contain only supported
widths greater than the requested width. Zero means exact shape. A preferred
four-lane shape can permit eight lanes with mask `8u`.

Creation prepares all permitted templates and computes the maximum host workspace
requirement once. This adds cold compilation work; templates persist for the
handle lifetime. Each system tries available widths in ascending order during
specialization. Only `REGISTER_PRESSURE` causes a retry. Invalid ASTs, malformed
compiler layouts, patch-capacity failures and CUDA errors remain explicit errors.
A failed batch drains admitted work and all its outputs must be discarded.

Every shape uses the same two streams/events, caller buffers and `[start,
permutation]` indexing. Different systems in one batch may use different widths.
`odezza_lm_pipeline_shape_report` reports requested/allowed/available/attempted/used
width masks, template results/register counts, setup time, system counts and
specialization register rejections by width. These are aggregate counts, not a
per-system ordered trace. The original descriptor and run-report layouts are
unchanged. This is resource fallback, not an automatic speed tuner; wider lanes
can cost more for small systems.

The native search adapter defaults to one lane with wider fallback enabled.
Requests can override that independently of `toggle_width`:

```json
{"fit": {"backend": "lm_toggle", "lanes_per_fit": 2, "shape_fallback": true}}
```

Set `shape_fallback` to `false` to enforce the requested shape. These controls
require `ODEZZA_CORE_LIBRARY`; the historical Python backend rejects them.
Search profiles include the native shape report and `shape_fallback_used`.
The Python binding also permits exact control:

```python
LmPipeline(library, sm, LmShape(8, 8, 256, 2), fallback_lanes=(4, 8))
```

The native generator now updates all coupled RK4 sensitivities simultaneously.
Consequently old native LM optimization paths and timing comparisons need a
fresh baseline; verified primal MSEs remain valid. The cooperative validation
also found and repaired compiler-inserted predicate save/restore instructions
inside patch sites. See [the incident and regression evidence](../docs/incidents/2026-09-10-native-cooperative-lm.md).

## CUB score reduction

The reducer uses only installed NVIDIA CUB block primitives, compiled by NVRTC.
The public boundary remains C99; no host CUDA C++ adapter, vendor-header patches
or benchmark glibc compatibility flags are used. Make supplies the include path
from `CUDA_PATH`; CMake uses `CUDAToolkit_INCLUDE_DIRS`. Matching CUDA/CCCL headers
must be available when creating the handle. Missing/incompatible headers produce
an explicit compiler diagnostic, with no custom-reducer fallback.

Create once for the current SM and initial k, then use
`odezza_score_reducer_set_k(handle, k)` between completed operations for k=1..256.
Query requirements with that k before running, and keep k unchanged until gather
finishes. This setter allocates nothing and triggers no compilation/module load.
The handle retains its stream and events; runs consume caller-owned buffers and
honor the producer's input-ready event. No allocation occurs in run/gather.

Top-1 uses CUB block reduction, top-2..4 repeated CUB argmin over a resident tile,
and larger k uses stable CUB block radix sorting. Tiles cover at most 2,048
contiguous elements; larger groups use an exact hierarchy with two caller-owned
scratch levels. One-tile groups write final outputs directly without a second
merge launch. The requirements query keeps a conservative nonzero scratch size
for compatibility even in that one-tile case. The old 256-tile cap is replaced
by checked 32-bit CUDA-grid/size bounds; pool limits belong to the caller.

The core knows only launch-layout groups and original indices. Family/tag and
raw-versus-distinct policies remain in the C runtime. MSE/index ties, signed-zero
bits, invalid/negative counts and exact coefficient gathering are preserved.
See [integration validation](../benchmarks/score_topk/integration/README.md).

Single-system scoring templates may have one fewer physical patch instruction
when NVRTC elides the final branch. Template validation accepts this specific
shortfall; specialization still uses the inspected physical capacity, never the
larger requested capacity. Creation and artifact round-trip have regression tests.

## Compilation units

| Files | Responsibility |
|---|---|
| `odezza.h` | Sole public header: scoring, LM, score reduction and RNG pool contracts |
| `o_odezza_internal.h` | Shared private types and function declarations |
| `o_generate_scoring_cuda.c`, `o_generate_lm_cuda.c`, `o_lm_source.c` | CUDA generation and fixed LM numerical body |
| `o_elf.c` | Checked shared ELF reads, sections and symbol lookup |
| `o_inspect_scoring_cubin.c`, `o_inspect_lm_cubin.c` | Physical marker/site inspection |
| `o_ast.c`, `o_lm_ast.c` | Postorder validation and analytic differentiation |
| `o_sass_assembler.c`, `o_sass.h` | Shared register/barrier-aware lowering and instruction encoding |
| `o_specialize_scoring_cubin.c`, `o_specialize_lm_cubin.c` | Checked template patching and register metadata |
| `o_scoring_template.c` | Context-independent compilation, inspection and opaque artifacts |
| `o_nvrtc_compilation.c` | NVRTC ownership, CUBIN and compiler diagnostics |
| `o_scoring_pipeline.c`, `o_lm_pipeline.c` | Handle lifecycle, bulk scheduling and launch ABI |
| `o_philox.*`, `o_score_reduce.*`, sampled-constant helpers | Reusable GPU data primitives used by the kernel pipeline |
| `o_writer.h`, `o_sha256.h` | Private bounded source writing and artifact identity |

The historical `src/` paths are compatibility symlinks into this folder; there
is one authoritative implementation. Root Make/CMake builds use `core/` directly.
Research snapshots under `scratch/*/native` remain historical snapshots and are
not edited by this move. A downstream build must link the current library and
use its current headers to validate the new core, rather than compiling snapshot
copies over the top of it.

## Build and validation

Use the repository's existing CUDA toolkit and build tools. No runtime Python,
compiler subprocess, external symbolic package or service is required by C99.

```sh
make all test-c
make shared test-public-api
make test-scoring-template
make test-lm-core
make test-lm-failures
CUDA_MODULE_LOADING=EAGER make test-cuda
```

`test-lm-core` is a public-header-only C99 caller: the test owns a context and data,
submits three packs with partial CTAs, verifies coefficient recovery and MSE,
checks the workspace contract and reuses the handle. Existing scoring/SASS tests
protect the shared-code extraction. CUDA tests must run on a supported NVIDIA
host; host or source checks do not stand in for GPU execution.

The [public C99 caller](../tests/o_lm_core_test.c) is a complete executable example.
The shared library exports only the 34 functions declared in `odezza.h`; the
export test rejects accidentally exposed internal helpers. Make tracks included
headers so changing a shared implementation header rebuilds dependent units.

Native and downstream validation passed on rack1, including independent CPU
replay, two-GPU GP/checkpoint execution and CUDA memory checking. See
[validation and limitations](VALIDATION.md). This is numerical and integration
validation, not a claim of maximum throughput or universal shape support.

## Using the current core from the fitting trial

The thin binding is [python/odezza/native.py](../python/odezza/native.py). The
[trial adapter](../scratch/fitting_batch_trial/lm_toggle/native_service.py) owns
search reports and device-data conversion. Its C99 handle owns execution.

From the repository root on a configured CUDA host:

```sh
make all shared
make -C scratch/fitting_batch_trial core-runner
CUDA_MODULE_LOADING=EAGER ODEZZA_CORE_LIBRARY="$PWD/build/libodezza.so" \
  python3 scratch/fitting_batch_trial/service.py \
    --root "$PWD/scratch/native-core-service" \
    --backend "$PWD/scratch/fitting_batch_trial/bin/odezza-fit-core-run" \
    --port 8772 --devices 0,1
```

Use the repository's existing Python environment. `ODEZZA_CORE_LIBRARY` explicitly
selects native LM for requests selecting the `lm_toggle` fitter. An invalid path
or unsupported native shape fails explicitly. Omitting this variable preserves
the historical Python adapter. `core-runner` links the current scoring library;
the historical `runner` target compiles snapshot overrides for old experiments.

The native LM lifecycle follows scoring: templates persist; specialized modules
are loaded and retired in bounded slots. The older Python trial retained a
module cache. Reports record this distinction (`module_cache_hit=false` for
native LM); warm timings between these backends are not assumed equivalent.
No existing service was restarted or switched as part of this refactor.

Scoring run reports now expose per-module host timing and optional CUDA interval
profiling (`OdezzaScoringLaunch.profile_timing`). Events remain handle-owned; the
steady-state call still creates no resources or allocations. This extends the C
launch/report ABI: recompile callers. See the
[timing definitions](../benchmarks/request_sizing/README.md); summed module time
and CUDA interval union must not be described as SM occupancy.
