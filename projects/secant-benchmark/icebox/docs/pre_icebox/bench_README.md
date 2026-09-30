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

# SECANT Benchmarks

SECANT keeps compilation and execution in separate executables so their timing
boundaries are explicit.

Both executables use the shared AST construction in `bench_ast.c`. ALU mode
cycles through the generated C representation of the versioned portable corpus in
[`corpus/portable_alu_v1.json`](corpus/portable_alu_v1.json), which is also used
by the external backend collectors. The corpus generator also provides
MUFU-heavy, mixed-depth, constant-bearing, and protected-math JSON profiles.
Regenerating `portable_alu_v1.h` selects one of those profiles for the compiled
SECANT and native AVX benchmarks. Each result prints its stable `corpus` and
`corpus_hash`, plus an `expanded_corpus_hash` for the exact repeated program
array. Simple and MUFU modes remain deterministic generated workloads, but
they are not presented as exact external-backend comparisons. MUFU safe
division is reciprocal then multiply and uses the same safe square-root
routines for every SECANT backend. GPU native approximate operations are
compared against the CPU interpreter with tolerance rather than assumed to be
bit-identical.

External runtime baselines for PySR, EvoGP, Kozax, Operon, and native AVX2/FMA
C are described in [`baselines/README.md`](baselines/README.md). Those collectors write a
shared CSV schema and can be rendered together without changing SECANT's
compile or runtime benchmark formats.

## Compile

`secant_compile_bench` measures the backend's AST-to-native-binary call:

- CUDA: ASTs, CUDA generation, and NVRTC compilation to CUBIN.
- PTX: AST lowering, PTX Inject rendering, and nvPTXCompiler compilation to CUBIN.
- HIP: ASTs, HIP generation, and HIPRTC compilation to HSACO.
- CUBIN: direct AST-to-SASS generation into a caller-owned CUBIN template.
- HSACO: direct AST-to-AMDGPU-ISA generation into a caller-owned HSACO template.

Backend setup, PTX template creation, AST generation, worker scratch allocation,
warmups, module loading, execution, and destruction are outside the timer.
CUDA, PTX, and HIP return distinct owned native binaries. For CUBIN and HSACO,
template measurement, allocation, and copying happen before timing; the timed
call is only `secant_cubin_specialize_into` or
`secant_hsaco_specialize_into`. Each lowers ASTs and patches a prepared
caller-owned image in place. The executable retains the binaries until timing
ends, then loads a configurable subset and compares its outputs against
the CPU interpreter declared in `secant_cpu.h`.

OpenMP workers share immutable prepared state and use independent ASTs, scratch,
logs, and outputs. CUBIN and HSACO each inspect one native template before
timing and share the resulting immutable plan across workers. Every worker
patches a distinct caller-owned native image directly; specialization has no
scratch buffer or allocation.

```sh
./build/secant_compile_bench \
  --backend ptx \
  --shape materialize \
  --ast-mode mufu \
  --warmups 2 \
  --iterations 48 \
  --workers 24 \
  --kernels 16 \
  --asts-per-kernel 8 \
  --opt-level 1 \
  --source-sm 80 \
  --target-sm 120 \
  --check-modules 2
```

Use `--backend cuda` for direct NVRTC compilation, `--backend hip` for HIPRTC,
`--backend cubin` for direct SASS patching, or `--backend hsaco` for direct
AMDGPU ISA patching. CUBIN and HSACO require `--opt-level 1`, which describes
the cold template compiler configuration; native AST patching does not invoke
an optimizer. HIP and HSACO benchmarks detect the selected device architecture;
`--hip-arch` overrides it. A backend absent from the configured build is
rejected without requiring its headers or libraries.

### Compile Matrix

`compile_matrix.py` runs a common module-shape cross product serially and writes
one aggregate AST/s matrix CSV per backend. Rows are worker count, kernel shape,
optimization, and AST workload configurations. Columns are identical module
shapes for every row. The `--output` path is a base name: the command below
writes files ending in `_cuda.csv`, `_ptx.csv`, and `_cubin.csv`.

```sh
python3 bench/compile_matrix.py \
  --benchmark /home/cdurham/code/build/secant/secant_compile_bench \
  --output docs/compile_matrix_2026-07-24.csv
```

The default module shapes are `8x24`, `16x24`, `32x24`, `64x8`, `64x16`, and
`64x64`, where each value is `kernels/module x ASTs/kernel`. Each shape is run
with 1, 12, and 24 workers. Each worker performs one untimed warmup compile and
two timed compiles: the benchmark therefore gets 1/2, 12/24, and 24/48
warmup/timed modules. CUBIN emits only O1 rows because its optimization level
describes the fixed template compilation, not the direct SASS patch. The script
writes the affected backend CSV after every completed cell. Running the same
command resumes missing cells independently in each file; use `--force` to
start over. `--dry-run` prints the complete command set without invoking the
benchmark.

Render the resulting CSV with module shapes on the horizontal axis and PTX/CUDA
as the only table rows:

```sh
.venv/bin/python bench/compile_matrix_graph.py \
  docs/compile_matrix_2026-07-24_ptx.csv \
  docs/compile_matrix_2026-07-24_cuda.csv \
  docs/compile_matrix_2026-07-24_cubin.csv \
  --output docs/compile_matrix_2026-07-24.svg

python3 bench/compile_matrix_report.py \
  docs/compile_matrix_2026-07-24_ptx.csv \
  docs/compile_matrix_2026-07-24_cuda.csv \
  docs/compile_matrix_2026-07-24_cubin.csv \
  --graph docs/compile_matrix_2026-07-24.svg \
  --output docs/compile_matrix_2026-07-24.md
```

## Runtime

`secant_runtime_bench` prepares and loads one module, verifies every kernel
against the CPU interpreter, then uses GPU events to time kernel launches only.
Compilation or template creation, AST patching, loading, allocation, copies,
output clearing, warmups, and verification are excluded.

```sh
./build/secant_runtime_bench \
  --backend ptx \
  --shape sse \
  --ast-mode mufu \
  --kernels 16 \
  --asts-per-kernel 8 \
  --tile-rows 4096 \
  --threads 128 \
  --streams 1 \
  --opt-level 1 \
  --run-rows 1048576 \
  --run-iterations 20 \
  --check-rows 257 \
  --source-sm 80 \
  --target-sm 120
```

For SSE, row evaluations count one AST evaluated for one row. Targets do not
multiply the reported row-evaluation count. The runtime benchmark defaults to
4,096 rows per CTA and 128 threads, amortizing each CTA's reduction and atomic
epilogue across 32 rows per thread. Both values are included in each result.

`--streams N` assigns kernels round-robin to `N` nonblocking streams. Streams,
start/stop events, and per-stream completion events are created before the hot
loop. The timing stream releases the other streams with a start event, waits
for every completion event, and then records the stop event. Stream count is
reported with each result and cannot exceed the number of kernels.

For HSACO SSE, `--sse-atomic-stride N` writes each AST's atomic result into a
temporary output with an `N`-float leading dimension. After every timed kernel
batch, one generated module-wide kernel compacts the temporary results into
the ordinary dense output. Compaction is included in runtime timing and CPU
verification reads the compacted output.

Run and graph the fixed-packing SSE tile sweep with:

```sh
python3 bench/runtime_tile_sweep.py \
  --output docs/runtime_tile_sweep_2026-07-24.csv

.venv/bin/python bench/runtime_tile_graph.py \
  docs/runtime_tile_sweep_2026-07-24_ptx.csv \
  docs/runtime_tile_sweep_2026-07-24_cuda.csv \
  docs/runtime_tile_sweep_2026-07-24_cubin.csv \
  --output docs/runtime_tile_sweep_2026-07-24.svg
```

Sweep AST packing for materialize and SSE, then render each shape on its own
scale:

```sh
python3 bench/runtime_packing_sweep.py \
  --output docs/runtime_packing_sweep_2026-07-24.csv

.venv/bin/python bench/runtime_packing_graph.py \
  docs/runtime_packing_sweep_2026-07-24_ptx.csv \
  docs/runtime_packing_sweep_2026-07-24_cuda.csv \
  docs/runtime_packing_sweep_2026-07-24_cubin.csv \
  --shape materialize \
  --output docs/runtime_materialize_packing_2026-07-24.svg

.venv/bin/python bench/runtime_packing_graph.py \
  docs/runtime_packing_sweep_2026-07-24_ptx.csv \
  docs/runtime_packing_sweep_2026-07-24_cuda.csv \
  docs/runtime_packing_sweep_2026-07-24_cubin.csv \
  --shape sse \
  --output docs/runtime_sse_packing_2026-07-24.svg
```

Build the complete SSE cross product of AST packing and logical tile size,
reusing the one-dimensional sweeps as seeds:

```sh
python3 bench/runtime_sse_surface_sweep.py \
  --packing-csv docs/runtime_packing_sweep_2026-07-24.csv \
  --tile-csv docs/runtime_tile_sweep_2026-07-24.csv \
  --output docs/runtime_sse_surface_2026-07-24.csv

.venv/bin/python bench/runtime_sse_surface_graph.py \
  docs/runtime_sse_surface_2026-07-24_ptx.csv \
  docs/runtime_sse_surface_2026-07-24_cuda.csv \
  docs/runtime_sse_surface_2026-07-24_cubin.csv \
  --output docs/runtime_sse_surface_2026-07-24.svg
```

Every runtime sweep treats `--output` as a base name and writes one CSV per
backend. CUBIN records only O1. The default materialize sweep stops at 128
ASTs/kernel, while the SSE sweeps include 96, 128, and 192 ASTs/kernel. A
backend failure aborts the sweep instead of silently filtering a requested
packing. Use `--backends` to run a subset.
`--opt-level 0|1` selects ptxas optimization for
direct CUDA, nvPTXCompiler optimization for PTX, and HIPRTC optimization for
HIP.

### Dynamic-Constant SSE

`secant_dynamic_constant_sse_bench` is shared by CUDA, PTX, HIP, CUBIN, and
HSACO. Each thread owns a constant setting and accumulates SSE for every packed
AST and target. The same deterministic ASTs, columns, constants, targets,
output layout, CPU oracle, and timing procedure are used for every backend.
It verifies four settings over up to 257 rows before timing. One row evaluation
is one AST evaluated for one setting and one row; targets do not multiply the
reported count.

Without `--tile-rows`, the benchmark uses measured backend defaults:
64 for NVIDIA, 1024 for native HIP, and 256 for direct HSACO. These differ
because the Radeon paths are more sensitive to tile size, contended atomic
updates, and frontend loop optimization. AMD defaults to 100 warmup launches
so the measurement does not include the transition from the card's idle clock.

Select a backend and configure both the expression inputs and constant-setting
matrix at runtime:

```sh
CUDA_MODULE_LOADING=EAGER \
  ./build/secant_dynamic_constant_sse_bench \
    --backend cubin \
    --columns 4 \
    --constants 4 \
    --settings 256 \
    --targets 2
```

SSE generators support `--reduction atomic` and `--reduction workspace`.
Atomic mode adds each tile's partial SSE directly to the result. Workspace mode
writes one partial per result and tile, then launches the generated reducer
kernel once per iteration. The benchmark reports the workspace size and
includes the reducer in runtime timing.

Use the MUFU-heavy configuration with:

```sh
CUDA_MODULE_LOADING=EAGER \
  ./build/secant_dynamic_constant_sse_bench \
    --backend cubin \
    --ast-mode mufu \
    --asts-per-kernel 48 \
    --settings 256 \
    --tile-rows 64 \
    --threads 128
```

PTX template creation and CUBIN/HSACO source compilation plus inspection are
reported as `template_prepare_seconds` and remain outside `compile_seconds`.
For direct ISA backends, `compile_seconds` is only in-place AST
specialization. The result also includes AST-setting-row evaluations/s, atomic
updates/s, registers per thread, static shared memory, local memory, active
blocks/SM, and module-load time. Device allocation, copies, output clearing,
warmups, and CPU verification are excluded from runtime timing.

Generate the compile graph and every runtime graph together from any selected
combination of CUDA, PTX, and CUBIN CSVs:

```sh
.venv/bin/python bench/benchmark_graphs.py \
  --compile-csv \
    docs/compile_matrix_2026-07-23_ptx.csv \
    docs/compile_matrix_2026-07-23_cuda.csv \
    docs/compile_matrix_2026-07-24_cubin.csv \
  --runtime-packing-csv \
    docs/runtime_packing_sweep_2026-07-24_ptx.csv \
    docs/runtime_packing_sweep_2026-07-24_cuda.csv \
    docs/runtime_packing_sweep_2026-07-24_cubin.csv \
  --runtime-tile-csv \
    docs/runtime_tile_sweep_2026-07-24_ptx.csv \
    docs/runtime_tile_sweep_2026-07-24_cuda.csv \
    docs/runtime_tile_sweep_2026-07-24_cubin.csv \
  --runtime-surface-csv \
    docs/runtime_sse_surface_2026-07-24_ptx.csv \
    docs/runtime_sse_surface_2026-07-24_cuda.csv \
    docs/runtime_sse_surface_2026-07-24_cubin.csv \
  --backends ptx cuda cubin \
  --output-dir docs \
  --prefix secant_2026-07-24
```

Add `--log-scale` for logarithmic Y axes. Omit an input group to skip those
graphs, or pass a subset such as `--backends cubin` to isolate one backend.
The individual graph scripts accept the same `--backends` and `--log-scale`
options.

The RTX 5090 results are recorded in
[`docs/compile_matrix_2026-07-24.md`](../docs/compile_matrix_2026-07-24.md) and
[`docs/benchmark_2026-07-23.md`](../docs/benchmark_2026-07-23.md). Direct CUBIN
patch and runtime results are in
[`docs/cubin_benchmark_2026-07-24.md`](../docs/cubin_benchmark_2026-07-24.md).
The frozen kernel runtime sweep is defined in
[`docs/runtime_limit_protocol.md`](../docs/runtime_limit_protocol.md).

## Compile-to-Execution Frontier

`secant_end_to_end_bench` measures one AST batch through three distinct
phases:

1. AST-specific compilation or binary specialization.
2. Driver module load and function resolution.
3. Kernel execution with input and target arrays already resident on device.

Cold PTX/CUBIN/HSACO template creation, template inspection, template copying,
memory allocation, host-to-device transfers, correctness checks, and teardown
are outside these phase timings. CUBIN and HSACO compilation therefore mean
AST code generation plus in-place specialization of an already inspected
template. CUDA and HIP compilation mean AST-specific source generation plus
native compiler invocation. PTX compilation means AST injection and
nvPTXCompiler invocation after the PTX template handle has been created.

Generate one-core NVIDIA data with CUDA and PTX O0/O1 plus CUBIN:

```sh
python3 bench/end_to_end_sweep.py \
  --platform nvidia \
  --output docs/end_to_end_rtx5090_2026-07-26.csv
```

Generate the equivalent HIP O0/O1 plus HSACO data on the Radeon host:

```sh
python3 bench/end_to_end_sweep.py \
  --platform amd \
  --system 'Radeon RX 9070 XT' \
  --hip-arch gfx1201 \
  --output docs/end_to_end_rx9070xt_2026-07-26.csv
```

The sweep pins every benchmark subprocess and its compiler children to one
CPU, physically executes every listed row count, verifies each artifact
against the CPU backend, and writes both median and raw-sample CSVs. Seeds vary
across samples while every backend receives the same seed for a given row and
sample.

Render either platform independently:

```sh
uv run --with matplotlib python bench/end_to_end_graph.py \
  docs/end_to_end_rtx5090_2026-07-26.csv \
  --output docs/end_to_end_rtx5090_2026-07-26.svg
```

Backend color is fixed. Solid lines include AST compilation, module load, and
execution; dotted lines include execution only. O0 and O1 use different point
markers. Every point is an actual benchmark run rather than an extrapolated
row count.

## Concurrent Compile/Run Pipeline

`secant_pipeline_bench` overlaps CPU compilation with GPU execution across
many modules through the persistent CUDA, PTX, CUBIN, HIP, and HSACO runner
APIs. Each runner owns a fixed pthread worker pool and a bounded
compile-completion queue. The NVIDIA consumer uses
`CUDA_MODULE_LOADING=EAGER`:

1. Launch the current module across the pre-created streams.
2. Load the next completed CUDA, PTX, or CUBIN artifact while the current
   module executes.
3. Wait for the joined stop event and read its runtime.
4. Unload the current module.
5. Promote and launch the staged module.

The benchmark sets eager mode before CUDA initialization and verifies that the
driver reports `CU_MODULE_EAGER_LOADING`. CUDA function handles are enumerated
and mapped by their numeric name suffix, but the already-eager functions do
not receive redundant `cuFuncLoad` calls. At most two modules are resident.

HIP and HSACO use the same worker, queue, stream, and blocking `run_all`
contract. Their module transition remains wait, unload, then load; CUDA's
measured eager-loading overlap is not assumed to apply to ROCm. HIPRTC workers
own independent compile scratch, while HSACO workers specialize reusable
template slots in place.

Inputs and targets remain resident on the GPU. AST construction, handle
creation, template inspection, CUBIN template copies, memory allocation, data
transfers, and correctness verification are outside the pipeline timer. The
result reports both summed GPU-event runtime and the overlapped pipeline wall
time. Compile-window time is wall time until the last worker finishes;
because runners use bounded slot pools, it may include time workers spend
waiting for module consumption. Compile-critical time is the maximum
cumulative callback time of any worker and is used for compiler-capacity
AST/s. Compile-work time is the sum of callback time across all workers. Use
`secant_compile_bench` for isolated raw compiler throughput. Both materialize
and SSE kernels are supported.
Materialize writes distinct AST-row results within each module and reuses that
output region for later modules because the benchmark does not consume
materialized values. SSE retains each AST's reduction against the configured
targets.

The sweep defaults are 128 ASTs/kernel, 64 kernels/module, 48 modules, and 24
OpenMP workers. This runs 393,216 ASTs per campaign. SSE defaults to 8,192
rows/CTA. The default row range stops at 262,144 rows. Run the NVIDIA backends
with:

```sh
python3 bench/pipeline_sweep.py \
  --platform nvidia \
  --output docs/pipeline_rtx5090_2026-07-26.csv

.venv/bin/python bench/pipeline_graph.py \
  docs/pipeline_rtx5090_2026-07-26.csv \
  --output docs/pipeline_rtx5090_2026-07-26.svg
```

Run the corresponding AMD backends with:

```sh
python3 bench/pipeline_sweep.py \
  --platform amd \
  --benchmark /path/to/secant_pipeline_bench \
  --hip-arch gfx1201 \
  --opt-levels 1 \
  --output docs/pipeline_rx9070xt_2026-07-26.csv
```

The NVIDIA sweep executes CUDA O0/O1, PTX O0/O1, and CUBIN with eager module
streaming. The AMD sweep
executes HIP O0/O1 and HSACO. Both cover materialize/SSE and ALU/MUFU. Solid
graph lines are actual compile/run pipeline wall throughput; dotted lines are
GPU-event kernel throughput only. `cuModuleLoadData` remains a synchronous
host call, but already-enqueued CUDA work continues while that call blocks.
Its elapsed duration therefore overlaps GPU runtime and is not additive with
the other reported phases. Overall pipeline wall time and summed GPU-event
runtime are the primary measurements.

The benchmark does not issue a device-wide synchronization before the first
load or between normal module transitions. The joined stop event depends on
every launch stream; a module is unloaded only after that event completes.
The measured loading strategies and recommendation are recorded in
[`docs/cuda_eager_module_streaming_2026-07-27.md`](../docs/cuda_eager_module_streaming_2026-07-27.md).

An optional build-time CUBIN template can remove the cold NVRTC template build
from benchmark startup:

```sh
cmake -S . -B build -DSECANT_EMBEDDED_CUBIN_SM=120
cmake --build build --target secant_pipeline_bench -j
```

The generated CUBIN is compiled through the same NVRTC template path and
embedded with `incbin.h`. It is selected only when the requested SSE recipe
exactly matches the embedded SM, kernel count, AST capacity, input/target
counts, tile shape, thread count, and patch capacity. The result reports
`template_source=embedded_cubin` or `template_source=runtime_nvrtc`. Template
preparation remains outside the pipeline timer in both cases.

The current 64-kernel, 128-AST CUBIN tile sweep is in
[`docs/runtime_sse_tile_rtx5090_2026-07-26.svg`](../docs/runtime_sse_tile_rtx5090_2026-07-26.svg).
It measures 8,192 rows/CTA as the peak of the tested range. The current
pipeline SVGs use 8,192 rows/CTA and contain six measured row counts per
backend, kernel shape, and AST mode.

The raw-sample CSV is updated after every case. Re-running an interrupted sweep
with the same dimensions resumes missing cases; use `--force` to discard the
existing raw samples. `--opt-levels` restricts native compiler configurations.
At 128 ASTs/kernel, RX 9070 XT HIP O0 materialize MUFU generated about 35 KiB
of private storage per thread and exhausted ROCm's queue backing for the
262,144-row, eight-stream endpoint. The command above deliberately compares
HIP O1 and HSACO at the full shape; reducing the O0 workload is not treated as
an equivalent result.

Measured reports are in
[`docs/pipeline_rtx5090_2026-07-26.md`](../docs/pipeline_rtx5090_2026-07-26.md)
and
[`docs/pipeline_rx9070xt_2026-07-26.md`](../docs/pipeline_rx9070xt_2026-07-26.md).

## Current Backend Observation

On the Radeon RX 9070 XT (`gfx1201`, ROCm 7.1), direct HSACO patching reached
9.28 million materialize ALU AST/s on one CPU worker and 63.3 million AST/s
with 24 workers. An SSE MUFU configuration reached 3.49 million AST/s on one
worker and 41.2 million AST/s with 24 workers. Handle creation and HIPRTC
template compilation are outside these hot-path measurements.
