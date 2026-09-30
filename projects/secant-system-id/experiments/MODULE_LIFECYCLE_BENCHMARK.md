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

# Standalone CUDA module lifecycle benchmark

This benchmark isolates the single-context module pipeline used by Secant
System ID. It measures worker-owned CUBIN copying, module loading, function
lookup, kernel launch, event joining, completion polling, and module unloading
without trajectory integration or AST arithmetic.

The runner is plain C99 and dynamically loads the installed NVIDIA driver. The
fixture generator is Python because template generation and inspection remain
the Python-side setup boundary in the current project.

## Build

```sh
make module-lifecycle-benchmark
```

No CUDA headers or CUDA runtime library are required to build the C runner. It
uses the same driver-level operations as `csrc/ssid_pipeline.c`.

## Generate a fixture

The following creates an approximately 3.34 MB module with eight independently
resolvable entry points:

```sh
PYTHONPATH=src python3 -m secant_system_id.module_lifecycle \
  --target-cubin-bytes 3341248 \
  --padding-mode brkpt \
  --kernels 8 \
  --arch sm_120 \
  --output-directory generated/module_lifecycle/example_k8
```

`brkpt` is the code-heavy mode. The generator compiles one marker instruction
per function to PTX, expands the marker into repeated BRKPT instructions, then
assembles the CUBIN with the installed `ptxas`. Padding is distributed across
entry points. A safety guard rejects more than 65,536 BRKPT instructions in one
function because larger monolithic dummy functions caused pathological ptxas
memory use on Rohini.

`global` is the low-compile-cost image-size control:

```sh
PYTHONPATH=src python3 -m secant_system_id.module_lifecycle \
  --target-cubin-bytes 6674600 \
  --padding-mode global \
  --kernels 16 \
  --arch sm_120 \
  --output-directory generated/module_lifecycle/example_global_k16
```

Global-data padding is not equivalent to a code-heavy System ID CUBIN. It is
provided to distinguish raw image-size costs from executable-code metadata,
lookup, and unload costs.

## Run one fixture

The fixture manifest supplies the nonce offset and value:

```sh
build/module_lifecycle_benchmark \
  --cubin generated/module_lifecycle/example_k8/module_lifecycle.cubin \
  --kernel secant_module_lifecycle_kernel \
  --kernels 8 \
  --nonce-offset 7920 \
  --nonce-magic 0xd43c7a91e5b6028f \
  --modules 512 \
  --warmup-modules 8 \
  --workers 1 \
  --streams 4 \
  --loaded-modules 16 \
  --blocks 1 \
  --threads 32 \
  --wait-clocks 0 \
  --poll-us 0 \
  --identity unique
```

Loaded-module depth and stream count are independent. `--loaded-modules D`
limits retained in-flight `CUmodule` objects; it does not count functions. For
`K` entry points and `B` blocks per launch, the maximum represented work is:

```text
queued function launches = D * K
queued blocks             = D * K * B
dependency events/module  = min(K, stream_count)
```

Packing more functions therefore reduces the module depth needed to keep the
stream queue populated. Extra depth still retains more module handles, events,
and queued launches, so it should not be increased after throughput saturates.
The runner reports these effective queue quantities in every JSON result.

The sweep helper reads those manifest fields automatically:

```sh
python3 experiments/run_module_lifecycle_sweep.py \
  --runner build/module_lifecycle_benchmark \
  --fixture generated/module_lifecycle/example_k8 \
  --modules 512 \
  --workers 1,4 \
  --loaded-depths 16 \
  --stream-counts 1,2,4,8 \
  --wait-clocks 0,500000,2000000 \
  --identities unique,identical \
  --poll-us 0 \
  --output generated/module_lifecycle/example_sweep.json
```

## Kernel behavior

Every entry point:

1. loads a volatile gate from global memory;
2. exits immediately when gate bit zero is set;
3. writes the module nonce for a separate GPU identity check when bit one is
   set; and
4. reaches BRKPT padding only when both bits are clear.

`--wait-clocks N` changes the timed early-exit path into a runtime-selected
`clock64()` busy wait before returning. The same CUBIN is reused for every wait
duration. Zero retains the ideal early-exit path. A nonzero value keeps one
warp per launched block resident, but it does not reproduce useful-kernel
arithmetic, memory traffic, or register pressure. Use it to measure lifecycle
overlap and latency masking, not as a substitute for a real RK4 kernel.

The timed path sets the gate to one. On `sm_120`, disassembly confirms an
`LDG.E.STRONG.SYS`, a predicated `EXIT`, a branch around the verification path,
and unreachable `BPT.TRAP` padding.

## Cache controls

- `--identity identical` copies and loads the exact same image repeatedly. It
  is an explicit cache-favorable control.
- `--identity unique` patches a different 64-bit constant-data nonce into each
  worker-owned image. The preflight kernel reads the patched nonce back on the
  GPU, and the report records differing whole-image hashes.

Unique mode makes every submitted CUBIN byte-distinct and semantically distinct,
but the SASS instruction bytes remain the same. A real Secant specialization
changes SASS as well, so production unique-CUBIN sweeps remain the final
cache-proof comparison.

## Lifecycle parity with System ID

The C runner mirrors the current runtime topology:

- one dedicated CUDA context;
- persistent nonblocking streams;
- one fixed-size CUBIN copy per POSIX worker;
- worker reuse immediately after `cuModuleLoadData` returns;
- all populated functions resolved per module;
- functions distributed round-robin across streams;
- one dependency event per used stream and one joined completion event;
- later modules loaded while earlier launches remain in flight; and
- module unload only after completion is observed.

`--poll-us 0` is the ideal busy-poll ceiling. `--poll-us 1000` approximates the
current runtime's one-millisecond timed wait when no completion is immediately
available.

## Timed and excluded work

Included in the reported wall time:

- worker image copy and unique nonce patch;
- module load;
- lookup of every entry point;
- event creation;
- all kernel launches and dependency joins;
- completion queries and optional polling delay;
- event destruction; and
- module unload.

Excluded:

- fixture compilation;
- process startup and CUDA context creation;
- stream and device-buffer allocation;
- the preflight GPU identity check;
- untimed warmup modules; and
- useful AST evaluation or trajectory integration.

The fixture manifest and every sweep report contain a `material_deviations`
section. Do not compare global-data padding, identical-image controls, or
resident-only kernel timings as though they were equivalent to byte-distinct
code-heavy lifecycle measurements.
