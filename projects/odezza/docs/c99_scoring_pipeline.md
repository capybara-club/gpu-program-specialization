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

# C99 scoring pipeline

## Supported lifecycle

The supported public lifecycle is deliberately narrow:

```text
higher layer makes a compatible CUDA context current on caller thread
  -> create Odezza handle from explicit SM + reusable capacities + active counts + fixed RHS ASTs
     -> generate CUDA
     -> NVRTC compile directly to native CUBIN
     -> inspect physical patch sites
     -> specialize the fixed RHS into an immutable retained CUBIN
     -> create CPU workers and bounded queues
     -> create persistent streams and events in the current context
  -> query exact steady-state workspace
  -> bulk-run candidate system ASTs with that same context current
     -> specialize/load/execute/unload on the caller thread
  -> destroy handle with that same context current
     -> destroy persistent streams and events
  -> higher layer releases or destroys its context
```

There is no public sequence in which an application generates source, manages an NVRTC object, inspects a CUBIN, patches individual modules, or submits internal tickets. Those pieces exist behind the handle and in a private test header only.

CUDA and NVRTC are mandatory linked dependencies. The pipeline calls `cuModuleLoadData`, `cuModuleGetFunction`, `cuLaunchKernel`, event/stream functions, and `cuModuleUnload` directly. It does not use `dlopen`, symbol probing, a disabled-CUDA compilation mode, or any CUDA context-management function.

## Ownership

The caller owns:

- CUDA context selection and lifetime for handle creation, every run, and destruction;
- AST byte spans for the duration of a synchronous run;
- constant, reference, and output device allocations;
- the queried run workspace.

The handle owns:

- generated source and the compiled, fixed-prespecialized template CUBIN;
- copied physical inspection metadata;
- worker threads and bounded index queues;
- nonblocking streams and completion events created once in the caller's current context;
- a 512-byte last-error buffer.

Handle creation may allocate and creates the persistent CUDA resources. Bulk execution allocates nothing through Odezza and creates no streams, events, or other persistent CUDA resources. The run workspace stores `worker_count * slots_per_worker` exact CUBIN copies followed by one specialization workspace per slot. Each worker resets a free slot from the retained template, performs transactional AST assembly and patching, and passes that same memory to `cuModuleLoadData`. No queue copies a CUBIN.

## Concurrency

Specialization workers and the synchronous run caller share one bounded slot pool. The workers perform CPU-only specialization and never touch CUDA state. The calling thread performs every stream, event, module, and launch operation under the context supplied implicitly by the higher layer. A CUBIN slot becomes free immediately after `cuModuleLoadData` returns because the driver has consumed the image. The loaded module then executes independently on an internal stream until its event completes; only then is it unloaded.

The loaded-module limit and stream count are distinct from host CUBIN depth. They are selected internally; only private diagnostics can override them for controlled sweeps. For the short trajectory kernel measured so far, the practical knee has generally been:

```text
specialization workers: 1
CUBIN slots per worker: 2
execution streams:      2
loaded-module limit:    2
```

This is a measured starting point, not an ABI assumption. Large CUBINs can make concurrent module loading serialize with execution, and longer kernels can benefit from different depth. The internal benchmark reports module, system, and configuration throughput plus specialization/load/unload time and peak in-flight modules.

## Eager loading

`CUDA_MODULE_LOADING=EAGER` is process policy and must be established by the higher layer before CUDA initialization. Handle creation checks `cuModuleGetLoadingMode` before creating the persistent streams and events. Odezza does not mutate the environment or silently accept lazy mode, because that changes where loading work is paid and makes the specialize/load/execute/unload pipeline incomparable.

## Bulk partitioning and output

One generated module can hold `system_capacity` candidate systems. A public run may provide any positive system count. Odezza partitions the input into module-sized batches and adjusts system-major constant and output device pointers for each batch. Output remains `[system, configuration]` across the whole request.

All tickets in a wave complete before the next wave reuses the handle's bounded ticket set. The synchronous call returns only after every submitted module has completed and unloaded. The returned time covers specialization, reset copies, module load, lookup, launch/execution, event retirement, and module unload. It excludes handle creation/compilation and one-time stream/event creation, caller context management, device allocation/upload, and result download.

## Reusable template and runtime trajectory ABI

Native CUDA compilation depends on six dimensions: SM version, state capacity, constant capacity, system capacity, shared patch capacity, and per-system patch capacity. It does not depend on state names, active counts, trajectory count, point count, offsets, times, observations, RK4 step count, constant-bank count, toggle bit positions, or toggle wrapping. A future raw-CUBIN cache can therefore key on those six compiled dimensions; fixed RHS programs are applied afterward to form a handle-specific prespecialized CUBIN.

Patch capacity is the dominant cold-compile dimension. On the RTX 5090, a 256-system template with 64 shared and 64 per-system instruction slots produced a roughly 296 KB CUBIN and cold-created in about 2.6 seconds. Raising both budgets to 384 produced a 1.62 MB CUBIN and took about 81 seconds. The internal tools therefore default to 64; callers must choose a larger explicit bucket only when AST analysis shows it is needed. Odezza reports specialization capacity failure and never silently changes the bucket.

The run-time trajectory representation is ragged dense-state data:

```text
trajectory_offsets[trajectory_count + 1]
trajectory_times[trajectory_point_count]
reference[active_state_count][trajectory_point_count]
```

Offsets partition concatenated nonempty trajectories. Each first point initializes its trajectory, later points contribute to MSE, and adjacent times determine the RK4 interval. All active states are observed at every point. The complete representation is staged in dynamic shared memory per CTA; launch fails rather than silently choosing another path when it exceeds the device limit.

Each configuration selects a toggle permutation from its low bits and a constant bank from the remaining high bits. An AST toggle's bit index is specialized directly into its SASS bit-test mask; no runtime permutation table or modulo is needed. The active toggle-bit count is a runtime uniform used by the constant-bank shift, so changing between 2, 4, 8, or 4,096 permutations reuses the same CUBIN. There is no per-CUBIN toggle capacity; the common ABI exposes all 32 bits of the configuration word. With one active bit, even configurations select choice zero, odd configurations select choice one, and each adjacent pair shares a constant bank. Reusing an index couples sites. Inactive state and constant capacity registers remain available to the SASS assembler as temporary registers.

This replaces the earlier runtime shift-array launch ABI. Throughput measurements from that older path and from templates that exposed toggle bits as floating-point input registers are not directly comparable and must be rerun with specialized per-site masks and the uniform constant-bank shift.

## Errors and failed creation

Every public function returns `OdezzaResult`. A post-allocation create failure leaves a diagnostic handle in `pipeline_ret`; only error retrieval and destruction are valid on it. Runtime failures copy the first consequential ticket error into the handle. A second call overwrites the previous message, so callers that need a durable log should copy it after a failure.

Register pressure, patch capacity, invalid AST encoding, inconsistent RHS coverage, NVRTC failure, non-eager loading, CUDA failure, undersized/misaligned workspace, and concurrent use are distinct results or diagnostics. None silently falls back to a different kernel or execution path.

## Validation boundary

The repository keeps low-level generation, inspection, and specialization contract tests through `o_odezza_internal.h`, plus a real public-handle CUDA smoke test. Its higher-level harness activates a primary context, creates the pipeline and its persistent streams/events, executes the run, destroys the pipeline under the same context, and requires the planted system to replay near machine precision.

On Ada's current CUDA 13.1 stack, the non-GPU contract tests pass under AddressSanitizer plus UndefinedBehaviorSanitizer, but an ASan-instrumented executable cannot retain the CUDA primary context, with or without leak detection. The CUDA smoke therefore runs in the strict release build on both Ada and Rohini; this sanitizer/runtime incompatibility is not treated as a passing GPU sanitizer result.

The first remote verification attempt for the generic-toggle revision copied changed directory contents into the repository root instead of their corresponding subdirectories, so that build used the previous source. Its results are invalid for this revision and were not retained as final measurements. The stray copies were removed, the exact subdirectories were resynchronized, and the rebuilt revision passed all contract/CUDA tests plus the same-pipeline 1/2/3/12-bit runtime-width regression test.

Historical reports written for the earlier ticket-level public API remain useful for kernel-packing and module-depth observations, but their host API is obsolete. New comparisons must use the same systems/module, configurations/system, CUBIN capacities, RK4 work, and timing boundary before claiming a throughput change.
