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

# Module Loading, Eager Mode, and Sustained Throughput in Secant

## The lifecycle problem

Direct SASS specialization removes native compilation from Secant's per-AST
path, but a specialized CUBIN must still become an executable CUDA module.
`cuModuleLoadData` is a synchronous host call, function residency may add more
driver work, and a module cannot be unloaded until every kernel using it has
finished. A serialized load-launch-wait-unload loop can therefore leave the GPU
idle even when specialization and the kernels themselves are fast.

This distinction is important:

- **Kernel throughput** measures GPU execution after a module is resident.
- **Pipeline throughput** includes specialization, module loading, execution,
  completion waits, and unloading.

Module scheduling does not make an individual kernel faster. It keeps the GPU
running close to its runtime-only rate by hiding lifecycle work behind useful
execution.

## Why Secant requires eager loading

CUDA supports lazy and eager module loading. In lazy mode,
`cuModuleLoadData` may leave individual functions unresolved until
`cuFuncLoad` or their first launch. A module containing many generated kernels
can therefore expose two separate host-side phases: loading the module and
making all of its functions resident. First-launch loading is especially
undesirable because it inserts unpredictable stalls into the execution path.

With `CUDA_MODULE_LOADING=EAGER`, `cuModuleLoadData` performs the module and
function-residency work in one synchronous host call. Although that call blocks
the calling CPU thread, previously enqueued GPU work continues to execute. This
makes the complete loading cost of the next module overlapable with the current
module's kernels.

Eager loading and execution overlap are separate mechanisms. Eager mode alone
would still leave the GPU idle if the runner waited for loading to finish before
launching useful work. Secant explicitly launches module A first and then calls
`cuModuleLoadData` for module B while A is executing. The load is not an
asynchronous background driver API—the host thread is blocked inside the
driver—but it proceeds concurrently with already-enqueued work on the GPU.
Eager mode is valuable because it combines module loading and function
residency into one host phase that this scheduler can hide.

Eager mode must be selected before CUDA initializes, for example:

```sh
CUDA_MODULE_LOADING=EAGER ./application
```

The Secant runner queries `cuModuleGetLoadingMode` during creation and returns
`SECANT_ERROR_EAGER_LOADING_REQUIRED` instead of silently running with a
different lifecycle.

## The streaming scheduler

The production runner uses persistent specialization workers, a ready queue,
one active module, and one staging module:

1. CPU workers specialize independent CUBIN copies in reusable slots and
   publish completed modules to a ready queue.
2. The GPU consumer eagerly loads the first ready module and enumerates its
   functions.
3. Its kernels are distributed across pre-created nonblocking CUDA streams.
4. While those kernels execute, the host calls `cuModuleLoadData` for the next
   ready module. The host is blocked, but the GPU continues running the active
   module.
5. Per-stream completion events are joined onto stream zero. After the active
   module's stop event completes, the runner records its elapsed GPU time and
   unloads it.
6. The staged module is launched immediately, and the process repeats.

At steady state, three different generations of work coexist:

```text
CPU workers:       specialize modules C, D, ...
Host loader:       eagerly load module B
GPU streams:       execute module A
```

The runner allocates `num_workers + 2` reusable slots so specialization can
remain ahead of the consumer. Only the active and staging modules need to be
loaded, avoiding the memory cost of preregistering an entire campaign.

## Keeping GPU execution efficient

Lifecycle overlap works because the kernel side also supplies enough useful
work:

- **Compiler-optimized skeletons.** CUDA compilation still controls data
  movement, tiling, synchronization, and reductions. Secant changes only the
  expression instructions.
- **Native register-local ASTs.** Specialized expressions execute as SASS with
  register intermediates. A branch skips unused patch capacity.
- **AST packing.** Many ASTs are packed into each kernel and many kernels into
  each module, amortizing one load over substantial work.
- **Multiple streams.** Kernels are assigned round-robin across pre-created
  streams, giving the GPU independent work and avoiding a serial launch chain.
- **Resident data.** Inputs, targets, settings, and output storage are prepared
  before module streaming. Module transitions do not recopy the dataset.
- **Fused outputs.** SSE, affine, Gram, and optimizer shapes reduce inside the
  kernel instead of writing and rereading full intermediate feature matrices.
- **Runtime settings.** One resident AST kernel can evaluate many column
  bindings and constant values, increasing work per specialization and load.
- **Batch reuse.** Ordinary modules are specialized and loaded once, then
  launched across all requested datasets or trajectories before unloading.
  Optimizer modules remain resident across iterations whenever their code does
  not change.
- **Persistent resources.** Worker threads, streams, events, binary buffers,
  and pinned feedback storage are created with the runner rather than allocated
  in the execution loop.

The scheduler therefore hides module preparation, while packing, settings,
resident data, and fusion make each residency interval long enough to hide it.

## Measured effect

An RTX 5090 probe streamed 128 modules containing 1,048,576 total ASTs, with 64
kernels per module, 128 ASTs per kernel, eight streams, and 131,072 rows.

| Strategy | One pass per module | Eight passes per module |
|---|---:|---:|
| Lazy serialized | 1,830.0 ms | 6,843.8 ms |
| Lazy module-load overlap | 1,195.6 ms | 6,207.5 ms |
| Lazy split-work overlap | 946.3 ms | 5,957.1 ms |
| Eager streaming | 878.5 ms | 5,894.2 ms |
| GPU event time | 719.0 ms | 5,734.4 ms |

The lazy module-load result shows that overlap is independently useful: it
removed 634.4 ms, about two-thirds of the total one-pass improvement from lazy
serialized execution to eager streaming. Lazy split-work went further by
placing module loading and explicit function residency behind separate batches
of GPU work. It came within 7.7% of eager streaming for one pass and 1.1% for
eight passes, but required more launches, events, and scheduler complexity.

Eager streaming reduced one-pass wall time by 52% and raised GPU-time
utilization from 39.3% to 81.8%. With eight passes, it reached 97.3%. The
residual non-GPU difference was about 1.25 ms per module, or 0.15 microseconds
per AST. The peak number of loaded modules remained two. These results show
that overlap is the fundamental throughput mechanism; eager mode makes the
entire residency cost simpler and more predictable to overlap.

In the larger SSE campaign at 262,144 rows, GPU execution was effectively
unchanged—0.3572 seconds before and 0.3586 seconds afterward—while pipeline
wall time fell from 2.6680 to 0.8251 seconds. Compile-inclusive throughput rose
from `1.932e11` to `6.246e11` row evaluations per second. The improvement came
from eliminating exposed lifecycle time, not from changing the kernel.

## Measurement and portability contract

Secant reports specialization work, module-load time, completion waits,
module-unload time, GPU-event runtime, and total wall time separately. Load
durations may overlap execution and must not be added as if they were disjoint
phases; total pipeline wall time is the authoritative end-to-end measurement.

The observed overlap is a measured driver behavior, not a promise that every
future CUDA driver, GPU, or operating system will behave identically. Secant
therefore checks the active loading mode and benchmarks both runtime-only and
full-pipeline throughput. If eager loading is unavailable, a lazy scheduler can
split useful work to overlap module load and explicit function residency, but
it requires more launches and remained slower in the measured experiments.
