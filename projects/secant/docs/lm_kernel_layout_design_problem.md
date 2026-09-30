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

# LM Kernel Layout Design Problem

## Purpose

This document is a self-contained design brief for deciding how Secant should
execute Levenberg-Marquardt (LM) constant optimization on an NVIDIA GPU. It is
intended to be given to an LLM or engineer with no prior Secant context.

The central question is:

> Should one CUDA function contain one specialized AST and loop over many LM
> settings, or should it contain several independent AST patch sites so those
> ASTs can share one row tile and one launch?

The answer must account for registers, occupancy, shared-memory traffic,
atomics, instruction and module size, useful setting counts, module loading,
and concurrent CUDA streams. The goal is not merely the highest reported
row-evaluation rate. The useful metric is end-to-end LM progress across a large
population of unrelated ASTs.

## Secant Background

Secant generates a CUDA skeleton, compiles it once to a CUBIN, inspects the
CUBIN, and then specializes postorder AST bytecode directly into reserved SASS
regions. Specialization is much faster than compiling each expression with
NVRTC or PTXAS.

A module normally contains many CUDA functions. Each function can contain one
or more reserved AST sites. The runner:

1. Specializes many ASTs into a copy of the skeleton CUBIN.
2. Loads the module eagerly.
3. Launches its functions over several nonblocking CUDA streams.
4. Keeps the module resident while its useful work is issued.
5. Synchronizes and unloads only at a module transition.

"One AST per kernel" must therefore be interpreted carefully. It can mean one
AST per CUDA function while still having 64-128 functions in one module and
many CTAs per function. It must not mean one AST per module or one CTA total.

## Existing Tile-Static Dynamic Kernel

Secant's current dynamic-leaf and dynamic-constant SSE kernels use this general
mapping:

```text
grid.x                     = row tiles
thread within a CTA        = one runtime setting
shared memory              = cached input columns and targets for one row tile
thread-local registers     = SSE accumulators for its setting
```

Conceptually:

```cpp
load the CTA's row tile into shared memory;

for (setting = threadIdx.x;
     setting < num_settings;
     setting += blockDim.x) {
    initialize setting registers;
    initialize SSE accumulators;

    for (row = 0; row < valid_rows; ++row) {
        load the row's values from shared memory;
        evaluate specialized ASTs;
        accumulate SSE;
    }

    atomically add this tile's partial SSE;
}
```

This shape is fast because thousands of settings reuse each globally loaded
row tile. Settings are parameter vectors or dynamic leaf assignments; they are
not rows.

## Proposed LM Work

Each AST has up to `K = 8` optimizable constants. The AST would be specialized
as a forward-mode automatic-differentiation program. For every row and setting,
it produces a prediction and `K` derivatives.

For one LM setting, a CTA tile must accumulate:

```text
SSE                                  1 value
J^T r                                K values
upper triangle of J^T J              K(K + 1) / 2 values
```

At `K = 8`, this is:

```text
1 + 8 + 8 * 9 / 2 = 45 f32 statistics
```

It is convenient to pad this to 48 floats for addressing and alignment. A
thread that owns one setting therefore needs approximately:

```text
45 LM accumulator registers
8 derivative registers
AST primal and scratch registers
constant, pointer, and loop registers
```

The resulting total may be roughly 70-110 registers per thread, depending on
the AST. This is an estimate and must be measured from the specialized binary.

After all row-tile CTAs have contributed, another kernel solves the small
damped normal equations for every AST and setting. A proposed constant vector
is evaluated in another statistics pass. Acceptance or rejection updates the
LM damping state. The module should remain resident across these iterations.

## Shared-Memory Budget

Assume up to 32 input columns, one target, and an odd shared-memory column
stride used to avoid systematic bank conflicts. A representative allocation is
34 floats per row:

| Tile rows | Approximate shared memory |
| ---: | ---: |
| 128 | 17,408 bytes |
| 256 | 34,816 bytes |
| 512 | 69,632 bytes |
| 1,024 | 139,264 bytes |

A 128-row tile should permit more resident CTAs. A 256-row tile halves the
number of atomic contributions and amortizes tile loading better, but consumes
twice the shared memory. Both must be tested. A 1,024-row tile is unlikely to
be the right default with 32 columns because it severely restricts occupancy.

## Candidate A: One AST Site Per CUDA Function

All CTA threads evaluate settings for the same AST:

```cpp
load row tile into shared memory;

for (setting = threadIdx.x;
     setting < num_settings;
     setting += blockDim.x) {
    initialize 45 statistics;

    for (row = 0; row < valid_rows; ++row) {
        evaluate one forward-AD AST;
        accumulate SSE, JTr, and JTJ;
    }

    atomicAdd 45 tile statistics;
}
```

Advantages:

- One patch site and the smallest CUBIN function.
- Straightforward inspection and specialization.
- All threads follow the same AST instruction stream.
- Registers represent one AST and one setting, not a packed cohort.
- One row traversal per setting.

Disadvantages:

- Every AST function loads its own copy of the row tile from global memory.
- The function needs enough settings to occupy all 128 threads usefully.
- If LM normally uses only 1-8 starting points per AST, most lanes are idle.
- Many function launches are required, although many functions can share a
  module and be queued over several streams.

This is the clean reference implementation and may be optimal when hundreds of
LM starts per AST are algorithmically useful.

## Candidate B: Sequential Independent AST Sites

One CTA loads the row tile once. All CTA threads then evaluate AST 0 over their
settings and rows, flush its 45 statistics, reuse the same registers for AST 1,
and continue through several independent sites:

```cpp
load row tile once;

evaluate all settings and rows for AST site 0;
atomically flush AST 0 statistics;

evaluate all settings and rows for AST site 1;
atomically flush AST 1 statistics;

...
```

Advantages:

- One global tile load is amortized over several ASTs.
- One function launch performs more useful work.
- Statistic and AST scratch registers can be reused between sites, so register
  count should be determined by the largest site rather than the sum of sites.

Disadvantages:

- Shared memory is traversed again for every AST.
- Each site needs its own patch region, entry scheduling, and branch over
  unused capacity.
- Function and CUBIN size grow with site count.
- The CTA processes sites serially, which increases its duration and may make
  load balancing less flexible.

The repeated shared-memory traversal is probably acceptable for LM. It is much
cheaper than repeating global column loads, and forward AD plus 45 statistic
updates creates substantial arithmetic work per row.

## Candidate C: One Independent AST Site Per Warp

With 128 threads, the CTA has four warps. Each warp selects a different AST
site, and each lane owns one setting for that AST:

```text
warp 0 -> AST 0, settings 0-31
warp 1 -> AST 1, settings 0-31
warp 2 -> AST 2, settings 0-31
warp 3 -> AST 3, settings 0-31
```

For more than 32 settings, each warp advances by 32:

```cpp
warp_id = threadIdx.x / 32;
lane = threadIdx.x % 32;

select AST site once using warp_id;

for (setting = lane; setting < num_settings; setting += 32) {
    initialize 45 statistics;

    for (row = 0; row < valid_rows; ++row) {
        evaluate this warp's forward-AD AST;
        accumulate LM statistics;
    }

    atomically flush statistics;
}
```

Advantages:

- Four ASTs share one global tile load and one function launch.
- The four ASTs execute concurrently rather than serially.
- Each warp takes a uniform branch before its row loop.
- Only 32 settings per AST are needed to fill a warp.
- The same scratch-register range can potentially be used by every site.

Disadvantages:

- Four independent SASS sites are needed.
- Instruction footprint is larger and may affect the instruction cache.
- Each warp independently traverses the same shared-memory tile.
- An AST-tail branch and conservative load wait remain in each site's hot row
  loop.
- Fewer than 32 useful settings per AST still leaves lanes idle.

This is currently the strongest candidate when the optimizer has at least 32
useful starts or parameter settings per AST.

A possible extension assigns two ASTs sequentially to each warp, giving eight
sites per function. That further amortizes the tile load but doubles each
warp's shared-memory traversal and code footprint.

## Candidate D: CTA-Selected AST Site

Another option is to use a second grid dimension or `blockIdx` to select one of
several AST sites. Every CTA then works on only one AST, but multiple ASTs share
one function launch:

```text
blockIdx.y selects AST site
blockIdx.x selects row tile
all CTA threads process settings for the selected AST
```

This amortizes launch and module metadata but does not share the row tile across
ASTs: each AST's CTA loads the data again. It is useful as a control experiment
and may simplify scheduling, but it sacrifices the primary reason to introduce
multiple sites.

## Historical Multi-Site Evidence

An earlier, much cheaper cuSR SSE experiment compared separate AST sites with
one pooled site on an RTX 5090:

```text
separate sites: approximately 3.695e12 row-evaluations/s
one pooled site: approximately 4.720e12 row-evaluations/s
```

The pooled form was about 28% faster, while release-mode patching improved from
approximately 0.557 to 0.476 microseconds per AST. The causes included fewer
site waits, fewer branches, and simpler patch bookkeeping.

This result must not be treated as proof that independent LM sites are wrong.
The old SSE body did relatively little work per row. Forward AD and 45 LM
updates substantially reduce the fraction of runtime represented by one branch
or conservative wait. The new layouts require direct measurement.

## Atomic Layout

The simplest design directly atomically accumulates every tile's 45 statistics
into global memory. The preferred physical layout is statistic-major within a
warp-sized setting group:

```text
[AST][setting_group][statistic][lane]
```

where:

```text
setting_group = setting / 32
lane          = setting % 32
statistics    = 48 padded floats
```

For each statistic, the 32 lanes then atomically update contiguous addresses.
A warp issues 45 coalesced sets of atomic operations after completing the tile.

At 128 tile rows, this is 45 atomic contributions per 128 setting-row
evaluations. LM performs enough arithmetic that direct atomics may remain a
minor cost. Existing Secant SSE kernels also found direct atomic reduction
preferable to a workspace and separate reduction kernel.

If contention is unexpectedly significant, use 2-4 striped statistic bins per
AST and setting, followed by a small reducer. Do not start with this added
complexity without a measurement showing that atomics are limiting throughput.

## Occupancy Considerations

The initial configuration should use 128 threads per CTA. With approximately
70-110 registers per thread, 256 threads may make the register allocation per
CTA too large and sharply reduce resident blocks. Sixty-four threads may expose
too little parallelism and too few settings.

The actual decision requires these measurements from the specialized CUBIN:

- Registers per thread.
- Shared memory per CTA.
- Active CTAs and warps per SM.
- Local-memory spills.
- Instruction count and function size.
- Runtime for ALU-heavy, MUFU-heavy, shallow, and deep ASTs.

The patcher must also ensure that adding sites does not conservatively increase
the function register count as if every site's scratch range were distinct.
Mutually exclusive sites should use the same scratch-register pool where safe.

## CUDA Streams And Module Residency

Streams cannot repair poor occupancy in a kernel that already has a large grid,
but they can interleave many small function launches when row counts are low.

Recommended initial runner policy:

```text
functions per module: 32-64 for four-site functions
ASTs per module:      128-256
streams:              8, with 4 and 16 included in the sweep
module loading:       eager
module lifetime:      all LM iterations for that module
```

Functions should be distributed round-robin across streams without a device
synchronization between ASTs. An LM iteration does require a dependency between
statistics generation and the batched solve/update kernel. Use CUDA events and
stream waits rather than a host or device-wide synchronization where possible.

For example:

```text
statistics kernels on worker streams
        -> record worker events
control stream waits for those events
        -> batched LM solve/update
        -> record update event
worker streams wait for update event
        -> next statistics pass
```

There should be one such dependency boundary per LM iteration, not one per AST.

## Important Algorithmic Qualification

Settings represent independent LM parameter vectors or starting points. GPU
occupancy alone is not a reason to evaluate hundreds of starts per AST if the
search algorithm only benefits from one to eight.

This creates three operating regimes:

1. **At least 128 useful settings per AST:** one site may be sufficient and is
   the cleanest baseline.
2. **At least 32 useful settings per AST:** four warp-owned sites can fill a
   128-thread CTA while sharing the row tile.
3. **Only 1-8 useful settings per AST:** both mappings waste lanes. A different
   row-parallel or AST-parallel layout may be required, or LM should be applied
   only to enough shortlisted ASTs simultaneously to fill the machine.

The benchmark must therefore report optimization quality per unit time in
addition to raw row-evaluation throughput. Artificially adding useless settings
can make the kernel look fast while making the optimizer less efficient.

## Required Benchmark Matrix

Hold total AST-setting-row evaluations constant where possible and sweep:

```text
AST sites per function:       1, 2, 4, 8
site mapping:                 sequential, warp-owned, CTA-selected
threads per CTA:              64, 128, 256
tile rows:                    128, 256
settings per AST:             1, 4, 8, 32, 64, 128, 256, 512
constants K:                  2, 4, 8
AST class:                    shallow ALU, deep ALU, MUFU, imbalanced
rows:                         4,096; 10,000; 65,536; 262,144
streams:                      1, 4, 8, 16
modules:                      enough to expose load and pipeline costs
```

Report:

- Runtime row-evaluations per second.
- AST-setting evaluations per second.
- Complete LM iterations per second.
- Time to reach a fixed constant-fitting error.
- Statistics-kernel time, solve time, and dependency-wait time.
- Specialization, module-load, and end-to-end pipeline time.
- Register count, spills, shared memory, and occupancy.
- CUBIN bytes and specialization cost per AST.
- Correctness against a CPU double-precision oracle.

## Decision To Make

The initial implementation should probably use:

```text
four warp-owned independent AST sites
128 threads
128- or 256-row tiles
32-256 settings per AST
48-float padded direct-atomic statistics
32-64 functions per module
8 CUDA streams
```

However, this is a hypothesis, not an established result. The one-site version
must be implemented as the reference, and fixed-work comparisons must determine
whether four independent sites recover more from tile and launch reuse than
they lose to additional branches, waits, code size, and shared-memory traffic.

An independent analysis should answer:

1. Which layout maximizes useful LM steps per second, not merely row-evaluations
   per second?
2. At what setting count does each layout saturate the RTX 5090?
3. Does a four-site warp mapping retain enough occupancy with 45 live
   statistics per thread?
4. Does direct atomic accumulation remain faster than striped workspace bins?
5. Is 128 or 256 tile rows the better balance of atomics, shared memory, and
   occupancy?
6. How many sites can be added before instruction footprint and patch-site
   overhead erase tile reuse?
7. What layout should be used when the algorithm needs only 1-8 LM starts per
   AST?

