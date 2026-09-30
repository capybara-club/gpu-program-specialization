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

# Dynamic shared-memory leaf bindings

This experiment measures the cost of using per-setting state bindings inside
the score-only RK4 kernel. It keeps the rate-law skeleton fixed while allowing
each thread to select the state value used by eight leaf sites.

## Kernel shape

Each thread loads and retains:

- six FP32 rate constants; and
- eight unsigned state indices.

At every RK4 stage it writes its current `[X, S1, S2, P]` into a state-major
shared-memory tile:

```text
state_tile[state_index * blockDim.x + threadIdx.x]
```

It then dynamically gathers eight leaves and evaluates:

```text
specific_growth_1 = c0*leaf0 / ((leaf1 + c1*leaf2)*(1 + c2*leaf3))
specific_growth_2 = c3*leaf4 / ((leaf5 + c4*leaf6)*(1 + c5*leaf7))
growth_1 = specific_growth_1 * X
growth_2 = specific_growth_2 * X
```

State indices are `0=X`, `1=S1`, `2=S2`, and `3=P`. Setting zero uses the
correct binding vector:

```text
[1, 1, 0, 2,  2, 2, 0, 1]
```

Other settings vary the binding indices. Product may appear in numerator and
inhibition leaves; the two additive-denominator inputs are restricted to
`X/S1/S2` to avoid creating an initial zero denominator solely as a benchmark
artifact. Every setting executes the full integration even if its trajectory
later becomes invalid.

No CTA barrier is required because a thread only reads the four locations it
wrote itself. For CTA sizes divisible by 32, the state-plane stride is also a
multiple of the 32 shared-memory banks:

```text
(binding * blockDim.x + lane) mod 32 = lane
```

Consequently, different bindings in every lane remain bank-conflict-free.

## Resources and generated code

Both `sm_89` and `sm_120` CUBINs use:

- 40 registers/thread, versus 38 for fixed leaves;
- zero stack, spills, and local memory;
- zero barriers; and
- `4 * blockDim.x * sizeof(float)` dynamic shared memory.

This is 512 B, 1 KiB, 2 KiB, or 4 KiB for 32, 64, 128, or 256 threads.

The `sm_89` SASS contains 14 global setting loads before the integration loop:
six constants and eight bindings. The RK4 body contains four `STS` and eight
indexed `LDS` instructions per stage, or 16 stores and 32 loads per RK4 step.
There are no global setting reloads inside the loop.

For a 192-step trajectory, each thread therefore executes 3,072 shared stores
and 6,144 indexed shared loads. This store/gather dependency, not occupancy or
bank conflicts, is the principal new cost.

## Sustained comparison

The apples-to-apples sustained comparison used 262,144 settings, 128 threads
per CTA, 786,432 trajectories per launch, and the same architecture-specific
CUBIN loading and timing boundary.

| GPU | Kernel | ms/launch | Trajectories/s | Time/trajectory | Time/3 trajectories |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 5090 | Fixed leaves | 0.3809 | 2.065e9 | 0.484 ns | 1.453 ns |
| RTX 5090 | Dynamic shared leaves | 0.9998 | **7.866e8** | **1.271 ns** | **3.814 ns** |
| RTX 4090 | Fixed leaves | 0.5085 | 1.546e9 | 0.647 ns | 1.940 ns |
| RTX 4090 | Dynamic shared leaves | 1.3282 | **5.921e8** | **1.689 ns** | **5.067 ns** |

The dynamic kernel retains 38.1% of fixed-leaf throughput on the 5090 and
38.3% on the 4090: a **2.62x slowdown on both GPUs**.

The best shorter 5090 sample reached 8.204e8 trajectories/s, or 1.219 ns per
trajectory and 3.657 ns per complete three-trajectory setting. CTA size was not
important: 32, 64, and 128 threads were within approximately 0.3% in the same
sweep, while 256 threads was about 1.6% slower.

With 80 RK4 steps per observation interval:

| GPU | Fixed leaves trajectories/s | Dynamic leaves trajectories/s | Slowdown |
| --- | ---: | ---: | ---: |
| RTX 5090 | 4.514e8 | 1.634e8 | 2.76x |
| RTX 4090 | 3.124e8 | 1.187e8 | 2.63x |

## Correctness

The planted binding and published constants passed all three trajectory checks
on both GPUs. At 16 steps per observation interval, the maximum final-state
error against the same-step double CPU reference remained `4.14e-6` absolute
and at most `2.63e-7` after scaling by `max(abs(reference), 1)`.

## Interpretation

Dynamic shared-memory bindings remain extraordinarily fast, but they are not a
small perturbation to this compact rate law. The fixed kernel has very little
work besides arithmetic, so adding 9,216 shared-memory operations per
trajectory is visible even though those operations are conflict-free and the
kernel remains spill-free.

This suggests a useful two-level policy: use the dynamic kernel when leaf
permutation is genuinely part of the setting search, then specialize or rewrite
the winning bindings into fixed register references for later generations,
constant optimization, or final evaluation. A register-select implementation
for four states is also worth comparing separately; this experiment isolates
the shared-memory topology requested here.
