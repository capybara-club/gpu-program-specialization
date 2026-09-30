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

# Reusing selected leaves in packed toggle kernels

**Status: experimental, not the default writer.** The user requested arbitrary
AST packing with transient selection results. The default was restored exactly
to the preceding writer. The [patch and tests](selector-reuse-experiment/README.md)
are preserved separately, and the frozen comparison binaries remain unchanged.
The shared-fixture speedup below must not be attributed to the restored default.

The experimental writer retains up to eight repeated selected-leaf values within
one patch. The main eight-AST shared-selector regression is repaired. This is
compiler common-subexpression elimination; the public AST, bank/permutation
layout, kernel template, arithmetic, runner, and GP policy are unchanged.

On rack1 GPU 1 (RTX 5080, CUDA 13.1), all 32 original paired fixtures passed:
**14,811,136 scores were byte-for-byte identical between settings and toggles**.
Every score also passed the independent CPU oracle and signed prediction checks.
The same original settings executables were used, one backend at a time.

Rates include expression evaluation, squared error and tile reduction, with
prepared resident kernels. They exclude specialization/loading. The runner-call
column includes specialization, module loading, scoring and unloading.

| Eight ASTs/kernel; 8,192 rows; 4,096 configs/AST | Settings billion rows/s | Reused toggles billion rows/s | Speed ratio | Settings runner | Toggle runner |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared selectors, arithmetic | 1,271.8 | **1,333.2** | 1.05× | 3.681 ms | **3.593 ms** |
| Shared selectors, sine | 750.4 | **859.1** | 1.14× | 6.577 ms | **5.451 ms** |
| Diverse selectors, arithmetic | 625.4 | **687.2** | 1.10× | 7.209 ms | **6.761 ms** |
| Diverse selectors, sine | 406.9 | **589.2** | 1.45× | 11.056 ms | **7.805 ms** |

The preceding toggle writer measured 626.9 billion rows/s and 7.287 ms for the
shared arithmetic fixture. The new measurement is about **2.13× the engine rate**
and **2.03× the runner rate**. These before/after values are separate sweeps on
the same GPU; the settings/new-toggle columns above are the newly paired sweep.
The toggle resource count remains 53 registers/thread, zero local bytes, at eight
ASTs/kernel. Original settings uses 55 registers in the shared case.

Not every workload beats settings. At 65,536 rows, shared arithmetic still runs
at about 0.73× settings speed. Four-AST shared arithmetic is about 0.81×, and
32-AST diverse arithmetic about 0.83×. The 32-AST SM120 scaffold still reports
168 registers/thread and 16 local bytes. These remaining cases are retained in
[all 32 measurements](results-selector-reuse-20260919.csv); they are not excluded
from the result. Achieved occupancy has not been profiled.

## Compiler contract

- A bounded lookahead considers the first 64 distinct direct selector encodings
  in a packed group and chooses up to eight repeated keys by occurrence count.
  It does not enumerate runtime configurations or allocate per-selector storage.
- Identity includes opcode, ordered alternatives, exact literal bits, and ordered
  bit IDs. Swapped alternatives, different bit IDs, and +0/-0 keys do not alias.
- Values are retained only when at least eight scratch registers remain. Retained
  operands are borrowed, immutable registers, so arithmetic and routine argument
  reclamation cannot overwrite or free them.
- Selection still executes inside the row loop. Reuse spans ASTs within one row;
  values are recomputed next row/configuration. No request or module value cache
  is introduced, and constant-only hoisting is not implemented here.
- Planning inspects top-level direct selectors. Routines remain fully validated
  and compiled; matching selectors inside them can reuse a planned value. There
  is no separate routine-expansion planning pass.
- If retention causes register overflow or insufficient patch space, only that
  patch is retried using ordinary emission. This compiler optimization never
  changes the requested programs or drops configurations to fit a resource limit.

## Validation and artifacts

Seven core host tests passed normally and under ASan/UBSan. New white-box tests
verify selector identity, single emission across repeated uses, immutable
ownership through arithmetic/transcendentals, scratch-budget behavior and patch
isolation. The forced mutation/crossover tests in Secant-SR also pass normally
and under sanitizers; search production code was not modified.

Core CUDA regression and the Secant-SR scoring adapter pass on both RTX 5080
and RTX 4090. The 5080 core run under Compute Sanitizer memcheck reports zero
errors. Tests retain partial packings, changed banks, multiple targets, routines,
bit 31, rejected ASTs, random selectors, and all four retained static shapes.

Remote experiment: `rack1:/home/cdurham/experiments/secant-selector-reuse-20260919/`.
`full-shared/` and `full-diverse/` retain manifests, binary hashes, raw scores and
logs. Source snapshot and test/benchmark executables are alongside them. The
original source is also retained locally in `scratch/selector-reuse-20260919/`.

A separate matched-data/time search campaign compares the old settings policy,
the preceding toggle binary and the updated toggle binary. Engine improvements
in this report do not establish improved symbolic recovery or universal GP speed.

## Fixed GP initial populations

An additional check used the identical generation-zero populations (8,192 AST
occurrences × 4,096 configurations × 10,000 rows), with the actual Secant-SR
before/after executables and seed 23654. Arms alternated for four repetitions;
the table gives median GPU event intervals after the first repetition. This ran
on rack1 GPU 0 while the independent policy campaign used GPU 1. Thus it checks
device scoring cost; concurrent host work limits interpretation of full-call
timings. Dataset preparation, configuration and candidate generation were equal.

| Dataset | Before scoring GPU events | After scoring GPU events |
| --- | ---: | ---: |
| Feynman I.39.11 | 133.40 ms | 133.50 ms |
| Feynman III.7.38 | 133.56 ms | 133.41 ms |
| Feynman test 16 | 140.51 ms | 140.46 ms |

There is no meaningful speed change in this small initial-population probe.
Random independent alternatives offer far fewer common selectors than the
shared-choice fixture. Native winner audits passed for every probe; later
evolution can differ from floating-point reduction ties, so comparing whole
search durations without the scored-generation counts would be misleading.
Full JSON/logs are under the remote experiment's `generation-zero/` directory.
