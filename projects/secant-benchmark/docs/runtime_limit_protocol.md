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

# Runtime Limit Protocol

## Constraint

The generated materialize and SSE kernel algorithms remain fixed while their
runtime ceilings are measured. Do not add occupancy queries, persistent work
schedulers, instrumentation, CUB machinery, or additional kernel branches.
Those changes would alter both execution and ptxas compilation cost.

AST packing necessarily changes the amount of expression code in a kernel.
Logical rows per CTA may change a generated constant and launch dimensions, but
the load, evaluation, reduction, and store algorithms must remain identical.

## Correct Output Contract

Every timed kernel writes to its own output region. Materialize therefore
measures distinct generated values rather than repeatedly overwriting an
L2-resident buffer. SSE uses a distinct accumulator region per kernel even
though its output is small.

Compilation, module loading, allocation, copies, output initialization,
verification, and warmups remain outside GPU event timing.

## Materialize Sweep

Freeze threads, compiler options, AST definitions, and kernel implementation.
Sweep:

```text
ASTs/kernel: 1, 2, 4, 8, 12, 16, 24, 32
Rows:        65,536; 262,144; 1,048,576; 4,194,304
AST mode:    ALU, MUFU
Backend:     PTX, CUDA
Opt level:   O0, O1
```

Keep enough kernels to stabilize launch timing, subject to:

```text
output bytes = kernels * ASTs/kernel * rows * sizeof(float)
```

Record row-evals/s and `4 * row-evals/s` as the minimum output-store byte rate.
The plateau across increasing rows and AST packing is the materialize ceiling.
Smaller inputs also show the L2-resident regime, which must be reported
separately from the bulk-output regime.

## SSE Sweep

Freeze the reduction and atomic epilogue. Start with 128 threads and sweep:

```text
ASTs/kernel:  1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192
Rows/CTA:     128, 256, 512, 1,024, 2,048, 4,096, 8,192
Rows:         65,536; 262,144; 1,048,576; 4,194,304
AST mode:     ALU, MUFU
Backend:      PTX, CUDA
Opt level:    O0, O1
```

The primary relationship is:

```text
CTAs/kernel = ceil(rows / rows_per_CTA)
```

No occupancy query is required. The explicit rows/CTA sweep reveals the point
where reduction and atomic amortization stop improving throughput because too
few CTAs remain.

After identifying the best rows/CTA neighborhood, compare 64, 128, and 256
threads only within that neighborhood. This avoids multiplying the full sweep
by a weak dimension.

## Measurement Order

1. Establish the ASTs/kernel plateau at 1,048,576 rows.
2. Establish rows/CTA for SSE using the best packing neighborhood.
3. Check row-count scaling and separate cache-resident from bulk-data results.
4. Check 64/128/256 threads around the SSE winner.
5. Repeat only the winning and neighboring cells for final reported values.

This process finds limits through controlled launch and packing experiments
without making the production kernels more complicated.
