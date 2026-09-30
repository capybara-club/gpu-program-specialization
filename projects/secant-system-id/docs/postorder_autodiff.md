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

# Postorder autodiff specialization

## Decision

The derivative is represented as data before it becomes code. The canonical
artifact is a versioned postorder AD tape derived deterministically from the
original postorder AST. Direct CUDA, PTX, or SASS is a lowering of that tape,
not the only copy of the derivative.

This preserves:

- a stable byte representation that Python and C99 can compare exactly;
- a human-readable inspection containing node indices, operands, forward
  arguments, reverse rules, scratch bounds, and a SHA-256 identity;
- repeated leaf occurrences, which must accumulate into one leaf partial;
- dynamic leaf bindings, whose leaf partials are later accumulated into state
  and active-constant derivatives; and
- freedom to replace the lowering without changing the genome or checkpoint
  format.

LM rejects nonsmooth abs, min, and max ASTs for now. Their subgradient policy
must be explicit before they enter a differentiable promotion stage.

## Current lowering

For each missing site, the initial backend derives a site-major output bundle:

1. the primal value;
2. one local partial for each leaf slot.

The scalar postorder programs are still inspectable. Before SASS emission, the
backend hash-conses their expression DAG and emits common subexpressions only
once. The planted fed-batch genome has two 15-node ASTs. Its primal site has
two outputs and emits 18 SASS instructions. Its statistics-only site has 16
partial outputs and emits 96 instructions. Independent scalar compilation of
all primal and partial programs would emit 334 instructions.

The trajectory code uses each local partial with the setting's dynamic binding:
partials bound to state slots contribute to the ODE state Jacobian; partials
bound to active constant slots contribute to the parameter Jacobian. Two leaf
slots that bind to the same state or constant are summed, not overwritten.

## Scratch policy

The current 15-node fed-batch shape fits a register-only CSE schedule and
therefore allocates no AD scratch. The generated CUBIN uses no local memory,
stack, or spills.

The canonical tape also records a bounded fallback layout with two node-major
planes:

    value[node][thread]
    adjoint[node][thread]

Both planes are node-major so a warp accesses adjacent per-thread values. One
workspace is reused sequentially across missing ASTs; it is sized for the
largest site, not the sum of all sites. The shared-memory estimate is:

    reference bytes
    + bank slots * threads * 4
    + 2 * maximum site nodes * threads * 4

At 256 threads, a 30-node maximum site plus the current fed-batch reference and
bank needs 77,056 bytes. The planted 15-node sites need 46,336 bytes if this
fallback is selected. Register-only lowering remains preferable when it fits.

If the bounded shared allocation exceeds the device or occupancy budget, the
backend must reject that kernel shape or deliberately choose a different
topology. It must not silently spill an unbounded tape to local memory.

## File policy

Tape inspection is in-memory and prints JSON to stdout by default. Persisting a
tape, generated include, CUBIN, or inspection report requires an explicit
output path. Repository builds place these artifacts only below generated/,
which is ignored by Git. No source checkout, package cache, or ad-hoc temporary
tree is created as part of specialization.

## Split evaluation ABI

The trajectory LM kernel contains two disjoint marker identities and patch
reserves:

- a primal-only site used during every trajectory evaluation; and
- a partial-only site entered only while accumulating LM statistics.

Both are specialized from the same canonical tape and inspection records each
marker identity, output count, reserve size, and emitted instruction counts.
There are no additional kernel launches. On the RTX 4090 this changes
saturated throughput from 34,963 fits/s for the combined site to 61,998 fits/s,
or about 65% of the 95,618 fits/s hand-written derivative control.

The final CUBIN still uses 255 registers with no spills. Tests at 64, 128, and
256 threads per CTA show that 256-thread thread ownership remains fastest for
this four-state, six-parameter problem. A direct reverse-tape SASS lowering or
one partial site per missing AST should be considered only if measurements on
larger ASTs justify their extra memory operations or code complexity.

## GP integration gates

The split kernel is suitable for an experimental fed-batch promotion path, but
the following boundaries remain explicit:

- the CUDA control is fixed to two eight-leaf sites, four states, six active
  constants, 16 dense aligned trajectories, and 12 observations;
- the multi-output CSE specializer is Python-only, while C99 currently builds
  and verifies the canonical AD tape and owns the persistent module queue;
- abs, min, and max genomes are ineligible for LM;
- the production split sites pass a 64-genome by 32-point randomized GPU
  primal/gradient differential gate on both sm_89 and sm_120, including all
  smooth opcodes and repeated leaves;
- one specialized CUBIN currently represents one complete system genome; and
- the experimental recovery command selects scored GP candidates, runs dynamic
  binding settings and starts through LM, and writes back constants/bindings
  only when MSE improves. The default command remains SSE-only.

The derivative CSE scheduler may use marker output registers as temporary
storage before those outputs are materialized. If a shared value is still live
when its register becomes the next fixed output, the scheduler relocates that
value with one `MOV` and then pins the output. ASTs that genuinely exhaust all
legal registers remain ineligible for LM; promotion continues with the next
ranked candidate and records the rejection in the durable campaign log.

Promotion batches now pass several Python-specialized genomes through the
eager C99 loader with a shared reference allocation and fixed device buffers.
The remaining optimization boundary is the Python SASS specialization itself,
not module/context creation or per-candidate device allocation. It is not a
reason to change the current thread-owned LM topology.
