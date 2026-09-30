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

# Candidate-evaluation performance estimate

This note distinguishes published measurements, arithmetic derived from those
measurements, measurements of this reproduction, and projections for a future
Secant GPU implementation. They should not be treated as the same kind of
evidence.

## What the paper measured

For the concentration-space PySR approach, Riezzo et al. report:

- 500 evolutionary iterations
- more than 120 hours of wall-clock time
- 32 CPU cores from two 16-core Intel Xeon Gold 6130 processors at 2.10 GHz
- “hundreds of thousands” of candidate expressions generated per iteration
- RK4 integration of every candidate model over all three experiments

The paper does not report an exact candidate count, RK4 internal step size, or
per-candidate timer. Consequently, an exact per-candidate duration cannot be
recovered from the publication.

### What is known about the PySR integration path

The concentration-space method substituted two PySR-generated expressions into
the known kinetic backbone, ran RK4 for every resulting candidate model over all
three experiments, and compared the predicted concentration profiles with the
observations. The paper does not publish its implementation or describe its
allocation, cache, or intermediate-storage policy; its data-availability
statement exposes only the supporting dataset. It therefore supports the claim
that predicted concentration values were produced for scoring, but not the
stronger claim that complete trajectories, every RK4 stage, or every
expression-tree node were written to main memory.

[PySR's custom-objective interface](https://ai.damtp.cam.ac.uk/pysr/v1.5.9/api.html)
normally evaluates trees through Julia's `eval_tree_array`. That interface
returns a prediction vector, and
[DynamicExpressions' evaluator](https://github.com/SymbolicML/DynamicExpressions.jl/blob/master/src/Evaluate.jl)
recursively uses array-valued subtree results. The current evaluator reduces
this traffic by fusing small two- and three-operation branches and can reuse
arrays from an arena. The paper does not say whether its custom
concentration-space objective used this path at each RK4 stage, exported a
scalar callable, or implemented a different evaluator. Any exact memory-traffic
comparison must therefore be made against a reproduced baseline or the authors'
source rather than inferred from the 120-hour result alone.

This is materially different from the proposed Secant kernel: after
specialization, each candidate expression is ordinary register-to-register SASS
inside the RK4 stage loop. State, stage derivatives, and expression intermediates
can remain in registers; the kernel reads observations and writes one final loss
per parameter setting.

Using 100,000-300,000 candidates per iteration as the literal working range for
“hundreds of thousands” gives 50-150 million candidate evaluations. At the
reported 120-hour baseline, this implies:

| Quantity | Derived range |
| --- | ---: |
| Candidate throughput across 32 cores | 116-347 candidates/s |
| Amortized wall time per candidate | 2.88-8.64 ms |
| CPU time per candidate at full utilization | 92-276 ms |
| Wall time per evolutionary iteration | more than 14.4 min |
| Total compute consumption | more than 3,840 core-hours |

The midpoint interpretation is 200,000 candidates per iteration, or 100 million
over the run. It corresponds to 231 candidates/s, 4.32 ms of amortized wall time,
and about 138 ms of CPU time per candidate. These figures include whatever PySR,
Julia, integration, allocation, expression interpretation, and constant tuning
work occurred inside the reported run; they are not pure RHS timings.

## CPU cost of the reproduced model

Here one candidate means evaluating one fixed pair of kinetic expressions over
all three 96-hour experiments. The benchmark is single-threaded, compiled C99 on
an Apple M4, and excludes evolutionary-search and compilation overhead:

| Maximum RK4 step | RK4 steps/candidate | RHS evaluations/candidate | Time/candidate |
| ---: | ---: | ---: | ---: |
| 0.5 h | 576 | 2,304 | 14.596 us |
| 0.1 h | 2,880 | 11,520 | 73.871 us |
| 0.01 h | 28,800 | 115,200 | 735.097 us |

The 0.01-hour case sustains about 156.7 million complete RHS evaluations per
second. It is much faster than the per-candidate cost inferred from the paper
because it is a small compiled function with fixed expressions, no PySR object
machinery, no candidate construction, and no constant optimization.

Run the local measurement with:

```sh
make bench
```

## Secant GPU projection

The time recurrence within one trajectory is sequential, so the GPU cannot
parallelize the 9,600 time steps of one experiment. The available parallelism is
across many candidate expressions, initial conditions, and constant settings.

At the conservative effective range of 10-100 billion complete RK4 RHS
evaluations per second on the RTX 5090, deliberately far below Secant's simpler
independent-row throughput, the 0.01-hour workload projects to:

| Quantity | Projection |
| --- | ---: |
| Effective time per batched candidate | 1.15-11.52 us |
| Pure integration time for 50 million candidates | 58 s-9.6 min |
| Pure integration time for 150 million candidates | 2.9-28.8 min |

This is an estimate of a resident, well-filled scoring kernel, not an end-to-end
search measurement. It assumes FP32 candidate state and enough simultaneous
candidates to hide the long dependency chains and division latency. FP64, a
smaller candidate batch, more complicated expressions, rejected/invalid
trajectories, or a smaller integration step would reduce throughput.

A defensible early end-to-end expectation is therefore tens of minutes to about
two hours for work comparable to the paper's 500-iteration concentration-space
run, if candidate generation, specialization, module preparation, constant
handling, selection, and data movement remain batched. The scoring portion alone
could plausibly be roughly 8,000-240,000 times faster than the CPU time per
candidate inferred from the paper. The complete search speedup will be much
smaller and must be measured after the search path exists.

The estimate fails if each candidate causes a separate synchronous module load or
kernel launch. Secant's eager loading and multi-candidate kernel organization are
therefore part of the eventual end-to-end experiment, even though they are not
part of this reproduction milestone.
