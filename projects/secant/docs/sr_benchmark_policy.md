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

# Secant-SR Benchmark Policy and Dataset Plan

This document records the evaluation policy for Secant-SR and the benchmark
suites that are relevant to the Secant paper. It is intended to prevent the
paper from drifting toward artificial resource normalization or evaluator-only
comparisons that do not reflect how Secant searches.

## Core Decision

The primary comparison is **end-to-end wall-clock time to solution quality**.

There is no conversion from one CPU core-hour to an equivalent number of RTX
5090 seconds. Secant is designed to make rapidly changing symbolic programs
practical on a GPU, so using that GPU is part of the system being evaluated.
Every result must state the hardware explicitly rather than describe the
hardware as equal.

The main curves should show quality as a function of elapsed `fit()` time, with
useful checkpoints such as:

```text
10 seconds
30 seconds
1 minute
5 minutes
1 hour
8 hours when a legacy protocol requires it
```

The timer must include all per-run work needed by the search:

- population generation, mutation, crossover, and selection;
- SASS specialization;
- CUDA module and kernel lifecycle;
- host/device transfers and synchronization;
- SSE evaluation;
- settings and binding search;
- constant optimization and LM;
- winner materialization and scoring needed during the search.

One-time environment installation is not part of `fit()`. Dataset loading and
process startup should be reported consistently with the benchmark harness. A
separate systems section may decompose time, but kernel-only throughput is not
the primary symbolic-regression comparison.

## Primary Metrics

For ground-truth problems, report:

- fraction of runs with a symbolically verified solution versus wall time;
- fraction of runs with held-out `R^2 > 0.999` versus wall time;
- median time to first exact or accurate solution;
- final expression size and the accuracy/complexity Pareto front;
- medians and confidence intervals over the prescribed seeds.

For black-box problems, report held-out accuracy and expression complexity
versus wall time. An anytime accuracy/complexity hypervolume or performance
profile is preferable to choosing only one terminal budget.

The search log should also retain diagnostic counters such as proposed ASTs,
AST/settings evaluated, rows evaluated, and LM work. These counters explain the
system but are not used to equate unlike algorithms.

## Operon Comparison

The legacy 2021 SRBench ground-truth protocol used 10 trials, a 75/25 split,
four target-noise levels, and termination at one million evaluations or eight
hours, whichever came first. The study did not exploit parallelism. A run under
that protocol is useful for reproducing the historical Operon comparison, but
its evaluation counter is not treated as a hardware-neutral unit of work.

The direct paper comparison should run Operon and Secant on explicitly named
hardware and plot their observed quality against wall time. Operon's official
single-core configuration can be retained as the legacy reproduction. An
all-physical-core Operon result is an optional practical CPU baseline, clearly
labelled as a different configuration.

Published Operon results may be used as historical context and as external
quality reference points. They are not sufficient for a direct wall-clock
speedup claim because the hardware, software revision, timer scope, and stopping
behavior differ. A direct speedup claim requires rerunning Operon under the same
dataset splits, seeds, success criteria, and timing harness used for Secant.

## EvoGP Comparison

EvoGP is relevant only as an **end-to-end same-GPU search baseline**. Secant and
EvoGP should run on the same RTX 5090 and host, with the same training data,
held-out split, random seeds, wall-clock deadlines, and result metrics. Each
system should be allowed to use its natural search representation and
implementation, including Secant's settings, binding search, and LM.

An identical frozen-tree evaluator comparison is not a primary experiment.
That workload would suppress the settings-based search that is central to
Secant and favor EvoGP's tensorized population representation. The paper is
about the useful search system, not merely the time needed to replay a fixed
tree list.

EvoGP's published paper used four useful smoke-test datasets:

- Daily Demand Forecasting;
- Auto MPG;
- California Housing;
- Feynman I.9.18.

It used 100 generations, ten repetitions, maximum tree size 512, tournament
size 20, crossover probability 0.9, mutation probability 0.1, and the function
set `{+, -, *, /, sin, cos, tan}`. Its reported end-to-end numbers were measured
on an RTX 4090, so its published times may be cited as context but not used as
the denominator of a 5090 speedup. EvoGP must be rerun on the 5090 for the
direct comparison.

## Benchmark Suites to Retain

### 1. SRBench 2025

This should be the primary broad benchmark. The 2025 `srbench_2025` protocol
compares 25 methods on 24 selected problems split between black-box and
phenomenological/first-principles tracks. It uses 30 independent runs, a
six-hour hyperparameter-search allowance, and a one-hour final-training
allowance. Jobs cannot use multiple CPU cores; GPU-capable methods receive a
GPU.

Use the name **SRBench 2025** in the paper. The project has used overlapping
`v2.0` and `SRBench 2.0` terminology for different editions, so year-based names
are less ambiguous.

References:

- <https://github.com/cavalab/srbench/tree/srbench_2025>
- <https://arxiv.org/abs/2505.03977>

### 2. Legacy SRBench Feynman and Strogatz

Keep the legacy ground-truth suite for continuity with widely cited results:

- 116 Feynman regression problems;
- 14 Strogatz regression problems;
- 10 trials per problem;
- target-noise levels `0`, `0.001`, `0.01`, and `0.1` when reproducing the full
  legacy protocol.

Strogatz contains the two derivative components from seven two-state systems.
Each problem predicts one state derivative from the current two states. It is
therefore compatible with ordinary Secant-SR tabular fitting, but it is not a
closed-loop ODE rollout benchmark.

References:

- <https://github.com/cavalab/srbench>
- <https://github.com/lacava/ode-strogatz>

### 3. SRBench++

SRBench++ is a separate task-driven extension rather than the current SRBench
edition. Its synthetic track tests exact rediscovery, feature selection, noisy
local optima, extrapolation, and target noise. These tasks are useful secondary
robustness checks, particularly for Secant's mixed bindings and constant
optimization. The original competition also included a domain-expert COVID-19
track, which is not necessary for the Secant paper.

Reference: <https://doi.org/10.1109/TEVC.2024.3423681>

### 4. SRSD

The Symbolic Regression for Scientific Discovery benchmark provides 120
physics problems with more physically motivated sampling distributions than
the original uniformly sampled Feynman data. Its dummy-variable variants are
particularly relevant to Secant's column-binding search. This is a strong
scientific-discovery supplement if experiment capacity permits.

Reference: <https://github.com/omron-sinicx/srsd-benchmark>

### 5. Closed-Loop Iterated Maps

A fixed, public collection of discrete maps should provide the application in
which Secant's fused recurrent evaluation is uniquely visible. This suite must
split by complete trajectories or initial conditions and report short-horizon
training, long-horizon rollout error, exact recovery, parameter error, and
dynamical statistics where appropriate. It remains distinct from Strogatz,
which is derivative regression.

Candidate source: <https://github.com/GilpinLab/dysts>

### 6. Optional Later Suites

- ODEBench, if continuous-time integrated search becomes paper-ready.
- LLM-SRBench, if LLM-proposed function families become an evaluated part of
  the system rather than a discussion or demonstration.

These are not required for the first Secant paper.

## Use of Published Results

Published results are sufficient for:

- describing the existing state of the field;
- showing historical quality reference lines;
- motivating the chosen competitors and datasets;
- comparing Secant with the wider set of methods when the exact published
  protocol and raw result files are used and the comparison is labelled
  external.

Published results are not sufficient for:

- a direct CPU-versus-GPU wall-clock speedup number;
- a direct Secant-versus-EvoGP speedup on an RTX 5090;
- time-to-quality curves with a shared timer definition;
- claims affected by different software versions, operators, splits, or
  stopping rules.

It is not necessary to rerun every published SR method. The minimum credible
direct baseline set is:

1. Secant-SR on the selected benchmark suites;
2. Operon under the same splits, seeds, deadlines, and timing harness;
3. EvoGP end-to-end on the same RTX 5090 for the datasets it can support
   cleanly.

The remaining published SRBench methods can be presented as external reference
results. Primary direct comparisons should use Rohini so that Secant and EvoGP
share the RTX 5090 and host. Ada can be used for development, replication, and
additional campaigns, but results from its different GPU should not be mixed
into the same timing curve.

## Intended Paper Claim

Existing systems already run GP on GPUs. The claim to test is narrower and
stronger:

> Secant enables native, code-specialized GPU execution for rapidly changing
> symbolic programs, including fused scoring, settings search, constant
> optimization, and recurrent computation, while retaining practical
> end-to-end time to discovery.

Operon tests the CPU-to-GPU application benefit. EvoGP tests the end-to-end
benefit relative to another GPU GP design. SRBench and SRSD test general
symbolic-search quality; Strogatz provides a recognized dynamical-system
bridge; closed-loop maps expose the benefit of fused recurrence.
