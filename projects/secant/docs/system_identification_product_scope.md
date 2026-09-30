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

# Secant System-Identification Product Scope

## Product thesis

The strongest product is not a universal ODE solver or an unrestricted
symbolic-regression service. It is a mechanistic model-completion system:

> Given a mostly trusted dynamical model, several experiments, and one or a few
> explicitly unknown rate laws, search constrained expression families, fit
> their constants against complete trajectories, and return a validated,
> interpretable shortlist.

This is where Secant's direct specialization is most differentiated. Each
candidate expression is evaluated repeatedly inside a stable integration,
observation, reduction, and constant-optimization kernel. Candidate values,
trajectory states, sensitivities, and LM statistics can remain fused on the
GPU rather than being materialized between an expression engine and an ODE
solver.

## Initial computational scope

| Dimension | Preferred range | Initial boundary |
|---|---:|---:|
| ODE states | 2--16 | 32 |
| Unknown RHS terms or equations | 1--3 | Approximately 4 |
| Distinct fitted constants per candidate | 2--16 | 32 |
| Candidate AST size | 5--30 nodes | Approximately 50 nodes |
| Experimental trajectories | 2--32 | Stream larger collections |
| Observations per experiment | 10--1,000 | Tile larger target tables |
| Integration work per evaluation | 100--10,000 RK steps | Longer if numerically stable |
| Runtime leaf settings | Hundreds to tens of thousands | Sample enormous products |
| Numeric regime | Smooth, non-stiff, float or mixed precision | Validate finalists in double |

Targets do not have to fit completely in shared memory. Small target tables
receive a shared-memory fast path; larger tables can be consumed in observation
tiles while the simulated state remains resident.

The active fitted-parameter count, not the number of AST occurrences, controls
the LM topology. Repeated references to one dynamic constant still represent
one parameter. Dynamic leaves bound to states, controls, or known constants are
discrete settings rather than LM dimensions.

## Kernel regimes

The design space can be covered by five principal kernel families rather than
dozens of unrelated implementations.

| Kernel family | Use |
|---|---|
| Thread-owned trajectory scoring | Broad search with abundant AST, binding, and constant settings |
| Trajectory-subgroup scoring | Few settings and several independent experiments per setting |
| Thread-owned LM | Small state and parameter counts with abundant independent settings |
| Parameter-subgroup LM with 4, 8, 16, or 32 lanes | Modest settings, larger parameter counts, or latency-sensitive fitting |
| Tiled or matrix-free large-system optimizer | More than 32 parameters or state pressure beyond the register-resident regime |

For the measured 12-state, eight-parameter fixed ODE, the eight-lane LM kernel
used 96 registers without spills. The direct thread-owned kernel used 255
registers. Parameter-subgroup ownership was faster below approximately
8,000--12,000 concurrent settings, while thread ownership won once settings
were abundant. The crossover should move toward subgroup ownership as the
number of states or fitted constants increases.

The relevant setting count is the total ready work across ASTs, bindings,
starts, modules, and streams, not merely the starts attached to one AST. A
runtime dispatcher can therefore choose a topology from:

```text
[AST skeleton][leaf binding][constant start]
```

A representative promoted candidate might use 64 leaf bindings and two starts,
giving 128 useful LM settings while still amortizing specialization, module
loading, and target staging.

## Trajectory grouping

Four-thread RK4 groups are useful when four lanes own four independent
experiments for the same candidate and parameter setting. Each lane integrates
a complete trajectory and the subgroup reduces their losses. This creates
parallelism when the candidate population is small.

They are generally not useful for splitting one ordinary coupled four-state
trajectory across four lanes. That requires communication at every RK stage.

For forward-sensitivity LM, lanes instead correspond to parameter directions:

| Fitted constants | Natural layout for four experiments |
|---:|---|
| 4 | Four 4-lane trajectory groups can fit in half a warp; additional work can occupy the other half |
| 8 | Four 8-lane trajectory groups fill one warp |
| 16 | Two 16-lane groups process the experiments in two batches |
| 32 | One warp-wide group processes four experiments sequentially |
| More than 32 | Use parameter tiles, multiple warps, or a different optimizer |

At every observation, residuals and exact sensitivities are accumulated online:

\[
r=w(x(t)-y(t)),\qquad
g_j=w\frac{\partial x(t)}{\partial\theta_j},
\]

\[
J^TJ\mathrel{+}=gg^T,\qquad
J^Tr\mathrel{+}=gr,\qquad
SSE\mathrel{+}=r^2.
\]

The Jacobian is never materialized. More observations increase integration and
accumulation work but do not enlarge the small LM solve.

## Best real-world problem shapes

| Domain | Suitable task |
|---|---|
| Bioprocessing and fermentation | Missing growth, uptake, production, inhibition, or death law |
| Pharmacokinetics and pharmacodynamics | Unknown absorption, clearance, saturation, inhibition, or drug-response term |
| Chemical and catalytic kinetics | Missing reaction-rate or interaction term in a mostly known network |
| Battery degradation | Compact unknown hysteresis, side-reaction, or degradation law |
| Ecology | Missing competition, predation, environmental-response, or carrying-capacity term |
| Industrial processes | Missing dynamics in reactors, tanks, flow, thermal, and other low-order systems |

The first target should remain gray-box bioprocess kinetic-law discovery. It
combines controlled experiments, small state vectors, strong physical
constraints, interpretable rate laws, and a direct opportunity to reduce wet-lab
experimentation. Pharmacometric PK/PD is potentially higher value but has more
population variability, latent-state, credibility, and regulatory complexity.

## Explicit initial exclusions

| Out-of-scope setup | Reason |
|---|---|
| Discovering every equation of a large system from scratch | Combinatorial search and observational equivalence dominate evaluator speed |
| Spatial PDE grids, CFD, or weather models | The state is a large field rather than a small register-resident vector |
| Strongly stiff ODEs or DAEs requiring implicit integration | Linear solves and adaptive stepping fundamentally change the kernel |
| Frequent discontinuities, contacts, switches, or state resets | Divergent control flow and piecewise sensitivities require a separate design |
| Long-horizon chaotic trajectory matching | Pointwise trajectory gradients become unstable; use windows or statistical objectives |
| Hundreds of fitted constants | Forward sensitivities and the quadratic LM matrix cease to be appropriate |
| Strict FP64 or severely ill-conditioned fitting | Consumer-GPU throughput and the current numerical design are poor matches |
| Huge unrestricted expression languages | Search, identifiability, and interpretation fail before evaluation throughput matters |
| A single known model requiring one ordinary parameter fit | Established optimizers suffice; there is too little repeated specialization |
| Neural ODE training | This is a tensor/reverse-mode weight-optimization workload |
| Full Bayesian posterior inference | Useful downstream of discovery, but a separate computational product |
| Nonsmooth expressions passed directly to LM | `min`, `max`, `abs`, thresholds, and branches do not provide reliable local derivatives |

Nonsmooth candidates can still participate in score-only search. They should
skip LM or enter an explicitly piecewise optimization path.

## User workflow

A useful system asks the domain expert to provide:

- state variables, units, and known RHS equations;
- unknown insertion sites and allowed dependencies;
- experimental initial conditions, controls, and interventions;
- observed trajectories, missing-value masks, and measurement weights;
- permitted smooth operators and dimensional constraints;
- parameter bounds and sign constraints; and
- complete experiments reserved for validation.

The system returns:

- a Pareto-ranked shortlist of candidate missing mechanisms;
- fitted constants and uncertainty or stability diagnostics;
- training and held-out trajectory metrics;
- parameter and structure stability across experiments and random seeds;
- invalid-trajectory and identifiability warnings;
- experiment conditions that best distinguish competing candidates; and
- portable equations and a complete, auditable search record.

The product should present candidates as hypotheses for expert review, not as
automatically established scientific truth.

## Evidence that there is a real user need

The surrounding activities already have institutional and commercial value:

- The FDA's [Model-Informed Drug Development Paired Meeting
  Program](https://www.fda.gov/drugs/development-resources/model-informed-drug-development-paired-meeting-program)
  explicitly supports exposure-based, biological, statistical, and mechanistic
  models used for dosing, trial design, safety, and regulatory decisions.
- The FDA's final [ICH M15 General Principles for Model-Informed Drug
  Development](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/m15-general-principles-model-informed-drug-development)
  establishes expectations for planning, evaluating, documenting, and reporting
  model-informed evidence. This means credibility and auditability must be core
  product features.
- The EMA states that mechanistic models are increasingly used throughout drug
  research and development and is developing additional guidance for their
  assessment and reporting: [mechanistic-model guideline concept
  paper](https://www.ema.europa.eu/en/guideline-assessment-reporting-mechanistic-models-used-context-model-informed-drug-development).
- NIST is actively developing [scalable bioprocess models for controlled cell
  culture](https://www.nist.gov/programs-projects/bioprocess-modeling-expansion-cell-culture)
  and connects them to reproducibility, quality assurance, and digital twins.
- NIIMBL describes an industry need for predictive mechanistic digital twins
  that reduce experimental burden and accelerate bioprocess development in its
  [integrated upstream-process project](https://www.niimbl.org/projects/leveraging-multidimensional-partner-data-to-develop-predictive-digital-twins-for-integrated-upstream-processes/).

This evidence validates demand for mechanistic models and better model-building
workflows. It does not by itself prove demand for automatic equation discovery.
The product must demonstrate that constrained structure search produces useful
hypotheses faster, reduces experiment cycles, or reveals a mechanism that the
existing manual workflow misses.

## Likely users and product form

The initial users are specialist teams rather than a mass self-service market:

- pharmacometricians and quantitative systems pharmacology scientists;
- bioprocess modeling and process-development groups;
- computational chemical and systems-biology researchers;
- system-identification consultants; and
- research groups maintaining mechanistic digital twins.

The best initial offering is likely an expert-assisted search service or a
local, auditable technical tool rather than an opaque hosted optimizer. Real
datasets are sensitive, the problem grammar requires domain judgment, and
users must inspect competing mechanisms and validation evidence. An MCP or API
layer can let an LLM and domain expert propose constrained families, submit
searches, inspect failure modes, and refine the next batch while Secant remains
the high-throughput backend.

## Product validation gates

The product case becomes credible only after demonstrating all of the following:

1. Recover blinded laws on recognized systems across repeated seeds.
2. Beat strong baselines on wall-clock time to a correct or equally predictive
   structure, not only expression evaluations per second.
3. Predict complete held-out experiments with unseen initial conditions or
   control schedules.
4. Provide parameter stability, identifiability, physical-validity, and
   complexity diagnostics.
5. Demonstrate one expert workflow where the result changes an experiment,
   model, or process-development decision.
6. Show that final candidates reproduce in a trusted high-accuracy CPU solver.

The first commercial proof should be a paid or closely partnered pilot on a
small gray-box model. The decisive value metric is not GPU cost. It is the time
or experiments saved in reaching a trustworthy mechanistic hypothesis.
