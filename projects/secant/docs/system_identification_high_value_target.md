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

# High-Value Target for Secant System Identification

See also the standalone [system-identification product scope](system_identification_product_scope.md)
for supported regimes, explicit exclusions, product workflow, and validation
gates.

## Recommendation

The best first target is **gray-box bioprocess kinetic-law discovery**:
fermentation, cell-culture, or catalytic-reaction systems in which the mass
balances and experimental inputs are known, but one or two nonlinear rate laws
are unknown.

This target offers an unusually good combination of:

- real value in pharmaceutical, biochemical, food, and renewable-chemical
  manufacturing;
- small state vectors that can remain in registers;
- controlled experimental inputs and initial conditions;
- strong physical and dimensional constraints;
- rapidly changing expressions evaluated repeatedly inside an ODE integrator;
- recent symbolic-regression baselines with supporting data; and
- considerably less regulatory and statistical complexity than clinical
  pharmacometrics.

The objective should not be unrestricted discovery of every system equation.
The strongest and most attainable formulation is:

> Preserve a trusted mechanistic backbone and discover one or two missing
> nonlinear kinetic terms through fused full-trajectory evaluation.

## Representative system

A fed-batch fermentation model might contain

\[
\begin{aligned}
D &= F/V \\
\dot X &= \big(\mu(S,P,T)-k_d-D\big)X \\
\dot S &= D(S_f-S)-\frac{1}{Y_{X/S}}\mu(S,P,T)X \\
\dot P &= q_P(S,P,T)X-DP \\
\dot V &= F,
\end{aligned}
\]

where:

- \(X\) is biomass or cell concentration;
- \(S\) is substrate concentration;
- \(P\) is product or an inhibitory byproduct;
- \(V\) is reactor volume;
- \(F\) is feed rate;
- \(T\) is temperature;
- \(\mu\) is an unknown growth-rate law; and
- \(q_P\) is an unknown product-formation law.

The mass balances remain fixed. Secant searches expressions such as

\[
\mu(S,P,T)
=
\frac{\mu_{\max}S}{K_S+S}
\frac{K_I}{K_I+P}
e^{-E_a/(RT)}
\]

without being given that exact structure.

The stable CUDA kernel should own:

- experiment and observation loading;
- feed, dosing, and other external events;
- known mass-balance equations;
- RK4 integration;
- positivity, bounds, and nonfinite handling;
- observation scoring; and
- aggregation over multiple experiments.

Secant should specialize only the changing growth, production, inhibition, or
death-rate expressions. Those expressions are evaluated at every RK4 stage,
integration step, experiment, candidate, and parameter setting.

## Why this is a timely external target

Riezzo et al. published a closely aligned study in August 2026. It applies
PySR to unknown kinetic terms inside a known yeast-fermentation model backbone,
compares direct trajectory-space discovery with kinetic-profile discovery, and
provides its data in the supporting information:

- [Automated Data-Efficient Symbolic Regression for Interpretable Bioprocess
  Model Development](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/bit.70328)

This supplies a current external baseline, expression grammar, experimental
budget, ground-truth system, model-selection procedure, and trajectory metrics.
It also exposes an important difficulty: structurally different expressions
can produce similar integrated trajectories. The study found its gradient- or
profile-space approach more effective for recovering the underlying kinetic
structure than unconstrained trajectory fitting alone.

That suggests a hybrid Secant pipeline:

```text
physical constraints and candidate generation
                    |
                    v
Secant feature/Gram or weak-form evaluation
                    |
                    v
STLSQ or kinetic-profile screening
                    |
                    v
small set of candidate kinetic expressions
                    |
                    v
Secant multi-site RK4 trajectory scoring
                    |
                    v
LM refinement of nonlinear parameters
                    |
                    v
AICc, complexity, identifiability, and validation
```

The sparse or gradient stage removes the easy linear combinatorial dimension.
Full integration then distinguishes and refines candidates under the objective
that ultimately matters.

## What would constitute a surprising result

Raw expression-evaluation throughput is insufficient. A convincing result
would have the following form:

> From the same limited fermentation experiments, recover the correct hidden
> kinetic laws reliably across repeated seeds, validate them through complete
> integrated trajectories under unseen operating conditions, and reduce the
> search from hundreds of evolutionary iterations to seconds or minutes.

The evaluation should report:

- wall-clock time to the first correct structure;
- exact structural-recovery rate across independent seeds;
- held-out complete experiments rather than randomly held-out time points;
- prediction under unseen feed profiles, initial conditions, and temperatures;
- long-rollout stability and physical validity;
- parameter error when ground truth exists;
- expression complexity and Pareto fronts;
- AICc or another small-sample model-selection criterion;
- total search time and component timing;
- comparison with PySR under the same data and expression language; and
- ablations for gradient-only, trajectory-only, and hybrid search.

A particularly valuable extension would close the experimental-design loop:

1. Discover several plausible kinetic laws.
2. Find an initial condition or feed schedule for which their predictions
   differ maximally.
3. Perform or simulate that experiment.
4. Use the new observations to eliminate incorrect structures.

This would make the system an experiment-discrimination tool rather than only
a faster curve fitter. In wet-lab settings, reducing the number of experiments
can be more valuable than reducing compute time.

## Target comparison

| Target | Ease | Commercial value | Secant fit | Potential expert impact |
|---|---:|---:|---:|---:|
| Bioprocess or fermentation rate laws | 4.5/5 | 4.5/5 | 5/5 | 4.5/5 |
| Public nonlinear-control benchmarks | 5/5 | 3/5 | 4/5 | 3/5 |
| Battery degradation or hysteresis | 3/5 | 5/5 | 4.5/5 | 4.5/5 |
| PK/PD missing mechanisms | 3/5 | 5/5 | 5/5 | 4.5/5 |
| Interatomic potentials | 2.5/5 | 5/5 | 5/5 | 5/5 |
| Material constitutive laws | 2.5/5 | 5/5 | 5/5 | 5/5 |
| Turbulence and PDE closures | 1.5/5 | 5/5 | 5/5 | 5/5 |

### Public nonlinear-control benchmarks

The [Silverbox benchmark](https://www.nonlinearbenchmark.org/benchmarks/silverbox)
is a measured electronic implementation of a Duffing-like nonlinear oscillator
with public data, evaluation support, and curated results. It is a good early
test for external control inputs, partial observation, one-step versus rollout
loss, and long-horizon stability. It should be treated as a system-ID
conformance test rather than the headline industrial result.

### Battery degradation

A low-order gray-box battery model could retain state of charge, polarization,
temperature, hysteresis, capacity loss, and resistance growth while Secant
searches one degradation or hysteresis rate law. This is economically valuable
and has strong repeated-integration structure, but important internal states
are latent and detailed electrochemical models become stiff DAEs or PDEs.

Useful public resources include:

- the [NASA randomized and recommissioned battery
  dataset](https://catalog.data.gov/dataset/randomized-and-recommissioned-battery-dataset),
  which covers 26 battery packs under constant and randomized loading; and
- [PyBaMM](https://docs.pybamm.org/en/stable/), which provides an open library
  of battery ODE, DAE, and spatial models and parameter sets.

Battery degradation is a strong second vertical after the small-state
bioprocess infrastructure is validated.

### Pharmacometric PK/PD

PK/PD has extremely high decision value and a compatible computational shape:
known compartment equations, dosing events, a missing nonlinear response,
many subjects and covariates, and trajectory likelihood evaluation. It is not
the easiest first result because clinical data are sparse, population
variability matters, important states can be unobserved, and observationally
equivalent mechanisms are common.

### Interatomic potentials

Interatomic-potential discovery may be the purest eventual Secant workload: a
stable neighbor-list and integration kernel repeatedly evaluates a changing
analytic force expression for many atoms and timesteps. Symbolic regression
has already produced compact many-body potentials, including evaluation in a
ten-million-step molecular-dynamics benchmark:

- [Fast, accurate, and transferable many-body interatomic potentials by
  symbolic regression](https://www.nature.com/articles/s41524-019-0249-1)

This has exceptional potential impact, but descriptors, rotational and
permutation invariance, force derivatives, neighbor lists, conservation, and
training data make it a substantially larger first project.

### Constitutive laws and PDE closures

Material constitutive updates inside explicit finite-element timesteppers and
turbulence closures inside CFD solvers maximize Secant's architectural
advantage: the stable surrounding kernel is large, the changing expression is
small, and it executes at every integration point or grid location and
timestep. These are high-impact follow-on domains rather than quick first
targets because physical admissibility and extrapolation are difficult.

Relevant examples include:

- [Automated data-driven discovery of material models based on symbolic
  regression](https://www.sciencedirect.com/science/article/pii/S174270612400521X)
- [Data-enabled discovery of specific and generalisable turbulence
  closures](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/dataenabled-discovery-of-specific-and-generalisable-turbulence-closures/4077B2AAC80E5386F5A941ED13FBB451)

## Recommended implementation and validation path

### 1. General small-state trajectory kernel

Build the reusable multi-site specialization infrastructure and a stable RK4
kernel supporting:

- several state variables;
- external controls and experimental conditions;
- known and candidate RHS equations;
- repeated instances of one RHS program at all RK4 stages;
- several experiments per candidate;
- dynamic constants, bindings, and parameter starts;
- invalid-trajectory penalties; and
- complete trajectory loss.

### 2. Conformance tests

Validate the machinery on synthetic kinetic systems and recognized nonlinear
benchmarks such as Silverbox. Establish CPU, CUDA, PTX, and Secant agreement,
then test one-step and rollout scoring separately.

### 3. Reproduce the 2026 yeast-fermentation study

Use the published data, grammar, experimental conditions, AICc protocol, and
ground truth. Reproduce its PySR baseline before making performance or recovery
claims.

### 4. Add the hybrid search

Use Secant Gram or weak-form evaluation and STLSQ to screen structures, then
compose selected terms into candidate ASTs for full integrated scoring. Use LM
only for constants that enter nonlinearly; solve linear coefficients directly
where possible.

### 5. Validate under interventions

Hold out entire operating regimes, including different feed schedules,
substrate concentrations, initial biomass, temperatures, or inhibition
conditions. A correct mechanism should predict interventions, not only
interpolate one observed trajectory.

### 6. Move to experimental bioprocess data

After the method is reproduced and characterized on known ground truth, seek a
bioprocess collaboration or a well-documented public experimental dataset.
At this stage, experimental design, uncertainty, and identifiability are more
important than simply expanding the expression search budget.

## Central conclusion

The quickest path to a system that could surprise domain experts is:

```text
Silverbox and synthetic correctness
                |
                v
published yeast-fermentation reproduction
                |
                v
Secant hybrid structure discovery
                |
                v
held-out operating-condition prediction
                |
                v
real bioprocess model-discovery collaboration
```

Secant's strongest claim would not be that it tries more trees. It would be
that it makes full trajectory-level structure search cheap enough to remain in
the discovery loop while preserving a large, native, physically constrained
simulation kernel.
