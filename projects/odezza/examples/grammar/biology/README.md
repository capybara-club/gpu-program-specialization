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

# Gene expression examples for the ODE grammar compiler

These examples compile with the standalone `odegrammar` package delivered earlier in this conversation. They demonstrate real modeling questions using illustrative equations, scaled coefficients, and parameter ranges. They do not contain experimental data, measured fit scores, or claims of model recovery. The compiler was used to validate and enumerate every example; `plan_results.json` records the resulting counts.

Extract this folder beside the previously extracted `ode_grammar_compiler` folder, then run:

```bash
cd ode_grammar_compiler
python -m odegrammar plan ../biology_grammar_examples/01_gene_feedback.json
python -m odegrammar compile ../biology_grammar_examples/03_gene_network.json --compact -o gene_network.jsonl
```

## Is the grammar expressive enough

It can represent many small deterministic biological ODEs: Hill regulation, mass-action terms, Michaelis–Menten removal, combinations of interactions, explicit hidden species, maturation compartments, and alternative source-state assignments. The larger shortcomings are experimental observation models, time-varying inputs, parameter initialization and scope, and scientific constraints. Expressing an RHS does not provide those features or establish that its parameters are identifiable.

Every example fixes a known biological framework and searches uncertain mechanisms inside it. A broad unconstrained operator grammar would generate many expressions with no useful biological interpretation.

## Case 1 Feedback versus protein removal

A researcher measures the time course after starting expression of a repressor protein. They have calibrated protein abundance measurements and several mRNA measurements, potentially irregularly spaced and missing at some times. They want to know whether feedback is needed to explain the dynamics, or whether production and removal suffice.

Assumptions for this example:

- Conditions are constant after induction at the start of the modeled interval. Arbitrary inducer schedules are not supplied through this compiler.
- The engine receives the initial conditions, observation times, masks, and state-to-measurement mapping separately.
- Basal transcription, translation, and dilution are fixed from external information for this illustration. mRNA loss is profiled over three discrete possibilities.
- The supplied numbers use illustrative rescaled units; they are not universal biological rate ranges. The RK4 step size must be checked against the actual scales and candidate dynamics.

The vector field is:

`dm/dt = b + a*R(p) - dm_rate*m`

`dp/dt = kt*m - mu*p - D(p)`

Four regulation choices are `1`, `1/(1+p/K)`, `1/(1+(p/K)^2)`, and `1/(1+(p/K)^4)`. Three additional removal choices are `dp_rate*p`, `V*p/(Kd+p)`, and their sum. The constant dilution term is present in every protein equation.

Negative autoregulation is an experimentally grounded mechanism affecting response time. This motivates comparing such hypotheses; it does not mean that the shape of one curve uniquely reveals feedback. [Rosenfeld, Elowitz and Alon](https://pubmed.ncbi.nlm.nih.gov/12417193/)

`01_gene_feedback.json` produces 12 whole-system skeletons, each with three mRNA-loss grid values and 64 joint random coefficient vectors: 2,304 configuration visits. All random coefficient components use the same draw axis and separate streams, so there are 64 joint vectors, not 64 raised to the number of coefficients.

The LLM would send a request such as:

> Keep translation and dilution fixed. Compare no feedback with three Hill repression forms and three removal laws. Preserve the best distinct models under each mechanism tag. Evaluate against observed entries across the prepared experiments with shared mechanism coefficients.

The engine must aggregate those experiment losses consistently; the grammar compiler does not implement the observation objective or multi-experiment fitting.

After screening, the LLM requests separate LM refinement of retained configurations. It examines errors by experiment and observed state, checks finalist integration accuracy, and selects using held-out data. If feedback alternatives improve one fitted trajectory but fail another experiment, the proper next step is to revise or reject the hypothesis rather than claim a discovered regulator.

The file carries `no_feedback`, `negative_feedback`, Hill-exponent, and removal-mechanism tags. Top-k policies preserve alternatives under these tags. The compiler forwards that policy; your scoring backend must implement it.

## A current-syntax bridge from RNG screening to LM

The compiler presently accepts a numerical `theta` initial value rather than a pooled initializer. The examples use:

```text
a = rng.a * exp(theta.log_a)
```

with `theta.log_a` initially zero and `rng.a` sampled from a positive log-uniform range. The expression is placed in the shared named shape `shape.a`.

During screening, `a` equals the sampled coefficient. During LM, the sampled `rng.a` stays fixed and `theta.log_a` is fitted. This is one fitted scalar and keeps `a` positive. The backend must differentiate the exponential correctly. It does not enforce an upper biological bound or prevent numerical overflow for extreme trial parameters.

The same construction is used for the threshold and removal coefficients. The largest active fit in case 1 uses five scalars: `log_a`, `log_K`, `log_dp`, `log_V`, and `log_Kd`. Inactive branches use fewer. The sampled reference value is not an additional fitted variable.

`05_lm_request_template.json` is a separate LM request for a negative-feedback/mixed-removal candidate. Its candidate ID is visibly a placeholder; replace it with an actual scored configuration ID. The real coefficients after fitting are the sampled reference coefficients multiplied by the corresponding exponentiated corrections. LM does not vary the selected mRNA-loss grid entry.

A first-class parameter declaration with `initial_from`, a transform, and `fit` would be a worthwhile API improvement. The current workaround adds constant-only exponential expressions to the RHS. A future compiler could hoist those expressions once per LM trial evaluation or iteration when their parameter values change; it must not freeze them across the entire LM run.

## Case 2 Fluorescence maturation can explain apparent delay

Raw fluorescence is not necessarily the instantaneous total abundance of the functional protein. Maturation dynamics can bias inferred promoter activity if omitted. [Pavlou and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC9675035/)

`02_reporter_maturation.json` adds an immature protein state:

```text
dm/dt          = b + a*R(immature + f) - dm_rate*m
d(immature)/dt = kt*m - (km + dp_rate + mu)*immature
df/dt          = km*immature - (dp_rate + mu)*f
```

Here `f` is mature fluorescent protein and `immature` is a modeled, potentially unobserved state. Feedback depends on total protein in this illustrative model, assuming regulatory activity does not await fluorophore maturation. A different biological assumption could make feedback depend only on mature protein. This is a one-stage maturation model, not a universal model for every fluorescent protein.

The LLM would ask:

> Does the apparent delay still require a different feedback law after accounting for reporter maturation? Score the measured mature reporter and available mRNA entries, and retain both explanations if they remain compatible.

The engine supplies `fluorescence = scale*f + background` or calibrated observations, with consistent loss normalization. It also needs an initial condition for the hidden immature state; missing its measurements does not set that initial condition to zero. Comparing the two-state and three-state requests requires separate state/preparation contexts but the same measurement objective.

This example generates four skeletons and 768 configuration visits. Its largest active fit has four scalars. It currently uses first-order protein removal only. Therefore it is an illustrative maturation comparison, not a complete factorial comparison of every maturation and removal model from case 1.

## Case 3 Unknown interactions in a three-gene circuit

Small synthetic transcription circuits provide a concrete setting for structure discovery. The original repressilator is an experimental example with three transcriptional repressors and a fluorescent readout; the example here is a generic search space, not a fitted reproduction of that experiment. [Elowitz and Leibler](https://www.nature.com/articles/35002125)

The six dynamic states are three mRNAs and three proteins. Each gene has:

```text
dm_i/dt = b + a_i*Promoter(selected_protein, K_i) - dm_rate*m_i
dp_i/dt = kt*m_i - dp_rate*p_i
```

The promoter alternatives are activation or repression with Hill exponent 1 or 2. Three independent four-choice rules produce `4^3 = 64` operator skeletons. Regulator identity is selected by one binary toggle leaf per gene from the three protein states.

`03_gene_network.json` requests every pair among the three proteins for each regulator leaf. There are `C(3,2)^3 = 27` group variants per skeleton. Each variant has `2^3 = 8` state assignments and 32 joint coefficient vectors. The full compilation therefore produces:

`64 skeletons * 27 variants * 8 assignments * 32 vectors = 442368 configuration visits`.

The pair groups overlap: 216 state-assignment visits per skeleton cover 27 distinct regulator assignments eight times. These numbers must not be called 442,368 distinct network topologies. The coefficient vectors and whole-model parameter sharing remain fixed across whatever prepared experiments the engine jointly scores.

`04_gene_network_sampled.json` selects five joint combinations of pair groups per skeleton, yielding 81,920 configuration visits. This is five complete group combinations rather than five groups independently per leaf.

The largest fit has six scalar corrections: production and threshold for each gene. Shared mRNA loss, translation, and protein loss are fixed modeling assumptions in this illustration, not facts about arbitrary circuits. Relaxing those assumptions can exceed the LM fitted-dimension limit.

Tags include explicit combinations such as `gene_0:repression`, allowing the scoring backend to retain top models where gene 0 is repressed. A global tag such as `repression` means that a repression production occurs somewhere in the model; it does not assert that every gene is repressed.

The LLM's search could begin with these single-regulator mechanisms, then add biologically motivated two-regulator logic or alternative removal laws if the experiment supports that extra complexity. The engine's cheap structural exploration becomes valuable as those mechanism combinations grow and as the researcher repeats the search across additional experimental conditions. Independent time-resolved measurements are much more appropriate for this workflow than treating unrelated RNA-seq snapshots or pseudotime as one directly observed trajectory.

## Scaling to a million structural proposals

A possible extension, not an additional executable example in this archive, is four genes with mRNA and protein states: eight states total. Give each promoter two candidate regulators. For each regulator choose activation or repression and Hill exponent 1 or 2; combine the two bounded response functions with either `f*g` or `f+g-f*g`. That is `4*4*2 = 32` promoter alternatives and `32^4 = 1,048,576` whole-system structural derivations before deduplication. Regulator identities remain in toggle pools.

These are phenomenological logic hypotheses. Both combiners are symmetric, so swapping response laws together with regulator identities creates equivalent resolved models. Repeated regulator identities can create further overlap. A million derivations must not be advertised as a million distinct biological mechanisms.

Using only production `a_i` and one shared threshold `K_i` per promoter gives eight fitted scalars. This requires fixing all other rates and initial conditions and deliberately sharing a response threshold between the two regulators, with compatible concentration units or normalization. It is a restrictive search family, not a general four-gene fitting capability. Independent thresholds, unknown initial conditions, or additional kinetic rates would require fixing or profiling some quantities or raising the fitted-dimension limit.

## What I would improve next

1. A named parameter lifecycle covering sampled initialization, fixed values, fit selection, transforms, and reporting of fitted physical values.
2. Explicit observation mappings, calibration, noise weights, and experiment-level parameter/initial-condition scopes in the engine interface.
3. Known input time series and event handling for inducer pulses, with declared interpolation and discontinuities.
4. Reusable biological constructors with units, allowed interaction edges, and shared reaction fluxes. These should lower to the current AST representation.
5. A supported stiff integrator before advertising general stiff biological model fitting. The current grammar can write such equations, but RK4 is still the only executable integration declaration.

Run-scoped RNG in these requests explores uncertain coefficients; it does not model transcriptional bursts or stochastic reactions during integration. Continuous positive coefficients also do not guarantee that a numerical trajectory remains positive. These are substantive distinctions for biological use, not additional arithmetic syntax problems.
