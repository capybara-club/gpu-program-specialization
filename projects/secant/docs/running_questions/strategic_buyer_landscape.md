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

# Strategic Buyer Landscape for Secant System Identification

Last reviewed: 2026-08-25

## Current conclusion

Secant has plausible strategic buyers well beyond pharmacometrics. The common
opportunity is an incumbent that already owns trusted simulation models,
libraries, workflows, and customer relationships but lacks fast, interpretable
discovery of missing dynamical laws.

The product is not most valuable as a general ODE solver. Its differentiated
form is a model-completion engine:

> Given a mostly trusted dynamical model, experimental trajectories, and one or
> a few unknown rate laws, search constrained expression families, fit each
> candidate's constants, and return a validated shortlist of interpretable
> mechanisms.

The strongest initial market outside pharmacology is fed-batch fermentation and
bioprocess development. Chemical and industrial process modeling is the largest
nearby expansion. Batteries are a good second demonstration. Reservoir and
large multiphysics applications show strong acquisition demand, but much of
their computation is outside the current small, register-resident ODE regime.

## Market map

| Domain | Candidate unknowns | Representative strategic buyers | Technical fit | Acquisition potential |
|---|---|---|---:|---:|
| Pharmacology, QSP, and PBPK | Absorption, clearance, saturation, inhibition, and response laws | Certara, Simulations Plus, PumasAI | Excellent | Excellent |
| Bioprocess and fermentation | Growth, uptake, production, inhibition, and death laws | Sartorius, Siemens, Emerson/AspenTech, Cytiva | Excellent | Excellent |
| Chemical and process plants | Reaction kinetics, fouling, transfer, and degradation laws | Siemens, Emerson/AspenTech, Honeywell, Modelon | Excellent | Excellent |
| Batteries and electrochemistry | Aging, thermal, resistance, hysteresis, and reaction laws | Synopsys/Ansys, Siemens, AVL, MathWorks, Modelon | Good | Good |
| Oil and gas | Production, injection, well, and reduced-order reservoir dynamics | SLB, Emerson/AspenTech, Honeywell | Mixed | Excellent |
| Automotive and aerospace | Component, thermal, actuator, and degradation dynamics | Siemens, Synopsys/Ansys, AVL, MathWorks, Dassault Systemes | Good | Moderate to high |
| Energy and HVAC | Heat-pump, turbine, building, and storage dynamics | Siemens, Emerson, Honeywell, Modelon | Good | Moderate |
| Robotics and control | Interpretable plant, actuator, and fault dynamics | MathWorks, Siemens, dSPACE, AVL | Moderate | Moderate |
| Ecology and epidemiology | Population, competition, and transmission mechanisms | Primarily research-tool vendors | Good | Low |
| CFD, weather, and general spatial PDEs | Unknown spatial constitutive laws | Synopsys/Ansys, Siemens, Dassault Systemes | Poor for the present kernel | Long-term |

The ratings concern the present Secant trajectory-search shape, not the total
market size. A domain can have enormous simulation spending while still being a
poor fit for forward-sensitivity LM over a small ODE state.

## 1. Pharmacology, QSP, and PBPK

### Product fit

Existing packages simulate trusted mechanistic models, virtual populations,
clinical scenarios, and fitted parameters. Secant's novel insertion point is
model-structure discovery: hold the trusted physiology and known equations
fixed while searching a missing absorption, clearance, transport, inhibition,
or response term.

The relevant strategic buyers are:

- **Simulations Plus**, including GastroPlus, Monolix, and its QSP portfolio;
- **Certara**, including Simcyp and its broader model-informed drug-development
  platform; and
- **PumasAI**, whose platform covers nonlinear mixed-effects modeling, QSP,
  Bayesian estimation, and clinical-trial simulation.

There is direct acquisition precedent. Simulations Plus acquired Immunetrics
for $15.5 million at closing plus up to $8 million in earn-outs to add QSP
software, models, services, and domain expertise:
[Immunetrics acquisition](https://www.simulations-plus.com/resource/simulations-plus-acquires-immunetrics-to-expand-its-immunology-and-oncology-drug-development-capabilities/).
It previously acquired Lixoft to add the Monolix population-PK/PD platform:
[Lixoft acquisition](https://www.simulations-plus.com/resource/simulations-plus-acquires-lixoft-expanding-modeling-software-offerings-and-broadening-presence-in-europe/).

Certara also buys complementary scientific software. It paid $90 million for
Chemaxon in 2024 after a ten-year partnership:
[Chemaxon acquisition rationale and price](https://ir.certara.com/static-files/4579c7ba-08e6-4a05-83a3-75431c0cae3b).

### Constraint

This market has unusually high validation, auditability, and regulatory
expectations. A fast evaluator alone has little value unless its recovered
models reproduce in trusted solvers, generalize to held-out experiments, and
produce a complete search and fitting record.

## 2. Bioprocess and fermentation

### Why this is the best initial wedge

Fed-batch bioprocess models commonly have:

- small state vectors;
- several controlled experimental trajectories;
- known mass balances with uncertain biological rate laws;
- expensive physical experiments and failed batches;
- strong demand for interpretable mechanisms; and
- enough repeated simulation and fitting to reward specialization.

This matches Secant more closely than a large, stiff, spatially distributed
engineering model. The most useful unknowns include substrate uptake, biomass
growth, inhibition, product formation, oxygen limitation, maintenance, and
cell-death laws.

Sartorius already sells statistical digital-twin and bioprocess modeling
products. Its SIMCA platform covers multivariate modeling, batch-process
analysis, prediction, and real-time monitoring:
[SIMCA](https://www.sartorius.com/en/products/process-analytical-technology/data-analytics-software/mvda-software/simca).
Sartorius explicitly names bioprocessing software as a category in which it
acquires complementary technologies:
[Sartorius acquisition strategy](https://www.sartorius.com/en/company/innovation/acquisitions).

Other plausible buyers include Siemens through gPROMS, Emerson/AspenTech,
Cytiva, and vendors building bioreactor digital twins or process-development
software.

### Product pitch

> Existing tools calibrate a model selected by a scientist. Secant searches for
> the missing biological mechanism and calibrates each candidate against full
> trajectories while keeping the simulation, sensitivities, and objective
> fused on the GPU.

## 3. Chemical and industrial process modeling

Siemens gPROMS, AspenTech, and Honeywell already provide high-fidelity process
models, parameter estimation, validation, optimization, and operational digital
twins. Their typical workflow still begins with an engineer selecting or
writing the model structure.

- gPROMS supports calibration against experimental or plant data, model
  validation, global analysis, and multivariate optimization:
  [gPROMS Process](https://www.siemens.com/en-us/products/gproms/process/).
- Aspen Plus Dynamics includes parameter fitting, data reconciliation, and
  steady-state and dynamic optimization:
  [Aspen Plus Dynamics](https://home.aspentech.com/en/products/pages/aspen-plus-dynamics).
- Honeywell Process Digital Twin combines process simulation and plant data for
  monitoring, prediction, scenario analysis, and troubleshooting:
  [Honeywell Process Digital Twin](https://process.honeywell.com/us/en/products/industrial-software/process-optimization/process-digital-twin).

The acquisition pattern is established. Siemens acquired Process Systems
Enterprise specifically for the gPROMS advanced process-modeling platform:
[Siemens and PSE](https://press.siemens.com/global/en/pressrelease/siemens-plans-acquire-process-systems-enterprise).
Siemens later acquired Altair for approximately $10 billion to expand its
simulation, HPC, data-science, and AI portfolio:
[Siemens and Altair](https://newsroom.sw.siemens.com/en-US/siemens-altair-engineering-closing/).
Emerson combined its industrial software businesses with AspenTech to create a
broader industrial software platform:
[Emerson and AspenTech](https://www.emerson.com/en/corporate/news/2022/emerson-and-aspentech-complete-transaction).

The promising Secant scope is low-order reactor, tank, thermal, catalytic,
fouling, degradation, and transfer dynamics. Large flowsheets with stiff DAEs,
events, and hundreds of states require additional solver regimes.

## 4. Batteries and electrochemistry

Battery tools already perform model calibration against experimental
trajectories. Ansys, for example, exposes Levenberg-Marquardt fitting for fixed
equivalent-circuit and thermal-abuse model structures:
[Ansys battery parameter estimation](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/flu_ug/flu_bat_MSMD_sec_use_battery_model.html).

Secant could add structure discovery for compact resistance, hysteresis,
thermal-generation, side-reaction, and aging laws. Likely buyers include
Synopsys/Ansys, Siemens Simcenter, AVL, MathWorks, and Modelon.

Modelon is a particularly clear integration candidate. Modelon Impact is an
open, Modelica-based system-simulation platform, while its current calibration
library documents important limitations: Nelder-Mead only, small calibration
problems, and no trajectory-based calibration beyond comparing the final
result point:
[Modelon calibration](https://help.modelon.com/latest/library_documentation/users_guide/calibration/general_information/).

Battery electrochemistry can become stiff or spatially distributed. The first
demonstration should therefore use a compact equivalent-circuit or lumped
electrothermal model, not a full porous-electrode PDE.

## 5. Oil, gas, and subsurface modeling

This sector has strong appetite for fast physics-guided optimization. In May
2026, SLB announced an agreement to acquire Tachyus, a company specializing in
high-speed, physics-based reservoir modeling and optimization. SLB highlighted
the ability to evaluate thousands of reservoir scenarios in minutes and to
connect the technology to closed-loop reservoir and production workflows:
[SLB and Tachyus](https://www.slb.com/newsroom/updates/2026/2026-0528-slb-tachyus-update).

This is a close strategic precedent: an incumbent with trusted workflows buys a
small differentiated modeling engine that makes rapid exploration practical.

Full reservoir simulation is not an immediate Secant target because it is
normally a large spatial PDE problem. Suitable entry points are reduced-order
production models, well dynamics, injection response, and other compact lumped
systems.

## 6. Automotive, aerospace, energy, and HVAC

These markets use Modelica, Simulink, system simulation, reduced-order models,
and digital twins extensively. Candidate buyers include Siemens, Synopsys/Ansys,
AVL, MathWorks, Dassault Systemes, and Modelon.

MathWorks already offers linear and nonlinear system identification,
grey-box parameter estimation, nonlinear ARX and Hammerstein-Wiener models, and
neural ODEs:
[System Identification Toolbox](https://www.mathworks.com/products/sysid.html).
Secant would need to show that constrained symbolic ODE completion finds
smaller, interpretable models faster or more reliably than those existing
families.

This is a large market but a harder initial sale. Models often contain DAEs,
events, switching, contact, multiple physical domains, and certification
requirements. A compact actuator, thermal-management, battery, fuel-cell, or
degradation example is more credible than claiming arbitrary vehicle or
aircraft discovery.

## Domains to defer

| Domain or setup | Why it should not lead the project |
|---|---|
| Full CFD, weather, and spatial PDE discovery | State fields do not fit the present register-resident topology |
| Large stiff reaction networks and general DAEs | Require implicit solves, adaptive stepping, and different sensitivities |
| Long-horizon chaotic systems | Pointwise trajectory loss and gradients become unstable |
| Molecular dynamics | A different force-evaluation and neighbor-list workload |
| Neural ODE training | Tensor and reverse-mode weight optimization dominate |
| Finance | Weak mechanistic identifiability and limited value from interpretable physical laws |
| Broad ecology and epidemiology tooling | Scientifically suitable but generally weaker software acquisition budgets |
| Unrestricted discovery of an entire large system | Search ambiguity dominates evaluator throughput |

## Who is most approachable

The companies with the most money are not necessarily the best early contacts.
Large vendors generally wait for domain validation, customers, and low
integration risk.

| Practical order | Organization | Reason |
|---:|---|---|
| 1 | Simulations Plus | Direct domain fit and repeated acquisitions of small modeling businesses |
| 2 | Sartorius | Excellent fermentation fit and an explicit bioprocess-software acquisition strategy |
| 3 | Modelon | Open equation-based platform with a visible trajectory-calibration gap |
| 4 | PumasAI | Technically sophisticated pharmacometrics platform and plausible partnership scale |
| 5 | Certara | Enormous strategic fit but high regulatory and integration expectations |
| 6 | Siemens gPROMS | Excellent process-modeling fit after industrial validation |
| 7 | Emerson/AspenTech | Strong customer access, but likely to require plant or process evidence |
| 8 | Synopsys/Ansys or AVL | Credible after a battery or thermal-system demonstration |
| 9 | SLB | Relevant after a reduced-order production or reservoir demonstration |

## What a buyer would actually acquire

A source tree alone is the least valuable transaction form. More credible forms
are:

1. An embeddable engine with a stable problem grammar and API.
2. A field-limited or exclusive license with continuing engineering support.
3. A small company containing the IP, its inventor, benchmarks, and domain
   demonstrations.
4. A product with one or two paid design partners and a repeatable workflow.

Direct SASS specialization strengthens the performance moat but creates buyer
concerns about support across GPU generations. Documentation, generated-code
validation, architecture-specific tests, a portable fallback, and continued
access to the inventor materially reduce that risk.

## Evidence required before outreach

The shortest path from promising technology to a buyable capability is:

1. Recover several blinded laws in the fed-batch fermentation benchmark across
   repeated seeds.
2. Compare wall-clock time to a correct held-out model against PySR, standard
   parameter fitting, and a reasonable manual or enumerative baseline.
3. Reproduce finalists in an independent, high-accuracy CPU solver.
4. Demonstrate a second vertical, preferably compact PK/PD or battery dynamics,
   using the same problem grammar and execution engine.
5. Package known equations, unknown sites, allowed operators, units, bounds,
   experiments, and validation splits through a stable YAML or API schema.
6. Produce complete audit records, invalid-trajectory diagnostics,
   identifiability checks, and parameter stability results.
7. Establish clean code and data ownership and review patent strategy before
   disclosing the implementation in detail.
8. Obtain one expert evaluation, design partnership, or paid pilot.

The decisive customer metric is not raw row evaluations per second. It is the
time, experiments, or failed development cycles saved in reaching a trustworthy
mechanistic hypothesis.

## Open questions

- Does constrained structure discovery solve a recurring problem that domain
  teams will pay to avoid, or is manual structure selection rarely the
  bottleneck?
- Which first benchmark has sufficiently credible published trajectories and
  blinded laws to persuade an industrial modeler?
- Is the best initial transaction a search service, an SDK license, a product
  partnership, or a company acquisition?
- How much portability and long-term GPU support will an incumbent require
  before accepting direct SASS specialization?
- Can the same grammar and audit record support bioprocess, PK/PD, and battery
  demonstrations without becoming an unrestricted modeling language?

These questions should be resolved with domain interviews and pilots rather
than further evaluator microbenchmarks alone.
