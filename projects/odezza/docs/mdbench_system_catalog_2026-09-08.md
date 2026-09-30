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

# MDBench complete system catalog — 2026-09-08

Catalog of the official 63 ODE definitions and 14 PDE datasets. Descriptions explain the implemented mathematics; names and regime labels are those of the repository, not independently verified attractor classifications. No benchmark runs were launched.

Sources: [official repository](https://github.com/gryaklab/mdbench), [ODE definitions](https://raw.githubusercontent.com/gryaklab/mdbench/main/scripts/strogatz_ode.py), [ODE generator](https://github.com/gryaklab/mdbench/blob/main/scripts/generate_ode.py), and [paper, Appendix B/Table A1](https://arxiv.org/html/2509.20529v2#A2).

ODE definition source SHA-256: `fe39de6cf002d62e62c3f1d7e026b514ac5046b8c96778a610ff5ff9dc8f0958`.

## Sampling and interpretation

- ODE generator: one exported trajectory per system, 150 requested uniform observation times from 0 to 10 inclusive; spacing 10/149 = 0.06711409396. Every state is observed. The definition lists two IC vectors, but make_dataset selects solutions[0][0], the first parameter set and first initial condition.
- Each system has a clean version plus multiplicative-noise versions at nominal SNR 40, 30, 20 and 10 dB. These are perturbations of the same path, not independent initial conditions. Total: 315 ODE datasets across 63 system labels.
- Paper protocol: first 60% train, next 20% validation, last 20% test; 90/30/30 samples for a 150-point trajectory. Final fitting may use the first 120 after hyperparameter selection. The paper protocol and existing Secant exports distinguish this from a helper split implementation.
- Uniform spacing does not guarantee temporal resolution relative to every system's dynamics. The output sampling interval is not the adaptive integration solver's internal step. This is a modest temporal dataset with only one IC, not broad state-space coverage.
- PDEs are space-time fields with a spatial grid and time axis, not hundreds of independent ODE trajectories. Their processed sample counts appear below.
- MDBench evaluates derivative prediction and equation fidelity/complexity. A trajectory-rollout recovery experiment is a related but different training/evaluation protocol.
- ODE dimensions: 23 scalar, 28 two-state, 10 three-state and 2 four-state entries. It cannot by itself measure scaling to 12 or 24 ODE states.
- Several labels share functional structure: logistic growth/autocatalysis/laser, three Lorenz entries, and three Rössler entries. Entry 40 (Duffing label) implements the same form as Van der Pol entry 37. Do not count all 63 labels as structurally distinct equation families.

## ODE definitions

Each RHS is listed in state order. The notation ^ means exponentiation. Parameters c_0, c_1, ... are shown in order; equations are preserved from the benchmark, including its sign conventions. In particular the Gompertz-labelled equation has the positive x*log(b*x) convention shown below. All entries use the ODE sampling protocol above.

### 1. RC-circuit — 1 state

Capacitor charge relaxes toward the source-voltage equilibrium through a resistor; affine linear relaxation.

```text
dx_0/dt = (c_0 - x_0 / c_1) / c_2
```

Parameters: `[0.7, 1.2, 2.31]`. Exported initial state: `[10.0]`.

### 2. Population growth naive — 1 state

Population grows exponentially at a fixed per-capita rate; the simplest multiplicative growth model.

```text
dx_0/dt = c_0 * x_0
```

Parameters: `[0.23]`. Exported initial state: `[4.78]`.

### 3. Population growth carrying capacity — 1 state

Logistic population growth slows as the population approaches carrying capacity; linear and quadratic terms.

```text
dx_0/dt = c_0 * x_0 * (1 - x_0 / c_1)
```

Parameters: `[0.79, 74.3]`. Exported initial state: `[7.3]`.

### 4. RC-circuit non-linear resistor — 1 state

Capacitor model with a sigmoid current-voltage relation; exponential inside a reciprocal, rather than a linear resistor.

```text
dx_0/dt = 1 / (1 + exp(c_0 - x_0 / c_1)) - 0.5
```

Parameters: `[0.5, 0.96]`. Exported initial state: `[0.8]`.

### 5. Velocity falling object — 1 state

A falling body's velocity balances gravitational acceleration against quadratic drag, approaching terminal speed.

```text
dx_0/dt = c_0 - c_1 * x_0^2
```

Parameters: `[9.81, 0.0021175]`. Exported initial state: `[0.5]`.

### 6. Autocatalysis — 1 state

A chemical species promotes its own production while a quadratic loss limits its concentration.

```text
dx_0/dt = c_0 * x_0 - c_1 * x_0^2
```

Parameters: `[2.1, 0.5]`. Exported initial state: `[0.13]`.

### 7. Gompertz law tumor growth — 1 state

Tumour-growth-labelled model with x*log(b*x). The implemented positive sign and coefficient convention must be preserved.

```text
dx_0/dt = c_0 * x_0 * log(c_1 * x_0)
```

Parameters: `[0.032, 2.29]`. Exported initial state: `[1.73]`.

### 8. Logistic equation Allee effect — 1 state

Logistic growth with a low-population threshold: insufficient population can decline instead of recovering; cubic polynomial.

```text
dx_0/dt = c_0 * x_0 * (1 - x_0 / c_1) * (x_0 / c_2 - 1)
```

Parameters: `[0.14, 130.0, 4.4]`. Exported initial state: `[6.123]`.

### 9. Language death model — 1 state

Two-language population share changes through fixed switching rates; an affine linear balance law.

```text
dx_0/dt = (1 - x_0) * c_0 - x_0 * c_1
```

Parameters: `[0.32, 0.28]`. Exported initial state: `[0.14]`.

### 10. Refined language death model — 1 state

Language competition with frequency-dependent switching and a noninteger exponent of 1.2.

```text
dx_0/dt = (1 - x_0) * c_0 * x_0^c_1 - x_0 * (1 - c_0) * (1 - x_0)^c_1
```

Parameters: `[0.2, 1.2]`. Exported initial state: `[0.83]`.

### 11. Naive critical slowing down — 1 state

Cubic relaxation dx/dt=-x^3 slows markedly near zero, where the linear restoring term vanishes.

```text
dx_0/dt = - x_0^3
```

Parameters: `[]`. Exported initial state: `[3.4]`.

### 12. Photons in a laser — 1 state

Simplified photon production and depletion in a laser; the same linear-minus-quadratic structure as logistic growth.

```text
dx_0/dt = c_0 * x_0 - c_1 * x_0^2
```

Parameters: `[1.8, 0.1107]`. Exported initial state: `[11.0]`.

### 13. Overdamped bead — 1 state

Angular motion of a strongly damped bead on a spinning hoop; coupled sine and cosine factors.

```text
dx_0/dt = c_0 * sin(x_0) * (c_1 * cos(x_0) - 1)
```

Parameters: `[0.0981, 9.7]`. Exported initial state: `[3.1]`.

### 14. Budworm outbreak model — 1 state

Insect population growth opposed by saturating predation; logistic terms plus a rational response.

```text
dx_0/dt = c_0 * x_0 * (1 - x_0 / c_1) - c_3 * x_0^2 / (c_2^2 + x_0^2)
```

Parameters: `[0.78, 81.0, 21.2, 0.9]`. Exported initial state: `[2.76]`.

### 15. Budworm outbreak predation — 1 state

Nondimensional budworm model retaining logistic growth and saturating predation, with fewer free constants.

```text
dx_0/dt = c_0 * x_0 * (1 - x_0 / c_1) - x_0^2 / (1 + x_0^2)
```

Parameters: `[0.4, 95.0]`. Exported initial state: `[44.3]`.

### 16. Landau equation — 1 state

Order-parameter evolution with linear, cubic and quintic terms; a model for transitions and multiple equilibria.

```text
dx_0/dt = c_0 * x_0 - c_1 * x_0^3 - c_2 * x_0^5
```

Parameters: `[0.1, -0.04, 0.001]`. Exported initial state: `[0.94]`.

### 17. Logistic equation harvesting — 1 state

Logistic population growth with a fixed removal rate, representing harvesting or fishing.

```text
dx_0/dt = c_0 * x_0 * (1 - x_0 / c_1) - c_2
```

Parameters: `[0.4, 100.0, 0.3]`. Exported initial state: `[14.3]`.

### 18. Improved logistic equation harvesting — 1 state

Logistic growth with saturating population-dependent harvesting, avoiding a fixed removal rate at tiny populations.

```text
dx_0/dt = c_0 * x_0 * (1 - x_0 / c_1) - c_2 * x_0 / (c_3 + x_0)
```

Parameters: `[0.4, 100.0, 0.24, 50.0]`. Exported initial state: `[21.1]`.

### 19. Improved logistic equation harvesting dimensionless — 1 state

Rescaled version of saturating harvesting, retaining the rational x/(b+x) term.

```text
dx_0/dt = x_0 * (1 - x_0) - c_0 * x_0 / (c_1 + x_0)
```

Parameters: `[0.08, 0.8]`. Exported initial state: `[0.13]`.

### 20. Autocatalytic gene switching — 1 state

Basal production, linear degradation and saturating positive feedback in gene-product concentration.

```text
dx_0/dt = c_0 - c_1 * x_0 + x_0^2 / (1 + x_0^2)
```

Parameters: `[0.1, 0.55]`. Exported initial state: `[0.002]`.

### 21. Dimensionally reduced SIR — 1 state

Scalar epidemic reduction containing constant input, linear loss and exp(-x); not the full compartmental SIR system.

```text
dx_0/dt = c_0 - c_1 * x_0 - exp(-x_0)
```

Parameters: `[1.2, 0.2]`. Exported initial state: `[0.0]`.

### 22. Protein expression — 1 state

Protein production with basal expression and a fifth-power saturating activation term, opposed by linear decay.

```text
dx_0/dt = c_0 + c_1 * x_0^5 / (c_2 + x_0^5) - c_3 * x_0
```

Parameters: `[1.4, 0.4, 123.0, 0.89]`. Exported initial state: `[3.1]`.

### 23. Overdamped pendulum — 1 state

Strongly damped phase dynamics under constant torque; a constant term competes with sin(angle).

```text
dx_0/dt = c_0 - sin(x_0)
```

Parameters: `[0.21]`. Exported initial state: `[-2.74]`.

### 24. Harmonic oscillator — 2 states

Position and velocity of an ideal spring-mass oscillator; mutual linear coupling without damping.

```text
dx_0/dt = x_1
dx_1/dt = - c_0 * x_0
```

Parameters: `[2.1]`. Exported initial state: `[0.4, -0.03]`.

### 25. Harmonic oscillator damping — 2 states

Spring-mass motion with velocity-dependent linear damping; oscillation amplitude decays.

```text
dx_0/dt = x_1
dx_1/dt = - c_0 * x_0 - c_1 * x_1
```

Parameters: `[4.5, 0.43]`. Exported initial state: `[0.12, 0.043]`.

### 26. Lotka-Volterra competition — 2 states

Two species compete for resources; both self-limitation and cross-species inhibition enter quadratically.

```text
dx_0/dt = x_0 * (c_0 - x_0 - c_1 * x_1)
dx_1/dt = x_1 * (c_2 - x_0 - x_1)
```

Parameters: `[3.0, 2.0, 2.0]`. Exported initial state: `[5.0, 4.3]`.

### 27. Lotka-Volterra simple — 2 states

Predator-prey populations coupled by encounter products, with prey growth and predator mortality.

```text
dx_0/dt = x_0 * (c_0 - c_1 * x_1)
dx_1/dt = - x_1 * (c_2 - c_3 * x_0)
```

Parameters: `[1.84, 1.45, 3.0, 1.62]`. Exported initial state: `[8.3, 3.4]`.

### 28. Pendulum without friction — 2 states

Pendulum angle and angular velocity with a full sine restoring force; nonlinear oscillations without friction.

```text
dx_0/dt = x_1
dx_1/dt = - c_0 * sin(x_0)
```

Parameters: `[0.9]`. Exported initial state: `[-1.9, 0.0]`.

### 29. Dipole fixed point — 2 states

A planar quadratic vector field illustrating a degenerate fixed-point geometry; xy and squared-state terms.

```text
dx_0/dt = c_0 * x_0 * x_1
dx_1/dt = x_1^2 - x_0^2
```

Parameters: `[0.65]`. Exported initial state: `[3.2, 1.4]`.

### 30. Catalyzing RNA molecules — 2 states

Two RNA species promote each other's replication, with nonlinear saturation through shared products.

```text
dx_0/dt = x_0 * (x_1 - c_0 * x_0 * x_1)
dx_1/dt = x_1 * (x_0 - c_0 * x_0 * x_1)
```

Parameters: `[1.61]`. Exported initial state: `[0.3, 0.04]`.

### 31. SIR infection — 2 states

Susceptible and infected populations: infection transfers people between compartments; recovery removes infected people.

```text
dx_0/dt = - c_0 * x_0 * x_1
dx_1/dt = c_0 * x_0 * x_1 - c_1 * x_1
```

Parameters: `[0.4, 0.314]`. Exported initial state: `[7.2, 0.98]`.

### 32. Damped double well oscillator — 2 states

Position and velocity in a two-well potential with damping; linear and cubic restoring terms.

```text
dx_0/dt = x_1
dx_1/dt = - c_0 * x_1 + x_0 - x_0^3
```

Parameters: `[0.18]`. Exported initial state: `[-1.8, -1.8]`.

### 33. Glider — 2 states

Glider speed and flight angle; trigonometric forces, quadratic drag and an inverse-speed term.

```text
dx_0/dt = - sin(x_1) - c_0 * x_0^2
dx_1/dt = x_0 - cos(x_1) / x_0
```

Parameters: `[0.08]`. Exported initial state: `[5.0, 0.7]`.

### 34. Frictionless bead — 2 states

A bead's angle and angular velocity on a rotating hoop; inertia plus a sine-times-cosine force.

```text
dx_0/dt = x_1
dx_1/dt = sin(x_0) * (cos(x_0) - c_0)
```

Parameters: `[0.93]`. Exported initial state: `[2.1, 0.0]`.

### 35. Rotational dynamics — 2 states

Orientation angles of an object in shear flow; cotangent and trigonometric products introduce coordinate singularities.

```text
dx_0/dt = cot(x_1) * cos(x_0)
dx_1/dt = sin(x_0) * (cos(x_1)^2 + c_0 * sin(x_1)^2)
```

Parameters: `[4.2]`. Exported initial state: `[1.13, -0.3]`.

### 36. Pendulum non-linear damping — 2 states

Pendulum motion with angle-dependent damping, including a cos(angle)*angular-velocity interaction.

```text
dx_0/dt = x_1
dx_1/dt = - sin(x_0) - x_1 - c_0 * cos(x_0) * x_1
```

Parameters: `[0.07]`. Exported initial state: `[0.45, 0.9]`.

### 37. Van der Pol oscillator — 2 states

Self-excited oscillator: damping changes sign with amplitude, supporting a sustained oscillation.

```text
dx_0/dt = x_1
dx_1/dt = - x_0 - c_0 * (x_0^2 - 1) * x_1
```

Parameters: `[0.43]`. Exported initial state: `[2.2, 0.0]`.

### 38. Van der Pol oscillator simplified — 2 states

Alternative Van der Pol coordinates with cubic dynamics in one equation and a linear second equation.

```text
dx_0/dt = c_0 * (x_1 - x_0^3 / 3 + x_0)
dx_1/dt = - x_0 / c_0
```

Parameters: `[3.37]`. Exported initial state: `[0.7, 0.0]`.

### 39. Glycolytic oscillator — 2 states

Two metabolite concentrations coupled through cubic reaction terms; the benchmark's glycolysis oscillator model.

```text
dx_0/dt = - x_0 + c_0 * x_1 + x_0^2 * x_1
dx_1/dt = c_1 - c_0 * x_0 - x_0^2 * x_1
```

Parameters: `[2.4, 0.07]`. Exported initial state: `[0.4, 0.31]`.

### 40. Duffing equation — 2 states

Named Duffing, but its implemented RHS is algebraically the same form as entry 37, with a different coefficient.

```text
dx_0/dt = x_1
dx_1/dt = - x_0 + c_0 * x_1 * (1 - x_0^2)
```

Parameters: `[0.886]`. Exported initial state: `[0.63, -0.03]`.

### 41. Cell cycle model — 2 states

Two interacting cell-cycle regulators with nonlinear feedback and separated parameter scales.

```text
dx_0/dt = c_0 * (x_1 - x_0) * (c_1 + x_0^2) - x_0
dx_1/dt = c_2 - x_0
```

Parameters: `[15.3, 0.001, 0.3]`. Exported initial state: `[0.8, 0.3]`.

### 42. CDIMA reaction — 2 states

Reduced chemical reaction model with shared rational saturation terms involving 1+x^2.

```text
dx_0/dt = c_0 - x_0 - c_1 * x_0 * x_1 / (1 + x_0^2)
dx_1/dt = c_2 * x_0 * (1 - x_1 / (1 + x_0^2))
```

Parameters: `[8.9, 4.0, 1.4]`. Exported initial state: `[0.2, 0.35]`.

### 43. Driven pendulum linear damping — 2 states

Pendulum/Josephson-junction dynamics under constant driving torque with linear velocity damping.

```text
dx_0/dt = x_1
dx_1/dt = c_0 - sin(x_0) - c_1 * x_1
```

Parameters: `[1.67, 0.64]`. Exported initial state: `[1.47, -0.2]`.

### 44. Driven pendulum quadratic damping — 2 states

Same constant-drive setting, with signed quadratic drag v*abs(v) instead of linear damping.

```text
dx_0/dt = x_1
dx_1/dt = c_0 - sin(x_0) - c_1 * x_1 * abs(x_1)
```

Parameters: `[1.67, 0.64]`. Exported initial state: `[1.47, -0.2]`.

### 45. Gray-Scott model — 2 states

Two-species autocatalytic chemistry with a shared x*y^2 reaction term, feed and depletion; spatially homogeneous ODE version.

```text
dx_0/dt = c_0 * (1 - x_0) - x_0 * x_1^2
dx_1/dt = x_0 * x_1^2 - c_1 * x_1
```

Parameters: `[0.5, 0.02]`. Exported initial state: `[1.4, 0.2]`.

### 46. Interacting bar magnets — 2 states

Two magnet angles interact through sine of their angle difference and individual restoring torques.

```text
dx_0/dt = c_0 * sin(x_0 - x_1) - sin(x_0)
dx_1/dt = c_0 * sin(x_1 - x_0) - sin(x_1)
```

Parameters: `[0.33]`. Exported initial state: `[0.54, -0.1]`.

### 47. Binocular rivalry model — 2 states

Two competing perceptual activities inhibit each other through sigmoid responses; no adaptation state.

```text
dx_0/dt = - x_0 + 1 / (1 + exp(c_0 * x_1 - c_1))
dx_1/dt = - x_1 + 1 / (1 + exp(c_0 * x_0 - c_1))
```

Parameters: `[4.89, 1.4]`. Exported initial state: `[0.65, 0.59]`.

### 48. Bacterial respiration model — 2 states

Nutrient and oxygen concentrations linked through a shared nonlinear rational consumption rate.

```text
dx_0/dt = c_0 - x_0 - x_0 * x_1 / (1 + c_1 * x_0^2)
dx_1/dt = c_2 - x_0 * x_1 / (1 + c_1 * x_0^2)
```

Parameters: `[18.3, 0.48, 11.23]`. Exported initial state: `[0.1, 30.4]`.

### 49. Brusselator — 2 states

A chemical oscillator with feed, linear conversion and the cubic autocatalytic term x^2*y.

```text
dx_0/dt = 1 - (c_0 + 1) * x_0 + c_1 * x_0^2 * x_1
dx_1/dt = c_0 * x_0 - c_1 * x_0^2 * x_1
```

Parameters: `[3.03, 3.1]`. Exported initial state: `[0.7, -1.4]`.

### 50. Schnackenberg model — 2 states

Two-species chemical kinetics with constant feeds and an x^2*y conversion term.

```text
dx_0/dt = c_0 - x_0 + x_0^2 * x_1
dx_1/dt = c_1 - x_0^2 * x_1
```

Parameters: `[0.24, 1.43]`. Exported initial state: `[0.14, 0.6]`.

### 51. Oscillator death model — 2 states

Two phase variables have different constant drives and a shared sin(y)*cos(x) term; their difference has constant derivative.

```text
dx_0/dt = c_0 + sin(x_1) * cos(x_0)
dx_1/dt = c_1 + sin(x_1) * cos(x_0)
```

Parameters: `[1.432, 0.972]`. Exported initial state: `[2.2, 0.67]`.

### 52. Maxwell-Bloch equations — 3 states

Laser field, atomic polarization and population inversion coupled by bilinear products and decay/pumping terms.

```text
dx_0/dt = c_0 * (x_1 - x_0)
dx_1/dt = c_1 * (x_0 * x_2 - x_1)
dx_2/dt = c_2 * (c_3 + 1 - x_2 - c_3 * x_0 * x_1)
```

Parameters: `[0.1, 0.21, 0.34, 3.1]`. Exported initial state: `[1.3, 1.1, 0.89]`.

### 53. Apoptosis model — 3 states

Cell-death reaction model with several rational kinetic terms; the last two derivatives cancel, conserving their state sum.

```text
dx_0/dt = c_0 - c_5 * x_1 * x_0 / (c_9 + x_0) - c_4 * x_0
dx_1/dt = c_1 * x_2 * (c_8 + x_1) - c_2 * x_1 / (c_6 + x_1) - c_3 * x_0 * x_1 / (c_7 + x_1)
dx_2/dt = - c_1 * x_2 * (c_8 + x_1) + c_2 * x_1 / (c_6 + x_1) + c_3 * x_0 * x_1 / (c_7 + x_1)
```

Parameters: `[0.1, 0.6, 0.2, 7.95, 0.05, 0.4, 0.1, 2.0, 0.1, 0.1]`. Exported initial state: `[0.005, 0.26, 2.15]`.

### 54. Lorenz equations periodic — 3 states

Lorenz vector field at the lower-driving parameter set (sigma,rho,beta)=(5.1,12,1.67); repository label is periodic.

```text
dx_0/dt = c_0 * (x_1 - x_0)
dx_1/dt = c_1 * x_0 - x_1 - x_0 * x_2
dx_2/dt = x_0 * x_1 - c_2 * x_2
```

Parameters: `[5.1, 12.0, 1.67]`. Exported initial state: `[2.3, 8.1, 12.4]`.

### 55. Lorenz equations complex periodic — 3 states

The same Lorenz structure at (10,99.96,8/3), labelled complex periodic by the repository.

```text
dx_0/dt = c_0 * (x_1 - x_0)
dx_1/dt = c_1 * x_0 - x_1 - x_0 * x_2
dx_2/dt = x_0 * x_1 - c_2 * x_2
```

Parameters: `[10.0, 99.96, 2.6666666666666665]`. Exported initial state: `[2.3, 8.1, 12.4]`.

### 56. Lorenz equations chaotic — 3 states

Lorenz system at its familiar chaotic parameters (10,28,8/3); simple quadratic equations with sensitive trajectories.

```text
dx_0/dt = c_0 * (x_1 - x_0)
dx_1/dt = c_1 * x_0 - x_1 - x_0 * x_2
dx_2/dt = x_0 * x_1 - c_2 * x_2
```

Parameters: `[10.0, 28.0, 2.6666666666666665]`. Exported initial state: `[2.3, 8.1, 12.4]`.

### 57. Rössler fixed point — 3 states

Rössler system at a stable-fixed-point-labelled setting; mostly linear equations plus one bilinear coupling.

```text
dx_0/dt = c_3 * (- x_1 - x_2)
dx_1/dt = c_3 * (x_0  + c_0 * x_1)
dx_2/dt = c_3 * (c_1 + x_2 * (x_0 - c_2))
```

Parameters: `[-0.2, 0.2, 5.7, 5.0]`. Exported initial state: `[2.3, 1.1, 0.8]`.

### 58. Rössler attractor periodic — 3 states

The same Rössler structure in the periodic-labelled parameter setting.

```text
dx_0/dt = c_3 * (- x_1 - x_2)
dx_1/dt = c_3 * (x_0  + c_0 * x_1)
dx_2/dt = c_3 * (c_1 + x_2 * (x_0 - c_2))
```

Parameters: `[0.1, 0.2, 5.7, 5.0]`. Exported initial state: `[2.3, 1.1, 0.8]`.

### 59. Rössler attractor chaotic — 3 states

The same Rössler structure in the chaotic-labelled parameter setting; changing coefficients changes behavior.

```text
dx_0/dt = c_3 * (- x_1 - x_2)
dx_1/dt = c_3 * (x_0  + c_0 * x_1)
dx_2/dt = c_3 * (c_1 + x_2 * (x_0 - c_2))
```

Parameters: `[0.2, 0.2, 5.7, 5.0]`. Exported initial state: `[2.3, 1.1, 0.8]`.

### 60. Aizawa attractor — 3 states

Chaotic three-state vector field with rotational coupling and cubic/quartic polynomial interactions; structurally richer than Lorenz.

```text
dx_0/dt = x_0 * (x_2 - c_1) - c_3 * x_1
dx_1/dt = c_3 * x_0 + x_1 * (x_2 - c_1)
dx_2/dt = c_2 + c_0 * x_2 - x_2^3 / 3. - (x_0^2 + x_1^2) * (1 + c_4 * x_2) + c_5 * x_2 * x_0^3
```

Parameters: `[0.95, 0.7, 0.65, 3.5, 0.25, 0.1]`. Exported initial state: `[0.1, 0.05, 0.05]`.

### 61. Chen-Lee attractor — 3 states

A rigid-body feedback-inspired chaotic model using angular-velocity-like states and quadratic cross-products.

```text
dx_0/dt = c_0 * x_0 - x_1 * x_2
dx_1/dt = c_1 * x_1 + x_0 * x_2
dx_2/dt = c_2 * x_2 + x_0 * x_1 / c_3
```

Parameters: `[5.0, -10.0, -3.8, 3.0]`. Exported initial state: `[15, -15, -15]`.

### 62. Binocular rivalry adaptation — 4 states

Competing perceptual activities plus an adaptation variable for each; sigmoid inhibition and delayed feedback.

```text
dx_0/dt = - x_0 + 1 / (1 + exp(c_0 * x_2 + c_1 * x_1 - c_2))
dx_1/dt = c_3 * (x_0 - x_1)
dx_2/dt = - x_2 + 1 / (1 + exp(c_0 * x_0 + c_1 * x_3 - c_2))
dx_3/dt = c_3 * (x_2 - x_3)
```

Parameters: `[0.89, 0.4, 1.4, 1.0]`. Exported initial state: `[2.25, -0.5, -1.13, 0.4]`.

### 63. SEIR infection — 4 states

Susceptible, exposed, infected and recovered compartments; sparse infection/progression/recovery transfers conserve total population.

```text
dx_0/dt = - c_1 * x_0 * x_2
dx_1/dt = c_1 * x_0 * x_2 - c_0 * x_1
dx_2/dt = c_0 * x_1 - c_2 * x_2
dx_3/dt = c_2 * x_2
```

Parameters: `[0.47, 0.28, 0.3]`. Exported initial state: `[0.6, 0.3, 0.09, 0.01]`.

## All 14 PDE datasets

Counts below describe the processed benchmark data in the paper's Table A1, not the larger original simulation outputs. Each row is one space-time dataset. Fields are state channels; spatial grid cells are not independent initial-condition trajectories.

| Dataset | Fields | Time samples | Time spacing | Spatial grid | Description |
|---|---:|---:|---:|---|---|
| Advection | 1 | 201 | 0.01 | 1024 | Transport of a scalar wave/profile at constant speed; first spatial derivative. |
| Burgers | 1 | 101 | 0.1 | 256 | Nonlinear transport competes with viscous smoothing; steepening fronts and diffusion. |
| Korteweg–de Vries | 1 | 201 | 0.1 | 512 | Nonlinear dispersive waves; combines u*u_x and a third spatial derivative. |
| Kuramoto–Sivashinsky | 1 | 251 | 0.4 | 1024 | Nonlinear transport with second- and fourth-order terms; irregular pattern dynamics. |
| Advection–diffusion | 1 | 61 | 0.1 | 51 × 51 | A scalar field transported and spread in two spatial dimensions. |
| Heat (Solar) 1D | 1 | 576 | 300 | 51 | Temperature diffuses through soil depth under a time-varying surface temperature. |
| Heat (Solar) 2D | 1 | 576 | 300 | 51 × 51 | The same thermal process on a two-dimensional spatial domain. |
| Heat (Solar) 3D | 1 | 20 | 300 | 51 × 51 × 11 | Three-dimensional thermal field; the benchmark uses only twenty time slices. |
| Heat (Laser) | 1 | 20 | 0.1 | 201 × 201 × 3 | A moving localized laser heats a three-dimensional plate; spatially and temporally varying forcing. |
| Nonlinear Schrödinger | 2 | 251 | 0.0126 | 256 | A complex wave represented by two real fields; dispersion and cubic amplitude interactions. |
| Reaction–diffusion | 2 | 100 | 0.101 | 32 × 32 | Two coupled nonlinear fields diffuse in two dimensions and produce patterns. |
| Navier–Stokes channel | 2 | 50 | 0.02 | 9 × 9 | Pressure-driven incompressible channel flow; the processed dataset has two state channels. |
| Navier–Stokes cylinder | 3 | 51 | 0.02 | 100 × 30 | Velocity and pressure around an obstacle; includes a wake and vortex shedding. |
| Reaction–diffusion cylinder | 6 | 51 | 0.02 | 100 × 30 | Three reacting chemical concentrations combined with the flow/pressure fields; transport, diffusion and reaction. |

## Relevance to Odezza and feature screening

The rational biological/chemical systems, nonlinear resistor, binocular rivalry and trigonometric mechanics broaden the operator coverage beyond the recent polynomial/sin/cos/tanh depth-two pilots. SEIR and adaptation dynamics provide useful sparse dependencies, while apoptosis has a conserved combination that can create input ambiguity from one trajectory. Chaotic cases test long-horizon fidelity even when the algebra is small.

For dependency inference, the official ODE trajectories provide limited initial-condition coverage. Regenerating multiple ICs from these public equations would be a valuable, explicitly separate experiment. Preserve official data and scoring for published benchmark comparisons. PDEs require spatial-derivative handling, boundary/forcing semantics and a spatial solver or suitable discretization; they do not plug directly into the present ODE rollout path.

## Attribution

The equation definitions and parameter metadata are adapted from gryaklab/mdbench (itself adapted from ODEFormer's ODEBench definitions). The source repository license follows. Descriptions and benchmark implications above are this review's analysis.

```text
MIT License

Copyright (c) 2025 Jonathan Gryak

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
```
