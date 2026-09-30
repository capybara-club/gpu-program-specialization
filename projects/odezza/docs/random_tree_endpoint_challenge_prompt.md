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

# Random-tree endpoint challenge: generator handoff

Proposed 2026-09-14. This is a generation specification, not an implemented
grammar or evidence of a CPU/GPU speed advantage. Copy the text below to the
independent generating LLM. Give the solving LLM only the resulting public JSON.

---

Create one reproducible, randomly generated ODE discovery challenge. Implement
and run a standalone Python generator using NumPy and SciPy. Generate actual
trajectories by numerical integration; do not invent observations or manually
choose the hidden expressions. The purpose is to compare recovery methods on
random expression trees, without selecting cases according to which method wins.

1. System and observations

Use six states x0 through x5. Publish these four known equations exactly:

    x0' = -0.5*x0 + 0.8*tanh(x2 + 0.4*x4*x5) + 0.3*sin(x1)
    x1' = -0.5*x1 + 0.7*tanh(x3 - 0.5*x2*x5) + 0.3*sin(x0)
    x2' = -0.6*x2 + 0.8*tanh(x5 + 0.4*x0*x3)
    x3' = -0.6*x3 + 0.7*tanh(x4 + 0.5*x1*x2)

The two unknown equations are:

    x4' = -0.5*x4 + A4*E4(x)
    x5' = -0.4*x5 + A5*E5(x)

The damping, amplitude range, and expression grammar are public. Independently
sample the two hidden expression trees and their coefficients. Observe only x0
and x1 at t=0 and t=2. Supply all six initial state values for every trajectory.
Do not supply time derivatives, intermediate observations, or later hidden-state
values.

2. Exact random-tree grammar

For this first tier, each expression body E has maximum depth exactly 3: at
least one root-to-leaf path reaches depth 3, but other paths may stop earlier.
A leaf has depth 0; a unary or binary operator has depth one plus the maximum
depth of its children. The fixed damping and outer amplitude A are outside the
body-depth count. Arithmetic involving a constant inside E counts normally.

To generate a body with remaining depth d:

- At d=0, generate a leaf.
- Otherwise choose binary with probability 0.65, unary with probability 0.25,
  or leaf with probability 0.10.
- Binary: uniformly choose +, -, or *, then independently generate two ordered
  children with remaining depth d-1.
- Unary: uniformly choose sin, cos, or tanh, then generate one child with
  remaining depth d-1.
- Leaf: choose a state with probability 0.80, otherwise a fresh independent
  numeric constant. Each state leaf chooses uniformly from x0,...,x5; repetitions
  are allowed.

Sample each numeric constant's magnitude uniformly from [0.3,1.2] and its sign
uniformly from {-1,+1}. Sample each outer amplitude's magnitude uniformly from
[0.4,1.0] with an independent random sign. Different constant occurrences are
independent, not shared parameters.

Reject and redraw a body unless it has depth exactly 3, at least three state-leaf
occurrences, at least two distinct state indices, and at most three constant
leaves. E4 must contain x5; E5 must contain x4. Thus each RHS has one to four
unknown coefficients including its amplitude. Publish all these restrictions.

This is a recursive sampling distribution conditioned on the stated rejection
rules; do not describe it as uniform over all ASTs or mathematical functions.
Do not impose a numerator/denominator template, require a particular unary root,
or require every binary branch to reach the depth limit. Do not add implicit
protected operators, clipping, normalization, or coefficients at every node.

3. Dataset generation

Draw 96 training ICs and 32 independent validation ICs, each coordinate uniform
on [-1.5,1.5]. Fix these IC sets for the generation attempt sequence, independently
of the proposed equations. Integrate every accepted system over [0,2] using
float64 DOP853 with rtol=1e-11 and atol=1e-13. Start with noiseless observations.

Before accepting a system, apply these declared checks:

- Reject failed or nonfinite integrations, or absolute state values exceeding
  20 at the reference solver's accepted time points.
- Reintegrate all reference trajectories at rtol=1e-13, atol=1e-15 and require
  the maximum absolute difference in observed endpoints to be below 1e-8.
- Using training trajectories, remove A4*E4 and then, separately, A5*E5, while
  retaining damping and the other unknown law. Require the RMS observed endpoint
  change to be at least 0.02 for each ablation. Reject if an ablated integration
  fails or is nonfinite.

Record rejection reasons privately. Use the tighter integrations as the saved
targets. Accept the first system passing these checks. Do not run a discovery
solver to select an instance. Do not condition on CPU/GPU recovery time. Stop
with a clear failure after a configurable maximum of 500 joint-system proposals;
do not silently relax any rule.

These checks establish numerical consistency and measurable influence, not
unique structural identifiability. Redundant coefficient parameterizations
can arise naturally; report any coefficient-sensitivity diagnostic privately
and do not introduce an unannounced rank-based rejection rule.

4. Public and private output

Obtain the default generation seed from operating-system randomness. Allow an
optional explicit seed for reproducing a private generation. Keep the seed and
its derived RNG state out of the public output. Use independently derived RNG
streams for ICs and expression proposals, and retain the private seed and
generator/library versions needed for reproduction.

Write one public file named random-trees-depth3-<UTC-timestamp>.json, with no zip
or extra output directory. Include:

- format_version, state names, unknown_rhs_indices=[4,5];
- the four known equations, known damping, and public unknown-law form A_j*E_j;
- the complete grammar, coefficient ranges, depth convention, sampling and
  acceptance rules;
- observed_state_indices=[0,1], observation_times=[0,2], solver settings;
- 96 training records, each with id, initial_state in [x0,...,x5] order, and
  observations shaped [2 times][2 observed states];
- 32 validation records with id and complete initial_state only.

The public file must contain enough metadata to interpret every array. It must
not contain the seed, hidden ASTs, hidden coefficients, validation targets,
realized tree depths/counts beyond the public constraints, realized dependencies,
or any hint derived from the particular hidden equations.

Print the private generation seed, all six equations, exact hidden ASTs and
full-precision coefficients, validation targets, and generation diagnostics to
stdout for the challenge owner to retain separately. Do not put them in the
public JSON. Return the generator source and a link to the public file. The owner
will give the solving LLM only the public file, not your private stdout.

5. Grading

Compare solvers using the same public file and priors. Freeze a candidate before
consulting the private answer. Report predictive error on private validation
trajectories and structural recovery separately. Accept algebraically equivalent
functions with fixed state identities; raw AST equality is not required. Handle
equivalent coefficient parameterizations explicitly. Numerical RHS probes can
reject false matches but do not prove symbolic equivalence. Label inconclusive
symbolic grading as inconclusive. Do not supply private-answer-guided hints to
an ongoing search or promise that this generator guarantees a GPU advantage.
