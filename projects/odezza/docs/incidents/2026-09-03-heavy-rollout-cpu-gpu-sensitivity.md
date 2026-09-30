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

# Heavy-rollout CPU/GPU sensitivity

## Status

Open numerical-policy issue. The scoring and specialization paths remain operational.

## Observation

On 2026-09-03, a strict CPU replay of the four-RHS exact-depth-8 structural-random corpus failed at system 585, configuration 159:

- CPU MSE: `6.20561367e11`
- GPU MSE: `6.22952776e11`
- Absolute difference: `2.39140864e9`
- Relative difference: approximately 0.38%

The ASTs used only add, subtract, multiply, and negate. No candidate was routed through a division/transcendental skip policy. The system remained finite but had an explosive free rollout that amplified ordinary FP32 evaluation differences.

## Affected evidence

- Depth-2, depth-4, and depth-6 results remain strictly checked with 64 complete CPU/GPU trajectory replays each.
- The full 4,096-system depth-8 throughput result is performance-valid but only has one passing strict sample in that timed run.
- A separate 64-system depth-8 run passed 64 complete replays, including four mutually invalid CPU/GPU scores, with a maximum finite relative difference of `2.00455338e-6`.

No specialization failure, CUDA failure, capacity fallback, or register-spill fallback was observed.

## Current mitigation

Do not silently relax the global replay tolerance and do not describe the full depth-8 corpus as universally CPU-equivalent. Report the focused validation separately from the full throughput run.

## Next action

Define a numerical agreement policy that records stable finite agreement, mutually invalid rollouts, and sensitivity-amplified finite disagreement separately. Add a deliberately unstable regression fixture before changing acceptance thresholds.
