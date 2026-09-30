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

# Sizing policy validation

2026-09-12, item 6 development, before deployment.

The initial policy used at most 32 systems per module for every rollout. The
matched one-GPU grid exposed regressions of 11.9% (6 states / 8 constants) and
20.9% (12 states / 24 constants / longer AST) on its longer trajectories. Exact
winner IDs, programs, coefficients, MSEs and evaluated counts agreed throughout;
these are valid sizing comparisons, not equivalent performance claims.

The initial policy was rejected before deployment. The revised policy favors
64-system modules when a configuration integrates at least 32 RK4 steps and a
skeleton has at least 128 configurations in the tile. Short screens favor 32.
Explicit module sizes are preserved. This is a measured 5080 starting policy,
not a hardware-independent optimum or an occupancy guarantee.

The grid's historical case labels 4/32 select short/long trajectory fixtures;
they are not the actual integration step counts. Reports give the exact FP32
layout, and summary tables must use `integration.steps_per_configuration`.
Profiling runs add CUDA events and are excluded from uninstrumented medians.

The first correction reduced the one-GPU long-rollout gap to 2–4%, while
128-variant pages still required more tile/reduction calls than the original
256-variant pages. Long-rollout page ceilings were raised from 16 to 32 MiB
within the same one-eighth host-budget share, restoring 256-variant pages for
these fixtures. Short-rollout pages are unchanged.

The full million-AST × 2,048-bank comparison exposed another overly conservative
page limit: 256-AST pages caused 3,907 tiles instead of 977. Automatic execution
was 1.396 s versus 1.229 s with explicit 1,024-AST pages (13.5% slower); counts
and winners agreed. Families reserving at least 65,536 variants now receive a
64 MiB page ceiling, still constrained by the same one-eighth shared host budget.
This restores 1,024-AST pages in that fixture without changing tile dependencies.

The isolated network fixture `family_overrides_late_tags` exposed a one-system
shape with two states and no active constants that NVRTC produced in a form the
existing inspector rejected (`template inspection measurement failed`, result 4).
No configurations were scored. Automatic packing now starts at two systems to
retain dispatch; actual launch counts remain one when only one candidate exists.
This is a documented shape-selection restriction, not a silent retry or changed
search population. Explicit shapes remain unchanged. General one-system inspector
coverage remains a core-validation follow-up; the complete network fixture is
rerun before deployment.

Resolution evidence and final single/dual-GPU measurements belong in SUMMARY.md.
Raw logs/reports belong outside version control.

During deployment, worker 437805 started before the source-refresh rebuild had
finished. All ten live requests passed, but `/proc/437805/maps` showed the runtime
library mapping marked deleted after relinking. Those initial live results are
provisional and excluded from the final deployment evidence. The worker was
stopped with an empty queue, restarted after build completion, and the entire
live contract suite rerun. Final process/library identities and the rerun result
are recorded in deployment.json. This was a deployment-order error; no customer
job was active, and the previous deployment remained available for rollback.
