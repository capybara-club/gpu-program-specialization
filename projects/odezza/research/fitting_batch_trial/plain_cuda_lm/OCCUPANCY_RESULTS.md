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

# Occupancy-controlled LM fitting results — 2026-09-09

The 1,024-fit table was dominated by insufficient work and must not set throughput
defaults. A one-lane launch had 32 CTAs: at most 0.79% of this GPU's warp slots.
Eight lanes supplied 256 CTAs and raised that upper bound to 6.35%. Those are
grid-based upper bounds, not measured achieved occupancy.

## What was measured

On rack1 device 0 (RTX 5080, 84 SMs), reused all 48 compiled kernels from the
3/6/12/16-state × 3/6/8-coefficient × 1/2/4/8-lane grid. No AST generation or
compilation occurs in the scaling run. Larger banks tile the same 1,024 starts,
holding the distribution of optimizer work constant. All outputs and counters
must match the corresponding repeated reference byte for byte.

The main run completed 281 population/shape cells, three samples each, totaling
66,600,960 repeated LM fit executions. It took 3,521.95 seconds wall time;
3,463.11 seconds were measured launch-event durations. The remainder includes
warm-up launches, preparation, transfers and checks, which are not separately
profiled. These are complete multi-iteration fits, not individual scoring trials
or newly distinct AST/configuration samples.

Every shape reached at least 4.88 predicted resident waves. Nineteen met the
strict raw-throughput plateau criterion. Forty-seven had locally stable linear
cost models; the remaining 3-state/3-coefficient one-lane curve was resolved in
a separate follow-up at 262,144/524,288/786,432 starts. All four follow-up widths
passed both plateau and linear-cost checks. The known 3-state/6-coefficient
width-2 optimizer-path difference remains excluded from cross-width selection.

## Corrected lane choices

These are the lowest measured latency at the largest population common to all
comparable widths in each main-run case (131,072 or 262,144 fits). They are useful
large-bank measurements, not universal N/P-only defaults. Near ties are marked.

| States | 3 coefficients | 6 coefficients | 8 coefficients |
|---:|---:|---:|---:|
| 3 | 1–2 near tie | 4, with 1 close; 2 excluded | 1–2 near tie |
| 6 | 1 | 4, with 2/8 close | 4, with 2 close |
| 12 | 1 | 2–4 near tie; 2 spills | 4 |
| 16 | 2 | 4–8 near tie | 4; spills |

At 786,432 starts the small 3-state/3-coefficient follow-up delivers approximately
1.389M / 1.334M / 0.938M / 0.461M fits/s at widths 1/2/4/8. All four have plateau
evidence. One lane is around three times the throughput of eight lanes, despite
41.7% versus 50% theoretical occupancy. The small one/two difference should not
be treated as a hardware-independent rule.

For 16 states and 8 coefficients at a matched 131,072 starts, four lanes deliver
9.39k fits/s while eight deliver 7.09k. The four-lane kernel spills; the eight-lane
kernel does not. This is a direct reason to retain spilled shapes in selection.

## How the API reports occupancy

`occupancy_geometry` in benchmark and trial specialized-LM profiles now reports:

- CTA count, fits/CTA, resident capacity and resident waves;
- upper bounds on SM coverage, resident-slot coverage and global warp occupancy;
- fits required for one CTA/SM, one resident wave and four resident waves;
- fits/s/SM, amortized SM µs/fit, CTA/s/SM and amortized SM µs/CTA;
- achieved occupancy as unknown unless it is actually profiled.

The driver reports register/local/shared usage, actual block size, maximum
resident blocks/warps, and theoretical occupancy using CUDA's occupancy API.
With 32-thread CTAs and this device's 24-block/SM limit, the block-size ceiling
alone is 50% warp occupancy. Tested resource footprints lower it to 16.7–50%.
Increasing population fills the shape's available capacity; it cannot raise
that resource ceiling. Higher occupancy did not consistently mean higher speed.

For an intrinsic-cost estimate, the report fits `T(N) = fixed_cost + N × cost`
to the largest three populations and checks repeat variability, linearity and
sufficient grid residency. Multiplying the slope by SM count gives incremental
SM-time/fit. This is a qualified local estimate. Dividing by theoretical occupancy
would not correctly remove latency hiding, memory stalls or CTA-tail effects.
CTA completion rate is also not the latency of an individual overlapping CTA.

## Validation and limits

All 281 main-run and 12 follow-up cells match their tiled references. Sixteen
runner/service/occupancy unit tests pass. The specialized adapter's one/four-stream
test passes with identical candidate results and work counts, per-pack cleanup,
borrowed-pool reuse, and the new occupancy fields. The 26 native snapshot files
still match the fork baseline; no scoring or fitting kernel math changed.

Nsight Compute is installed but hardware counters are denied (`ERR_NVGPUCTRPERM`).
Actual warp residency and issue utilization remain unmeasured; no administrator
setting was changed. GPU clocks were not locked; per-cell telemetry is retained.
This is a prepared ordinary-CUDA fitting calibration with known correct structures,
not a blind recovery or production cooperative-specialization rollout.

Detailed evidence: [main curves and per-SM tables](../validation/occupancy-sweep-01/OCCUPANCY.md),
[small-state follow-up](../validation/occupancy-small-followup-01/OCCUPANCY.md),
[adapter checks](../validation/occupancy-adapter-check/checks.json),
[API](../pipeline_runner/README.md).
