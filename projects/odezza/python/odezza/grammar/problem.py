# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Prepared observed-entry objective, with explicit FP32 and RK4 semantics."""
from __future__ import annotations
from copy import deepcopy
import math

from odegrammar.compiler import content_id
from .lowering import CompatibilityError, f32


def prepare_problem(spec):
    if set(spec) - {"states", "trajectories", "known_rhs", "integration", "description"}:
        raise ValueError("Unknown prepared-problem fields")
    states = spec["states"]
    if not isinstance(states, list) or not states or len(states) > 126 or len(set(states)) != len(states):
        raise CompatibilityError("Provide 1..126 unique ordered states (actual native resources may impose a lower limit)")
    if any(not isinstance(s, str) or not s.isidentifier() or s == "t" for s in states):
        raise ValueError("Invalid state name")
    if not isinstance(spec.get("trajectories"), list) or not spec["trajectories"]:
        raise ValueError("At least one trajectory is required")
    offsets, times, rows, scored = [0], [], [], 0
    observed_by_state = [0] * len(states)
    for trajectory in spec["trajectories"]:
        if set(trajectory) - {"times", "values", "initial", "mask"}:
            raise ValueError("Unknown trajectory fields")
        ts = [f32(t) for t in trajectory["times"]]
        values = deepcopy(trajectory["values"])
        initial = [f32(x) for x in trajectory["initial"]]
        if len(initial) != len(states) or not ts or len(values) != len(ts):
            raise ValueError("Complete initial vector and one observation row per time are required")
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError("Times must increase strictly after FP32 conversion")
        mask = trajectory.get("mask")
        if mask is not None and (len(mask) != len(ts) or any(len(r) != len(states) or any(type(v) is not bool for v in r) for r in mask)):
            raise ValueError("Mask must be a boolean [time,state] array")
        for j, row in enumerate(values):
            if len(row) != len(states):
                raise ValueError("Observation rows must follow the declared state order")
            result = []
            for s, value in enumerate(row):
                if value is not None:
                    value = f32(value)
                visible = value is not None and (mask is None or mask[j][s])
                if j == 0:
                    if visible and value != initial[s]:
                        raise ValueError("Initial observation disagrees with explicit initial state")
                    result.append(initial[s])
                else:
                    result.append(value if visible else None)
                    scored += int(visible)
                    observed_by_state[s] += int(visible)
            rows.append(result)
        times.extend(ts)
        offsets.append(len(times))
    if not scored:
        raise ValueError("At least one noninitial observed scalar is required")
    known = spec.get("known_rhs", {})
    if not isinstance(known, dict) or set(known) - set(states) or any(not isinstance(v, str) for v in known.values()):
        raise ValueError("known_rhs must map declared states to expression strings")
    result = dict(states=states, offsets=offsets, times=times, rows=rows,
                  observed_scalars=scored, observed_by_state=observed_by_state,
                  known_rhs=known, integration=spec.get("integration", {}),
                  objective="pooled_observed_mse_excluding_initials_v1", dtype="float32")
    result["id"] = content_id(result)
    return result


def attach_knowns(request, problem):
    request = deepcopy(request)
    if request["states"] != problem["states"]:
        raise CompatibilityError("Grammar state order differs from prepared problem")
    if request.get("problem_id") not in (None, problem["id"]):
        raise CompatibilityError("Grammar problem_id differs from prepared revision")
    request["problem_id"] = problem["id"]
    # The canonical compiler still receives the complete RHS vector.
    for family in request.get("families", [request]):
        rhs = {**request.get("rhs", {}), **family.get("rhs", {})}
        for state, expr in problem["known_rhs"].items():
            if state in rhs and rhs[state] != expr:
                raise CompatibilityError("Grammar attempts to replace prepared known RHS " + state)
            rhs[state] = expr
        family["rhs"] = rhs
    return request


def resolve_integration(settings, problem):
    if settings.get("method", "rk4") != "rk4" or settings.get("stiff", False) or "rtol" in settings or "atol" in settings:
        raise CompatibilityError("Only fixed-subdivision RK4 is executable; no solver fallback")
    dt = settings.get("dt", problem["integration"].get("dt"))
    if dt is None or f32(dt) <= 0:
        raise CompatibilityError("Specify a positive integration.dt in the grammar or prepared problem")
    intervals = [problem["times"][i] - problem["times"][i-1]
                 for start, end in zip(problem["offsets"], problem["offsets"][1:])
                 for i in range(start + 1, end)]
    steps = max(1, math.ceil(max(intervals) / f32(dt)))
    if steps > 2**32 - 1 or any(f32(f32(h)/steps) <= 0 for h in intervals):
        raise CompatibilityError("RK4 subdivision count/step exceeds native range")
    total_steps = steps * len(intervals)
    if "max_steps" in settings and total_steps > settings["max_steps"]:
        raise CompatibilityError("Resolved RK4 work exceeds integration.max_steps per configuration")
    return dict(method="rk4", dt_max=f32(dt), steps_per_observation=steps,
                steps_per_configuration=total_steps, observation_policy="land_on_each_observation",
                dt_semantics="common_subdivisions_ensuring_maximum_step", dtype="float32")


def cpu_score(problem, programs, *, steps, breakdown=False):
    """FP64 reference evaluation of the *resolved* FP32 native AST coefficients."""
    from .reference import evaluate
    total, count = 0., 0
    state_sums, state_counts, trajectories = [0.]*len(programs), [0]*len(programs), []
    try:
        for start, end in zip(problem["offsets"], problem["offsets"][1:]):
            trajectory_sum, trajectory_count = 0., 0
            state = list(problem["rows"][start])
            for i in range(start + 1, end):
                h = (problem["times"][i] - problem["times"][i-1]) / steps
                for _ in range(steps):
                    def rhs(y): return [evaluate(p, y) for p in programs]
                    k1 = rhs(state)
                    k2 = rhs([y + h*.5*k for y, k in zip(state, k1)])
                    k3 = rhs([y + h*.5*k for y, k in zip(state, k2)])
                    k4 = rhs([y + h*k for y, k in zip(state, k3)])
                    state = [y + h/6*(a+2*b+2*c+d) for y, a, b, c, d in zip(state, k1, k2, k3, k4)]
                    if not all(math.isfinite(v) for v in state):
                        return None
                for s, (y, expected) in enumerate(zip(state, problem["rows"][i])):
                    if expected is not None:
                        residual = (y-expected)**2
                        total += residual; trajectory_sum += residual; state_sums[s] += residual
                        trajectory_count += 1; state_counts[s] += 1
                        count += 1
            trajectories.append(dict(observed_scalars=trajectory_count, mse=trajectory_sum/trajectory_count if trajectory_count else None))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    mse = total / count if math.isfinite(total) else None
    if breakdown and mse is not None:
        return dict(mse=mse, by_state={name: dict(observed_scalars=n, mse=s/n if n else None)
                                      for name, s, n in zip(problem["states"], state_sums, state_counts)},
                    by_trajectory=trajectories)
    return mse
