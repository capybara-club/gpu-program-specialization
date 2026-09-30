#!/usr/bin/env python3
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

import csv
import io
import math
import subprocess
import sys


def rhs(state):
    biomass, glucose, sucrose, _ = state

    glucose_growth = 0.43 * glucose * biomass / ((glucose + 63.7 * biomass) * (1.0 + 5.8 * sucrose))
    sucrose_growth = 0.132 * sucrose * biomass / ((sucrose + 3.68 * biomass) * (1.0 + 0.0 * glucose))

    return (
        glucose_growth + sucrose_growth - 0.0055 * biomass,
        -2.58 * glucose_growth,
        -1.71 * sucrose_growth,
        0.0 * glucose_growth + 0.0 * sucrose_growth + 0.21 * biomass - 0.0466 * biomass * biomass,
    )


def add(state, h, terms):
    return tuple(value + h * sum(coefficient * derivative[index] for coefficient, derivative in terms)
                 for index, value in enumerate(state))


def dopri_step(state, h):
    k1 = rhs(state)
    k2 = rhs(add(state, h, [(1.0 / 5.0, k1)]))
    k3 = rhs(add(state, h, [(3.0 / 40.0, k1), (9.0 / 40.0, k2)]))
    k4 = rhs(add(state, h, [(44.0 / 45.0, k1), (-56.0 / 15.0, k2), (32.0 / 9.0, k3)]))
    k5 = rhs(add(state, h, [(19372.0 / 6561.0, k1), (-25360.0 / 2187.0, k2),
                            (64448.0 / 6561.0, k3), (-212.0 / 729.0, k4)]))
    k6 = rhs(add(state, h, [(9017.0 / 3168.0, k1), (-355.0 / 33.0, k2),
                            (46732.0 / 5247.0, k3), (49.0 / 176.0, k4),
                            (-5103.0 / 18656.0, k5)]))
    fifth = add(state, h, [(35.0 / 384.0, k1), (500.0 / 1113.0, k3),
                           (125.0 / 192.0, k4), (-2187.0 / 6784.0, k5),
                           (11.0 / 84.0, k6)])
    k7 = rhs(fifth)
    fourth = add(state, h, [(5179.0 / 57600.0, k1), (7571.0 / 16695.0, k3),
                            (393.0 / 640.0, k4), (-92097.0 / 339200.0, k5),
                            (187.0 / 2100.0, k6), (1.0 / 40.0, k7)])
    return fifth, tuple(fifth[index] - fourth[index] for index in range(4))


def reference_trajectory(initial_state, observation_times):
    state = initial_state
    current_time = 0.0
    h = 0.05
    output = []

    for observation_time in observation_times:
        while current_time < observation_time:
            h = min(h, observation_time - current_time)
            candidate, error = dopri_step(state, h)
            normalized_error = max(
                abs(error[index]) / (1.0e-12 + 1.0e-11 * max(abs(state[index]), abs(candidate[index])))
                for index in range(4)
            )

            if normalized_error <= 1.0:
                state = candidate
                current_time += h

            factor = 5.0 if normalized_error == 0.0 else 0.9 * normalized_error ** -0.2
            h *= min(5.0, max(0.2, factor))

        output.append(state)

    return output


def run_c(program, max_step):
    completed = subprocess.run([program, str(max_step)], check=True, text=True, capture_output=True)
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    return {
        (int(row["experiment"]), float(row["time_h"])): (
            float(row["biomass_g_L"]),
            float(row["glucose_g_L"]),
            float(row["sucrose_g_L"]),
            float(row["astaxanthin_mg_L"]),
        )
        for row in rows
    }


def maximum_scaled_error(actual, expected):
    return max(abs(a - e) / max(1.0, abs(e)) for a, e in zip(actual, expected))


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} ASTAXANTHIN_REPRODUCTION_PROGRAM", file=sys.stderr)
        return 2

    program = sys.argv[1]
    observation_times = [float(time) for time in range(0, 97, 8)]
    initial_states = [
        (0.1, 10.0, 5.0, 0.0),
        (0.1, 20.0, 5.0, 0.0),
        (0.2, 5.0, 2.5, 0.0),
    ]
    reference = {
        (experiment, time): state
        for experiment, initial_state in enumerate(initial_states, start=1)
        for time, state in zip(observation_times, reference_trajectory(initial_state, observation_times))
    }

    coarse = run_c(program, 0.5)
    fine = run_c(program, 0.25)
    production = run_c(program, 0.01)
    coarse_error = max(maximum_scaled_error(coarse[key], reference[key]) for key in reference)
    fine_error = max(maximum_scaled_error(fine[key], reference[key]) for key in reference)
    production_error = max(maximum_scaled_error(production[key], reference[key]) for key in reference)

    if not math.isfinite(production_error) or production_error > 2.0e-9:
        print(f"FAIL: maximum scaled error {production_error:.3e} exceeds 2.0e-9")
        return 1

    if fine_error > coarse_error * 0.2:
        print(f"FAIL: RK4 refinement did not show fourth-order convergence ({coarse_error:.3e} -> {fine_error:.3e})")
        return 1

    print(f"PASS: C99 RK4 matches independent Dormand-Prince reference")
    print(f"maximum scaled error, 0.50 h step: {coarse_error:.3e}")
    print(f"maximum scaled error, 0.25 h step: {fine_error:.3e}")
    print(f"maximum scaled error, 0.01 h production step: {production_error:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
