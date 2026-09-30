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

import argparse
import subprocess

import numpy as np


def driver_run(driver: str, case: tuple[int, ...]) -> dict[str, np.ndarray | list[float]]:
    process = subprocess.run([driver, *(str(value) for value in case)], check=True, capture_output=True, text=True)
    parsed: dict[str, np.ndarray | list[float]] = {}
    for line in process.stdout.splitlines():
        fields = line.split()
        name = fields[0]
        if name == "meta":
            parsed[name] = [float(value) for value in fields[1:]]
            continue
        count = int(fields[1])
        values = fields[2:]
        if len(values) != count:
            raise AssertionError(f"{name}: expected {count} values, received {len(values)}")
        dtype = np.uint32 if name == "active_masks" else np.int32 if name in {
            "solve_info", "active_counts", "iteration_counts"
        } else np.float64
        parsed[name] = np.asarray(values, dtype=dtype)
    return parsed


def system_solve(normalized: np.ndarray, rhs_base: np.ndarray, alpha: float, active_mask: int) -> np.ndarray:
    capacity = normalized.shape[0]
    active = np.asarray([(active_mask >> idx) & 1 for idx in range(capacity)], dtype=bool)
    matrix = np.eye(capacity, dtype=np.float64)
    matrix[np.ix_(active, active)] = normalized[np.ix_(active, active)]
    matrix[np.diag_indices(capacity)] = np.where(active, 1.0 + alpha, 1.0)
    rhs = np.where(active, rhs_base, 0.0)
    lower = np.linalg.cholesky(matrix)
    return np.linalg.solve(lower.T, np.linalg.solve(lower, rhs))


def oracle_compute(parsed: dict[str, np.ndarray | list[float]]) -> dict[str, np.ndarray]:
    capacity, num_asts, num_rows, num_targets, num_sweeps, max_iterations, cohorts, scale_epsilon = parsed["meta"]
    capacity = int(capacity)
    num_asts = int(num_asts)
    num_rows = int(num_rows)
    num_targets = int(num_targets)
    num_sweeps = int(num_sweeps)
    max_iterations = int(max_iterations)
    cohorts = int(cohorts)
    stats_per_cohort = capacity + capacity * capacity + capacity * num_targets
    statistics = np.asarray(parsed["statistics"]).reshape(cohorts, stats_per_cohort)
    target_stats = np.asarray(parsed["target_stats"]).reshape(num_targets, 2)
    alphas = np.asarray(parsed["alphas"])
    thresholds = np.asarray(parsed["thresholds"])
    output_count = num_sweeps * cohorts * num_targets
    coefficients = np.zeros((output_count, capacity), dtype=np.float64)
    intercepts = np.zeros(output_count, dtype=np.float64)
    sse = np.zeros(output_count, dtype=np.float64)
    solve_info = np.zeros(output_count, dtype=np.int32)
    active_masks = np.zeros(output_count, dtype=np.uint32)
    active_counts = np.zeros(output_count, dtype=np.int32)
    iteration_counts = np.zeros(output_count, dtype=np.int32)

    for cohort in range(cohorts):
        active_features = min(capacity, num_asts - cohort * capacity)
        stats = statistics[cohort]
        sums = stats[:capacity]
        gram = stats[capacity:capacity + capacity * capacity].reshape(capacity, capacity)
        cross = stats[capacity + capacity * capacity:].reshape(capacity, num_targets)
        means = np.zeros(capacity, dtype=np.float64)
        scales = np.ones(capacity, dtype=np.float64)
        means[:active_features] = sums[:active_features] / num_rows
        variances = (np.diag(gram)[:active_features] - sums[:active_features] ** 2 / num_rows) / num_rows
        scales[:active_features] = np.sqrt(np.maximum(variances, scale_epsilon * scale_epsilon))
        normalized = np.eye(capacity, dtype=np.float64)
        centered_gram = gram - np.outer(sums, sums) / num_rows
        normalized[:active_features, :active_features] = centered_gram[:active_features, :active_features] / (
            num_rows * np.outer(scales[:active_features], scales[:active_features])
        )
        np.fill_diagonal(normalized, 1.0)

        for sweep in range(num_sweeps):
            for target in range(num_targets):
                output = (sweep * cohorts + cohort) * num_targets + target
                centered_cross = cross[:, target] - sums * target_stats[target, 0] / num_rows
                rhs_base = centered_cross / (num_rows * scales)
                rhs_base[active_features:] = 0.0
                active_mask = (1 << active_features) - 1 if active_features < 32 else 0xFFFFFFFF
                solution = np.zeros(capacity, dtype=np.float64)
                iterations = 0

                while active_mask != 0 and iterations < max_iterations:
                    previous_mask = active_mask
                    solution = system_solve(normalized, rhs_base, alphas[sweep], active_mask)
                    next_mask = 0
                    for feature in range(active_features):
                        if (active_mask >> feature) & 1 and abs(solution[feature]) >= thresholds[sweep]:
                            next_mask |= 1 << feature
                    iterations += 1
                    active_mask = next_mask
                    if next_mask == previous_mask or next_mask == 0:
                        break

                if active_mask != 0:
                    solution = system_solve(normalized, rhs_base, alphas[sweep], active_mask)
                else:
                    solution.fill(0.0)
                raw = np.zeros(capacity, dtype=np.float64)
                raw[:active_features] = solution[:active_features] / scales[:active_features]
                intercept = target_stats[target, 0] / num_rows - np.dot(raw, means)
                linear_sum = np.dot(raw, sums)
                cross_sum = np.dot(raw, cross[:, target])
                quadratic_sum = raw[:active_features] @ gram[:active_features, :active_features] @ raw[:active_features]
                score = target_stats[target, 1] - 2.0 * cross_sum + quadratic_sum - 2.0 * intercept * (
                    target_stats[target, 0]
                ) + 2.0 * intercept * linear_sum + num_rows * intercept * intercept
                coefficients[output] = raw
                intercepts[output] = intercept
                sse[output] = max(score, 0.0)
                active_masks[output] = active_mask
                active_counts[output] = active_mask.bit_count()
                iteration_counts[output] = iterations

    return {
        "coefficients": coefficients.reshape(-1),
        "intercepts": intercepts,
        "sse": sse,
        "solve_info": solve_info,
        "active_masks": active_masks,
        "active_counts": active_counts,
        "iteration_counts": iteration_counts,
    }


def compare_case(driver: str, case: tuple[int, ...]) -> None:
    parsed = driver_run(driver, case)
    expected = oracle_compute(parsed)
    for name in ("solve_info", "active_masks", "active_counts", "iteration_counts"):
        np.testing.assert_array_equal(np.asarray(parsed[name]), expected[name], err_msg=f"{name}, case={case}")
    np.testing.assert_allclose(parsed["coefficients"], expected["coefficients"], rtol=3.0e-4, atol=3.0e-5,
                               err_msg=f"coefficients, case={case}")
    np.testing.assert_allclose(parsed["intercepts"], expected["intercepts"], rtol=3.0e-4, atol=3.0e-5,
                               err_msg=f"intercepts, case={case}")
    np.testing.assert_allclose(parsed["sse"], expected["sse"], rtol=8.0e-4, atol=2.0e-2,
                               err_msg=f"sse, case={case}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    args = parser.parse_args()
    cases = [
        (1, 3, 7, 129, 1, 3, 3),
        (7, 4, 9, 257, 2, 4, 4),
        (19, 8, 21, 513, 3, 4, 8),
        (41, 16, 31, 193, 4, 3, 16),
        (97, 32, 37, 321, 2, 4, 32),
    ]
    for case in cases:
        compare_case(args.driver, case)
    print(f"secant-sindy NumPy oracle matched {len(cases)} randomized STLSQ cases")


if __name__ == "__main__":
    main()
