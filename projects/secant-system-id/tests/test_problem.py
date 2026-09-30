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
from __future__ import annotations

from dataclasses import replace
import math
import unittest

from secant_system_id.fed_batch import make_recovery_problem, reference_data
from secant_system_id.odebench import ODEBENCH_REFERENCE_SYSTEMS, make_reference_problem


class RecoveryProblemTest(unittest.TestCase):
    def test_fed_batch_legacy_reference_matches_generic_problem(self) -> None:
        problem = make_recovery_problem(steps_per_observation=4)
        self.assertEqual(problem.trajectory_kernel_blockers(), ())
        self.assertEqual(problem.make_trajectory_shape().ast_leaf_counts, (8, 8))
        self.assertEqual(problem.dense_reference_array(), reference_data(4))

    def test_missing_values_are_preserved_and_rejected_by_dense_kernel(self) -> None:
        problem = make_recovery_problem(observation_count=2, steps_per_observation=2)
        original = problem.training[0]
        values = list(original.values)
        row = list(values[1])
        row[2] = None
        values[1] = tuple(row)
        sparse = replace(original, values=tuple(values))
        sparse_problem = replace(problem, training=(sparse,))
        self.assertEqual(sparse_problem.trajectory_kernel_blockers(), ("missing state observations",))
        with self.assertRaisesRegex(ValueError, "missing state observations"):
            sparse_problem.dense_reference_array()

    def test_four_reference_systems_are_finite_and_converged(self) -> None:
        self.assertEqual(
            {definition.name for definition in ODEBENCH_REFERENCE_SYSTEMS},
            {"lotka_volterra", "lorenz_63", "van_der_pol", "rossler"},
        )
        for definition in ODEBENCH_REFERENCE_SYSTEMS:
            with self.subTest(system=definition.name):
                coarse = make_reference_problem(definition, maximum_step=0.005)
                fine = make_reference_problem(definition, maximum_step=0.0025)
                self.assertEqual(len(fine.training[0].times), 150)
                self.assertEqual(fine.model.state_count, len(definition.equations))
                self.assertTrue(
                    all(
                        math.isfinite(value)
                        for series in fine.training + fine.test
                        for row in series.values
                        for value in row
                    )
                )
                errors = [
                    lhs - rhs
                    for coarse_series, fine_series in zip(
                        coarse.training + coarse.test, fine.training + fine.test
                    )
                    for coarse_row, fine_row in zip(coarse_series.values, fine_series.values)
                    for lhs, rhs in zip(coarse_row, fine_row)
                ]
                mean_squared_error = sum(error * error for error in errors) / len(errors)
                self.assertLess(mean_squared_error, 1.0e-5)


if __name__ == "__main__":
    unittest.main()
