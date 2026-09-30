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

import unittest

from secant_system_id.ast import Program, input_slot
from secant_system_id.fed_batch import (
    FED_BATCH_SHAPE,
    PLANTED_BINDINGS,
    PLANTED_CONSTANTS,
    planted_programs,
)
from secant_system_id.recovery import (
    held_out_trajectory_mse,
    rate_surface_metrics,
    resolved_expression,
    site_structure_matches,
    strict_structure_match,
)


class RecoveryTest(unittest.TestCase):
    def test_planted_laws_pass_all_recovery_metrics(self) -> None:
        programs = planted_programs()
        self.assertTrue(strict_structure_match(programs, PLANTED_BINDINGS, FED_BATCH_SHAPE))
        rates = rate_surface_metrics(
            programs,
            PLANTED_CONSTANTS,
            PLANTED_BINDINGS,
            FED_BATCH_SHAPE,
        )
        self.assertLess(rates.joint_nrmse, 1.0e-6)
        self.assertEqual(rates.valid_points, 100)
        self.assertLess(
            held_out_trajectory_mse(
                programs,
                PLANTED_CONSTANTS,
                PLANTED_BINDINGS,
                FED_BATCH_SHAPE,
            ),
            1.0e-10,
        )

    def test_leaf_only_candidate_is_not_a_structural_match(self) -> None:
        programs = (
            Program.from_expression(input_slot(0)),
            Program.from_expression(input_slot(8)),
        )
        self.assertFalse(strict_structure_match(programs, PLANTED_BINDINGS, FED_BATCH_SHAPE))
        self.assertGreater(
            rate_surface_metrics(
                programs,
                PLANTED_CONSTANTS,
                PLANTED_BINDINGS,
                FED_BATCH_SHAPE,
            ).joint_nrmse,
            0.1,
        )

    def test_zero_inhibition_law_accepts_the_simplified_structure(self) -> None:
        first = planted_programs()[0]
        leaves = [input_slot(8 + index) for index in range(5)]
        second = Program.from_expression(leaves[0] * leaves[1] / (leaves[2] + leaves[3] * leaves[4]))
        matches = site_structure_matches((first, second), PLANTED_BINDINGS, FED_BATCH_SHAPE)
        self.assertEqual(matches, (True, True))
        self.assertTrue(strict_structure_match((first, second), PLANTED_BINDINGS, FED_BATCH_SHAPE))

    def test_resolved_expression_names_states_and_parameters(self) -> None:
        expression = resolved_expression(
            planted_programs()[0],
            PLANTED_CONSTANTS,
            PLANTED_BINDINGS,
            FED_BATCH_SHAPE,
        )
        self.assertIn("X", expression)
        self.assertIn("G", expression)
        self.assertIn("S", expression)
        self.assertIn("c0[", expression)


if __name__ == "__main__":
    unittest.main()
