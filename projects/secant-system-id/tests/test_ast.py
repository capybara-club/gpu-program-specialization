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

from secant_system_id.ast import InstructionType, Program, input_slot
from secant_system_id.bindings import LeafKind, constant_leaf, decode_leaf, state_leaf
from secant_system_id.fed_batch import PLANTED_CONSTANTS, planted_programs


class AstTest(unittest.TestCase):
    def test_leaf_sources_share_one_encoded_bank(self) -> None:
        self.assertEqual(state_leaf(3).encode(), 3)
        self.assertEqual(constant_leaf(0).encode(), 4)
        self.assertEqual(decode_leaf(11).kind, LeafKind.CONSTANT)
        self.assertEqual(decode_leaf(11).index, 7)

    def test_secant_abi_and_postorder_evaluation(self) -> None:
        expression = (input_slot(0) + 2.0 * input_slot(1)) / input_slot(2)
        program = Program.from_expression(expression)
        self.assertEqual(program.data[0], InstructionType.STATIC_COLUMN_INPUT_F32)
        self.assertEqual(program.data[-1], InstructionType.RETURN_F32)
        self.assertAlmostEqual(program.evaluate((3.0, 4.0, 2.0)), 5.5)

    def test_planted_rate_programs(self) -> None:
        first, second = planted_programs()
        state = (0.1, 10.0, 5.0, 0.0)
        bank = state + PLANTED_CONSTANTS
        first_bindings = (4, 1, 1, 5, 0, 10, 6, 2)
        second_bindings = (7, 2, 2, 8, 0, 10, 9, 1)
        rate1 = first.evaluate([bank[index] for index in first_bindings])
        rate2 = second.evaluate([0.0] * 8 + [bank[index] for index in second_bindings])
        self.assertAlmostEqual(rate1, 0.43 * 10.0 / ((10.0 + 63.7 * 0.1) * (1.0 + 5.8 * 5.0)))
        self.assertAlmostEqual(rate2, 0.132 * 5.0 / ((5.0 + 3.68 * 0.1) * (1.0 + 0.0 * 10.0)))


if __name__ == "__main__":
    unittest.main()
