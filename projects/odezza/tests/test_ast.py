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

from odezza import (
    KernelShape,
    OutputPrograms,
    Program,
    SystemGroup,
    constant_source,
    source,
    state_source,
    toggle2,
    toggle4,
)


class AstTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = KernelShape(system_capacity=2)

    def test_toggle_bit_can_be_reused(self) -> None:
        left = toggle2(0, state_source(0), state_source(1))
        right = toggle2(0, constant_source(0), constant_source(1))
        program = Program.from_expression(left + right, self.shape)
        self.assertEqual(program.required_toggle_bits, 1)
        self.assertEqual(program.evaluate((2.0, 7.0), (11.0, 13.0, 17.0, 19.0), 0, self.shape), 13.0)
        self.assertEqual(program.evaluate((2.0, 7.0), (11.0, 13.0, 17.0, 19.0), 1, self.shape), 20.0)

    def test_two_bit_toggle_selects_four_register_sources(self) -> None:
        expression = toggle4(0, 1, state_source(0), state_source(1), constant_source(0), constant_source(1))
        program = Program.from_expression(expression, self.shape)
        values = [
            program.evaluate((2.0, 7.0), (11.0, 13.0, 17.0, 19.0), index, self.shape)
            for index in range(4)
        ]
        self.assertEqual(values, [2.0, 7.0, 11.0, 13.0])

    def test_toggle_out_of_bounds_is_rejected(self) -> None:
        highest = Program.from_expression(toggle2(31, state_source(0), state_source(1)), self.shape)
        self.assertEqual(highest.evaluate((2.0, 7.0), (11.0, 13.0, 17.0, 19.0), 1 << 31, self.shape), 7.0)
        with self.assertRaisesRegex(ValueError, "exceed"):
            Program.from_expression(toggle2(32, state_source(0), state_source(1)), self.shape)
        with self.assertRaisesRegex(ValueError, "exceed"):
            Program.from_expression(
                toggle4(0, 32, state_source(0), state_source(1), constant_source(0), constant_source(1)),
                self.shape,
            )

    def test_toggle_choices_can_include_inline_literals(self) -> None:
        program = Program.from_expression(toggle2(0, state_source(0), -1.0), self.shape)
        self.assertEqual(program.evaluate((2.0, 7.0), (11.0, 13.0, 17.0, 19.0), 0, self.shape), 2.0)
        self.assertEqual(program.evaluate((2.0, 7.0), (11.0, 13.0, 17.0, 19.0), 1, self.shape), -1.0)

    def test_toggle_rejects_computed_choices(self) -> None:
        with self.assertRaisesRegex(TypeError, "direct"):
            toggle2(0, source(state_source(0)) + 1.0, state_source(1))

    def test_system_group_document_is_identity_bound(self) -> None:
        program = Program.from_expression(source(state_source(0)), self.shape)
        group = SystemGroup(
            OutputPrograms((0,), (program,)),
            (OutputPrograms((1,), (program,)),),
        )
        document = group.to_document(self.shape)
        self.assertEqual(SystemGroup.from_document(document, self.shape), group)
        document["required_toggle_bits"] = 1
        with self.assertRaisesRegex(ValueError, "toggle-bit declaration"):
            SystemGroup.from_document(document, self.shape)


if __name__ == "__main__":
    unittest.main()
