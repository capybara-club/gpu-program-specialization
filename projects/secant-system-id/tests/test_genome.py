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
from secant_system_id.genome import SystemGenome
from secant_system_id.shape import KernelShape


class GenomeTest(unittest.TestCase):
    def test_programs_are_site_local(self) -> None:
        shape = KernelShape(ast_leaf_counts=(2, 3), state_count=2)
        genome = SystemGenome(
            (
                Program.from_expression(input_slot(0) + input_slot(1)),
                Program.from_expression(input_slot(2) * input_slot(4)),
            )
        )
        genome.validate(shape)
        with self.assertRaisesRegex(ValueError, "another missing site"):
            genome.replace_site(0, Program.from_expression(input_slot(2)), shape)


if __name__ == "__main__":
    unittest.main()
