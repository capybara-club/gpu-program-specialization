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

import secant
from benchmark_programs import generate_unique_balanced_programs


class BenchmarkProgramTests(unittest.TestCase):
    def test_unique_balanced_programs_are_valid_unique_and_deterministic(self) -> None:
        for mode in ("alu", "mufu"):
            data, offsets = generate_unique_balanced_programs(4096, mode, seed=17)
            repeated_data, repeated_offsets = generate_unique_balanced_programs(4096, mode, seed=17)
            different_data, _ = generate_unique_balanced_programs(4096, mode, seed=18)

            self.assertEqual(data, repeated_data)
            self.assertEqual(offsets.tolist(), repeated_offsets.tolist())
            self.assertNotEqual(data, different_data)

            programs = {
                bytes(data[int(offsets[index]) : int(offsets[index + 1])])
                for index in range(4096)
            }
            self.assertEqual(len(programs), 4096)
            for program in programs:
                self.assertEqual(secant.validate_program(program), program)

    def test_unique_balanced_programs_reject_exhausted_space(self) -> None:
        with self.assertRaisesRegex(ValueError, "space of"):
            generate_unique_balanced_programs(
                17,
                "alu",
                seed=1,
                num_inputs=1,
                num_leaves=2,
            )


if __name__ == "__main__":
    unittest.main()
