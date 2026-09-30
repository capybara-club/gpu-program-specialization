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

import numpy as np

import secant
from secant.routines import DEFAULT_ROUTINES


class AstTests(unittest.TestCase):
    def test_variable_length_encoding_matches_secant_header(self) -> None:
        program = secant.Program(secant.input(0) + secant.constant(1.5))
        self.assertEqual(program.bytecode, bytes((0xB6, 0x00, 0x81, 0x00, 0x00, 0xC0, 0x3F, 0x85, 0x83)))

    def test_expression_and_routine_interpreter(self) -> None:
        x0 = secant.input(0)
        x1 = secant.input(1)
        programs = (
            secant.Program(secant.sin(x0) * secant.cos(x1)),
            secant.Program(secant.safe_div(x0, x1)),
            secant.Program(secant.fma(x0, x1, 0.5)),
            secant.Program(secant.exp(x0) + secant.log(x1)),
        )
        values = np.array(
            (
                (0.25, 1.5),
                (0.75, 2.0),
                (1.25, 2.5),
            ),
            dtype=np.float32,
        )
        actual = np.stack(tuple(secant.evaluate(program, values, routines=DEFAULT_ROUTINES) for program in programs))
        expected = np.stack(
            (
                np.sin(values[:, 0]) * np.cos(values[:, 1]),
                values[:, 0] * values[:, 1] / (values[:, 1] * values[:, 1] + np.float32(0.01)),
                values[:, 0] * values[:, 1] + np.float32(0.5),
                np.exp(values[:, 0]) + np.log(values[:, 1]),
            )
        ).astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)

    def test_materialize_sse_and_dynamic_constant_layouts(self) -> None:
        rng = np.random.default_rng(0x5EC0)
        rows = 17
        inputs = rng.uniform(0.5, 3.0, size=(4, rows)).astype(np.float32)
        targets = rng.uniform(-2.0, 2.0, size=(2, rows)).astype(np.float32)
        constants = rng.uniform(0.5, 2.0, size=(2, 5)).astype(np.float32)
        programs = (
            secant.Program(secant.input(0) + secant.input(1)),
            secant.Program(secant.sin(secant.input(2)) * secant.input(3)),
        )
        dynamic_programs = (
            secant.Program(secant.input(0) + secant.dynamic_constant(0)),
            secant.Program(secant.sin(secant.input(1)) * secant.dynamic_constant(1)),
        )

        values = secant.materialize(programs, inputs)
        self.assertEqual(values.shape, (2, rows))
        self.assertEqual(secant.sse(programs, inputs, targets).shape, (2, 2))
        self.assertEqual(
            secant.dynamic_constant_sse(dynamic_programs, inputs, constants, targets).shape,
            (2, 2, 5),
        )

    def test_dynamic_leaf_layout_and_kind_checks(self) -> None:
        inputs = np.arange(3 * 11, dtype=np.float32).reshape(3, 11) / np.float32(7.0)
        targets = np.zeros((1, 11), dtype=np.float32)
        constant_bits = np.asarray((1.25, -0.5), dtype=np.float32).view(np.uint32)
        leaf_masks = np.asarray((0b01, 0b10), dtype=np.uint32)
        leaf_words = np.asarray(((2, constant_bits[0]), (constant_bits[1], 1)), dtype=np.uint32)
        programs = (
            secant.Program(secant.dynamic_constant_or_column(0) + secant.dynamic_constant_or_column(1)),
            secant.Program(secant.dynamic_constant_or_column(0) * secant.dynamic_constant_or_column(1)),
        )

        actual = secant.dynamic_leaf_sse(programs, inputs, leaf_masks, leaf_words, targets)
        expected0 = np.sum((inputs[2] + np.float32(1.25)) ** 2, dtype=np.float32)
        expected1 = np.sum((np.float32(-0.5) * inputs[1]) ** 2, dtype=np.float32)
        self.assertEqual(actual.shape, (2, 1, 2))
        np.testing.assert_allclose(actual[0, 0, 0], expected0, rtol=2.0e-6)
        np.testing.assert_allclose(actual[1, 0, 1], expected1, rtol=2.0e-6)
        dedicated = secant.Program(secant.dynamic_column(0) + secant.dynamic_constant(1))
        secant.dynamic_leaf_sse((dedicated,), inputs, leaf_masks[:1], leaf_words[:1], targets)
        with self.assertRaisesRegex(ValueError, "contains a constant"):
            secant.dynamic_leaf_sse((dedicated,), inputs, leaf_masks[1:], leaf_words[1:], targets)
        with self.assertRaisesRegex(ValueError, "static column index 0 is out of range"):
            secant.dynamic_leaf_sse((secant.Program(secant.input(0)),), inputs, leaf_masks, leaf_words, targets)
        mixed = secant.Program(secant.input(0) + secant.dynamic_constant_or_column(1))
        mixed_actual = secant.dynamic_leaf_sse(
            (mixed,),
            inputs,
            leaf_masks,
            leaf_words,
            targets,
            num_static_input_columns=inputs.shape[0],
        )
        mixed_expected = np.sum((inputs[0] + np.float32(1.25)) ** 2, dtype=np.float32)
        np.testing.assert_allclose(mixed_actual[0, 0, 0], mixed_expected, rtol=2.0e-6)
        with self.assertRaisesRegex(ValueError, "dynamic leaf index 2"):
            secant.dynamic_leaf_sse(
                (secant.Program(secant.dynamic_constant_or_column(2)),),
                inputs,
                leaf_masks,
                leaf_words,
                targets,
            )


if __name__ == "__main__":
    unittest.main()
