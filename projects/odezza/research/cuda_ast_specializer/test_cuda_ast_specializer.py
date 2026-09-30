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

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).with_name("cuda_ast_specializer.py")
SPEC = importlib.util.spec_from_file_location("cuda_ast_specializer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DecodeTests(unittest.TestCase):
    def test_binary_expression_is_emitted_in_postorder(self) -> None:
        encoded = bytes((MODULE.OP_STATE, 0, MODULE.OP_CONSTANT, 1, MODULE.OP_MUL, MODULE.OP_RETURN))
        program = MODULE.decode_program(encoded)
        self.assertEqual(program.operator_count, 1)
        self.assertEqual(program.toggle_count, 0)
        self.assertIn("stage_state[0]", program.statements[0])
        self.assertIn("constant1", program.statements[0])

    def test_toggle_preserves_declared_bit(self) -> None:
        encoded = bytes((MODULE.OP_STATE, 0, MODULE.OP_CONSTANT, 1, MODULE.OP_TOGGLE2, 7, MODULE.OP_RETURN))
        program = MODULE.decode_program(encoded)
        self.assertEqual(program.toggle_count, 1)
        self.assertIn("permutation >> 7u", program.statements[0])

    def test_generated_source_has_one_case_per_system(self) -> None:
        first = MODULE.decode_program(bytes((MODULE.OP_STATE, 0, MODULE.OP_RETURN)))
        second = MODULE.decode_program(bytes((MODULE.OP_CONSTANT, 0, MODULE.OP_RETURN)))
        source = MODULE.generate_cuda([first, second])
        self.assertIn("#define SYSTEM_CAPACITY 2u", source)
        self.assertEqual(source.count("case 0u:"), 1)
        self.assertEqual(source.count("case 1u:"), 1)


if __name__ == "__main__":
    unittest.main()
