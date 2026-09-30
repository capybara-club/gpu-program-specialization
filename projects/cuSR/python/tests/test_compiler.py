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

from cuda.core import Device

from cusr import (
    TileStaticEvalInstantiation,
    TileStaticMseInstantiation,
    compile_tile_static_eval_cubin,
    compile_tile_static_mse_cubin,
    make_tile_static_eval_instantiations,
    make_tile_static_mse_instantiations,
)


class CompilerTest(unittest.TestCase):
    def test_compile_template_instantiation_list(self) -> None:
        instantiations = make_tile_static_mse_instantiations(
            2,
            ast_capacity=8,
            tile_rows=64,
            threads_per_cta=128,
        )
        result = compile_tile_static_mse_cubin(instantiations, arch=f"sm_{Device(0).arch}")

        self.assertTrue(result.cubin.startswith(b"\x7fELF"))
        self.assertEqual(result.size, len(result.cubin))
        self.assertEqual(result.name_expressions, tuple(item.name_expression for item in instantiations))
        self.assertEqual(len(result.lowered_names), len(instantiations))
        self.assertEqual(len(set(result.lowered_names)), len(instantiations))
        self.assertNotIn("\0", result.log)
        self.assertNotIn(b"cusr_tile_static_mse_f32_000", result.cubin)
        for lowered_name in result.lowered_names:
            self.assertIn(lowered_name.encode("ascii"), result.cubin)

    def test_reject_nonpositional_kernel_indices(self) -> None:
        instantiation = TileStaticMseInstantiation(0, 8, 64, 128)
        with self.assertRaisesRegex(ValueError, "consecutive and ordered"):
            compile_tile_static_mse_cubin((instantiation, instantiation), arch="sm_120")

    def test_compile_eval_template_instantiation_list(self) -> None:
        instantiations = make_tile_static_eval_instantiations(
            2,
            ast_capacity=8,
            tile_rows=64,
            threads_per_cta=128,
        )
        result = compile_tile_static_eval_cubin(instantiations, arch=f"sm_{Device(0).arch}")

        self.assertTrue(result.cubin.startswith(b"\x7fELF"))
        self.assertEqual(result.size, len(result.cubin))
        self.assertEqual(result.name_expressions, tuple(item.name_expression for item in instantiations))
        self.assertEqual(len(result.lowered_names), len(instantiations))
        self.assertEqual(len(set(result.lowered_names)), len(instantiations))
        self.assertNotIn("\0", result.log)
        for lowered_name in result.lowered_names:
            self.assertIn(lowered_name.encode("ascii"), result.cubin)

    def test_reject_nonpositional_eval_kernel_indices(self) -> None:
        instantiation = TileStaticEvalInstantiation(0, 8, 64, 128)
        with self.assertRaisesRegex(ValueError, "consecutive and ordered"):
            compile_tile_static_eval_cubin((instantiation, instantiation), arch="sm_120")


if __name__ == "__main__":
    unittest.main()
