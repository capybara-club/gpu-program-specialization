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

from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from cuda.core import Device

from cusr import (
    CubinInspectError,
    CubinInspectErrorCode,
    compile_tile_static_eval_cubin,
    compile_tile_static_mse_cubin,
    inspect_cubin,
    make_tile_static_eval_instantiations,
    make_tile_static_mse_instantiations,
    print_cubin_inspection,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_C_INSPECTOR = _REPO_ROOT / "build" / "cusr_inspect_sites_cubin"


def _c_function_pattern(lowered_names: tuple[str, ...]) -> str:
    prefix, separator, suffix = lowered_names[0].partition("ILj0E")
    if not separator:
        raise AssertionError("unexpected template mangling for kernel index zero")
    if lowered_names[1] != f"{prefix}ILj1E{suffix}":
        raise AssertionError("template symbols differ by more than the kernel index")
    return f"{prefix}ILj%dE{suffix}"


class InspectionTest(unittest.TestCase):
    def test_reject_malformed_elf(self) -> None:
        with self.assertRaises(CubinInspectError) as caught:
            inspect_cubin(b"not an ELF", ("kernel",), ast_capacity=8)
        self.assertEqual(caught.exception.code, CubinInspectErrorCode.BAD_ELF)

    def test_python_structured_output(self) -> None:
        instantiations = make_tile_static_mse_instantiations(2, ast_capacity=8)
        compiled = compile_tile_static_mse_cubin(instantiations, arch=f"sm_{Device(0).arch}")
        inspection = inspect_cubin(
            compiled.cubin,
            compiled.lowered_names,
            ast_capacity=8,
            expected_occurrences=1,
        )
        output = StringIO()
        print_cubin_inspection(inspection, output)

        structured = json.loads(output.getvalue())
        self.assertEqual(structured, inspection.to_dict())
        self.assertEqual(structured["sass_arch"], int(Device(0).arch))
        self.assertEqual(structured["num_kernels"], 2)
        self.assertEqual(structured["num_sites"], 2)
        self.assertEqual(structured["num_regcount_records"], 4)
        self.assertEqual(len(structured["kernels"][0]["sites"][0]["input_regs"]), 8)
        self.assertEqual(len(structured["kernels"][0]["sites"][0]["output_regs"]), 8)

    def test_python_matches_c_inspector(self) -> None:
        if not _C_INSPECTOR.is_file():
            self.skipTest("build/cusr_inspect_sites_cubin is required for the cross-implementation check")

        arch = f"sm_{Device(0).arch}"
        for ast_capacity, tile_rows in ((8, 64), (16, 128), (32, 256)):
            with self.subTest(ast_capacity=ast_capacity, tile_rows=tile_rows):
                instantiations = make_tile_static_mse_instantiations(
                    2,
                    ast_capacity=ast_capacity,
                    tile_rows=tile_rows,
                )
                compiled = compile_tile_static_mse_cubin(instantiations, arch=arch)
                python_result = inspect_cubin(
                    compiled.cubin,
                    compiled.lowered_names,
                    ast_capacity=ast_capacity,
                    expected_occurrences=1,
                ).to_dict()

                with tempfile.TemporaryDirectory() as directory:
                    cubin_path = Path(directory) / "template.cubin"
                    cubin_path.write_bytes(compiled.cubin)
                    command = (
                        str(_C_INSPECTOR),
                        "--json",
                        str(cubin_path),
                        _c_function_pattern(compiled.lowered_names),
                        "2",
                        str(ast_capacity),
                        "0x7fc0ffee",
                        "1",
                    )
                    c_result = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)

                self.assertEqual(python_result, c_result)

    def test_eval_template_python_matches_c_inspector(self) -> None:
        if not _C_INSPECTOR.is_file():
            self.skipTest("build/cusr_inspect_sites_cubin is required for the cross-implementation check")

        ast_capacity = 8
        instantiations = make_tile_static_eval_instantiations(2, ast_capacity=ast_capacity)
        compiled = compile_tile_static_eval_cubin(instantiations, arch=f"sm_{Device(0).arch}")
        python_result = inspect_cubin(
            compiled.cubin,
            compiled.lowered_names,
            ast_capacity=ast_capacity,
            expected_occurrences=1,
        ).to_dict()

        with tempfile.TemporaryDirectory() as directory:
            cubin_path = Path(directory) / "eval_template.cubin"
            cubin_path.write_bytes(compiled.cubin)
            command = (
                str(_C_INSPECTOR),
                "--json",
                str(cubin_path),
                _c_function_pattern(compiled.lowered_names),
                "2",
                str(ast_capacity),
                "0x7fc0ffee",
                "1",
            )
            c_result = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)

        self.assertEqual(python_result, c_result)


if __name__ == "__main__":
    unittest.main()
