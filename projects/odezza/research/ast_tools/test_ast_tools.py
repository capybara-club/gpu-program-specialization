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

import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from odezza.ast import Program
from odezza.model import KernelShape


TOOL = Path(os.environ.get("ODEZZA_AST_TOOL", Path(__file__).parent / "build" / "odezza-ast-tools"))


def run_tool(*arguments: str) -> tuple[list[dict[str, object]], str]:
    completed = subprocess.run(
        [str(TOOL), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [json.loads(line) for line in completed.stdout.splitlines()], completed.stderr


class AstToolTests(unittest.TestCase):
    def test_exhaustive_programs_match_python_cpu_reference(self) -> None:
        records, summary = run_tool(
            "--states", "2",
            "--constants", "1",
            "--operators", "add,mul",
            "--depth", "1",
            "--depth-mode", "up-to",
            "--mode", "exhaustive",
            "--state-values", "1.25,-2",
            "--constant-values", "3",
        )
        self.assertEqual(len(records), 21)
        self.assertEqual(len({record["program"] for record in records}), 21)
        self.assertIn("space=21 selected=21 emitted=21", summary)
        shape = KernelShape(state_names=("x", "y"), constant_count=1)
        for record in records:
            program = Program(bytes.fromhex(str(record["program"])))
            program.validate(shape)
            expected = program.evaluate((1.25, -2.0), (3.0,), 0, shape)
            expected_bits = f"{struct.unpack('<I', struct.pack('<f', expected))[0]:08x}"
            self.assertEqual(record["value_bits"], expected_bits)

    def test_toggle_permutation_matches_python(self) -> None:
        arguments = (
            "--states", "2",
            "--constants", "1",
            "--operators", "none",
            "--toggle-bits", "2",
            "--include-toggle2",
            "--depth", "0",
            "--depth-mode", "exact",
            "--mode", "exhaustive",
            "--state-values", "2,5",
            "--constant-values", "7",
            "--permutation", "2",
        )
        records, _ = run_tool(*arguments)
        self.assertEqual(len(records), 21)
        shape = KernelShape(state_names=("x", "y"), constant_count=1)
        for record in records:
            program = Program(bytes.fromhex(str(record["program"])))
            expected = program.evaluate((2.0, 5.0), (7.0,), 2, shape)
            expected_bits = f"{struct.unpack('<I', struct.pack('<f', expected))[0]:08x}"
            self.assertEqual(record["value_bits"], expected_bits)

    def test_four_way_toggle_matches_python(self) -> None:
        records, _ = run_tool(
            "--states", "1",
            "--constants", "1",
            "--operators", "none",
            "--toggle-bits", "2",
            "--include-toggle4",
            "--depth", "0",
            "--depth-mode", "exact",
            "--mode", "exhaustive",
            "--state-values", "2",
            "--constant-values", "7",
            "--permutation", "3",
        )
        self.assertEqual(len(records), 34)
        self.assertEqual(len({record["program"] for record in records}), 34)
        shape = KernelShape(state_names=("x",), constant_count=1)
        for record in records:
            program = Program(bytes.fromhex(str(record["program"])))
            expected = program.evaluate((2.0,), (7.0,), 3, shape)
            expected_bits = f"{struct.unpack('<I', struct.pack('<f', expected))[0]:08x}"
            self.assertEqual(record["value_bits"], expected_bits)

    def test_every_operator_matches_python_on_safe_inputs(self) -> None:
        records, _ = run_tool(
            "--states", "2",
            "--constants", "1",
            "--operators", "add,sub,mul,div,neg,sqrt,rcp,abs,min,max,fma,sin,cos,ex2,lg2,rsqrt,tanh,exp,log",
            "--depth", "1",
            "--depth-mode", "exact",
            "--mode", "exhaustive",
            "--state-values", "1.25,2",
            "--constant-values", "3",
        )
        shape = KernelShape(state_names=("x", "y"), constant_count=1)
        for record in records:
            program = Program(bytes.fromhex(str(record["program"])))
            expected = program.evaluate((1.25, 2.0), (3.0,), 0, shape)
            expected_bits = f"{struct.unpack('<I', struct.pack('<f', expected))[0]:08x}"
            self.assertEqual(record["value_bits"], expected_bits)

    def test_random_selection_is_unique_and_repeatable(self) -> None:
        arguments = (
            "--states", "3",
            "--constants", "1",
            "--operators", "add,sub,mul",
            "--depth", "2",
            "--mode", "random",
            "--count", "32",
            "--seed", "12345",
        )
        first, _ = run_tool(*arguments)
        second, _ = run_tool(*arguments)
        first_ranks = [record["rank"] for record in first]
        self.assertEqual(first, second)
        self.assertEqual(len(first_ranks), 32)
        self.assertEqual(len(set(first_ranks)), 32)
        self.assertNotEqual(first_ranks, list(range(32)))

    def test_structural_random_generates_complete_heavy_systems(self) -> None:
        arguments = (
            "--states", "4",
            "--rhs-count", "4",
            "--constants", "4",
            "--literal", "0",
            "--literal", "1",
            "--operators", "add,sub,mul,div,neg,sin,cos",
            "--depth", "4",
            "--depth-mode", "exact",
            "--mode", "structural-random",
            "--count", "16",
            "--seed", "2468",
            "--no-evaluate",
        )
        first, first_summary = run_tool(*arguments)
        second, second_summary = run_tool(*arguments)
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(len(first), 16)
        self.assertIn("rhs_per_system=4", first_summary)
        self.assertIn("ranked_selection=false", first_summary)
        shape = KernelShape(state_names=("x0", "x1", "x2", "x3"), constant_count=4)
        for system_index, system in enumerate(first):
            self.assertEqual(system["generation_index"], system_index)
            self.assertIsNone(system["rank"])
            rhs = system["rhs"]
            self.assertEqual(len(rhs), 4)
            for state_index, expression in enumerate(rhs):
                self.assertEqual(expression["state_index"], state_index)
                self.assertEqual(expression["depth"], 4)
                self.assertIsNone(expression["expression_rank"])
                Program(bytes.fromhex(str(expression["program"]))).validate(shape)

    def test_structural_random_single_rhs_allows_saturated_space(self) -> None:
        records, summary = run_tool(
            "--states", "4",
            "--constants", "4",
            "--operators", "add,sub,mul,neg",
            "--depth", "4",
            "--depth-mode", "exact",
            "--mode", "structural-random",
            "--count", "8",
            "--seed", "99",
            "--no-evaluate",
        )
        self.assertEqual(len(records), 8)
        self.assertTrue(all(record["depth"] == 4 for record in records))
        self.assertIn("space=18446744073709551615+", summary)

    def test_heavy_systems_flatten_to_binary_ast_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "systems.oast"
            records, summary = run_tool(
                "--states", "4",
                "--rhs-count", "4",
                "--constants", "4",
                "--operators", "add,sub,mul,neg",
                "--depth", "4",
                "--depth-mode", "exact",
                "--mode", "structural-random",
                "--count", "16",
                "--seed", "123",
                "--binary-output", str(path),
            )
            self.assertEqual(records, [])
            self.assertIn("systems=16 rhs_per_system=4 records=64", summary)
            header = path.read_bytes()[:24]
            self.assertEqual(struct.unpack_from("<Q", header, 16)[0], 64)
            _, verify_summary = run_tool("--verify-binary", str(path))
            self.assertIn("verified_records=64", verify_summary)

    def test_exact_depth_excludes_shallower_trees(self) -> None:
        records, summary = run_tool(
            "--states", "2",
            "--constants", "0",
            "--operators", "add",
            "--depth", "2",
            "--depth-mode", "exact",
            "--mode", "exhaustive",
        )
        self.assertEqual(len(records), 32)
        self.assertEqual(len({record["program"] for record in records}), 32)
        self.assertTrue(all(record["depth"] == 2 for record in records))
        self.assertIn("space=32 selected=32 emitted=32", summary)

    def test_binary_file_round_trip(self) -> None:
        generation = (
            "--states", "3",
            "--constants", "2",
            "--literal", "0",
            "--literal", "1",
            "--operators", "add,sub,mul,neg",
            "--toggle-bits", "3",
            "--include-toggle2",
            "--depth", "2",
            "--depth-mode", "exact",
            "--mode", "random",
            "--count", "200",
            "--seed", "90210",
        )
        evaluation = (
            "--states", "3",
            "--constants", "2",
            "--toggle-bits", "3",
            "--state-values", "1.25,-2,4",
            "--constant-values", "0.5,3",
            "--permutation", "5",
        )
        expected, _ = run_tool(*generation, *evaluation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "programs.oast"
            records, write_summary = run_tool(*generation, "--binary-output", str(path))
            self.assertEqual(records, [])
            self.assertIn("records=200", write_summary)
            header = path.read_bytes()[:24]
            self.assertEqual(header[:8], b"ODEZZAAS")
            self.assertEqual(struct.unpack_from("<I", header, 8)[0], 1)
            self.assertGreater(struct.unpack_from("<I", header, 12)[0], 0)
            self.assertEqual(struct.unpack_from("<Q", header, 16)[0], 200)
            records, verify_summary = run_tool("--verify-binary", str(path))
            self.assertEqual(records, [])
            self.assertIn("verified_records=200", verify_summary)
            actual, _ = run_tool(*evaluation, "--binary-input", str(path))
            for index, record in enumerate(expected):
                record["rank"] = index
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
