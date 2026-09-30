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
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

from secant_system_id.mdbench import (
    add_multiplicative_snr_noise,
    dataset_from_problem,
    load_npz,
    run_reference_pilot,
)
from secant_system_id.odebench import ODEBENCH_REFERENCE_SYSTEMS, load_solutions_json, make_reference_problem
from secant_system_id.problem import RecoveryProtocol


def _npy(shape: tuple[int, ...], values: tuple[float, ...]) -> bytes:
    header_text = repr({"descr": "<f8", "fortran_order": False, "shape": shape})
    prefix_size = 10
    padding = (16 - ((prefix_size + len(header_text) + 1) % 16)) % 16
    header = (header_text + " " * padding + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack(
        "<" + "d" * len(values), *values
    )


class MDBenchAdapterTest(unittest.TestCase):
    def test_loads_dependency_free_npz_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "system_1.npz"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("t.npy", _npy((3,), (0.0, 0.5, 1.0)))
                archive.writestr("u.npy", _npy((3, 2), (1.0, 2.0, 1.5, 3.0, 2.0, 4.0)))
                archive.writestr("du.npy", _npy((3, 2), (1.0, 2.0, 1.0, 2.0, 1.0, 2.0)))
            dataset = load_npz(path)
        self.assertEqual(dataset.times, (0.0, 0.5, 1.0))
        self.assertEqual(dataset.states[1], (1.5, 3.0))
        self.assertEqual(dataset.approximate_derivatives[1], (1.0, 2.0))

    def test_noise_changes_states_but_retains_clean_derivative_targets(self) -> None:
        problem = make_reference_problem(
            ODEBENCH_REFERENCE_SYSTEMS[0],
            protocol=RecoveryProtocol.DERIVATIVE_REGRESSION,
        )
        clean = dataset_from_problem(problem)
        noisy = add_multiplicative_snr_noise(clean, 20.0, seed=7)
        self.assertNotEqual(noisy.states, clean.states)
        self.assertEqual(noisy.true_derivatives, clean.true_derivatives)
        train, test = noisy.split()
        self.assertEqual((len(train.times), len(test.times)), (120, 30))

    def test_clean_noisy_pilot_is_explicitly_not_a_search_result(self) -> None:
        result = run_reference_pilot()
        self.assertEqual(result["kind"], "adapter_correctness_pilot_not_search")
        self.assertFalse(result["official_artifact_used"])
        self.assertEqual(len(result["systems"]), 4)
        self.assertTrue(all(system["clean_oracle_test_nmse"] == 0.0 for system in result["systems"]))
        self.assertTrue(all(system["noisy_oracle_test_nmse"] > 0.0 for system in result["systems"]))

    def test_odebench_json_adapter_can_require_all_63(self) -> None:
        record = {
            "id": 1,
            "eq": "c_0*x_0",
            "dim": 1,
            "eq_description": "growth",
            "solutions": [[{
                "success": True,
                "t": [0.0, 0.5, 1.0],
                "y": [[1.0, 1.5, 2.0]],
                "consts": [0.5],
                "init": [1.0],
                "noise_amplitude": 0.0,
                "random_seed": 42,
            }]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "solutions.json"
            path.write_text(json.dumps([record]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected all 63"):
                load_solutions_json(path)
            problem = load_solutions_json(path, require_all_63=False)[0]
        self.assertEqual(problem.provenance.system_id, "1")
        self.assertEqual(problem.training[0].values[-1], (2.0,))


if __name__ == "__main__":
    unittest.main()
