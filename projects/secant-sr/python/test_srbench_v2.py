#!/usr/bin/env python3
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
"""Deterministic tests for the frozen SRBench v2.0 compatibility path."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import unittest
import uuid

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from srbench_data import (
    HEADER,
    prepare_srbench_v2_blackbox_dataset,
    prepare_srbench_v2_groundtruth_dataset,
)
from srbench_symbolic import assess_expression
from srbench_v2 import SRBENCH_V2_BLACKBOX_PROTOCOL, SRBENCH_V2_PROTOCOL, SRBENCH_V2_TRAIN_ROWS
from run_suite import dynamic_leaf_fallback_reason


class SRBenchV2Test(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.work = root / "scratch" / f"srbench_v2_test_{uuid.uuid4().hex}"
        self.dataset_dir = self.work / "datasets" / "test_problem"
        self.dataset_dir.mkdir(parents=True)
        self.source = self.dataset_dir / "test_problem.tsv.gz"
        self.metadata = self.dataset_dir / "metadata.yaml"

    def tearDown(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)

    def _dataset_write(self, num_rows: int = 10_001) -> tuple[np.ndarray, np.ndarray]:
        row = np.arange(num_rows, dtype=np.float64)
        features = np.column_stack((row * 0.25 - 3.0, row * -0.5 + 7.0))
        target = features[:, 0] + 2.0 * features[:, 1]
        frame = pd.DataFrame({"a": features[:, 0], "b": features[:, 1], "target": target})
        with gzip.open(self.source, "wt", encoding="utf-8", newline="") as output:
            frame.to_csv(output, sep="\t", index=False)
        self.metadata.write_text(
            "description: |\n  f = a + 2*b\nfeatures:\n  - name: a\n  - name: b\n",
            encoding="utf-8",
        )
        return features, target

    @staticmethod
    def _binary_read(path: Path):
        data = path.read_bytes()
        _, _, num_inputs, num_train_rows, num_validation_rows = HEADER.unpack_from(data)
        values = np.frombuffer(data, dtype="<f4", offset=HEADER.size)
        input_count = num_inputs * num_train_rows
        train_input = values[:input_count].reshape(num_inputs, num_train_rows)
        offset = input_count
        train_target = values[offset:offset + num_train_rows]
        offset += num_train_rows
        validation_count = num_inputs * num_validation_rows
        validation_input = values[offset:offset + validation_count].reshape(num_inputs, num_validation_rows)
        offset += validation_count
        validation_target = values[offset:offset + num_validation_rows]
        return train_input, train_target, validation_input, validation_target

    def test_data_path_matches_v2_reference(self) -> None:
        features, target = self._dataset_write()
        seed = 23654
        noise = 0.01
        prepared = prepare_srbench_v2_groundtruth_dataset(self.source, self.work / "prepared", seed, noise)

        np.random.seed(seed)
        train_x, validation_x, train_y, validation_y = train_test_split(
            features, target, train_size=0.75, test_size=0.25, random_state=seed)
        sample_idx = np.random.choice(np.arange(len(train_x)), size=SRBENCH_V2_TRAIN_ROWS)
        train_x = train_x[sample_idx]
        train_y = train_y[sample_idx]
        train_y = train_y + np.random.normal(
            0.0, noise * np.sqrt(np.mean(np.square(train_y))), size=len(train_y))
        actual = self._binary_read(prepared.path)

        self.assertEqual(prepared.protocol, SRBENCH_V2_PROTOCOL)
        self.assertEqual(prepared.num_train_rows, SRBENCH_V2_TRAIN_ROWS)
        self.assertEqual(prepared.num_validation_rows, len(validation_y))
        np.testing.assert_array_equal(actual[0], train_x.astype(np.float32).T)
        np.testing.assert_array_equal(actual[1], train_y.astype(np.float32))
        np.testing.assert_array_equal(actual[2], validation_x.astype(np.float32).T)
        np.testing.assert_array_equal(actual[3], validation_y.astype(np.float32))

    def test_blackbox_data_path_matches_v2_reference(self) -> None:
        features, target = self._dataset_write()
        seed = 23654
        prepared = prepare_srbench_v2_blackbox_dataset(self.source, self.work / "prepared", seed)

        np.random.seed(seed)
        train_x, validation_x, train_y, validation_y = train_test_split(
            features, target, train_size=0.75, test_size=0.25, random_state=seed)
        sample_idx = np.random.choice(np.arange(len(train_x)), size=SRBENCH_V2_TRAIN_ROWS)
        train_x = train_x[sample_idx]
        train_y = train_y[sample_idx]
        scaler_x = StandardScaler()
        train_x = scaler_x.fit_transform(train_x)
        validation_x = scaler_x.transform(validation_x)
        scaler_y = StandardScaler()
        train_y = scaler_y.fit_transform(train_y.reshape(-1, 1)).ravel()
        validation_y = scaler_y.transform(validation_y.reshape(-1, 1)).ravel()
        actual = self._binary_read(prepared.path)

        self.assertEqual(prepared.protocol, SRBENCH_V2_BLACKBOX_PROTOCOL)
        self.assertTrue(prepared.scale_x)
        self.assertTrue(prepared.scale_y)
        self.assertEqual(prepared.num_train_rows, SRBENCH_V2_TRAIN_ROWS)
        self.assertEqual(prepared.num_validation_rows, len(validation_y))
        np.testing.assert_array_equal(actual[0], train_x.astype(np.float32).T)
        np.testing.assert_array_equal(actual[1], train_y.astype(np.float32))
        np.testing.assert_array_equal(actual[2], validation_x.astype(np.float32).T)
        np.testing.assert_array_equal(actual[3], validation_y.astype(np.float32))

    def test_wide_blackbox_problems_route_away_from_mixed_kernel(self) -> None:
        self.assertIsNone(dynamic_leaf_fallback_reason(48, 8, 32))
        self.assertIsNone(dynamic_leaf_fallback_reason(50, 8, 32))
        self.assertEqual(
            dynamic_leaf_fallback_reason(100, 8, 32),
            "mixed_kernel_register_budget_exceeds_256",
        )
        self.assertEqual(
            dynamic_leaf_fallback_reason(117, 8, 32),
            "mixed_kernel_register_budget_exceeds_256",
        )
        self.assertEqual(
            dynamic_leaf_fallback_reason(124, 8, 32),
            "static_plus_dynamic_inputs_exceed_128",
        )

    def test_symbolic_assessment_matches_srbench_rules(self) -> None:
        self._dataset_write(16)
        self.metadata.write_text(
            "description: |\n  f = a\nfeatures:\n  - name: a\n  - name: b\n",
            encoding="utf-8",
        )

        exact = assess_expression(self.source, "x0", 1.0)
        offset = assess_expression(self.source, "(x0 + 3)", 1.0)
        scaled = assess_expression(self.source, "(2 * x0)", 1.0)
        wrong = assess_expression(self.source, "(x0 * x1)", 0.9)
        inaccurate = assess_expression(self.source, "x0", 0.5)
        squared = assess_expression(self.source, "square(x0)", 1.0)

        self.assertTrue(exact.symbolic_error_is_zero)
        self.assertTrue(offset.symbolic_error_is_constant)
        self.assertTrue(scaled.symbolic_fraction_is_constant)
        self.assertFalse(wrong.symbolic_solution)
        self.assertFalse(inaccurate.symbolic_solution)
        self.assertFalse(squared.symbolic_solution)

        self.metadata.write_text(
            "description: |\n  f = a**2\nfeatures:\n  - name: a\n  - name: b\n",
            encoding="utf-8",
        )
        self.assertTrue(assess_expression(self.source, "square(x0)", 1.0).symbolic_solution)
        self.metadata.write_text(
            "description: |\n  f = a**3\nfeatures:\n  - name: a\n  - name: b\n",
            encoding="utf-8",
        )
        self.assertTrue(assess_expression(self.source, "cube(x0)", 1.0).symbolic_solution)

    def test_report_accepts_deferred_symbolic_assessment(self) -> None:
        results = self.work / "results.csv"
        fields = ("problem", "seed", "target_noise", "accuracy_solution", "symbolic_solution", "elapsed_seconds")
        rows = (
            ("first", 1, 0.0, 1, "", 1.0),
            ("first", 2, 0.0, 1, "", 2.0),
            ("second", 1, 0.0, 1, "", 3.0),
            ("second", 2, 0.0, 0, "", 4.0),
        )
        with results.open("w", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(fields)
            writer.writerows(rows)

        completed = subprocess.run(
            (sys.executable, str(Path(__file__).with_name("srbench_v2_report.py")), str(results)),
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("| 0 | 2 | 2 | 75.00% | 75.00% | not assessed | 10.000 s | 2.500 s |", completed.stdout)

    def test_blackbox_report_excludes_wide_rows(self) -> None:
        results = self.work / "blackbox_results.csv"
        fields = ("problem", "seed", "validation_r2", "nodes", "elapsed_seconds", "num_inputs")
        rows = (
            ("narrow", 1, 0.75, 7, 2.0, 32),
            ("wide", 1, 1.0, 9, 3.0, 33),
        )
        with results.open("w", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(fields)
            writer.writerows(rows)

        completed = subprocess.run(
            (
                sys.executable,
                str(Path(__file__).with_name("srbench_v2_blackbox_report.py")),
                str(results),
                "--max-inputs", "32",
            ),
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("Scope: Secant datasets with at most 32 inputs", completed.stdout)
        self.assertIn("| 1 | 1 | 0.750000 |", completed.stdout)


if __name__ == "__main__":
    unittest.main()
