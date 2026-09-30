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

from __future__ import annotations

import unittest

import pandas as pd

from srbench_v2_targets import targets_select


class SRBenchV2TargetsTest(unittest.TestCase):
    def test_selects_and_normalizes_requested_trials(self) -> None:
        results = pd.DataFrame([
            {
                "algorithm": "Operon",
                "dataset": "feynman_a",
                "random_state": 20,
                "target_noise": 0.0,
                "data_group": "Feynman",
                "training time (s)": 3.5,
                "r2_test": 0.9995,
                "model_size": 7,
                "symbolic_solution": True,
                "symbolic_model": "x0 + 1",
            },
            {
                "algorithm": "Other",
                "dataset": "feynman_a",
                "random_state": 20,
                "target_noise": 0.0,
                "data_group": "Feynman",
                "training time (s)": 1.0,
                "r2_test": 1.0,
                "model_size": 1,
                "symbolic_solution": False,
                "symbolic_model": "0",
            },
        ])

        records = targets_select(results, "Operon", "Feynman", 0.0, {"feynman_a"})

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["random_state"], 20)
        self.assertEqual(records[0]["training_seconds"], 3.5)
        self.assertEqual(records[0]["symbolic_solution"], 1)


if __name__ == "__main__":
    unittest.main()
